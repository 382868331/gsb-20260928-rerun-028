"""Departure-window analysis: safe/blocked/undetermined intervals over a
user-specified range of departure times on the unchanged original route.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vehicle_planning.cli import main
from vehicle_planning.departure import (
    BLOCKED,
    SAFE,
    UNDETERMINED,
    evaluate_departure_window,
    review_saved_window,
    save_window_report,
    select_departure,
)
from vehicle_planning.plan import load_plan, save_plan
from vehicle_planning.replay import replay
from vehicle_planning.scene import scene_from_dict
from vehicle_planning.scheduler import Scheduler, SchedulerConfig
from vehicle_planning.vehicle import load_vehicle

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "data/scenes/pedestrian_crossing.json"

# Obstacle crosses the corridor inside the original ~4.2-4.4s passage window.
EARLY = [[0.0, 11.0, 4.0, 0.0], [4.0, 11.0, 4.0, 0.0], [8.0, -8.0, 4.0, 0.0]]


@pytest.fixture(scope="module")
def base():
    vehicle = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    data = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    base_scene = scene_from_dict(data)
    geometric = replace(
        load_plan(ROOT / "outputs/reverse_into_bay.json"),
        scene=base_scene.name,
    )
    plan = Scheduler(vehicle, base_scene, SchedulerConfig()).schedule(geometric)
    assert replay(plan, vehicle, base_scene).valid
    return vehicle, data, plan


def changed_scene(data, keyframes):
    payload = copy.deepcopy(data)
    payload["dynamic_obstacles"][0]["keyframes"] = keyframes
    return scene_from_dict(payload)


def test_window_splits_safe_and_blocked_intervals_with_obstacle_names(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_departure_window(
        plan, vehicle, scene, earliest=0.0, latest=6.0, step=0.5
    )
    statuses = {interval.status for interval in assessment.intervals}
    assert SAFE in statuses
    assert BLOCKED in statuses
    assert UNDETERMINED not in statuses
    blocked = [i for i in assessment.intervals if i.status == BLOCKED]
    assert all("trolley" in i.obstacles for i in blocked)
    # Departure at 0s hits the early obstacle; a late departure clears it.
    assert assessment.earliest_status == BLOCKED
    assert assessment.latest_status == SAFE
    assert assessment.budget_exhausted is False
    assert assessment.checks_used == 13


def test_boundary_departures_are_reported_explicitly(base):
    vehicle, data, plan = base
    scene = scene_from_dict(data)
    assessment = evaluate_departure_window(
        plan, vehicle, scene, earliest=0.0, latest=4.0, step=1.0
    )
    assert assessment.earliest_status == SAFE
    assert assessment.latest_status == SAFE
    boundary = assessment.to_dict()["boundary"]
    assert boundary["earliest"] == {"time": 0.0, "status": SAFE}
    assert boundary["latest"] == {"time": 4.0, "status": SAFE}


def test_budget_exhaustion_marks_tail_undetermined_never_safe(base):
    vehicle, data, plan = base
    scene = scene_from_dict(data)
    assessment = evaluate_departure_window(
        plan, vehicle, scene, earliest=0.0, latest=10.0, step=0.5, max_checks=5
    )
    assert assessment.budget_exhausted is True
    assert assessment.checks_used == 5
    undetermined = [i for i in assessment.intervals if i.status == UNDETERMINED]
    assert undetermined, "unevaluated departures must stay undetermined"
    # Undetermined intervals report zero verified points and no safe claim.
    assert all(i.checked_points == 0 for i in undetermined)
    # Boundaries are probed first, so both ends are known even on a budget.
    assert assessment.earliest_status == SAFE
    assert assessment.latest_status == SAFE
    # Selecting inside an undetermined region is refused.
    with pytest.raises(ValueError, match="not a verified safe offset"):
        select_departure(assessment, plan, vehicle, scene, 5.0)


def test_select_departure_from_safe_interval_verifies_full_route(base, tmp_path):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_departure_window(
        plan, vehicle, scene, earliest=0.0, latest=6.0, step=0.5
    )
    with pytest.raises(ValueError, match="not a verified safe offset"):
        select_departure(assessment, plan, vehicle, scene, 0.0)
    chosen = select_departure(assessment, plan, vehicle, scene, 6.0)
    assert replay(chosen, vehicle, scene).valid
    assert chosen.scene_signature == scene.signature()
    assert chosen.actions[0].type == "wait"
    assert chosen.actions[0].end_time == pytest.approx(6.0)
    # The original success record is untouched by the whole flow.
    original_path = tmp_path / "original.json"
    save_plan(plan, original_path)
    before = original_path.read_bytes()
    save_window_report(assessment, tmp_path / "window/report.json", selected=(6.0, chosen))
    assert original_path.read_bytes() == before
    saved = load_plan(tmp_path / "window/report_depart_6.00s.json")
    assert replay(saved, vehicle, scene).valid


def test_saved_report_distinguishes_recheck_from_historical(base, tmp_path):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_departure_window(
        plan, vehicle, scene, earliest=0.0, latest=6.0, step=0.5
    )
    report_path = tmp_path / "window.json"
    save_window_report(assessment, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))

    # Same timetable: intervals stay current.
    review = review_saved_window(saved, scene)
    assert review["timetable_changed"] is False
    assert all(i["review"] == "current" for i in review["intervals"])

    # Changed obstacle schedule: every old interval needs re-checking.
    newer = changed_scene(data, [[0.0, 11.0, 5.0, 0.0], [9.0, 11.0, 5.0, 0.0], [14.0, -8.0, 5.0, 0.0]])
    review = review_saved_window(saved, newer)
    assert review["timetable_changed"] is True
    assert review["intervals"], "expected historical intervals"
    assert all(i["review"] == "needs_recheck" for i in review["intervals"])
    assert all("historical" in i["basis"] for i in review["intervals"])


def test_cli_window_writes_report_and_selected_plan(base, tmp_path):
    _, data, plan = base
    original_path = tmp_path / "original.json"
    save_plan(plan, original_path)
    before = original_path.read_bytes()
    scene_path = tmp_path / "scene_early.json"
    scene_path.write_text(json.dumps(changed_scene(data, EARLY).to_dict()), encoding="utf-8")
    report_path = tmp_path / "window/report.json"

    exit_code = main([
        "window",
        "--vehicle", str(ROOT / "data/vehicles/compact_car.json"),
        "--scene", str(scene_path),
        "--plan", str(original_path),
        "--earliest", "0",
        "--latest", "6",
        "--step", "0.5",
        "--select", "6",
        "--output", str(report_path),
    ])
    assert exit_code == 0
    assert original_path.read_bytes() == before
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "departure_window_v1"
    assert report["budget_exhausted"] is False
    assert {i["status"] for i in report["intervals"]} >= {SAFE, BLOCKED}
    assert report["selected"]["departure"] == 6.0
    assert report["selected"]["verified"] is True

    # A second run on a changed timetable reviews the saved report.
    newer_path = tmp_path / "scene_newer.json"
    newer = changed_scene(data, [[0.0, 11.0, 5.0, 0.0], [9.0, 11.0, 5.0, 0.0], [14.0, -8.0, 5.0, 0.0]])
    newer_path.write_text(json.dumps(newer.to_dict()), encoding="utf-8")
    report2 = tmp_path / "window/report2.json"
    exit_code = main([
        "window",
        "--vehicle", str(ROOT / "data/vehicles/compact_car.json"),
        "--scene", str(newer_path),
        "--plan", str(original_path),
        "--earliest", "0",
        "--latest", "6",
        "--step", "0.5",
        "--output", str(report2),
        "--previous", str(report_path),
    ])
    assert exit_code == 0
    payload = json.loads(report2.read_text(encoding="utf-8"))
    review = payload["previous_review"]
    assert review["timetable_changed"] is True
    assert all(i["review"] == "needs_recheck" for i in review["intervals"])
