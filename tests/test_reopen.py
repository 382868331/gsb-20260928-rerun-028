"""Reopen workflow: a timetable change replans the whole question, never
silently re-validates or rewrites the old success record.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vehicle_planning.cli import main
from vehicle_planning.plan import load_plan, save_plan
from vehicle_planning.reopen import (
    evaluate_reopen,
    find_delayed_plan,
    same_route,
    same_timing,
    save_reopen_bundle,
    scan_conflicts,
)
from vehicle_planning.replay import replay
from vehicle_planning.scene import scene_from_dict
from vehicle_planning.scheduler import Scheduler, SchedulerConfig
from vehicle_planning.vehicle import load_vehicle

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "data/scenes/pedestrian_crossing.json"

# Obstacle arrives early and crosses the corridor entrance inside the
# original ~4.2-4.4s passage window.
EARLY = [[0.0, 11.0, 4.0, 0.0], [4.0, 11.0, 4.0, 0.0], [8.0, -8.0, 4.0, 0.0]]
# Obstacle is still up north and crosses later than the original schedule.
LATE = [[0.0, 11.0, 4.5, 0.0], [6.0, 11.0, 4.5, 0.0], [12.0, -15.0, 4.5, 0.0]]
# Obstacle leaves before the vehicle arrives, but the timetable hash differs.
EARLY_LEAVE = [[0.0, -8.0, 4.5, 0.0], [3.0, -2.0, 4.5, 0.0], [6.0, -15.0, 4.5, 0.0]]
# Obstacle timetable stays far away from the route at all times.
AVOID = [[0.0, -15.0, 9.5, 0.0], [10.0, -15.0, 9.5, 0.0], [20.0, -25.0, 9.5, 0.0]]


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


@pytest.fixture()
def cheap_replan(base, monkeypatch):
    """Re-plan on the new timetable using the same saved geometry.

    Keeps the route identical on purpose so tests can assert that route and
    timing changes are reported independently; the real geometric search is
    covered by test_motion_planning.
    """
    import vehicle_planning.reopen as reopen_module

    _, _, plan = base

    def run_full_replan(vehicle, new_scene, **kwargs):
        scheduler = Scheduler(
            vehicle, new_scene, kwargs.get("scheduler_config") or SchedulerConfig()
        )
        return scheduler.schedule(replace(plan, actions=(), scene=new_scene.name))

    monkeypatch.setattr(reopen_module, "run_full_replan", run_full_replan)


def test_changed_timetable_cannot_be_revalidated_even_when_obstacle_moves_away(base, cheap_replan):
    vehicle, data, plan = base
    for frames in (EARLY_LEAVE, AVOID):
        scene = changed_scene(data, frames)
        assert replay(plan, vehicle, scene).valid is False
        assessment = evaluate_reopen(plan, vehicle, scene, max_delay=0.0)
        assert assessment.timetable_changed is True
        assert assessment.original_feasible is False
        assert "re-planned" in assessment.option("original").message
        assert assessment.conflicts == ()


def test_conflict_intervals_are_reported_without_touching_route(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    conflicts = scan_conflicts(plan, vehicle, scene)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.obstacles == ("trolley",)
    assert 4.0 < conflict.start_time < conflict.end_time < 5.5
    assert conflict.samples >= 5
    assert replay(plan, vehicle, scene_from_dict(data)).valid


def test_delay_is_tried_before_route_change_and_changes_only_timing(base, cheap_replan):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_reopen(plan, vehicle, scene, max_delay=10, delay_step=0.5)

    assert assessment.original_feasible is False
    delayed = assessment.option("delayed")
    assert delayed.feasible is True
    assert delayed.route_changed is False
    assert delayed.same_route is True
    assert delayed.timing_changed is True
    assert delayed.departure_delay == pytest.approx(0.5)
    assert delayed.final_time == pytest.approx(plan.final_time + delayed.departure_delay)
    assert assessment.recommendation == "delayed"

    delayed_plan = find_delayed_plan(plan, vehicle, scene, max_delay=10, delay_step=0.5)[1]
    assert same_route(plan, delayed_plan)
    assert not same_timing(plan, delayed_plan)
    assert replay(delayed_plan, vehicle, scene).valid


def test_late_obstacle_conflicts_on_original_schedule_but_delay_resolves(base, cheap_replan):
    vehicle, data, plan = base
    scene = changed_scene(data, LATE)
    assessment = evaluate_reopen(plan, vehicle, scene, max_delay=10, delay_step=0.5)

    assert assessment.original_feasible is False
    assert len(assessment.conflicts) >= 1
    delayed = assessment.option("delayed")
    assert delayed.feasible is True
    assert delayed.departure_delay > 0.0
    assert delayed.route_changed is False
    assert delayed.timing_changed is True


def test_replanned_option_uses_new_timetable_and_flags_route_vs_timing(base, cheap_replan):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_reopen(plan, vehicle, scene, max_delay=0.1)

    assert assessment.option("delayed").feasible is False
    replanned = assessment.option("replanned")
    assert replanned.feasible is True
    # The cheap replanner keeps geometry but regenerates the schedule.
    assert replanned.same_route is True
    assert replanned.route_changed is False
    assert replanned.scene_signature == scene.signature()


def test_bundle_preserves_old_success_and_binds_each_option_to_new_timetable(
    base, cheap_replan, tmp_path
):
    vehicle, data, plan = base
    scene = changed_scene(data, LATE)
    assessment = replace(
        evaluate_reopen(plan, vehicle, scene, max_delay=10, delay_step=0.5),
        original_plan_path="outputs/pedestrian_crossing.json",
    )
    delayed_plan = assessment.option_plans["delayed"]
    replanned = assessment.option_plans["replanned"]

    original_path = tmp_path / "original_success.json"
    save_plan(plan, original_path)
    before = original_path.read_bytes()

    report_path = tmp_path / "reopen/report.json"
    save_reopen_bundle(assessment, None, report_path)

    assert original_path.read_bytes() == before
    saved_delayed = load_plan(tmp_path / "reopen/report_delayed.json")
    saved_replanned = load_plan(tmp_path / "reopen/report_replanned.json")
    assert saved_delayed.scene_signature == scene.signature()
    assert saved_replanned.scene_signature == scene.signature()
    assert replay(saved_delayed, vehicle, scene).valid
    assert replay(saved_replanned, vehicle, scene).valid
    assert not replay(saved_delayed, vehicle, scene_from_dict(data)).valid

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "reopen_assessment_v1"
    assert set(report["option_plan_files"]) == {"delayed", "replanned"}
    for option in report["options"].values():
        assert option["scene_signature"] == scene.signature()


def test_cli_reopen_writes_bundle_without_modifying_plan_file(base, cheap_replan, tmp_path):
    _, data, plan = base
    original_path = tmp_path / "original.json"
    save_plan(plan, original_path)
    before = original_path.read_bytes()

    scene_path = tmp_path / "scene_late.json"
    scene_path.write_text(json.dumps(changed_scene(data, LATE).to_dict()), encoding="utf-8")
    report_path = tmp_path / "reopen_report.json"

    exit_code = main([
        "reopen",
        "--vehicle", str(ROOT / "data/vehicles/compact_car.json"),
        "--scene", str(scene_path),
        "--plan", str(original_path),
        "--output", str(report_path),
        "--max-delay", "10",
        "--delay-step", "0.5",
    ])
    assert exit_code == 0
    assert original_path.read_bytes() == before
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["timetable_changed"] is True
    assert report["recommendation"] == "delayed"
    assert (tmp_path / "reopen_report_delayed.json").exists()
    assert (tmp_path / "reopen_report_replanned.json").exists()
