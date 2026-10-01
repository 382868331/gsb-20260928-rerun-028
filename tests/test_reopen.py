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
    analyze_departure_windows,
    attach_historical_windows,
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


def test_changed_timetable_route_faces_new_schedule_on_its_own_merits(base, cheap_replan):
    vehicle, data, plan = base
    for frames in (EARLY_LEAVE, AVOID):
        scene = changed_scene(data, frames)
        # The signature-bound old plan is not reusable as-is ...
        assert replay(plan, vehicle, scene).valid is False
        assessment = evaluate_reopen(plan, vehicle, scene, max_delay=0.0)
        assert assessment.timetable_changed is True
        # ... but the original route, judged independently on the new timetable
        # with zero conflicts, must get the same verdict as a zero-second
        # "delay" (whose timing is unchanged).
        assert assessment.conflicts == ()
        assert assessment.original_feasible is True
        assert assessment.option("original").timing_changed is False
        assert assessment.option("delayed").feasible is True
        assert assessment.option("delayed").departure_delay == 0.0
        assert assessment.recommendation == "original"
        # The saved copy is rebound to the new timetable and replays on it.
        rebound = assessment.option_plans["original"]
        assert rebound.scene_signature == scene.signature()
        assert same_route(rebound, plan) and same_timing(rebound, plan)
        assert replay(rebound, vehicle, scene).valid
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


def test_departure_windows_classify_safe_blocked_and_boundaries(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
    )
    assert analysis.probes_requested == 17
    assert analysis.budget_exhausted is False
    labels = {item.label: item for item in analysis.boundaries}
    assert labels["earliest"].departure_time == 0.0
    assert labels["latest"].departure_time == 8.0
    kinds = [window.kind for window in analysis.windows]
    assert "safe" in kinds and "blocked" in kinds and "unverified" not in kinds
    blocked = [window for window in analysis.windows if window.kind == "blocked"]
    assert all(window.obstacles for window in blocked)
    # A time selected from a safe window replays the whole route successfully.
    assert analysis.selected_departure is not None
    delayed_plan = find_delayed_plan(
        plan, vehicle, scene, max_delay=8, delay_step=0.5
    )[1]
    assert replay(delayed_plan, vehicle, scene).valid
    assert delayed_plan.actions[0].type == "wait" or analysis.selected_departure == 0.0


def test_departure_window_budget_leaves_unverified_tail_not_safe(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=10.0, step=0.5, max_probes=4,
    )
    assert analysis.budget_exhausted is True
    assert analysis.probes_completed == 4
    tail = [w for w in analysis.windows if w.kind == "unverified"]
    assert len(tail) == 1
    assert tail[0].end_time == 10.0


def test_departure_windows_all_safe_when_obstacle_moves_away(base):
    vehicle, data, plan = base
    scene = changed_scene(data, AVOID)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=5.0, step=0.5,
    )
    assert all(w.kind == "safe" for w in analysis.windows)
    assert all(b.feasible for b in analysis.boundaries)
    assert analysis.selected_departure == 0.0


def test_historical_report_is_tagged_never_adopted(base, tmp_path):
    vehicle, data, plan = base
    scene_late = changed_scene(data, LATE)
    analysis = analyze_departure_windows(
        plan, vehicle, scene_late,
        earliest_departure=0.0, latest_departure=6.0, step=0.5,
    )
    same = attach_historical_windows(
        analysis,
        {"departure_windows": {"schedule_signature": scene_late.signature()}},
    )
    assert same.historical_status == "same_timetable"
    other_scene = changed_scene(data, EARLY)
    stale = attach_historical_windows(
        analysis,
        {"departure_windows": {"schedule_signature": other_scene.signature()}},
    )
    assert stale.historical_status == "needs_recheck"
    # Current safe/blocked verdicts come from fresh probes, never from history.
    assert all(w.kind != "unverified" for w in stale.windows)


def test_reopen_includes_selected_window_plan_and_preserves_history(base, cheap_replan, tmp_path):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_reopen(
        plan, vehicle, scene, max_delay=10, delay_step=0.5,
        earliest_departure=0.0, latest_departure=8.0, departure_step=0.5,
    )
    windows = assessment.departure_windows
    selected = windows.selected_departure
    assert selected is not None
    selected_plan = assessment.option_plans["selected_window"]
    assert selected_plan.scene_signature == scene.signature()
    assert replay(selected_plan, vehicle, scene).valid

    report_path = tmp_path / "report.json"
    save_reopen_bundle(assessment, None, report_path)
    assert (tmp_path / "report_selected_window.json").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["departure_windows"]["selected_verified"] is True

    # Reopen again on a changed arrangement, handing in the previous report:
    # its windows must be flagged as historical / needing recheck.
    scene2 = changed_scene(data, LATE)
    assessment2 = evaluate_reopen(
        plan, vehicle, scene2, max_delay=10, delay_step=0.5,
        earliest_departure=0.0, latest_departure=8.0, departure_step=0.5,
        historical_report=payload,
    )
    assert assessment2.departure_windows.historical_status == "needs_recheck"

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
