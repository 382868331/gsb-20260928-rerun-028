import json
from dataclasses import replace
from pathlib import Path

import pytest

from vehicle_planning.plan import load_plan, save_plan
from vehicle_planning.replay import replay
from vehicle_planning.scene import load_scene, scene_from_dict
from vehicle_planning.scheduler import Scheduler, SchedulerConfig, SchedulerStatus
from vehicle_planning.vehicle import load_vehicle


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def assets():
    return (
        load_vehicle(ROOT / "data/vehicles/compact_car.json"),
        load_scene(ROOT / "data/scenes/pedestrian_crossing.json"),
        load_plan(ROOT / "outputs/reverse_into_bay.json"),
    )


def timed_plan(assets, config=None):
    vehicle, scene, geometric = assets
    geometric = replace(geometric, scene=scene.name)
    return vehicle, scene, Scheduler(vehicle, scene, config or SchedulerConfig()).schedule(geometric)


def test_dynamic_obstacle_interpolates_between_keyframes(assets):
    _, scene, _ = assets
    obstacle = scene.dynamic_obstacles[0]
    start = obstacle.keyframes[0]
    end = obstacle.keyframes[1]
    mid_time = (start[0] + end[0]) / 2.0
    x, y, _ = obstacle.pose_at(mid_time)
    assert x == pytest.approx(start[1])
    assert y == pytest.approx((start[2] + end[2]) / 2.0)
    last = obstacle.keyframes[-1]
    held = obstacle.pose_at(last[0] + 100.0)
    assert held[:2] == last[1:3]


def test_plan_records_wait_reason_gear_transitions_and_timed_actions(assets):
    vehicle, scene, plan = timed_plan(assets)
    assert plan.status == SchedulerStatus.PATH_FOUND
    types = [action.type for action in plan.actions]
    assert "wait" in types
    assert "gear_switch" in types
    wait = next(action for action in plan.actions if action.type == "wait")
    assert wait.end_time > wait.start_time
    assert "trolley" in wait.reason and "等待" in wait.reason
    switches = [action for action in plan.actions if action.type == "gear_switch"]
    assert all(action.end_time - action.start_time == pytest.approx(1.5) for action in switches)
    # Every forward/reverse change in the geometric plan becomes a transition.
    directions = [motion.direction for motion in plan.motions]
    changes = sum(1 for a, b in zip(directions, directions[1:]) if a != b)
    assert (
        sum(1 for action in switches if (action.start_time, action.direction))
        == changes + (1 if directions[0] == -1 else 0)
    )


def test_independent_timed_replay_checks_vehicle_and_obstacle_every_moment(assets):
    vehicle, scene, plan = timed_plan(assets)
    report = replay(plan, vehicle, scene)
    assert report.valid, report.message
    assert report.final_time == pytest.approx(plan.final_time)
    assert report.samples > len(plan.motions) * 5


def test_changed_timetable_invalidates_previously_successful_plan(assets):
    vehicle, scene, plan = timed_plan(assets)
    assert replay(plan, vehicle, scene).valid

    data = json.loads((ROOT / "data/scenes/pedestrian_crossing.json").read_text(encoding="utf-8"))
    data["dynamic_obstacles"][0]["keyframes"][1][0] = 3.0
    changed_scene = scene_from_dict(data)
    report = replay(plan, vehicle, changed_scene)
    assert not report.valid
    assert "timetable" in report.message or "re-planned" in report.message


def test_untimed_plan_is_not_reused_when_scene_has_scheduled_obstacles(assets):
    vehicle, scene, geometric = assets
    geometric = replace(geometric, scene=scene.name)
    report = replay(geometric, vehicle, scene)
    assert not report.valid
    assert "scheduled obstacles" in report.message


def test_impossible_wait_returns_schedule_failed_status(assets):
    vehicle, scene, geometric = assets
    geometric = replace(geometric, scene=scene.name)
    # Blocking forever: obstacle holds a pose that overlaps the whole approach.
    data = json.loads((ROOT / "data/scenes/pedestrian_crossing.json").read_text(encoding="utf-8"))
    data["dynamic_obstacles"][0]["keyframes"] = [
        [0.0, 3.0, 3.0, 0.0],
        [1.0, 3.0, 3.0, 0.0],
        [99.0, 3.0, 3.0, 0.0],
    ]
    blocked_scene = scene_from_dict(data)
    plan = Scheduler(
        vehicle,
        blocked_scene,
        SchedulerConfig(max_wait=2.0, wait_step=0.5),
    ).schedule(geometric)
    assert plan.status == SchedulerStatus.SCHEDULE_FAILED


def test_roundtrip_timed_plan_file(assets, tmp_path):
    vehicle, scene, plan = timed_plan(assets)
    path = tmp_path / "plan.json"
    save_plan(plan, path)
    reloaded = load_plan(path)
    assert reloaded.scene_signature == scene.signature()
    assert [action.type for action in reloaded.actions] == [
        action.type for action in plan.actions
    ]
    assert replay(reloaded, vehicle, scene).valid
