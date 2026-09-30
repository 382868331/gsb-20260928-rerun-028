import json
from dataclasses import replace
from pathlib import Path

import pytest

from vehicle_planning.motion import standard_primitives
from vehicle_planning.plan import load_plan, save_plan
from vehicle_planning.planner import PlannerConfig
from vehicle_planning.replan import (
    compare_alternatives,
    evaluate_original_schedule,
    timetable_changed,
    try_delayed_schedule,
)
from vehicle_planning.replay import replay
from vehicle_planning.scene import load_scene, scene_from_dict
from vehicle_planning.scheduler import Scheduler, SchedulerConfig
from vehicle_planning.vehicle import load_vehicle


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def bundle():
    vehicle = load_vehicle(ROOT / 'data/vehicles/compact_car.json')
    scene = load_scene(ROOT / 'data/scenes/pedestrian_crossing.json')
    geometric = replace(
        load_plan(ROOT / 'outputs/reverse_into_bay.json'), scene=scene.name
    )
    old_plan = Scheduler(
        vehicle, scene, SchedulerConfig(variant='original')
    ).schedule(geometric)
    return vehicle, scene, geometric, old_plan


def _scene_with_keyframes(keyframes):
    data = json.loads(
        (ROOT / 'data/scenes/pedestrian_crossing.json').read_text(encoding='utf-8')
    )
    data['dynamic_obstacles'][0]['keyframes'] = keyframes
    return data, scene_from_dict(data)


def _primitives(vehicle):
    return standard_primitives(vehicle, lengths=(0.4, 0.75), steering_steps=2)


def test_timetable_change_always_requires_replanning_even_when_obstacle_moves_away(bundle):
    vehicle, scene, _, old_plan = bundle
    _, far_scene = _scene_with_keyframes(
        [
            [0.0, 20.0, 20.0, 0.0],
            [4.5, 20.0, 20.0, 0.0],
            [7.5, 25.0, 25.0, 0.0],
            [16.0, 30.0, 30.0, 0.0],
        ]
    )

    assert timetable_changed(old_plan, far_scene)
    report = replay(old_plan, vehicle, far_scene)
    assert not report.valid
    assert 'timetable' in report.message or 're-planned' in report.message

    evaluation = evaluate_original_schedule(old_plan, vehicle, far_scene)
    assert evaluation.feasible
    assert evaluation.conflicts == ()
    assert evaluation.route_changed is False
    assert evaluation.timing_changed is False
    assert old_plan.timetable_signature == scene.timetable_signature()


def test_failed_evaluation_reports_conflict_intervals_and_keeps_success_record(bundle):
    vehicle, scene, _, old_plan = bundle
    _, late_scene = _scene_with_keyframes(
        [
            [0.0, 11.0, 4.5, 0.0],
            [9.0, 11.0, 4.5, 0.0],
            [14.0, -1.5, 4.5, 0.0],
            [22.0, -8.0, 4.5, 0.0],
        ]
    )
    evaluation = evaluate_original_schedule(old_plan, vehicle, late_scene)
    assert not evaluation.feasible
    assert evaluation.conflicts
    first = evaluation.conflicts[0]
    assert first.end_time > first.start_time
    assert first.obstacle == 'trolley'
    assert first.actions

    assert old_plan.status == 'path_found'
    assert old_plan.timetable_signature == scene.timetable_signature()
    assert replay(old_plan, vehicle, scene).valid


def test_delayed_alternative_changes_only_timing_not_route(bundle):
    vehicle, _, _, old_plan = bundle
    _, late_scene = _scene_with_keyframes(
        [
            [0.0, 11.0, 4.5, 0.0],
            [9.0, 11.0, 4.5, 0.0],
            [14.0, -1.5, 4.5, 0.0],
            [22.0, -8.0, 4.5, 0.0],
        ]
    )
    delayed = try_delayed_schedule(
        old_plan, vehicle, late_scene, SchedulerConfig(max_wait=30, wait_step=0.2)
    )
    assert delayed.feasible
    assert delayed.route_changed is False
    assert delayed.timing_changed is True
    assert delayed.plan.route_signature == old_plan.route_signature
    assert delayed.plan.timetable_signature == late_scene.timetable_signature()
    assert delayed.plan.variant == 'delayed'
    assert replay(delayed.plan, vehicle, late_scene).valid


def test_early_arrival_is_covered_and_compared_read_only(bundle):
    vehicle, scene, _, old_plan = bundle
    _, early_scene = _scene_with_keyframes(
        [
            [0.0, 11.0, 4.5, 0.0],
            [1.0, 11.0, 4.5, 0.0],
            [4.0, -1.5, 4.5, 0.0],
            [12.0, -8.0, 4.5, 0.0],
        ]
    )
    comparison = compare_alternatives(
        old_plan,
        vehicle,
        early_scene,
        _primitives(vehicle),
        scheduler_config=SchedulerConfig(max_wait=30, wait_step=0.2),
        allow_replan=False,
    )
    assert comparison.timetable_changed
    assert comparison.delayed is not None
    assert comparison.recommended in {'original', 'delayed'}
    assert comparison.original.plan.timetable_signature == scene.timetable_signature()


def test_replanned_alternative_marks_route_change_and_signature(bundle):
    vehicle, _, _, old_plan = bundle
    _, blocked_scene = _scene_with_keyframes(
        [
            [0.0, 7.0, 6.6, 1.5707963267948966],
            [1.0, 7.0, 6.6, 1.5707963267948966],
            [99.0, 7.0, 6.6, 1.5707963267948966],
        ]
    )
    comparison = compare_alternatives(
        old_plan,
        vehicle,
        blocked_scene,
        _primitives(vehicle),
        scheduler_config=SchedulerConfig(max_wait=1.0, wait_step=0.5),
        planner_config=PlannerConfig(max_expanded_nodes=4000),
    )
    assert comparison.delayed.feasible is False
    assert comparison.delayed.route_changed is False
    if comparison.replanned is not None and comparison.replanned.feasible:
        plan = comparison.replanned.plan
        assert comparison.replanned.route_changed is True
        assert plan.variant == 'replanned'
        assert plan.timetable_signature == blocked_scene.timetable_signature()
        assert replay(plan, vehicle, blocked_scene).valid


def test_each_saved_variant_keeps_its_own_timetable_after_reopen(bundle, tmp_path):
    vehicle, scene, _, old_plan = bundle
    _, late_scene = _scene_with_keyframes(
        [
            [0.0, 11.0, 4.5, 0.0],
            [9.0, 11.0, 4.5, 0.0],
            [14.0, -1.5, 4.5, 0.0],
            [22.0, -8.0, 4.5, 0.0],
        ]
    )
    delayed = try_delayed_schedule(
        old_plan, vehicle, late_scene, SchedulerConfig(max_wait=30, wait_step=0.2)
    ).plan

    old_path = tmp_path / 'original.json'
    delayed_path = tmp_path / 'delayed.json'
    save_plan(old_plan, old_path)
    save_plan(delayed, delayed_path)

    reloaded_old = load_plan(old_path)
    reloaded_delayed = load_plan(delayed_path)
    assert reloaded_old.timetable_signature == scene.timetable_signature()
    assert reloaded_delayed.timetable_signature == late_scene.timetable_signature()
    assert reloaded_old.timetable_signature != reloaded_delayed.timetable_signature
    assert reloaded_old.route_signature == reloaded_delayed.route_signature
    assert replay(reloaded_old, vehicle, scene).valid
    assert not replay(reloaded_old, vehicle, late_scene).valid
    assert replay(reloaded_delayed, vehicle, late_scene).valid
    assert not replay(reloaded_delayed, vehicle, scene).valid
