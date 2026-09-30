import math
from pathlib import Path

from vehicle_planning.geometry import angle_distance
from vehicle_planning.motion import integrate, standard_primitives
from vehicle_planning.plan import Motion, load_plan
from vehicle_planning.planner import Planner, PlannerConfig, PlannerStatus
from vehicle_planning.replay import replay
from vehicle_planning.scene import Scene, load_scene
from vehicle_planning.vehicle import Vehicle, load_vehicle


ROOT = Path(__file__).resolve().parents[1]


def vehicle() -> Vehicle:
    return Vehicle(
        name="test_car",
        length=4.0,
        width=1.8,
        wheelbase=2.5,
        front_overhang=0.8,
        rear_overhang=0.7,
        max_steering_angle=0.5,
    )


def test_straight_reverse_moves_backward_without_changing_heading():
    pose = (1.0, 2.0, 0.3)
    result = integrate(pose, -1, 0.0, 1.0, 2.5)
    assert result[0] == pose[0] - math.cos(0.3)
    assert result[1] == pose[1] - math.sin(0.3)
    assert angle_distance(result[2], 0.3) < 1e-12


def test_full_footprint_collision_blocks_even_when_rear_axle_is_free():
    car = vehicle()
    scene = Scene(
        name="blocked",
        bounds=(0.0, 0.0, 10.0, 10.0),
        obstacles=(((4.1, -1.0), (5.0, -1.0), (5.0, 11.0), (4.1, 11.0)),),
        start=(2.0, 2.0, 0.0),
        goal=(3.0, 2.0, 0.0),
        goal_xy_tolerance=0.1,
        goal_theta_tolerance=0.1,
    )
    assert not scene.point_collides(4.0, 2.0)
    assert scene.footprint_collides(car.footprint(4.0, 2.0, 0.0))


def test_no_path_and_search_limit_are_distinct_statuses():
    car = vehicle()
    blocked_goal = Scene(
        name="blocked_goal",
        bounds=(0.0, 0.0, 20.0, 20.0),
        obstacles=(((1.0, 1.0), (1.4, 1.0), (1.4, 1.4), (1.0, 1.4)),),
        start=(2.0, 2.0, 0.0),
        goal=(1.2, 1.2, 0.0),
        goal_xy_tolerance=0.05,
        goal_theta_tolerance=0.05,
    )
    planner = Planner(car, blocked_goal, standard_primitives(car), PlannerConfig())
    assert planner.plan().status == PlannerStatus.NO_PATH

    open_scene = Scene(
        name="open",
        bounds=(-20.0, -20.0, 20.0, 20.0),
        obstacles=(),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, math.pi),
        goal_xy_tolerance=0.1,
        goal_theta_tolerance=0.1,
    )
    limited = Planner(
        car,
        open_scene,
        standard_primitives(car),
        PlannerConfig(max_expanded_nodes=5),
    )
    assert limited.plan().status == PlannerStatus.SEARCH_LIMIT_REACHED


def test_generated_examples_replay_collision_free_and_at_orientation_goal():
    car = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    for scene_name in ("narrow_uturn", "reverse_into_bay"):
        scene = load_scene(ROOT / f"data/scenes/{scene_name}.json")
        plan = load_plan(ROOT / f"outputs/{scene_name}.json")
        report = replay(plan, car, scene)
        assert report.valid, report.message
        assert report.goal_xy_error <= scene.goal_xy_tolerance
        assert report.goal_theta_error <= scene.goal_theta_tolerance
        assert plan.motions
        assert any(motion.direction == -1 for motion in plan.motions)


def test_saved_motion_must_reproduce_saved_state():
    car = vehicle()
    scene = Scene(
        name="tampered",
        bounds=(0.0, 0.0, 10.0, 10.0),
        obstacles=(),
        start=(0.0, 0.0, 0.0),
        goal=(1.0, 0.0, 0.0),
        goal_xy_tolerance=0.2,
        goal_theta_tolerance=0.2,
    )
    from vehicle_planning.plan import PlanResult

    plan = PlanResult(
        status=PlannerStatus.PATH_FOUND,
        vehicle=car.name,
        scene=scene.name,
        start_pose=scene.start,
        goal_pose=scene.goal,
        states=((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
        motions=(Motion(1, 0.0, 1.0),),
        nodes_expanded=1,
        max_nodes=10,
        message="",
    )
    assert not replay(plan, car, scene).valid


import dataclasses

from vehicle_planning.scene import DynamicObstacle, KeyFrame, load_scene
from vehicle_planning.plan import load_plan
from vehicle_planning.motion import (
    earliest_clear_departure,
    stationary_collision_free,
    sweep_collision_free,
)
from vehicle_planning.planner import Planner, PlannerConfig, PlannerStatus
from vehicle_planning.replay import replay_timed


def _crossing_cart() -> DynamicObstacle:
    return DynamicObstacle(
        name="crossing_cart",
        footprint=((-1.0, -0.6), (1.0, -0.6), (1.0, 0.6), (-1.0, 0.6)),
        keyframes=(KeyFrame(0.0, 8.0, 1.0, 0.0), KeyFrame(6.0, 8.0, 7.0, 0.0)),
    )


def test_dynamic_obstacle_interpolates_position_and_vanishes_outside_window():
    cart = _crossing_cart()
    assert cart.pose_at(0.0) == (8.0, 1.0, 0.0)
    mid = cart.pose_at(3.0)
    assert mid is not None and mid[0] == 8.0 and abs(mid[1] - 4.0) < 1e-9
    assert cart.pose_at(-0.5) is None and cart.pose_at(6.5) is None


def test_timed_plan_waits_then_drives_and_records_reason():
    car = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    scene = load_scene(ROOT / "data/scenes/timed_crossing.json")
    plan = load_plan(ROOT / "outputs/timed_crossing.json")
    assert plan.found and plan.waits
    wait = plan.waits[0]
    assert wait.t1 > wait.t0
    assert "crossing_cart" in wait.reason
    report = replay_timed(plan, car, scene)
    assert report.valid and not report.timetable_changed
    drive = next(action for action in plan.timeline if action.kind == "drive" if action.t0 >= wait.t1)
    assert drive.t0 >= wait.t1 - 1e-9


def test_sweep_check_uses_each_timestamp_not_the_final_obstacle_pose():
    car = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    scene = load_scene(ROOT / "data/scenes/timed_crossing.json")
    from vehicle_planning.plan import Motion

    # Long straight crossing of x in [2, 9]; mid-travel the cart crosses the lane.
    motion = Motion(1, 0.0, 7.0)
    assert sweep_collision_free(car, scene, (2.0, 3.5, 0.0), motion, 0.0, 1.0) is False
    departure, blockers = earliest_clear_departure(
        car, scene, (2.0, 3.5, 0.0), motion, 0.0, speed=1.0
    )
    assert departure is not None and departure >= 2.0
    assert sweep_collision_free(car, scene, (2.0, 3.5, 0.0), motion, departure, 1.0)
    assert "crossing_cart" in blockers


def test_gear_shift_is_a_stationary_timed_transition():
    car = vehicle()
    base_cart = _crossing_cart()
    cart = dataclasses.replace(
        base_cart,
        keyframes=tuple(
            dataclasses.replace(frame, x=16.0) for frame in base_cart.keyframes
        ),
    )
    scene = Scene(
        name="shift_scene",
        bounds=(0.0, 0.0, 20.0, 20.0),
        obstacles=(),
        start=(4.0, 3.5, 0.0),
        goal=(2.0, 3.5, 0.0),
        goal_xy_tolerance=0.3,
        goal_theta_tolerance=0.3,
        dynamic_obstacles=(cart,),
    )
    config = PlannerConfig(
        xy_resolution=0.4,
        theta_resolution=0.3,
        max_expanded_nodes=20000,
        gear_shift_time=2.0,
        max_makespan=60.0,
    )
    plan = Planner(car, scene, standard_primitives(car, lengths=(1.0,)), config).plan()
    assert plan.status == PlannerStatus.PATH_FOUND
    shifts = [action for action in plan.timeline if action.kind == "shift"]
    assert shifts and all(action.duration == 2.0 for action in shifts)
    assert any(action.direction == -1 for action in shifts)
    report = replay_timed(plan, car, scene)
    assert report.valid, report.message


def test_changed_timetable_forces_revalidation_even_if_plan_was_valid():
    car = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    scene = load_scene(ROOT / "data/scenes/timed_crossing.json")
    plan = load_plan(ROOT / "outputs/timed_crossing.json")

    cart = scene.dynamic_obstacles[0]
    shifted = dataclasses.replace(
        cart,
        keyframes=tuple(dataclasses.replace(frame, time=frame.time + 8.0) for frame in cart.keyframes),
    )
    changed_scene = dataclasses.replace(scene, dynamic_obstacles=(shifted,))
    assert changed_scene.timetable_signature() != plan.timetable_signature

    report = replay_timed(plan, car, changed_scene)
    assert report.timetable_changed is True
    assert report.valid is False
    assert "crossing_cart" in report.message

    safe = dataclasses.replace(
        cart,
        keyframes=tuple(dataclasses.replace(frame, time=frame.time - 2.0) for frame in cart.keyframes),
    )
    safe_scene = dataclasses.replace(scene, dynamic_obstacles=(safe,))
    safe_report = replay_timed(plan, car, safe_scene)
    assert safe_report.valid is True and safe_report.timetable_changed is True


def test_stationary_wait_inside_obstacle_path_is_rejected():
    car = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    scene = load_scene(ROOT / "data/scenes/timed_crossing.json")
    # At (8, 3.5) the cart sweeps directly over the parked ego footprint.
    assert not stationary_collision_free(car, scene, (8.0, 3.5, 0.0), 2.0, 4.0)
    assert stationary_collision_free(car, scene, (2.0, 3.5, 0.0), 0.0, 1.0)


def test_tampered_timed_action_is_rejected_despite_success_status():
    car = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    scene = load_scene(ROOT / "data/scenes/timed_crossing.json")
    plan = load_plan(ROOT / "outputs/timed_crossing.json")
    from vehicle_planning.plan import TimedAction

    tampered = dataclasses.replace(
        plan,
        timeline=tuple(
            dataclasses.replace(action, t1=action.t1 + 5.0, t0=action.t0 + 5.0)
            if index == 0
            else action
            for index, action in enumerate(plan.timeline)
        ),
    )
    report = replay_timed(tampered, car, scene)
    assert not report.valid
