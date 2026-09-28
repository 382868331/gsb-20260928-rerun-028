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
