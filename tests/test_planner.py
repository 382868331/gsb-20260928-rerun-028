import math
import unittest

from vehicle_planner.collision import pose_collision_free
from vehicle_planner.models import (
    GoalTolerance,
    MotionCommand,
    Pose,
    RectangleObstacle,
    Scenario,
    Vehicle,
)
from vehicle_planner.planner import MotionPlanner, SearchLimits
from vehicle_planner.replay import replay_plan


VEHICLE = Vehicle(
    name="test_car",
    wheelbase=2.6,
    length=4.2,
    width=1.8,
    rear_overhang=0.8,
    max_steering_angle=0.83,
)


class PlannerTests(unittest.TestCase):
    def test_forward_goal_requires_heading(self):
        scenario = Scenario(
            name="straight",
            start=Pose(0, 0, 0),
            goal=Pose(1.6, 0, 0),
            obstacles=(),
            goal_tolerance=GoalTolerance(0.1, 0.1),
        )
        result = MotionPlanner(VEHICLE, heading_bins=24).plan(scenario)
        self.assertEqual(result.status, "solved")
        self.assertLess(abs(result.goal_pose.theta), 0.1)

    def test_wrong_heading_is_not_accepted(self):
        scenario = Scenario(
            name="wrong_heading",
            start=Pose(0, 0, 0),
            goal=Pose(0.8, 0, math.pi),
            obstacles=(),
            goal_tolerance=GoalTolerance(0.2, 0.1),
            bounds=(-1.5, -1.3, 1.5, 1.3),
        )
        result = MotionPlanner(VEHICLE, heading_bins=24).plan(scenario)
        self.assertNotEqual(result.status, "solved")

    def test_full_body_collision_blocks_goal(self):
        scenario = Scenario(
            name="body_block",
            start=Pose(0, 0, 0),
            goal=Pose(2, 0, 0),
            obstacles=(
                RectangleObstacle("barrier", -0.1, -2.0, 3.0, 2.0),
            ),
            goal_tolerance=GoalTolerance(0.1, 0.1),
        )
        self.assertFalse(pose_collision_free(scenario.start, VEHICLE, scenario))
        result = MotionPlanner(VEHICLE).plan(scenario)
        self.assertEqual(result.status, "no_path")

    def test_reverse_maneuver_replays_with_collision_checks(self):
        scenario = Scenario(
            name="reverse",
            start=Pose(0, 0, math.pi / 2),
            goal=Pose(0, -1.6, math.pi / 2),
            obstacles=(
                RectangleObstacle("left", -3, -3.5, -1.1, 0.5),
                RectangleObstacle("right", 1.1, -3.5, 3, 0.5),
            ),
            goal_tolerance=GoalTolerance(0.1, 0.1),
        )
        planner = MotionPlanner(VEHICLE, heading_bins=24)
        plan = planner.make_plan(scenario)
        self.assertEqual(plan.status, "solved")
        self.assertTrue(any(command.velocity < 0 for command in plan.commands))
        report = replay_plan(plan, VEHICLE, scenario)
        self.assertTrue(report.valid, report)

    def test_expansion_limit_is_distinct_from_no_path(self):
        scenario = Scenario(
            name="open",
            start=Pose(0, 0, 0),
            goal=Pose(5, 5, math.pi),
            obstacles=(),
            goal_tolerance=GoalTolerance(0.1, 0.1),
        )
        planner = MotionPlanner(VEHICLE, limits=SearchLimits(max_expansions=1))
        result = planner.plan(scenario)
        self.assertEqual(result.status, "search_incomplete")

    def test_arc_command_is_not_pretended_to_be_straight(self):
        command = MotionCommand(VEHICLE.max_steering_angle, 1.0, 1.0)
        from vehicle_planner.kinematics import integrate

        pose = integrate(Pose(0, 0, 0), command, VEHICLE)
        self.assertGreater(abs(pose.y), 0.1)
        self.assertGreater(abs(pose.theta), 0.1)


if __name__ == "__main__":
    unittest.main()
