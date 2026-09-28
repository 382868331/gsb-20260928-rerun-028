from __future__ import annotations

import math
from dataclasses import dataclass

from .collision import command_collision_free
from .geometry import angle_distance
from .kinematics import trace_path
from .models import Plan, Scenario, Vehicle


@dataclass
class ReplayReport:
    valid: bool
    final_position_error: float
    final_heading_error: float
    collision_free: bool
    pose_count: int
    command_count: int
    reason: str = ""


def replay_plan(
    plan: Plan,
    vehicle: Vehicle,
    scenario: Scenario,
    sample_step: float = 0.05,
) -> ReplayReport:
    if plan.vehicle_name != vehicle.name or plan.scenario_name != scenario.name:
        return ReplayReport(False, math.inf, math.inf, False, 0, len(plan.commands), "identity mismatch")
    if plan.start != scenario.start:
        return ReplayReport(False, math.inf, math.inf, False, 0, len(plan.commands), "start mismatch")

    collision_free = True
    current = scenario.start
    for command in plan.commands:
        if abs(command.steering_angle) > vehicle.max_steering_angle + 1e-9:
            return ReplayReport(False, math.inf, math.inf, False, 0, len(plan.commands), "steering limit violated")
        if not command_collision_free(current, command, vehicle, scenario, sample_step):
            collision_free = False
            break
        from .kinematics import integrate

        current = integrate(current, command, vehicle)

    poses = trace_path(scenario.start, plan.commands, vehicle, sample_step)
    final_pose = poses[-1]
    position_error = math.hypot(final_pose.x - scenario.goal.x, final_pose.y - scenario.goal.y)
    heading_error = angle_distance(final_pose.theta, scenario.goal.theta)
    reached_goal = (
        position_error <= scenario.goal_tolerance.position
        and heading_error <= scenario.goal_tolerance.heading
    )
    valid = collision_free and reached_goal
    reason = "ok" if valid else "collision" if not collision_free else "goal pose not reached"
    return ReplayReport(
        valid=valid,
        final_position_error=position_error,
        final_heading_error=heading_error,
        collision_free=collision_free,
        pose_count=len(poses),
        command_count=len(plan.commands),
        reason=reason,
    )


def trajectory_json(plan: Plan, vehicle: Vehicle, sample_step: float = 0.05) -> list[dict]:
    if plan.start is None:
        raise ValueError("plan does not contain a start pose")
    poses = trace_path(plan.start, plan.commands, vehicle, sample_step)
    return [
        {"x": pose.x, "y": pose.y, "theta": pose.theta}
        for pose in poses
    ]
