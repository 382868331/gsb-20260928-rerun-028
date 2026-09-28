from __future__ import annotations

import math

from .geometry import obstacle_vertices, rectangles_intersect
from .kinematics import integrate, vehicle_corners
from .models import MotionCommand, Pose, Scenario, Vehicle


def pose_in_bounds(pose: Pose, vehicle: Vehicle, scenario: Scenario) -> bool:
    if scenario.bounds is None:
        return True
    min_x, min_y, max_x, max_y = scenario.bounds
    return all(
        min_x <= x <= max_x and min_y <= y <= max_y
        for x, y in vehicle_corners(pose, vehicle)
    )


def pose_collision_free(pose: Pose, vehicle: Vehicle, scenario: Scenario) -> bool:
    if not pose_in_bounds(pose, vehicle, scenario):
        return False
    body = vehicle_corners(pose, vehicle)
    return not any(
        rectangles_intersect(body, obstacle_vertices(obstacle))
        for obstacle in scenario.obstacles
    )


def command_collision_free(
    start: Pose,
    command: MotionCommand,
    vehicle: Vehicle,
    scenario: Scenario,
    sample_step: float = 0.05,
) -> bool:
    length = abs(command.velocity) * command.duration
    count = max(1, math.ceil(length / sample_step))
    dt = command.duration / count
    current = start
    sampled = MotionCommand(command.steering_angle, command.velocity, dt)
    for _ in range(count + 1):
        if not pose_collision_free(current, vehicle, scenario):
            return False
        current = integrate(current, sampled, vehicle)
    return True
