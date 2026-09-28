from __future__ import annotations

import math

from .geometry import rotate
from .models import MotionCommand, Pose, Vehicle


def vehicle_corners(pose: Pose, vehicle: Vehicle) -> list[tuple[float, float]]:
    front_overhang = vehicle.length - vehicle.wheelbase - vehicle.rear_overhang
    rear_x = -vehicle.rear_overhang
    front_x = vehicle.wheelbase + front_overhang
    half_width = vehicle.width / 2.0
    local = [
        (rear_x, -half_width),
        (front_x, -half_width),
        (front_x, half_width),
        (rear_x, half_width),
    ]
    return [
        (pose.x + rx, pose.y + ry)
        for rx, ry in (rotate(x, y, pose.theta) for x, y in local)
    ]


def integrate(pose: Pose, command: MotionCommand, vehicle: Vehicle, dt: float | None = None) -> Pose:
    steps = 1 if dt is None else max(1, math.ceil(command.duration / dt))
    step_dt = command.duration / steps
    x, y, theta = pose.x, pose.y, pose.theta
    curvature = math.tan(command.steering_angle) / vehicle.wheelbase

    for _ in range(steps):
        if abs(curvature) < 1e-12:
            x += command.velocity * math.cos(theta) * step_dt
            y += command.velocity * math.sin(theta) * step_dt
        else:
            theta_delta = command.velocity * curvature * step_dt
            x += (math.sin(theta + theta_delta) - math.sin(theta)) / curvature
            y += (math.cos(theta) - math.cos(theta + theta_delta)) / curvature
            theta += theta_delta

    return Pose(x, y, theta)


def trace_path(start: Pose, commands, vehicle: Vehicle, sample_step: float = 0.05) -> list[Pose]:
    poses = [start]
    current = start
    for command in commands:
        count = max(1, math.ceil(command.duration / sample_step))
        dt = command.duration / count
        for _ in range(count):
            current = integrate(
                current,
                MotionCommand(command.steering_angle, command.velocity, dt),
                vehicle,
            )
            poses.append(current)
    return poses
