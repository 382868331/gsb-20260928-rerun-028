from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import angle_distance, normalize_angle
from .plan import Motion, Pose
from .scene import Scene
from .vehicle import Vehicle


@dataclass(frozen=True)
class Primitive:
    direction: int
    steering: float
    distance: float

    def as_motion(self) -> Motion:
        return Motion(self.direction, self.steering, self.distance)


def integrate(
    pose: Pose,
    direction: int,
    steering: float,
    distance: float,
    wheelbase: float,
) -> Pose:
    x, y, theta = pose
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    curvature = math.tan(steering) / wheelbase
    theta_end = theta + direction * curvature * distance
    if abs(curvature) < 1e-12:
        return (
            x + direction * distance * math.cos(theta),
            y + direction * distance * math.sin(theta),
            normalize_angle(theta_end),
        )
    return (
        x + (math.sin(theta_end) - math.sin(theta)) / curvature,
        y + (-math.cos(theta_end) + math.cos(theta)) / curvature,
        normalize_angle(theta_end),
    )


def sample_arc(
    pose: Pose,
    direction: int,
    steering: float,
    distance: float,
    wheelbase: float,
    step: float = 0.08,
) -> list[Pose]:
    count = max(1, int(math.ceil(distance / step)))
    actual_step = distance / count
    return [
        integrate(pose, direction, steering, actual_step * index, wheelbase)
        for index in range(count + 1)
    ]


def motion_is_collision_free(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    motion: Motion | Primitive,
    collision_step: float = 0.08,
) -> bool:
    samples = sample_arc(
        pose,
        motion.direction,
        motion.steering,
        motion.distance,
        vehicle.wheelbase,
        collision_step,
    )
    for state in samples:
        footprint = vehicle.footprint(*state)
        if scene.footprint_collides(footprint):
            return False
    return True


def standard_primitives(
    vehicle: Vehicle,
    lengths: tuple[float, ...] = (0.75,),
    steering_steps: int = 2,
) -> tuple[Primitive, ...]:
    values = [0.0]
    for index in range(1, steering_steps + 1):
        fraction = index / steering_steps
        values.extend((fraction * vehicle.max_steering_angle, -fraction * vehicle.max_steering_angle))
    return tuple(
        Primitive(direction=direction, steering=steering, distance=distance)
        for direction in (1, -1)
        for steering in values
        for distance in lengths
    )


def endpoint_pose(vehicle: Vehicle, pose: Pose, primitive: Primitive) -> Pose:
    return integrate(
        pose,
        primitive.direction,
        primitive.steering,
        primitive.distance,
        vehicle.wheelbase,
    )


def goal_during_motion(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    primitive: Primitive,
) -> tuple[Pose, float] | None:
    samples = sample_arc(
        pose,
        primitive.direction,
        primitive.steering,
        primitive.distance,
        vehicle.wheelbase,
    )
    for index, state in enumerate(samples[1:], start=1):
        footprint = vehicle.footprint(*state)
        if scene.footprint_collides(footprint):
            return None
        if scene.at_goal(state):
            return state, primitive.distance * index / len(samples[1:])
    return None


def heuristic(pose: Pose, scene: Scene) -> float:
    x, y, theta = pose
    gx, gy, gtheta = scene.goal
    position = math.hypot(x - gx, y - gy)
    angular = max(0.0, angle_distance(theta, gtheta) - scene.goal_theta_tolerance)
    return position + 0.65 * angular
