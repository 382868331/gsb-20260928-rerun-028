from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import angle_distance, normalize_angle, polygons_intersect
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


def _footprint_collides_static(vehicle: Vehicle, scene: Scene, pose: Pose) -> bool:
    footprint = vehicle.footprint(*pose)
    if not all(scene.contains_point(x, y) for x, y in footprint):
        return True
    return any(polygons_intersect(footprint, obstacle) for obstacle in scene.obstacles)


def _arc_poses(
    vehicle: Vehicle,
    pose: Pose,
    motion: Motion | Primitive,
    collision_step: float,
) -> list[Pose]:
    return sample_arc(
        pose,
        motion.direction,
        motion.steering,
        motion.distance,
        vehicle.wheelbase,
        collision_step,
    )


def arc_static_collision(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    motion: Motion | Primitive,
    collision_step: float = 0.08,
) -> bool:
    """True when the swept arc hits a static obstacle or the workspace bounds."""
    for state in _arc_poses(vehicle, pose, motion, collision_step):
        if _footprint_collides_static(vehicle, scene, state):
            return True
    return False


def stationary_collision_free(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    t_start: float,
    t_end: float,
    time_step: float = 0.1,
) -> bool:
    """Holding ``pose`` over [t_start, t_end] never overlaps a moving obstacle."""
    footprint = vehicle.footprint(*pose)
    if scene._collides_static(footprint):
        return False
    steps = max(1, int(math.ceil(max(0.0, t_end - t_start) / time_step)))
    for index in range(steps + 1):
        time = t_start + (t_end - t_start) * index / steps
        for obstacle in scene.dynamic_obstacles:
            polygon = obstacle.polygon_at(time)
            if polygon is not None and polygons_intersect(footprint, polygon):
                return False
    return True


def sweep_collision_free(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    motion: Motion | Primitive,
    departure_time: float,
    speed: float,
    collision_step: float = 0.08,
    time_step: float = 0.1,
) -> bool:
    """Whole-vehicle sweep versus dynamic obstacles along a timed arc.

    Arc length maps linearly onto the clock via ``speed``; every spatial
    sample is checked against every obstacle active at its own timestamp,
    instead of stamping an obstacle's final position onto the map.
    """
    samples = _arc_poses(vehicle, pose, motion, collision_step)
    total = max(motion.distance, 1e-12)
    last_time = departure_time
    for index, state in enumerate(samples):
        time = departure_time + motion.distance * index / (len(samples) - 1) / speed
        footprint = vehicle.footprint(*state)
        for obstacle in scene.dynamic_obstacles:
            sub_steps = max(1, int(math.ceil(max(0.0, time - last_time) / time_step)))
            for sub in range(sub_steps + 1):
                sample_time = last_time + (time - last_time) * sub / sub_steps
                polygon = obstacle.polygon_at(sample_time)
                if polygon is not None and polygons_intersect(footprint, polygon):
                    return False
        last_time = time
    return True


def blocking_obstacles(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    motion: Motion | Primitive | None,
    time: float,
    speed: float = 1.0,
) -> list[str]:
    """Names of dynamic obstacles overlapping the (possibly swept) footprint."""
    if motion is None or motion.distance <= 0.0:
        states: list[Pose] = [pose]
        fractions = [0.0]
    else:
        states = _arc_poses(vehicle, pose, motion, 0.08)
        fractions = [index / (len(states) - 1) for index in range(len(states))]
    names: list[str] = []
    for state, fraction in zip(states, fractions):
        footprint = vehicle.footprint(*state)
        stamp = time + fraction * motion.distance / speed if motion is not None else time
        for obstacle in scene.dynamic_obstacles:
            polygon = obstacle.polygon_at(stamp)
            if polygon is not None and polygons_intersect(footprint, polygon) and obstacle.name not in names:
                names.append(obstacle.name)
    return names


def earliest_clear_departure(
    vehicle: Vehicle,
    scene: Scene,
    pose: Pose,
    motion: Motion | Primitive,
    earliest_time: float,
    speed: float,
    shift_duration: float = 0.0,
    max_wait: float = 30.0,
    time_step: float = 0.25,
    collision_step: float = 0.08,
) -> tuple[float | None, list[str]]:
    """Smallest departure time making the stationary hold and timed arc safe.

    ``shift_duration`` is a stationary gear-change interval before driving.
    Returns ``(None, blockers)`` when no departure exists inside the horizon.
    """
    blockers = blocking_obstacles(vehicle, scene, pose, motion, earliest_time, speed)
    attempt = 0
    while True:
        departure = earliest_time + attempt * time_step
        if departure - earliest_time > max_wait + 1e-9:
            return None, blockers
        movement_start = departure + shift_duration
        hold_ok = stationary_collision_free(
            vehicle, scene, pose, earliest_time, movement_start, time_step
        )
        arc_ok = hold_ok and sweep_collision_free(
            vehicle,
            scene,
            pose,
            motion,
            movement_start,
            speed,
            collision_step,
            time_step,
        )
        if arc_ok:
            return departure, blockers
        attempt += 1
