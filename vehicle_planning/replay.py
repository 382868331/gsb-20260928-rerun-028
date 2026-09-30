"""Independent replay of saved plans.

Timed plans are replayed action by action on a common clock: drive actions
integrate the bicycle model, wait/gear_switch actions hold the vehicle still.
At every sampled instant the full vehicle rectangle is tested against both
static geometry and the scene's dynamic-obstacle timetable. A signature over
the whole scene (including that timetable) invalidates a plan saved for an
older schedule, even if it was previously marked successful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .geometry import angle_distance
from .motion import sample_arc
from .plan import PlanResult
from .scene import Scene
from .vehicle import Vehicle


@dataclass(frozen=True)
class ReplayReport:
    valid: bool
    samples: int
    goal_xy_error: float
    goal_theta_error: float
    message: str
    final_time: float = 0.0


def replay(plan: PlanResult, vehicle: Vehicle, scene: Scene, collision_step: float = 0.03) -> ReplayReport:
    if not plan.found:
        return ReplayReport(False, 0, math.inf, math.inf, f"plan status is {plan.status}")
    if vehicle.name != plan.vehicle or scene.name != plan.scene:
        return ReplayReport(False, 0, math.inf, math.inf, "plan does not match vehicle/scene names")

    if plan.timed:
        return _replay_timed(plan, vehicle, scene, collision_step)
    if scene.dynamic_obstacles:
        return ReplayReport(
            False,
            0,
            math.inf,
            math.inf,
            "untimed plan cannot be reused for a scene with scheduled obstacles",
        )
    return _replay_spatial(plan, vehicle, scene, collision_step)


def _replay_spatial(plan, vehicle, scene, collision_step):
    current = plan.start_pose
    checked = 0
    endpoint_index = 1
    for motion in plan.motions:
        arc = sample_arc(
            current,
            motion.direction,
            motion.steering,
            motion.distance,
            vehicle.wheelbase,
            collision_step,
        )
        for pose in arc:
            if scene.footprint_collides(vehicle.footprint(*pose)):
                return ReplayReport(False, checked, math.inf, math.inf, f"collision at sampled pose {pose}")
            checked += 1
        current = arc[-1]

        if endpoint_index < len(plan.states):
            saved = plan.states[endpoint_index]
            xy_error = math.hypot(current[0] - saved[0], current[1] - saved[1])
            if xy_error > 1e-5 or angle_distance(current[2], saved[2]) > 1e-5:
                return ReplayReport(False, checked, math.inf, math.inf, "saved state is not reachable by its saved motion")
        endpoint_index += 1

    xy_error = math.hypot(current[0] - scene.goal[0], current[1] - scene.goal[1])
    theta_error = angle_distance(current[2], scene.goal[2])
    if xy_error > scene.goal_xy_tolerance or theta_error > scene.goal_theta_tolerance:
        return ReplayReport(False, checked, xy_error, theta_error, "final pose misses goal tolerance")
    return ReplayReport(True, checked, xy_error, theta_error, "replayed path is collision-free and reaches the target pose")


def _replay_timed(plan, vehicle, scene, collision_step):
    expected_signature = scene.signature()
    if plan.scene_signature != expected_signature:
        return ReplayReport(
            False,
            0,
            math.inf,
            math.inf,
            "scene or dynamic-obstacle timetable changed; saved plan must be re-planned",
        )

    time_now = 0.0
    pose = plan.start_pose
    checked = 0
    previous_end = 0.0

    def check_pose(at_pose, at_time, context):
        footprint = vehicle.footprint(*at_pose)
        if scene.footprint_collides(footprint):
            return f"static collision at t={at_time:.2f}s ({context})"
        blocking = scene.blocking_dynamic_obstacles(footprint, at_time)
        if blocking:
            return (
                f"collision with dynamic obstacle {', '.join(blocking)} "
                f"at t={at_time:.2f}s ({context})"
            )
        return None

    error = check_pose(pose, time_now, "start")
    if error:
        return ReplayReport(False, checked, math.inf, math.inf, error)
    checked += 1

    for index, action in enumerate(plan.actions):
        expected_start = time_now
        # A leading wait action is an explicit departure delay: it may start
        # at t=0 while the vehicle holds the start pose before driving off.
        if index == 0 and action.type == "wait":
            expected_start = 0.0
        if action.start_time < expected_start - 1e-9:
            return ReplayReport(False, checked, math.inf, math.inf, "action timeline is not ordered")
        if action.end_time < action.start_time - 1e-9:
            return ReplayReport(False, checked, math.inf, math.inf, f"{action.type} action has negative duration")
        duration = action.end_time - action.start_time

        if action.type in ("wait", "gear_switch"):
            if abs(action.start_time - expected_start) > 1e-9:
                return ReplayReport(False, checked, math.inf, math.inf, f"time gap before {action.type} at index {index}")
            probes = max(1, int(math.ceil(duration / 0.1)))
            for sub in range(probes + 1):
                probe_time = min(action.start_time + duration * sub / probes, action.end_time)
                error = check_pose(pose, probe_time, action.type)
                if error:
                    return ReplayReport(False, checked, math.inf, math.inf, error)
                checked += 1
            time_now = action.end_time
            previous_end = time_now
            continue

        if action.type != "drive":
            return ReplayReport(False, checked, math.inf, math.inf, f"unknown action type {action.type}")

        if abs(action.start_time - expected_start) > 1e-9:
            return ReplayReport(False, checked, math.inf, math.inf, f"time gap before drive at index {index}")
        distance = action.distance
        arc = sample_arc(
            pose,
            action.direction,
            action.steering,
            distance,
            vehicle.wheelbase,
            collision_step,
        )
        segment_time = abs(distance) / 1.2
        if abs(segment_time - duration) > 0.05:
            return ReplayReport(
                False,
                checked,
                math.inf,
                math.inf,
                f"drive duration {duration:.2f}s does not match cruise speed (expected {segment_time:.2f}s)",
            )
        for sub_index, sampled in enumerate(arc):
            fraction = sub_index / (len(arc) - 1)
            sample_time = action.start_time + duration * fraction
            error = check_pose(sampled, sample_time, "drive sweep")
            if error:
                return ReplayReport(False, checked, math.inf, math.inf, error)
            checked += 1
        pose = arc[-1]
        time_now = action.end_time
        previous_end = time_now

    xy_error = math.hypot(pose[0] - scene.goal[0], pose[1] - scene.goal[1])
    theta_error = angle_distance(pose[2], scene.goal[2])
    if xy_error > scene.goal_xy_tolerance or theta_error > scene.goal_theta_tolerance:
        return ReplayReport(False, checked, xy_error, theta_error, "final pose misses goal tolerance")
    return ReplayReport(
        True,
        checked,
        xy_error,
        theta_error,
        f"timed replay verified against static and moving obstacles, final time {time_now:.2f}s",
        final_time=time_now,
    )


def densify(plan: PlanResult, vehicle: Vehicle, step: float = 0.05) -> list[tuple[float, float, float, int]]:
    output = []
    current = plan.start_pose
    output.append((current[0], current[1], current[2], 0))
    motions = plan.motions
    for motion in motions:
        arc = sample_arc(
            current,
            motion.direction,
            motion.steering,
            motion.distance,
            vehicle.wheelbase,
            step,
        )[1:]
        output.extend((x, y, theta, motion.direction) for x, y, theta in arc)
        current = arc[-1]
    return output


def render_svg(plan: PlanResult, vehicle: Vehicle, scene: Scene, path: str | Path, step: float = 0.05) -> None:
    min_x, min_y, max_x, max_y = scene.bounds
    scale = 720.0 / max(max_x - min_x, max_y - min_y)
    margin = 35.0

    def sx(x):
        return margin + (x - min_x) * scale

    def sy(y):
        return margin + (max_y - y) * scale

    width = (max_x - min_x) * scale + 2 * margin
    height = (max_y - min_y) * scale + 2 * margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        "<rect width='100%' height='100%' fill='white'/>",
        f"<rect x='{sx(min_x):.1f}' y='{sy(max_y):.1f}' width='{(max_x-min_x)*scale:.1f}' height='{(max_y-min_y)*scale:.1f}' fill='#f7f7f7' stroke='black' stroke-width='2'/>",
    ]
    for obstacle in scene.obstacles:
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in obstacle)
        parts.append(f"<polygon points='{points}' fill='#b8b8b8' stroke='#555'/>")

    trajectory = densify(plan, vehicle, step)
    for color, direction in (("#1f77b4", 1), ("#d62728", -1)):
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y, _, d in trajectory if d == direction)
        if points:
            parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5' stroke-linecap='round'/>")

    for obstacle in scene.dynamic_obstacles:
        for frame in obstacle.keyframes:
            t, x, y, theta = frame
            polygon = vehicle_footprint_like(obstacle.footprint, x, y, theta)
            points = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in polygon)
            parts.append(
                f"<polygon points='{points}' fill='#ff9f43' fill-opacity='0.22' stroke='#e67e22' stroke-dasharray='5,4'/>"
            )

    def draw_pose(pose, fill):
        polygon = vehicle.footprint(*pose)
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in polygon)
        parts.append(f"<polygon points='{points}' fill='{fill}' fill-opacity='0.28' stroke='{fill}' stroke-width='2'/>")

    draw_pose(scene.start, "#2ca02c")
    draw_pose(scene.goal, "#ff7f0e")
    subtitle = "blue=forward red=reverse; orange dashed=scheduled obstacle keyframes"
    parts.append(f"<text x='10' y='22' font-family='sans-serif' font-size='14'>{subtitle}</text>")
    parts.append("</svg>")
    Path(path).write_text(chr(10).join(parts), encoding="utf-8")


def vehicle_footprint_like(local_polygon, x, y, theta):
    from .geometry import transform_points

    return transform_points(local_polygon, x, y, theta)
