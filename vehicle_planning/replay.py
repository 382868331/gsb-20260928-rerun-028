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


def replay(plan: PlanResult, vehicle: Vehicle, scene: Scene, collision_step: float = 0.03) -> ReplayReport:
    if not plan.found:
        return ReplayReport(False, 0, math.inf, math.inf, f"plan status is {plan.status}")
    if vehicle.name != plan.vehicle or scene.name != plan.scene:
        return ReplayReport(False, 0, math.inf, math.inf, "plan does not match vehicle/scene names")

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


def densify(plan: PlanResult, vehicle: Vehicle, step: float = 0.05) -> list[tuple[float, float, float, int]]:
    output: list[tuple[float, float, float, int]] = []
    current = plan.start_pose
    output.append((current[0], current[1], current[2], 0))
    for motion in plan.motions:
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

    def sx(x: float) -> float:
        return margin + (x - min_x) * scale

    def sy(y: float) -> float:
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

    def draw_pose(pose, fill: str) -> None:
        polygon = vehicle.footprint(*pose)
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in polygon)
        parts.append(f"<polygon points='{points}' fill='{fill}' fill-opacity='0.28' stroke='{fill}' stroke-width='2'/>")

    draw_pose(scene.start, "#2ca02c")
    draw_pose(scene.goal, "#ff7f0e")
    parts.append(
        "<text x='10' y='22' font-family='sans-serif' font-size='14'>blue=forward red=reverse; green=start orange=goal</text>"
    )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
