from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import angle_distance, polygons_intersect
from .motion import sample_arc
from .plan import PlanResult
from .scene import Scene
from .vehicle import Vehicle


@dataclass(frozen=True)
class TimedSample:
    time: float
    x: float
    y: float
    theta: float
    kind: str


@dataclass(frozen=True)
class ReplayReport:
    valid: bool
    samples: int
    goal_xy_error: float
    goal_theta_error: float
    message: str
    timetable_changed: bool = False
    events: tuple[TimedSample, ...] = ()
    timeline_valid: bool = False


def _collision_message(
    scene: Scene,
    vehicle: Vehicle,
    pose: tuple[float, float, float],
    time: float,
    phase: str,
) -> str | None:
    footprint = vehicle.footprint(*pose)
    for obstacle in scene.dynamic_obstacles:
        polygon = obstacle.polygon_at(time)
        if polygon is not None and polygons_intersect(footprint, polygon):
            return (
                f"collision with dynamic obstacle '{obstacle.name}' at t={time:.2f}s "
                f"during {phase}"
            )
    return None


def replay_timed(
    plan: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    speed: float = 1.0,
    step: float = 0.05,
    time_step: float = 0.05,
) -> ReplayReport:
    """Independently replay a timed plan against the current scene timetable.

    The obstacle position is recomputed from the scene at every timestamp:
    stored plan geometry is never trusted as an obstacle source.
    """
    if not plan.found:
        return ReplayReport(False, 0, math.inf, math.inf, f"plan status is {plan.status}")
    current_signature = scene.timetable_signature()
    changed = bool(plan.timetable_signature) and plan.timetable_signature != current_signature

    drive_actions = [action for action in plan.timeline if action.kind == "drive"]
    if len(drive_actions) != len(plan.motions):
        return ReplayReport(
            False, 0, math.inf, math.inf,
            "timeline drive actions do not match saved motions", changed,
        )
    for action, motion in zip(drive_actions, plan.motions):
        if (
            action.direction != motion.direction
            or abs(action.steering - motion.steering) > 1e-9
            or abs(action.distance - motion.distance) > 1e-9
        ):
            return ReplayReport(
                False, 0, math.inf, math.inf,
                "a timeline drive action disagrees with its saved motion", changed,
            )

    clock = 0.0
    for action in plan.timeline:
        if action.t0 + 1e-6 < clock or action.t1 < action.t0 - 1e-9:
            return ReplayReport(
                False, 0, math.inf, math.inf,
                "timeline actions are not monotonically time ordered", changed,
            )
        clock = action.t1

    events: list[TimedSample] = []
    current = plan.start_pose
    expected = plan.start_pose
    checked = 0
    collision_step = step

    for action in plan.timeline:
        if action.kind in ("wait", "shift"):
            steps = max(1, int(math.ceil(action.duration / time_step)))
            for index in range(steps + 1):
                time = action.t0 + action.duration * index / steps
                footprint = vehicle.footprint(*current)
                if scene.footprint_collides(footprint):
                    return ReplayReport(False, checked, math.inf, math.inf, f"static collision at t={time:.2f}s during {action.kind}", changed, tuple(events), False)
                message = _collision_message(scene, vehicle, current, time, action.kind)
                if message is not None:
                    return ReplayReport(False, checked, math.inf, math.inf, message, changed, tuple(events), False)
                checked += 1
                events.append(TimedSample(time, current[0], current[1], current[2], action.kind))
            continue

        if action.kind != "drive":
            return ReplayReport(False, checked, math.inf, math.inf, f"unknown action kind {action.kind}", changed, tuple(events), False)

        duration = action.distance / speed
        if abs(duration - action.duration) > 1e-4:
            return ReplayReport(
                False,
                checked,
                math.inf,
                math.inf,
                f"drive timing {action.duration:.3f}s disagrees with distance/speed {duration:.3f}s",
                changed,
                tuple(events),
                False,
            )
        arc = sample_arc(
            current,
            action.direction,
            action.steering,
            action.distance,
            vehicle.wheelbase,
            collision_step,
        )
        for index, pose in enumerate(arc):
            fraction = index / (len(arc) - 1)
            time = action.t0 + action.duration * fraction
            footprint = vehicle.footprint(*pose)
            if scene.footprint_collides(footprint):
                return ReplayReport(False, checked, math.inf, math.inf, f"static collision at t={time:.2f}s during drive", changed, tuple(events), False)
            message = _collision_message(scene, vehicle, pose, time, "drive")
            if message is not None:
                return ReplayReport(False, checked, math.inf, math.inf, message, changed, tuple(events), False)
            checked += 1
            events.append(TimedSample(time, pose[0], pose[1], pose[2], "drive"))
        current = arc[-1]

    saved_final = plan.states[-1]
    if (
        math.hypot(current[0] - saved_final[0], current[1] - saved_final[1]) > 1e-5
        or angle_distance(current[2], saved_final[2]) > 1e-5
    ):
        return ReplayReport(
            False, checked, math.inf, math.inf,
            "replayed final pose disagrees with saved state", changed, tuple(events), False,
        )
    xy_error = math.hypot(current[0] - scene.goal[0], current[1] - scene.goal[1])
    theta_error = angle_distance(current[2], scene.goal[2])
    if xy_error > scene.goal_xy_tolerance or theta_error > scene.goal_theta_tolerance:
        return ReplayReport(
            False, checked, xy_error, theta_error,
            "final pose misses goal tolerance", changed, tuple(events), False,
        )

    base = "timed replay is collision-free and reaches the target pose"
    if changed:
        base = "timed replay re-verified against a changed timetable; " + base
    return ReplayReport(True, checked, xy_error, theta_error, base, changed, tuple(events), True)


def replay(plan: PlanResult, vehicle: Vehicle, scene: Scene, collision_step: float = 0.03) -> ReplayReport:
    if not plan.found:
        return ReplayReport(False, 0, math.inf, math.inf, f"plan status is {plan.status}")
    if vehicle.name != plan.vehicle or scene.name != plan.scene:
        return ReplayReport(False, 0, math.inf, math.inf, "plan does not match vehicle/scene names")

    if plan.timeline:
        return replay_timed(plan, vehicle, scene, step=collision_step, time_step=collision_step)

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

    report = replay(plan, vehicle, scene) if plan.timeline else None
    if report is not None and report.valid:
        drive_forward: list[tuple[float, float]] = []
        drive_reverse: list[tuple[float, float]] = []
        wait_points: list[tuple[float, float, float, float, str]] = []
        for event in report.events:
            if event.kind == "drive":
                # Recover gear from the active drive action via nearest timeline match.
                gear = _gear_at(plan, event.time)
                (drive_forward if gear == 1 else drive_reverse).append((event.x, event.y))
            elif event.kind in ("wait", "shift"):
                wait_points.append((event.x, event.y, event.time, event.time, event.kind))
        for color, pts in (("#1f77b4", drive_forward), ("#d62728", drive_reverse)):
            if pts:
                polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
                parts.append(f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2.5' stroke-linecap='round'/>")
        shown: set[tuple[float, float, str]] = set()
        for x, y, _, _, kind in wait_points:
            key = (round(x, 3), round(y, 3), kind)
            if key in shown:
                continue
            shown.add(key)
            fill = "#ff9f1c" if kind == "wait" else "#8e44ad"
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' fill='{fill}'/>")
    else:
        trajectory = densify(plan, vehicle, step)
        for color, direction in (("#1f77b4", 1), ("#d62728", -1)):
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y, _, d in trajectory if d == direction)
            if points:
                parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5' stroke-linecap='round'/>")

    for obstacle in scene.dynamic_obstacles:
        trail = " ".join(f"{sx(f.x):.1f},{sy(f.y):.1f}" for f in obstacle.keyframes)
        parts.append(f"<polyline points='{trail}' fill='none' stroke='#2ca02c' stroke-width='1.5' stroke-dasharray='6 4'/>")
        for index, frame in enumerate(obstacle.keyframes):
            polygon = obstacle.polygon_at(frame.time)
            assert polygon is not None
            points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in polygon)
            parts.append(f"<polygon points='{points}' fill='#2ca02c' fill-opacity='0.18' stroke='#2ca02c' stroke-dasharray='4 3'/>")
            parts.append(
                f"<text x='{sx(frame.x):.1f}' y='{sy(frame.y) - 8:.1f}' font-family='sans-serif' font-size='11' fill='#1b6e1b'>t={frame.time:g}s</text>"
            )

    def draw_pose(pose, fill: str) -> None:
        polygon = vehicle.footprint(*pose)
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in polygon)
        parts.append(f"<polygon points='{points}' fill='{fill}' fill-opacity='0.28' stroke='{fill}' stroke-width='2'/>")

    draw_pose(scene.start, "#2ca02c")
    draw_pose(scene.goal, "#ff7f0e")
    parts.append(
        "<text x='10' y='22' font-family='sans-serif' font-size='14'>blue=forward red=reverse orange=wait purple=shift green dashed=moving obstacle</text>"
    )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def _gear_at(plan: PlanResult, time: float) -> int:
    current = 0
    for action in plan.timeline:
        if action.kind == "drive":
            current = action.direction
            if action.t0 - 1e-9 <= time <= action.t1 + 1e-9:
                return current
    return current
