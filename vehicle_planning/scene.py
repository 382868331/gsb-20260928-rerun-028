from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .geometry import Polygon, normalize_angle, point_in_polygon, polygons_intersect, transform_points

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class DynamicObstacle:
    """Polygon obstacle moving through a timed keyframe schedule.

    Keyframes are ``(time, x, y, theta)``. Pose is linearly interpolated
    between keyframes and held outside the scheduled range.
    """

    name: str
    footprint: Polygon
    keyframes: tuple[tuple[float, float, float, float], ...]

    def pose_at(self, time: float) -> Pose:
        frames = self.keyframes
        if time <= frames[0][0]:
            first = frames[0]
            return (first[1], first[2], first[3])
        if time >= frames[-1][0]:
            last = frames[-1]
            return (last[1], last[2], last[3])
        for index in range(len(frames) - 1):
            t0, x0, y0, a0 = frames[index]
            t1, x1, y1, a1 = frames[index + 1]
            if t0 <= time <= t1:
                fraction = (time - t0) / (t1 - t0) if t1 > t0 else 0.0
                angle = a0 + fraction * normalize_angle(a1 - a0)
                return (
                    x0 + fraction * (x1 - x0),
                    y0 + fraction * (y1 - y0),
                    normalize_angle(angle),
                )
        last = frames[-1]
        return (last[1], last[2], last[3])

    def polygon_at(self, time: float) -> Polygon:
        x, y, theta = self.pose_at(time)
        return transform_points(self.footprint, x, y, theta)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "footprint": [[x, y] for x, y in self.footprint],
            "keyframes": [list(frame) for frame in self.keyframes],
        }


@dataclass(frozen=True)
class Scene:
    name: str
    bounds: tuple[float, float, float, float]
    obstacles: tuple[Polygon, ...]
    start: Pose
    goal: Pose
    goal_xy_tolerance: float
    goal_theta_tolerance: float
    dynamic_obstacles: tuple[DynamicObstacle, ...] = ()

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        min_x, min_y, max_x, max_y = self.bounds
        return (
            min_x + margin <= x <= max_x - margin
            and min_y + margin <= y <= max_y - margin
        )

    def footprint_collides(self, footprint: Polygon) -> bool:
        if not all(self.contains_point(x, y) for x, y in footprint):
            return True
        return any(polygons_intersect(footprint, obstacle) for obstacle in self.obstacles)

    def blocking_dynamic_obstacles(self, footprint: Polygon, time: float) -> list[str]:
        """Names of scheduled obstacles overlapping the footprint at ``time``."""
        return [
            obstacle.name
            for obstacle in self.dynamic_obstacles
            if polygons_intersect(footprint, obstacle.polygon_at(time))
        ]

    def collides_at(self, footprint: Polygon, time: float) -> bool:
        """Spatiotemporal collision check against static and scheduled bodies."""
        return self.footprint_collides(footprint) or bool(
            self.blocking_dynamic_obstacles(footprint, time)
        )

    def point_collides(self, x: float, y: float) -> bool:
        if not self.contains_point(x, y):
            return True
        return any(point_in_polygon((x, y), obstacle) for obstacle in self.obstacles)

    def at_goal(self, pose: Pose) -> bool:
        x, y, theta = pose
        gx, gy, gtheta = self.goal
        dx = x - gx
        dy = y - gy
        # Angle comparison is completed by motion primitives using geometry.normalize_angle.
        from .geometry import angle_distance

        return (
            dx * dx + dy * dy <= self.goal_xy_tolerance**2
            and angle_distance(theta, gtheta) <= self.goal_theta_tolerance
        )

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "bounds": list(self.bounds),
            "obstacles": [[[x, y] for x, y in polygon] for polygon in self.obstacles],
            "start": list(self.start),
            "goal": list(self.goal),
            "goal_tolerance": {
                "xy": self.goal_xy_tolerance,
                "theta": self.goal_theta_tolerance,
            },
        }
        if self.dynamic_obstacles:
            data["dynamic_obstacles"] = [item.to_dict() for item in self.dynamic_obstacles]
        return data

    def signature(self) -> str:
        """Hash covering geometry and the dynamic obstacle timetable."""
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _polygon(data: Sequence[Sequence[float]]) -> Polygon:
    polygon = tuple((float(point[0]), float(point[1])) for point in data)
    if len(polygon) < 3:
        raise ValueError("each obstacle must contain at least three points")
    return polygon


def _dynamic_obstacle(data: dict) -> DynamicObstacle:
    frames = tuple(
        (
            float(frame[0]),
            float(frame[1]),
            float(frame[2]),
            float(frame[3]),
        )
        for frame in data["keyframes"]
    )
    if len(frames) < 2:
        raise ValueError("dynamic obstacle keyframes must contain at least two entries")
    times = [frame[0] for frame in frames]
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError("dynamic obstacle keyframe times must be strictly increasing")
    return DynamicObstacle(
        name=str(data["name"]),
        footprint=_polygon(data["footprint"]),
        keyframes=frames,
    )


def scene_from_dict(data: dict) -> Scene:
    tolerance = data.get("goal_tolerance", {})
    return Scene(
        name=data["name"],
        bounds=tuple(float(v) for v in data["bounds"]),
        obstacles=tuple(_polygon(item) for item in data["obstacles"]),
        start=tuple(float(v) for v in data["start"]),
        goal=tuple(float(v) for v in data["goal"]),
        goal_xy_tolerance=float(tolerance.get("xy", 0.15)),
        goal_theta_tolerance=float(tolerance.get("theta", 0.12)),
        dynamic_obstacles=tuple(
            _dynamic_obstacle(item) for item in data.get("dynamic_obstacles", ())
        ),
    )


def load_scene(path: str | Path) -> Scene:
    return scene_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_scene(scene: Scene, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(scene.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
