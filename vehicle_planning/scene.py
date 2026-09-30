from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .geometry import (
    Polygon,
    normalize_angle,
    point_in_polygon,
    polygons_intersect,
    transform_points,
)

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class KeyFrame:
    """Position/heading of a moving obstacle at one point in time (seconds)."""

    time: float
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class DynamicObstacle:
    """Low-speed moving obstacle with a fixed footprint and a keyframe timetable.

    Poses are interpolated linearly in x/y and along the shortest angular arc.
    Before ``start_time`` and after ``end_time`` the obstacle is absent, which
    models a temporary object only passing through the workspace.
    """

    name: str
    footprint: Polygon
    keyframes: tuple[KeyFrame, ...]
    start_time: float | None = None
    end_time: float | None = None

    def __post_init__(self) -> None:
        if len(self.keyframes) < 2:
            raise ValueError("dynamic obstacle needs at least two keyframes")
        times = [frame.time for frame in self.keyframes]
        if any(before >= after for before, after in zip(times, times[1:])):
            raise ValueError("dynamic obstacle keyframes must be strictly increasing in time")
        if self.start_time is not None and self.end_time is not None:
            if self.end_time < self.start_time:
                raise ValueError("dynamic obstacle end_time must be >= start_time")

    @property
    def active_start(self) -> float:
        return self.start_time if self.start_time is not None else self.keyframes[0].time

    @property
    def active_end(self) -> float:
        return self.end_time if self.end_time is not None else self.keyframes[-1].time

    def is_active(self, time: float) -> bool:
        return self.active_start <= time <= self.active_end

    def pose_at(self, time: float) -> Pose | None:
        """Interpolated pose at ``time``; ``None`` when the obstacle is absent."""
        if not self.is_active(time):
            return None
        if time <= self.keyframes[0].time:
            frame = self.keyframes[0]
            return (frame.x, frame.y, normalize_angle(frame.theta))
        if time >= self.keyframes[-1].time:
            frame = self.keyframes[-1]
            return (frame.x, frame.y, normalize_angle(frame.theta))
        for left, right in zip(self.keyframes, self.keyframes[1:]):
            if left.time <= time <= right.time:
                ratio = (time - left.time) / (right.time - left.time)
                theta = left.theta + ratio * normalize_angle(right.theta - left.theta)
                return (
                    left.x + ratio * (right.x - left.x),
                    left.y + ratio * (right.y - left.y),
                    normalize_angle(theta),
                )
        return None

    def polygon_at(self, time: float) -> Polygon | None:
        pose = self.pose_at(time)
        if pose is None:
            return None
        return transform_points(self.footprint, pose[0], pose[1], pose[2])

    def to_dict(self) -> dict:
        data: dict = {
            "name": self.name,
            "footprint": [[x, y] for x, y in self.footprint],
            "timetable": [
                {"time": frame.time, "x": frame.x, "y": frame.y, "theta": frame.theta}
                for frame in self.keyframes
            ],
        }
        if self.start_time is not None:
            data["start_time"] = self.start_time
        if self.end_time is not None:
            data["end_time"] = self.end_time
        return data


@dataclass(frozen=True)
class Scene:
    name: str
    bounds: tuple[float, float, float, float]
    obstacles: tuple[Polygon, ...]
    start: Pose
    goal: Pose
    goal_xy_tolerance: float
    goal_theta_tolerance: float
    dynamic_obstacles: tuple[DynamicObstacle, ...] = field(default=())

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        min_x, min_y, max_x, max_y = self.bounds
        return (
            min_x + margin <= x <= max_x - margin
            and min_y + margin <= y <= max_y - margin
        )

    def _collides_static(self, footprint: Polygon) -> bool:
        return any(polygons_intersect(footprint, obstacle) for obstacle in self.obstacles)

    def footprint_collides(self, footprint: Polygon, time: float | None = None) -> bool:
        if not all(self.contains_point(x, y) for x, y in footprint):
            return True
        if self._collides_static(footprint):
            return True
        if time is None:
            return False
        for obstacle in self.dynamic_obstacles:
            polygon = obstacle.polygon_at(time)
            if polygon is not None and polygons_intersect(footprint, polygon):
                return True
        return False

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

    def timetable_signature(self) -> str:
        """Stable hash of the dynamic-obstacle timetable.

        A saved plan records this signature; replay recomputes it from the
        scene file so an old success marker cannot be reused after the
        timetable changes.
        """
        payload = json.dumps(
            [obstacle.to_dict() for obstacle in self.dynamic_obstacles],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

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


def _polygon(data: Sequence[Sequence[float]]) -> Polygon:
    polygon = tuple((float(point[0]), float(point[1])) for point in data)
    if len(polygon) < 3:
        raise ValueError("each obstacle must contain at least three points")
    return polygon


def _dynamic_obstacle(data: dict) -> DynamicObstacle:
    keyframes = tuple(
        KeyFrame(
            time=float(frame["time"]),
            x=float(frame["x"]),
            y=float(frame["y"]),
            theta=float(frame.get("theta", 0.0)),
        )
        for frame in data["timetable"]
    )
    return DynamicObstacle(
        name=str(data.get("name", "dynamic_obstacle")),
        footprint=_polygon(data["footprint"]),
        keyframes=keyframes,
        start_time=float(data["start_time"]) if "start_time" in data else None,
        end_time=float(data["end_time"]) if "end_time" in data else None,
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
            _dynamic_obstacle(item) for item in data.get("dynamic_obstacles", [])
        ),
    )


def load_scene(path: str | Path) -> Scene:
    return scene_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_scene(scene: Scene, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(scene.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
