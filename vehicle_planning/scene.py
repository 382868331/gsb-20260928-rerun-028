from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .geometry import Polygon, point_in_polygon, polygons_intersect

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class Scene:
    name: str
    bounds: tuple[float, float, float, float]
    obstacles: tuple[Polygon, ...]
    start: Pose
    goal: Pose
    goal_xy_tolerance: float
    goal_theta_tolerance: float

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
        return {
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


def _polygon(data: Sequence[Sequence[float]]) -> Polygon:
    polygon = tuple((float(point[0]), float(point[1])) for point in data)
    if len(polygon) < 3:
        raise ValueError("each obstacle must contain at least three points")
    return polygon


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
    )


def load_scene(path: str | Path) -> Scene:
    return scene_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_scene(scene: Scene, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(scene.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
