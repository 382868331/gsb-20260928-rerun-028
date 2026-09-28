from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .geometry import Polygon, transform_points


@dataclass(frozen=True)
class Vehicle:
    name: str
    length: float
    width: float
    wheelbase: float
    front_overhang: float
    rear_overhang: float
    max_steering_angle: float
    max_speed: float = 2.0

    @property
    def min_turning_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steering_angle)

    def footprint(self, x: float, y: float, theta: float) -> Polygon:
        rear = -self.rear_overhang
        front = self.wheelbase + self.front_overhang
        half = self.width / 2.0
        local = [
            (rear, -half),
            (front, -half),
            (front, half),
            (rear, half),
        ]
        return transform_points(local, x, y, theta)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "wheelbase": self.wheelbase,
            "front_overhang": self.front_overhang,
            "rear_overhang": self.rear_overhang,
            "max_steering_angle": self.max_steering_angle,
            "max_speed": self.max_speed,
            "min_turning_radius": self.min_turning_radius,
        }


def load_vehicle(path: str | Path) -> Vehicle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    values = {
        key: data[key]
        for key in (
            "name",
            "length",
            "width",
            "wheelbase",
            "front_overhang",
            "rear_overhang",
            "max_steering_angle",
        )
    }
    values["max_speed"] = data.get("max_speed", 2.0)
    vehicle = Vehicle(**values)
    if abs(vehicle.front_overhang + vehicle.wheelbase + vehicle.rear_overhang - vehicle.length) > 1e-6:
        raise ValueError("front overhang + wheelbase + rear overhang must equal length")
    return vehicle
