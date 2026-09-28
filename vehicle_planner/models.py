from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Vehicle:
    name: str
    wheelbase: float
    length: float
    width: float
    rear_overhang: float
    max_steering_angle: float

    @property
    def min_turning_radius(self) -> float:
        import math

        return self.wheelbase / math.tan(self.max_steering_angle)


@dataclass(frozen=True)
class RectangleObstacle:
    name: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class GoalTolerance:
    position: float
    heading: float


@dataclass(frozen=True)
class Scenario:
    name: str
    start: Pose
    goal: Pose
    obstacles: tuple[RectangleObstacle, ...]
    goal_tolerance: GoalTolerance
    bounds: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class MotionCommand:
    steering_angle: float
    velocity: float
    duration: float


@dataclass
class Plan:
    vehicle_name: str
    scenario_name: str
    status: str
    commands: list[MotionCommand] = field(default_factory=list)
    start: Pose | None = None
    goal: Pose | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def dataclass_to_dict(value: Any) -> Any:
    if isinstance(value, tuple):
        return [dataclass_to_dict(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(getattr(value, key)) for key in value.__dataclass_fields__}
    return value


def vehicle_from_dict(data: dict[str, Any]) -> Vehicle:
    return Vehicle(**data)


def obstacle_from_dict(data: dict[str, Any]) -> RectangleObstacle:
    return RectangleObstacle(**data)


def pose_from_dict(data: dict[str, Any]) -> Pose:
    return Pose(float(data["x"]), float(data["y"]), float(data["theta"]))


def tolerance_from_dict(data: dict[str, Any]) -> GoalTolerance:
    return GoalTolerance(float(data["position"]), float(data["heading"]))


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    return Scenario(
        name=data["name"],
        start=pose_from_dict(data["start"]),
        goal=pose_from_dict(data["goal"]),
        obstacles=tuple(obstacle_from_dict(item) for item in data["obstacles"]),
        goal_tolerance=tolerance_from_dict(data["goal_tolerance"]),
        bounds=tuple(data["bounds"]) if data.get("bounds") is not None else None,
    )


def command_from_dict(data: dict[str, Any]) -> MotionCommand:
    return MotionCommand(**data)


def plan_from_dict(data: dict[str, Any]) -> Plan:
    return Plan(
        vehicle_name=data["vehicle_name"],
        scenario_name=data["scenario_name"],
        status=data["status"],
        commands=[command_from_dict(item) for item in data.get("commands", [])],
        start=pose_from_dict(data["start"]) if data.get("start") else None,
        goal=pose_from_dict(data["goal"]) if data.get("goal") else None,
        metadata=data.get("metadata", {}),
    )


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    return asdict(plan)
