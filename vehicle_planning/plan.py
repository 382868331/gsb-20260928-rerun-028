from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class Motion:
    direction: int
    steering: float
    distance: float


@dataclass(frozen=True)
class Action:
    """One timed control action: drive, stop-and-wait, or gear transition."""

    type: str
    start_time: float
    end_time: float
    direction: int = 0
    steering: float = 0.0
    distance: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "direction": self.direction,
            "steering": self.steering,
            "distance": self.distance,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlanResult:
    status: str
    vehicle: str
    scene: str
    start_pose: Pose
    goal_pose: Pose
    states: tuple[Pose, ...]
    motions: tuple[Motion, ...]
    nodes_expanded: int
    max_nodes: int
    message: str
    actions: tuple[Action, ...] = ()
    scene_signature: str = ""

    @property
    def found(self) -> bool:
        return self.status == "path_found"

    @property
    def total_distance(self) -> float:
        if self.actions:
            return sum(abs(item.distance) for item in self.actions if item.type == "drive")
        return sum(abs(item.distance) for item in self.motions)

    @property
    def timed(self) -> bool:
        return bool(self.actions)

    @property
    def final_time(self) -> float:
        return self.actions[-1].end_time if self.actions else 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "vehicle": self.vehicle,
            "scene": self.scene,
            "start_pose": list(self.start_pose),
            "goal_pose": list(self.goal_pose),
            "nodes_expanded": self.nodes_expanded,
            "max_nodes": self.max_nodes,
            "total_distance": self.total_distance,
            "motions": [
                {
                    "direction": item.direction,
                    "steering": item.steering,
                    "distance": item.distance,
                }
                for item in self.motions
            ],
            "states": [[x, y, theta] for x, y, theta in self.states],
            "actions": [item.to_dict() for item in self.actions],
            "scene_signature": self.scene_signature,
        }


def _actions_from_dict(data: dict) -> tuple[Action, ...]:
    return tuple(
        Action(
            type=str(item["type"]),
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            direction=int(item.get("direction", 0)),
            steering=float(item.get("steering", 0.0)),
            distance=float(item.get("distance", 0.0)),
            reason=str(item.get("reason", "")),
        )
        for item in data.get("actions", ())
    )


def load_plan(path: str | Path) -> PlanResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlanResult(
        status=data["status"],
        vehicle=data["vehicle"],
        scene=data["scene"],
        start_pose=tuple(data["start_pose"]),
        goal_pose=tuple(data["goal_pose"]),
        states=tuple(tuple(item) for item in data["states"]),
        motions=tuple(
            Motion(
                direction=int(item["direction"]),
                steering=float(item["steering"]),
                distance=float(item["distance"]),
            )
            for item in data["motions"]
        ),
        nodes_expanded=int(data["nodes_expanded"]),
        max_nodes=int(data["max_nodes"]),
        message=data.get("message", ""),
        actions=_actions_from_dict(data),
        scene_signature=str(data.get("scene_signature", "")),
    )


def save_plan(plan: PlanResult, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_states_csv(states: Sequence[Pose], path: str | Path) -> None:
    lines = ["x,y,theta,direction_hint"]
    for index, (x, y, theta) in enumerate(states):
        lines.append(f"{x:.6f},{y:.6f},{theta:.6f},{index}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
