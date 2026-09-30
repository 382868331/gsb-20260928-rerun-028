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
class TimedAction:
    """A control action placed on a global clock.

    ``kind`` is one of:
      * ``drive``  - execute ``motion`` from ``t0`` to ``t1``;
      * ``shift``  - stationary forward/reverse (or reverse/forward) transition;
      * ``wait``   - hold position for a low-speed obstacle to pass.
    """

    kind: str
    t0: float
    t1: float
    direction: int
    steering: float
    distance: float
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "t0": round(self.t0, 6),
            "t1": round(self.t1, 6),
            "duration": round(self.duration, 6),
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
    timeline: tuple[TimedAction, ...] = ()
    timetable_signature: str = ""
    makespan: float = 0.0

    @property
    def found(self) -> bool:
        return self.status == "path_found"

    @property
    def total_distance(self) -> float:
        return sum(abs(item.distance) for item in self.motions)

    @property
    def waits(self) -> tuple[TimedAction, ...]:
        return tuple(item for item in self.timeline if item.kind == "wait")

    def to_dict(self) -> dict:
        data = {
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
        }
        if self.timeline:
            data["timetable_signature"] = self.timetable_signature
            data["makespan"] = round(self.makespan, 6)
            data["timeline"] = [item.to_dict() for item in self.timeline]
        return data


def _action_from_dict(item: dict) -> TimedAction:
    return TimedAction(
        kind=item["kind"],
        t0=float(item["t0"]),
        t1=float(item["t1"]),
        direction=int(item["direction"]),
        steering=float(item["steering"]),
        distance=float(item["distance"]),
        reason=item.get("reason", ""),
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
        timeline=tuple(_action_from_dict(item) for item in data.get("timeline", [])),
        timetable_signature=data.get("timetable_signature", ""),
        makespan=float(data.get("makespan", 0.0)),
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
