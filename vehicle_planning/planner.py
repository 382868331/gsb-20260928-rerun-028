from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

from .geometry import angle_distance, normalize_angle
from .motion import (
    Primitive,
    endpoint_pose,
    heuristic,
    motion_is_collision_free,
    goal_during_motion,
    standard_primitives,
)
from .plan import Motion, PlanResult, Pose
from .scene import Scene
from .vehicle import Vehicle


class PlannerStatus:
    PATH_FOUND = "path_found"
    NO_PATH = "no_path"
    SEARCH_LIMIT_REACHED = "search_limit_reached"


@dataclass(frozen=True)
class PlannerConfig:
    xy_resolution: float = 0.30
    theta_resolution: float = math.radians(12.0)
    max_expanded_nodes: int = 60_000
    heuristic_weight: float = 1.35
    collision_step: float = 0.075
    switch_gear_penalty: float = 0.15


class _Node:
    __slots__ = ("pose", "parent", "motion", "g", "direction")

    def __init__(
        self,
        pose: Pose,
        parent: "_Node | None",
        motion: Primitive | None,
        g: float,
        direction: int,
    ):
        self.pose = pose
        self.parent = parent
        self.motion = motion
        self.g = g
        self.direction = direction


class Planner:
    def __init__(
        self,
        vehicle: Vehicle,
        scene: Scene,
        primitives: Sequence[Primitive] | None = None,
        config: PlannerConfig | None = None,
    ):
        self.vehicle = vehicle
        self.scene = scene
        self.config = config or PlannerConfig()
        self.primitives = tuple(primitives or standard_primitives(vehicle))

    def _cell(self, pose: Pose) -> tuple[int, int, int]:
        x, y, theta = pose
        return (
            int(round(x / self.config.xy_resolution)),
            int(round(y / self.config.xy_resolution)),
            int(round(normalize_angle(theta) / self.config.theta_resolution)),
        )

    def plan(self) -> PlanResult:
        scene = self.scene
        if scene.footprint_collides(self.vehicle.footprint(*scene.start)):
            return self._failure(PlannerStatus.NO_PATH, 0, "start pose is in collision or outside bounds")
        if scene.footprint_collides(self.vehicle.footprint(*scene.goal)):
            return self._failure(PlannerStatus.NO_PATH, 0, "goal pose is in collision or outside bounds")

        start = scene.start
        root = _Node(start, None, None, 0.0, 1)
        open_heap: list[tuple[float, int, _Node]] = []
        counter = 0
        heapq.heappush(open_heap, (self.config.heuristic_weight * heuristic(start, scene), counter, root))
        best_g = {self._cell(start): 0.0}
        expanded = 0

        while open_heap:
            _, _, node = heapq.heappop(open_heap)
            cell = self._cell(node.pose)
            if best_g.get(cell, math.inf) + 1e-9 < node.g:
                continue
            if scene.at_goal(node.pose):
                return self._success(node, expanded)

            expanded += 1
            if expanded > self.config.max_expanded_nodes:
                return self._failure(
                    PlannerStatus.SEARCH_LIMIT_REACHED,
                    expanded,
                    "search stopped after expanding the configured node limit",
                )

            for primitive in self.primitives:
                if not motion_is_collision_free(
                    self.vehicle,
                    scene,
                    node.pose,
                    primitive,
                    self.config.collision_step,
                ):
                    continue

                goal_hit = goal_during_motion(self.vehicle, scene, node.pose, primitive)
                if goal_hit is not None:
                    goal_pose, goal_distance = goal_hit
                    goal_motion = Primitive(
                        primitive.direction,
                        primitive.steering,
                        goal_distance,
                    )
                    child = _Node(goal_pose, node, goal_motion, node.g + goal_distance, primitive.direction)
                    return self._success(child, expanded)

                next_pose = endpoint_pose(self.vehicle, node.pose, primitive)
                next_cell = self._cell(next_pose)
                gear_penalty = (
                    self.config.switch_gear_penalty
                    if node.motion is not None and node.motion.direction != primitive.direction
                    else 0.0
                )
                next_g = node.g + primitive.distance + gear_penalty
                if next_g + 1e-9 >= best_g.get(next_cell, math.inf):
                    continue
                best_g[next_cell] = next_g
                f = next_g + self.config.heuristic_weight * heuristic(next_pose, scene)
                counter += 1
                heapq.heappush(open_heap, (f, counter, _Node(next_pose, node, primitive, next_g, primitive.direction)))

        return self._failure(PlannerStatus.NO_PATH, expanded, "motion-primitive state space is exhausted")

    def _success(self, node: _Node, expanded: int) -> PlanResult:
        reverse_nodes: list[_Node] = []
        current = node
        while current.parent is not None:
            reverse_nodes.append(current)
            current = current.parent
        reverse_nodes.reverse()

        states = [self.scene.start]
        motions: list[Motion] = []
        for item in reverse_nodes:
            states.append(item.pose)
            assert item.motion is not None
            motions.append(item.motion.as_motion())

        return PlanResult(
            status=PlannerStatus.PATH_FOUND,
            vehicle=self.vehicle.name,
            scene=self.scene.name,
            start_pose=self.scene.start,
            goal_pose=self.scene.goal,
            states=tuple(states),
            motions=tuple(motions),
            nodes_expanded=expanded,
            max_nodes=self.config.max_expanded_nodes,
            message="kinematically feasible path found",
        )

    def _failure(self, status: str, expanded: int, message: str) -> PlanResult:
        return PlanResult(
            status=status,
            vehicle=self.vehicle.name,
            scene=self.scene.name,
            start_pose=self.scene.start,
            goal_pose=self.scene.goal,
            states=tuple(),
            motions=tuple(),
            nodes_expanded=expanded,
            max_nodes=self.config.max_expanded_nodes,
            message=message,
        )
