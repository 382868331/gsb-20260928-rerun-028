from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

from .geometry import normalize_angle
from .motion import (
    Primitive,
    arc_static_collision,
    earliest_clear_departure,
    endpoint_pose,
    goal_during_motion,
    heuristic,
    standard_primitives,
)
from .plan import Motion, PlanResult, Pose, TimedAction
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
    speed: float = 1.0
    gear_shift_time: float = 2.0
    max_wait_time: float = 25.0
    wait_step: float = 0.25
    max_makespan: float = 90.0


class _Node:
    __slots__ = (
        "pose",
        "time",
        "gear",
        "parent",
        "wait",
        "shift",
        "motion",
        "move_start",
        "move_end",
        "g",
    )

    def __init__(
        self,
        pose: Pose,
        time: float,
        gear: int,
        parent: "_Node | None",
        wait: float,
        shift: float,
        motion: Primitive | None,
        move_start: float,
        move_end: float,
        g: float,
    ):
        self.pose = pose
        self.time = time
        self.gear = gear
        self.parent = parent
        self.wait = wait
        self.shift = shift
        self.motion = motion
        self.move_start = move_start
        self.move_end = move_end
        self.g = g


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

    def _cell(self, pose: Pose, gear: int) -> tuple[int, int, int, int]:
        x, y, theta = pose
        return (
            int(round(x / self.config.xy_resolution)),
            int(round(y / self.config.xy_resolution)),
            int(round(normalize_angle(theta) / self.config.theta_resolution)),
            gear,
        )

    def plan(self) -> PlanResult:
        scene = self.scene
        config = self.config
        if scene.footprint_collides(self.vehicle.footprint(*scene.start)):
            return self._failure(PlannerStatus.NO_PATH, 0, "start pose is in collision or outside bounds")
        if scene.footprint_collides(self.vehicle.footprint(*scene.goal)):
            return self._failure(PlannerStatus.NO_PATH, 0, "goal pose is in collision or outside bounds")

        start = scene.start
        root = _Node(start, 0.0, 1, None, 0.0, 0.0, None, 0.0, 0.0, 0.0)
        open_heap: list[tuple[float, int, _Node]] = []
        counter = 0
        heapq.heappush(open_heap, (config.heuristic_weight * heuristic(start, scene), counter, root))
        # Earlier arrival at the same discretized pose/gear dominates later arrival.
        best_time = {self._cell(start, 1): 0.0}
        expanded = 0

        while open_heap:
            _, _, node = heapq.heappop(open_heap)
            if best_time.get(self._cell(node.pose, node.gear), math.inf) + 1e-9 < node.time:
                continue
            if scene.at_goal(node.pose):
                return self._success(node, expanded)

            expanded += 1
            if expanded > config.max_expanded_nodes:
                return self._failure(
                    PlannerStatus.SEARCH_LIMIT_REACHED,
                    expanded,
                    "search stopped after expanding the configured node limit",
                )

            for primitive in self.primitives:
                if arc_static_collision(
                    self.vehicle, scene, node.pose, primitive, config.collision_step
                ):
                    continue

                goal_hit = goal_during_motion(self.vehicle, scene, node.pose, primitive)
                candidate = primitive
                if goal_hit is not None:
                    goal_pose, goal_distance = goal_hit
                    candidate = Primitive(primitive.direction, primitive.steering, goal_distance)
                else:
                    goal_pose = endpoint_pose(self.vehicle, node.pose, primitive)

                shift = config.gear_shift_time if node.gear != candidate.direction else 0.0
                departure, blockers = earliest_clear_departure(
                    self.vehicle,
                    scene,
                    node.pose,
                    candidate,
                    node.time,
                    config.speed,
                    shift_duration=shift,
                    max_wait=config.max_wait_time,
                    time_step=config.wait_step,
                    collision_step=config.collision_step,
                )
                if departure is None:
                    continue
                move_start = departure + shift
                move_end = move_start + candidate.distance / config.speed  # noqa: E501
                if move_end > config.max_makespan:
                    continue

                wait = departure - node.time
                child = _Node(
                    goal_pose,
                    move_end,
                    candidate.direction,
                    node,
                    wait,
                    shift,
                    candidate,
                    move_start,
                    move_end,
                    node.g + wait + shift + candidate.distance,
                )
                if goal_hit is not None:
                    return self._success(child, expanded)
                next_cell = self._cell(goal_pose, candidate.direction)
                if move_end + 1e-9 >= best_time.get(next_cell, math.inf):
                    continue
                best_time[next_cell] = move_end
                f = move_end + config.heuristic_weight * heuristic(goal_pose, scene)
                counter += 1
                heapq.heappush(open_heap, (f, counter, child))

        return self._failure(PlannerStatus.NO_PATH, expanded, "timed motion-primitive state space is exhausted")

    def _success(self, node: _Node, expanded: int) -> PlanResult:
        reverse_nodes: list[_Node] = []
        current = node
        while current.parent is not None:
            reverse_nodes.append(current)
            current = current.parent
        reverse_nodes.reverse()

        states = [self.scene.start]
        motions: list[Motion] = []
        timeline: list[TimedAction] = []
        for edge in reverse_nodes:
            assert edge.motion is not None
            if edge.wait > 1e-9:
                blockers = self._wait_blockers(edge)
                timeline.append(
                    TimedAction(
                        kind="wait",
                        t0=round(edge.move_start - edge.shift - edge.wait, 6),
                        t1=round(edge.move_start - edge.shift, 6),
                        direction=0,
                        steering=0.0,
                        distance=0.0,
                        reason=(
                            "hold position for passing dynamic obstacle(s): "
                            + ", ".join(blockers)
                        ),
                    )
                )
            if edge.shift > 1e-9:
                timeline.append(
                    TimedAction(
                        kind="shift",
                        t0=round(edge.move_start - edge.shift, 6),
                        t1=round(edge.move_start, 6),
                        direction=edge.motion.direction,
                        steering=0.0,
                        distance=0.0,
                        reason=(
                            ("forward to reverse" if edge.motion.direction == -1 else "reverse to forward")
                            + " gear transition"
                        ),
                    )
                )
            timeline.append(
                TimedAction(
                    kind="drive",
                    t0=round(edge.move_start, 6),
                    t1=round(edge.move_end, 6),
                    direction=edge.motion.direction,
                    steering=edge.motion.steering,
                    distance=edge.motion.distance,
                    reason=("reverse" if edge.motion.direction == -1 else "forward")
                    + " driving",
                )
            )
            states.append(edge.pose)
            motions.append(edge.motion.as_motion())

        makespan = round(node.time, 6)
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
            message="kinematically feasible timed path found",
            timeline=tuple(timeline),
            timetable_signature=self.scene.timetable_signature(),
            makespan=makespan,
        )

    def _wait_blockers(self, edge: _Node) -> list[str]:
        assert edge.parent is not None and edge.motion is not None
        from .motion import blocking_obstacles

        names = blocking_obstacles(
            self.vehicle,
            self.scene,
            edge.parent.pose,
            None,
            edge.move_start - edge.shift - edge.wait,
        )
        if not names:
            names = blocking_obstacles(
                self.vehicle,
                self.scene,
                edge.parent.pose,
                edge.motion,
                edge.move_start - edge.shift - edge.wait,
                self.config.speed,
            )
        return names or ["dynamic_obstacle"]

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
            timetable_signature=self.scene.timetable_signature(),
        )
