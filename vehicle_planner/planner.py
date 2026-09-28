from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .collision import command_collision_free
from .geometry import angle_distance, wrap_angle
from .kinematics import integrate
from .models import MotionCommand, Plan, Pose, Scenario, Vehicle


@dataclass(frozen=True)
class SearchLimits:
    max_expansions: int = 60_000
    max_queue_size: int = 200_000
    collision_sample_step: float = 0.05


@dataclass
class PlanningResult:
    status: str
    commands: list[MotionCommand] = field(default_factory=list)
    expansions: int = 0
    goal_pose: Pose | None = None
    reason: str = ""


class MotionPlanner:
    def __init__(
        self,
        vehicle: Vehicle,
        heading_bins: int = 24,
        position_resolution: float = 0.25,
        straight_length: float = 0.8,
        gear_switch_cost: float = 4.0,
        steering_switch_cost: float = 0.25,
        limits: SearchLimits | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.heading_bins = heading_bins
        self.position_resolution = position_resolution
        self.straight_length = straight_length
        self.gear_switch_cost = gear_switch_cost
        self.steering_switch_cost = steering_switch_cost
        self.limits = limits or SearchLimits()
        self.heading_step = 2.0 * math.pi / heading_bins
        self.arc_duration = self.heading_step * vehicle.min_turning_radius

    def plan(self, scenario: Scenario) -> PlanningResult:
        start = scenario.start
        start_key = self._node_key(start)
        queue = [(self._heuristic(start, scenario.goal), 0, start_key)]
        counter = 0
        scores = {start_key: 0.0}
        parents = {start_key: (None, MotionCommand(0.0, 1.0, 0.0), start, 1, 0)}
        expansions = 0

        while queue and expansions < self.limits.max_expansions:
            _, _, key = heapq.heappop(queue)
            _, _, pose, gear, steering_index = parents[key]
            current_score = scores[key]
            expansions += 1

            if self._at_goal(pose, scenario):
                return PlanningResult(
                    status="solved",
                    commands=self._reconstruct(parents, key),
                    expansions=expansions,
                    goal_pose=pose,
                )

            for next_pose, command, next_gear, next_steering, edge_cost in self._neighbors(
                pose, gear, steering_index, scenario
            ):
                next_key = self._node_key(next_pose)
                next_score = current_score + edge_cost
                if next_score >= scores.get(next_key, math.inf):
                    continue
                scores[next_key] = next_score
                parents[next_key] = (key, command, next_pose, next_gear, next_steering)
                priority = next_score + self._heuristic(next_pose, scenario.goal)
                counter += 1
                heapq.heappush(queue, (priority, counter, next_key))
                if len(queue) > self.limits.max_queue_size:
                    return PlanningResult(
                        status="search_incomplete",
                        expansions=expansions,
                        reason="search queue size limit reached",
                    )

        if expansions >= self.limits.max_expansions:
            return PlanningResult(
                status="search_incomplete",
                expansions=expansions,
                reason="node expansion limit reached before proving no path",
            )
        return PlanningResult(status="no_path", expansions=expansions, reason="search space exhausted")

    def make_plan(self, scenario: Scenario) -> Plan:
        result = self.plan(scenario)
        return Plan(
            vehicle_name=self.vehicle.name,
            scenario_name=scenario.name,
            status=result.status,
            commands=result.commands,
            start=scenario.start,
            goal=scenario.goal,
            metadata={
                "expansions": result.expansions,
                "reason": result.reason,
                "goal_pose": result.goal_pose.__dict__ if result.goal_pose else None,
            },
        )

    def _neighbors(self, pose: Pose, gear: int, steering_index: int, scenario: Scenario):
        steerings = (-self.vehicle.max_steering_angle, 0.0, self.vehicle.max_steering_angle)
        durations = (self.arc_duration, self.straight_length, self.arc_duration)
        for velocity in (-1.0, 1.0):
            next_gear = 1 if velocity > 0 else -1
            for index, (steering, duration) in enumerate(zip(steerings, durations)):
                command = MotionCommand(steering, velocity, duration)
                if not command_collision_free(
                    pose, command, self.vehicle, scenario, self.limits.collision_sample_step
                ):
                    continue
                next_pose = integrate(pose, command, self.vehicle)
                edge_cost = duration
                if next_gear != gear:
                    edge_cost += self.gear_switch_cost
                if index != steering_index:
                    edge_cost += self.steering_switch_cost
                yield next_pose, command, next_gear, index, edge_cost

    def _node_key(self, pose: Pose) -> str:
        heading_bin = int(round(wrap_angle(pose.theta) / self.heading_step)) % self.heading_bins
        x_bin = int(math.floor(pose.x / self.position_resolution))
        y_bin = int(math.floor(pose.y / self.position_resolution))
        return f"{x_bin},{y_bin},{heading_bin}"

    def _at_goal(self, pose: Pose, scenario: Scenario) -> bool:
        goal = scenario.goal
        tolerance = scenario.goal_tolerance
        distance = math.hypot(pose.x - goal.x, pose.y - goal.y)
        return (
            distance <= tolerance.position
            and angle_distance(pose.theta, goal.theta) <= tolerance.heading
        )

    @staticmethod
    def _heuristic(pose: Pose, goal: Pose) -> float:
        position = math.hypot(pose.x - goal.x, pose.y - goal.y)
        heading = angle_distance(pose.theta, goal.theta)
        return position + 0.05 * heading

    @staticmethod
    def _reconstruct(parents, key: str) -> list[MotionCommand]:
        commands = []
        while True:
            parent_key, command, _, _, _ = parents[key]
            if parent_key is None:
                break
            commands.append(command)
            key = parent_key
        commands.reverse()
        return commands
