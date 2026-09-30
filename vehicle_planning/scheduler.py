"""Turn a kinematic path into a timed control schedule.

The geometric planner ignores time. The scheduler replays that path at a low
cruise speed and inserts stationary control actions:

* wait: a temporary object occupies the corridor at the current time;
* gear_switch: every forward/reverse direction change costs transition
  time while the vehicle remains stopped.

At every moment the full vehicle rectangle is checked against static geometry
and against the interpolated positions of scheduled obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .motion import sample_arc
from .plan import Action, PlanResult
from .scene import Scene
from .vehicle import Vehicle


class SchedulerStatus:
    PATH_FOUND = "path_found"
    SCHEDULE_FAILED = "schedule_failed"


@dataclass(frozen=True)
class SchedulerConfig:
    cruise_speed: float = 1.2
    collision_step: float = 0.08
    wait_step: float = 0.2
    gear_switch_time: float = 1.5
    max_wait: float = 60.0
    time_collision_step: float = 0.1


@dataclass
class _ScheduleFailure(Exception):
    message: str


def _wait_reason(names, seconds):
    joined = ", ".join(names)
    return f"通道被动态障碍 {joined} 临时占用，原地等待 {seconds:.1f}s 后继续"


class Scheduler:
    def __init__(self, vehicle, scene, config=None):
        self.vehicle = vehicle
        self.scene = scene
        self.config = config or SchedulerConfig()

    def schedule(self, plan):
        if not plan.found:
            return plan
        cfg = self.config

        pose = plan.start_pose
        time_now = 0.0
        gear = 1
        actions = []
        try:
            for motion_index, motion in enumerate(plan.motions):
                if motion.direction != gear:
                    time_now = self._wait_if_blocked(
                        pose,
                        time_now,
                        "换挡停车点被动态障碍占用",
                        actions,
                    )
                    actions.append(
                        Action(
                            type="gear_switch",
                            start_time=time_now,
                            end_time=time_now + cfg.gear_switch_time,
                            direction=motion.direction,
                            reason=(
                                "前进切倒车，计入换挡过渡时间"
                                if gear == 1
                                else "倒车切前进，计入换挡过渡时间"
                            ),
                        )
                    )
                    time_now += cfg.gear_switch_time
                    gear = motion.direction

                pose, time_now = self._schedule_drive(
                    pose,
                    time_now,
                    motion.direction,
                    motion.steering,
                    motion.distance,
                    actions,
                )

                saved = plan.states[motion_index + 1]
                if math.hypot(pose[0] - saved[0], pose[1] - saved[1]) > 1e-6:
                    raise _ScheduleFailure("scheduled drive drifted from saved state")
        except _ScheduleFailure as failure:
            return self._failure(plan, failure.message)

        return PlanResult(
            status=SchedulerStatus.PATH_FOUND,
            vehicle=plan.vehicle,
            scene=plan.scene,
            start_pose=plan.start_pose,
            goal_pose=plan.goal_pose,
            states=plan.states,
            motions=plan.motions,
            nodes_expanded=plan.nodes_expanded,
            max_nodes=plan.max_nodes,
            message=f"timed schedule found, total time {time_now:.2f}s",
            actions=tuple(actions),
            scene_signature=self.scene.signature(),
        )

    def _failure(self, plan, message):
        return PlanResult(
            status=SchedulerStatus.SCHEDULE_FAILED,
            vehicle=plan.vehicle,
            scene=plan.scene,
            start_pose=plan.start_pose,
            goal_pose=plan.goal_pose,
            states=plan.states,
            motions=plan.motions,
            nodes_expanded=plan.nodes_expanded,
            max_nodes=plan.max_nodes,
            message=message,
            actions=(),
            scene_signature=self.scene.signature(),
        )

    def _wait_if_blocked(self, pose, time_now, context, actions):
        """Stay stopped until the gear-switch hold window is clear."""
        cfg = self.config
        footprint = self.vehicle.footprint(*pose)
        if self.scene.footprint_collides(footprint):
            raise _ScheduleFailure("stationary pose is in static collision")

        waited = 0.0
        start = time_now
        names = set()
        while True:
            probe = time_now
            window_clear = True
            while probe <= time_now + cfg.gear_switch_time + 1e-9:
                blocking = self.scene.blocking_dynamic_obstacles(footprint, probe)
                if blocking:
                    window_clear = False
                    names.update(blocking)
                probe += cfg.time_collision_step
            if window_clear:
                break
            time_now += cfg.wait_step
            waited += cfg.wait_step
            if waited > cfg.max_wait:
                raise _ScheduleFailure(
                    f"dynamic obstacle never leaves within {cfg.max_wait:.0f}s"
                )

        if waited > 0.0:
            label = ", ".join(sorted(names))
            actions.append(
                Action(
                    type="wait",
                    start_time=start,
                    end_time=time_now,
                    reason=(
                        f"{context}（障碍：{label}），等待 {waited:.1f}s 再执行过渡动作"
                    ),
                )
            )
        return time_now

    def _schedule_drive(self, start_pose, start_time, direction, steering, distance, actions):
        cfg = self.config
        arc = sample_arc(
            start_pose,
            direction,
            steering,
            distance,
            self.vehicle.wheelbase,
            cfg.collision_step,
        )
        step_length = distance / (len(arc) - 1)
        footprints = [self.vehicle.footprint(*sampled) for sampled in arc]
        for footprint in footprints:
            if self.scene.footprint_collides(footprint):
                raise _ScheduleFailure("scheduled sweep is in static collision")

        n = len(arc)
        hold_footprint = footprints[0]

        def wait_window_blocked(begin, end):
            steps = max(1, int(math.ceil((end - begin) / cfg.time_collision_step)))
            names = set()
            for j in range(steps + 1):
                at_time = begin + (end - begin) * j / steps
                names.update(
                    self.scene.blocking_dynamic_obstacles(hold_footprint, at_time)
                )
            return names

        def sweep_blocked(release_time):
            names = set()
            for k in range(n):
                at_time = release_time + k * step_length / cfg.cruise_speed
                names.update(
                    self.scene.blocking_dynamic_obstacles(footprints[k], at_time)
                )
            return names

        wait_start = start_time
        delay = 0.0
        names = set()
        while True:
            release_time = wait_start + delay
            blocking = wait_window_blocked(wait_start, release_time)
            blocking.update(sweep_blocked(release_time))
            if not blocking:
                break
            names.update(blocking)
            delay += cfg.wait_step
            if delay > cfg.max_wait:
                raise _ScheduleFailure(
                    f"dynamic obstacle never leaves the corridor within {cfg.max_wait:.0f}s"
                )

        clock = release_time
        if delay > 1e-9:
            actions.append(
                Action(
                    type="wait",
                    start_time=wait_start,
                    end_time=clock,
                    reason=_wait_reason(sorted(names), delay),
                )
            )

        drive_time = (n - 1) * step_length / cfg.cruise_speed
        actions.append(
            Action(
                type="drive",
                start_time=clock,
                end_time=clock + drive_time,
                direction=direction,
                steering=steering,
                distance=distance,
            )
        )
        return arc[-1], clock + drive_time
