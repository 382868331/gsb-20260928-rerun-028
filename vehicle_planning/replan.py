"""Non-destructive replanning after the obstacle timetable changes.

A timetable change can never be "re-validated away": the saved plan keeps its
old signature and success record. This module offers three clearly separated
alternatives against the *new* timetable:

* original: same route, same timing; evaluated read-only, with concrete
  conflict time intervals reported. It never overwrites the saved record.
* delayed: same route; only the departure / waiting timing is shifted.
* replanned: a brand-new geometric route scheduled for the new timetable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .motion import sample_arc
from .plan import Action
from .plan import PlanResult
from .planner import Planner, PlannerConfig
from .scheduler import Scheduler, SchedulerConfig, SchedulerStatus
from .scene import Scene
from .vehicle import Vehicle


@dataclass(frozen=True)
class ConflictInterval:
    """One continuous time range where the old schedule collides.

    Consecutive drive / wait / gear-switch actions touching each other in time
    for the same obstacle are merged into one episode; ``actions`` records all
    involved action indices and their types.
    """

    start_time: float
    end_time: float
    obstacle: str
    actions: tuple[tuple[int, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "obstacle": self.obstacle,
            "actions": [[index, kind] for index, kind in self.actions],
        }

    @property
    def action_index(self) -> int:
        return self.actions[0][0] if self.actions else -1

    @property
    def action_type(self) -> str:
        return ",".join(kind for _, kind in self.actions)


@dataclass(frozen=True)
class Alternative:
    """One of original / delayed / replanned alternatives."""

    key: str
    feasible: bool
    route_changed: bool
    timing_changed: bool
    plan: PlanResult | None = None
    delay: float = 0.0
    conflicts: tuple[ConflictInterval, ...] = ()
    message: str = ""

    @property
    def status(self) -> str:
        if self.plan is None:
            return SchedulerStatus.SCHEDULE_FAILED
        return self.plan.status

    @property
    def final_time(self) -> float:
        return self.plan.final_time if self.plan is not None else math.inf


@dataclass(frozen=True)
class ReplanReport:
    """Comparison of the three alternatives for one timetable revision."""

    old_timetable_signature: str
    new_timetable_signature: str
    timetable_changed: bool
    original: Alternative
    delayed: Alternative | None
    replanned: Alternative | None
    recommended: str
    message: str

    def to_dict(self) -> dict:
        def alt(alternative):
            if alternative is None:
                return None
            return {
                "key": alternative.key,
                "feasible": alternative.feasible,
                "route_changed": alternative.route_changed,
                "timing_changed": alternative.timing_changed,
                "delay": alternative.delay,
                "final_time": alternative.final_time,
                "conflicts": [item.to_dict() for item in alternative.conflicts],
                "message": alternative.message,
            }

        return {
            "old_timetable_signature": self.old_timetable_signature,
            "new_timetable_signature": self.new_timetable_signature,
            "timetable_changed": self.timetable_changed,
            "original": alt(self.original),
            "delayed": alt(self.delayed),
            "replanned": alt(self.replanned),
            "recommended": self.recommended,
            "message": self.message,
        }


def timetable_changed(old_plan: PlanResult, scene: Scene) -> bool:
    """Any change to the obstacle timetable blocks plain re-validation."""

    if not old_plan.timetable_signature:
        return True
    return old_plan.timetable_signature != scene.timetable_signature()


def _merge_conflicts(raw, step):
    if not raw:
        return ()
    raw.sort(key=lambda item: (item[0], item[1]))
    intervals = []
    # First bound the collision inside each action by (action, obstacle).
    grouped: dict[tuple[int, str], dict] = {}
    for at_time, obstacle, index, kind in raw:
        key = (index, obstacle)
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                "start": at_time,
                "end": at_time,
                "index": index,
                "kind": kind,
                "name": obstacle,
            }
        else:
            entry["start"] = min(entry["start"], at_time)
            entry["end"] = max(entry["end"], at_time)
    for entry in grouped.values():
        intervals.append(
            ConflictInterval(
                entry["start"],
                entry["end"],
                entry["name"],
                ((entry["index"], entry["kind"]),),
            )
        )
    intervals.sort(key=lambda item: (item.start_time, item.obstacle))

    # Merge touching intervals for the same obstacle across consecutive actions.
    merged: list[ConflictInterval] = []
    for interval in intervals:
        if (
            merged
            and merged[-1].obstacle == interval.obstacle
            and interval.start_time <= merged[-1].end_time + step + 1e-9
        ):
            previous = merged[-1]
            merged[-1] = ConflictInterval(
                start_time=previous.start_time,
                end_time=max(previous.end_time, interval.end_time),
                obstacle=previous.obstacle,
                actions=tuple(previous.actions) + tuple(interval.actions),
            )
        else:
            merged.append(interval)
    return tuple(merged)


def evaluate_original_schedule(
    old_plan: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    sample_step: float = 0.1,
) -> Alternative:
    """Read-only check of the saved route with its saved timing.

    The saved plan object is never mutated and no success marker is reused or
    overwritten. Conflicts are returned as concrete time intervals.
    """

    if not old_plan.timed:
        return Alternative(
            key="original",
            feasible=False,
            route_changed=False,
            timing_changed=False,
            conflicts=(),
            message="saved plan has no timed schedule; it cannot be re-validated against a timetable",
        )

    raw = []
    pose = old_plan.start_pose
    for index, action in enumerate(old_plan.actions):
        duration = action.end_time - action.start_time
        if action.type in ("wait", "gear_switch"):
            probes = max(1, int(math.ceil(duration / sample_step)))
            for sub in range(probes + 1):
                at_time = action.start_time + duration * sub / probes
                for obs_name in scene.blocking_dynamic_obstacles(vehicle.footprint(*pose), at_time):
                    raw.append((at_time, obs_name, index, action.type))
            continue

        arc = sample_arc(
            pose,
            action.direction,
            action.steering,
            action.distance,
            vehicle.wheelbase,
        )
        for sub_index, sampled in enumerate(arc):
            fraction = sub_index / max(1, len(arc) - 1)
            at_time = action.start_time + duration * fraction
            for obs_name in scene.blocking_dynamic_obstacles(vehicle.footprint(*sampled), at_time):
                raw.append((at_time, obs_name, index, "drive"))
        pose = arc[-1]

    conflicts = _merge_conflicts(raw, sample_step)
    feasible = not conflicts
    if feasible:
        message = (
            "原路线与原时间安排在新时间表下仍可执行；障碍已完全避开路线，"
            "但原成功记录保持不变，本结论仅为只读评估"
        )
    else:
        first = conflicts[0]
        message = (
            "原时间安排在新时间表下不可执行，首个冲突区间 "
            f"t={first.start_time:.2f}-{first.end_time:.2f}s（{first.obstacle}）；"
            "必须重新规划，不能单独重新验证，原成功记录不被改写"
        )
    return Alternative(
        key="original",
        feasible=feasible,
        route_changed=False,
        timing_changed=False,
        plan=old_plan,
        conflicts=conflicts,
        message=message,
    )


def _reschedule_config(cfg: SchedulerConfig, variant: str, initial_delay: float = 0.0):
    return SchedulerConfig(
        cruise_speed=cfg.cruise_speed,
        collision_step=cfg.collision_step,
        wait_step=cfg.wait_step,
        gear_switch_time=cfg.gear_switch_time,
        max_wait=cfg.max_wait,
        time_collision_step=cfg.time_collision_step,
        initial_delay=initial_delay,
        variant=variant,
    )


def _shifted_plan(old_plan: PlanResult, delay: float, variant: str, scene: Scene) -> PlanResult:
    shifted = tuple(
        Action(
            type=action.type,
            start_time=action.start_time + delay,
            end_time=action.end_time + delay,
            direction=action.direction,
            steering=action.steering,
            distance=action.distance,
            reason=action.reason,
        )
        for action in old_plan.actions
    )
    actions = (
        Action(
            type="wait",
            start_time=0.0,
            end_time=delay,
            reason=f"障碍时间表变化，整体延后 {delay:.1f}s 出发，路线保持不变",
        ),
    ) + shifted
    return PlanResult(
        status=old_plan.status,
        vehicle=old_plan.vehicle,
        scene=old_plan.scene,
        start_pose=old_plan.start_pose,
        goal_pose=old_plan.goal_pose,
        states=old_plan.states,
        motions=old_plan.motions,
        nodes_expanded=old_plan.nodes_expanded,
        max_nodes=old_plan.max_nodes,
        message=old_plan.message,
        actions=actions,
        scene_signature=scene.signature(),
        route_signature=old_plan.route_signature,
        timetable_signature=scene.timetable_signature(),
        variant=variant,
    )


def _try_pure_departure_delay(
    old_plan: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    cfg: SchedulerConfig,
) -> PlanResult | None:
    """Keep every relative action time; only shift the whole departure later."""

    probe = evaluate_original_schedule(old_plan, vehicle, scene, cfg.time_collision_step)
    if probe.feasible:
        return old_plan

    delay = 0.0
    while delay <= cfg.max_wait + 1e-9:
        delay += cfg.wait_step
        candidate = _shifted_plan(old_plan, delay, "delayed", scene)
        check = evaluate_original_schedule(candidate, vehicle, scene, cfg.time_collision_step)
        if check.feasible:
            return candidate
    return None


def try_delayed_schedule(
    old_plan: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    config: SchedulerConfig | None = None,
) -> Alternative:
    """Keep the geometric route fixed; allow departure/wait timing to shift."""

    cfg = config or SchedulerConfig()
    pure_delay = _try_pure_departure_delay(old_plan, vehicle, scene, cfg)
    if pure_delay is not None and pure_delay is not old_plan:
        candidate = pure_delay
        delay_strategy = "departure_delay"
    else:
        candidate = Scheduler(vehicle, scene, _reschedule_config(cfg, "delayed")).schedule(old_plan)
        delay_strategy = "segment_waits"
    if candidate.status != SchedulerStatus.PATH_FOUND:
        return Alternative(
            key="delayed",
            feasible=False,
            route_changed=False,
            timing_changed=True,
            plan=None,
            conflicts=(),
            message=f"保持原路线、仅调整出发/等待时间仍无法避开：{candidate.message}",
        )

    delay = candidate.final_time - old_plan.final_time
    timing_changed = any(
        new_item.start_time != old_item.start_time
        or new_item.end_time != old_item.end_time
        or new_item.type != old_item.type
        for new_item, old_item in zip(candidate.actions, old_plan.actions)
    ) or len(candidate.actions) != len(old_plan.actions)
    message = (
        f"保持原路线，仅调整时间安排即可执行（{delay_strategy}，时间轴变化约 {delay:.2f}s），"
        "route_signature 不变，timetable_signature 指向新时间表"
    )
    return Alternative(
        key="delayed",
        feasible=True,
        route_changed=False,
        timing_changed=timing_changed or abs(delay) > 1e-9,
        plan=candidate,
        delay=delay,
        conflicts=(),
        message=message,
    )


def replan_route(
    vehicle: Vehicle,
    scene: Scene,
    primitives,
    planner_config: PlannerConfig | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> Alternative:
    """Search a new geometric route and schedule it on the new timetable."""

    geometric = Planner(
        vehicle, scene, primitives, planner_config or PlannerConfig()
    ).plan()
    if not geometric.found:
        return Alternative(
            key="replanned",
            feasible=False,
            route_changed=True,
            timing_changed=True,
            plan=None,
            message=f"新时间表下找不到新的行驶路径：{geometric.message}",
        )
    cfg = scheduler_config or SchedulerConfig()
    scheduled = Scheduler(
        vehicle, scene, _reschedule_config(cfg, "replanned")
    ).schedule(geometric)
    if scheduled.status != SchedulerStatus.PATH_FOUND:
        return Alternative(
            key="replanned",
            feasible=False,
            route_changed=True,
            timing_changed=True,
            plan=None,
            message=f"新路径存在但无法排出时间方案：{scheduled.message}",
        )
    return Alternative(
        key="replanned",
        feasible=True,
        route_changed=True,
        timing_changed=True,
        plan=scheduled,
        message="已改变行驶路径，并针对新时间表重新排定时间",
    )


def compare_alternatives(
    old_plan: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    primitives,
    scheduler_config: SchedulerConfig | None = None,
    planner_config: PlannerConfig | None = None,
    allow_replan: bool = True,
) -> ReplanReport:
    """Evaluate original, delayed and replanned alternatives read-only."""

    old_timetable = old_plan.timetable_signature
    new_timetable = scene.timetable_signature()
    changed = timetable_changed(old_plan, scene)

    original = evaluate_original_schedule(old_plan, vehicle, scene)
    delayed = None
    replanned = None

    if changed:
        delayed = try_delayed_schedule(old_plan, vehicle, scene, scheduler_config)
        if not delayed.feasible and allow_replan:
            replanned = replan_route(
                vehicle, scene, primitives, planner_config, scheduler_config
            )

    if original.feasible:
        recommended = "original"
    elif delayed is not None and delayed.feasible:
        recommended = "delayed"
    elif replanned is not None and replanned.feasible:
        recommended = "replanned"
    else:
        recommended = "none"

    if not changed:
        message = "时间表未变化，沿用已保存方案"
    elif recommended == "original":
        message = "障碍已远离/完全避开路线，原方案只读评估通过；记录不改动"
    elif recommended == "delayed":
        message = "推荐仅调整出发/等待时间，路线不变"
    elif recommended == "replanned":
        message = "时间调整无效，推荐改用重新规划的行驶路径"
    else:
        message = "三种方案均无法在新时间表下执行"

    return ReplanReport(
        old_timetable_signature=old_timetable,
        new_timetable_signature=new_timetable,
        timetable_changed=changed,
        original=original,
        delayed=delayed,
        replanned=replanned,
        recommended=recommended,
        message=message,
    )
