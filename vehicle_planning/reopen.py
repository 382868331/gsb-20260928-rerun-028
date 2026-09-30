"""Reopen an already successful plan after the obstacle timetable changes.

A saved plan is bound to one scene signature. When the timetable changes the
old route cannot simply be re-validated: the whole planning question is
reopened against the new timetable. This module keeps the original success
record untouched and evaluates three clearly separated options:

* original schedule on the new timetable (same route, same timing);
* delayed execution (same route, shifted departure time);
* full re-plan (a fresh geometric route scheduled on the new timetable).

Route changes (motions/geometry) and timing changes (departure shift, waits,
final time) are reported independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .motion import sample_arc
from .plan import Action, PlanResult, load_plan, save_plan
from .replay import replay
from .scheduler import Scheduler, SchedulerConfig, SchedulerStatus
from .scene import Scene, load_scene
from .vehicle import Vehicle, load_vehicle


@dataclass(frozen=True)
class ConflictInterval:
    """One continuous time window where the original route is blocked."""

    start_time: float
    end_time: float
    obstacles: tuple[str, ...]
    context: str
    samples: int

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.end_time - self.start_time, 3),
            "obstacles": list(self.obstacles),
            "context": self.context,
            "samples": self.samples,
        }


def _timed_footprints(plan: PlanResult, vehicle: Vehicle, collision_step: float = 0.05):
    """Yield (time, footprint, context) samples along a saved timed plan."""
    pose = plan.start_pose
    yield 0.0, vehicle.footprint(*pose), "start"
    for action in plan.actions:
        if action.type in ("wait", "gear_switch"):
            steps = max(1, int(round((action.end_time - action.start_time) / 0.1)))
            for index in range(1, steps + 1):
                at_time = action.start_time + (action.end_time - action.start_time) * index / steps
                yield at_time, vehicle.footprint(*pose), action.type
            continue
        if action.type != "drive":
            raise ValueError(f"unknown action type {action.type}")
        arc = sample_arc(
            pose,
            action.direction,
            action.steering,
            action.distance,
            vehicle.wheelbase,
            collision_step,
        )
        duration = action.end_time - action.start_time
        for index, sampled in enumerate(arc):
            fraction = index / (len(arc) - 1)
            at_time = action.start_time + duration * fraction
            yield at_time, vehicle.footprint(*sampled), "drive sweep"
        pose = arc[-1]


def scan_conflicts(
    plan: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    merge_gap: float = 0.25,
) -> list[ConflictInterval]:
    """Return concrete conflict intervals of a timed route on a new timetable.

    The route itself is never modified; static collisions are reported with an
    empty obstacle list, moving-obstacle collisions with the obstacle names.
    Neighbouring samples within merge_gap seconds merge into one window.
    """
    intervals: list[ConflictInterval] = []
    current_start = None
    current_end = 0.0
    current_names: set[str] = set()
    current_context = ""
    current_samples = 0

    def flush():
        if current_start is None:
            return
        intervals.append(
            ConflictInterval(
                current_start,
                current_end,
                tuple(sorted(current_names)),
                current_context,
                current_samples,
            )
        )

    previous_time = None
    for at_time, footprint, context in _timed_footprints(plan, vehicle):
        names = scene.blocking_dynamic_obstacles(footprint, at_time)
        blocked = bool(names) or scene.footprint_collides(footprint)
        if blocked:
            if current_start is None or previous_time is None or at_time - current_end > merge_gap:
                flush()
                current_start = at_time
                current_names = set()
                current_samples = 0
            current_end = at_time
            current_names.update(names)
            current_context = context
            current_samples += 1
        previous_time = at_time
    flush()
    return intervals


def _shift_actions(actions: tuple[Action, ...], delay: float, reason: str) -> tuple[Action, ...]:
    if delay <= 1e-9:
        return actions
    shifted = [
        replace(action, start_time=action.start_time + delay, end_time=action.end_time + delay)
        for action in actions
    ]
    shifted.insert(
        0,
        Action(type="wait", start_time=0.0, end_time=delay, reason=reason),
    )
    return tuple(shifted)


def same_route(plan_a: PlanResult, plan_b: PlanResult, tolerance: float = 1e-6) -> bool:
    """True when both plans use identical geometric motions (ignoring timing)."""
    if len(plan_a.motions) != len(plan_b.motions):
        return False
    for motion_a, motion_b in zip(plan_a.motions, plan_b.motions):
        if motion_a.direction != motion_b.direction:
            return False
        if abs(motion_a.steering - motion_b.steering) > tolerance:
            return False
        if abs(motion_a.distance - motion_b.distance) > tolerance:
            return False
    return True


def same_timing(plan_a: PlanResult, plan_b: PlanResult, tolerance: float = 1e-6) -> bool:
    """True when both plans carry the same timed action schedule."""
    if len(plan_a.actions) != len(plan_b.actions):
        return False
    for action_a, action_b in zip(plan_a.actions, plan_b.actions):
        if action_a.type != action_b.type:
            return False
        if abs(action_a.start_time - action_b.start_time) > tolerance:
            return False
        if abs(action_a.end_time - action_b.end_time) > tolerance:
            return False
    return True


def _wait_total(plan: PlanResult) -> float:
    return sum(
        action.end_time - action.start_time
        for action in plan.actions
        if action.type == "wait"
    )


def find_delayed_plan(
    original: PlanResult,
    vehicle: Vehicle,
    new_scene: Scene,
    *,
    max_delay: float = 30.0,
    delay_step: float = 0.2,
) -> tuple[float | None, PlanResult | None, str]:
    """Try departure-only delays, smallest feasible delay wins.

    The geometric route is untouched; the whole original timeline is shifted
    by an initial wait and independently replayed against the new timetable.
    """
    new_signature = new_scene.signature()
    aligned = (
        original
        if original.scene == new_scene.name
        else replace(original, scene=new_scene.name)
    )
    found_delay = None
    delayed_plan = None
    message = "未能在最大延后时间内避开所有冲突"
    delay = 0.0
    while delay <= max_delay + 1e-9:
        actions = _shift_actions(
            aligned.actions,
            delay,
            f"障碍时间表变化，整体延后 {delay:.1f}s 出发（路线不变）",
        )
        candidate = replace(aligned, actions=actions, scene_signature=new_signature)
        report = replay(candidate, vehicle, new_scene)
        if report.valid:
            return delay, candidate, report.message
        message = report.message
        delay += delay_step
    return found_delay, delayed_plan, message


def run_full_replan(
    vehicle: Vehicle,
    new_scene: Scene,
    *,
    primitives=None,
    planner_config=None,
    scheduler_config: SchedulerConfig | None = None,
) -> PlanResult:
    """Search a fresh geometric path and schedule it on the new timetable."""
    from .planner import Planner, PlannerConfig as _PlannerConfig

    planner = Planner(vehicle, new_scene, primitives, planner_config or _PlannerConfig())
    geometric = planner.plan()
    if not geometric.found:
        return geometric
    return Scheduler(
        vehicle, new_scene, scheduler_config or SchedulerConfig()
    ).schedule(geometric)


@dataclass(frozen=True)
class ReopenOption:
    key: str
    description: str
    feasible: bool
    route_changed: bool
    timing_changed: bool
    same_route: bool
    departure_delay: float
    final_time: float
    total_wait: float
    message: str
    scene_signature: str

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "feasible": self.feasible,
            "route_changed": self.route_changed,
            "timing_changed": self.timing_changed,
            "same_route": self.same_route,
            "departure_delay": round(self.departure_delay, 3),
            "final_time": round(self.final_time, 3),
            "total_wait": round(self.total_wait, 3),
            "message": self.message,
            "scene_signature": self.scene_signature,
        }


@dataclass(frozen=True)
class ReopenAssessment:
    original_plan_path: str
    old_scene_signature: str
    new_scene_signature: str
    timetable_changed: bool
    original_feasible: bool
    conflicts: tuple[ConflictInterval, ...]
    options: tuple[ReopenOption, ...]
    recommendation: str
    option_plans: dict[str, PlanResult] | None = None

    def option(self, key: str) -> ReopenOption:
        return next(item for item in self.options if item.key == key)

    def to_dict(self) -> dict:
        return {
            "schema": "reopen_assessment_v1",
            "original_plan_path": self.original_plan_path,
            "old_scene_signature": self.old_scene_signature,
            "new_scene_signature": self.new_scene_signature,
            "timetable_changed": self.timetable_changed,
            "original_feasible": self.original_feasible,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "options": {item.key: item.to_dict() for item in self.options},
            "recommendation": self.recommendation,
        }


def evaluate_reopen(
    original: PlanResult,
    vehicle: Vehicle,
    new_scene: Scene,
    *,
    max_delay: float = 30.0,
    delay_step: float = 0.2,
    primitives=None,
    planner_config=None,
    scheduler_config: SchedulerConfig | None = None,
) -> ReopenAssessment:
    """Reopen a successful plan against a changed obstacle timetable.

    The original plan is read-only. The original timing is checked against the
    new timetable first (with explicit conflict windows), then a departure
    delay is tried before any geometric re-plan.
    """
    if not original.timed:
        raise ValueError("reopen requires a timed plan; re-plan untimed plans on the new scene")
    if original.vehicle != vehicle.name:
        raise ValueError("reopen requires the same vehicle as the original plan")
    new_signature = new_scene.signature()
    aligned = (
        original
        if original.scene == new_scene.name
        else replace(original, scene=new_scene.name)
    )
    timetable_changed = original.scene_signature != new_signature
    conflicts = tuple(scan_conflicts(original, vehicle, new_scene))

    original_report = replay(aligned, vehicle, new_scene)
    original_feasible = original_report.valid

    options: list[ReopenOption] = [
        ReopenOption(
            key="original",
            description="保留原路线与原时间安排，直接面对新时间表",
            feasible=original_feasible,
            route_changed=False,
            timing_changed=False,
            same_route=True,
            departure_delay=0.0,
            final_time=original.final_time if original_feasible else 0.0,
            total_wait=_wait_total(original) if original_feasible else 0.0,
            message=original_report.message,
            scene_signature=new_signature,
        )
    ]

    found_delay, delayed_plan, delay_report_message = find_delayed_plan(
        original,
        vehicle,
        new_scene,
        max_delay=max_delay,
        delay_step=delay_step,
    )

    options.append(
        ReopenOption(
            key="delayed",
            description="保留原路线，仅延后出发（整体平移时间轴，先试时间再试路径）",
            feasible=found_delay is not None,
            route_changed=False,
            timing_changed=(found_delay or 0.0) > 0.0,
            same_route=True,
            departure_delay=found_delay if found_delay is not None else 0.0,
            final_time=delayed_plan.final_time if delayed_plan else 0.0,
            total_wait=_wait_total(delayed_plan) if delayed_plan else 0.0,
            message=delay_report_message,
            scene_signature=new_signature,
        )
    )

    replanned = run_full_replan(
        vehicle,
        new_scene,
        primitives=primitives,
        planner_config=planner_config,
        scheduler_config=scheduler_config,
    )
    replan_feasible = replanned.status == SchedulerStatus.PATH_FOUND
    route_changed = not same_route(aligned, replanned)
    options.append(
        ReopenOption(
            key="replanned",
            description="在新时间表上重新搜索行驶路径并重新排时",
            feasible=replan_feasible,
            route_changed=route_changed,
            timing_changed=replan_feasible and not same_timing(original, replanned),
            same_route=not route_changed,
            departure_delay=0.0,
            final_time=replanned.final_time if replan_feasible else 0.0,
            total_wait=_wait_total(replanned) if replan_feasible else 0.0,
            message=replanned.message,
            scene_signature=new_signature,
        )
    )

    if original_feasible:
        recommendation = "original"
    elif found_delay is not None:
        recommendation = "delayed"
    elif replan_feasible:
        recommendation = "replanned"
    else:
        recommendation = "none"

    return ReopenAssessment(
        original_plan_path="",
        old_scene_signature=original.scene_signature,
        new_scene_signature=new_signature,
        timetable_changed=timetable_changed,
        original_feasible=original_feasible,
        conflicts=conflicts,
        options=tuple(options),
        recommendation=recommendation,
        option_plans={
            **({"delayed": delayed_plan} if delayed_plan is not None else {}),
            **({"replanned": replanned} if replan_feasible else {}),
        },
    )


def save_reopen_bundle(
    assessment: ReopenAssessment,
    plans: dict[str, PlanResult] | None,
    report_path: str | Path,
) -> None:
    """Persist the assessment report and each option plan next to it.

    Each saved plan carries the new timetable signature; the original success
    file is referenced by path and is never rewritten.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = assessment.to_dict()
    saved_paths: dict[str, str] = {}
    for key, plan in (plans or assessment.option_plans or {}).items():
        plan_path = report_path.with_name(f"{report_path.stem}_{key}.json")
        save_plan(plan, plan_path)
        saved_paths[key] = plan_path.name
    payload["option_plan_files"] = saved_paths
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reopen_from_files(
    original_plan_path: str | Path,
    vehicle_path: str | Path,
    new_scene_path: str | Path,
    **kwargs,
) -> tuple[PlanResult, Vehicle, Scene, ReopenAssessment]:
    original = load_plan(original_plan_path)
    vehicle = load_vehicle(vehicle_path)
    new_scene = load_scene(new_scene_path)
    assessment = evaluate_reopen(original, vehicle, new_scene, **kwargs)
    assessment = replace(assessment, original_plan_path=str(original_plan_path))
    return original, vehicle, new_scene, assessment
