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

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .motion import sample_arc
from .geometry import polygons_intersect
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
            duration = action.end_time - action.start_time
            steps = max(1, int(math.ceil(duration / 0.1)))
            for index in range(steps + 1):
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


@dataclass(frozen=True)
class DepartureWindow:
    """A point or range backed by one clearly identified kind of evidence."""

    kind: str  # safe | blocked | unverified
    start_time: float
    end_time: float
    obstacles: tuple[str, ...] = ()
    note: str = ""
    evidence_type: str = "point"  # point | continuous | unproven_range
    basis: str = ""
    boundary_conditions: tuple[str, ...] = ()
    start_inclusive: bool = True
    end_inclusive: bool = True

    def to_dict(self) -> dict:
        payload = {
            "kind": self.kind,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.end_time - self.start_time, 3),
        }
        if self.obstacles:
            payload["obstacles"] = list(self.obstacles)
        payload["evidence_type"] = self.evidence_type
        if self.basis:
            payload["basis"] = self.basis
        if self.boundary_conditions:
            payload["boundary_conditions"] = list(self.boundary_conditions)
        payload["start_inclusive"] = self.start_inclusive
        payload["end_inclusive"] = self.end_inclusive
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class BoundaryProbe:
    label: str
    departure_time: float
    feasible: bool | None
    obstacles: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "departure_time": round(self.departure_time, 3),
            "feasible": self.feasible,
            "obstacles": list(self.obstacles),
        }


@dataclass(frozen=True)
class ProbeEvidence:
    departure_time: float
    feasible: bool
    obstacles: tuple[str, ...]
    check_count: int
    scene_signature: str
    route_signature: str
    replay_samples: int
    result_message: str
    first_checked: int
    last_checked: int

    def to_dict(self) -> dict:
        return {
            "departure_time": round(self.departure_time, 9),
            "feasible": self.feasible,
            "obstacles": list(self.obstacles),
            "check_count": self.check_count,
            "scene_signature": self.scene_signature,
            "route_signature": self.route_signature,
            "replay_samples": self.replay_samples,
            "result_message": self.result_message,
            "first_checked": self.first_checked,
            "last_checked": self.last_checked,
        }


@dataclass(frozen=True)
class DepartureWindowAnalysis:
    """Result of scanning a user-bounded departure-time range."""

    earliest_departure: float
    latest_departure: float
    step: float
    probes_requested: int
    probes_completed: int
    budget_exhausted: bool
    boundaries: tuple[BoundaryProbe, ...]
    windows: tuple[DepartureWindow, ...]
    checked_probes: tuple[ProbeEvidence, ...] = ()
    continuous_ranges: tuple[DepartureWindow, ...] = ()
    newly_checked_departures: tuple[float, ...] = ()
    selected_departure: float | None = None
    selected_verified: bool = False
    selection_status: str = "none"
    selected_evidence: dict | None = None
    schedule_signature: str = ""
    route_signature: str = ""
    evidence_reuse_status: str = "no_prior"  # no_prior | reused | scene_changed | route_changed
    stale_evidence_count: int = 0
    search_state: dict | None = None
    historical_signature: str | None = None
    historical_status: str = "none"  # none | same_timetable | needs_recheck

    def to_dict(self) -> dict:
        return {
            "earliest_departure": round(self.earliest_departure, 3),
            "latest_departure": round(self.latest_departure, 3),
            "step": round(self.step, 3),
            "probes_requested": self.probes_requested,
            "probes_completed": self.probes_completed,
            "budget_exhausted": self.budget_exhausted,
            "boundaries": [item.to_dict() for item in self.boundaries],
            "windows": [item.to_dict() for item in self.windows],
            "checked_probes": [item.to_dict() for item in self.checked_probes],
            "continuous_ranges": [item.to_dict() for item in self.continuous_ranges],
            "newly_checked_departures": [round(item, 9) for item in self.newly_checked_departures],
            "selected_departure": (
                round(self.selected_departure, 3)
                if self.selected_departure is not None
                else None
            ),
            "selected_verified": self.selected_verified,
            "selection_status": self.selection_status,
            "selected_evidence": self.selected_evidence,
            "schedule_signature": self.schedule_signature,
            "route_signature": self.route_signature,
            "evidence_reuse_status": self.evidence_reuse_status,
            "stale_evidence_count": self.stale_evidence_count,
            "search_state": self.search_state,
            "historical_signature": self.historical_signature,
            "historical_status": self.historical_status,
        }


def _probe_departure(
    aligned: PlanResult,
    vehicle: Vehicle,
    new_scene: Scene,
    departure: float,
) -> tuple[bool, tuple[str, ...], PlanResult, int, str]:
    """Check one departure time over the whole unchanged route on the new timetable.

    The candidate carries the new timetable signature and is replayed in the
    normal (signature-enforcing) mode, so a "safe" probe is independently
    verifiable and never inherited from an old success mark.
    """
    new_signature = new_scene.signature()
    actions = _shift_actions(
        aligned.actions,
        departure,
        f"障碍时间表变化，延后 {departure:.1f}s 出发（路线不变）",
    )
    candidate = replace(aligned, actions=actions, scene_signature=new_signature)
    conflicts = scan_conflicts(candidate, vehicle, new_scene)
    if conflicts:
        names = tuple(
            sorted({name for item in conflicts for name in item.obstacles})
        )
        return False, names, candidate, sum(item.samples for item in conflicts), "detected concrete conflict interval"
    report = replay(candidate, vehicle, new_scene)
    return report.valid, (), candidate, report.samples, report.message


def _route_signature(plan: PlanResult) -> str:
    payload = {
        "vehicle": plan.vehicle,
        "start_pose": plan.start_pose,
        "goal": plan.goal_pose,
        "states": plan.states,
        "motions": [
            {
                "direction": motion.direction,
                "steering": motion.steering,
                "distance": motion.distance,
            }
            for motion in plan.motions
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _route_footprints(plan: PlanResult, vehicle: Vehicle) -> list:
    pose = plan.start_pose
    footprints = [vehicle.footprint(*pose)]
    for action in plan.actions:
        if not hasattr(action, "distance"):
            continue
        if action.type in ("wait", "gear_switch"):
            continue
        arc = sample_arc(
            pose,
            action.direction,
            action.steering,
            action.distance,
            vehicle.wheelbase,
            0.03,
        )
        footprints.extend(vehicle.footprint(*sampled) for sampled in arc[1:])
        pose = arc[-1]
    return footprints


def _continuous_safe_range(
    aligned: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    earliest: float,
    latest: float,
) -> DepartureWindow | None:
    """Return a range only when a timetable-wide proof is available.

    Discrete departure probes never become ranges. This conservative proof
    checks static geometry and every dynamic obstacle's held endpoint plus a
    dense sweep of every keyframe interval against every route footprint.
    """
    timed = list(_timed_footprints(aligned, vehicle))
    for _route_time, footprint, _context in timed:
        if scene.footprint_collides(footprint):
            return None
    if not scene.dynamic_obstacles:
        return DepartureWindow(
            kind="safe",
            start_time=earliest,
            end_time=latest,
            note="无动态障碍；静态几何全程核对通过",
            evidence_type="continuous",
            basis="static route sweep",
            boundary_conditions=(
                f"included lower endpoint {earliest:g}",
                f"included upper endpoint {latest:g}",
            ),
        )
    for route_time, footprint, _context in timed:
        window_begin = max(0.0, route_time + earliest)
        window_end = max(window_begin, route_time + latest)
        critical = {window_begin, window_end}
        for obstacle in scene.dynamic_obstacles:
            key_times = [frame[0] for frame in obstacle.keyframes]
            critical.update(value for value in key_times if window_begin <= value <= window_end)
            for left, right in zip(key_times, key_times[1:]):
                if right < window_begin or left > window_end:
                    continue
                begin = max(window_begin, left)
                end = min(window_end, right)
                steps = max(1, int(math.ceil((end - begin) / 0.05)))
                critical.update(begin + (end - begin) * i / steps for i in range(steps + 1))
            for at_time in critical:
                if polygons_intersect(footprint, obstacle.polygon_at(at_time)):
                    return None
            if polygons_intersect(footprint, obstacle.polygon_at(window_begin)):
                return None
            if polygons_intersect(footprint, obstacle.polygon_at(window_end)):
                return None
    return DepartureWindow(
        kind="safe",
        start_time=earliest,
        end_time=latest,
        note="动态障碍在路线时间轴上的保持端点、关键帧端点和关键帧间连续扫掠均不与路线相交",
        evidence_type="continuous",
        basis="static replay plus dense obstacle timetable/route sweep",
        boundary_conditions=(
            f"included lower endpoint {earliest:g}",
            f"included upper endpoint {latest:g}",
            "obstacle keyframe instants included",
            "obstacle held before first and after last keyframe checked",
        ),
    )


def analyze_departure_windows(
    original: PlanResult,
    vehicle: Vehicle,
    new_scene: Scene,
    *,
    earliest_departure: float = 0.0,
    latest_departure: float,
    step: float = 0.2,
    max_probes: int = 300,
    search_state: dict | None = None,
    extra_candidates: tuple[float, ...] | None = None,
) -> DepartureWindowAnalysis:
    """Classify a departure-time range into safe / blocked / unverified windows.

    Probes are exact point evidence. They never imply safety or failure at a
    neighbouring time, even when the two endpoints of an interval have both
    been checked. Persisted state deduplicates repeat probes and records the
    scene/route for which evidence remains applicable.
    """
    if earliest_departure < -1e-9:
        raise ValueError("earliest departure time must be non-negative")
    if latest_departure < earliest_departure - 1e-9:
        raise ValueError("latest departure time must be >= earliest departure time")
    if step <= 0.0:
        raise ValueError("departure step must be positive")
    if max_probes < 0:
        raise ValueError("departure budget must be non-negative")

    aligned = (
        original
        if original.scene == new_scene.name
        else replace(original, scene=new_scene.name)
    )
    new_signature = new_scene.signature()

    grid = [earliest_departure]
    interior_count = max(
        0, int(math.floor((latest_departure - earliest_departure) / step + 1e-9)) - 1
    )
    for index in range(1, interior_count + 1):
        grid.append(earliest_departure + index * step)
    if abs(grid[-1] - latest_departure) > 1e-9:
        grid.append(latest_departure)
    requested = {round(value, 9): value for value in grid}
    for value in extra_candidates or ():
        if value < earliest_departure - 1e-9 or value > latest_departure + 1e-9:
            raise ValueError("extra departure candidate must lie inside the search range")
        requested[round(value, 9)] = float(value)

    route_sig = _route_signature(aligned)
    previous = search_state or {}
    inherited: dict[float, ProbeEvidence] = {}
    stale_count = len(previous.get("probes", ())) if previous else 0
    evidence_reuse_status = "no_prior" if not previous else "scene_changed"
    if previous:
        if previous.get("scene_signature") != new_signature:
            evidence_reuse_status = "scene_changed"
        elif previous.get("route_signature") != route_sig:
            evidence_reuse_status = "route_changed"
        else:
            evidence_reuse_status = "reused"
    if evidence_reuse_status == "reused":
        for item in previous.get("probes", ()):
            evidence = ProbeEvidence(
                departure_time=float(item["departure_time"]),
                feasible=bool(item["feasible"]),
                obstacles=tuple(item.get("obstacles", ())),
                check_count=int(item.get("check_count", 1)),
                scene_signature=item["scene_signature"],
                route_signature=item["route_signature"],
                replay_samples=int(item.get("replay_samples", 0)),
                result_message=str(item.get("result_message", "")),
                first_checked=int(item.get("first_checked", 0)),
                last_checked=int(item.get("last_checked", 0)),
            )
            inherited[round(evidence.departure_time, 9)] = evidence
    all_evidence = dict(inherited)
    probes = {
        departure: evidence
        for departure, evidence in inherited.items()
        if earliest_departure - 1e-9 <= evidence.departure_time <= latest_departure + 1e-9
    }

    explicit = {round(value, 9) for value in (extra_candidates or ())}
    candidates = sorted(explicit) + sorted(set(requested) - explicit)
    new_checks: list[float] = []
    turn_offset = int(previous.get("next_check_index", 0))
    budget_used_this_turn = 0
    budget_exhausted = False
    for departure in candidates:
        if departure in probes:
            continue
        if budget_used_this_turn >= max_probes:
            budget_exhausted = True
            break
        feasible, names, _candidate, samples, message = _probe_departure(
            aligned, vehicle, new_scene, requested[departure]
        )
        check_index = turn_offset + budget_used_this_turn + 1
        probes[departure] = ProbeEvidence(
            departure_time=requested[departure],
            feasible=feasible,
            obstacles=names,
            check_count=1,
            scene_signature=new_signature,
            route_signature=route_sig,
            replay_samples=samples,
            result_message=message,
            first_checked=check_index,
            last_checked=check_index,
        )
        all_evidence[departure] = probes[departure]
        new_checks.append(departure)
        budget_used_this_turn += 1

    checked = tuple(probes[key] for key in sorted(probes))
    boundaries = tuple(
        BoundaryProbe(
            label=("earliest" if index == 0 else "latest"),
            departure_time=value,
            feasible=(probes[round(value, 9)].feasible if round(value, 9) in probes else None),
            obstacles=(probes[round(value, 9)].obstacles if round(value, 9) in probes else ()),
        )
        for index, value in enumerate((earliest_departure, latest_departure))
    )

    continuous_range = _continuous_safe_range(
        aligned, vehicle, new_scene, earliest_departure, latest_departure
    )
    windows: list[DepartureWindow] = []
    if continuous_range is not None:
        windows.append(continuous_range)
    else:
        sorted_checked = sorted(probes)
        for departure in sorted_checked:
            evidence = probes[departure]
            windows.append(
                DepartureWindow(
                    kind="safe" if evidence.feasible else "blocked",
                    start_time=departure,
                    end_time=departure,
                    obstacles=evidence.obstacles,
                    note=(
                        "该具体出发时刻已沿当前场景、当前路线全程核对通过"
                        if evidence.feasible
                        else "该具体出发时刻核对到具体障碍占用"
                    ),
                    evidence_type="point",
                    basis=f"independent replay/sweep, {evidence.replay_samples} samples",
                )
            )
        unchecked: list[tuple[float, float, bool, bool]] = []
        if not sorted_checked:
            unchecked.append((earliest_departure, latest_departure, True, True))
        else:
            first = sorted_checked[0]
            last = sorted_checked[-1]
            if first > earliest_departure + 1e-9:
                unchecked.append((earliest_departure, first, True, False))
            for left, right in zip(sorted_checked, sorted_checked[1:]):
                if right > left + 1e-9:
                    unchecked.append((left, right, False, False))
            if last < latest_departure - 1e-9:
                unchecked.append((last, latest_departure, False, True))
        for left, right, start_inclusive, end_inclusive in unchecked:
            windows.append(
                DepartureWindow(
                    kind="unverified",
                    start_time=left,
                    end_time=right,
                    note="范围边界或采样点通过不外推到内部；这些时刻尚未独立核对",
                    evidence_type="unproven_range",
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                )
            )
        windows.sort(key=lambda item: (item.start_time, item.end_time, item.kind))

    feasible_points = tuple(item.departure_time for item in checked if item.feasible)
    selected = feasible_points[0] if feasible_points else None
    if continuous_range is not None:
        selected = earliest_departure
        selection_status = "executable"
        selected_evidence = {
            **continuous_range.to_dict(),
            "departure_time": round(selected, 9),
            "scene_signature": new_signature,
            "route_signature": route_sig,
        }
    elif selected is not None:
        selection_status = "executable_at_checked_point"
        selected_evidence = probes[selected].to_dict()
    else:
        selection_status = "provisional"
        selected_evidence = None

    state = {
        "schema": "departure_search_state_v1",
        "scene_signature": new_signature,
        "route_signature": route_sig,
        "earliest_departure": earliest_departure,
        "latest_departure": latest_departure,
        "step": step,
        "next_check_index": turn_offset + budget_used_this_turn,
        "probes": [item.to_dict() for item in (all_evidence[key] for key in sorted(all_evidence))],
    }

    return DepartureWindowAnalysis(
        earliest_departure=earliest_departure,
        latest_departure=latest_departure,
        step=step,
        probes_requested=len(requested),
        probes_completed=len([item for item in checked if earliest_departure - 1e-9 <= item.departure_time <= latest_departure + 1e-9]),
        budget_exhausted=budget_exhausted,
        boundaries=boundaries,
        windows=tuple(windows),
        checked_probes=checked,
        continuous_ranges=((continuous_range,) if continuous_range is not None else ()),
        newly_checked_departures=tuple(new_checks),
        selected_departure=selected,
        selected_verified=continuous_range is not None or selected in probes,
        selection_status=selection_status,
        selected_evidence=selected_evidence,
        schedule_signature=new_signature,
        route_signature=route_sig,
        evidence_reuse_status=evidence_reuse_status,
        stale_evidence_count=(0 if evidence_reuse_status == "reused" else stale_count),
        search_state=state,
    )


def attach_historical_windows(
    analysis: DepartureWindowAnalysis,
    historical_report: dict | None,
) -> DepartureWindowAnalysis:
    """Tag a previous report's windows without adopting them as current truth.

    A previous conclusion is only comparable when it was produced for the
    exact same timetable signature; any other old report is a historical
    conclusion that must be re-checked.
    """
    if not historical_report:
        return analysis
    old = historical_report.get("departure_windows") or {}
    old_signature = old.get("schedule_signature")
    status = (
        "same_timetable"
        if old_signature == analysis.schedule_signature
        else "needs_recheck"
    )
    return replace(
        analysis,
        historical_signature=old_signature,
        historical_status=status,
    )


def _departure_state(historical_report: dict | None, continued_report: dict | None) -> dict | None:
    for report in (continued_report, historical_report):
        if not report:
            continue
        windows = report.get("departure_windows") or {}
        state = windows.get("search_state")
        if state:
            return state
    return None


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
    departure_windows: DepartureWindowAnalysis | None = None

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
            **(
                {"departure_windows": self.departure_windows.to_dict()}
                if self.departure_windows is not None
                else {}
            ),
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
    earliest_departure: float | None = None,
    latest_departure: float | None = None,
    departure_step: float = 0.2,
    departure_budget: int = 300,
    historical_report: dict | None = None,
    continued_report: dict | None = None,
    departure_candidates: tuple[float, ...] = (),
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

    # The original route faces the NEW timetable on its own merits: replay in
    # signature-bypass mode so a timetable hash change alone cannot reject a
    # route that has zero conflicts. This keeps the verdict consistent with
    # the delayed option at delay 0 and with scan_conflicts().
    original_report = replay(aligned, vehicle, new_scene, enforce_signature=False)
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

    # A feasible original schedule is re-bound to the NEW timetable signature
    # when saved, so it independently replays against the new timetable while
    # the original success file on disk is referenced and never rewritten.
    original_resigned = None
    if original_feasible:
        original_resigned = replace(aligned, scene_signature=new_signature)

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

    departure_analysis = None
    if earliest_departure is not None and latest_departure is not None:
        departure_analysis = analyze_departure_windows(
            original,
            vehicle,
            new_scene,
            earliest_departure=earliest_departure,
            latest_departure=latest_departure,
            step=departure_step,
            max_probes=departure_budget,
            search_state=_departure_state(historical_report, continued_report),
            extra_candidates=tuple(departure_candidates),
        )
        departure_analysis = attach_historical_windows(
            departure_analysis, continued_report or historical_report
        )

    selected_plan = None
    if (
        departure_analysis is not None
        and departure_analysis.selected_departure is not None
        and departure_analysis.selection_status != "provisional"
    ):
        selected_plan = replace(
            aligned,
            actions=_shift_actions(
                aligned.actions,
                departure_analysis.selected_departure,
                f"采用已核对的出发时刻 {departure_analysis.selected_departure:.1f}s（路线不变）",
            ),
            scene_signature=new_signature,
        )

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
            **({"original": original_resigned} if original_resigned is not None else {}),
            **({"delayed": delayed_plan} if delayed_plan is not None else {}),
            **({"selected_window": selected_plan} if selected_plan is not None else {}),
            **({"replanned": replanned} if replan_feasible else {}),
        },
        departure_windows=departure_analysis,
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
