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
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .motion import sample_arc
from .plan import Action, PlanResult, load_plan, save_plan
from .replay import replay
from .scheduler import Scheduler, SchedulerConfig, SchedulerStatus
from .scene import Scene, load_scene
from .vehicle import Vehicle, load_vehicle


# Continuous coverage is independent evidence, never derived from the
# user's sampling grid: gap intervals are replayed at this fixed fine
# spacing, so enlarging --departure-step cannot manufacture a safe span.
DEFAULT_COVERAGE_STEP = 0.05


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


def route_fingerprint(plan: PlanResult) -> str:
    """Hash of the geometric route (start pose + motion primitives).

    Departure-time evidence is only reusable while both the scene signature
    and this route fingerprint stay unchanged; any route or timetable edit
    forces a fresh check instead of stitching old and new verdicts together.
    """
    payload = {
        "start": [round(float(value), 9) for value in plan.start_pose],
        "motions": [
            [motion.direction, round(motion.steering, 9), round(motion.distance, 9)]
            for motion in plan.motions
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    """One classified sub-interval of the user's departure-time range."""

    kind: str  # "safe" | "blocked" | "unverified"
    start_time: float
    end_time: float
    obstacles: tuple[str, ...] = ()
    note: str = ""
    probes: int = 0

    def to_dict(self) -> dict:
        payload = {
            "kind": self.kind,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.end_time - self.start_time, 3),
        }
        if self.probes:
            payload["probes"] = self.probes
        if self.obstacles:
            payload["obstacles"] = list(self.obstacles)
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class DepartureProbe:
    """One actually checked departure time: the atomic piece of evidence."""

    departure_time: float
    feasible: bool
    obstacles: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "departure_time": round(self.departure_time, 3),
            "feasible": self.feasible,
            "obstacles": list(self.obstacles),
        }

    @staticmethod
    def from_dict(payload: dict) -> "DepartureProbe":
        return DepartureProbe(
            departure_time=float(payload["departure_time"]),
            feasible=bool(payload["feasible"]),
            obstacles=tuple(str(name) for name in payload.get("obstacles", ())),
        )


@dataclass(frozen=True)
class BoundaryProbe:
    label: str
    departure_time: float
    feasible: bool | None
    obstacles: tuple[str, ...] = ()
    probed: bool = True

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "departure_time": round(self.departure_time, 3),
            "feasible": self.feasible,
            "probed": self.probed,
            "obstacles": list(self.obstacles),
        }


@dataclass(frozen=True)
class DepartureWindowAnalysis:
    """Result of scanning a user-bounded departure-time range."""

    earliest_departure: float
    latest_departure: float
    step: float
    coverage_step: float
    probes_requested: int
    probes_completed: int
    budget_exhausted: bool
    boundaries: tuple[BoundaryProbe, ...]
    windows: tuple[DepartureWindow, ...]
    selected_departure: float | None
    selected_verified: bool
    schedule_signature: str
    probes: tuple[DepartureProbe, ...] = ()
    sweeps: tuple[DepartureProbe, ...] = ()
    newly_probed: tuple[float, ...] = ()
    newly_swept: tuple[float, ...] = ()
    probes_this_round: int = 0
    scene_name: str = ""
    route_fingerprint: str = ""
    continued: bool = False
    historical_signature: str | None = None
    historical_status: str = "none"  # none | same_timetable | needs_recheck
    prior_selected_departure: float | None = None
    prior_selection_status: str = "none"  # none | kept | invalidated
    prior_selection_note: str = ""

    def to_dict(self) -> dict:
        selection = None
        if self.selected_departure is not None:
            selection = {
                "departure_time": round(self.selected_departure, 3),
                "scene": self.scene_name,
                "scene_signature": self.schedule_signature,
                "route_fingerprint": self.route_fingerprint,
                "verified": True,
                "executable": True,
            }
        return {
            "earliest_departure": round(self.earliest_departure, 3),
            "latest_departure": round(self.latest_departure, 3),
            "step": round(self.step, 3),
            "coverage_step": round(self.coverage_step, 6),
            "probes_requested": self.probes_requested,
            "probes_completed": self.probes_completed,
            "probes_this_round": self.probes_this_round,
            "newly_probed": [round(value, 3) for value in self.newly_probed],
            "newly_swept": [round(value, 3) for value in self.newly_swept],
            "continued": self.continued,
            "budget_exhausted": self.budget_exhausted,
            "boundaries": [item.to_dict() for item in self.boundaries],
            "windows": [item.to_dict() for item in self.windows],
            "probes": [item.to_dict() for item in self.probes],
            "coverage_probes": [item.to_dict() for item in self.sweeps],
            "selected_departure": (
                round(self.selected_departure, 3)
                if self.selected_departure is not None
                else None
            ),
            "selected_verified": self.selected_verified,
            "selection": selection,
            "schedule_signature": self.schedule_signature,
            "scene": self.scene_name,
            "route_fingerprint": self.route_fingerprint,
            "historical_signature": self.historical_signature,
            "historical_status": self.historical_status,
            "prior_selected_departure": (
                round(self.prior_selected_departure, 3)
                if self.prior_selected_departure is not None
                else None
            ),
            "prior_selection_status": self.prior_selection_status,
            "prior_selection_note": self.prior_selection_note,
        }


def _probe_departure(
    aligned: PlanResult,
    vehicle: Vehicle,
    new_scene: Scene,
    departure: float,
) -> tuple[bool, tuple[str, ...], PlanResult]:
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
        return False, names, candidate
    report = replay(candidate, vehicle, new_scene)
    return report.valid, (), candidate


def analyze_departure_windows(
    original: PlanResult,
    vehicle: Vehicle,
    new_scene: Scene,
    *,
    earliest_departure: float = 0.0,
    latest_departure: float,
    step: float = 0.2,
    coverage_step: float = DEFAULT_COVERAGE_STEP,
    max_probes: int = 300,
    prior: "DepartureWindowAnalysis | dict | None" = None,
) -> DepartureWindowAnalysis:
    """Classify a departure-time range into safe / blocked / unverified windows.

    Evidence model: a probe is a point verdict for exactly one departure
    time and, however dense the requested sampling grid happens to be,
    never proves anything between two checked moments. Three evidence
    levels are reported strictly separately:

    * probed points: every point verdict is a zero-width window at its
      own moment and is never extrapolated;
    * continuous intervals: only an independent coverage sweep that
      replays the route at a fixed, user-independent fine spacing
      (coverage_step) throughout an interval supplies continuous
      evidence; adjacent checked moments no more than coverage_step
      apart then share a cell boundary at their midpoint;
    * unknown ranges: gaps without that independent sweep (budget tail,
      gaps between point probes, gaps left by a changed sampling step,
      range edges) are reported as unverified and are never promoted to
      safe or failure merely because the requested step was reached.

    Boundary probes that were not reached are reported as unprobed, never
    as infeasible.

    The prior argument (a previous analysis object or a saved report dict)
    continues the same search with extra budget: its probes are reused
    verbatim when both the timetable signature and the route fingerprint
    still match, only the missing grid points consume the new budget, and
    newly_probed lists exactly what this round checked. Re-submitting an
    already checked candidate costs nothing and does not inflate the
    independent-check count. Any scene or route change marks the old
    evidence needs_recheck and starts fresh instead of stitching old and
    new verdicts into one interval.
    """
    if earliest_departure < -1e-9:
        raise ValueError("earliest departure time must be non-negative")
    if latest_departure < earliest_departure - 1e-9:
        raise ValueError("latest departure time must be >= earliest departure time")
    if step <= 0.0:
        raise ValueError("departure step must be positive")
    if coverage_step <= 0.0:
        raise ValueError("coverage step must be positive")

    aligned = (
        original
        if original.scene == new_scene.name
        else replace(original, scene=new_scene.name)
    )
    new_signature = new_scene.signature()
    fingerprint = route_fingerprint(aligned)

    historical_signature = None
    historical_status = "none"
    continued = False
    point_evidence: dict[float, DepartureProbe] = {}
    coverage_evidence: dict[float, DepartureProbe] = {}
    prior_selected: float | None = None
    if prior is not None:
        prior_payload = (
            prior.to_dict() if isinstance(prior, DepartureWindowAnalysis) else prior
        )
        if "departure_windows" in prior_payload and "probes" not in prior_payload:
            prior_payload = prior_payload.get("departure_windows") or {}
        raw_selected = prior_payload.get("selected_departure")
        if raw_selected is None:
            raw_selected = (prior_payload.get("selection") or {}).get("departure_time")
        if raw_selected is not None:
            prior_selected = float(raw_selected)
        historical_signature = prior_payload.get("schedule_signature")
        prior_route = prior_payload.get("route_fingerprint")
        if historical_signature == new_signature and prior_route in (None, "", fingerprint):
            # Same timetable and same route: old point verdicts still apply.
            # Narrowing the range simply filters them; each kept probe stays
            # evidence for its exact departure time only.
            point_payloads = prior_payload.get("probes", ())
            for payload in point_payloads:
                probe = DepartureProbe.from_dict(payload)
                if (
                    earliest_departure - 1e-9
                    <= probe.departure_time
                    <= latest_departure + 1e-9
                ):
                    key = round(probe.departure_time, 9)
                    point_evidence.setdefault(key, probe)
                    coverage_evidence.setdefault(key, probe)
            # Saved continuous-sweep facts stay reusable, but their spacing
            # is re-judged under the current coverage step: a coarse old
            # sweep never proves gaps against a stricter current density.
            for payload in prior_payload.get("coverage_probes", ()):
                probe = DepartureProbe.from_dict(payload)
                if (
                    earliest_departure - 1e-9
                    <= probe.departure_time
                    <= latest_departure + 1e-9
                ):
                    coverage_evidence.setdefault(round(probe.departure_time, 9), probe)
            continued = bool(point_evidence or coverage_evidence)
            historical_status = "same_timetable"
        else:
            historical_status = "needs_recheck"

    interior_count = max(
        0, int(math.floor((latest_departure - earliest_departure) / step + 1e-9)) - 1
    )
    interior = [earliest_departure + index * step for index in range(1, interior_count + 1)]
    # Boundary moments are probed FIRST, then the interior in order: a tiny
    # budget still characterizes both edges and leaves the middle unknown
    # instead of silently spending everything on one end of the range.
    grid = [earliest_departure, latest_departure, *interior]
    # Zero-width ranges and duplicate grid points are probed once, never twice.
    grid = list(dict.fromkeys(round(value, 9) for value in grid))
    probes_requested = len(grid)

    newly_probed: list[float] = []
    budget_exhausted = False
    checks_used = 0
    for departure in grid:
        if departure in point_evidence:
            continue  # already checked in a previous round: not counted again
        if checks_used >= max_probes:
            budget_exhausted = True
            break
        feasible, names, _ = _probe_departure(aligned, vehicle, new_scene, departure)
        probe = DepartureProbe(departure, feasible, names)
        point_evidence[departure] = probe
        coverage_evidence.setdefault(departure, probe)
        newly_probed.append(departure)
        checks_used += 1

    # Independent continuous evidence: fill each interval between adjacent
    # nominal grid points on a fixed fine lattice anchored at the earliest
    # departure, regardless of the user's sampling step. Filling starts
    # from both edges of the whole range (smallest gaps first) so added
    # budget only turns an interval into a conclusion when the sweep really
    # reaches across it. A half-swept gap ends at the last checked moment.
    sorted_grid = sorted(grid)
    intervals = [
        (left, right)
        for left, right in zip(sorted_grid, sorted_grid[1:])
        if left in point_evidence and right in point_evidence
    ]
    newly_swept: list[float] = []
    for left, right in intervals if math.isfinite(coverage_step) else ():
        sweep_moments: list[float] = [left]
        index = 1
        while True:
            moment = round(earliest_departure + index * coverage_step, 9)
            if moment >= right - 1e-9:
                break
            if left < moment < right:
                sweep_moments.append(moment)
            index += 1
        sweep_moments.append(right)
        for moment in sweep_moments:
            if moment in coverage_evidence:
                continue
            if checks_used >= max_probes:
                budget_exhausted = True
                break
            feasible, names, _ = _probe_departure(
                aligned, vehicle, new_scene, moment
            )
            probe = DepartureProbe(moment, feasible, names)
            coverage_evidence[moment] = probe
            if moment not in point_evidence:
                newly_swept.append(moment)
            checks_used += 1
        if budget_exhausted:
            break

    boundaries = tuple(
        BoundaryProbe(
            label=("earliest" if index == 0 else "latest"),
            departure_time=value,
            feasible=(point_evidence[round(value, 9)].feasible
                      if round(value, 9) in point_evidence else None),
            obstacles=(point_evidence[round(value, 9)].obstacles
                       if round(value, 9) in point_evidence else ()),
            probed=round(value, 9) in point_evidence,
        )
        for index, value in enumerate((earliest_departure, latest_departure))
    )

    ordered = [coverage_evidence[key] for key in sorted(coverage_evidence)]
    windows: list[DepartureWindow] = []
    cursor = earliest_departure

    def emit_unverified(start: float, end: float) -> None:
        if end - start > 1e-9:
            windows.append(
                DepartureWindow(
                    kind="unverified",
                    start_time=start,
                    end_time=end,
                    note="未核对范围：证据不足（预算耗尽或采样间隔变化留下的间隙），既非安全也非失败",
                )
            )

    # Only the independent coverage sweep can bridge two moments. The
    # user's sampling step is intentionally irrelevant here: a chain of
    # point probes at exactly that spacing is still just isolated points.
    # Coverage moments spaced no more than coverage_step apart link at
    # their midpoint; every other gap stays unverified.
    covered_keys = set(coverage_evidence)

    def densely_covered(previous_time: float, next_time: float) -> bool:
        gap = next_time - previous_time
        # An infinite coverage step requests points only (no independent
        # sweep); zero-distance duplicates still carry no width.
        if gap <= 1e-9:
            return True
        if not math.isfinite(coverage_step) or gap > coverage_step + 1e-9:
            return False
        right_key = round(next_time, 9)
        offset = 1
        expected = round(previous_time + coverage_step, 9)
        while expected < right_key - 1e-9:
            if expected not in covered_keys:
                return False
            offset += 1
            expected = round(previous_time + offset * coverage_step, 9)
        return True

    cells: list[tuple[float, float]] = []
    for index, probe in enumerate(ordered):
        at_time = probe.departure_time
        left = at_time
        right = at_time
        if index > 0:
            previous_time = ordered[index - 1].departure_time
            if densely_covered(previous_time, at_time):
                left = (previous_time + at_time) / 2.0
        if index < len(ordered) - 1:
            next_time = ordered[index + 1].departure_time
            if densely_covered(at_time, next_time):
                right = (at_time + next_time) / 2.0
        left = min(max(left, earliest_departure), latest_departure)
        right = min(max(right, earliest_departure), latest_departure)
        cells.append((left, right))

    for probe, (left, right) in zip(ordered, cells):
        if left > cursor + 1e-9:
            emit_unverified(cursor, left)
        left = max(left, cursor)
        kind = "safe" if probe.feasible else "blocked"
        if (
            windows
            and windows[-1].kind == kind
            and windows[-1].obstacles == probe.obstacles
            and abs(windows[-1].end_time - left) <= 1e-9
        ):
            previous = windows[-1]
            windows[-1] = replace(previous, end_time=right, probes=previous.probes + 1)
        else:
            windows.append(
                DepartureWindow(
                    kind=kind,
                    start_time=left,
                    end_time=right,
                    obstacles=probe.obstacles,
                    probes=1,
                )
            )
        cursor = max(cursor, right)
    emit_unverified(cursor, latest_departure)

    # Annotate what each verified window actually covers. A merged window
    # rests on the independent fixed-spacing sweep; a lone point verdict
    # never implies its unchecked neighbours.
    annotated: list[DepartureWindow] = []
    for window in windows:
        if window.kind in ("safe", "blocked"):
            verdict = "可行" if window.kind == "safe" else "冲突"
            if window.probes > 1:
                note = (
                    f"连续{verdict}区间：覆盖 [{window.start_time:.3f}, {window.end_time:.3f}]s 内"
                    f"独立连续扫描间隔不超过 {coverage_step}s 的 {window.probes} 个已核对时刻；"
                    "该结论不依赖用户采样步长，点间空隙已被实际逐一全程扫描"
                )
            elif window.end_time - window.start_time <= 1e-9:
                note = (
                    f"已测点：仅出发时刻 {window.start_time:.3f}s 本身已核对（{verdict}），"
                    "两侧未核对范围保持未知，不外推"
                )
            else:
                note = f"已测点：出发时刻已核对（{verdict}），相邻时刻由独立连续扫描支持"
            if window.kind == "blocked":
                note += "；全程扫描到具体障碍占用"
            window = replace(window, note=note)
        annotated.append(window)

    selected_departure = next(
        (probe.departure_time for probe in ordered if probe.feasible), None
    )

    # A previous round's selected departure is never silently carried over:
    # the report states explicitly whether it still holds or lapsed.
    prior_selection_status = "none"
    prior_selection_note = ""
    if prior_selected is not None:
        if historical_status == "needs_recheck":
            prior_selection_status = "invalidated"
            prior_selection_note = (
                f"旧选择 {prior_selected:.3f}s 失效：场景签名或路线指纹已变化，"
                "旧证据全部重新核对，旧选择未沿用"
            )
        else:
            old_probe = coverage_evidence.get(round(prior_selected, 9))
            if old_probe is not None and old_probe.feasible:
                prior_selection_status = "kept"
                prior_selection_note = (
                    f"旧选择 {prior_selected:.3f}s 仍然有效：该时刻在当前证据中"
                    "已核对可行"
                )
            elif old_probe is not None:
                prior_selection_status = "invalidated"
                prior_selection_note = (
                    f"旧选择 {prior_selected:.3f}s 失效：当前证据表明该时刻存在冲突"
                )
            else:
                prior_selection_status = "invalidated"
                prior_selection_note = (
                    f"旧选择 {prior_selected:.3f}s 失效：该时刻不在当前已核对集合内"
                    "（范围缩小或采样网格变化），无法确认，未沿用"
                )

    return DepartureWindowAnalysis(
        earliest_departure=earliest_departure,
        latest_departure=latest_departure,
        step=step,
        coverage_step=coverage_step,
        probes_requested=probes_requested,
        probes_completed=len(point_evidence),
        probes_this_round=len(newly_probed) + len(newly_swept),
        newly_probed=tuple(newly_probed),
        newly_swept=tuple(newly_swept),
        continued=continued,
        budget_exhausted=budget_exhausted,
        boundaries=boundaries,
        windows=tuple(annotated),
        probes=tuple(
            point_evidence[key] for key in sorted(point_evidence)
        ),
        sweeps=tuple(ordered),
        selected_departure=selected_departure,
        selected_verified=selected_departure is not None,
        schedule_signature=new_signature,
        scene_name=new_scene.name,
        route_fingerprint=fingerprint,
        historical_signature=historical_signature,
        historical_status=historical_status,
        prior_selected_departure=prior_selected,
        prior_selection_status=prior_selection_status,
        prior_selection_note=prior_selection_note,
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
    departure_coverage_step: float = DEFAULT_COVERAGE_STEP,
    departure_budget: int = 300,
    historical_report: dict | None = None,
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
            coverage_step=departure_coverage_step,
            max_probes=departure_budget,
            prior=historical_report,
        )

    selected_plan = None
    if departure_analysis is not None and departure_analysis.selected_departure is not None:
        _, _, selected_plan = _probe_departure(
            aligned,
            vehicle,
            new_scene,
            departure_analysis.selected_departure,
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
