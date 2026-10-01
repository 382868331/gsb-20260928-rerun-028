"""Safe departure-window analysis for a saved timed plan.

Instead of trying a single delayed departure, the user provides an earliest
and latest departure offset. Every candidate offset shifts the original
timeline (route geometry untouched) and is independently replayed against
the current obstacle timetable. Consecutive results merge into intervals:

* "safe"         - verified collision-free departures;
* "blocked"      - departures that collide, with the obstacle names that
  break them;
* "undetermined" - offsets never evaluated because the check budget ran
  out. Undetermined is never reported as safe.

Reports carry the scene signature of the timetable they were computed on.
After the obstacle schedule changes, a saved report can be reviewed against
the new scene: old intervals are explicitly marked as historical
(needs_recheck) instead of being silently reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .plan import PlanResult, load_plan, save_plan
from .replay import replay
from .reopen import _shift_actions, scan_conflicts
from .scene import Scene, load_scene
from .vehicle import Vehicle, load_vehicle

SAFE = "safe"
BLOCKED = "blocked"
UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class DepartureInterval:
    """A contiguous run of departure offsets sharing one status."""

    start: float
    end: float
    status: str
    obstacles: tuple[str, ...]
    checked_points: int

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "status": self.status,
            "obstacles": list(self.obstacles),
            "checked_points": self.checked_points,
        }


@dataclass(frozen=True)
class DepartureWindowAssessment:
    original_plan_path: str
    scene: str
    scene_signature: str
    earliest: float
    latest: float
    step: float
    max_checks: int
    checks_used: int
    budget_exhausted: bool
    earliest_status: str
    latest_status: str
    intervals: tuple[DepartureInterval, ...]
    point_status: tuple[tuple[float, str], ...]

    def status_at(self, departure: float, tolerance: float = 1e-6) -> str | None:
        for point, status in self.point_status:
            if abs(point - departure) <= tolerance:
                return status
        return None

    def to_dict(self) -> dict:
        return {
            "schema": "departure_window_v1",
            "original_plan_path": self.original_plan_path,
            "scene": self.scene,
            "scene_signature": self.scene_signature,
            "earliest": round(self.earliest, 3),
            "latest": round(self.latest, 3),
            "step": round(self.step, 3),
            "max_checks": self.max_checks,
            "checks_used": self.checks_used,
            "budget_exhausted": self.budget_exhausted,
            "boundary": {
                "earliest": {"time": round(self.earliest, 3), "status": self.earliest_status},
                "latest": {"time": round(self.latest, 3), "status": self.latest_status},
            },
            "intervals": [item.to_dict() for item in self.intervals],
        }


def _departure_grid(earliest: float, latest: float, step: float) -> list[float]:
    if earliest < -1e-9:
        raise ValueError("earliest departure offset must be >= 0")
    if latest < earliest - 1e-9:
        raise ValueError("latest departure must be >= earliest departure")
    if step <= 0:
        raise ValueError("departure step must be positive")
    points = []
    value = earliest
    while value < latest - 1e-9:
        points.append(round(value, 9))
        value += step
    points.append(round(latest, 9))
    return points


def _evaluation_order(count: int) -> list[int]:
    """Boundaries first, then interior points left to right."""
    if count <= 2:
        return list(range(count))
    return [0, count - 1, *range(1, count - 1)]


def evaluate_departure_window(
    original: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    *,
    earliest: float,
    latest: float,
    step: float = 0.5,
    max_checks: int = 200,
) -> DepartureWindowAssessment:
    """Classify each candidate departure offset as safe/blocked/undetermined.

    The original plan is read-only; every candidate is a time-shifted copy
    re-signed with the current timetable and fully replayed. Each replay
    consumes one unit of the check budget; offsets never reached stay
    undetermined and are never reported as safe.
    """
    if not original.timed:
        raise ValueError("departure window analysis requires a timed plan")
    if original.vehicle != vehicle.name:
        raise ValueError("departure window requires the same vehicle as the original plan")
    if max_checks < 1:
        raise ValueError("max_checks must be at least 1")

    signature = scene.signature()
    aligned = (
        original
        if original.scene == scene.name
        else replace(original, scene=scene.name)
    )
    grid = _departure_grid(earliest, latest, step)
    status_by_index: dict[int, str] = {}
    obstacles_by_index: dict[int, tuple[str, ...]] = {}
    checks_used = 0

    for index in _evaluation_order(len(grid)):
        if checks_used >= max_checks:
            break
        delay = grid[index]
        actions = _shift_actions(
            aligned.actions,
            delay,
            f"departure-window probe, shifted by {delay:.2f}s (route unchanged)",
        )
        candidate = replace(aligned, actions=actions, scene_signature=signature)
        checks_used += 1
        if replay(candidate, vehicle, scene).valid:
            status_by_index[index] = SAFE
            obstacles_by_index[index] = ()
        else:
            status_by_index[index] = BLOCKED
            names: set[str] = set()
            for conflict in scan_conflicts(candidate, vehicle, scene):
                names.update(conflict.obstacles)
            obstacles_by_index[index] = tuple(sorted(names))

    point_status = tuple(
        (grid[index], status_by_index.get(index, UNDETERMINED))
        for index in range(len(grid))
    )

    intervals: list[DepartureInterval] = []
    for index, (point, status) in enumerate(point_status):
        obstacles = obstacles_by_index.get(index, ())
        if intervals and intervals[-1].status == status and (
            status != BLOCKED or intervals[-1].obstacles == obstacles
        ):
            last = intervals[-1]
            intervals[-1] = DepartureInterval(
                last.start,
                point,
                status,
                last.obstacles,
                last.checked_points + (0 if status == UNDETERMINED else 1),
            )
        else:
            intervals.append(
                DepartureInterval(
                    point,
                    point,
                    status,
                    obstacles,
                    0 if status == UNDETERMINED else 1,
                )
            )

    budget_exhausted = any(status == UNDETERMINED for _, status in point_status)
    return DepartureWindowAssessment(
        original_plan_path="",
        scene=scene.name,
        scene_signature=signature,
        earliest=earliest,
        latest=latest,
        step=step,
        max_checks=max_checks,
        checks_used=checks_used,
        budget_exhausted=budget_exhausted,
        earliest_status=point_status[0][1],
        latest_status=point_status[-1][1],
        intervals=tuple(intervals),
        point_status=point_status,
    )


def select_departure(
    assessment: DepartureWindowAssessment,
    original: PlanResult,
    vehicle: Vehicle,
    scene: Scene,
    departure: float,
) -> PlanResult:
    """Build a verified plan departing at a chosen safe offset.

    The chosen offset must come from a safe point of an assessment made on
    the same timetable; the shifted plan is then fully replayed once more
    against the current scene before it is returned.
    """
    if assessment.scene_signature != scene.signature():
        raise ValueError(
            "assessment belongs to an older timetable; rerun the window analysis"
        )
    if assessment.status_at(departure) != SAFE:
        raise ValueError(
            f"departure {departure:.2f}s is not a verified safe offset "
            f"(status: {assessment.status_at(departure) or 'outside evaluated grid'})"
        )
    aligned = (
        original
        if original.scene == scene.name
        else replace(original, scene=scene.name)
    )
    actions = _shift_actions(
        aligned.actions,
        departure,
        f"selected departure inside safe window at {departure:.2f}s (route unchanged)",
    )
    candidate = replace(aligned, actions=actions, scene_signature=scene.signature())
    report = replay(candidate, vehicle, scene)
    if not report.valid:
        raise ValueError(f"selected departure failed final verification: {report.message}")
    return candidate


def review_saved_window(saved_report: dict, scene: Scene) -> dict:
    """Re-check a saved window report against the current timetable.

    Intervals computed under a different obstacle schedule are historical
    conclusions and are marked needs_recheck; only a matching signature
    keeps them current.
    """
    if saved_report.get("schema") != "departure_window_v1":
        raise ValueError("not a departure_window_v1 report")
    current_signature = scene.signature()
    matches = saved_report.get("scene_signature") == current_signature
    reviewed = []
    for interval in saved_report.get("intervals", []):
        reviewed.append(
            {
                **interval,
                "review": "current" if matches else "needs_recheck",
                "basis": (
                    "same timetable"
                    if matches
                    else "historical conclusion under previous timetable "
                    f"{saved_report.get('scene_signature', '')[:12]}..."
                ),
            }
        )
    return {
        "schema": "departure_window_review_v1",
        "scene_signature": current_signature,
        "saved_scene_signature": saved_report.get("scene_signature"),
        "timetable_changed": not matches,
        "intervals": reviewed,
    }


def save_window_report(
    assessment: DepartureWindowAssessment,
    report_path: str | Path,
    *,
    selected: tuple[float, PlanResult] | None = None,
    previous_report_path: str | Path | None = None,
    scene: Scene | None = None,
) -> dict:
    """Persist the window report, an optional selected plan, and an optional
    review of a previously saved report against the current timetable."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = assessment.to_dict()
    if selected is not None:
        departure, plan = selected
        plan_path = report_path.with_name(f"{report_path.stem}_depart_{departure:.2f}s.json")
        save_plan(plan, plan_path)
        payload["selected"] = {
            "departure": round(departure, 3),
            "plan_file": plan_path.name,
            "verified": True,
            "scene_signature": plan.scene_signature,
        }
    if previous_report_path is not None:
        if scene is None:
            raise ValueError("scene is required to review a previous report")
        saved = json.loads(Path(previous_report_path).read_text(encoding="utf-8"))
        payload["previous_review"] = review_saved_window(saved, scene)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def window_from_files(
    plan_path: str | Path,
    vehicle_path: str | Path,
    scene_path: str | Path,
    **kwargs,
) -> tuple[PlanResult, Vehicle, Scene, DepartureWindowAssessment]:
    original = load_plan(plan_path)
    vehicle = load_vehicle(vehicle_path)
    scene = load_scene(scene_path)
    assessment = evaluate_departure_window(original, vehicle, scene, **kwargs)
    assessment = replace(assessment, original_plan_path=str(plan_path))
    return original, vehicle, scene, assessment
