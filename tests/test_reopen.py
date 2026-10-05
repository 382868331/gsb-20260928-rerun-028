"""Reopen workflow: a timetable change replans the whole question, never
silently re-validates or rewrites the old success record.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vehicle_planning.cli import main
from vehicle_planning.plan import load_plan, save_plan
from vehicle_planning.reopen import (
    analyze_departure_windows,
    attach_historical_windows,
    evaluate_reopen,
    find_delayed_plan,
    same_route,
    same_timing,
    save_reopen_bundle,
    scan_conflicts,
)
from vehicle_planning.replay import replay
from vehicle_planning.scene import scene_from_dict
from vehicle_planning.scheduler import Scheduler, SchedulerConfig
from vehicle_planning.vehicle import load_vehicle

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "data/scenes/pedestrian_crossing.json"

# Obstacle arrives early and crosses the corridor entrance inside the
# original ~4.2-4.4s passage window.
EARLY = [[0.0, 11.0, 4.0, 0.0], [4.0, 11.0, 4.0, 0.0], [8.0, -8.0, 4.0, 0.0]]
# Obstacle is still up north and crosses later than the original schedule.
LATE = [[0.0, 11.0, 4.5, 0.0], [6.0, 11.0, 4.5, 0.0], [12.0, -15.0, 4.5, 0.0]]
# Obstacle leaves before the vehicle arrives, but the timetable hash differs.
EARLY_LEAVE = [[0.0, -8.0, 4.5, 0.0], [3.0, -2.0, 4.5, 0.0], [6.0, -15.0, 4.5, 0.0]]
# Obstacle timetable stays far away from the route at all times.
AVOID = [[0.0, -15.0, 9.5, 0.0], [10.0, -15.0, 9.5, 0.0], [20.0, -25.0, 9.5, 0.0]]
# Obstacle briefly blocks the corridor around t=19.5-21.0s, so only middle
# departure shifts (~1-5.5s) collide; both range endpoints are feasible.
MIDDLE = [
    [0.0, -30.0, 6.0, 0.0],
    [19.4, -30.0, 6.0, 0.0],
    [19.5, 6.9, 6.0, 0.0],
    [21.0, 6.9, 6.0, 0.0],
    [21.1, -30.0, 6.0, 0.0],
    [30.0, -30.0, 6.0, 0.0],
]


@pytest.fixture(scope="module")
def base():
    vehicle = load_vehicle(ROOT / "data/vehicles/compact_car.json")
    data = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    base_scene = scene_from_dict(data)
    geometric = replace(
        load_plan(ROOT / "outputs/reverse_into_bay.json"),
        scene=base_scene.name,
    )
    plan = Scheduler(vehicle, base_scene, SchedulerConfig()).schedule(geometric)
    assert replay(plan, vehicle, base_scene).valid
    return vehicle, data, plan


def changed_scene(data, keyframes):
    payload = copy.deepcopy(data)
    payload["dynamic_obstacles"][0]["keyframes"] = keyframes
    return scene_from_dict(payload)


@pytest.fixture()
def cheap_replan(base, monkeypatch):
    """Re-plan on the new timetable using the same saved geometry.

    Keeps the route identical on purpose so tests can assert that route and
    timing changes are reported independently; the real geometric search is
    covered by test_motion_planning.
    """
    import vehicle_planning.reopen as reopen_module

    _, _, plan = base

    def run_full_replan(vehicle, new_scene, **kwargs):
        scheduler = Scheduler(
            vehicle, new_scene, kwargs.get("scheduler_config") or SchedulerConfig()
        )
        return scheduler.schedule(replace(plan, actions=(), scene=new_scene.name))

    monkeypatch.setattr(reopen_module, "run_full_replan", run_full_replan)


def test_changed_timetable_route_faces_new_schedule_on_its_own_merits(base, cheap_replan):
    vehicle, data, plan = base
    for frames in (EARLY_LEAVE, AVOID):
        scene = changed_scene(data, frames)
        # The signature-bound old plan is not reusable as-is ...
        assert replay(plan, vehicle, scene).valid is False
        assessment = evaluate_reopen(plan, vehicle, scene, max_delay=0.0)
        assert assessment.timetable_changed is True
        # ... but the original route, judged independently on the new timetable
        # with zero conflicts, must get the same verdict as a zero-second
        # "delay" (whose timing is unchanged).
        assert assessment.conflicts == ()
        assert assessment.original_feasible is True
        assert assessment.option("original").timing_changed is False
        assert assessment.option("delayed").feasible is True
        assert assessment.option("delayed").departure_delay == 0.0
        assert assessment.recommendation == "original"
        # The saved copy is rebound to the new timetable and replays on it.
        rebound = assessment.option_plans["original"]
        assert rebound.scene_signature == scene.signature()
        assert same_route(rebound, plan) and same_timing(rebound, plan)
        assert replay(rebound, vehicle, scene).valid
        assert assessment.conflicts == ()


def test_conflict_intervals_are_reported_without_touching_route(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    conflicts = scan_conflicts(plan, vehicle, scene)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.obstacles == ("trolley",)
    assert 4.0 < conflict.start_time < conflict.end_time < 5.5
    assert conflict.samples >= 5
    assert replay(plan, vehicle, scene_from_dict(data)).valid


def test_delay_is_tried_before_route_change_and_changes_only_timing(base, cheap_replan):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_reopen(plan, vehicle, scene, max_delay=10, delay_step=0.5)

    assert assessment.original_feasible is False
    delayed = assessment.option("delayed")
    assert delayed.feasible is True
    assert delayed.route_changed is False
    assert delayed.same_route is True
    assert delayed.timing_changed is True
    assert delayed.departure_delay == pytest.approx(0.5)
    assert delayed.final_time == pytest.approx(plan.final_time + delayed.departure_delay)
    assert assessment.recommendation == "delayed"

    delayed_plan = find_delayed_plan(plan, vehicle, scene, max_delay=10, delay_step=0.5)[1]
    assert same_route(plan, delayed_plan)
    assert not same_timing(plan, delayed_plan)
    assert replay(delayed_plan, vehicle, scene).valid


def test_late_obstacle_conflicts_on_original_schedule_but_delay_resolves(base, cheap_replan):
    vehicle, data, plan = base
    scene = changed_scene(data, LATE)
    assessment = evaluate_reopen(plan, vehicle, scene, max_delay=10, delay_step=0.5)

    assert assessment.original_feasible is False
    assert len(assessment.conflicts) >= 1
    delayed = assessment.option("delayed")
    assert delayed.feasible is True
    assert delayed.departure_delay > 0.0
    assert delayed.route_changed is False
    assert delayed.timing_changed is True


def test_departure_windows_classify_safe_blocked_and_boundaries(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
    )
    assert analysis.probes_requested == 17
    assert analysis.budget_exhausted is False
    labels = {item.label: item for item in analysis.boundaries}
    assert labels["earliest"].departure_time == 0.0
    assert labels["latest"].departure_time == 8.0
    kinds = [window.kind for window in analysis.windows]
    assert "safe" in kinds and "blocked" in kinds and "unverified" not in kinds
    blocked = [window for window in analysis.windows if window.kind == "blocked"]
    assert all(window.obstacles for window in blocked)
    # A time selected from a safe window replays the whole route successfully.
    assert analysis.selected_departure is not None
    delayed_plan = find_delayed_plan(
        plan, vehicle, scene, max_delay=8, delay_step=0.5
    )[1]
    assert replay(delayed_plan, vehicle, scene).valid
    assert delayed_plan.actions[0].type == "wait" or analysis.selected_departure == 0.0


def test_departure_window_budget_leaves_unverified_tail_not_safe(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=10.0, step=0.5, max_probes=4,
    )
    assert analysis.budget_exhausted is True
    assert analysis.probes_completed == 4
    tail = [w for w in analysis.windows if w.kind == "unverified"]
    assert len(tail) == 1
    assert tail[0].end_time == 10.0


def _assert_partition(analysis):
    """Windows tile the range exactly once: no overlaps, no gaps."""
    cursor = analysis.earliest_departure
    for window in analysis.windows:
        assert window.start_time == pytest.approx(cursor, abs=1e-6)
        assert window.end_time >= window.start_time - 1e-9
        cursor = window.end_time
    assert cursor == pytest.approx(analysis.latest_departure, abs=1e-6)


def test_single_probe_budget_gives_no_contradictory_verdicts(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=10.0, step=0.5, max_probes=1,
    )
    assert analysis.budget_exhausted is True
    assert analysis.probes_completed == 1
    _assert_partition(analysis)
    # The lone probed point and the unverified tail never overlap: a single
    # probe stands for its own moment only, not a half-step cell.
    verified = [w for w in analysis.windows if w.kind != "unverified"]
    assert len(verified) == 1
    assert verified[0].start_time == 0.0 and verified[0].end_time == pytest.approx(0.0)
    tail = [w for w in analysis.windows if w.kind == "unverified"]
    assert len(tail) == 1 and tail[0].start_time == pytest.approx(0.0)
    assert tail[0].end_time == pytest.approx(10.0)
    # ... the unreached latest boundary is unknown, not a failure ...
    labels = {item.label: item for item in analysis.boundaries}
    assert labels["earliest"].probed is True
    assert labels["latest"].probed is False
    assert labels["latest"].feasible is None
    # ... and no executable departure is selected without evidence.
    assert analysis.selected_departure is None
    assert analysis.selected_verified is False
    assert analysis.to_dict()["selection"] is None


def test_budget_topup_continues_search_and_keeps_evidence(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    first = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5, max_probes=4,
    )
    assert first.budget_exhausted is True
    resumed = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
        max_probes=300, prior=first,
    )
    assert resumed.continued is True
    assert resumed.historical_status == "same_timetable"
    assert resumed.probes_completed == 17
    assert resumed.probes_this_round == 13
    # Old evidence is reused verbatim, only new positions consumed budget.
    old = {p.departure_time: p for p in first.probes}
    new = {p.departure_time: p for p in resumed.probes}
    for at_time, probe in old.items():
        assert new[at_time].feasible == probe.feasible
    assert set(resumed.newly_probed).isdisjoint(old)
    assert all(w.kind != "unverified" for w in resumed.windows)
    _assert_partition(resumed)


def test_resubmission_and_restore_do_not_inflate_check_counts(base, tmp_path):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    first = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
    )
    # Re-running the same search with the same candidates adds nothing.
    again = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
        max_probes=300, prior=first,
    )
    assert again.probes_this_round == 0
    assert again.newly_probed == ()
    assert again.probes_completed == first.probes_completed
    # A saved-and-restored report continues identically (no phantom checks).
    payload = json.loads(json.dumps(first.to_dict()))
    restored = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
        max_probes=300, prior={"departure_windows": payload},
    )
    assert restored.probes_this_round == 0
    assert restored.probes_completed == first.probes_completed


def test_changing_step_leaves_intermediate_times_unverified(base):
    vehicle, data, plan = base
    scene = changed_scene(data, AVOID)
    coarse = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=4.0, step=1.0,
    )
    assert all(w.kind == "safe" for w in coarse.windows)
    # Finer step, zero extra budget: probed integer times stay safe, the
    # half-integer gaps are unverified -- endpoints passing does not imply
    # the middle passes.
    refined = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=4.0, step=0.5,
        max_probes=0, prior=coarse,
    )
    assert refined.probes_this_round == 0
    kinds = [w.kind for w in refined.windows]
    assert "unverified" in kinds and "safe" in kinds
    _assert_partition(refined)
    gaps = [w for w in refined.windows if w.kind == "unverified"]
    # Each integer probe is an isolated point verdict; everything between
    # the integers is unknown, not split at midpoints.
    assert any(
        g.start_time == pytest.approx(0.0) and g.end_time == pytest.approx(1.0)
        for g in gaps
    )
    points = [w for w in refined.windows if w.kind == "safe"]
    assert all(w.start_time == pytest.approx(w.end_time) for w in points)


def test_zero_width_range_probes_single_moment(base):
    vehicle, data, plan = base
    scene = changed_scene(data, AVOID)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=2.0, latest_departure=2.0, step=0.5,
    )
    assert analysis.probes_requested == 1
    assert analysis.probes_completed == 1
    assert len(analysis.windows) == 1
    window = analysis.windows[0]
    assert window.kind == "safe"
    assert window.start_time == window.end_time == 2.0
    assert all(b.probed and b.feasible for b in analysis.boundaries)
    assert analysis.selected_departure == 2.0


def test_endpoints_only_budget_leaves_middle_unknown(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5, max_probes=2,
    )
    assert analysis.probes_completed == 2
    assert analysis.budget_exhausted is True
    _assert_partition(analysis)
    # Only the two endpoint moments carry verdicts; the whole interior is
    # unknown -- never split at the midpoint into blocked/safe halves.
    verified = [w for w in analysis.windows if w.kind != "unverified"]
    assert len(verified) == 2
    assert all(w.start_time == pytest.approx(w.end_time) for w in verified)
    assert verified[0].start_time == 0.0 and verified[0].kind == "blocked"
    assert verified[1].start_time == 8.0 and verified[1].kind == "safe"
    middle = [w for w in analysis.windows if w.kind == "unverified"]
    assert len(middle) == 1
    assert middle[0].start_time == pytest.approx(0.0)
    assert middle[0].end_time == pytest.approx(8.0)
    # The probed feasible endpoint is still a valid, verified selection.
    assert analysis.selected_departure == 8.0


def test_narrow_middle_conflict_not_hidden_by_sparse_probes(base):
    vehicle, data, plan = base
    scene = changed_scene(data, MIDDLE)
    sparse = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5, max_probes=2,
    )
    # Both endpoints probe safe, yet the middle must stay unknown: endpoint
    # evidence never promotes the unprobed interior to safe.
    labels = {item.label: item for item in sparse.boundaries}
    assert labels["earliest"].feasible is True
    assert labels["latest"].feasible is True
    middle = [w for w in sparse.windows if w.kind == "unverified"]
    assert len(middle) == 1
    assert middle[0].start_time == pytest.approx(0.0)
    assert middle[0].end_time == pytest.approx(8.0)
    assert all(w.kind != "safe" or w.start_time == w.end_time for w in sparse.windows)
    # ... and topping up the budget resolves exactly the previously
    # unknown middle, exposing the narrow blocked window hiding there.
    resumed = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
        max_probes=300, prior=sparse,
    )
    assert resumed.probes_this_round == 15
    assert all(w.kind != "unverified" for w in resumed.windows)
    blocked = [w for w in resumed.windows if w.kind == "blocked"]
    assert blocked and all(w.obstacles for w in blocked)
    assert any(w.start_time > 0.0 and w.end_time < 8.0 for w in blocked)


def test_coarse_step_endpoints_only_never_cover_whole_range(base):
    vehicle, data, plan = base
    scene = changed_scene(data, MIDDLE)
    # A step as large as the whole span requests only the two endpoints.
    # Both endpoints are feasible, but enlarging the step must not turn the
    # unprobed interior into one continuous safe window: with no interior
    # moment ever checked, the endpoints stay isolated point verdicts.
    for coarse_step in (8.0, 20.0):
        analysis = analyze_departure_windows(
            plan, vehicle, scene,
            earliest_departure=0.0, latest_departure=8.0, step=coarse_step,
        )
        assert analysis.probes_completed == 2
        verified = [w for w in analysis.windows if w.kind != "unverified"]
        assert len(verified) == 2
        assert all(w.start_time == pytest.approx(w.end_time) for w in verified)
        assert all(w.kind == "safe" for w in verified)
        unknown = [w for w in analysis.windows if w.kind == "unverified"]
        assert len(unknown) == 1
        assert unknown[0].start_time == pytest.approx(0.0)
        assert unknown[0].end_time == pytest.approx(8.0)
        _assert_partition(analysis)

    # Non-integer bounds behave identically: a step larger than the span
    # probes the two boundary moments only; between them stays unknown.
    shifted = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.3, latest_departure=3.7, step=9.9,
    )
    assert shifted.probes_requested == 2 and shifted.probes_completed == 2
    unknown = [w for w in shifted.windows if w.kind == "unverified"]
    assert len(unknown) == 1
    assert unknown[0].start_time == pytest.approx(0.3)
    assert unknown[0].end_time == pytest.approx(3.7)

    # A zero-budget reopen changes no evidence: the two isolated endpoint
    # points are reused for free and the interior remains unknown.
    unchanged = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=8.0,
        max_probes=0, prior=analysis,
    )
    assert unchanged.probes_this_round == 0
    assert unchanged.probes_completed == 2
    assert [w.kind for w in unchanged.windows] == ["safe", "unverified", "safe"]

    # Extra budget at a real sampling step expands only the territory the
    # new probes actually cover; the narrow middle conflict surfaces
    # instead of being inherited from the endpoint agreement.
    expanded = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
        max_probes=300, prior=analysis,
    )
    assert expanded.probes_this_round == 15
    blocked = [w for w in expanded.windows if w.kind == "blocked"]
    assert blocked and all(w.obstacles for w in blocked)
    assert any(0.0 < w.start_time and w.end_time < 8.0 for w in blocked)


def test_non_integer_start_and_step_resume_without_double_counting(base):
    vehicle, data, plan = base
    scene = changed_scene(data, AVOID)
    first = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.3, latest_departure=7.8, step=0.7, max_probes=3,
    )
    assert first.probes_completed == 3
    assert first.budget_exhausted is True
    resumed = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.3, latest_departure=7.8, step=0.7,
        max_probes=300, prior=first,
    )
    assert resumed.probes_completed == resumed.probes_requested
    assert resumed.probes_this_round == resumed.probes_requested - 3
    assert set(resumed.newly_probed).isdisjoint(p.departure_time for p in first.probes)
    # Re-submitting the restored result costs nothing more.
    again = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.3, latest_departure=7.8, step=0.7,
        max_probes=300, prior=resumed.to_dict(),
    )
    assert again.probes_this_round == 0
    assert again.probes_completed == resumed.probes_completed


def test_prior_selection_invalidation_is_explicit(base):
    vehicle, data, plan = base
    scene = changed_scene(data, AVOID)
    first = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=5.0, step=0.5,
    )
    assert first.selected_departure == 0.0

    # Same timetable, selection still verified: explicitly kept.
    kept = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=5.0, step=0.5,
        max_probes=300, prior=first,
    )
    assert kept.prior_selected_departure == 0.0
    assert kept.prior_selection_status == "kept"

    # Same timetable but the old selection falls outside the checked set:
    # explicitly invalidated, never silently dropped.
    narrowed = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=2.0, latest_departure=5.0, step=0.5,
        max_probes=300, prior=first,
    )
    assert narrowed.prior_selected_departure == 0.0
    assert narrowed.prior_selection_status == "invalidated"
    assert narrowed.prior_selection_note

    # Changed timetable: old selection lapses with the rest of the evidence.
    other = changed_scene(data, LATE)
    rechecked = analyze_departure_windows(
        plan, vehicle, other,
        earliest_departure=0.0, latest_departure=5.0, step=0.5,
        max_probes=300, prior=first,
    )
    assert rechecked.historical_status == "needs_recheck"
    assert rechecked.prior_selection_status == "invalidated"
    assert str(rechecked.prior_selected_departure) in rechecked.prior_selection_note


def test_scene_change_invalidates_prior_evidence(base):
    vehicle, data, plan = base
    scene_early = changed_scene(data, EARLY)
    first = analyze_departure_windows(
        plan, vehicle, scene_early,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
    )
    scene_late = changed_scene(data, LATE)
    resumed = analyze_departure_windows(
        plan, vehicle, scene_late,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
        max_probes=300, prior=first,
    )
    # Old and new timetables are never stitched into one safe interval.
    assert resumed.historical_status == "needs_recheck"
    assert resumed.continued is False
    assert resumed.probes_this_round == resumed.probes_completed
    assert resumed.schedule_signature == scene_late.signature()


def test_selection_records_scene_route_and_verdict(base):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=8.0, step=0.5,
    )
    selection = analysis.to_dict()["selection"]
    assert selection["departure_time"] == pytest.approx(analysis.selected_departure)
    assert selection["scene"] == scene.name
    assert selection["scene_signature"] == scene.signature()
    assert selection["route_fingerprint"] == analysis.route_fingerprint
    assert selection["verified"] is True and selection["executable"] is True


def test_departure_windows_all_safe_when_obstacle_moves_away(base):
    vehicle, data, plan = base
    scene = changed_scene(data, AVOID)
    analysis = analyze_departure_windows(
        plan, vehicle, scene,
        earliest_departure=0.0, latest_departure=5.0, step=0.5,
    )
    assert all(w.kind == "safe" for w in analysis.windows)
    assert all(b.feasible for b in analysis.boundaries)
    assert analysis.selected_departure == 0.0


def test_historical_report_is_tagged_never_adopted(base, tmp_path):
    vehicle, data, plan = base
    scene_late = changed_scene(data, LATE)
    analysis = analyze_departure_windows(
        plan, vehicle, scene_late,
        earliest_departure=0.0, latest_departure=6.0, step=0.5,
    )
    same = attach_historical_windows(
        analysis,
        {"departure_windows": {"schedule_signature": scene_late.signature()}},
    )
    assert same.historical_status == "same_timetable"
    other_scene = changed_scene(data, EARLY)
    stale = attach_historical_windows(
        analysis,
        {"departure_windows": {"schedule_signature": other_scene.signature()}},
    )
    assert stale.historical_status == "needs_recheck"
    # Current safe/blocked verdicts come from fresh probes, never from history.
    assert all(w.kind != "unverified" for w in stale.windows)


def test_reopen_includes_selected_window_plan_and_preserves_history(base, cheap_replan, tmp_path):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_reopen(
        plan, vehicle, scene, max_delay=10, delay_step=0.5,
        earliest_departure=0.0, latest_departure=8.0, departure_step=0.5,
    )
    windows = assessment.departure_windows
    selected = windows.selected_departure
    assert selected is not None
    selected_plan = assessment.option_plans["selected_window"]
    assert selected_plan.scene_signature == scene.signature()
    assert replay(selected_plan, vehicle, scene).valid

    report_path = tmp_path / "report.json"
    save_reopen_bundle(assessment, None, report_path)
    assert (tmp_path / "report_selected_window.json").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["departure_windows"]["selected_verified"] is True

    # Reopen again on a changed arrangement, handing in the previous report:
    # its windows must be flagged as historical / needing recheck.
    scene2 = changed_scene(data, LATE)
    assessment2 = evaluate_reopen(
        plan, vehicle, scene2, max_delay=10, delay_step=0.5,
        earliest_departure=0.0, latest_departure=8.0, departure_step=0.5,
        historical_report=payload,
    )
    assert assessment2.departure_windows.historical_status == "needs_recheck"

def test_replanned_option_uses_new_timetable_and_flags_route_vs_timing(base, cheap_replan):
    vehicle, data, plan = base
    scene = changed_scene(data, EARLY)
    assessment = evaluate_reopen(plan, vehicle, scene, max_delay=0.1)

    assert assessment.option("delayed").feasible is False
    replanned = assessment.option("replanned")
    assert replanned.feasible is True
    # The cheap replanner keeps geometry but regenerates the schedule.
    assert replanned.same_route is True
    assert replanned.route_changed is False
    assert replanned.scene_signature == scene.signature()


def test_bundle_preserves_old_success_and_binds_each_option_to_new_timetable(
    base, cheap_replan, tmp_path
):
    vehicle, data, plan = base
    scene = changed_scene(data, LATE)
    assessment = replace(
        evaluate_reopen(plan, vehicle, scene, max_delay=10, delay_step=0.5),
        original_plan_path="outputs/pedestrian_crossing.json",
    )
    delayed_plan = assessment.option_plans["delayed"]
    replanned = assessment.option_plans["replanned"]

    original_path = tmp_path / "original_success.json"
    save_plan(plan, original_path)
    before = original_path.read_bytes()

    report_path = tmp_path / "reopen/report.json"
    save_reopen_bundle(assessment, None, report_path)

    assert original_path.read_bytes() == before
    saved_delayed = load_plan(tmp_path / "reopen/report_delayed.json")
    saved_replanned = load_plan(tmp_path / "reopen/report_replanned.json")
    assert saved_delayed.scene_signature == scene.signature()
    assert saved_replanned.scene_signature == scene.signature()
    assert replay(saved_delayed, vehicle, scene).valid
    assert replay(saved_replanned, vehicle, scene).valid
    assert not replay(saved_delayed, vehicle, scene_from_dict(data)).valid

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "reopen_assessment_v1"
    assert set(report["option_plan_files"]) == {"delayed", "replanned"}
    for option in report["options"].values():
        assert option["scene_signature"] == scene.signature()


def test_cli_reopen_writes_bundle_without_modifying_plan_file(base, cheap_replan, tmp_path):
    _, data, plan = base
    original_path = tmp_path / "original.json"
    save_plan(plan, original_path)
    before = original_path.read_bytes()

    scene_path = tmp_path / "scene_late.json"
    scene_path.write_text(json.dumps(changed_scene(data, LATE).to_dict()), encoding="utf-8")
    report_path = tmp_path / "reopen_report.json"

    exit_code = main([
        "reopen",
        "--vehicle", str(ROOT / "data/vehicles/compact_car.json"),
        "--scene", str(scene_path),
        "--plan", str(original_path),
        "--output", str(report_path),
        "--max-delay", "10",
        "--delay-step", "0.5",
    ])
    assert exit_code == 0
    assert original_path.read_bytes() == before
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["timetable_changed"] is True
    assert report["recommendation"] == "delayed"
    assert (tmp_path / "reopen_report_delayed.json").exists()
    assert (tmp_path / "reopen_report_replanned.json").exists()
