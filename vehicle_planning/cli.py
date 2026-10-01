from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .motion import standard_primitives
from .plan import load_plan, save_plan
from .planner import Planner, PlannerConfig
from .replay import render_svg, replay
from .departure import (
    evaluate_departure_window,
    save_window_report,
    select_departure,
)
from .reopen import (
    evaluate_reopen,
    save_reopen_bundle,
)
from .scheduler import Scheduler, SchedulerConfig
from .scene import load_scene
from .vehicle import load_vehicle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-speed kinematic vehicle motion planner")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="search a forward/reverse path")
    plan_parser.add_argument("--vehicle", required=True)
    plan_parser.add_argument("--scene", required=True)
    plan_parser.add_argument("--output", required=True)
    plan_parser.add_argument("--max-nodes", type=int, default=60_000)
    plan_parser.add_argument("--primitive-length", type=float, nargs="+", default=[0.4, 0.75])
    plan_parser.add_argument("--xy-resolution", type=float, default=0.25)
    plan_parser.add_argument("--theta-resolution", type=float, default=0.12)
    plan_parser.add_argument(
        "--timed",
        action="store_true",
        help="build a timed schedule with waits for dynamic obstacles",
    )
    plan_parser.add_argument("--cruise-speed", type=float, default=1.2)
    plan_parser.add_argument("--gear-switch-time", type=float, default=1.5)

    replay_parser = sub.add_parser("replay", help="replay a saved plan and render SVG")
    replay_parser.add_argument("--vehicle", required=True)
    replay_parser.add_argument("--scene", required=True)
    replay_parser.add_argument("--plan", required=True)
    replay_parser.add_argument("--svg")

    reopen_parser = sub.add_parser(
        "reopen",
        help="reopen a successful plan against a changed obstacle timetable",
    )
    reopen_parser.add_argument("--vehicle", required=True)
    reopen_parser.add_argument("--scene", required=True, help="new timetable scene file")
    reopen_parser.add_argument("--plan", required=True, help="original successful plan (never rewritten)")
    reopen_parser.add_argument("--output", required=True, help="assessment report json path")
    reopen_parser.add_argument("--max-delay", type=float, default=30.0)
    reopen_parser.add_argument("--delay-step", type=float, default=0.2)
    reopen_parser.add_argument("--primitive-length", type=float, nargs="+", default=[0.4, 0.75])
    reopen_parser.add_argument("--cruise-speed", type=float, default=1.2)
    reopen_parser.add_argument("--gear-switch-time", type=float, default=1.5)

    window_parser = sub.add_parser(
        "window",
        help="evaluate a range of departure times on the original route",
    )
    window_parser.add_argument("--vehicle", required=True)
    window_parser.add_argument("--scene", required=True, help="current timetable scene file")
    window_parser.add_argument("--plan", required=True, help="original successful plan (never rewritten)")
    window_parser.add_argument("--earliest", type=float, required=True, help="earliest departure offset (s)")
    window_parser.add_argument("--latest", type=float, required=True, help="latest departure offset (s)")
    window_parser.add_argument("--step", type=float, default=0.5, help="departure sampling step (s)")
    window_parser.add_argument("--max-checks", type=int, default=200, help="replay budget for departure probes")
    window_parser.add_argument("--output", required=True, help="window report json path")
    window_parser.add_argument("--select", type=float, default=None, help="departure offset chosen from a safe interval")
    window_parser.add_argument("--previous", default=None, help="earlier window report to review against the current timetable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vehicle = load_vehicle(args.vehicle)
    scene = load_scene(args.scene)

    if args.command == "plan":
        primitives = standard_primitives(
            vehicle,
            lengths=tuple(args.primitive_length),
            steering_steps=2,
        )
        planner = Planner(
            vehicle,
            scene,
            primitives,
            PlannerConfig(
                max_expanded_nodes=args.max_nodes,
                xy_resolution=args.xy_resolution,
                theta_resolution=args.theta_resolution,
            ),
        )
        result = planner.plan()
        if args.timed or scene.dynamic_obstacles:
            scheduler = Scheduler(
                vehicle,
                scene,
                SchedulerConfig(
                    cruise_speed=args.cruise_speed,
                    gear_switch_time=args.gear_switch_time,
                ),
            )
            result = scheduler.schedule(result)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_plan(result, output)
        print(result.status, result.message)
        print(f"nodes_expanded={result.nodes_expanded}, segments={len(result.motions)}")
        if result.timed:
            waits = [action for action in result.actions if action.type == "wait"]
            switches = [action for action in result.actions if action.type == "gear_switch"]
            print(
                f"actions={len(result.actions)}, waits={len(waits)}, "
                f"gear_switches={len(switches)}, final_time={result.final_time:.2f}s"
            )
            for action in result.actions:
                if action.type != "drive":
                    print(
                        f"  {action.type} {action.start_time:6.2f}-{action.end_time:6.2f}s: {action.reason}"
                    )
        return 0

    if args.command == "reopen":
        original = load_plan(args.plan)
        primitives = standard_primitives(
            vehicle,
            lengths=tuple(args.primitive_length),
            steering_steps=2,
        )
        scheduler_config = SchedulerConfig(
            cruise_speed=args.cruise_speed,
            gear_switch_time=args.gear_switch_time,
        )
        assessment = evaluate_reopen(
            original,
            vehicle,
            scene,
            max_delay=args.max_delay,
            delay_step=args.delay_step,
            primitives=primitives,
            scheduler_config=scheduler_config,
        )
        assessment = replace(assessment, original_plan_path=args.plan)
        save_reopen_bundle(assessment, None, args.output)
        print(f"timetable_changed={assessment.timetable_changed}")
        print(
            f"original feasible={assessment.original_feasible}, "
            f"conflict_intervals={len(assessment.conflicts)}"
        )
        for conflict in assessment.conflicts:
            names = ",".join(conflict.obstacles) or "static"
            print(
                f"  conflict {conflict.start_time:6.2f}-{conflict.end_time:6.2f}s "
                f"({names}, {conflict.context}, samples={conflict.samples})"
            )
        for option in assessment.options:
            print(
                f"{option.key:9s} feasible={option.feasible} "
                f"route_changed={option.route_changed} timing_changed={option.timing_changed} "
                f"delay={option.departure_delay:.2f}s final={option.final_time:.2f}s"
            )
        print(f"recommendation={assessment.recommendation}")
        return 0

    if args.command == "window":
        original = load_plan(args.plan)
        assessment = evaluate_departure_window(
            original,
            vehicle,
            scene,
            earliest=args.earliest,
            latest=args.latest,
            step=args.step,
            max_checks=args.max_checks,
        )
        assessment = replace(assessment, original_plan_path=args.plan)
        selected = None
        if args.select is not None:
            selected_plan = select_departure(assessment, original, vehicle, scene, args.select)
            selected = (args.select, selected_plan)
        payload = save_window_report(
            assessment,
            args.output,
            selected=selected,
            previous_report_path=args.previous,
            scene=scene,
        )
        print(
            f"window [{assessment.earliest:.2f}, {assessment.latest:.2f}]s "
            f"step={assessment.step:.2f}s checks={assessment.checks_used}/{assessment.max_checks} "
            f"budget_exhausted={assessment.budget_exhausted}"
        )
        print(
            f"boundary earliest={assessment.earliest_status} latest={assessment.latest_status}"
        )
        for interval in assessment.intervals:
            names = ",".join(interval.obstacles) or "-"
            print(
                f"  {interval.status:12s} {interval.start:7.2f}-{interval.end:7.2f}s "
                f"obstacles={names} checked={interval.checked_points}"
            )
        if selected is not None:
            print(f"selected departure {args.select:.2f}s verified, plan saved")
        if payload.get("previous_review"):
            review = payload["previous_review"]
            print(f"previous report timetable_changed={review['timetable_changed']}")
            for interval in review["intervals"]:
                print(
                    f"  previous {interval['status']:12s} "
                    f"{interval['start']:7.2f}-{interval['end']:7.2f}s -> {interval['review']}"
                )
        return 0

    result = load_plan(args.plan)
    report = replay(result, vehicle, scene)
    print(f"valid={report.valid} samples={report.samples}")
    print(f"xy_error={report.goal_xy_error:.4f} m, theta_error={report.goal_theta_error:.4f} rad")
    print(report.message)
    if args.svg and result.found:
        render_svg(result, vehicle, scene, args.svg)
        print(f"svg={args.svg}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
