from __future__ import annotations

import argparse
import json
from pathlib import Path

from .motion import standard_primitives
from .plan import load_plan, save_plan
from .planner import Planner, PlannerConfig
from .replay import render_svg, replay
from .replan import compare_alternatives
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
    replay_parser.add_argument("--report")

    replan_parser = sub.add_parser(
        "replan",
        help="re-evaluate a saved plan against a changed timetable",
    )
    replan_parser.add_argument("--vehicle", required=True)
    replan_parser.add_argument("--scene", required=True)
    replan_parser.add_argument("--plan", required=True, help="previously successful plan (kept untouched)")
    replan_parser.add_argument("--primitive-length", type=float, nargs="+", default=[0.4, 0.75])
    replan_parser.add_argument("--cruise-speed", type=float, default=1.2)
    replan_parser.add_argument("--gear-switch-time", type=float, default=1.5)
    replan_parser.add_argument("--max-wait", type=float, default=60.0)
    replan_parser.add_argument("--max-nodes", type=int, default=60_000)
    replan_parser.add_argument("--report", help="write the three-alternative comparison as JSON")
    replan_parser.add_argument(
        "--output",
        help="save the recommended delayed/replanned variant (the original file is never overwritten)",
    )
    replan_parser.add_argument("--no-route-replan", action="store_true")
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

    result = load_plan(args.plan)
    if args.command == "replan":
        return _run_replan(args, vehicle, scene, result)

    report = replay(result, vehicle, scene)
    print(f"valid={report.valid} samples={report.samples}")
    print(f"xy_error={report.goal_xy_error:.4f} m, theta_error={report.goal_theta_error:.4f} rad")
    print(report.message)
    if args.svg and result.found:
        render_svg(result, vehicle, scene, args.svg)
        print(f"svg={args.svg}")
    if args.report:
        Path(args.report).write_text(json.dumps(vars(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report={args.report}")
    return 0 if report.valid else 1


def _run_replan(args, vehicle, scene, old_plan):
    primitives = standard_primitives(
        vehicle,
        lengths=tuple(args.primitive_length),
        steering_steps=2,
    )
    comparison = compare_alternatives(
        old_plan,
        vehicle,
        scene,
        primitives,
        scheduler_config=SchedulerConfig(
            cruise_speed=args.cruise_speed,
            gear_switch_time=args.gear_switch_time,
            max_wait=args.max_wait,
        ),
        planner_config=PlannerConfig(max_expanded_nodes=args.max_nodes),
        allow_replan=not args.no_route_replan,
    )

    print(f"timetable_changed={comparison.timetable_changed}")
    print(comparison.message)
    for alternative in (
        comparison.original,
        comparison.delayed,
        comparison.replanned,
    ):
        if alternative is None:
            continue
        print(
            f"[{alternative.key}] feasible={alternative.feasible} "
            f"route_changed={alternative.route_changed} "
            f"timing_changed={alternative.timing_changed} "
            f"final_time={alternative.final_time:.2f}"
        )
        print(f"    {alternative.message}")
        for conflict in alternative.conflicts:
            print(
                "    conflict "
                f"t={conflict.start_time:.2f}-{conflict.end_time:.2f}s "
                f"obstacle={conflict.obstacle} during {conflict.action_type}"
            )

    if args.report:
        Path(args.report).write_text(
            json.dumps(comparison.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report={args.report}")

    if args.output:
        chosen = {
            "delayed": comparison.delayed,
            "replanned": comparison.replanned,
        }.get(comparison.recommended)
        if chosen is None or not chosen.feasible or chosen.plan is None:
            print("no feasible delayed/replanned variant to save; original plan left untouched")
            return 1
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_plan(chosen.plan, output)
        print(f"saved {comparison.recommended} variant -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
