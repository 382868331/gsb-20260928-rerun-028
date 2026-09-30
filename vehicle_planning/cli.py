from __future__ import annotations

import argparse
from pathlib import Path

from .motion import standard_primitives
from .plan import load_plan, save_plan
from .planner import Planner, PlannerConfig
from .replay import render_svg, replay
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
    plan_parser.add_argument("--speed", type=float, default=1.0)
    plan_parser.add_argument("--gear-shift-time", type=float, default=2.0)
    plan_parser.add_argument("--max-wait-time", type=float, default=25.0)
    plan_parser.add_argument("--max-makespan", type=float, default=90.0)

    replay_parser = sub.add_parser("replay", help="replay a saved plan and render SVG")
    replay_parser.add_argument("--vehicle", required=True)
    replay_parser.add_argument("--scene", required=True)
    replay_parser.add_argument("--plan", required=True)
    replay_parser.add_argument("--svg")
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
                speed=args.speed,
                gear_shift_time=args.gear_shift_time,
                max_wait_time=args.max_wait_time,
                max_makespan=args.max_makespan,
            ),
        )
        result = planner.plan()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_plan(result, output)
        print(result.status, result.message)
        print(f"nodes_expanded={result.nodes_expanded}, segments={len(result.motions)}")
        if result.found:
            print(f"makespan={result.makespan:.2f}s, waits={len(result.waits)}")
            for action in result.waits:
                print(f"  wait {action.t0:.2f}-{action.t1:.2f}s: {action.reason}")
        return 0

    result = load_plan(args.plan)
    report = replay(result, vehicle, scene)
    print(f"valid={report.valid} samples={report.samples}")
    print(f"xy_error={report.goal_xy_error:.4f} m, theta_error={report.goal_theta_error:.4f} rad")
    if report.timetable_changed:
        print("timetable_changed=True: plan re-verified against the current obstacle timetable")
    print(report.message)
    if args.svg and result.found:
        render_svg(result, vehicle, scene, args.svg)
        print(f"svg={args.svg}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
