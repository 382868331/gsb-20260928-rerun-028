from __future__ import annotations

import argparse

from .models import (
    load_json,
    plan_from_dict,
    plan_to_dict,
    save_json,
    scenario_from_dict,
    vehicle_from_dict,
)
from .planner import MotionPlanner
from .planner import SearchLimits
from .replay import replay_plan, trajectory_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-speed Ackermann vehicle motion planner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="search a feasible path")
    plan.add_argument("--vehicle", required=True)
    plan.add_argument("--scenario", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--heading-bins", type=int, default=24)
    plan.add_argument("--position-resolution", type=float, default=0.25)
    plan.add_argument("--max-expansions", type=int, default=60000)

    replay = subparsers.add_parser("replay", help="re-integrate and validate a saved plan")
    replay.add_argument("--vehicle", required=True)
    replay.add_argument("--scenario", required=True)
    replay.add_argument("--plan", required=True)
    replay.add_argument("--trajectory", help="write replayed trajectory JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vehicle = vehicle_from_dict(load_json(args.vehicle))
    scenario = scenario_from_dict(load_json(args.scenario))

    if args.command == "plan":
        planner = MotionPlanner(
            vehicle,
            heading_bins=args.heading_bins,
            position_resolution=args.position_resolution,
            limits=SearchLimits(max_expansions=args.max_expansions),
        )
        plan = planner.make_plan(scenario)
        save_json(args.output, plan_to_dict(plan))
        print(plan.status, plan.metadata)
        return 0 if plan.status == "solved" else 2

    plan = plan_from_dict(load_json(args.plan))
    report = replay_plan(plan, vehicle, scenario)
    if args.trajectory:
        save_json(args.trajectory, trajectory_json(plan, vehicle))
    print(report)
    return 0 if report.valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
