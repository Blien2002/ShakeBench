"""Run one ShakeBench episode against a WebSocket policy server.

Thin wrapper: the shared runner in shakebench.scripts.evaluate owns the
rollout, result contract, and deadlines; this file only supplies the policy
factory and hosts the documented --host/--port arguments.
"""

from __future__ import annotations

import argparse

from shakebench import models
from shakebench.scripts.evaluate import main as evaluate_main


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--states", default=str(Path(models.assets_root, "shakebench_states_dev.json")))
    parser.add_argument("--state-id", default="shakebench-dev-v0-000")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--inference-timeout-s", type=float, default=None, help="Hard per-request deadline")
    parser.add_argument("--dataset", default=None, help="Collected dataset; enables train/eval split checking")
    parser.add_argument("--output", default=None, help="Result JSON path; default: print the record")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    forwarded = [
        "--policy",
        "shakebench.utils.websocket_policy:make_policy",
        "--policy-id",
        f"websocket:{args.host}:{args.port}",
        "--policy-arg",
        f"host={args.host}",
        "--policy-arg",
        f"port={args.port}",
        "--states",
        args.states,
        "--state-ids",
        args.state_id,
        "--gamma",
        str(args.gamma),
        "--horizon-steps",
        str(args.horizon_steps),
        "--action-horizon",
        str(args.action_horizon),
    ]
    if args.inference_timeout_s is not None:
        forwarded += ["--inference-timeout-s", str(args.inference_timeout_s)]
    if args.dataset is not None:
        forwarded += ["--dataset", args.dataset]
    if args.output is not None:
        forwarded += ["--output", args.output]
    return evaluate_main(forwarded)


if __name__ == "__main__":
    main()
