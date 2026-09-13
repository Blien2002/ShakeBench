"""Run a ShakeBench episode against the official StarVLA WebSocket policy server."""

import argparse
import json

from robosuite.scripts.shakebench_run_oracle import load_dev_states
from robosuite.utils.shakebench_starvla import StarVLAEnvironment, StarVLAPolicy, rollout_starvla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--states", default="robosuite/models/assets/shakebench_states_dev.json")
    parser.add_argument("--state-id", default="shakebench-dev-v0-000")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--action-horizon", type=int, default=1)
    args = parser.parse_args()
    states = {state["state_id"]: state for state in load_dev_states(args.states)}
    if args.state_id not in states:
        parser.error(f"unknown state ID: {args.state_id}")
    task = StarVLAEnvironment(states[args.state_id], gamma=args.gamma, horizon=args.horizon_steps)
    policy = None
    try:
        policy = StarVLAPolicy(host=args.host, port=args.port)
        print(json.dumps(rollout_starvla(task, policy, action_horizon=args.action_horizon), indent=2))
    finally:
        try:
            task.close()
        finally:
            if policy is not None:
                policy.close()


if __name__ == "__main__":
    main()
