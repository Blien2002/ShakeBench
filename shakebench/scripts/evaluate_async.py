"""Evaluate continuous action chunks without changing the synchronous runner.

Example (external model source must be on PYTHONPATH)::

    python -m shakebench.scripts.evaluate_async \
        --policy shakebench.utils.async_policy:make_chunk_policy \
        --policy-arg checkpoint=/path/to/finetuned_checkpoint \
        --dataset /path/to/success_dataset --gamma 0.5 --output out/async.json

--time-scale sets wall seconds per simulation second and scales measured model
latency by the same factor. Overruns are reported, never disguised as real time.
"""

import json
import sys

from shakebench.scripts.evaluate import (
    build_parser,
    observation_config_from_collection,
    parse_policy_args,
    requested_state_ids,
    summarize,
)
from shakebench.scripts.run_oracle import load_state_asset
from shakebench.utils.artifacts import write_json
from shakebench.utils.async_policy import AsyncPolicy, PacedTask
from shakebench.utils.rollout import ShakeBenchTaskEnv, invalid_episode_result, rollout_policy
from shakebench.utils.train_states import assert_split_disjoint, split_overlap


def main(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.sway_v1:
        from shakebench.utils.excitation import SWAY_V1

        if args.mode_params or args.mode != "multisine_v1":
            raise ValueError("--sway-v1 cannot be combined with --mode or --mode-params")
        args.mode, args.mode_params = SWAY_V1["mode"], SWAY_V1["mode_params"]
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.horizon_steps < 1 or args.action_horizon != 1:
        raise ValueError("async evaluation needs a positive horizon and --action-horizon 1")
    states = requested_state_ids(load_state_asset(args.states)["states"], args.state_ids)
    if not states:
        raise ValueError("no states selected")
    cameras = {
        "main_camera": args.main_camera or "task_close",
        "height": args.height or 256,
        "width": args.width or 256,
    }
    split = {"status": "unavailable", "reason": "no --dataset was given"}
    if args.dataset is not None:
        trained_on = observation_config_from_collection(args.dataset)
        train_states = trained_on.pop("train_states")
        cameras = {
            "main_camera": args.main_camera or trained_on["main_camera"],
            "height": args.height or trained_on["height"],
            "width": args.width or trained_on["width"],
        }
        if not args.allow_train_states:
            assert_split_disjoint(train_states, states)
        split = {
            **trained_on,
            "status": "train_states_allowed" if args.allow_train_states else "checked",
            "overlap": len(split_overlap(train_states, states)),
            "train_state_count": len(train_states),
        }
    timeout = 10.0 if args.inference_timeout_s is None else args.inference_timeout_s
    arguments = parse_policy_args(args.policy_arg)
    episodes = []
    policy_id = args.policy_id or args.policy

    def payload():
        return {
            "schema_id": "shakebench.policy_evaluation_async",
            "schema_version": 1,
            "scoreable": False,
            "scoreable_reason": "asynchronous policy evaluation evidence",
            "policy": {"spec": args.policy, "policy_id": policy_id, "arguments": arguments},
            "run_contract": {
                "timing": "asynchronous_continuous_chunks",
                "gamma": args.gamma,
                "mode": args.mode,
                "mode_params": args.mode_params,
                "horizon_steps": args.horizon_steps,
                "action_horizon": 1,
                "control_frequency_hz": 20,
                "time_scale": args.time_scale,
                "inference_timeout_s": timeout,
                "timeout_semantics": "maximum wall interval without a completed chunk, scaled by time_scale",
                "warmup": "initial prediction before episode clock",
                "missing_action": "zero_osc_hold",
                "chunk_merge": "latest_prediction_overwrites_overlap; expired_prefix_discarded",
                "observation_source": args.observation_source,
                "observation": cameras,
                "state_asset": str(args.states),
            },
            "train_split_check": split,
            "episodes": episodes,
            "summary": summarize(episodes),
        }

    def save():
        if args.output is not None:
            write_json(args.output, payload())

    policy = None
    try:
        for state in states:
            task = None
            try:
                if policy is None:
                    policy = AsyncPolicy(
                        args.policy,
                        arguments,
                        time_scale=args.time_scale,
                        timeout_s=timeout,
                        startup_timeout_s=args.startup_timeout_s,
                    )
                task = PacedTask(
                    ShakeBenchTaskEnv(
                        state,
                        gamma=args.gamma,
                        mode=args.mode,
                        mode_params=args.mode_params,
                        horizon=args.horizon_steps,
                        observation_source=args.observation_source,
                        **cameras,
                    ),
                    policy,
                    time_scale=args.time_scale,
                )
                result = rollout_policy(task, policy, action_horizon=1, policy_id=policy_id)
                result.update(
                    {
                        "timing": "asynchronous_continuous_chunks",
                        "inference_timeout_s": timeout,
                        "async": {
                            **policy.stats,
                            "time_scale": args.time_scale,
                            "simulation_overrun_steps": task.overruns,
                            "max_simulation_overrun_s": task.max_overrun_s,
                            "episode_wall_time_s": task.wall_time_s,
                            "timing_valid": task.overruns == 0 and result["steps"] > 0,
                        },
                    }
                )
                # The shared loop's inference counter measures polling, not background inference.
                result["poll_wall_time_s"] = result["inference_wall_time_s"]
                result["inference_wall_time_s"] = policy.stats["inference_wall_time_s"]
                episodes.append(result)
                if task.overruns:
                    print(
                        f"{state['state_id']}: {task.overruns} simulation overruns; increase --time-scale",
                        file=sys.stderr,
                    )
                if result["termination_cause"] in {"policy_error", "invalid_execution"}:
                    policy.close()
                    policy = None
            except Exception as exc:
                result = invalid_episode_result(
                    state_id=state.get("state_id"),
                    reason=f"{type(exc).__name__}: {exc}",
                    task=task,
                    policy_id=policy_id,
                    gamma=args.gamma,
                    horizon_steps=args.horizon_steps,
                    action_horizon=1,
                    inference_timeout_s=timeout,
                )
                result["timing"] = "asynchronous_continuous_chunks"
                episodes.append(result)
                if policy is not None:
                    policy.close()
                    policy = None
            finally:
                if task is not None:
                    task.close()
            save()
    finally:
        if policy is not None:
            policy.close()
    if args.output is None:
        print(json.dumps(payload(), indent=2))
    else:
        print(json.dumps(payload()["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
