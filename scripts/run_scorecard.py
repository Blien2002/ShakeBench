#!/usr/bin/env python3
"""Run a reproducible ShakeBench scorecard from committed init states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import shakebench
from shakebench.benchmark.scorecard import aggregate_episodes
from shakebench.config import workpiece_dimensions_m
from shakebench.policies import make_policy, policy_env_kwargs, privilege_label
from shakebench.utils.paths import PROJECT_ROOT


DEFAULT_RATE = {
    "oracle_full": 1000,
    "oracle_phase": 10,
    "oracle_reactive": 10,
    "classical": 200,
    "random": 10,
}


def _policy(name: str, control_freq: int):
    if name == "random":
        return make_policy(name, action_dim=13, seed=0)
    return make_policy(name, control_freq=control_freq)


def _selected_tasks(suite, args) -> list[Any]:
    result = []
    defaults = {
        "gamma": 0.50,
        "frequency_scale": 1.0,
        "bandwidth_ratio": 0.10,
        "control_freq": 10.0,
    }
    for task in suite.tasks:
        config = task.config
        selectors = (
            ("gamma", args.gamma),
            ("frequency_scale", args.frequency_scale),
            ("bandwidth_ratio", args.bandwidth_ratio),
            ("control_freq", args.task_control_freq),
        )
        if all(
            value is None or float(config.get(key, defaults[key])) == value
            for key, value in selectors
        ):
            result.append(task)
    if not result:
        raise ValueError("no suite tasks match the requested selectors")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(shakebench.benchmark.get_benchmark_dict()), required=True)
    parser.add_argument("--policy", choices=tuple(DEFAULT_RATE), required=True)
    parser.add_argument("--control-freq", type=int)
    parser.add_argument("--task-control-freq", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--frequency-scale", type=float)
    parser.add_argument("--bandwidth-ratio", type=float)
    parser.add_argument("--init-count", type=int, default=10)
    parser.add_argument("--physics-profile", choices=("official", "training"), default="official")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-contract-backend",
        action="store_true",
        help="pipeline smoke test only; all state-contract episodes remain voided",
    )
    args = parser.parse_args()
    if not 1 <= args.init_count <= 50:
        raise ValueError("init-count must be in [1, 50]")

    suite = shakebench.benchmark.get_benchmark_dict()[args.suite]()
    tasks = _selected_tasks(suite, args)
    episodes: list[dict[str, Any]] = []
    declared_policy = _policy(args.policy, args.control_freq or DEFAULT_RATE[args.policy])
    for task_index, task in enumerate(suite.tasks):
        if task not in tasks:
            continue
        task_rate = int(task.config.get("control_freq", args.control_freq or DEFAULT_RATE[args.policy]))
        if args.control_freq is not None:
            task_rate = args.control_freq
        if args.policy == "oracle_full":
            task_rate = 1000
        policy = _policy(args.policy, task_rate)
        controller = shakebench.load_controller_config("VARIABLE_IMPEDANCE")
        controller["intra_step_mode"] = getattr(policy, "intra_step_mode", "zoh")
        env_kwargs = {
            **task.config,
            **policy_env_kwargs(policy),
            "controller_configs": controller,
            "control_freq": task_rate,
            "physics_profile": args.physics_profile,
            "use_camera_obs": args.policy == "oracle_phase",
        }
        env = shakebench.make(**env_kwargs)
        try:
            states = suite.get_task_init_states(task_index)[: args.init_count]
            for init_index, state in enumerate(states):
                env.set_init_state(state)
                policy.reset()
                observation, info = env.reset()
                if not info.get("scoreable", False) and not args.allow_contract_backend:
                    raise RuntimeError(
                        "the default state-contract backend is not scoreable; attach the "
                        "Isaac/Newton Gym backend, or use --allow-contract-backend for a "
                        "voided pipeline smoke test"
                    )
                while True:
                    action = policy.act(observation)
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                frequency_scale = float(task.config.get("frequency_scale", 1.0))
                episodes.append(
                    {
                        **info,
                        "task_name": task.name,
                        "init_state_index": init_index,
                        "gamma_expected": float(state["gamma_realized"]),
                        "gamma_target": float(task.config.get("gamma", 0.50)),
                        "frequency_scale": frequency_scale,
                        "center_frequency_hz": 8.0 * frequency_scale,
                        "bandwidth_ratio": task.config.get("bandwidth_ratio"),
                    }
                )
        finally:
            env.close()

    dimensions = [
        min(workpiece_dimensions_m(task.config["workpiece"], 0.75)) for task in tasks
    ]
    penetration_limit_mm = 0.01 * 1000.0 * min(dimensions)
    aggregate = aggregate_episodes(
        episodes, penetration_limit_mm=penetration_limit_mm, bootstrap_seed=0
    )
    varying = lambda key: sorted({episode.get(key) for episode in episodes if episode.get(key) is not None})
    payload = {
        "policy_name": args.policy,
        "requires_privileged": list(declared_policy.requires_privileged),
        "privilege_label": privilege_label(declared_policy),
        "control_freq": args.control_freq or DEFAULT_RATE[args.policy],
        "gamma_target": varying("gamma_target"),
        "frequency_scale": varying("frequency_scale"),
        "bandwidth_ratio": varying("bandwidth_ratio"),
        "physics_profile": args.physics_profile,
        "penetration_limit_mm": penetration_limit_mm,
        **aggregate,
        "episodes": episodes,
    }
    output = args.output or PROJECT_ROOT / "out" / f"scorecard_{args.policy}_{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
