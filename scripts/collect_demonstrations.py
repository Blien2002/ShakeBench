#!/usr/bin/env python3
"""Collect staged oracle_phase demonstrations in robomimic HDF5 format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import shakebench
from shakebench.envs.wrappers import DataCollectionWrapper
from shakebench.policies import OraclePhasePolicy, policy_env_kwargs


WORKPIECES = ("cracker_box", "sugar_box", "soup_can", "mustard_bottle")
STAGE_EPISODES = {"A": 25, "B": 200, "C": 500}


def validate_fidelity_gate(path: Path) -> None:
    """Require machine-readable evidence that training physics is eligible."""

    if not path.is_file():
        raise RuntimeError(
            f"missing fidelity gate {path}; run the official/training correlation "
            "experiment before collecting training data"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rho = float(payload.get("spearman_rho", 0.0))
    if rho <= 0.9 or not payload.get("training_eligible", False):
        raise RuntimeError(
            f"training physics failed the fidelity gate: rho={rho:.4f}, "
            f"training_eligible={payload.get('training_eligible', False)}"
        )


def collect_one(
    output: Path,
    *,
    mode: str,
    workpiece: str,
    episodes: int,
    seed: int,
    gamma: float | None = None,
    allow_contract_backend: bool = False,
) -> Path:
    policy = OraclePhasePolicy(control_freq=10)
    gamma = 0.50 if gamma is None else float(gamma)
    env = shakebench.make(
        "PickPlace",
        controller_configs=shakebench.load_controller_config("VARIABLE_IMPEDANCE"),
        intra_step_mode="feedforward",
        control_freq=10,
        physics_profile="training",
        vibration_mode="off" if mode == "static" else "spectral",
        gamma=gamma,
        workpiece=workpiece,
        use_camera_obs=True,
        **policy_env_kwargs(policy),
    )
    if not getattr(env.unwrapped, "scoreable", False) and not allow_contract_backend:
        env.close()
        raise RuntimeError(
            "the default state-contract backend cannot produce paper datasets; "
            "attach the Isaac/Newton Gym backend, or pass --allow-contract-backend "
            "only for a schema smoke test"
        )
    wrapped = DataCollectionWrapper(env, output_path=output)
    try:
        for episode in range(episodes):
            policy.reset()
            observation, _ = wrapped.reset(seed=seed + episode)
            while True:
                action = policy.act(observation)
                observation, _, terminated, truncated, _ = wrapped.step(action)
                if terminated or truncated:
                    break
        return wrapped.flush()
    finally:
        # flush() above owns the final write; avoid a second write from the
        # wrapper while still releasing the underlying environment.
        env.close()


def merge_datasets(paths: list[Path], output: Path, *, mode: str) -> Path:
    """Merge deterministic per-workpiece/Γ shards into one training set."""

    if not paths:
        raise ValueError("at least one shard is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    demos: list[str] = []
    total = 0
    with h5py.File(paths[0], "r") as first:
        source_env_args = json.loads(first["data"].attrs["env_args"])
    with h5py.File(temporary, "w") as destination:
        data = destination.create_group("data")
        controller_config = shakebench.load_controller_config("VARIABLE_IMPEDANCE")
        controller_config["intra_step_mode"] = "feedforward"
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "PickPlace",
                "backend": source_env_args.get("backend", "unknown"),
                "scoreable": bool(source_env_args.get("scoreable", False)),
                "env_kwargs": {
                    "gamma": None if mode == "static" else [0.15, 0.30, 0.50],
                    "frequency_scale": 1.0,
                    "control_freq": 10,
                    "level_scale": None if mode == "static" else "per_episode",
                    "physics_profile": "training",
                    "controller_config": controller_config,
                },
            },
            sort_keys=True,
        )
        for path in paths:
            with h5py.File(path, "r") as source:
                source_data = source["data"]
                for old_name in sorted(key for key in source_data if key.startswith("demo_")):
                    new_name = f"demo_{len(demos)}"
                    source.copy(source_data[old_name], data, name=new_name)
                    demos.append(new_name)
                    total += int(source_data[old_name].attrs["num_samples"])
        data.attrs["total"] = total
        mask = destination.create_group("mask")
        valid_count = min(int(round(0.1 * len(demos))), max(0, len(demos) - 1))
        split = len(demos) - valid_count
        dtype = h5py.string_dtype("utf-8")
        mask.create_dataset("train", data=np.asarray(demos[:split], dtype=dtype))
        mask.create_dataset("valid", data=np.asarray(demos[split:], dtype=dtype))
    temporary.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_EPISODES), default="A")
    parser.add_argument("--dataset", choices=("static", "shaken"), required=True)
    parser.add_argument("--workpiece", choices=("all", *WORKPIECES), default="all")
    parser.add_argument("--episodes", type=int, help="override per-workpiece stage count")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fidelity-gate",
        type=Path,
        default=Path("docs/reports/fidelity_throughput_correlation.json"),
    )
    parser.add_argument(
        "--allow-contract-backend",
        action="store_true",
        help="schema smoke test only; output is not eligible for training or paper results",
    )
    args = parser.parse_args()
    if not args.allow_contract_backend:
        validate_fidelity_gate(args.fidelity_gate)
    episodes = args.episodes or STAGE_EPISODES[args.stage]
    output = args.output or Path("datasets") / f"demo_{args.dataset}_stage_{args.stage.lower()}.hdf5"
    workpieces = WORKPIECES if args.workpiece == "all" else (args.workpiece,)
    with tempfile.TemporaryDirectory(prefix="shakebench_dataset_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        shards: list[Path] = []
        for workpiece_index, workpiece in enumerate(workpieces):
            if args.dataset == "static":
                distributions = ((None, episodes),)
            else:
                base, remainder = divmod(episodes, 3)
                distributions = tuple(
                    (gamma, base + int(index < remainder))
                    for index, gamma in enumerate((0.15, 0.30, 0.50))
                )
            offset = 0
            for gamma, count in distributions:
                if count == 0:
                    continue
                label = "off" if gamma is None else str(gamma).replace(".", "p")
                shard = temporary_root / f"{workpiece}_{label}.hdf5"
                collect_one(
                    shard,
                    mode=args.dataset,
                    workpiece=workpiece,
                    episodes=count,
                    seed=args.seed + 10_000 * workpiece_index + offset,
                    gamma=gamma,
                    allow_contract_backend=args.allow_contract_backend,
                )
                shards.append(shard)
                offset += count
        result = merge_datasets(shards, output, mode=args.dataset)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
