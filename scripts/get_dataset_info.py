#!/usr/bin/env python3
"""Inspect a ShakeBench demonstration HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def dataset_info(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demos = sorted(key for key in data if key.startswith("demo_"))
        if not demos:
            raise ValueError(f"{path} contains no demo groups")
        observation_keys = sorted(data[demos[0]]["obs"].keys())
        privileged = [key for key in observation_keys if key.startswith("privileged_")]
        return {
            "path": str(path),
            "episodes": len(demos),
            "total_samples": int(data.attrs["total"]),
            "env_args": json.loads(data.attrs["env_args"]),
            "observation_keys": observation_keys,
            "privileged_keys": privileged,
            "train_episodes": len(handle["mask/train"]),
            "valid_episodes": len(handle["mask/valid"]),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    info = dataset_info(args.dataset)
    if args.as_json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print(f"dataset: {info['path']}")
        print(f"episodes: {info['episodes']}  samples: {info['total_samples']}")
        print(
            f"split: train={info['train_episodes']} valid={info['valid_episodes']}"
        )
        keys = info["privileged_keys"]
        print(f"该数据集包含 {len(keys)} 个特权键")
        for key in keys:
            print(f"  - {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
