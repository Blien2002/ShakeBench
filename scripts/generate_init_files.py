#!/usr/bin/env python3
"""Generate committed LIBERO-style initial-state files for all suites."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from shakebench.benchmark import get_benchmark_dict
from shakebench.config import SpectralBand, VibrationConfig
from shakebench.vibration import calibrate_level_scale, validate_deck_displacement_gate

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "src" / "shakebench" / "init_files"


def _vibration_config(config: dict, seed: int, t0: float) -> VibrationConfig:
    vibration = VibrationConfig(
        seed=seed,
        t0=t0,
        gamma=float(config.get("gamma", 0.50)),
        frequency_scale=float(config.get("frequency_scale", 1.0)),
    )
    if "bandwidth_ratio" in config:
        ratio = float(config["bandwidth_ratio"])
        bands = {
            axis: tuple(
                SpectralBand(band.center_hz, band.accel_rms, ratio, band.tones)
                for band in axis_bands
            )
            for axis, axis_bands in vibration.bands.items()
        }
        vibration = replace(vibration, bands=bands)
    return vibration


def _placement(row: int) -> np.ndarray:
    angle = 2.0 * np.pi * row / 50.0
    object_pose = np.array(
        [0.08 + 0.015 * np.cos(angle), -0.13 + 0.015 * np.sin(angle), 0.47, 1, 0, 0, 0],
        dtype=np.float64,
    )
    target_pose = np.array([0.08, 0.17, 0.376, 1, 0, 0, 0], dtype=np.float64)
    return np.stack((object_pose, target_pose))


def generate(count: int = 50, suites: tuple[str, ...] = ()) -> None:
    if count != 50:
        raise ValueError("official init files contain exactly 50 states per task")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0_values = np.linspace(0.05, 99.95, count, dtype=np.float64)
    cache: dict[tuple, tuple[float, float]] = {}
    for suite_name, suite_type in get_benchmark_dict().items():
        if suites and suite_name not in suites:
            continue
        suite = suite_type()
        output = OUTPUT_DIR / f"{suite_name}.hdf5"
        with h5py.File(output, "w") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["suite"] = suite_name
            for task_index, task in enumerate(suite.tasks):
                group = handle.create_group(f"task_{task_index:03d}")
                placements = np.stack([_placement(row) for row in range(count)])
                seeds = np.arange(1000, 1000 + count, dtype=np.int64)
                scales = np.empty(count, dtype=np.float64)
                gammas = np.empty(count, dtype=np.float64)
                profile = tuple(
                    sorted(
                        (key, value)
                        for key, value in task.config.items()
                        if key not in ("workpiece", "env_name", "control_freq")
                    )
                )
                for row, (seed, t0) in enumerate(zip(seeds, t0_values)):
                    key = (profile, int(seed), float(t0))
                    if key not in cache:
                        vibration = _vibration_config(task.config, int(seed), float(t0))
                        scale, report = calibrate_level_scale(vibration, 1000, 16.0)
                        validate_deck_displacement_gate(
                            vibration,
                            vibration.gamma,
                            float(report["peak_deck_displacement_m"]),
                        )
                        cache[key] = (float(scale), float(report["gamma_realized"]))
                    scales[row], gammas[row] = cache[key]
                group.create_dataset("object_placement", data=placements)
                string_type = h5py.string_dtype("utf-8")
                group.create_dataset(
                    "workpiece", data=np.asarray([task.config["workpiece"]] * count, dtype=object),
                    dtype=string_type,
                )
                group.create_dataset("t0", data=t0_values)
                group.create_dataset("level_scale", data=scales)
                group.create_dataset("gamma_realized", data=gammas)
                group.create_dataset("seed", data=seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--suite", action="append", choices=tuple(get_benchmark_dict()), default=[]
    )
    args = parser.parse_args()
    generate(args.count, tuple(args.suite))


if __name__ == "__main__":
    main()
