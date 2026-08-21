"""LIBERO-style ShakeBench task suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .task_map import BenchmarkTask

WORKPIECES = ("cracker_box", "sugar_box", "soup_can", "mustard_bottle")


def _language(workpiece: str) -> str:
    return f"pick up the {workpiece.replace('_', ' ')} and place it in the bin"


@dataclass
class BenchmarkSuite:
    name: str
    tasks: tuple[BenchmarkTask, ...]

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def get_task(self, index: int) -> BenchmarkTask:
        return self.tasks[index]

    @property
    def init_file(self) -> Path:
        return Path(__file__).resolve().parents[1] / "init_files" / f"{self.name}.hdf5"

    def get_task_init_states(self, index: int) -> list[dict]:
        if not self.init_file.is_file():
            raise FileNotFoundError(
                f"missing {self.init_file}; run scripts/generate_init_files.py"
            )
        key = f"task_{index:03d}"
        states: list[dict] = []
        with h5py.File(self.init_file, "r") as handle:
            group = handle[key]
            count = len(group["t0"])
            for row in range(count):
                workpiece = group["workpiece"][row]
                if isinstance(workpiece, bytes):
                    workpiece = workpiece.decode("utf-8")
                states.append(
                    {
                        "object_placement": group["object_placement"][row].astype(np.float64),
                        "workpiece": str(workpiece),
                        "t0": float(group["t0"][row]),
                        "level_scale": float(group["level_scale"][row]),
                        "gamma_realized": float(group["gamma_realized"][row]),
                        "seed": int(group["seed"][row]),
                    }
                )
        return states


def _cross_tasks(name: str, axis: str, values: Iterable[float | int]) -> tuple[BenchmarkTask, ...]:
    tasks = []
    for value in values:
        for workpiece in WORKPIECES:
            label = str(value).replace(".", "p")
            tasks.append(
                BenchmarkTask(
                    name=f"{name}_{axis}_{label}_{workpiece}",
                    language=_language(workpiece),
                    config={"env_name": "PickPlace", "workpiece": workpiece, axis: value},
                )
            )
    return tuple(tasks)


class ShakeBenchLadder(BenchmarkSuite):
    def __init__(self) -> None:
        super().__init__(
            "shakebench_ladder", _cross_tasks("ladder", "gamma", (0.15, 0.30, 0.50, 0.75, 0.95))
        )


class ShakeBenchSweep(BenchmarkSuite):
    def __init__(self) -> None:
        tasks = tuple(
            BenchmarkTask(
                name=task.name,
                language=task.language,
                config={**task.config, "gamma": 0.15},
            )
            for task in _cross_tasks(
                "sweep", "frequency_scale", (0.25, 0.5, 1.0, 2.0, 4.0)
            )
        )
        super().__init__(
            "shakebench_sweep", tasks
        )


class ShakeBenchBandwidth(BenchmarkSuite):
    def __init__(self) -> None:
        super().__init__(
            "shakebench_bandwidth", _cross_tasks("bandwidth", "control_freq", (2, 5, 10, 20, 50, 200))
        )


class ShakeBenchPredictability(BenchmarkSuite):
    def __init__(self) -> None:
        super().__init__(
            "shakebench_predictability",
            _cross_tasks("predictability", "bandwidth_ratio", (0.0, 0.10, 0.40)),
        )
