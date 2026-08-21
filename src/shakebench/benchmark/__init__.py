"""Task-suite registry."""

from .suites import (
    BenchmarkSuite,
    ShakeBenchBandwidth,
    ShakeBenchLadder,
    ShakeBenchPredictability,
    ShakeBenchSweep,
)
from .scorecard import (
    aggregate_episodes,
    bootstrap_interval,
    critical_value,
    decision_point,
    paired_bootstrap_difference,
)


def get_benchmark_dict() -> dict[str, type[BenchmarkSuite]]:
    return {
        "shakebench_ladder": ShakeBenchLadder,
        "shakebench_sweep": ShakeBenchSweep,
        "shakebench_bandwidth": ShakeBenchBandwidth,
        "shakebench_predictability": ShakeBenchPredictability,
    }


__all__ = [
    "BenchmarkSuite",
    "aggregate_episodes",
    "bootstrap_interval",
    "critical_value",
    "decision_point",
    "get_benchmark_dict",
    "paired_bootstrap_difference",
]
