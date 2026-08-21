"""Task-suite registry."""

from .suites import (
    BenchmarkSuite,
    ShakeBenchBandwidth,
    ShakeBenchLadder,
    ShakeBenchPredictability,
    ShakeBenchSweep,
)


def get_benchmark_dict() -> dict[str, type[BenchmarkSuite]]:
    return {
        "shakebench_ladder": ShakeBenchLadder,
        "shakebench_sweep": ShakeBenchSweep,
        "shakebench_bandwidth": ShakeBenchBandwidth,
        "shakebench_predictability": ShakeBenchPredictability,
    }


__all__ = ["BenchmarkSuite", "get_benchmark_dict"]
