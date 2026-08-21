import numpy as np
import pytest

from shakebench import benchmark
from shakebench.config import SpectralBand, VibrationConfig
from shakebench.vibration import calibrate_level_scale


@pytest.mark.parametrize("suite_name", benchmark.get_benchmark_dict())
def test_init_file_schema_phase_coverage_and_replay(suite_name: str) -> None:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    states = suite.get_task_init_states(0)
    task = suite.get_task(0)
    assert len(states) == 50
    t0 = np.asarray([state["t0"] for state in states])
    assert 0.0 <= t0.min() < 1.0
    assert 99.0 < t0.max() < 100.0
    for state in (states[0], states[24], states[-1]):
        cfg = VibrationConfig(
            seed=state["seed"], t0=state["t0"],
            gamma=float(task.config.get("gamma", 0.50)),
            frequency_scale=float(task.config.get("frequency_scale", 1.0)),
        )
        if "bandwidth_ratio" in task.config:
            ratio = float(task.config["bandwidth_ratio"])
            cfg = VibrationConfig(
                seed=state["seed"], t0=state["t0"], gamma=0.50,
                bands={
                    axis: tuple(
                        SpectralBand(b.center_hz, b.accel_rms, ratio, b.tones)
                        for b in bands
                    )
                    for axis, bands in cfg.bands.items()
                },
            )
        level, report = calibrate_level_scale(cfg, 1000, 16.0)
        assert level == pytest.approx(state["level_scale"], abs=1e-6)
        assert report["gamma_realized"] == pytest.approx(state["gamma_realized"], abs=1e-6)
