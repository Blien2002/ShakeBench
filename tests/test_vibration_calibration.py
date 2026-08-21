from dataclasses import replace

import pytest

from shakebench.config import VibrationConfig
from shakebench.vibration import calibrate_level_scale


@pytest.mark.parametrize("seed,t0", [(17, 0.05), (31, 25.0), (73, 99.95)])
def test_gamma_calibration_is_exact_for_seed_and_time_window(seed: int, t0: float) -> None:
    cfg = VibrationConfig(seed=seed, t0=t0, gamma=0.50)
    level, report = calibrate_level_scale(cfg, 1000, 16.0)
    assert level > 0.0
    assert report["gamma_realized"] == pytest.approx(0.50, abs=1e-12)


def test_t0_changes_level_but_not_realized_gamma() -> None:
    base = VibrationConfig(seed=17, gamma=0.75)
    level_a, report_a = calibrate_level_scale(replace(base, t0=0.1), 1000, 16.0)
    level_b, report_b = calibrate_level_scale(replace(base, t0=81.3), 1000, 16.0)
    assert level_a != level_b
    assert report_a["gamma_realized"] == pytest.approx(report_b["gamma_realized"], abs=1e-12)
