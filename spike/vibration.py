"""NumPy-only wrapper around ShakeBench's authored vibration spectrum."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The spike environment deliberately does not install Isaac Lab or PyTorch.
# ShakeBench's spectral module only needs the Tensor attribute for annotations;
# scipy may also probe it while importing optional array backends.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from shakebench.config import AXES, BenchmarkConfig, VibrationConfig  # noqa: E402
from shakebench.vibration import spectral as sp  # noqa: E402


class NumpyVibration:
    """Analytic per-physics-step evaluator of the authored six-axis motion."""

    def __init__(self, config: VibrationConfig, phase_offset_rad: float = 0.0):
        self.config = config
        self.phase_offset_rad = phase_offset_rad
        self._lines: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for axis_index, axis in enumerate(config.active_axes):
            coordinate = AXES.index(axis)
            self._lines[coordinate] = sp._synthesize_axis_lines(
                config,
                env_id=0,
                axis=axis,
                axis_index=axis_index,
                level_scale=config.level_scale,
                seed=config.seed,
            )

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return analytic displacement, velocity, and acceleration at time."""

        q = np.zeros(6, dtype=np.float64)
        qd = np.zeros(6, dtype=np.float64)
        qdd = np.zeros(6, dtype=np.float64)
        spectral_time = time_s + self.config.t0
        for coordinate, bands in self._lines.items():
            for accel_amp, omega, phase in bands:
                angle = spectral_time * omega + phase + self.phase_offset_rad
                q[coordinate] += np.sum(accel_amp / omega**2 * np.sin(angle))
                qd[coordinate] += np.sum(accel_amp / omega * np.cos(angle))
                qdd[coordinate] -= np.sum(accel_amp * np.sin(angle))

        ramp, ramp_d, ramp_dd = _smoothstep5(time_s, self.config.ramp_s)
        return (
            ramp * q,
            ramp_d * q + ramp * qd,
            ramp_dd * q + 2.0 * ramp_d * qd + ramp * qdd,
        )


def calibrated_vibration(
    gamma: float,
    *,
    seed: int,
    physics_hz: int,
    episode_s: float,
    t0: float = 0.0,
    phase_offset_rad: float = 0.0,
) -> tuple[NumpyVibration, dict]:
    """Create a calibrated VibrationConfig without touching Isaac modules."""

    base: VibrationConfig = BenchmarkConfig().vibration
    config = dataclasses.replace(
        base,
        gamma=gamma,
        seed=seed,
        t0=t0,
        ballistic_allowed=gamma >= 1.0,
    )
    level_scale, report = sp.calibrate_level_scale(config, physics_hz, episode_s)
    config = dataclasses.replace(config, level_scale=level_scale)
    return NumpyVibration(config, phase_offset_rad=phase_offset_rad), report


def _smoothstep5(time_s: float, ramp_s: float) -> tuple[float, float, float]:
    """Fifth-order ramp and its two analytic derivatives."""

    if time_s >= ramp_s:
        return 1.0, 0.0, 0.0
    u = max(0.0, time_s / ramp_s)
    value = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    first = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / ramp_s
    second = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / ramp_s**2
    return value, first, second
