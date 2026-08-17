"""Deterministic, GPU-resident six-axis random-spectrum vibration."""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import AXES, VibrationConfig


def estimated_peak_velocity_m_s(cfg: VibrationConfig, mount_radius_m: float = 0.93) -> float:
    """Estimate the 3.5-sigma peak surface speed used for timestep audits."""

    if cfg.mode == "off":
        return 0.0
    if cfg.mode == "sine":
        speed = abs(cfg.sine_amplitude * 2.0 * math.pi * cfg.sine_frequency_hz)
        return speed if cfg.sine_axis in ("tx", "ty", "tz") else speed * mount_radius_m
    axis_speeds: list[float] = []
    for axis in cfg.active_axes:
        variance = sum(
            ((2.0 * math.pi * band.center_hz) * band.rms) ** 2
            for band in cfg.bands.get(axis, ())
        )
        speed = math.sqrt(variance)
        axis_speeds.append(speed if axis.startswith("t") else mount_radius_m * speed)
    return cfg.spectral_scale * 3.5 * max(axis_speeds, default=0.0)


def estimated_substep_displacement_m(
    cfg: VibrationConfig,
    physics_hz: int,
    substeps: int,
    mount_radius_m: float = 0.93,
) -> float:
    """Peak speed estimate multiplied by the effective solver timestep."""

    if physics_hz <= 0 or substeps <= 0:
        raise ValueError("physics_hz and substeps must be positive")
    return estimated_peak_velocity_m_s(cfg, mount_radius_m) / (physics_hz * substeps)


def validate_impulsive_timestep(
    cfg: VibrationConfig,
    physics_hz: int,
    substeps: int,
    limit_m: float = 0.0002,
) -> float:
    """Reject waveforms whose estimated per-substep travel is unsafe.

    The spectral estimate is a conservative 3.5-sigma bound rather than a
    deterministic maximum, but exceeding it by several times caused actual
    contact tunnelling in the 240 Hz training profile.  All modes therefore
    share the configured startup gate.
    """

    displacement = estimated_substep_displacement_m(cfg, physics_hz, substeps)
    if cfg.mode != "off" and displacement > limit_m:
        raise ValueError(
            f"unsafe {cfg.mode} excitation: estimated substep displacement "
            f"{1000.0 * displacement:.3f} mm exceeds {1000.0 * limit_m:.3f} mm; "
            "raise physics_hz/substeps or reduce amplitude/frequency"
        )
    return displacement


class SpectralVibration:
    """Synthesizes displacement, velocity and acceleration analytically.

    Randomness is episode-seeded and therefore exactly reproducible.  Each
    environment receives independent line jitter and phase while sharing the
    configured PSD bands.
    """

    def __init__(self, cfg: VibrationConfig, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._amplitude: dict[str, torch.Tensor] = {}
        self._omega: dict[str, torch.Tensor] = {}
        self._phase: dict[str, torch.Tensor] = {}
        self.reseed(cfg.seed)

    def reseed(self, seed: int) -> None:
        self._amplitude.clear()
        self._omega.clear()
        self._phase.clear()
        if self.cfg.mode != "spectral":
            return
        for axis in self.cfg.active_axes:
            env_a: list[np.ndarray] = []
            env_w: list[np.ndarray] = []
            env_p: list[np.ndarray] = []
            for env_id in range(self.num_envs):
                rng = np.random.default_rng(int(seed) + 1009 * env_id)
                amplitudes: list[np.ndarray] = []
                omegas: list[np.ndarray] = []
                phases: list[np.ndarray] = []
                for band in self.cfg.bands.get(axis, ()):
                    lo = band.center_hz * (1.0 - band.bandwidth_ratio)
                    hi = band.center_hz * (1.0 + band.bandwidth_ratio)
                    frequency = np.linspace(lo, hi, band.tones)
                    if hi > lo:
                        frequency += rng.uniform(-1.0, 1.0, band.tones) * (hi - lo) / (2 * band.tones)
                    amplitudes.append(
                        np.full(
                            band.tones,
                            self.cfg.spectral_scale * band.rms * math.sqrt(2.0 / band.tones),
                        )
                    )
                    omegas.append(2.0 * math.pi * frequency)
                    phases.append(rng.uniform(0.0, 2.0 * math.pi, band.tones))
                if amplitudes:
                    env_a.append(np.concatenate(amplitudes))
                    env_w.append(np.concatenate(omegas))
                    env_p.append(np.concatenate(phases))
            if env_a:
                self._amplitude[axis] = torch.as_tensor(np.stack(env_a), dtype=torch.float32, device=self.device)
                self._omega[axis] = torch.as_tensor(np.stack(env_w), dtype=torch.float32, device=self.device)
                self._phase[axis] = torch.as_tensor(np.stack(env_p), dtype=torch.float32, device=self.device)

    def _ramp(self, time_s: float) -> tuple[float, float, float]:
        if time_s >= self.cfg.ramp_s:
            return 1.0, 0.0, 0.0
        u = max(0.0, time_s / self.cfg.ramp_s)
        r = 10 * u**3 - 15 * u**4 + 6 * u**5
        rd = (30 * u**2 - 60 * u**3 + 30 * u**4) / self.cfg.ramp_s
        rdd = (60 * u - 180 * u**2 + 120 * u**3) / self.cfg.ramp_s**2
        return r, rd, rdd

    def sample(self, time_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        qd = torch.zeros_like(q)
        qdd = torch.zeros_like(q)
        if self.cfg.mode == "sine":
            index = AXES.index(self.cfg.sine_axis)
            omega = 2.0 * math.pi * self.cfg.sine_frequency_hz
            q[:, index] = self.cfg.sine_amplitude * math.sin(omega * time_s)
            qd[:, index] = self.cfg.sine_amplitude * omega * math.cos(omega * time_s)
            qdd[:, index] = -self.cfg.sine_amplitude * omega**2 * math.sin(omega * time_s)
        elif self.cfg.mode == "spectral":
            for index, axis in enumerate(AXES):
                if axis not in self._amplitude:
                    continue
                angle = self._omega[axis] * time_s + self._phase[axis]
                q[:, index] = torch.sum(self._amplitude[axis] * torch.sin(angle), dim=1)
                qd[:, index] = torch.sum(self._amplitude[axis] * self._omega[axis] * torch.cos(angle), dim=1)
                qdd[:, index] = -torch.sum(self._amplitude[axis] * self._omega[axis].square() * torch.sin(angle), dim=1)
        r, rd, rdd = self._ramp(time_s)
        return r * q, rd * q + r * qd, rdd * q + 2.0 * rd * qd + r * qdd
