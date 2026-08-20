"""Deterministic, GPU-resident six-axis random-spectrum vibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from .config import AXES, VibrationConfig


@dataclass(frozen=True)
class SupportTravelReport:
    """Deterministic full-episode support travel replay.

    ``max_v_dt_m`` is the teleport bound if written velocities were ignored;
    ``max_half_a_dt2_m`` is the residual of first-order-hold integration;
    ``max_substep_travel_m`` is the value compared against the startup gate.
    """

    max_v_dt_m: float
    max_half_a_dt2_m: float
    max_outer_travel_m: float
    max_substep_travel_m: float
    worst_member: str
    worst_time_s: float


def _ramp_values(time_s: np.ndarray, ramp_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized fifth-order smoothstep ramp and its first two derivatives."""

    r = np.ones_like(time_s)
    rd = np.zeros_like(time_s)
    rdd = np.zeros_like(time_s)
    ramp = time_s < ramp_s
    u = np.maximum(0.0, time_s[ramp] / ramp_s)
    r[ramp] = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    rd[ramp] = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / ramp_s
    rdd[ramp] = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / ramp_s**2
    return r, rd, rdd


def _euler_rotation_matrices(q: np.ndarray) -> np.ndarray:
    """Batched Rz(q5)·Ry(q4)·Rx(q3), matching ``quat_from_euler_xyz``."""

    rx, ry, rz = q[:, 3], q[:, 4], q[:, 5]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.zeros((q.shape[0], 3, 3))
    rx_m[:, 0, 0] = 1.0
    rx_m[:, 1, 1] = cx
    rx_m[:, 1, 2] = -sx
    rx_m[:, 2, 1] = sx
    rx_m[:, 2, 2] = cx
    ry_m = np.zeros_like(rx_m)
    ry_m[:, 0, 0] = cy
    ry_m[:, 0, 2] = sy
    ry_m[:, 1, 1] = 1.0
    ry_m[:, 2, 0] = -sy
    ry_m[:, 2, 2] = cy
    rz_m = np.zeros_like(rx_m)
    rz_m[:, 0, 0] = cz
    rz_m[:, 0, 1] = -sz
    rz_m[:, 1, 0] = sz
    rz_m[:, 1, 1] = cz
    rz_m[:, 2, 2] = 1.0
    return rz_m @ ry_m @ rx_m


def _synthesize_episode(
    cfg: VibrationConfig,
    physics_hz: int,
    episode_s: float,
    env_id: int = 0,
):
    """Return analytic deck (q, qd, qdd) arrays on the outer-step grid."""

    num_steps = max(1, int(round(physics_hz * episode_s)))
    time_s = np.arange(0, num_steps, dtype=np.float64) / physics_hz
    q = np.zeros((num_steps, 6), dtype=np.float64)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    if cfg.mode == "off":
        return time_s, q, qd, qdd
    if cfg.mode == "sine":
        index = AXES.index(cfg.sine_axis)
        omega = 2.0 * math.pi * cfg.sine_frequency_hz
        q[:, index] = cfg.sine_amplitude * np.sin(omega * time_s)
        qd[:, index] = cfg.sine_amplitude * omega * np.cos(omega * time_s)
        qdd[:, index] = -cfg.sine_amplitude * omega**2 * np.sin(omega * time_s)
    else:
        for axis_index, axis in enumerate(cfg.active_axes):
            rng = np.random.default_rng([int(cfg.seed), env_id, axis_index])
            for band in cfg.bands.get(axis, ()):
                lo = band.center_hz * (1.0 - band.bandwidth_ratio)
                hi = band.center_hz * (1.0 + band.bandwidth_ratio)
                frequency = np.linspace(lo, hi, band.tones)
                if hi > lo:
                    frequency += rng.uniform(-1.0, 1.0, band.tones) * (hi - lo) / (2 * band.tones)
                phases = rng.uniform(0.0, 2.0 * math.pi, band.tones)
                amplitude = cfg.spectral_scale * band.rms * math.sqrt(2.0 / band.tones)
                omega = 2.0 * math.pi * frequency
                angle = np.outer(time_s, omega) + phases
                index = AXES.index(axis)
                q[:, index] += amplitude * np.sin(angle).sum(axis=1)
                qd[:, index] += (amplitude * omega * np.cos(angle)).sum(axis=1)
                qdd[:, index] += -(amplitude * omega**2 * np.sin(angle)).sum(axis=1)
    r, rd, rdd = _ramp_values(time_s, cfg.ramp_s)
    return (
        time_s,
        r[:, None] * q,
        rd[:, None] * q + r[:, None] * qd,
        rdd[:, None] * q + 2.0 * rd[:, None] * qd + r[:, None] * qdd,
    )


def offline_support_travel_report(
    cfg: Any,
    groups: tuple[Any, ...],
    physics_hz: int,
    substeps: int,
) -> SupportTravelReport:
    """Replay the current seed's full episode and return true travel bounds.

    The member table is the same ``SupportGroup`` table used by the runtime
    writer, so there is no second source of geometry truth.  Angular
    acceleration for the first-order-hold residual uses centred differences
    on 1 kHz analytic velocity samples; at the 32 Hz upper band this is
    accurate to far below the reported millimetre quantities.
    """

    from .supports import SupportGroup

    if physics_hz <= 0 or substeps <= 0:
        raise ValueError("physics_hz and substeps must be positive")
    for group in groups:
        if not isinstance(group, SupportGroup):
            raise TypeError("offline replay expects SupportGroup instances")
        if group.motion_source != "deck":
            raise NotImplementedError(
                f"replay gate does not yet cover motion_source={group.motion_source!r}"
            )

    dt = 1.0 / physics_hz
    worst_speed = 0.0
    worst_accel = 0.0
    worst_member = "none"
    worst_time = 0.0

    for env_id in range(max(1, cfg.num_envs)):
        time_s, q, qd, qdd = _synthesize_episode(
            cfg.vibration, physics_hz, cfg.episode_s, env_id
        )
        if time_s.size == 0:
            continue

        rates = qd[:, 3:6]
        rx, ry, rz = q[:, 3], q[:, 4], q[:, 5]
        rx_dot, ry_dot, rz_dot = rates[:, 0], rates[:, 1], rates[:, 2]
        sin_rz, cos_rz = np.sin(rz), np.cos(rz)
        sin_ry, cos_ry = np.sin(ry), np.cos(ry)
        omega = np.stack(
            (
                rx_dot * cos_rz * cos_ry - ry_dot * sin_rz,
                rx_dot * sin_rz * cos_ry + ry_dot * cos_rz,
                -rx_dot * sin_ry + rz_dot,
            ),
            axis=1,
        )
        alpha = np.zeros_like(omega)
        alpha[1:-1] = (omega[2:] - omega[:-2]) / (2.0 * dt)
        alpha[0] = (omega[1] - omega[0]) / dt
        alpha[-1] = (omega[-1] - omega[-2]) / dt

        rotation = _euler_rotation_matrices(q)

        omega_norm = np.linalg.norm(omega, axis=1)
        alpha_norm = np.linalg.norm(alpha, axis=1)
        for group in groups:
            anchor = np.asarray(group.rotation_anchor, dtype=np.float64)
            for member in group.members:
                local = np.asarray(member.local, dtype=np.float64)
                bound = float(member.bound_radius_m)
                offset0 = local - anchor
                r_center = np.einsum("nij,j->ni", rotation, offset0)
                v_center = qd[:, :3] + np.cross(omega, r_center)
                a_center = (
                    qdd[:, :3]
                    + np.cross(alpha, r_center)
                    + np.cross(omega, np.cross(omega, r_center))
                )
                speed_bound = (
                    np.linalg.norm(v_center, axis=1) + omega_norm * bound
                )
                accel_bound = (
                    np.linalg.norm(a_center, axis=1)
                    + alpha_norm * bound
                    + omega_norm * omega_norm * bound
                )
                step = int(np.argmax(speed_bound))
                speed = float(speed_bound[step])
                accel = float(np.max(accel_bound))
                if speed > worst_speed:
                    worst_speed = speed
                    worst_member = member.name
                    worst_time = float(time_s[step])
                if accel > worst_accel:
                    worst_accel = accel

    substep_dt = dt / substeps
    return SupportTravelReport(
        max_v_dt_m=worst_speed * dt,
        max_half_a_dt2_m=0.5 * worst_accel * dt * dt,
        max_outer_travel_m=worst_speed * dt,
        max_substep_travel_m=worst_speed * substep_dt,
        worst_member=worst_member,
        worst_time_s=worst_time,
    )


class SpectralVibration:
    """Synthesizes displacement, velocity and acceleration analytically.

    Randomness is episode-seeded and therefore exactly reproducible.  Each
    environment and each active axis receives independent line jitter and
    phase while sharing the configured PSD bands.
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
        for axis_index, axis in enumerate(self.cfg.active_axes):
            env_a: list[np.ndarray] = []
            env_w: list[np.ndarray] = []
            env_p: list[np.ndarray] = []
            for env_id in range(self.num_envs):
                rng = np.random.default_rng([int(seed), env_id, axis_index])
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
