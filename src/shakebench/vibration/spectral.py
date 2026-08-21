"""Deterministic, GPU-resident six-axis random-spectrum vibration.

The spectral family is authored as *relative acceleration PSD shapes* whose
absolute level is solved by :func:`calibrate_level_scale`.  That calibration
makes the realized peak vertical acceleration at the workpiece equal
``gamma * G`` exactly (one replay per seed, because ``a_z`` is linear in the
level scale).  Displacement is derived per spectral line with ``1 / omega^2``
so the in-band acceleration PSD stays flat.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import torch

from ..config import AXES, G, GAMMA_CEIL, SpectralBand, VibrationConfig

# ``sweep`` is a single-axis, single narrow band whose absolute level is still
# fixed by the Gamma calibration.  ``sine`` is a deprecated alias.
_SWEEP_TONES = 12
_SWEEP_BANDWIDTH_RATIO = 0.10


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


@dataclass(frozen=True)
class DeckMotionReport:
    """Peak deck-centre translation and speed over the full replay."""

    peak_displacement_m: float
    peak_velocity_m_s: float


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


def _mode_axes(cfg: VibrationConfig) -> tuple[str, ...]:
    """Axes that receive a spectral contribution for the configured mode."""

    if cfg.mode == "off":
        return ()
    if cfg.mode in ("sweep", "sine"):
        return (cfg.sine_axis,)
    return tuple(cfg.active_axes)


def _axis_bands(cfg: VibrationConfig, axis: str) -> tuple[SpectralBand, ...]:
    """Bands authored or synthesized for one axis in the configured mode."""

    if cfg.mode in ("sweep", "sine"):
        if axis != cfg.sine_axis:
            return ()
        return (SpectralBand(cfg.sine_frequency_hz, 1.0, _SWEEP_BANDWIDTH_RATIO, _SWEEP_TONES),)
    if cfg.mode == "spectral":
        return tuple(cfg.bands.get(axis, ()))
    return ()


def _synthesize_axis_lines(
    cfg: VibrationConfig,
    env_id: int,
    axis: str,
    axis_index: int,
    level_scale: float,
    *,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (accel_amp, omega, phase) arrays for one axis and environment.

    Randomness is episode-seeded and exactly reproducible.  Each active axis
    and each environment receives independent line jitter and phase while
    sharing the configured acceleration-PSD bands.  All line acceleration
    amplitudes inside one band are equal, which keeps the in-band acceleration
    PSD flat; displacement amplitude is derived per line by ``1 / omega^2``.
    """

    episode_seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng([int(episode_seed), env_id, axis_index])
    lines: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for band in _axis_bands(cfg, axis):
        fc = band.center_hz * cfg.frequency_scale
        lo, hi = fc * (1.0 - band.bandwidth_ratio), fc * (1.0 + band.bandwidth_ratio)
        frequency = np.linspace(lo, hi, band.tones)
        if hi > lo:
            frequency += rng.uniform(-1.0, 1.0, band.tones) * (hi - lo) / (2 * band.tones)
        omega = 2.0 * math.pi * frequency
        accel_amp = np.full(
            band.tones,
            level_scale * band.accel_rms * math.sqrt(2.0 / band.tones),
            dtype=np.float64,
        )
        phase = rng.uniform(0.0, 2.0 * math.pi, band.tones)
        lines.append((accel_amp, omega, phase))
    return lines


def _add_axis_contribution(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    time_s: np.ndarray,
    index: int,
    lines: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Accumulate analytic displacement/velocity/acceleration for one axis."""

    for accel_amp, omega, phase in lines:
        disp_amp = accel_amp / omega**2
        angle = np.outer(time_s, omega) + phase
        q[:, index] += (disp_amp * np.sin(angle)).sum(axis=1)
        qd[:, index] += ((accel_amp / omega) * np.cos(angle)).sum(axis=1)
        qdd[:, index] += -(accel_amp * np.sin(angle)).sum(axis=1)


def _synthesize_episode(
    cfg: VibrationConfig,
    physics_hz: int,
    episode_s: float,
    env_id: int = 0,
    level_scale: float = 1.0,
    *,
    ramp: bool = True,
):
    """Return analytic deck (q, qd, qdd) arrays on the outer-step grid.

    ``ramp=False`` returns the steady-state spectra over the full grid, which
    is the statistically correct window for reporting per-axis RMS values.
    """

    if physics_hz <= 0 or episode_s <= 0.0:
        raise ValueError("physics_hz and episode_s must be positive")
    num_steps = max(1, int(round(physics_hz * episode_s)))
    time_s = np.arange(0, num_steps, dtype=np.float64) / physics_hz
    q = np.zeros((num_steps, 6), dtype=np.float64)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for axis_index, axis in enumerate(_mode_axes(cfg)):
        index = AXES.index(axis)
        lines = _synthesize_axis_lines(cfg, env_id, axis, axis_index, level_scale)
        _add_axis_contribution(q, qd, qdd, time_s + cfg.t0, index, lines)
    if not ramp:
        return time_s, q, qd, qdd
    r, rd, rdd = _ramp_values(time_s, cfg.ramp_s)
    return (
        time_s,
        r[:, None] * q,
        rd[:, None] * q + r[:, None] * qd,
        rdd[:, None] * q + 2.0 * rd[:, None] * qd + r[:, None] * qdd,
    )


def _workpiece_vertical_accel(qdd: np.ndarray, r_wp: np.ndarray) -> np.ndarray:
    """Vertical acceleration at the workpiece position, including rotation.

    Small-angle rigid-body contribution ``(alpha x r_wp)_z`` is added to the
    deck vertical acceleration.  ``qdd`` has translation in columns 0..2 and
    angular acceleration in columns 3..5.
    """

    return qdd[:, 2] + np.cross(qdd[:, 3:6], r_wp)[:, 2]


def _config_view(cfg: Any) -> tuple[VibrationConfig, int]:
    """Accept either a BenchmarkConfig or a bare VibrationConfig."""

    vibration = getattr(cfg, "vibration", cfg)
    num_envs = max(1, int(getattr(cfg, "num_envs", getattr(vibration, "num_envs", 1))))
    return vibration, num_envs


def calibrate_level_scale(cfg: Any, physics_hz: int, episode_s: float) -> tuple[float, dict]:
    """Solve the single scalar that makes the realized peak vertical
    acceleration at the workpiece equal ``gamma * g``.

    ``a_z`` is linear in the level scale, so the solve is exact and needs
    exactly one replay per seed -- no iteration.  Multi-env runs take the
    minimum scale so no environment exceeds the target.  Returns the scale
    and a report dict that goes straight into the metrics JSON.
    """

    vibration, num_envs = _config_view(cfg)
    if vibration.mode == "off":
        raise ValueError("cannot Gamma-calibrate vibration mode 'off'")
    if physics_hz <= 0 or episode_s <= 0.0:
        raise ValueError("physics_hz and episode_s must be positive")

    r_wp = np.asarray(vibration.workpiece_offset_m, dtype=np.float64)
    env_az: list[np.ndarray] = []
    env_az_peak: list[float] = []
    env_deck_disp_peak: list[float] = []
    env_deck_vel_peak: list[float] = []

    for env_id in range(num_envs):
        _, q, qd, qdd = _synthesize_episode(
            vibration, physics_hz, episode_s, env_id, level_scale=1.0
        )
        az = _workpiece_vertical_accel(qdd, r_wp)
        peak = float(np.max(np.abs(az)))
        if peak <= 0.0:
            raise ValueError("degenerate excitation: zero vertical acceleration")
        env_az.append(az)
        env_az_peak.append(peak)
        env_deck_disp_peak.append(float(np.max(np.linalg.norm(q[:, :3], axis=1))))
        env_deck_vel_peak.append(float(np.max(np.linalg.norm(qd[:, :3], axis=1))))

    # The scale is analytic.  The minimum over environments is the only
    # multi-env choice that never exceeds the target Gamma.
    level_scale = vibration.gamma * G / max(env_az_peak)
    gamma_realized = level_scale * max(env_az_peak) / G
    airborne_fraction = max(
        float(np.mean(level_scale * az < -G))
        for az in env_az
    )

    # Per-axis statistics describe the steady-state spectrum, so use the raw
    # (pre-ramp) replay over the complete grid.  The limiting environment is
    # the one whose unit-level peak set the calibration scale.
    limiting_env = int(np.argmax(env_az_peak))
    _, raw_q, _, raw_qdd = _synthesize_episode(
        vibration,
        physics_hz,
        episode_s,
        limiting_env,
        level_scale=level_scale,
        ramp=False,
    )
    raw_az = _workpiece_vertical_accel(raw_qdd, r_wp)
    az_rms = float(np.sqrt(np.mean(raw_az * raw_az)))
    peak_az = level_scale * max(env_az_peak)

    per_axis_disp_rms: dict[str, float] = {}
    per_axis_accel_rms: dict[str, float] = {}
    per_axis_accel_peak_abs: dict[str, float] = {}
    for index, axis in enumerate(AXES):
        per_axis_disp_rms[axis] = float(np.sqrt(np.mean(raw_q[:, index] ** 2)))
        per_axis_accel_rms[axis] = float(np.sqrt(np.mean(raw_qdd[:, index] ** 2)))
        per_axis_accel_peak_abs[axis] = float(np.max(np.abs(raw_qdd[:, index])))

    report = {
        "gamma_target": float(vibration.gamma),
        "gamma_realized": gamma_realized,
        "level_scale": level_scale,
        "airborne_fraction": airborne_fraction,
        "peak_factor_kappa": peak_az / az_rms if az_rms > 0.0 else float("inf"),
        "per_axis_disp_rms": per_axis_disp_rms,
        "per_axis_accel_rms": per_axis_accel_rms,
        "per_axis_accel_peak_abs": per_axis_accel_peak_abs,
        "peak_deck_displacement_m": level_scale * max(env_deck_disp_peak),
        "peak_deck_velocity_m_s": level_scale * max(env_deck_vel_peak),
        "workpiece_az_rms_m_s2": az_rms,
        "workpiece_az_peak_m_s2": peak_az,
    }
    return level_scale, report


def offline_deck_motion_report(
    cfg: Any,
    physics_hz: int,
    episode_s: float,
    level_scale: float | None = None,
) -> DeckMotionReport:
    """Replay the calibrated deck motion and return peak translation/speed.

    Deck displacement is the magnitude of the deck-centre translation vector
    ``||q[:3]||``; deck speed is ``||qd[:3]||``.  Both are maxima over every
    environment and the full episode (including the ramp, matching runtime).
    """

    vibration, num_envs = _config_view(cfg)
    scale = vibration.level_scale if level_scale is None else float(level_scale)
    worst_disp = 0.0
    worst_vel = 0.0
    for env_id in range(num_envs):
        _, q, qd, _ = _synthesize_episode(
            vibration, physics_hz, episode_s, env_id, level_scale=scale
        )
        worst_disp = max(worst_disp, float(np.max(np.linalg.norm(q[:, :3], axis=1))))
        worst_vel = max(worst_vel, float(np.max(np.linalg.norm(qd[:, :3], axis=1))))
    return DeckMotionReport(peak_displacement_m=worst_disp, peak_velocity_m_s=worst_vel)


def displacement_gate_gamma_max(
    cfg: Any,
    gamma_target: float,
    peak_deck_displacement_m: float,
) -> float:
    """Return the largest Gamma feasible at the configured frequency.

    The feasibility wedge is the standard shaker specification: displacement
    limited at low frequency and acceleration limited above.  Deck peak
    displacement is linear in Gamma, so the unit-level peak for the current
    replay is ``peak / gamma_target``.
    """

    vibration = getattr(cfg, "vibration", cfg)
    if gamma_target <= 0.0 or peak_deck_displacement_m <= 0.0:
        return GAMMA_CEIL
    unit_peak = peak_deck_displacement_m / gamma_target
    return min(GAMMA_CEIL, vibration.max_deck_displacement_m / unit_peak)


def validate_deck_displacement_gate(
    cfg: Any,
    gamma_target: float,
    peak_deck_displacement_m: float,
) -> float:
    """Reject startup when the calibrated deck stroke exceeds its cap.

    Returns the feasible Gamma ceiling for the current frequency; raises
    ValueError with that ceiling included in the message when the configured
    Gamma exceeds it.
    """

    vibration = getattr(cfg, "vibration", cfg)
    cap = vibration.max_deck_displacement_m
    if vibration.mode == "off" or peak_deck_displacement_m <= cap:
        return displacement_gate_gamma_max(cfg, gamma_target, peak_deck_displacement_m)
    gamma_max = displacement_gate_gamma_max(cfg, gamma_target, peak_deck_displacement_m)
    centres = _default_band_centres(vibration)
    if vibration.mode in ("sweep", "sine"):
        centre_label = vibration.sine_axis
        centre_hz = vibration.sine_frequency_hz
    else:
        centre_label = "tz"
        centre_hz = centres.get("tz", float("nan"))
    raise ValueError(
        "unsafe deck displacement: calibrated peak "
        f"{1000.0 * peak_deck_displacement_m:.1f} mm exceeds "
        f"max_deck_displacement_m={1000.0 * cap:.1f} mm at {centre_label} centre frequency "
        f"{vibration.frequency_scale * centre_hz:.2f} Hz; "
        f"feasible Gamma at this frequency is Gamma_max={gamma_max:.4f} "
        f"(Gamma_ceil={GAMMA_CEIL})"
    )


def _default_band_centres(cfg: VibrationConfig) -> dict[str, float]:
    """Centre frequencies of the authored/default band table, unscaled."""

    centres: dict[str, float] = {}
    for axis in AXES:
        bands = _axis_bands(cfg, axis)
        if bands:
            centres[axis] = bands[0].center_hz
    if "tz" not in centres and cfg.mode in ("sweep", "sine"):
        centres["tz"] = cfg.sine_frequency_hz
    return centres


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

    from ..models.supports.base import SupportGroup

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
            cfg.vibration,
            physics_hz,
            cfg.episode_s,
            env_id,
            level_scale=cfg.vibration.level_scale,
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
    phase while sharing the configured acceleration-PSD bands.  The absolute
    level comes from ``cfg.level_scale`` (Gamma-calibrated by the CLI).
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
        if self.cfg.mode == "off":
            return
        for axis_index, axis in enumerate(_mode_axes(self.cfg)):
            env_a: list[np.ndarray] = []
            env_w: list[np.ndarray] = []
            env_p: list[np.ndarray] = []
            for env_id in range(self.num_envs):
                lines = _synthesize_axis_lines(
                    self.cfg,
                    env_id,
                    axis,
                    axis_index,
                    self.cfg.level_scale,
                    seed=seed,
                )
                amplitudes: list[np.ndarray] = []
                omegas: list[np.ndarray] = []
                phases: list[np.ndarray] = []
                for accel_amp, omega, phase in lines:
                    amplitudes.append(accel_amp / omega**2)
                    omegas.append(omega)
                    phases.append(phase)
                env_a.append(np.concatenate(amplitudes))
                env_w.append(np.concatenate(omegas))
                env_p.append(np.concatenate(phases))
            self._amplitude[axis] = torch.as_tensor(
                np.stack(env_a), dtype=torch.float32, device=self.device
            )
            self._omega[axis] = torch.as_tensor(
                np.stack(env_w), dtype=torch.float32, device=self.device
            )
            self._phase[axis] = torch.as_tensor(
                np.stack(env_p), dtype=torch.float32, device=self.device
            )

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
        if self.cfg.mode != "off":
            for axis in self._amplitude:
                index = AXES.index(axis)
                angle = self._omega[axis] * (time_s + self.cfg.t0) + self._phase[axis]
                q[:, index] = torch.sum(self._amplitude[axis] * torch.sin(angle), dim=1)
                qd[:, index] = torch.sum(
                    self._amplitude[axis] * self._omega[axis] * torch.cos(angle), dim=1
                )
                qdd[:, index] = -torch.sum(
                    self._amplitude[axis] * self._omega[axis].square() * torch.sin(angle),
                    dim=1,
                )
        r, rd, rdd = self._ramp(time_s)
        return r * q, rd * q + r * qd, rdd * q + 2.0 * rd * qd + r * qdd
