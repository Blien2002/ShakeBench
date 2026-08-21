"""Three disclosed oracle tiers for the Round-7 decision experiments.

The policies emit the public 13-D ``VARIABLE_IMPEDANCE`` action: Cartesian
equilibrium delta, diagonal stiffness, and grip force.  They deliberately do
not attach the workpiece or modify simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


EQUILIBRIUM_LIMIT = np.asarray((0.05, 0.05, 0.05, 0.5, 0.5, 0.5), np.float32)
STIFFNESS_MIN = np.asarray((10, 10, 10, 1, 1, 1), np.float32)
STIFFNESS_MAX = np.asarray((3000, 3000, 3000, 300, 300, 300), np.float32)
RELATIVE_ACCEL_RMS = np.asarray(
    (0.500, 0.350, 1.000, 0.30 / 0.65, 0.30 / 0.65, 0.15 / 0.65),
    np.float32,
)


def _array(observation: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return np.asarray(observation[key], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"oracle requires observation key {key!r}") from exc


def variable_impedance_action(
    equilibrium_delta: np.ndarray,
    stiffness: np.ndarray,
    grip_force_n: float,
) -> np.ndarray:
    """Encode physical slow variables into the normalized public action."""

    delta = np.asarray(equilibrium_delta, dtype=np.float32)
    gains = np.asarray(stiffness, dtype=np.float32)
    if delta.shape != (6,) or gains.shape != (6,):
        raise ValueError("equilibrium_delta and stiffness must both have shape (6,)")
    action = np.empty(13, np.float32)
    action[:6] = np.clip(delta / EQUILIBRIUM_LIMIT, -1.0, 1.0)
    action[6:12] = np.clip(
        2.0 * (gains - STIFFNESS_MIN) / (STIFFNESS_MAX - STIFFNESS_MIN) - 1.0,
        -1.0,
        1.0,
    )
    action[12] = np.clip(float(grip_force_n) / 35.0 - 1.0, -1.0, 1.0)
    return action


def workpiece_acceleration(qdd: np.ndarray, lever_m: np.ndarray) -> np.ndarray:
    """Small-angle workpiece acceleration ``a + alpha x r``."""

    value = np.asarray(qdd, dtype=np.float32).reshape(6)
    lever = np.asarray(lever_m, dtype=np.float32).reshape(3)
    return value[:3] + np.cross(value[3:], lever)


def required_grip_force(
    acceleration: np.ndarray,
    mass_kg: float,
    friction: float,
    safety_factor: float = 1.3,
) -> float:
    """Coulomb lower bound used by the oracle grip-force feedforward."""

    if mass_kg <= 0.0 or friction <= 0.0 or safety_factor < 1.0:
        raise ValueError("mass/friction must be positive and safety_factor must be >= 1")
    return safety_factor * mass_kg * float(np.linalg.norm(acceleration)) / friction


@dataclass(frozen=True)
class _MotionPhase:
    end_fraction: float
    mode: str
    gripping: bool


_PHASES = (
    _MotionPhase(0.08, "settle", False),
    _MotionPhase(0.20, "approach", False),
    _MotionPhase(0.32, "descend", False),
    _MotionPhase(0.43, "grasp", True),
    _MotionPhase(0.57, "lift", True),
    _MotionPhase(0.72, "transfer", True),
    _MotionPhase(0.84, "place", True),
    _MotionPhase(0.91, "release", False),
    _MotionPhase(1.00, "retreat", False),
)


class _OracleBase:
    controller_name = "VARIABLE_IMPEDANCE"
    intra_step_mode = "feedforward"
    stiffness = np.asarray((1800, 1800, 1500, 140, 140, 100), np.float32)

    def __init__(
        self,
        *,
        control_freq: int,
        episode_s: float = 16.0,
        nominal_mass_kg: float = 0.6,
        nominal_friction: float = 0.8,
        nominal_grip_force_n: float = 12.0,
    ) -> None:
        if control_freq <= 0 or episode_s <= 0.0:
            raise ValueError("control_freq and episode_s must be positive")
        self.control_freq = int(control_freq)
        self.episode_s = float(episode_s)
        self.nominal_mass_kg = float(nominal_mass_kg)
        self.nominal_friction = float(nominal_friction)
        self.nominal_grip_force_n = float(nominal_grip_force_n)
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._settle_eef: np.ndarray | None = None

    @property
    def time_s(self) -> float:
        return self._step / self.control_freq

    def _phase(self) -> _MotionPhase:
        fraction = min(self.time_s / self.episode_s, 1.0)
        return next(phase for phase in _PHASES if fraction <= phase.end_fraction)

    def _nominal_object_target(
        self, observation: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        # The phase oracle's perception input is RGB; this deterministic
        # fallback is the authored reset geometry. A camera pose estimator can
        # replace it without changing the oracle privilege contract.
        obj = np.asarray((0.08, -0.13, 0.47), np.float32)
        target = np.asarray((0.08, 0.17, 0.376), np.float32)
        if "object_pos" in observation:
            obj = _array(observation, "object_pos")
        if "target_pos" in observation:
            target = _array(observation, "target_pos")
        return obj, target

    def _equilibrium_delta(
        self, observation: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, bool]:
        eef = _array(observation, "robot0_eef_pos")
        if self._settle_eef is None:
            self._settle_eef = eef.copy()
        obj, target = self._nominal_object_target(observation)
        phase = self._phase()
        if phase.mode == "settle":
            desired = self._settle_eef
        elif phase.mode == "approach":
            desired = obj + np.asarray((0.0, 0.0, 0.18), np.float32)
        elif phase.mode in ("descend", "grasp"):
            desired = obj + np.asarray((0.0, 0.0, 0.08), np.float32)
        elif phase.mode == "lift":
            desired = obj + np.asarray((0.0, 0.0, 0.25), np.float32)
        elif phase.mode == "transfer":
            desired = target + np.asarray((0.0, 0.0, 0.25), np.float32)
        elif phase.mode in ("place", "release"):
            desired = target + np.asarray((0.0, 0.0, 0.10), np.float32)
        else:
            desired = target + np.asarray((0.0, 0.0, 0.25), np.float32)
        delta = np.zeros(6, np.float32)
        delta[:3] = desired - eef
        return delta, phase.gripping

    def _finish_action(self, delta: np.ndarray, gripping: bool, force_n: float) -> np.ndarray:
        force = max(self.nominal_grip_force_n, force_n) if gripping else 0.0
        action = variable_impedance_action(delta, self.stiffness, force)
        self._step += 1
        return action


class OracleFullPolicy(_OracleBase):
    """Absolute upper bound: full truth and a 1 kHz policy loop."""

    requires_privileged = ("object", "vibration")

    def __init__(self, *, control_freq: int = 1000, **kwargs) -> None:
        if control_freq != 1000:
            raise ValueError("oracle_full must run at 1000 Hz")
        super().__init__(control_freq=control_freq, **kwargs)

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        delta, gripping = self._equilibrium_delta(observation)
        acceleration = workpiece_acceleration(
            _array(observation, "vibration_qdd"), np.asarray((0.65, 0.0, 0.0))
        )
        mass = float(_array(observation, "object_mass").reshape(-1)[0])
        friction = float(_array(observation, "object_mu").reshape(-1)[0])
        force = required_grip_force(acceleration, mass, friction)
        return self._finish_action(delta, gripping, force)


class OraclePhasePolicy(_OracleBase):
    """Slow policy with true spectral phase and real sensor observations only."""

    requires_privileged = ("phase",)

    def __init__(self, *, control_freq: int = 10, **kwargs) -> None:
        super().__init__(control_freq=control_freq, **kwargs)

    def _analytic_qdd(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        phase = _array(observation, "vibration_phase")
        mask = _array(observation, "vibration_line_mask")
        level_scale = float(
            _array(observation, "vibration_level_scale").reshape(-1)[0]
        )
        line_count = np.maximum(mask.sum(axis=1), 1.0)
        line_amplitude = (
            level_scale * RELATIVE_ACCEL_RMS * np.sqrt(2.0 / line_count)
        )
        amplitude = line_amplitude[:, None] * mask
        raw_qdd = -np.sum(amplitude * np.sin(phase) * mask, axis=1)
        # During the C2 ramp, reconstruct the full analytic derivative rather
        # than pretending the steady-state acceleration starts at t=0.
        time_s = float(_array(observation, "vibration_time").reshape(-1)[0])
        ramp_s = float(_array(observation, "vibration_ramp_s").reshape(-1)[0])
        if time_s >= ramp_s:
            return raw_qdd.astype(np.float32)
        omega = _array(observation, "vibration_omega")
        safe_omega = np.where(mask > 0, omega, 1.0)
        raw_q = np.sum(amplitude / safe_omega**2 * np.sin(phase) * mask, axis=1)
        raw_qd = np.sum(amplitude / safe_omega * np.cos(phase) * mask, axis=1)
        u = max(0.0, time_s / ramp_s)
        r = 10 * u**3 - 15 * u**4 + 6 * u**5
        rd = (30 * u**2 - 60 * u**3 + 30 * u**4) / ramp_s
        rdd = (60 * u - 180 * u**2 + 120 * u**3) / ramp_s**2
        return (rdd * raw_q + 2.0 * rd * raw_qd + r * raw_qdd).astype(np.float32)

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        delta, gripping = self._equilibrium_delta(observation)
        acceleration = workpiece_acceleration(
            self._analytic_qdd(observation), np.asarray((0.65, 0.0, 0.0))
        )
        force = required_grip_force(
            acceleration, self.nominal_mass_kg, self.nominal_friction
        )
        return self._finish_action(delta, gripping, force)


class OracleReactivePolicy(_OracleBase):
    """Full instantaneous state feedback with no phase, history, or prediction."""

    requires_privileged = ("object", "instantaneous_load")
    intra_step_mode = "zoh"

    def __init__(self, *, control_freq: int = 10, **kwargs) -> None:
        super().__init__(control_freq=control_freq, **kwargs)

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        # This method intentionally keeps no observation history and performs
        # no frequency estimation or extrapolation.
        delta, gripping = self._equilibrium_delta(observation)
        acceleration = workpiece_acceleration(
            _array(observation, "vibration_qdd"), np.asarray((0.65, 0.0, 0.0))
        )
        mass = float(_array(observation, "object_mass").reshape(-1)[0])
        friction = float(_array(observation, "object_mu").reshape(-1)[0])
        force = required_grip_force(acceleration, mass, friction)
        return self._finish_action(delta, gripping, force)
