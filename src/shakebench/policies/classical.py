"""Non-learning fixed-impedance + filtered-x LMS reference controller."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .oracle import _OracleBase, variable_impedance_action


class FxLMSFilter:
    """Small deterministic multi-channel FxLMS adaptive feedforward filter."""

    def __init__(
        self,
        n_channels: int = 3,
        n_taps: int = 32,
        learning_rate: float = 2.0e-4,
        leakage: float = 1.0e-5,
        secondary_path: np.ndarray | None = None,
    ) -> None:
        if n_channels < 1 or n_taps < 2 or learning_rate <= 0.0:
            raise ValueError("invalid FxLMS dimensions or learning rate")
        self.n_channels = int(n_channels)
        self.n_taps = int(n_taps)
        self.learning_rate = float(learning_rate)
        self.leakage = float(leakage)
        secondary = np.asarray(
            secondary_path if secondary_path is not None else (0.72, 0.20, 0.08),
            dtype=np.float32,
        )
        if secondary.ndim != 1 or not np.any(secondary):
            raise ValueError("secondary_path must be a nonzero 1-D impulse response")
        self.secondary_path = secondary
        self.reset()

    def reset(self) -> None:
        self.weights = np.zeros((self.n_channels, self.n_taps), np.float32)
        self.reference = np.zeros((self.n_channels, self.n_taps), np.float32)
        self.filtered_reference = np.zeros_like(self.reference)

    def step(self, reference: np.ndarray, error: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, dtype=np.float32).reshape(self.n_channels)
        error = np.asarray(error, dtype=np.float32).reshape(self.n_channels)
        self.reference[:, 1:] = self.reference[:, :-1]
        self.reference[:, 0] = reference
        for channel in range(self.n_channels):
            filtered = np.convolve(
                self.reference[channel], self.secondary_path, mode="full"
            )[: self.n_taps]
            self.filtered_reference[channel] = filtered
        output = np.sum(self.weights * self.reference, axis=1)
        self.weights *= 1.0 - self.leakage
        self.weights -= self.learning_rate * error[:, None] * self.filtered_reference
        np.clip(self.weights, -0.5, 0.5, out=self.weights)
        return output.astype(np.float32)


class ClassicalPolicy(_OracleBase):
    """FxLMS deck-IMU cancellation with fixed Cartesian impedance.

    It declares no privileged inputs. The reference is the synthetic deck IMU
    and the adaptation error is measured robot tracking/wrist load; no truth
    vibration state, phase, object pose, or future sample is consumed.
    """

    requires_privileged: tuple[str, ...] = ()
    controller_name = "VARIABLE_IMPEDANCE"
    intra_step_mode = "zoh"

    def __init__(
        self,
        *,
        control_freq: int = 200,
        n_taps: int = 32,
        learning_rate: float = 2.0e-4,
        grip_force_n: float = 18.0,
    ) -> None:
        if control_freq <= 0:
            raise ValueError("control_freq must be positive")
        self.grip_force_n = float(grip_force_n)
        self.filter = FxLMSFilter(n_taps=n_taps, learning_rate=learning_rate)
        self.stiffness = np.asarray((1600, 1600, 1400, 120, 120, 90), np.float32)
        super().__init__(
            control_freq=control_freq,
            nominal_grip_force_n=self.grip_force_n,
        )

    def reset(self) -> None:
        super().reset()
        self.filter.reset()
        self._reference_eef: np.ndarray | None = None

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        imu = np.asarray(observation["deck_imu"], dtype=np.float32)
        wrist = np.asarray(observation["robot0_wrist_force"], dtype=np.float32)
        delta, gripping = self._equilibrium_delta(observation)
        if self._reference_eef is None:
            self._reference_eef = eef.copy()
        tracking_error = eef - self._reference_eef
        # The wrist term makes the adaptive update respond to contact loads as
        # well as kinematic error, while staying entirely sensor based.
        error = tracking_error + 2.0e-5 * wrist
        cancellation = self.filter.step(imu[:3], error)
        delta[:3] -= np.clip(cancellation, -0.03, 0.03)
        self._reference_eef = eef + delta[:3]
        self._step += 1
        return variable_impedance_action(
            delta, self.stiffness, self.grip_force_n if gripping else 0.0
        )
