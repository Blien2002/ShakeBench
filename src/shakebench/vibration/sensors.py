"""Deterministic synthetic sensors derived from deck motion."""

from __future__ import annotations

import numpy as np

from ..config import G


class SyntheticDeckIMU:
    """Six-channel consumer-MEMS model with bias, noise, quantization, and delay."""

    accel_noise_density = 150e-6 * G
    accel_bias_initial_std = 0.02
    accel_bias_walk = 1e-4
    gyro_noise_density = np.deg2rad(0.005)
    gyro_bias_initial_std = np.deg2rad(0.05)
    gyro_bias_walk = np.deg2rad(1e-4)

    def __init__(self, physics_hz: int = 1000, seed: int = 0) -> None:
        self.physics_hz = int(physics_hz)
        if self.physics_hz <= 0:
            raise ValueError("physics_hz must be positive")
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._accel_bias = self._rng.normal(0.0, self.accel_bias_initial_std, 3)
        self._gyro_bias = self._rng.normal(0.0, self.gyro_bias_initial_std, 3)
        self._delayed = np.zeros(6, dtype=np.float64)

    @staticmethod
    def _quantize(value: np.ndarray, full_scale: float) -> np.ndarray:
        step = 2.0 * full_scale / (2**16 - 1)
        return np.clip(np.round(value / step) * step, -full_scale, full_scale)

    def sample(
        self,
        linear_accel_deck: np.ndarray,
        angular_velocity_deck: np.ndarray,
        *,
        angular_accel_deck: np.ndarray | None = None,
        r_imu_deck: np.ndarray | None = None,
        rotation_body_from_deck: np.ndarray | None = None,
    ) -> np.ndarray:
        dt = 1.0 / self.physics_hz
        alpha = np.zeros(3) if angular_accel_deck is None else np.asarray(angular_accel_deck)
        lever = np.zeros(3) if r_imu_deck is None else np.asarray(r_imu_deck)
        rotation = np.eye(3) if rotation_body_from_deck is None else np.asarray(rotation_body_from_deck)
        accel = rotation @ (np.asarray(linear_accel_deck) + np.cross(alpha, lever))
        gyro = rotation @ np.asarray(angular_velocity_deck)
        self._accel_bias += self._rng.normal(0.0, self.accel_bias_walk * np.sqrt(dt), 3)
        self._gyro_bias += self._rng.normal(0.0, self.gyro_bias_walk * np.sqrt(dt), 3)
        accel += self._accel_bias + self._rng.normal(
            0.0, self.accel_noise_density * np.sqrt(self.physics_hz / 2.0), 3
        )
        gyro += self._gyro_bias + self._rng.normal(
            0.0, self.gyro_noise_density * np.sqrt(self.physics_hz / 2.0), 3
        )
        current = np.concatenate(
            (self._quantize(accel, 16.0 * G), self._quantize(gyro, np.deg2rad(2000.0)))
        )
        delayed = self._delayed.copy()
        self._delayed = current
        return delayed.astype(np.float32)
