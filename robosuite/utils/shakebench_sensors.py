"""Canonical ShakeBench deck IMU model.

This module owns the measurement model used by the State-V1 observation.  The
physics equation is deliberately separate from the runtime sampling plumbing
so it can be tested with small synthetic rigid-body examples as well as with a
live MuJoCo simulation.

The public policy signal is the delayed, quantized ``float32`` window.  The
``IMUSample`` and ``trace`` interfaces expose the decomposition needed by the
privileged recorder, but the environment never puts those fields in its
ordinary observation dictionary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import pi
from typing import Any, Iterable, Optional

import mujoco
import numpy as np


G0_M_S2 = 9.80665
GRAVITY_WORLD_M_S2 = (0.0, 0.0, -G0_M_S2)

IMU_CHANNELS = (
    "specific_force_x",
    "specific_force_y",
    "specific_force_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)
IMU_SAMPLE_RATE_HZ = 200.0
IMU_DT_S = 1.0 / IMU_SAMPLE_RATE_HZ
IMU_POLICY_RATE_HZ = 20.0
IMU_WINDOW_SAMPLES = 10
IMU_WINDOW_SHAPE = (IMU_WINDOW_SAMPLES, len(IMU_CHANNELS))
IMU_DELIVERY_DELAY_SAMPLES = 1

BUTTERWORTH_ORDER = 2
BUTTERWORTH_CUTOFF_HZ = 40.0
BUTTERWORTH_B = (0.20657208382614792, 0.41314416765229584, 0.20657208382614792)
BUTTERWORTH_A = (1.0, -0.36952737735124142, 0.19581571265583306)
BUTTERWORTH_ENBW_HZ = 40.7618155662

ACCEL_RANGE_M_S2 = 16.0 * G0_M_S2
GYRO_RANGE_RAD_S = 2000.0 * pi / 180.0
ACCEL_NOISE_DENSITY_M_S2_SQRT_HZ = 150.0e-6 * G0_M_S2
GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ = 0.005 * pi / 180.0
ACCEL_INITIAL_BIAS_STD_M_S2 = 0.02
GYRO_INITIAL_BIAS_STD_RAD_S = 0.05 * pi / 180.0
ACCEL_BIAS_DIFFUSION_M_S2_SQRT_S = 1.0e-4
GYRO_BIAS_DIFFUSION_RAD_S_SQRT_S = 1.0e-4 * pi / 180.0
IMU_BITS = 16
IMU_PROFILE_ID = "canonical_midgrade_v1"


class ShakeBenchSensorError(ValueError):
    """Raised when an IMU input or configuration violates the sensor contract."""


def _finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ShakeBenchSensorError(f"{name} must contain {length} finite values") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ShakeBenchSensorError(f"{name} must contain {length} finite values")
    return np.array(result, dtype=float, copy=True)


def _finite_matrix(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ShakeBenchSensorError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ShakeBenchSensorError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return np.array(result, dtype=float, copy=True)


def _normalise_quaternion_wxyz(value: Any, name: str = "quaternion_wxyz") -> np.ndarray:
    quaternion = _finite_vector(name, value, 4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ShakeBenchSensorError(f"{name} must have non-zero norm")
    return quaternion / norm


def _quat_wxyz_to_matrix(value: Any) -> np.ndarray:
    w, x, y, z = _normalise_quaternion_wxyz(value)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _matrix_to_quaternion_wxyz(matrix: Any) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized ``wxyz`` quaternion."""

    rotation = _finite_matrix("rotation", matrix, (3, 3))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        result = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ),
            dtype=float,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1e-16, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            result = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ),
                dtype=float,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1e-16, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
            result = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ),
                dtype=float,
            )
        else:
            scale = 2.0 * np.sqrt(max(1e-16, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
            result = np.asarray(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ),
                dtype=float,
            )
    return _normalise_quaternion_wxyz(result)


def rigid_body_point_acceleration(
    origin_acceleration_world_m_s2: Iterable[float],
    angular_acceleration_world_rad_s2: Iterable[float],
    angular_velocity_world_rad_s: Iterable[float],
    lever_arm_world_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return acceleration at a rigidly attached point in world coordinates.

    The returned value implements the exact terms required by the canonical
    profile:

    ``a_P = a_O + alpha x r_OP + omega x (omega x r_OP)``.

    There is no relative or Coriolis term because the sensor is rigidly
    attached to the deck.
    """

    origin = _finite_vector("origin_acceleration_world_m_s2", origin_acceleration_world_m_s2, 3)
    alpha = _finite_vector("angular_acceleration_world_rad_s2", angular_acceleration_world_rad_s2, 3)
    omega = _finite_vector("angular_velocity_world_rad_s", angular_velocity_world_rad_s, 3)
    lever = _finite_vector("lever_arm_world_m", lever_arm_world_m, 3)
    return origin + np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))


def specific_force_from_rigid_body_motion(
    origin_acceleration_world_m_s2: Iterable[float],
    angular_acceleration_world_rad_s2: Iterable[float],
    angular_velocity_world_rad_s: Iterable[float],
    lever_arm_world_m: Iterable[float] = (0.0, 0.0, 0.0),
    gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
    rotation_world_to_sensor: Iterable[Iterable[float]] = np.eye(3),
) -> np.ndarray:
    """Compute sensor-frame accelerometer specific force.

    ``rotation_world_to_sensor`` is ``R_SW``.  Gravity is subtracted from
    inertial point acceleration before the frame rotation, so an aligned
    stationary sensor reports ``[0, 0, +g0]`` and a freely falling sensor
    reports zero.
    """

    gravity = _finite_vector("gravity_world_m_s2", gravity_world_m_s2, 3)
    rotation = _finite_matrix("rotation_world_to_sensor", rotation_world_to_sensor, (3, 3))
    point_acceleration = rigid_body_point_acceleration(
        origin_acceleration_world_m_s2,
        angular_acceleration_world_rad_s2,
        angular_velocity_world_rad_s,
        lever_arm_world_m,
    )
    return rotation.dot(point_acceleration - gravity)


def gyro_from_rigid_body_motion(
    angular_velocity_world_rad_s: Iterable[float],
    rotation_world_to_sensor: Iterable[Iterable[float]] = np.eye(3),
) -> np.ndarray:
    """Return spatial angular velocity in sensor coordinates."""

    omega = _finite_vector("angular_velocity_world_rad_s", angular_velocity_world_rad_s, 3)
    rotation = _finite_matrix("rotation_world_to_sensor", rotation_world_to_sensor, (3, 3))
    return rotation.dot(omega)


def imu_measurement_from_rigid_body_motion(
    origin_acceleration_world_m_s2: Iterable[float],
    angular_acceleration_world_rad_s2: Iterable[float],
    angular_velocity_world_rad_s: Iterable[float],
    lever_arm_world_m: Iterable[float] = (0.0, 0.0, 0.0),
    gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
    rotation_world_to_sensor: Iterable[Iterable[float]] = np.eye(3),
) -> np.ndarray:
    """Return the six clean channels in canonical accelerometer/gyro order."""

    force = specific_force_from_rigid_body_motion(
        origin_acceleration_world_m_s2,
        angular_acceleration_world_rad_s2,
        angular_velocity_world_rad_s,
        lever_arm_world_m,
        gravity_world_m_s2,
        rotation_world_to_sensor,
    )
    gyro = gyro_from_rigid_body_motion(angular_velocity_world_rad_s, rotation_world_to_sensor)
    return np.concatenate((force, gyro))


# Short aliases are useful in physics-only tests and keep the equation seam
# discoverable without making callers depend on one spelling.
compute_specific_force = specific_force_from_rigid_body_motion
compute_gyro_measurement = gyro_from_rigid_body_motion
compute_imu_measurement = imu_measurement_from_rigid_body_motion
specific_force = specific_force_from_rigid_body_motion
spatial_angular_velocity = gyro_from_rigid_body_motion
rigid_point_acceleration = rigid_body_point_acceleration


@dataclass(frozen=True)
class CanonicalIMUProfile:
    """Frozen numeric and semantic parameters for the canonical IMU."""

    profile_id: str = IMU_PROFILE_ID
    sample_rate_hz: float = IMU_SAMPLE_RATE_HZ
    policy_rate_hz: float = IMU_POLICY_RATE_HZ
    window_samples: int = IMU_WINDOW_SAMPLES
    delivery_delay_samples: int = IMU_DELIVERY_DELAY_SAMPLES
    filter_order: int = BUTTERWORTH_ORDER
    cutoff_3db_hz: float = BUTTERWORTH_CUTOFF_HZ
    enbw_hz: float = BUTTERWORTH_ENBW_HZ
    filter_b: tuple[float, float, float] = tuple(float(x) for x in BUTTERWORTH_B)
    filter_a: tuple[float, float, float] = tuple(float(x) for x in BUTTERWORTH_A)
    accel_range_m_s2: float = ACCEL_RANGE_M_S2
    accel_bits: int = IMU_BITS
    accel_noise_density_m_s2_sqrt_hz: float = ACCEL_NOISE_DENSITY_M_S2_SQRT_HZ
    accel_initial_bias_std_m_s2: float = ACCEL_INITIAL_BIAS_STD_M_S2
    accel_bias_diffusion_m_s2_sqrt_s: float = ACCEL_BIAS_DIFFUSION_M_S2_SQRT_S
    gyro_range_rad_s: float = GYRO_RANGE_RAD_S
    gyro_bits: int = IMU_BITS
    gyro_noise_density_rad_s_sqrt_hz: float = GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ
    gyro_initial_bias_std_rad_s: float = GYRO_INITIAL_BIAS_STD_RAD_S
    gyro_bias_diffusion_rad_s_sqrt_s: float = GYRO_BIAS_DIFFUSION_RAD_S_SQRT_S
    g0_m_s2: float = G0_M_S2

    def __post_init__(self) -> None:
        if self.profile_id != IMU_PROFILE_ID:
            raise ShakeBenchSensorError(f"profile_id must be {IMU_PROFILE_ID!r}")
        for name in ("sample_rate_hz", "policy_rate_hz", "cutoff_3db_hz", "enbw_hz", "g0_m_s2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ShakeBenchSensorError(f"{name} must be finite and positive")
        if self.sample_rate_hz != IMU_SAMPLE_RATE_HZ or self.policy_rate_hz != IMU_POLICY_RATE_HZ:
            raise ShakeBenchSensorError("the canonical IMU rates are fixed at 200 Hz and 20 Hz")
        if int(self.window_samples) != IMU_WINDOW_SAMPLES:
            raise ShakeBenchSensorError("the canonical IMU policy window is fixed at 10 samples")
        if int(self.delivery_delay_samples) != IMU_DELIVERY_DELAY_SAMPLES:
            raise ShakeBenchSensorError("the canonical IMU delivery delay is fixed at one sample")
        if int(self.filter_order) != BUTTERWORTH_ORDER:
            raise ShakeBenchSensorError("the canonical IMU filter order is fixed at two")
        try:
            filter_b = tuple(float(value) for value in self.filter_b)
            filter_a = tuple(float(value) for value in self.filter_a)
        except (TypeError, ValueError) as exc:
            raise ShakeBenchSensorError("filter coefficients must be finite") from exc
        if len(filter_b) != 3 or len(filter_a) != 3 or filter_a[0] != 1.0:
            raise ShakeBenchSensorError("the canonical filter must be a normalized second-order IIR")
        if not np.all(np.isfinite(filter_b)) or not np.all(np.isfinite(filter_a)):
            raise ShakeBenchSensorError("filter coefficients must be finite")
        object.__setattr__(self, "filter_b", filter_b)
        object.__setattr__(self, "filter_a", filter_a)
        for name in (
            "accel_range_m_s2",
            "accel_noise_density_m_s2_sqrt_hz",
            "accel_initial_bias_std_m_s2",
            "accel_bias_diffusion_m_s2_sqrt_s",
            "gyro_range_rad_s",
            "gyro_noise_density_rad_s_sqrt_hz",
            "gyro_initial_bias_std_rad_s",
            "gyro_bias_diffusion_rad_s_sqrt_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or (value <= 0.0 if "range" in name else value < 0.0):
                requirement = "positive" if "range" in name else "non-negative"
                raise ShakeBenchSensorError(f"{name} must be finite and {requirement}")
        if int(self.accel_bits) != IMU_BITS or int(self.gyro_bits) != IMU_BITS:
            raise ShakeBenchSensorError("the canonical accelerometer and gyro are fixed at 16 bit")

    @property
    def dt_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def window_shape(self) -> tuple[int, int]:
        return self.window_samples, len(IMU_CHANNELS)

    @property
    def accel_quantization_step_m_s2(self) -> float:
        return 2.0 * self.accel_range_m_s2 / (2**self.accel_bits)

    @property
    def gyro_quantization_step_rad_s(self) -> float:
        return 2.0 * self.gyro_range_rad_s / (2**self.gyro_bits)

    @property
    def accel_code_min(self) -> int:
        return -(2 ** (self.accel_bits - 1))

    @property
    def accel_code_max(self) -> int:
        return 2 ** (self.accel_bits - 1) - 1

    @property
    def gyro_code_min(self) -> int:
        return -(2 ** (self.gyro_bits - 1))

    @property
    def gyro_code_max(self) -> int:
        return 2 ** (self.gyro_bits - 1) - 1

    @property
    def quantization_steps(self) -> np.ndarray:
        return np.asarray(
            (self.accel_quantization_step_m_s2,) * 3 + (self.gyro_quantization_step_rad_s,) * 3,
            dtype=float,
        )

    @property
    def noise_density(self) -> np.ndarray:
        return np.asarray(
            (self.accel_noise_density_m_s2_sqrt_hz,) * 3 + (self.gyro_noise_density_rad_s_sqrt_hz,) * 3,
            dtype=float,
        )

    @property
    def ranges(self) -> np.ndarray:
        return np.asarray((self.accel_range_m_s2,) * 3 + (self.gyro_range_rad_s,) * 3, dtype=float)

    @property
    def bias_initial_std(self) -> np.ndarray:
        return np.asarray(
            (self.accel_initial_bias_std_m_s2,) * 3 + (self.gyro_initial_bias_std_rad_s,) * 3,
            dtype=float,
        )

    @property
    def bias_diffusion(self) -> np.ndarray:
        return np.asarray(
            (self.accel_bias_diffusion_m_s2_sqrt_s,) * 3 + (self.gyro_bias_diffusion_rad_s_sqrt_s,) * 3,
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "frame": {
                "parent": "robot_base",
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "output": {
                "sample_rate_hz": self.sample_rate_hz,
                "policy_rate_hz": self.policy_rate_hz,
                "window_samples": self.window_samples,
                "window_order": "oldest_to_newest",
                "channels": list(IMU_CHANNELS),
                "units": ["m/s2"] * 3 + ["rad/s"] * 3,
                "delivery_delay_samples": self.delivery_delay_samples,
                "timestamp_semantics": "acquisition_time",
            },
            "lowpass": {
                "type": "butterworth",
                "order": self.filter_order,
                "cutoff_3db_hz": self.cutoff_3db_hz,
                "enbw_hz": self.enbw_hz,
                "b": list(self.filter_b),
                "a": list(self.filter_a),
            },
            "accelerometer": {
                "range_m_s2": self.accel_range_m_s2,
                "bits": self.accel_bits,
                "code_min": self.accel_code_min,
                "code_max": self.accel_code_max,
                "lsb_m_s2": self.accel_quantization_step_m_s2,
                "noise_density_m_s2_sqrt_hz": self.accel_noise_density_m_s2_sqrt_hz,
                "initial_residual_bias_std_m_s2": self.accel_initial_bias_std_m_s2,
                "bias_diffusion_m_s2_sqrt_s": self.accel_bias_diffusion_m_s2_sqrt_s,
            },
            "gyroscope": {
                "range_rad_s": self.gyro_range_rad_s,
                "bits": self.gyro_bits,
                "code_min": self.gyro_code_min,
                "code_max": self.gyro_code_max,
                "lsb_rad_s": self.gyro_quantization_step_rad_s,
                "noise_density_rad_s_sqrt_hz": self.gyro_noise_density_rad_s_sqrt_hz,
                "initial_residual_bias_std_rad_s": self.gyro_initial_bias_std_rad_s,
                "bias_diffusion_rad_s_sqrt_s": self.gyro_bias_diffusion_rad_s_sqrt_s,
            },
            "omitted_effects": [
                "scale_factor_error",
                "axis_misalignment",
                "temperature_dependence",
                "vibration_rectification_error",
                "packet_loss",
                "timing_jitter",
            ],
        }


CANONICAL_IMU_PROFILE = CanonicalIMUProfile()


def canonical_imu_profile_hash(profile: CanonicalIMUProfile = CANONICAL_IMU_PROFILE) -> str:
    """Return the stable SHA-256 of a serialized IMU profile."""

    if not isinstance(profile, CanonicalIMUProfile):
        raise ShakeBenchSensorError("profile must be a CanonicalIMUProfile")
    import hashlib
    import json

    encoded = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CANONICAL_IMU_PROFILE_HASH = canonical_imu_profile_hash()


class ButterworthLowpass:
    """Causal second-order IIR filter with the frozen canonical coefficients."""

    def __init__(
        self,
        channels: int = len(IMU_CHANNELS),
        *,
        b: Iterable[float] = BUTTERWORTH_B,
        a: Iterable[float] = BUTTERWORTH_A,
    ) -> None:
        if isinstance(channels, (bool, np.bool_)) or int(channels) != channels or int(channels) <= 0:
            raise ShakeBenchSensorError("channels must be a positive integer")
        self.channels = int(channels)
        self.b = _finite_vector("filter b", b, 3)
        self.a = _finite_vector("filter a", a, 3)
        if self.a[0] == 0.0:
            raise ShakeBenchSensorError("filter a[0] must be non-zero")
        self.b /= self.a[0]
        self.a /= self.a[0]
        self.x1 = np.zeros(self.channels, dtype=float)
        self.x2 = np.zeros(self.channels, dtype=float)
        self.y1 = np.zeros(self.channels, dtype=float)
        self.y2 = np.zeros(self.channels, dtype=float)

    def reset(self, steady_state: Optional[Iterable[float]] = None) -> None:
        if steady_state is None:
            value = np.zeros(self.channels, dtype=float)
        else:
            value = _finite_vector("steady_state", steady_state, self.channels)
        # For a DC input, setting both input and output histories to the DC
        # value avoids synthetic zero frames at episode start.
        self.x1 = value.copy()
        self.x2 = value.copy()
        self.y1 = value.copy()
        self.y2 = value.copy()

    def step(self, value: Iterable[float]) -> np.ndarray:
        current = _finite_vector("filter input", value, self.channels)
        output = self.b[0] * current + self.b[1] * self.x1 + self.b[2] * self.x2
        output -= self.a[1] * self.y1 + self.a[2] * self.y2
        self.x2 = self.x1.copy()
        self.x1 = current.copy()
        self.y2 = self.y1.copy()
        self.y1 = output.copy()
        return output

    def filter(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        samples = np.asarray(values, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ShakeBenchSensorError(f"filter input must have shape (n, {self.channels})")
        return np.asarray([self.step(value) for value in samples], dtype=float)

    filter_samples = filter

    def state(self) -> np.ndarray:
        result = np.concatenate((self.x1, self.x2, self.y1, self.y2))
        result.setflags(write=False)
        return result

    def frequency_response(self, frequency_hz: Any, sample_rate_hz: float = IMU_SAMPLE_RATE_HZ) -> np.ndarray:
        frequencies = np.asarray(frequency_hz, dtype=float)
        if (
            not np.all(np.isfinite(frequencies))
            or np.any(frequencies < 0.0)
            or np.any(frequencies > sample_rate_hz / 2.0)
        ):
            raise ShakeBenchSensorError("frequency_hz must lie in the finite Nyquist interval")
        z_inverse = np.exp(-2j * pi * frequencies / sample_rate_hz)
        numerator = self.b[0] + self.b[1] * z_inverse + self.b[2] * z_inverse**2
        denominator = self.a[0] + self.a[1] * z_inverse + self.a[2] * z_inverse**2
        return numerator / denominator

    def enbw_hz(self, sample_rate_hz: float = IMU_SAMPLE_RATE_HZ, points: int = 200001) -> float:
        if sample_rate_hz != IMU_SAMPLE_RATE_HZ and (not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0):
            raise ShakeBenchSensorError("sample_rate_hz must be finite and positive")
        if (
            sample_rate_hz == IMU_SAMPLE_RATE_HZ
            and np.allclose(self.b, BUTTERWORTH_B)
            and np.allclose(self.a, BUTTERWORTH_A)
        ):
            return BUTTERWORTH_ENBW_HZ
        frequencies = np.linspace(0.0, sample_rate_hz / 2.0, int(points))
        response = self.frequency_response(frequencies, sample_rate_hz=sample_rate_hz)
        dc = abs(response[0])
        return float(np.trapz(np.abs(response / dc) ** 2, frequencies))

    response = frequency_response
    equivalent_noise_bandwidth_hz = enbw_hz


@dataclass(frozen=True)
class IMUSample:
    """One acquisition plus its privileged signal decomposition."""

    acquisition_time_s: float
    delivery_time_s: float
    delivered_acquisition_time_s: float
    clean_measurement: np.ndarray
    bias: np.ndarray
    bias_increment: np.ndarray
    white_noise: np.ndarray
    prefilter_measurement: np.ndarray
    filtered_measurement: np.ndarray
    clipped_measurement: np.ndarray
    quantized_codes: np.ndarray
    quantized_measurement: np.ndarray
    clipping: np.ndarray
    delivered_measurement: np.ndarray
    filter_state: np.ndarray
    quantizer_saturation: np.ndarray = field(default_factory=lambda: np.zeros(len(IMU_CHANNELS), dtype=bool))

    def __post_init__(self) -> None:
        for name in (
            "clean_measurement",
            "bias",
            "bias_increment",
            "white_noise",
            "prefilter_measurement",
            "filtered_measurement",
            "clipped_measurement",
            "quantized_measurement",
            "delivered_measurement",
        ):
            value = _finite_vector(name, getattr(self, name), len(IMU_CHANNELS))
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        raw_codes = np.asarray(self.quantized_codes)
        if raw_codes.shape != (len(IMU_CHANNELS),) or not np.issubdtype(raw_codes.dtype, np.number):
            raise ShakeBenchSensorError("quantized_codes must have six channels")
        if not np.all(np.isfinite(raw_codes)) or not np.all(raw_codes == np.rint(raw_codes)):
            raise ShakeBenchSensorError("quantized_codes must be integral")
        codes = np.asarray(raw_codes, dtype=np.int64)
        if np.any(codes < -(2 ** (IMU_BITS - 1))) or np.any(codes > 2 ** (IMU_BITS - 1) - 1):
            raise ShakeBenchSensorError("quantized_codes must fit signed 16-bit range")
        codes = np.array(codes, dtype=np.int16, copy=True)
        codes.setflags(write=False)
        object.__setattr__(self, "quantized_codes", codes)
        clipping = np.asarray(self.clipping, dtype=bool)
        if clipping.shape != (len(IMU_CHANNELS),):
            raise ShakeBenchSensorError("clipping must have six channels")
        clipping = np.array(clipping, copy=True)
        clipping.setflags(write=False)
        object.__setattr__(self, "clipping", clipping)
        state = np.asarray(self.filter_state, dtype=float)
        if state.shape != (4 * len(IMU_CHANNELS),) or not np.all(np.isfinite(state)):
            raise ShakeBenchSensorError("filter_state must have 24 finite values")
        state = np.array(state, copy=True)
        state.setflags(write=False)
        object.__setattr__(self, "filter_state", state)
        saturation = np.asarray(self.quantizer_saturation, dtype=bool)
        if saturation.shape != (len(IMU_CHANNELS),):
            raise ShakeBenchSensorError("quantizer_saturation must have six channels")
        saturation = np.array(saturation, copy=True)
        saturation.setflags(write=False)
        object.__setattr__(self, "quantizer_saturation", saturation)
        for name in ("acquisition_time_s", "delivery_time_s", "delivered_acquisition_time_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ShakeBenchSensorError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def measurement(self) -> np.ndarray:
        """The value delivered by the one-sample queue."""

        return self.delivered_measurement

    @property
    def clipping_flags(self) -> np.ndarray:
        return self.clipping.copy()

    @property
    def quantizer_saturation_flags(self) -> np.ndarray:
        return self.quantizer_saturation.copy()

    @property
    def quantized(self) -> np.ndarray:
        return self.quantized_measurement.copy()

    @property
    def clean(self) -> np.ndarray:
        return self.clean_measurement.copy()

    @property
    def noise(self) -> np.ndarray:
        return self.white_noise.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "acquisition_time_s": self.acquisition_time_s,
            "delivery_time_s": self.delivery_time_s,
            "delivered_acquisition_time_s": self.delivered_acquisition_time_s,
            "clean_measurement": self.clean_measurement.copy(),
            "bias": self.bias.copy(),
            "bias_increment": self.bias_increment.copy(),
            "white_noise": self.white_noise.copy(),
            "prefilter_measurement": self.prefilter_measurement.copy(),
            "filtered_measurement": self.filtered_measurement.copy(),
            "clipped_measurement": self.clipped_measurement.copy(),
            "quantized_codes": self.quantized_codes.copy(),
            "quantized_measurement": self.quantized_measurement.copy(),
            "clipping": self.clipping.copy(),
            "quantizer_saturation": self.quantizer_saturation.copy(),
            "delivered_measurement": self.delivered_measurement.copy(),
            "filter_state": self.filter_state.copy(),
        }


@dataclass(frozen=True)
class _DeliveryItem:
    acquisition_time_s: float
    measurement: np.ndarray

    def __post_init__(self) -> None:
        timestamp = float(self.acquisition_time_s)
        if not np.isfinite(timestamp):
            raise ShakeBenchSensorError("delivery acquisition timestamp must be finite")
        value = _finite_vector("delivery measurement", self.measurement, len(IMU_CHANNELS))
        value.setflags(write=False)
        object.__setattr__(self, "acquisition_time_s", timestamp)
        object.__setattr__(self, "measurement", value)


class _TraceView(dict):
    """Mapping that also supports the historical ``trace()`` spelling."""

    def __call__(self):
        return self


def _raw_model_data(sim: Any, data: Any = None) -> tuple[Any, Any]:
    model = getattr(sim, "model", sim)
    if data is None:
        data = getattr(sim, "data", None)
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    if raw_model is None or raw_data is None:
        raise ShakeBenchSensorError("a live simulation with model and data is required")
    return raw_model, raw_data


def _body_spatial_kinematics(
    sim: Any,
    body_name: str,
    gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
    data: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``position, R_WB, [v, omega], [a, alpha]`` for a body."""

    model, raw_data = _raw_model_data(sim, data=data)
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ShakeBenchSensorError(f"compiled model is missing body {body_name!r}")
    position = np.asarray(raw_data.xpos[body_id], dtype=float).copy()
    rotation = np.asarray(raw_data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    velocity_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, raw_data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity_rotlin, 0)
    twist = np.concatenate((velocity_rotlin[3:], velocity_rotlin[:3]))
    mujoco.mj_rnePostConstraint(model, raw_data)
    acceleration_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectAcceleration(model, raw_data, mujoco.mjtObj.mjOBJ_BODY, body_id, acceleration_rotlin, 0)
    acceleration = np.concatenate((acceleration_rotlin[3:], acceleration_rotlin[:3]))
    # MuJoCo's object-acceleration API reports the linear acceleration in its
    # gravity-inclusive convention.  Convert it back to inertial acceleration
    # before applying the specific-force equation.  With world gravity
    # ``[0, 0, -g0]`` this changes a resting body's reported ``+g0`` back to
    # zero and leaves a freely falling body's acceleration at ``-g0``.
    acceleration[:3] += _finite_vector("gravity_world_m_s2", gravity_world_m_s2, 3)
    if not np.all(np.isfinite(np.concatenate((position, rotation.reshape(-1), twist, acceleration)))):
        raise ShakeBenchSensorError("MuJoCo body kinematics contain non-finite values")
    return position, rotation, twist, acceleration


def clean_imu_measurement_from_sim(
    sim: Any,
    body_name: str = "deck",
    *,
    data: Any = None,
    sensor_position_body_m: Iterable[float] = (0.0, 0.0, 0.0),
    sensor_quat_body_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
    gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build clean IMU channels from a realized MuJoCo rigid-body state.

    ``sensor_quat_body_wxyz`` describes the sensor frame orientation in the
    body frame (sensor-to-body); the returned ``R_SW`` therefore maps world
    vectors into sensor coordinates.
    """

    gravity = _finite_vector("gravity_world_m_s2", gravity_world_m_s2, 3)
    position, body_rotation, twist, acceleration = _body_spatial_kinematics(sim, body_name, gravity, data=data)
    sensor_position = _finite_vector("sensor_position_body_m", sensor_position_body_m, 3)
    sensor_quat = _normalise_quaternion_wxyz(sensor_quat_body_wxyz, "sensor_quat_body_wxyz")
    sensor_to_body = _quat_wxyz_to_matrix(sensor_quat)
    sensor_from_body = sensor_to_body.T
    rotation_world_to_sensor = sensor_from_body.dot(body_rotation.T)
    lever_world = body_rotation.dot(sensor_position)
    point_acceleration = rigid_body_point_acceleration(acceleration[:3], acceleration[3:], twist[3:], lever_world)
    clean_force = rotation_world_to_sensor.dot(point_acceleration - gravity)
    clean_gyro = rotation_world_to_sensor.dot(twist[3:])
    clean = np.concatenate((clean_force, clean_gyro))
    return clean, {
        "body_position_world_m": position,
        "body_rotation_world": body_rotation,
        "origin_acceleration_world_m_s2": acceleration[:3].copy(),
        "angular_acceleration_world_rad_s2": acceleration[3:].copy(),
        "angular_velocity_world_rad_s": twist[3:].copy(),
        "lever_arm_world_m": lever_world,
        "point_acceleration_world_m_s2": point_acceleration,
        "rotation_world_to_sensor": rotation_world_to_sensor,
        "specific_force_sensor_m_s2": clean_force.copy(),
        "angular_velocity_sensor_rad_s": clean_gyro.copy(),
    }


class CanonicalIMU:
    """Seeded canonical IMU acquisition, filtering, delay, and window model."""

    VALID_MODES = ("ideal_smoke", "canonical_noisy_v1")

    def __init__(
        self,
        seed: int = 0,
        *,
        mode: str = "canonical_noisy_v1",
        profile: CanonicalIMUProfile = CANONICAL_IMU_PROFILE,
    ) -> None:
        if not isinstance(profile, CanonicalIMUProfile):
            raise ShakeBenchSensorError("profile must be a CanonicalIMUProfile")
        if mode not in self.VALID_MODES:
            raise ShakeBenchSensorError("mode must be ideal_smoke or canonical_noisy_v1")
        if isinstance(seed, (bool, np.bool_)):
            raise ShakeBenchSensorError("seed must be a non-negative integer")
        try:
            seed_int = int(seed)
        except (TypeError, ValueError) as exc:
            raise ShakeBenchSensorError("seed must be a non-negative integer") from exc
        if seed_int != seed or seed_int < 0:
            raise ShakeBenchSensorError("seed must be a non-negative integer")
        self.profile = profile
        self.mode = mode
        self.seed = seed_int
        self.rng = np.random.default_rng(seed_int)
        self.filter = ButterworthLowpass(b=profile.filter_b, a=profile.filter_a)
        self._window: deque[_DeliveryItem] = deque(maxlen=profile.window_samples)
        self._delivery_queue: deque[_DeliveryItem] = deque()
        self._prefill_item = _DeliveryItem(0.0, np.zeros(6, dtype=float))
        self._records: list[IMUSample] = []
        self._last_sample: Optional[IMUSample] = None
        self._last_timestamp_s: Optional[float] = None
        self._episode_start_time_s = 0.0
        self._bias = np.zeros(6, dtype=float)
        self._initial_bias = np.zeros(6, dtype=float)
        self._last_kinematics: dict[str, np.ndarray] = {}
        self.reset()

    @property
    def is_noisy(self) -> bool:
        return self.mode == "canonical_noisy_v1"

    @property
    def acquisition_rate_hz(self) -> float:
        return self.profile.sample_rate_hz

    @property
    def sample_rate_hz(self) -> float:
        return self.profile.sample_rate_hz

    @property
    def dt_s(self) -> float:
        return self.profile.dt_s

    @property
    def delivery_delay_samples(self) -> int:
        return self.profile.delivery_delay_samples

    @property
    def initial_bias(self) -> np.ndarray:
        return self._initial_bias.copy()

    @property
    def bias(self) -> np.ndarray:
        return self._bias.copy()

    @property
    def last_sample(self) -> Optional[IMUSample]:
        return self._last_sample

    @property
    def last_kinematics(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._last_kinematics.items()}

    def reset(
        self,
        *,
        initial_clean_measurement: Optional[Iterable[float]] = None,
        initial_measurement: Optional[Iterable[float]] = None,
        timestamp_s: float = 0.0,
    ) -> None:
        if not np.isfinite(float(timestamp_s)):
            raise ShakeBenchSensorError("timestamp_s must be finite")
        if initial_clean_measurement is not None and initial_measurement is not None:
            raise ShakeBenchSensorError("provide only one of initial_clean_measurement and initial_measurement")
        if initial_clean_measurement is None:
            initial_clean_measurement = initial_measurement
        # A sensor seed identifies an episode replay stream.  Resetting the
        # same sensor therefore restarts the exact bias/noise sequence instead
        # of consuming a different continuation from the previous episode.
        self.rng = np.random.default_rng(self.seed)
        initial_clean = (
            np.asarray((0.0, 0.0, G0_M_S2, 0.0, 0.0, 0.0), dtype=float)
            if initial_clean_measurement is None
            else _finite_vector("initial_clean_measurement", initial_clean_measurement, len(IMU_CHANNELS))
        )
        if self.is_noisy:
            self._initial_bias = self.rng.normal(0.0, self.profile.bias_initial_std, size=6)
        else:
            self._initial_bias = np.zeros(6, dtype=float)
        self._bias = self._initial_bias.copy()
        prefill_signal = initial_clean + self._bias
        self.filter.reset(prefill_signal)
        if self.is_noisy:
            prefill_signal = np.clip(prefill_signal, -self.profile.ranges, self.profile.ranges)
            prefill_codes = np.rint(prefill_signal / self.profile.quantization_steps).astype(np.int64)
            prefill_codes[:3] = np.clip(prefill_codes[:3], self.profile.accel_code_min, self.profile.accel_code_max)
            prefill_codes[3:] = np.clip(prefill_codes[3:], self.profile.gyro_code_min, self.profile.gyro_code_max)
            prefill_signal = prefill_codes * self.profile.quantization_steps
        self._window.clear()
        self._delivery_queue.clear()
        self._records = []
        self._last_sample = None
        self._last_timestamp_s = None
        self._episode_start_time_s = float(timestamp_s)
        self._last_kinematics = {}
        # The reset window is a real static acquisition history ending at
        # t=-dt.  The queue then contains the static acquisition at t=0, so
        # the first live acquisition at t=dt delivers t=0 and no 10 ms gap is
        # manufactured at the policy boundary.
        prefill_time = float(timestamp_s) - self.profile.window_samples * self.profile.dt_s
        self._prefill_item = _DeliveryItem(float(timestamp_s), np.array(prefill_signal, dtype=float, copy=True))
        for index in range(self.profile.window_samples):
            self._window.append(
                _DeliveryItem(
                    prefill_time + index * self.profile.dt_s,
                    np.array(prefill_signal, dtype=float, copy=True),
                )
            )
        for index in range(self.profile.delivery_delay_samples):
            self._delivery_queue.append(
                _DeliveryItem(
                    float(timestamp_s),
                    np.array(prefill_signal, dtype=float, copy=True),
                )
            )

    def _quantize(self, filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ranges = self.profile.ranges
        clipped = np.clip(filtered, -ranges, ranges)
        clipping = (filtered < -ranges) | (filtered > ranges)
        if not self.is_noisy:
            return clipped, np.zeros(6, dtype=np.int64), clipped.copy(), clipping, np.zeros(6, dtype=bool)
        steps = self.profile.quantization_steps
        codes = np.rint(clipped / steps).astype(np.int64)
        quantizer_saturation = np.zeros(6, dtype=bool)
        quantizer_saturation[:3] = (codes[:3] < self.profile.accel_code_min) | (codes[:3] > self.profile.accel_code_max)
        quantizer_saturation[3:] = (codes[3:] < self.profile.gyro_code_min) | (codes[3:] > self.profile.gyro_code_max)
        codes[:3] = np.clip(codes[:3], self.profile.accel_code_min, self.profile.accel_code_max)
        codes[3:] = np.clip(codes[3:], self.profile.gyro_code_min, self.profile.gyro_code_max)
        quantized = codes * steps
        return clipped, codes.astype(np.int16), quantized, clipping, quantizer_saturation

    def quantize_measurement(self, filtered_measurement: Iterable[float]) -> dict[str, np.ndarray]:
        """Apply the canonical clip-and-round stage to a filtered reading."""

        filtered = _finite_vector("filtered_measurement", filtered_measurement, len(IMU_CHANNELS))
        clipped, codes, quantized, clipping, quantizer_saturation = self._quantize(filtered)
        return {
            "clipped_measurement": clipped,
            "quantized_codes": codes,
            "quantized_measurement": quantized,
            "clipping": clipping,
            "quantizer_saturation": quantizer_saturation,
        }

    def acquire(self, clean_measurement: Iterable[float], timestamp_s: Optional[float] = None) -> IMUSample:
        clean = _finite_vector("clean_measurement", clean_measurement, len(IMU_CHANNELS))
        if timestamp_s is None:
            timestamp = (
                self._episode_start_time_s + self.profile.dt_s
                if self._last_timestamp_s is None
                else self._last_timestamp_s + self.profile.dt_s
            )
        else:
            timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ShakeBenchSensorError("acquisition timestamp must be finite")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ShakeBenchSensorError("acquisition timestamps must be strictly increasing")
        bias = self._bias.copy()
        if self.is_noisy:
            bias_increment = self.profile.bias_diffusion * np.sqrt(self.profile.dt_s) * self.rng.normal(size=6)
            white_noise = (
                self.profile.noise_density * np.sqrt(self.profile.sample_rate_hz / 2.0) * self.rng.normal(size=6)
            )
        else:
            bias_increment = np.zeros(6, dtype=float)
            white_noise = np.zeros(6, dtype=float)
        prefilter = clean + bias + white_noise
        filtered = self.filter.step(prefilter)
        clipped, codes, quantized, clipping, quantizer_saturation = self._quantize(filtered)
        self._bias = bias + bias_increment
        self._last_timestamp_s = timestamp
        item = _DeliveryItem(timestamp, quantized.copy())
        self._delivery_queue.append(item)
        if len(self._delivery_queue) > self.profile.delivery_delay_samples:
            delivered_item = self._delivery_queue.popleft()
        else:
            delivered_item = self._prefill_item
        self._window.append(_DeliveryItem(delivered_item.acquisition_time_s, delivered_item.measurement.copy()))
        sample = IMUSample(
            acquisition_time_s=timestamp,
            delivery_time_s=timestamp,
            delivered_acquisition_time_s=delivered_item.acquisition_time_s,
            clean_measurement=clean,
            bias=bias,
            bias_increment=bias_increment,
            white_noise=white_noise,
            prefilter_measurement=prefilter,
            filtered_measurement=filtered,
            clipped_measurement=clipped,
            quantized_codes=codes,
            quantized_measurement=quantized,
            clipping=clipping,
            delivered_measurement=delivered_item.measurement,
            filter_state=self.filter.state(),
            quantizer_saturation=quantizer_saturation,
        )
        self._last_sample = sample
        self._records.append(sample)
        return sample

    def acquire_from_sim(
        self,
        sim: Any,
        timestamp_s: Optional[float] = None,
        body_name: str = "deck",
        *,
        sensor_position_body_m: Iterable[float] = (0.0, 0.0, 0.0),
        sensor_quat_body_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
    ) -> IMUSample:
        clean, kinematics = clean_imu_measurement_from_sim(
            sim,
            body_name,
            sensor_position_body_m=sensor_position_body_m,
            sensor_quat_body_wxyz=sensor_quat_body_wxyz,
            gravity_world_m_s2=gravity_world_m_s2,
        )
        self._last_kinematics = kinematics
        return self.acquire(clean, timestamp_s=timestamp_s)

    sample = acquire
    process = acquire

    def window(self) -> np.ndarray:
        if len(self._window) != self.profile.window_samples:
            raise ShakeBenchSensorError("IMU window is not initialized")
        return np.asarray([item.measurement for item in self._window], dtype=np.float32)

    get_window = window

    @property
    def acquisition_timestamps_s(self) -> np.ndarray:
        return np.asarray([sample.acquisition_time_s for sample in self._records], dtype=float)

    @property
    def delivered_acquisition_timestamps_s(self) -> np.ndarray:
        return np.asarray([sample.delivered_acquisition_time_s for sample in self._records], dtype=float)

    @property
    def window_acquisition_timestamps_s(self) -> np.ndarray:
        return np.asarray([item.acquisition_time_s for item in self._window], dtype=float)

    @property
    def pending_delivery_acquisition_timestamps_s(self) -> np.ndarray:
        """Acquisition timestamps still waiting in the delivery queue."""

        return np.asarray([item.acquisition_time_s for item in self._delivery_queue], dtype=float)

    @property
    def records(self) -> tuple[IMUSample, ...]:
        return tuple(self._records)

    @property
    def trace(self) -> dict[str, np.ndarray]:
        fields = (
            "clean_measurement",
            "bias",
            "bias_increment",
            "white_noise",
            "prefilter_measurement",
            "filtered_measurement",
            "clipped_measurement",
            "quantized_codes",
            "quantized_measurement",
            "clipping",
            "quantizer_saturation",
            "delivered_measurement",
            "filter_state",
        )
        timestamps = {
            "acquisition_time_s": self.acquisition_timestamps_s,
            "delivery_time_s": np.asarray([sample.delivery_time_s for sample in self._records], dtype=float),
            "delivered_acquisition_time_s": self.delivered_acquisition_timestamps_s,
        }
        timestamps["acquisition_timestamps_s"] = timestamps["acquisition_time_s"].copy()
        timestamps["delivery_timestamps_s"] = timestamps["delivery_time_s"].copy()
        result = dict(timestamps)
        for field in fields:
            if field in {"clipping", "quantizer_saturation"}:
                value = np.asarray([getattr(sample, field) for sample in self._records], dtype=bool)
            elif field == "quantized_codes":
                value = np.asarray([getattr(sample, field) for sample in self._records], dtype=np.int16)
            else:
                value = np.asarray([getattr(sample, field) for sample in self._records])
            value = np.array(value, copy=True)
            value.setflags(write=False)
            result[field] = value
        for value in result.values():
            value.setflags(write=False)
        return _TraceView(result)

    def get_trace(self) -> dict[str, np.ndarray]:
        return self.trace

    def to_policy_observation(self) -> dict[str, np.ndarray]:
        return {
            "deck_imu_window": self.window(),
            "deck_imu_dt_s": np.asarray(self.profile.dt_s, dtype=np.float32),
        }


# Names used by callers that describe the physical installation rather than
# the profile implementation.  They intentionally refer to the same class.
CanonicalDeckIMU = CanonicalIMU
DeckIMU = CanonicalIMU
CanonicalIMUSensor = CanonicalIMU
DeckIMUSensor = CanonicalIMU
IMUSensor = CanonicalIMU
ButterworthFilter = ButterworthLowpass


__all__ = [
    "ACCEL_BIAS_DIFFUSION_M_S2_SQRT_S",
    "ACCEL_INITIAL_BIAS_STD_M_S2",
    "ACCEL_NOISE_DENSITY_M_S2_SQRT_HZ",
    "ACCEL_RANGE_M_S2",
    "BUTTERWORTH_A",
    "BUTTERWORTH_B",
    "BUTTERWORTH_CUTOFF_HZ",
    "BUTTERWORTH_ENBW_HZ",
    "CanonicalDeckIMU",
    "CanonicalIMUSensor",
    "CanonicalIMU",
    "CanonicalIMUProfile",
    "CANONICAL_IMU_PROFILE",
    "CANONICAL_IMU_PROFILE_HASH",
    "DeckIMU",
    "DeckIMUSensor",
    "IMUSensor",
    "G0_M_S2",
    "GRAVITY_WORLD_M_S2",
    "GYRO_BIAS_DIFFUSION_RAD_S_SQRT_S",
    "GYRO_INITIAL_BIAS_STD_RAD_S",
    "GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ",
    "GYRO_RANGE_RAD_S",
    "IMU_CHANNELS",
    "IMU_DELIVERY_DELAY_SAMPLES",
    "IMU_DT_S",
    "IMU_POLICY_RATE_HZ",
    "IMU_PROFILE_ID",
    "IMU_SAMPLE_RATE_HZ",
    "IMU_WINDOW_SAMPLES",
    "IMU_WINDOW_SHAPE",
    "IMUSample",
    "ButterworthLowpass",
    "ButterworthFilter",
    "ShakeBenchSensorError",
    "clean_imu_measurement_from_sim",
    "canonical_imu_profile_hash",
    "compute_gyro_measurement",
    "compute_imu_measurement",
    "compute_specific_force",
    "gyro_from_rigid_body_motion",
    "imu_measurement_from_rigid_body_motion",
    "rigid_body_point_acceleration",
    "rigid_point_acceleration",
    "specific_force",
    "specific_force_from_rigid_body_motion",
    "spatial_angular_velocity",
]
