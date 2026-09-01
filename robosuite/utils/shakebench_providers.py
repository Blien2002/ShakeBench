"""Policy-safe vibration information providers for ShakeBench State tiers.

The providers deliberately expose only the fields allowed at each tier.  They
do not return MuJoCo ``qpos`` / ``qvel`` arrays and they do not retain support
state history.  Runtime truth needed for evaluation is collected separately by
``shakebench_privilege.PrivilegedRecorder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import mujoco
import numpy as np

from robosuite.utils.shakebench_excitation import ExcitationProgram, MotionSample, build_excitation_program
from robosuite.utils.shakebench_sensors import (
    CANONICAL_IMU_PROFILE,
    CanonicalIMU,
    CanonicalIMUProfile,
    GRAVITY_WORLD_M_S2,
    ShakeBenchSensorError,
    _matrix_to_quaternion_wxyz,
    _normalise_quaternion_wxyz,
    _quat_wxyz_to_matrix,
)


OBSERVATION_TIERS = ("V0", "V1", "V2", "V3")
COMMON_STATE_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos_robot_base",
    "robot0_eef_quat_robot_base",
    "robot0_gripper_state",
    "robot0_wrist_force",
    "robot0_wrist_torque",
    "robot0_fingertip_pos_robot_base",
    "can_pos_robot_base",
    "can_quat_robot_base",
    "goal_center_robot_base",
    "goal_half_extents_robot_base",
    "goal_z_bounds_robot_base",
    "goal_orientation_mask",
)
V0_POLICY_KEYS = ()
V1_POLICY_KEYS = ("deck_imu_window", "deck_imu_dt_s")
V2_POLICY_KEYS = (
    "deck_pose_in_nominal_frame",
    "deck_twist_in_nominal_frame",
    "deck_accel_in_nominal_frame",
    "table_pose_in_deck_frame",
    "table_twist_in_deck_frame",
    "table_accel_in_deck_frame",
)
V3_POLICY_KEYS = (
    "line_accel_amplitude",
    "line_omega_rad_s",
    "line_phase_at_episode_zero",
    "line_mask",
    "episode_time_s",
    "ramp_type",
    "ramp_duration_s",
    "program_frame",
)
TIER_ADDED_KEYS = MappingProxyType(
    {
        "V0": V0_POLICY_KEYS,
        "V1": V1_POLICY_KEYS,
        "V2": V2_POLICY_KEYS,
        "V3": V3_POLICY_KEYS,
    }
)
TIER_POLICY_KEYS = MappingProxyType(
    {
        "V0": V0_POLICY_KEYS,
        "V1": V1_POLICY_KEYS,
        "V2": V1_POLICY_KEYS + V2_POLICY_KEYS,
        "V3": V1_POLICY_KEYS + V2_POLICY_KEYS + V3_POLICY_KEYS,
    }
)
POLICY_FIELD_CONTRACT = MappingProxyType(
    {
        "robot0_joint_pos": {"shape": (7,), "dtype": "float64", "units": "rad", "frame": "robot_base"},
        "robot0_joint_vel": {"shape": (7,), "dtype": "float64", "units": "rad/s", "frame": "robot_base"},
        "robot0_eef_pos_robot_base": {"shape": (3,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "robot0_eef_quat_robot_base": {"shape": (4,), "dtype": "float32", "units": "unitless", "frame": "robot_base"},
        "robot0_gripper_state": {"shape": (4,), "dtype": "float32", "units": "rad,rad/s", "frame": "robot_base"},
        "robot0_wrist_force": {"shape": (3,), "dtype": "float32", "units": "N", "frame": "wrist_sensor"},
        "robot0_wrist_torque": {"shape": (3,), "dtype": "float32", "units": "N*m", "frame": "wrist_sensor"},
        "robot0_fingertip_pos_robot_base": {"shape": (6,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "can_pos_robot_base": {"shape": (3,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "can_quat_robot_base": {"shape": (4,), "dtype": "float32", "units": "unitless", "frame": "robot_base"},
        "goal_center_robot_base": {"shape": (3,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "goal_half_extents_robot_base": {"shape": (2,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "goal_z_bounds_robot_base": {"shape": (2,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "goal_orientation_mask": {"shape": (3,), "dtype": "bool", "units": "unitless", "frame": "robot_base"},
        "deck_imu_window": {"shape": (10, 6), "dtype": "float32", "units": "m/s2,rad/s", "frame": "sensor"},
        "deck_imu_dt_s": {"shape": (), "dtype": "float32", "units": "s", "frame": "acquisition_time"},
        "deck_pose_in_nominal_frame": {"shape": (7,), "dtype": "float32", "units": "m,unitless", "frame": "nominal"},
        "deck_twist_in_nominal_frame": {"shape": (6,), "dtype": "float32", "units": "m/s,rad/s", "frame": "nominal"},
        "deck_accel_in_nominal_frame": {"shape": (6,), "dtype": "float32", "units": "m/s2,rad/s2", "frame": "nominal"},
        "table_pose_in_deck_frame": {"shape": (7,), "dtype": "float32", "units": "m,unitless", "frame": "deck"},
        "table_twist_in_deck_frame": {"shape": (6,), "dtype": "float32", "units": "m/s,rad/s", "frame": "deck"},
        "table_accel_in_deck_frame": {"shape": (6,), "dtype": "float32", "units": "m/s2,rad/s2", "frame": "deck"},
        "line_accel_amplitude": {"shape": (6, 12), "dtype": "float32", "units": "m/s2,rad/s2", "frame": "deck_program"},
        "line_omega_rad_s": {"shape": (6, 12), "dtype": "float32", "units": "rad/s", "frame": "deck_program"},
        "line_phase_at_episode_zero": {"shape": (6, 12), "dtype": "float32", "units": "rad", "frame": "deck_program"},
        "line_mask": {"shape": (6, 12), "dtype": "bool", "units": "unitless", "frame": "deck_program"},
        "episode_time_s": {"shape": (), "dtype": "float32", "units": "s", "frame": "episode"},
        "ramp_type": {"shape": (), "dtype": "str", "units": "label", "frame": "deck_program"},
        "ramp_duration_s": {"shape": (), "dtype": "float32", "units": "s", "frame": "deck_program"},
        "program_frame": {"shape": (), "dtype": "str", "units": "label", "frame": "deck_program"},
    }
)


class ShakeBenchProviderError(ValueError):
    """Raised when a vibration provider cannot satisfy its tier contract."""


def normalize_observation_tier(tier: Any) -> str:
    if not isinstance(tier, str) or tier not in OBSERVATION_TIERS:
        raise ShakeBenchProviderError("observation_tier must be exactly one of V0, V1, V2, V3")
    return tier


def policy_keys_for_tier(tier: str, *, include_common: bool = False) -> tuple[str, ...]:
    normalized = normalize_observation_tier(tier)
    vibration = TIER_POLICY_KEYS[normalized]
    return tuple(COMMON_STATE_KEYS) + vibration if include_common else tuple(vibration)


def observation_contract_for_tier(tier: str) -> dict[str, dict[str, Any]]:
    """Return a copy of the declared public shape/unit/frame contract."""

    keys = policy_keys_for_tier(tier, include_common=True)
    return {key: dict(POLICY_FIELD_CONTRACT[key]) for key in keys}


def _reconstruction_ramp(time_s: Any, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(time_s, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ShakeBenchProviderError("reconstruction query times must be finite")
    duration = float(duration_s)
    if not np.isfinite(duration) or duration < 0.0:
        raise ShakeBenchProviderError("ramp_duration_s must be finite and non-negative")
    if duration == 0.0:
        return np.ones_like(times), np.zeros_like(times), np.zeros_like(times)
    scaled = times / duration
    ramp = np.zeros_like(times)
    first = np.zeros_like(times)
    second = np.zeros_like(times)
    after = scaled >= 1.0
    middle = (scaled > 0.0) & (scaled < 1.0)
    ramp[after] = 1.0
    s = scaled[middle]
    ramp[middle] = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    first[middle] = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    second[middle] = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    return ramp, first, second


def reconstruct_authored_motion(payload: Mapping[str, Any], time_s: Any) -> MotionSample:
    """Rebuild V3 authored ``q/qdot/qdd`` from public payload fields only."""

    if not isinstance(payload, Mapping):
        raise ShakeBenchProviderError("V3 reconstruction payload must be a mapping")
    required = {
        "line_accel_amplitude",
        "line_omega_rad_s",
        "line_phase_at_episode_zero",
        "line_mask",
        "ramp_type",
        "ramp_duration_s",
        "program_frame",
    }
    missing = required - set(payload)
    if missing:
        raise ShakeBenchProviderError("V3 reconstruction is missing: " + ", ".join(sorted(missing)))
    amplitudes = np.asarray(payload["line_accel_amplitude"], dtype=float)
    omegas = np.asarray(payload["line_omega_rad_s"], dtype=float)
    phases = np.asarray(payload["line_phase_at_episode_zero"], dtype=float)
    mask = np.asarray(payload["line_mask"], dtype=bool)
    if amplitudes.shape != omegas.shape or amplitudes.shape != phases.shape or amplitudes.shape != mask.shape:
        raise ShakeBenchProviderError("V3 line arrays must have matching shape (6, max_lines)")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != 6:
        raise ShakeBenchProviderError("V3 line arrays must have shape (6, max_lines)")
    if not all(np.all(np.isfinite(value)) for value in (amplitudes, omegas, phases)):
        raise ShakeBenchProviderError("V3 line arrays must be finite")
    if np.any(omegas[mask] <= 0.0) or np.any(amplitudes[~mask] != 0.0):
        raise ShakeBenchProviderError("V3 line mask contains invalid active/inactive lines")
    ramp_type = str(np.asarray(payload["ramp_type"]).reshape(-1)[0])
    program_frame = str(np.asarray(payload["program_frame"]).reshape(-1)[0])
    if ramp_type != "quintic_smoothstep":
        raise ShakeBenchProviderError("unsupported V3 ramp_type")
    if program_frame != "deck":
        raise ShakeBenchProviderError("unsupported V3 program_frame")
    times = np.asarray(time_s, dtype=float)
    time_ndim = times.ndim
    expand = (slice(None), slice(None)) + (None,) * time_ndim
    omega = omegas[expand]
    phase = phases[expand]
    amplitude = amplitudes[expand]
    omega_safe = np.where(mask, omegas, 1.0)[expand]
    theta = omega * times + phase
    carrier_q = np.sum(-amplitude / omega_safe**2 * np.sin(theta), axis=1)
    carrier_qdot = np.sum(-amplitude / omega_safe * np.cos(theta), axis=1)
    carrier_qdd = np.sum(amplitude * np.sin(theta), axis=1)
    carrier = MotionSample(
        q=np.moveaxis(carrier_q, 0, -1),
        qdot=np.moveaxis(carrier_qdot, 0, -1),
        qdd=np.moveaxis(carrier_qdd, 0, -1),
    )
    ramp, ramp_first, ramp_second = _reconstruction_ramp(times, payload["ramp_duration_s"])
    return MotionSample(
        q=carrier.q * ramp[..., None],
        qdot=carrier.qdot * ramp[..., None] + carrier.q * ramp_first[..., None],
        qdd=(
            carrier.qdd * ramp[..., None]
            + 2.0 * carrier.qdot * ramp_first[..., None]
            + carrier.q * ramp_second[..., None]
        ),
    )


@dataclass(frozen=True)
class RigidBodyState:
    """Backend-neutral body state used by the V2 coordinate transform seam."""

    position_world_m: np.ndarray
    rotation_world: np.ndarray
    twist_world: np.ndarray
    acceleration_world: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_world_m, dtype=float)
        rotation = np.asarray(self.rotation_world, dtype=float)
        twist = np.asarray(self.twist_world, dtype=float)
        acceleration = np.asarray(self.acceleration_world, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3) or twist.shape != (6,) or acceleration.shape != (6,):
            raise ShakeBenchProviderError("RigidBodyState has invalid field shapes")
        if not all(np.all(np.isfinite(value)) for value in (position, rotation, twist, acceleration)):
            raise ShakeBenchProviderError("RigidBodyState fields must be finite")
        for name, value in (
            ("position_world_m", position),
            ("rotation_world", rotation),
            ("twist_world", twist),
            ("acceleration_world", acceleration),
        ):
            value = np.array(value, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RelativePoseTwistAcceleration:
    """A body's pose, twist, and acceleration expressed in a moving frame."""

    pose: np.ndarray
    twist: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=float)
        twist = np.asarray(self.twist, dtype=float)
        acceleration = np.asarray(self.acceleration, dtype=float)
        if pose.shape != (7,) or twist.shape != (6,) or acceleration.shape != (6,):
            raise ShakeBenchProviderError("relative state fields must have shapes (7,), (6,), and (6,)")
        if not all(np.all(np.isfinite(value)) for value in (pose, twist, acceleration)):
            raise ShakeBenchProviderError("relative state fields must be finite")
        pose = np.array(pose, dtype=float, copy=True)
        pose[3:] = _normalise_quaternion_wxyz(pose[3:])
        for name, value in (("pose", pose), ("twist", twist), ("acceleration", acceleration)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SupportState:
    """The exact six V2 public support-state fields."""

    deck_pose_in_nominal_frame: np.ndarray
    deck_twist_in_nominal_frame: np.ndarray
    deck_accel_in_nominal_frame: np.ndarray
    table_pose_in_deck_frame: np.ndarray
    table_twist_in_deck_frame: np.ndarray
    table_accel_in_deck_frame: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "deck_pose_in_nominal_frame",
            "table_pose_in_deck_frame",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                raise ShakeBenchProviderError(f"{name} must have shape (7,) and finite values")
            value = np.array(value, dtype=float, copy=True)
            value[3:] = _normalise_quaternion_wxyz(value[3:])
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in (
            "deck_twist_in_nominal_frame",
            "deck_accel_in_nominal_frame",
            "table_twist_in_deck_frame",
            "table_accel_in_deck_frame",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ShakeBenchProviderError(f"{name} must have shape (6,) and finite values")
            value = np.array(value, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "deck_pose_in_nominal_frame": self.deck_pose_in_nominal_frame.copy(),
            "deck_twist_in_nominal_frame": self.deck_twist_in_nominal_frame.copy(),
            "deck_accel_in_nominal_frame": self.deck_accel_in_nominal_frame.copy(),
            "table_pose_in_deck_frame": self.table_pose_in_deck_frame.copy(),
            "table_twist_in_deck_frame": self.table_twist_in_deck_frame.copy(),
            "table_accel_in_deck_frame": self.table_accel_in_deck_frame.copy(),
        }

    def to_policy_observation(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value, dtype=np.float32) for key, value in self.to_dict().items()}


def _relative_state(
    child: RigidBodyState,
    frame: RigidBodyState,
) -> RelativePoseTwistAcceleration:
    """Compute a moving-frame relative state without using joint coordinates."""

    frame_rotation = frame.rotation_world
    frame_rotation_transpose = frame_rotation.T
    relative_position_world = child.position_world_m - frame.position_world_m
    relative_position = frame_rotation_transpose.dot(relative_position_world)
    relative_rotation = frame_rotation_transpose.dot(child.rotation_world)

    frame_linear = frame.twist_world[:3]
    frame_angular = frame.twist_world[3:]
    child_linear = child.twist_world[:3]
    child_angular = child.twist_world[3:]
    frame_angular_in_frame = frame_rotation_transpose.dot(frame_angular)
    relative_linear_world_without_rotation = child_linear - frame_linear
    relative_linear = frame_rotation_transpose.dot(relative_linear_world_without_rotation)
    relative_linear -= np.cross(frame_angular_in_frame, relative_position)
    relative_angular = frame_rotation_transpose.dot(child_angular - frame_angular)

    frame_alpha = frame.acceleration_world[3:]
    child_alpha = child.acceleration_world[3:]
    frame_angular_cross_position = np.cross(frame_angular, relative_position_world)
    coordinate_velocity_world = child_linear - frame_linear - frame_angular_cross_position
    relative_linear_acceleration = child.acceleration_world[:3] - frame.acceleration_world[:3]
    relative_linear_acceleration -= np.cross(frame_alpha, relative_position_world)
    relative_linear_acceleration -= np.cross(frame_angular, np.cross(frame_angular, relative_position_world))
    relative_linear_acceleration -= 2.0 * np.cross(frame_angular, coordinate_velocity_world)
    relative_linear_acceleration = frame_rotation_transpose.dot(relative_linear_acceleration)
    relative_angular_acceleration = frame_rotation_transpose.dot(child_alpha - frame_alpha)
    relative_angular_acceleration -= np.cross(frame_angular_in_frame, relative_angular)

    return RelativePoseTwistAcceleration(
        pose=np.concatenate((relative_position, _matrix_to_quaternion_wxyz(relative_rotation))),
        twist=np.concatenate((relative_linear, relative_angular)),
        acceleration=np.concatenate((relative_linear_acceleration, relative_angular_acceleration)),
    )


relative_pose_twist_acceleration = _relative_state


def _raw_model_data(sim: Any, data: Any = None) -> tuple[Any, Any]:
    model = getattr(sim, "model", sim)
    if data is None:
        data = getattr(sim, "data", None)
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    if raw_model is None or raw_data is None:
        raise ShakeBenchProviderError("a live simulation with model and data is required")
    return raw_model, raw_data


def _body_state_from_sim(sim: Any, body_name: str, *, data: Any = None, assume_static: bool = False) -> RigidBodyState:
    model, raw_data = _raw_model_data(sim, data=data)
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ShakeBenchProviderError(f"compiled model is missing body {body_name!r}")
    position = np.asarray(raw_data.xpos[body_id], dtype=float).copy()
    rotation = np.asarray(raw_data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    velocity_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, raw_data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity_rotlin, 0)
    twist = np.concatenate((velocity_rotlin[3:], velocity_rotlin[:3]))
    mujoco.mj_rnePostConstraint(model, raw_data)
    acceleration_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectAcceleration(model, raw_data, mujoco.mjtObj.mjOBJ_BODY, body_id, acceleration_rotlin, 0)
    acceleration = np.concatenate((acceleration_rotlin[3:], acceleration_rotlin[:3]))
    # Normalize MuJoCo's gravity-inclusive linear object acceleration to the
    # inertial acceleration used by the V2 physical state contract.
    acceleration[:3] += GRAVITY_WORLD_M_S2
    if assume_static:
        acceleration = np.zeros(6, dtype=float)
    return RigidBodyState(position, rotation, twist, acceleration)


def support_state_from_sim(
    sim: Any,
    *,
    deck_body_name: str = "deck",
    table_body_name: str = "worktable",
    nominal_frame_position_m: Iterable[float] = (0.0, 0.0, 0.0),
    nominal_frame_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
    assume_static: bool = False,
    data: Any = None,
) -> SupportState:
    """Return current realized deck/table state in the V2 frames.

    The function accepts a robosuite simulator, a MuJoCo binding simulator,
    or a tiny object exposing ``model`` and ``data``.  It reads compiled body
    kinematics through MuJoCo's object API and never publishes raw joint
    coordinates.
    """

    deck = _body_state_from_sim(sim, deck_body_name, data=data, assume_static=assume_static)
    table = _body_state_from_sim(sim, table_body_name, data=data, assume_static=assume_static)
    nominal_position = np.asarray(nominal_frame_position_m, dtype=float)
    if nominal_position.shape != (3,) or not np.all(np.isfinite(nominal_position)):
        raise ShakeBenchProviderError("nominal_frame_position_m must have three finite values")
    nominal_rotation = _quat_wxyz_to_matrix(nominal_frame_quat_wxyz)
    world_to_nominal = nominal_rotation.T
    deck_pose = np.concatenate(
        (
            world_to_nominal.dot(deck.position_world_m - nominal_position),
            _matrix_to_quaternion_wxyz(world_to_nominal.dot(deck.rotation_world)),
        )
    )
    deck_twist = np.concatenate(
        (world_to_nominal.dot(deck.twist_world[:3]), world_to_nominal.dot(deck.twist_world[3:]))
    )
    deck_acceleration = np.concatenate(
        (world_to_nominal.dot(deck.acceleration_world[:3]), world_to_nominal.dot(deck.acceleration_world[3:]))
    )
    table_relative = _relative_state(table, deck)
    return SupportState(
        deck_pose_in_nominal_frame=deck_pose,
        deck_twist_in_nominal_frame=deck_twist,
        deck_accel_in_nominal_frame=deck_acceleration,
        table_pose_in_deck_frame=table_relative.pose,
        table_twist_in_deck_frame=table_relative.twist,
        table_accel_in_deck_frame=table_relative.acceleration,
    )


class VibrationProvider:
    """Base class for the additive vibration-information lane."""

    tier = "V0"

    @property
    def policy_keys(self) -> tuple[str, ...]:
        return tuple(TIER_POLICY_KEYS[self.tier])

    def reset(self, sim: Any = None, *, timestamp_s: float = 0.0) -> None:
        if not np.isfinite(float(timestamp_s)) or float(timestamp_s) < 0.0:
            raise ShakeBenchProviderError("timestamp_s must be finite and non-negative")

    def on_physics_sample(self, sim: Any, sample_time_s: float, policy_step: bool = False) -> None:
        del sim, policy_step
        if not np.isfinite(float(sample_time_s)) or float(sample_time_s) < 0.0:
            raise ShakeBenchProviderError("sample_time_s must be finite and non-negative")

    def observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        del sim
        return {}

    def get_observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        return self.observation(sim)

    def policy_observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        return self.observation(sim)

    def get_observations(self, sim: Any = None) -> dict[str, np.ndarray]:
        return self.observation(sim)

    def audit_compiled_mount(self, sim: Any) -> None:
        del sim
        return None

    def privileged_snapshot(self, sim: Any = None, *, time_s: Optional[float] = None) -> dict[str, Any]:
        del sim, time_s
        return {}


class V0Provider(VibrationProvider):
    """No dedicated vibration channel."""

    tier = "V0"


class V1Provider(VibrationProvider):
    """Delayed/noisy canonical IMU provider mounted at robot-base origin."""

    tier = "V1"

    def __init__(
        self,
        *,
        seed: int = 0,
        imu_mode: str = "canonical_noisy_v1",
        imu_profile: CanonicalIMUProfile = CANONICAL_IMU_PROFILE,
        imu_body_name: Optional[str] = None,
        sensor_body_name: Optional[str] = None,
        deck_body_name: Optional[str] = None,
        sensor_position_body_m: Iterable[float] = (0.0, 0.0, 0.0),
        sensor_quat_body_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
    ) -> None:
        try:
            self.imu = CanonicalIMU(seed=seed, mode=imu_mode, profile=imu_profile)
        except (ShakeBenchSensorError, TypeError, ValueError) as exc:
            raise ShakeBenchProviderError(str(exc)) from exc
        if sensor_body_name is not None and imu_body_name is not None and str(sensor_body_name) != str(imu_body_name):
            raise ShakeBenchProviderError("sensor_body_name and imu_body_name specify different bodies")
        selected_imu_body_name = sensor_body_name or imu_body_name or deck_body_name or "robot0_base"
        self.imu_body_name = str(selected_imu_body_name)
        self.sensor_body_name = self.imu_body_name
        self.deck_body_name = "deck" if deck_body_name is None else str(deck_body_name)
        self.sensor_position_body_m = np.array(sensor_position_body_m, dtype=float, copy=True)
        self.sensor_quat_body_wxyz = np.array(sensor_quat_body_wxyz, dtype=float, copy=True)
        self.gravity_world_m_s2 = np.array(gravity_world_m_s2, dtype=float, copy=True)
        if self.sensor_position_body_m.shape != (3,) or not np.all(np.isfinite(self.sensor_position_body_m)):
            raise ShakeBenchProviderError("sensor_position_body_m must have three finite values")
        if self.sensor_quat_body_wxyz.shape != (4,) or not np.all(np.isfinite(self.sensor_quat_body_wxyz)):
            raise ShakeBenchProviderError("sensor_quat_body_wxyz must have four finite values")
        if self.gravity_world_m_s2.shape != (3,) or not np.all(np.isfinite(self.gravity_world_m_s2)):
            raise ShakeBenchProviderError("gravity_world_m_s2 must have three finite values")
        self.sensor_position_body_m.setflags(write=False)
        self.sensor_quat_body_wxyz.setflags(write=False)
        self.gravity_world_m_s2.setflags(write=False)
        self._next_acquisition_time_s = self.imu.profile.dt_s
        self._last_physics_time_s = 0.0
        self._compiled_imu_mount = None
        self.reset()

    def audit_compiled_mount(self, sim: Any) -> dict[str, Any]:
        """Audit the canonical robot-base IMU body and save deck extrinsics."""

        model = getattr(sim, "model", sim)
        model = getattr(model, "_model", model)
        sensor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.imu_body_name))
        deck_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.deck_body_name))
        if sensor_id < 0:
            raise ShakeBenchProviderError(f"compiled model is missing IMU body {self.imu_body_name!r}")
        if deck_id < 0:
            raise ShakeBenchProviderError(f"compiled model is missing deck body {self.deck_body_name!r}")
        parent_id = int(model.body_parentid[sensor_id])
        if parent_id != deck_id:
            raise ShakeBenchProviderError(
                f"IMU body {self.imu_body_name!r} must be a rigid child of {self.deck_body_name!r}"
            )
        local_position = np.asarray(model.body_pos[sensor_id], dtype=float).copy()
        local_quaternion = _normalise_quaternion_wxyz(model.body_quat[sensor_id], "compiled IMU body quaternion")
        metadata_position = np.asarray(self.sensor_position_body_m, dtype=float)
        metadata_quaternion = _normalise_quaternion_wxyz(self.sensor_quat_body_wxyz, "sensor_quat_body_wxyz")
        if not np.allclose(metadata_position, 0.0, rtol=0.0, atol=1e-12) or not np.allclose(
            metadata_quaternion, (1.0, 0.0, 0.0, 0.0), rtol=0.0, atol=1e-12
        ):
            raise ShakeBenchProviderError(
                "canonical IMU extrinsics must be zero position and identity orientation in robot_base"
            )
        mount = {
            "sensor_body_name": self.imu_body_name,
            "sensor_frame_parent": "robot_base",
            "deck_body_name": self.deck_body_name,
            "parent_body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
            "robot_base_pose_in_deck": np.concatenate((local_position, local_quaternion)),
            "sensor_position_m_in_robot_base": metadata_position.copy(),
            "sensor_quaternion_wxyz_in_robot_base": metadata_quaternion.copy(),
        }
        self._compiled_imu_mount = mount
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in mount.items()}

    def reset(self, sim: Any = None, *, timestamp_s: float = 0.0) -> None:
        super().reset(sim, timestamp_s=timestamp_s)
        if sim is None:
            self.imu.reset(timestamp_s=timestamp_s)
        else:
            try:
                self.audit_compiled_mount(sim)
                from robosuite.utils.shakebench_sensors import clean_imu_measurement_from_sim

                clean, kinematics = clean_imu_measurement_from_sim(
                    sim,
                    self.imu_body_name,
                    sensor_position_body_m=self.sensor_position_body_m,
                    sensor_quat_body_wxyz=self.sensor_quat_body_wxyz,
                    gravity_world_m_s2=self.gravity_world_m_s2,
                )
                # At t=0 MuJoCo has populated accelerations from the initial
                # constraint solve, which is not an acquired physical sample
                # and can contain a transient.  The episode starts from the
                # authored zero-motion deck state, so seed static history from
                # that state while retaining the compiled orientation.
                data = getattr(sim, "data", None)
                if data is not None and abs(float(data.time) - float(timestamp_s)) <= 1e-12:
                    rotation_world_to_sensor = kinematics["rotation_world_to_sensor"]
                    clean = np.concatenate(
                        (
                            rotation_world_to_sensor.dot(-self.gravity_world_m_s2),
                            np.zeros(3, dtype=float),
                        )
                    )
                    kinematics["origin_acceleration_world_m_s2"] = np.zeros(3, dtype=float)
                    kinematics["angular_acceleration_world_rad_s2"] = np.zeros(3, dtype=float)
                    kinematics["angular_velocity_world_rad_s"] = np.zeros(3, dtype=float)
                    kinematics["point_acceleration_world_m_s2"] = np.zeros(3, dtype=float)
                    kinematics["specific_force_sensor_m_s2"] = clean[:3].copy()
                    kinematics["angular_velocity_sensor_rad_s"] = np.zeros(3, dtype=float)
                self.imu.reset(initial_clean_measurement=clean, timestamp_s=timestamp_s)
                self.imu._last_kinematics = kinematics
            except (ShakeBenchSensorError, TypeError, ValueError) as exc:
                raise ShakeBenchProviderError(str(exc)) from exc
        self._next_acquisition_time_s = float(timestamp_s) + self.imu.profile.dt_s
        self._last_physics_time_s = float(timestamp_s)

    def on_physics_sample(self, sim: Any, sample_time_s: float, policy_step: bool = False) -> None:
        super().on_physics_sample(sim, sample_time_s, policy_step=policy_step)
        sample_time = float(sample_time_s)
        if sample_time < self._last_physics_time_s - 1e-12:
            self.reset(sim, timestamp_s=sample_time)
        while sample_time + 1e-12 >= self._next_acquisition_time_s:
            self.imu.acquire_from_sim(
                sim,
                timestamp_s=self._next_acquisition_time_s,
                body_name=self.imu_body_name,
                sensor_position_body_m=self.sensor_position_body_m,
                sensor_quat_body_wxyz=self.sensor_quat_body_wxyz,
                gravity_world_m_s2=self.gravity_world_m_s2,
            )
            self._next_acquisition_time_s += self.imu.profile.dt_s
        self._last_physics_time_s = sample_time

    def observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        del sim
        return self.imu.to_policy_observation()

    def privileged_snapshot(self, sim: Any = None, *, time_s: Optional[float] = None) -> dict[str, Any]:
        del sim, time_s
        trace = self.imu.trace
        result = {f"imu_{key}": value for key, value in trace.items()}
        if trace["clean_measurement"].ndim == 2:
            result["imu_clean_specific_force_m_s2"] = trace["clean_measurement"][:, :3]
            result["imu_clean_angular_velocity_rad_s"] = trace["clean_measurement"][:, 3:]
            result["imu_clean_measurement_mixed_channels"] = trace["clean_measurement"]
            result["imu_filtered_measurement_mixed_channels"] = trace["filtered_measurement"]
        result["imu_window_acquisition_time_s"] = self.imu.window_acquisition_timestamps_s
        result["imu_initial_bias"] = self.imu.initial_bias
        result["imu_current_bias"] = self.imu.bias
        result["imu_last_kinematics"] = self.imu.last_kinematics
        result["imu_body_name"] = self.imu_body_name
        result["imu_compiled_mount"] = (
            None
            if self._compiled_imu_mount is None
            else {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._compiled_imu_mount.items()
            }
        )
        return result


class V2Provider(V1Provider):
    """V1 plus current realized deck/table support state."""

    tier = "V2"

    def __init__(
        self,
        *,
        deck_body_name: str = "deck",
        imu_body_name: Optional[str] = "robot0_base",
        table_body_name: str = "worktable",
        nominal_frame_position_m: Iterable[float] = (0.0, 0.0, 0.0),
        nominal_frame_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        **kwargs: Any,
    ) -> None:
        super().__init__(deck_body_name=deck_body_name, imu_body_name=imu_body_name, **kwargs)
        self.table_body_name = str(table_body_name)
        self.nominal_frame_position_m = np.array(nominal_frame_position_m, dtype=float, copy=True)
        self.nominal_frame_quat_wxyz = np.array(nominal_frame_quat_wxyz, dtype=float, copy=True)
        if self.nominal_frame_position_m.shape != (3,) or not np.all(np.isfinite(self.nominal_frame_position_m)):
            raise ShakeBenchProviderError("nominal_frame_position_m must have three finite values")
        _normalise_quaternion_wxyz(self.nominal_frame_quat_wxyz, "nominal_frame_quat_wxyz")
        self.nominal_frame_position_m.setflags(write=False)
        self.nominal_frame_quat_wxyz.setflags(write=False)
        self._current_support_state: Optional[SupportState] = None

    def reset(self, sim: Any = None, *, timestamp_s: float = 0.0) -> None:
        super().reset(sim, timestamp_s=timestamp_s)
        if sim is None:
            self._current_support_state = None
        else:
            data = getattr(sim, "data", None)
            assume_static = data is not None and abs(float(data.time) - float(timestamp_s)) <= 1e-12
            self._current_support_state = support_state_from_sim(
                sim,
                deck_body_name=self.deck_body_name,
                table_body_name=self.table_body_name,
                nominal_frame_position_m=self.nominal_frame_position_m,
                nominal_frame_quat_wxyz=self.nominal_frame_quat_wxyz,
                assume_static=assume_static,
            )

    def on_physics_sample(self, sim: Any, sample_time_s: float, policy_step: bool = False) -> None:
        super().on_physics_sample(sim, sample_time_s, policy_step=policy_step)
        # This is intentionally a single current snapshot, not a history or an
        # online estimator.  The policy path recomputes it at read time too.
        self._current_support_state = support_state_from_sim(
            sim,
            deck_body_name=self.deck_body_name,
            table_body_name=self.table_body_name,
            nominal_frame_position_m=self.nominal_frame_position_m,
            nominal_frame_quat_wxyz=self.nominal_frame_quat_wxyz,
        )

    def observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        result = super().observation(sim)
        if sim is not None:
            data = getattr(sim, "data", None)
            assume_static = data is not None and abs(float(data.time)) <= 1e-12
            self._current_support_state = support_state_from_sim(
                sim,
                deck_body_name=self.deck_body_name,
                table_body_name=self.table_body_name,
                nominal_frame_position_m=self.nominal_frame_position_m,
                nominal_frame_quat_wxyz=self.nominal_frame_quat_wxyz,
                assume_static=assume_static,
            )
        if self._current_support_state is None:
            if sim is not None:
                raise ShakeBenchProviderError("V2 support state is not initialized")
            self._current_support_state = SupportState(
                deck_pose_in_nominal_frame=np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
                deck_twist_in_nominal_frame=np.zeros(6),
                deck_accel_in_nominal_frame=np.zeros(6),
                table_pose_in_deck_frame=np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
                table_twist_in_deck_frame=np.zeros(6),
                table_accel_in_deck_frame=np.zeros(6),
            )
        result.update(self._current_support_state.to_policy_observation())
        return result

    @property
    def current_support_state(self) -> Optional[SupportState]:
        return self._current_support_state

    def privileged_snapshot(self, sim: Any = None, *, time_s: Optional[float] = None) -> dict[str, Any]:
        result = super().privileged_snapshot(sim, time_s=time_s)
        if sim is not None:
            data = getattr(sim, "data", None)
            assume_static = data is not None and abs(float(data.time)) <= 1e-12
            self._current_support_state = support_state_from_sim(
                sim,
                deck_body_name=self.deck_body_name,
                table_body_name=self.table_body_name,
                nominal_frame_position_m=self.nominal_frame_position_m,
                nominal_frame_quat_wxyz=self.nominal_frame_quat_wxyz,
                assume_static=assume_static,
            )
        if self._current_support_state is not None:
            result["support_state"] = self._current_support_state.to_dict()
        return result


class V3Provider(V2Provider):
    """V2 plus a self-describing authored future excitation program."""

    tier = "V3"

    def __init__(
        self,
        *,
        program: Optional[ExcitationProgram] = None,
        excitation_program: Optional[ExcitationProgram] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        if program is not None and excitation_program is not None and program is not excitation_program:
            raise ShakeBenchProviderError("program and excitation_program specify different objects")
        selected_program = program if program is not None else excitation_program
        if selected_program is None:
            selected_program = build_excitation_program(seed=seed, level_scale=0.0, active_axes=())
        if not isinstance(selected_program, ExcitationProgram):
            raise ShakeBenchProviderError("V3 requires an ExcitationProgram")
        self.program = selected_program
        self._episode_time_s = 0.0
        super().__init__(seed=seed, **kwargs)

    @property
    def episode_time_s(self) -> float:
        return float(self._episode_time_s)

    @property
    def program_payload(self) -> dict[str, Any]:
        return self.program.to_runtime_payload(self._episode_time_s)

    def public_program_payload(self) -> dict[str, Any]:
        """Return exactly the payload fields available to a V3 policy."""

        return {
            "line_accel_amplitude": np.asarray(self.program.line_accel_amplitude, dtype=np.float32).copy(),
            "line_omega_rad_s": np.asarray(self.program.line_omega_rad_s, dtype=np.float32).copy(),
            "line_phase_at_episode_zero": np.asarray(self.program.line_phase_at_episode_zero, dtype=np.float32).copy(),
            "line_mask": np.asarray(self.program.line_mask, dtype=np.bool_).copy(),
            "episode_time_s": np.asarray(self._episode_time_s, dtype=np.float32),
            "ramp_type": np.asarray("quintic_smoothstep", dtype="<U20"),
            "ramp_duration_s": np.asarray(self.program.config.ramp_duration_s, dtype=np.float32),
            "program_frame": np.asarray("deck", dtype="<U4"),
        }

    def reset(self, sim: Any = None, *, timestamp_s: float = 0.0) -> None:
        super().reset(sim, timestamp_s=timestamp_s)
        self._episode_time_s = float(timestamp_s)

    def on_physics_sample(self, sim: Any, sample_time_s: float, policy_step: bool = False) -> None:
        super().on_physics_sample(sim, sample_time_s, policy_step=policy_step)
        self._episode_time_s = float(sample_time_s)

    def observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        result = super().observation(sim)
        if sim is not None:
            data = getattr(sim, "data", None)
            if data is not None and hasattr(data, "time"):
                self._episode_time_s = float(data.time)
        result.update(self.public_program_payload())
        return result

    def privileged_snapshot(self, sim: Any = None, *, time_s: Optional[float] = None) -> dict[str, Any]:
        result = super().privileged_snapshot(sim, time_s=time_s)
        result["authored_program"] = self.program.to_runtime_payload(
            self._episode_time_s if time_s is None else float(time_s)
        )
        return result

    def evaluate_authored(self, time_s: Any):
        """Evaluate the authored command at an arbitrary query time.

        Reconstruction consumes only the public V3 arrays and metadata; it
        never calls the provider-held program evaluator or consults a realized
        simulator state/contact result.
        """

        return reconstruct_authored_motion(self.public_program_payload(), time_s)


def make_vibration_provider(
    tier: str,
    *,
    seed: int = 0,
    imu_mode: str = "canonical_noisy_v1",
    imu_profile: CanonicalIMUProfile = CANONICAL_IMU_PROFILE,
    program: Optional[ExcitationProgram] = None,
    excitation_program: Optional[ExcitationProgram] = None,
    deck_body_name: str = "deck",
    imu_body_name: str = "robot0_base",
    table_body_name: str = "worktable",
    nominal_frame_position_m: Iterable[float] = (0.0, 0.0, 0.0),
    nominal_frame_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
    **kwargs: Any,
) -> VibrationProvider:
    normalized = normalize_observation_tier(tier)
    common = {
        "seed": seed,
        "imu_mode": imu_mode,
        "imu_profile": imu_profile,
        "deck_body_name": deck_body_name,
        "imu_body_name": imu_body_name,
        "table_body_name": table_body_name,
        "nominal_frame_position_m": nominal_frame_position_m,
        "nominal_frame_quat_wxyz": nominal_frame_quat_wxyz,
    }
    common.update(kwargs)
    if normalized == "V0":
        return V0Provider()
    if normalized == "V1":
        return V1Provider(
            **{
                key: value
                for key, value in common.items()
                if key not in {"table_body_name", "nominal_frame_position_m", "nominal_frame_quat_wxyz"}
            }
        )
    if normalized == "V2":
        return V2Provider(**common)
    return V3Provider(program=program, excitation_program=excitation_program, **common)


create_vibration_provider = make_vibration_provider
VibrationProviderFactory = make_vibration_provider
VibrationProviderV0 = V0Provider
VibrationProviderV1 = V1Provider
VibrationProviderV2 = V2Provider
VibrationProviderV3 = V3Provider
TIER_KEYS = TIER_POLICY_KEYS


__all__ = [
    "COMMON_STATE_KEYS",
    "OBSERVATION_TIERS",
    "POLICY_FIELD_CONTRACT",
    "TIER_ADDED_KEYS",
    "TIER_POLICY_KEYS",
    "V0_POLICY_KEYS",
    "V1_POLICY_KEYS",
    "V2_POLICY_KEYS",
    "V3_POLICY_KEYS",
    "RelativePoseTwistAcceleration",
    "RigidBodyState",
    "ShakeBenchProviderError",
    "SupportState",
    "V0Provider",
    "V1Provider",
    "V2Provider",
    "V3Provider",
    "VibrationProvider",
    "VibrationProviderFactory",
    "VibrationProviderV0",
    "VibrationProviderV1",
    "VibrationProviderV2",
    "VibrationProviderV3",
    "TIER_KEYS",
    "create_vibration_provider",
    "make_vibration_provider",
    "normalize_observation_tier",
    "observation_contract_for_tier",
    "reconstruct_authored_motion",
    "relative_pose_twist_acceleration",
    "policy_keys_for_tier",
    "support_state_from_sim",
]
