"""Public-observation reference controller for the ShakeBench State track.

The controller is intentionally a small, inspectable baseline.  It owns no
MuJoCo object and accepts only the State observation, public static context,
and a provider payload.  Simulator contact and evaluator truth remain on the
recorder side of the boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.shakebench_isolator import relative_transfer_function
from robosuite.utils.shakebench_providers import (
    OBSERVATION_TIERS,
    TIER_POLICY_KEYS,
    reconstruct_authored_motion,
)
from robosuite.utils.shakebench_sensors import CANONICAL_IMU_PROFILE, G0_M_S2


class ShakeBenchOracleError(ValueError):
    """Raised when the public controller contract is malformed."""


class TaskPhase(str, Enum):
    SETTLE = "settle"
    APPROACH = "approach"
    CLEARANCE_LIFT = "clearance_lift"
    LATERAL_ALIGN_ABOVE_CAN = "lateral_align_above_can"
    ALIGN_SETTLE = "align_settle"
    VERTICAL_DESCEND = "vertical_descend"
    DESCEND = "descend"
    GRASP = "grasp"
    GRASP_CLOSE = "grasp_close"
    PRELIFT_VERIFY = "prelift_verify"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    VERIFY = "verify"
    RECOVERY_HOLD = "recovery_hold_close"
    RECOVERY_HOLD_CLOSE = "recovery_hold_close"
    REVERSE_TO_PRELIFT_START = "recovery_hold_close"
    RECOVERY_LOWER_IF_NEEDED = "recovery_lower_if_needed"
    LOWER_TO_VERIFIED_TABLE_HEIGHT = "recovery_lower_if_needed"
    WAIT_PUBLIC_SETTLE = "wait_public_settle"
    RECOVERY_OPEN = "recovery_open"
    OPEN_IF_FULLY_SUPPORTED_AND_IN_BOUNDS = "recovery_open"
    CLEARANCE_RETREAT = "clearance_retreat"
    RETREAT = "clearance_retreat"
    RE_ALIGN = "re_align"
    COMPLETE = "complete"
    ABORTED = "aborted"
    # Compatibility alias.  New runner code must use ``abort_reason`` and
    # never infer a task result from a controller phase.
    FAILED = "aborted"


class MotionCapability(str, Enum):
    """The physical action policy associated with an execution phase."""

    HOLD = "hold"
    FREE_SPACE = "free_space"
    CLEARANCE_TRANSLATE = "clearance_translate"
    CONTACT_APPROACH = "contact_approach"
    GRIPPER_CLOSE = "gripper_close"
    OBJECT_HELD = "object_held"
    RECOVERY_LOWER = "recovery_lower"
    RECOVERY_RETREAT = "recovery_retreat"
    TERMINAL = "terminal"


# Phase names describe the state-machine history; capabilities are the one
# source of action limits and gripper semantics.  Keeping this table explicit
# makes a newly added phase fail during profile construction instead of
# silently receiving the free-space policy.
MOTION_CAPABILITY_BY_PHASE = MappingProxyType(
    {
        TaskPhase.SETTLE: MotionCapability.HOLD,
        TaskPhase.APPROACH: MotionCapability.FREE_SPACE,
        TaskPhase.CLEARANCE_LIFT: MotionCapability.CLEARANCE_TRANSLATE,
        TaskPhase.LATERAL_ALIGN_ABOVE_CAN: MotionCapability.FREE_SPACE,
        TaskPhase.ALIGN_SETTLE: MotionCapability.HOLD,
        TaskPhase.VERTICAL_DESCEND: MotionCapability.CONTACT_APPROACH,
        TaskPhase.DESCEND: MotionCapability.CONTACT_APPROACH,
        TaskPhase.GRASP: MotionCapability.GRIPPER_CLOSE,
        TaskPhase.GRASP_CLOSE: MotionCapability.GRIPPER_CLOSE,
        TaskPhase.PRELIFT_VERIFY: MotionCapability.GRIPPER_CLOSE,
        TaskPhase.LIFT: MotionCapability.OBJECT_HELD,
        TaskPhase.TRANSPORT: MotionCapability.OBJECT_HELD,
        TaskPhase.PLACE: MotionCapability.OBJECT_HELD,
        TaskPhase.RELEASE: MotionCapability.HOLD,
        TaskPhase.VERIFY: MotionCapability.HOLD,
        TaskPhase.RECOVERY_HOLD: MotionCapability.RECOVERY_RETREAT,
        TaskPhase.RECOVERY_LOWER_IF_NEEDED: MotionCapability.RECOVERY_LOWER,
        TaskPhase.WAIT_PUBLIC_SETTLE: MotionCapability.HOLD,
        TaskPhase.RECOVERY_OPEN: MotionCapability.TERMINAL,
        TaskPhase.CLEARANCE_RETREAT: MotionCapability.RECOVERY_RETREAT,
        TaskPhase.RE_ALIGN: MotionCapability.FREE_SPACE,
        TaskPhase.COMPLETE: MotionCapability.TERMINAL,
        TaskPhase.ABORTED: MotionCapability.TERMINAL,
    }
)


def motion_capability_for_phase(phase: TaskPhase) -> MotionCapability:
    """Return the unique action capability for a canonical task phase."""

    try:
        return MOTION_CAPABILITY_BY_PHASE[TaskPhase(phase)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ShakeBenchOracleError(f"phase {phase!r} has no motion capability") from exc


def validate_motion_capability_map() -> None:
    """Fail closed if the phase machine and capability table drift apart."""

    phases = set(TaskPhase)
    if set(MOTION_CAPABILITY_BY_PHASE) != phases:
        missing = sorted(phase.value for phase in phases - set(MOTION_CAPABILITY_BY_PHASE))
        extra = sorted(phase.value for phase in set(MOTION_CAPABILITY_BY_PHASE) - phases)
        raise ShakeBenchOracleError(f"motion capability map drifted; missing={missing}, extra={extra}")
    if any(not isinstance(capability, MotionCapability) for capability in MOTION_CAPABILITY_BY_PHASE.values()):
        raise ShakeBenchOracleError("motion capability map contains an invalid capability")


@dataclass(frozen=True)
class WorktableTaskContext:
    """Public immutable task geometry shared by every observation tier."""

    worktable_size_xy_m: tuple[float, float] = (0.65, 0.60)
    target_frame_origin_in_worktable_m: tuple[float, float, float] = (-0.10, 0.17, 0.042)
    table_surface_z_in_worktable_m: float = 0.03
    can_collision_radius_m: float = 0.02509177806572465
    can_collision_lower_support_m: float = -0.040297003330440104
    can_collision_upper_support_m: float = 0.03970300217508332
    finger_pad_tool_support_offsets_m: tuple[float, ...] = (0.0, 0.0, 0.0934)
    support_topology_id: str = "deck_robot_base_plus_isolated_worktable"
    # The robot base is a rigid child of the deck.  This compiled, episode-
    # static transform lets V2/V3 express table motion at the same public
    # robot-base reference point as V0/V1.
    deck_to_robot_base_position_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    deck_to_robot_base_quaternion_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        size = np.asarray(self.worktable_size_xy_m, dtype=float)
        origin = np.asarray(self.target_frame_origin_in_worktable_m, dtype=float)
        offsets = np.asarray(self.finger_pad_tool_support_offsets_m, dtype=float)
        deck_position = np.asarray(self.deck_to_robot_base_position_m, dtype=float)
        deck_quaternion = np.asarray(self.deck_to_robot_base_quaternion_wxyz, dtype=float)
        if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ShakeBenchOracleError("worktable_size_xy_m must be finite positive length two")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ShakeBenchOracleError("target_frame_origin_in_worktable_m must be finite length three")
        if offsets.shape != (3,) or not np.all(np.isfinite(offsets)):
            raise ShakeBenchOracleError("finger_pad_tool_support_offsets_m must be finite length three")
        if deck_position.shape != (3,) or not np.all(np.isfinite(deck_position)):
            raise ShakeBenchOracleError("deck_to_robot_base_position_m must be finite length three")
        if deck_quaternion.shape != (4,) or not np.all(np.isfinite(deck_quaternion)):
            raise ShakeBenchOracleError("deck_to_robot_base_quaternion_wxyz must be finite length four")
        if float(np.linalg.norm(deck_quaternion)) <= 1.0e-12:
            raise ShakeBenchOracleError("deck_to_robot_base_quaternion_wxyz must be nonzero")
        for name in (
            "table_surface_z_in_worktable_m",
            "can_collision_radius_m",
            "can_collision_lower_support_m",
            "can_collision_upper_support_m",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ShakeBenchOracleError(f"{name} must be finite")
        if (
            self.can_collision_radius_m <= 0.0
            or self.can_collision_lower_support_m >= self.can_collision_upper_support_m
        ):
            raise ShakeBenchOracleError("Can collision support bounds are invalid")
        if not isinstance(self.support_topology_id, str) or not self.support_topology_id:
            raise ShakeBenchOracleError("support_topology_id must be nonempty")

    @property
    def worktable_half_extents_xy_m(self) -> tuple[float, float]:
        return tuple(float(value) for value in np.asarray(self.worktable_size_xy_m, dtype=float) / 2.0)

    @property
    def sha256(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "WorktableTaskContext":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ShakeBenchOracleError("task context must be a mapping")
        default = cls()
        worktable = value.get("worktable", value)
        can = value.get("can", value)
        target = value.get("target_container", value)
        envelope = can.get("collision_envelope", {}) if isinstance(can, Mapping) else {}
        motion = value.get("support_motion", {})
        if not isinstance(motion, Mapping):
            motion = {}
        imu = value.get("imu", {})
        if not isinstance(imu, Mapping):
            imu = {}
        compiled_pose = motion.get(
            "deck_to_robot_base_pose_in_deck",
            value.get("deck_to_robot_base_pose_in_deck", imu.get("compiled_robot_base_pose_in_deck")),
        )
        if compiled_pose is None:
            compiled_position = motion.get(
                "deck_to_robot_base_position_m",
                value.get("deck_to_robot_base_position_m", default.deck_to_robot_base_position_m),
            )
            compiled_quaternion = motion.get(
                "deck_to_robot_base_quaternion_wxyz",
                value.get("deck_to_robot_base_quaternion_wxyz", default.deck_to_robot_base_quaternion_wxyz),
            )
        else:
            compiled_pose = np.asarray(compiled_pose, dtype=float)
            if compiled_pose.shape != (7,):
                raise ShakeBenchOracleError("deck_to_robot_base_pose_in_deck must have seven values")
            compiled_position = compiled_pose[:3]
            compiled_quaternion = compiled_pose[3:]
        return cls(
            worktable_size_xy_m=tuple(
                np.asarray(
                    worktable.get("dimensions_m", value.get("worktable_size_xy_m", default.worktable_size_xy_m)),
                    dtype=float,
                )[:2]
            ),
            target_frame_origin_in_worktable_m=tuple(
                value.get(
                    "target_frame_origin_in_worktable_m",
                    target.get("frame_origin_m", default.target_frame_origin_in_worktable_m),
                )
            ),
            table_surface_z_in_worktable_m=float(
                value.get("table_surface_z_in_worktable_m", default.table_surface_z_in_worktable_m)
            ),
            can_collision_radius_m=float(
                envelope.get("support_radius_m", value.get("can_collision_radius_m", default.can_collision_radius_m))
            ),
            can_collision_lower_support_m=float(
                envelope.get(
                    "lower_support_z_m",
                    value.get("can_collision_lower_support_m", default.can_collision_lower_support_m),
                )
            ),
            can_collision_upper_support_m=float(
                envelope.get(
                    "upper_support_z_m",
                    value.get("can_collision_upper_support_m", default.can_collision_upper_support_m),
                )
            ),
            finger_pad_tool_support_offsets_m=tuple(
                value.get("finger_pad_tool_support_offsets_m", default.finger_pad_tool_support_offsets_m)
            ),
            support_topology_id=str(value.get("support_topology_id", default.support_topology_id)),
            deck_to_robot_base_position_m=tuple(compiled_position),
            deck_to_robot_base_quaternion_wxyz=tuple(compiled_quaternion),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["worktable_half_extents_xy_m"] = list(self.worktable_half_extents_xy_m)
        result["context_sha256"] = self.sha256
        return result


RELATIVE_SUPPORT_ESTIMATE_SCHEMA_ID = "shakebench.relative_support_motion_estimate"
RELATIVE_SUPPORT_ESTIMATE_SCHEMA_VERSION = 1
RELATIVE_SUPPORT_SEMANTIC_QUANTITY = "worktable_relative_to_robot_base_motion"
RELATIVE_SUPPORT_FRAME = "robot_base"
RELATIVE_SUPPORT_REFERENCE_POINT = "target_origin"


@dataclass(frozen=True)
class RelativeSupportMotionEstimate:
    """One typed meaning for every V0--V3 disturbance estimate.

    The canonical values are the motion of the rigid worktable/target
    assembly relative to the robot base, expressed at the target-frame origin.
    The shared law consumes only ``current_relative_twist_robot_base`` and
    ``current_relative_acceleration_robot_base`` plus the final row of the
    predicted acceleration trajectory.  The ``tier`` field is provenance; it
    is deliberately not consulted by the law.
    """

    tier: str
    current_relative_pose_robot_base: np.ndarray
    current_relative_twist_robot_base: np.ndarray
    current_relative_acceleration_robot_base: np.ndarray
    measurement_timestamp_s: float
    policy_timestamp_s: float
    prediction_timestamps_s: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    predicted_relative_pose_robot_base: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), dtype=float))
    predicted_relative_twist_robot_base: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=float))
    predicted_relative_acceleration_robot_base: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6), dtype=float)
    )
    frame: str = RELATIVE_SUPPORT_FRAME
    reference_point: str = RELATIVE_SUPPORT_REFERENCE_POINT
    semantic_quantity: str = RELATIVE_SUPPORT_SEMANTIC_QUANTITY
    units: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(
            {
                "pose_translation": "m",
                "pose_rotation": "xyzw unit quaternion",
                "linear_twist": "m/s",
                "angular_twist": "rad/s",
                "linear_acceleration": "m/s2",
                "angular_acceleration": "rad/s2",
                "time": "s",
                "linear": "m/s2",
                "angular": "rad/s2",
            }
        )
    )
    latency_s: float = 0.0
    validity: bool = True
    prediction_validity: bool = True
    confidence: float = 1.0
    source: str = "neutral"
    future_program_available: bool = False
    context_sha256: Optional[str] = None
    legacy_current_linear_accel_m_s2: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.tier not in OBSERVATION_TIERS:
            raise ShakeBenchOracleError("estimate tier must be V0, V1, V2, or V3")
        if self.frame != RELATIVE_SUPPORT_FRAME:
            raise ShakeBenchOracleError("relative support estimate frame must be robot_base")
        if self.reference_point != RELATIVE_SUPPORT_REFERENCE_POINT:
            raise ShakeBenchOracleError("relative support estimate reference point must be target_origin")
        if self.semantic_quantity != RELATIVE_SUPPORT_SEMANTIC_QUANTITY:
            raise ShakeBenchOracleError("relative support estimate has an unsupported semantic quantity")
        pose = np.asarray(self.current_relative_pose_robot_base, dtype=float)
        twist = np.asarray(self.current_relative_twist_robot_base, dtype=float)
        acceleration = np.asarray(self.current_relative_acceleration_robot_base, dtype=float)
        times = np.asarray(self.prediction_timestamps_s, dtype=float)
        predicted_pose = np.asarray(self.predicted_relative_pose_robot_base, dtype=float)
        predicted_twist = np.asarray(self.predicted_relative_twist_robot_base, dtype=float)
        predicted_acceleration = np.asarray(self.predicted_relative_acceleration_robot_base, dtype=float)
        legacy_linear = (
            None
            if self.legacy_current_linear_accel_m_s2 is None
            else np.asarray(self.legacy_current_linear_accel_m_s2, dtype=float)
        )
        if pose.shape != (7,) or twist.shape != (6,) or acceleration.shape != (6,):
            raise ShakeBenchOracleError("relative support current state has shapes (7,), (6,), and (6,)")
        if times.ndim != 1 or predicted_pose.shape != (times.size, 7):
            raise ShakeBenchOracleError("prediction pose grid must have shape (N, 7)")
        if predicted_twist.shape != (times.size, 6) or predicted_acceleration.shape != (times.size, 6):
            raise ShakeBenchOracleError("prediction twist/acceleration grid has an invalid shape")
        if legacy_linear is not None and (legacy_linear.shape != (3,) or not np.all(np.isfinite(legacy_linear))):
            raise ShakeBenchOracleError("legacy current linear acceleration must be finite length three")
        if not all(
            np.all(np.isfinite(value))
            for value in (pose, twist, acceleration, times, predicted_pose, predicted_twist, predicted_acceleration)
        ):
            raise ShakeBenchOracleError("relative support estimate values must be finite")
        pose = pose.copy()
        pose_norm = float(np.linalg.norm(pose[3:]))
        if pose_norm <= 1.0e-12:
            raise ShakeBenchOracleError("current relative pose quaternion must be nonzero")
        pose[3:] /= pose_norm
        predicted_pose = predicted_pose.copy()
        if predicted_pose.size:
            norms = np.linalg.norm(predicted_pose[:, 3:], axis=1)
            if np.any(norms <= 1.0e-12):
                raise ShakeBenchOracleError("predicted relative poses must have nonzero quaternions")
            predicted_pose[:, 3:] /= norms[:, None]
        measurement = float(self.measurement_timestamp_s)
        policy = float(self.policy_timestamp_s)
        latency = float(self.latency_s)
        confidence = float(self.confidence)
        if not np.isfinite(measurement) or measurement < 0.0 or not np.isfinite(policy) or policy < 0.0:
            raise ShakeBenchOracleError("relative support timestamps must be finite and non-negative")
        if measurement > policy + 1.0e-9:
            raise ShakeBenchOracleError("measurement timestamp cannot be after policy timestamp")
        if not np.isfinite(latency) or latency < 0.0:
            raise ShakeBenchOracleError("relative support latency must be finite and non-negative")
        if not np.isclose(policy - measurement, latency, rtol=0.0, atol=5.0e-5):
            raise ShakeBenchOracleError("relative support latency must equal policy minus measurement time")
        if times.size and (np.any(times <= policy) or np.any(np.diff(times) <= 0.0)):
            raise ShakeBenchOracleError("prediction queries must be strictly after policy time")
        if not 0.0 <= confidence <= 1.0:
            raise ShakeBenchOracleError("relative support confidence must lie in [0, 1]")
        if (
            not isinstance(self.validity, (bool, np.bool_))
            or not isinstance(self.prediction_validity, (bool, np.bool_))
            or not isinstance(self.future_program_available, (bool, np.bool_))
        ):
            raise ShakeBenchOracleError("relative support validity flags must be boolean")
        if not isinstance(self.units, Mapping) or not self.units:
            raise ShakeBenchOracleError("relative support units must be a nonempty mapping")
        if not isinstance(self.source, str) or not self.source:
            raise ShakeBenchOracleError("relative support source must be nonempty")
        if self.context_sha256 is not None and (
            not isinstance(self.context_sha256, str) or len(self.context_sha256) != 64
        ):
            raise ShakeBenchOracleError("relative support context hash must be a SHA-256 hex digest")
        for name, value in (
            ("current_relative_pose_robot_base", pose),
            ("current_relative_twist_robot_base", twist),
            ("current_relative_acceleration_robot_base", acceleration),
            ("prediction_timestamps_s", times),
            ("predicted_relative_pose_robot_base", predicted_pose),
            ("predicted_relative_twist_robot_base", predicted_twist),
            ("predicted_relative_acceleration_robot_base", predicted_acceleration),
        ):
            value = np.array(value, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "measurement_timestamp_s", measurement)
        object.__setattr__(self, "policy_timestamp_s", policy)
        object.__setattr__(self, "latency_s", latency)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        if legacy_linear is not None:
            legacy_linear = legacy_linear.copy()
            legacy_linear.setflags(write=False)
        object.__setattr__(self, "legacy_current_linear_accel_m_s2", legacy_linear)

    # Compatibility properties retain the R4 spelling while the canonical
    # interface above keeps one physical meaning for all tiers.
    @property
    def timestamp_s(self) -> float:
        return self.measurement_timestamp_s

    @property
    def current_linear_accel_m_s2(self) -> np.ndarray:
        if self.legacy_current_linear_accel_m_s2 is not None:
            return self.legacy_current_linear_accel_m_s2
        return self.current_relative_acceleration_robot_base[:3]

    @property
    def current_angular_velocity_rad_s(self) -> np.ndarray:
        return self.current_relative_twist_robot_base[3:]

    @property
    def future_query_time_s(self) -> Optional[float]:
        return float(self.prediction_timestamps_s[-1]) if self.prediction_timestamps_s.size else None

    @property
    def future_linear_accel_m_s2(self) -> np.ndarray:
        if not self.predicted_relative_acceleration_robot_base.size:
            return np.zeros(3, dtype=float)
        return self.predicted_relative_acceleration_robot_base[-1, :3]

    @property
    def future_angular_accel_rad_s2(self) -> np.ndarray:
        if not self.predicted_relative_acceleration_robot_base.size:
            return np.zeros(3, dtype=float)
        return self.predicted_relative_acceleration_robot_base[-1, 3:]

    @property
    def current_relative_pose(self) -> np.ndarray:
        return self.current_relative_pose_robot_base

    @property
    def current_relative_twist(self) -> np.ndarray:
        return self.current_relative_twist_robot_base

    @property
    def current_relative_acceleration(self) -> np.ndarray:
        return self.current_relative_acceleration_robot_base

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_id": RELATIVE_SUPPORT_ESTIMATE_SCHEMA_ID,
            "schema_version": RELATIVE_SUPPORT_ESTIMATE_SCHEMA_VERSION,
            "tier": self.tier,
            "semantic_quantity": self.semantic_quantity,
            "frame": self.frame,
            "reference_point": self.reference_point,
            "units": dict(self.units),
            "current_relative_pose_robot_base": self.current_relative_pose_robot_base.tolist(),
            "current_relative_twist_robot_base": self.current_relative_twist_robot_base.tolist(),
            "current_relative_acceleration_robot_base": self.current_relative_acceleration_robot_base.tolist(),
            "measurement_timestamp_s": self.measurement_timestamp_s,
            "policy_timestamp_s": self.policy_timestamp_s,
            "latency_s": self.latency_s,
            "prediction_timestamps_s": self.prediction_timestamps_s.tolist(),
            "predicted_relative_pose_robot_base": self.predicted_relative_pose_robot_base.tolist(),
            "predicted_relative_twist_robot_base": self.predicted_relative_twist_robot_base.tolist(),
            "predicted_relative_acceleration_robot_base": self.predicted_relative_acceleration_robot_base.tolist(),
            "validity": bool(self.validity),
            "prediction_validity": bool(self.prediction_validity),
            "confidence": self.confidence,
            "source": self.source,
            "future_program_available": bool(self.future_program_available),
            "context_sha256": self.context_sha256,
            "legacy_current_linear_accel_m_s2": (
                None
                if self.legacy_current_linear_accel_m_s2 is None
                else self.legacy_current_linear_accel_m_s2.tolist()
            ),
        }
        # Deprecated aliases are emitted during the explicit R4 -> R5 schema
        # transition so old trace readers cannot silently reinterpret values.
        result.update(
            {
                "timestamp_s": self.measurement_timestamp_s,
                "measurement_timestamp_s": self.measurement_timestamp_s,
                "current_linear_accel_m_s2": self.current_linear_accel_m_s2.tolist(),
                "current_angular_velocity_rad_s": self.current_angular_velocity_rad_s.tolist(),
                "future_linear_accel_m_s2": self.future_linear_accel_m_s2.tolist(),
                "future_angular_accel_rad_s2": self.future_angular_accel_rad_s2.tolist(),
                "future_query_time_s": self.future_query_time_s,
            }
        )
        return result


# Public compatibility name retained for callers that imported the R4 type.
VibrationEstimate = RelativeSupportMotionEstimate


def _rotation_from_rotvec(rotation_vector: np.ndarray) -> np.ndarray:
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotation_vector / angle
    skew = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)), dtype=float)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew.dot(skew)


def gravity_compensated_imu_window(
    imu_window: np.ndarray,
    dt_s: float,
    *,
    initial_rotation_control_from_sensor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use all public IMU samples to estimate gravity-free control-frame motion.

    ``initial_rotation_control_from_sensor`` is the declared nominal attitude
    at the beginning of the delayed public window. Gyro samples integrate the
    attitude forward; each specific-force sample then has its frame-aware
    gravity projection removed before averaging.
    """

    window = np.asarray(imu_window, dtype=float)
    if window.shape != (10, 6) or not np.all(np.isfinite(window)):
        raise ShakeBenchOracleError("IMU window must be finite with shape (10, 6)")
    if not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ShakeBenchOracleError("IMU dt must be finite and positive")
    rotation = (
        np.eye(3)
        if initial_rotation_control_from_sensor is None
        else np.asarray(initial_rotation_control_from_sensor, dtype=float)
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ShakeBenchOracleError("initial IMU attitude must be a finite 3x3 matrix")
    gravity_control = np.asarray((0.0, 0.0, 9.81), dtype=float)
    motions = []
    angular = []
    for sample in window:
        acceleration_sensor, gyro_sensor = sample[:3], sample[3:]
        gravity_sensor = rotation.T.dot(gravity_control)
        motions.append(rotation.dot(acceleration_sensor - gravity_sensor))
        angular.append(rotation.dot(gyro_sensor))
        rotation = rotation.dot(_rotation_from_rotvec(gyro_sensor * float(dt_s)))
    return np.mean(motions, axis=0), np.mean(angular, axis=0)


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project a numerically integrated rotation back to SO(3)."""

    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    result = u.dot(vt)
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u.dot(vt)
    return result


class V1IMUEstimator:
    """Controller-owned stateful estimator for the canonical public IMU lane."""

    def __init__(self, profile: "OracleControllerProfile") -> None:
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        self.attitude_control_from_sensor = np.eye(3, dtype=float)
        self.last_policy_time_s: Optional[float] = None
        self.last_window_end_time_s: Optional[float] = None
        self.gravity_control_estimate_m_s2 = np.asarray((0.0, 0.0, G0_M_S2), dtype=float)
        self.gravity_filter_state = np.zeros(3, dtype=float)
        self.reset_generation = getattr(self, "reset_generation", 0) + 1

    def estimate(
        self,
        imu_window: np.ndarray,
        dt_s: float,
        *,
        policy_time_s: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        policy_time = float(policy_time_s)
        if not np.isfinite(policy_time) or policy_time < 0.0:
            raise ShakeBenchOracleError("V1 policy time must be finite and non-negative")
        if self.last_policy_time_s is not None and policy_time <= self.last_policy_time_s + 1.0e-12:
            self.reset()
        window = np.asarray(imu_window, dtype=float)
        if window.shape != (10, 6) or not np.all(np.isfinite(window)):
            raise ShakeBenchOracleError("V1 IMU window must be finite with shape (10, 6)")
        dt = float(dt_s)
        if not np.isfinite(dt) or not np.isclose(dt, 1.0 / CANONICAL_IMU_PROFILE.sample_rate_hz, rtol=0.0, atol=1e-7):
            raise ShakeBenchOracleError("V1 IMU sample interval must be the canonical 0.005 s")

        rotation = self.attitude_control_from_sensor.copy()
        motion_samples = []
        angular_samples = []
        for sample in window:
            specific_force_sensor = sample[:3]
            gyro_sensor = sample[3:]
            gravity_sensor = rotation.T.dot(self.gravity_control_estimate_m_s2)
            motion_samples.append(rotation.dot(specific_force_sensor - gravity_sensor))
            angular_samples.append(rotation.dot(gyro_sensor))
            rotation = _orthonormalize_rotation(rotation.dot(_rotation_from_rotvec(gyro_sensor * dt)))

            # Keep an explicit gravity filter state, but only adapt it when the
            # public signal is compatible with quasi-static gravity.  This
            # avoids turning a transient acceleration into a new gravity axis.
            gravity_measurement = rotation.dot(specific_force_sensor)
            if np.linalg.norm(gyro_sensor) < 0.25 and abs(np.linalg.norm(gravity_measurement) - G0_M_S2) < 0.5:
                self.gravity_filter_state = 0.98 * self.gravity_filter_state + 0.02 * (
                    gravity_measurement - self.gravity_control_estimate_m_s2
                )
                self.gravity_control_estimate_m_s2 = _orthonormalize_gravity(
                    self.gravity_control_estimate_m_s2 + 0.02 * self.gravity_filter_state
                )
        self.attitude_control_from_sensor = rotation
        self.last_policy_time_s = policy_time
        measurement_time = max(0.0, policy_time - self.profile.imu_end_to_end_latency_s)
        self.last_window_end_time_s = measurement_time
        return np.mean(motion_samples, axis=0), np.mean(angular_samples, axis=0), measurement_time


def _orthonormalize_gravity(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.asarray((0.0, 0.0, G0_M_S2), dtype=float)
    return vector * (G0_M_S2 / norm)


DEFAULT_FUTURE_ISOLATOR_FN_HZ = (4.0, 4.0, 4.0, 3.0, 3.0, 2.0)
DEFAULT_FUTURE_ISOLATOR_ZETA = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
DEFAULT_DECK_TO_CONTROL_ROTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
DIAGNOSTIC_MODES = (
    "main",
    "task_executive_only",
    "compensation_off",
    "common_public_history",
    "current_state_compensation",
    "preview_off",
    "preview_on",
)


def future_relative_acceleration_from_public_program(
    observation: Mapping[str, Any],
    query_time_s: float,
    *,
    natural_frequency_hz: tuple[float, ...] = DEFAULT_FUTURE_ISOLATOR_FN_HZ,
    damping_ratio: tuple[float, ...] = DEFAULT_FUTURE_ISOLATOR_ZETA,
    deck_to_control_rotation: tuple[float, ...] = DEFAULT_DECK_TO_CONTROL_ROTATION,
    reference_point_offset_m: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Predict common-frame support acceleration from public authored lines.

    The transfer function is applied to each line's *ramped* acceleration.
    Thus the input to the transfer contains ``qdd*ramp + 2*qdot*ramp' +
    q*ramp''``.  This is a causal preview of authored forcing only; no future
    realized simulator state is consulted.
    """

    query = float(query_time_s)
    if not np.isfinite(query) or query < 0.0:
        raise ShakeBenchOracleError("future query time must be finite and non-negative")
    payload = dict(observation)
    # The public reconstruction is the schema-validation seam and keeps the
    # preview algebra exactly aligned with the authored q/qdot/qdd contract.
    reconstruct_authored_motion(payload, query)
    amplitudes = np.asarray(payload["line_accel_amplitude"], dtype=float)
    omegas = np.asarray(payload["line_omega_rad_s"], dtype=float)
    phases = np.asarray(payload["line_phase_at_episode_zero"], dtype=float)
    mask = np.asarray(payload["line_mask"], dtype=bool)
    natural = np.asarray(natural_frequency_hz, dtype=float)
    damping = np.asarray(damping_ratio, dtype=float)
    deck_rotation = np.asarray(deck_to_control_rotation, dtype=float)
    reference_offset = np.asarray(reference_point_offset_m, dtype=float)
    if natural.shape != (6,) or damping.shape != (6,) or np.any(natural <= 0.0) or np.any(damping < 0.0):
        raise ShakeBenchOracleError("future isolator parameters must contain six valid axes")
    if deck_rotation.shape != (9,) or not np.all(np.isfinite(deck_rotation)):
        raise ShakeBenchOracleError("deck-to-control rotation must contain nine finite values")
    if reference_offset.shape != (3,) or not np.all(np.isfinite(reference_offset)):
        raise ShakeBenchOracleError("reference_point_offset_m must be finite length three")
    deck_rotation = deck_rotation.reshape(3, 3)
    ramp_duration = float(np.asarray(payload["ramp_duration_s"]))
    if ramp_duration == 0.0 or query >= ramp_duration:
        ramp, ramp_first, ramp_second = 1.0, 0.0, 0.0
    elif query <= 0.0:
        ramp, ramp_first, ramp_second = 0.0, 0.0, 0.0
    else:
        s = query / ramp_duration
        ramp = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        ramp_first = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / ramp_duration
        ramp_second = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / ramp_duration**2

    predicted_deck = np.zeros(6, dtype=float)
    predicted_velocity_deck = np.zeros(6, dtype=float)
    for axis in range(6):
        for line in np.flatnonzero(mask[axis]):
            amplitude = amplitudes[axis, line]
            omega = omegas[axis, line]
            theta = omega * query + phases[axis, line]
            carrier_qdot = -amplitude / omega * np.cos(theta)
            # Decompose the ramped carrier into sine/cosine coefficients so
            # the complex transfer preserves both phase and ramp derivatives.
            accel_sine = amplitude * ramp - amplitude / omega**2 * ramp_second
            accel_cosine = -2.0 * amplitude / omega * ramp_first
            velocity_sine = -amplitude / omega**2 * ramp_first
            velocity_cosine = -amplitude / omega * ramp
            transfer = complex(relative_transfer_function(omega / (2.0 * np.pi), natural[axis], damping[axis]))
            predicted_deck[axis] += accel_sine * (
                transfer.real * np.sin(theta) + transfer.imag * np.cos(theta)
            ) + accel_cosine * (-transfer.imag * np.sin(theta) + transfer.real * np.cos(theta))
            # qdot*ramp + q*ramp' with the same deterministic frequency-domain
            # operator; the result is used only for reference-point lever-arm
            # terms, not as a second independent disturbance correction.
            predicted_velocity_deck[axis] += velocity_sine * (
                transfer.real * np.sin(theta) + transfer.imag * np.cos(theta)
            ) + velocity_cosine * (-transfer.imag * np.sin(theta) + transfer.real * np.cos(theta))
    predicted = predicted_deck.copy()
    predicted_velocity = predicted_velocity_deck.copy()
    predicted[:3] = deck_rotation.dot(predicted_deck[:3])
    predicted[3:] = deck_rotation.dot(predicted_deck[3:])
    predicted_velocity[:3] = deck_rotation.dot(predicted_velocity_deck[:3])
    predicted_velocity[3:] = deck_rotation.dot(predicted_velocity_deck[3:])
    reference_control = deck_rotation.dot(reference_offset)
    predicted[:3] += np.cross(predicted[3:], reference_control)
    predicted[:3] += np.cross(predicted_velocity[3:], np.cross(predicted_velocity[3:], reference_control))
    if not np.all(np.isfinite(predicted)):
        raise ShakeBenchOracleError("future relative acceleration is non-finite")
    return predicted[:3], predicted[3:]


def _target_pose_from_public_observation(observation: Mapping[str, Any]) -> np.ndarray:
    position = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
    quaternion = np.asarray(observation["goal_frame_quat_robot_base"], dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(position)):
        raise ShakeBenchOracleError("public target frame pose is malformed")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ShakeBenchOracleError("public target frame quaternion is malformed")
    return np.concatenate((position, quaternion / norm))


def _prediction_grid(policy_time_s: float, profile: "OracleControllerProfile") -> np.ndarray:
    offsets = np.asarray(profile.prediction_grid_s, dtype=float)
    return policy_time_s + offsets


def _integrate_prediction(
    initial_pose: np.ndarray,
    initial_twist: np.ndarray,
    initial_acceleration: np.ndarray,
    start_time_s: float,
    prediction_times: np.ndarray,
    predicted_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a short public prediction trajectory from its current state."""

    poses = np.empty((prediction_times.size, 7), dtype=float)
    twists = np.empty((prediction_times.size, 6), dtype=float)
    pose = np.asarray(initial_pose, dtype=float).copy()
    twist = np.asarray(initial_twist, dtype=float).copy()
    previous_acceleration = np.asarray(initial_acceleration, dtype=float).copy()
    rotation = _quat_xyzw_to_matrix(pose[3:])
    previous_time = float(start_time_s)
    for index, absolute_time in enumerate(prediction_times):
        dt = float(absolute_time - previous_time)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ShakeBenchOracleError("prediction time grid must advance from policy time")
        acceleration = np.asarray(predicted_acceleration[index], dtype=float)
        average_acceleration = 0.5 * (previous_acceleration + acceleration)
        pose[:3] += twist[:3] * dt + 0.5 * average_acceleration[:3] * dt**2
        angular_delta = twist[3:] * dt + 0.5 * average_acceleration[3:] * dt**2
        rotation = rotation.dot(_rotation_from_rotvec(angular_delta))
        pose[3:] = T.mat2quat(rotation)
        twist += average_acceleration * dt
        poses[index] = pose
        twists[index] = twist
        previous_acceleration = acceleration
        previous_time = float(absolute_time)
    return poses, twists


def _support_state_from_public_observation(
    observation: Mapping[str, Any], context: WorktableTaskContext
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert public V2 table state to the common target-origin/base frame."""

    deck_pose = np.asarray(observation["deck_pose_in_nominal_frame"], dtype=float)
    table_pose = np.asarray(observation["table_pose_in_deck_frame"], dtype=float)
    table_twist = np.asarray(observation["table_twist_in_deck_frame"], dtype=float)
    table_acceleration = np.asarray(observation["table_accel_in_deck_frame"], dtype=float)
    if (
        deck_pose.shape != (7,)
        or table_pose.shape != (7,)
        or table_twist.shape != (6,)
        or table_acceleration.shape != (6,)
    ):
        raise ShakeBenchOracleError("V2 support state has invalid shapes")
    if not all(np.all(np.isfinite(value)) for value in (deck_pose, table_pose, table_twist, table_acceleration)):
        raise ShakeBenchOracleError("V2 support state must be finite")
    deck_rotation = _quat_xyzw_to_matrix(deck_pose[3:][[1, 2, 3, 0]])
    table_rotation = _quat_xyzw_to_matrix(table_pose[3:][[1, 2, 3, 0]])
    deck_to_table = np.eye(4, dtype=float)
    deck_to_table[:3, :3] = table_rotation
    deck_to_table[:3, 3] = table_pose[:3]
    deck_to_base = np.eye(4, dtype=float)
    deck_to_base[:3, :3] = _quat_xyzw_to_matrix(
        np.asarray(context.deck_to_robot_base_quaternion_wxyz, dtype=float)[[1, 2, 3, 0]]
    )
    deck_to_base[:3, 3] = np.asarray(context.deck_to_robot_base_position_m, dtype=float)
    base_to_deck = np.linalg.inv(deck_to_base)
    table_to_target = np.eye(4, dtype=float)
    table_to_target[:3, 3] = np.asarray(context.target_frame_origin_in_worktable_m, dtype=float)
    base_to_target = base_to_deck.dot(deck_to_table).dot(table_to_target)
    pose = np.concatenate((base_to_target[:3, 3], T.mat2quat(base_to_target[:3, :3])))

    target_offset_deck = table_rotation.dot(np.asarray(context.target_frame_origin_in_worktable_m, dtype=float))
    relative_angular = table_twist[3:]
    target_linear_deck = table_twist[:3] + np.cross(relative_angular, target_offset_deck)
    target_angular_deck = relative_angular
    target_alpha_deck = table_acceleration[3:]
    target_acceleration_deck = table_acceleration[:3]
    target_acceleration_deck += np.cross(target_alpha_deck, target_offset_deck)
    target_acceleration_deck += np.cross(relative_angular, np.cross(relative_angular, target_offset_deck))
    base_rotation_from_deck = base_to_deck[:3, :3]
    twist = np.concatenate(
        (base_rotation_from_deck.dot(target_linear_deck), base_rotation_from_deck.dot(target_angular_deck))
    )
    acceleration = np.concatenate(
        (base_rotation_from_deck.dot(target_acceleration_deck), base_rotation_from_deck.dot(target_alpha_deck))
    )
    return pose, twist, acceleration


def relative_support_motion_from_public_observation(
    observation: Mapping[str, Any], context: WorktableTaskContext | None = None
) -> dict[str, np.ndarray]:
    """Return the V2 public support state in the common target-origin frame."""

    active_context = context or WorktableTaskContext()
    pose, twist, acceleration = _support_state_from_public_observation(observation, active_context)
    return {
        "pose_robot_base": pose,
        "twist_robot_base": twist,
        "acceleration_robot_base": acceleration,
    }


def vibration_estimate_from_public_observation(
    tier: str,
    observation: Mapping[str, Any],
    *,
    future_query_horizon_s: float = 0.100,
    policy_time_s: Optional[float] = None,
    v1_estimator: Optional[V1IMUEstimator] = None,
    future_natural_frequency_hz: tuple[float, ...] = DEFAULT_FUTURE_ISOLATOR_FN_HZ,
    future_damping_ratio: tuple[float, ...] = DEFAULT_FUTURE_ISOLATOR_ZETA,
    future_deck_to_control_rotation: tuple[float, ...] = DEFAULT_DECK_TO_CONTROL_ROTATION,
    task_context: WorktableTaskContext | None = None,
    relative_kinematics: Any = None,
    profile: Optional["OracleControllerProfile"] = None,
) -> VibrationEstimate:
    """Derive the same common physical quantity from a public tier payload."""

    if tier not in OBSERVATION_TIERS:
        raise ShakeBenchOracleError("tier must be V0, V1, V2, or V3")
    if not isinstance(observation, Mapping):
        raise ShakeBenchOracleError("observation must be a mapping")
    forbidden = [key for key in observation if str(key).startswith("privileged_")]
    if forbidden:
        raise ShakeBenchOracleError("privileged keys are not valid controller input")
    expected = set(TIER_POLICY_KEYS[tier])
    missing = expected - set(observation)
    if missing:
        raise ShakeBenchOracleError("missing tier payload: " + ", ".join(sorted(missing)))
    profile = profile or OracleControllerProfile()
    context = task_context or WorktableTaskContext()
    if not np.isfinite(float(future_query_horizon_s)) or float(future_query_horizon_s) <= 0.0:
        raise ShakeBenchOracleError("future_query_horizon_s must be finite and positive")
    policy_time = float(np.asarray(observation.get("episode_time_s", 0.0) if policy_time_s is None else policy_time_s))
    if not np.isfinite(policy_time) or policy_time < 0.0:
        raise ShakeBenchOracleError("policy_time_s must be finite and non-negative")
    current_pose = _target_pose_from_public_observation(observation)
    current_twist = np.zeros(6, dtype=float)
    current_acceleration = np.zeros(6, dtype=float)
    measurement_time = policy_time
    source = "public_neutral"
    validity = True
    prediction_validity = True
    confidence = 0.0
    legacy_current_linear = None
    if (
        profile.public_history_estimator_enabled
        and isinstance(relative_kinematics, RelativeKinematicsSnapshot)
        and relative_kinematics.history_valid
    ):
        # The target pose history is common State information, so V0 may use
        # this explicitly named low-confidence baseline without receiving any
        # dedicated vibration field.
        current_pose = relative_kinematics.target_frame_pose_robot_base.copy()
        if tier == "V0":
            current_twist = relative_kinematics.target_frame_twist_robot_base.copy()
            current_acceleration = relative_kinematics.target_frame_acceleration_robot_base.copy()
            source = "public_task_history_baseline"
            confidence = 0.20

    if tier == "V1":
        imu = np.asarray(observation["deck_imu_window"], dtype=float)
        if imu.shape != (10, 6):
            raise ShakeBenchOracleError("V1 deck_imu_window must have shape (10, 6)")
        if v1_estimator is None:
            v1_estimator = V1IMUEstimator(profile)
        motion, angular, measurement_time = v1_estimator.estimate(
            imu,
            float(np.asarray(observation["deck_imu_dt_s"])),
            policy_time_s=policy_time,
        )
        # The IMU observes the rigid deck/base.  A causal, fixed-frequency
        # relative isolator response maps it to the same table/target support
        # quantity used by V2; unknown transient state is represented by a
        # confidence below one rather than by a different semantic field.
        transfer = abs(
            complex(
                relative_transfer_function(
                    profile.v1_model_frequency_hz,
                    profile.future_isolator_fn_hz[0],
                    profile.future_isolator_zeta[0],
                )
            )
        )
        current_acceleration[:3] = motion * transfer
        current_acceleration[3:] = angular * transfer
        legacy_current_linear = motion.copy()
        if (
            profile.public_history_estimator_enabled
            and isinstance(relative_kinematics, RelativeKinematicsSnapshot)
            and relative_kinematics.history_valid
        ):
            current_twist = relative_kinematics.target_frame_twist_robot_base.copy()
        source = "causal_noisy_delayed_imu_relative_model"
        confidence = 0.35
    elif tier in {"V2", "V3"}:
        legacy_current_linear = np.asarray(observation["table_accel_in_deck_frame"], dtype=float)[:3].copy()
        current_pose, current_twist, current_acceleration = _support_state_from_public_observation(observation, context)
        source = "current_realized_support" if tier == "V2" else "current_realized_support_plus_authored_preview"
        confidence = 1.0

    prediction_times = _prediction_grid(policy_time, profile)
    predicted_acceleration = np.zeros((prediction_times.size, 6), dtype=float)
    if tier == "V3":
        try:
            for index, query_time in enumerate(prediction_times):
                linear, angular = future_relative_acceleration_from_public_program(
                    observation,
                    float(query_time),
                    natural_frequency_hz=future_natural_frequency_hz,
                    damping_ratio=future_damping_ratio,
                    deck_to_control_rotation=future_deck_to_control_rotation,
                    reference_point_offset_m=context.target_frame_origin_in_worktable_m,
                )
                predicted_acceleration[index] = np.concatenate((linear, angular))
        except (ShakeBenchOracleError, ValueError, TypeError):
            # A malformed model/program invalidates only the preview.  The
            # current V2-equivalent estimate remains usable and the law's
            # prediction-validity gate disables just its future correction.
            predicted_acceleration.fill(0.0)
            prediction_validity = False
    predicted_pose, predicted_twist = _integrate_prediction(
        current_pose,
        current_twist,
        current_acceleration,
        policy_time,
        prediction_times,
        predicted_acceleration,
    )
    actual_latency = max(0.0, policy_time - float(measurement_time))
    return VibrationEstimate(
        tier=tier,
        current_relative_pose_robot_base=current_pose,
        current_relative_twist_robot_base=current_twist,
        current_relative_acceleration_robot_base=current_acceleration,
        measurement_timestamp_s=float(measurement_time),
        policy_timestamp_s=policy_time,
        prediction_timestamps_s=prediction_times,
        predicted_relative_pose_robot_base=predicted_pose,
        predicted_relative_twist_robot_base=predicted_twist,
        predicted_relative_acceleration_robot_base=predicted_acceleration,
        latency_s=actual_latency,
        validity=validity,
        prediction_validity=prediction_validity,
        confidence=confidence,
        source=source,
        future_program_available=tier == "V3",
        context_sha256=context.sha256,
        legacy_current_linear_accel_m_s2=legacy_current_linear,
    )


@dataclass(frozen=True)
class OracleControllerProfile:
    """One profile applied unchanged to V0--V3."""

    profile_id: str = "shakebench.reference_oracle.v3"
    policy_rate_hz: float = 20.0
    position_action_range_m: float = 0.05
    orientation_action_range_rad: float = 0.5
    # The waypoint is measured from the Can centre; 0.16 m leaves the full
    # finger-pad support points above the Can collision top during lateral
    # alignment, before the vertical descend is allowed.
    approach_height_m: float = 0.16
    grasp_height_m: float = 0.085
    transport_height_m: float = 0.16
    placement_height_m: float = 0.105
    position_tolerance_m: float = 0.018
    approach_tolerance_m: float = 0.015
    descend_tolerance_m: float = 0.060
    descend_vertical_tolerance_m: float = 0.020
    settle_s: float = 0.25
    grasp_s: float = 0.75
    prelift_height_m: float = 0.010
    prelift_s: float = 0.40
    prelift_min_follow_m: float = 0.004
    prelift_geometry_tolerance_m: float = 0.010
    prelift_required_samples: int = 2
    clearance_lift_height_m: float = 0.075
    clearance_lift_s: float = 0.30
    align_settle_s: float = 0.20
    align_required_samples: int = 3
    align_eef_speed_limit_m_s: float = 0.025
    table_edge_safety_margin_m: float = 0.008
    tool_clearance_m: float = 0.012
    anchor_drift_tolerance_m: float = 0.010
    lift_s: float = 0.75
    release_s: float = 0.30
    verify_s: float = 0.55
    # Panda's public single-channel action contract is fixed: -1=open,
    # +1=close.  Use explicit names so phase wiring cannot invert the hold.
    gripper_open_action: float = -1.0
    gripper_close_action: float = 1.0
    completion_evaluator_settle_s: float = 1.0
    phase_deadline_s: float = 8.0
    current_linear_accel_gain_s2: float = 0.002
    current_angular_velocity_gain_s: float = 0.002
    future_query_horizon_s: float = 0.100
    future_linear_accel_gain_s2: float = 0.0005
    future_angular_accel_gain_s2: float = 0.0005
    max_current_linear_compensation_m: float = 0.010
    max_current_angular_compensation_rad: float = 0.010
    max_future_linear_compensation_m: float = 0.010
    max_future_angular_compensation_rad: float = 0.010
    imu_filter_group_delay_s: float = 0.00487
    imu_delivery_delay_s: float = 0.005
    imu_end_to_end_latency_s: float = 0.00987
    descend_position_action_limit: float = 0.30
    transport_position_action_limit: float = 0.35
    public_workspace_radius_m: float = 1.20
    can_collision_envelope_radius_m: float = 0.02509177806572465
    # Pad geometry is measured in the public observation.  The nominal
    # gripper-to-Can transform is shared robot / Can geometry, never a state
    # specific waypoint.
    # Compiled Panda pad midpoint is approximately z=0.0934 m in the EEF
    # frame; the Can centre sits just below that midpoint when bilaterally
    # captured.  This geometry-derived transform is independent of state ID.
    expected_eef_can_translation_m: tuple[float, ...] = (-0.005, 0.0, 0.080)
    grasp_hold_min_opening_rad: float = 0.006
    # Can diameter (2 * collision radius) plus a 10 mm compiled pad/measurement
    # allowance; bilateral geometry and expected-transform gates remain mandatory.
    grasp_hold_max_opening_rad: float = 0.0602
    grasp_establishment_min_opening_rad: float = 0.006
    grasp_corridor_margin_m: float = 0.004
    grasp_expected_transform_tolerance_m: float = 0.018
    grasp_wrench_confidence_min_N: float = 0.5
    # 18 mm is below the 25.0918 mm collision radius while leaving the
    # measured policy/actuator noise margin for a rigid carry.
    grasp_slip_tolerance_m: float = 0.018
    grasp_severe_slip_fraction_of_can_radius: float = 0.75
    grasp_slip_tolerance_rad: float = 0.35
    grasp_loss_confirm_samples: int = 2
    recovery_table_height_tolerance_m: float = 0.055
    recovery_max_downward_speed_m_s: float = 0.12
    recovery_lower_s: float = 0.30
    recovery_hold_s: float = 0.30
    recovery_velocity_brake_s: float = 0.0
    recovery_open_s: float = 0.20
    recovery_retreat_s: float = 0.30
    recovery_settle_s: float = 1.20
    recovery_stable_samples: int = 3
    placement_support_tolerance_m: float = 0.008
    public_risk_margin_m: float = 0.012
    verify_linear_speed_limit_m_s: float = 0.020
    verify_angular_speed_limit_rad_s: float = 0.20
    verify_stability_samples: int = 2
    future_isolator_fn_hz: tuple[float, ...] = DEFAULT_FUTURE_ISOLATOR_FN_HZ
    future_isolator_zeta: tuple[float, ...] = DEFAULT_FUTURE_ISOLATOR_ZETA
    future_deck_to_control_rotation: tuple[float, ...] = DEFAULT_DECK_TO_CONTROL_ROTATION
    # R5 prediction grid and the causal V1 model operating point are frozen
    # profile inputs, not values selected from positive-Gamma outcomes.
    prediction_grid_s: tuple[float, ...] = (0.020, 0.040, 0.060, 0.080, 0.100)
    v1_model_frequency_hz: float = 4.0
    diagnostic_mode: str = "main"
    compensation_enabled: bool = True
    current_state_compensation_enabled: bool = True
    preview_enabled: bool = True
    public_history_estimator_enabled: bool = True
    recovery_budget: int = 3

    def __post_init__(self) -> None:
        validate_motion_capability_map()
        for key, value in asdict(self).items():
            if key in {
                "profile_id",
                "recovery_budget",
                "verify_stability_samples",
                "prelift_required_samples",
                "grasp_loss_confirm_samples",
                "align_required_samples",
                "recovery_stable_samples",
                "recovery_velocity_brake_s",
                "future_isolator_fn_hz",
                "future_isolator_zeta",
                "future_deck_to_control_rotation",
                "prediction_grid_s",
                "expected_eef_can_translation_m",
                "diagnostic_mode",
                "compensation_enabled",
                "current_state_compensation_enabled",
                "preview_enabled",
                "public_history_estimator_enabled",
            }:
                continue
            if key in {"gripper_open_action", "gripper_close_action"}:
                if not np.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
                    raise ShakeBenchOracleError(f"profile {key} must lie in [-1, 1]")
            elif not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ShakeBenchOracleError(f"profile {key} must be finite and positive")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ShakeBenchOracleError("profile_id must be nonempty")
        if self.diagnostic_mode not in DIAGNOSTIC_MODES:
            raise ShakeBenchOracleError("diagnostic_mode is not registered")
        for name in (
            "compensation_enabled",
            "current_state_compensation_enabled",
            "preview_enabled",
            "public_history_estimator_enabled",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ShakeBenchOracleError(f"{name} must be boolean")
        if not isinstance(self.verify_stability_samples, int) or self.verify_stability_samples < 2:
            raise ShakeBenchOracleError("verify_stability_samples must be an integer >= 2")
        for name, expected_shape, positive, nonnegative in (
            ("future_isolator_fn_hz", (6,), True, False),
            ("future_isolator_zeta", (6,), False, True),
            ("future_deck_to_control_rotation", (9,), False, False),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ShakeBenchOracleError(f"profile {name} has invalid shape or non-finite values")
            if positive and np.any(value <= 0.0):
                raise ShakeBenchOracleError(f"profile {name} must be positive")
            if nonnegative and np.any(value < 0.0):
                raise ShakeBenchOracleError(f"profile {name} must be non-negative")
        if not np.isclose(
            self.imu_filter_group_delay_s + self.imu_delivery_delay_s,
            self.imu_end_to_end_latency_s,
            rtol=0.0,
            atol=5.0e-5,
        ):
            raise ShakeBenchOracleError("IMU end-to-end latency must equal filter plus delivery delay")
        if not np.isclose(
            self.imu_delivery_delay_s,
            CANONICAL_IMU_PROFILE.delivery_delay_samples * CANONICAL_IMU_PROFILE.dt_s,
            atol=1e-12,
        ):
            raise ShakeBenchOracleError("IMU delivery delay must equal one canonical sample interval")
        if self.recovery_budget < 0:
            raise ShakeBenchOracleError("recovery_budget must be non-negative")
        prediction_grid = np.asarray(self.prediction_grid_s, dtype=float)
        if (
            prediction_grid.ndim != 1
            or prediction_grid.size == 0
            or not np.all(np.isfinite(prediction_grid))
            or np.any(prediction_grid <= 0.0)
            or np.any(np.diff(prediction_grid) <= 0.0)
        ):
            raise ShakeBenchOracleError("prediction_grid_s must be a strictly increasing positive grid")
        if not np.isfinite(self.v1_model_frequency_hz) or self.v1_model_frequency_hz <= 0.0:
            raise ShakeBenchOracleError("v1_model_frequency_hz must be finite and positive")
        if self.grasp_hold_max_opening_rad < self.grasp_hold_min_opening_rad:
            raise ShakeBenchOracleError("grasp aperture upper bound must exceed lower bound")
        if self.gripper_open_action != -1.0 or self.gripper_close_action != 1.0:
            raise ShakeBenchOracleError("Panda gripper contract requires open=-1.0 and close=+1.0")
        expected_translation = np.asarray(self.expected_eef_can_translation_m, dtype=float)
        if expected_translation.shape != (3,) or not np.all(np.isfinite(expected_translation)):
            raise ShakeBenchOracleError("expected_eef_can_translation_m must be finite length three")
        if self.prelift_required_samples < 2 or self.grasp_loss_confirm_samples < 2:
            raise ShakeBenchOracleError("prelift and loss confirmation require at least two samples")
        if self.align_required_samples < 2 or self.recovery_stable_samples < 2:
            raise ShakeBenchOracleError("alignment and recovery confirmation require at least two samples")
        if not np.isfinite(self.recovery_velocity_brake_s) or self.recovery_velocity_brake_s < 0.0:
            raise ShakeBenchOracleError("recovery_velocity_brake_s must be finite and non-negative")
        if not 0.0 < self.grasp_severe_slip_fraction_of_can_radius <= 1.0:
            raise ShakeBenchOracleError("severe slip fraction must lie in (0, 1]")

    @property
    def sha256(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile_sha256"] = self.sha256
        return result


def profile_for_diagnostic_mode(
    mode: str, base_profile: Optional[OracleControllerProfile] = None
) -> OracleControllerProfile:
    """Create a hashed, explicitly named diagnostic profile."""

    if mode not in DIAGNOSTIC_MODES:
        raise ShakeBenchOracleError("diagnostic_mode is not registered")
    base = base_profile or OracleControllerProfile()
    if mode == "main":
        return base
    flags = {
        "compensation_enabled": base.compensation_enabled,
        "current_state_compensation_enabled": base.current_state_compensation_enabled,
        "preview_enabled": base.preview_enabled,
        "public_history_estimator_enabled": base.public_history_estimator_enabled,
    }
    if mode in {"task_executive_only", "compensation_off"}:
        flags["compensation_enabled"] = False
    elif mode == "common_public_history":
        flags["public_history_estimator_enabled"] = True
    elif mode == "current_state_compensation":
        flags["current_state_compensation_enabled"] = True
        flags["preview_enabled"] = False
    elif mode == "preview_off":
        flags["preview_enabled"] = False
    elif mode == "preview_on":
        flags["preview_enabled"] = True
    return dataclasses.replace(
        base,
        profile_id=f"{base.profile_id}.{mode}",
        diagnostic_mode=mode,
        **flags,
    )


@dataclass(frozen=True)
class MotionCapabilityPolicy:
    """Action limits selected from one phase capability."""

    capability: MotionCapability
    max_translation_normalized: float
    max_orientation_normalized: float
    gripper_action: float
    vibration_feedforward_allowed: bool


def motion_capability_policy(phase: TaskPhase, profile: OracleControllerProfile) -> MotionCapabilityPolicy:
    """Resolve all phase action constraints at one centralized seam."""

    capability = motion_capability_for_phase(phase)
    translation_limits = {
        MotionCapability.HOLD: 1.0,
        MotionCapability.FREE_SPACE: 1.0,
        MotionCapability.CLEARANCE_TRANSLATE: 0.30,
        MotionCapability.CONTACT_APPROACH: profile.descend_position_action_limit,
        MotionCapability.GRIPPER_CLOSE: profile.descend_position_action_limit,
        MotionCapability.OBJECT_HELD: profile.transport_position_action_limit,
        MotionCapability.RECOVERY_LOWER: profile.descend_position_action_limit,
        MotionCapability.RECOVERY_RETREAT: profile.transport_position_action_limit,
        MotionCapability.TERMINAL: 1.0 if phase is TaskPhase.COMPLETE else 0.0,
    }
    gripper_open_phases = {
        TaskPhase.SETTLE,
        TaskPhase.APPROACH,
        TaskPhase.CLEARANCE_LIFT,
        TaskPhase.LATERAL_ALIGN_ABOVE_CAN,
        TaskPhase.ALIGN_SETTLE,
        TaskPhase.VERTICAL_DESCEND,
        TaskPhase.DESCEND,
        TaskPhase.RELEASE,
        TaskPhase.VERIFY,
        TaskPhase.RECOVERY_OPEN,
        TaskPhase.CLEARANCE_RETREAT,
        TaskPhase.RE_ALIGN,
        TaskPhase.COMPLETE,
        TaskPhase.FAILED,
    }
    gripper = profile.gripper_open_action if phase in gripper_open_phases else profile.gripper_close_action
    return MotionCapabilityPolicy(
        capability=capability,
        max_translation_normalized=float(translation_limits[capability]),
        max_orientation_normalized=(
            1.0 if phase is TaskPhase.COMPLETE else (0.0 if capability is MotionCapability.TERMINAL else 1.0)
        ),
        gripper_action=gripper,
        vibration_feedforward_allowed=capability is not MotionCapability.TERMINAL,
    )


@dataclass(frozen=True)
class ControlLawOutput:
    desired_delta_base: np.ndarray
    compensated_delta_base: np.ndarray
    estimate: VibrationEstimate

    def __post_init__(self) -> None:
        for key in ("desired_delta_base", "compensated_delta_base"):
            value = np.asarray(getattr(self, key), dtype=float)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ShakeBenchOracleError(f"{key} must be a finite six-vector")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, key, value)


class SharedVibrationControlLaw:
    """Single tier-invariant law: providers only change the typed estimate."""

    def __init__(self, profile: OracleControllerProfile):
        self.profile = profile

    def apply(self, desired_delta_base: np.ndarray, estimate: VibrationEstimate) -> ControlLawOutput:
        desired = np.asarray(desired_delta_base, dtype=float)
        if desired.shape != (6,) or not np.all(np.isfinite(desired)):
            raise ShakeBenchOracleError("desired_delta_base must be a finite six-vector")
        # Estimates already have one common frame/reference-point meaning. The
        # law must never infer their representation from tier labels or subtract
        # gravity a second time.
        if estimate.validity and self.profile.compensation_enabled:
            if self.profile.current_state_compensation_enabled:
                linear_correction = (
                    -self.profile.current_linear_accel_gain_s2 * estimate.current_relative_acceleration_robot_base[:3]
                )
                linear_correction = np.clip(
                    linear_correction,
                    -self.profile.max_current_linear_compensation_m,
                    self.profile.max_current_linear_compensation_m,
                )
                angular_correction = (
                    -self.profile.current_angular_velocity_gain_s * estimate.current_relative_twist_robot_base[3:]
                )
                angular_correction = np.clip(
                    angular_correction,
                    -self.profile.max_current_angular_compensation_rad,
                    self.profile.max_current_angular_compensation_rad,
                )
            else:
                linear_correction = np.zeros(3, dtype=float)
                angular_correction = np.zeros(3, dtype=float)
            if self.profile.preview_enabled and estimate.prediction_validity:
                future_linear_correction = -self.profile.future_linear_accel_gain_s2 * estimate.future_linear_accel_m_s2
                future_linear_correction = np.clip(
                    future_linear_correction,
                    -self.profile.max_future_linear_compensation_m,
                    self.profile.max_future_linear_compensation_m,
                )
                future_angular_correction = (
                    -self.profile.future_angular_accel_gain_s2 * estimate.future_angular_accel_rad_s2
                )
                future_angular_correction = np.clip(
                    future_angular_correction,
                    -self.profile.max_future_angular_compensation_rad,
                    self.profile.max_future_angular_compensation_rad,
                )
            else:
                future_linear_correction = np.zeros(3, dtype=float)
                future_angular_correction = np.zeros(3, dtype=float)
        else:
            linear_correction = np.zeros(3, dtype=float)
            angular_correction = np.zeros(3, dtype=float)
            future_linear_correction = np.zeros(3, dtype=float)
            future_angular_correction = np.zeros(3, dtype=float)
        compensated = desired.copy()
        compensated[:3] += linear_correction + future_linear_correction
        compensated[3:] += angular_correction + future_angular_correction
        return ControlLawOutput(desired_delta_base=desired, compensated_delta_base=compensated, estimate=estimate)


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import xyzw_to_matrix

    return xyzw_to_matrix(quaternion, name="goal quaternion", error_type=ShakeBenchOracleError)



def _rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import matrix_to_rotation_vector

    return matrix_to_rotation_vector(rotation, error_type=ShakeBenchOracleError)



def target_local_to_robot_base(observation: Mapping[str, Any], local_position: np.ndarray) -> np.ndarray:
    """Transform a target-local point through the complete public goal frame."""

    frame_position = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
    rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
    local_position = np.asarray(local_position, dtype=float)
    if frame_position.shape != (3,) or local_position.shape != (3,):
        raise ShakeBenchOracleError("goal frame and local target positions must be length three")
    return frame_position + rotation.dot(local_position)


def _worktable_transform_robot_base(observation: Mapping[str, Any], context: WorktableTaskContext) -> np.ndarray:
    """Return the current public worktable-frame pose in robot-base."""

    target_position = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
    target_rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
    origin = np.asarray(context.target_frame_origin_in_worktable_m, dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = target_rotation
    transform[:3, 3] = target_position - target_rotation.dot(origin)
    return transform


def worktable_local_to_robot_base(
    observation: Mapping[str, Any], local_position: np.ndarray, context: WorktableTaskContext | None = None
) -> np.ndarray:
    """Transform a point from the current worktable frame to robot base."""

    context = context or WorktableTaskContext()
    point = np.asarray(local_position, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ShakeBenchOracleError("worktable local point must be finite length three")
    transform = _worktable_transform_robot_base(observation, context)
    return transform[:3, 3] + transform[:3, :3].dot(point)


def robot_base_to_worktable_local(
    observation: Mapping[str, Any], robot_base_position: np.ndarray, context: WorktableTaskContext | None = None
) -> np.ndarray:
    """Transform a robot-base point into the current worktable frame."""

    context = context or WorktableTaskContext()
    point = np.asarray(robot_base_position, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ShakeBenchOracleError("robot-base point must be finite length three")
    transform = _worktable_transform_robot_base(observation, context)
    return transform[:3, :3].T.dot(point - transform[:3, 3])


def target_local_to_worktable_local(
    local_position: np.ndarray, context: WorktableTaskContext | None = None
) -> np.ndarray:
    """Convert a target-local point to the rigid worktable frame."""

    context = context or WorktableTaskContext()
    point = np.asarray(local_position, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ShakeBenchOracleError("target local point must be finite length three")
    return point + np.asarray(context.target_frame_origin_in_worktable_m, dtype=float)


def _pose_transform_from_observation(
    observation: Mapping[str, Any], position_key: str, quaternion_key: str
) -> np.ndarray:
    position = np.asarray(observation[position_key], dtype=float)
    quaternion = np.asarray(observation[quaternion_key], dtype=float)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
    ):
        raise ShakeBenchOracleError(f"{position_key}/{quaternion_key} must be finite pose fields")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quat_xyzw_to_matrix(quaternion)
    transform[:3, 3] = position
    return transform


@dataclass(frozen=True)
class RelativeKinematicsSnapshot:
    """Public relative SE(3) state at one policy timestamp."""

    worktable_can_pose: np.ndarray
    worktable_can_twist: np.ndarray
    worktable_can_acceleration: np.ndarray
    target_can_pose: np.ndarray
    target_can_twist: np.ndarray
    target_can_acceleration: np.ndarray
    target_frame_pose_robot_base: np.ndarray
    target_frame_twist_robot_base: np.ndarray
    target_frame_acceleration_robot_base: np.ndarray
    timestamp_s: float
    dt_s: float
    history_valid: bool
    reset_generation: int
    reset_reason: Optional[str] = None

    def __post_init__(self) -> None:
        shapes = {
            "worktable_can_pose": ((7,), self.worktable_can_pose),
            "target_can_pose": ((7,), self.target_can_pose),
            "target_frame_pose_robot_base": ((7,), self.target_frame_pose_robot_base),
            "worktable_can_twist": ((6,), self.worktable_can_twist),
            "worktable_can_acceleration": ((6,), self.worktable_can_acceleration),
            "target_can_twist": ((6,), self.target_can_twist),
            "target_can_acceleration": ((6,), self.target_can_acceleration),
            "target_frame_twist_robot_base": ((6,), self.target_frame_twist_robot_base),
            "target_frame_acceleration_robot_base": ((6,), self.target_frame_acceleration_robot_base),
        }
        for name, (shape, value) in shapes.items():
            array = np.asarray(value, dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ShakeBenchOracleError(f"{name} must be finite with shape {shape}")
            array = array.copy()
            if shape == (7,):
                norm = float(np.linalg.norm(array[3:]))
                if norm <= 1.0e-12:
                    raise ShakeBenchOracleError(f"{name} quaternion must be nonzero")
                array[3:] /= norm
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not np.isfinite(float(self.timestamp_s)) or self.timestamp_s < 0.0:
            raise ShakeBenchOracleError("relative kinematics timestamp must be finite and non-negative")
        if not np.isfinite(float(self.dt_s)) or self.dt_s < 0.0:
            raise ShakeBenchOracleError("relative kinematics dt must be finite and non-negative")
        if not isinstance(self.reset_generation, int) or self.reset_generation < 1:
            raise ShakeBenchOracleError("relative kinematics reset generation must be positive")

    @property
    def worktable_can_linear_speed_m_s(self) -> float:
        return float(np.linalg.norm(self.worktable_can_twist[:3]))

    @property
    def worktable_can_angular_speed_rad_s(self) -> float:
        return float(np.linalg.norm(self.worktable_can_twist[3:]))

    @property
    def target_can_linear_speed_m_s(self) -> float:
        return float(np.linalg.norm(self.target_can_twist[:3]))

    @property
    def target_can_angular_speed_rad_s(self) -> float:
        return float(np.linalg.norm(self.target_can_twist[3:]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "worktable_can_pose": self.worktable_can_pose.tolist(),
            "worktable_can_twist": self.worktable_can_twist.tolist(),
            "worktable_can_acceleration": self.worktable_can_acceleration.tolist(),
            "target_can_pose": self.target_can_pose.tolist(),
            "target_can_twist": self.target_can_twist.tolist(),
            "target_can_acceleration": self.target_can_acceleration.tolist(),
            "target_frame_pose_robot_base": self.target_frame_pose_robot_base.tolist(),
            "target_frame_twist_robot_base": self.target_frame_twist_robot_base.tolist(),
            "target_frame_acceleration_robot_base": self.target_frame_acceleration_robot_base.tolist(),
            "timestamp_s": self.timestamp_s,
            "dt_s": self.dt_s,
            "history_valid": bool(self.history_valid),
            "reset_generation": self.reset_generation,
            "reset_reason": self.reset_reason,
        }


class PublicRelativeKinematicsTracker:
    """Deep public-only seam for moving-frame stability and anchor checks."""

    def __init__(self, context: WorktableTaskContext | None = None) -> None:
        self.context = context or WorktableTaskContext()
        self.reset_generation = 0
        self.reset()

    def reset(self, reason: Optional[str] = "episode_reset") -> None:
        self._previous_worktable_can: Optional[np.ndarray] = None
        self._previous_target_can: Optional[np.ndarray] = None
        self._previous_target_frame: Optional[np.ndarray] = None
        self._previous_worktable_can_twist: Optional[np.ndarray] = None
        self._previous_target_can_twist: Optional[np.ndarray] = None
        self._previous_target_frame_twist: Optional[np.ndarray] = None
        self._previous_time_s: Optional[float] = None
        self.reset_generation += 1
        self._last_reset_reason = reason

    @staticmethod
    def _relative_transform(frame: np.ndarray, child: np.ndarray) -> np.ndarray:
        return np.linalg.inv(frame).dot(child)

    @staticmethod
    def _pose_from_transform(transform: np.ndarray) -> np.ndarray:
        return np.concatenate((transform[:3, 3], np.asarray(T.mat2quat(transform[:3, :3]), dtype=float)))

    @staticmethod
    def _finite_time(time_s: Any) -> float:
        time = float(time_s)
        if not np.isfinite(time) or time < 0.0:
            raise ShakeBenchOracleError("relative kinematics time must be finite and non-negative")
        return time

    def update(self, observation: Mapping[str, Any], time_s: float) -> RelativeKinematicsSnapshot:
        if not isinstance(observation, Mapping):
            raise ShakeBenchOracleError("relative kinematics observation must be a mapping")
        time = self._finite_time(time_s)
        target_frame = _pose_transform_from_observation(
            observation, "goal_frame_pos_robot_base", "goal_frame_quat_robot_base"
        )
        can = _pose_transform_from_observation(observation, "can_pos_robot_base", "can_quat_robot_base")
        worktable = _worktable_transform_robot_base(observation, self.context)
        worktable_can = self._relative_transform(worktable, can)
        target_can = self._relative_transform(target_frame, can)
        history_valid = (
            self._previous_worktable_can is not None
            and self._previous_target_can is not None
            and self._previous_target_frame is not None
            and self._previous_time_s is not None
        )
        reset_reason = None
        if history_valid and time <= self._previous_time_s + 1.0e-12:
            self.reset("non_monotonic_time")
            history_valid = False
            reset_reason = "non_monotonic_time"
        if not history_valid:
            dt = 0.0
            worktable_twist = np.zeros(6, dtype=float)
            target_twist = np.zeros(6, dtype=float)
            target_frame_twist = np.zeros(6, dtype=float)
            worktable_acceleration = np.zeros(6, dtype=float)
            target_acceleration = np.zeros(6, dtype=float)
            target_frame_acceleration = np.zeros(6, dtype=float)
            if reset_reason is None:
                reset_reason = self._last_reset_reason
        else:
            dt = time - float(self._previous_time_s)
            worktable_twist = self._pose_finite_difference(self._previous_worktable_can, worktable_can, dt)
            target_twist = self._pose_finite_difference(self._previous_target_can, target_can, dt)
            target_frame_twist = self._pose_finite_difference(self._previous_target_frame, target_frame, dt)
            worktable_acceleration = self._vector_finite_difference(
                self._previous_worktable_can_twist, worktable_twist, dt
            )
            target_acceleration = self._vector_finite_difference(self._previous_target_can_twist, target_twist, dt)
            target_frame_acceleration = self._vector_finite_difference(
                self._previous_target_frame_twist, target_frame_twist, dt
            )
        self._previous_worktable_can = worktable_can.copy()
        self._previous_target_can = target_can.copy()
        self._previous_target_frame = target_frame.copy()
        self._previous_worktable_can_twist = worktable_twist.copy()
        self._previous_target_can_twist = target_twist.copy()
        self._previous_target_frame_twist = target_frame_twist.copy()
        self._previous_time_s = time
        return RelativeKinematicsSnapshot(
            worktable_can_pose=self._pose_from_transform(worktable_can),
            worktable_can_twist=worktable_twist,
            worktable_can_acceleration=worktable_acceleration,
            target_can_pose=self._pose_from_transform(target_can),
            target_can_twist=target_twist,
            target_can_acceleration=target_acceleration,
            target_frame_pose_robot_base=self._pose_from_transform(target_frame),
            target_frame_twist_robot_base=target_frame_twist,
            target_frame_acceleration_robot_base=target_frame_acceleration,
            timestamp_s=time,
            dt_s=dt,
            history_valid=history_valid,
            reset_generation=self.reset_generation,
            reset_reason=reset_reason,
        )

    @staticmethod
    def _pose_finite_difference(previous: Optional[np.ndarray], current: np.ndarray, dt: float) -> np.ndarray:
        if previous is None or dt <= 0.0:
            return np.zeros(6, dtype=float)
        position_delta = current[:3, 3] - previous[:3, 3]
        angular_delta = _rotation_vector_from_matrix(previous[:3, :3].T.dot(current[:3, :3]))
        return np.concatenate((position_delta / dt, angular_delta / dt))

    @staticmethod
    def _vector_finite_difference(previous: Optional[np.ndarray], current: np.ndarray, dt: float) -> np.ndarray:
        if previous is None or dt <= 0.0:
            return np.zeros(6, dtype=float)
        return (np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)) / dt

class ShakeBenchOracleController:
    """One public-input controller emitting robosuite's normalized 7D action."""

    def __init__(
        self,
        tier: str,
        profile: Optional[OracleControllerProfile] = None,
        task_context: WorktableTaskContext | Mapping[str, Any] | None = None,
    ):
        if tier not in OBSERVATION_TIERS:
            raise ShakeBenchOracleError("tier must be V0, V1, V2, or V3")
        self.tier = tier
        self.profile = profile or OracleControllerProfile()
        self.task_context = (
            task_context
            if isinstance(task_context, WorktableTaskContext)
            else WorktableTaskContext.from_mapping(task_context)
        )
        self.control_law = SharedVibrationControlLaw(self.profile)
        from robosuite.utils.oracle_executive import TaskExecutive

        self.executive = TaskExecutive(self.profile, self.task_context)
        self.v1_estimator = V1IMUEstimator(self.profile)
        self._last_policy_time_s: Optional[float] = None
        self.last_trace: Optional[dict[str, Any]] = None

    @property
    def profile_sha256(self) -> str:
        return self.profile.sha256

    @property
    def task_context_sha256(self) -> str:
        return self.task_context.sha256

    @property
    def abort_requested(self) -> bool:
        """Return the controller's structured decision to stop acting."""

        return self.executive.abort_reason is not None

    @property
    def controller_context_hash(self) -> str:
        return hashlib.sha256(f"{self.profile_sha256}:{self.task_context_sha256}".encode("ascii")).hexdigest()

    def reset(self) -> None:
        self.executive.reset()
        self.v1_estimator.reset()
        self._last_policy_time_s = None
        self.last_trace = None

    def action(self, observation: Mapping[str, Any], *, time_s: Optional[float] = None) -> np.ndarray:
        expected = set(TIER_POLICY_KEYS[self.tier])
        missing = expected - set(observation)
        if missing:
            raise ShakeBenchOracleError("missing tier payload: " + ", ".join(sorted(missing)))
        if time_s is None:
            time_s = float(observation.get("episode_time_s", 0.0))
        policy_time = float(time_s)
        if self._last_policy_time_s is not None and policy_time <= self._last_policy_time_s + 1.0e-12:
            self.reset()
        desired_delta, gripper = self.executive.command(observation, policy_time)
        estimate = vibration_estimate_from_public_observation(
            self.tier,
            observation,
            future_query_horizon_s=self.profile.future_query_horizon_s,
            policy_time_s=policy_time,
            v1_estimator=self.v1_estimator,
            future_natural_frequency_hz=self.profile.future_isolator_fn_hz,
            future_damping_ratio=self.profile.future_isolator_zeta,
            future_deck_to_control_rotation=self.profile.future_deck_to_control_rotation,
            task_context=self.task_context,
            relative_kinematics=self.executive._relative_kinematics,
            profile=self.profile,
        )
        law_output = self.control_law.apply(desired_delta, estimate)
        capability_policy = motion_capability_policy(self.executive.phase, self.profile)
        pre_capability_delta = law_output.compensated_delta_base.copy()
        normalized = np.empty(7, dtype=np.float32)
        normalized[:3] = np.clip(pre_capability_delta[:3] / self.profile.position_action_range_m, -1.0, 1.0)
        normalized[3:6] = np.clip(pre_capability_delta[3:] / self.profile.orientation_action_range_rad, -1.0, 1.0)
        normalized[:3] = np.clip(
            normalized[:3],
            -capability_policy.max_translation_normalized,
            capability_policy.max_translation_normalized,
        )
        normalized[3:6] = np.clip(
            normalized[3:6],
            -capability_policy.max_orientation_normalized,
            capability_policy.max_orientation_normalized,
        )
        normalized[6] = float(np.clip(capability_policy.gripper_action, -1.0, 1.0))
        if not np.all(np.isfinite(normalized)):
            raise ShakeBenchOracleError("controller produced a non-finite action")
        phase_diagnostics = dict(self.executive.diagnostics())
        phase_diagnostics["task_context_sha256"] = self.task_context_sha256
        phase_diagnostics["table_edge_margin_m"] = self.executive._table_edge_margin_m(observation)
        phase_diagnostics["anchor_error_m"] = (
            None
            if self.executive._anchor_worktable_can_transform is None
            else self.executive._anchor_error_m(observation)
        )
        phase_diagnostics["tool_clearance"] = self.executive.public_tool_clearance_certificate(observation)
        self.last_trace = {
            "policy_time_s": policy_time,
            "measurement_time_s": estimate.timestamp_s,
            "latency_s": estimate.latency_s,
            "phase": phase_diagnostics,
            "relative_kinematics": (
                self.executive._relative_kinematics.to_dict()
                if self.executive._relative_kinematics is not None
                else None
            ),
            "task_context_sha256": self.task_context_sha256,
            "controller_context_hash": self.controller_context_hash,
            "provider_payload_keys": sorted(expected),
            "provider_payload": {key: np.asarray(observation[key]).copy() for key in sorted(expected)},
            "estimate": estimate.to_dict(),
            "control_law": {
                "desired_delta_base": law_output.desired_delta_base.tolist(),
                "compensated_delta_base": law_output.compensated_delta_base.tolist(),
                "correction_delta_base": (law_output.compensated_delta_base - law_output.desired_delta_base).tolist(),
            },
            "task_desired_action": desired_delta.copy(),
            "pre_capability_compensated_action": pre_capability_delta,
            "phase_capability": capability_policy.capability.value,
            "capability_limits": {
                "max_translation_normalized": capability_policy.max_translation_normalized,
                "max_orientation_normalized": capability_policy.max_orientation_normalized,
                "gripper_action": capability_policy.gripper_action,
                "vibration_feedforward_allowed": capability_policy.vibration_feedforward_allowed,
            },
            "post_capability_normalized_action": normalized.copy(),
            "normalized_action": normalized.copy(),
        }
        self._last_policy_time_s = policy_time
        return normalized


def __getattr__(name: str):
    if name == "TaskExecutive":
        from robosuite.utils.oracle_executive import TaskExecutive

        return TaskExecutive
    raise AttributeError(name)


__all__ = [
    "ControlLawOutput",
    "MOTION_CAPABILITY_BY_PHASE",
    "MotionCapability",
    "MotionCapabilityPolicy",
    "OracleControllerProfile",
    "PublicRelativeKinematicsTracker",
    "RelativeKinematicsSnapshot",
    "RelativeSupportMotionEstimate",
    "ShakeBenchOracleController",
    "ShakeBenchOracleError",
    "SharedVibrationControlLaw",
    "TaskExecutive",
    "TaskPhase",
    "V1IMUEstimator",
    "VibrationEstimate",
    "WorktableTaskContext",
    "future_relative_acceleration_from_public_program",
    "motion_capability_for_phase",
    "motion_capability_policy",
    "relative_support_motion_from_public_observation",
    "robot_base_to_worktable_local",
    "target_local_to_robot_base",
    "target_local_to_worktable_local",
    "vibration_estimate_from_public_observation",
    "validate_motion_capability_map",
    "worktable_local_to_robot_base",
]
