"""Public-observation reference controller for the ShakeBench State track.

The controller is intentionally a small, inspectable baseline.  It owns no
MuJoCo object and accepts only the current single public observation contract
(task state plus the under-table IMU payload that the recorder keeps as public
provenance) and public static context.  Simulator contact and evaluator truth
remain on the recorder side of the boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

import robosuite.utils.transform_utils as T
from shakebench.utils.providers import COMMON_STATE_KEYS, TABLE_IMU_POLICY_KEYS

if TYPE_CHECKING:  # pragma: no cover - the runtime name is served by __getattr__
    from shakebench.utils.oracle_executive import TaskExecutive

# The oracle wire contract addresses the canonical field contract by the names
# the task executive, the GPU collector and the recorder already publish.
ORACLE_TASK_KEYS = tuple(COMMON_STATE_KEYS)


def _public_copy(value: Any) -> Any:
    """Copy a public observation value, keeping scalars scalar.

    A zero-dimensional array would reach the artifact writer as something it
    tries to iterate, so the scalar IMU interval stays a plain value.
    """

    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.copy()


class ShakeBenchOracleError(ValueError):
    """Raised when the public controller contract is malformed."""


class TaskPhase(str, Enum):
    SETTLE = "settle"
    APPROACH = "approach"
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
    ABORTED = "aborted"
    # Compatibility alias.  New runner code must use ``abort_reason`` and
    # never infer a task result from a controller phase.
    FAILED = "aborted"


class MotionCapability(str, Enum):
    """The physical action policy associated with an execution phase."""

    HOLD = "hold"
    FREE_SPACE = "free_space"
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
    """Public immutable task geometry of the current world-fixed assembly."""

    worktable_size_xy_m: tuple[float, float] = (0.65, 0.60)
    target_frame_origin_in_worktable_m: tuple[float, float, float] = (-0.10, 0.17, 0.042)
    table_surface_z_in_worktable_m: float = 0.03
    object_collision_radius_m: float = 0.02509177806572465
    object_collision_lower_support_m: float = -0.040297003330440104
    object_collision_upper_support_m: float = 0.03970300217508332
    finger_pad_tool_support_offsets_m: tuple[float, ...] = (0.0, 0.0, 0.0934)
    support_topology_id: str = "deck_robot_base_plus_isolated_worktable"

    def __post_init__(self) -> None:
        size = np.asarray(self.worktable_size_xy_m, dtype=float)
        origin = np.asarray(self.target_frame_origin_in_worktable_m, dtype=float)
        offsets = np.asarray(self.finger_pad_tool_support_offsets_m, dtype=float)
        if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ShakeBenchOracleError("worktable_size_xy_m must be finite positive length two")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ShakeBenchOracleError("target_frame_origin_in_worktable_m must be finite length three")
        if offsets.shape != (3,) or not np.all(np.isfinite(offsets)):
            raise ShakeBenchOracleError("finger_pad_tool_support_offsets_m must be finite length three")
        for name in (
            "table_surface_z_in_worktable_m",
            "object_collision_radius_m",
            "object_collision_lower_support_m",
            "object_collision_upper_support_m",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ShakeBenchOracleError(f"{name} must be finite")
        if (
            self.object_collision_radius_m <= 0.0
            or self.object_collision_lower_support_m >= self.object_collision_upper_support_m
        ):
            raise ShakeBenchOracleError("Can collision support bounds are invalid")
        if not isinstance(self.support_topology_id, str) or not self.support_topology_id:
            raise ShakeBenchOracleError("support_topology_id must be nonempty")

    @property
    def worktable_half_extents_xy_m(self) -> tuple[float, float]:
        return tuple(float(value) for value in np.asarray(self.worktable_size_xy_m, dtype=float) / 2.0)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "WorktableTaskContext":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ShakeBenchOracleError("task context must be a mapping")
        default = cls()
        worktable = value.get("worktable", value)
        can = value.get("object", value)
        target = value.get("target_container", value)
        envelope = can.get("collision_envelope", {}) if isinstance(can, Mapping) else {}
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
            object_collision_radius_m=float(
                envelope.get(
                    "support_radius_m", value.get("object_collision_radius_m", default.object_collision_radius_m)
                )
            ),
            object_collision_lower_support_m=float(
                envelope.get(
                    "lower_support_z_m",
                    value.get("object_collision_lower_support_m", default.object_collision_lower_support_m),
                )
            ),
            object_collision_upper_support_m=float(
                envelope.get(
                    "upper_support_z_m",
                    value.get("object_collision_upper_support_m", default.object_collision_upper_support_m),
                )
            ),
            finger_pad_tool_support_offsets_m=tuple(
                value.get("finger_pad_tool_support_offsets_m", default.finger_pad_tool_support_offsets_m)
            ),
            support_topology_id=str(value.get("support_topology_id", default.support_topology_id)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["worktable_half_extents_xy_m"] = list(self.worktable_half_extents_xy_m)
        return result


RELATIVE_SUPPORT_ESTIMATE_SCHEMA_ID = "shakebench.relative_support_motion_estimate"
RELATIVE_SUPPORT_ESTIMATE_SCHEMA_VERSION = 1
RELATIVE_SUPPORT_SEMANTIC_QUANTITY = "worktable_relative_to_robot_base_motion"
RELATIVE_SUPPORT_FRAME = "robot_base"
RELATIVE_SUPPORT_REFERENCE_POINT = "target_origin"


@dataclass(frozen=True)
class RelativeSupportMotionEstimate:
    """One typed meaning for the current public support-motion estimate.

    The canonical values are the motion of the rigid worktable/target
    assembly relative to the robot base, expressed at the target-frame origin
    at one policy timestamp.  The current contract applies no feedforward, so
    this record is public provenance for the recorder and never an authority
    over the action.
    """

    current_relative_pose_robot_base: np.ndarray
    current_relative_twist_robot_base: np.ndarray
    current_relative_acceleration_robot_base: np.ndarray
    measurement_timestamp_s: float
    policy_timestamp_s: float
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
    confidence: float = 1.0
    source: str = "neutral"

    def __post_init__(self) -> None:
        if self.frame != RELATIVE_SUPPORT_FRAME:
            raise ShakeBenchOracleError("relative support estimate frame must be robot_base")
        if self.reference_point != RELATIVE_SUPPORT_REFERENCE_POINT:
            raise ShakeBenchOracleError("relative support estimate reference point must be target_origin")
        if self.semantic_quantity != RELATIVE_SUPPORT_SEMANTIC_QUANTITY:
            raise ShakeBenchOracleError("relative support estimate has an unsupported semantic quantity")
        pose = np.asarray(self.current_relative_pose_robot_base, dtype=float)
        twist = np.asarray(self.current_relative_twist_robot_base, dtype=float)
        acceleration = np.asarray(self.current_relative_acceleration_robot_base, dtype=float)
        if pose.shape != (7,) or twist.shape != (6,) or acceleration.shape != (6,):
            raise ShakeBenchOracleError("relative support current state has shapes (7,), (6,), and (6,)")
        if not all(np.all(np.isfinite(value)) for value in (pose, twist, acceleration)):
            raise ShakeBenchOracleError("relative support estimate values must be finite")
        pose = pose.copy()
        pose_norm = float(np.linalg.norm(pose[3:]))
        if pose_norm <= 1.0e-12:
            raise ShakeBenchOracleError("current relative pose quaternion must be nonzero")
        pose[3:] /= pose_norm
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
        if not 0.0 <= confidence <= 1.0:
            raise ShakeBenchOracleError("relative support confidence must lie in [0, 1]")
        if not isinstance(self.validity, (bool, np.bool_)):
            raise ShakeBenchOracleError("relative support validity must be boolean")
        if not isinstance(self.units, Mapping) or not self.units:
            raise ShakeBenchOracleError("relative support units must be a nonempty mapping")
        if not isinstance(self.source, str) or not self.source:
            raise ShakeBenchOracleError("relative support source must be nonempty")
        for name, value in (
            ("current_relative_pose_robot_base", pose),
            ("current_relative_twist_robot_base", twist),
            ("current_relative_acceleration_robot_base", acceleration),
        ):
            value = np.array(value, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "measurement_timestamp_s", measurement)
        object.__setattr__(self, "policy_timestamp_s", policy)
        object.__setattr__(self, "latency_s", latency)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    # Compatibility property retains the R4 spelling of the measurement time.
    @property
    def timestamp_s(self) -> float:
        return self.measurement_timestamp_s

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_id": RELATIVE_SUPPORT_ESTIMATE_SCHEMA_ID,
            "schema_version": RELATIVE_SUPPORT_ESTIMATE_SCHEMA_VERSION,
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
            "validity": bool(self.validity),
            "confidence": self.confidence,
            "source": self.source,
        }
        return result


# Public compatibility name retained for callers that imported the R4 type.
VibrationEstimate = RelativeSupportMotionEstimate


def _target_pose_from_public_observation(observation: Mapping[str, Any]) -> np.ndarray:
    position = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
    quaternion = np.asarray(observation["goal_frame_quat_robot_base"], dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(position)):
        raise ShakeBenchOracleError("public target frame pose is malformed")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ShakeBenchOracleError("public target frame quaternion is malformed")
    return np.concatenate((position, quaternion / norm))


def vibration_estimate_from_public_observation(
    observation: Mapping[str, Any],
    *,
    policy_time_s: Optional[float] = None,
    relative_kinematics: Any = None,
) -> VibrationEstimate:
    """Type the current public support motion at one policy timestamp.

    The current contract is the task observation plus the under-table IMU
    payload.  This record reads the public task history, applies no feedforward
    and requires no removed per-observation-tier field.
    """

    if not isinstance(observation, Mapping):
        raise ShakeBenchOracleError("observation must be a mapping")
    forbidden = [key for key in observation if str(key).startswith("privileged_")]
    if forbidden:
        raise ShakeBenchOracleError("privileged keys are not valid controller input")
    missing = [key for key in ORACLE_TASK_KEYS if key not in observation]
    if missing:
        raise ShakeBenchOracleError("missing observation payload: " + ", ".join(missing))
    policy_time = float(np.asarray(observation.get("episode_time_s", 0.0) if policy_time_s is None else policy_time_s))
    if not np.isfinite(policy_time) or policy_time < 0.0:
        raise ShakeBenchOracleError("policy_time_s must be finite and non-negative")
    current_pose = _target_pose_from_public_observation(observation)
    current_twist = np.zeros(6, dtype=float)
    current_acceleration = np.zeros(6, dtype=float)
    source = "public_neutral"
    confidence = 0.0
    if isinstance(relative_kinematics, RelativeKinematicsSnapshot) and relative_kinematics.history_valid:
        # The target pose history is common task information, so the current
        # contract may use this explicitly named low-confidence baseline.
        current_pose = relative_kinematics.target_frame_pose_robot_base.copy()
        current_twist = relative_kinematics.target_frame_twist_robot_base.copy()
        current_acceleration = relative_kinematics.target_frame_acceleration_robot_base.copy()
        source = "public_task_history_baseline"
        confidence = 0.20
    return VibrationEstimate(
        current_relative_pose_robot_base=current_pose,
        current_relative_twist_robot_base=current_twist,
        current_relative_acceleration_robot_base=current_acceleration,
        measurement_timestamp_s=policy_time,
        policy_timestamp_s=policy_time,
        latency_s=0.0,
        validity=True,
        confidence=confidence,
        source=source,
    )


@dataclass(frozen=True)
class OracleControllerProfile:
    """One profile applied unchanged to the current observation contract."""

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
    grasp_corridor_margin_m: float = 0.004
    grasp_expected_transform_tolerance_m: float = 0.018
    # 18 mm is below the 25.0918 mm collision radius while leaving the
    # measured policy/actuator noise margin for a rigid carry.
    grasp_slip_tolerance_m: float = 0.018
    grasp_severe_slip_fraction_of_can_radius: float = 0.75
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
    verify_linear_speed_limit_m_s: float = 0.020
    verify_angular_speed_limit_rad_s: float = 0.20
    verify_stability_samples: int = 2
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
                "expected_eef_can_translation_m",
            }:
                continue
            if key in {"gripper_open_action", "gripper_close_action"}:
                if not np.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
                    raise ShakeBenchOracleError(f"profile {key} must lie in [-1, 1]")
            elif not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ShakeBenchOracleError(f"profile {key} must be finite and positive")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ShakeBenchOracleError("profile_id must be nonempty")
        if not isinstance(self.verify_stability_samples, int) or self.verify_stability_samples < 2:
            raise ShakeBenchOracleError("verify_stability_samples must be an integer >= 2")
        if self.recovery_budget < 0:
            raise ShakeBenchOracleError("recovery_budget must be non-negative")
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionCapabilityPolicy:
    """Action limits selected from one phase capability."""

    capability: MotionCapability
    max_translation_normalized: float
    max_orientation_normalized: float
    gripper_action: float


def motion_capability_policy(phase: TaskPhase, profile: OracleControllerProfile) -> MotionCapabilityPolicy:
    """Resolve all phase action constraints at one centralized seam."""

    capability = motion_capability_for_phase(phase)
    translation_limits = {
        MotionCapability.HOLD: 1.0,
        MotionCapability.FREE_SPACE: 1.0,
        MotionCapability.CONTACT_APPROACH: profile.descend_position_action_limit,
        MotionCapability.GRIPPER_CLOSE: profile.descend_position_action_limit,
        MotionCapability.OBJECT_HELD: profile.transport_position_action_limit,
        MotionCapability.RECOVERY_LOWER: profile.descend_position_action_limit,
        MotionCapability.RECOVERY_RETREAT: profile.transport_position_action_limit,
        MotionCapability.TERMINAL: 0.0,
    }
    gripper_open_phases = {
        TaskPhase.SETTLE,
        TaskPhase.APPROACH,
        TaskPhase.LATERAL_ALIGN_ABOVE_CAN,
        TaskPhase.ALIGN_SETTLE,
        TaskPhase.VERTICAL_DESCEND,
        TaskPhase.DESCEND,
        TaskPhase.RELEASE,
        TaskPhase.VERIFY,
        TaskPhase.RECOVERY_OPEN,
        TaskPhase.CLEARANCE_RETREAT,
        TaskPhase.RE_ALIGN,
        TaskPhase.FAILED,
    }
    gripper = profile.gripper_open_action if phase in gripper_open_phases else profile.gripper_close_action
    return MotionCapabilityPolicy(
        capability=capability,
        max_translation_normalized=float(translation_limits[capability]),
        max_orientation_normalized=(0.0 if capability is MotionCapability.TERMINAL else 1.0),
        gripper_action=gripper,
    )


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    from shakebench.utils.rotations import xyzw_to_matrix

    return xyzw_to_matrix(quaternion, name="goal quaternion", error_type=ShakeBenchOracleError)


def _rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    from shakebench.utils.rotations import matrix_to_rotation_vector

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
        can = _pose_transform_from_observation(observation, "object_pos_robot_base", "object_quat_robot_base")
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
        profile: Optional[OracleControllerProfile] = None,
        task_context: WorktableTaskContext | Mapping[str, Any] | None = None,
    ):
        self.profile = profile or OracleControllerProfile()
        self.task_context = (
            task_context
            if isinstance(task_context, WorktableTaskContext)
            else WorktableTaskContext.from_mapping(task_context)
        )
        from shakebench.utils.oracle_executive import TaskExecutive

        self.executive = TaskExecutive(self.profile, self.task_context)
        self._last_policy_time_s: Optional[float] = None
        self.last_trace: Optional[dict[str, Any]] = None

    @property
    def abort_requested(self) -> bool:
        """Return the controller's structured decision to stop acting."""

        return self.executive.abort_reason is not None

    def reset(self) -> None:
        self.executive.reset()
        self._last_policy_time_s = None
        self.last_trace = None

    def action(self, observation: Mapping[str, Any], *, time_s: Optional[float] = None) -> np.ndarray:
        missing = [key for key in ORACLE_TASK_KEYS if key not in observation]
        if missing:
            raise ShakeBenchOracleError("missing observation payload: " + ", ".join(missing))
        if time_s is None:
            time_s = float(observation.get("episode_time_s", 0.0))
        policy_time = float(time_s)
        if self._last_policy_time_s is not None and policy_time <= self._last_policy_time_s + 1.0e-12:
            self.reset()
        # The single-channel gripper command comes from the phase capability.
        desired_delta = self.executive.command(observation, policy_time)
        estimate = vibration_estimate_from_public_observation(
            observation,
            policy_time_s=policy_time,
            relative_kinematics=self.executive._relative_kinematics,
        )
        capability_policy = motion_capability_policy(self.executive.phase, self.profile)
        normalized = np.empty(7, dtype=np.float32)
        normalized[:3] = np.clip(desired_delta[:3] / self.profile.position_action_range_m, -1.0, 1.0)
        normalized[3:6] = np.clip(desired_delta[3:] / self.profile.orientation_action_range_rad, -1.0, 1.0)
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
        phase_diagnostics["table_edge_margin_m"] = self.executive._table_edge_margin_m(observation)
        phase_diagnostics["anchor_error_m"] = (
            None
            if self.executive._anchor_worktable_can_transform is None
            else self.executive._anchor_error_m(observation)
        )
        phase_diagnostics["tool_clearance"] = self.executive.public_tool_clearance_certificate(observation)
        provider_keys = sorted(key for key in TABLE_IMU_POLICY_KEYS if key in observation)
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
            "provider_payload_keys": provider_keys,
            "provider_payload": {key: _public_copy(observation[key]) for key in provider_keys},
            "estimate": estimate.to_dict(),
            "task_desired_action": desired_delta.copy(),
            "phase_capability": capability_policy.capability.value,
            "capability_limits": {
                "max_translation_normalized": capability_policy.max_translation_normalized,
                "max_orientation_normalized": capability_policy.max_orientation_normalized,
                "gripper_action": capability_policy.gripper_action,
            },
            "post_capability_normalized_action": normalized.copy(),
            "normalized_action": normalized.copy(),
        }
        self._last_policy_time_s = policy_time
        return normalized


def __getattr__(name: str):
    if name == "TaskExecutive":
        from shakebench.utils.oracle_executive import TaskExecutive

        return TaskExecutive
    raise AttributeError(name)


__all__ = [
    "MOTION_CAPABILITY_BY_PHASE",
    "MotionCapability",
    "MotionCapabilityPolicy",
    "ORACLE_TASK_KEYS",
    "OracleControllerProfile",
    "PublicRelativeKinematicsTracker",
    "RelativeKinematicsSnapshot",
    "RelativeSupportMotionEstimate",
    "ShakeBenchOracleController",
    "ShakeBenchOracleError",
    "TaskExecutive",
    "TaskPhase",
    "VibrationEstimate",
    "WorktableTaskContext",
    "motion_capability_for_phase",
    "motion_capability_policy",
    "robot_base_to_worktable_local",
    "target_local_to_robot_base",
    "vibration_estimate_from_public_observation",
    "validate_motion_capability_map",
    "worktable_local_to_robot_base",
]
