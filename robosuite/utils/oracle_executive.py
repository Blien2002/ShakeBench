"""Public-state ShakeBench task execution state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from robosuite.utils.shakebench_oracle import (
    MotionCapabilityPolicy,
    OracleControllerProfile,
    PublicRelativeKinematicsTracker,
    RelativeKinematicsSnapshot,
    ShakeBenchOracleError,
    TaskPhase,
    WorktableTaskContext,
    _pose_transform_from_observation,
    _quat_xyzw_to_matrix,
    _rotation_vector_from_matrix,
    _worktable_transform_robot_base,
    motion_capability_policy,
    robot_base_to_worktable_local,
    target_local_to_robot_base,
    validate_motion_capability_map,
    worktable_local_to_robot_base,
)


@dataclass
class TaskExecutive:
    """A public-state phase machine with deadlines and bounded recovery."""

    profile: OracleControllerProfile
    context: WorktableTaskContext = field(default_factory=WorktableTaskContext)
    phase: TaskPhase = TaskPhase.SETTLE
    phase_entered_s: float = 0.0
    # ``failure_reason`` remains a read-only compatibility projection for old
    # diagnostic consumers.  It is never a task outcome authority.
    failure_reason: Optional[str] = None
    abort_reason: Optional[str] = None
    last_recovery_reason: Optional[str] = None
    recovery_count: int = 0
    recovery_transition: Optional[str] = field(default=None, init=False, repr=False)
    _eef_reference_rotation: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _initial_target_rotation: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _grasp_eef_can_transform: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_public_can_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_public_eef_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_public_can_rotation: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_public_time_s: Optional[float] = field(default=None, init=False, repr=False)
    _public_linear_speed_m_s: float = field(default=0.0, init=False, repr=False)
    _public_angular_speed_rad_s: float = field(default=0.0, init=False, repr=False)
    _public_eef_speed_m_s: float = field(default=0.0, init=False, repr=False)
    _public_eef_velocity_m_s: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float), init=False, repr=False
    )
    _public_stable_samples: int = field(default=0, init=False, repr=False)
    _last_slip_translation_m: float = field(default=0.0, init=False, repr=False)
    _last_slip_rotation_rad: float = field(default=0.0, init=False, repr=False)
    _grasp_loss_samples: int = field(default=0, init=False, repr=False)
    _grasp_confidence: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False)
    _grasp_reference_established: bool = field(default=False, init=False, repr=False)
    _prelift_start_eef_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prelift_start_can_height_m: Optional[float] = field(default=None, init=False, repr=False)
    _prelift_follow_samples: int = field(default=0, init=False, repr=False)
    _recovery_events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _anchor_worktable_can_transform: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _anchor_worktable_eef_goal: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _anchor_clearance_certificate: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False)
    _anchor_timestamp_s: Optional[float] = field(default=None, init=False, repr=False)
    _clearance_start_eef_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _placement_start_eef_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _release_eef_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _recovery_start_eef_position: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _recovery_lower_goal: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _recovery_reverse_goal: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _align_stable_samples: int = field(default=0, init=False, repr=False)
    _relative_kinematics_tracker: PublicRelativeKinematicsTracker = field(init=False, repr=False)
    _relative_kinematics: Optional[RelativeKinematicsSnapshot] = field(default=None, init=False, repr=False)
    _last_swept_clearance_certificate: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_motion_capability_map()
        self._relative_kinematics_tracker = PublicRelativeKinematicsTracker(self.context)
        self.reset()

    def reset(self) -> None:
        self.phase = TaskPhase.SETTLE
        self.phase_entered_s = 0.0
        self.failure_reason = None
        self.abort_reason = None
        self.last_recovery_reason = None
        self.recovery_count = 0
        self.recovery_transition = None
        self._eef_reference_rotation = None
        self._initial_target_rotation = None
        self._grasp_eef_can_transform = None
        self._last_public_can_position = None
        self._last_public_eef_position = None
        self._last_public_can_rotation = None
        self._last_public_time_s = None
        self._public_linear_speed_m_s = 0.0
        self._public_angular_speed_rad_s = 0.0
        self._public_eef_speed_m_s = 0.0
        self._public_eef_velocity_m_s = np.zeros(3, dtype=float)
        self._public_stable_samples = 0
        self._last_slip_translation_m = 0.0
        self._last_slip_rotation_rad = 0.0
        self._grasp_loss_samples = 0
        self._grasp_confidence = {}
        self._grasp_reference_established = False
        self._prelift_start_eef_position = None
        self._prelift_start_can_height_m = None
        self._prelift_follow_samples = 0
        self._recovery_events = []
        self._anchor_worktable_can_transform = None
        self._anchor_worktable_eef_goal = None
        self._anchor_clearance_certificate = {}
        self._anchor_timestamp_s = None
        self._clearance_start_eef_position = None
        self._placement_start_eef_position = None
        self._release_eef_position = None
        self._recovery_start_eef_position = None
        self._recovery_lower_goal = None
        self._recovery_reverse_goal = None
        self._align_stable_samples = 0
        self._relative_kinematics = None
        self._relative_kinematics_tracker.reset()
        self._last_swept_clearance_certificate = {}

    @property
    def relative_kinematics(self) -> Optional[RelativeKinematicsSnapshot]:
        """Last public relative-motion snapshot, or ``None`` before a call."""

        return self._relative_kinematics

    def _transition(self, phase: TaskPhase, time_s: float) -> None:
        self.phase = phase
        self.phase_entered_s = float(time_s)

    def _current_can_worktable_transform(self, observation: Mapping[str, Any]) -> np.ndarray:
        can_base = _pose_transform_from_observation(observation, "can_pos_robot_base", "can_quat_robot_base")
        table_base = _worktable_transform_robot_base(observation, self.context)
        return np.linalg.inv(table_base).dot(can_base)

    def _capture_grasp_anchor(self, observation: Mapping[str, Any], time_s: float) -> None:
        """Freeze Can and grasp waypoints in the rigid worktable frame."""

        self._anchor_worktable_can_transform = self._current_can_worktable_transform(observation)
        can_position = self._anchor_worktable_can_transform[:3, 3]
        table_base = _worktable_transform_robot_base(observation, self.context)
        eef_base = _pose_transform_from_observation(
            observation, "robot0_eef_pos_robot_base", "robot0_eef_quat_robot_base"
        )
        eef_worktable = np.linalg.inv(table_base).dot(eef_base)
        eef_worktable[:3, 3] = can_position + np.array((0.0, 0.0, self.profile.grasp_height_m), dtype=float)
        self._anchor_worktable_eef_goal = eef_worktable
        self._anchor_clearance_certificate = dict(self.public_tool_clearance_certificate(observation))
        self._anchor_timestamp_s = float(time_s)

    def _begin_lateral_alignment(self, observation: Mapping[str, Any], time_s: float) -> None:
        """Freeze the Can and enter the already-clear lateral standoff move."""

        self._capture_grasp_anchor(observation, time_s)
        self._align_stable_samples = 0
        self._transition(TaskPhase.LATERAL_ALIGN_ABOVE_CAN, time_s)
        lateral_goal = self._phase_goal(observation)[0]
        self._last_swept_clearance_certificate = dict(
            self._swept_clearance_to_goal(observation, lateral_goal, self._anchored_can_base(observation))
        )
        if not self._last_swept_clearance_certificate["passed"]:
            self._recover_or_fail(observation, time_s, "public_tool_clearance")

    def _anchored_can_base(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._anchor_worktable_can_transform is None:
            raise ShakeBenchOracleError("grasp anchor has not been captured")
        table_base = _worktable_transform_robot_base(observation, self.context)
        return table_base[:3, 3] + table_base[:3, :3].dot(self._anchor_worktable_can_transform[:3, 3])

    def _anchor_error_m(self, observation: Mapping[str, Any]) -> float:
        if self._anchor_worktable_can_transform is None:
            return float("inf")
        if self._relative_kinematics is not None:
            current_position = self._relative_kinematics.worktable_can_pose[:3]
        else:
            current_position = self._current_can_worktable_transform(observation)[:3, 3]
        return float(np.linalg.norm(current_position - self._anchor_worktable_can_transform[:3, 3]))

    def _capability_policy(self) -> MotionCapabilityPolicy:
        return motion_capability_policy(self.phase, self.profile)

    def _swept_clearance_to_goal(
        self, observation: Mapping[str, Any], goal: np.ndarray, can_position: np.ndarray | None = None
    ) -> Mapping[str, Any]:
        eef = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        tips = np.asarray(observation["robot0_fingertip_pos_robot_base"], dtype=float)
        translation = np.asarray(goal, dtype=float) - eef
        endpoint = tips + np.tile(translation, 2)
        certificate = dict(
            self.public_swept_tool_clearance_certificate(observation, tips, endpoint, can_position=can_position)
        )
        certificate.update(
            {
                "segment_start_eef_position": eef.tolist(),
                "segment_end_eef_position": np.asarray(goal, dtype=float).tolist(),
                "segment_start_fingertips": tips.tolist(),
                "segment_end_fingertips": endpoint.tolist(),
            }
        )
        return certificate

    def _table_edge_margin_m(self, observation: Mapping[str, Any], position: np.ndarray | None = None) -> float:
        point = (
            np.asarray(observation["can_pos_robot_base"], dtype=float)
            if position is None
            else np.asarray(position, dtype=float)
        )
        local = robot_base_to_worktable_local(observation, point, self.context)
        half = np.asarray(self.context.worktable_half_extents_xy_m, dtype=float)
        return float(np.min(half - np.abs(local[:2]) - self.context.can_collision_radius_m))

    def public_tool_clearance_certificate(
        self, observation: Mapping[str, Any], can_position: np.ndarray | None = None
    ) -> Mapping[str, Any]:
        """Check the complete public finger-pad support points above the Can."""

        can = (
            np.asarray(observation["can_pos_robot_base"], dtype=float)
            if can_position is None
            else np.asarray(can_position, dtype=float)
        )
        tips = np.asarray(observation["robot0_fingertip_pos_robot_base"], dtype=float)
        if tips.shape != (6,) or not np.all(np.isfinite(tips)):
            raise ShakeBenchOracleError("public fingertip positions must be finite length six")
        can_local = robot_base_to_worktable_local(observation, can, self.context)
        tip_local = np.asarray(
            [robot_base_to_worktable_local(observation, tip, self.context) for tip in tips.reshape(2, 3)]
        )
        can_top = can_local[2] + self.context.can_collision_upper_support_m
        lowest_tip = float(np.min(tip_local[:, 2]))
        clearance = lowest_tip - can_top
        half = np.asarray(self.context.worktable_half_extents_xy_m, dtype=float)
        tips_in_table = bool(np.all(np.abs(tip_local[:, :2]) <= half + self.profile.tool_clearance_m))
        return {
            "can_top_z_worktable_m": float(can_top),
            "lowest_fingertip_z_worktable_m": lowest_tip,
            "clearance_m": float(clearance),
            "required_clearance_m": self.profile.tool_clearance_m,
            "tips_in_worktable_envelope": tips_in_table,
            "passed": bool(clearance >= self.profile.tool_clearance_m and tips_in_table),
        }

    def public_swept_tool_clearance_certificate(
        self,
        observation: Mapping[str, Any],
        start_fingertips: np.ndarray,
        end_fingertips: np.ndarray,
        can_position: np.ndarray | None = None,
        samples: int = 11,
    ) -> Mapping[str, Any]:
        """Check pad support clearance over a public linear swept segment."""

        start = np.asarray(start_fingertips, dtype=float)
        end = np.asarray(end_fingertips, dtype=float)
        if start.shape != (6,) or end.shape != (6,) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ShakeBenchOracleError("swept fingertips must be finite length-six vectors")
        if not isinstance(samples, int) or samples < 2:
            raise ShakeBenchOracleError("swept clearance requires at least two samples")
        certificates: list[Mapping[str, Any]] = []
        clearances = []
        passed = True
        tips_in_table = True
        for fraction in np.linspace(0.0, 1.0, samples):
            observation_at_fraction = dict(observation)
            observation_at_fraction["robot0_fingertip_pos_robot_base"] = start + fraction * (end - start)
            certificate = self.public_tool_clearance_certificate(observation_at_fraction, can_position=can_position)
            certificates.append(certificate)
            clearances.append(float(certificate["clearance_m"]))
            passed = passed and bool(certificate["passed"])
            tips_in_table = tips_in_table and bool(certificate["tips_in_worktable_envelope"])
        return {
            "samples": samples,
            "minimum_clearance_m": float(min(clearances)),
            "required_clearance_m": self.profile.tool_clearance_m,
            "tips_in_worktable_envelope": tips_in_table,
            "passed": passed,
            "endpoint_certificates": certificates,
        }

    def _public_can_stable_and_in_bounds(self, observation: Mapping[str, Any]) -> bool:
        recoverability = self._public_recoverability(observation)
        return bool(
            recoverability["table_envelope_ok"]
            and recoverability["edge_margin_m"] >= self.profile.table_edge_safety_margin_m
            and self._public_stable_samples >= self.profile.align_required_samples
        )

    def _public_grasp_geometry(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Evaluate a bilateral pad corridor using only public geometry.

        The Can centre must lie between the pads along their closing axis and
        near the shared Panda/Can grasp transform.  This intentionally rejects
        a single-side push, a distant wrench event, and an empty closed hand.
        """

        gripper_state = np.asarray(observation["robot0_gripper_state"], dtype=float)
        if gripper_state.shape != (4,) or not np.all(np.isfinite(gripper_state)):
            raise ShakeBenchOracleError("robot0_gripper_state must be a finite four-vector")
        opening = float(np.sum(np.abs(gripper_state[:2])))
        tips = np.asarray(observation["robot0_fingertip_pos_robot_base"], dtype=float)
        can = np.asarray(observation["can_pos_robot_base"], dtype=float)
        eef = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        wrist_force = np.asarray(observation["robot0_wrist_force"], dtype=float)
        wrist_torque = np.asarray(observation["robot0_wrist_torque"], dtype=float)
        if (
            tips.shape != (6,)
            or can.shape != (3,)
            or eef.shape != (3,)
            or wrist_force.shape != (3,)
            or wrist_torque.shape != (3,)
        ):
            raise ShakeBenchOracleError("public grasp signals have invalid shapes")
        if not all(np.all(np.isfinite(value)) for value in (tips, can, eef, wrist_force, wrist_torque)):
            raise ShakeBenchOracleError("public grasp signals must be finite")
        first_tip, second_tip = tips.reshape(2, 3)
        fingertip_midpoint = 0.5 * (first_tip + second_tip)
        closing_vector = second_tip - first_tip
        pad_span = float(np.linalg.norm(closing_vector))
        if pad_span <= 1.0e-6:
            # A degenerate public pad measurement is fail-closed as an empty
            # gripper, rather than becoming an exception path that skips the
            # controller's recoverability policy.
            return {
                "aperture": 0.0,
                "fingertip_geometry": 0.0,
                "opening_rad": opening,
                "aperture_ok": False,
                "pad_span_m": pad_span,
                "along_closing_axis_m": float("inf"),
                "perpendicular_to_closing_axis_m": float("inf"),
                "corridor_half_width_m": 0.0,
                "corridor_ok": False,
                "expected_eef_can_error_m": float("inf"),
                "expected_transform_ok": False,
                "can_to_eef_distance_m": float(np.linalg.norm(can - eef)),
                "wrench_magnitude_public": float(np.linalg.norm(wrist_force) + np.linalg.norm(wrist_torque)),
                "wrench_used_for_hold_verdict": False,
            }
        closing_axis = closing_vector / pad_span
        can_from_midpoint = can - fingertip_midpoint
        along_closing_axis = float(np.dot(can_from_midpoint, closing_axis))
        perpendicular_to_axis = float(np.linalg.norm(can_from_midpoint - along_closing_axis * closing_axis))
        half_corridor = 0.5 * pad_span + self.context.can_collision_radius_m + self.profile.grasp_corridor_margin_m
        corridor_ok = abs(along_closing_axis) <= half_corridor
        base_eef = _pose_transform_from_observation(
            observation, "robot0_eef_pos_robot_base", "robot0_eef_quat_robot_base"
        )
        expected_eef_can = np.asarray(self.profile.expected_eef_can_translation_m, dtype=float)
        can_in_eef = base_eef[:3, :3].T.dot(can - eef)
        expected_error = float(np.linalg.norm(can_in_eef - expected_eef_can))
        can_to_eef = float(np.linalg.norm(can - eef))
        aperture_ok = self.profile.grasp_hold_min_opening_rad <= opening <= self.profile.grasp_hold_max_opening_rad
        expected_ok = expected_error <= self.profile.grasp_expected_transform_tolerance_m
        aperture_confidence = float(np.clip(opening / self.profile.grasp_hold_max_opening_rad, 0.0, 1.0))
        geometry_confidence = float(
            np.clip(1.0 - expected_error / self.profile.grasp_expected_transform_tolerance_m, 0.0, 1.0)
        )
        wrench_magnitude = float(np.linalg.norm(wrist_force) + np.linalg.norm(wrist_torque))
        return {
            "aperture": aperture_confidence,
            "fingertip_geometry": geometry_confidence,
            "opening_rad": opening,
            "aperture_ok": aperture_ok,
            "pad_span_m": pad_span,
            "along_closing_axis_m": along_closing_axis,
            "perpendicular_to_closing_axis_m": perpendicular_to_axis,
            "corridor_half_width_m": half_corridor,
            "corridor_ok": corridor_ok,
            "expected_eef_can_error_m": expected_error,
            "expected_transform_ok": expected_ok,
            "can_to_eef_distance_m": can_to_eef,
            "wrench_magnitude_public": wrench_magnitude,
            "wrench_used_for_hold_verdict": False,
        }

    def _public_grasp_held(self, observation: Mapping[str, Any]) -> bool:
        """Require aperture plus bilateral geometry; wrench never grants hold authority."""

        geometry = self._public_grasp_geometry(observation)
        self._grasp_confidence = geometry
        if self._grasp_reference_established:
            # Once a candidate reference exists, the reference SE(3) slip
            # check is authoritative.  Requiring the static establishment
            # transform again would reject a rigid Can when the EEF rotates
            # along the placement path.
            return bool(
                geometry["aperture_ok"] and geometry["corridor_ok"] and geometry["can_to_eef_distance_m"] <= 0.14
            )
        return bool(geometry["aperture_ok"] and geometry["corridor_ok"] and geometry["expected_transform_ok"])

    def _public_gripper_open(self, observation: Mapping[str, Any]) -> bool:
        state = np.asarray(observation["robot0_gripper_state"], dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ShakeBenchOracleError("robot0_gripper_state must be a finite four-vector")
        opening = float(np.sum(np.abs(state[:2])))
        return bool(opening >= self.profile.grasp_hold_min_opening_rad)

    def _record_grasp_reference(self, observation: Mapping[str, Any]) -> None:
        base_eef = _pose_transform_from_observation(
            observation, "robot0_eef_pos_robot_base", "robot0_eef_quat_robot_base"
        )
        base_can = _pose_transform_from_observation(observation, "can_pos_robot_base", "can_quat_robot_base")
        self._grasp_eef_can_transform = np.linalg.inv(base_eef).dot(base_can)
        self._grasp_reference_established = True

    def _target_local_z(self, observation: Mapping[str, Any], position: np.ndarray) -> float:
        frame = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
        rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
        return float(rotation.T.dot(np.asarray(position, dtype=float) - frame)[2])

    def _begin_prelift_verify(self, observation: Mapping[str, Any], time_s: float) -> None:
        self._record_grasp_reference(observation)
        self._prelift_start_eef_position = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float).copy()
        if self._anchor_worktable_eef_goal is not None:
            self._prelift_start_eef_position = worktable_local_to_robot_base(
                observation,
                self._anchor_worktable_eef_goal[:3, 3],
                self.context,
            )
        self._prelift_start_can_height_m = self._target_local_z(
            observation, np.asarray(observation["can_pos_robot_base"], dtype=float)
        )
        self._prelift_follow_samples = 0
        self._grasp_loss_samples = 0
        self._transition(TaskPhase.PRELIFT_VERIFY, time_s)

    def _public_grasp_slip(self, observation: Mapping[str, Any]) -> tuple[float, float]:
        if self._grasp_eef_can_transform is None:
            return 0.0, 0.0
        base_eef = _pose_transform_from_observation(
            observation, "robot0_eef_pos_robot_base", "robot0_eef_quat_robot_base"
        )
        base_can = _pose_transform_from_observation(observation, "can_pos_robot_base", "can_quat_robot_base")
        current = np.linalg.inv(base_eef).dot(base_can)
        translation = float(np.linalg.norm(current[:3, 3] - self._grasp_eef_can_transform[:3, 3]))
        rotation = float(
            np.linalg.norm(_rotation_vector_from_matrix(self._grasp_eef_can_transform[:3, :3].T.dot(current[:3, :3])))
        )
        return translation, rotation

    def _public_container_risk(self, observation: Mapping[str, Any]) -> bool:
        """Conservative public fingertip/EEF wall-risk check in target frame."""
        if not self._public_can_inside_target(observation):
            return False
        tips = np.asarray(observation["robot0_fingertip_pos_robot_base"], dtype=float).reshape(2, 3)
        frame = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
        rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
        extents = np.asarray(observation["goal_inner_half_extents_target"], dtype=float)
        local = (tips - frame).dot(rotation)
        return bool(np.any(np.abs(local[:, :2]) > extents + self.profile.public_risk_margin_m))

    def _public_recoverability(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        can = np.asarray(observation["can_pos_robot_base"], dtype=float)
        eef = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        local_can = robot_base_to_worktable_local(observation, can, self.context)
        lower_support = float(local_can[2] + self.context.can_collision_lower_support_m)
        edge_margin = self._table_edge_margin_m(observation, can)
        envelope_inside = bool(edge_margin >= 0.0)
        table_height_error = abs(lower_support - self.context.table_surface_z_in_worktable_m)
        in_workspace = bool(
            np.linalg.norm(can) <= self.profile.public_workspace_radius_m
            and np.linalg.norm(eef) <= self.profile.public_workspace_radius_m
        )
        held = self._public_grasp_held(observation)
        near_table = table_height_error <= self.profile.recovery_table_height_tolerance_m
        falling = (
            self._public_linear_speed_m_s > self.profile.recovery_max_downward_speed_m_s
            and lower_support < self.context.table_surface_z_in_worktable_m
        )
        edge_unrecoverable = edge_margin < self.profile.table_edge_safety_margin_m
        return {
            "finite": bool(np.all(np.isfinite(can)) and np.all(np.isfinite(eef))),
            "can_worktable_position_m": local_can.tolist(),
            "can_lower_support_z_worktable_m": lower_support,
            "table_surface_z_worktable_m": self.context.table_surface_z_in_worktable_m,
            "table_height_error_m": table_height_error,
            "edge_margin_m": edge_margin,
            "table_envelope_ok": envelope_inside,
            "in_workspace": in_workspace,
            "held": held,
            "near_table": near_table,
            "falling_fast_below_table": falling,
            "edge_unrecoverable": edge_unrecoverable,
            "recoverable": bool(
                envelope_inside and not edge_unrecoverable and in_workspace and (held or (near_table and not falling))
            ),
        }

    def _public_safe_to_open(self, observation: Mapping[str, Any]) -> bool:
        state = self._public_recoverability(observation)
        return bool(
            state["table_envelope_ok"]
            and not state["edge_unrecoverable"]
            and state["near_table"]
            and not state["falling_fast_below_table"]
            and self._public_stable_samples >= self.profile.recovery_stable_samples
            and self._public_linear_speed_m_s <= self.profile.recovery_max_downward_speed_m_s
        )

    @staticmethod
    def _event_type(reason: str) -> str:
        event_type = {
            "phase_deadline": "phase_deadline",
            "public_grasp_not_established": "grasp_not_established",
            "public_grasp_loss": "grasp_loss",
            "public_grasp_slip": "grasp_slip",
            "public_anchor_drift": "anchor_drift",
            "public_tool_clearance": "tool_clearance_blocked",
            "public_placement_loss": "placement_loss",
            "public_placement_rebound": "placement_rebound",
            "edge_risk": "edge_risk",
            "workspace_risk": "workspace_risk",
        }.get(reason)
        if event_type is None:
            raise ShakeBenchOracleError(f"unregistered recovery reason: {reason}")
        return event_type

    def _record_event(self, event_type: str, observation: Mapping[str, Any], time_s: float, decision: str) -> None:
        """Record a public controller observation without changing outcome state."""

        self._recovery_events.append(
            {
                "event_type": event_type,
                "time_s": float(time_s),
                "phase": self.phase.value,
                "decision": decision,
                "observation_summary": {
                    "can_pos_robot_base": np.asarray(observation["can_pos_robot_base"], dtype=float).tolist(),
                    "eef_pos_robot_base": np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float).tolist(),
                    "recovery_count": self.recovery_count,
                },
            }
        )

    def _abort(self, observation: Mapping[str, Any], time_s: float, reason: str) -> None:
        """End this controller's action stream; the runner owns episode result."""

        self._record_event(self._event_type(reason), observation, time_s, "policy_abort")
        self.phase = TaskPhase.ABORTED
        self.abort_reason = reason
        self.failure_reason = "policy_abort"  # legacy diagnostic projection
        self.recovery_transition = "policy_abort"

    def _recover_or_fail(self, observation: Mapping[str, Any], time_s: float, reason: str) -> None:
        """Choose a recovery primitive or a structured policy abort.

        The historical name is retained only for source compatibility.  No
        branch here creates an environment/task failure.
        """

        self.last_recovery_reason = reason
        precondition = self._public_recoverability(observation)
        self._record_event(self._event_type(reason), observation, time_s, "evaluate_recovery")
        self._recovery_start_eef_position = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float).copy()
        brake = np.clip(
            self.profile.recovery_velocity_brake_s * self._public_eef_velocity_m_s,
            -0.03,
            0.03,
        )
        self._recovery_start_eef_position -= brake
        self._recovery_reverse_goal = (
            self._prelift_start_eef_position.copy()
            if self._prelift_start_eef_position is not None
            else (
                self._anchor_worktable_eef_goal[:3, 3].copy()
                if self._anchor_worktable_eef_goal is not None
                else self._recovery_start_eef_position.copy()
            )
        )
        target_rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
        can_worktable = robot_base_to_worktable_local(
            observation, np.asarray(observation["can_pos_robot_base"], dtype=float), self.context
        )
        current_lower_support = can_worktable[2] + self.context.can_collision_lower_support_m
        lower_distance = max(0.0, current_lower_support - self.context.table_surface_z_in_worktable_m)
        self._recovery_lower_goal = self._recovery_reverse_goal.copy()
        if lower_distance > 0.0 and self._prelift_start_eef_position is None:
            self._recovery_lower_goal = self._recovery_start_eef_position - target_rotation[:, 2] * lower_distance
        if reason == "workspace_risk":
            self._abort(observation, time_s, "workspace_risk")
        elif precondition["edge_unrecoverable"]:
            self._abort(observation, time_s, "edge_risk")
        elif not precondition["recoverable"]:
            self._abort(observation, time_s, "workspace_risk")
        elif self.recovery_count < self.profile.recovery_budget:
            self.recovery_count += 1
            self.recovery_transition = "recovery_hold_close"
            self._transition(TaskPhase.RECOVERY_HOLD, time_s)
        else:
            self._abort(observation, time_s, reason)

    def _target_base(self, observation: Mapping[str, Any], local_position: np.ndarray) -> np.ndarray:
        return target_local_to_robot_base(observation, local_position)

    def _public_can_inside_target(self, observation: Mapping[str, Any]) -> bool:
        can = np.asarray(observation["can_pos_robot_base"], dtype=float)
        frame = np.asarray(observation["goal_frame_pos_robot_base"], dtype=float)
        extents = np.asarray(observation["goal_inner_half_extents_target"], dtype=float)
        if can.shape != (3,) or frame.shape != (3,) or extents.shape != (2,):
            raise ShakeBenchOracleError("public Can and target fields have invalid shapes")
        local = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float)).T.dot(
            can - frame
        )
        # The profile radius is the conservative horizontal support envelope
        # of the compiled Can collision geometry, not a visual/COM proxy.
        radius = float(self.context.can_collision_radius_m)
        return bool(np.all(np.abs(local[:2]) + radius <= extents))

    def _public_can_at_target_support(self, observation: Mapping[str, Any]) -> bool:
        """Public height gate for a release: lower Can support near target bottom."""

        can = np.asarray(observation["can_pos_robot_base"], dtype=float)
        support_z_target = self._target_local_z(observation, can) + self.context.can_collision_lower_support_m
        return bool(abs(support_z_target) <= self.profile.placement_support_tolerance_m)

    def _update_public_kinematics(self, observation: Mapping[str, Any], time_s: float) -> None:
        can_position = np.asarray(observation["can_pos_robot_base"], dtype=float)
        eef_position = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        if can_position.shape != (3,) or not np.all(np.isfinite(can_position)):
            raise ShakeBenchOracleError("can_pos_robot_base must be finite length three")
        if eef_position.shape != (3,) or not np.all(np.isfinite(eef_position)):
            raise ShakeBenchOracleError("robot0_eef_pos_robot_base must be finite length three")
        self._relative_kinematics = self._relative_kinematics_tracker.update(observation, float(time_s))
        self._public_linear_speed_m_s = self._relative_kinematics.target_can_linear_speed_m_s
        self._public_angular_speed_rad_s = self._relative_kinematics.target_can_angular_speed_rad_s
        self._public_eef_velocity_m_s = np.zeros(3, dtype=float)
        self._public_eef_speed_m_s = 0.0
        if self._last_public_eef_position is not None and self._last_public_time_s is not None:
            dt = float(time_s) - self._last_public_time_s
            if dt > 0.0:
                self._public_eef_velocity_m_s = (eef_position - self._last_public_eef_position) / dt
                self._public_eef_speed_m_s = float(np.linalg.norm(self._public_eef_velocity_m_s))
        if self._relative_kinematics.history_valid:
            if (
                self._public_linear_speed_m_s <= self.profile.verify_linear_speed_limit_m_s
                and self._public_angular_speed_rad_s <= self.profile.verify_angular_speed_limit_rad_s
            ):
                self._public_stable_samples += 1
            else:
                self._public_stable_samples = 0
        else:
            self._public_stable_samples = 1
        self._last_public_can_position = can_position.copy()
        self._last_public_eef_position = eef_position.copy()
        self._last_public_can_rotation = _quat_xyzw_to_matrix(
            np.asarray(observation["can_quat_robot_base"], dtype=float)
        )
        self._last_public_time_s = float(time_s)

    def _public_verify_ok(self, observation: Mapping[str, Any]) -> bool:
        return bool(
            self._public_can_inside_target(observation)
            and self._public_gripper_open(observation)
            and self._public_linear_speed_m_s <= self.profile.verify_linear_speed_limit_m_s
            and self._public_angular_speed_rad_s <= self.profile.verify_angular_speed_limit_rad_s
            and self._public_stable_samples >= self.profile.verify_stability_samples
        )

    def _phase_goal(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        can = np.asarray(observation["can_pos_robot_base"], dtype=float)
        eef = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        if can.shape != (3,):
            raise ShakeBenchOracleError("can_pos_robot_base must be length three")
        z_bounds = np.asarray(observation["goal_z_bounds_target"], dtype=float)
        if z_bounds.shape != (2,):
            raise ShakeBenchOracleError("goal_z_bounds_target must be length two")
        target_top = float(z_bounds[1])
        if self.phase in {TaskPhase.SETTLE, TaskPhase.APPROACH}:
            return can + np.array((0.0, 0.0, self.profile.approach_height_m)), self.profile.gripper_open_action
        target_rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
        target_z = target_rotation[:, 2]
        if self.phase == TaskPhase.CLEARANCE_LIFT:
            if self._clearance_start_eef_position is None:
                self._clearance_start_eef_position = eef.copy()
            return (
                self._clearance_start_eef_position + target_z * self.profile.clearance_lift_height_m,
                self.profile.gripper_open_action,
            )
        if self.phase == TaskPhase.LATERAL_ALIGN_ABOVE_CAN:
            anchor_can = self._anchored_can_base(observation)
            return anchor_can + target_z * self.profile.approach_height_m, self.profile.gripper_open_action
        if self.phase == TaskPhase.ALIGN_SETTLE:
            anchor_can = self._anchored_can_base(observation)
            return anchor_can + target_z * self.profile.approach_height_m, self.profile.gripper_open_action
        if self.phase in {TaskPhase.DESCEND, TaskPhase.VERTICAL_DESCEND}:
            anchor_can = (
                self._anchored_can_base(observation) if self._anchor_worktable_can_transform is not None else can
            )
            return anchor_can + target_z * self.profile.grasp_height_m, self.profile.gripper_open_action
        if self.phase in {TaskPhase.GRASP, TaskPhase.GRASP_CLOSE}:
            anchor_can = (
                self._anchored_can_base(observation) if self._anchor_worktable_can_transform is not None else can
            )
            return anchor_can + target_z * self.profile.grasp_height_m, self.profile.gripper_close_action
        if self.phase == TaskPhase.PRELIFT_VERIFY:
            if self._prelift_start_eef_position is None:
                raise ShakeBenchOracleError("PRELIFT_VERIFY requires a saved EEF reference")
            target_rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
            return (
                self._prelift_start_eef_position + target_z * self.profile.prelift_height_m,
                self.profile.gripper_close_action,
            )
        if self.phase == TaskPhase.LIFT:
            anchor_can = (
                self._anchored_can_base(observation) if self._anchor_worktable_can_transform is not None else can
            )
            return anchor_can + target_z * self.profile.transport_height_m, self.profile.gripper_close_action
        target = self._target_base(observation, np.array((0.0, 0.0, target_top + self.profile.transport_height_m)))
        if self.phase == TaskPhase.TRANSPORT:
            return target, self.profile.gripper_close_action
        if self.phase == TaskPhase.PLACE:
            if self._placement_start_eef_position is not None:
                # Descend on the already aligned transport x/y.  A lateral
                # recentering command while lowering pulls a gripped Can out
                # of the shallow target and creates an avoidable slip.
                support_error = self._target_local_z(observation, can) + self.context.can_collision_lower_support_m
                normal_correction = -support_error * target_z
                placement = eef + normal_correction
                return (
                    np.array(
                        (
                            self._placement_start_eef_position[0] + normal_correction[0],
                            self._placement_start_eef_position[1] + normal_correction[1],
                            placement[2],
                        )
                    ),
                    self.profile.gripper_close_action,
                )
            return (
                self._target_base(observation, np.array((0.0, 0.0, self.profile.placement_height_m))),
                self.profile.gripper_close_action,
            )
        if self.phase == TaskPhase.RELEASE:
            return (
                self._release_eef_position.copy() if self._release_eef_position is not None else eef,
                self.profile.gripper_open_action,
            )
        if self.phase == TaskPhase.RECOVERY_HOLD:
            hold = self._recovery_start_eef_position if self._recovery_start_eef_position is not None else eef
            return hold, self.profile.gripper_close_action
        if self.phase == TaskPhase.RECOVERY_LOWER_IF_NEEDED:
            lower = (
                self._recovery_lower_goal
                if self._recovery_lower_goal is not None
                else can + target_z * self.profile.grasp_height_m
            )
            return lower, self.profile.gripper_close_action
        if self.phase == TaskPhase.WAIT_PUBLIC_SETTLE:
            return eef, self.profile.gripper_close_action
        if self.phase == TaskPhase.RECOVERY_OPEN:
            return eef, self.profile.gripper_open_action
        if self.phase in {TaskPhase.CLEARANCE_RETREAT, TaskPhase.RETREAT}:
            return eef + target_z * self.profile.clearance_lift_height_m, self.profile.gripper_open_action
        if self.phase == TaskPhase.RE_ALIGN:
            return can + target_z * self.profile.approach_height_m, self.profile.gripper_open_action
        return target, self.profile.gripper_open_action

    def _desired_eef_rotation(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._eef_reference_rotation is None:
            raise ShakeBenchOracleError("EEF orientation is not initialized")
        target_rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
        if self._initial_target_rotation is None:
            self._initial_target_rotation = target_rotation
        if self.phase in {TaskPhase.TRANSPORT, TaskPhase.PLACE, TaskPhase.RELEASE, TaskPhase.VERIFY}:
            # Preserve the frozen tool-to-target orientation while aligning
            # the tool's transport axis with the moving target frame.
            return target_rotation.dot(self._initial_target_rotation.T).dot(self._eef_reference_rotation)
        return self._eef_reference_rotation

    def _advance(self, observation: Mapping[str, Any], time_s: float) -> None:
        elapsed = float(time_s) - self.phase_entered_s
        can = np.asarray(observation["can_pos_robot_base"], dtype=float)
        if float(np.linalg.norm(can)) > self.profile.public_workspace_radius_m:
            self._recover_or_fail(observation, time_s, "workspace_risk")
            return
        if (
            self._anchor_worktable_can_transform is not None
            and self.phase
            in {
                TaskPhase.LATERAL_ALIGN_ABOVE_CAN,
                TaskPhase.ALIGN_SETTLE,
                TaskPhase.DESCEND,
                TaskPhase.VERTICAL_DESCEND,
                TaskPhase.GRASP,
                TaskPhase.GRASP_CLOSE,
                TaskPhase.PRELIFT_VERIFY,
            }
            and self._anchor_error_m(observation) > self.profile.anchor_drift_tolerance_m
        ):
            # A moving Can is never chased by the frozen grasp waypoint.  The
            # recovery primitive keeps the Panda closed, reverses the latest
            # safe path, and only re-anchors after a public stable/bounds gate.
            self._recover_or_fail(observation, time_s, "public_anchor_drift")
            return
        exempt = {
            TaskPhase.SETTLE,
            TaskPhase.GRASP,
            TaskPhase.GRASP_CLOSE,
            TaskPhase.PRELIFT_VERIFY,
            TaskPhase.ALIGN_SETTLE,
            TaskPhase.RELEASE,
            TaskPhase.VERIFY,
            TaskPhase.RECOVERY_HOLD,
            TaskPhase.RECOVERY_LOWER_IF_NEEDED,
            TaskPhase.WAIT_PUBLIC_SETTLE,
            TaskPhase.RECOVERY_OPEN,
            TaskPhase.CLEARANCE_RETREAT,
            TaskPhase.RE_ALIGN,
        }
        if self.phase not in exempt and elapsed > self.profile.phase_deadline_s:
            self._recover_or_fail(observation, time_s, "phase_deadline")
            return

        if self.phase in {TaskPhase.PRELIFT_VERIFY, TaskPhase.LIFT, TaskPhase.TRANSPORT, TaskPhase.PLACE}:
            held = self._public_grasp_held(observation)
            slip_translation, slip_rotation = self._public_grasp_slip(observation)
            self._last_slip_translation_m, self._last_slip_rotation_rad = slip_translation, slip_rotation
            loss = not held or slip_translation > self.profile.grasp_slip_tolerance_m
            self._grasp_loss_samples = self._grasp_loss_samples + 1 if loss else 0
            severe_slip = slip_translation >= (
                self.profile.grasp_severe_slip_fraction_of_can_radius * self.context.can_collision_radius_m
            )
            if severe_slip or self._grasp_loss_samples >= self.profile.grasp_loss_confirm_samples:
                self._recover_or_fail(
                    observation,
                    time_s,
                    (
                        "public_grasp_slip"
                        if slip_translation > self.profile.grasp_slip_tolerance_m
                        else "public_grasp_loss"
                    ),
                )
                return
            if self.phase in {TaskPhase.TRANSPORT, TaskPhase.PLACE} and self._public_container_risk(observation):
                self.last_recovery_reason = "public_container_risk"

        eef = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        goal, _ = self._phase_goal(observation)
        if self.phase in {TaskPhase.DESCEND, TaskPhase.VERTICAL_DESCEND}:
            error = eef - goal
            close = bool(
                np.linalg.norm(error[:2]) <= self.profile.descend_tolerance_m
                and abs(float(error[2])) <= self.profile.descend_vertical_tolerance_m
            )
        else:
            tolerance = (
                self.profile.approach_tolerance_m
                if self.phase == TaskPhase.APPROACH
                else self.profile.position_tolerance_m
            )
            close = bool(np.linalg.norm(eef - goal) <= tolerance)

        if self.phase == TaskPhase.SETTLE and elapsed >= self.profile.settle_s:
            self._transition(TaskPhase.APPROACH, time_s)
        elif self.phase == TaskPhase.APPROACH and close:
            self._last_swept_clearance_certificate = dict(self._swept_clearance_to_goal(observation, goal, can))
            if not self._last_swept_clearance_certificate["passed"]:
                self._recover_or_fail(observation, time_s, "public_tool_clearance")
                return
            # ``approach_height_m`` is itself the certified lateral standoff:
            # the complete fingertip support points already clear the Can.
            # An unconditional extra lift here caused a down-up-down reversal
            # before every nominal grasp without adding a stronger gate.
            self._begin_lateral_alignment(observation, time_s)
        elif self.phase == TaskPhase.CLEARANCE_LIFT and close and elapsed >= self.profile.clearance_lift_s:
            self._last_swept_clearance_certificate = dict(
                self._swept_clearance_to_goal(observation, self._phase_goal(observation)[0], can)
            )
            if not self._last_swept_clearance_certificate["passed"]:
                self._recover_or_fail(observation, time_s, "public_tool_clearance")
                return
            self._begin_lateral_alignment(observation, time_s)
        elif self.phase == TaskPhase.LATERAL_ALIGN_ABOVE_CAN and close:
            self._transition(TaskPhase.ALIGN_SETTLE, time_s)
        elif self.phase == TaskPhase.ALIGN_SETTLE:
            certificate = self.public_tool_clearance_certificate(observation, self._anchored_can_base(observation))
            if self._public_can_stable_and_in_bounds(observation) and certificate["passed"]:
                self._align_stable_samples += 1
            else:
                self._align_stable_samples = 0
            if (
                close
                and elapsed >= self.profile.align_settle_s
                and self._align_stable_samples >= self.profile.align_required_samples
            ):
                self._transition(TaskPhase.VERTICAL_DESCEND, time_s)
        elif self.phase in {TaskPhase.DESCEND, TaskPhase.VERTICAL_DESCEND} and close:
            if self._anchor_worktable_can_transform is None:
                self._capture_grasp_anchor(observation, time_s)
            self._transition(TaskPhase.GRASP_CLOSE, time_s)
        elif self.phase in {TaskPhase.GRASP, TaskPhase.GRASP_CLOSE} and elapsed >= self.profile.grasp_s:
            if self._public_grasp_held(observation):
                self._begin_prelift_verify(observation, time_s)
            else:
                self._recover_or_fail(observation, time_s, "public_grasp_not_established")
        elif self.phase == TaskPhase.PRELIFT_VERIFY:
            if self._prelift_start_eef_position is None or self._prelift_start_can_height_m is None:
                raise ShakeBenchOracleError("PRELIFT_VERIFY missing candidate state")
            rotation = _quat_xyzw_to_matrix(np.asarray(observation["goal_frame_quat_robot_base"], dtype=float))
            eef_lift = float(np.dot(eef - self._prelift_start_eef_position, rotation[:, 2]))
            can_lift = self._target_local_z(observation, can) - self._prelift_start_can_height_m
            follows = (
                eef_lift >= self.profile.prelift_min_follow_m
                and can_lift >= eef_lift - self.profile.prelift_geometry_tolerance_m
            )
            stable = (
                self._public_grasp_held(observation)
                and self._last_slip_translation_m <= self.profile.prelift_geometry_tolerance_m
            )
            self._prelift_follow_samples = self._prelift_follow_samples + 1 if follows and stable else 0
            if self._prelift_follow_samples >= self.profile.prelift_required_samples:
                self._transition(TaskPhase.LIFT, time_s)
            elif elapsed >= self.profile.prelift_s:
                self._recover_or_fail(observation, time_s, "public_grasp_not_established")
        elif self.phase == TaskPhase.LIFT and elapsed >= self.profile.lift_s:
            self._transition(TaskPhase.TRANSPORT, time_s)
        elif self.phase == TaskPhase.TRANSPORT and close:
            self._placement_start_eef_position = eef.copy()
            self._transition(TaskPhase.PLACE, time_s)
        # Containment alone is not a release waypoint: the EEF must first
        # reach the frozen low placement height while the Can remains held.
        # Otherwise the gripper opens one policy sample at transport height
        # and the Can falls through the shallow target.
        elif self.phase == TaskPhase.PLACE and close:
            if self._public_can_inside_target(observation) and self._public_can_at_target_support(observation):
                self._release_eef_position = eef.copy()
                self._transition(TaskPhase.RELEASE, time_s)
        elif self.phase == TaskPhase.RELEASE and elapsed >= self.profile.release_s:
            self._transition(TaskPhase.VERIFY, time_s)
        elif self.phase == TaskPhase.VERIFY and elapsed >= self.profile.verify_s:
            public_outside, public_unstable = not self._public_can_inside_target(
                observation
            ), not self._public_verify_ok(observation)
            if public_outside or (
                public_unstable and elapsed >= self.profile.verify_s + self.profile.completion_evaluator_settle_s
            ):
                self._recover_or_fail(
                    observation, time_s, "public_placement_rebound" if public_outside else "public_placement_loss"
                )
        elif self.phase == TaskPhase.RECOVERY_HOLD and elapsed >= self.profile.recovery_hold_s:
            self._transition(TaskPhase.RECOVERY_LOWER_IF_NEEDED, time_s)
        elif self.phase == TaskPhase.RECOVERY_LOWER_IF_NEEDED and elapsed >= self.profile.recovery_lower_s:
            self._transition(TaskPhase.WAIT_PUBLIC_SETTLE, time_s)
        elif self.phase == TaskPhase.WAIT_PUBLIC_SETTLE:
            if self._public_safe_to_open(observation):
                self._transition(TaskPhase.RECOVERY_OPEN, time_s)
            elif elapsed >= self.profile.recovery_settle_s:
                state = self._public_recoverability(observation)
                self._record_event("recovery_settle_timeout", observation, time_s, "reassess_recovery")
                # A local settle deadline is not an episode deadline.  Keep
                # the same physical rollout alive whenever a safe primitive
                # remains; the global runner horizon is the only budget.
                if state["recoverable"]:
                    self.phase_entered_s = float(time_s)
                    self.recovery_transition = "recovery_wait_continue"
                else:
                    self._abort(observation, time_s, "edge_risk" if state["edge_unrecoverable"] else "workspace_risk")
        elif self.phase == TaskPhase.RECOVERY_OPEN and elapsed >= self.profile.recovery_open_s:
            self._transition(TaskPhase.CLEARANCE_RETREAT, time_s)
        elif (
            self.phase in {TaskPhase.CLEARANCE_RETREAT, TaskPhase.RETREAT}
            and elapsed >= self.profile.recovery_retreat_s
        ):
            self._transition(TaskPhase.RE_ALIGN, time_s)
        elif self.phase == TaskPhase.RE_ALIGN and elapsed >= self.profile.clearance_lift_s:
            self._anchor_worktable_can_transform = None
            self._anchor_worktable_eef_goal = None
            self._grasp_eef_can_transform = None
            self._grasp_reference_established = False
            self._recovery_reverse_goal = None
            self._transition(TaskPhase.APPROACH, time_s)

    def command(self, observation: Mapping[str, Any], time_s: float) -> tuple[np.ndarray, float]:
        if not np.isfinite(float(time_s)) or time_s < 0.0:
            raise ShakeBenchOracleError("time_s must be finite and non-negative")
        if self.phase in {TaskPhase.COMPLETE, TaskPhase.ABORTED}:
            return np.zeros(6), self.profile.gripper_open_action
        eef = np.asarray(observation["robot0_eef_pos_robot_base"], dtype=float)
        if eef.shape != (3,) or not np.all(np.isfinite(eef)):
            raise ShakeBenchOracleError("robot0_eef_pos_robot_base must be a finite three-vector")
        if self._eef_reference_rotation is None:
            quat = np.asarray(observation["robot0_eef_quat_robot_base"], dtype=float)
            if quat.shape != (4,) or not np.all(np.isfinite(quat)):
                raise ShakeBenchOracleError("robot0_eef_quat_robot_base must be a finite quaternion")
            self._eef_reference_rotation = _quat_xyzw_to_matrix(quat)
            self._initial_target_rotation = _quat_xyzw_to_matrix(
                np.asarray(observation["goal_frame_quat_robot_base"], dtype=float)
            )
        self._update_public_kinematics(observation, float(time_s))
        self._advance(observation, float(time_s))
        if self.phase is TaskPhase.ABORTED:
            return np.zeros(6), self.profile.gripper_open_action
        if self.phase == TaskPhase.SETTLE:
            return np.zeros(6), self.profile.gripper_open_action
        goal, gripper = self._phase_goal(observation)
        gripper = self._capability_policy().gripper_action
        delta = np.zeros(6)
        delta[:3] = goal - eef
        current_rotation = _quat_xyzw_to_matrix(np.asarray(observation["robot0_eef_quat_robot_base"], dtype=float))
        delta[3:] = _rotation_vector_from_matrix(self._desired_eef_rotation(observation).dot(current_rotation.T))
        return delta, gripper

    def diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "phase": self.phase.value,
                "capability": self._capability_policy().capability.value,
                "capability_policy": {
                    "max_translation_normalized": self._capability_policy().max_translation_normalized,
                    "max_orientation_normalized": self._capability_policy().max_orientation_normalized,
                    "gripper_action": self._capability_policy().gripper_action,
                    "vibration_feedforward_allowed": self._capability_policy().vibration_feedforward_allowed,
                },
                "phase_entered_s": self.phase_entered_s,
                "failure_reason": self.failure_reason,
                "abort_reason": self.abort_reason,
                "last_recovery_reason": self.last_recovery_reason,
                "recovery_count": self.recovery_count,
                "recovery_transition": self.recovery_transition,
                "public_linear_speed_m_s": self._public_linear_speed_m_s,
                "public_angular_speed_rad_s": self._public_angular_speed_rad_s,
                "public_eef_speed_m_s": self._public_eef_speed_m_s,
                "public_stable_samples": self._public_stable_samples,
                "slip_translation_m": self._last_slip_translation_m,
                "slip_rotation_rad": self._last_slip_rotation_rad,
                "grasp_confidence": dict(self._grasp_confidence),
                "prelift_follow_samples": self._prelift_follow_samples,
                "align_stable_samples": self._align_stable_samples,
                "anchor_timestamp_s": self._anchor_timestamp_s,
                "anchor_worktable_can_transform": (
                    self._anchor_worktable_can_transform.tolist()
                    if self._anchor_worktable_can_transform is not None
                    else None
                ),
                "anchor_worktable_eef_grasp_goal": (
                    self._anchor_worktable_eef_goal.tolist() if self._anchor_worktable_eef_goal is not None else None
                ),
                "anchor_clearance_certificate": dict(self._anchor_clearance_certificate),
                "recovery_start_eef_position": (
                    self._recovery_start_eef_position.tolist()
                    if self._recovery_start_eef_position is not None
                    else None
                ),
                "recovery_lower_goal": (
                    self._recovery_lower_goal.tolist() if self._recovery_lower_goal is not None else None
                ),
                "recovery_reverse_goal": (
                    self._recovery_reverse_goal.tolist() if self._recovery_reverse_goal is not None else None
                ),
                "recovery_events": list(self._recovery_events),
                "relative_kinematics": (
                    self._relative_kinematics.to_dict() if self._relative_kinematics is not None else None
                ),
                "swept_clearance_certificate": dict(self._last_swept_clearance_certificate),
            }
        )

    @property
    def controller_events(self) -> list[dict[str, Any]]:
        """Canonical event sequence for the runner artifact."""

        return [dict(event) for event in self._recovery_events]

    def record_evaluator_not_latched(self, observation: Mapping[str, Any], time_s: float) -> None:
        """Expose the post-verify diagnostic without allowing it to terminate."""

        self._record_event("evaluator_not_latched_after_public_verify", observation, time_s, "continue_rollout")
