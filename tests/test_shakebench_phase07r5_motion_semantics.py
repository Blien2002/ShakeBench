"""Public-seam regressions for Phase 07R5 motion semantics."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.utils.shakebench_oracle import (
    MOTION_CAPABILITY_BY_PHASE,
    MotionCapability,
    PublicRelativeKinematicsTracker,
    ShakeBenchOracleController,
    TaskPhase,
    VibrationEstimate,
    WorktableTaskContext,
    vibration_estimate_from_public_observation,
)
from tests.shakebench_test_helpers import build_current_observation


def _observation() -> dict[str, np.ndarray]:
    return build_current_observation(
        goal_frame_pos=(0.0, 0.0, 0.0),
        can_pos=(0.0, 0.0, 0.04),
        eef_pos=(-0.04, 0.0, 0.125),
        fingertip_pos=(0.0, -0.015, 0.0934, 0.0, 0.015, 0.0934),
        gripper_state=(0.015, -0.015, 0.0, 0.0),
    )


def _z_quaternion(angle: float) -> np.ndarray:
    return np.asarray((0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)), dtype=np.float32)


def test_every_production_phase_has_one_explicit_motion_capability():
    assert set(MOTION_CAPABILITY_BY_PHASE) == set(TaskPhase)
    assert all(isinstance(value, MotionCapability) for value in MOTION_CAPABILITY_BY_PHASE.values())
    assert len(MOTION_CAPABILITY_BY_PHASE) == len(set(MOTION_CAPABILITY_BY_PHASE))


def test_contact_alias_phases_share_translation_and_gripper_limits():
    observation = _observation()
    observation["robot0_eef_pos_robot_base"] = np.asarray((-0.10, 0.0, 0.125), dtype=np.float32)
    results = []
    for phase in (TaskPhase.DESCEND, TaskPhase.VERTICAL_DESCEND):
        controller = ShakeBenchOracleController()
        controller.executive.phase = phase
        results.append(controller.action(observation, time_s=0.1))
    np.testing.assert_allclose(results[0], results[1], atol=1e-7)
    assert results[0][0] == pytest.approx(0.30, abs=1e-7)
    assert results[0][6] == pytest.approx(-1.0)


def test_target_relative_motion_is_zero_for_rigid_common_translation_and_rotation():
    controller = ShakeBenchOracleController()
    first = _observation()
    controller.action(first, time_s=0.0)

    translated = {key: value.copy() for key, value in first.items()}
    translated["goal_frame_pos_robot_base"] += np.array((0.005, -0.002, 0.0), dtype=np.float32)
    translated["object_pos_robot_base"] += np.array((0.005, -0.002, 0.0), dtype=np.float32)
    controller.action(translated, time_s=0.05)
    assert controller.executive.diagnostics()["public_linear_speed_m_s"] == pytest.approx(0.0, abs=1e-7)
    assert controller.executive.diagnostics()["public_angular_speed_rad_s"] == pytest.approx(0.0, abs=1e-7)

    rotated = {key: value.copy() for key, value in translated.items()}
    rotation = _z_quaternion(np.pi / 6.0)
    rotated["goal_frame_quat_robot_base"] = rotation
    rotated["object_quat_robot_base"] = rotation.copy()
    controller.action(rotated, time_s=0.10)
    assert controller.executive.diagnostics()["public_linear_speed_m_s"] == pytest.approx(0.0, abs=1e-6)
    assert controller.executive.diagnostics()["public_angular_speed_rad_s"] == pytest.approx(0.0, abs=1e-6)


def test_relative_motion_distinguishes_can_and_target_motion_and_resets_bad_time():
    tracker = PublicRelativeKinematicsTracker(WorktableTaskContext())
    first = _observation()
    tracker.update(first, 0.0)

    can_only = {key: value.copy() for key, value in first.items()}
    can_only["object_pos_robot_base"][0] += 0.005
    snapshot = tracker.update(can_only, 0.05)
    assert snapshot.target_can_twist[0] == pytest.approx(0.10, abs=1e-6)
    assert np.all(np.isfinite(snapshot.target_can_acceleration))

    repeated = tracker.update(can_only, 0.05)
    assert repeated.reset_generation > snapshot.reset_generation
    assert repeated.history_valid is False
    assert np.all(np.isfinite(repeated.target_can_twist))

    backwards = tracker.update(first, 0.0)
    assert backwards.reset_generation > repeated.reset_generation
    assert np.all(np.isfinite(backwards.target_can_twist))


def test_anchor_drift_is_a_phase_transition_guard():
    controller = ShakeBenchOracleController()
    observation = _observation()
    controller.executive._capture_grasp_anchor(observation, 0.0)
    controller.executive.phase = TaskPhase.VERTICAL_DESCEND
    displaced = {key: value.copy() for key, value in observation.items()}
    displaced["object_pos_robot_base"][0] += controller.profile.anchor_drift_tolerance_m * 2.0
    controller.action(displaced, time_s=0.1)
    assert controller.executive.phase in {
        TaskPhase.RECOVERY_HOLD,
        TaskPhase.RECOVERY_OPEN,
        TaskPhase.FAILED,
    }
    assert controller.executive.last_recovery_reason == "public_anchor_drift"


def test_current_estimate_publishes_one_typed_support_contract():
    estimate = vibration_estimate_from_public_observation(_observation(), policy_time_s=0.2)
    assert isinstance(estimate, VibrationEstimate)
    assert estimate.semantic_quantity == "worktable_relative_to_robot_base_motion"
    assert estimate.frame == "robot_base"
    assert estimate.reference_point == "target_origin"
    assert estimate.current_relative_pose_robot_base.shape == (7,)
    assert estimate.current_relative_twist_robot_base.shape == (6,)
    assert estimate.current_relative_acceleration_robot_base.shape == (6,)
    assert estimate.policy_timestamp_s == pytest.approx(0.2)
    assert estimate.latency_s == pytest.approx(0.0)
    assert estimate.units["linear"] == "m/s2"


def test_current_estimate_is_reproducible_and_reads_only_the_public_task_state():
    observation = _observation()
    first = vibration_estimate_from_public_observation(observation, policy_time_s=0.2)
    second = vibration_estimate_from_public_observation(observation, policy_time_s=0.2)
    assert first.to_dict() == second.to_dict()
    moved = {key: value.copy() for key, value in observation.items()}
    moved["table_imu_window"] = np.linspace(-20.0, 20.0, 60, dtype=np.float32).reshape(10, 6)
    assert vibration_estimate_from_public_observation(moved, policy_time_s=0.2).to_dict() == first.to_dict()
