"""Public-seam regressions for Phase 07R5 motion semantics."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.utils.shakebench_isolator import relative_transfer_function
from robosuite.utils.shakebench_oracle import (
    MOTION_CAPABILITY_BY_PHASE,
    MotionCapability,
    OracleControllerProfile,
    PublicRelativeKinematicsTracker,
    ShakeBenchOracleController,
    TaskPhase,
    VibrationEstimate,
    WorktableTaskContext,
    future_relative_acceleration_from_public_program,
    vibration_estimate_from_public_observation,
)
from tests.shakebench_test_helpers import build_tier_observation


def _observation(tier: str = "V0") -> dict[str, np.ndarray]:
    return build_tier_observation(
        tier,
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
        controller = ShakeBenchOracleController("V0")
        controller.executive.phase = phase
        results.append(controller.action(observation, time_s=0.1))
    np.testing.assert_allclose(results[0], results[1], atol=1e-7)
    assert results[0][0] == pytest.approx(0.30, abs=1e-7)
    assert results[0][6] == pytest.approx(-1.0)


def test_target_relative_motion_is_zero_for_rigid_common_translation_and_rotation():
    controller = ShakeBenchOracleController("V0")
    first = _observation()
    controller.action(first, time_s=0.0)

    translated = {key: value.copy() for key, value in first.items()}
    translated["goal_frame_pos_robot_base"] += np.array((0.005, -0.002, 0.0), dtype=np.float32)
    translated["can_pos_robot_base"] += np.array((0.005, -0.002, 0.0), dtype=np.float32)
    controller.action(translated, time_s=0.05)
    assert controller.executive.diagnostics()["public_linear_speed_m_s"] == pytest.approx(0.0, abs=1e-7)
    assert controller.executive.diagnostics()["public_angular_speed_rad_s"] == pytest.approx(0.0, abs=1e-7)

    rotated = {key: value.copy() for key, value in translated.items()}
    rotation = _z_quaternion(np.pi / 6.0)
    rotated["goal_frame_quat_robot_base"] = rotation
    rotated["can_quat_robot_base"] = rotation.copy()
    controller.action(rotated, time_s=0.10)
    assert controller.executive.diagnostics()["public_linear_speed_m_s"] == pytest.approx(0.0, abs=1e-6)
    assert controller.executive.diagnostics()["public_angular_speed_rad_s"] == pytest.approx(0.0, abs=1e-6)


def test_relative_motion_distinguishes_can_and_target_motion_and_resets_bad_time():
    tracker = PublicRelativeKinematicsTracker(WorktableTaskContext())
    first = _observation()
    tracker.update(first, 0.0)

    can_only = {key: value.copy() for key, value in first.items()}
    can_only["can_pos_robot_base"][0] += 0.005
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
    controller = ShakeBenchOracleController("V0")
    observation = _observation()
    controller.executive._capture_grasp_anchor(observation, 0.0)
    controller.executive.phase = TaskPhase.VERTICAL_DESCEND
    displaced = {key: value.copy() for key, value in observation.items()}
    displaced["can_pos_robot_base"][0] += controller.profile.anchor_drift_tolerance_m * 2.0
    controller.action(displaced, time_s=0.1)
    assert controller.executive.phase in {
        TaskPhase.RECOVERY_HOLD,
        TaskPhase.RECOVERY_OPEN,
        TaskPhase.FAILED,
    }
    assert controller.executive.last_recovery_reason == "public_anchor_drift"


@pytest.mark.parametrize("tier", ("V0", "V1", "V2", "V3"))
def test_all_tiers_publish_one_typed_relative_support_contract(tier):
    estimate = vibration_estimate_from_public_observation(tier, _observation(tier), policy_time_s=0.2)
    assert isinstance(estimate, VibrationEstimate)
    assert estimate.semantic_quantity == "worktable_relative_to_robot_base_motion"
    assert estimate.frame == "robot_base"
    assert estimate.reference_point == "target_origin"
    assert estimate.current_relative_pose_robot_base.shape == (7,)
    assert estimate.current_relative_twist_robot_base.shape == (6,)
    assert estimate.current_relative_acceleration_robot_base.shape == (6,)
    assert estimate.prediction_timestamps_s.ndim == 1
    assert np.all(estimate.prediction_timestamps_s > estimate.policy_timestamp_s)
    assert estimate.predicted_relative_acceleration_robot_base.shape == (
        estimate.prediction_timestamps_s.size,
        6,
    )
    assert estimate.units["linear"] == "m/s2"


def test_shared_law_is_independent_of_provider_tier_for_the_same_estimate():
    observation = _observation("V2")
    observation["table_accel_in_deck_frame"] = np.asarray((1.0, -2.0, 0.5, 0.1, -0.2, 0.3), dtype=np.float32)
    first = vibration_estimate_from_public_observation("V2", observation, policy_time_s=0.2)
    second = vibration_estimate_from_public_observation("V2", observation, policy_time_s=0.2)
    assert first.to_dict() == second.to_dict()
    controller = ShakeBenchOracleController("V2")
    np.testing.assert_allclose(
        controller.control_law.apply(np.zeros(6), first).compensated_delta_base,
        controller.control_law.apply(np.zeros(6), second).compensated_delta_base,
    )


def test_v3_preview_applies_quintic_ramp_derivative_terms_and_is_causal():
    observation = _observation("V3")
    observation["line_mask"][:] = False
    observation["line_mask"][0, 0] = True
    observation["line_accel_amplitude"][:] = 0.0
    observation["line_accel_amplitude"][0, 0] = 2.0
    observation["line_omega_rad_s"][:] = 1.0
    omega = 2.0 * np.pi * 2.0
    observation["line_omega_rad_s"][0, 0] = omega
    observation["ramp_duration_s"] = np.float32(1.0)
    query = 0.5
    profile = OracleControllerProfile()
    actual_linear, _ = future_relative_acceleration_from_public_program(
        observation,
        query,
        natural_frequency_hz=profile.future_isolator_fn_hz,
        damping_ratio=profile.future_isolator_zeta,
    )
    s = query
    ramp = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    ramp_first = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    ramp_second = 60.0 * s - 180.0 * s**2 + 120.0 * s**3
    theta = omega * query
    carrier_q = -2.0 / omega**2 * np.sin(theta)
    carrier_qdot = -2.0 / omega * np.cos(theta)
    carrier_qdd = 2.0 * np.sin(theta)
    expected_input = carrier_qdd * ramp + 2.0 * carrier_qdot * ramp_first + carrier_q * ramp_second
    transfer = relative_transfer_function(2.0, profile.future_isolator_fn_hz[0], profile.future_isolator_zeta[0])
    expected = (2.0 * ramp - 2.0 / omega**2 * ramp_second) * (
        transfer.real * np.sin(theta) + transfer.imag * np.cos(theta)
    ) + (-2.0 * 2.0 / omega * ramp_first) * (-transfer.imag * np.sin(theta) + transfer.real * np.cos(theta))
    old_shortcut = (transfer.real * np.sin(theta) + transfer.imag * np.cos(theta)) * ramp
    np.testing.assert_allclose(actual_linear[0], expected, atol=1.0e-8)
    assert abs(float(actual_linear[0]) - float(old_shortcut)) > 1.0e-8
    assert np.isfinite(expected_input)

    before = vibration_estimate_from_public_observation("V3", observation, policy_time_s=0.1)
    observation["line_accel_amplitude"][0, 0] = 4.0
    after = vibration_estimate_from_public_observation("V3", observation, policy_time_s=0.1)
    np.testing.assert_allclose(
        before.current_relative_acceleration_robot_base,
        after.current_relative_acceleration_robot_base,
    )
    assert not np.allclose(
        before.predicted_relative_acceleration_robot_base, after.predicted_relative_acceleration_robot_base
    )
    assert before.prediction_timestamps_s[0] > before.policy_timestamp_s
