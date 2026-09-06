"""Phase 07 public-seam tests for the shared State Oracle controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robosuite.scripts.shakebench_run_oracle import load_dev_states
from robosuite.utils.shakebench_isolator import relative_transfer_function
from robosuite.utils.shakebench_metrics import SuccessSnapshot, VibrationSuccessEvaluator
from robosuite.utils.shakebench_oracle import (
    OracleControllerProfile,
    ShakeBenchOracleController,
    ShakeBenchOracleError,
    TaskPhase,
    WorktableTaskContext,
    future_relative_acceleration_from_public_program,
    robot_base_to_worktable_local,
    target_local_to_robot_base,
    vibration_estimate_from_public_observation,
    worktable_local_to_robot_base,
)
from robosuite.utils.shakebench_providers import COMMON_STATE_KEYS, TIER_POLICY_KEYS
from tests.shakebench_test_helpers import build_tier_observation


def _public_observation(tier="V0", *, goal_quat=(0.0, 0.0, 0.0, 1.0)):
    values = build_tier_observation(
        tier,
        goal_frame_pos=(0.2, 0.3, 0.02),
        gripper_state=(0.0, 0.0, 0.0, 0.0),
        table_twist=np.arange(6, dtype=np.float32),
        table_accel=np.arange(6, dtype=np.float32),
        imu_window=np.ones((10, 6), dtype=np.float32),
    )
    values["goal_frame_quat_robot_base"] = np.asarray(goal_quat, dtype=np.float32)
    return values


def _snapshot(*, finger=False):
    return SuccessSnapshot(
        support_points_target_xy=np.array(((-0.01, -0.01), (0.01, 0.01))),
        target_inner_half_extents_m=(0.082, 0.072),
        target_bottom_contact_present=True,
        target_bottom_support_force_N=1.0,
        lower_support_z_m=0.0,
        finger_can_contact_present=finger,
        relative_linear_speed_m_s=0.0,
        relative_angular_speed_rad_s=0.0,
        illegal_penetration_m=0.0,
    )


def _bilateral_grasp_observation():
    """A public-only nominal Panda-pad / Can geometry fixture."""

    observation = _public_observation()
    observation["robot0_eef_pos_robot_base"] = np.zeros(3, dtype=np.float32)
    observation["robot0_eef_quat_robot_base"] = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    observation["robot0_fingertip_pos_robot_base"] = np.array(
        (0.0, -0.015, 0.0934, 0.0, 0.015, 0.0934), dtype=np.float32
    )
    observation["robot0_gripper_state"] = np.array((0.015, -0.015, 0.0, 0.0), dtype=np.float32)
    observation["can_pos_robot_base"] = np.array((-0.005, 0.0, 0.080), dtype=np.float32)
    return observation


def test_complete_public_goal_frame_distinguishes_rotated_targets_and_transforms_points():
    identity = _public_observation(goal_quat=(0.0, 0.0, 0.0, 1.0))
    rotated = _public_observation(goal_quat=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    local = np.array((0.04, 0.0, 0.01))
    np.testing.assert_allclose(target_local_to_robot_base(identity, local), (0.24, 0.3, 0.03), atol=1e-7)
    np.testing.assert_allclose(target_local_to_robot_base(rotated, local), (0.2, 0.34, 0.03), atol=1e-7)
    assert not np.array_equal(identity["goal_frame_quat_robot_base"], rotated["goal_frame_quat_robot_base"])
    for tier in ("V0", "V1", "V2", "V3"):
        observation = _public_observation(tier)
        assert set(COMMON_STATE_KEYS) <= set(observation)
        assert set(TIER_POLICY_KEYS[tier]) <= set(observation)
        for key in (
            "goal_frame_pos_robot_base",
            "goal_frame_quat_robot_base",
            "goal_inner_half_extents_target",
            "goal_z_bounds_target",
            "goal_orientation_mask",
        ):
            np.testing.assert_array_equal(observation[key], identity[key])


def test_one_internal_step_failure_resets_the_half_second_candidate_window():
    evaluator = VibrationSuccessEvaluator()
    dt = 0.0002
    for step in range(1251):
        assert not evaluator.evaluate(_snapshot(), step * dt).passed
    # A policy-rate-only sampler would not see this 0.2 ms failure and would
    # incorrectly latch at 0.50 s.  Physics-step semantics reset the window.
    assert not evaluator.evaluate(_snapshot(finger=True), 1251 * dt).passed
    for step in range(1252, 3752):
        assert not evaluator.evaluate(_snapshot(), step * dt).passed
    assert evaluator.evaluate(_snapshot(), 3752 * dt).passed


def test_provider_estimates_are_public_only_and_v2_is_current_only():
    for tier in ("V0", "V1", "V2", "V3"):
        estimate = vibration_estimate_from_public_observation(tier, _public_observation(tier))
        assert estimate.tier == tier
        assert estimate.future_program_available is (tier == "V3")
    v2 = vibration_estimate_from_public_observation("V2", _public_observation("V2"))
    np.testing.assert_array_equal(v2.current_linear_accel_m_s2, (0.0, 1.0, 2.0))
    with pytest.raises(ShakeBenchOracleError, match="privileged"):
        vibration_estimate_from_public_observation("V0", {**_public_observation(), "privileged_contacts": {}})


def test_shared_profile_and_neutral_estimate_produce_identical_actions():
    profile = OracleControllerProfile()
    v0 = ShakeBenchOracleController("V0", profile)
    v0_again = ShakeBenchOracleController("V0", profile)
    action = v0.action(_public_observation(), time_s=0.0)
    np.testing.assert_array_equal(action, v0_again.action(_public_observation(), time_s=0.0))
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)
    assert v0.profile_sha256 == v0_again.profile_sha256


def test_stationary_v1_specific_force_does_not_create_a_gravity_control_bias():
    profile = OracleControllerProfile()
    v0 = ShakeBenchOracleController("V0", profile)
    v1 = ShakeBenchOracleController("V1", profile)
    v1_observation = _public_observation("V1")
    v1_observation["deck_imu_window"][:] = 0.0
    v1_observation["deck_imu_window"][:, :3] = np.array((0.0, 0.0, 9.80665), dtype=np.float32)
    np.testing.assert_allclose(
        v0.action(_public_observation(), time_s=0.0),
        v1.action(v1_observation, time_s=0.0),
        atol=1e-7,
    )


def test_v1_reset_on_time_reversal_returns_to_nominal_estimator_state():
    controller = ShakeBenchOracleController("V1")
    observation = _public_observation("V1")
    observation["deck_imu_window"][:] = 0.0
    observation["deck_imu_window"][:, :3] = np.array((0.0, 0.0, 9.80665), dtype=np.float32)
    controller.action(observation, time_s=0.1)
    generation = controller.v1_estimator.reset_generation
    controller.action(observation, time_s=0.0)
    assert controller.v1_estimator.reset_generation > generation
    np.testing.assert_allclose(controller.v1_estimator.attitude_control_from_sensor, np.eye(3), atol=1e-12)
    assert controller.last_trace["estimate"]["timestamp_s"] == 0.0


def test_v3_future_transfer_matches_public_isolator_formula_at_three_frequency_regions():
    profile = OracleControllerProfile()
    for frequency_hz in (0.5, 4.0, 20.0):
        observation = _public_observation("V3")
        observation["line_mask"][:] = False
        observation["line_accel_amplitude"][0, 0] = 1.0
        observation["line_mask"][0, 0] = True
        observation["line_omega_rad_s"][0, 0] = 2.0 * np.pi * frequency_hz
        observation["line_phase_at_episode_zero"][0, 0] = 0.37
        observation["ramp_duration_s"] = np.float32(0.0)
        query = 0.23
        linear, angular = future_relative_acceleration_from_public_program(
            observation,
            query,
            natural_frequency_hz=profile.future_isolator_fn_hz,
            damping_ratio=profile.future_isolator_zeta,
            deck_to_control_rotation=profile.future_deck_to_control_rotation,
        )
        transfer = relative_transfer_function(
            frequency_hz, profile.future_isolator_fn_hz[0], profile.future_isolator_zeta[0]
        )
        theta = 2.0 * np.pi * frequency_hz * query + 0.37
        expected = transfer.real * np.sin(theta) + transfer.imag * np.cos(theta)
        np.testing.assert_allclose(linear[0], expected, atol=1e-6)
        np.testing.assert_allclose(linear[1:], np.zeros(2), atol=1e-10)
        np.testing.assert_allclose(angular, np.zeros(3), atol=1e-10)


def test_neutral_v3_program_is_action_equivalent_to_v2_and_frame_mapping_is_explicit():
    v2 = _public_observation("V2")
    v3 = _public_observation("V3")
    v2["table_accel_in_deck_frame"] = np.array((0.2, -0.1, 0.3, 0.01, -0.02, 0.03), dtype=np.float32)
    v2["table_twist_in_deck_frame"] = np.array((0.0, 0.0, 0.0, 0.1, -0.1, 0.2), dtype=np.float32)
    v3["table_accel_in_deck_frame"] = v2["table_accel_in_deck_frame"].copy()
    v3["table_twist_in_deck_frame"] = v2["table_twist_in_deck_frame"].copy()
    v3["line_mask"][:] = False
    v2_controller = ShakeBenchOracleController("V2")
    v3_controller = ShakeBenchOracleController("V3")
    np.testing.assert_allclose(v2_controller.action(v2, time_s=0.1), v3_controller.action(v3, time_s=0.1), atol=1e-7)

    rotated = dict(v3)
    rotated["line_mask"] = np.zeros((6, 12), dtype=np.bool_)
    rotated["line_mask"][0, 0] = True
    rotated["line_accel_amplitude"] = np.zeros((6, 12), dtype=np.float32)
    rotated["line_accel_amplitude"][0, 0] = 4.0
    rotated["line_omega_rad_s"][0, 0] = 2.0 * np.pi * 1.0
    rotated["line_phase_at_episode_zero"] = np.zeros((6, 12), dtype=np.float32)
    rotated["ramp_duration_s"] = np.float32(0.0)
    estimate = vibration_estimate_from_public_observation(
        "V3",
        rotated,
        policy_time_s=0.23,
        future_query_horizon_s=0.1,
        future_deck_to_control_rotation=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert abs(float(estimate.future_linear_accel_m_s2[1])) > 1.0e-6
    assert abs(float(estimate.future_linear_accel_m_s2[0])) < 1.0e-6


def test_public_gripper_closure_triggers_bounded_recovery_not_false_transport():
    controller = ShakeBenchOracleController("V0")
    controller.executive.phase = TaskPhase.LIFT
    observation = _public_observation()
    observation["robot0_gripper_state"] = np.array((0.03, -0.03, 0.0, 0.0), dtype=np.float32)
    observation["can_pos_robot_base"] = np.array((0.3, -0.1, 0.04), dtype=np.float32)
    controller.action(observation, time_s=1.0)
    controller.action(observation, time_s=1.05)
    assert controller.executive.phase is TaskPhase.RECOVERY_HOLD
    assert controller.executive.recovery_transition == "recovery_hold_close"


def test_phase_to_gripper_wiring_uses_the_public_panda_contract():
    observation = _bilateral_grasp_observation()
    for phase in (TaskPhase.SETTLE, TaskPhase.APPROACH, TaskPhase.DESCEND, TaskPhase.RELEASE, TaskPhase.VERIFY):
        controller = ShakeBenchOracleController("V0")
        controller.executive.phase = phase
        assert controller.action(observation, time_s=0.1)[6] == pytest.approx(-1.0)
    for phase in (TaskPhase.GRASP, TaskPhase.PRELIFT_VERIFY, TaskPhase.LIFT, TaskPhase.TRANSPORT, TaskPhase.PLACE):
        controller = ShakeBenchOracleController("V0")
        controller.executive.phase = phase
        if phase is TaskPhase.PRELIFT_VERIFY:
            controller.executive._prelift_start_eef_position = observation["robot0_eef_pos_robot_base"].copy()
            controller.executive._prelift_start_can_height_m = 0.085
        assert controller.action(observation, time_s=0.1)[6] == pytest.approx(1.0)
    controller = ShakeBenchOracleController("V0")
    controller.executive.phase = TaskPhase.FAILED
    assert controller.action(observation, time_s=0.1)[6] == pytest.approx(-1.0)


def test_false_grasp_wrench_or_single_side_contact_cannot_enter_lift():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    observation["can_pos_robot_base"] += np.array((0.0, 0.10, 0.0), dtype=np.float32)
    observation["robot0_wrist_force"] = np.array((100.0, 0.0, 0.0), dtype=np.float32)
    controller.executive.phase = TaskPhase.GRASP
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is not TaskPhase.LIFT
    assert controller.executive._grasp_eef_can_transform is None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda observation: observation.__setitem__("robot0_gripper_state", np.zeros(4, dtype=np.float32)),
        lambda observation: observation["can_pos_robot_base"].__setitem__(1, 0.10),
        lambda observation: observation["can_pos_robot_base"].__setitem__(0, 0.20),
    ),
)
def test_public_grasp_negative_controls_reject_empty_single_side_and_distant_can(mutate):
    observation = _bilateral_grasp_observation()
    observation["robot0_wrist_force"] = np.array((120.0, 0.0, 0.0), dtype=np.float32)
    mutate(observation)
    executive = ShakeBenchOracleController("V0").executive
    assert not executive._public_grasp_held(observation)
    assert executive._grasp_confidence["wrench_used_for_hold_verdict"] is False


def test_prelift_rejects_a_can_that_stays_on_the_table():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    controller.executive.phase = TaskPhase.GRASP
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is TaskPhase.PRELIFT_VERIFY
    observation["robot0_eef_pos_robot_base"][2] = 0.012
    for time_s in (0.80, 0.85, 1.20):
        controller.action(observation, time_s=time_s)
    assert controller.executive.phase is not TaskPhase.LIFT


def test_prelift_requires_two_public_rigid_follow_samples_before_lift():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    controller.executive.phase = TaskPhase.GRASP
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is TaskPhase.PRELIFT_VERIFY
    for time_s, lift in ((0.80, 0.007), (0.85, 0.013)):
        observation["robot0_eef_pos_robot_base"][2] = lift
        observation["can_pos_robot_base"][2] = 0.085 + lift
        controller.action(observation, time_s=time_s)
    assert controller.executive.phase is TaskPhase.LIFT


def test_early_slip_enters_safe_recovery_before_one_can_radius():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.085)
    controller.executive._record_grasp_reference(observation)
    controller.executive.phase = TaskPhase.LIFT
    observation["can_pos_robot_base"][0] += 0.020
    controller.action(observation, time_s=0.10)
    controller.action(observation, time_s=0.15)
    assert controller.executive.phase is TaskPhase.RECOVERY_HOLD
    assert controller.executive.last_recovery_reason == "public_grasp_slip"


def test_descend_anchor_stops_chasing_a_contact_displaced_can_and_tracks_table_motion():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["can_pos_robot_base"][:] = (0.0, 0.0, 0.04)
    observation["robot0_eef_pos_robot_base"][:] = (0.0, 0.0, 0.125)
    controller.executive.phase = TaskPhase.DESCEND
    first = controller.action(observation, time_s=0.10)
    assert controller.executive.phase is TaskPhase.GRASP_CLOSE
    anchor = controller.executive._anchor_worktable_can_transform.copy()
    observation["can_pos_robot_base"][0] += 0.05
    second = controller.action(observation, time_s=0.15)
    np.testing.assert_allclose(second[:2], first[:2], atol=1e-6)
    observation["goal_frame_pos_robot_base"][0] += 0.10
    moved = controller.action(observation, time_s=0.20)
    assert controller.executive._anchor_worktable_can_transform is not None
    np.testing.assert_allclose(
        controller.executive._anchored_can_base(observation),
        np.array((0.10, 0.0, 0.04)),
        atol=1e-6,
    )
    assert np.all(np.isfinite(moved))
    np.testing.assert_allclose(controller.executive._anchor_worktable_can_transform, anchor, atol=1e-6)


def test_clearance_certificate_checks_fingertip_support_points_not_only_eef_origin():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["can_pos_robot_base"][:] = (0.0, 0.0, 0.04)
    observation["robot0_fingertip_pos_robot_base"][:,] = np.array(
        (0.0, -0.015, 0.085, 0.0, 0.015, 0.085), dtype=np.float32
    )
    low = controller.executive.public_tool_clearance_certificate(observation)
    assert not low["passed"]
    observation["robot0_fingertip_pos_robot_base"][:,] = np.array(
        (0.0, -0.015, 0.15, 0.0, 0.015, 0.15), dtype=np.float32
    )
    high = controller.executive.public_tool_clearance_certificate(observation)
    assert high["passed"]


def test_swept_clearance_certificate_rejects_a_low_midpath_pad_segment():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["can_pos_robot_base"][:] = (0.0, 0.0, 0.04)
    start = np.array((0.0, -0.015, 0.15, 0.0, 0.015, 0.15))
    end = np.array((0.0, -0.015, 0.085, 0.0, 0.015, 0.085))
    swept = controller.executive.public_swept_tool_clearance_certificate(observation, start, end)
    assert swept["samples"] == 11
    assert not swept["passed"]
    assert swept["minimum_clearance_m"] < swept["required_clearance_m"]


def test_worktable_bounds_use_can_envelope_and_survive_target_frame_rotation():
    context = WorktableTaskContext()
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = context.target_frame_origin_in_worktable_m
    observation["can_pos_robot_base"][:] = worktable_local_to_robot_base(
        observation, (0.29, 0.0, 0.0702970033), context
    )
    executive = ShakeBenchOracleController("V0", task_context=context).executive
    assert executive._public_recoverability(observation)["recoverable"]
    observation["can_pos_robot_base"][:] = worktable_local_to_robot_base(
        observation, (0.31, 0.0, 0.0702970033), context
    )
    assert not executive._public_recoverability(observation)["recoverable"]
    assert executive._public_recoverability(observation)["edge_unrecoverable"]
    rotated = dict(observation)
    rotated["goal_frame_quat_robot_base"] = np.array((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    rotated["goal_frame_pos_robot_base"] = np.array((-0.17, -0.10, 0.042))
    rotated["can_pos_robot_base"] = np.array((0.0, 0.29, 0.0702970033))
    local = robot_base_to_worktable_local(rotated, rotated["can_pos_robot_base"], context)
    assert local[0] == pytest.approx(0.29, abs=1e-6)


def test_recovery_holds_close_until_public_table_support_is_stable():
    controller = ShakeBenchOracleController("V0")
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["can_pos_robot_base"][:] = (0.0, 0.0, 0.0702970033)
    observation["robot0_eef_pos_robot_base"][:] = (0.04, 0.0, -0.014703)
    observation["robot0_fingertip_pos_robot_base"][:] = np.array(
        (0.04, -0.015, 0.078697, 0.04, 0.015, 0.078697), dtype=np.float32
    )
    controller.executive._record_grasp_reference(observation)
    controller.executive.phase = TaskPhase.LIFT
    observation["can_pos_robot_base"][0] = 0.020
    controller.action(observation, time_s=0.10)
    action = controller.action(observation, time_s=0.15)
    assert controller.executive.phase is TaskPhase.RECOVERY_HOLD
    assert action[6] == pytest.approx(1.0)
    controller.action(observation, time_s=0.45)
    controller.action(observation, time_s=0.75)
    assert controller.executive.phase is TaskPhase.WAIT_PUBLIC_SETTLE
    controller.action(observation, time_s=1.60)
    assert controller.executive.phase is TaskPhase.RECOVERY_OPEN


def test_verify_uses_public_target_geometry_for_placement_recovery():
    controller = ShakeBenchOracleController("V0")
    controller.executive.phase = TaskPhase.VERIFY
    controller.executive.phase_entered_s = 0.0
    observation = _public_observation()
    observation["can_pos_robot_base"] = np.array((0.5, 0.5, 0.04), dtype=np.float32)
    controller.action(observation, time_s=controller.profile.verify_s)
    assert controller.executive.phase is TaskPhase.FAILED
    assert controller.executive.failure_reason == "public_object_edge_unrecoverable"


def test_phase07_has_a_frozen_unselected_ten_state_dev_manifest():
    artifact = Path(__file__).parents[1] / "robosuite/models/assets/shakebench_states_dev.json"
    states = load_dev_states(artifact)
    assert [state["state_id"] for state in states] == [f"shakebench-dev-v0-{index:03d}" for index in range(10)]
    xy = np.asarray([state["can_xy_m"] for state in states])
    assert np.all(xy >= np.asarray((-0.12, -0.15)))
    assert np.all(xy <= np.asarray((-0.08, -0.11)))
