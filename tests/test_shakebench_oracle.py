"""Phase 07 public-seam tests for the shared State Oracle controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robosuite.scripts.shakebench_run_oracle import load_dev_states
from robosuite.utils.shakebench_metrics import SuccessSnapshot, VibrationSuccessEvaluator
from robosuite.utils.shakebench_oracle import (
    ORACLE_TASK_KEYS,
    OracleControllerProfile,
    ShakeBenchOracleController,
    ShakeBenchOracleError,
    TaskPhase,
    WorktableTaskContext,
    robot_base_to_worktable_local,
    target_local_to_robot_base,
    vibration_estimate_from_public_observation,
    worktable_local_to_robot_base,
)
from robosuite.utils.shakebench_providers import TABLE_IMU_POLICY_KEYS
from tests.shakebench_test_helpers import build_current_observation


def _public_observation(*, goal_quat=(0.0, 0.0, 0.0, 1.0)):
    values = build_current_observation(
        goal_frame_pos=(0.2, 0.3, 0.02),
        gripper_state=(0.0, 0.0, 0.0, 0.0),
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
    observation["object_pos_robot_base"] = np.array((-0.005, 0.0, 0.080), dtype=np.float32)
    return observation


def test_complete_public_goal_frame_distinguishes_rotated_targets_and_transforms_points():
    identity = _public_observation(goal_quat=(0.0, 0.0, 0.0, 1.0))
    rotated = _public_observation(goal_quat=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    local = np.array((0.04, 0.0, 0.01))
    np.testing.assert_allclose(target_local_to_robot_base(identity, local), (0.24, 0.3, 0.03), atol=1e-7)
    np.testing.assert_allclose(target_local_to_robot_base(rotated, local), (0.2, 0.34, 0.03), atol=1e-7)
    assert not np.array_equal(identity["goal_frame_quat_robot_base"], rotated["goal_frame_quat_robot_base"])
    observation = _public_observation()
    assert set(ORACLE_TASK_KEYS) <= set(observation)
    assert set(TABLE_IMU_POLICY_KEYS) <= set(observation)
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


def test_current_estimate_is_public_only_and_neutral_without_history():
    estimate = vibration_estimate_from_public_observation(_public_observation(), policy_time_s=0.2)
    assert estimate.source == "public_neutral"
    assert estimate.validity is True
    assert estimate.measurement_timestamp_s == pytest.approx(0.2)
    assert estimate.latency_s == pytest.approx(0.0)
    np.testing.assert_array_equal(estimate.current_relative_acceleration_robot_base, np.zeros(6))
    with pytest.raises(ShakeBenchOracleError, match="privileged"):
        vibration_estimate_from_public_observation({**_public_observation(), "privileged_contacts": {}})
    with pytest.raises(ShakeBenchOracleError, match="missing observation payload"):
        vibration_estimate_from_public_observation({"goal_frame_pos_robot_base": np.zeros(3)})


def test_shared_profile_and_neutral_estimate_produce_identical_actions():
    profile = OracleControllerProfile()
    v0 = ShakeBenchOracleController(profile)
    v0_again = ShakeBenchOracleController(profile)
    action = v0.action(_public_observation(), time_s=0.0)
    np.testing.assert_array_equal(action, v0_again.action(_public_observation(), time_s=0.0))
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)
    assert v0.profile_sha256 == v0_again.profile_sha256


def test_safe_approach_transitions_directly_to_lateral_alignment_without_vertical_reversal():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    approach = observation["object_pos_robot_base"] + np.array(
        (0.0, 0.0, controller.profile.approach_height_m), dtype=np.float32
    )
    observation["robot0_eef_pos_robot_base"] = approach
    observation["robot0_fingertip_pos_robot_base"] = np.array(
        (
            approach[0],
            approach[1] - 0.015,
            approach[2],
            approach[0],
            approach[1] + 0.015,
            approach[2],
        ),
        dtype=np.float32,
    )
    controller.executive.phase = TaskPhase.APPROACH

    action = controller.action(observation, time_s=0.1)

    assert controller.executive.phase is TaskPhase.LATERAL_ALIGN_ABOVE_CAN
    assert abs(float(action[2])) < 1.0e-6
    assert controller.executive.diagnostics()["swept_clearance_certificate"]["passed"]


def test_time_reversal_resets_the_public_controller_state():
    controller = ShakeBenchOracleController()
    observation = _public_observation()
    first = controller.action(observation, time_s=0.1)
    controller.action(observation, time_s=0.05)
    assert controller.last_trace["estimate"]["measurement_timestamp_s"] == 0.05
    assert controller.last_trace["measurement_time_s"] == 0.05
    np.testing.assert_array_equal(controller.action(observation, time_s=0.1), first)


def test_worktable_imu_payload_stays_public_provenance_and_never_moves_the_action():
    controller = ShakeBenchOracleController()
    observation = _public_observation()
    quiet = controller.action(observation, time_s=0.0)
    assert controller.last_trace["provider_payload_keys"] == sorted(TABLE_IMU_POLICY_KEYS)
    noisy = {key: value.copy() for key, value in observation.items()}
    noisy["table_imu_window"] = np.linspace(-40.0, 40.0, 60, dtype=np.float32).reshape(10, 6)
    comparison = ShakeBenchOracleController()
    np.testing.assert_array_equal(quiet, comparison.action(noisy, time_s=0.0))


def test_public_gripper_closure_triggers_bounded_recovery_not_false_transport():
    controller = ShakeBenchOracleController()
    controller.executive.phase = TaskPhase.LIFT
    observation = _public_observation()
    observation["robot0_gripper_state"] = np.array((0.03, -0.03, 0.0, 0.0), dtype=np.float32)
    observation["object_pos_robot_base"] = np.array((0.3, -0.1, 0.04), dtype=np.float32)
    controller.action(observation, time_s=1.0)
    controller.action(observation, time_s=1.05)
    assert controller.executive.phase is TaskPhase.RECOVERY_HOLD
    assert controller.executive.recovery_transition == "recovery_hold_close"


def test_phase_to_gripper_wiring_uses_the_public_panda_contract():
    observation = _bilateral_grasp_observation()
    for phase in (TaskPhase.SETTLE, TaskPhase.APPROACH, TaskPhase.DESCEND, TaskPhase.RELEASE, TaskPhase.VERIFY):
        controller = ShakeBenchOracleController()
        controller.executive.phase = phase
        assert controller.action(observation, time_s=0.1)[6] == pytest.approx(-1.0)
    for phase in (TaskPhase.GRASP, TaskPhase.PRELIFT_VERIFY, TaskPhase.LIFT, TaskPhase.TRANSPORT, TaskPhase.PLACE):
        controller = ShakeBenchOracleController()
        controller.executive.phase = phase
        if phase is TaskPhase.PRELIFT_VERIFY:
            controller.executive._prelift_start_eef_position = observation["robot0_eef_pos_robot_base"].copy()
            controller.executive._prelift_start_can_height_m = 0.085
        assert controller.action(observation, time_s=0.1)[6] == pytest.approx(1.0)
    controller = ShakeBenchOracleController()
    controller.executive.phase = TaskPhase.FAILED
    assert controller.action(observation, time_s=0.1)[6] == pytest.approx(-1.0)


def test_false_grasp_wrench_or_single_side_contact_cannot_enter_lift():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    observation["object_pos_robot_base"] += np.array((0.0, 0.10, 0.0), dtype=np.float32)
    observation["robot0_wrist_force"] = np.array((100.0, 0.0, 0.0), dtype=np.float32)
    controller.executive.phase = TaskPhase.GRASP
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is not TaskPhase.LIFT
    assert controller.executive._grasp_eef_can_transform is None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda observation: observation.__setitem__("robot0_gripper_state", np.zeros(4, dtype=np.float32)),
        lambda observation: observation["object_pos_robot_base"].__setitem__(1, 0.10),
        lambda observation: observation["object_pos_robot_base"].__setitem__(0, 0.20),
    ),
)
def test_public_grasp_negative_controls_reject_empty_single_side_and_distant_can(mutate):
    observation = _bilateral_grasp_observation()
    observation["robot0_wrist_force"] = np.array((120.0, 0.0, 0.0), dtype=np.float32)
    mutate(observation)
    executive = ShakeBenchOracleController().executive
    assert not executive._public_grasp_held(observation)
    assert executive._grasp_confidence["wrench_used_for_hold_verdict"] is False


def test_prelift_rejects_a_can_that_stays_on_the_table():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    controller.executive.phase = TaskPhase.GRASP
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is TaskPhase.PRELIFT_VERIFY
    observation["robot0_eef_pos_robot_base"][2] = 0.012
    for time_s in (0.80, 0.85, 1.20):
        controller.action(observation, time_s=time_s)
    assert controller.executive.phase is not TaskPhase.LIFT


def test_prelift_requires_two_public_rigid_follow_samples_before_lift():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    controller.executive.phase = TaskPhase.GRASP
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is TaskPhase.PRELIFT_VERIFY
    for time_s, lift in ((0.80, 0.007), (0.85, 0.013)):
        observation["robot0_eef_pos_robot_base"][2] = lift
        observation["object_pos_robot_base"][2] = 0.085 + lift
        controller.action(observation, time_s=time_s)
    assert controller.executive.phase is TaskPhase.LIFT


def test_early_slip_enters_safe_recovery_before_one_can_radius():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.085)
    controller.executive._record_grasp_reference(observation)
    controller.executive.phase = TaskPhase.LIFT
    observation["object_pos_robot_base"][0] += 0.020
    controller.action(observation, time_s=0.10)
    controller.action(observation, time_s=0.15)
    assert controller.executive.phase is TaskPhase.RECOVERY_HOLD
    assert controller.executive.last_recovery_reason == "public_grasp_slip"


def test_descend_anchor_stops_chasing_a_contact_displaced_can_and_tracks_table_motion():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["object_pos_robot_base"][:] = (0.0, 0.0, 0.04)
    observation["robot0_eef_pos_robot_base"][:] = (0.0, 0.0, 0.125)
    controller.executive.phase = TaskPhase.DESCEND
    first = controller.action(observation, time_s=0.10)
    assert controller.executive.phase is TaskPhase.GRASP_CLOSE
    anchor = controller.executive._anchor_worktable_can_transform.copy()
    observation["object_pos_robot_base"][0] += 0.05
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
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["object_pos_robot_base"][:] = (0.0, 0.0, 0.04)
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
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["object_pos_robot_base"][:] = (0.0, 0.0, 0.04)
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
    observation["object_pos_robot_base"][:] = worktable_local_to_robot_base(
        observation, (0.29, 0.0, 0.0702970033), context
    )
    executive = ShakeBenchOracleController(task_context=context).executive
    assert executive._public_recoverability(observation)["recoverable"]
    observation["object_pos_robot_base"][:] = worktable_local_to_robot_base(
        observation, (0.31, 0.0, 0.0702970033), context
    )
    assert not executive._public_recoverability(observation)["recoverable"]
    assert executive._public_recoverability(observation)["edge_unrecoverable"]
    rotated = dict(observation)
    rotated["goal_frame_quat_robot_base"] = np.array((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    rotated["goal_frame_pos_robot_base"] = np.array((-0.17, -0.10, 0.042))
    rotated["object_pos_robot_base"] = np.array((0.0, 0.29, 0.0702970033))
    local = robot_base_to_worktable_local(rotated, rotated["object_pos_robot_base"], context)
    assert local[0] == pytest.approx(0.29, abs=1e-6)


def test_recovery_holds_close_until_public_table_support_is_stable():
    controller = ShakeBenchOracleController()
    observation = _bilateral_grasp_observation()
    observation["goal_frame_pos_robot_base"][:] = (0.0, 0.0, 0.0)
    observation["object_pos_robot_base"][:] = (0.0, 0.0, 0.0702970033)
    observation["robot0_eef_pos_robot_base"][:] = (0.04, 0.0, -0.014703)
    observation["robot0_fingertip_pos_robot_base"][:] = np.array(
        (0.04, -0.015, 0.078697, 0.04, 0.015, 0.078697), dtype=np.float32
    )
    controller.executive._record_grasp_reference(observation)
    controller.executive.phase = TaskPhase.LIFT
    observation["object_pos_robot_base"][0] = 0.020
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
    controller = ShakeBenchOracleController()
    controller.executive.phase = TaskPhase.VERIFY
    controller.executive.phase_entered_s = 0.0
    observation = _public_observation()
    observation["object_pos_robot_base"] = np.array((0.5, 0.5, 0.04), dtype=np.float32)
    controller.action(observation, time_s=controller.profile.verify_s)
    assert controller.executive.phase is TaskPhase.FAILED
    assert controller.executive.failure_reason == "policy_abort"
    assert controller.executive.abort_reason == "edge_risk"


def test_unknown_recovery_reason_fails_closed_instead_of_mislabeling_the_event():
    with pytest.raises(ShakeBenchOracleError, match="unregistered recovery reason"):
        ShakeBenchOracleController().executive._event_type("typo")


def test_phase07_has_a_frozen_unselected_ten_state_dev_manifest():
    artifact = Path(__file__).parents[1] / "robosuite/models/assets/shakebench_states_dev.json"
    states = load_dev_states(artifact)
    assert [state["state_id"] for state in states] == [f"shakebench-dev-v0-{index:03d}" for index in range(10)]
    xy = np.asarray([state["object_xy_m"] for state in states])
    assert np.all(xy >= np.asarray((-0.12, -0.15)))
    assert np.all(xy <= np.asarray((-0.08, -0.11)))
