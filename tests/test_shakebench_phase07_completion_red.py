"""Red feedback for the Phase 07 completion remediation review findings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.scripts.shakebench_run_oracle import (
    _dev_state_anchor,
    _digest,
    _json_ready,
    load_dev_states,
    run_episode,
    verify_determinism_manifest,
    verify_run_artifact,
)
from robosuite.utils.shakebench_oracle import (
    OracleControllerProfile,
    ShakeBenchOracleController,
    SharedVibrationControlLaw,
    TaskPhase,
    _quat_xyzw_to_matrix,
    gravity_compensated_imu_window,
    vibration_estimate_from_public_observation,
)
from tests.shakebench_test_helpers import build_tier_observation


def _observation(tier: str) -> dict[str, np.ndarray]:
    return build_tier_observation(tier, goal_frame_pos=(0.0, 0.0, 0.0))


def _make_env():
    return robosuite.make(
        "VibrationPickPlaceCan",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        observation_tier="V0",
        imu_mode="ideal_smoke",
        horizon=1,
        seed=5,
    )


def test_red_target_quaternion_is_already_xyzw_before_environment_conversion():
    np.testing.assert_allclose(T.mat2quat(np.eye(3)), (0.0, 0.0, 0.0, 1.0), atol=1e-12)
    env = _make_env()
    try:
        observation = env.reset()
        np.testing.assert_allclose(observation["goal_frame_quat_robot_base"], (0.0, 0.0, 0.0, 1.0), atol=1e-7)
    finally:
        env.close()


def test_public_quaternion_reconstructs_known_robot_base_target_rotations():
    expected = {
        "x": np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))),
        "y": np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
        "z": np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
    }
    quaternions = {
        "x": (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
        "y": (0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)),
        "z": (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
    }
    for axis in expected:
        np.testing.assert_allclose(_quat_xyzw_to_matrix(quaternions[axis]), expected[axis], atol=1e-12)


def test_red_zero_v2_acceleration_must_not_create_vertical_correction():
    estimate = vibration_estimate_from_public_observation("V2", _observation("V2"))
    output = SharedVibrationControlLaw(OracleControllerProfile()).apply(np.zeros(6), estimate)
    np.testing.assert_allclose(output.compensated_delta_base, np.zeros(6), atol=1e-12)


def test_tilted_stationary_imu_is_not_horizontal_motion():
    rotation_control_from_sensor = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    gravity_sensor = rotation_control_from_sensor.T.dot(np.array((0.0, 0.0, 9.81)))
    window = np.tile(np.r_[gravity_sensor, (0.0, 0.0, 0.0)], (10, 1))
    motion, angular = gravity_compensated_imu_window(
        window,
        0.005,
        initial_rotation_control_from_sensor=rotation_control_from_sensor,
    )
    np.testing.assert_allclose(motion, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(angular, np.zeros(3), atol=1e-12)


def _tilted_window(angle_rad: float, *, stationary: bool = False) -> np.ndarray:
    """Construct a public, frame-aware IMU window from an independent invariant."""

    gravity_control = np.array((0.0, 0.0, 9.80665))
    dt_s = 0.005
    gyro = np.zeros(3) if stationary else np.array((angle_rad / (10.0 * dt_s), 0.0, 0.0))
    rotation = np.eye(3)
    if stationary:
        rotation = _rotation_from_rotvec_for_test(np.array((angle_rad, 0.0, 0.0)))
    samples = []
    for _ in range(10):
        samples.append(np.r_[rotation.T.dot(gravity_control), gyro])
        if not stationary:
            rotation = rotation.dot(_rotation_from_rotvec_for_test(gyro * dt_s))
    return np.asarray(samples, dtype=np.float32)


def _rotation_from_rotvec_for_test(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew.dot(skew)


def test_red_v1_attitude_and_latency_persist_across_policy_windows():
    first = _observation("V1")
    second = _observation("V1")
    first["deck_imu_window"] = _tilted_window(np.pi / 3.0)
    second["deck_imu_window"] = _tilted_window(np.pi / 3.0, stationary=True)
    controller = ShakeBenchOracleController("V1")
    controller.action(first, time_s=0.05)
    first_estimate = controller.last_trace["estimate"]
    controller.action(second, time_s=0.10)
    second_estimate = controller.last_trace["estimate"]
    np.testing.assert_allclose(second_estimate["current_linear_accel_m_s2"], np.zeros(3), atol=1e-3)
    assert second_estimate["timestamp_s"] > first_estimate["timestamp_s"]
    assert second_estimate["latency_s"] == pytest.approx(0.00987, abs=5e-5)
    assert np.allclose(
        controller.v1_estimator.attitude_control_from_sensor,
        _rotation_from_rotvec_for_test(np.array((np.pi / 3.0, 0.0, 0.0))),
        atol=3e-2,
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def test_red_grasp_slip_uses_complete_can_in_eef_se3_transform():
    controller = ShakeBenchOracleController("V0")
    observation = _observation("V0")
    observation["robot0_eef_pos_robot_base"] = np.zeros(3, dtype=np.float32)
    observation["can_pos_robot_base"] = np.array((-0.005, 0.0, 0.080), dtype=np.float32)
    observation["goal_frame_pos_robot_base"] = np.array((0.0, 0.0, 0.085), dtype=np.float32)
    observation["robot0_fingertip_pos_robot_base"] = np.array((0.0, -0.015, 0.0934, 0.0, 0.015, 0.0934))
    observation["robot0_gripper_state"] = np.array((0.015, -0.015, 0.0, 0.0), dtype=np.float32)
    controller.executive.phase = TaskPhase.GRASP
    controller.executive.phase_entered_s = 0.0
    controller.action(observation, time_s=controller.profile.grasp_s)
    reference = controller.executive._grasp_eef_can_transform.copy()

    eef_rotation = _rotation_from_rotvec_for_test(np.array((0.0, 0.0, np.pi / 2.0)))
    eef_position = np.array((0.2, 0.3, 0.4))
    eef = _make_transform(eef_rotation, eef_position)
    can = eef.dot(reference)
    observation["robot0_eef_pos_robot_base"] = eef[:3, 3]
    observation["robot0_eef_quat_robot_base"] = np.array((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    observation["can_pos_robot_base"] = can[:3, 3]
    observation["can_quat_robot_base"] = np.array((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    translation_slip, rotation_slip = controller.executive._public_grasp_slip(observation)
    assert translation_slip == pytest.approx(0.0, abs=1e-6)
    assert rotation_slip == pytest.approx(0.0, abs=1e-6)

    relative_slip = _make_transform(_rotation_from_rotvec_for_test(np.array((0.0, np.pi / 4.0, 0.0))), (0.03, 0.0, 0.0))
    slipped = eef.dot(reference).dot(relative_slip)
    observation["can_pos_robot_base"] = slipped[:3, 3]
    observation["can_quat_robot_base"] = T.mat2quat(slipped[:3, :3])
    translation_slip, rotation_slip = controller.executive._public_grasp_slip(observation)
    assert translation_slip > 0.02
    assert rotation_slip == pytest.approx(np.pi / 4.0, abs=1e-5)


def test_red_verify_checks_can_envelope_rebound_and_stability_before_completion():
    controller = ShakeBenchOracleController("V0")
    controller.executive.phase = TaskPhase.VERIFY
    controller.executive.phase_entered_s = 0.0
    observation = _observation("V0")
    observation["can_pos_robot_base"] = np.array((0.075, 0.0, 0.04), dtype=np.float32)
    observation["robot0_gripper_state"] = np.array((0.03, -0.03, 0.0, 0.0), dtype=np.float32)
    controller.action(observation, time_s=controller.profile.verify_s)
    assert controller.executive.phase is not TaskPhase.COMPLETE
    assert controller.executive.last_recovery_reason in {"public_placement_rebound", "public_placement_loss"}

    stable_controller = ShakeBenchOracleController("V0")
    stable_controller.executive.phase = TaskPhase.VERIFY
    stable_controller.executive.phase_entered_s = 0.0
    stable = _observation("V0")
    stable["can_pos_robot_base"] = np.array((0.0, 0.0, 0.04), dtype=np.float32)
    stable["robot0_gripper_state"] = np.array((0.03, -0.03, 0.0, 0.0), dtype=np.float32)
    stable["robot0_fingertip_pos_robot_base"] = np.array((0.4, 0.4, 0.4, 0.4, 0.4, 0.4), dtype=np.float32)
    stable_controller.action(stable, time_s=0.0)
    stable_controller.action(stable, time_s=stable_controller.profile.verify_s)
    assert stable_controller.executive.phase is TaskPhase.VERIFY


def test_public_container_risk_is_diagnostic_and_recovery_budget_exhaustion_is_bounded():
    controller = ShakeBenchOracleController("V0")
    controller.executive.phase = TaskPhase.TRANSPORT
    observation = _observation("V0")
    observation["can_pos_robot_base"] = np.array((0.0, 0.0, 0.04), dtype=np.float32)
    observation["robot0_fingertip_pos_robot_base"] = np.array((0.0, 0.0, 0.04, 0.25, 0.0, 0.04))
    controller.action(observation, time_s=0.1)
    assert controller.executive.phase is TaskPhase.TRANSPORT
    assert controller.executive.last_recovery_reason == "public_container_risk"
    assert controller.executive.recovery_count == 0

    exhausted = ShakeBenchOracleController("V0", OracleControllerProfile(recovery_budget=1))
    exhausted.executive.phase = TaskPhase.VERIFY
    exhausted.executive.phase_entered_s = 0.0
    bad = _observation("V0")
    bad["can_pos_robot_base"] = np.array((0.2, 0.2, 0.04), dtype=np.float32)
    bad["robot0_gripper_state"] = np.array((0.03, -0.03, 0.0, 0.0), dtype=np.float32)
    exhausted.action(bad, time_s=exhausted.profile.verify_s)
    exhausted.executive.phase = TaskPhase.VERIFY
    exhausted.executive.phase_entered_s = 0.0
    exhausted.action(bad, time_s=exhausted.profile.verify_s + 0.01)
    assert exhausted.executive.phase is TaskPhase.FAILED
    assert exhausted.executive.failure_reason == "public_object_edge_unrecoverable"


def _reseal(payload: dict) -> None:
    for episode in payload["episodes"]:
        episode["trace_sha256"] = _digest(episode["trace"])
    payload["payload_sha256"] = _digest({key: value for key, value in payload.items() if key != "payload_sha256"})


def _valid_run_payload(episode: dict, *, state_ids: list[str]) -> dict:
    profile = OracleControllerProfile()
    payload = {
        "schema_id": "shakebench.phase07.oracle_run",
        "schema_version": 3,
        "tier": "V0",
        "gamma_commanded": 0.0,
        "controller_profile": profile.to_dict(),
        "evaluator_post_complete_settle_s": profile.completion_evaluator_settle_s,
        "dev_state_anchor": _dev_state_anchor("robosuite/models/assets/shakebench_states_dev.json"),
        "physics_authority": {
            "profile_id": "shakebench.official.physics.v2",
            "profile_sha256": "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c",
        },
        "episodes": [episode],
    }
    payload["run_id"] = _digest(
        {
            "tier": "V0",
            "gamma_commanded": 0.0,
            "controller_profile": profile.to_dict(),
            "state_ids": state_ids,
        }
    )
    _reseal(payload)
    return payload


@pytest.mark.parametrize(
    "mutation",
    (
        "delete_policy_input",
        "delete_applied_force",
        "actuator_name",
        "actuator_id",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "normalized_action",
        "decoded_action",
        "clipped_action",
        "delete_middle_row",
        "swap_middle_rows",
        "controller_profile",
        "state_hash",
        "physics_hash",
        "termination",
        "tier_key",
    ),
)
def test_red_resealed_semantic_mutations_are_rejected(tmp_path, mutation):
    state = load_dev_states("robosuite/models/assets/shakebench_states_dev.json")[0]
    horizon = 3 if mutation in {"delete_middle_row", "swap_middle_rows"} else 1
    episode = run_episode(
        state, tier="V0", gamma_commanded=0.0, profile=OracleControllerProfile(), horizon_steps=horizon
    )
    payload = _valid_run_payload(episode, state_ids=[state["state_id"]])
    if mutation == "delete_policy_input":
        del episode["trace"][0]["policy_input"]
    elif mutation == "delete_applied_force":
        del episode["trace"][0]["applied_actuator_force"]
    elif mutation == "actuator_name":
        episode["actuators"][0]["name"] = "wrong"
    elif mutation == "actuator_id":
        episode["actuators"][0]["id"] = 99
    elif mutation == "actuator_ctrlrange":
        episode["actuators"][0]["ctrlrange"][0] = -1.0
    elif mutation == "actuator_forcerange":
        episode["actuators"][0]["forcerange"][1] = 1.0
    elif mutation in {"normalized_action", "decoded_action", "clipped_action"}:
        episode["trace"][0][mutation][0] += 0.1
    elif mutation == "delete_middle_row":
        episode["trace"].pop(1)
    elif mutation == "swap_middle_rows":
        episode["trace"][0], episode["trace"][1] = episode["trace"][1], episode["trace"][0]
    elif mutation == "controller_profile":
        payload["controller_profile"]["position_action_range_m"] *= 2.0
    elif mutation == "state_hash":
        episode["state_sha256"] = "0" * 64
    elif mutation == "physics_hash":
        episode["physics_profile"]["profile_sha256"] = "0" * 64
    elif mutation == "termination":
        episode["termination_category"] = "environment_success"
        episode["failure_reason"] = None
    elif mutation == "tier_key":
        episode["trace"][0]["policy_input"]["forbidden_key"] = 1.0
    _reseal(payload)
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(_json_ready(payload)), encoding="utf-8")
    assert not verify_run_artifact(path)["passed"]


def test_red_true_three_process_manifest_rejects_reused_identity_and_path():
    path = Path("out/phase07_invalidated/remediation_raw/determinism_manifest_final.json")
    if not path.is_file():
        pytest.skip("historical invalidated raw evidence requires the explicit evidence archive")
    result = verify_determinism_manifest(path)
    assert not result["passed"]
    assert any("process" in error or "path" in error for error in result["errors"])


def test_red_v3_future_program_changes_action_with_same_current_state():
    first = _observation("V3")
    second = _observation("V3")
    second["line_accel_amplitude"][0, 0] = 5.0
    first_controller = ShakeBenchOracleController("V3")
    second_controller = ShakeBenchOracleController("V3")
    first_controller.executive.phase = TaskPhase.COMPLETE
    second_controller.executive.phase = TaskPhase.COMPLETE
    first_action = first_controller.action(first, time_s=0.5)
    second_action = second_controller.action(second, time_s=0.5)
    assert not np.allclose(first_action, second_action)


def test_six_axis_target_rotation_drives_positive_and_negative_osc_channels():
    positive = _observation("V0")
    negative = _observation("V0")
    controller_positive = ShakeBenchOracleController("V0")
    controller_negative = ShakeBenchOracleController("V0")
    controller_positive.action(positive, time_s=0.0)
    controller_negative.action(negative, time_s=0.0)
    controller_positive.executive.phase = TaskPhase.TRANSPORT
    controller_negative.executive.phase = TaskPhase.TRANSPORT
    positive["goal_frame_quat_robot_base"] = np.array((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    negative["goal_frame_quat_robot_base"] = np.array((0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    assert controller_positive.action(positive, time_s=0.1)[5] > 0.0
    assert controller_negative.action(negative, time_s=0.1)[5] < 0.0


def test_public_in_target_grasp_can_release_when_grip_site_waypoint_is_constrained():
    controller = ShakeBenchOracleController("V0")
    observation = _observation("V0")
    observation["can_pos_robot_base"] = np.array((0.0, 0.0, 0.04), dtype=np.float32)
    observation["robot0_eef_pos_robot_base"] = np.array((0.04, 0.0, 0.105), dtype=np.float32)
    observation["robot0_fingertip_pos_robot_base"] = np.array(
        (0.04, -0.015, 0.105, 0.04, 0.015, 0.105), dtype=np.float32
    )
    observation["robot0_gripper_state"] = np.array((0.015, -0.015, 0.0, 0.0), dtype=np.float32)
    controller.executive.phase = TaskPhase.PLACE
    controller.executive._record_grasp_reference(observation)
    controller.executive._placement_start_eef_position = observation["robot0_eef_pos_robot_base"].copy()
    controller.action(observation, time_s=0.1)
    assert controller.executive.phase is TaskPhase.RELEASE


def test_public_workspace_escape_stops_controller_without_simulator_truth():
    controller = ShakeBenchOracleController("V0")
    observation = _observation("V0")
    observation["can_pos_robot_base"] = np.array((2.0, 0.0, 0.0), dtype=np.float32)
    np.testing.assert_allclose(
        controller.action(observation, time_s=0.1), np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    )
    assert controller.executive.phase is TaskPhase.FAILED
    assert controller.executive.failure_reason == "public_object_out_of_workspace"


def test_public_fingertip_geometry_and_wrench_reference_drives_slip_recovery():
    controller = ShakeBenchOracleController("V0", OracleControllerProfile(grasp_slip_tolerance_m=0.05))
    observation = _observation("V0")
    observation["robot0_eef_pos_robot_base"] = np.zeros(3, dtype=np.float32)
    observation["can_pos_robot_base"] = np.array((-0.005, 0.0, 0.080), dtype=np.float32)
    observation["goal_frame_pos_robot_base"] = np.array((0.0, 0.0, 0.085), dtype=np.float32)
    observation["robot0_fingertip_pos_robot_base"] = np.array(
        (0.0, -0.015, 0.0934, 0.0, 0.015, 0.0934), dtype=np.float32
    )
    observation["robot0_gripper_state"] = np.array((0.015, -0.015, 0.0, 0.0), dtype=np.float32)
    controller.executive.phase = TaskPhase.GRASP
    controller.executive.phase_entered_s = 0.0
    controller.action(observation, time_s=controller.profile.grasp_s)
    assert controller.executive.phase is TaskPhase.PRELIFT_VERIFY
    controller.executive.phase = TaskPhase.LIFT
    observation["can_pos_robot_base"] = np.array((0.06, 0.0, 0.085), dtype=np.float32)
    observation["robot0_fingertip_pos_robot_base"] += np.array((0.1, 0.0, 0.0, 0.1, 0.0, 0.0), dtype=np.float32)
    controller.action(observation, time_s=controller.profile.grasp_s + 0.1)
    controller.action(observation, time_s=controller.profile.grasp_s + 0.15)
    assert controller.executive.phase in {TaskPhase.RECOVERY_HOLD, TaskPhase.RECOVERY_OPEN}
    assert controller.executive.last_recovery_reason == "public_grasp_slip"


def test_red_horizon_failure_has_a_nonempty_reason():
    state = load_dev_states("robosuite/models/assets/shakebench_states_dev.json")[0]
    episode = run_episode(state, tier="V0", gamma_commanded=0.0, profile=OracleControllerProfile(), horizon_steps=1)
    assert episode["success"] is False
    assert episode["failure_reason"] == "horizon_exhausted"


def test_run_artifact_verifier_rejects_trace_mutation(tmp_path):
    state = load_dev_states("robosuite/models/assets/shakebench_states_dev.json")[0]
    episode = run_episode(state, tier="V0", gamma_commanded=0.0, profile=OracleControllerProfile(), horizon_steps=1)
    payload = _valid_run_payload(episode, state_ids=[state["state_id"]])
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_json_ready(payload)), encoding="utf-8")
    assert verify_run_artifact(path)["passed"]
    payload["episodes"][0]["trace"][0]["normalized_action"][0] = 123.0
    path.write_text(json.dumps(_json_ready(payload)), encoding="utf-8")
    assert not verify_run_artifact(path)["passed"]


def test_lightweight_success_snapshot_matches_full_metrics_update():
    env = _make_env()
    try:
        env.reset()
        lightweight = env.metrics.success_snapshot(env.sim).to_dict()
        full = env.metrics.update(env.sim).success_snapshot.to_dict()
        for key in (
            "containment",
            "target_bottom_contact_present",
            "target_bottom_support_force_N",
            "lower_support_z_m",
            "finger_can_contact_present",
            "relative_linear_speed_m_s",
            "relative_angular_speed_rad_s",
            "illegal_penetration_m",
        ):
            if isinstance(lightweight[key], np.ndarray):
                np.testing.assert_allclose(lightweight[key], full[key], atol=1e-12)
            else:
                assert lightweight[key] == full[key]
    finally:
        env.close()
