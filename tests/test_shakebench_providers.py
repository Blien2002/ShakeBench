"""Tests for the Phase 05 V0--V3 policy provider ladder."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.shakebench_deck import DeckCommand
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.shakebench_providers import (
    COMMON_STATE_KEYS,
    TIER_ADDED_KEYS,
    TIER_POLICY_KEYS,
    RigidBodyState,
    ShakeBenchProviderError,
    V3Provider,
    make_vibration_provider,
    policy_keys_for_tier,
    reconstruct_authored_motion,
    relative_pose_twist_acceleration,
)
from robosuite.utils.shakebench_sensors import (
    GRAVITY_WORLD_M_S2,
    clean_imu_measurement_from_sim,
    specific_force_from_rigid_body_motion,
)


def _make_env(tier, **kwargs):
    options = {
        "robots": "Panda",
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": False,
        "control_freq": 20,
        "model_timestep": 0.0002,
        "horizon": 2,
        "seed": 17,
        "physics_profile": "probe",
        "observation_tier": tier,
        "imu_mode": "ideal_smoke",
    }
    options.update(kwargs)
    return robosuite.make("VibrationPickPlaceCan", **options)


def test_tier_key_sets_are_exactly_cumulative():
    previous = set(COMMON_STATE_KEYS)
    for tier in ("V0", "V1", "V2", "V3"):
        expected = set(COMMON_STATE_KEYS) | set(TIER_POLICY_KEYS[tier])
        assert set(policy_keys_for_tier(tier, include_common=True)) == expected
        assert set(TIER_POLICY_KEYS[tier]) == set(TIER_POLICY_KEYS["V0"]) | set(
            key
            for level in ("V1", "V2", "V3")[: ("V0", "V1", "V2", "V3").index(tier)]
            for key in TIER_ADDED_KEYS[level]
        )
        assert previous <= expected
        previous = expected


def test_real_environment_observations_add_only_the_declared_tier_fields():
    observations_by_tier = {}
    for tier in ("V0", "V1", "V2", "V3"):
        env = _make_env(tier)
        try:
            observations = env.reset()
            observations_by_tier[tier] = set(observations)
            expected_keys = set(COMMON_STATE_KEYS) | set(TIER_POLICY_KEYS[tier])
            assert set(observations) == expected_keys
            assert set(env.policy_observation_keys) == expected_keys
            assert not any(key.startswith("privileged_") for key in observations)
            assert "deck_driver" not in observations
            assert "table_pose" not in observations
            assert "qpos" not in observations
            assert observations["robot0_eef_pos_robot_base"].shape == (3,)
            assert observations["robot0_eef_quat_robot_base"].shape == (4,)
            assert observations["goal_frame_pos_robot_base"].shape == (3,)
            assert observations["goal_frame_quat_robot_base"].shape == (4,)
            assert observations["goal_inner_half_extents_target"].shape == (2,)
            assert observations["goal_z_bounds_target"].shape == (2,)
            assert observations["goal_orientation_mask"].dtype == np.bool_
            if tier != "V0":
                assert env.vibration_provider.imu_body_name == "robot0_base"
                assert env._imu_mount_audit["parent_body_name"] == "deck"
                np.testing.assert_allclose(
                    env.vibration_provider.imu.window_acquisition_timestamps_s,
                    np.arange(-0.050, 0.0, 0.005),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(env.vibration_provider.imu.pending_delivery_acquisition_timestamps_s, [0.0])
                assert observations["deck_imu_window"].shape == (10, 6)
                assert observations["deck_imu_window"].dtype == np.float32
                assert observations["deck_imu_dt_s"].dtype == np.float32
            if tier in {"V2", "V3"}:
                for key in TIER_ADDED_KEYS["V2"]:
                    width = 7 if "pose" in key else 6
                    assert observations[key].shape == (width,)
                    assert observations[key].dtype == np.float32
            if tier == "V3":
                assert observations["line_accel_amplitude"].shape == (6, 12)
                assert observations["line_omega_rad_s"].shape == (6, 12)
                assert observations["line_phase_at_episode_zero"].shape == (6, 12)
                assert observations["line_mask"].shape == (6, 12)
                assert observations["line_mask"].dtype == np.bool_
                assert observations["ramp_type"].item() == "quintic_smoothstep"
                assert observations["program_frame"].item() == "deck"
            stepped, _, _, _ = env.step(np.zeros(env.action_dim))
            assert set(stepped) == set(observations)
            if tier != "V0":
                trace = env.vibration_provider.imu.trace
                assert trace["acquisition_time_s"].shape == (10,)
                np.testing.assert_allclose(trace["acquisition_time_s"], np.arange(0.005, 0.055, 0.005))
                np.testing.assert_allclose(trace["delivered_acquisition_time_s"], np.arange(0.0, 0.050, 0.005))
                np.testing.assert_allclose(
                    env.vibration_provider.imu.pending_delivery_acquisition_timestamps_s, [0.050]
                )
                env.step(np.zeros(env.action_dim))
                assert env.vibration_provider.imu.trace["acquisition_time_s"].shape == (20,)
                np.testing.assert_allclose(
                    env.vibration_provider.imu.trace["acquisition_time_s"][-10:],
                    np.arange(0.055, 0.105, 0.005),
                )
            if tier == "V3":
                assert stepped["episode_time_s"] == pytest.approx(0.05, abs=1e-6)
        finally:
            env.close()

    assert observations_by_tier["V0"] < observations_by_tier["V1"]
    assert observations_by_tier["V1"] < observations_by_tier["V2"]
    assert observations_by_tier["V2"] < observations_by_tier["V3"]
    assert observations_by_tier["V1"] - observations_by_tier["V0"] == set(TIER_ADDED_KEYS["V1"])
    assert observations_by_tier["V2"] - observations_by_tier["V1"] == set(TIER_ADDED_KEYS["V2"])
    assert observations_by_tier["V3"] - observations_by_tier["V2"] == set(TIER_ADDED_KEYS["V3"])


def test_robot_base_imu_matches_real_rotational_deck_lever_arm():
    amplitude = 0.03
    frequency_hz = 2.0
    omega = 2.0 * np.pi * frequency_hz

    def trajectory(time_s):
        angle = amplitude * np.sin(omega * time_s)
        rate = amplitude * omega * np.cos(omega * time_s)
        acceleration = -amplitude * omega**2 * np.sin(omega * time_s)
        return DeckCommand(
            pose=np.asarray((0.0, 0.0, 0.0, 0.0, angle, 0.0)),
            twist=np.asarray((0.0, 0.0, 0.0, 0.0, rate, 0.0)),
            acceleration=np.asarray((0.0, 0.0, 0.0, 0.0, acceleration, 0.0)),
        )

    env = _make_env("V1", deck_trajectory=trajectory, horizon=2)
    try:
        env.step(np.zeros(env.action_dim))
        sample = env.vibration_provider.imu.last_sample
        trace = env.deck_driver.trace
        assert sample is not None
        trace_index = int(np.argmin(np.abs(trace.sample_time_s - sample.acquisition_time_s)))
        deck_pose = trace.actual_pose[trace_index]
        deck_rotation = T.quat2mat(deck_pose[3:][[1, 2, 3, 0]])
        deck_twist = trace.actual_twist[trace_index]
        deck_acceleration = trace.actual_acceleration[trace_index].copy()
        # DeckDriver's trace retains its established object-acceleration
        # convention; convert its linear portion back to inertial acceleration
        # only at this independent comparison boundary.
        deck_acceleration[:3] += np.asarray(GRAVITY_WORLD_M_S2)
        base_id = env.sim.model.body_name2id("robot0_base")
        lever_arm_world = deck_rotation.dot(np.asarray(env.sim.model.body_pos[base_id], dtype=float))
        base_rotation_world = env.vibration_provider.imu.last_kinematics["rotation_world_to_sensor"].T
        expected = np.concatenate(
            (
                specific_force_from_rigid_body_motion(
                    deck_acceleration[:3],
                    deck_acceleration[3:],
                    deck_twist[3:],
                    lever_arm_world,
                    GRAVITY_WORLD_M_S2,
                    base_rotation_world.T,
                ),
                base_rotation_world.T.dot(deck_twist[3:]),
            )
        )
        deck_origin = np.concatenate(
            (
                specific_force_from_rigid_body_motion(
                    deck_acceleration[:3],
                    deck_acceleration[3:],
                    deck_twist[3:],
                    np.zeros(3),
                    GRAVITY_WORLD_M_S2,
                    deck_rotation.T,
                ),
                deck_rotation.T.dot(deck_twist[3:]),
            )
        )
        np.testing.assert_allclose(sample.clean_measurement, expected, rtol=0.0, atol=2e-9)
        assert np.linalg.norm(sample.clean_measurement[:3] - deck_origin[:3]) > 1.0
        assert np.linalg.norm(lever_arm_world) > 0.9
        # Confirm the implementation really sampled robot0_base, not the deck.
        direct_clean, _ = clean_imu_measurement_from_sim(env.sim, "robot0_base")
        np.testing.assert_allclose(direct_clean, sample.clean_measurement, rtol=0.0, atol=1e-10)
    finally:
        env.close()


@pytest.mark.parametrize("hard_reset", (False, True), ids=("nonhard", "hard"))
def test_live_acquisition_timeline_restarts_on_reset_without_gaps(hard_reset):
    env = _make_env("V1", hard_reset=hard_reset, horizon=4)
    try:
        env.reset()
        for _ in range(2):
            env.step(np.zeros(env.action_dim))
        trace = env.vibration_provider.imu.trace
        assert trace["acquisition_time_s"].shape == (20,)
        np.testing.assert_allclose(trace["acquisition_time_s"][:10], np.arange(0.005, 0.055, 0.005))
        np.testing.assert_allclose(trace["acquisition_time_s"][10:], np.arange(0.055, 0.105, 0.005))

        env.reset()
        assert env.vibration_provider.imu.trace["acquisition_time_s"].shape == (0,)
        np.testing.assert_allclose(
            env.vibration_provider.imu.window_acquisition_timestamps_s,
            np.arange(-0.050, 0.0, 0.005),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(env.vibration_provider.imu.pending_delivery_acquisition_timestamps_s, [0.0])
        env.step(np.zeros(env.action_dim))
        np.testing.assert_allclose(
            env.vibration_provider.imu.trace["delivered_acquisition_time_s"],
            np.arange(0.0, 0.050, 0.005),
        )
    finally:
        env.close()


def test_state_policy_boundary_remains_exact_with_renderer_flags():
    env = _make_env("V0", render_collision_mesh=True, render_visual_mesh=False)
    try:
        expected = set(COMMON_STATE_KEYS)
        assert set(env.reset()) == expected
        stepped, _, _, _ = env.step(np.zeros(env.action_dim))
        assert set(stepped) == expected
        assert "robot0_joint_acc" not in stepped
        assert "robot0_proprio-state" not in stepped
        assert "shakebench_task-state" not in stepped
    finally:
        env.close()


def test_public_goal_frame_matches_the_evaluator_target_frame_convention():
    env = _make_env("V0")
    try:
        observation = env.reset()
        target_world = env.target_frame_world_position()
        base_id = env.sim.model.body_name2id(env.robot_base_body_name)
        base_position = np.asarray(env.sim.data.xpos[base_id], dtype=float)
        base_rotation = np.asarray(env.sim.data.xmat[base_id], dtype=float).reshape(3, 3)
        table_id = env.sim.model.body_name2id(env.worktable_body_name)
        target_rotation = np.asarray(env.sim.data.xmat[table_id], dtype=float).reshape(3, 3)
        np.testing.assert_allclose(
            observation["goal_frame_pos_robot_base"], base_rotation.T.dot(target_world - base_position), atol=1e-7
        )
        # ``mat2quat`` is already xyzw; converting it again would reinterpret
        # the components as wxyz and invalidate the public target frame.
        expected_quat = T.mat2quat(base_rotation.T.dot(target_rotation))
        np.testing.assert_allclose(observation["goal_frame_quat_robot_base"], expected_quat, atol=1e-7)
        np.testing.assert_allclose(observation["goal_z_bounds_target"], (0.0, 0.035), atol=1e-7)
    finally:
        env.close()


def test_v2_transform_is_current_only_and_removes_rigid_frame_motion():
    frame = RigidBodyState(
        position_world_m=np.zeros(3),
        rotation_world=np.eye(3),
        twist_world=np.array([0.2, -0.1, 0.3, 0.0, 0.0, 2.0]),
        acceleration_world=np.array([0.1, 0.2, -0.3, 0.0, 0.0, 0.4]),
    )
    lever = np.array([1.0, 0.0, 0.0])
    child = RigidBodyState(
        position_world_m=lever,
        rotation_world=np.eye(3),
        twist_world=frame.twist_world + np.r_[np.cross(frame.twist_world[3:], lever), [0.0, 0.0, 0.0]],
        acceleration_world=frame.acceleration_world
        + np.r_[
            np.cross(frame.acceleration_world[3:], lever)
            + np.cross(frame.twist_world[3:], np.cross(frame.twist_world[3:], lever)),
            [0.0, 0.0, 0.0],
        ],
    )
    relative = relative_pose_twist_acceleration(child, frame)
    np.testing.assert_allclose(relative.pose[:3], lever)
    np.testing.assert_allclose(relative.twist, np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(relative.acceleration, np.zeros(6), atol=1e-12)


def test_v3_program_is_analytic_and_does_not_need_realized_state():
    program = build_excitation_program(seed=31, t0=0.23, level_scale=0.4)
    provider = V3Provider(program=program, seed=31, imu_mode="ideal_smoke")
    payload = provider.observation()
    np.testing.assert_array_equal(payload["line_mask"], program.line_mask)
    np.testing.assert_allclose(payload["line_accel_amplitude"], program.line_accel_amplitude, atol=1e-7)
    query_times = (0.0, 0.37, 1.5, 7.0)
    reference = [program.evaluate(query_time) for query_time in query_times]
    with mock.patch.object(type(program), "evaluate", side_effect=AssertionError("provider program was used")):
        for query_time, expected in zip(query_times, reference):
            actual = reconstruct_authored_motion(provider.public_program_payload(), query_time)
            np.testing.assert_allclose(actual.q, expected.q, rtol=2e-5, atol=1e-9)
            np.testing.assert_allclose(actual.qdot, expected.qdot, rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(actual.qdd, expected.qdd, rtol=5e-5, atol=5e-6)
    for query_time in query_times:
        expected = program.evaluate(query_time)
        actual = provider.evaluate_authored(query_time)
        np.testing.assert_allclose(actual.q, expected.q, rtol=2e-5, atol=1e-9)
    assert "seed" not in payload
    assert "t0" not in payload
    assert "actual_table" not in payload


def test_provider_rejects_unknown_tiers_and_non_self_describing_v3_trajectory():
    with pytest.raises(ShakeBenchProviderError, match="V0, V1, V2, V3"):
        make_vibration_provider("v1")
    with pytest.raises(ShakeBenchProviderError, match="ExcitationProgram"):
        V3Provider(program={})
