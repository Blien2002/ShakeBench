"""Current-scene expert regression and optional real CUDA consistency check.

Run CUDA with SHAKEBENCH_TEST_DEVICE=cuda:0 and requirements-gpu.txt installed.
"""

import os

import numpy as np
import pytest

from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.utils.shakebench_expert import ORACLE_STATE_KEYS, oracle_observation
from robosuite.utils.shakebench_oracle import (
    OracleControllerProfile,
    ShakeBenchOracleController,
    WorktableTaskContext,
)
from robosuite.utils.shakebench_providers import TABLE_IMU_POLICY_KEYS


@pytest.mark.parametrize("gamma", [0.0, 0.15])
def test_current_scene_expert_and_optional_cuda(gamma):
    state = {"seed": 42, "object_xy_m": [-0.08, -0.13]}
    env, program = make_environment(
        state, gamma=gamma, horizon=3, physics_profile=os.environ.get("SHAKEBENCH_TEST_PHYSICS_PROFILE", "official")
    )
    try:
        public = env._get_observations(force_update=True)
        public_before = {key: np.asarray(value).copy() for key, value in public.items()}
        obs = oracle_observation(env, public)
        assert set(obs) == set(ORACLE_STATE_KEYS)
        assert "robot0_eef_pos_robot_base" not in public
        assert "object_pos_robot_base" not in public  # truth is collection-only
        for key in public:
            np.testing.assert_array_equal(public[key], public_before[key])
        assert env.geometry_profile["profile_id"] == "world_fixed_arm_v1"
        assert env.table_imu_provider.audit_compiled_mount(env.sim)["parent_body_name"] == "worktable"
        sample = program.evaluate(np.linspace(0, 3, 61))
        if gamma == 0:
            for values in (sample.q, sample.qdot, sample.qdd):
                np.testing.assert_array_equal(values, np.zeros_like(values))
        else:
            assert np.max(np.abs(sample.q)) > 0
        controller = ShakeBenchOracleController(
            task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
        )
        batch = None
        if os.environ.get("SHAKEBENCH_TEST_DEVICE"):
            # An explicitly requested CUDA check must fail, never skip or fall
            # back to CPU, if the dependency/device is unavailable.
            from robosuite.utils.shakebench_mjwarp import MJWarpBatch

            batch = MJWarpBatch([env], [program], device=os.environ["SHAKEBENCH_TEST_DEVICE"])
            np.testing.assert_allclose(
                batch.model.opt.tolerance.numpy(), env.sim.model.opt.tolerance, rtol=1e-6, atol=0
            )
            initial = batch.reset()[0]
            assert set(initial) == set(ORACLE_STATE_KEYS) | set(TABLE_IMU_POLICY_KEYS)
            assert batch.e.imu_body == env.worktable_body_id
            batch.enable_rendering(["agentview"], resolution=(64, 64))
            render_buffer = batch.render_buffers["agentview"]
            frames = batch.render_rgb()["agentview"]
            next_frames = batch.render_rgb()["agentview"]
            assert frames.shape == (1, 64, 64, 3) and frames.dtype == np.uint8
            assert next_frames.shape == frames.shape and next_frames.dtype == frames.dtype
            assert np.ptp(frames) > 20  # a rendered scene, not a flat buffer
            assert batch.render_buffers["agentview"] is render_buffer
        base = env.sim.data.xpos[env.robot_base_body_id].copy()
        for step in range(3):
            action = controller.action(obs, time_s=step / 20)
            if step == 1:
                action[:3] = [0.05, -0.05, 0.05]  # exercise OSC motion as well as hold
            assert action.shape == (7,) and np.isfinite(action).all()
            env.step(np.clip(action, -1, 1))
            obs = oracle_observation(env)
            np.testing.assert_array_equal(env.sim.data.xpos[env.robot_base_body_id], base)
            if batch is not None:
                gpu_obs, metrics = batch.step(np.asarray([action]))
                assert not metrics["invalid"][0]
                np.testing.assert_allclose(metrics["qpos"][0], env.sim.data.qpos, atol=1e-3, rtol=0)
                for key in ("robot0_eef_pos_robot_base", "object_pos_robot_base", "goal_frame_pos_robot_base"):
                    np.testing.assert_allclose(gpu_obs[0][key], obs[key], atol=1e-3, rtol=0)
                cpu_imu = env.table_imu_provider.observation()
                np.testing.assert_allclose(
                    gpu_obs[0]["table_imu_window"], cpu_imu["table_imu_window"], atol=0.1, rtol=0.01
                )
                np.testing.assert_allclose(
                    gpu_obs[0]["table_imu_timestamps_s"], cpu_imu["table_imu_timestamps_s"], atol=1e-12, rtol=0
                )
                # Isolate sensor decoding from cross-backend trajectory error:
                # native RNE on the downloaded GPU state must give the same IMU.
                import mujoco
                import mujoco_warp as mjw

                from robosuite.utils.shakebench_sensors import clean_imu_measurement_from_sim

                raw = env.sim.model._model
                downloaded = mujoco.MjData(raw)
                mjw.get_data_into(downloaded, raw, batch.data, world_id=0)
                clean, _ = clean_imu_measurement_from_sim(
                    raw,
                    "worktable",
                    data=downloaded,
                    sensor_position_body_m=env.table_imu_position_m,
                    sensor_quat_body_wxyz=env.table_imu_quat_wxyz,
                )
                np.testing.assert_allclose(batch.e.imu.numpy()[0, -1], clean, atol=1e-5, rtol=1e-5)
    finally:
        env.close()


def test_recorded_phase_names_are_labels_not_trace_dictionaries():
    """Regression: the NPZ phase array stored truncated diagnostics text."""

    from types import SimpleNamespace

    from robosuite.scripts.shakebench_gpu_batch import phase_name

    controller = SimpleNamespace(executive=SimpleNamespace(phase=SimpleNamespace(value="settle")))
    controller.last_trace = {"phase": {"phase": "settle", "task_context_sha256": "0" * 64}}

    recorded = phase_name(controller)

    assert recorded == "settle"
    assert not recorded.startswith("{")
    assert np.asarray([recorded, "grasp"], dtype="U64").tolist() == ["settle", "grasp"]


def test_device_rollout_latches_success():
    """Regression: an official device oracle rollout must reach success_latched."""

    device = os.environ.get("SHAKEBENCH_TEST_DEVICE")
    if not device:
        pytest.skip("set SHAKEBENCH_TEST_DEVICE=cuda:0 with requirements-gpu.txt installed")

    import mujoco

    from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
    from robosuite.utils.shakebench_mjwarp import (
        DEVICE_FINGER_CAN_CONTACT_SOLREF,
        DEVICE_SUPPORT_CONTACT_SOLREF,
        MJWarpBatch,
    )

    state = {"seed": 42, "object_xy_m": [-0.08, -0.13]}
    horizon = 300
    env, program = make_environment(state, gamma=0.0, horizon=horizon, physics_profile="official")
    try:
        batch = MJWarpBatch([env], [program], device=device)
        host = env.sim.model._model
        device_solref = batch.model.pair_solref.numpy()
        solref = device_solref.reshape(host.npair, device_solref.shape[-1])
        assert solref.shape == (host.npair, 2)

        def geom_ids(names):
            return {mujoco.mj_name2id(host, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names}

        can_ids = geom_ids(env.can.contact_geoms)
        finger_ids = geom_ids(env.finger_pad_geom_names)
        support_ids = geom_ids(env.table_contact_geom_names + env.target_bottom_geom_names + env.target_wall_geom_names)
        calibrated = {"finger": 0, "support": 0}
        for index in range(host.npair):
            pair = {int(host.pair_geom1[index]), int(host.pair_geom2[index])}
            if not pair & can_ids:
                continue
            if pair & finger_ids:
                np.testing.assert_allclose(solref[index], DEVICE_FINGER_CAN_CONTACT_SOLREF, rtol=1e-6, atol=0)
                calibrated["finger"] += 1
            elif pair & support_ids:
                np.testing.assert_allclose(solref[index], DEVICE_SUPPORT_CONTACT_SOLREF, rtol=1e-6, atol=0)
                calibrated["support"] += 1
        assert calibrated["finger"] == len(finger_ids)
        assert calibrated["support"] == len(support_ids)
        controller = ShakeBenchOracleController(
            OracleControllerProfile(),
            task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"]),
        )
        observation = batch.reset()[0]
        latched_at = None
        penetration_m = 0.0
        for step in range(horizon):
            action = np.clip(controller.action(observation, time_s=step / 20), -1, 1)
            observations, metrics = batch.step([action])
            observation = observations[0]
            assert not metrics["invalid"][0]
            penetration_m = max(penetration_m, float(metrics["contacts"][0][0]))
            if metrics["success"][0]:
                latched_at = step
                break
        assert latched_at is not None, "device rollout reached the tray but never latched success"
        assert penetration_m < DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m
    finally:
        env.close()
