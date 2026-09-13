from __future__ import annotations

import numpy as np

import robosuite
from robosuite.utils.shakebench_sensors import CANONICAL_IMU_PROFILE, clean_imu_measurement_from_sim


def test_table_imu_profile_and_live_observation_contract():
    frame = CANONICAL_IMU_PROFILE.to_dict()["frame"]
    assert frame == {
        "parent": "worktable",
        "position_m": [0.0, 0.0, -0.03],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }

    env = robosuite.make(
        "VibrationPickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="probe",
        imu_mode="ideal_smoke",
        horizon=2,
        seed=17,
    )
    try:
        observation = env.reset()
        assert set(env.vibration_observation_keys) == {
            "table_imu_window",
            "table_imu_dt_s",
            "table_imu_timestamps_s",
        }
        assert not any(key.startswith("deck_imu") for key in observation)
        assert observation["table_imu_window"].shape == (10, 6)
        assert observation["table_imu_dt_s"] == np.float32(0.005)
        np.testing.assert_allclose(observation["table_imu_timestamps_s"], np.arange(-0.05, 0.0, 0.005))
        clean, _ = clean_imu_measurement_from_sim(
            env.sim, "worktable", sensor_position_body_m=env.table_imu_position_m,
        )
        np.testing.assert_allclose(
            observation["table_imu_window"], np.broadcast_to(clean, (10, 6)), atol=1e-6,
        )
        assert env._imu_mount_audit["sensor_parent"] == "worktable"
        assert env._imu_mount_audit["parent_body_name"] == "worktable"
        assert len(env._imu_mount_audit["sensor_config_sha256"]) == 64

        stepped, _, _, _ = env.step(np.zeros(env.action_dim))
        np.testing.assert_allclose(stepped["table_imu_timestamps_s"], np.arange(0.0, 0.05, 0.005))
        assert env.table_imu_provider.imu.last_kinematics["lever_arm_world_m"].shape == (3,)
    finally:
        env.close()
