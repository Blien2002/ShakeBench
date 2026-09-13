"""Reset must remove loaded-support transients before the episode starts."""

import numpy as np

from robosuite.environments.manipulation.vibration_pick_place import VibrationPickPlace
from robosuite.utils.shakebench_sensors import clean_imu_measurement_from_sim


def test_loaded_reset_is_quiet_and_does_not_advance_episode():
    env = VibrationPickPlace(
        physics_profile="probe", vibration={"gamma": 0.0}, seed=42,
        load_model_on_init=False, hard_reset=False, use_camera_obs=False,
        has_renderer=False, has_offscreen_renderer=False, use_object_obs=False,
    )
    try:
        for _ in range(2):
            env.reset()
            assert env.sim.data.time == env.cur_time == env.timestep == env._physics_step_index == 0
            # The analytic payload equilibrium seeds the support, so the
            # bounded settle only absorbs weld and contact residuals.
            assert 0.1 <= env.reset_settle_duration_s <= 1.5
            assert not env.table_imu_provider.imu.records
            np.testing.assert_array_equal(env.sim.data.qvel, 0.0)
            robot = env.robots[0]
            np.testing.assert_allclose(env.sim.data.qpos[robot._ref_joint_pos_indexes], robot.init_qpos)
            clean, state = clean_imu_measurement_from_sim(
                env.sim, "worktable", sensor_position_body_m=env.table_imu_position_m,
            )
            resting_force = state["rotation_world_to_sensor"] @ -env.sim.model.opt.gravity
            np.testing.assert_allclose(clean[:3], resting_force, atol=0.003)
            for _ in range(4):
                env.step(np.zeros(env.action_dim))
            samples = env.table_imu_provider.imu.trace["clean_measurement"]
            np.testing.assert_allclose(samples[:, :3], np.broadcast_to(resting_force, samples[:, :3].shape), atol=0.003)
            np.testing.assert_allclose(samples[:, 3:], 0.0, atol=1e-4)
            assert len(samples) == 40
    finally:
        env.close()
