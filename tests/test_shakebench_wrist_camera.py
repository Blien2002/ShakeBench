"""Exercise wrist images and mounting through the compiled ShakeBench scene."""

import numpy as np
import pytest

import robosuite


@pytest.mark.renderer
def test_default_visual_observations_include_a_moving_wrist_camera():
    env = robosuite.make(
        "VibrationPickPlace",
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_heights=[64, 80],
        camera_widths=[96, 80],
        camera_depths=[False, True],
        seed=17,
    )
    try:
        obs = env.reset()
        assert obs["agentview_image"].shape == (64, 96, 3)
        wrist = obs["robot0_eye_in_hand_image"]
        assert wrist.shape == (80, 80, 3)
        assert wrist.dtype == np.uint8
        assert np.ptp(wrist) > 30
        depth = obs["robot0_eye_in_hand_depth"]
        assert depth.shape == (80, 80, 1)
        assert np.isfinite(depth).all()
        assert np.ptp(depth) > 0

        model, data = env.sim.model, env.sim.data
        camera_id = model.camera_name2id("robot0_eye_in_hand")
        hand_id = model.body_name2id("robot0_right_hand")
        assert model.cam_bodyid[camera_id] == hand_id
        assert env.scene_config.scene_id == "shakebench.scene.world_fixed_arm.v1"
        assert "shakebench_camera_overview" in model.camera_names
        assert env.geometry_profile["mount_type"] == "NullMount"
        before = data.cam_xpos[camera_id].copy()
        before_rotation = data.cam_xmat[camera_id].copy()
        external_id = model.camera_name2id("agentview")
        external = data.cam_xpos[external_id].copy()
        # Move a real arm joint, then check that the camera remains rigid in
        # the hand frame while its world pose and rendered image change.
        data.qpos[env.robots[0]._ref_joint_pos_indexes[5]] += 0.2
        env.sim.forward()
        rotation = data.body_xmat[hand_id].reshape(3, 3)
        np.testing.assert_allclose(
            rotation.T @ (data.cam_xpos[camera_id] - data.body_xpos[hand_id]),
            model.cam_pos[camera_id],
            atol=1e-8,
        )
        assert not np.allclose(data.cam_xpos[camera_id], before)
        assert not np.allclose(data.cam_xmat[camera_id], before_rotation)
        np.testing.assert_allclose(data.cam_xpos[external_id], external)
        # reset() force-samples observables; allow the next sampling period
        # to start before checking the delivered image for a change.
        for _ in range(2):
            stepped, _, _, _ = env.step(np.zeros(env.action_dim))
        assert not np.array_equal(stepped["robot0_eye_in_hand_image"], wrist)
        assert env.reset()["robot0_eye_in_hand_image"].shape == (80, 80, 3)
    finally:
        env.close()


@pytest.mark.renderer
def test_explicit_single_camera_remains_supported():
    env = robosuite.make(
        "VibrationPickPlace",
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_names="robot0_eye_in_hand",
        camera_heights=64,
        camera_widths=64,
        hard_reset=False,
        seed=17,
    )
    try:
        obs = env.reset()
        assert obs["robot0_eye_in_hand_image"].shape == (64, 64, 3)
        assert "agentview_image" not in obs
    finally:
        env.close()
