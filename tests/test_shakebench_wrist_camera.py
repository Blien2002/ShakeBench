"""Exercise wrist images and mounting through the compiled ShakeBench scene."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import robosuite
from robosuite.demos import demo_shakebench_oracle_video as video


@pytest.mark.renderer
@pytest.mark.parametrize("geometry_profile", ["canonical", "direct_mount_v1"])
def test_default_visual_observations_include_a_moving_wrist_camera(geometry_profile):
    env = robosuite.make(
        "VibrationPickPlaceCan",
        use_camera_obs=True,
        has_offscreen_renderer=True,
        camera_heights=[64, 80],
        camera_widths=[96, 80],
        camera_depths=[False, True],
        seed=17,
        geometry_profile=geometry_profile,
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
        if geometry_profile == "direct_mount_v1":
            assert env.scene_config.scene_id == "shakebench.scene.direct_mount.v1"
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
        "VibrationPickPlaceCan",
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


@pytest.mark.renderer
def test_video_inset_renders_without_changing_state_or_policy_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "FFmpegVideoWriter", Mock())
    env = robosuite.make(
        "VibrationPickPlaceCan", observation_tier="V0", geometry_profile="direct_mount_v1", hard_reset=False, seed=17
    )
    observer = video.VideoObserver(
        tmp_path / "wrist.mp4",
        camera="presentation",
        width=512,
        height=512,
        fps=20,
        tier="V0",
        gamma=0.0,
        state_id="test",
        policy_rate_hz=20.0,
        scene_config=env.scene_config,
    )
    controller = SimpleNamespace(last_trace=None, executive=SimpleNamespace(phase=SimpleNamespace(value="reset")))
    try:
        obs = env.reset()
        before = env.sim.get_state().flatten().copy()
        keys = set(obs)
        observer(env, 0, obs, controller)
        overview = observer.last_frame.copy()
        observer.wrist_inset = True
        observer(env, 0, obs, controller)
        combined = observer.last_frame
        assert combined.shape == (512, 512, 3)
        # GPU rasterization can round a few channel values differently on
        # successive renders; the overview must otherwise remain unchanged.
        np.testing.assert_allclose(combined[:342], overview[:342], atol=1, rtol=0)
        assert not np.array_equal(combined[342:, 342:], overview[342:, 342:])
        np.testing.assert_array_equal(env.sim.get_state().flatten(), before)
        assert set(obs) == keys == set(env.policy_observation_keys)
        assert not any(key.endswith("_image") for key in obs)
    finally:
        observer.close(success=False, hold_seconds=0)
        env.close()
