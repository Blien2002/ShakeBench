import os

import numpy as np
import pytest

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

pytestmark = pytest.mark.renderer


def is_display_available() -> bool:
    return "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ


@pytest.mark.skipif(not is_display_available(), reason="No display available for on-screen rendering.")
def test_mujoco_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mujoco",
    )

    env.reset()

    low, high = env.action_spec

    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()


@pytest.mark.skipif(not is_display_available(), reason="No display available for on-screen rendering.")
def test_multiple_mujoco_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mujoco",
    )

    env.reset()
    camera_name = ["agentview", "birdview"]
    env.viewer.set_camera(camera_name=camera_name, width=1080, height=720)
    low, high = env.action_spec

    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()


@pytest.mark.skipif(not is_display_available(), reason="No display available for on-screen rendering.")
def test_mjviewer_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )

    # ``launch_passive`` owns a native GLFW thread. Closing explicitly is
    # required so its teardown completes before pytest exits; otherwise a test
    # can report PASS and still crash the interpreter during finalization.
    try:
        env.reset()

        low, high = env.action_spec

        for i in range(10):
            action = np.random.uniform(low, high)
            obs, reward, done, _ = env.step(action)
            env.render()
    finally:
        env.close()


def test_offscreen_renderer():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
    )

    env.reset()

    low, high = env.action_spec

    for i in range(10):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        assert obs["agentview_image"].shape == (256, 256, 3)
