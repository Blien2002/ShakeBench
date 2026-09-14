"""Tests for the qualitative ShakeBench oracle video demo."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robosuite.demos.demo_shakebench_oracle_video import _task_close_camera, annotate_frame, build_parser
from robosuite.utils.shakebench_geometry import load_geometry_profile


def test_demo_defaults_are_a_fixed_reproducible_scenario():
    args = build_parser().parse_args([])
    assert args.tier == "V0"
    assert args.gamma == 0.15
    assert args.state_id == "shakebench-dev-v0-000"
    assert args.camera == "presentation"
    assert args.geometry_profile == "world_fixed_arm_v1"
    assert args.wrist_inset is False
    assert args.fps == 20


def test_overlay_preserves_frame_contract_and_adds_status_pixels():
    source = np.zeros((180, 320, 3), dtype=np.uint8)
    rendered = annotate_frame(
        source,
        tier="V0",
        gamma=0.15,
        state_id="shakebench-dev-v0-000",
        step=12,
        policy_rate_hz=20.0,
        phase="transport",
        success=False,
    )
    assert rendered.shape == source.shape
    assert rendered.dtype == np.uint8
    assert np.any(rendered != 0)
    assert not np.shares_memory(rendered, source)


def test_task_close_camera_holds_the_libero_agentview_pose():
    """The main view is LIBERO's agentview, scaled so the deck fills the frame like LIBERO's table.

    Upstream source: libero/libero/envs/bddl_base_domain.py::BenchmarkEnv._setup_camera,
    which sets the observation camera for a tabletop centred at (0, 0, 0.8).
    """
    profile = load_geometry_profile()
    deck_top = np.asarray(profile["table_top_pos_m"], dtype=float)
    scale = float(profile["retained_table_dimensions_m"][0]) / 0.8
    libero_pos = np.array([0.5886131746834771, 0.0, 1.4903500240372423])
    libero_quat = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
    camera = _task_close_camera()
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    rotated = np.zeros(9)
    mujoco.mju_quat2Mat(rotated, np.array(libero_quat))
    rotated = rotated.reshape(3, 3)
    assert abs(rotated[2, 0]) < 1e-6  # LIBERO's view has no roll, so a free camera can hold it
    forward = -rotated[:, 2]
    azimuth, elevation = np.radians(camera.azimuth), np.radians(camera.elevation)
    # MuJoCo free-camera convention: pos = lookat - distance * (cos el cos az, cos el sin az, sin el).
    direction = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    np.testing.assert_allclose(direction, forward, atol=1e-6)  # mju_quat2Mat is single precision
    np.testing.assert_allclose(
        camera.lookat - camera.distance * direction, deck_top + scale * (libero_pos - [0.0, 0.0, 0.8]), atol=1e-6
    )
    assert camera.lookat[2] == pytest.approx(deck_top[2])  # optical axis lands on the deck plane
    # The deck-to-table size ratio is what makes the deck span LIBERO's share of the frame.
    assert camera.distance == pytest.approx(scale * (libero_pos[2] - 0.8) / -forward[2], abs=1e-9)
    assert abs(camera.azimuth) == pytest.approx(180.0)  # camera stands on +x looking back at the deck
    assert camera.elevation == pytest.approx(-38.922, abs=1e-3)
