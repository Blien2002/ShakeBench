"""Tests for the qualitative ShakeBench oracle video demo."""

from __future__ import annotations

import numpy as np

from robosuite.demos.demo_shakebench_oracle_video import annotate_frame, build_parser


def test_demo_defaults_are_a_fixed_reproducible_scenario():
    args = build_parser().parse_args([])
    assert args.tier == "V0"
    assert args.gamma == 0.15
    assert args.state_id == "shakebench-dev-v0-000"
    assert args.camera == "presentation"
    assert args.geometry_profile == "direct_mount_v1"
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
