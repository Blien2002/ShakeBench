"""Execution-boundary state normalization: committed can_* schemas must run."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.scripts.shakebench_run_oracle import load_state_asset
from robosuite.utils.shakebench_state_schema import StateSchemaError, normalize_state

OFFICIAL_ASSET = "robosuite/models/assets/shakebench_states_official.json"


def _committed_record(**overrides):
    record = {
        "state_id": "official-0000",
        "split": "official",
        "can_xy_m": [-0.10, -0.13],
        "can_pose_worktable": [-0.10, -0.13, 0.07, 1.0, 0.0, 0.0, 0.0],
        "can_yaw_rad": 0.0,
        "can_initial_velocity": [0.0] * 6,
        "excitation_seed": 11,
        "imu_seed": 12,
        "t0_s": 0.25,
        "canonical_payload_sha256": "deadbeef",
    }
    record.update(overrides)
    return record


def test_committed_names_gain_execution_names_and_keep_their_identity():
    normalized = normalize_state(_committed_record())

    assert normalized["object_xy_m"] == [-0.10, -0.13]
    assert normalized["object_pose_worktable"] == normalized["can_pose_worktable"]
    assert normalized["object_yaw_rad"] == 0.0 and normalized["object_initial_velocity"] == [0.0] * 6
    assert normalized["can_xy_m"] == [-0.10, -0.13] and normalized["split"] == "official"
    assert normalized["canonical_payload_sha256"] == "deadbeef"
    assert "object_xy_m" not in _committed_record(), "normalization must copy, never mutate"


def test_dev_names_pass_through_unchanged():
    dev = {"state_id": "shakebench-dev-v0-000", "object_xy_m": [0.0, 0.0], "split": "dev"}

    assert normalize_state(dev) == dev


@pytest.mark.parametrize(
    "record",
    [
        {"state_id": "x", "can_xy_m": [0.0]},
        {"state_id": "x", "can_xy_m": [0.0, float("inf")]},
        {"state_id": "x", "can_xy_m": None},
    ],
)
def test_normalization_fails_closed(record):
    with pytest.raises(StateSchemaError):
        normalize_state(record)


def test_verified_official_asset_builds_and_steps_an_environment():
    """Regression: committed can_* states used to fail with KeyError object_xy_m."""

    state = load_state_asset(OFFICIAL_ASSET)["states"][0]
    assert "can_xy_m" in state and "object_xy_m" not in state

    env, _ = make_environment(state, gamma=0.0, horizon=2)
    try:
        assert np.isfinite(float(env.get_metrics()["max_illegal_penetration_m"]))
        _, reward, _, _ = env.step(np.zeros(env.action_dim))
        assert np.isfinite(float(reward))
    finally:
        env.close()
