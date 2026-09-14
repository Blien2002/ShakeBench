"""Train-split state generation, verification, and split disjointness."""

from __future__ import annotations

import json

import pytest

from robosuite.scripts.shakebench_generate_train_states import main as generate_main
from robosuite.scripts.shakebench_run_oracle import load_state_asset
from robosuite.utils.shakebench_train_states import (
    TrainStateError,
    assert_split_disjoint,
    build_train_state_artifact,
    generate_train_states,
    verify_train_state_artifact,
)


def test_train_pool_is_deterministic_unique_and_in_bounds():
    first = build_train_state_artifact(12, seed=5)
    second = build_train_state_artifact(12, seed=5)

    assert first == second, "the same seed must rebuild the same pool"
    assert first["split"] == "train" and first["scoreable"] is False
    states = first["states"]
    assert len(states) == 12 and len({row["state_id"] for row in states}) == 12
    from robosuite.utils.shakebench_dev_states import CAN_NOMINAL_XY_M

    for row in states:
        for axis in (0, 1):
            assert abs(row["object_xy_m"][axis] - CAN_NOMINAL_XY_M[axis]) <= 0.02 + 1e-12
        assert row["object_yaw_rad"] == 0.0 and row["object_initial_velocity"] == [0.0] * 6
    assert build_train_state_artifact(12, seed=6)["states"] != states


def test_tampered_or_malformed_train_artifacts_fail_closed():
    payload = build_train_state_artifact(3, seed=1)
    tampered = json.loads(json.dumps(payload))
    tampered["states"][0]["object_xy_m"] = [-0.30, -0.13]
    assert not verify_train_state_artifact(tampered)["passed"]
    assert not verify_train_state_artifact({})["passed"]
    assert not verify_train_state_artifact({**payload, "scoreable": True})["passed"]
    with pytest.raises(TrainStateError):
        build_train_state_artifact(3, seed=1, half_range_m=0.5)


def test_generator_cli_writes_a_verifiable_asset(tmp_path, capsys):
    target = tmp_path / "train.json"

    assert generate_main(["--output", str(target), "--count", "4", "--seed", "9"]) == 0
    payload = json.loads(target.read_text())
    assert verify_train_state_artifact(payload)["passed"]
    assert "4 train states" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        generate_main(["--output", str(target), "--count", "4", "--seed", "9"])


def test_load_state_asset_accepts_the_train_pool_and_never_scores_it(tmp_path):
    target = tmp_path / "train.json"
    generate_main(["--output", str(target), "--count", "5", "--seed", "3"])

    asset = load_state_asset(target)

    assert asset["authority"]["kind"] == "train" and asset["authority"]["scoreable"] is False
    assert [row["state_id"] for row in asset["states"]] == [f"shakebench-train-v0-{index:04d}" for index in range(5)]


def test_overlapping_train_and_evaluation_states_are_rejected():
    assert_split_disjoint(["a", "b"], ["c"])
    with pytest.raises(TrainStateError, match="overlap"):
        assert_split_disjoint(["a", "b"], ["b", "c"])


def test_collection_limit_larger_than_the_pool_fails_closed(tmp_path):
    from robosuite.scripts.shakebench_collect_lerobot import main as collect_main

    target = tmp_path / "train.json"
    generate_main(["--output", str(target), "--count", "3", "--seed", "2"])
    with pytest.raises(ValueError, match="exceeds the 3 selected states"):
        collect_main(
            [
                "--output",
                str(tmp_path / "dataset"),
                "--states",
                str(target),
                "--limit",
                "100",
            ]
        )
    assert generate_train_states(3, seed=2)[0]["state_id"] == "shakebench-train-v0-0000"
