"""Train-split state generation, verification, and split disjointness."""

from __future__ import annotations

import json

import pytest

from robosuite.scripts.shakebench_generate_train_states import main as generate_main
from robosuite.scripts.shakebench_run_oracle import load_state_asset
from robosuite.utils.shakebench_dev_states import build_phase07_dev_state_artifact
from robosuite.utils.shakebench_train_states import (
    TrainStateError,
    assert_split_disjoint,
    build_train_state_artifact,
    execution_state_fingerprint,
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
    assert [row["state_id"] for row in asset["states"]] == [
        row["state_id"] for row in generate_train_states(5, seed=3)
    ]


def test_overlapping_train_and_evaluation_states_are_rejected():
    assert_split_disjoint(["a", "b"], ["c"])
    with pytest.raises(TrainStateError, match="overlap"):
        assert_split_disjoint(["a", "b"], ["b", "c"])


def test_train_generator_never_reproduces_the_frozen_dev_execution_states():
    """Regression: the train generator once shared the dev namespace word for word."""

    train = build_train_state_artifact(10, seed=20260905)["states"]
    dev = build_phase07_dev_state_artifact()["states"]

    assert not set(map(execution_state_fingerprint, train)) & set(map(execution_state_fingerprint, dev))
    fields = ("object_xy_m", "excitation_seed", "imu_seed", "t0_s")
    assert not any(
        all(train_row[field] == dev_row[field] for field in fields) for train_row, dev_row in zip(train, dev)
    )


def test_split_check_compares_execution_states_not_only_names():
    """A renamed copy of one execution state is still the same training state."""

    dev = build_phase07_dev_state_artifact()["states"][0]
    renamed = {**dev, "state_id": "some-other-pool-0000"}

    with pytest.raises(TrainStateError, match="overlap"):
        assert_split_disjoint([renamed], [dev])
    assert execution_state_fingerprint(renamed) == execution_state_fingerprint(dev)


def test_state_ids_bind_the_generation_request():
    """Different pools must not share state IDs, or merges replace the wrong episode."""

    first = {row["state_id"] for row in build_train_state_artifact(2, seed=1)["states"]}
    second = {row["state_id"] for row in build_train_state_artifact(2, seed=2)["states"]}
    narrower = {row["state_id"] for row in build_train_state_artifact(2, seed=1, half_range_m=0.01)["states"]}

    assert first == {row["state_id"] for row in build_train_state_artifact(2, seed=1)["states"]}
    assert first.isdisjoint(second) and first.isdisjoint(narrower)
    assert sorted(first)[0] == "shakebench-train-v0-s1-r0.02-0000"


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
    assert generate_train_states(3, seed=2)[0]["state_id"] == "shakebench-train-v0-s2-r0.02-0000"
