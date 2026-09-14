"""Policy-agnostic evaluation runner: factory, observation contract, split guard."""

from __future__ import annotations

import json

import pytest

from robosuite.scripts.shakebench_evaluate import (
    build_policy,
    load_policy_factory,
    main,
    observation_config_from_collection,
    parse_policy_args,
    requested_state_ids,
    summarize,
)
from robosuite.utils.shakebench_train_states import TrainStateError


def _collection(tmp_path, *, state_ids, main_camera="task_close", height=64, width=64, complete=True):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"features": {"observation.images.main": {"shape": [height, width, 3]}}})
    )
    (meta / "shakebench_collection.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "cameras": {"observation.images.main": main_camera},
                "episodes": [{"state": {"state_id": state_id}} for state_id in state_ids],
                "sft_subset": {"selection_rule": "termination_cause == success_latched"},
            }
        )
    )
    return dataset


def test_policy_factory_is_imported_by_module_path():
    factory = load_policy_factory("robosuite.scripts.shakebench_evaluate:load_policy_factory")

    assert callable(factory)
    for bad in ("no_colon", ":missing_module_attribute", "robosuite.scripts.shakebench_evaluate:nope"):
        with pytest.raises((ValueError, ModuleNotFoundError, AttributeError)):
            load_policy_factory(bad)


def test_policy_arguments_and_call_signature_are_respected():
    def factory(*, port, inference_timeout_s=None):
        return type(
            "P", (), {"chunk_size": 4, "predict": lambda self, obs: None, "port": port, "deadline": inference_timeout_s}
        )()

    arguments = parse_policy_args(["port=10093", "host=127.0.0.1"])

    assert arguments == {"port": 10093, "host": "127.0.0.1"}
    policy = build_policy(factory, arguments={"port": 10093}, inference_timeout_s=5.0)
    assert policy.port == 10093 and policy.chunk_size == 4 and policy.deadline == 5.0
    with pytest.raises(ValueError, match="does not accept"):
        build_policy(factory, arguments={"nope": 1}, inference_timeout_s=None)
    with pytest.raises(ValueError, match="chunk_size"):
        build_policy(
            lambda: type("Q", (), {"chunk_size": 0, "predict": None})(), arguments={}, inference_timeout_s=None
        )


def test_observation_config_comes_from_the_training_dataset(tmp_path):
    dataset = _collection(tmp_path, state_ids=["shakebench-train-v0-0000"], main_camera="robot0_eye_in_hand", height=48)

    config = observation_config_from_collection(dataset)

    assert config["main_camera"] == "robot0_eye_in_hand"
    assert (config["height"], config["width"]) == (48, 64)
    assert config["train_state_ids"] == ["shakebench-train-v0-0000"]
    assert len(config["manifest_sha256"]) == 64
    assert observation_config_from_collection(dataset / "meta" / "shakebench_collection.json")["height"] == 48
    with pytest.raises(ValueError, match="complete"):
        observation_config_from_collection(_collection(tmp_path / "b", state_ids=[], complete=False))


def test_state_selection_rejects_unknown_and_duplicate_ids():
    states = [{"state_id": "a"}, {"state_id": "b"}]

    assert requested_state_ids(states, None) == states
    assert requested_state_ids(states, ["b,a"]) == [states[1], states[0]]
    with pytest.raises(ValueError, match="unknown state IDs"):
        requested_state_ids(states, ["c"])
    with pytest.raises(ValueError, match="duplicate"):
        requested_state_ids(states, ["a", "a"])


def test_summary_keeps_the_failure_denominator():
    episodes = [
        {
            "episode_validity": "valid",
            "score_outcome": "success",
            "termination_cause": "success_latched",
            "policy_errors": [],
        },
        {
            "episode_validity": "valid",
            "score_outcome": "unsuccessful",
            "termination_cause": "horizon_exhausted",
            "policy_errors": [],
        },
        {
            "episode_validity": "valid",
            "score_outcome": "unsuccessful",
            "termination_cause": "policy_error",
            "policy_errors": [{"error_type": "policy_timeout"}],
        },
        {
            "episode_validity": "invalid",
            "score_outcome": None,
            "termination_cause": "invalid_execution",
            "policy_errors": [],
        },
    ]

    summary = summarize(episodes)

    assert summary["attempted_episodes"] == 4 and summary["valid_episodes"] == 3
    assert summary["invalid_execution_episodes"] == 1
    assert summary["success"] == 1 and summary["unsuccessful"] == 2
    assert summary["success_rate_over_attempts"] == 0.25
    assert summary["success_rate_over_valid"] == pytest.approx(1 / 3)
    assert summary["policy_error_types"] == {"policy_timeout": 1}
    assert summary["termination_causes"]["invalid_execution"] == 1


def test_overlapping_train_states_stop_the_run_before_any_policy_is_built(tmp_path):
    dataset = _collection(tmp_path, state_ids=["shakebench-dev-v0-000"])

    with pytest.raises(TrainStateError, match="overlap"):
        main(
            [
                "--policy",
                "robosuite.scripts.shakebench_evaluate:load_policy_factory",
                "--dataset",
                str(dataset),
                "--state-ids",
                "shakebench-dev-v0-000",
                "--output",
                str(tmp_path / "results.json"),
            ]
        )
    assert not (tmp_path / "results.json").exists()


def test_results_are_never_overwritten(tmp_path):
    existing = tmp_path / "results.json"
    existing.write_text("{}")

    with pytest.raises(FileExistsError):
        main(["--policy", "m:f", "--output", str(existing)])
