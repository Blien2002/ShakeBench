"""Policy-agnostic evaluation runner: factory, observation contract, split guard."""

from __future__ import annotations

import json
import runpy
from unittest.mock import patch

import numpy as np
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
from robosuite.utils.shakebench_dev_states import build_phase07_dev_state_artifact
from robosuite.utils.shakebench_train_states import TrainStateError, execution_state_fingerprint

_STUB_TASK = runpy.run_path("tests/test_shakebench_rollout.py")["_StubTask"]
# The frozen dev records, i.e. exactly what a dev-state collection manifest holds.
_DEV_STATES = build_phase07_dev_state_artifact()["states"]


def _state(state_id, *, xy=(-0.10, -0.13)):
    return {
        "state_id": state_id,
        "object_xy_m": list(xy),
        "excitation_seed": 7,
        "imu_seed": 8,
        "t0_s": 0.25,
    }


def _collection(tmp_path, *, states, main_camera="task_close", height=64, width=64, complete=True):
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
                "episodes": [{"state": state} for state in states],
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
    dataset = _collection(
        tmp_path,
        states=[_state("shakebench-train-v0-s2-r0.02-0000")],
        main_camera="robot0_eye_in_hand",
        height=48,
    )

    config = observation_config_from_collection(dataset)

    assert config["main_camera"] == "robot0_eye_in_hand"
    assert (config["height"], config["width"]) == (48, 64)
    assert [state["state_id"] for state in config["train_states"]] == ["shakebench-train-v0-s2-r0.02-0000"]
    assert config["subset_provenance"] is None
    assert len(config["manifest_sha256"]) == 64
    assert observation_config_from_collection(dataset / "meta" / "shakebench_collection.json")["height"] == 48
    with pytest.raises(ValueError, match="complete"):
        observation_config_from_collection(_collection(tmp_path / "b", states=[], complete=False))


def test_subset_provenance_links_the_export_to_its_source_manifest(tmp_path):
    dataset = _collection(tmp_path, states=[_state("state-0")])
    provenance = {"schema_id": "shakebench.sft_subset", "source_manifest_sha256": "a" * 64}
    (dataset / "meta" / "shakebench_sft_subset.json").write_text(json.dumps(provenance))

    config = observation_config_from_collection(dataset)

    assert config["subset_provenance"] == provenance
    assert config["manifest_sha256"] != provenance["source_manifest_sha256"]


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
    dev_state = _DEV_STATES[0]
    dataset = _collection(tmp_path, states=[dev_state])

    with pytest.raises(TrainStateError, match="overlap"):
        main(
            [
                "--policy",
                "robosuite.scripts.shakebench_evaluate:load_policy_factory",
                "--dataset",
                str(dataset),
                "--state-ids",
                dev_state["state_id"],
                "--output",
                str(tmp_path / "results.json"),
            ]
        )
    assert not (tmp_path / "results.json").exists()


def test_split_guard_also_rejects_a_renamed_training_execution_state(tmp_path):
    """The training state leaked through a different state ID, so IDs alone cannot guard."""

    dev_state = _DEV_STATES[1]
    dataset = _collection(tmp_path, states=[{**dev_state, "state_id": "some-other-pool-0000"}])

    with pytest.raises(TrainStateError, match="overlap"):
        main(
            [
                "--policy",
                "robosuite.scripts.shakebench_evaluate:load_policy_factory",
                "--dataset",
                str(dataset),
                "--state-ids",
                dev_state["state_id"],
                "--output",
                str(tmp_path / "results.json"),
            ]
        )


def _write_states(path, states):
    path.write_text(json.dumps({"states": states}))
    return path


def test_a_failing_reset_cannot_end_the_batch_or_lose_finished_episodes(tmp_path):
    """Regression: reset used to escape the batch and the results were never written."""

    visited = []

    class EvaluationTask(_STUB_TASK):
        def __init__(self, state, **kwargs):
            super().__init__(terminate_after=1)
            self.state = state

        def reset(self):
            visited.append(self.state["state_id"])
            if self.state["state_id"] == "b":
                raise RuntimeError("second reset failed")
            return super().reset()

        def close(self):
            pass

    class Policy:
        chunk_size = 1

        def predict(self, observation):
            return np.zeros((1, 7))

    states = _write_states(tmp_path / "states.json", [_state(name) for name in "abc"])
    output = tmp_path / "results.json"
    with (
        patch(
            "robosuite.scripts.shakebench_evaluate.load_state_asset",
            return_value={"states": [_state(name) for name in "abc"]},
        ),
        patch("robosuite.scripts.shakebench_evaluate.load_policy_factory", return_value=Policy),
        patch("robosuite.scripts.shakebench_evaluate.ShakeBenchTaskEnv", EvaluationTask),
    ):
        assert main(["--policy", "stub:Policy", "--states", str(states), "--output", str(output)]) == 0

    payload = json.loads(output.read_text())
    assert visited == ["a", "b", "c"]
    assert payload["summary"]["attempted_episodes"] == 3
    assert payload["summary"]["invalid_execution_episodes"] == 1
    assert payload["summary"]["success"] == 2
    assert [episode["termination_cause"] for episode in payload["episodes"]] == [
        "success_latched",
        "invalid_execution",
        "success_latched",
    ]
    assert "second reset failed" in payload["episodes"][1]["invalid_execution_reason"]
    assert payload["episodes"][1]["policy_errors"] == []


def test_states_that_fingerprint_the_same_are_compared_as_the_same_initial_state():
    assert execution_state_fingerprint(_state("a")) == execution_state_fingerprint(_state("b", xy=(-0.10, -0.13)))
    assert execution_state_fingerprint(_state("a")) != execution_state_fingerprint(_state("a", xy=(-0.11, -0.13)))


def test_results_are_never_overwritten(tmp_path):
    existing = tmp_path / "results.json"
    existing.write_text("{}")

    with pytest.raises(FileExistsError):
        main(["--policy", "m:f", "--output", str(existing)])
