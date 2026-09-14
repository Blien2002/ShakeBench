"""Model-agnostic rollout contract: typed policy failures and action evidence."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite.utils.shakebench_outcomes import validate_outcome
from robosuite.utils.shakebench_rollout import (
    PolicyTimeoutError,
    action_evidence,
    episode_outcome,
    rollout_policy,
)


class _StubTask:
    """Duck-typed task: only reset/step and the identity hooks are used."""

    def __init__(self, *, step_error=None, terminate_after=2):
        self.state = {"state_id": "stub-state"}
        self.gamma, self.horizon = 0.15, 3
        self.step_error, self.terminate_after = step_error, terminate_after
        self.steps, self.done = 0, True
        self.actions = []

    def reset(self):
        self.steps, self.done, self.actions = 0, False, []
        return {"task": "stub"}, {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float).copy())
        if self.step_error is not None:
            raise self.step_error
        self.steps += 1
        terminated = self.steps >= self.terminate_after
        return (
            {"task": "stub"},
            1.0 if terminated else 0.0,
            terminated,
            False,
            {
                "termination_cause": "success_latched" if terminated else None,
                "success": terminated,
            },
        )

    def observation_identity(self):
        return {"source": "stub"}

    def get_task_context(self):
        return {"task_id": "stub"}


def _policy(predict, *, chunk_size=2, deadline_s=None, reset=None):
    policy = SimpleNamespace(chunk_size=chunk_size, predict=predict, deadline_s=deadline_s)
    if reset is not None:
        policy.reset = reset
    return policy


def test_policy_output_violation_ends_the_episode_with_a_record():
    record = rollout_policy(_StubTask(), _policy(lambda _: np.full((2, 7), 2.0)), action_horizon=1)

    assert record["termination_cause"] == "policy_error"
    assert record["episode_validity"] == "valid" and record["score_outcome"] == "unsuccessful"
    assert [error["error_type"] for error in record["policy_errors"]] == ["policy_output_violation"]
    assert record["executed_action_count"] == 0 and record["policy_calls"] == 0
    validate_outcome(
        episode_validity=record["episode_validity"],
        score_outcome=record["score_outcome"],
        termination_cause=record["termination_cause"],
    )


def test_policy_exception_and_timeout_are_distinguished():
    def explode(_):
        raise RuntimeError("adapter exploded")

    def stall(_):
        time.sleep(0.05)
        return np.zeros((2, 7))

    def deadline(_):
        raise PolicyTimeoutError("server did not answer within 1s")

    exception_record = rollout_policy(_StubTask(), _policy(explode))
    timeout_record = rollout_policy(_StubTask(), _policy(deadline))
    late_record = rollout_policy(_StubTask(), _policy(stall, deadline_s=0.01), inference_timeout_s=0.01)

    assert exception_record["policy_errors"][0]["error_type"] == "policy_exception"
    assert "RuntimeError" in exception_record["policy_errors"][0]["message"]
    assert timeout_record["policy_errors"][0]["error_type"] == "policy_timeout"
    assert late_record["policy_errors"][0]["error_type"] == "policy_timeout"
    assert "past the 0.01s deadline it declares" in late_record["policy_errors"][0]["message"]
    assert late_record["episode_validity"] == "valid"


def test_inference_timeout_needs_a_deadline_the_policy_can_enforce():
    """A blocking predict() cannot be interrupted from outside, so say so instead of pretending."""

    stalling = _policy(lambda _: np.zeros((1, 7)))
    with pytest.raises(ValueError, match="enforceable deadline"):
        rollout_policy(_StubTask(), stalling, inference_timeout_s=0.01)
    with pytest.raises(ValueError, match="exceeds the configured inference_timeout_s"):
        rollout_policy(_StubTask(), _policy(lambda _: np.zeros((1, 7)), deadline_s=5.0), inference_timeout_s=0.01)
    with pytest.raises(ValueError, match="positive number of seconds"):
        rollout_policy(_StubTask(), _policy(lambda _: np.zeros((1, 7)), deadline_s=0.0))


def test_each_episode_starts_with_a_policy_reset():
    """Regression: a stateful policy used to carry its state across episodes."""

    class StatefulPolicy:
        chunk_size = 1

        def __init__(self):
            self.resets, self.calls = 0, 0

        def reset(self):
            self.resets, self.calls = self.resets + 1, 0

        def predict(self, observation):
            self.calls += 1
            return np.full((1, 7), self.calls / 10.0)

    policy = StatefulPolicy()
    first_task, second_task = _StubTask(terminate_after=1), _StubTask(terminate_after=1)

    rollout_policy(first_task, policy)
    rollout_policy(second_task, policy)

    assert policy.resets == 2
    assert first_task.actions[0][0] == pytest.approx(0.1)
    assert second_task.actions[0][0] == pytest.approx(0.1)


def test_episode_setup_failure_keeps_its_place_in_the_batch():
    """Regression: a reset exception used to escape instead of recording invalid_execution."""

    class BrokenReset(_StubTask):
        def reset(self):
            raise RuntimeError("reset initialization failed")

    record = rollout_policy(BrokenReset(), _policy(lambda _: np.zeros((1, 7))))

    assert record["termination_cause"] == "invalid_execution"
    assert record["episode_validity"] == "invalid" and record["score_outcome"] is None
    assert "reset initialization failed" in record["invalid_execution_reason"]
    assert record["policy_calls"] == 0 and record["executed_action_count"] == 0
    assert record["policy_errors"] == [] and record["steps"] == 0
    validate_outcome(
        episode_validity=record["episode_validity"],
        score_outcome=record["score_outcome"],
        termination_cause=record["termination_cause"],
    )


def test_environment_failure_is_recorded_as_invalid_execution():
    record = rollout_policy(_StubTask(step_error=RuntimeError("sim died")), _policy(lambda _: np.zeros((1, 7))))

    assert record["termination_cause"] == "invalid_execution"
    assert record["episode_validity"] == "invalid" and record["score_outcome"] is None
    assert "sim died" in record["invalid_execution_reason"]
    assert record["policy_errors"] == []


def test_successful_chunk_executes_only_the_requested_horizon():
    record = rollout_policy(
        _StubTask(), _policy(lambda _: np.arange(14, dtype=float).reshape(2, 7) / 100.0), action_horizon=2
    )

    assert record["termination_cause"] == "success_latched" and record["success"] is True
    assert record["steps"] == 2 and record["policy_calls"] == 1
    assert record["executed_action_count"] == 2
    assert record["action_horizon"] == 2 and record["timing"].startswith("synchronous")


def test_action_evidence_digest_changes_with_the_commanded_motion():
    first = action_evidence([np.zeros(7), np.ones(7) * 0.5])
    second = action_evidence([np.zeros(7), np.ones(7) * 0.5])
    third = action_evidence([np.zeros(7), np.ones(7) * 0.25])

    assert first == second and first != third
    assert first["executed_action_count"] == 2 and len(first["executed_actions_sha256"]) == 64


def test_chunk_horizon_cannot_exceed_the_policy_chunk():
    with pytest.raises(ValueError, match="chunk size"):
        rollout_policy(_StubTask(), _policy(lambda _: np.zeros((2, 7))), action_horizon=3)
    with pytest.raises(ValueError, match="positive number of seconds"):
        rollout_policy(_StubTask(), _policy(lambda _: np.zeros((2, 7))), inference_timeout_s=0.0)


def test_episode_outcome_matches_the_frozen_contract():
    pairs = {
        "success_latched": ("valid", "success"),
        "horizon_exhausted": ("valid", "unsuccessful"),
        "task_rule_violation": ("valid", "unsuccessful"),
        "policy_error": ("valid", "unsuccessful"),
        "invalid_execution": ("invalid", None),
    }
    for cause, expected in pairs.items():
        assert episode_outcome(cause) == expected
        validate_outcome(episode_validity=expected[0], score_outcome=expected[1], termination_cause=cause)


def test_camera_identity_records_the_compiled_pose_not_only_a_name():
    from robosuite.utils.shakebench_rollout import ShakeBenchCameraObservation

    observation = object.__new__(ShakeBenchCameraObservation)
    positions = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    observation.env = SimpleNamespace(
        sim=SimpleNamespace(
            model=SimpleNamespace(
                camera_name2id=lambda name: {"observation.images.wrist": 0}[name],
                cam_pos=positions,
                cam_quat=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
            )
        )
    )
    observation.main_camera = "task_close"
    observation.cameras = {"observation.images.main": "task_close"}

    compiled = observation._camera_identity("observation.images.wrist")
    preset = observation._camera_identity(
        SimpleNamespace(lookat=np.array([1.0, 2.0, 3.0]), distance=0.9, azimuth=-180.0, elevation=-38.9)
    )

    assert compiled["kind"] == "compiled_camera" and compiled["name"] == "observation.images.wrist"
    assert compiled["position_m"] == [0.1, 0.2, 0.3] and compiled["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert preset["kind"] == "preset_camera" and preset["name"] == "task_close"
    assert preset["lookat"] == [1.0, 2.0, 3.0] and preset["distance"] == 0.9


def test_task_environment_requires_an_authenticated_state_identity():
    from robosuite.utils.shakebench_rollout import ShakeBenchTaskEnv

    with pytest.raises(ValueError, match="state_id"):
        ShakeBenchTaskEnv({"object_xy_m": [0.0, 0.0]})
    with pytest.raises(ValueError):
        ShakeBenchTaskEnv({"state_id": "x", "object_xy_m": [0.0]})
