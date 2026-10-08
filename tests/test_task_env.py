"""The policy-facing task environment and rollout boundary used by evaluation."""

import json

import numpy as np
import pytest

from shakebench.rollout.outcomes import TERMINATION_CAUSES
from shakebench.rollout.task_env import ShakeBenchTaskEnv, rollout_policy


class ZeroPolicy:
    chunk_size = 2

    def predict(self, observation):
        return np.zeros((self.chunk_size, 7), dtype=np.float32)


@pytest.fixture(scope="module")
def ring_task():
    from shakebench.environments.ring_on_peg import default_state

    task = ShakeBenchTaskEnv(default_state(), gamma=0.5, horizon=6, observation_source="contract")
    yield task
    task.close()


@pytest.mark.slow
def test_readme_quick_start(ring_task):
    observation, info = ring_task.reset()
    for _ in range(3):
        observation, reward, terminated, truncated, info = ring_task.step(np.zeros(7))
    assert observation
    assert terminated in (True, False) and truncated in (True, False)


@pytest.mark.slow
def test_rollout_reports_a_termination_cause(ring_task):
    result = json.dumps(rollout_policy(ring_task, ZeroPolicy()), default=str)
    assert any(cause in result for cause in TERMINATION_CAUSES)
