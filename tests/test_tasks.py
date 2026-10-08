"""Every task builds, steps under vibration and reports a latched success flag."""

import numpy as np
import pytest

from shakebench.tasks.runtime import make_environment

TASKS = [
    "pick_place",
    "panel_operation",
    "push_t",
    "ring_on_peg",
    "ring_on_rail",
    "stack_blocks",
    "upright",
    "wine_rack",
]


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_task_steps_under_vibration(task, initial_state):
    env, _ = make_environment(initial_state(task), gamma=0.6, horizon=10)
    try:
        rng = np.random.default_rng(0)
        for _ in range(3):
            observation, _, _, _ = env.step(rng.uniform(-0.5, 0.5, env.action_dim))
        for key, value in observation.items():
            array = np.asarray(value)
            if array.dtype.kind in "fc":
                assert np.isfinite(array).all(), key
        assert env.get_metrics()["success"]["passed"] in (True, False)
    finally:
        env.close()
