"""Hold-still policy for smoke-testing evaluation runs: ``--policy tools.vibration_scenarios.zero_policy:make_policy``."""

import numpy as np


class HoldPolicy:
    chunk_size = 1

    def predict(self, observation):
        action = np.zeros((1, 7), dtype=np.float32)
        action[0, -1] = -1.0  # keep the gripper open
        return action


def make_policy():
    return HoldPolicy()
