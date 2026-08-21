"""Controller contracts for slow policy actions and fast inner-loop execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ControllerOutput:
    command: np.ndarray
    clipped: bool = False


class Controller:
    """Base class shared by all public ShakeBench controllers."""

    action_dim = 0

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.previous_action = self.last_action.copy()

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.previous_action.fill(0.0)

    def normalize_action(self, action: np.ndarray) -> ControllerOutput:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {array.shape}")
        clipped = np.clip(array, -1.0, 1.0)
        return ControllerOutput(clipped, not np.array_equal(array, clipped))

    def pre_action(
        self,
        action: np.ndarray,
        policy_step: bool,
        *,
        alpha: float = 1.0,
        phase: float = 0.0,
    ) -> ControllerOutput:
        del phase
        result = self.normalize_action(action)
        if policy_step:
            self.previous_action = self.last_action.copy()
            self.last_action = result.command.copy()
        command = (1.0 - alpha) * self.previous_action + alpha * self.last_action
        return ControllerOutput(command.astype(np.float32), result.clipped)
