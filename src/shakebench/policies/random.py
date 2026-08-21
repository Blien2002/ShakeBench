"""Uniform-random lower-bound sentinel."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    requires_privileged: tuple[str, ...] = ()

    def __init__(self, action_dim: int = 8, seed: int = 0) -> None:
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, observation) -> np.ndarray:
        del observation
        return self._rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
