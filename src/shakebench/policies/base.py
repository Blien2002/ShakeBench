"""Policy protocol used by evaluation and future oracle baselines."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    requires_privileged: tuple[str, ...]

    def reset(self) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray: ...
