"""Round-7 data-collection extension point."""

from __future__ import annotations

import gymnasium as gym


class DataCollectionWrapper(gym.Wrapper):
    """Interface-compatible skeleton; persistence is intentionally deferred."""

    def __init__(self, env: gym.Env, directory: str | None = None) -> None:
        super().__init__(env)
        self.directory = directory

    def flush(self) -> None:
        """Flush a future trajectory buffer."""
