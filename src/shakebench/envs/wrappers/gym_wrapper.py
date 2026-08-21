"""Compatibility wrapper for task backends that already expose Gymnasium."""

import gymnasium as gym


class GymWrapper(gym.Wrapper):
    """Named wrapper matching the robosuite/LIBERO integration surface."""

    pass
