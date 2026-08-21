from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import shakebench


def test_gymnasium_checker() -> None:
    env = shakebench.make("PickPlace", use_camera_obs=False)
    check_env(env, skip_render_check=True)


def test_action_clip_is_disclosed() -> None:
    env = shakebench.make("PickPlace", use_camera_obs=False)
    env.reset(seed=17)
    _, _, _, _, info = env.step(np.full(env.action_space.shape, 2.0, np.float32))
    assert info["action_clipped"] is True


def test_horizon_is_derived_and_rate_must_divide_physics() -> None:
    env = shakebench.make("PickPlace", control_freq=5, episode_s=16.0, horizon=80)
    assert env.horizon == 80
    with pytest.raises(ValueError, match="horizon is derived"):
        shakebench.make("PickPlace", control_freq=5, episode_s=16.0, horizon=81)
    with pytest.raises(ValueError, match="integer multiple"):
        shakebench.make("PickPlace", control_freq=30)
