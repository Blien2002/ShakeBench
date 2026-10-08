"""ShakeBenchEnv: owned timestep, physics-step hooks and shadowless hidden sites."""

import numpy as np
import pytest

from shakebench.environments.base import validate_model_timestep


@pytest.mark.parametrize("value", [None, 0.002, 2e-4, 1])
def test_validate_model_timestep_accepts(value):
    assert validate_model_timestep(value) == (None if value is None else float(value))


@pytest.mark.parametrize("value", [0, -1e-3, float("nan"), float("inf"), True, "fast"])
def test_validate_model_timestep_rejects(value):
    with pytest.raises(ValueError):
        validate_model_timestep(value)


@pytest.fixture(scope="module")
def push_t(initial_state):
    from shakebench.tasks.runtime import make_environment

    env, _ = make_environment(initial_state("push_t"), gamma=0.6, horizon=50)
    yield env
    env.close()


def test_environment_owns_its_timestep(push_t):
    assert push_t.sim.model.opt.timestep == pytest.approx(2e-4)
    assert push_t.model_timestep == pytest.approx(2e-4)
    assert push_t._control_steps * push_t.model_timestep == pytest.approx(push_t.control_timestep)


def test_hooks_run_before_every_physics_step(push_t):
    seen = []

    def hook(physics_time_s, policy_step):
        seen.append((physics_time_s, policy_step))

    push_t.add_pre_physics_step_hook(hook)
    try:
        push_t.step(np.zeros(push_t.action_dim))
    finally:
        push_t._pre_physics_step_hooks.remove(hook)
    assert len(seen) == push_t._control_steps
    assert seen[0][1] is True and not any(policy_step for _, policy_step in seen[1:])
    times = [time_s for time_s, _ in seen]
    assert np.all(np.diff(times) > 0)


def test_hidden_sites_use_zero_alpha(push_t):
    managed = push_t._visualization_site_ids()
    assert (push_t.sim.model.site_rgba[:, 3] >= 0).all()
    push_t.visualize({name: True for name in push_t._visualizations})
    shown = push_t.sim.model.site_rgba[managed, 3].copy()
    push_t.visualize({name: False for name in push_t._visualizations})
    hidden = push_t.sim.model.site_rgba[managed, 3]
    assert (shown > 0).any()
    assert (hidden == 0).all()
    assert (push_t.sim.model.site_rgba[:, 3] >= 0).all()
