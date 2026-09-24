"""Shared CPU environment construction for registered 20 Hz Panda tasks."""

from robosuite.controllers import load_composite_controller_config
from shakebench.utils.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.utils.task_registry import get_task_definition, prepare_task_state, task_type


def make_environment(state, *, gamma, horizon, mode="multisine_v1", physics_profile="official", free_ring_peg=False):
    """Build/reset a registered task and return (environment, excitation program).

    Task adapters own task arguments; the runtime owns controller, timing,
    excitation and sensor seeds. Conflicting arguments fail before construction.
    """
    state = prepare_task_state(state)
    definition = get_task_definition(state)
    seed = int(state.get("excitation_seed", state.get("seed", 0)))
    if free_ring_peg and task_type(state) != "ring_on_peg":
        raise ValueError("free_ring_peg requires a ring_on_peg state")
    task_kwargs = definition.env_kwargs(state)
    if free_ring_peg:
        task_kwargs["free_peg"] = True
    env = definition.env_factory(
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile=physics_profile,
        geometry_profile=DEFAULT_GEOMETRY_PROFILE,
        imu_mode="canonical_noisy_v1",
        vibration={"mode": mode, "gamma": gamma, "seed": seed, "t0_s": float(state.get("t0_s", 0.0))},
        **task_kwargs,
        imu_seed=int(state.get("imu_seed", seed)),
        horizon=horizon,
        ignore_done=True,
        seed=seed,
        hard_reset=False,
    )
    try:
        env.reset()
        return env, env.deck_driver.trajectory
    except BaseException:
        env.close()
        raise
