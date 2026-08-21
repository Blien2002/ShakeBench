import shakebench


def _run(control_freq: int) -> tuple[int, int, float]:
    env = shakebench.make(
        "PickPlace", control_freq=control_freq, episode_s=16.0,
        use_camera_obs=False, use_imu_obs=False,
    )
    env.reset(seed=17)
    calls = 0
    while True:
        _, _, _, truncated, info = env.step(env.action_space.sample())
        calls += 1
        if truncated:
            return calls, info["physics_steps"], info["episode_s"]


def test_policy_bandwidth_does_not_change_physical_budget() -> None:
    slow = _run(5)
    fast = _run(200)
    assert fast[0] == 40 * slow[0]
    assert fast[1] == slow[1] == 16_000
    assert slow[2] == fast[2] == 16.0
