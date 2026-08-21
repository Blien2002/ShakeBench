import numpy as np

import shakebench


def _trajectory(state: dict) -> list[dict[str, np.ndarray]]:
    env = shakebench.make(
        "PickPlace", use_camera_obs=False, use_object_obs=True,
        use_vibration_obs=True, use_imu_obs=True, workpiece=state["workpiece"],
    )
    env.set_init_state(state)
    observation, _ = env.reset()
    result = [observation]
    action = np.linspace(-0.5, 0.5, env.action_space.shape[0], dtype=np.float32)
    for _ in range(8):
        observation, *_ = env.step(action)
        result.append(observation)
    return result


def test_same_init_state_has_bitwise_identical_trajectory() -> None:
    suite = shakebench.benchmark.get_benchmark_dict()["shakebench_ladder"]()
    state = suite.get_task_init_states(0)[7]
    first, second = _trajectory(state), _trajectory(state)
    for lhs, rhs in zip(first, second):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            assert np.array_equal(lhs[key], rhs[key]), key
