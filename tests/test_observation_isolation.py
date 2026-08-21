import shakebench


PRIVILEGED = {
    "object_pos", "object_quat", "target_pos", "target_quat", "penetration_mm",
    "mount_delta_z", "object_mass", "object_mu", "object_com_offset",
    "vibration_q", "vibration_qd", "vibration_qdd", "vibration_level_scale", "vibration_t0",
}


def test_default_policy_observation_contains_no_truth_leak() -> None:
    env = shakebench.make(
        "PickPlace", use_camera_obs=False, use_object_obs=False,
        use_vibration_obs=False, use_imu_obs=True,
    )
    observation, _ = env.reset(seed=17)
    assert PRIVILEGED.isdisjoint(observation)
    assert not any(key.endswith("_w") for key in observation)
    assert "deck_imu" in observation


def test_privileged_groups_are_explicit() -> None:
    env = shakebench.make(
        "PickPlace", use_camera_obs=False, use_object_obs=True, use_vibration_obs=True
    )
    observation, info = env.reset(seed=17)
    assert PRIVILEGED <= set(observation)
    assert info["privileged_observations"] == ("object", "vibration")
