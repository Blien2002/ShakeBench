import json

import h5py
import numpy as np
import pytest

import shakebench
from shakebench.benchmark.scorecard import (
    aggregate_episodes,
    critical_value,
    paired_bootstrap_difference,
)
from shakebench.envs.wrappers import DataCollectionWrapper
from shakebench.policies import (
    ClassicalPolicy,
    OracleFullPolicy,
    OraclePhasePolicy,
    OracleReactivePolicy,
    policy_env_kwargs,
)


def test_oracle_privilege_tiers_are_distinct_and_automatic() -> None:
    full = OracleFullPolicy()
    phase = OraclePhasePolicy()
    reactive = OracleReactivePolicy()
    classical = ClassicalPolicy()
    assert full.requires_privileged == ("object", "vibration")
    assert phase.requires_privileged == ("phase",)
    assert reactive.requires_privileged == ("object", "instantaneous_load")
    assert classical.requires_privileged == ()
    assert policy_env_kwargs(phase)["use_phase_obs"] is True
    assert policy_env_kwargs(phase)["use_object_obs"] is False
    with pytest.raises(ValueError, match="1000 Hz"):
        OracleFullPolicy(control_freq=200)

    classical_observation = {
        "robot0_eef_pos": np.array([0.45, 0.0, 0.45], np.float32),
        "robot0_wrist_force": np.zeros(3, np.float32),
        "deck_imu": np.zeros(6, np.float32),
    }
    assert classical.act(classical_observation).shape == (13,)


def test_phase_oracle_reconstructs_truth_without_qdd_observation() -> None:
    policy = OraclePhasePolicy(control_freq=10)
    env = shakebench.make(
        "PickPlace",
        controller_configs=shakebench.load_controller_config("VARIABLE_IMPEDANCE"),
        control_freq=10,
        use_camera_obs=False,
        **policy_env_kwargs(policy),
    )
    observation, _ = env.reset(seed=17)
    assert "vibration_qdd" not in observation
    observation, *_ = env.step(policy.act(observation))
    reconstructed = policy._analytic_qdd(observation)
    truth = env.privileged_observation()["privileged_vibration_qdd"]
    assert reconstructed == pytest.approx(truth, abs=2e-5)


def test_dataset_schema_and_privileged_prefix_isolation(tmp_path) -> None:
    policy = OraclePhasePolicy(control_freq=5, episode_s=0.2)
    env = shakebench.make(
        "PickPlace",
        controller_configs=shakebench.load_controller_config("VARIABLE_IMPEDANCE"),
        intra_step_mode="feedforward",
        control_freq=5,
        episode_s=0.2,
        vibration_mode="off",
        use_camera_obs=False,
        **policy_env_kwargs(policy),
    )
    path = tmp_path / "demo_static.hdf5"
    wrapped = DataCollectionWrapper(env, output_path=path)
    observation, _ = wrapped.reset(seed=17)
    while True:
        observation, _, terminated, truncated, _ = wrapped.step(policy.act(observation))
        if terminated or truncated:
            break
    assert wrapped.flush() == path
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        assert data.attrs["total"] == 1
        env_args = json.loads(data.attrs["env_args"])
        assert env_args["env_kwargs"]["level_scale"] is None
        demo = data["demo_0"]
        assert demo.attrs["num_samples"] == 1
        privileged = sorted(
            key for key in demo["obs"] if key.startswith("privileged_")
        )
        assert privileged == sorted(
            (
                "privileged_object_pose",
                "privileged_vibration_qdd",
                "privileged_phase",
                "privileged_object_mu",
                "privileged_object_mass",
            )
        )
        assert "vibration_phase" not in demo["obs"]
        assert "object_pos" not in demo["obs"]


def test_scorecard_validity_and_critical_interpolation() -> None:
    assert critical_value([(0.15, 1.0), (0.5, 0.75), (0.95, 0.25)]) == pytest.approx(0.725)
    base = {
        "scoreable": True,
        "support_geometry_valid": True,
        "grasp_assist_used": False,
        "max_penetration_mm": 0.01,
        "gamma_realized": 0.5,
        "gamma_expected": 0.5,
        "ee_tracking_error_rms_m": 0.002,
        "grip_margin_min": 1.2,
        "grip_excess": 1.1,
        "max_grasp_slip_m": 0.001,
        "gamma_target": 0.5,
        "control_freq": 10,
    }
    episodes = [{**base, "success": True}, {**base, "success": False}]
    result = aggregate_episodes(episodes, penetration_limit_mm=0.1)
    assert result["n_valid"] == 2
    assert result["n_voided"] == 0
    assert result["metrics"]["SR"]["mean"] == 0.5
    phase = [
        {**episodes[0], "task_name": "a", "init_state_index": 0, "success": True},
        {**episodes[0], "task_name": "a", "init_state_index": 1, "success": True},
    ]
    reactive = [
        {**episodes[0], "task_name": "a", "init_state_index": 0, "success": False},
        {**episodes[0], "task_name": "a", "init_state_index": 1, "success": False},
    ]
    difference = paired_bootstrap_difference(phase, reactive, "success")
    assert difference == {"mean": 1.0, "ci95": [1.0, 1.0], "n_pairs": 2}
