"""Privilege-boundary tests for the Phase 05 State observation API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import robosuite
from robosuite.utils.shakebench_privilege import (
    PrivilegedRecorder,
    ShakeBenchPrivilegeError,
    assert_policy_observation_is_clean,
    audit_policy_observation,
    make_privileged_recorder,
    verify_phase05_observation_artifact,
)


def _make_env(**kwargs):
    options = {
        "robots": "Panda",
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": False,
        "control_freq": 20,
        "model_timestep": 0.0002,
        "horizon": 2,
        "seed": 29,
        "observation_tier": "V1",
        "imu_mode": "ideal_smoke",
    }
    options.update(kwargs)
    return robosuite.make("VibrationPickPlaceCan", **options)


def _write_gymnasium_test_package(package_root):
    package = package_root / "gymnasium"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            __version__ = "0.29.0"


            class Env:
                metadata = None


            class Box:
                def __init__(self, low, high, shape=None, dtype=np.float32):
                    low_array = np.asarray(low)
                    high_array = np.asarray(high)
                    if shape is None:
                        shape = np.broadcast(low_array, high_array).shape
                    self.shape = tuple(shape)
                    self.dtype = np.dtype(dtype)
                    self.low = np.broadcast_to(
                        np.asarray(low, dtype=self.dtype), self.shape
                    ).copy()
                    self.high = np.broadcast_to(
                        np.asarray(high, dtype=self.dtype), self.shape
                    ).copy()

                def contains(self, value):
                    array = np.asarray(value)
                    if array.shape != self.shape:
                        return False
                    try:
                        array = array.astype(self.dtype, copy=False)
                    except (TypeError, ValueError):
                        return False
                    return bool(np.all(array >= self.low) and np.all(array <= self.high))


            class MultiBinary:
                def __init__(self, n):
                    self.shape = (n,) if isinstance(n, int) else tuple(n)
                    self.dtype = np.dtype(np.int8)

                def contains(self, value):
                    array = np.asarray(value)
                    return bool(
                        array.shape == self.shape
                        and np.all(array >= 0)
                        and np.all(array <= 1)
                    )


            class Text:
                def __init__(self, max_length, min_length=1):
                    self.max_length = int(max_length)
                    self.min_length = int(min_length)

                def contains(self, value):
                    return isinstance(value, str) and self.min_length <= len(value) <= self.max_length


            class Dict:
                def __init__(self, spaces):
                    self.spaces = dict(spaces)

                def contains(self, value):
                    return (
                        isinstance(value, dict)
                        and set(value) == set(self.spaces)
                        and all(self.spaces[key].contains(value[key]) for key in self.spaces)
                    )


            class _Spaces:
                Box = Box
                Dict = Dict
                MultiBinary = MultiBinary
                Text = Text


            spaces = _Spaces()
            """
        ),
        encoding="utf-8",
    )


def _write_old_gym_test_package(package_root):
    package = package_root / "gym"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '__version__ = "0.25.2"\nclass Env: pass\nspaces = object()\n',
        encoding="utf-8",
    )


def _run_gym_wrapper_subprocess(tmp_path, code, *, gymnasium=False, old_gym=False):
    package_root = None
    if gymnasium or old_gym:
        package_root = tmp_path / ("gymnasium_shim" if gymnasium else "old_gym_shim")
        if gymnasium:
            _write_gymnasium_test_package(package_root)
        else:
            _write_old_gym_test_package(package_root)

    environment = os.environ.copy()
    if package_root is None:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = os.pathsep.join((str(package_root), str(Path(__file__).resolve().parents[1])))
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_recorder_is_namespace_scoped_and_defensively_copied():
    received = []
    recorder = PrivilegedRecorder(callback=lambda payload: received.append(payload))
    mutable = np.array([1.0, 2.0])
    record = recorder.record({"privileged_signal": mutable})
    mutable[0] = 99.0
    record["privileged_signal"][1] = 99.0
    assert recorder.latest["privileged_signal"].tolist() == [1.0, 2.0]
    assert received[0]["privileged_signal"].tolist() == [1.0, 2.0]
    assert recorder.to_dict()["namespace"] == "privileged_"
    assert len(recorder) == 1

    with pytest.raises(ShakeBenchPrivilegeError, match="privileged_"):
        recorder.record({"clean_signal": np.zeros(1)})
    with pytest.raises(ShakeBenchPrivilegeError, match="not a flag"):
        make_privileged_recorder(True)


def test_recorder_rejects_uncopyable_values_and_separates_nested_copies():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("deliberately uncopyable")

    recorder = PrivilegedRecorder()
    with pytest.raises(ShakeBenchPrivilegeError, match="defensively copy"):
        recorder.record({"privileged_bad": Uncopyable()})

    source = {"privileged_nested": {"array": np.array([1.0, 2.0]), "items": [{"x": 3.0}]}}
    returned = recorder.record(source)
    latest = recorder.latest
    history = recorder.records[0]
    callback = []
    separate = PrivilegedRecorder(callback=lambda value: callback.append(value))
    separate_returned = separate.record(source)
    source["privileged_nested"]["array"][0] = 99.0
    returned["privileged_nested"]["array"][1] = 98.0
    latest["privileged_nested"]["items"][0]["x"] = 97.0
    history["privileged_nested"]["array"][0] = 96.0
    separate_returned["privileged_nested"]["array"][0] = 95.0
    callback[0]["privileged_nested"]["array"][0] = 94.0
    assert recorder.latest["privileged_nested"]["array"].tolist() == [1.0, 2.0]
    assert recorder.latest["privileged_nested"]["items"][0]["x"] == 3.0
    assert separate.latest["privileged_nested"]["array"].tolist() == [1.0, 2.0]


def test_policy_audit_rejects_only_privileged_namespace_keys():
    clean = audit_policy_observation({"deck_imu_window": np.zeros((10, 6))})
    assert clean.passed
    assert_policy_observation_is_clean({"ordinary": np.zeros(1)})
    leaked = audit_policy_observation({"ordinary": 1, "privileged_bias": np.zeros(6)})
    assert not leaked.passed
    assert leaked.leaked_keys == ("privileged_bias",)
    with pytest.raises(ShakeBenchPrivilegeError, match="privileged_bias"):
        assert_policy_observation_is_clean({"privileged_bias": np.zeros(6)})


def test_environment_recorder_receives_truth_but_policy_observation_does_not():
    recorder = PrivilegedRecorder()
    env = _make_env(privileged_recorder=recorder)
    try:
        observation = env.reset()
        assert_policy_observation_is_clean(observation)
        observation, _, _, _ = env.step(np.zeros(env.action_dim))
        assert_policy_observation_is_clean(observation)
        assert len(recorder) == 1
        latest = recorder.latest
        assert latest is not None
        assert all(key.startswith("privileged_") for key in latest)
        assert "privileged_provider" in latest
        assert "privileged_contacts" in latest
        assert "privileged_parameters" in latest
        assert "privileged_actions" in latest
        assert "privileged_can_pose_world" in latest
        assert "privileged_bias" not in observation
        assert "privileged_clean_measurement" not in observation
    finally:
        env.close()


def test_gym_wrapper_cannot_leak_recorder_namespace(tmp_path):
    _run_gym_wrapper_subprocess(
        tmp_path,
        """
        import numpy as np
        import robosuite
        from robosuite.utils.shakebench_privilege import PrivilegedRecorder
        from robosuite.wrappers import GymWrapper

        received = []
        recorder = PrivilegedRecorder(callback=lambda payload: received.append(payload))
        env = robosuite.make(
            "VibrationPickPlaceCan",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq=20,
            model_timestep=0.0002,
            horizon=2,
            seed=29,
            observation_tier="V1",
            imu_mode="ideal_smoke",
            privileged_recorder=recorder,
        )
        wrapper = GymWrapper(env, keys=["deck_imu_window", "deck_imu_dt_s"], flatten_obs=False)
        try:
            observation, _ = wrapper.reset()
            assert set(observation) == {"deck_imu_window", "deck_imu_dt_s"}
            assert wrapper.observation_space.contains(observation)
            assert not any(key.startswith("privileged_") for key in observation)
            action = np.zeros(wrapper.action_space.shape, dtype=wrapper.action_space.dtype)
            assert wrapper.action_space.contains(action)
            observation, _, _, _, _ = wrapper.step(action)
            assert wrapper.observation_space.contains(observation)
            assert not any(key.startswith("privileged_") for key in observation)
            assert received
            assert all(key.startswith("privileged_") for key in received[-1])
        finally:
            wrapper.close()
        """,
        gymnasium=True,
    )


def test_gym_wrapper_fail_closes_privileged_and_non_policy_keys(tmp_path):
    _run_gym_wrapper_subprocess(
        tmp_path,
        """
        import robosuite
        from robosuite.wrappers import GymWrapper

        for bad_key in (
            "privileged_bias",
            "privileged_clean_measurement",
            "robot0_proprio-state",
            "object-state",
            "debug",
            "renderer",
        ):
            env = robosuite.make(
                "VibrationPickPlaceCan",
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=False,
                control_freq=20,
                model_timestep=0.0002,
                horizon=2,
                seed=29,
                observation_tier="V1",
                imu_mode="ideal_smoke",
            )
            try:
                try:
                    GymWrapper(env, keys=[bad_key], flatten_obs=False)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"GymWrapper accepted forbidden key: {bad_key}")
            finally:
                env.close()
        """,
        gymnasium=True,
    )
    _run_gym_wrapper_subprocess(
        tmp_path,
        """
        try:
            from robosuite.wrappers.gym_wrapper import GymWrapper
        except ImportError as exc:
            assert "requires gymnasium or gym>=0.26" in str(exc)
        else:
            raise AssertionError("GymWrapper imported without a Gym dependency")
        """,
    )
    _run_gym_wrapper_subprocess(
        tmp_path,
        """
        try:
            from robosuite.wrappers.gym_wrapper import GymWrapper
        except ImportError as exc:
            assert "gym>=0.26" in str(exc)
            assert "found gym==0.25.2" in str(exc)
        else:
            raise AssertionError("GymWrapper accepted gym<0.26")
        """,
        old_gym=True,
    )


def test_gym_wrapper_default_state_keys_match_each_explicit_tier(tmp_path):
    source = Path(__file__).resolve().parents[1] / "robosuite/wrappers/gym_wrapper.py"
    production_source = source.read_text(encoding="utf-8")
    assert "_Fallback" not in production_source

    _run_gym_wrapper_subprocess(
        tmp_path,
        """
        import numpy as np
        import robosuite
        from robosuite.wrappers import GymWrapper

        for tier in ("V0", "V3"):
            env = robosuite.make(
                "VibrationPickPlaceCan",
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=False,
                control_freq=20,
                model_timestep=0.0002,
                horizon=2,
                seed=29,
                observation_tier=tier,
                imu_mode="ideal_smoke",
            )
            wrapper = GymWrapper(env, flatten_obs=False)
            try:
                observation, _ = wrapper.reset()
                assert set(observation) == set(env.policy_observation_keys)
                assert wrapper.observation_space.contains(observation)
                assert not any(key.startswith("privileged_") for key in observation)
                action = np.zeros(wrapper.action_space.shape, dtype=wrapper.action_space.dtype)
                assert wrapper.action_space.contains(action)
                observation, _, _, _, _ = wrapper.step(action)
                assert set(observation) == set(env.policy_observation_keys)
                assert wrapper.observation_space.contains(observation)
                if tier == "V3":
                    assert observation["ramp_type"] == "quintic_smoothstep"
                    assert observation["program_frame"] == "deck"
            finally:
                wrapper.close()
        """,
        gymnasium=True,
    )


def test_state_tier_rejects_vision_and_unknown_combinations():
    with pytest.raises(ValueError, match="V0, V1, V2, V3"):
        _make_env(observation_tier="V9")
    with pytest.raises(ValueError, match="use_camera_obs=False"):
        _make_env(use_camera_obs=True)
    with pytest.raises(ValueError, match="control_freq=20"):
        _make_env(control_freq=10)


def test_phase05_observation_artifact_is_read_only_and_hash_protected(tmp_path):
    artifact = Path(__file__).with_name("shakebench_phase_05_observation.json")
    summary = verify_phase05_observation_artifact(artifact)
    assert summary["passed"]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["timeline"]["first_step_acquisition_s"][0] = 0.006
    mutated = tmp_path / artifact.name
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    mutation_summary = verify_phase05_observation_artifact(mutated)
    assert not mutation_summary["passed"]
    assert not mutation_summary["integrity_valid"]
