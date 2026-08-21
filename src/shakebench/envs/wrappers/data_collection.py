"""Robomimic-compatible HDF5 trajectory collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import h5py
import numpy as np


_POLICY_PRIVILEGED_KEYS = {
    "object_pos",
    "object_quat",
    "target_pos",
    "target_quat",
    "penetration_mm",
    "mount_delta_z",
    "object_mass",
    "object_mu",
    "object_com_offset",
    "vibration_q",
    "vibration_qd",
    "vibration_qdd",
    "vibration_level_scale",
    "vibration_t0",
    "vibration_omega",
    "vibration_phase",
    "vibration_line_mask",
    "vibration_time",
    "vibration_ramp_s",
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


@dataclass
class _Episode:
    init_state: dict[str, Any]
    observations: dict[str, list[np.ndarray]] = field(default_factory=dict)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.actions)


class DataCollectionWrapper(gym.Wrapper):
    """Buffer trajectories and atomically write the Round-7 HDF5 schema.

    Truth labels come from ``env.unwrapped.privileged_observation()`` rather
    than the policy observation. This permits ``oracle_phase`` collection
    while preserving the policy's input isolation.
    """

    def __init__(
        self,
        env: gym.Env,
        directory: str | Path | None = None,
        *,
        output_path: str | Path | None = None,
        env_args: Mapping[str, Any] | None = None,
        validation_fraction: float = 0.1,
    ) -> None:
        super().__init__(env)
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if output_path is None:
            base = Path(directory or ".")
            output_path = base if base.suffix == ".hdf5" else base / "demo.hdf5"
        self.output_path = Path(output_path)
        self.directory = str(self.output_path.parent)
        self.validation_fraction = float(validation_fraction)
        self.env_args = dict(env_args or self._infer_env_args())
        self._episodes: list[_Episode] = []
        self._current: _Episode | None = None
        self._last_observation: dict[str, np.ndarray] | None = None

    def _infer_env_args(self) -> dict[str, Any]:
        base = self.env.unwrapped
        vibration = getattr(base, "_vibration_cfg", None)
        level = None if vibration is None or vibration.mode == "off" else vibration.level_scale
        return {
            "env_name": getattr(base, "env_name", type(base).__name__),
            "backend": getattr(base, "backend_name", type(base).__name__),
            "scoreable": bool(getattr(base, "scoreable", False)),
            "env_kwargs": {
                "gamma": None if vibration is None or vibration.mode == "off" else vibration.gamma,
                "frequency_scale": getattr(vibration, "frequency_scale", None),
                "control_freq": getattr(base, "control_freq", None),
                "level_scale": level,
                "physics_profile": (
                    "official" if getattr(base, "physics_hz", 0) == 1000 else "training"
                ),
                "controller_config": getattr(base, "controller_config", None),
            },
        }

    def _init_state(self) -> dict[str, Any]:
        base = self.env.unwrapped
        state = getattr(base, "_init_state", None)
        if state is not None:
            return dict(state)
        vibration = getattr(base, "_vibration_cfg", None)
        return {
            "object_placement": np.stack(
                (
                    np.concatenate((base._object_pos, base._object_quat)),
                    np.concatenate((base._target_pos, base._target_quat)),
                )
            ),
            "workpiece": base.cfg.assets.workpiece,
            "t0": getattr(vibration, "t0", 0.0),
            "level_scale": (
                None if vibration is None or vibration.mode == "off" else vibration.level_scale
            ),
            "gamma_realized": getattr(base, "gamma_realized", 0.0),
            "seed": getattr(base, "_base_seed", 0),
        }

    def _recording_observation(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        result = {
            key: np.asarray(value).copy()
            for key, value in observation.items()
            if key not in _POLICY_PRIVILEGED_KEYS
        }
        truth_provider = getattr(self.env.unwrapped, "privileged_observation", None)
        if truth_provider is not None:
            for key, value in truth_provider().items():
                if not key.startswith("privileged_"):
                    raise ValueError(f"truth label {key!r} must use privileged_ prefix")
                result[key] = np.asarray(value).copy()
        return result

    @staticmethod
    def _state_vector(observation: Mapping[str, np.ndarray]) -> np.ndarray:
        keys = (
            "robot0_joint_pos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        return np.concatenate(
            [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in keys]
        )

    def reset(self, **kwargs):
        if self._current is not None and self._current.num_samples:
            self._episodes.append(self._current)
        observation, info = self.env.reset(**kwargs)
        self._current = _Episode(self._init_state())
        self._last_observation = self._recording_observation(observation)
        return observation, info

    def step(self, action):
        if self._current is None or self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        before = self._last_observation
        observation, reward, terminated, truncated, info = self.env.step(action)
        for key, value in before.items():
            self._current.observations.setdefault(key, []).append(np.asarray(value).copy())
        self._current.states.append(self._state_vector(before))
        self._current.actions.append(np.asarray(action, dtype=np.float32).copy())
        self._current.rewards.append(float(reward))
        self._current.dones.append(bool(terminated or truncated))
        self._last_observation = self._recording_observation(observation)
        if terminated or truncated:
            self._episodes.append(self._current)
            self._current = None
            self._last_observation = None
        return observation, reward, terminated, truncated, info

    def _completed_episodes(self) -> list[_Episode]:
        result = list(self._episodes)
        if self._current is not None and self._current.num_samples:
            result.append(self._current)
        return result

    def flush(self) -> Path:
        """Write all buffered trajectories and return the final file path."""

        episodes = self._completed_episodes()
        if not episodes:
            raise ValueError("no trajectory samples have been collected")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with h5py.File(temporary, "w") as handle:
            data = handle.create_group("data")
            data.attrs["env_args"] = json.dumps(
                self.env_args, sort_keys=True, default=_json_default
            )
            data.attrs["total"] = sum(episode.num_samples for episode in episodes)
            demo_names: list[str] = []
            for index, episode in enumerate(episodes):
                name = f"demo_{index}"
                demo_names.append(name)
                demo = data.create_group(name)
                demo.attrs["num_samples"] = episode.num_samples
                demo.attrs["init_state"] = json.dumps(
                    episode.init_state, sort_keys=True, default=_json_default
                )
                demo.create_dataset("states", data=np.stack(episode.states), compression="gzip")
                demo.create_dataset("actions", data=np.stack(episode.actions), compression="gzip")
                demo.create_dataset("rewards", data=np.asarray(episode.rewards, np.float32))
                demo.create_dataset("dones", data=np.asarray(episode.dones, np.bool_))
                obs_group = demo.create_group("obs")
                for key, values in sorted(episode.observations.items()):
                    obs_group.create_dataset(key, data=np.stack(values), compression="gzip")
            mask = handle.create_group("mask")
            valid_count = int(round(len(demo_names) * self.validation_fraction))
            valid_count = min(valid_count, max(0, len(demo_names) - 1))
            split = len(demo_names) - valid_count
            string_dtype = h5py.string_dtype("utf-8")
            mask.create_dataset("train", data=np.asarray(demo_names[:split], dtype=string_dtype))
            mask.create_dataset("valid", data=np.asarray(demo_names[split:], dtype=string_dtype))
        temporary.replace(self.output_path)
        return self.output_path

    def close(self) -> None:
        if self._completed_episodes():
            self.flush()
        super().close()
