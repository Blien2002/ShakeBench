"""Gymnasium-compatible policy environment and environment registry."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..config import (
    AXES,
    AssetConfig,
    BenchmarkConfig,
    SpectralBand,
    VibrationConfig,
    workpiece_mass_kg,
)
from ..controllers import make_controller
from ..vibration import SpectralVibration, SyntheticDeckIMU, calibrate_level_scale

_ENV_REGISTRY: dict[str, type["ShakeBenchEnv"]] = {}


def register_env(*names: str) -> Callable[[type["ShakeBenchEnv"]], type["ShakeBenchEnv"]]:
    """Register an environment class under one or more case-insensitive names."""

    def decorator(cls: type["ShakeBenchEnv"]) -> type["ShakeBenchEnv"]:
        for name in names:
            key = name.lower()
            if key in _ENV_REGISTRY and _ENV_REGISTRY[key] is not cls:
                raise ValueError(f"environment {name!r} is already registered")
            _ENV_REGISTRY[key] = cls
        return cls

    return decorator


def registered_envs() -> tuple[str, ...]:
    return tuple(sorted(_ENV_REGISTRY))


@register_env("PickPlace", "pick_place", "PanelOperation", "panel_operation")
class ShakeBenchEnv(gym.Env):
    """Deterministic single-environment policy API.

    The class owns the policy-rate/physics-rate scheduling, observation
    isolation, controller scaling, and initial-state semantics. Isaac scene
    execution can be attached behind the same contract; the default state
    backend keeps API checks and policy integration usable without launching a
    renderer or GPU application.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    backend_name = "state_contract"
    scoreable = False

    def __init__(
        self,
        env_name: str = "PickPlace",
        robots: str = "Panda",
        controller_configs: dict[str, Any] | None = None,
        control_freq: int = 5,
        horizon: int | None = None,
        episode_s: float = 16.0,
        gamma: float = 0.50,
        vibration_mode: str = "spectral",
        frequency_scale: float = 1.0,
        bandwidth_ratio: float | None = None,
        physics_profile: str = "official",
        physics_hz: int | None = None,
        intra_step_mode: str | None = None,
        use_object_obs: bool = False,
        use_camera_obs: bool = True,
        use_vibration_obs: bool = False,
        use_phase_obs: bool = False,
        use_instantaneous_load_obs: bool = False,
        use_imu_obs: bool = True,
        reward_shaping: bool = True,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        ignore_done: bool = False,
        hard_reset: bool = True,
        render_mode: str | None = None,
        seed: int = 17,
        workpiece: str = "sugar_box",
        camera_height: int = 84,
        camera_width: int = 84,
        **_: Any,
    ) -> None:
        super().__init__()
        normalized_name = env_name.lower()
        if normalized_name not in _ENV_REGISTRY:
            raise ValueError(f"unknown env_name {env_name!r}; available={registered_envs()}")
        if robots.lower() not in ("panda", "franka", "franka_panda"):
            raise ValueError("ShakeBench currently supports only the Franka Panda")
        if physics_profile not in ("official", "training"):
            raise ValueError("physics_profile must be 'official' or 'training'")
        self.env_name = "PanelOperation" if "panel" in normalized_name else "PickPlace"
        self.task_name = "panel_operation" if self.env_name == "PanelOperation" else "pick_place"
        self.robots = robots
        self.control_freq = int(control_freq)
        self.physics_hz = int(physics_hz or (1000 if physics_profile == "official" else 240))
        self.episode_s = float(episode_s)
        if self.control_freq <= 0 or self.physics_hz % self.control_freq != 0:
            raise ValueError(
                f"physics_hz={self.physics_hz} must be an integer multiple of "
                f"control_freq={self.control_freq}"
            )
        self.horizon = int(round(self.episode_s * self.control_freq))
        if self.horizon < 1:
            raise ValueError("episode_s*control_freq must produce at least one policy step")
        if horizon is not None and int(horizon) != self.horizon:
            raise ValueError(
                f"horizon is derived from episode_s*control_freq ({self.horizon}); "
                f"received independent value {horizon}"
            )
        self.control_timestep = 1.0 / self.control_freq
        self.model_timestep = 1.0 / self.physics_hz
        self.physics_steps_per_action = self.physics_hz // self.control_freq
        config = dict(controller_configs or {"name": "JOINT_POSITION", "action_dim": 8})
        if intra_step_mode is not None:
            config["intra_step_mode"] = intra_step_mode
        mode = config.get("intra_step_mode", "zoh")
        if mode not in ("zoh", "interp", "feedforward"):
            raise ValueError("intra_step_mode must be zoh, interp, or feedforward")
        if mode == "feedforward" and config.get("name") != "VARIABLE_IMPEDANCE":
            raise ValueError("feedforward is only valid for VARIABLE_IMPEDANCE")
        self.intra_step_mode = mode
        self.controller_config = config
        self.controller = make_controller(config)
        self.action_space = spaces.Box(-1.0, 1.0, (self.controller.action_dim,), np.float32)
        self.use_object_obs = bool(use_object_obs)
        self.use_camera_obs = bool(use_camera_obs)
        self.use_vibration_obs = bool(use_vibration_obs)
        self.use_phase_obs = bool(use_phase_obs)
        self.use_instantaneous_load_obs = bool(use_instantaneous_load_obs)
        self.use_imu_obs = bool(use_imu_obs)
        self.reward_shaping = bool(reward_shaping)
        self.has_renderer = bool(has_renderer)
        self.has_offscreen_renderer = bool(has_offscreen_renderer)
        self.ignore_done = bool(ignore_done)
        self.hard_reset = bool(hard_reset)
        self.render_mode = render_mode
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self._base_seed = int(seed)
        self._vibration_cfg = VibrationConfig(
            mode=vibration_mode,
            seed=self._base_seed,
            gamma=gamma,
            frequency_scale=frequency_scale,
        )
        if bandwidth_ratio is not None:
            ratio = float(bandwidth_ratio)
            bands = {
                axis: tuple(
                    SpectralBand(band.center_hz, band.accel_rms, ratio, band.tones)
                    for band in axis_bands
                )
                for axis, axis_bands in self._vibration_cfg.bands.items()
            }
            self._vibration_cfg = replace(self._vibration_cfg, bands=bands)
        if self._vibration_cfg.mode == "off":
            self._vibration_cfg = replace(self._vibration_cfg, level_scale=0.0)
            self.gamma_realized = 0.0
        else:
            level_scale, report = calibrate_level_scale(
                self._vibration_cfg, self.physics_hz, self.episode_s
            )
            self._vibration_cfg = replace(self._vibration_cfg, level_scale=level_scale)
            self.gamma_realized = float(report["gamma_realized"])
        self.cfg = BenchmarkConfig(
            dt=1.0 / self.physics_hz,
            episode_s=self.episode_s,
            task=self.task_name,
            assets=AssetConfig(workpiece=workpiece),
            vibration=self._vibration_cfg,
            support_config="C2" if self.task_name == "panel_operation" else "C2_CLITE",
            solver_substeps=5 if self.physics_hz == 1000 else 4,
            contact_solref=(0.00060, 1.0) if self.physics_hz == 1000 else (0.0025, 1.0),
        )
        self._object_mass_kg = workpiece_mass_kg(
            self.cfg.assets.workpiece, self.cfg.assets.workpiece_scale
        )
        self._vibration = SpectralVibration(self._vibration_cfg, 1, "cpu")
        self._phase_tones = max(
            (band.tones for bands in self._vibration_cfg.bands.values() for band in bands),
            default=1,
        )
        self._imu = SyntheticDeckIMU(self.physics_hz, self._base_seed)
        self.observation_space = spaces.Dict(self._observation_spaces())
        self._init_state: dict[str, Any] | None = None
        self._reset_state()

    def _observation_spaces(self) -> dict[str, spaces.Space]:
        # Finite machine-safe limits keep Gymnasium's checker quiet while
        # remaining far outside any physically reachable benchmark value.
        unbounded = lambda shape: spaces.Box(-1.0e10, 1.0e10, shape, np.float32)
        result: dict[str, spaces.Space] = {
            "robot0_joint_pos": unbounded((7,)),
            "robot0_joint_vel": unbounded((7,)),
            "robot0_eef_pos": unbounded((3,)),
            "robot0_eef_quat": unbounded((4,)),
            "robot0_gripper_qpos": unbounded((1,)),
            "robot0_wrist_force": unbounded((3,)),
            "robot0_wrist_torque": unbounded((3,)),
        }
        if self.use_imu_obs:
            result["deck_imu"] = unbounded((6,))
        if self.use_camera_obs:
            result["wrist_camera_image"] = spaces.Box(
                0, 255, (self.camera_height, self.camera_width, 3), np.uint8
            )
        if self.use_object_obs:
            result.update(
                {
                    "object_pos": unbounded((3,)),
                    "object_quat": unbounded((4,)),
                    "target_pos": unbounded((3,)),
                    "target_quat": unbounded((4,)),
                    "penetration_mm": unbounded((1,)),
                    "mount_delta_z": unbounded((1,)),
                    "object_mass": unbounded((1,)),
                    "object_mu": unbounded((1,)),
                    "object_com_offset": unbounded((3,)),
                }
            )
        if self.use_vibration_obs:
            result.update(
                {
                    "vibration_q": unbounded((6,)),
                    "vibration_qd": unbounded((6,)),
                    "vibration_qdd": unbounded((6,)),
                    "vibration_level_scale": unbounded((1,)),
                    "vibration_t0": unbounded((1,)),
                }
            )
        if self.use_instantaneous_load_obs and not self.use_vibration_obs:
            result["vibration_qdd"] = unbounded((6,))
        if self.use_phase_obs:
            result.update(
                {
                    "vibration_omega": unbounded((6, self._phase_tones)),
                    "vibration_phase": unbounded((6, self._phase_tones)),
                    "vibration_line_mask": spaces.Box(
                        0, 1, (6, self._phase_tones), np.uint8
                    ),
                    "vibration_level_scale": unbounded((1,)),
                    "vibration_time": unbounded((1,)),
                    "vibration_ramp_s": unbounded((1,)),
                }
            )
        return result

    def _reset_state(self) -> None:
        self.elapsed_steps = 0
        self.physics_steps = 0
        self.time_s = 0.0
        self.controller.reset()
        self._imu.reset(self._base_seed)
        self._joint_pos = np.zeros(7, np.float32)
        self._joint_vel = np.zeros(7, np.float32)
        self._eef_pos = np.array([0.45, 0.0, 0.45], np.float32)
        self._eef_quat = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        self._gripper = np.array([0.04], np.float32)
        self._object_pos = np.array([0.08, -0.13, 0.47], np.float32)
        self._object_quat = self._eef_quat.copy()
        self._target_pos = np.array([0.08, 0.17, 0.376], np.float32)
        self._target_quat = self._eef_quat.copy()
        self._wrist_force = np.zeros(3, np.float32)
        self._wrist_torque = np.zeros(3, np.float32)
        self._last_motion = tuple(np.zeros((1, 6), np.float32) for _ in range(3))
        self._last_motion_time_s = 0.0
        self._last_imu = np.zeros(6, np.float32)
        self._grip_force_sum = 0.0
        self._grip_force_count = 0
        self._required_grip_force_min = float("inf")
        self._grip_margin_min = float("inf")
        self._grip_excess_sum = 0.0
        self._grip_ratio_count = 0
        self._ee_tracking_error_sq = 0.0
        self._ee_tracking_error_count = 0
        if self._init_state is not None:
            placement = np.asarray(self._init_state["object_placement"], dtype=np.float32)
            if placement.shape[0] >= 1:
                self._object_pos, self._object_quat = placement[0, :3], placement[0, 3:7]
            if placement.shape[0] >= 2:
                self._target_pos, self._target_quat = placement[1, :3], placement[1, 3:7]

    def set_init_state(self, init_state: dict[str, Any]) -> None:
        required = {
            "object_placement", "workpiece", "t0", "level_scale", "gamma_realized", "seed"
        }
        missing = required - set(init_state)
        if missing:
            raise ValueError(f"init_state missing fields: {sorted(missing)}")
        placement = np.asarray(init_state["object_placement"])
        if placement.ndim != 2 or placement.shape[1] != 7:
            raise ValueError("object_placement must have shape (N, 7)")
        if str(init_state["workpiece"]) != self.cfg.assets.workpiece:
            raise ValueError(
                f"init_state workpiece={init_state['workpiece']!r} does not match "
                f"environment workpiece={self.cfg.assets.workpiece!r}"
            )
        self._init_state = dict(init_state)
        self._base_seed = int(init_state["seed"])
        candidate = replace(
            self._vibration_cfg,
            seed=self._base_seed,
            t0=float(init_state["t0"]),
            gamma=float(init_state["gamma_realized"]),
            level_scale=1.0,
        )
        scale, report = calibrate_level_scale(candidate, self.physics_hz, self.episode_s)
        if not math.isclose(scale, float(init_state["level_scale"]), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("init_state level_scale does not match its (seed, t0) calibration")
        if not math.isclose(
            float(report["gamma_realized"]), float(init_state["gamma_realized"]),
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError("init_state gamma_realized failed replay validation")
        self._vibration_cfg = replace(candidate, level_scale=scale)
        self._vibration = SpectralVibration(self._vibration_cfg, 1, "cpu")
        self.gamma_realized = float(report["gamma_realized"])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        if seed is not None and self._init_state is None:
            self._base_seed = int(seed)
            candidate = replace(self._vibration_cfg, seed=self._base_seed, level_scale=1.0)
            if candidate.mode == "off":
                scale, realized = 0.0, 0.0
            else:
                scale, report = calibrate_level_scale(candidate, self.physics_hz, self.episode_s)
                realized = float(report["gamma_realized"])
            self._vibration_cfg = replace(candidate, level_scale=scale)
            self._vibration = SpectralVibration(self._vibration_cfg, 1, "cpu")
            self.gamma_realized = realized
        self._reset_state()
        return self._observation(), self._info(False)

    def _physics_step(self, command: np.ndarray) -> None:
        previous = self._joint_pos.copy()
        if self.controller_config.get("name", "JOINT_POSITION") == "JOINT_POSITION":
            self._joint_pos += 0.02 * (command[:7] - self._joint_pos)
            self._gripper[0] = 0.04 * (1.0 - 0.5 * (command[7] + 1.0))
        else:
            self._eef_pos += command[:3] / self.physics_steps_per_action
            self._gripper[0] = max(0.0, self._gripper[0] - 1e-4 * float(command[-1]))
        self._joint_vel = (self._joint_pos - previous) * self.physics_hz
        self._last_motion_time_s = self.time_s
        q, qd, qdd = self._vibration.sample(self._last_motion_time_s)
        self._last_motion = tuple(value.numpy() for value in (q, qd, qdd))
        self._last_imu = self._imu.sample(
            self._last_motion[2][0, :3],
            self._last_motion[1][0, 3:],
            angular_accel_deck=self._last_motion[2][0, 3:],
            r_imu_deck=np.asarray(self.cfg.resolved_robot_base) - np.asarray(self.cfg.platform_center),
        )
        if self.controller_config.get("name") == "VARIABLE_IMPEDANCE":
            applied = max(0.0, float(command[12]))
            r_wp = np.asarray(self._vibration_cfg.workpiece_offset_m, dtype=np.float32)
            acceleration = self._last_motion[2][0, :3] + np.cross(
                self._last_motion[2][0, 3:], r_wp
            )
            required = self._object_mass_kg * float(np.linalg.norm(acceleration)) / self.cfg.material_mu
            self._grip_force_sum += applied
            self._grip_force_count += 1
            if required > 1.0e-9:
                ratio = applied / required
                self._required_grip_force_min = min(self._required_grip_force_min, required)
                self._grip_margin_min = min(self._grip_margin_min, ratio)
                self._grip_excess_sum += ratio
                self._grip_ratio_count += 1
        self.physics_steps += 1
        self.time_s = self.physics_steps / self.physics_hz

    def step(self, action: np.ndarray):
        action_array = np.asarray(action, dtype=np.float32)
        clipped_any = bool(np.any(action_array < -1.0) or np.any(action_array > 1.0))
        for index in range(self.physics_steps_per_action):
            policy_step = index == 0
            alpha = (index + 1) / self.physics_steps_per_action if self.intra_step_mode == "interp" else 1.0
            phase = self._dominant_phase()
            output = self.controller.pre_action(
                action_array, policy_step, alpha=alpha, phase=phase
            )
            clipped_any |= output.clipped
            self._physics_step(output.command)
        self.elapsed_steps += 1
        distance = float(np.linalg.norm(self._object_pos - self._target_pos))
        reward = -distance if self.reward_shaping else float(distance < 0.05)
        terminated = bool(distance < 0.05 and not self.ignore_done)
        truncated = bool(self.elapsed_steps >= self.horizon and not self.ignore_done)
        return self._observation(), float(reward), terminated, truncated, self._info(clipped_any)

    def _camera_image(self) -> np.ndarray:
        image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        image[..., 0] = np.uint8(np.clip(80 + 100 * self._eef_pos[0], 0, 255))
        image[..., 1] = np.uint8(np.clip(60 + 100 * self._eef_pos[2], 0, 255))
        image[..., 2] = 110
        return image

    def _observation(self) -> dict[str, np.ndarray]:
        q, qd, qdd = self._last_motion
        obs = {
            "robot0_joint_pos": self._joint_pos.astype(np.float32).copy(),
            "robot0_joint_vel": self._joint_vel.astype(np.float32).copy(),
            "robot0_eef_pos": self._eef_pos.astype(np.float32).copy(),
            "robot0_eef_quat": self._eef_quat.copy(),
            "robot0_gripper_qpos": self._gripper.copy(),
            "robot0_wrist_force": self._wrist_force.copy(),
            "robot0_wrist_torque": self._wrist_torque.copy(),
        }
        if self.use_imu_obs:
            obs["deck_imu"] = self._last_imu.copy()
        if self.use_camera_obs:
            obs["wrist_camera_image"] = self._camera_image()
        if self.use_object_obs:
            obs.update(
                {
                    "object_pos": self._object_pos.copy(),
                    "object_quat": self._object_quat.copy(),
                    "target_pos": self._target_pos.copy(),
                    "target_quat": self._target_quat.copy(),
                    "penetration_mm": np.zeros(1, np.float32),
                    "mount_delta_z": np.zeros(1, np.float32),
                    "object_mass": np.array([self._object_mass_kg], np.float32),
                    "object_mu": np.array([self.cfg.material_mu], np.float32),
                    "object_com_offset": np.zeros(3, np.float32),
                }
            )
        if self.use_vibration_obs:
            obs.update(
                {
                    "vibration_q": q[0].astype(np.float32).copy(),
                    "vibration_qd": qd[0].astype(np.float32).copy(),
                    "vibration_qdd": qdd[0].astype(np.float32).copy(),
                    "vibration_level_scale": np.array([self._vibration_cfg.level_scale], np.float32),
                    "vibration_t0": np.array([self._vibration_cfg.t0], np.float32),
                }
            )
        elif self.use_instantaneous_load_obs:
            obs["vibration_qdd"] = qdd[0].astype(np.float32).copy()
        if self.use_phase_obs:
            obs.update(self._phase_observation())
        return obs

    def _phase_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return padded analytic line parameters without exposing state truth."""

        omega = np.zeros((6, self._phase_tones), np.float32)
        phase = np.zeros_like(omega)
        mask = np.zeros_like(omega, dtype=np.uint8)
        for axis, values in self._vibration._omega.items():
            index = AXES.index(axis)
            count = min(self._phase_tones, values.shape[1])
            omega[index, :count] = values[0, :count].cpu().numpy()
            phase[index, :count] = self._vibration._phase[axis][0, :count].cpu().numpy()
            mask[index, :count] = 1
        return omega, phase, mask

    def _phase_observation(self) -> dict[str, np.ndarray]:
        omega, phase0, mask = self._phase_arrays()
        current = np.remainder(
            omega * np.float32(self._last_motion_time_s + self._vibration_cfg.t0) + phase0,
            np.float32(2.0 * math.pi),
        ).astype(np.float32)
        return {
            "vibration_omega": omega,
            "vibration_phase": current,
            "vibration_line_mask": mask,
            "vibration_level_scale": np.array([self._vibration_cfg.level_scale], np.float32),
            "vibration_time": np.array([self._last_motion_time_s], np.float32),
            "vibration_ramp_s": np.array([self._vibration_cfg.ramp_s], np.float32),
        }

    def _dominant_phase(self) -> float:
        if not self._vibration._omega:
            return 0.0
        axis = "tz" if "tz" in self._vibration._omega else next(iter(self._vibration._omega))
        omega = float(self._vibration._omega[axis][0, 0].item())
        phase = float(self._vibration._phase[axis][0, 0].item())
        return omega * (self.time_s + self._vibration_cfg.t0) + phase

    def privileged_observation(self) -> dict[str, np.ndarray]:
        """Truth labels for recorders; this mapping is never returned to a policy."""

        phase = self._phase_observation()["vibration_phase"]
        return {
            "privileged_object_pose": np.concatenate(
                (self._object_pos, self._object_quat)
            ).astype(np.float32),
            "privileged_vibration_qdd": self._last_motion[2][0].astype(np.float32).copy(),
            "privileged_phase": phase,
            "privileged_object_mu": np.array([self.cfg.material_mu], np.float32),
            "privileged_object_mass": np.array([self._object_mass_kg], np.float32),
        }

    def _info(self, action_clipped: bool) -> dict[str, Any]:
        grip_count = max(self._grip_force_count, 1)
        return {
            "backend": "state_contract",
            "scoreable": False,
            "action_clipped": bool(action_clipped),
            "control_freq": self.control_freq,
            "physics_hz": self.physics_hz,
            "physics_steps": self.physics_steps,
            "episode_s": self.episode_s,
            "horizon": self.horizon,
            "intra_step_mode": self.intra_step_mode,
            "gamma_realized": self.gamma_realized,
            "success": bool(np.linalg.norm(self._object_pos - self._target_pos) < 0.05),
            "grasp_assist_used": False,
            "support_geometry_valid": True,
            "max_penetration_mm": 0.0,
            "max_grasp_slip_m": 0.0,
            "ee_tracking_error_rms_m": (
                math.sqrt(self._ee_tracking_error_sq / self._ee_tracking_error_count)
                if self._ee_tracking_error_count else 0.0
            ),
            "mean_grip_force_n": self._grip_force_sum / grip_count,
            "min_required_grip_force_n": (
                self._required_grip_force_min
                if math.isfinite(self._required_grip_force_min) else 0.0
            ),
            "grip_margin_min": (
                self._grip_margin_min if math.isfinite(self._grip_margin_min) else 0.0
            ),
            "grip_excess": (
                self._grip_excess_sum / self._grip_ratio_count
                if self._grip_ratio_count else 0.0
            ),
            "privileged_observations": tuple(
                name
                for enabled, name in (
                    (self.use_object_obs, "object"),
                    (self.use_vibration_obs, "vibration"),
                    (self.use_phase_obs, "phase"),
                    (self.use_instantaneous_load_obs, "instantaneous_load"),
                )
                if enabled
            ),
        }

    def render(self):
        return self._camera_image()

    def close(self) -> None:
        return None


def make_env(env_name: str, **kwargs: Any) -> ShakeBenchEnv:
    key = env_name.lower()
    try:
        cls = _ENV_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown env_name {env_name!r}; available={registered_envs()}") from exc
    return cls(env_name=env_name, **kwargs)
