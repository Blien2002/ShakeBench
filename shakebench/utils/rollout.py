"""Model-agnostic ShakeBench rollout boundary: observation, actions, outcomes, evidence.

Any object with "chunk_size" and "predict(observation)" is a policy here.  Model
adapters (wire message layout, normalization keys, server resize) stay outside
this module, so evaluation does not depend on one model implementation.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from robosuite.utils import transform_utils as T
from shakebench.demos.demo_oracle_video import _task_close_camera
from shakebench.utils.outcomes import resolve_termination_cause, validate_outcome
from shakebench.utils.privilege import assert_policy_observation_is_clean
from shakebench.utils.task_registry import describe_task, prepare_task_state, task_type

TASK = "Pick up the object from the phenolic table and place it in the target crate."
CAMERAS = {"observation.images.main": "task_close", "observation.images.wrist": "robot0_eye_in_hand"}
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "left_finger_qpos", "right_finger_qpos"]
OBSERVATION_SOURCES = ("cameras", "contract")


def proprioception_metadata():
    """Define the measured 8D state, following OpenPI's LIBERO field layout."""
    return {
        "key": "observation.state",
        "dtype": "float32",
        "shape": [8],
        "names": list(STATE_NAMES),
        "timing": "state_t, synchronized with images_t, before applying action_t",
        "normalization": "none; physical units stored, model loader handles normalization",
        "eef_position": {"slice": [0, 3], "units": "m", "frame": "robot_base", "body": "gripper_body"},
        "eef_orientation": {
            "slice": [3, 6],
            "units": "rad",
            "frame": "robot_base",
            "representation": "rotation_vector (unit axis * angle), not Euler angles",
        },
        "gripper_position": {
            "slice": [6, 8],
            "units": "m",
            "representation": "measured signed slide-joint positions",
            "joint_order": ["finger_joint1", "finger_joint2"],
            "joint_limits": [[0.0, 0.04], [-0.04, 0.0]],
        },
        "reference": "https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py",
    }


class PolicyTimeoutError(TimeoutError):
    """Raised when a policy request exceeds its deadline."""


class PolicyOutputError(ValueError):
    """Raised when a policy returns actions that violate the action contract."""


ERROR_TAXONOMY = {
    "policy_output_violation": "model returned actions outside the 7D [-1,1] contract",
    "policy_timeout": "model service or inference exceeded the configured deadline",
    "policy_exception": "adapter raised while serving one request",
    "invalid_execution": "environment or stepping failure; not a model result",
}


def task_description(state: Mapping[str, Any]) -> dict[str, str]:
    return describe_task(state)


def observation_features(height, width, *, include_imu=True):
    """LeRobot-compatible array schema; language uses its task metadata table."""
    features = {
        **{
            key: {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
            for key in CAMERAS
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    }
    if include_imu:
        features.update(
            {
                "observation.table_imu_window": {"dtype": "float32", "shape": (10, 6), "names": None},
                "observation.table_imu_timestamps_s": {"dtype": "float64", "shape": (10,), "names": None},
                "observation.table_imu_dt_s": {"dtype": "float32", "shape": (1,), "names": None},
            }
        )
    return features


def validated_actions(value):
    """Accept one action or a nonempty chunk; reject wrong units/ranges before stepping."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise PolicyOutputError("actions must be finite real numbers")
    if array.shape == (7,):
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 7 or len(array) == 0 or np.any(np.abs(array) > 1):
        raise PolicyOutputError("expected normalized actions [7] or [N,7], N>0, within [-1,1]")
    return array.astype(np.float32, copy=True)


def action_evidence(actions: Sequence[np.ndarray]) -> dict[str, Any]:
    """Count of the actions actually executed by the environment."""
    array = np.asarray(actions, dtype=np.float64).reshape(-1, 7)
    return {
        "executed_action_count": int(len(array)),
        "units": "normalized [-1,1] robot-base OSC delta plus gripper",
    }


def episode_outcome(termination_cause: str) -> tuple[str, str | None]:
    """Map a termination cause to the frozen (episode_validity, score_outcome) pair."""
    if termination_cause == "invalid_execution":
        return "invalid", None
    return "valid", ("success" if termination_cause == "success_latched" else "unsuccessful")


class ShakeBenchCameraObservation:
    """Shared camera and sensor extraction for collection and policy evaluation."""

    def __init__(self, env, *, height=256, width=256, main_camera="task_close", include_imu=True):
        self.env = env
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(main_camera, str) or not main_camera:
            raise ValueError("main_camera must be a non-empty camera name")
        self.height, self.width, self.main_camera, self.include_imu = height, width, main_camera, include_imu
        model = env.sim.model._model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.cameras = {**CAMERAS, "observation.images.main": main_camera}
        if main_camera == "task_close":
            self.cameras["observation.images.main"] = _task_close_camera()
        else:
            try:
                env.sim.model.camera_name2id(main_camera)
            except (KeyError, ValueError, mujoco.FatalError) as exc:
                raise ValueError(f"unknown main camera {main_camera!r}") from exc
        try:
            env.sim.model.camera_name2id("robot0_eye_in_hand")
        except (KeyError, ValueError, mujoco.FatalError) as exc:
            raise ValueError("compiled scene is missing robot0_eye_in_hand") from exc
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0], self.option.geomgroup[1] = 0, 1
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def _camera_identity(self, camera: Any) -> dict[str, Any]:
        """Names alone drift, so record the compiled pose that produced the pixels."""
        if isinstance(camera, str):
            camera_id = self.env.sim.model.camera_name2id(camera)
            return {
                "kind": "compiled_camera",
                "name": camera,
                "position_m": [float(value) for value in self.env.sim.model.cam_pos[camera_id]],
                "quaternion_wxyz": [float(value) for value in self.env.sim.model.cam_quat[camera_id]],
            }
        return {
            "kind": "preset_camera",
            "name": self.main_camera,
            "lookat": [float(value) for value in camera.lookat],
            "distance": float(camera.distance),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
        }

    def contract(self) -> dict[str, Any]:
        """Executable observation identity for one evaluation result."""
        return {
            "source": "shakebench.camera_observation",
            "height": self.height,
            "width": self.width,
            "proprioception": proprioception_metadata(),
            "cameras": {key: self._camera_identity(camera) for key, camera in self.cameras.items()},
        }

    def read(self, observation):
        result = {}
        for key, camera in self.cameras.items():
            self.renderer.update_scene(self.env.sim.data._data, camera=camera, scene_option=self.option)
            image = self.renderer.render().copy()
            if image.shape != (self.height, self.width, 3):
                raise ValueError(f"camera {key} returned {image.shape}, expected {(self.height, self.width, 3)}")
            result[key] = image
        robot = self.env.robots[0]
        pose = robot.pose_in_base_from_name(self.env.gripper_body_name)
        result["observation.state"] = np.concatenate(
            (
                pose[:3, 3],
                T.quat2axisangle(T.mat2quat(pose[:3, :3])),
                self.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]],
            )
        ).astype(np.float32)
        if self.include_imu:
            for name in ("table_imu_window", "table_imu_timestamps_s", "table_imu_dt_s"):
                dtype = np.float64 if name == "table_imu_timestamps_s" else np.float32
                result[f"observation.{name}"] = np.atleast_1d(np.asarray(observation[name], dtype=dtype)).copy()
        if result["observation.state"].shape != (8,):
            raise ValueError("observation.state must have shape (8,)")
        if self.include_imu and result["observation.table_imu_window"].shape != (10, 6):
            raise ValueError("table_imu_window must have shape (10, 6)")
        if self.include_imu and result["observation.table_imu_timestamps_s"].shape != (10,):
            raise ValueError("table_imu_timestamps_s must have shape (10,)")
        if self.include_imu and result["observation.table_imu_dt_s"].shape != (1,):
            raise ValueError("table_imu_dt_s must have shape (1,)")
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("non-finite policy observation")
        return result

    def sync_device_state(self, device_data, world: int) -> None:
        """Adopt one MJWarp world's state so host pixels show the same scene.

        Device physics owns qpos/qvel and the mocap-driven shaker deck.  Host rendering is
        still required because MJWarp's renderer maps textures only onto plane and mesh
        geoms, so every textured box in this scene would render as a flat colour.
        """

        data = self.env.sim.data
        data.qpos[:] = device_data.qpos.numpy()[world]
        data.qvel[:] = device_data.qvel.numpy()[world]
        data.mocap_pos[:] = device_data.mocap_pos.numpy()[world]
        data.mocap_quat[:] = device_data.mocap_quat.numpy()[world]
        mujoco.mj_forward(self.env.sim.model._model, data._data)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class ShakeBenchTaskEnv:
    """Gymnasium-style reset/step without a Gym dependency; state fixes all episode seeds.

    Policy receives only the observation dict. Evaluator info and raw env are not
    policy inputs. "observation_source" selects the declared camera schema or the
    environment's own contracted state fields; both keep the same action contract.
    Recreating the environment on reset deliberately preserves initial-state
    reproducibility.
    """

    def __init__(
        self,
        state,
        *,
        gamma=0.0,
        horizon=1200,
        height=256,
        width=256,
        main_camera="task_close",
        mode="multisine_v1",
        physics_profile="official",
        observation_source="cameras",
    ):
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (horizon, height, width)):
            raise ValueError("horizon and image dimensions must be integers")
        if min(horizon, height, width) <= 0:
            raise ValueError("horizon and image dimensions must be positive")
        try:
            gamma = float(gamma)
        except (TypeError, ValueError) as exc:
            raise ValueError("gamma must be finite and nonnegative") from exc
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if observation_source not in OBSERVATION_SOURCES:
            raise ValueError(f"observation_source must be one of {OBSERVATION_SOURCES}")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping loaded from an authenticated state asset")
        if not isinstance(state.get("state_id"), str) or not state["state_id"]:
            raise ValueError("state must include a non-empty state_id")
        # Verified committed assets name the object can_*; the shared boundary
        # normalizes both schemas once, before any rollout starts.
        self.state = prepare_task_state(state)
        self.description = task_description(self.state)
        self.gamma, self.horizon = float(gamma), horizon
        self.mode, self.physics_profile = mode, physics_profile
        self.observation_source = observation_source
        self.camera_options = {"height": height, "width": width, "main_camera": main_camera}
        self.env = self.reader = None
        self.steps, self.done = 0, True

    @property
    def action_spec(self):
        return -np.ones(7, dtype=np.float32), np.ones(7, dtype=np.float32)

    @property
    def task_id(self):
        return self.description["task_id"]

    @property
    def instruction(self):
        return self.description["instruction"]

    def observation_contract(self):
        contract = {
            "observation_source": self.observation_source,
            "task": {"dtype": "string", "shape": ()},
            "timestamp": {"dtype": "float32", "shape": (1,)},
        }
        if self.observation_source == "cameras":
            return {**observation_features(self.camera_options["height"], self.camera_options["width"]), **contract}
        contract["observation"] = {} if self.env is None else self.env.observation_contract()
        return contract

    def observation_identity(self) -> dict[str, Any]:
        """Executable observation identity recorded next to each evaluation result."""
        if self.observation_source == "cameras" and self.reader is not None:
            return self.reader.contract()
        declared = getattr(self.env, "observation_contract", None)
        return {
            "source": "shakebench.contract_observation",
            "fields": declared() if callable(declared) else {},
        }

    def get_task_context(self):
        context = {
            **self.description,
            "state_id": self.state.get("state_id"),
            "gamma": self.gamma,
            "horizon": self.horizon,
            "observation_source": self.observation_source,
            "action_space": {"shape": [7], "normalized_bounds": [-1.0, 1.0], "names": ACTION_NAMES},
        }
        environment_context = getattr(self.env, "get_policy_task_context", None)
        if callable(environment_context):
            context["environment"] = environment_context()
        return deepcopy(context)

    def _observation(self, observation):
        if self.reader is not None:
            payload = self.reader.read(observation)
        else:
            payload = {
                key: np.asarray(observation[key], dtype=float).copy() for key in self.env.policy_observation_keys
            }
        assert_policy_observation_is_clean(payload)
        return {
            **payload,
            "task": self.description["instruction"],
            "timestamp": np.array([self.steps / 20], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        from shakebench.utils.task_runtime import make_environment

        if seed is not None:
            raise ValueError("episode seeds are fixed by the authenticated state asset")
        if options not in (None, {}):
            raise ValueError("ShakeBenchTaskEnv uses the supplied authenticated state; reset options are unsupported")
        self.close()
        try:
            self.env, _ = make_environment(
                self.state,
                gamma=self.gamma,
                horizon=self.horizon,
                mode=self.mode,
                physics_profile=self.physics_profile,
                free_ring_peg=task_type(self.state) == "ring_on_peg",
            )
            if self.env.control_freq != 20 or self.env.action_dim != 7:
                raise ValueError("VLA interface requires the 20 Hz, 7D Panda controller")
            if self.observation_source == "cameras":
                self.reader = ShakeBenchCameraObservation(self.env, **self.camera_options)
            self.steps, self.done = 0, False
            return self._observation(self.env._get_observations()), {
                **self.description,
                "state_id": self.state.get("state_id"),
                "scoreable": False,
            }
        except BaseException:
            self.close()
            raise

    def step(self, action):
        if self.done:
            raise RuntimeError("call reset() before stepping a new or completed episode")
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        if np.asarray(action).shape != (7,):
            raise ValueError("step accepts a single [7] action; execute chunks one action at a time")
        actions = validated_actions(action)
        observation, reward, _, _ = self.env.step(actions[0])
        self.steps += 1
        metrics = self.env.get_metrics()
        cause = resolve_termination_cause(
            prior_cause=None,
            task_rule_violation=bool(metrics.get("task_rule_violation", False)),
            success_latched=bool(metrics["success"]["passed"]),
            policy_abort=False,
            horizon_exhausted=self.steps >= self.horizon,
        )
        self.done = cause is not None
        truncated = cause == "horizon_exhausted"
        info = {
            "success": cause == "success_latched",
            "termination_cause": cause,
            "steps": self.steps,
            "scoreable": False,
        }
        return self._observation(observation), float(reward), self.done and not truncated, truncated, info

    def close(self):
        try:
            if self.reader is not None:
                self.reader.close()
        finally:
            self.reader = None
            if self.env is not None:
                self.env.close()
                self.env = None
            self.done = True


def _predict(policy, observation, *, inference_timeout_s, policy_deadline_s):
    """One policy call plus a check that the adapter honored its own deadline.

    A synchronous ``predict()`` cannot be interrupted once it has started, so the
    enforceable bound is the transport deadline the adapter declares; this
    measurement only proves the adapter kept it.  ``rollout_policy`` refuses to
    configure an inference timeout without such a declaration.
    """
    started = time.perf_counter()
    try:
        actions = policy.predict(observation)
    except PolicyTimeoutError:
        raise
    elapsed = time.perf_counter() - started
    if inference_timeout_s is not None and elapsed > inference_timeout_s:
        raise PolicyTimeoutError(
            f"policy answered after {elapsed:.3f}s, past the {inference_timeout_s}s deadline it declares "
            f"for itself (adapter deadline_s {policy_deadline_s}); the adapter must enforce its own deadline"
        )
    return validated_actions(actions), elapsed


def declared_deadline_s(policy) -> float | None:
    """The executable per-request deadline a policy declares, if any."""

    deadline = getattr(policy, "deadline_s", None)
    if deadline is None:
        return None
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not np.isfinite(deadline)
        or deadline <= 0
    ):
        raise ValueError("policy deadline_s must be a positive number of seconds when declared")
    return float(deadline)


def _task_hook(task, name):
    hook = getattr(task, name, None)
    return hook() if callable(hook) else None


def invalid_episode_result(
    *,
    state_id,
    reason,
    task=None,
    policy_id=None,
    gamma=None,
    horizon_steps=None,
    action_horizon=None,
    inference_timeout_s=None,
) -> dict[str, Any]:
    """Result record for an episode whose task could not be built or started.

    The episode keeps its place in the attempted denominator as an invalid
    execution.  It is not a model result, so ``policy_errors`` stays empty and
    every execution field is zero.
    """

    validity, score_outcome = episode_outcome("invalid_execution")
    validate_outcome(episode_validity=validity, score_outcome=score_outcome, termination_cause="invalid_execution")
    return {
        "state_id": state_id,
        "gamma": gamma,
        "horizon_steps": horizon_steps,
        "policy_id": policy_id,
        "success": False,
        "episode_validity": validity,
        "score_outcome": score_outcome,
        "termination_cause": "invalid_execution",
        "steps": 0,
        "terminated": False,
        "truncated": False,
        "invalid_execution_reason": reason,
        "reward_sum": 0.0,
        "policy_calls": 0,
        "policy_errors": [],
        "error_taxonomy": ERROR_TAXONOMY,
        "inference_wall_time_s": 0.0,
        "action_horizon": action_horizon,
        "inference_timeout_s": inference_timeout_s,
        "observation_identity": _task_hook(task, "observation_identity"),
        "task_context": _task_hook(task, "get_task_context"),
        "timing": "synchronous_simulation_paused_during_inference",
        **action_evidence([]),
    }


def rollout_policy(
    task,
    policy,
    *,
    action_horizon=1,
    inference_timeout_s=None,
    policy_id=None,
    error_taxonomy=None,
):
    """Execute policy chunks with synchronous, paused-during-inference simulation.

    Failures are typed and recorded instead of raised: policy output violations and
    request timeouts end the episode as "policy_error", while an environment or
    stepping failure is "invalid_execution".  Episode setup failures (policy reset,
    task reset) are recorded the same way, so one broken state cannot lose the
    rest of a batch.

    ``inference_timeout_s`` requires the policy to declare an enforceable
    ``deadline_s``; a blocking ``predict()`` can only be bounded from inside the
    adapter.  A policy with an optional ``reset()`` gets it called once per episode.
    """
    if not isinstance(action_horizon, int) or isinstance(action_horizon, bool) or action_horizon < 1:
        raise ValueError("action_horizon must be a positive integer")
    if action_horizon > policy.chunk_size:
        raise ValueError("action_horizon exceeds the checkpoint chunk size")
    if inference_timeout_s is not None and (
        isinstance(inference_timeout_s, bool)
        or not isinstance(inference_timeout_s, (int, float))
        or not np.isfinite(inference_timeout_s)
        or inference_timeout_s <= 0
    ):
        raise ValueError("inference_timeout_s must be a positive number of seconds")
    policy_deadline_s = declared_deadline_s(policy)
    if inference_timeout_s is not None:
        if policy_deadline_s is None:
            raise ValueError(
                "inference_timeout_s needs an enforceable deadline: this policy declares no deadline_s, and a "
                "synchronous predict() call cannot be interrupted once it has started"
            )
        if policy_deadline_s > inference_timeout_s:
            raise ValueError(
                f"policy deadline_s {policy_deadline_s} exceeds the configured inference_timeout_s "
                f"{inference_timeout_s}; nothing can bound the call past the adapter's own deadline"
            )
    predictions, inference_s, reward_sum = 0, 0.0, 0.0
    executed, policy_errors = [], []
    termination_cause, invalid_reason = None, None
    observation = None

    def record(error_type, message):
        policy_errors.append({"step": len(executed), "error_type": error_type, "message": message})
        return "policy_error"

    try:
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        observation, _ = task.reset()
    except Exception as exc:  # noqa: BLE001 - episode setup failure, not a model result
        termination_cause, invalid_reason = "invalid_execution", f"{type(exc).__name__}: {exc}"

    while termination_cause is None:
        try:
            actions, elapsed = _predict(
                policy,
                observation,
                inference_timeout_s=inference_timeout_s,
                policy_deadline_s=policy_deadline_s,
            )
        except PolicyTimeoutError as exc:
            termination_cause = record("policy_timeout", str(exc))
            break
        except PolicyOutputError as exc:
            termination_cause = record("policy_output_violation", str(exc))
            break
        except Exception as exc:  # noqa: BLE001 - one adapter failure must not lose the episode record
            termination_cause = record("policy_exception", f"{type(exc).__name__}: {exc}")
            break
        inference_s += elapsed
        predictions += 1
        for action in actions[:action_horizon]:
            try:
                observation, reward, terminated, truncated, info = task.step(action)
            except Exception as exc:  # noqa: BLE001 - infrastructure failure, not a model result
                termination_cause, invalid_reason = "invalid_execution", f"{type(exc).__name__}: {exc}"
                break
            executed.append(np.asarray(action, dtype=np.float64))
            reward_sum += reward
            if terminated or truncated:
                termination_cause = info["termination_cause"]
                break

    validity, score_outcome = episode_outcome(termination_cause)
    validate_outcome(episode_validity=validity, score_outcome=score_outcome, termination_cause=termination_cause)
    return {
        "state_id": getattr(task, "state", {}).get("state_id"),
        "gamma": getattr(task, "gamma", None),
        "horizon_steps": getattr(task, "horizon", None),
        "policy_id": policy_id,
        "success": score_outcome == "success",
        "episode_validity": validity,
        "score_outcome": score_outcome,
        "termination_cause": termination_cause,
        "steps": len(executed),
        "terminated": termination_cause in {"success_latched", "task_rule_violation", "policy_abort", "policy_error"},
        "truncated": termination_cause == "horizon_exhausted",
        "invalid_execution_reason": invalid_reason,
        "reward_sum": reward_sum,
        "policy_calls": predictions,
        "policy_errors": policy_errors,
        "error_taxonomy": error_taxonomy or ERROR_TAXONOMY,
        "inference_wall_time_s": inference_s,
        "action_horizon": action_horizon,
        "inference_timeout_s": inference_timeout_s,
        "observation_identity": task.observation_identity() if hasattr(task, "observation_identity") else None,
        "task_context": task.get_task_context() if hasattr(task, "get_task_context") else None,
        "timing": "synchronous_simulation_paused_during_inference",
        **action_evidence(executed),
    }


__all__ = [
    "ACTION_NAMES",
    "CAMERAS",
    "ERROR_TAXONOMY",
    "OBSERVATION_SOURCES",
    "PolicyOutputError",
    "PolicyTimeoutError",
    "STATE_NAMES",
    "ShakeBenchCameraObservation",
    "ShakeBenchTaskEnv",
    "TASK",
    "action_evidence",
    "declared_deadline_s",
    "episode_outcome",
    "invalid_episode_result",
    "observation_features",
    "rollout_policy",
    "task_description",
    "validated_actions",
]
