"""StarVLA environment boundary and official WebSocket policy client; no model implementation."""

import time
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np
from PIL import Image

from robosuite.demos.demo_shakebench_oracle_video import _task_close_camera
from robosuite.utils import transform_utils as T
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_outcomes import resolve_termination_cause
from robosuite.utils.shakebench_tasks import TaskSpec

TASK = "Pick up the can from the table and place it in the target tray."
CAMERAS = {"observation.images.main": "task_close", "observation.images.wrist": "robot0_eye_in_hand"}
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "left_finger_qpos", "right_finger_qpos"]


def task_description(state):
    if "task" not in state:
        return {"task_id": "pick_place.can", "instruction": TASK}
    spec = TaskSpec.from_mapping(state["task"])
    names = {
        "food_can": "food can",
        "cookie_box": "cookie box",
        "bread": "bread",
        "light_wood_block": "wooden block",
    }
    surface = "mat" if spec.surface_id == "mat" else "table"
    return {
        "task_id": spec.variant_id,
        "instruction": f"Pick up the {names[spec.object_id]} from the {surface} and place it in the target basket.",
    }


def observation_features(height, width):
    """LeRobot-compatible array schema; language uses its task metadata table."""
    return {
        **{
            key: {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
            for key in CAMERAS
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "observation.table_imu_window": {"dtype": "float32", "shape": (10, 6), "names": None},
        "observation.table_imu_timestamps_s": {"dtype": "float64", "shape": (10,), "names": None},
        "observation.table_imu_dt_s": {"dtype": "float32", "shape": (1,), "names": None},
    }


def validated_actions(value):
    """Accept one action or a nonempty chunk; reject wrong units/ranges before stepping."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("actions must be finite real numbers")
    if array.shape == (7,):
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 7 or len(array) == 0 or np.any(np.abs(array) > 1):
        raise ValueError("expected normalized actions [7] or [N,7], N>0, within [-1,1]")
    return array.astype(np.float32, copy=True)


class StarVLAObservation:
    """Shared camera and sensor extraction for both collection and policy evaluation."""

    def __init__(self, env, *, height=256, width=256, main_camera="task_close"):
        self.env = env
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(main_camera, str) or not main_camera:
            raise ValueError("main_camera must be a non-empty camera name")
        self.height, self.width = height, width
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
        for name in ("table_imu_window", "table_imu_timestamps_s", "table_imu_dt_s"):
            dtype = np.float64 if name == "table_imu_timestamps_s" else np.float32
            result[f"observation.{name}"] = np.atleast_1d(np.asarray(observation[name], dtype=dtype)).copy()
        if result["observation.state"].shape != (8,):
            raise ValueError("observation.state must have shape (8,)")
        if result["observation.table_imu_window"].shape != (10, 6):
            raise ValueError("table_imu_window must have shape (10, 6)")
        if result["observation.table_imu_timestamps_s"].shape != (10,):
            raise ValueError("table_imu_timestamps_s must have shape (10,)")
        if result["observation.table_imu_dt_s"].shape != (1,):
            raise ValueError("table_imu_dt_s must have shape (1,)")
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("non-finite policy observation")
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class StarVLAEnvironment:
    """Gymnasium-style reset/step without a Gym dependency; state fixes all episode seeds.

    Policy receives only the observation dict. Evaluator info and raw env are not policy inputs.
    Recreating the environment on reset deliberately preserves initial-state reproducibility.
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
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping loaded from an authenticated state asset")
        if not isinstance(state.get("state_id"), str) or not state["state_id"]:
            raise ValueError("state must include a non-empty state_id")
        coordinate_key = "object_xy_m" if "task" in state else "object_xy_m"
        xy = np.asarray(state.get(coordinate_key), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"state must include finite {coordinate_key}[2]")
        self.state = deepcopy(state)
        self.description = task_description(state)
        self.gamma, self.horizon = float(gamma), horizon
        self.mode, self.physics_profile = mode, physics_profile
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
        return {
            **observation_features(self.camera_options["height"], self.camera_options["width"]),
            "task": {"dtype": "string", "shape": ()},
            "timestamp": {"dtype": "float32", "shape": (1,)},
        }

    def get_task_context(self):
        context = {
            **self.description,
            "state_id": self.state.get("state_id"),
            "gamma": self.gamma,
            "horizon": self.horizon,
            "action_space": {"shape": [7], "normalized_bounds": [-1.0, 1.0], "names": ACTION_NAMES},
        }
        if self.env is not None:
            context["environment"] = self.env.get_policy_task_context()
        return deepcopy(context)

    def _observation(self, observation):
        return {
            **self.reader.read(observation),
            "task": self.description["instruction"],
            "timestamp": np.array([self.steps / 20], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        from robosuite.scripts.shakebench_gpu_batch import make_environment

        if seed is not None:
            raise ValueError("episode seeds are fixed by the authenticated state asset")
        if options not in (None, {}):
            raise ValueError("StarVLAEnvironment uses the supplied authenticated state; reset options are unsupported")
        self.close()
        try:
            self.env, _ = make_environment(
                self.state, gamma=self.gamma, horizon=self.horizon, mode=self.mode, physics_profile=self.physics_profile
            )
            if self.env.control_freq != 20 or self.env.action_dim != 7:
                raise ValueError("VLA interface requires the 20 Hz, 7D Panda controller")
            self.reader = StarVLAObservation(self.env, **self.camera_options)
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
            raise ValueError("step accepts a single [7] action; execute StarVLA chunks one action at a time")
        actions = validated_actions(action)
        observation, reward, _, _ = self.env.step(actions[0])
        self.steps += 1
        metrics = self.env.get_metrics()
        cause = resolve_termination_cause(
            prior_cause=None,
            task_rule_violation=metrics["max_illegal_penetration_m"]
            >= DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m,
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


def modality_metadata():
    """StarVLA's GR00T loader mapping, alongside standard LeRobot v2.1 metadata."""
    return {
        "video": {name: {"original_key": f"observation.images.{name}"} for name in ("main", "wrist")},
        "state": {"proprio": {"original_key": "observation.state", "start": 0, "end": 8}},
        "action": {
            "osc": {"original_key": "action", "start": 0, "end": 6, "absolute": False},
            "gripper": {"original_key": "action", "start": 6, "end": 7, "absolute": True},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


class StarVLAPolicy:
    """Image/language baseline using StarVLA's official client and server normalization.

    Train with the supplied ShakeBench registry (include_state=false). Proprioception
    and IMU are recorded for external StarVLA research, but are not model inputs here.
    """

    def __init__(self, *, host="127.0.0.1", port=10093):
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host, port)
        try:
            meta = self.client.get_server_metadata()
            if "new_embodiment" not in meta.get("available_unnorm_keys", []):
                raise ValueError("server checkpoint must use the ShakeBench data registry and statistics")
            if meta.get("action_keys") != ["action.osc", "action.gripper"]:
                raise ValueError("server action contract does not match the ShakeBench registry")
            if meta.get("state_keys"):
                raise ValueError("this client requires the image/language ShakeBench registry (include_state=false)")
            self.chunk_size = int(meta["action_chunk_size"])
            if self.chunk_size < 1:
                raise ValueError("invalid server action_chunk_size")
        except BaseException:
            self.client.close()
            raise

    def predict(self, observation):
        # Match the pinned StarVLA loader's PIL resize exactly (including interpolation).
        example = {
            "image": [np.asarray(Image.fromarray(observation[key]).resize((224, 224))) for key in CAMERAS],
            "lang": observation["task"],
        }
        response = self.client.predict_action({"examples": [example], "unnorm_key": "new_embodiment"})
        if response.get("ok") is not True:
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"])
        if actions.shape != (1, self.chunk_size, 7):
            raise ValueError(f"StarVLA returned {actions.shape}; expected (1, {self.chunk_size}, 7)")
        # Server restores the original dataset units: normalized OSC commands, not meters.
        return validated_actions(actions[0])

    def close(self):
        self.client.close()


def rollout_starvla(task, policy, *, action_horizon=1):
    """Execute StarVLA chunks with synchronous, paused-during-inference simulation."""
    if not isinstance(action_horizon, int) or isinstance(action_horizon, bool) or action_horizon < 1:
        raise ValueError("action_horizon must be a positive integer")
    if action_horizon > policy.chunk_size:
        raise ValueError("action_horizon exceeds the checkpoint chunk size")
    observation, _ = task.reset()
    predictions, inference_s, reward_sum = 0, 0.0, 0.0
    while True:
        started = time.perf_counter()
        actions = policy.predict(observation)
        inference_s += time.perf_counter() - started
        predictions += 1
        for action in actions[:action_horizon]:
            observation, reward, terminated, truncated, info = task.step(action)
            reward_sum += reward
            if terminated or truncated:
                return {
                    **info,
                    "terminated": terminated,
                    "truncated": truncated,
                    "reward_sum": reward_sum,
                    "policy_calls": predictions,
                    "inference_wall_time_s": inference_s,
                    "action_horizon": action_horizon,
                    "timing": "synchronous_simulation_paused_during_inference",
                }
