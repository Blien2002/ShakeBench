"""Three-dimensional, non-grasping Push-T on the moving worktable (CPU)."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np

from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.models.objects.push_t import HALF_HEIGHT_M, RECTANGLES, coverage, make_tee, projected_geometry
from shakebench.utils.calibration import build_vibration_program
from shakebench.utils.deck import DeckDriver
from shakebench.utils.geometry import (
    DEFAULT_GEOMETRY_PROFILE,
    geometry_scene_path,
    load_geometry_profile,
    worktable_mount,
)
from shakebench.utils.physics import resolve_physics_profile
from shakebench.utils.privilege import assert_policy_observation_is_clean
from shakebench.utils.providers import POLICY_FIELD_CONTRACT, TableIMUProvider
from shakebench.utils.scene import DECK_VISUAL_BODY_NAME, configure_scene_rendering
from shakebench.utils.task_registry import TaskDefinition, register_state_loader, register_task

STATE_SCHEMA = "shakebench.push_t.states"


def default_state():
    """Explicit stationary start, in metres relative to the tabletop."""
    return {
        "state_id": "push-t-000",
        "task": {"task_type": "push_t", "version": 1},
        "object_xy_m": [-0.14, -0.12],
        "object_yaw_rad": 0.0,
        "target_xy_m": [0.02, 0.10],
        "target_yaw_rad": 0.0,
        "coverage_threshold": 0.95,
        "success_hold_s": 0.5,
        "lift_height_m": 0.005,
        "lift_duration_s": 0.2,
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Reject hidden fields, invalid geometry, thresholds and random seeds."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError("push_t state fields must match version 1")
    if state["task"] != {"task_type": "push_t", "version": 1} or type(state["task"]["version"]) is not int:
        raise ValueError("expected push_t task version 1")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be nonempty")
    result = deepcopy(dict(state))
    for key in (
        "object_yaw_rad",
        "target_yaw_rad",
        "coverage_threshold",
        "success_hold_s",
        "lift_height_m",
        "lift_duration_s",
        "t0_s",
    ):
        if isinstance(state[key], bool) or not isinstance(state[key], (int, float)) or not np.isfinite(state[key]):
            raise ValueError(f"{key} must be finite")
        result[key] = float(state[key])
    if not 0 < result["coverage_threshold"] <= 1 or result["t0_s"] < 0:
        raise ValueError("invalid coverage threshold or time offset")
    if any(result[k] <= 0 for k in ("success_hold_s", "lift_height_m", "lift_duration_s")):
        raise ValueError("hold durations and lift height must be positive")
    for prefix in ("object", "target"):
        xy = np.asarray(state[f"{prefix}_xy_m"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("positions must be finite 2D coordinates")
        yaw = result[f"{prefix}_yaw_rad"]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        polygons, _ = projected_geometry(np.r_[xy, HALF_HEIGHT_M], rotation)
        if any(np.any(np.abs(p) > [0.325, 0.30]) for p in polygons):
            raise ValueError("the entire T must lie on the tabletop")
        result[f"{prefix}_xy_m"] = xy.tolist()
    for key in ("excitation_seed", "imu_seed"):
        if type(state[key]) is not int or not 0 <= state[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    return result


def sample_state(rng, *, state_id=None):
    """Sample separated starts, recording every source of randomness."""
    state = default_state()
    state.update(
        object_xy_m=rng.uniform([-0.20, -0.19], [-0.08, -0.10]).tolist(),
        target_xy_m=rng.uniform([-0.04, 0.08], [0.08, 0.18]).tolist(),
        object_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
        target_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
        excitation_seed=int(rng.integers(2**32)),
        imu_seed=int(rng.integers(2**32)),
    )
    if state_id is not None:
        state["state_id"] = state_id
    return validate_state(state)


def load_states(payload):
    """Validate an explicit development state asset."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("unsupported push_t schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("states must be a nonempty list")
    states = [validate_state(s) for s in payload["states"]]
    if len({s["state_id"] for s in states}) != len(states):
        raise ValueError("duplicate state_id")
    return {"states": states, "authority": {"kind": STATE_SCHEMA, "scoreable": False}}


class PushT(ManipulationEnv):
    """Push the wooden T onto the matte green tabletop target without lifting."""

    def __init__(
        self,
        robots="Panda",
        *,
        task_state=None,
        physics_profile="official",
        geometry_profile=DEFAULT_GEOMETRY_PROFILE,
        vibration=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        use_object_obs=True,
        controller_configs=None,
        **kwargs,
    ):
        self.task_state = validate_state(default_state() if task_state is None else task_state)
        self.physics_profile = resolve_physics_profile(physics_profile)
        self.geometry_profile = load_geometry_profile(geometry_profile)
        self.worktable_mount = worktable_mount(geometry_profile)
        self.scene_path = geometry_scene_path(geometry_profile)
        self.table_imu_provider = TableIMUProvider(
            seed=self.task_state["imu_seed"] if imu_seed is None else imu_seed, imu_mode=imu_mode
        )
        self.deck_driver = DeckDriver(
            trajectory=build_vibration_program(
                vibration
                if vibration is not None
                else {
                    "mode": "multisine_v1",
                    "gamma": 0.0,
                    "seed": self.task_state["excitation_seed"],
                    "t0_s": self.task_state["t0_s"],
                }
            ),
            config=self.physics_profile.deck_driver_config(),
            body_handles={"isolated_worktable": "worktable", "deck_visual": DECK_VISUAL_BODY_NAME},
            required_roles=("isolated_worktable", "deck_visual"),
        )
        self._success = False
        self._violation = False
        self._lift_since = None
        self._candidate_since = None
        self._conditions = {}
        self.use_object_obs = use_object_obs
        eager = kwargs.pop("load_model_on_init", True)
        if kwargs.pop("control_freq", 20) != 20:
            raise ValueError("PushT requires control_freq=20")
        for key, value in {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "initialization_noise": None,
            "hard_reset": False,
            "seed": self.task_state["excitation_seed"],
        }.items():
            kwargs.setdefault(key, value)
        super().__init__(
            robots=robots,
            controller_configs=controller_configs or load_composite_controller_config(robot="Panda"),
            base_types=self.geometry_profile["mount_type"],
            control_freq=20,
            model_timestep=self.physics_profile.model_timestep_s,
            load_model_on_init=False,
            **kwargs,
        )
        self.set_xml_processor(self.physics_profile.process_xml)
        self.deck_driver.install(self)
        self.add_post_physics_step_hook(self._record_post_physics_metrics)
        self.add_post_physics_step_hook(self._update_phase05_provider)
        self.load_model_on_init = eager
        if eager:
            self.reset()

    def _check_robot_configuration(self, robots):
        if (list(robots) if isinstance(robots, (list, tuple)) else [robots]) != ["Panda"]:
            raise ValueError("PushT supports exactly one Panda")

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.robot_base_body_name = robot.robot_model.root_body
        self.gripper_body_name = robot.robot_model.eef_name["right"]
        if type(robot.gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("PushT requires PandaGripper")
        self.arena = ShakeBenchArena(
            table_offset=self.geometry_profile["table_top_pos_m"],
            isolator_config=self.physics_profile.isolator_config(),
            worktable_mount=self.worktable_mount,
            scene_config=self.scene_path,
        )
        self.worktable_body_name = self.arena.worktable_body_name
        support = (
            ET.parse(xml_path_completion(self.geometry_profile["robot_support_mjcf"]))
            .getroot()
            .find("./worldbody/body[@name='robot_support']")
        )
        self.arena.worldbody.append(support)
        target = ET.SubElement(
            self.arena.worktable_body,
            "body",
            name="push_t_target",
            pos=array_to_string([*self.task_state["target_xy_m"], self.arena.table_half_size[2]]),
            quat=array_to_string(
                [np.cos(self.task_state["target_yaw_rad"] / 2), 0, 0, np.sin(self.task_state["target_yaw_rad"] / 2)]
            ),
        )
        ET.SubElement(
            self.arena.asset,
            "material",
            name="push_t_green",
            rgba="0.32 0.58 0.38 1",
            emission="0",
            specular="0.05",
            shininess="0.1",
        )
        # A 0.1 mm visual decal: no collision, no mass, rigidly attached to the table.
        for index, (center, size) in enumerate(RECTANGLES):
            ET.SubElement(
                target,
                "geom",
                name=f"push_t_target_{index}",
                type="box",
                pos=array_to_string([*center, 0.00005]),
                size=array_to_string([*size, 0.00005]),
                material="push_t_green",
                group="1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        self.tee = make_tee()
        self.model = ManipulationTask(self.arena, [robot.robot_model], [self.tee])
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        pads = robot.gripper["right"].important_geoms
        finger_names = pads["left_fingerpad"] + pads["right_fingerpad"]
        for geom in self.tee.contact_geoms:
            for partner in ["table_collision", *finger_names]:
                finger = partner in finger_names
                attributes = self.physics_profile.pair_attributes(
                    float(self.physics_profile.contact["sliding_mu"]["finger_object" if finger else "table_object"]),
                    finger_contact=finger,
                )
                ET.SubElement(self.model.contact, "pair", geom1=geom, geom2=partner, **attributes)

    def _setup_references(self):
        super()._setup_references()
        self.tee_body_id = self.sim.model.body_name2id(self.tee.root_body)
        self.target_body_id = self.sim.model.body_name2id("push_t_target")
        self.tee_geom_ids = {self.sim.model.geom_name2id(name) for name in self.tee.contact_geoms}
        self.table_geom_id = self.sim.model.geom_name2id("table_collision")

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            yaw = self.task_state["object_yaw_rad"]
            self.sim.data.set_joint_qpos(
                self.tee.joints[0],
                [
                    *(self.arena.table_top_abs + [*self.task_state["object_xy_m"], HALF_HEIGHT_M + 0.001]),
                    np.cos(yaw / 2),
                    0,
                    0,
                    np.sin(yaw / 2),
                ],
            )
            self._settle()
        self.sim.forward()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=float(self.sim.data.time))
        self._success = self._violation = False
        self._candidate_since = self._lift_since = None
        self._conditions = {}
        self._record_post_physics_metrics(float(self.sim.data.time))

    def _settle(self):
        """Settle the support and T block with robot joints fixed, before the clock starts."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        # Distinguish translational drift (m/s) from rotational drift (rad/s).
        velocity_limits = np.full(model.nv, 0.001)
        adr = model.jnt_dofadr[self.sim.model.joint_name2id(self.tee.joints[0])]
        velocity_limits[adr + 3 : adr + 6] = 0.01
        quiet = 0
        stride = max(1, round(0.1 / model.opt.timestep))
        previous_pose = data.qpos.copy()
        mean_velocity = np.zeros(model.nv)
        for step in range(round(5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qpos_ids], data.qvel[qvel_ids] = fixed_pose, 0
            if (step + 1) % stride:
                continue
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("non-finite T block reset")
            # Use pose drift over 100 ms, not instantaneous soft-contact jitter.
            mujoco.mj_differentiatePos(model, mean_velocity, stride * model.opt.timestep, previous_pose, data.qpos)
            previous_pose[:] = data.qpos
            quiet = quiet + stride if np.all(np.abs(mean_velocity) < velocity_limits) else 0
            if quiet * model.opt.timestep >= 0.2:
                break
        else:
            raise RuntimeError("T block reset did not settle within 5 simulation seconds")
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _update_phase05_provider(self, sample_time_s, policy_step=False):
        self.table_imu_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        data = self.sim.data._data
        target_rotation = data.xmat[self.target_body_id].reshape(3, 3)
        position = target_rotation.T @ (data.xpos[self.tee_body_id] - data.xpos[self.target_body_id])
        rotation = target_rotation.T @ data.xmat[self.tee_body_id].reshape(3, 3)
        polygons, lowest = projected_geometry(position, rotation)
        overlap = coverage(polygons)
        supported = any(
            c.dist <= 0.001
            and (
                (c.geom1 in self.tee_geom_ids and c.geom2 == self.table_geom_id)
                or (c.geom2 in self.tee_geom_ids and c.geom1 == self.table_geom_id)
            )
            for c in data.contact[: data.ncon]
        )
        lifted = lowest > self.task_state["lift_height_m"]
        if lifted:
            if self._lift_since is None:
                self._lift_since = sample_time_s
            if sample_time_s - self._lift_since >= self.task_state["lift_duration_s"] - 1e-12:
                self._violation = True
        else:
            self._lift_since = None
        ready = overlap >= self.task_state["coverage_threshold"] and supported and not lifted
        if ready and not self._violation:
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= self.task_state["success_hold_s"] - 1e-12:
                self._success = True
        else:
            self._candidate_since = None
        if self._violation:
            self._success = False
        self._conditions = {
            "coverage": overlap,
            "lowest_height_m": lowest,
            "supported": bool(supported),
            "lift_elapsed_s": 0.0 if self._lift_since is None else sample_time_s - self._lift_since,
        }

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        obs.update(self.table_imu_provider.observation(self.sim))
        assert_policy_observation_is_clean(obs)
        return obs

    @property
    def policy_observation_keys(self):
        return tuple(self.observation_contract())

    def observation_contract(self):
        contract = {key: dict(POLICY_FIELD_CONTRACT[key]) for key in self.table_imu_provider.policy_keys}
        for key, shape, units, frame in (
            ("robot0_joint_pos", (7,), "rad", "robot_base"),
            ("robot0_joint_vel", (7,), "rad/s", "robot_base"),
            ("robot0_gripper_qpos", (2,), "m", "gripper"),
            ("robot0_gripper_qvel", (2,), "m/s", "gripper"),
        ):
            contract[key] = {"shape": shape, "dtype": "float64", "units": units, "frame": frame}
        return {
            key: {
                **value,
                "time": "delayed acquisition window" if key.startswith("table_imu") else "current control step",
            }
            for key, value in contract.items()
        }

    def get_policy_task_context(self):
        return {
            "task_type": "push_t",
            "version": 1,
            "scoreable": False,
            **{
                key: self.task_state[key]
                for key in ("coverage_threshold", "success_hold_s", "lift_height_m", "lift_duration_s")
            },
        }

    def get_metrics(self):
        return {
            "success": {"passed": bool(self._success)},
            "task_rule_violation": bool(self._violation),
            **deepcopy(self._conditions),
        }

    def _check_success(self):
        return bool(self._success)

    def reward(self, action=None):
        return float(self._success)


register_task(
    "push_t",
    TaskDefinition(
        env_factory=PushT,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"task_state": state},
        describe=lambda state: {
            "task_id": "push_t",
            "instruction": "Push the light wooden T onto the matte green T target without lifting it.",
        },
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
