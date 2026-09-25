"""Stand a side-lying can upright on the moving worktable (CPU MuJoCo)."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np

from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.utils.calibration import build_vibration_program
from shakebench.utils.deck import DeckDriver
from shakebench.utils.geometry import (
    DEFAULT_GEOMETRY_PROFILE,
    geometry_scene_path,
    load_geometry_profile,
    worktable_mount,
)
from shakebench.utils.metrics import pose_twist_in_frame
from shakebench.utils.physics import resolve_physics_profile
from shakebench.utils.privilege import assert_policy_observation_is_clean
from shakebench.utils.providers import POLICY_FIELD_CONTRACT, TableIMUProvider
from shakebench.utils.scene import DECK_VISUAL_BODY_NAME, configure_scene_rendering
from shakebench.utils.task_registry import TaskDefinition, register_state_loader, register_task
from shakebench.utils.tasks import (
    OBJECTS,
    UPRIGHT_AXIS_COSINE_MIN,
    TaskSpec,
    make_task_object,
    registered_rest_pose,
)

STATE_SCHEMA = "shakebench.upright.states"
TASK_VERSION = 1
SCHEMA_VERSION = 1
UP_COSINE_MIN = UPRIGHT_AXIS_COSINE_MIN
LINEAR_SPEED_MAX_M_S = 0.05
ANGULAR_SPEED_MAX_RAD_S = 0.2
SUCCESS_HOLD_S = 0.5
START_X_RANGE_M = (-0.20, -0.12)
START_Y_RANGE_M = (-0.10, 0.10)


def default_state():
    """One explicit, side-lying can start on the bare worktable."""
    return {
        "state_id": "upright-can-000",
        "task": {"task_type": "upright", "version": TASK_VERSION, "object_id": "can"},
        "object_xy_m": [-0.16, 0.0],
        "object_yaw_rad": 0.0,
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Accept only reproducible starts in the Panda-visible tabletop region."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError("upright state fields must match version 1")
    if (
        state["task"] != default_state()["task"]
        or not isinstance(state["task"], Mapping)
        or type(state["task"].get("version")) is not int
    ):
        raise ValueError("upright version 1 supports the side-lying can")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be nonempty")
    if "split" in state and state["split"] not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    try:
        xy = np.asarray(state["object_xy_m"], dtype=float)
        yaw = float(state["object_yaw_rad"])
        t0 = float(state["t0_s"])
    except (TypeError, ValueError):
        raise ValueError("object pose and t0_s must be finite numbers") from None
    if (
        xy.shape != (2,)
        or not np.isfinite(xy).all()
        or not START_X_RANGE_M[0] <= xy[0] <= START_X_RANGE_M[1]
        or not START_Y_RANGE_M[0] <= xy[1] <= START_Y_RANGE_M[1]
        or not np.isfinite(yaw)
        or not -np.pi <= yaw <= np.pi
        or not np.isfinite(t0)
        or t0 < 0
    ):
        raise ValueError("upright start pose or time is outside its bounds")
    if isinstance(state["object_yaw_rad"], bool) or isinstance(state["t0_s"], bool):
        raise ValueError("object_yaw_rad and t0_s must be numbers")
    for key in ("excitation_seed", "imu_seed"):
        if type(state[key]) is not int or not 0 <= state[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    result = deepcopy(dict(state))
    result["object_xy_m"], result["object_yaw_rad"], result["t0_s"] = xy.tolist(), yaw, t0
    return result


def sample_state(rng, *, split="train", state_id=None):
    """Sample a recorded XY and yaw without changing the object or task goal."""
    if split not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    state = default_state()
    state.update(
        state_id=state["state_id"] if state_id is None else state_id,
        split=split,
        object_xy_m=[float(rng.uniform(*START_X_RANGE_M)), float(rng.uniform(*START_Y_RANGE_M))],
        object_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
        excitation_seed=int(rng.integers(2**32)),
        imu_seed=int(rng.integers(2**32)),
    )
    return validate_state(state)


def load_states(payload):
    """Load an explicit development state set; it has no official score authority."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported upright state schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("upright states must be a nonempty list")
    generator = payload.get("generator")
    if (
        not isinstance(generator, Mapping)
        or set(generator) != {"seed", "split"}
        or type(generator["seed"]) is not int
        or not 0 <= generator["seed"] < 2**32
        or generator["split"] not in ("train", "eval")
    ):
        raise ValueError("upright states require a valid generator record")
    states = [validate_state(state) for state in payload["states"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate upright state_id")
    rng = np.random.default_rng(generator["seed"])
    expected = [
        sample_state(
            rng,
            split=generator["split"],
            state_id=f"upright-can-s{generator['seed']}-{index:04d}",
        )
        for index in range(len(states))
    ]
    if states != expected:
        raise ValueError("upright states do not match deterministic generation")
    return {"states": states, "authority": {"kind": STATE_SCHEMA, "scoreable": False}}


class Upright(ManipulationEnv):
    """Use the existing can asset, Panda, deck, table and camera scene."""

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
        self.use_object_obs = use_object_obs
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
        self._candidate_since = None
        self._conditions = {}
        eager = kwargs.pop("load_model_on_init", True)
        if kwargs.pop("control_freq", 20) != 20:
            raise ValueError("Upright requires control_freq=20")
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
            raise ValueError("Upright supports exactly one Panda")

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.gripper_body_name = robot.robot_model.eef_name["right"]
        if type(robot.gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("Upright requires PandaGripper")
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
        self.can = make_task_object(TaskSpec(object_id="can"), name="upright_can")
        for name in self.can.contact_geoms:
            geom = self.can.get_obj().find(f".//geom[@name='{name}']")
            geom.set("contype", "2")
            geom.set("conaffinity", "0")
            geom.attrib.pop("density", None)
            geom.set("mass", str(OBJECTS["can"]["mass_kg"]))
        self.model = ManipulationTask(self.arena, [robot.robot_model], [self.can])
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        table_mu = OBJECTS["can"]["table_mu"]
        for name in self.can.contact_geoms:
            for partner in ["table_collision", *robot.gripper["right"].contact_geoms]:
                geom = self.model.worldbody.find(f".//geom[@name='{partner}']")
                geom.set("conaffinity", "2")
                attributes = self.physics_profile.pair_attributes(
                    table_mu if partner == "table_collision" else 1.0,
                    finger_contact=partner != "table_collision",
                )
                ET.SubElement(self.model.contact, "pair", geom1=name, geom2=partner, **attributes)

    def _setup_references(self):
        super()._setup_references()
        self.can_body_id = self.sim.model.body_name2id(self.can.root_body)
        self.table_body_id = self.sim.model.body_name2id(self.arena.worktable_body_name)
        self.can_geom_ids = {self.sim.model.geom_name2id(name) for name in self.can.contact_geoms}
        self.table_geom_id = self.sim.model.geom_name2id("table_collision")
        self.gripper_geom_ids = {
            self.sim.model.geom_name2id(name) for name in self.robots[0].gripper["right"].contact_geoms
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            registered_quat, lower = registered_rest_pose(TaskSpec(object_id="can"), "side")
            yaw = self.task_state["object_yaw_rad"]
            yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw_quat, np.asarray(registered_quat))
            self.sim.data.set_joint_qpos(
                self.can.joints[0],
                [*(self.arena.table_top_abs + [*self.task_state["object_xy_m"], -lower + 0.001]), *quat],
            )
            self._settle()
        self.sim.forward()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=float(self.sim.data.time))
        self._success = False
        self._candidate_since = None
        self._conditions = {}
        self._record_post_physics_metrics(float(self.sim.data.time))

    def _settle(self):
        """Settle the can with the Panda fixed before the episode clock starts."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        for _ in range(round(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qpos_ids], data.qvel[qvel_ids] = fixed_pose, 0
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _update_phase05_provider(self, sample_time_s, policy_step=False):
        self.table_imu_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        data = self.sim.data._data
        table_rotation = data.xmat[self.table_body_id].reshape(3, 3)
        can_rotation = data.xmat[self.can_body_id].reshape(3, 3)
        up_cosine = float(np.dot(table_rotation[:, 2], can_rotation[:, 2]))
        relative = pose_twist_in_frame(self.sim, self.can.root_body, self.arena.worktable_body_name)
        touching_table = False
        touching_gripper = False
        for contact in data.contact[: data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if pair & self.can_geom_ids:
                touching_table |= self.table_geom_id in pair and contact.dist <= 0.001
                touching_gripper |= bool(pair & self.gripper_geom_ids) and contact.dist <= 0.001
        on_table = abs(relative.position_m[0]) < 0.28 and abs(relative.position_m[1]) < 0.25
        still = (
            np.linalg.norm(relative.linear_velocity_m_s) < LINEAR_SPEED_MAX_M_S
            and np.linalg.norm(relative.angular_velocity_rad_s) < ANGULAR_SPEED_MAX_RAD_S
        )
        ready = up_cosine >= UP_COSINE_MIN and touching_table and not touching_gripper and on_table and still
        if ready:
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= SUCCESS_HOLD_S - 1e-12:
                self._success = True
        else:
            self._candidate_since = None
        self._conditions = {
            "up_cosine": up_cosine,
            "touching_table": touching_table,
            "touching_gripper": touching_gripper,
            "on_table": bool(on_table),
            "still": bool(still),
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
            "task_type": "upright",
            "version": TASK_VERSION,
            "object_id": "can",
            "object_asset": OBJECTS["can"]["asset"],
            "scoreable": False,
            "up_cosine_min": UP_COSINE_MIN,
            "success_hold_s": SUCCESS_HOLD_S,
        }

    def get_metrics(self):
        return {"success": {"passed": bool(self._success)}, **deepcopy(self._conditions)}

    def _check_success(self):
        return bool(self._success)

    def reward(self, action=None):
        return float(self._success)


register_task(
    "upright",
    TaskDefinition(
        env_factory=Upright,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"task_state": state},
        describe=lambda state: {"task_id": "upright.can", "instruction": "Stand the fallen can upright on the table."},
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
