"""Stand a fallen mug, wine bottle, boxed drink, or pot on the moving worktable."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np

from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
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
from shakebench.utils.tasks import OBJECTS, UPRIGHT_AXIS_COSINE_MIN, TaskSpec, registered_rest_pose

STATE_SCHEMA = "shakebench.upright.states"
TASK_VERSION = 4
SCHEMA_VERSION = 4
UP_COSINE_MIN = UPRIGHT_AXIS_COSINE_MIN
LINEAR_SPEED_MAX_M_S = 0.05
ANGULAR_SPEED_MAX_RAD_S = 0.2
SUCCESS_HOLD_S = 0.5
START_X_RANGE_M = (-0.16, -0.12)
START_Y_RANGE_M = (-0.10, 0.10)
_SIDE_QUAT = (2**-0.5, 0.0, -(2**-0.5), 0.0)
_MUG_SIDE_QUAT, _MUG_SIDE_LOWER_Z = registered_rest_pose(TaskSpec(object_id="mug"), "side_double_wall")
OBJECT_IDS = ("mug", "wine_bottle", "boxed_drink", "pot")
UPRIGHT_OBJECTS = {
    "mug": {
        "asset": OBJECTS["mug"]["asset"],
        "mass_kg": OBJECTS["mug"]["mass_kg"],
        "table_mu": OBJECTS["mug"]["table_mu"],
        "start_quat_wxyz": _MUG_SIDE_QUAT,
        "start_lower_z_m": _MUG_SIDE_LOWER_Z,
        "upright_quat_wxyz": OBJECTS["mug"]["start_quat_wxyz"],
        "upright_lower_z_m": OBJECTS["mug"]["start_pose_support"][0],
        "instruction": "mug",
    },
    "wine_bottle": {
        "asset": "objects/robocasa/wine/wine_3/model.xml",
        "mass_kg": 1.1,
        "table_mu": 0.25,
        "start_quat_wxyz": _SIDE_QUAT,
        "start_lower_z_m": -0.033682,
        "upright_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "upright_lower_z_m": -0.128000,
        "instruction": "wine bottle",
    },
    "boxed_drink": {
        "asset": "objects/robocasa/boxed_drink/boxed_drink_0/model.xml",
        "mass_kg": 0.20,
        "table_mu": 0.30,
        "start_quat_wxyz": _SIDE_QUAT,
        "start_lower_z_m": -0.02610909,
        "upright_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "upright_lower_z_m": -0.04245595,
        "instruction": "boxed drink",
    },
    "pot": {
        "asset": "objects/robocasa/pot/pot_061/model.xml",
        "scale": 0.62,
        "mass_kg": 0.25,
        "table_mu": 0.35,
        "start_quat_wxyz": (2**-0.5, 2**-0.5, 0.0, 0.0),
        "start_lower_z_m": -0.08641002,
        "upright_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "upright_lower_z_m": -0.00133815,
        "instruction": "pot",
    },
}


def default_state():
    """One explicit, side-lying mug start on the bare worktable."""
    return {
        "state_id": "upright-mug-000",
        "task": {"task_type": "upright", "version": TASK_VERSION, "object_id": "mug"},
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
        raise ValueError("upright state fields must match version 4")
    if (
        not isinstance(state["task"], Mapping)
        or set(state["task"]) != {"task_type", "version", "object_id"}
        or state["task"]["task_type"] != "upright"
        or type(state["task"].get("version")) is not int
        or state["task"]["version"] != TASK_VERSION
        or state["task"]["object_id"] not in UPRIGHT_OBJECTS
    ):
        raise ValueError("upright version 4 supports only mug, wine_bottle, boxed_drink, and pot")
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


def sample_state(rng, *, object_id="mug", split="train", state_id=None):
    """Sample a recorded XY and yaw without changing the object or task goal."""
    if split not in ("train", "eval") or object_id not in UPRIGHT_OBJECTS:
        raise ValueError("split or upright object_id is unsupported")
    state = default_state()
    state["task"]["object_id"] = object_id
    state.update(
        state_id=f"upright-{object_id}-000" if state_id is None else state_id,
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
            object_id=OBJECT_IDS[index % len(OBJECT_IDS)],
            split=generator["split"],
            state_id=f"upright-{OBJECT_IDS[index % len(OBJECT_IDS)]}-s{generator['seed']}-{index:04d}",
        )
        for index in range(len(states))
    ]
    if states != expected:
        raise ValueError("upright states do not match deterministic generation")
    return {"states": states, "authority": {"kind": STATE_SCHEMA, "scoreable": False}}


class Upright(ManipulationEnv):
    """Use the existing Panda, deck, table, camera scene and RoboCasa assets."""

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
        spec = UPRIGHT_OBJECTS[self.task_state["task"]["object_id"]]
        self.object = MujocoXMLObject(
            xml_path_completion(spec["asset"]),
            name=f"upright_{self.task_state['task']['object_id']}",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=spec.get("scale"),
        )
        self._set_object_inertial(spec["mass_kg"])
        for name in self.object.contact_geoms:
            geom = self.object.get_obj().find(f".//geom[@name='{name}']")
            geom.set("contype", "2")
            geom.set("conaffinity", "0")
        self.model = ManipulationTask(self.arena, [robot.robot_model], [self.object])
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        for name in self.object.contact_geoms:
            for partner in ["table_collision", *robot.gripper["right"].contact_geoms]:
                geom = self.model.worldbody.find(f".//geom[@name='{partner}']")
                geom.set("conaffinity", "2")
                attributes = self.physics_profile.pair_attributes(
                    spec["table_mu"] if partner == "table_collision" else 1.0,
                    finger_contact=partner != "table_collision",
                )
                ET.SubElement(self.model.contact, "pair", geom1=name, geom2=partner, **attributes)

    def _set_object_inertial(self, mass_kg):
        """Scale collision-only inertia; visual and region geoms carry no task mass."""
        body = deepcopy(self.object.get_obj())
        names = set(self.object.contact_geoms)
        for parent in body.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("name") not in names:
                    parent.remove(geom)
        probe = ET.Element("mujoco", model="upright_object_probe")
        probe.append(deepcopy(self.object.asset))
        ET.SubElement(probe, "worldbody").append(body)
        model = mujoco.MjModel.from_xml_string(ET.tostring(probe, encoding="unicode"))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.object.root_body)
        if model.body_mass[body_id] <= 0:
            raise ValueError("upright object has no collision-derived mass")
        self.object.get_obj().insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": array_to_string(model.body_ipos[body_id]),
                    "quat": array_to_string(model.body_iquat[body_id]),
                    "mass": str(mass_kg),
                    "diaginertia": array_to_string(model.body_inertia[body_id] * (mass_kg / model.body_mass[body_id])),
                },
            ),
        )

    def _setup_references(self):
        super()._setup_references()
        self.object_body_id = self.sim.model.body_name2id(self.object.root_body)
        self.table_body_id = self.sim.model.body_name2id(self.arena.worktable_body_name)
        self.object_geom_ids = {self.sim.model.geom_name2id(name) for name in self.object.contact_geoms}
        self.table_geom_id = self.sim.model.geom_name2id("table_collision")
        self.gripper_geom_ids = {
            self.sim.model.geom_name2id(name) for name in self.robots[0].gripper["right"].contact_geoms
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            spec = UPRIGHT_OBJECTS[self.task_state["task"]["object_id"]]
            yaw = self.task_state["object_yaw_rad"]
            yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw_quat, np.asarray(spec["start_quat_wxyz"]))
            self.sim.data.set_joint_qpos(
                self.object.joints[0],
                [
                    *(self.arena.table_top_abs + [*self.task_state["object_xy_m"], -spec["start_lower_z_m"] + 0.001]),
                    *quat,
                ],
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
        """Settle the object with the Panda fixed before the episode clock starts."""
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
        object_rotation = data.xmat[self.object_body_id].reshape(3, 3)
        up_cosine = float(np.dot(table_rotation[:, 2], object_rotation[:, 2]))
        relative = pose_twist_in_frame(self.sim, self.object.root_body, self.arena.worktable_body_name)
        touching_table = False
        touching_gripper = False
        for contact in data.contact[: data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if pair & self.object_geom_ids:
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
        object_id = self.task_state["task"]["object_id"]
        return {
            "task_type": "upright",
            "version": TASK_VERSION,
            "object_id": object_id,
            "object_asset": UPRIGHT_OBJECTS[object_id]["asset"],
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
        describe=lambda state: {
            "task_id": f"upright.{state['task']['object_id']}",
            "instruction": f"Stand the fallen {UPRIGHT_OBJECTS[state['task']['object_id']]['instruction']} upright on the table.",
        },
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
