"""Stand a fallen mug, wine bottle, boxed drink, or power drill on the moving worktable."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from shakebench.environments.base import ShakeBenchTask
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.scene.config import configure_scene_rendering
from shakebench.scene.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.sensors.providers import POLICY_FIELD_CONTRACT
from shakebench.tasks.catalog import OBJECTS, UPRIGHT_AXIS_COSINE_MIN, TaskSpec, registered_rest_pose
from shakebench.tasks.metrics import pose_twist_in_frame
from shakebench.tasks.privilege import assert_policy_observation_is_clean
from shakebench.tasks.registry import TaskDefinition, register_state_loader, register_task

STATE_SCHEMA = "shakebench.upright.states"
TASK_VERSION = 6
SCHEMA_VERSION = 7
UP_COSINE_MIN = UPRIGHT_AXIS_COSINE_MIN
LINEAR_SPEED_MAX_M_S = 0.05
ANGULAR_SPEED_MAX_RAD_S = 0.2
SUCCESS_HOLD_S = 0.5
# The fallen drill has a measured horizontal radius below 0.159 m, including its visual mesh.
# This box leaves at least 26 mm to either table edge for every yaw. Official gamma=0 plans
# also pass a 10 mm expansion around the box; retain that reach margin when changing it.
START_X_RANGE_M = (-0.14, -0.09)
START_Y_RANGE_M = (-0.115, 0.115)
_SIDE_QUAT = (2**-0.5, 0.0, -(2**-0.5), 0.0)
_MUG_SIDE_QUAT, _MUG_SIDE_LOWER_Z = registered_rest_pose(TaskSpec(object_id="mug"), "side_double_wall")
OBJECT_IDS = ("mug", "wine_bottle", "boxed_drink", "power_drill")
# The narrow Y-Z face rests on -X; the broad X-Z face rests on -Y.
# Support heights are measured from the authored collision vertices.
BOXED_DRINK_REST_POSES = {
    "narrow_side": (_SIDE_QUAT, -0.02610909),
    "broad_side": ((2**-0.5, 2**-0.5, 0.0, 0.0), -0.01677489492),
}
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
    "power_drill": {
        "asset": "objects/ycb_sim/power_drill/model.xml",
        "mass_kg": 0.895,
        "table_mu": 0.35,
        "start_quat_wxyz": (2**-0.5, -(2**-0.5), 0.0, 0.0),
        "start_lower_z_m": 0.001,
        "upright_quat_wxyz": (2**-0.5, 0.0, 0.0, 2**-0.5),
        "upright_lower_z_m": -0.069,
        "instruction": "power drill",
    },
}


def default_state():
    """One explicit, side-lying mug start on the bare worktable."""
    return {
        "state_id": "upright-mug-000",
        "task": {"task_type": "upright", "version": TASK_VERSION, "object_id": "mug"},
        "object_xy_m": [-0.115, 0.0],
        "object_yaw_rad": 0.0,
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Accept only reproducible starts in the Panda-visible tabletop region."""
    required = set(default_state())
    if (
        not isinstance(state, Mapping)
        or not required <= set(state)
        or set(state) - required - {"split", "object_rest_pose"}
    ):
        raise ValueError(f"upright state fields must match version {TASK_VERSION}")
    if (
        not isinstance(state["task"], Mapping)
        or set(state["task"]) != {"task_type", "version", "object_id"}
        or state["task"]["task_type"] != "upright"
        or type(state["task"].get("version")) is not int
        or state["task"]["version"] != TASK_VERSION
        or state["task"]["object_id"] not in UPRIGHT_OBJECTS
    ):
        raise ValueError(f"upright version {TASK_VERSION} supports only mug, wine_bottle, boxed_drink, and power_drill")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be nonempty")
    if "split" in state and state["split"] not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    if "object_rest_pose" in state and (
        state["task"]["object_id"] != "boxed_drink"
        or not isinstance(state["object_rest_pose"], str)
        or state["object_rest_pose"] not in BOXED_DRINK_REST_POSES
    ):
        raise ValueError("object_rest_pose supports only boxed_drink narrow_side or broad_side")
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


def sample_state(rng, *, object_id="mug", split="train", state_id=None, object_rest_pose=None):
    """Sample XY, yaw and a boxed-drink resting face, or use an explicit resting face."""
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
    if object_id == "boxed_drink":
        state["object_rest_pose"] = (
            str(rng.choice(tuple(BOXED_DRINK_REST_POSES))) if object_rest_pose is None else object_rest_pose
        )
    elif object_rest_pose is not None:
        raise ValueError("object_rest_pose supports only boxed_drink")
    return validate_state(state)


def build_state_artifact(*, count, seed, split="train", schema_version=SCHEMA_VERSION):
    """Generate a balanced pool; schema 6 reproduces the legacy, single-face starts."""
    if (
        type(count) is not int
        or count <= 0
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or split not in ("train", "eval")
        or type(schema_version) is not int
        or schema_version not in (6, SCHEMA_VERSION)
    ):
        raise ValueError("invalid upright generator request")
    rng = np.random.default_rng(seed)
    states = []
    for index in range(count):
        object_id = OBJECT_IDS[index % len(OBJECT_IDS)]
        rest_pose = None
        if object_id == "boxed_drink":
            rest_pose = (
                tuple(BOXED_DRINK_REST_POSES)[index // len(OBJECT_IDS) % 2] if schema_version == 7 else "narrow_side"
            )
        pose_suffix = f"-{rest_pose}" if rest_pose is not None and schema_version == 7 else ""
        state = sample_state(
            rng,
            object_id=object_id,
            split=split,
            state_id=f"upright-{object_id}{pose_suffix}-s{seed}-{index:04d}",
            object_rest_pose=rest_pose,
        )
        if schema_version == 6:
            state.pop("object_rest_pose", None)
        states.append(state)
    return {
        "schema_id": STATE_SCHEMA,
        "schema_version": schema_version,
        "generator": {"seed": seed, "split": split},
        "states": states,
    }


def load_states(payload):
    """Load an explicit development state set; it has no official score authority."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] not in (6, SCHEMA_VERSION)
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
    expected = build_state_artifact(
        count=len(states), seed=generator["seed"], split=generator["split"], schema_version=payload["schema_version"]
    )["states"]
    if states != expected:
        raise ValueError("upright states do not match deterministic generation")
    return {"states": states}


class Upright(ShakeBenchTask):
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
        self._init_worktable(self.task_state, physics_profile, geometry_profile, vibration, imu_mode, imu_seed)
        self._success = False
        self._candidate_since = None
        self._conditions = {}
        self._init_robosuite(robots, controller_configs, self.task_state, kwargs)

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
            start_quat, lower_z = spec["start_quat_wxyz"], spec["start_lower_z_m"]
            if self.task_state["task"]["object_id"] == "boxed_drink":
                start_quat, lower_z = BOXED_DRINK_REST_POSES[self.task_state.get("object_rest_pose", "narrow_side")]
            yaw = self.task_state["object_yaw_rad"]
            yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, yaw_quat, np.asarray(start_quat))
            self.sim.data.set_joint_qpos(
                self.object.joints[0],
                [
                    *(self.arena.table_top_abs + [*self.task_state["object_xy_m"], -lower_z + 0.001]),
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


def _state_fingerprint(state):
    """Treat legacy boxed-drink starts and explicit narrow-face starts as the same execution."""
    result = {key: value for key, value in state.items() if key not in {"state_id", "split"}}
    if state["task"]["object_id"] == "boxed_drink":
        result["object_rest_pose"] = state.get("object_rest_pose", "narrow_side")
    return result


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
        fingerprint=_state_fingerprint,
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
