"""RLBench-inspired middle/left/right wine placement on the moving worktable.

Import this module to register the CPU task and its explicit development states.
The rack is native MuJoCo geometry; both bottles reuse packaged RoboCasa assets.
"""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import mujoco
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from shakebench.environments.base import ShakeBenchTask
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.models.objects.water_inertia import WATER_INERTIA_PROFILE, bottom_weighted_inertial
from shakebench.models.objects.water_support import WATER_SUPPORT_PROFILE, add_flat_water_sole
from shakebench.models.objects.wine_rack import BOTTLE_AXIS_HEIGHT_M, SLOT_Y_M, VISUAL_REVISION, add_wine_rack
from shakebench.scene.config import configure_scene_rendering
from shakebench.scene.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.sensors.providers import POLICY_FIELD_CONTRACT
from shakebench.tasks.metrics import collision_support_points_in_frame, pose_twist_in_frame
from shakebench.tasks.privilege import assert_policy_observation_is_clean
from shakebench.tasks.registry import TaskDefinition, register_state_loader, register_task

TASK_TYPE = "wine_rack"
TASK_VERSION = 10
STATE_SCHEMA = "shakebench.wine_rack.states"
SCHEMA_VERSION = 10
SOURCE_URL = "https://github.com/MohitShridhar/RLBench/blob/peract/rlbench/tasks/place_wine_at_rack_location.py"
# Match RLBench's variation indices. Left is +y when facing the rack from the Panda base.
LOCATIONS = ("middle", "left", "right")
BOTTLE_ASSETS = {"wine_3": "wine/wine_3", "bottled_water_16": "bottled_water/bottled_water_16"}
BOTTLE_VARIANTS = tuple(BOTTLE_ASSETS)
RACK_XY_M = (0.045, 0.0)
# Tabletop-relative centre bounds; the full rack stays clear of starts and table edges.
RACK_XY_MIN_M = (0.035, -0.08)
RACK_XY_MAX_M = (0.135, 0.08)
RACK_YAW_LIMIT_RAD = np.deg2rad(10.0)
RACK_HALF_DEPTH_M = 0.15
RACK_MIN_BOTTLE_Z_M = 0.020
RACK_HEIGHT_M = 0.125
# Approximate hardwood-frame mass and inertia about its center of mass.
RACK_MASS_KG = 0.9
RACK_COM_Z_M = 0.04
RACK_INERTIA_KG_M2 = (0.008, 0.005, 0.012)
SLOT_HALF_WIDTH_M = 0.046
CONTAINMENT_TOLERANCE_M = 0.001
SUCCESS_HOLD_S = 0.5
BOTTLE_MASSES_KG = {"wine_3": 1.1, "bottled_water_16": 0.7}
START_X_RANGE_M = (-0.25, -0.17)
# Keep tall bottle starts clear of the default Panda wrist.
START_ABS_Y_RANGE_M = (0.15, 0.22)


def default_state(location="middle", bottle_variant="wine_3"):
    """An upright bottle start; rack geometry and all randomness are explicit."""
    if not isinstance(location, str) or location not in LOCATIONS:
        raise ValueError("location must be middle, left or right")
    if bottle_variant not in BOTTLE_VARIANTS:
        raise ValueError(f"bottle_variant must be one of {BOTTLE_VARIANTS}")
    return {
        "state_id": f"wine-rack-{location}-000",
        "task": {"task_type": TASK_TYPE, "version": TASK_VERSION},
        "location": location,
        "bottle_variant": bottle_variant,
        "object_xy_m": [-0.21, -0.19],
        "rack_xy_m": list(RACK_XY_M),
        "rack_yaw_rad": 0.0,
        "object_yaw_rad": 0.0,
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Accept recorded bottle/rack starts clear of each other and table edges."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError(f"wine rack state fields must match version {TASK_VERSION}")
    if state["task"] != {"task_type": TASK_TYPE, "version": TASK_VERSION} or type(state["task"]["version"]) is not int:
        raise ValueError(f"expected {TASK_TYPE} task version {TASK_VERSION}")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be nonempty")
    if not isinstance(state["location"], str) or state["location"] not in LOCATIONS:
        raise ValueError("location must be middle, left or right")
    if state["bottle_variant"] not in BOTTLE_VARIANTS:
        raise ValueError(f"bottle_variant must be one of {BOTTLE_VARIANTS}")
    if "split" in state and state["split"] not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    for key in ("object_xy_m", "rack_xy_m"):
        xy = state[key]
        if not isinstance(xy, (list, tuple, np.ndarray)) or np.shape(xy) != (2,):
            raise ValueError(f"{key} must contain two finite numbers")
    for key, values in (
        ("object_xy_m", state["object_xy_m"]),
        ("rack_xy_m", state["rack_xy_m"]),
        ("object_yaw_rad", [state["object_yaw_rad"]]),
        ("rack_yaw_rad", [state["rack_yaw_rad"]]),
        ("t0_s", [state["t0_s"]]),
    ):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numbers")
    xy = np.asarray(state["object_xy_m"], dtype=float)
    rack_xy = np.asarray(state["rack_xy_m"], dtype=float)
    if np.any(rack_xy < RACK_XY_MIN_M) or np.any(rack_xy > RACK_XY_MAX_M):
        raise ValueError("rack_xy_m must lie in the rack-safe tabletop region")
    if (
        not START_X_RANGE_M[0] <= xy[0] <= START_X_RANGE_M[1]
        or not START_ABS_Y_RANGE_M[0] <= abs(xy[1]) <= START_ABS_Y_RANGE_M[1]
    ):
        raise ValueError("bottle start must lie in the rack-clear tabletop region")
    if not -RACK_YAW_LIMIT_RAD <= state["rack_yaw_rad"] <= RACK_YAW_LIMIT_RAD:
        raise ValueError("rack_yaw_rad must lie within +/-10 degrees")
    if not -np.pi <= state["object_yaw_rad"] <= np.pi or state["t0_s"] < 0:
        raise ValueError("object yaw or t0_s is outside its bounds")
    for key in ("excitation_seed", "imu_seed"):
        if type(state[key]) is not int or not 0 <= state[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    result = deepcopy(dict(state))
    result.update(
        object_xy_m=xy.tolist(),
        rack_xy_m=rack_xy.tolist(),
        rack_yaw_rad=float(state["rack_yaw_rad"]),
        object_yaw_rad=float(state["object_yaw_rad"]),
        t0_s=float(state["t0_s"]),
    )
    return result


def sample_state(rng, *, location=None, bottle_variant=None, split="train", state_id=None):
    """Sample recorded bottle poses and seeds, preserving the requested rack location."""
    state = default_state(LOCATIONS[int(rng.integers(len(LOCATIONS)))] if location is None else location)
    state.update(
        split=split,
        object_xy_m=[
            float(rng.uniform(*START_X_RANGE_M)),
            float(rng.choice((-1, 1)) * rng.uniform(*START_ABS_Y_RANGE_M)),
        ],
        rack_xy_m=rng.uniform(RACK_XY_MIN_M, RACK_XY_MAX_M).tolist(),
        rack_yaw_rad=float(rng.uniform(-RACK_YAW_LIMIT_RAD, RACK_YAW_LIMIT_RAD)),
        object_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
        excitation_seed=int(rng.integers(2**32)),
        imu_seed=int(rng.integers(2**32)),
    )
    state["bottle_variant"] = rng.choice(BOTTLE_VARIANTS) if bottle_variant is None else bottle_variant
    if state_id is not None:
        state["state_id"] = state_id
    return validate_state(state)


def load_states(payload):
    """Validate complete explicit development states, without score authority."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported wine rack state schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("states must be a nonempty list")
    states = [validate_state(state) for state in payload["states"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate state_id")
    return {"states": states}


class WineRack(ShakeBenchTask):
    """Lay the selected bottle across the instructed body/neck cradles and release it."""

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
        self._randomize_rack = task_state is None
        self._placement_rng = np.random.default_rng(kwargs.get("seed"))
        self.task_state = validate_state(default_state() if task_state is None else task_state)
        self._init_worktable(self.task_state, physics_profile, geometry_profile, vibration, imu_mode, imu_seed)
        self.use_object_obs = use_object_obs
        self._success, self._candidate_since, self._conditions = False, None, {}
        self._init_robosuite(robots, controller_configs, self.task_state, kwargs)

    def reset(self):
        """Sample a new rack pose, or replay the supplied state's recorded pose."""
        if not self.deterministic_reset and self._randomize_rack:
            self.task_state["rack_xy_m"] = self._placement_rng.uniform(RACK_XY_MIN_M, RACK_XY_MAX_M).tolist()
            self.task_state["rack_yaw_rad"] = float(
                self._placement_rng.uniform(-RACK_YAW_LIMIT_RAD, RACK_YAW_LIMIT_RAD)
            )
        return super().reset()

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.robot_base_body_name = robot.robot_model.root_body
        self.gripper_body_name = robot.robot_model.eef_name["right"]
        if type(robot.gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("WineRack requires PandaGripper")
        self.arena = ShakeBenchArena(
            table_offset=self.geometry_profile["table_top_pos_m"],
            isolator_config=self.physics_profile.isolator_config(),
            worktable_mount=self.worktable_mount,
            scene_config=self.scene_path,
        )
        self.worktable_body_name = self.arena.worktable_body_name
        self.arena.worldbody.append(
            ET.parse(xml_path_completion(self.geometry_profile["robot_support_mjcf"]))
            .getroot()
            .find("./worldbody/body[@name='robot_support']")
        )
        self._add_rack()
        self.bottle = MujocoXMLObject(
            xml_path_completion(f"objects/robocasa/{BOTTLE_ASSETS[self.task_state['bottle_variant']]}/model.xml"),
            name="wine_bottle",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        # Derive inertia from the collision meshes, excluding visual and region geoms.
        body = deepcopy(self.bottle.get_obj())
        for parent in body.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("name") not in self.bottle.contact_geoms:
                    parent.remove(geom)
        probe = ET.Element("mujoco")
        probe.append(deepcopy(self.bottle.asset))
        ET.SubElement(probe, "worldbody").append(body)
        model = mujoco.MjModel.from_xml_string(ET.tostring(probe, encoding="unicode"))
        bid = model.body(self.bottle.root_body).id
        self.bottle.get_obj().insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": array_to_string(model.body_ipos[bid]),
                    "quat": array_to_string(model.body_iquat[bid]),
                    "mass": str(BOTTLE_MASSES_KG[self.task_state["bottle_variant"]]),
                    "diaginertia": array_to_string(
                        model.body_inertia[bid]
                        * BOTTLE_MASSES_KG[self.task_state["bottle_variant"]]
                        / model.body_mass[bid]
                    ),
                },
            ),
        )
        if self.task_state["bottle_variant"] == "bottled_water_16":
            attributes = bottom_weighted_inertial(
                model.body_ipos[bid],
                model.body_inertia[bid] * BOTTLE_MASSES_KG["bottled_water_16"] / model.body_mass[bid],
                model.body_iquat[bid],
                BOTTLE_MASSES_KG["bottled_water_16"],
                float(self.bottle.bottom_offset[2]),
            )
            inertial = self.bottle.get_obj().find("inertial")
            inertial.attrib.clear()
            inertial.attrib.update(attributes)
            probe_data = mujoco.MjData(model)
            mujoco.mj_forward(model, probe_data)
            points = collision_support_points_in_frame(
                model, self.bottle.root_body, self.bottle.contact_geoms, self.bottle.root_body, data=probe_data
            )
            add_flat_water_sole(self.bottle, float(points[:, 2].min()))
        for geom in self.bottle.get_obj().iter("geom"):
            if geom.get("name") not in self.bottle.contact_geoms:
                geom.set("mass", "0")
        self.model = ManipulationTask(self.arena, [robot.robot_model], [self.bottle])
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        rack_contact = self.physics_profile.pair_attributes(self.physics_profile.contact["sliding_mu"]["table_object"])
        rack_friction = rack_contact["friction"].split()
        rack_friction[1] = rack_friction[0]
        rack_contact["friction"] = " ".join(rack_friction)
        for geom in self.rack_geom_names:
            ET.SubElement(self.model.contact, "pair", geom1=geom, geom2="table_collision", **rack_contact)
        pads = robot.gripper["right"].important_geoms
        finger_names = pads["left_fingerpad"] + pads["right_fingerpad"]
        for geom in self.bottle.contact_geoms:
            for partner in ["table_collision", *self.rack_geom_names, *finger_names]:
                finger = partner in finger_names
                attributes = self.physics_profile.pair_attributes(
                    float(self.physics_profile.contact["sliding_mu"]["finger_object" if finger else "table_object"]),
                    finger_contact=finger,
                )
                if partner in self.rack_geom_names:
                    # MuJoCo pairs have two sliding axes; both timber directions use mu=0.30.
                    friction = attributes["friction"].split()
                    friction[1] = friction[0]
                    attributes["friction"] = " ".join(friction)
                ET.SubElement(self.model.contact, "pair", geom1=geom, geom2=partner, **attributes)

    def _add_rack(self):
        """Place the free oak frame and its three pairs of curved bottle cradles."""
        rack = ET.SubElement(
            self.arena.worldbody,
            "body",
            name="wine_rack",
            pos=array_to_string(self.arena.table_top_abs + [*self.task_state["rack_xy_m"], 0]),
            quat=array_to_string(
                [np.cos(self.task_state["rack_yaw_rad"] / 2), 0, 0, np.sin(self.task_state["rack_yaw_rad"] / 2)]
            ),
        )
        ET.SubElement(rack, "freejoint", name="wine_rack_free")
        ET.SubElement(
            rack,
            "inertial",
            pos=f"0 0 {RACK_COM_Z_M}",
            mass=str(RACK_MASS_KG),
            diaginertia=array_to_string(RACK_INERTIA_KG_M2),
        )
        self.rack_geom_names, self.slot_support_names = add_wine_rack(self.arena.asset, rack)
        for location, y in SLOT_Y_M.items():
            ET.SubElement(
                rack,
                "site",
                name=f"wine_rack_{location}_target",
                pos=array_to_string([0, y, BOTTLE_AXIS_HEIGHT_M]),
                size="0.005",
                rgba="0 0 0 0",
            )

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self.bottle_body_id = model.body_name2id(self.bottle.root_body)
        self.rack_body_id = model.body_name2id("wine_rack")
        self.bottle_geom_ids = {model.geom_name2id(name) for name in self.bottle.contact_geoms}
        self.slot_support_ids = {
            location: {rail: {model.geom_name2id(name) for name in names} for rail, names in rails.items()}
            for location, rails in self.slot_support_names.items()
        }
        self.robot_geom_ids = {
            model.geom_name2id(name) for name in model.geom_names if name.startswith(("robot0_", "gripper0_"))
        }
        self.sim.forward()
        self.bottle_points = collision_support_points_in_frame(
            self.sim, self.bottle.root_body, self.bottle.contact_geoms, self.bottle.root_body
        )

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            rack_yaw = self.task_state["rack_yaw_rad"]
            self.sim.data.set_joint_qpos(
                "wine_rack_free",
                [
                    *(self.arena.table_top_abs + [*self.task_state["rack_xy_m"], 0]),
                    np.cos(rack_yaw / 2),
                    0,
                    0,
                    np.sin(rack_yaw / 2),
                ],
            )
            yaw = self.task_state["object_yaw_rad"]
            self.sim.data.set_joint_qpos(
                self.bottle.joints[0],
                [
                    *(
                        self.arena.table_top_abs
                        + [*self.task_state["object_xy_m"], -self.bottle_points[:, 2].min() + 0.001]
                    ),
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
        self._success, self._candidate_since, self._conditions = False, None, {}
        self._record_post_physics_metrics(float(self.sim.data.time))

    def _settle(self):
        """Settle the bottle with the Panda fixed before starting the episode clock."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        for _ in range(round(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qpos_ids], data.qvel[qvel_ids] = fixed_pose, 0
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("non-finite wine rack reset")
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        model, data = self.sim.model._model, self.sim.data._data
        rack_rotation = data.xmat[self.rack_body_id].reshape(3, 3)
        bottle_rotation = data.xmat[self.bottle_body_id].reshape(3, 3)
        points = (
            self.bottle_points @ bottle_rotation.T + data.xpos[self.bottle_body_id] - data.xpos[self.rack_body_id]
        ) @ rack_rotation
        location = self.task_state["location"]
        points[:, 1] -= SLOT_Y_M[location]
        lower = np.array([-RACK_HALF_DEPTH_M, -SLOT_HALF_WIDTH_M, RACK_MIN_BOTTLE_Z_M])
        # The broad water-bottle shoulder sits above the divider crown while both rails carry its weight.
        shoulder_clearance = 0.015 if self.task_state["bottle_variant"] == "bottled_water_16" else 0.0
        upper = np.array([RACK_HALF_DEPTH_M, SLOT_HALF_WIDTH_M, RACK_HEIGHT_M + shoulder_clearance])
        contained = bool(
            np.all(points >= lower - CONTAINMENT_TOLERANCE_M) and np.all(points <= upper + CONTAINMENT_TOLERANCE_M)
        )
        support_forces, released = {"body": 0.0, "neck": 0.0}, True
        force = np.zeros(6)
        for index, contact in enumerate(data.contact[: data.ncon]):
            if contact.geom1 in self.bottle_geom_ids:
                other = contact.geom2
            elif contact.geom2 in self.bottle_geom_ids:
                other = contact.geom1
            else:
                continue
            if other in self.robot_geom_ids and contact.dist <= 0.001:
                released = False
            for rail, ids in self.slot_support_ids[location].items():
                if other in ids:
                    mujoco.mj_contactForce(model, data, index, force)
                    support_forces[rail] += max(0.0, float(force[0]))
        axis = rack_rotation.T @ bottle_rotation[:, 2]
        self._conditions = {
            "inside_selected_slot": contained,
            "aligned": bool(axis[0] >= np.cos(np.deg2rad(15))),
            "supported": all(force > 0.05 for force in support_forces.values()),
            "released": released,
        }
        if all(self._conditions.values()):
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= SUCCESS_HOLD_S - 1e-12:
                self._success = True
        else:
            self._candidate_since = None

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        obs.update(self.table_imu_provider.observation(self.sim))
        if self.use_object_obs:
            for prefix, body in (("wine_bottle", self.bottle.root_body), ("wine_rack", "wine_rack")):
                pose = pose_twist_in_frame(self.sim, body, self.robot_base_body_name)
                obs[f"{prefix}_pos_robot_base"] = pose.position_m
                obs[f"{prefix}_quat_wxyz_robot_base"] = pose.quaternion_wxyz
        assert_policy_observation_is_clean(obs)
        return obs

    def observation_contract(self):
        contract = {key: dict(POLICY_FIELD_CONTRACT[key]) for key in self.table_imu_provider.policy_keys}
        fields = [
            ("robot0_joint_pos", (7,), "rad", "robot_base"),
            ("robot0_joint_vel", (7,), "rad/s", "robot_base"),
            ("robot0_gripper_qpos", (2,), "m", "gripper"),
            ("robot0_gripper_qvel", (2,), "m/s", "gripper"),
        ]
        if self.use_object_obs:
            for prefix in ("wine_bottle", "wine_rack"):
                fields.extend(
                    [
                        (f"{prefix}_pos_robot_base", (3,), "m", "robot_base"),
                        (f"{prefix}_quat_wxyz_robot_base", (4,), "unit quaternion wxyz", "robot_base"),
                    ]
                )
        for key, shape, units, frame in fields:
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
            "task_type": TASK_TYPE,
            "version": TASK_VERSION,
            "scoreable": False,
            "location": self.task_state["location"],
            "bottle_variant": self.task_state["bottle_variant"],
            "bottle_physics": {
                "revision": (
                    "water_physics_v2" if self.task_state["bottle_variant"] == "bottled_water_16" else "mesh_inertia_v1"
                ),
                "mass_kg": BOTTLE_MASSES_KG[self.task_state["bottle_variant"]],
                "inertia_profile": (
                    WATER_INERTIA_PROFILE
                    if self.task_state["bottle_variant"] == "bottled_water_16"
                    else "collision_mesh_inertia_v1"
                ),
                "support_profile": (
                    WATER_SUPPORT_PROFILE
                    if self.task_state["bottle_variant"] == "bottled_water_16"
                    else "original_collision_meshes"
                ),
            },
            "variation_index": LOCATIONS.index(self.task_state["location"]),
            "instruction": (
                f"Place the bottle in the {self.task_state['location']} slot of the rack, "
                "with its neck pointing away from the robot base."
            ),
            "rack_xy_m": list(self.task_state["rack_xy_m"]),
            "rack_yaw_rad": self.task_state["rack_yaw_rad"],
            "slot_y_m": dict(SLOT_Y_M),
            "bottle_axis_height_m": BOTTLE_AXIS_HEIGHT_M,
            "neck_direction_rack": "+x",
            "success_hold_s": SUCCESS_HOLD_S,
            "containment_tolerance_m": CONTAINMENT_TOLERANCE_M,
            "reference": SOURCE_URL,
            "visual_revision": VISUAL_REVISION,
        }

    def get_metrics(self):
        return {
            "success": {"passed": bool(self._success), "subconditions": deepcopy(self._conditions)},
            "location": self.task_state["location"],
        }


register_task(
    TASK_TYPE,
    TaskDefinition(
        env_factory=WineRack,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"task_state": state},
        describe=lambda state: {
            "task_id": f"{TASK_TYPE}.{state['location']}",
            "instruction": (
                f"Place the bottle in the {state['location']} slot of the rack, "
                "with its neck pointing away from the robot base."
            ),
        },
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
