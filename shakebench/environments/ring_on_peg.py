"""Ordered large-then-small ring stacking on the shaken worktable; import to register the task.

The ring uses robosuite's box-built hollow cylinder. Poses in state assets are
relative to the tabletop; the peg is rigidly attached to that moving table.
Only explicit CPU development states are supported, never certified scores.
"""

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
from shakebench.models.objects.rings import make_ring
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

RINGS = {
    "large": {"outer_radius": 0.055, "inner_radius": 0.025, "rgba": [0.94, 0.42, 0.055, 1]},
    "small": {"outer_radius": 0.040, "inner_radius": 0.020, "rgba": [0.035, 0.48, 0.55, 1]},
}
RING_HALF_HEIGHT_M = 0.008
RING_SEGMENTS = 16
RING_WALL_NORMALS = np.array(
    [
        [np.cos(np.pi - i * 2 * np.pi / RING_SEGMENTS), np.sin(np.pi - i * 2 * np.pi / RING_SEGMENTS)]
        for i in range(RING_SEGMENTS)
    ]
)
PEG_RADIUS_M = 0.012
PEG_HEIGHT_M = 0.100
HOLD_DURATION_S = 0.5
HOLE_PENETRATION_TOLERANCE_M = 0.0005
STATE_SCHEMA = "shakebench.ring_on_peg.states"


def default_state():
    """Return a complete deterministic development episode."""
    return {
        "state_id": "ring-stack-000",
        "task": {"task_type": "ring_on_peg", "version": 2},
        "large_ring_xy_m": [-0.10, -0.15],
        "small_ring_xy_m": [-0.10, 0.0],
        "large_ring_yaw_rad": 0.0,
        "small_ring_yaw_rad": 0.0,
        "peg_xy_m": [-0.10, 0.15],
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Validate explicit horizontal, stationary starts without hidden randomness."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError("ring state fields must match the explicit version 2 schema")
    if state["task"] != {"task_type": "ring_on_peg", "version": 2} or type(state["task"]["version"]) is not int:
        raise ValueError("expected ring_on_peg task version 2")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be a nonempty string")
    result = deepcopy(dict(state))
    footprints = {f"{name}_ring_xy_m": spec["outer_radius"] + 0.005 for name, spec in RINGS.items()}
    footprints["peg_xy_m"] = 0.017
    for key, radius in footprints.items():
        value = np.asarray(state[key], dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all() or np.any(np.abs(value) + radius > [0.325, 0.30]):
            raise ValueError(f"{key} must lie entirely on the tabletop")
        result[key] = value.tolist()
    keys = list(footprints)
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            if (
                np.linalg.norm(np.subtract(result[first], result[second]))
                < footprints[first] + footprints[second] + 0.01
            ):
                raise ValueError("ring and peg starts must be separated")
    for key in ("large_ring_yaw_rad", "small_ring_yaw_rad", "t0_s"):
        if isinstance(state[key], bool) or not isinstance(state[key], (int, float)) or not np.isfinite(state[key]):
            raise ValueError(f"{key} must be finite")
        result[key] = float(state[key])
    if result["t0_s"] < 0:
        raise ValueError("t0_s must be nonnegative")
    for key in ("excitation_seed", "imu_seed"):
        if type(state[key]) is not int or not 0 <= state[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    return result


def load_states(payload):
    """Load explicit development records; validate every pose and seed."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 2
    ):
        raise ValueError("unsupported ring state schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("ring states must be a nonempty list")
    states = [validate_state(state) for state in payload["states"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate ring state_id")
    return {"states": states, "authority": {"kind": STATE_SCHEMA, "scoreable": False}}


class RingOnPeg(ManipulationEnv):
    """Stack the large ring first, then the small ring, on the moving-table peg."""

    def __init__(
        self,
        robots="Panda",
        *,
        ring_state=None,
        physics_profile="official",
        geometry_profile=DEFAULT_GEOMETRY_PROFILE,
        vibration=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        use_object_obs=True,
        controller_configs=None,
        **kwargs,
    ):
        self.ring_state = validate_state(default_state() if ring_state is None else ring_state)
        self.physics_profile = resolve_physics_profile(physics_profile)
        self.geometry_profile = load_geometry_profile(geometry_profile)
        self.worktable_mount = worktable_mount(geometry_profile)
        self.scene_path = geometry_scene_path(geometry_profile)
        self.table_imu_provider = TableIMUProvider(
            seed=self.ring_state["imu_seed"] if imu_seed is None else imu_seed, imu_mode=imu_mode
        )
        self.deck_driver = DeckDriver(
            trajectory=build_vibration_program(
                vibration
                if vibration is not None
                else {
                    "mode": "multisine_v1",
                    "gamma": 0.0,
                    "seed": self.ring_state["excitation_seed"],
                    "t0_s": self.ring_state["t0_s"],
                }
            ),
            config=self.physics_profile.deck_driver_config(),
            body_handles={"isolated_worktable": "worktable", "deck_visual": DECK_VISUAL_BODY_NAME},
            required_roles=("isolated_worktable", "deck_visual"),
        )
        self._success = False
        self._stage = 0
        self._candidate_since = None
        self._conditions = {}
        self.use_object_obs = use_object_obs
        eager = kwargs.pop("load_model_on_init", True)
        if kwargs.pop("control_freq", 20) != 20:
            raise ValueError("RingOnPeg requires control_freq=20")
        for key, value in {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "initialization_noise": None,
            "hard_reset": False,
            "seed": self.ring_state["excitation_seed"],
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
        self.add_post_physics_step_hook(self._after_physics)
        self.load_model_on_init = eager
        if eager:
            self.reset()

    def _check_robot_configuration(self, robots):
        if (list(robots) if isinstance(robots, (list, tuple)) else [robots]) != ["Panda"]:
            raise ValueError("RingOnPeg supports exactly one Panda")

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.robot_base_body_name = robot.robot_model.root_body
        self.gripper_body_name = robot.robot_model.eef_name["right"]
        if type(robot.gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("RingOnPeg requires PandaGripper")
        self.arena = ShakeBenchArena(
            table_offset=self.geometry_profile["table_top_pos_m"],
            isolator_config=self.physics_profile.isolator_config(),
            worktable_mount=self.worktable_mount,
            scene_config=self.scene_path,
        )
        support = (
            ET.parse(xml_path_completion(self.geometry_profile["robot_support_mjcf"]))
            .getroot()
            .find("./worldbody/body[@name='robot_support']")
        )
        self.arena.worldbody.append(support)
        self.peg_origin = np.array([*self.ring_state["peg_xy_m"], self.arena.table_half_size[2]])
        peg = ET.SubElement(self.arena.worktable_body, "body", name="ring_peg", pos=array_to_string(self.peg_origin))
        # The fixture belongs to the existing 32 kg worktable assembly.
        for name, rgba, specular, shininess in (
            ("peg_steel", "0.62 0.66 0.70 1", "0.7", "0.65"),
            ("peg_base", "0.12 0.15 0.18 1", "0.4", "0.35"),
        ):
            ET.SubElement(self.arena.asset, "material", name=name, rgba=rgba, specular=specular, shininess=shininess)
        shaft_height = PEG_HEIGHT_M - PEG_RADIUS_M
        shapes = (
            ("shaft", "cylinder", f"{PEG_RADIUS_M} {shaft_height / 2}", f"0 0 {shaft_height / 2}", "peg_steel"),
            ("cap", "sphere", str(PEG_RADIUS_M), f"0 0 {shaft_height}", "peg_steel"),
            ("base", "cylinder", "0.017 0.002", "0 0 0.002", "peg_base"),
        )
        self.peg_contact_names = []
        for name, kind, size, pos, material in shapes:
            self.peg_contact_names.append(f"ring_peg_{name}_collision")
            for visual in (False, True):
                ET.SubElement(
                    peg,
                    "geom",
                    name=f"ring_peg_{name}_{'visual' if visual else 'collision'}",
                    type=kind,
                    size=size,
                    pos=pos,
                    group=str(int(visual)),
                    contype="0" if visual else "1",
                    conaffinity="0" if visual else "1",
                    mass="0",
                    material=material,
                )
        # Recessed screw heads stay within the base footprint and have no collision.
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
            xy = 0.0145 * np.array([np.cos(angle), np.sin(angle)])
            ET.SubElement(
                peg,
                "geom",
                name=f"peg_screw_{i}",
                type="cylinder",
                size="0.0018 0.0002",
                pos=array_to_string([*xy, 0.004]),
                material="peg_steel",
                group="1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        self.rings = {
            name: make_ring(f"{name}_ring", **spec, half_height=RING_HALF_HEIGHT_M, segments=RING_SEGMENTS)
            for name, spec in RINGS.items()
        }
        self.model = ManipulationTask(self.arena, [robot.robot_model], list(self.rings.values()))
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        pads = robot.gripper["right"].important_geoms
        finger_names = pads["left_fingerpad"] + pads["right_fingerpad"]
        for ring in self.rings.values():
            for ring_geom in ring.contact_geoms:
                for partner in ["table_collision", *self.peg_contact_names, *finger_names]:
                    finger = partner in finger_names
                    attributes = self.physics_profile.pair_attributes(
                        float(
                            self.physics_profile.contact["sliding_mu"]["finger_object" if finger else "table_object"]
                        ),
                        finger_contact=finger,
                    )
                    ET.SubElement(self.model.contact, "pair", geom1=ring_geom, geom2=partner, **attributes)
        for large in self.rings["large"].contact_geoms:
            for small in self.rings["small"].contact_geoms:
                ET.SubElement(
                    self.model.contact,
                    "pair",
                    geom1=large,
                    geom2=small,
                    **self.physics_profile.pair_attributes(
                        float(self.physics_profile.contact["sliding_mu"]["table_object"])
                    ),
                )

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self.ring_body_ids = {name: model.body_name2id(ring.root_body) for name, ring in self.rings.items()}
        self.peg_body_id = model.body_name2id("ring_peg")
        self.ring_geom_ids = {
            name: {model.geom_name2id(geom) for geom in ring.contact_geoms} for name, ring in self.rings.items()
        }
        self.peg_geom_ids = {model.geom_name2id(name) for name in self.peg_contact_names}
        self.table_geom_id = model.geom_name2id("table_collision")
        self.robot_geom_ids = {
            model.geom_name2id(name) for name in model.geom_names if name.startswith(("robot0_", "gripper0_"))
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for name, ring in self.rings.items():
                yaw = self.ring_state[f"{name}_ring_yaw_rad"]
                self.sim.data.set_joint_qpos(
                    ring.joints[0],
                    [
                        *(
                            self.arena.table_top_abs
                            + [*self.ring_state[f"{name}_ring_xy_m"], RING_HALF_HEIGHT_M + 0.001]
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
        self._success, self._stage, self._candidate_since, self._conditions = False, 0, None, {}

    def _settle(self):
        """Settle the support and ring with robot joints fixed, before the clock starts."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        quiet = 0
        stride = max(1, round(0.005 / model.opt.timestep))
        for step in range(round(5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qpos_ids], data.qvel[qvel_ids] = fixed_pose, 0
            if (step + 1) % stride:
                continue
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("non-finite ring reset")
            quiet = quiet + stride if np.max(np.abs(data.qvel)) < 0.001 else 0
            if quiet * model.opt.timestep >= 0.1:
                break
        else:
            raise RuntimeError("ring reset did not settle within 5 simulation seconds")
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _after_physics(self, sample_time_s, policy_step=False):
        self.table_imu_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)
        self._conditions = {name: self._success_conditions(name) for name in RINGS}
        if self._success:
            return
        large, small = self._conditions["large"], self._conditions["small"]
        # Stage 1 only starts with the small ring off the peg. A wrong-order or
        # simultaneous placement can be corrected by removing the small ring.
        ready = all(large.values()) and (not small["on_peg"] if self._stage == 0 else all(small.values()))
        if ready:
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= HOLD_DURATION_S - 1e-12:
                self._stage += 1
                self._candidate_since = None
                self._success = self._stage == 2
        else:
            self._candidate_since = None

    def _success_conditions(self, name):
        model, data = self.sim.model._model, self.sim.data._data
        ring = self.rings[name]
        body_id = self.ring_body_ids[name]
        geom_ids = self.ring_geom_ids[name]
        pose = pose_twist_in_frame(self.sim, ring.root_body, "ring_peg")
        rotation = data.xmat[body_id].reshape(3, 3).T @ data.xmat[self.peg_body_id].reshape(3, 3)
        axis = rotation[:, 2]
        peg_in_ring = -rotation @ pose.position_m
        cosine = abs(axis[2])
        # Check every inner wall, including the tilted cylinder's elliptical
        # cross-section and axis drift through the ring thickness. Wall contact
        # is valid; an inscribed-circle test would reject correctly seated rings.
        clearance = -np.inf
        if cosine > 0.5:
            intersection = peg_in_ring[:2] - peg_in_ring[2] * axis[:2] / axis[2]
            slope = RING_WALL_NORMALS @ axis[:2] / axis[2]
            extent = (
                RING_WALL_NORMALS @ intersection
                + PEG_RADIUS_M * np.sqrt(1 + slope**2)
                + RING_HALF_HEIGHT_M * np.abs(slope)
            )
            clearance = RINGS[name]["inner_radius"] * np.cos(np.pi / RING_SEGMENTS) - np.max(extent)
        support_force, any_support_force, released = 0.0, 0.0, True
        support_ids = {self.table_geom_id} if name == "large" else self.ring_geom_ids["large"]
        all_support_ids = (
            {self.table_geom_id} | self.peg_geom_ids | self.ring_geom_ids["large"] | self.ring_geom_ids["small"]
        )
        force = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.geom1 in geom_ids:
                other = contact.geom2
            elif contact.geom2 in geom_ids:
                other = contact.geom1
            else:
                continue
            if other in self.robot_geom_ids and contact.dist <= 0.001:
                released = False
            if other in all_support_ids:
                mujoco.mj_contactForce(model, data, index, force)
                any_support_force += max(0.0, force[0])
                if other in support_ids:
                    support_force += max(0.0, force[0])
        return {
            "on_peg": bool(
                clearance >= -HOLE_PENETRATION_TOLERANCE_M
                and released
                and 0 < pose.position_m[2] < PEG_HEIGHT_M
                and any_support_force > 0.01
            ),
            "threaded": bool(clearance >= -HOLE_PENETRATION_TOLERANCE_M),
            "upright": bool(cosine >= np.cos(np.deg2rad(10))),
            "seated": bool(abs(pose.position_m[2] - RING_HALF_HEIGHT_M * (1 if name == "large" else 3)) <= 0.003),
            "supported": bool(support_force > 0.01),
            "released": released,
            "stable": bool(
                np.linalg.norm(pose.linear_velocity_m_s) <= 0.02 and np.linalg.norm(pose.angular_velocity_rad_s) <= 0.3
            ),
        }

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        obs.update(self.table_imu_provider.observation(self.sim))
        for prefix, body in [(f"{name}_ring", ring.root_body) for name, ring in self.rings.items()] + [
            ("peg", "ring_peg")
        ]:
            pose = pose_twist_in_frame(self.sim, body, self.robot_base_body_name)
            obs[f"{prefix}_pos_robot_base"] = pose.position_m
            obs[f"{prefix}_quat_wxyz_robot_base"] = pose.quaternion_wxyz
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
            *((f"{prefix}_pos_robot_base", (3,), "m", "robot_base") for prefix in ("large_ring", "small_ring", "peg")),
            *(
                (f"{prefix}_quat_wxyz_robot_base", (4,), "unit quaternion wxyz", "robot_base")
                for prefix in ("large_ring", "small_ring", "peg")
            ),
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
            "task_type": "ring_on_peg",
            "version": 2,
            "scoreable": False,
            "rings": deepcopy(RINGS),
            "placement_order": ["large", "small"],
            "ring_half_height_m": RING_HALF_HEIGHT_M,
            "ring_segments": RING_SEGMENTS,
            "peg_radius_m": PEG_RADIUS_M,
            "peg_height_m": PEG_HEIGHT_M,
            "hold_duration_s": HOLD_DURATION_S,
            "hole_penetration_tolerance_m": HOLE_PENETRATION_TOLERANCE_M,
            "geometry_profile": self.geometry_profile["profile_id"],
        }

    def get_metrics(self):
        return {
            "success": {
                "passed": bool(self._success),
                "stage": self._stage,
                "subconditions": deepcopy(self._conditions),
            }
        }

    def _check_success(self):
        return bool(self._success)

    def reward(self, action=None):
        return float(self._success)


register_task(
    "ring_on_peg",
    TaskDefinition(
        env_factory=RingOnPeg,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"ring_state": state},
        describe=lambda state: {
            "task_id": "ring_on_peg.v2",
            "instruction": "Place the large orange ring on the metal peg first, then stack the small teal ring on top.",
        },
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
