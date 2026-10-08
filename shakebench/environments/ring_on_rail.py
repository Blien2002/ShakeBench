"""Single blue-ring placement on the shaken worktable; import to register the task.

The ring uses robosuite's box-built hollow cylinder. Poses in state assets are
relative to the tabletop; the peg moves on a low-profile cross rail.
Explicit v6 states support CPU evaluation and rigid-mount CUDA collection.
"""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from shakebench.environments.base import ShakeBenchTask
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.models.objects.ring_board import BOARD_HEIGHT_M, BOARD_RADIUS_M, add_ring_board
from shakebench.models.objects.rings import make_ring
from shakebench.scene.config import configure_scene_rendering
from shakebench.scene.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.sensors.providers import POLICY_FIELD_CONTRACT
from shakebench.tasks.metrics import pose_twist_in_frame
from shakebench.tasks.privilege import assert_policy_observation_is_clean
from shakebench.tasks.registry import TaskDefinition, register_state_loader, register_task

RINGS = {
    "large": {"outer_radius": 0.055, "inner_radius": 0.030, "rgba": [0.08, 0.28, 0.90, 1]},
}
RING_HALF_HEIGHT_M = 0.008
RING_SEGMENTS = 16
RING_WALL_NORMALS = np.array(
    [
        [np.cos(np.pi - i * 2 * np.pi / RING_SEGMENTS), np.sin(np.pi - i * 2 * np.pi / RING_SEGMENTS)]
        for i in range(RING_SEGMENTS)
    ]
)
PEG_RADIUS_M = 0.018
PEG_TOP_RADIUS_M = 0.009
PEG_HEIGHT_M = 0.120
PEG_HEAD_RADIUS_M = 0.0125
HOLD_DURATION_S = 0.5
HOLE_PENETRATION_TOLERANCE_M = 0.0005
STATE_SCHEMA = "shakebench.ring_on_rail.states"
DEFAULT_PEG_XY_M = (-0.060, -0.150)
RING_SPAWN_LOW_M = (-0.200, 0.040)
RING_SPAWN_HIGH_M = (0.060, 0.190)
PEG_TRAVEL_M = 0.08
PEG_OFFSET_LIMIT_M = 0.04
RAIL_HEIGHT_M = 0.016
RAIL_FOOTPRINT_HALF_M = PEG_TRAVEL_M + BOARD_RADIUS_M
PEG_SLIDE_JOINTS = ("peg_slide_x", "peg_slide_y")
FLOATING_MOUNT = {"fn_hz": 1.4, "zeta": 0.15}


def rail_clearance(xy, radius):
    """Distance from a ring edge to the full-travel assembly bounding box."""
    delta = np.maximum(np.abs(np.asarray(xy) - DEFAULT_PEG_XY_M) - RAIL_FOOTPRINT_HALF_M, 0)
    return float(np.linalg.norm(delta) - radius)


def validate_mount(mount):
    """Accept a locked carriage or explicit positive spring/damper parameters."""
    if isinstance(mount, str) and mount == "rigid":
        return mount
    if not isinstance(mount, Mapping) or set(mount) != {"fn_hz", "zeta"}:
        raise ValueError('peg_mount must be "rigid" or {"fn_hz": positive, "zeta": nonnegative}')
    result = {}
    for key, value in mount.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError(f"peg_mount.{key} must be finite")
        if value < 0 or (key == "fn_hz" and value == 0):
            raise ValueError(f"invalid peg_mount.{key}")
        result[key] = float(value)
    return result


def default_state():
    """Return a complete deterministic development episode."""
    return {
        "state_id": "ring-rail-000",
        "task": {"task_type": "ring_on_rail", "version": 6},
        "large_ring_xy_m": [-0.110, 0.115],
        "large_ring_yaw_rad": 0.0,
        "peg_xy_m": list(DEFAULT_PEG_XY_M),
        "peg_offset_m": [0.0, 0.0],
        "peg_mount": "rigid",
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Validate explicit horizontal, stationary starts without hidden randomness."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError("ring state fields must match the explicit version 6 schema")
    if state["task"] != {"task_type": "ring_on_rail", "version": 6} or type(state["task"]["version"]) is not int:
        raise ValueError("expected ring_on_rail task version 6")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be a nonempty string")
    result = deepcopy(dict(state))
    peg_xy = np.asarray(state["peg_xy_m"], dtype=float)
    if peg_xy.shape != (2,) or not np.array_equal(peg_xy, DEFAULT_PEG_XY_M):
        raise ValueError("rail centre must remain at (-0.06, -0.15)")
    result["peg_xy_m"] = peg_xy.tolist()
    offset = np.asarray(state["peg_offset_m"], dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > PEG_OFFSET_LIMIT_M):
        raise ValueError("peg_offset_m must contain two finite offsets in [-0.04, 0.04]")
    result["peg_offset_m"] = offset.tolist()
    result["peg_mount"] = validate_mount(state["peg_mount"])
    for name, spec in RINGS.items():
        key = f"{name}_ring_xy_m"
        xy = np.asarray(state[key], dtype=float)
        radius = spec["outer_radius"]
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"{key} must be a finite 2D position")
        if np.any(np.abs(xy) + radius > [0.325, 0.30]):
            raise ValueError("rings must lie entirely on the tabletop")
        if rail_clearance(xy, radius) < 0.01:
            raise ValueError("initial ring must clear the full-travel rail assembly by 1 cm")
        result[key] = xy.tolist()
    for key in (*[f"{name}_ring_yaw_rad" for name in RINGS], "t0_s"):
        if isinstance(state[key], bool) or not isinstance(state[key], (int, float)) or not np.isfinite(state[key]):
            raise ValueError(f"{key} must be finite")
        result[key] = float(state[key])
    if result["t0_s"] < 0:
        raise ValueError("t0_s must be nonnegative")
    for key in ("excitation_seed", "imu_seed"):
        if type(state[key]) is not int or not 0 <= state[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    return result


def sample_state(rng, *, base_state=None, state_id=None):
    """Sample a blue ring clear of the full-travel rails and random peg offsets.

    The split keeps the placement target clear while explicit sampled states make
    collection retries and evaluation reproducible.
    """
    state = deepcopy(default_state() if base_state is None else base_state)
    state["peg_offset_m"] = rng.uniform(-PEG_OFFSET_LIMIT_M, PEG_OFFSET_LIMIT_M, 2).tolist()
    for name, spec in RINGS.items():
        for _ in range(1000):
            xy = rng.uniform(RING_SPAWN_LOW_M, RING_SPAWN_HIGH_M)
            if rail_clearance(xy, spec["outer_radius"]) < 0.01:
                continue
            state[f"{name}_ring_xy_m"] = xy.tolist()
            break
        else:
            raise ValueError("could not sample a ring clear of the rails")
    if state_id is not None:
        state["state_id"] = state_id
    return validate_state(state)


def load_states(payload):
    """Load explicit development records; validate every pose and seed."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 6
    ):
        raise ValueError("unsupported ring state schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("ring states must be a nonempty list")
    states = [validate_state(state) for state in payload["states"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate ring state_id")
    return {"states": states}


class RingOnRail(ShakeBenchTask):
    """Place the blue ring on the moving-table wooden peg."""

    def __init__(
        self,
        robots="Panda",
        *,
        ring_state=None,
        free_peg=False,
        physics_profile="official",
        geometry_profile=DEFAULT_GEOMETRY_PROFILE,
        vibration=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        use_object_obs=True,
        controller_configs=None,
        **kwargs,
    ):
        self.free_peg = bool(free_peg)
        self._randomize_rings = ring_state is None
        self._placement_rng = np.random.default_rng(kwargs.get("seed"))
        self.ring_state = validate_state(default_state() if ring_state is None else ring_state)
        if self.free_peg and self.ring_state["peg_mount"] != "rigid":
            raise ValueError("floating peg_mount and free_peg are mutually exclusive")
        self._init_worktable(self.ring_state, physics_profile, geometry_profile, vibration, imu_mode, imu_seed)
        if self.worktable_mount != "rigid":
            raise ValueError("RingOnRail v6 requires a rigid worktable with only the two rail slide joints")
        self._success = False
        self._stage = 0
        self._candidate_since = None
        self._conditions = {}
        self._diagnostics = {}
        self.use_object_obs = use_object_obs
        self._init_robosuite(robots, controller_configs, self.ring_state, kwargs)

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.robot_base_body_name = robot.robot_model.root_body
        self.gripper_body_name = robot.robot_model.eef_name["right"]
        if type(robot.gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("RingOnRail requires PandaGripper")
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
        self.peg_origin = np.array([*DEFAULT_PEG_XY_M, self.arena.table_half_size[2] + RAIL_HEIGHT_M + BOARD_HEIGHT_M])
        if self.free_peg:
            peg_position = self.arena.table_top_abs + [*self.ring_state["peg_xy_m"], BOARD_HEIGHT_M]
            peg = ET.SubElement(self.arena.worldbody, "body", name="ring_peg", pos=array_to_string(peg_position))
            ET.SubElement(peg, "freejoint", name="ring_peg_free_joint")
        else:
            rail = ET.SubElement(
                self.arena.worktable_body,
                "body",
                name="rail_base",
                pos=array_to_string([*DEFAULT_PEG_XY_M, self.arena.table_half_size[2]]),
            )

            # The two rail levels and moving wood base occupy 28 mm above the table.
            def box(body, name, size, pos, rgba="0.30 0.34 0.39 1"):
                for visual in (False, True):
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{name}_{'visual' if visual else 'collision'}",
                        type="box",
                        size=array_to_string(size),
                        pos=array_to_string(pos),
                        rgba=rgba,
                        mass="0",
                        friction="0.3 0.005 0.0001",
                        group=str(int(visual)),
                        contype="0" if visual else "1",
                        conaffinity="0" if visual else "1",
                    )

            parent = rail
            for index, (axis, mass, effective_mass) in enumerate((("x", 1.0, 3.0), ("y", 2.0, 2.0))):
                z = 0.003 if index == 0 else 0.011
                for side in (-1, 1):
                    size = [0.115, 0.004, 0.003] if axis == "x" else [0.004, 0.115, 0.003]
                    pos = [0, side * 0.018, z] if axis == "x" else [side * 0.018, 0, z]
                    box(parent, f"rail_{axis}_{side}", size, pos, "0.56 0.60 0.65 1")
                    stop_pos = [side * 0.12, 0, z] if axis == "x" else [0, side * 0.12, z]
                    stop_size = [0.005, 0.027, 0.004] if axis == "x" else [0.027, 0.005, 0.004]
                    box(parent, f"rail_{axis}_stop_{side}", stop_size, stop_pos)
                stage = ET.SubElement(parent, "body", name=f"{axis}stage")
                ET.SubElement(
                    stage, "inertial", mass=str(mass), pos=f"0 0 {z + 0.004}", diaginertia="0.004 0.004 0.004"
                )
                mount = self.ring_state["peg_mount"]
                omega = 0.0 if mount == "rigid" else 2 * np.pi * mount["fn_hz"]
                damping = 0.0 if mount == "rigid" else 2 * mount["zeta"] * effective_mass * omega
                ET.SubElement(
                    stage,
                    "joint",
                    name=PEG_SLIDE_JOINTS[index],
                    type="slide",
                    axis="1 0 0" if index == 0 else "0 1 0",
                    limited="true",
                    range=f"{-PEG_TRAVEL_M} {PEG_TRAVEL_M}",
                    stiffness=str(effective_mass * omega**2),
                    damping=str(damping),
                    frictionloss="50" if mount == "rigid" else "0",
                    springref="0",
                )
                box(stage, f"{axis}stage_plate", [0.03, 0.03, 0.001], [0, 0, z + 0.004])
                parent = stage
            peg = ET.SubElement(parent, "body", name="ring_peg", pos=f"0 0 {RAIL_HEIGHT_M + BOARD_HEIGHT_M}")
        contact = self.physics_profile.pair_attributes(
            float(self.physics_profile.contact["sliding_mu"]["table_object"])
        )
        # Geom friction uses three coefficients, unlike an explicit five-coefficient pair.
        contact["friction"] = array_to_string(
            [
                float(self.physics_profile.contact["sliding_mu"]["table_object"]),
                self.physics_profile.contact_torsional_mu,
                self.physics_profile.contact_rolling_mu,
            ]
        )
        contact["priority"] = "1"
        self.board_contact_names = add_ring_board(
            self.arena.asset, peg, contact, mass_kg=0.09 if self.free_peg else 0.0
        )
        shaft_height = PEG_HEIGHT_M - PEG_HEAD_RADIUS_M
        # A convex frustum uses the same mesh for collision and appearance.
        # Duplicate the seam positions so the two edges have independent texture coordinates.
        angles = 2 * np.pi * (np.arange(65) % 64) / 64 + np.pi
        vertices = [
            [radius * np.cos(a), radius * np.sin(a), z]
            for radius, z in ((PEG_RADIUS_M, 0), (PEG_TOP_RADIUS_M, shaft_height))
            for a in angles
        ]
        faces = []
        for i in range(64):
            j = i + 1
            faces.extend([[i, j, j + 65], [i, j + 65, i + 65]])
        for i in range(1, 63):
            faces.extend([[0, i + 1, i], [65, 65 + i, 66 + i]])
        # Develop the frustum into an annular sector: equal physical texture scale
        # along and around the taper, rather than stretching a rectangular UV strip.
        slant = np.hypot(shaft_height, PEG_RADIUS_M - PEG_TOP_RADIUS_M)
        sector = np.linspace(-np.pi, np.pi, 65) * (PEG_RADIUS_M - PEG_TOP_RADIUS_M) / slant
        radii = np.array([PEG_RADIUS_M, PEG_TOP_RADIUS_M]) * slant / (PEG_RADIUS_M - PEG_TOP_RADIUS_M)
        # The source is 1024x512: a 160x80 mm patch keeps its texels square.
        uv = [[(r * np.cos(a) - radii.mean()) / 0.16 + 0.5, r * np.sin(a) / 0.08 + 0.5] for r in radii for a in sector]
        ET.SubElement(
            self.arena.asset,
            "mesh",
            name="ring_peg_frustum",
            vertex=array_to_string(np.asarray(vertices).ravel()),
            face=array_to_string(np.asarray(faces).ravel()),
            texcoord=array_to_string(np.asarray(uv).ravel()),
        )
        ET.SubElement(
            self.arena.asset,
            "texture",
            name="peg_fine_wood_texture",
            type="2d",
            file=xml_path_completion("textures/ambientcg_wood095_color_1k.png"),
        )
        ET.SubElement(
            self.arena.asset,
            "material",
            name="peg_fine_wood",
            texture="peg_fine_wood_texture",
            texrepeat="1 1",
            rgba="0.94 0.98 1 1",
            emission="0.26",
            specular="0.08",
            shininess="0.12",
        )
        shapes = (
            ("shaft", {"type": "mesh", "mesh": "ring_peg_frustum"}),
            ("cap", {"type": "sphere", "size": str(PEG_HEAD_RADIUS_M), "pos": f"0 0 {shaft_height}"}),
        )
        self.peg_contact_names = []
        peg_masses = {"shaft": 0.04, "cap": 0.005}
        for name, attributes in shapes:
            self.peg_contact_names.append(f"ring_peg_{name}_collision")
            for visual in (False, True):
                ET.SubElement(
                    peg,
                    "geom",
                    name=f"ring_peg_{name}_{'visual' if visual else 'collision'}",
                    **attributes,
                    group=str(int(visual)),
                    contype="0" if visual else "1",
                    conaffinity="0" if visual else "1",
                    mass=str(peg_masses[name] if self.free_peg and not visual else 0),
                    material="peg_fine_wood",
                )
        self.rings = {
            name: make_ring(f"{name}_ring", **spec, half_height=RING_HALF_HEIGHT_M, segments=RING_SEGMENTS)
            for name, spec in RINGS.items()
        }
        self.model = ManipulationTask(self.arena, [robot.robot_model], list(self.rings.values()))
        # MJWarp requires dense inertia storage for this task.
        self.model.root.find("option").set("jacobian", "dense")
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

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self.ring_body_ids = {name: model.body_name2id(ring.root_body) for name, ring in self.rings.items()}
        self.peg_body_id = model.body_name2id("ring_peg")
        self.peg_slide_qpos_ids = (
            []
            if self.free_peg
            else [int(model._model.jnt_qposadr[model.joint_name2id(name)]) for name in PEG_SLIDE_JOINTS]
        )
        self.ring_geom_ids = {
            name: {model.geom_name2id(geom) for geom in ring.contact_geoms} for name, ring in self.rings.items()
        }
        self.peg_geom_ids = {model.geom_name2id(name) for name in self.peg_contact_names}
        self.board_geom_ids = {model.geom_name2id(name) for name in self.board_contact_names}
        self.table_geom_id = model.geom_name2id("table_collision")
        self.robot_geom_ids = {
            model.geom_name2id(name) for name in model.geom_names if name.startswith(("robot0_", "gripper0_"))
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            if self.free_peg:
                self.sim.data.set_joint_qpos(
                    "ring_peg_free_joint",
                    [*(self.arena.table_top_abs + [*self.ring_state["peg_xy_m"], BOARD_HEIGHT_M]), 1, 0, 0, 0],
                )
            if self._randomize_rings:
                self.ring_state = sample_state(self._placement_rng, base_state=self.ring_state)
            if not self.free_peg:
                if self.ring_state["peg_mount"] != "rigid":
                    self.sim.model._model.qpos_spring[self.peg_slide_qpos_ids] = self.ring_state["peg_offset_m"]
                self.sim.data.qpos[self.peg_slide_qpos_ids] = self.ring_state["peg_offset_m"]
            for name, ring in self.rings.items():
                yaw = self.ring_state[f"{name}_ring_yaw_rad"]
                self.sim.data.set_joint_qpos(
                    ring.joints[0],
                    [
                        *(
                            self.arena.table_top_abs
                            + [
                                *self.ring_state[f"{name}_ring_xy_m"],
                                BOARD_HEIGHT_M + RING_HALF_HEIGHT_M + 0.001,
                            ]
                        ),
                        np.cos(yaw / 2),
                        0,
                        0,
                        np.sin(yaw / 2),
                    ],
                )
            self._settle()
            if not self.free_peg:
                self.sim.data.qpos[self.peg_slide_qpos_ids] = self.ring_state["peg_offset_m"]
        self.sim.forward()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=float(self.sim.data.time))
        self._success, self._stage, self._candidate_since, self._conditions, self._diagnostics = False, 0, None, {}, {}

    def _settle(self):
        """Settle the support and ring with robot joints fixed, before the clock starts."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        # Rim-supported rings can rock slightly: distinguish m/s from rad/s.
        velocity_limits = np.full(model.nv, 0.001)
        for name, ring in self.rings.items():
            adr = model.jnt_dofadr[self.sim.model.joint_name2id(ring.joints[0])]
            velocity_limits[adr + 3 : adr + 6] = 0.001 / RINGS[name]["outer_radius"]
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
                raise RuntimeError("non-finite ring reset")
            # Use pose drift over 100 ms, not instantaneous soft-contact jitter.
            mujoco.mj_differentiatePos(model, mean_velocity, stride * model.opt.timestep, previous_pose, data.qpos)
            previous_pose[:] = data.qpos
            quiet = quiet + stride if np.all(np.abs(mean_velocity) < velocity_limits) else 0
            if quiet * model.opt.timestep >= 0.2:
                break
        else:
            raise RuntimeError("ring reset did not settle within 5 simulation seconds")
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        self._conditions = {name: self._success_conditions(name) for name in RINGS}
        if self._success:
            return
        ready = all(all(conditions.values()) for conditions in self._conditions.values())
        if ready:
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= HOLD_DURATION_S - 1e-12:
                self._stage += 1
                self._candidate_since = None
                self._success = self._stage == len(RINGS)
        else:
            self._candidate_since = None

    def _success_conditions(self, name):
        model, data = self.sim.model._model, self.sim.data._data
        ring = self.rings[name]
        body_id = self.ring_body_ids[name]
        geom_ids = self.ring_geom_ids[name]
        pose = pose_twist_in_frame(self.sim, ring.root_body, "ring_peg")
        self._diagnostics[name] = {
            "stable": bool(
                np.linalg.norm(pose.linear_velocity_m_s) <= 0.02 and np.linalg.norm(pose.angular_velocity_rad_s) <= 0.3
            ),
            "relative_linear_velocity_m_s": pose.linear_velocity_m_s.tolist(),
            "relative_angular_velocity_rad_s": pose.angular_velocity_rad_s.tolist(),
        }
        rotation = data.xmat[body_id].reshape(3, 3).T @ data.xmat[self.peg_body_id].reshape(3, 3)
        axis = rotation[:, 2]
        peg_in_ring = -rotation @ pose.position_m
        cosine = abs(axis[2])
        # Bound the taper by its widest radius across the ring slab, then check
        # every polygon wall including tilt. This is conservative for tilted rings.
        clearance = -np.inf
        if cosine > 0.5:
            intersection = peg_in_ring[:2] - peg_in_ring[2] * axis[:2] / axis[2]
            slope = RING_WALL_NORMALS @ axis[:2] / axis[2]
            lowest_z = (
                pose.position_m[2]
                - (RING_HALF_HEIGHT_M + RINGS[name]["inner_radius"] * np.sqrt(max(0.0, 1 - cosine**2))) / cosine
            )
            shaft_height = PEG_HEIGHT_M - PEG_HEAD_RADIUS_M
            radius = PEG_RADIUS_M + (PEG_TOP_RADIUS_M - PEG_RADIUS_M) * np.clip(lowest_z / shaft_height, 0, 1)
            extent = (
                RING_WALL_NORMALS @ intersection + radius * np.sqrt(1 + slope**2) + RING_HALF_HEIGHT_M * np.abs(slope)
            )
            clearance = RINGS[name]["inner_radius"] * np.cos(np.pi / RING_SEGMENTS) - np.max(extent)
        support_force, any_support_force, released = 0.0, 0.0, True
        support_ids = self.board_geom_ids
        all_support_ids = {self.table_geom_id} | self.board_geom_ids | self.peg_geom_ids
        for ids in self.ring_geom_ids.values():
            all_support_ids |= ids
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
            "seated": bool(abs(pose.position_m[2] - RING_HALF_HEIGHT_M) <= 0.003),
            "supported": bool(support_force > 0.01),
            "released": released,
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

    def observation_contract(self):
        contract = {key: dict(POLICY_FIELD_CONTRACT[key]) for key in self.table_imu_provider.policy_keys}
        for key, shape, units, frame in (
            ("robot0_joint_pos", (7,), "rad", "robot_base"),
            ("robot0_joint_vel", (7,), "rad/s", "robot_base"),
            ("robot0_gripper_qpos", (2,), "m", "gripper"),
            ("robot0_gripper_qvel", (2,), "m/s", "gripper"),
            *((f"{prefix}_pos_robot_base", (3,), "m", "robot_base") for prefix in ("large_ring", "peg")),
            *(
                (f"{prefix}_quat_wxyz_robot_base", (4,), "unit quaternion wxyz", "robot_base")
                for prefix in ("large_ring", "peg")
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
            "task_type": "ring_on_rail",
            "version": 6,
            "scoreable": False,
            "rings": deepcopy(RINGS),
            "placement_order": list(RINGS),
            "board_radius_m": BOARD_RADIUS_M,
            "board_height_m": BOARD_HEIGHT_M,
            "peg_head_radius_m": PEG_HEAD_RADIUS_M,
            "ring_half_height_m": RING_HALF_HEIGHT_M,
            "ring_segments": RING_SEGMENTS,
            "peg_base_radius_m": PEG_RADIUS_M,
            "peg_top_radius_m": PEG_TOP_RADIUS_M,
            "peg_height_m": PEG_HEIGHT_M,
            "peg_free_on_table": self.free_peg,
            "hold_duration_s": HOLD_DURATION_S,
            "hole_penetration_tolerance_m": HOLE_PENETRATION_TOLERANCE_M,
            "geometry_profile": self.geometry_profile["profile_id"],
        }

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        info["privileged"] = self.get_privileged_state()
        return reward, done, info

    def get_privileged_state(self):
        """Current control-step ground truth; excluded from policy observations."""
        data = self.sim.data
        rotation = data.xmat[self.peg_body_id].reshape(3, 3)
        return {
            "peg_head_world_m": (data.xpos[self.peg_body_id] + rotation @ [0, 0, PEG_HEIGHT_M]).tolist(),
            "peg_slide_qpos_m": data.qpos[self.peg_slide_qpos_ids].tolist(),
        }

    def get_metrics(self):
        return {
            "privileged": self.get_privileged_state(),
            "diagnostics": deepcopy(self._diagnostics),
            "success": {
                "passed": bool(self._success),
                "stage": self._stage,
                "subconditions": deepcopy(self._conditions),
            },
        }


register_task(
    "ring_on_rail",
    TaskDefinition(
        env_factory=RingOnRail,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"ring_state": state},
        describe=lambda state: {
            "task_id": "ring_on_rail",
            "instruction": "Place the blue ring on the wooden peg.",
        },
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
