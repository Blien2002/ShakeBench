"""Ordered three-block stacking on the shaken worktable.

Blue, green and yellow 50 mm blocks start apart on the worktable and nothing
is fixed to it, so the blue block is the free base of the tower. The policy
stacks the green block on the blue block, then the yellow block on the green
block. Stage conditions follow RingOnPeg, except that stability is judged by
pose drift relative to the support over a window instead of instantaneous
twist, which misreads standing towers as unstable under vibration. Both joint
variants share the same outer block shape and explicit state schema.
"""

import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Mapping
from copy import deepcopy

import mujoco
import numpy as np
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from shakebench.environments.base import ShakeBenchTask
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.models.objects import stack_parts as parts
from shakebench.scene.config import configure_scene_rendering
from shakebench.scene.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.sensors.providers import POLICY_FIELD_CONTRACT
from shakebench.tasks.metrics import pose_twist_in_frame
from shakebench.tasks.privilege import assert_policy_observation_is_clean
from shakebench.tasks.registry import TaskDefinition, register_state_loader, register_task

BLOCK_HALF_M = parts.BLOCK_HALF_M
BLOCK_NAMES = parts.PARTS
BLOCK_DENSITY_KG_M3 = 650.0
WOOD_WOOD_MU = 0.45
HOLD_DURATION_S = 0.5
ALIGN_TOLERANCE_M = 0.0125
SEAT_TOLERANCE_M = 0.003
TILT_MAX_RAD = np.deg2rad(10.0)
DRIFT_WINDOW_S = 0.5
DRIFT_POSITION_M = 0.003
DRIFT_ANGLE_RAD = np.deg2rad(2.0)
BASE_BLOCK_XY_M = (-0.060, -0.150)
BLUE_BLOCK_SPAWN_LOW_M = (-0.140, -0.210)
BLUE_BLOCK_SPAWN_HIGH_M = (0.020, -0.090)
BLOCK_SPAWN_LOW_M = (-0.240, 0.040)
BLOCK_SPAWN_HIGH_M = (0.020, 0.220)
MIN_BLOCK_SPACING_M = 0.12
MAX_START_YAW_RAD = np.pi / 4
METRIC_PERIOD_S = 0.01  # stage conditions at 100 Hz; the 0.5 s holds need no 5 kHz resolution
PLAIN_SUCCESS_RULE = "ordered_released_supported_stable_v2"
STATE_SCHEMA = "shakebench.stack_blocks.states"
SCHEMA_VERSION = 1


def plain_footprints_overlap(relative, rotation):
    """Positive area overlap of projected cube footprints, without centre/yaw alignment.

    Separating axes are the support square axes and the normals of the projected
    cube edges. Touching only an edge is not an overlapping support footprint.
    """
    projected = np.asarray(rotation)[:2, :]
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    axes += [np.array([-edge[1], edge[0]]) for edge in projected.T]
    for axis in axes:
        if np.linalg.norm(axis) < 1e-12:
            continue
        support_extent = BLOCK_HALF_M * np.sum(np.abs(axis))
        block_extent = BLOCK_HALF_M * np.sum(np.abs(axis @ projected))
        if abs(float(axis @ relative[:2])) >= support_extent + block_extent - 1e-9:
            return False
    return True


def default_state():
    """Return a complete deterministic development episode."""
    return {
        "state_id": "stack-blocks-000",
        "task": {"task_type": "stack_blocks", "version": 1},
        "joint": "plain",
        "blue_block_xy_m": list(BASE_BLOCK_XY_M),
        "blue_block_yaw_rad": 0.0,
        "green_block_xy_m": [-0.110, 0.115],
        "green_block_yaw_rad": 0.0,
        "yellow_block_xy_m": [0.040, 0.120],
        "yellow_block_yaw_rad": 0.0,
        "order": list(BLOCK_NAMES),
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Validate explicit stationary starts without hidden randomness."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError("stack state fields must match the explicit version 1 schema")
    if state["task"] != {"task_type": "stack_blocks", "version": 1}:
        raise ValueError("expected stack_blocks task version 1")
    result = deepcopy(dict(state))
    if result["joint"] not in parts.JOINT_SPECS:
        raise ValueError(f"joint must name one of {sorted(parts.JOINT_SPECS)}")
    if not isinstance(result["state_id"], str) or not result["state_id"]:
        raise ValueError("state_id must be a nonempty string")
    for seed in ("excitation_seed", "imu_seed"):
        if type(result[seed]) is not int or not 0 <= result[seed] < 2**32:
            raise ValueError(f"{seed} must be a 32-bit unsigned integer")
    if list(result["order"]) != list(BLOCK_NAMES):
        raise ValueError("order must list blue, green, yellow from bottom to top in version 1")
    points = []
    for name in BLOCK_NAMES:
        xy = np.asarray(result[f"{name}_block_xy_m"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(np.abs(xy) + 0.04 > [0.325, 0.30]):
            raise ValueError("blocks must start on the tabletop with an edge margin")
        if any(np.linalg.norm(xy - other) < MIN_BLOCK_SPACING_M for other in points):
            raise ValueError("blocks must start clear of each other")
        points.append(xy)
        yaw = float(result[f"{name}_block_yaw_rad"])
        if not np.isfinite(yaw) or abs(yaw) > MAX_START_YAW_RAD + 1e-9:
            raise ValueError("block yaw must lie within 45 degrees of the table axes")
        result[f"{name}_block_yaw_rad"] = yaw
    result["t0_s"] = float(result["t0_s"])
    if not np.isfinite(result["t0_s"]):
        raise ValueError("t0_s must be finite")
    return result


def load_states(payload):
    """Load one geometry variant per explicit state file."""
    if (
        payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported stack_blocks state schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("stack states must be a nonempty list")
    states = [validate_state(state) for state in payload["states"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate stack state_id")
    if len({state["joint"] for state in states}) != 1:
        raise ValueError("one state file must contain one joint specification")
    return {"states": states}


def _yaw_quat(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


class StackBlocks(ShakeBenchTask):
    """Stack the green block on the free blue block, then the yellow block on the green block."""

    def __init__(
        self,
        robots="Panda",
        *,
        stack_state=None,
        physics_profile="official",
        geometry_profile=DEFAULT_GEOMETRY_PROFILE,
        vibration=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        use_object_obs=True,
        controller_configs=None,
        **kwargs,
    ):
        self.stack_state = validate_state(default_state() if stack_state is None else stack_state)
        self.joint = self.stack_state["joint"]
        self.use_object_obs = use_object_obs
        self._init_worktable(self.stack_state, physics_profile, geometry_profile, vibration, imu_mode, imu_seed)
        self._success, self._stage, self._candidate_since, self._conditions = False, 0, None, {}
        self._history = {name: deque() for name in BLOCK_NAMES}
        self._last_metric_time = -np.inf
        self._task_rule_violation = False
        self._diagnostics = self._new_diagnostics()
        self._wrong_contact_active = set()
        self._previous_base_xy = None
        self._grasp_time = {name: None for name in BLOCK_NAMES[1:]}
        self._seat_relative_xy = {name: None for name in BLOCK_NAMES[1:]}
        self._previous_placed_xy = {name: None for name in BLOCK_NAMES[1:]}
        self._was_seated = {name: False for name in BLOCK_NAMES[1:]}
        self._ever_seated = {name: False for name in BLOCK_NAMES[1:]}
        self._entered_joint = {name: False for name in BLOCK_NAMES[1:]}
        self._unseated_since = {name: None for name in BLOCK_NAMES[1:]}
        self._jam_since = {name: None for name in BLOCK_NAMES[1:]}
        self._current_joint_force = {}
        self._init_robosuite(robots, controller_configs, self.stack_state, kwargs)

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.robot_base_body_name = robot.robot_model.root_body
        self.gripper_body_name = robot.robot_model.eef_name["right"]
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
        meshes = parts.add_stack_assets(self.arena.asset, self.joint)
        self.blocks = {}
        for name in BLOCK_NAMES:
            block = BoxObject(f"{name}_block", size=[BLOCK_HALF_M] * 3, density=BLOCK_DENSITY_KG_M3, rgba=[1, 1, 1, 1])
            body = block.get_obj()
            original = next(g for g in body if g.tag == "geom" and g.get("group") == "0")
            attributes = {
                key: value for key, value in original.attrib.items() if key not in {"name", "type", "size", "pos"}
            }
            body.remove(original)
            parts.add_collision_geoms(self.arena.asset, body, name, self.joint, f"{name}_block_", **attributes)
            block._contact_geoms = [shape["name"] for shape in parts.collision_shapes(name, self.joint)]
            for geom in [g for g in body if g.tag == "geom" and g.get("group") == "1"]:
                body.remove(geom)
            parts.visual_geom(body, f"{name}_block_visual", meshes[name], f"stack_wood_{name}", [0, 0, 0])
            # Keep robosuite's segmentation metadata in sync with the replaced visual geom (unprefixed name).
            block._visual_geoms = ["visual"]
            self.blocks[name] = block
        self.model = ManipulationTask(self.arena, [robot.robot_model], list(self.blocks.values()))
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        pads = robot.gripper["right"].important_geoms
        finger_names = pads["left_fingerpad"] + pads["right_fingerpad"]
        table_mu = float(self.physics_profile.contact["sliding_mu"]["table_object"])
        finger_mu = float(self.physics_profile.contact["sliding_mu"]["finger_object"])
        partners = [("table_collision", table_mu, False)] + [(finger, finger_mu, True) for finger in finger_names]
        for block in self.blocks.values():
            for geom in block.contact_geoms:
                for partner, mu, finger in partners:
                    attributes = self.physics_profile.pair_attributes(mu, finger_contact=finger)
                    ET.SubElement(self.model.contact, "pair", geom1=geom, geom2=partner, **attributes)
        names = list(self.blocks)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                for a in self.blocks[first].contact_geoms:
                    for b in self.blocks[second].contact_geoms:
                        attributes = self.physics_profile.pair_attributes(WOOD_WOOD_MU)
                        ET.SubElement(self.model.contact, "pair", geom1=a, geom2=b, **attributes)

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self.block_body_ids = {name: model.body_name2id(block.root_body) for name, block in self.blocks.items()}
        self.block_geom_ids = {
            name: {model.geom_name2id(geom) for geom in block.contact_geoms} for name, block in self.blocks.items()
        }
        self.table_geom_id = model.geom_name2id("table_collision")
        self._joint_wall_geoms = {
            "blue_green": {model.geom_name2id("blue_block_tenon_collision")} if self.joint != "plain" else set(),
            "green_yellow": (
                {model.geom_name2id("green_block_round_tenon_collision")} if self.joint != "plain" else set()
            ),
        }
        self.table_body_id = int(model._model.geom_bodyid[self.table_geom_id])
        self.robot_geom_ids = {
            model.geom_name2id(name) for name in model.geom_names if name.startswith(("robot0_", "gripper0_"))
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for name, block in self.blocks.items():
                xy = self.stack_state[f"{name}_block_xy_m"]
                self.sim.data.set_joint_qpos(
                    block.joints[0],
                    [
                        *(self.arena.table_top_abs + [*xy, BLOCK_HALF_M + 0.001]),
                        *_yaw_quat(self.stack_state[f"{name}_block_yaw_rad"]),
                    ],
                )
            self._settle()
        self.sim.forward()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=float(self.sim.data.time))
        self._success, self._stage, self._candidate_since, self._conditions = False, 0, None, {}
        self._history = {name: deque() for name in BLOCK_NAMES}
        self._last_metric_time = -np.inf
        self._task_rule_violation = False
        self._diagnostics = self._new_diagnostics()
        self._wrong_contact_active = set()
        self._previous_base_xy = None
        self._grasp_time = {name: None for name in BLOCK_NAMES[1:]}
        self._seat_relative_xy = {name: None for name in BLOCK_NAMES[1:]}
        self._previous_placed_xy = {name: None for name in BLOCK_NAMES[1:]}
        self._was_seated = {name: False for name in BLOCK_NAMES[1:]}
        self._ever_seated = {name: False for name in BLOCK_NAMES[1:]}
        self._entered_joint = {name: False for name in BLOCK_NAMES[1:]}
        self._unseated_since = {name: None for name in BLOCK_NAMES[1:]}
        self._jam_since = {name: None for name in BLOCK_NAMES[1:]}
        self._current_joint_force = {}

    def _settle(self):
        """Settle the cubes with the robot fixed, before the clock starts."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        stride = max(1, round(0.1 / model.opt.timestep))
        previous_pose = data.qpos.copy()
        mean_velocity = np.zeros(model.nv)
        quiet = 0
        velocity_limits = np.full(model.nv, 0.001)
        for block in self.blocks.values():
            adr = model.jnt_dofadr[self.sim.model.joint_name2id(block.joints[0])]
            velocity_limits[adr + 3 : adr + 6] = 0.001 / BLOCK_HALF_M
        for step in range(round(5.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qpos_ids], data.qvel[qvel_ids] = fixed_pose, 0
            if (step + 1) % stride:
                continue
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("non-finite stack reset")
            mujoco.mj_differentiatePos(model, mean_velocity, stride * model.opt.timestep, previous_pose, data.qpos)
            previous_pose[:] = data.qpos
            quiet = quiet + stride if np.all(np.abs(mean_velocity) < velocity_limits) else 0
            if quiet * model.opt.timestep >= 0.2:
                break
        else:
            raise RuntimeError("stack reset did not settle within 5 simulation seconds")
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _support_ids(self, index):
        """Body, geoms and half height of the support of the block at ``index`` in the order (index >= 1)."""
        below = self.stack_state["order"][index - 1]
        return self.block_body_ids[below], self.block_geom_ids[below], BLOCK_HALF_M

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        if sample_time_s - self._last_metric_time < METRIC_PERIOD_S - 1e-9 and not policy_step:
            return
        self._last_metric_time = sample_time_s
        order = self.stack_state["order"]
        base, placed = order[0], order[1:]
        self._conditions = {base: self._base_conditions(base)}
        for index, name in enumerate(placed, start=1):
            self._conditions[name] = self._block_conditions(index, name, sample_time_s)
        self._sample_diagnostics(sample_time_s)
        if self._success:
            return
        # The free base must stand on the table and the correctly placed prefix of the order, counted from the
        # bottom, must reach past the current stage. A prefix count, not "later blocks stay off", so a block
        # that lands during the previous block's hold does not deadlock the stage machine.
        prefix = 0
        for name in placed:
            if not all(self._conditions[name].values()):
                break
            prefix += 1
        ready = all(self._conditions[base].values()) and prefix > self._stage
        if ready:
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= HOLD_DURATION_S - 1e-12:
                self._stage += 1
                self._candidate_since = None
                self._success = self._stage == len(placed)
        else:
            self._candidate_since = None

    def _contacts(self, name, support_geoms, support_up_world=None):
        """Normal force from ``support_geoms`` on a block, and whether no robot geom is within 1 mm."""
        model, data = self.sim.model._model, self.sim.data._data
        support_force, released = 0.0, True
        force = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 in self.block_geom_ids[name]:
                other = contact.geom2
            elif contact.geom2 in self.block_geom_ids[name]:
                other = contact.geom1
            else:
                continue
            if other in self.robot_geom_ids and contact.dist <= 0.001:
                released = False
            if other in support_geoms:
                mujoco.mj_contactForce(model, data, i, force)
                normal_component = 1.0
                if support_up_world is not None:
                    sign = -1.0 if contact.geom1 in self.block_geom_ids[name] else 1.0
                    normal_component = max(0.0, float(sign * np.dot(contact.frame[:3], support_up_world)))
                support_force += max(0.0, force[0]) * normal_component
        return support_force, released

    def _base_conditions(self, name):
        data = self.sim.data._data
        table_rot = data.xmat[self.table_body_id].reshape(3, 3)
        rotation = table_rot.T @ data.xmat[self.block_body_ids[name]].reshape(3, 3)
        tilt = float(np.arccos(np.clip(np.max(np.abs(rotation[2, :])), -1.0, 1.0)))
        table_force, released = self._contacts(
            name, {self.table_geom_id}, table_rot[:, 2] if self.joint == "plain" else None
        )
        return {"on_table": bool(table_force > 0.01), "upright": bool(tilt <= TILT_MAX_RAD), "released": released}

    def _block_conditions(self, index, name, sample_time_s):
        data = self.sim.data._data
        body = self.block_body_ids[name]
        support_body, support_geoms, support_half = self._support_ids(index)
        support_rot = data.xmat[support_body].reshape(3, 3)
        relative = support_rot.T @ (data.xpos[body] - data.xpos[support_body])
        rotation = support_rot.T @ data.xmat[body].reshape(3, 3)
        # A cube counts as upright when any face points up.
        history = self._history[name]
        history.append((sample_time_s, relative.copy(), rotation.copy()))
        while history and sample_time_s - history[0][0] > DRIFT_WINDOW_S + 1e-9:
            history.popleft()
        covered = sample_time_s - history[0][0] >= DRIFT_WINDOW_S - 0.02
        drift = max(np.linalg.norm(relative - p) for _, p, _ in history)
        turn = max(np.arccos(np.clip((np.trace(r.T @ rotation) - 1) / 2, -1.0, 1.0)) for _, _, r in history)
        recent = [(p, r) for t, p, r in history if sample_time_s - t <= 0.1 + 1e-9]
        heights = [abs(p[2] - (support_half + BLOCK_HALF_M)) for p, _ in recent]
        offsets = [np.linalg.norm(p[:2]) for p, _ in recent]
        tilts = [np.arccos(np.clip(np.max(np.abs(r[2, :])), -1.0, 1.0)) for _, r in recent]
        table_up = data.xmat[self.table_body_id].reshape(3, 3)[:, 2]
        support_force, released = self._contacts(name, support_geoms, table_up if self.joint == "plain" else None)
        seated = float(np.median(heights)) <= (0.003 if self.joint == "plain" else 0.001)
        # Plain accepts staggered/yawed cubes; physical overlap replaces the legacy centre alignment gate.
        aligned = True
        overlap = plain_footprints_overlap(relative, rotation) if self.joint == "plain" else True
        upright = float(np.median(tilts)) <= (TILT_MAX_RAD if self.joint == "plain" else np.deg2rad(3.0))
        return {
            "on_support": bool(support_force > 0.01 and seated and overlap and released),
            "supported": bool(support_force > 0.01),
            "aligned": bool(aligned),
            "seated": bool(seated),
            "upright": bool(upright),
            "released": released,
            "stable": bool(covered and drift <= DRIFT_POSITION_M and turn <= DRIFT_ANGLE_RAD),
        }

    def _new_diagnostics(self):
        """Create bounded per-episode placement diagnostics."""
        placed = BLOCK_NAMES[1:]
        return {
            "first_contact_offset_m": {name: None for name in placed},
            "first_contact_deck_phase": {name: None for name in placed},
            "time_to_seat_s": {name: None for name in placed},
            "seat_time_s": {name: None for name in placed},
            "seated_relative_yaw_deg": {name: None for name in placed},
            "max_joint_force_n": {name: 0.0 for name in placed},
            "post_seat_rattle_m": {name: 0.0 for name in placed},
            "drift_since_placement_m": {name: 0.0 for name in placed},
            "bottom_block_drift_m": {"cumulative_m": 0.0, "max_tilt_deg": 0.0},
            "wrong_order_events": {"count": 0, "events": [], "blocked_by_taper": 0},
            "physics_fault": None,
            "realized_peak_normal_accel_g": 0.0,
        }

    def _sample_diagnostics(self, sample_time_s):
        """Update diagnostic latches from the same 100 Hz contact sample as stage logic."""
        model, data = self.sim.model._model, self.sim.data._data
        diagnostics = self._diagnostics
        geoms = {geom: name for name, ids in self.block_geom_ids.items() for geom in ids}
        forces = {name: {} for name in BLOCK_NAMES}
        robot_touch = set()
        joint_forces = {name: 0.0 for name in BLOCK_NAMES}
        force = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist < -0.003 and diagnostics["physics_fault"] is None:
                diagnostics["physics_fault"] = float(sample_time_s)
            for geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                name = geoms.get(geom)
                if name is None:
                    continue
                if other in self.robot_geom_ids and contact.dist <= 0.001:
                    robot_touch.add(name)
                support = geoms.get(other)
                if other == self.table_geom_id:
                    support = "table"
                if support is None or support == name:
                    continue
                if (
                    support != "table"
                    and data.xpos[self.block_body_ids[support], 2] >= data.xpos[self.block_body_ids[name], 2]
                ):
                    continue
                mujoco.mj_contactForce(model, data, i, force)
                forces[name][support] = forces[name].get(support, 0.0) + max(0.0, float(force[0]))
                if self.joint != "plain" and (
                    (name == "green" and other in self._joint_wall_geoms["blue_green"])
                    or (name == "yellow" and other in self._joint_wall_geoms["green_yellow"])
                ):
                    joint_forces[name] = max(joint_forces[name], float(force[0]))
        self._current_joint_force = joint_forces
        for name in BLOCK_NAMES:
            body = self.block_body_ids[name]
            adr = model.jnt_dofadr[model.body_jntadr[body]]
            if (
                name not in robot_touch
                and np.linalg.norm(data.qvel[adr : adr + 3]) > 0.5
                and diagnostics["physics_fault"] is None
            ):
                diagnostics["physics_fault"] = float(sample_time_s)
            table_top = data.geom_xpos[self.table_geom_id, 2] + model.geom_size[self.table_geom_id, 2]
            if data.xpos[body, 2] < table_top - 0.05:
                self._task_rule_violation = True
        base = self.stack_state["order"][0]
        base_body = self.block_body_ids[base]
        base_xy = data.xpos[base_body, :2].copy()
        if self._previous_base_xy is not None:
            diagnostics["bottom_block_drift_m"]["cumulative_m"] += float(
                np.linalg.norm(base_xy - self._previous_base_xy)
            )
        self._previous_base_xy = base_xy
        base_rotation = data.xmat[self.table_body_id].reshape(3, 3).T @ data.xmat[base_body].reshape(3, 3)
        base_tilt = np.degrees(np.arccos(np.clip(np.max(np.abs(base_rotation[2, :])), -1.0, 1.0)))
        diagnostics["bottom_block_drift_m"]["max_tilt_deg"] = max(
            diagnostics["bottom_block_drift_m"]["max_tilt_deg"], float(base_tilt)
        )
        active_wrong = set()
        base_support = max(forces[base], key=forces[base].get) if forces[base] else None
        if base_support not in {None, "table"}:
            active_wrong.add((base, base_support))
        for index, name in enumerate(self.stack_state["order"][1:], start=1):
            expected = self.stack_state["order"][index - 1]
            actual = max(forces[name], key=forces[name].get) if forces[name] else None
            if actual not in {None, "table", expected}:
                active_wrong.add((name, actual))
            if self.joint != "plain" and name == "yellow" and actual == "blue" and not self._conditions[name]["seated"]:
                active_wrong.add(("yellow", "blue"))
                if ("yellow", "blue") not in self._wrong_contact_active:
                    diagnostics["wrong_order_events"]["blocked_by_taper"] += 1
            body, support_body = self.block_body_ids[name], self.block_body_ids[expected]
            support_rotation = data.xmat[support_body].reshape(3, 3)
            relative = support_rotation.T @ (data.xpos[body] - data.xpos[support_body])
            offset = float(np.linalg.norm(relative[:2]))
            if name in robot_touch and self._grasp_time[name] is None:
                self._grasp_time[name] = float(sample_time_s)
            first = diagnostics["first_contact_offset_m"][name]
            entered = (
                self.joint != "plain"
                and np.max(np.abs(relative[:2])) <= 0.03
                and 2 * BLOCK_HALF_M - 0.003 <= relative[2] <= 2 * BLOCK_HALF_M + parts.TENON_LENGTH_M
            )
            if entered:
                self._entered_joint[name] = True
                self._jam_since[name] = self._jam_since[name] or float(sample_time_s)
            if first is None and (
                (self.joint == "plain" and forces[name].get(expected, 0) > 0.01) or (self.joint != "plain" and entered)
            ):
                diagnostics["first_contact_offset_m"][name] = offset
                motion = self.deck_driver.trajectory.evaluate(sample_time_s)
                diagnostics["first_contact_deck_phase"][name] = {
                    "displacement_m": np.asarray(motion.q[:3], float).tolist(),
                    "acceleration_m_s2": np.asarray(motion.qdd[:3], float).tolist(),
                }
            seated = self._conditions[name]["seated"] and forces[name].get(expected, 0) > 0.01
            if seated:
                if not self._was_seated[name]:
                    diagnostics["seat_time_s"][name] = float(sample_time_s)
                    diagnostics["time_to_seat_s"][name] = (
                        None if self._grasp_time[name] is None else float(sample_time_s - self._grasp_time[name])
                    )
                    self._seat_relative_xy[name] = relative[:2].copy()
                self._ever_seated[name] = True
                self._unseated_since[name] = None
                self._jam_since[name] = None
                rotation = support_rotation.T @ data.xmat[body].reshape(3, 3)
                diagnostics["seated_relative_yaw_deg"][name] = float(
                    np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
                )
            elif self._ever_seated[name] and self._unseated_since[name] is None:
                self._unseated_since[name] = float(sample_time_s)
            self._was_seated[name] = bool(seated)
            if self._seat_relative_xy[name] is not None:
                if self.joint == "plain":
                    previous = self._previous_placed_xy[name]
                    if previous is not None:
                        diagnostics["drift_since_placement_m"][name] += float(np.linalg.norm(relative[:2] - previous))
                    self._previous_placed_xy[name] = relative[:2].copy()
                else:
                    value = float(np.linalg.norm(relative[:2] - self._seat_relative_xy[name]))
                    diagnostics["post_seat_rattle_m"][name] = max(diagnostics["post_seat_rattle_m"][name], value)
            diagnostics["max_joint_force_n"][name] = max(diagnostics["max_joint_force_n"][name], joint_forces[name])
        for name, support in active_wrong - self._wrong_contact_active:
            events = diagnostics["wrong_order_events"]
            events["count"] += 1
            events["events"].append({"time_s": float(sample_time_s), "block": name, "support": support})
        self._wrong_contact_active = active_wrong
        motion = self.deck_driver.trajectory.evaluate(sample_time_s)
        point = np.asarray(self.deck_driver.trajectory.config.workpiece_point_offset_m, float)
        normal_accel = motion.qdd[2] + np.cross(motion.qdd[3:], point)[2]
        diagnostics["realized_peak_normal_accel_g"] = max(
            diagnostics["realized_peak_normal_accel_g"], float(abs(normal_accel) / 9.81)
        )

    def _failure_category(self):
        """Return the highest-priority terminal failure seen so far."""
        if self._diagnostics["physics_fault"] is not None:
            return "physics_fault"
        if self._success:
            return None
        if self._task_rule_violation:
            return "fell_off_table"
        if self._conditions.get("blue", {}).get("upright") is False:
            return "base_toppled"
        if self.joint != "plain":
            if any(t is not None and self._last_metric_time - t > 0.3 for t in self._unseated_since.values()):
                return "popped_out"
            if any(t is not None and self._last_metric_time - t > 1.0 for t in self._jam_since.values()):
                return "jammed"
        if self._diagnostics["wrong_order_events"]["count"]:
            return "wrong_order"
        if self.joint != "plain" and any(not value for value in self._entered_joint.values()):
            return "missed"
        if self.joint == "plain" and any(self._ever_seated.values()):
            for name in BLOCK_NAMES[1:]:
                if self._ever_seated[name] and not self._conditions.get(name, {}).get("supported", False):
                    return "toppled" if not self._conditions[name].get("upright", False) else "slid_off"
        return "incomplete"

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        obs.update(self.table_imu_provider.observation(self.sim))
        for prefix, body in [(f"{name}_block", block.root_body) for name, block in self.blocks.items()]:
            pose = pose_twist_in_frame(self.sim, body, self.robot_base_body_name)
            obs[f"{prefix}_pos_robot_base"] = pose.position_m
            obs[f"{prefix}_quat_wxyz_robot_base"] = pose.quaternion_wxyz
        assert_policy_observation_is_clean(obs)
        return obs

    def observation_contract(self):
        """Describe the task's policy-visible object poses and IMU samples."""
        contract = {key: dict(POLICY_FIELD_CONTRACT[key]) for key in self.table_imu_provider.policy_keys}
        for key, shape, units, frame in (
            ("robot0_joint_pos", (7,), "rad", "robot_base"),
            ("robot0_joint_vel", (7,), "rad/s", "robot_base"),
            ("robot0_gripper_qpos", (2,), "m", "gripper"),
            ("robot0_gripper_qvel", (2,), "m/s", "gripper"),
        ):
            contract[key] = {"shape": shape, "dtype": "float64", "units": units, "frame": frame}
        for name in BLOCK_NAMES:
            for suffix, shape, units in (("pos", (3,), "m"), ("quat_wxyz", (4,), "unit quaternion wxyz")):
                contract[f"{name}_block_{suffix}_robot_base"] = {
                    "shape": shape,
                    "dtype": "float64",
                    "units": units,
                    "frame": "robot_base",
                }
        return {
            key: {
                **value,
                "time": "delayed acquisition window" if key.startswith("table_imu") else "current control step",
            }
            for key, value in contract.items()
        }

    def get_policy_task_context(self):
        return {
            "task_type": "stack_blocks",
            "version": 1,
            "scoreable": self.physics_profile.scoreable and self.joint in {"plain", "tenon"},
            "order": list(self.stack_state["order"]),
            "block_half_m": BLOCK_HALF_M,
            "joint": self.joint,
            "joint_geometry_version": parts.JOINT_GEOMETRY_VERSION if self.joint != "plain" else 1,
            "joint_geometry": parts.joint_geometry(self.joint),
            "hold_duration_s": HOLD_DURATION_S,
            "align_tolerance_m": None if self.joint == "plain" else ALIGN_TOLERANCE_M,
            "success_rule": PLAIN_SUCCESS_RULE if self.joint == "plain" else "native_tenon_v1",
            "plain_requires_centre_alignment": False,
            "geometry_profile": self.geometry_profile["profile_id"],
        }

    def get_metrics(self):
        return {
            "success": {
                "passed": bool(self._success),
                "stage": self._stage,
                "subconditions": deepcopy(self._conditions),
                "diagnostics": deepcopy(self._diagnostics),
            },
            "task_rule_violation": bool(self._task_rule_violation),
            "failure_category": self._failure_category(),
        }


register_task(
    "stack_blocks",
    TaskDefinition(
        env_factory=StackBlocks,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"stack_state": state},
        describe=lambda state: {
            "task_id": f"stack_blocks.{state['joint']}",
            "instruction": (
                "Stack the green block on the blue block, then the yellow block on the green block."
                if state["joint"] == "plain"
                else (
                    "Fit the green block onto the square tapered tenon of the blue block, "
                    "then fit the yellow block onto the round tapered tenon of the green block."
                )
            ),
        },
        fingerprint=lambda state: {
            **{key: value for key, value in state.items() if key not in {"state_id", "split"}},
            "joint_geometry_version": parts.JOINT_GEOMETRY_VERSION if state["joint"] != "plain" else 1,
        },
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
