"""ShakeBench Phase 04 open-table Can pick-and-place environment."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from copy import deepcopy

import mujoco
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import ShakeBenchArena
from robosuite.models.objects import CanObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.shakebench_deck import DeckDriver, DeckDriverConfig, audit_compiled_deck_model
from robosuite.utils.shakebench_metrics import (
    CANONICAL_CAN_COLLISION_ENVELOPE,
    CANONICAL_CAN_COM_M,
    CANONICAL_CAN_MASS_KG,
    CONTACT_INTERFACE_FINGER_OBJECT,
    CONTACT_INTERFACE_TABLE_OBJECT,
    CONTACT_INTERFACE_TARGET_OBJECT,
    DEFAULT_SUCCESS_THRESHOLDS,
    ShakeBenchMetrics,
    ShakeBenchMetricsError,
    VibrationSuccessEvaluator,
    audit_can_compiled_model,
    audit_contact_pairs,
    can_pose_twist_in_frame,
    equivalent_cylinder_inertia,
    extract_can_collision_envelope,
    frame_world_position,
)

CAN_START_XY_M = (-0.10, -0.13)
TARGET_CENTER_XY_M = (-0.10, 0.17)
DEFAULT_TABLE_OFFSET_M = (0.0, 0.0, 0.8)
DEFAULT_MODEL_TIMESTEP_S = 0.0002
DEFAULT_TABLE_FRICTION = (1.0, 0.005, 0.0001)
DEFAULT_CONTACT_TORSIONAL_MU = 0.005
DEFAULT_CONTACT_ROLLING_MU = 0.0001
DEFAULT_DECK_EQ_SOLREF = (2.0 * DEFAULT_MODEL_TIMESTEP_S, 0.5)
CAN_CONTACT_BIT = 2


def _finite_vector(name: str, value: Iterable[float], length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} finite values")
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite values") from exc
    if array.size != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(item) for item in array)


def _one_value(name: str, value, *, allowed: tuple[str, ...]) -> str:
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(values) != 1 or values[0] not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(f"{name} must select exactly one of {allowed_text}")
    return values[0]


class VibrationPickPlaceCan(ManipulationEnv):
    """Single-Panda open-table Can pick/place scene for ShakeBench v0.

    The task deliberately has no source bin.  The Can is a world child with a
    freejoint; the target is a shallow, five-geom assembly rigidly attached to
    the isolated worktable.  The Phase 02 deck processor is installed before
    the first compilation, so the robot base and worktable can be reparented
    by explicit roles without changing the Can topology.
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise=None,
        table_full_size=(0.65, 0.60, 0.06),
        table_friction=DEFAULT_TABLE_FRICTION,
        table_offset=DEFAULT_TABLE_OFFSET_M,
        target_container_friction=(0.30, DEFAULT_CONTACT_TORSIONAL_MU, DEFAULT_CONTACT_ROLLING_MU),
        can_start_xy=CAN_START_XY_M,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        deck_trajectory=None,
        deck_config=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        model_timestep=DEFAULT_MODEL_TIMESTEP_S,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        load_model_on_init=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        requested_robots = list(robots) if isinstance(robots, (list, tuple)) else [robots]
        if requested_robots != ["Panda"]:
            raise ValueError("VibrationPickPlaceCan currently supports exactly one Panda robot")
        if env_configuration != "default":
            raise ValueError("VibrationPickPlaceCan only supports env_configuration='default'")
        _one_value("base_types", base_types, allowed=("default",))
        _one_value("gripper_types", gripper_types, allowed=("default", "PandaGripper"))
        self.table_full_size = _finite_vector("table_full_size", table_full_size, 3)
        self.table_friction = _finite_vector("table_friction", table_friction, 3)
        self.table_offset = _finite_vector("table_offset", table_offset, 3)
        self.target_container_friction = _finite_vector("target_container_friction", target_container_friction, 3)
        self.can_start_xy = _finite_vector("can_start_xy", can_start_xy, 2)
        if any(value < 0.0 for value in self.table_friction + self.target_container_friction):
            raise ValueError("friction values must be non-negative")
        if model_timestep is None:
            model_timestep = DEFAULT_MODEL_TIMESTEP_S
        if (
            isinstance(model_timestep, (bool, np.bool_))
            or not np.isfinite(float(model_timestep))
            or float(model_timestep) <= 0.0
        ):
            raise ValueError("model_timestep must be finite and positive")
        self._phase04_model_timestep = float(model_timestep)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.policy_task_state_frame = "robot_base"
        self.table_object_sliding_mu = 0.30
        self.finger_object_sliding_mu = 1.00
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self._requested_load_model_on_init = bool(load_model_on_init)
        self._compiled_contract = None
        self.metrics = None
        self.success_evaluator = VibrationSuccessEvaluator(DEFAULT_SUCCESS_THRESHOLDS)

        if deck_config is not None and not isinstance(deck_config, DeckDriverConfig):
            raise ValueError("deck_config must be a DeckDriverConfig")
        if deck_config is None:
            deck_config = DeckDriverConfig(
                physics_timestep_s=self._phase04_model_timestep,
                eq_solref=(2.0 * self._phase04_model_timestep, DEFAULT_DECK_EQ_SOLREF[1]),
            )
        elif deck_config.physics_timestep_s is not None and not np.isclose(
            deck_config.physics_timestep_s, self._phase04_model_timestep, rtol=0.0, atol=1e-14
        ):
            raise ValueError("deck_config.physics_timestep_s must equal model_timestep")
        self.deck_config = deck_config
        self.deck_driver = DeckDriver(
            trajectory=deck_trajectory,
            config=self.deck_config,
            body_handles={"isolated_worktable": "worktable", "robot_base": "robot0_base"},
            required_roles=("isolated_worktable", "robot_base"),
        )

        # The driver must be installed before the first model compile.  The
        # parent supports load_model_on_init=False as an additive seam; the
        # public environment still preserves the usual eager construction
        # behavior by explicitly resetting below when requested.
        super().__init__(
            robots=requested_robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            load_model_on_init=False,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
            model_timestep=self._phase04_model_timestep,
        )
        # ``MujocoEnv`` fills ``model_timestep`` after compilation.  Keep the
        # requested value available during the pre-compile task assembly too.
        self.model_timestep = self._phase04_model_timestep
        self.load_model_on_init = self._requested_load_model_on_init
        self.deck_driver.install(self)
        self.add_sim_initialization_hook(self._audit_compiled_contract)
        self.add_post_physics_step_hook(self._record_post_physics_metrics)
        if self._requested_load_model_on_init:
            self.reset()

    def _configure_can(self):
        """Add the explicit canonical inertial and restrict contact to pairs."""

        can_body = self.can.get_obj()
        # Explicit contact pairs are the only Can interaction seam.  MuJoCo
        # still applies the contact bit test before considering a pair, so use
        # a dedicated bit for the listed partner geoms.  Ordinary world and
        # robot geoms keep bit 1 and cannot silently add Can contacts.
        for geom_name in self.can.contact_geoms:
            geom = can_body.find(f".//geom[@name='{geom_name}']")
            if geom is None:
                raise ShakeBenchMetricsError(f"Can collision geom {geom_name!r} is missing")
            geom.set("contype", str(CAN_CONTACT_BIT))
            geom.set("conaffinity", "0")
            geom.attrib.pop("density", None)
        self.can_inertia = None

    def _measure_can_collision_envelope(self):
        """Measure the compiled mesh support bounds before task assembly."""

        # CanObject keeps its free joint inside the extracted object body, so
        # compile a tiny valid world containing a copied asset/body subtree.
        # This is read-only and avoids treating the stock bottom_site as a
        # collision vertex when placing the object on the tabletop.
        root = ET.Element("mujoco", {"model": "shakebench_can_probe"})
        root.append(deepcopy(self.can.asset))
        worldbody = ET.SubElement(root, "worldbody")
        worldbody.append(deepcopy(self.can.get_obj()))
        probe_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        envelope = extract_can_collision_envelope(
            probe_model,
            self.can.root_body,
            self.can.contact_geoms,
        )
        envelope.assert_matches(CANONICAL_CAN_COLLISION_ENVELOPE)
        self.can_collision_envelope = envelope
        self.can_collision_envelope_bottom_m = envelope.lower_support_z_m
        self.can_collision_envelope_top_m = envelope.upper_support_z_m
        self.can_collision_envelope_radius_m = envelope.support_radius_m
        self.can_placement_z_offset_m = -envelope.lower_support_z_m
        inertia = equivalent_cylinder_inertia(
            CANONICAL_CAN_MASS_KG,
            envelope.support_radius_m,
            envelope.height_m,
        )
        self.can_inertia = tuple(float(value) for value in inertia)
        can_body = self.can.get_obj()
        if can_body.find("./inertial") is not None:
            raise ShakeBenchMetricsError("Can asset unexpectedly already contains an inertial")
        can_body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": array_to_string(CANONICAL_CAN_COM_M),
                    "mass": format(CANONICAL_CAN_MASS_KG, ".17g"),
                    "diaginertia": array_to_string(inertia),
                },
            ),
        )
        return envelope

    def _append_contact_pairs(self):
        can_geom_names = tuple(self.can.contact_geoms)
        table_geom_names = (self.arena.table_collision.get("name"),)
        target_collision_names = tuple(
            name for key, name in self.arena.target_container_geom_names.items() if not key.endswith("_visual")
        )
        target_bottom_names = tuple(
            name for key, name in self.arena.target_container_geom_names.items() if key == "bottom"
        )
        target_wall_names = tuple(
            name
            for key, name in self.arena.target_container_geom_names.items()
            if key.startswith("wall_") and not key.endswith("_visual")
        )
        finger_pad_names = tuple(self.finger_pad_geom_names)
        table_friction = array_to_string(
            (self.table_object_sliding_mu, DEFAULT_CONTACT_TORSIONAL_MU, DEFAULT_CONTACT_ROLLING_MU)
        )
        finger_friction = array_to_string(
            (self.finger_object_sliding_mu, DEFAULT_CONTACT_TORSIONAL_MU, DEFAULT_CONTACT_ROLLING_MU)
        )
        partner_names = table_geom_names + target_collision_names + finger_pad_names
        for geom_name in partner_names:
            geom = self.model.worldbody.find(f".//geom[@name='{geom_name}']")
            if geom is None:
                raise ShakeBenchMetricsError(f"contact partner geom {geom_name!r} is missing")
            geom.set("conaffinity", str(CAN_CONTACT_BIT))
        for can_name in can_geom_names:
            for table_name in table_geom_names:
                ET.SubElement(
                    self.model.contact, "pair", {"geom1": can_name, "geom2": table_name, "friction": table_friction}
                )
            for target_name in target_collision_names:
                ET.SubElement(
                    self.model.contact, "pair", {"geom1": can_name, "geom2": target_name, "friction": table_friction}
                )
            for finger_name in finger_pad_names:
                ET.SubElement(
                    self.model.contact, "pair", {"geom1": can_name, "geom2": finger_name, "friction": finger_friction}
                )
        self.table_contact_geom_names = table_geom_names
        self.target_collision_geom_names = target_collision_names
        self.target_bottom_geom_names = target_bottom_names
        self.target_wall_geom_names = target_wall_names
        self.contact_roles = {
            CONTACT_INTERFACE_TABLE_OBJECT: tuple(
                (can_name, table_name) for can_name in can_geom_names for table_name in table_geom_names
            ),
            CONTACT_INTERFACE_TARGET_OBJECT: tuple(
                (can_name, target_name) for can_name in can_geom_names for target_name in target_collision_names
            ),
            CONTACT_INTERFACE_FINGER_OBJECT: tuple(
                (can_name, finger_name) for can_name in can_geom_names for finger_name in finger_pad_names
            ),
        }

    def _load_model(self):
        super()._load_model()
        if len(self.robots) != 1 or self.robot_names != ["Panda"]:
            raise ValueError("VibrationPickPlaceCan requires exactly one Panda")

        self.arena = ShakeBenchArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            include_target_container=False,
            visual=True,
        )
        self.arena.add_target_container(friction=self.target_container_friction)
        base_position = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(np.asarray(base_position, dtype=float))
        self.robot_base_body_name = self.robots[0].robot_model.root_body
        if self.robot_base_body_name != "robot0_base":
            raise ValueError("Panda base role changed; refusing implicit deck body lookup")
        if type(self.robots[0].gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("VibrationPickPlaceCan requires the standard PandaGripper")
        self.finger_pad_geom_names = tuple(
            self.robots[0].gripper["right"].important_geoms["left_fingerpad"]
            + self.robots[0].gripper["right"].important_geoms["right_fingerpad"]
        )
        self.gripper_body_name = self.robots[0].robot_model.eef_name["right"]

        self.can = CanObject(name="can")
        self._configure_can()
        self._measure_can_collision_envelope()
        if self.placement_initializer is None:
            self.placement_initializer = UniformRandomSampler(
                name="OpenTableCanSampler",
                mujoco_objects=self.can,
                x_range=(self.can_start_xy[0], self.can_start_xy[0]),
                y_range=(self.can_start_xy[1], self.can_start_xy[1]),
                rotation=0.0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.arena.table_top_abs,
                z_offset=self.can_placement_z_offset_m,
                rng=self.rng,
            )
        else:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.can)

        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.can,
        )
        self._append_contact_pairs()
        target_spec = self.arena.target_container_spec
        target_frame_local_origin = np.array(
            [
                target_spec["center_xy_m"][0],
                target_spec["center_xy_m"][1],
                self.arena.table_half_size[2] + target_spec["bottom_thickness_m"],
            ],
            dtype=float,
        )
        self.metrics = ShakeBenchMetrics(
            can_body_name=self.can.root_body,
            can_geom_names=self.can.contact_geoms,
            robot_base_body_name=self.robot_base_body_name,
            worktable_body_name=self.arena.worktable_body_name,
            target_frame_local_origin_m=target_frame_local_origin,
            target_inner_xy_m=target_spec["inner_xy_m"],
            table_geom_names=self.table_contact_geom_names,
            target_bottom_geom_names=self.target_bottom_geom_names,
            target_wall_geom_names=self.target_wall_geom_names,
            finger_pad_geom_names=self.finger_pad_geom_names,
            gripper_body_name=self.gripper_body_name,
            deck_body_name=self.deck_config.deck_body_name,
            deck_driver=self.deck_driver,
            dt_s=self._phase04_model_timestep,
        )
        self.can_body_name = self.can.root_body
        self.can_geom_name = self.can.contact_geoms[0]
        self.worktable_body_name = self.arena.worktable_body_name
        self.deck_body_name = self.deck_config.deck_body_name
        self.target_container_geom_names = dict(self.arena.target_container_geom_names)
        self.target_frame_local_origin_m = target_frame_local_origin.copy()
        self.target_inner_xy_m = tuple(float(value) for value in target_spec["inner_xy_m"])

    def _setup_references(self):
        super()._setup_references()
        self.can_body_id = self.sim.model.body_name2id(self.can.root_body)
        self.worktable_body_id = self.sim.model.body_name2id(self.arena.worktable_body_name)
        self.robot_base_body_id = self.sim.model.body_name2id(self.robot_base_body_name)
        self.can_geom_ids = [self.sim.model.geom_name2id(name) for name in self.can.contact_geoms]
        self.can_geom_id = self.can_geom_ids[0]
        compiled_envelope = extract_can_collision_envelope(
            self.sim,
            self.can.root_body,
            self.can.contact_geoms,
        )
        compiled_envelope.assert_matches(self.can_collision_envelope)
        self.can_collision_envelope = compiled_envelope

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        @sensor(modality=modality)
        def can_pos_robot_base(obs_cache):
            return can_pose_twist_in_frame(
                self.sim,
                self.can.root_body,
                self.robot_base_body_name,
            ).position_m

        @sensor(modality=modality)
        def can_quat_robot_base(obs_cache):
            return T.convert_quat(
                can_pose_twist_in_frame(
                    self.sim,
                    self.can.root_body,
                    self.robot_base_body_name,
                ).quaternion_wxyz,
                to="xyzw",
            )

        @sensor(modality=modality)
        def can_to_target_pos(obs_cache):
            return can_pose_twist_in_frame(
                self.sim,
                self.can.root_body,
                self.arena.worktable_body_name,
                frame_local_origin_m=self.metrics.target_frame_local_origin_m,
            ).position_m

        for name, sensor_fn in (
            ("can_pos_robot_base", can_pos_robot_base),
            ("can_quat_robot_base", can_quat_robot_base),
            ("can_to_target_pos", can_to_target_pos),
        ):
            observables[name] = Observable(name=name, sensor=sensor_fn, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample(on_top=False)
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate((np.asarray(obj_pos), np.asarray(obj_quat))))
        if self.metrics is not None:
            self.metrics.reset()
        self.success_evaluator.reset()

    def _audit_compiled_contract(self, sim):
        # This callback executes after the Phase 02 driver has validated and
        # bound the compiled deck.  It is intentionally an assertion seam, not
        # a recovery path.
        self._compiled_contract = self.audit_compiled_model(sim)

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        if self.metrics is None:
            return
        # DeckDriver keeps its full-rate response trace.  Task/contact
        # metrics are sampled at the policy boundary (20 Hz by contract),
        # which is sufficient for the 0.50 s success window and avoids doing
        # expensive collision-mesh support-point extraction for every internal
        # model step.
        if self._control_steps and (self._physics_step_index + 1) % self._control_steps != 0:
            return
        snapshot = self.metrics.update(self.sim, time_s=sample_time_s)
        evaluation = self.success_evaluator.evaluate(snapshot.success_snapshot, time_s=sample_time_s)
        self.metrics.attach_success(evaluation)

    def _sample_metrics(self):
        if self.metrics is None:
            raise ShakeBenchMetricsError("metrics are unavailable before model construction")
        snapshot = self.metrics.update(self.sim, time_s=float(self.sim.data.time))
        evaluation = self.success_evaluator.evaluate(snapshot.success_snapshot, time_s=float(self.sim.data.time))
        return self.metrics.attach_success(evaluation)

    def target_frame_world_position(self):
        return frame_world_position(
            self.sim,
            self.arena.worktable_body_name,
            frame_local_origin_m=self.metrics.target_frame_local_origin_m,
        )

    def get_metrics(self, *, update=False):
        """Return the latest serializable metrics report."""

        if update or self.metrics.latest is None:
            self._sample_metrics()
        return self.metrics.to_dict()

    def audit_compiled_model(self, sim_or_model=None):
        """Audit compiled topology, canonical Can inertia, target, and pairs."""

        if sim_or_model is None:
            sim_or_model = self.sim
        model = getattr(sim_or_model, "model", sim_or_model)
        raw_model = getattr(model, "_model", model)

        def parent_name(body_name):
            body_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if body_id < 0:
                raise ShakeBenchMetricsError(f"compiled model is missing body {body_name!r}")
            parent_id = int(raw_model.body_parentid[body_id])
            return None if parent_id == 0 else mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        can_audit = audit_can_compiled_model(
            sim_or_model,
            self.can.root_body,
            self.can.contact_geoms,
            expected_mass_kg=CANONICAL_CAN_MASS_KG,
            expected_com_m=CANONICAL_CAN_COM_M,
            expected_inertia_kg_m2=self.can_inertia,
        )
        compiled_envelope = extract_can_collision_envelope(
            sim_or_model,
            self.can.root_body,
            self.can.contact_geoms,
        )
        compiled_envelope.assert_matches(self.can_collision_envelope)
        can_audit["collision_envelope"] = compiled_envelope.to_dict()
        can_audit["placement_correction_z_offset_m"] = float(self.can_placement_z_offset_m)
        contact_audit = audit_contact_pairs(
            sim_or_model,
            can_geom_names=self.can.contact_geoms,
            table_geom_names=self.table_contact_geom_names,
            target_bottom_geom_names=self.target_bottom_geom_names,
            target_wall_geom_names=self.target_wall_geom_names,
            finger_pad_geom_names=self.finger_pad_geom_names,
            table_sliding_mu=self.table_object_sliding_mu,
            finger_sliding_mu=self.finger_object_sliding_mu,
        )
        target_spec = self.arena.target_container_spec
        target_collision_names = list(self.target_collision_geom_names)
        target_geometry = {}
        target_body_names = {
            mujoco.mj_id2name(
                raw_model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(raw_model.geom_bodyid[int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name))]),
            )
            for name in target_collision_names
        }
        if target_body_names != {self.arena.worktable_body_name}:
            raise ShakeBenchMetricsError("target collision geoms are not rigid children of worktable")
        for name in target_collision_names:
            geom_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name))
            target_geometry[name] = {
                "size_m": np.asarray(raw_model.geom_size[geom_id], dtype=float).tolist(),
                "pos_m_in_worktable": np.asarray(raw_model.geom_pos[geom_id], dtype=float).tolist(),
            }
        if parent_name(self.can.root_body) is not None:
            raise ShakeBenchMetricsError("Can must remain a direct world child")
        if parent_name(self.robot_base_body_name) != self.deck_config.deck_body_name:
            raise ShakeBenchMetricsError("Panda base is not a rigid child of the dynamic deck")
        if parent_name(self.arena.worktable_body_name) != self.deck_config.deck_body_name:
            raise ShakeBenchMetricsError("worktable is not a rigid child of the dynamic deck")
        deck_audit = audit_compiled_deck_model(
            sim_or_model,
            self.deck_config,
            self.deck_driver.role_handles,
        )
        worktable_audit = self.arena.audit_compiled_model(sim_or_model)
        return {
            "topology": {
                "deck_body": self.deck_config.deck_body_name,
                "can_parent": parent_name(self.can.root_body),
                "robot_base_parent": parent_name(self.robot_base_body_name),
                "worktable_parent": parent_name(self.arena.worktable_body_name),
                "worktable_body": self.arena.worktable_body_name,
                "target_body": next(iter(target_body_names)),
                "can_freejoint": can_audit["freejoint_names"],
                "target_collision_geom_names": target_collision_names,
            },
            "deck": deck_audit,
            "worktable": worktable_audit,
            "can": can_audit,
            "target_container": {
                "center_xy_m": list(target_spec["center_xy_m"]),
                "outer_xy_m": list(target_spec["outer_xy_m"]),
                "inner_xy_m": list(target_spec["inner_xy_m"]),
                "wall_thickness_m": target_spec["wall_thickness_m"],
                "wall_height_m": target_spec["wall_height_m"],
                "bottom_thickness_m": target_spec["bottom_thickness_m"],
                "collision_geom_names": target_collision_names,
                "collision_geometry": target_geometry,
            },
            "contacts": contact_audit,
        }

    def reward(self, action=None):
        """Sparse reward using the same latched success definition."""

        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        return bool(self._sample_metrics().success)

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

    def _check_robot_configuration(self, robots):
        requested = list(robots) if isinstance(robots, (list, tuple)) else [robots]
        if requested != ["Panda"]:
            raise ValueError("VibrationPickPlaceCan currently supports exactly one Panda robot")


__all__ = ["VibrationPickPlaceCan"]
