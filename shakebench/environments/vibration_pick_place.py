"""ShakeBench open-table pick-and-place environment for the current scene."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path as P
from collections.abc import Iterable
from copy import deepcopy

import mujoco
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.utils.artifacts import payload_hash
from shakebench.utils.calibration import CalibrationError, build_vibration_program, vibration_record
from shakebench.utils.deck import DeckDriver, DeckDriverConfig, audit_compiled_deck_model
from shakebench.utils.geometry import geometry_scene_path, load_geometry_profile
from shakebench.utils.isolator import AXES as ISOLATOR_AXES
from shakebench.utils.isolator import static_equilibrium_offset
from shakebench.utils.metrics import (
    CANONICAL_OBJECT_COLLISION_ENVELOPE,
    CANONICAL_OBJECT_COM_M,
    CANONICAL_OBJECT_MASS_KG,
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
from shakebench.utils.physics import PhysicsProfileError, resolve_physics_profile
from shakebench.utils.privilege import (
    PRIVILEGED_NAMESPACE,
    ShakeBenchPrivilegeError,
    assert_policy_observation_is_clean,
    make_privileged_recorder,
)
from shakebench.utils.providers import (
    POLICY_FIELD_CONTRACT,
    ShakeBenchProviderError,
    TableIMUProvider,
)
from shakebench.utils.scene import (
    DECK_VISUAL_BODY_NAME,
    SceneVisualConfig,
    audit_compiled_scene,
    configure_scene_rendering,
    load_scene_visual_config,
    scene_clearance_report,
)
from shakebench.utils.sensors import (
    CANONICAL_IMU_PROFILE,
    CANONICAL_IMU_PROFILE_HASH,
    ShakeBenchSensorError,
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
WRIST_CAMERA = "robot0_eye_in_hand"
DEFAULT_CAMERA_NAMES = ("agentview", WRIST_CAMERA)


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


class VibrationPickPlace(ManipulationEnv):
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
        object_start_xy=CAN_START_XY_M,
        object_start_yaw_rad=0.0,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        deck_trajectory=None,
        deck_config=None,
        physics_profile=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        model_timestep=None,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        load_model_on_init=True,
        camera_names=DEFAULT_CAMERA_NAMES,
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        table_imu_position_m=(0.0, 0.0, -0.03),
        table_imu_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        excitation_program=None,
        vibration=None,
        privileged_recorder=None,
        scene_config=None,
        scene_visual=True,
        geometry_profile="world_fixed_arm_v1",
        task=None,
    ):
        from shakebench.utils.tasks import TaskSpec

        self.task_spec = None if task is None else TaskSpec.from_mapping(task)
        self.object_mass_kg = CANONICAL_OBJECT_MASS_KG if self.task_spec is None else self.task_spec.object_mass_kg
        requested_robots = list(robots) if isinstance(robots, (list, tuple)) else [robots]
        if requested_robots != ["Panda"]:
            raise ValueError("VibrationPickPlace currently supports exactly one Panda robot")
        if env_configuration != "default":
            raise ValueError("VibrationPickPlace only supports env_configuration='default'")
        try:
            self.physics_profile = resolve_physics_profile(physics_profile)
        except PhysicsProfileError as exc:
            raise ValueError(str(exc)) from exc
        profile_timestep = self.physics_profile.model_timestep_s
        if model_timestep is None:
            model_timestep = profile_timestep
        elif not np.isclose(float(model_timestep), profile_timestep, rtol=0.0, atol=1e-14):
            raise ValueError("model_timestep must equal the selected physics profile timestep")
        if not np.isclose(float(control_freq), self.physics_profile.control_freq_hz, rtol=0.0, atol=1e-12):
            raise ValueError("control_freq must equal control_freq=20 Hz in the selected physics profile scheduler")
        expected_target_friction = (
            float(self.physics_profile.contact["sliding_mu"]["table_object"]),
            float(self.physics_profile.contact["torsional_mu"]),
            float(self.physics_profile.contact["rolling_mu"]),
        )
        if not np.allclose(
            np.asarray(target_container_friction, dtype=float),
            np.asarray(expected_target_friction, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("target_container_friction must equal the selected physics profile")
        self.geometry_profile = load_geometry_profile(geometry_profile)
        geometry = self.geometry_profile
        if base_types not in ("default", geometry["mount_type"]):
            raise ValueError("base_types conflicts with the selected geometry profile")
        if not np.allclose(table_offset, DEFAULT_TABLE_OFFSET_M, atol=1e-12, rtol=0):
            raise ValueError("table_offset is owned by the selected geometry profile")
        if scene_config is not None:
            raise ValueError("scene_config is owned by the selected geometry profile")
        base_types = geometry["mount_type"]
        table_offset = geometry["table_top_pos_m"]
        scene_config = geometry_scene_path(geometry_profile)
        base_types = _one_value(
            "base_types",
            base_types,
            allowed=("default", "RethinkMount", "RethinkMinimalMount", "NullMount"),
        )
        _one_value("gripper_types", gripper_types, allowed=("default", "PandaGripper"))
        self.base_types = base_types
        self.scene_config: SceneVisualConfig = load_scene_visual_config(scene_config)
        if P(self.scene_config.source_path).name != self.geometry_profile["scene_config"]:
            raise ValueError("geometry profile scene config mismatch")
        if not isinstance(scene_visual, (bool, np.bool_)):
            raise ValueError("scene_visual must be boolean")
        self.scene_visual = bool(scene_visual)
        self.table_full_size = _finite_vector("table_full_size", table_full_size, 3)
        self.table_friction = _finite_vector("table_friction", table_friction, 3)
        self.table_offset = _finite_vector("table_offset", table_offset, 3)
        self.target_container_friction = _finite_vector("target_container_friction", target_container_friction, 3)
        self.object_start_xy = _finite_vector("object_start_xy", object_start_xy, 2)
        self.object_start_yaw_rad = float(object_start_yaw_rad)
        if not np.isfinite(self.object_start_yaw_rad):
            raise ValueError("object_start_yaw_rad must be finite")
        if any(value < 0.0 for value in self.table_friction + self.target_container_friction):
            raise ValueError("friction values must be non-negative")
        if (
            isinstance(model_timestep, (bool, np.bool_))
            or not np.isfinite(float(model_timestep))
            or float(model_timestep) <= 0.0
        ):
            raise ValueError("model_timestep must be finite and positive")
        self._phase04_model_timestep = float(model_timestep)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.table_object_sliding_mu = float(self.physics_profile.contact["sliding_mu"]["table_object"])
        self.finger_object_sliding_mu = float(self.physics_profile.contact["sliding_mu"]["finger_object"])
        self.target_object_sliding_mu = self.table_object_sliding_mu
        if self.task_spec is not None:
            self.table_object_sliding_mu = self.task_spec.table_sliding_mu
            self.target_object_sliding_mu = self.task_spec.target_sliding_mu
            self.finger_object_sliding_mu = 1.0
            self.target_container_friction = (self.target_object_sliding_mu, *expected_target_friction[1:])
        self.use_object_obs = use_object_obs
        self.use_camera_obs = use_camera_obs
        self.imu_mode = imu_mode
        self.table_imu_position_m = _finite_vector("table_imu_position_m", table_imu_position_m, 3)
        self.table_imu_quat_wxyz = _finite_vector("table_imu_quat_wxyz", table_imu_quat_wxyz, 4)
        if np.linalg.norm(self.table_imu_quat_wxyz) <= 0.0:
            raise ValueError("table_imu_quat_wxyz must have non-zero norm")
        self.table_imu_quat_wxyz = tuple(
            np.asarray(self.table_imu_quat_wxyz) / np.linalg.norm(self.table_imu_quat_wxyz)
        )
        requested_imu_seed = seed if imu_seed is None and seed is not None else (0 if imu_seed is None else imu_seed)
        if isinstance(requested_imu_seed, (bool, np.bool_)):
            raise ValueError("imu_seed must be a non-negative integer")
        try:
            self.imu_seed = int(requested_imu_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("imu_seed must be a non-negative integer") from exc
        if self.imu_seed != requested_imu_seed or self.imu_seed < 0:
            raise ValueError("imu_seed must be a non-negative integer")
        try:
            self.privileged_recorder = make_privileged_recorder(privileged_recorder)
        except ShakeBenchPrivilegeError as exc:
            raise ValueError(str(exc)) from exc
        self._phase05_last_action = np.zeros(0, dtype=np.float32)
        self._imu_mount_audit = None
        self._scene_audit = None
        self._scene_clearance = None
        self.policy_task_state_frame = "robot_base"
        self.placement_initializer = placement_initializer
        self._requested_load_model_on_init = bool(load_model_on_init)
        self._compiled_contract = None
        self.metrics = None
        self.success_evaluator = VibrationSuccessEvaluator(DEFAULT_SUCCESS_THRESHOLDS)
        self._last_success_evaluation = None

        if deck_config is not None and not isinstance(deck_config, DeckDriverConfig):
            raise ValueError("deck_config must be a DeckDriverConfig")
        if deck_config is None:
            deck_config = self.physics_profile.deck_driver_config()
        elif deck_config.physics_timestep_s is not None and not np.isclose(
            deck_config.physics_timestep_s, self._phase04_model_timestep, rtol=0.0, atol=1e-14
        ):
            raise ValueError("deck_config.physics_timestep_s must equal model_timestep")
        if self.physics_profile.scoreable:
            try:
                self.physics_profile.assert_matches_deck_config(deck_config)
            except PhysicsProfileError as exc:
                raise ValueError(str(exc)) from exc
        self.deck_config = deck_config
        if vibration is not None:
            if deck_trajectory is not None or excitation_program is not None:
                raise ValueError("vibration cannot be combined with deck_trajectory or excitation_program")
            try:
                excitation_program = build_vibration_program(vibration)
            except CalibrationError as exc:
                raise ValueError(str(exc)) from exc
        self.vibration_record = (
            None if excitation_program is None or vibration is None else vibration_record(excitation_program)
        )
        runtime_trajectory = deck_trajectory
        if excitation_program is not None:
            if runtime_trajectory is not None and runtime_trajectory is not excitation_program:
                raise ValueError("deck_trajectory and excitation_program specify different trajectories")
            runtime_trajectory = excitation_program
        self.deck_driver = DeckDriver(
            trajectory=runtime_trajectory,
            config=self.deck_config,
            body_handles={
                "isolated_worktable": "worktable",
                "deck_visual": DECK_VISUAL_BODY_NAME,
            },
            required_roles=("isolated_worktable", "deck_visual"),
        )
        try:
            self.table_imu_provider = TableIMUProvider(
                seed=self.imu_seed,
                imu_mode=imu_mode,
                sensor_position_body_m=self.table_imu_position_m,
                sensor_quat_body_wxyz=self.table_imu_quat_wxyz,
            )
        except (ShakeBenchProviderError, ShakeBenchSensorError) as exc:
            raise ValueError(str(exc)) from exc

        # The driver must be installed before the first model compile.  The
        # parent supports load_model_on_init=False as an additive seam; the
        # public environment still preserves the usual eager construction
        # behavior by explicitly resetting below when requested.
        super().__init__(
            robots=requested_robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
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
        # Set environment-owned solver/timestep options before the deck XML
        # processor generates the dynamic deck.  This profile is the only
        # source of score-affecting physics for the official environment.
        self.set_xml_processor(self.physics_profile.process_xml)
        self.deck_driver.install(self)
        self.add_sim_initialization_hook(self._audit_compiled_contract)
        self.add_post_physics_step_hook(self._record_post_physics_metrics)
        # The sole table IMU is sampled independently of model-input selection.
        self.add_post_physics_step_hook(self._update_phase05_provider)
        if self.privileged_recorder is not None:
            self.add_post_physics_step_hook(self._record_phase05_privileged)
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
        if self.task_spec is None:
            envelope.assert_matches(CANONICAL_OBJECT_COLLISION_ENVELOPE)
        if self.task_spec is not None:
            from shakebench.utils.tasks import OBJECT_SUPPORT

            expected_support = OBJECT_SUPPORT[self.task_spec.object_id]
            actual_support = (envelope.lower_support_z_m, envelope.upper_support_z_m, envelope.support_radius_m)
            if not np.allclose(actual_support, expected_support, atol=1e-10, rtol=0):
                raise ShakeBenchMetricsError("task object collision support differs from its state contract")
        self.can_collision_envelope = envelope
        self.can_collision_envelope_bottom_m = envelope.lower_support_z_m
        self.can_collision_envelope_top_m = envelope.upper_support_z_m
        self.can_collision_envelope_radius_m = envelope.support_radius_m
        self.can_placement_z_offset_m = -envelope.lower_support_z_m
        inertia = equivalent_cylinder_inertia(
            self.object_mass_kg,
            envelope.support_radius_m,
            envelope.height_m,
        )
        self.can_com = CANONICAL_OBJECT_COM_M
        inertial_quat = (1.0, 0.0, 0.0, 0.0)
        if self.task_spec is not None:
            body_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_BODY, self.can.root_body)
            inertia = probe_model.body_inertia[body_id] * (self.object_mass_kg / probe_model.body_mass[body_id])
            self.can_com = tuple(probe_model.body_ipos[body_id])
            inertial_quat = tuple(probe_model.body_iquat[body_id])
        self.can_inertia = tuple(float(value) for value in inertia)
        can_body = self.can.get_obj()
        if can_body.find("./inertial") is not None:
            raise ShakeBenchMetricsError("Can asset unexpectedly already contains an inertial")
        can_body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": array_to_string(self.can_com),
                    "quat": array_to_string(inertial_quat),
                    "mass": format(self.object_mass_kg, ".17g"),
                    "diaginertia": array_to_string(inertia),
                },
            ),
        )
        return envelope

    def _append_contact_pairs(self):
        can_geom_names = tuple(self.can.contact_geoms)
        table_geom_names = (self.arena.object_support_geom.get("name"),)
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
        table_pair_attributes = self.physics_profile.pair_attributes(self.table_object_sliding_mu)
        finger_pair_attributes = self.physics_profile.pair_attributes(self.finger_object_sliding_mu)
        target_pair_attributes = self.physics_profile.pair_attributes(self.target_object_sliding_mu)
        partner_names = table_geom_names + target_collision_names + finger_pad_names
        for geom_name in partner_names:
            geom = self.model.worldbody.find(f".//geom[@name='{geom_name}']")
            if geom is None:
                raise ShakeBenchMetricsError(f"contact partner geom {geom_name!r} is missing")
            geom.set("conaffinity", str(CAN_CONTACT_BIT))
        for can_name in can_geom_names:
            for table_name in table_geom_names:
                ET.SubElement(
                    self.model.contact,
                    "pair",
                    {"geom1": can_name, "geom2": table_name, **table_pair_attributes},
                )
            for target_name in target_collision_names:
                ET.SubElement(
                    self.model.contact,
                    "pair",
                    {"geom1": can_name, "geom2": target_name, **target_pair_attributes},
                )
            for finger_name in finger_pad_names:
                ET.SubElement(
                    self.model.contact,
                    "pair",
                    {"geom1": can_name, "geom2": finger_name, **finger_pair_attributes},
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
        self._policy_task_context_cache = None
        super()._load_model()
        if len(self.robots) != 1 or self.robot_names != ["Panda"]:
            raise ValueError("VibrationPickPlace requires exactly one Panda")
        if self.geometry_profile is not None:
            initial_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"], dtype=float)
            if initial_qpos.shape != (7,) or not np.all(np.isfinite(initial_qpos)):
                raise ValueError("geometry profile initial_joint_qpos_rad must be a finite seven-vector")
            self.robots[0].init_qpos = initial_qpos.copy()

        self.arena = ShakeBenchArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            isolator_config=self.physics_profile.isolator_config(),
            include_target_container=False,
            visual=self.scene_visual,
            scene_config=self.scene_config,
        )
        self.arena.table_imu_site.set("pos", array_to_string(self.table_imu_position_m))
        self.arena.table_imu_site.set("quat", array_to_string(self.table_imu_quat_wxyz))
        support = (
            ET.parse(xml_path_completion(self.geometry_profile["robot_support_mjcf"]))
            .getroot()
            .find("./worldbody/body[@name='robot_support']")
        )
        if support is None:
            raise ValueError("robot support MJCF must contain a worldbody/robot_support body")
        if not self.scene_visual:
            for geom in support.iter("geom"):
                if geom.get("rgba") is not None:
                    rgba = np.fromstring(geom.get("rgba"), sep=" ")
                    rgba[3] = 0.0
                    geom.set("rgba", array_to_string(rgba))
        self.arena.worldbody.append(support)
        self.arena.object_support_geom = self.arena.table_collision
        if self.task_spec is not None and self.task_spec.surface_id == "mat":
            self.arena.add_table_mat()
        self.arena.add_target_container(
            friction=self.target_container_friction,
            visual_style="basket" if self.task_spec is not None else "tray",
        )
        base_position = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        if self.geometry_profile is not None:
            base_position = self.geometry_profile["robot_base_pos_m"]
        self.robots[0].robot_model.set_base_xpos(np.asarray(base_position, dtype=float))
        self.robot_base_body_name = self.robots[0].robot_model.root_body
        if self.robot_base_body_name != "robot0_base":
            raise ValueError("Panda base role changed; refusing implicit deck body lookup")
        self.robot_mount_type = type(self.robots[0].robot_model.base).__name__
        if type(self.robots[0].gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("VibrationPickPlace requires the standard PandaGripper")
        self.finger_pad_geom_names = tuple(
            self.robots[0].gripper["right"].important_geoms["left_fingerpad"]
            + self.robots[0].gripper["right"].important_geoms["right_fingerpad"]
        )
        self.gripper_body_name = self.robots[0].robot_model.eef_name["right"]

        from shakebench.utils.tasks import make_task_object

        # Stable internal handles preserve the evaluator and legacy replay wire format.
        self.can = make_task_object(self.task_spec)
        self.task_object = self.can
        self._configure_can()
        self._measure_can_collision_envelope()
        if self.placement_initializer is None:
            self.placement_initializer = UniformRandomSampler(
                name="OpenTableCanSampler",
                mujoco_objects=self.can,
                x_range=(self.object_start_xy[0], self.object_start_xy[0]),
                y_range=(self.object_start_xy[1], self.object_start_xy[1]),
                rotation=self.object_start_yaw_rad,
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
        configure_scene_rendering(self.model.root, self.scene_config)
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
        self._imu_mount_audit = self.table_imu_provider.audit_compiled_mount(self.sim)

    def _setup_observables(self):
        observables = super()._setup_observables()
        for name in self.table_imu_provider.policy_keys:

            @sensor(modality=f"shakebench_{name}")
            def table_imu_sensor(obs_cache, provider_key=name):
                return self.table_imu_provider.observation(self.sim)[provider_key]

            observables[name] = Observable(name=name, sensor=table_imu_sensor, sampling_rate=self.control_freq)
        if not self.use_object_obs:
            return observables
        modality = "object"

        @sensor(modality=modality)
        def object_pos_robot_base(obs_cache):
            return can_pose_twist_in_frame(
                self.sim,
                self.can.root_body,
                self.robot_base_body_name,
            ).position_m

        @sensor(modality=modality)
        def object_quat_robot_base(obs_cache):
            return T.convert_quat(
                can_pose_twist_in_frame(
                    self.sim,
                    self.can.root_body,
                    self.robot_base_body_name,
                ).quaternion_wxyz,
                to="xyzw",
            )

        @sensor(modality=modality)
        def object_to_target_pos(obs_cache):
            return can_pose_twist_in_frame(
                self.sim,
                self.can.root_body,
                self.arena.worktable_body_name,
                frame_local_origin_m=self.metrics.target_frame_local_origin_m,
            ).position_m

        for name, sensor_fn in (
            ("object_pos_robot_base", object_pos_robot_base),
            ("object_quat_robot_base", object_quat_robot_base),
            ("object_to_target_pos", object_to_target_pos),
        ):
            observables[name] = Observable(name=name, sensor=sensor_fn, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.reset_settle_duration_s = 0.0
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample(on_top=False)
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate((np.asarray(obj_pos), np.asarray(obj_quat))))
            self._settle_reset_support()
        self.sim.forward()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=0.0)
        if self.privileged_recorder is not None:
            self.privileged_recorder.reset()
        self._phase05_last_action = np.zeros(self.action_dim if hasattr(self, "action_dim") else 0, dtype=np.float32)
        if self.metrics is not None:
            self.metrics.reset()
        self.success_evaluator.reset()
        self._last_success_evaluation = None

    def _settle_reset_support(self):
        """Find loaded contact equilibrium before starting the episode clock.

        Native integration includes the compliant deck weld and object contacts.
        Robot joints stay at their reset pose; no policy, excitation, sensor or
        metric hooks run during initialization.
        """
        model, data = self.sim.model._model, self.sim.data._data
        self._seed_loaded_support_equilibrium(model, data)
        moving_bodies = {int(model.body(self.deck_config.deck_body_name).id), self.can_body_id}
        for body in range(1, model.nbody):
            if int(model.body_parentid[body]) in moving_bodies:
                moving_bodies.add(body)
        moving = np.isin(model.dof_bodyid, list(moving_bodies))
        support = moving & (model.dof_bodyid != self.can_body_id)
        object_dofs = model.dof_bodyid == self.can_body_id
        fixed_qpos = np.ones(model.nq, dtype=bool)
        for joint in range(model.njnt):
            if int(model.jnt_bodyid[joint]) in moving_bodies:
                start = int(model.jnt_qposadr[joint])
                end = int(model.jnt_qposadr[joint + 1]) if joint + 1 < model.njnt else model.nq
                fixed_qpos[start:end] = False
        robot_pose = data.qpos[fixed_qpos].copy()
        quiet_steps = 0
        required_quiet_steps = int(np.ceil(0.1 / model.opt.timestep))
        sample_stride = max(1, int(round(0.005 / model.opt.timestep)))
        # ponytail: bounded native settle; use a contact equilibrium solver if
        # initialization cost becomes significant at large collection scale.
        for step in range(int(np.ceil(5.0 / model.opt.timestep))):
            mujoco.mj_step(model, data)
            data.qpos[fixed_qpos] = robot_pose
            data.qvel[~moving] = 0.0
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("reset equilibrium produced non-finite state")
            if (step + 1) % sample_stride:
                continue
            object_velocity = data.qvel[object_dofs]
            quiet = (
                np.max(np.abs(data.qvel[support])) < 1e-5
                and np.max(np.abs(data.qacc[support])) < 0.05
                and np.max(np.abs(object_velocity[:3])) < 0.001
                and np.max(np.abs(object_velocity[3:])) < 0.02
            )
            quiet_steps = quiet_steps + sample_stride if quiet else 0
            if quiet_steps >= required_quiet_steps:
                break
        else:
            raise RuntimeError(
                "reset support did not reach equilibrium within 5 simulation seconds: "
                f"qvel={data.qvel[moving]}, qacc={data.qacc[moving]}"
            )
        self.reset_settle_duration_s = (step + 1) * float(model.opt.timestep)
        data.qvel[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        data.time = 0.0

    def _seed_loaded_support_equilibrium(self, model, data):
        """Start the settle from the analytic loaded equilibrium.

        The isolator ``springref`` compensates only the 32 kg table, so the
        placed object would otherwise make the 5 Hz support ring down from the
        unloaded pose.  Seeding the six isolator coordinates removes that
        transient; the settle below only has to absorb what the linear model
        cannot express, namely the compliant deck weld and object contact.
        """

        self.sim.forward()
        table_id = int(model.body(self.arena.worktable_body_name).id)
        table_position = np.array(data.xpos[table_id], dtype=float, copy=True)
        table_rotation = np.array(data.xmat[table_id], dtype=float).reshape(3, 3)
        object_position = np.array(data.xpos[self.can_body_id], dtype=float, copy=True)
        object_rotation = np.array(data.xmat[self.can_body_id], dtype=float).reshape(3, 3)
        offset = static_equilibrium_offset(
            self.arena.isolator_parameters,
            payload_mass_kg=self.object_mass_kg,
            payload_com_m=object_position - table_position,
        )
        for index, axis in enumerate(ISOLATOR_AXES):
            joint_id = int(model.joint("isolator_" + axis).id)
            data.qpos[int(model.jnt_qposadr[joint_id])] = float(offset[index])
            data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
        mujoco.mj_forward(model, data)
        # The object rests on the support, so carry it with the same rigid
        # transform instead of letting it drop the few tenths of a millimetre
        # onto the seeded table pose.
        moved_rotation = np.array(data.xmat[table_id], dtype=float).reshape(3, 3)
        moved_position = np.array(data.xpos[table_id], dtype=float)
        delta_rotation = moved_rotation.dot(table_rotation.T)
        delta_position = moved_position - delta_rotation.dot(table_position)
        quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, delta_rotation.dot(object_rotation).reshape(-1))
        self.sim.data.set_joint_qpos(
            self.can.joints[0], np.concatenate((delta_rotation.dot(object_position) + delta_position, quaternion))
        )

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

    def _update_phase05_provider(self, sample_time_s, policy_step=False):
        self.table_imu_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)

    def _record_phase05_privileged(self, sample_time_s, policy_step=False):
        if self.privileged_recorder is None:
            return
        if self._control_steps and (self._physics_step_index + 1) % self._control_steps != 0:
            return
        can_pose_world = np.concatenate(
            (
                np.asarray(self.sim.data.xpos[self.can_body_id], dtype=float),
                np.asarray(self.sim.data.xquat[self.can_body_id], dtype=float),
            )
        )
        target_world = self.target_frame_world_position()
        metrics_report = self.metrics.to_dict() if self.metrics is not None and self.metrics.latest is not None else {}
        provider_truth = self.table_imu_provider.privileged_snapshot(self.sim, time_s=float(sample_time_s))
        success_report = metrics_report.get("success", {})
        if not isinstance(success_report, dict):
            success_report = {}
        driver_response = metrics_report.get("driver_response", {})
        table_response = metrics_report.get("table_response", {})
        snapshot = {
            "privileged_time_s": float(sample_time_s),
            "privileged_actions": self._phase05_last_action.copy(),
            "privileged_can_pose_world": can_pose_world,
            "privileged_goal_center_world": np.asarray(target_world, dtype=float),
            "privileged_support": {
                "deck_body_name": self.deck_body_name,
                "worktable_body_name": self.worktable_body_name,
                "metrics_driver_response": metrics_report.get("driver_response", {}),
                "metrics_table_response": metrics_report.get("table_response", {}),
            },
            "privileged_commanded_support": {
                key: driver_response[key]
                for key in ("command_pose", "command_twist", "command_acceleration")
                if key in driver_response
            },
            "privileged_actual_support": {
                "deck": driver_response.get("actual_pose"),
                "deck_twist": driver_response.get("actual_twist"),
                "deck_acceleration": driver_response.get("deck_acceleration"),
                "table": table_response,
            },
            "privileged_contacts": metrics_report.get("contacts", {}),
            "privileged_success_subconditions": success_report.get(
                "subconditions", metrics_report.get("success_snapshot", {})
            ),
            "privileged_success": success_report,
            "privileged_parameters": self.policy_task_context,
            "privileged_provider": provider_truth,
        }
        # Keep the nested provider record convenient for consumers while also
        # giving every truth field a directly auditable privileged_ name.
        for key, value in provider_truth.items():
            snapshot[f"{PRIVILEGED_NAMESPACE}{key}"] = value
        for key in (
            "command_pose",
            "command_twist",
            "command_acceleration",
            "actual_pose",
            "actual_twist",
            "deck_acceleration",
        ):
            if key in driver_response:
                snapshot[f"{PRIVILEGED_NAMESPACE}deck_{key}"] = driver_response[key]
        for key, value in table_response.items():
            snapshot[f"{PRIVILEGED_NAMESPACE}table_{key}"] = value
        if "support_state" in provider_truth:
            snapshot[f"{PRIVILEGED_NAMESPACE}support_state"] = provider_truth["support_state"]
        self.privileged_recorder.record(snapshot)

    def _audit_compiled_contract(self, sim):
        # This callback executes after the Phase 02 driver has validated and
        # bound the compiled deck.  It is intentionally an assertion seam, not
        # a recovery path.
        self._scene_audit = audit_compiled_scene(sim, self.scene_config)
        self._scene_clearance = scene_clearance_report(sim, self.scene_config)
        if not self._scene_clearance.passed:
            raise ShakeBenchMetricsError("world-fixed scene failed support/clearance audit")
        self._compiled_contract = self.audit_compiled_model(sim)
        self._compiled_contract["scene"] = self._scene_audit.to_dict()
        self._compiled_contract["scene_clearance"] = self._scene_clearance.to_dict()
        self._imu_mount_audit = self.table_imu_provider.audit_compiled_mount(sim)
        self._compiled_contract["imu_mount"] = self._imu_mount_audit

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        if self.metrics is None:
            return
        # Success is a physics-step contract.  Keep the full diagnostic report
        # at 20 Hz, but update the small evaluator input on every substep.
        self._last_success_evaluation = self.success_evaluator.evaluate(
            self.metrics.success_snapshot(self.sim), time_s=sample_time_s
        )
        if not self._control_steps or (self._physics_step_index + 1) % self._control_steps == 0:
            self.metrics.update(self.sim, time_s=sample_time_s)
            self.metrics.attach_success(self._last_success_evaluation)

    def _sample_metrics(self):
        if self.metrics is None:
            raise ShakeBenchMetricsError("metrics are unavailable before model construction")
        snapshot = self.metrics.update(self.sim, time_s=float(self.sim.data.time))
        evaluation = self.success_evaluator.evaluate(snapshot.success_snapshot, time_s=float(self.sim.data.time))
        self._last_success_evaluation = evaluation
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

    @property
    def policy_observation_keys(self):
        """Return the explicit State-track keys, excluding aggregate modality keys."""

        return tuple(self._observables.keys())

    @property
    def state_observation_keys(self):
        """Alias for the State-track policy key contract."""

        return self.policy_observation_keys

    @property
    def vibration_observation_keys(self):
        """Return only the dedicated vibration keys visible at this tier."""

        return tuple(self.table_imu_provider.policy_keys)

    @property
    def policy_task_context(self):
        """Return public static task/physics metadata, never current runtime truth."""

        cached = getattr(self, "_policy_task_context_cache", None)
        if cached is not None:
            return copy.deepcopy(cached)
        if not hasattr(self, "arena"):
            return {"policy_rate_hz": float(self.control_freq)}
        target_spec = self.arena.target_container_spec
        robot_base_position = np.asarray(self.geometry_profile["robot_base_pos_m"], dtype=float) - np.asarray(
            self.robots[0].robot_model.bottom_offset, dtype=float
        )
        task_context = dict(
            worktable_size_xy_m=tuple(np.asarray(self.table_full_size, dtype=float)[:2]),
            target_frame_origin_in_worktable_m=tuple(self.metrics.target_frame_local_origin_m),
            table_surface_z_in_worktable_m=float(self.arena.table_half_size[2]),
            object_collision_radius_m=float(self.can_collision_envelope.support_radius_m),
            object_collision_lower_support_m=float(self.can_collision_envelope.lower_support_z_m),
            object_collision_upper_support_m=float(self.can_collision_envelope.upper_support_z_m),
            finger_pad_tool_support_offsets_m=(0.0, 0.0, 0.0934),
            support_topology_id="world_fixed_arm_v1",
            world_to_robot_base_position_m=tuple(robot_base_position),
            world_to_robot_base_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        context = {
            "policy_rate_hz": float(self.control_freq),
            **({"geometry_profile": self.geometry_profile} if self.geometry_profile else {}),
            "scene_visual": {
                "scene_id": self.scene_config.scene_id,
                "geometry_variant": self.scene_config.geometry_variant,
                "physics_effect": self.scene_config.physics_effect,
                "enabled": self.scene_visual,
            },
            "robot_mount": {
                "requested_base_type": self.base_types,
                "compiled_mount_type": getattr(self, "robot_mount_type", None),
                "robot_base_body_name": self.robot_base_body_name,
                "initial_joint_qpos_rad": (
                    np.asarray(self.robots[0].init_qpos, dtype=float).tolist()
                    if self.geometry_profile is not None
                    else None
                ),
            },
            "worktable": {
                "dimensions_m": list(self.table_full_size),
                "mass_kg": float(self.arena.isolator_parameters.mass_kg),
                "inertia_kg_m2": list(self.arena.isolator_parameters.inertia_kg_m2),
            },
            "table_imu": {
                "sensor_parent": "worktable",
                "sensor_site_name": "table_imu_site",
                "position_m_in_worktable": list(self.table_imu_position_m),
                "quaternion_wxyz_in_worktable": list(self.table_imu_quat_wxyz),
                "profile_id": CANONICAL_IMU_PROFILE.profile_id,
                "profile_sha256": CANONICAL_IMU_PROFILE_HASH,
                "sensor_config_sha256": self._imu_mount_audit["sensor_config_sha256"],
            },
            "object": {
                "mass_kg": float(self.object_mass_kg),
                "inertia_kg_m2": list(self.can_inertia) if self.can_inertia is not None else None,
                "collision_envelope": (
                    self.can_collision_envelope.to_dict() if hasattr(self, "can_collision_envelope") else None
                ),
            },
            "isolator": self.arena.isolator_parameters.to_dict(),
            "friction": {
                "table_object_sliding_mu": float(self.table_object_sliding_mu),
                "finger_object_sliding_mu": float(self.finger_object_sliding_mu),
                "target_container": list(self.target_container_friction),
            },
            "physics_profile": {
                "profile_id": self.physics_profile.profile_id,
                "scoreable": self.physics_profile.scoreable
                and self._geometry_is_scoreable()
                and self.task_spec is None,
            },
            "target_container": target_spec,
            "task_context": task_context,
            "task_context_sha256": payload_hash(task_context),
            "support_motion": {
                "semantic_quantity": "worktable_relative_to_robot_base_motion",
                "frame": "robot_base",
                "reference_point": "target_origin",
                "isolator_fn_hz": self.arena.isolator_parameters.fn_hz,
                "isolator_zeta": self.arena.isolator_parameters.zeta,
                "isolator_k": self.arena.isolator_parameters.stiffness,
                "isolator_c": self.arena.isolator_parameters.damping,
            },
            "support_topology_id": "world_fixed_arm_v1",
            "success_semantics": "phase04_vibration_success_evaluator",
            **({"task": self.task_spec.contract()} if self.task_spec is not None else {}),
        }
        if hasattr(self, "sim"):
            self._policy_task_context_cache = copy.deepcopy(context)
        return context

    def get_policy_task_context(self):
        """Return the public static task context as a fresh mapping."""

        return self.policy_task_context

    def get_task_context(self):
        """Return the public static task context for tools and tests."""

        return self.get_policy_task_context()

    def _geometry_is_scoreable(self):
        """New topology requires fresh experimental certification."""
        return False

    def observation_contract(self):
        """Return the declared public State shape/unit/frame contract."""

        return {key: dict(POLICY_FIELD_CONTRACT[key]) for key in self.table_imu_provider.policy_keys}

    def _get_observations(self, force_update=False):
        observations = super()._get_observations(force_update=force_update)
        observations.update(self.table_imu_provider.observation(self.sim))
        assert_policy_observation_is_clean(observations)
        return observations

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
            expected_mass_kg=self.object_mass_kg,
            expected_com_m=self.can_com,
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
            target_sliding_mu=self.target_object_sliding_mu,
            finger_sliding_mu=self.finger_object_sliding_mu,
            contact_profile=self.physics_profile.contact,
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
        if parent_name(self.robot_base_body_name) is not None:
            raise ShakeBenchMetricsError("Panda base must be a direct world child")
        for name in (self.robot_base_body_name, "robot_support"):
            body_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, name))
            if (
                body_id < 0
                or raw_model.body_parentid[body_id] != 0
                or raw_model.body_jntnum[body_id]
                or raw_model.body_mocapid[body_id] >= 0
            ):
                raise ShakeBenchMetricsError(f"{name} must be rigidly fixed to world")
        if parent_name(self.arena.worktable_body_name) != self.deck_config.deck_body_name:
            raise ShakeBenchMetricsError("worktable is not a rigid child of the dynamic deck")
        deck_audit = audit_compiled_deck_model(
            sim_or_model,
            self.deck_config,
            self.deck_driver.role_handles,
        )
        worktable_audit = self.arena.audit_compiled_model(sim_or_model)
        robot_base_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, self.robot_base_body_name))

        def body_ancestors(body_id):
            ancestors = set()
            current = int(body_id)
            while current != 0:
                ancestors.add(current)
                current = int(raw_model.body_parentid[current])
            return ancestors

        mount_body_ids = [
            body_id
            for body_id in range(int(raw_model.nbody))
            if body_id != robot_base_id and robot_base_id in body_ancestors(body_id)
        ]
        mount_bodies = {}
        for body_id in mount_body_ids:
            body_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None or not str(body_name).startswith("fixed_mount"):
                continue
            mount_bodies[str(body_name)] = {
                "body_id": body_id,
                "parent": parent_name(str(body_name)),
                "mass_kg": float(raw_model.body_mass[body_id]),
                "inertia_kg_m2": np.asarray(raw_model.body_inertia[body_id], dtype=float).tolist(),
            }
        mount_geoms = {}
        for geom_id in range(int(raw_model.ngeom)):
            body_id = int(raw_model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            geom_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if body_name is None or geom_name is None or not str(body_name).startswith("fixed_mount"):
                continue
            mount_geoms[str(geom_name)] = {
                "body_name": str(body_name),
                "geom_id": geom_id,
                "type": int(raw_model.geom_type[geom_id]),
                "size_m": np.asarray(raw_model.geom_size[geom_id], dtype=float).tolist(),
                "contype": int(raw_model.geom_contype[geom_id]),
                "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
                "world_position_m": np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float).tolist(),
            }
        support_geometry = {}
        for part in ("foundation", "mount_plate"):
            geom_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, "robot_support_" + part))
            if geom_id < 0:
                raise ShakeBenchMetricsError(f"robot support MJCF is missing {part}")
            support_geometry[part + "_pos_m"] = raw_model.geom_pos[geom_id].tolist()
            support_geometry[part + "_half_size_m"] = raw_model.geom_size[geom_id].tolist()
        robot_mount_audit = {
            "requested_base_type": self.base_types,
            "compiled_mount_type": self.robot_mount_type,
            "robot_base_body_name": self.robot_base_body_name,
            "support_body_name": "robot_support",
            "support_parent": parent_name("robot_support"),
            "support_geometry": support_geometry,
            "robot_base_pose_in_world": {
                "pos_m": np.asarray(raw_model.body_pos[robot_base_id], dtype=float).tolist(),
                "quat_wxyz": np.asarray(raw_model.body_quat[robot_base_id], dtype=float).tolist(),
            },
            "bodies": mount_bodies,
            "collision_geoms": mount_geoms,
        }
        result = {
            "topology": {
                "deck_body": self.deck_config.deck_body_name,
                "can_parent": parent_name(self.can.root_body),
                "robot_base_parent": parent_name(self.robot_base_body_name),
                "robot_support_parent": parent_name("robot_support"),
                "worktable_parent": parent_name(self.arena.worktable_body_name),
                "worktable_body": self.arena.worktable_body_name,
                "target_body": next(iter(target_body_names)),
                "can_freejoint": can_audit["freejoint_names"],
                "target_collision_geom_names": target_collision_names,
            },
            "deck": deck_audit,
            "worktable": worktable_audit,
            "object": can_audit,
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
            **({"task": self.task_spec.contract()} if self.task_spec is not None else {}),
            "physics_profile": self.physics_profile.audit(),
            "robot_mount": robot_mount_audit,
            **({"geometry_profile": self.geometry_profile} if self.geometry_profile else {}),
            "scene_visual": {
                "scene_id": self.scene_config.scene_id,
                "geometry_variant": self.scene_config.geometry_variant,
                "physics_effect": self.scene_config.physics_effect,
                "enabled": self.scene_visual,
            },
        }
        result["imu_mount"] = self.table_imu_provider.audit_compiled_mount(sim_or_model)
        return result

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
            raise ValueError("VibrationPickPlace currently supports exactly one Panda robot")


__all__ = ["VibrationPickPlace"]
