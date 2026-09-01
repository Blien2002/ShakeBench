"""ShakeBench Phase 04 open-table Can pick-and-place environment."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import OrderedDict
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
from robosuite.utils.shakebench_physics import PhysicsProfileError, resolve_physics_profile
from robosuite.utils.shakebench_privilege import (
    PRIVILEGED_NAMESPACE,
    ShakeBenchPrivilegeError,
    assert_policy_observation_is_clean,
    make_privileged_recorder,
)
from robosuite.utils.shakebench_providers import (
    COMMON_STATE_KEYS,
    TIER_POLICY_KEYS,
    ShakeBenchProviderError,
    make_vibration_provider,
    normalize_observation_tier,
    observation_contract_for_tier,
)
from robosuite.utils.shakebench_sensors import (
    CANONICAL_IMU_PROFILE,
    CanonicalIMU,
    IMU_DT_S,
    ShakeBenchSensorError,
)
from robosuite.utils.shakebench_excitation import ExcitationProgram

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
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        observation_tier=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        excitation_program=None,
        privileged_recorder=None,
    ):
        requested_robots = list(robots) if isinstance(robots, (list, tuple)) else [robots]
        if requested_robots != ["Panda"]:
            raise ValueError("VibrationPickPlaceCan currently supports exactly one Panda robot")
        if env_configuration != "default":
            raise ValueError("VibrationPickPlaceCan only supports env_configuration='default'")
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
        if observation_tier is not None:
            try:
                observation_tier = normalize_observation_tier(observation_tier)
            except ShakeBenchProviderError as exc:
                raise ValueError(str(exc)) from exc
            if use_camera_obs:
                raise ValueError("State observation tiers require use_camera_obs=False")
            if imu_mode not in CanonicalIMU.VALID_MODES:
                raise ValueError("imu_mode must be ideal_smoke or canonical_noisy_v1")
            if float(control_freq) != 20.0:
                raise ValueError("State observation tiers require control_freq=20 Hz")
            if model_timestep is not None:
                try:
                    phase05_timestep = float(model_timestep)
                except (TypeError, ValueError) as exc:
                    raise ValueError("model_timestep must be compatible with the 200 Hz IMU") from exc
                if (
                    not np.isfinite(phase05_timestep)
                    or phase05_timestep <= 0.0
                    or not np.isclose(
                        phase05_timestep * round(IMU_DT_S / phase05_timestep), IMU_DT_S, rtol=0.0, atol=1e-12
                    )
                ):
                    raise ValueError("State observation tiers require model_timestep to divide the 5 ms IMU interval")
        _one_value("base_types", base_types, allowed=("default",))
        _one_value("gripper_types", gripper_types, allowed=("default", "PandaGripper"))
        self.table_full_size = _finite_vector("table_full_size", table_full_size, 3)
        self.table_friction = _finite_vector("table_friction", table_friction, 3)
        self.table_offset = _finite_vector("table_offset", table_offset, 3)
        self.target_container_friction = _finite_vector("target_container_friction", target_container_friction, 3)
        self.can_start_xy = _finite_vector("can_start_xy", can_start_xy, 2)
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
        self.use_object_obs = use_object_obs
        self.use_camera_obs = use_camera_obs
        self.observation_tier = observation_tier
        self.imu_mode = imu_mode
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
        self._phase05_policy_observation_keys = ()
        self._imu_mount_audit = None
        self.policy_task_state_frame = "robot_base"
        self.placement_initializer = placement_initializer
        self._requested_load_model_on_init = bool(load_model_on_init)
        self._compiled_contract = None
        self.metrics = None
        self.success_evaluator = VibrationSuccessEvaluator(DEFAULT_SUCCESS_THRESHOLDS)

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
        if observation_tier == "V3":
            if excitation_program is None and isinstance(deck_trajectory, ExcitationProgram):
                excitation_program = deck_trajectory
            elif excitation_program is None and deck_trajectory is not None:
                raise ValueError(
                    "V3 requires deck_trajectory to be an ExcitationProgram or an explicit excitation_program"
                )
            elif (
                excitation_program is not None
                and isinstance(deck_trajectory, ExcitationProgram)
                and excitation_program is not deck_trajectory
            ):
                raise ValueError("deck_trajectory and excitation_program specify different V3 programs")
        runtime_trajectory = deck_trajectory
        if excitation_program is not None:
            if runtime_trajectory is not None and runtime_trajectory is not excitation_program:
                raise ValueError("deck_trajectory and excitation_program specify different trajectories")
            runtime_trajectory = excitation_program
        self.deck_driver = DeckDriver(
            trajectory=runtime_trajectory,
            config=self.deck_config,
            body_handles={"isolated_worktable": "worktable", "robot_base": "robot0_base"},
            required_roles=("isolated_worktable", "robot_base"),
        )
        if observation_tier is None:
            self.vibration_provider = None
        else:
            try:
                self.vibration_provider = make_vibration_provider(
                    observation_tier,
                    seed=self.imu_seed,
                    imu_mode=imu_mode,
                    program=excitation_program,
                    deck_body_name=self.deck_config.deck_body_name,
                    imu_body_name="robot0_base",
                    table_body_name="worktable",
                    nominal_frame_position_m=self.deck_config.deck_pos_m,
                    nominal_frame_quat_wxyz=self.deck_config.deck_quat_wxyz,
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
        # Set environment-owned solver/timestep options before the deck XML
        # processor generates the dynamic deck.  This profile is the only
        # source of score-affecting physics for the official environment.
        self.set_xml_processor(self.physics_profile.process_xml)
        self.deck_driver.install(self)
        self.add_sim_initialization_hook(self._audit_compiled_contract)
        self.add_post_physics_step_hook(self._record_post_physics_metrics)
        if self.observation_tier is not None:
            # The provider runs after the DeckDriver has recorded the current
            # realized body state and before the policy observation is read.
            self.add_post_physics_step_hook(self._update_phase05_provider)
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
        table_pair_attributes = self.physics_profile.pair_attributes(self.table_object_sliding_mu)
        finger_pair_attributes = self.physics_profile.pair_attributes(self.finger_object_sliding_mu)
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
                    {"geom1": can_name, "geom2": target_name, **table_pair_attributes},
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
        super()._load_model()
        if len(self.robots) != 1 or self.robot_names != ["Panda"]:
            raise ValueError("VibrationPickPlaceCan requires exactly one Panda")

        self.arena = ShakeBenchArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            isolator_config=self.physics_profile.isolator_config(),
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
        if self.vibration_provider is not None:
            self._imu_mount_audit = self.vibration_provider.audit_compiled_mount(self.sim)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.observation_tier is None:
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

        # The stock robot observables report EEF pose in the world frame.  A
        # world-frame EEF value would make the State track leak deck motion, so
        # those three legacy observables are inactive only for an explicit
        # State tier.  Their names remain available in the underlying object
        # for compatibility with robosuite's internal robot bookkeeping.
        robot_prefix = self.robots[0].robot_model.naming_prefix
        for name in (
            f"{robot_prefix}eef_pos",
            f"{robot_prefix}eef_quat",
            f"{robot_prefix}eef_quat_site",
        ):
            if name in observables:
                observables[name].set_active(False)

        task_modality = "shakebench_task"
        vibration_modality = "shakebench_vibration"
        robot = self.robots[0]

        def _world_to_robot_base(point_world):
            base_position = np.asarray(self.sim.data.get_body_xpos(self.robot_base_body_name), dtype=float)
            base_rotation = np.asarray(self.sim.data.get_body_xmat(self.robot_base_body_name), dtype=float).reshape(
                3, 3
            )
            return base_rotation.T.dot(np.asarray(point_world, dtype=float) - base_position)

        @sensor(modality=task_modality)
        def eef_pos_robot_base(obs_cache):
            pose = robot.pose_in_base_from_name(self.gripper_body_name)
            return np.asarray(pose[:3, 3], dtype=np.float32)

        @sensor(modality=task_modality)
        def eef_quat_robot_base(obs_cache):
            pose = robot.pose_in_base_from_name(self.gripper_body_name)
            return np.asarray(T.mat2quat(pose[:3, :3]), dtype=np.float32)

        @sensor(modality=task_modality)
        def gripper_state(obs_cache):
            qpos = np.asarray(
                [self.sim.data.qpos[index] for index in robot._ref_gripper_joint_pos_indexes["right"]],
                dtype=np.float32,
            )
            qvel = np.asarray(
                [self.sim.data.qvel[index] for index in robot._ref_gripper_joint_vel_indexes["right"]],
                dtype=np.float32,
            )
            return np.concatenate((qpos, qvel))

        @sensor(modality=task_modality)
        def wrist_force(obs_cache):
            return np.asarray(robot.ee_force["right"], dtype=np.float32)

        @sensor(modality=task_modality)
        def wrist_torque(obs_cache):
            return np.asarray(robot.ee_torque["right"], dtype=np.float32)

        @sensor(modality=task_modality)
        def fingertip_pos_robot_base(obs_cache):
            positions = [
                _world_to_robot_base(self.sim.data.geom_xpos[self.sim.model.geom_name2id(name)])
                for name in self.finger_pad_geom_names
            ]
            return np.asarray(np.concatenate(positions), dtype=np.float32)

        @sensor(modality=task_modality)
        def can_pos_robot_base(obs_cache):
            return np.asarray(
                can_pose_twist_in_frame(self.sim, self.can.root_body, self.robot_base_body_name).position_m,
                dtype=np.float32,
            )

        @sensor(modality=task_modality)
        def can_quat_robot_base(obs_cache):
            return np.asarray(
                T.convert_quat(
                    can_pose_twist_in_frame(self.sim, self.can.root_body, self.robot_base_body_name).quaternion_wxyz,
                    to="xyzw",
                ),
                dtype=np.float32,
            )

        @sensor(modality=task_modality)
        def goal_center_robot_base(obs_cache):
            target_world = frame_world_position(
                self.sim,
                self.arena.worktable_body_name,
                frame_local_origin_m=self.metrics.target_frame_local_origin_m,
            )
            return np.asarray(_world_to_robot_base(target_world), dtype=np.float32)

        @sensor(modality=task_modality)
        def goal_half_extents_robot_base(obs_cache):
            return np.asarray(
                (self.target_inner_xy_m[0] / 2.0, self.target_inner_xy_m[1] / 2.0),
                dtype=np.float32,
            )

        @sensor(modality=task_modality)
        def goal_z_bounds_robot_base(obs_cache):
            target_world = frame_world_position(
                self.sim,
                self.arena.worktable_body_name,
                frame_local_origin_m=self.metrics.target_frame_local_origin_m,
            )
            worktable_id = self.sim.model.body_name2id(self.arena.worktable_body_name)
            worktable_rotation = np.asarray(self.sim.data.xmat[worktable_id], dtype=float).reshape(3, 3)
            wall_height = float(self.arena.target_container_spec["wall_height_m"])
            top_world = target_world + worktable_rotation.dot(np.asarray((0.0, 0.0, wall_height)))
            return np.asarray(
                (_world_to_robot_base(target_world)[2], _world_to_robot_base(top_world)[2]), dtype=np.float32
            )

        @sensor(modality=task_modality)
        def goal_orientation_mask(obs_cache):
            # v0 has no final-yaw requirement.  A false mask is explicit and
            # avoids pretending the shallow box has a unique target pose.
            return np.zeros(3, dtype=np.bool_)

        state_sensors = (
            ("robot0_eef_pos_robot_base", eef_pos_robot_base),
            ("robot0_eef_quat_robot_base", eef_quat_robot_base),
            ("robot0_gripper_state", gripper_state),
            ("robot0_wrist_force", wrist_force),
            ("robot0_wrist_torque", wrist_torque),
            ("robot0_fingertip_pos_robot_base", fingertip_pos_robot_base),
            ("can_pos_robot_base", can_pos_robot_base),
            ("can_quat_robot_base", can_quat_robot_base),
            ("goal_center_robot_base", goal_center_robot_base),
            ("goal_half_extents_robot_base", goal_half_extents_robot_base),
            ("goal_z_bounds_robot_base", goal_z_bounds_robot_base),
            ("goal_orientation_mask", goal_orientation_mask),
        )
        for name, sensor_fn in state_sensors:
            # State fields intentionally have heterogeneous shapes.  Giving
            # each field its own modality prevents robosuite's legacy
            # same-modality concatenation from trying to join e.g. [10, 6]
            # with a scalar while preserving the individual observation keys.
            sensor_fn.__modality__ = f"{task_modality}_{name}"
            observables[name] = Observable(name=name, sensor=sensor_fn, sampling_rate=self.control_freq)

        for name in self.vibration_provider.policy_keys:

            @sensor(modality=vibration_modality)
            def provider_sensor(obs_cache, provider_key=name):
                return self.vibration_provider.observation(self.sim)[provider_key]

            provider_sensor.__modality__ = f"{vibration_modality}_{name}"
            observables[name] = Observable(name=name, sensor=provider_sensor, sampling_rate=self.control_freq)

        self._phase05_policy_observation_keys = tuple(COMMON_STATE_KEYS) + tuple(
            TIER_POLICY_KEYS[self.observation_tier]
        )
        for key in observables:
            if key.startswith(PRIVILEGED_NAMESPACE):
                raise ShakeBenchPrivilegeError("privileged namespace cannot be installed as an Observable")
        for key in self._phase05_policy_observation_keys:
            if key not in observables:
                raise ShakeBenchPrivilegeError(f"State tier observable {key!r} was not constructed")
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample(on_top=False)
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate((np.asarray(obj_pos), np.asarray(obj_quat))))
        # Placement is applied after the parent reset housekeeping.  Refresh
        # derived MuJoCo state before using it to seed the IMU's static history
        # so the first delivered window contains a real physical reading.
        if self.observation_tier is not None:
            self.sim.forward()
            self.vibration_provider.reset(self.sim, timestamp_s=0.0)
            if self.privileged_recorder is not None:
                self.privileged_recorder.reset()
        self._phase05_last_action = np.zeros(self.action_dim if hasattr(self, "action_dim") else 0, dtype=np.float32)
        if self.metrics is not None:
            self.metrics.reset()
        self.success_evaluator.reset()

    def _pre_action(self, action, policy_step=False):
        if self.observation_tier is not None and policy_step:
            self._phase05_last_action = np.asarray(action, dtype=np.float32).copy()
        super()._pre_action(action, policy_step=policy_step)

    def _update_phase05_provider(self, sample_time_s, policy_step=False):
        if self.vibration_provider is not None:
            self.vibration_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)

    def _record_phase05_privileged(self, sample_time_s, policy_step=False):
        if self.privileged_recorder is None or self.observation_tier is None:
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
        provider_truth = self.vibration_provider.privileged_snapshot(self.sim, time_s=float(sample_time_s))
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
        self._compiled_contract = self.audit_compiled_model(sim)
        if self.vibration_provider is not None:
            self._imu_mount_audit = self.vibration_provider.audit_compiled_mount(sim)
            self._compiled_contract["imu_mount"] = self._imu_mount_audit

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

    @property
    def policy_observation_keys(self):
        """Return the explicit State-track keys, excluding aggregate modality keys."""

        if self.observation_tier is None:
            return tuple(self._observables.keys())
        return tuple(self._phase05_policy_observation_keys)

    @property
    def state_observation_keys(self):
        """Alias for the State-track policy key contract."""

        return self.policy_observation_keys

    @property
    def vibration_observation_keys(self):
        """Return only the dedicated vibration keys visible at this tier."""

        if self.observation_tier is None:
            return ()
        return tuple(TIER_POLICY_KEYS[self.observation_tier])

    @property
    def policy_task_context(self):
        """Return public static task/physics metadata, never current runtime truth."""

        if not hasattr(self, "arena"):
            return {
                "observation_tier": self.observation_tier,
                "policy_rate_hz": float(self.control_freq),
            }
        target_spec = self.arena.target_container_spec
        return {
            "observation_tier": self.observation_tier,
            "policy_rate_hz": float(self.control_freq),
            "worktable": {
                "dimensions_m": list(self.table_full_size),
                "mass_kg": float(self.arena.isolator_parameters.mass_kg),
                "inertia_kg_m2": list(self.arena.isolator_parameters.inertia_kg_m2),
            },
            "can": {
                "mass_kg": float(CANONICAL_CAN_MASS_KG),
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
                "profile_sha256": self.physics_profile.profile_sha256,
                "scoreable": self.physics_profile.scoreable,
            },
            "imu": {
                "profile_id": CANONICAL_IMU_PROFILE.profile_id,
                "sample_rate_hz": CANONICAL_IMU_PROFILE.sample_rate_hz,
                "policy_window_shape": list(CANONICAL_IMU_PROFILE.window_shape),
                "position_m_in_robot_base": [0.0, 0.0, 0.0],
                "quaternion_wxyz_in_robot_base": [1.0, 0.0, 0.0, 0.0],
                "compiled_robot_base_pose_in_deck": (
                    self._imu_mount_audit.get("robot_base_pose_in_deck").tolist()
                    if self._imu_mount_audit is not None
                    and isinstance(self._imu_mount_audit.get("robot_base_pose_in_deck"), np.ndarray)
                    else None
                ),
            },
            "target_container": target_spec,
            "support_topology_id": "deck_robot_base_plus_isolated_worktable",
            "success_semantics": "phase04_vibration_success_evaluator",
        }

    def get_policy_task_context(self):
        """Return the public static task context as a fresh mapping."""

        return self.policy_task_context

    def observation_contract(self):
        """Return the declared public State shape/unit/frame contract."""

        if self.observation_tier is None:
            return {}
        return observation_contract_for_tier(self.observation_tier)

    def _get_observations(self, force_update=False):
        observations = super()._get_observations(force_update=force_update)
        if self.observation_tier is None:
            assert_policy_observation_is_clean(observations)
            return observations

        # Base robosuite updates Observable objects before its post-physics
        # hooks.  The provider hook therefore runs just after the normal
        # update on the last internal step.  Refresh provider-owned values at
        # the policy boundary so V2 is genuinely current-only and the V1
        # window includes the newest delivered sample.
        observations.update(self.vibration_provider.observation(self.sim))
        expected_keys = tuple(self._phase05_policy_observation_keys)
        missing = set(expected_keys) - set(observations)
        if missing:
            raise ShakeBenchPrivilegeError("State policy observation is missing key(s): " + ", ".join(sorted(missing)))
        policy_observation = OrderedDict((key, np.array(observations[key], copy=True)) for key in expected_keys)
        if set(policy_observation) != set(expected_keys) or len(policy_observation) != len(expected_keys):
            raise ShakeBenchPrivilegeError("State policy observation key-set does not match its declared contract")
        if any(key.startswith(PRIVILEGED_NAMESPACE) for key in policy_observation):
            raise ShakeBenchPrivilegeError("privileged namespace cannot cross the State policy boundary")
        assert_policy_observation_is_clean(policy_observation)
        return policy_observation

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
        result = {
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
            "physics_profile": self.physics_profile.audit(),
        }
        if self.vibration_provider is not None:
            result["imu_mount"] = self.vibration_provider.audit_compiled_mount(sim_or_model)
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
            raise ValueError("VibrationPickPlaceCan currently supports exactly one Panda robot")


__all__ = ["VibrationPickPlaceCan"]
