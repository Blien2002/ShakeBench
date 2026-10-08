"""Control-panel manipulation task and optional interactive scene.

The knob switches between 0 and 120 degrees, the lever has +/-30 degree stops, and the button has
4 mm of travel and toggles between raised OFF and recessed ON on successive presses. Joint resistance
and contact proxies are in panel.xml; observations use radians for hinges and
metres for the slide.
Green, amber and red indicators show knob ON state, positive lever tilt
and button ON state respectively.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from shakebench.environments.base import ShakeBenchTask
from shakebench.environments.panel_state import (
    INSTRUCTIONS,
    PANEL_HALF_FOOTPRINT_M,
    STATE_SCHEMA,
    STEP_CONTROL,
    SUCCESS_HOLD_S,
    TASK_TYPE,
    TASK_VERSION,
    default_state,
    load_states,
    validate_state,
)
from shakebench.models import xml_path_completion
from shakebench.models.arenas import ShakeBenchArena
from shakebench.physics.calibration import build_vibration_program
from shakebench.physics.deck import DeckDriver
from shakebench.physics.profile import resolve_physics_profile
from shakebench.scene.config import DECK_VISUAL_BODY_NAME, configure_scene_rendering
from shakebench.scene.geometry import (
    DEFAULT_GEOMETRY_PROFILE,
    geometry_scene_path,
    load_geometry_profile,
    worktable_mount,
)
from shakebench.sensors.providers import POLICY_FIELD_CONTRACT, TableIMUProvider
from shakebench.tasks.privilege import assert_policy_observation_is_clean
from shakebench.tasks.registry import TaskDefinition, register_state_loader, register_task

# Every task module exposes default_state(); the panel defines its own in panel_state.
__all__ = ["PanelOperation", "default_state"]


class PanelOperation(ShakeBenchTask):
    """Panda control-panel scene with optional ordered actions."""

    panel_joint_names = ("panel_knob_joint", "panel_lever_joint", "panel_button_joint")
    KNOB_RANGE_RAD = np.deg2rad(120)
    KNOB_ON_THRESHOLD_RAD = np.deg2rad(115)
    KNOB_OFF_THRESHOLD_RAD = np.deg2rad(5)
    BUTTON_ON_DEPTH_M = -0.002
    BUTTON_OFF_SPRING_M = 0.0005
    BUTTON_PRESS_THRESHOLD_M = -0.0033
    BUTTON_RELEASE_THRESHOLD_M = -0.0025

    def __init__(
        self,
        robots="Panda",
        *,
        panel_xy_m=(0.0, 0.0),
        panel_yaw_rad=0.0,
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
        self.task_state = None if task_state is None else validate_state(task_state)
        if self.task_state is not None:
            if vibration is not None:
                vibration = {**vibration, "seed": self.task_state["excitation_seed"], "t0_s": self.task_state["t0_s"]}
            panel_xy_m = self.task_state["panel_xy_m"]
            panel_yaw_rad = self.task_state["panel_yaw_rad"]
            imu_seed = self.task_state["imu_seed"]
            kwargs["seed"] = self.task_state["excitation_seed"]
        self.panel_xy_m = np.asarray(panel_xy_m, dtype=float)
        self.panel_yaw_rad = float(panel_yaw_rad)
        if self.panel_xy_m.shape != (2,) or not np.isfinite(self.panel_xy_m).all():
            raise ValueError("panel_xy_m must contain two finite tabletop coordinates")
        if not np.isfinite(self.panel_yaw_rad):
            raise ValueError("panel_yaw_rad must be finite")
        self.physics_profile = resolve_physics_profile(physics_profile)
        self.geometry_profile = load_geometry_profile(geometry_profile)
        self.worktable_mount = worktable_mount(geometry_profile)
        self.scene_path = geometry_scene_path(geometry_profile)
        seed = kwargs.setdefault("seed", 17)
        self.table_imu_provider = TableIMUProvider(seed=seed if imu_seed is None else imu_seed, imu_mode=imu_mode)
        self.deck_driver = DeckDriver(
            trajectory=build_vibration_program(
                vibration
                if vibration is not None
                else {
                    "mode": "multisine_v1",
                    "gamma": 0.0,
                    "seed": seed if seed is not None else 17,
                    "t0_s": 0.0 if self.task_state is None else self.task_state["t0_s"],
                }
            ),
            config=self.physics_profile.deck_driver_config(),
            body_handles={"isolated_worktable": "worktable", "deck_visual": DECK_VISUAL_BODY_NAME},
            required_roles=("isolated_worktable", "deck_visual"),
        )
        self.use_object_obs = bool(use_object_obs)
        eager = kwargs.pop("load_model_on_init", True)
        if kwargs.pop("control_freq", 20) != 20:
            raise ValueError("PanelOperation requires control_freq=20")
        for key, value in {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "initialization_noise": None,
            "hard_reset": False,
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
        self.add_post_integration_refresh_hook(self.update_state)
        self.add_post_physics_step_hook(self._record_post_physics_metrics)
        self.add_post_physics_step_hook(self._sample_imu)
        self.load_model_on_init = eager
        if eager:
            self.reset()

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(
            self.geometry_profile["initial_joint_qpos_rad"]
            if self.task_state is None
            else self.task_state["arm_qpos_rad"]
        )
        robot.robot_model.set_base_xpos(self.geometry_profile["robot_base_pos_m"])
        self.robot_base_body_name = robot.robot_model.root_body
        self.gripper_body_name = robot.robot_model.eef_name["right"]
        if type(robot.gripper["right"]).__name__ != "PandaGripper":
            raise ValueError("PanelOperation requires PandaGripper")
        self.arena = ShakeBenchArena(
            table_offset=self.geometry_profile["table_top_pos_m"],
            isolator_config=self.physics_profile.isolator_config(),
            worktable_mount=self.worktable_mount,
            scene_config=self.scene_path,
        )
        self.worktable_body_name = self.arena.worktable_body_name
        support = ET.parse(xml_path_completion(self.geometry_profile["robot_support_mjcf"])).getroot()
        self.arena.worldbody.append(support.find("./worldbody/body[@name='robot_support']"))
        # Check the compact case footprint with clearance around the rounded faceplate.
        c, s = np.cos(self.panel_yaw_rad), np.sin(self.panel_yaw_rad)
        half_extent = np.abs([[c, -s], [s, c]]) @ PANEL_HALF_FOOTPRINT_M
        if np.any(np.abs(self.panel_xy_m) + half_extent > self.arena.table_half_size[:2]):
            raise ValueError("panel footprint must fit on the worktable")
        fixture = ET.parse(xml_path_completion("objects/panel/panel.xml")).getroot()
        for resource in fixture.findall("./asset/*[@file]"):
            resource.set("file", xml_path_completion("objects/panel/" + resource.get("file")))
        self.arena.asset.extend(fixture.find("asset"))
        panel = fixture.find("./worldbody/body[@name='panel']")
        panel.set("pos", array_to_string(self.arena.table_top_abs + [*self.panel_xy_m, 0.0001]))
        panel.set("quat", array_to_string([np.cos(self.panel_yaw_rad / 2), 0, 0, np.sin(self.panel_yaw_rad / 2)]))
        ET.SubElement(panel, "freejoint", name="panel_free")
        # Inertia of a 152 x 224 x 55 mm, 2 kg solid box about its centre.
        ET.SubElement(panel, "inertial", pos="0 0 0.02", mass="2", diaginertia="0.008867 0.004351 0.012203")
        self.arena.worldbody.append(panel)
        self.model = ManipulationTask(self.arena, [robot.robot_model], [])
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        table_contact = self.physics_profile.pair_attributes(1.0)
        friction = table_contact["friction"].split()
        friction[1] = friction[0]
        table_contact["friction"] = " ".join(friction)
        control_collisions = {
            kind: tuple(
                geom.get("name")
                for geom in panel.find(f".//body[@name='panel_{kind}']").findall("./geom[@name]")
                if geom.get("name").endswith("_collision")
            )
            for kind in ("knob", "lever", "button")
        }
        moving_control_names = {name for names in control_collisions.values() for name in names}
        for geom in panel.findall(".//geom[@name]"):
            name = geom.get("name")
            if name.endswith("_collision") and name not in moving_control_names:
                ET.SubElement(self.model.contact, "pair", geom1=name, geom2="table_collision", **table_contact)
        pads = robot.gripper["right"].important_geoms
        for kind in ("knob", "lever", "button"):
            for name in control_collisions[kind]:
                for finger in pads["left_fingerpad"] + pads["right_fingerpad"]:
                    ET.SubElement(
                        self.model.contact,
                        "pair",
                        geom1=name,
                        geom2=finger,
                        **self.physics_profile.pair_attributes(
                            float(self.physics_profile.contact["sliding_mu"]["finger_object"]), finger_contact=True
                        ),
                    )

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model._model
        joint_ids = [self.sim.model.joint_name2id(name) for name in self.panel_joint_names]
        self.panel_qpos_indexes = model.jnt_qposadr[joint_ids].copy()
        self.panel_qvel_indexes = model.jnt_dofadr[joint_ids].copy()
        self.panel_indicator_geom_ids = np.array([model.geom(f"panel_annunciator_lamp{i}_visual").id for i in range(3)])
        self.panel_indicator_material_ids = np.array(
            [[model.mat(f"panel_lens_{i}{suffix}").id for suffix in ("", "_on")] for i in range(3)]
        )

        self.panel_button_cap_geom_id = model.geom("panel_cap_visual").id
        self.panel_button_cap_material_ids = [
            model.mat(name).id for name in ("panel_button_finish", "panel_button_lit")
        ]
        self.panel_body_id = model.body("panel").id
        self.panel_joint_ids = {kind: model.joint(f"panel_{kind}_joint").id for kind in ("knob", "lever", "button")}
        self.panel_control_geom_ids = {
            kind: model.geom(f"panel_{kind}_collision").id for kind in ("knob", "lever", "button")
        }
        self.panel_control_geom_groups = {
            kind: frozenset(
                geom.id
                for geom in (model.geom(i) for i in range(model.ngeom))
                if geom.bodyid == model.body(f"panel_{kind}").id and geom.name.endswith("_collision")
            )
            for kind in ("knob", "lever", "button")
        }
        self.robot_geom_ids = {
            model.geom(name).id for name in self.sim.model.geom_names if name.startswith(("robot0_", "gripper0_"))
        }

    def _reset_internal(self):
        super()._reset_internal()
        state = self.task_state
        if state is not None or not self.deterministic_reset:
            yaw = self.panel_yaw_rad
            self.sim.data.set_joint_qpos(
                "panel_free",
                [*(self.arena.table_top_abs + [*self.panel_xy_m, 0.0001]), np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
            )
            if state is None:
                self.sim.data.qpos[self.panel_qpos_indexes] = 0.0
            else:
                controls = state["controls"]
                self.sim.data.qpos[self.panel_qpos_indexes] = [
                    self.KNOB_RANGE_RAD if controls["knob_on"] else 0.0,
                    np.pi / 6 if controls["lever_up"] else -np.pi / 6,
                    self.BUTTON_ON_DEPTH_M if controls["button_on"] else 0.0,
                ]
            self.sim.data.qvel[self.panel_qvel_indexes] = 0.0
            if state is not None:
                robot = self.robots[0]
                fingers = robot._ref_gripper_joint_pos_indexes["right"]
                self.sim.data.qpos[fingers] = robot.gripper["right"].init_qpos if state["gripper_open"] else 0.0
                self.sim.data.qvel[robot._ref_gripper_joint_vel_indexes["right"]] = 0.0
        self.sim.data.qfrc_applied[self.panel_qvel_indexes] = 0.0
        knob = self.sim.data.qpos[self.panel_qpos_indexes[0]]
        self.panel_knob_on = bool(state["controls"]["knob_on"]) if state is not None else False
        self.panel_button_on = bool(state["controls"]["button_on"]) if state is not None else False
        self.panel_button_armed = False
        self._button_flip_count = 0
        model = self.sim.model._model
        model.qpos_spring[self.panel_qpos_indexes[0]] = self.KNOB_RANGE_RAD if knob >= self.KNOB_RANGE_RAD / 2 else 0.0
        model.qpos_spring[self.panel_qpos_indexes[2]] = (
            self.BUTTON_ON_DEPTH_M if self.panel_button_on else self.BUTTON_OFF_SPRING_M
        )
        self.sim.forward()
        self.update_state()
        if state is not None or not self.deterministic_reset:
            self._settle()
        self.sim.forward()
        self.update_state()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=float(self.sim.data.time))
        self._success = False
        self._violation = None
        self._candidate_since = None
        self._conditions = {}
        self._completed_steps = 0
        self._step_results = (
            [] if state is None else [{"step": step, "passed": False, "time_s": None} for step in state["steps"]]
        )
        self._step_baseline = self._discrete_controls()
        self._record_post_physics_metrics(float(self.sim.data.time))

    def _settle(self):
        """Settle the free panel for 0.5 s while fixing the Panda joints and gripper."""
        model, data = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qpos_ids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        qvel_ids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed_pose = data.qpos[qpos_ids].copy()
        for _ in range(round(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qpos_ids], data.qvel[qvel_ids] = fixed_pose, 0
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("non-finite panel reset")
        data.qvel[:], data.qacc_warmstart[:], data.time = 0, 0, 0

    def _discrete_controls(self):
        _, lever, _ = self.sim.data.qpos[self.panel_qpos_indexes]
        return {
            "knob": self.panel_knob_on,
            # Positive lever angle lights the amber indicator: up means +q.
            "lever": True if lever > np.deg2rad(5) else False if lever < -np.deg2rad(5) else None,
            "button": bool(self.panel_button_on),
        }

    def _record_post_physics_metrics(self, sample_time_s, policy_step=False):
        if self.task_state is None or self._success or self._violation is not None:
            return
        target = STEP_CONTROL[self.task_state["steps"][self._completed_steps]]
        target_ids = self.panel_control_geom_groups[target]
        released = True
        for contact in self.sim.data._data.contact[: self.sim.data._data.ncon]:
            if contact.dist <= 0.001 and (
                (contact.geom1 in target_ids and contact.geom2 in self.robot_geom_ids)
                or (contact.geom2 in target_ids and contact.geom1 in self.robot_geom_ids)
            ):
                released = False
                break
        self._record_control_metrics(
            sample_time_s, self._discrete_controls(), self.sim.data.qpos[self.panel_qpos_indexes], released
        )

    def _record_control_metrics(self, sample_time_s, discrete, values, released):
        """Score one physics sample, shared by MuJoCo and the device trace replay."""
        if self.task_state is None or self._success or self._violation is not None:
            return
        steps = self.task_state["steps"]
        if self._completed_steps >= len(steps):
            return
        step = steps[self._completed_steps]
        target = STEP_CONTROL[step]
        for control in ("knob", "lever", "button"):
            if (
                control != target
                and discrete[control] is not None
                and discrete[control] != self._step_baseline[control]
            ):
                self._violation = {
                    "control": control,
                    "step_index": self._completed_steps,
                    "time_s": float(sample_time_s),
                }
                self._candidate_since = None
                self._conditions["no_violation"] = False
                return
        knob, lever, button = values
        reached = {
            "press": discrete["button"] != self._step_baseline["button"]
            and self._button_flip_count == 1
            and button > self.BUTTON_RELEASE_THRESHOLD_M,
            "toggle_up": lever >= np.deg2rad(20),
            "toggle_down": lever <= -np.deg2rad(20),
            "turn_on": knob >= self.KNOB_ON_THRESHOLD_RAD,
            "turn_off": knob <= self.KNOB_OFF_THRESHOLD_RAD,
        }[step]
        self._conditions = {"target_reached": bool(reached), "released": released, "no_violation": True}
        if reached and released:
            if self._candidate_since is None:
                self._candidate_since = sample_time_s
            if sample_time_s - self._candidate_since >= SUCCESS_HOLD_S - 1e-12:
                self._step_results[self._completed_steps] = {
                    "step": step,
                    "passed": True,
                    "time_s": float(sample_time_s),
                }
                self._completed_steps += 1
                self._candidate_since = None
                self._step_baseline = discrete
                self._button_flip_count = 0
                self._success = self._completed_steps == len(steps)
        else:
            self._candidate_since = None

    def update_state(self, sample_time_s=None, policy_step=False):
        """Update the push-push switch and visual feedback before camera observations."""
        model, data = self.sim.model._model, self.sim.data._data
        knob, lever, button = data.qpos[self.panel_qpos_indexes]
        if not self.panel_button_armed and button < self.BUTTON_PRESS_THRESHOLD_M:
            self.panel_button_on = not self.panel_button_on
            self._button_flip_count += 1
            self.panel_button_armed = True
        elif self.panel_button_armed and button > self.BUTTON_RELEASE_THRESHOLD_M:
            self.panel_button_armed = False
        model.qpos_spring[self.panel_qpos_indexes[2]] = (
            self.BUTTON_ON_DEPTH_M if self.panel_button_on else self.BUTTON_OFF_SPRING_M
        )
        model.geom_matid[self.panel_button_cap_geom_id] = self.panel_button_cap_material_ids[int(self.panel_button_on)]
        if knob >= self.KNOB_ON_THRESHOLD_RAD:
            self.panel_knob_on = True
        elif knob <= self.KNOB_OFF_THRESHOLD_RAD:
            self.panel_knob_on = False
        model.qpos_spring[self.panel_qpos_indexes[0]] = self.KNOB_RANGE_RAD if knob >= self.KNOB_RANGE_RAD / 2 else 0.0
        active = np.array([self.panel_knob_on, lever > np.deg2rad(5), self.panel_button_on])
        model.geom_matid[self.panel_indicator_geom_ids] = self.panel_indicator_material_ids[
            np.arange(3), active.astype(int)
        ]

    def _sample_imu(self, sample_time_s, policy_step=False):
        self.table_imu_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        obs.update(self.table_imu_provider.observation(self.sim))
        if self.use_object_obs:
            for i, kind in enumerate(("knob", "lever", "button")):
                obs[f"panel_{kind}_pos"] = self.sim.data.qpos[self.panel_qpos_indexes[i : i + 1]].copy()
                obs[f"panel_{kind}_vel"] = self.sim.data.qvel[self.panel_qvel_indexes[i : i + 1]].copy()
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
            for kind, unit in (("knob", "rad"), ("lever", "rad"), ("button", "m")):
                fields.extend(
                    [(f"panel_{kind}_pos", (1,), unit, "panel"), (f"panel_{kind}_vel", (1,), f"{unit}/s", "panel")]
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

    def current_instruction(self):
        """Return the action now requested, retaining the last after completion."""
        if self.task_state is None:
            return None
        steps = self.task_state["steps"]
        return INSTRUCTIONS[steps[min(self._completed_steps, len(steps) - 1)]]

    def get_policy_task_context(self):
        steps = [] if self.task_state is None else self.task_state["steps"]
        return {
            "task_type": TASK_TYPE,
            "version": TASK_VERSION,
            "scoreable": False,
            "success_defined": bool(steps),
            "instruction": self.current_instruction(),
            "steps": list(steps),
            "current_step_index": self._completed_steps,
            "controls_initial": None if self.task_state is None else deepcopy(self.task_state["controls"]),
            "panel_xy_m": self.panel_xy_m.tolist(),
            "panel_yaw_rad": self.panel_yaw_rad,
            "knob_range_rad": [0.0, self.KNOB_RANGE_RAD],
            "lever_range_rad": [-np.pi / 6, np.pi / 6],
            "button_range_m": [-0.004, 0.0],
            "geometry_profile": self.geometry_profile["profile_id"],
            "control_asset_revision": "original_controls",
        }

    def get_metrics(self):
        if self.task_state is None:
            return {"success": {"defined": False, "passed": False}}
        return {
            "success": {"passed": bool(self._success), "subconditions": deepcopy(self._conditions)},
            "completed_steps": self._completed_steps,
            "step_results": deepcopy(self._step_results),
            "violation": deepcopy(self._violation),
            "task_rule_violation": self._violation is not None,
            "current_step_index": self._completed_steps,
        }


register_task(
    TASK_TYPE,
    TaskDefinition(
        env_factory=PanelOperation,
        normalize_state=validate_state,
        env_kwargs=lambda state: {"task_state": state},
        describe=lambda state: {
            "task_id": TASK_TYPE + "." + "-".join(state["steps"]),
            "instruction": INSTRUCTIONS[state["steps"][0]],
        },
        fingerprint=lambda state: {key: value for key, value in state.items() if key not in {"state_id", "split"}},
    ),
)
register_state_loader(STATE_SCHEMA, load_states)
