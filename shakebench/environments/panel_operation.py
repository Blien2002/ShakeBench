"""Interactive control panel on the shaken worktable, with no task goals or Oracle.

The knob rotates freely, the lever has +/-30 degree stops, and the button has
4 mm of spring-return travel. Joint resistance and contact proxies are in the
packaged panel.xml; observations use radians for hinges and metres for the slide.
"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
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
from shakebench.utils.physics import resolve_physics_profile
from shakebench.utils.privilege import assert_policy_observation_is_clean
from shakebench.utils.providers import POLICY_FIELD_CONTRACT, TableIMUProvider
from shakebench.utils.scene import DECK_VISUAL_BODY_NAME, configure_scene_rendering


class PanelOperation(ManipulationEnv):
    """Panda interaction scene; reward stays zero and success is undefined."""

    panel_joint_names = ("panel_knob_joint", "panel_lever_joint", "panel_button_joint")

    def __init__(
        self,
        robots="Panda",
        *,
        panel_xy_m=(0.0, 0.0),
        panel_yaw_rad=0.0,
        physics_profile="official",
        geometry_profile=DEFAULT_GEOMETRY_PROFILE,
        vibration=None,
        imu_mode="canonical_noisy_v1",
        imu_seed=None,
        use_object_obs=True,
        controller_configs=None,
        **kwargs,
    ):
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
                else {"mode": "multisine_v1", "gamma": 0.0, "seed": seed if seed is not None else 17, "t0_s": 0.0}
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
        self.add_post_physics_step_hook(self._sample_imu)
        self.load_model_on_init = eager
        if eager:
            self.reset()

    def _check_robot_configuration(self, robots):
        if (list(robots) if isinstance(robots, (list, tuple)) else [robots]) != ["Panda"]:
            raise ValueError("PanelOperation supports exactly one Panda")

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.init_qpos = np.asarray(self.geometry_profile["initial_joint_qpos_rad"])
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
        # Include the edge cheeks when checking the rotated case footprint.
        c, s = np.cos(self.panel_yaw_rad), np.sin(self.panel_yaw_rad)
        half_extent = np.abs([[c, -s], [s, c]]) @ [0.105, 0.160]
        if np.any(np.abs(self.panel_xy_m) + half_extent > self.arena.table_half_size[:2]):
            raise ValueError("panel footprint must fit on the worktable")
        fixture = ET.parse(xml_path_completion("objects/panel/panel.xml")).getroot()
        for resource in fixture.findall("./asset/*[@file]"):
            resource.set("file", xml_path_completion("objects/panel/" + resource.get("file")))
        self.arena.asset.extend(fixture.find("asset"))
        panel = fixture.find("./worldbody/body[@name='panel']")
        panel.set("pos", array_to_string([*self.panel_xy_m, self.arena.table_half_size[2] + 0.090]))
        panel.set("euler", array_to_string([0, 0, self.panel_yaw_rad]))
        self.arena.worktable_body.append(panel)
        self.model = ManipulationTask(self.arena, [robot.robot_model], [])
        configure_scene_rendering(self.model.root, self.arena.scene_config)
        pads = robot.gripper["right"].important_geoms
        for kind in ("knob", "lever", "button"):
            for finger in pads["left_fingerpad"] + pads["right_fingerpad"]:
                ET.SubElement(
                    self.model.contact,
                    "pair",
                    geom1=f"panel_{kind}_collision",
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

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            self.sim.data.qpos[self.panel_qpos_indexes] = 0.0
            self.sim.data.qvel[self.panel_qvel_indexes] = 0.0
        self.sim.data.qfrc_applied[self.panel_qvel_indexes] = 0.0
        self.sim.forward()
        self.deck_driver.reset_trace()
        self.table_imu_provider.reset(self.sim, timestamp_s=float(self.sim.data.time))

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

    @property
    def policy_observation_keys(self):
        return tuple(self.observation_contract())

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

    def get_policy_task_context(self):
        return {
            "task_type": "panel_operation",
            "scoreable": False,
            "success_defined": False,
            "panel_xy_m": self.panel_xy_m.tolist(),
            "panel_yaw_rad": self.panel_yaw_rad,
            "lever_range_rad": [-np.pi / 6, np.pi / 6],
            "button_range_m": [-0.004, 0.0],
            "geometry_profile": self.geometry_profile["profile_id"],
        }

    def get_metrics(self):
        return {"success": {"defined": False, "passed": False}}

    def _check_success(self):
        return False

    def reward(self, action=None):
        return 0.0
