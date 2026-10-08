"""ShakeBench's seam on top of an unmodified robosuite ``ManipulationEnv``.

robosuite steps MuJoCo with the global ``macros.SIMULATION_TIMESTEP`` and offers
no callback inside its physics loop.  ShakeBench needs three things that robosuite
does not provide:

* an environment-owned MuJoCo timestep (the official physics profile runs at 5 kHz);
* callbacks around every physics step: the dynamic deck drives the shaken
  worktable before each step, and the table IMU samples after integration;
* hidden visualization sites that cast no shadow, so camera images do not depend
  on which debug sites a model defines.

Every ShakeBench task environment subclasses :class:`ShakeBenchEnv`.  The step
loop below reproduces robosuite's ``MujocoEnv.step`` and only adds the hooks, so
an environment that registers no hook steps exactly like robosuite.

Adapted from robosuite (https://github.com/ARISE-Initiative/robosuite), MIT
License, Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot
Perception and Learning Lab.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils import SimulationError
from robosuite.utils.binding_utils import MjSim

from shakebench.physics.calibration import build_vibration_program
from shakebench.physics.deck import DeckDriver
from shakebench.physics.profile import resolve_physics_profile
from shakebench.scene.config import DECK_VISUAL_BODY_NAME
from shakebench.scene.geometry import geometry_scene_path, load_geometry_profile, worktable_mount
from shakebench.sensors.providers import TableIMUProvider

PhysicsHook = Callable[[float, bool], None]


def validate_model_timestep(model_timestep):
    """Return ``None`` or the requested timestep as a finite positive float, in seconds."""

    if model_timestep is None:
        return None
    if isinstance(model_timestep, (bool, np.bool_)):
        raise ValueError("model_timestep must be a finite positive number")
    try:
        value = float(model_timestep)
    except (TypeError, ValueError) as exc:
        raise ValueError("model_timestep must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("model_timestep must be a finite positive number")
    return value


def _set_xml_model_timestep(xml_string, model_timestep):
    """Set a model-local timestep in an MJCF string without touching robosuite macros."""

    root = ET.fromstring(xml_string)
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", format(float(model_timestep), ".17g"))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _require_callable(hook, kind):
    if hook is None:
        raise ValueError(f"{kind} hook must be callable")
    if not callable(hook):
        raise TypeError(f"{kind} hook must be callable")
    return hook


class ShakeBenchEnv(ManipulationEnv):
    """``ManipulationEnv`` with an owned timestep, physics-step hooks and shadowless hidden sites.

    Args:
        *args: Forwarded to ``ManipulationEnv``.
        model_timestep (None or float): MuJoCo timestep in seconds for this environment only.  ``None`` keeps
            robosuite's ``macros.SIMULATION_TIMESTEP`` and its truncating step count.  An explicit timestep must
            divide the control period exactly.
        **kwargs: Forwarded to ``ManipulationEnv``.
    """

    def __init__(self, *args, model_timestep=None, **kwargs):
        # ManipulationEnv.__init__ may compile and reset the model, so the seam exists before it runs.
        self._requested_model_timestep = validate_model_timestep(model_timestep)
        self._control_steps = None
        self._pre_physics_step_hooks: list[PhysicsHook] = []
        self._post_physics_step_hooks: list[PhysicsHook] = []
        self._post_integration_refresh_hooks: list[PhysicsHook] = []
        self._sim_initialization_hooks: list[Callable] = []
        self._post_integration_refresh_requested = False
        self._post_integration_refresh_stride = 1
        self._physics_step_index = 0
        self._hidden_site_alpha: dict[int, float] = {}
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ timing

    def initialize_time(self, control_freq):
        """Set model and control timesteps from the compiled model, the requested timestep or robosuite's macro."""

        self.cur_time = 0
        if self.sim is not None:
            self.model_timestep = float(self.sim.model.opt.timestep)
        elif self._requested_model_timestep is not None:
            self.model_timestep = self._requested_model_timestep
        else:
            self.model_timestep = macros.SIMULATION_TIMESTEP
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError("Control frequency {} is invalid".format(control_freq))
        self.control_timestep = 1.0 / control_freq

        steps = self.control_timestep / self.model_timestep
        rounded_steps = int(round(steps))
        if self._requested_model_timestep is None:
            # robosuite's own path, including its truncation for unusual control/model frequency pairs.
            self._control_steps = int(steps)
        else:
            if rounded_steps < 1 or not np.isclose(steps, rounded_steps, rtol=1e-10, atol=1e-10):
                raise SimulationError(
                    "control timestep {} must be an integer multiple of model timestep {}".format(
                        self.control_timestep, self.model_timestep
                    )
                )
            self._control_steps = rounded_steps

    # ------------------------------------------------------------------- hooks

    def add_sim_initialization_hook(self, hook):
        """Call ``hook(sim)`` after every simulator compilation, once the model timestep is resolved."""

        self._sim_initialization_hooks.append(_require_callable(hook, "sim-initialization"))

    def request_post_integration_refresh(self, stride=1):
        """Opt in to a full derived-state refresh after every ``stride``-th physics step.

        With the default ``stride=1`` a ``sim.forward()`` follows every integration, before observables and
        post-physics hooks, so qpos, kinematics, constraints and the sample timestamp describe one state.  A larger
        stride records a decimated measurement stream without changing the integration.  Post-physics hooks run
        only on refreshed steps.
        """

        if isinstance(stride, (bool, np.bool_)):
            raise ValueError("post-integration refresh stride must be a positive integer")
        try:
            numeric_stride = float(stride)
            stride = int(numeric_stride)
        except (TypeError, ValueError) as exc:
            raise ValueError("post-integration refresh stride must be a positive integer") from exc
        if not np.isfinite(numeric_stride) or stride < 1 or numeric_stride != float(stride):
            raise ValueError("post-integration refresh stride must be a positive integer")
        self._post_integration_refresh_stride = stride
        self._post_integration_refresh_requested = True

    def add_post_integration_refresh_hook(self, hook):
        """Call ``hook(sample_time_s, policy_step)`` after integration and before the opt-in refresh."""

        self._post_integration_refresh_hooks.append(_require_callable(hook, "post-integration refresh"))

    def add_pre_physics_step_hook(self, hook):
        """Call ``hook(physics_time_s, policy_step)`` immediately before each MuJoCo step."""

        self._pre_physics_step_hooks.append(_require_callable(hook, "pre-physics"))

    def add_post_physics_step_hook(self, hook):
        """Call ``hook(sample_time_s, policy_step)`` after each refreshed physics step and observable update."""

        self._post_physics_step_hooks.append(_require_callable(hook, "post-physics"))

    def set_pre_physics_step_hook(self, hook):
        """Replace all pre-physics hooks with ``hook`` (or none)."""

        self._pre_physics_step_hooks = []
        if hook is not None:
            self.add_pre_physics_step_hook(hook)

    def set_post_physics_step_hook(self, hook):
        """Replace all post-physics hooks with ``hook`` (or none)."""

        self._post_physics_step_hooks = []
        if hook is not None:
            self.add_post_physics_step_hook(hook)

    def _pre_physics_step(self, physics_time_s, policy_step=False):
        """Subclass hook invoked before MuJoCo computes the next step."""

    def _post_physics_step(self, physics_time_s, policy_step=False):
        """Subclass hook invoked after MuJoCo and observable updates on refreshed steps."""

    # -------------------------------------------------------------- simulation

    def _initialize_sim(self, xml_string=None):
        """Compile the processed MJCF, applying the environment-owned timestep after every XML processor."""

        xml = xml_string if xml_string else self.model.get_xml()
        for processor in self._xml_processors:
            xml = processor(xml)
        if self._requested_model_timestep is not None:
            xml = _set_xml_model_timestep(xml, self._requested_model_timestep)

        self.sim = MjSim.from_xml_string(xml)
        self.sim.forward()
        self.initialize_time(self.control_freq)
        self._physics_step_index = 0
        self._hidden_site_alpha = {}
        for hook in self._sim_initialization_hooks:
            hook(self.sim)

    def _reset_internal(self):
        super()._reset_internal()
        self._physics_step_index = 0

    def step(self, action):
        """Advance one control period; same contract as ``robosuite.environments.base.MujocoEnv.step``."""

        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        # Only the first physics step of a control period carries a new policy action.
        policy_step = True
        for _ in range(self._control_steps):
            physics_time_s = float(self.sim.data.time)
            self._pre_physics_step(physics_time_s, policy_step)
            for hook in self._pre_physics_step_hooks:
                hook(physics_time_s, policy_step)
            if self.lite_physics:
                self.sim.step1()
            else:
                self.sim.forward()
            self._pre_action(action, policy_step)
            if self.lite_physics:
                self.sim.step2()
            else:
                self.sim.step()
            refresh_due = self._post_integration_refresh_requested and (
                (self._physics_step_index + 1) % self._post_integration_refresh_stride == 0
            )
            if refresh_due:
                refresh_time_s = float(self.sim.data.time)
                for hook in self._post_integration_refresh_hooks:
                    hook(refresh_time_s, policy_step)
                self.sim.forward()
            self._update_observables()
            if refresh_due:
                sample_time_s = float(self.sim.data.time)
                self._post_physics_step(sample_time_s, policy_step)
                for hook in self._post_physics_step_hooks:
                    hook(sample_time_s, policy_step)
            self._physics_step_index += 1
            policy_step = False

        # Accumulate once per control step to avoid floating point drift.
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
        elif self.has_renderer and self.renderer == "mjviewer" and self.viewer is None:
            # Relaunch a viewer that was closed by the user.
            self.initialize_renderer()
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info

    # ----------------------------------------------------------- visualization

    def visualize(self, vis_settings):
        """Toggle robosuite visualization sites, hiding them with zero alpha.

        robosuite hides a site by negating its alpha, and MuJoCo still renders the shadow of such a site.  The
        hidden alphas are handed back to robosuite before each toggle, so its visibility logic is unchanged.
        """

        rgba = self.sim.model.site_rgba
        for site_id, alpha in self._hidden_site_alpha.items():
            rgba[site_id, 3] = -alpha
        self._hidden_site_alpha = {}
        super().visualize(vis_settings)
        for site_id in self._visualization_site_ids():
            alpha = float(rgba[site_id, 3])
            if alpha < 0:
                self._hidden_site_alpha[site_id] = -alpha
                rgba[site_id, 3] = 0.0

    def _visualization_site_ids(self):
        """Sites whose visibility robosuite manages: task objects, robots and grippers."""

        models = list(getattr(self.model, "mujoco_objects", None) or ())
        for robot in self.robots:
            models.append(robot.robot_model)
            models.extend(gripper for gripper in getattr(robot, "gripper", {}).values() if gripper is not None)
        names = {name for model in models for name in model.sites}
        return sorted(self.sim.model.site_name2id(name) for name in names)


class ShakeBenchTask(ShakeBenchEnv):
    """A registered task on the shaken worktable: one Panda, 20 Hz OSC control, the dynamic deck and the table IMU.

    A subclass constructor validates its state and sets its task fields, calling :meth:`_init_worktable` before
    anything that needs the physics or geometry profile and :meth:`_init_robosuite` as its last step.  Subclasses
    define ``_record_post_physics_metrics`` and keep ``self._success`` as the latched success flag.
    """

    def _init_worktable(self, state, physics_profile, geometry_profile, vibration, imu_mode, imu_seed):
        """Resolve the physics and geometry profiles, then build the table IMU and the vibration-driven deck."""

        self.physics_profile = resolve_physics_profile(physics_profile)
        self.geometry_profile = load_geometry_profile(geometry_profile)
        self.worktable_mount = worktable_mount(geometry_profile)
        self.scene_path = geometry_scene_path(geometry_profile)
        self.table_imu_provider = TableIMUProvider(
            seed=state["imu_seed"] if imu_seed is None else imu_seed, imu_mode=imu_mode
        )
        if vibration is None:
            vibration = {"mode": "multisine_v1", "gamma": 0.0, "seed": state["excitation_seed"], "t0_s": state["t0_s"]}
        self.deck_driver = DeckDriver(
            trajectory=build_vibration_program(vibration),
            config=self.physics_profile.deck_driver_config(),
            body_handles={"isolated_worktable": "worktable", "deck_visual": DECK_VISUAL_BODY_NAME},
            required_roles=("isolated_worktable", "deck_visual"),
        )

    def _init_robosuite(self, robots, controller_configs, state, kwargs):
        """Construct the robosuite environment at 20 Hz, attach the deck and IMU hooks, and reset unless deferred."""

        eager = kwargs.pop("load_model_on_init", True)
        if kwargs.pop("control_freq", 20) != 20:
            raise ValueError(f"{type(self).__name__} requires control_freq=20")
        defaults = {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "initialization_noise": None,
            "hard_reset": False,
            "seed": state["excitation_seed"],
        }
        for key, value in defaults.items():
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
        self.add_post_physics_step_hook(self._record_post_physics_metrics)
        self.add_post_physics_step_hook(self._update_imu_provider)
        self.load_model_on_init = eager
        if eager:
            self.reset()

    def _check_robot_configuration(self, robots):
        if (list(robots) if isinstance(robots, (list, tuple)) else [robots]) != ["Panda"]:
            raise ValueError(f"{type(self).__name__} supports exactly one Panda")

    @property
    def policy_observation_keys(self):
        return tuple(self.observation_contract())

    def _check_success(self):
        return bool(self._success)

    def reward(self, action=None):
        return float(self._success)

    def _update_imu_provider(self, sample_time_s, policy_step=False):
        self.table_imu_provider.on_physics_sample(self.sim, sample_time_s, policy_step=policy_step)


# The base classes are not tasks: keep them out of robosuite's ``make`` registry.
for _base in (ShakeBenchEnv, ShakeBenchTask):
    REGISTERED_ENVS.pop(_base.__name__, None)
