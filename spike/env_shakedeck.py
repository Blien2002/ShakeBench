"""robosuite Lift with a mocap+weld ShakeBench deck."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import xml.etree.ElementTree as ET

import numpy as np

import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.panda_gripper import PandaGripperBase


ANCHOR = np.array((0.0, 0.0, 0.8), dtype=np.float64)
CONTACT_MARGIN_M = 1.0e-3
CONTACT_SOLREF = (6.0e-4, 1.0)
WELD_SOLREF = (4.0e-4, 1.0)
MATERIAL_MU = 1.5
DECK_BALLAST_KG = 400.0
PANDA_PAD_SOLREF = (1.0e-2, 0.5)
PANDA_PAD_FRICTION = (2.0, 5.0e-2, 1.0e-4)
CUBE_TABLE_PAIR_FRICTION = (1.5, 1.5, 5.0e-3, 1.0e-4, 1.0e-4)
DEFAULT_MOCAP_COMMAND_LEAD_STEPS = 1
PANDA_STOCK_GRIPPER_FORCE_LIMIT_N = 20.0

MotionSampler = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]
PhysicsStepCallback = Callable[["ShakeDeckLift"], None]


@register_gripper
class SpikeDirectPandaGripper(PandaGripperBase):
    """Panda geometry with two continuous targets for physical target latching."""


def _numbers(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _quat_wxyz_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """Quaternion for Rz(rz) Ry(ry) Rx(rx), in MuJoCo wxyz order."""

    rx, ry, rz = 0.5 * euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    quat = np.array(
        (
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ),
        dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)


def shake_deck_xml(
    xml_str: str,
    *,
    gripper_force_limit_n: float | None = None,
    gripper_acquisition_force_limit_n: float = PANDA_STOCK_GRIPPER_FORCE_LIMIT_N,
    pad_solref_dampratio: float = PANDA_PAD_SOLREF[1],
    cube_table_solref: tuple[float, float] = CONTACT_SOLREF,
    cube_table_sliding_mu: float = MATERIAL_MU,
    decoupled_table: bool = False,
) -> str:
    """Move Lift's table and Panda base under a welded dynamic deck."""

    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("robosuite XML has no worldbody")
    if worldbody.find("./body[@name='deck']") is not None:
        raise ValueError("ShakeDeck XML processor was applied twice")

    movable = {}
    for name in ("table", "robot0_base"):
        body = worldbody.find(f"./body[@name='{name}']")
        if body is None:
            raise ValueError(f"robosuite Lift XML has no top-level {name!r} body")
        movable[name] = body
        worldbody.remove(body)

    driver = ET.Element("body", name="deck_drv", mocap="true", pos=_numbers(ANCHOR))
    worldbody.append(driver)

    deck = ET.Element("body", name="deck", pos=_numbers(ANCHOR))
    ET.SubElement(deck, "freejoint", name="deck_freejoint")
    ET.SubElement(
        deck,
        "geom",
        name="deck_ballast",
        type="box",
        size="0.10 0.10 0.01",
        mass=f"{DECK_BALLAST_KG:g}",
        contype="0",
        conaffinity="0",
        rgba="0 0 0 0",
    )
    ET.SubElement(deck, "site", name="deck_anchor", pos="0 0 0", size="0.002")
    deck_members = ("robot0_base",) if decoupled_table else ("table", "robot0_base")
    for name in deck_members:
        body = movable[name]
        original = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body.set("pos", _numbers(original - ANCHOR))
        deck.append(body)
    worldbody.append(deck)

    if decoupled_table:
        table_driver = ET.Element(
            "body", name="table_deck_drv", mocap="true", pos=_numbers(ANCHOR)
        )
        worldbody.append(table_driver)
        table_deck = ET.Element("body", name="table_deck", pos=_numbers(ANCHOR))
        ET.SubElement(table_deck, "freejoint", name="table_deck_freejoint")
        ET.SubElement(
            table_deck,
            "geom",
            name="table_deck_ballast",
            type="box",
            size="0.10 0.10 0.01",
            mass=f"{DECK_BALLAST_KG:g}",
            contype="0",
            conaffinity="0",
            rgba="0 0 0 0",
        )
        ET.SubElement(
            table_deck, "site", name="table_deck_anchor", pos="0 0 0", size="0.002"
        )
        table = movable["table"]
        table_original = np.fromstring(table.get("pos", "0 0 0"), sep=" ")
        table.set("pos", _numbers(table_original - ANCHOR))
        table_deck.append(table)
        worldbody.append(table_deck)

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        name="deck_weld",
        body1="deck_drv",
        body2="deck",
        solref=_numbers(WELD_SOLREF),
    )
    if decoupled_table:
        ET.SubElement(
            equality,
            "weld",
            name="table_deck_weld",
            body1="table_deck_drv",
            body2="table_deck",
            solref=_numbers(WELD_SOLREF),
        )

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="table", body2="robot0_base")
    # These parameters belong to the workpiece-table interaction, not to every
    # collision geom in the model. An explicit pair avoids contaminating the
    # Panda pads through MuJoCo's per-contact parameter combination rules.
    ET.SubElement(
        contact,
        "pair",
        name="cube_table_contact",
        geom1="cube_g0",
        geom2="table_collision",
        margin=f"{CONTACT_MARGIN_M:g}",
        gap=f"{CONTACT_MARGIN_M:g}",
        solref=_numbers(cube_table_solref),
        friction=_numbers(
            (cube_table_sliding_mu, cube_table_sliding_mu, *CUBE_TABLE_PAIR_FRICTION[2:])
        ),
    )

    if pad_solref_dampratio <= 0.0:
        raise ValueError("pad solref damping ratio must be positive")
    pad_names = {
        "gripper0_right_finger1_pad_collision",
        "gripper0_right_finger2_pad_collision",
    }
    configured_pads = 0
    for geom in root.iter("geom"):
        if geom.get("name") in pad_names:
            geom.set("solref", _numbers((PANDA_PAD_SOLREF[0], pad_solref_dampratio)))
            configured_pads += 1
    if configured_pads != 2:
        raise ValueError(f"expected two Panda finger pads, configured {configured_pads}")

    if gripper_force_limit_n is not None:
        if not 0.0 < gripper_force_limit_n <= PANDA_STOCK_GRIPPER_FORCE_LIMIT_N:
            raise ValueError("gripper hold force limit must be in (0, 20] N")
        if not 0.0 < gripper_acquisition_force_limit_n <= PANDA_STOCK_GRIPPER_FORCE_LIMIT_N:
            raise ValueError("gripper acquisition force limit must be in (0, 20] N")
        configured = 0
        for actuator in root.iter("position"):
            name = actuator.get("name", "")
            if name.endswith(("gripper_finger_joint1", "gripper_finger_joint2")):
                actuator.set("forcelimited", "true")
                actuator.set(
                    "forcerange",
                    _numbers(
                        (
                            -gripper_acquisition_force_limit_n,
                            gripper_acquisition_force_limit_n,
                        )
                    ),
                )
                configured += 1
        if configured != 2:
            raise ValueError(f"expected two Panda finger actuators, configured {configured}")

    return ET.tostring(root, encoding="unicode")


class ShakeDeckLift(Lift):
    """Lift subclass whose model-step hook drives the welded mocap target."""

    motion_sampler: MotionSampler | None = None
    write_zero_motion: bool = True
    mocap_command_lead_steps: int = DEFAULT_MOCAP_COMMAND_LEAD_STEPS
    gripper_force_limit_n: float | None = None
    gripper_acquisition_force_limit_n: float = PANDA_STOCK_GRIPPER_FORCE_LIMIT_N
    pad_solref_dampratio: float = PANDA_PAD_SOLREF[1]
    cube_table_solref: tuple[float, float] = CONTACT_SOLREF
    cube_table_sliding_mu: float = MATERIAL_MU
    table_motion_sampler: MotionSampler | None = None
    decoupled_table: bool = False
    physics_step_callback: PhysicsStepCallback | None = None

    def configure_shakedeck(
        self,
        motion_sampler: MotionSampler | None,
        mocap_command_lead_steps: int = DEFAULT_MOCAP_COMMAND_LEAD_STEPS,
        gripper_force_limit_n: float | None = None,
        gripper_acquisition_force_limit_n: float = PANDA_STOCK_GRIPPER_FORCE_LIMIT_N,
        pad_solref_dampratio: float = PANDA_PAD_SOLREF[1],
        cube_table_solref: tuple[float, float] = CONTACT_SOLREF,
        cube_table_sliding_mu: float = MATERIAL_MU,
        table_motion_sampler: MotionSampler | None = None,
    ) -> None:
        self.motion_sampler = motion_sampler
        self.write_zero_motion = motion_sampler is None
        self.mocap_command_lead_steps = mocap_command_lead_steps
        self.gripper_force_limit_n = gripper_force_limit_n
        self.gripper_acquisition_force_limit_n = gripper_acquisition_force_limit_n
        self.gripper_force_phase = "acquisition"
        self.gripper_force_switch_history: list[dict] = []
        self.gripper_force_validation_history: list[dict] = []
        self.pad_solref_dampratio = pad_solref_dampratio
        self.cube_table_solref = cube_table_solref
        self.cube_table_sliding_mu = cube_table_sliding_mu
        self.table_motion_sampler = table_motion_sampler
        self.decoupled_table = table_motion_sampler is not None
        # Lift intentionally drops its cube from 10 mm. The spike measures
        # table-relative slip, so start in static contact instead of folding
        # that unrelated free-fall distance into the diagnostic.
        self.placement_initializer.z_offset = 0.0
        self.set_xml_processor(
            partial(
                shake_deck_xml,
                gripper_force_limit_n=gripper_force_limit_n,
                gripper_acquisition_force_limit_n=gripper_acquisition_force_limit_n,
                pad_solref_dampratio=pad_solref_dampratio,
                cube_table_solref=cube_table_solref,
                cube_table_sliding_mu=cube_table_sliding_mu,
                decoupled_table=self.decoupled_table,
            )
        )
        self.reset()
        self.deck_body_id = self.sim.model.body_name2id("deck")
        self.deck_site_id = self.sim.model.site_name2id("deck_anchor")
        driver_body_id = self.sim.model.body_name2id("deck_drv")
        self.deck_mocap_id = int(self.sim.model.body_mocapid[driver_body_id])
        if self.deck_mocap_id < 0:
            raise RuntimeError("deck_drv did not compile as a mocap body")
        if self.decoupled_table:
            self.table_deck_body_id = self.sim.model.body_name2id("table_deck")
            self.table_deck_site_id = self.sim.model.site_name2id("table_deck_anchor")
            table_driver_body_id = self.sim.model.body_name2id("table_deck_drv")
            self.table_deck_mocap_id = int(
                self.sim.model.body_mocapid[table_driver_body_id]
            )
            if self.table_deck_mocap_id < 0:
                raise RuntimeError("table_deck_drv did not compile as a mocap body")
        self._write_deck_target(0.0)
        self.sim.forward()

    def _write_deck_target(self, time_s: float) -> None:
        time_s += self.mocap_command_lead_steps * float(self.sim.model.opt.timestep)
        if self.motion_sampler is None:
            q = np.zeros(6, dtype=np.float64)
        else:
            q, _qd, _qdd = self.motion_sampler(time_s)
        self.sim.data.mocap_pos[self.deck_mocap_id] = ANCHOR + q[:3]
        self.sim.data.mocap_quat[self.deck_mocap_id] = _quat_wxyz_from_euler_xyz(q[3:])
        if self.decoupled_table:
            table_q, _table_qd, _table_qdd = self.table_motion_sampler(time_s)
            self.sim.data.mocap_pos[self.table_deck_mocap_id] = ANCHOR + table_q[:3]
            self.sim.data.mocap_quat[self.table_deck_mocap_id] = _quat_wxyz_from_euler_xyz(
                table_q[3:]
            )

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        self._write_deck_target(float(self.sim.data.time))

    def _update_observables(self, *args, **kwargs):
        super()._update_observables(*args, **kwargs)
        callback = getattr(self, "physics_step_callback", None)
        if callback is not None:
            callback(self)

    def commanded_deck_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.motion_sampler is None:
            q = np.zeros(6, dtype=np.float64)
        else:
            q, _qd, _qdd = self.motion_sampler(float(self.sim.data.time))
        return ANCHOR + q[:3], _quat_wxyz_from_euler_xyz(q[3:])

    def commanded_table_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.decoupled_table:
            return self.commanded_deck_pose()
        q, _qd, _qdd = self.table_motion_sampler(float(self.sim.data.time))
        return ANCHOR + q[:3], _quat_wxyz_from_euler_xyz(q[3:])

    def contact_configuration(self) -> dict:
        """Return the compiled contact parameters that guard this regression."""

        model = self.sim.model

        def geom_parameters(name: str) -> dict:
            geom_id = model.geom_name2id(name)
            return {
                "margin_m": float(model.geom_margin[geom_id]),
                "gap_m": float(model.geom_gap[geom_id]),
                "solref": np.asarray(model.geom_solref[geom_id]).tolist(),
                "friction": np.asarray(model.geom_friction[geom_id]).tolist(),
            }

        pair_id = next(
            index
            for index in range(model.npair)
            if {
                int(model.pair_geom1[index]),
                int(model.pair_geom2[index]),
            }
            == {
                model.geom_name2id("cube_g0"),
                model.geom_name2id("table_collision"),
            }
        )
        return {
            "cube_table_pair": {
                "margin_m": float(model.pair_margin[pair_id]),
                "gap_m": float(model.pair_gap[pair_id]),
                "solref": np.asarray(model.pair_solref[pair_id]).tolist(),
                "friction": np.asarray(model.pair_friction[pair_id]).tolist(),
            },
            "left_finger_pad": geom_parameters("gripper0_right_finger1_pad_collision"),
            "right_finger_pad": geom_parameters("gripper0_right_finger2_pad_collision"),
        }

    def gripper_actuator_configuration(self) -> dict:
        result = {}
        for actuator_id in range(self.sim.model.nu):
            name = self.sim.model.actuator_id2name(actuator_id) or ""
            if name.endswith(("gripper_finger_joint1", "gripper_finger_joint2")):
                result[name] = {
                    "force_limited": bool(self.sim.model.actuator_forcelimited[actuator_id]),
                    "force_range_n": np.asarray(
                        self.sim.model.actuator_forcerange[actuator_id]
                    ).tolist(),
                    "gain_parameters": np.asarray(
                        self.sim.model.actuator_gainprm[actuator_id]
                    ).tolist(),
                }
        return result

    def _gripper_actuator_ids(self) -> list[int]:
        return [
            actuator_id
            for actuator_id in range(self.sim.model.nu)
            if (self.sim.model.actuator_id2name(actuator_id) or "").endswith(
                ("gripper_finger_joint1", "gripper_finger_joint2")
            )
        ]

    def validate_gripper_force_configuration(
        self,
        expected_limit_n: float,
        *,
        phase: str,
    ) -> dict:
        expected = np.asarray((-expected_limit_n, expected_limit_n), dtype=np.float64)
        actuator_ids = self._gripper_actuator_ids()
        actual = [
            np.asarray(self.sim.model.actuator_forcerange[actuator_id]).copy()
            for actuator_id in actuator_ids
        ]
        passed = len(actual) == 2 and all(
            bool(self.sim.model.actuator_forcelimited[actuator_id])
            and np.array_equal(force_range, expected)
            for actuator_id, force_range in zip(actuator_ids, actual, strict=True)
        )
        result = {
            "phase": phase,
            "time_s": float(self.sim.data.time),
            "model_step_index": int(
                round(float(self.sim.data.time) / float(self.sim.model.opt.timestep))
            ),
            "expected_force_range_n": expected.tolist(),
            "actual_force_ranges_n": [force_range.tolist() for force_range in actual],
            "passed": passed,
        }
        self.gripper_force_validation_history.append(result)
        if not passed:
            raise RuntimeError(
                f"Panda finger actuator force limits are not {expected.tolist()}: {actual}"
            )
        return result

    def activate_hold_force_limit(
        self,
        *,
        policy_step_index: int,
        trigger: str,
    ) -> dict:
        if self.gripper_force_limit_n is None:
            raise RuntimeError("no gripper hold force limit is configured")
        if self.gripper_force_phase != "acquisition":
            raise RuntimeError(f"hold force limit already active: {self.gripper_force_phase}")
        before = self.gripper_actuator_configuration()
        target = np.asarray(
            (-self.gripper_force_limit_n, self.gripper_force_limit_n), dtype=np.float64
        )
        for actuator_id in self._gripper_actuator_ids():
            self.sim.model.actuator_forcerange[actuator_id] = target
        self.gripper_force_phase = "hold"
        validation = self.validate_gripper_force_configuration(
            self.gripper_force_limit_n,
            phase="hold",
        )
        event = {
            "event": "gripper_force_limit_switch",
            "trigger": trigger,
            "time_s": float(self.sim.data.time),
            "model_step_index": validation["model_step_index"],
            "policy_step_index": int(policy_step_index),
            "from_limit_n_per_finger": self.gripper_acquisition_force_limit_n,
            "to_limit_n_per_finger": self.gripper_force_limit_n,
            "before": before,
            "after": self.gripper_actuator_configuration(),
            "post_switch_validation": validation,
        }
        self.gripper_force_switch_history.append(event)
        return event

    def validate_contact_configuration(self) -> None:
        configuration = self.contact_configuration()
        pair = configuration["cube_table_pair"]
        if not (
            pair["margin_m"] == CONTACT_MARGIN_M
            and pair["gap_m"] == CONTACT_MARGIN_M
            and np.array_equal(pair["solref"], np.asarray(self.cube_table_solref))
            and np.array_equal(
                pair["friction"],
                np.asarray(
                    (
                        self.cube_table_sliding_mu,
                        self.cube_table_sliding_mu,
                        *CUBE_TABLE_PAIR_FRICTION[2:],
                    )
                ),
            )
        ):
            raise RuntimeError(f"cube-table contact pair is misconfigured: {pair}")
        for side in ("left_finger_pad", "right_finger_pad"):
            pad = configuration[side]
            if not (
                pad["margin_m"] == 0.0
                and pad["gap_m"] == 0.0
                and np.array_equal(
                    pad["solref"],
                    np.asarray((PANDA_PAD_SOLREF[0], self.pad_solref_dampratio)),
                )
                and np.array_equal(pad["friction"], np.asarray(PANDA_PAD_FRICTION))
            ):
                raise RuntimeError(f"{side} no longer has Panda stock contact parameters: {pad}")
        if self.gripper_force_limit_n is not None:
            self.validate_gripper_force_configuration(
                self.gripper_acquisition_force_limit_n,
                phase="acquisition",
            )


def make_env(
    *,
    seed: int,
    physics_timestep: float,
    motion_sampler: MotionSampler | None,
    control_freq: int = 20,
    horizon: int = 1000,
    direct_gripper: bool = False,
    mocap_command_lead_steps: int = DEFAULT_MOCAP_COMMAND_LEAD_STEPS,
    gripper_force_limit_n: float | None = None,
    gripper_acquisition_force_limit_n: float = PANDA_STOCK_GRIPPER_FORCE_LIMIT_N,
    pad_solref_dampratio: float = PANDA_PAD_SOLREF[1],
    cube_table_solref: tuple[float, float] = CONTACT_SOLREF,
    cube_table_sliding_mu: float = MATERIAL_MU,
    osc_kp: float = 150.0,
    table_motion_sampler: MotionSampler | None = None,
) -> ShakeDeckLift:
    """Build a headless deterministic Lift and then install the XML processor."""

    # robosuite reads this global while constructing both MJCF and controllers.
    macros.SIMULATION_TIMESTEP = physics_timestep
    controller = load_composite_controller_config(controller="BASIC", robot="Panda")
    controller["body_parts"]["right"]["kp"] = osc_kp
    env = ShakeDeckLift(
        robots="Panda",
        controller_configs=controller,
        gripper_types="SpikeDirectPandaGripper" if direct_gripper else "default",
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
        hard_reset=True,
        lite_physics=True,
        seed=seed,
    )
    env.configure_shakedeck(
        motion_sampler,
        mocap_command_lead_steps,
        gripper_force_limit_n,
        gripper_acquisition_force_limit_n,
        pad_solref_dampratio,
        cube_table_solref,
        cube_table_sliding_mu,
        table_motion_sampler,
    )
    env.validate_contact_configuration()
    return env
