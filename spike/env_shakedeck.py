"""robosuite Lift with a mocap+weld ShakeBench deck."""

from __future__ import annotations

from collections.abc import Callable
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

MotionSampler = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]


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


def shake_deck_xml(xml_str: str) -> str:
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
    for body in movable.values():
        original = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body.set("pos", _numbers(original - ANCHOR))
        deck.append(body)
    worldbody.append(deck)

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

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="table", body2="robot0_base")

    for geom in root.iter("geom"):
        if geom.get("contype", "1") == "0":
            continue
        geom.set("margin", f"{CONTACT_MARGIN_M:g}")
        geom.set("gap", f"{CONTACT_MARGIN_M:g}")
        geom.set("solref", _numbers(CONTACT_SOLREF))
        friction = geom.get("friction", "").split()
        rolling = friction[1:] if len(friction) >= 3 else ["0.005", "0.0001"]
        geom.set("friction", " ".join((f"{MATERIAL_MU:g}", *rolling[:2])))

    return ET.tostring(root, encoding="unicode")


class ShakeDeckLift(Lift):
    """Lift subclass whose model-step hook drives the welded mocap target."""

    motion_sampler: MotionSampler | None = None
    write_zero_motion: bool = True

    def configure_shakedeck(self, motion_sampler: MotionSampler | None) -> None:
        self.motion_sampler = motion_sampler
        self.write_zero_motion = motion_sampler is None
        # Lift intentionally drops its cube from 10 mm. The spike measures
        # table-relative slip, so start in static contact instead of folding
        # that unrelated free-fall distance into the diagnostic.
        self.placement_initializer.z_offset = 0.0
        self.set_xml_processor(shake_deck_xml)
        self.reset()
        self.deck_body_id = self.sim.model.body_name2id("deck")
        self.deck_site_id = self.sim.model.site_name2id("deck_anchor")
        driver_body_id = self.sim.model.body_name2id("deck_drv")
        self.deck_mocap_id = int(self.sim.model.body_mocapid[driver_body_id])
        if self.deck_mocap_id < 0:
            raise RuntimeError("deck_drv did not compile as a mocap body")
        self._write_deck_target(0.0)
        self.sim.forward()

    def _write_deck_target(self, time_s: float) -> None:
        if self.motion_sampler is None:
            q = np.zeros(6, dtype=np.float64)
        else:
            q, _qd, _qdd = self.motion_sampler(time_s)
        self.sim.data.mocap_pos[self.deck_mocap_id] = ANCHOR + q[:3]
        self.sim.data.mocap_quat[self.deck_mocap_id] = _quat_wxyz_from_euler_xyz(q[3:])

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        self._write_deck_target(float(self.sim.data.time))

    def commanded_deck_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.motion_sampler is None:
            q = np.zeros(6, dtype=np.float64)
        else:
            q, _qd, _qdd = self.motion_sampler(float(self.sim.data.time))
        return ANCHOR + q[:3], _quat_wxyz_from_euler_xyz(q[3:])


def make_env(
    *,
    seed: int,
    physics_timestep: float,
    motion_sampler: MotionSampler | None,
    control_freq: int = 20,
    horizon: int = 1000,
    direct_gripper: bool = False,
) -> ShakeDeckLift:
    """Build a headless deterministic Lift and then install the XML processor."""

    # robosuite reads this global while constructing both MJCF and controllers.
    macros.SIMULATION_TIMESTEP = physics_timestep
    controller = load_composite_controller_config(controller="BASIC", robot="Panda")
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
    env.configure_shakedeck(motion_sampler)
    return env
