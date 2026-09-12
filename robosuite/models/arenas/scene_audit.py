"""Compiled ShakeBench scene audits and clearance scans.

This is an explicit audit tool, separate from runtime scene configuration and
visual assembly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from robosuite.utils.shakebench_scene import (
    DECK_BODY_NAME,
    DECK_FREEJOINT_NAME,
    DECK_VISUAL_BODY_NAME,
    SCENE_BODY_PREFIX,
    WORKTABLE_BODY_NAME,
    SceneAuditError,
    SceneConfigError,
    SceneVisualConfig,
    _coerce_config,
    _require_mapping,
    _stewart_points,
    _thaw,
    rotation_vector_to_quat,
    scene_visual_geom_names,
)


def _raw_model_and_data(sim_or_model: Any) -> tuple[Any, Any]:
    model = getattr(sim_or_model, "model", sim_or_model)
    raw_model = getattr(model, "_model", model)
    data = getattr(sim_or_model, "data", None)
    raw_data = getattr(data, "_data", data)
    if raw_data is None:
        raw_data = mujoco.MjData(raw_model)
        mujoco.mj_forward(raw_model, raw_data)
    return raw_model, raw_data


def _mujoco_id(raw_model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(raw_model, object_type, name))
    if value < 0:
        raise SceneAuditError(f"compiled scene is missing {object_type} {name!r}")
    return value


def _body_ancestry(raw_model: Any, body_id: int) -> tuple[str, ...]:
    names: list[str] = []
    current = int(body_id)
    while current != 0:
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, current)
        if name is None:
            break
        names.append(str(name))
        current = int(raw_model.body_parentid[current])
    return tuple(reversed(names))


def _geom_body_name(raw_model: Any, geom_id: int) -> str:
    body_id = int(raw_model.geom_bodyid[geom_id])
    name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return str(name)


def _frame_for_body(raw_model: Any, body_id: int) -> str:
    ancestry = _body_ancestry(raw_model, body_id)
    if WORKTABLE_BODY_NAME in ancestry:
        return "isolated_worktable"
    if DECK_BODY_NAME in ancestry:
        return "dynamic_deck"
    if any(name.startswith("shakebench_") for name in ancestry):
        return "world"
    if mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id) == "can_main":
        return "world_free_body"
    return "world"


def _canonical_array(value: Any) -> list[float | int]:
    array = np.asarray(value)
    if array.dtype.kind in "iu":
        return [int(item) for item in array.reshape(-1)]
    return [float(item) for item in array.reshape(-1)]


def _physics_signature_payload(raw_model: Any) -> dict[str, Any]:
    def is_scene_body(name: str) -> bool:
        return name.startswith(SCENE_BODY_PREFIX)

    physical_bodies: dict[str, Any] = {}
    for body_id in range(int(raw_model.nbody)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None or is_scene_body(str(name)):
            continue
        parent_id = int(raw_model.body_parentid[body_id])
        physical_bodies[str(name)] = {
            "parent": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
            "pos": _canonical_array(raw_model.body_pos[body_id]),
            "quat": _canonical_array(raw_model.body_quat[body_id]),
            "mass": float(raw_model.body_mass[body_id]),
            "ipos": _canonical_array(raw_model.body_ipos[body_id]),
            "inertia": _canonical_array(raw_model.body_inertia[body_id]),
        }
    physical_geoms: dict[str, Any] = {}
    for geom_id in range(int(raw_model.ngeom)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            continue
        if not int(raw_model.geom_contype[geom_id]) and not int(raw_model.geom_conaffinity[geom_id]):
            continue
        physical_geoms[str(name)] = {
            "body": _geom_body_name(raw_model, geom_id),
            "type": int(raw_model.geom_type[geom_id]),
            "pos": _canonical_array(raw_model.geom_pos[geom_id]),
            "quat": _canonical_array(raw_model.geom_quat[geom_id]),
            "size": _canonical_array(raw_model.geom_size[geom_id]),
            "friction": _canonical_array(raw_model.geom_friction[geom_id]),
            "solref": _canonical_array(raw_model.geom_solref[geom_id]),
            "solimp": _canonical_array(raw_model.geom_solimp[geom_id]),
            "contype": int(raw_model.geom_contype[geom_id]),
            "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
            "condim": int(raw_model.geom_condim[geom_id]),
            "margin": float(raw_model.geom_margin[geom_id]),
            "gap": float(raw_model.geom_gap[geom_id]),
        }
    joints: dict[str, Any] = {}
    for joint_id in range(int(raw_model.njnt)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        dof_id = int(raw_model.jnt_dofadr[joint_id])
        qpos_id = int(raw_model.jnt_qposadr[joint_id])
        joints[str(name)] = {
            "body": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, int(raw_model.jnt_bodyid[joint_id])),
            "type": int(raw_model.jnt_type[joint_id]),
            "pos": _canonical_array(raw_model.jnt_pos[joint_id]),
            "axis": _canonical_array(raw_model.jnt_axis[joint_id]),
            "range": _canonical_array(raw_model.jnt_range[joint_id]),
            "limited": bool(raw_model.jnt_limited[joint_id]),
            "stiffness": float(raw_model.jnt_stiffness[joint_id]),
            "damping": float(raw_model.dof_damping[dof_id]),
            "springref": float(raw_model.qpos_spring[qpos_id]),
        }
    equalities: dict[str, Any] = {}
    for equality_id in range(int(raw_model.neq)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id)
        if name is None:
            continue
        equalities[str(name)] = {
            "type": int(raw_model.eq_type[equality_id]),
            "body1": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, int(raw_model.eq_obj1id[equality_id])),
            "body2": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, int(raw_model.eq_obj2id[equality_id])),
            "joint1": (
                mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, int(raw_model.eq_obj1id[equality_id]))
                if int(raw_model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT)
                else None
            ),
            "joint2": (
                mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, int(raw_model.eq_obj2id[equality_id]))
                if int(raw_model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT)
                else None
            ),
            "solref": _canonical_array(raw_model.eq_solref[equality_id]),
            "solimp": _canonical_array(raw_model.eq_solimp[equality_id]),
        }
    actuators: dict[str, Any] = {}
    for actuator_id in range(int(raw_model.nu)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name is None:
            continue
        trnid = _canonical_array(raw_model.actuator_trnid[actuator_id])
        target_name = None
        if int(raw_model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            target_name = mujoco.mj_id2name(
                raw_model, mujoco.mjtObj.mjOBJ_JOINT, int(raw_model.actuator_trnid[actuator_id, 0])
            )
        actuators[str(name)] = {
            "trntype": int(raw_model.actuator_trntype[actuator_id]),
            "trnid": trnid,
            "target": target_name,
            "ctrlrange": _canonical_array(raw_model.actuator_ctrlrange[actuator_id]),
            "ctrllimited": bool(raw_model.actuator_ctrllimited[actuator_id]),
        }
    contact_pairs = []
    for pair_id in range(int(raw_model.npair)):
        contact_pairs.append(
            {
                "geom1": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom1[pair_id])),
                "geom2": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom2[pair_id])),
                "friction": _canonical_array(raw_model.pair_friction[pair_id]),
                "solref": _canonical_array(raw_model.pair_solref[pair_id]),
                "solimp": _canonical_array(raw_model.pair_solimp[pair_id]),
                "margin": float(raw_model.pair_margin[pair_id]),
                "gap": float(raw_model.pair_gap[pair_id]),
                "dim": int(raw_model.pair_dim[pair_id]),
            }
        )
    options = {
        "timestep": float(raw_model.opt.timestep),
        "integrator": int(raw_model.opt.integrator),
        "solver": int(raw_model.opt.solver),
        "iterations": int(raw_model.opt.iterations),
        "tolerance": float(raw_model.opt.tolerance),
        "ls_iterations": int(raw_model.opt.ls_iterations),
        "noslip_iterations": int(raw_model.opt.noslip_iterations),
        "ccd_iterations": int(getattr(raw_model.opt, "ccd_iterations", 0)),
        "ccd_tolerance": float(getattr(raw_model.opt, "ccd_tolerance", 0.0)),
        "impratio": float(raw_model.opt.impratio),
        "cone": int(raw_model.opt.cone),
        "jacobian": int(raw_model.opt.jacobian),
        "disableflags": int(raw_model.opt.disableflags),
        "enableflags": int(raw_model.opt.enableflags),
    }
    return {
        "options": options,
        "bodies": physical_bodies,
        "geoms": physical_geoms,
        "joints": joints,
        "equalities": equalities,
        "actuators": actuators,
        "contact_pairs": contact_pairs,
    }


def compiled_physics_signature(sim_or_model: Any) -> dict[str, Any]:
    """Return a stable name-based signature excluding display-only geometry."""

    raw_model, _ = _raw_model_and_data(sim_or_model)
    payload = _physics_signature_payload(raw_model)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "payload": payload}


@dataclass(frozen=True)
class SceneAudit:
    """Compiled scene audit with stable names rather than raw MuJoCo IDs."""

    passed: bool
    config_sha256: str
    scene_id: str
    geometry_variant: str
    frame_ownership: Mapping[str, Any]
    role_audit: Mapping[str, Any]
    visual_geoms: Mapping[str, Any]
    visual_bodies: Mapping[str, Any]
    camera_audit: Mapping[str, Any]
    physics_signature: Mapping[str, Any]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "frame_ownership",
            "role_audit",
            "visual_geoms",
            "visual_bodies",
            "camera_audit",
            "physics_signature",
        ):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, MappingProxyType(_thaw(value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "config_sha256": self.config_sha256,
            "scene_id": self.scene_id,
            "geometry_variant": self.geometry_variant,
            "frame_ownership": _thaw(self.frame_ownership),
            "compiled_frame_ownership": {name: item["frame"] for name, item in self.visual_geoms.items()},
            "role_audit": _thaw(self.role_audit),
            "visual_geoms": _thaw(self.visual_geoms),
            "visual_bodies": _thaw(self.visual_bodies),
            "camera_audit": _thaw(self.camera_audit),
            "physics_signature": _thaw(self.physics_signature),
            "errors": list(self.errors),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def audit_compiled_scene(
    sim: Any, config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None
) -> SceneAudit:
    """Audit frame ownership, visual physics isolation, and cameras.

    Structural violations raise :class:`SceneAuditError`; a successful audit
    is also returned as a dict-like :class:`SceneAudit` for manifests.
    """

    scene_config = _coerce_config(config)
    raw_model, raw_data = _raw_model_and_data(sim)
    visual_names = scene_visual_geom_names(raw_model, scene_config)
    expected_roles = scene_config.section("role_handles")
    role_audit: dict[str, Any] = {}
    for role, body_name in expected_roles.items():
        body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        ancestry = _body_ancestry(raw_model, body_id)
        frame = _frame_for_body(raw_model, body_id)
        if role == "deck_visual" and frame != "dynamic_deck":
            raise SceneAuditError("deck_visual is not a descendant of dynamic deck")
        if role == "isolated_worktable" and WORKTABLE_BODY_NAME not in ancestry:
            raise SceneAuditError("isolated_worktable role is not attached to worktable")
        role_audit[role] = {"body_name": body_name, "ancestry": list(ancestry), "frame": frame, "body_id": body_id}
    for frame, content in scene_config.section("frame_ownership").items():
        if frame == "world_free_body":
            continue
        for body_name in content["bodies"]:
            body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            actual_frame = _frame_for_body(raw_model, body_id)
            if actual_frame != frame:
                raise SceneAuditError(f"frame ownership for body {body_name!r} is {actual_frame!r}, expected {frame!r}")

    visual_geoms: dict[str, Any] = {}
    for name in visual_names:
        geom_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body_name = _geom_body_name(raw_model, geom_id)
        if int(raw_model.geom_contype[geom_id]) != 0 or int(raw_model.geom_conaffinity[geom_id]) != 0:
            raise SceneAuditError(f"scene visual geom {name!r} contributes to contact")
        visual_geoms[name] = {
            "geom_id": geom_id,
            "body_name": body_name,
            "frame": _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom_id])),
            "type": int(raw_model.geom_type[geom_id]),
            "size_m": _canonical_array(raw_model.geom_size[geom_id]),
            "contype": int(raw_model.geom_contype[geom_id]),
            "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
        }
    if not any(name.startswith("shakebench_platen_") for name in visual_names):
        raise SceneAuditError("compiled scene is missing the visible vibration platen")
    for prefix, expected_frame in (
        ("shakebench_table_upper_", "isolated_worktable"),
        ("shakebench_table_lower_mount_", "dynamic_deck"),
        ("shakebench_stewart_outer_", "world"),
        ("shakebench_stewart_rod_", "dynamic_deck"),
    ):
        names = [name for name in visual_names if name.startswith(prefix)]
        if not names:
            raise SceneAuditError(f"compiled scene is missing visual role prefix {prefix!r}")
        wrong = [name for name in names if visual_geoms[name]["frame"] != expected_frame]
        if wrong:
            raise SceneAuditError(f"visual role {prefix!r} has wrong frame: {wrong}")

    visual_bodies: dict[str, Any] = {}
    for body_name in _body_name_set_from_model(raw_model):
        if not body_name.startswith(SCENE_BODY_PREFIX):
            continue
        body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_names = [
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(int(raw_model.njnt))
            if int(raw_model.jnt_bodyid[joint_id]) == body_id
        ]
        if joint_names:
            raise SceneAuditError(f"visual body {body_name!r} owns a joint: {joint_names}")
        if float(raw_model.body_subtreemass[body_id]) != 0.0:
            raise SceneAuditError(f"visual body {body_name!r} adds mass {raw_model.body_subtreemass[body_id]}")
        visual_bodies[body_name] = {
            "body_id": body_id,
            "frame": _frame_for_body(raw_model, body_id),
            "mass_kg": float(raw_model.body_mass[body_id]),
            "subtree_mass_kg": float(raw_model.body_subtreemass[body_id]),
            "inertia_kg_m2": _canonical_array(raw_model.body_inertia[body_id]),
            "joint_names": [name for name in joint_names if name is not None],
        }

    contact_pairs_with_scene_visuals = []
    visual_set = set(visual_names)
    for pair_id in range(int(raw_model.npair)):
        pair_names = (
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom1[pair_id])),
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom2[pair_id])),
        )
        if any(name in visual_set for name in pair_names):
            contact_pairs_with_scene_visuals.append(pair_names)
    if contact_pairs_with_scene_visuals:
        raise SceneAuditError(f"scene visuals appear in explicit contact pairs: {contact_pairs_with_scene_visuals}")
    active_visual_contacts = []
    for contact_id in range(int(raw_data.ncon)):
        pair_names = (
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_data.contact[contact_id].geom1)),
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_data.contact[contact_id].geom2)),
        )
        if any(name in visual_set for name in pair_names):
            active_visual_contacts.append(pair_names)
    if active_visual_contacts:
        raise SceneAuditError(f"scene visuals have active contacts: {active_visual_contacts}")

    camera_audit: dict[str, Any] = {}
    camera_names = {
        str(mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id))
        for camera_id in range(int(raw_model.ncam))
    }
    for key, camera in scene_config.section("cameras").items():
        name = str(camera["name"])
        if name not in camera_names:
            raise SceneAuditError(f"compiled scene is missing camera {name!r}")
        camera_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        camera_audit[key] = {
            "name": name,
            "camera_id": camera_id,
            "pos_m": _canonical_array(raw_model.cam_pos[camera_id]),
            "fovy_deg": float(raw_model.cam_fovy[camera_id]),
        }
    physics_signature = compiled_physics_signature(sim)
    return SceneAudit(
        passed=True,
        config_sha256=scene_config.config_sha256,
        scene_id=scene_config.scene_id,
        geometry_variant=scene_config.geometry_variant,
        frame_ownership=scene_config.section("frame_ownership"),
        role_audit=role_audit,
        visual_geoms=visual_geoms,
        visual_bodies=visual_bodies,
        camera_audit=camera_audit,
        physics_signature=physics_signature,
    )


def _body_name_set_from_model(raw_model: Any) -> tuple[str, ...]:
    return tuple(
        str(name)
        for body_id in range(int(raw_model.nbody))
        if (name := mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)) is not None
    )


def _geom_aabb(raw_model: Any, raw_data: Any, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    centre = np.asarray(raw_data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(raw_data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    geom_type = int(raw_model.geom_type[geom_id])
    size = np.asarray(raw_model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        local_half = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local_half = np.repeat(size[0], 3)
    elif geom_type in {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}:
        local_half = np.array(
            (size[0], size[0], size[1] + (size[0] if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0))
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        local_half = size[:3]
    else:
        radius = float(raw_model.geom_rbound[geom_id])
        local_half = np.repeat(radius, 3)
    extent = np.abs(rotation).dot(local_half)
    return centre - extent, centre + extent


def _geom_distance(raw_model: Any, raw_data: Any, geom1: int, geom2: int) -> tuple[float, np.ndarray]:
    support = _floor_cylinder_support_distance(raw_model, raw_data, geom1, geom2)
    if support is not None:
        return support
    fromto = np.zeros(6, dtype=np.float64)
    distance = float(mujoco.mj_geomDistance(raw_model, raw_data, int(geom1), int(geom2), 100.0, fromto))
    if not np.isfinite(distance):
        distance = -float("inf") if distance < 0.0 else float("inf")
    return distance, fromto.reshape(2, 3)


def _floor_cylinder_support_distance(model: Any, data: Any, geom1: int, geom2: int) -> tuple[float, np.ndarray] | None:
    """Exact top-face support distance for a cylinder contained over a floor slab.

    MuJoCo's convex distance can return millimetre-scale negative values for
    a tangent wheel on a broad, thin slab. Use the cylinder support function
    only when its entire XY bound lies inside a horizontal slab and its centre
    is above the slab. Penetration remains negative; no tolerance whitelist.
    """

    for slab, wheel in ((geom1, geom2), (geom2, geom1)):
        if not model.geom(slab).name.startswith("shakebench_floor_slab_"):
            continue
        if (
            model.geom_type[slab] != mujoco.mjtGeom.mjGEOM_BOX
            or model.geom_type[wheel] != mujoco.mjtGeom.mjGEOM_CYLINDER
        ):
            continue
        if not np.allclose(data.geom_xmat[slab].reshape(3, 3), np.eye(3), atol=1e-12, rtol=0):
            continue
        lower, upper = _geom_aabb(model, data, slab)
        wheel_lower, wheel_upper = _geom_aabb(model, data, wheel)
        centre = data.geom_xpos[wheel]
        if np.any(wheel_lower[:2] < lower[:2]) or np.any(wheel_upper[:2] > upper[:2]) or centre[2] < upper[2]:
            continue
        axis = data.geom_xmat[wheel].reshape(3, 3)[:, 2]
        radius, half_length = model.geom_size[wheel, :2]
        vertical = np.array([0.0, 0.0, 1.0])
        radial = vertical - axis[2] * axis
        norm = np.linalg.norm(radial)
        point = centre - half_length * np.sign(axis[2]) * axis
        if norm > 1e-12:
            point = point - radius * radial / norm
        floor_point = np.array([point[0], point[1], upper[2]])
        points = np.array([floor_point, point])
        return float(point[2] - upper[2]), points if slab == geom1 else points[::-1]
    return None


def _pair_whitelist(name1: str, name2: str, signed_distance_m: float | None = None) -> str | None:
    names = {name1, name2}
    if any(name.startswith("shakebench_pit_") for name in names) and any(
        name.startswith("shakebench_floor_slab_") for name in names
    ):
        return "pit_visual_boundary"
    if any(name.startswith("shakebench_pit_") for name in names) and any(
        name.startswith("shakebench_guardrail_") for name in names
    ):
        return "pit_safety_boundary"
    if any(name.startswith("shakebench_table_upper_") for name in names) and any(
        name.startswith("shakebench_table_lower_mount_") for name in names
    ):
        return "two_stage_isolator_mount_interface"
    if any(name.startswith("shakebench_stewart_rod") for name in names) and any(
        name.startswith("shakebench_platen_") for name in names
    ):
        return "stewart_rod_to_platen_joint_interface"
    if any(name.startswith("shakebench_stewart_outer") for name in names) and any(
        name.startswith("shakebench_stewart_rod") for name in names
    ):
        return "two_segment_stewart_overlap"
    if any(name.startswith("shakebench_shaker_foundation_") for name in names) and any(
        name.startswith("shakebench_pit_") for name in names
    ):
        return "shaker_foundation_inside_pit"
    return None


def _scene_collision_proxy_ids(raw_model: Any, config: SceneVisualConfig) -> tuple[int, ...]:
    prefixes = tuple(config.section("clearance")["mount_collision_prefixes"])
    ids = []
    for geom_id in range(int(raw_model.ngeom)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not (int(raw_model.geom_contype[geom_id]) or int(raw_model.geom_conaffinity[geom_id])):
            continue
        if str(name).startswith(prefixes):
            ids.append(geom_id)
    return tuple(ids)


def _candidate_pairs(
    raw_model: Any, raw_data: Any, config: SceneVisualConfig
) -> tuple[tuple[str, int, str, int, str | None], ...]:
    scene_names = scene_visual_geom_names(raw_model, config)
    scene_ids = {name: _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in scene_names}
    window = float(config.section("clearance")["candidate_window_m"])
    candidates: list[tuple[str, int, str, int, str | None]] = []
    for index, name1 in enumerate(scene_names):
        body1 = int(raw_model.geom_bodyid[scene_ids[name1]])
        for name2 in scene_names[index + 1 :]:
            geom2 = scene_ids[name2]
            if body1 == int(raw_model.geom_bodyid[geom2]):
                continue
            distance, _ = _geom_distance(raw_model, raw_data, scene_ids[name1], geom2)
            whitelist = _pair_whitelist(name1, name2)
            if whitelist is not None or distance <= window:
                candidates.append(("visual_visual", scene_ids[name1], name1, geom2, name2))
    proxy_ids = _scene_collision_proxy_ids(raw_model, config)
    for visual_name, visual_id in scene_ids.items():
        for proxy_id in proxy_ids:
            proxy_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, proxy_id)
            if proxy_name is None:
                continue
            distance, _ = _geom_distance(raw_model, raw_data, visual_id, proxy_id)
            whitelist = _pair_whitelist(visual_name, str(proxy_name))
            if whitelist is not None or distance <= window:
                candidates.append(("visual_collision_proxy", visual_id, visual_name, proxy_id, str(proxy_name)))
    support_ids = [i for i in proxy_ids if str(raw_model.geom(i).name).startswith("robot_support_")]
    existing = {(row[1], row[3]) for row in candidates}
    for support_id in support_ids:
        for geom_id in range(int(raw_model.ngeom)):
            if _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom_id])) not in {
                "dynamic_deck",
                "isolated_worktable",
            }:
                continue
            if (geom_id, support_id) not in existing:
                candidates.append(
                    (
                        "support_motion_clearance",
                        geom_id,
                        raw_model.geom(geom_id).name,
                        support_id,
                        raw_model.geom(support_id).name,
                    )
                )
    return tuple(candidates)


def _normalise_pose(value: Any, nominal_pose: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 7:
        pose = array.copy()
    elif array.size == 6:
        pose = np.concatenate((nominal_pose[:3] + array[:3], rotation_vector_to_quat(array[3:])))
    else:
        raise SceneConfigError("safe envelope poses must contain six relative or seven world-pose values")
    if not np.all(np.isfinite(pose)):
        raise SceneConfigError("safe envelope poses must be finite")
    norm = float(np.linalg.norm(pose[3:]))
    if norm <= 1.0e-12:
        raise SceneConfigError("safe envelope pose quaternion must be non-zero")
    pose[3:] /= norm
    return pose


def _safe_envelope_poses(config: SceneVisualConfig, nominal_pose: np.ndarray, envelope: Any) -> tuple[np.ndarray, ...]:
    if envelope is not None:
        if isinstance(envelope, Mapping):
            values = envelope.get("poses", envelope.get("deck_pose_samples"))
            if values is not None:
                return tuple(_normalise_pose(value, nominal_pose) for value in values)
            translations = envelope.get("translation_abs_m")
            rotations = envelope.get("rotation_abs_rad")
            if translations is not None or rotations is not None:
                source = dict(config.to_dict()["safe_excitation_envelope"])
                source.update(envelope)
                envelope = source
        elif hasattr(envelope, "evaluate"):
            times = np.linspace(
                0.0,
                float(config.section("safe_excitation_envelope")["duration_s"]),
                int(config.section("safe_excitation_envelope")["sample_count"]),
            )
            values = []
            for time_s in times:
                result = envelope.evaluate(float(time_s))
                q = np.asarray(getattr(result, "q", result), dtype=float).reshape(-1)
                values.append(_normalise_pose(q, nominal_pose))
            return tuple(values)
        else:
            return tuple(_normalise_pose(value, nominal_pose) for value in envelope)
    source = config.section("safe_excitation_envelope") if envelope is None else _require_mapping("envelope", envelope)
    translation_abs = np.asarray(source["translation_abs_m"], dtype=float)
    rotation_abs = np.asarray(source["rotation_abs_rad"], dtype=float)
    relative: list[np.ndarray] = [np.zeros(6)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            value = np.zeros(6)
            value[axis] = sign * translation_abs[axis]
            relative.append(value)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            value = np.zeros(6)
            value[3 + axis] = sign * rotation_abs[axis]
            relative.append(value)
    # Sample coupled translation/rotation corners as well as individual axes;
    # finite sampling does not prove clearance over a continuous envelope.
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        relative.append(np.asarray(signs) * np.concatenate((translation_abs, rotation_abs)))
    return tuple(_normalise_pose(value, nominal_pose) for value in relative)


def _support_report(raw_model: Any, raw_data: Any, config: SceneVisualConfig) -> dict[str, Any]:
    visual_names = scene_visual_geom_names(raw_model, config)

    def extrema(prefix: str, *, upper: bool = False) -> float | None:
        values = []
        for name in visual_names:
            if name.startswith(prefix):
                geom_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name)
                lower, upper_bound = _geom_aabb(raw_model, raw_data, geom_id)
                values.append(float(upper_bound[2] if upper else lower[2]))
        return (max(values) if upper else min(values)) if values else None

    # The lower plate is the table support's installation datum on the deck;
    # the moving upper foot is seated on top of the isolator sleeve.
    table_bottom = extrema("shakebench_table_lower_mount_plate_")
    platen_top = extrema("shakebench_platen_surface", upper=True)
    configured_top = float(config.section("platen")["nominal_top_z_m"])
    mount_bottom = support_top = None
    mount_names = []
    support_body = config.section("clearance").get("world_robot_support_body")
    if support_body is not None:
        mount_geoms = []
        for geom_id in range(int(raw_model.ngeom)):
            name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None or not str(name).startswith("fixed_mount"):
                continue
            if not int(raw_model.geom_contype[geom_id]) and not int(raw_model.geom_conaffinity[geom_id]):
                continue
            lower, _ = _geom_aabb(raw_model, raw_data, geom_id)
            mount_geoms.append((float(lower[2]), str(name)))
        if mount_geoms:
            mount_bottom = min(mount_geoms)[0]
        else:
            base_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, support_body)
            mount_bottom = float(raw_data.xpos[base_id, 2])
            mount_geoms = [(mount_bottom, support_body + ":authored_mount_datum")]
        beam_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, "robot_support_foundation")
        _, upper = _geom_aabb(raw_model, raw_data, beam_id)
        support_top = float(upper[2])
        mount_names = [name for _, name in mount_geoms]
    return {
        "method": "compiled table support extrema and world-fixed robot installation datum",
        "platen_nominal_top_z_m": platen_top,
        "configured_platen_nominal_top_z_m": configured_top,
        "derived_platen_nominal_top_z_m": table_bottom,
        "derivation_rule": "compiled worktable lower-mount plate only",
        "derivation_inputs_z_m": {"worktable_foot_lowest_support_z_m": table_bottom},
        "derived_vs_compiled_platen_top_error_m": (
            None if table_bottom is None or platen_top is None else platen_top - table_bottom
        ),
        "worktable_foot_lowest_support_z_m": table_bottom,
        "robot_mount_lowest_support_z_m": mount_bottom,
        "world_robot_support_top_z_m": support_top,
        "worktable_assembly_error_m": None if table_bottom is None or platen_top is None else table_bottom - platen_top,
        "robot_mount_assembly_error_m": None if mount_bottom is None else mount_bottom - support_top,
        "robot_mount_support_geoms": mount_names,
    }


@dataclass(frozen=True)
class ClearanceReport:
    """Signed-distance report for nominal and registered safe poses."""

    passed: bool
    config_sha256: str
    geometry_variant: str
    nominal: Mapping[str, Any]
    safe_envelope: Mapping[str, Any]
    visual_visual_candidates: tuple[Mapping[str, Any], ...]
    visual_collision_candidates: tuple[Mapping[str, Any], ...]
    active_physical_contacts: tuple[Mapping[str, Any], ...]
    support_interfaces: Mapping[str, Any]
    stewart: Mapping[str, Any]
    unexpected_penetration_pairs: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...] = ()

    @property
    def clearance_passed(self) -> bool:
        """Compatibility alias for the scene-clearance gate result."""

        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "clearance_passed": self.passed,
            "config_sha256": self.config_sha256,
            "geometry_variant": self.geometry_variant,
            "nominal": _thaw(self.nominal),
            "safe_envelope": _thaw(self.safe_envelope),
            "visual_visual_candidates": [_thaw(item) for item in self.visual_visual_candidates],
            "visual_collision_candidates": [_thaw(item) for item in self.visual_collision_candidates],
            "active_physical_contacts": [_thaw(item) for item in self.active_physical_contacts],
            "support_interfaces": _thaw(self.support_interfaces),
            "stewart": _thaw(self.stewart),
            "unexpected_penetration_pairs": [_thaw(item) for item in self.unexpected_penetration_pairs],
            "warnings": list(self.warnings),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _distance_record(
    raw_model: Any,
    raw_data: Any,
    category: str,
    geom1: int,
    name1: str,
    geom2: int,
    name2: str,
    sample_index: int,
    pose: np.ndarray,
) -> dict[str, Any]:
    distance, closest = _geom_distance(raw_model, raw_data, geom1, geom2)
    type1 = int(raw_model.geom_type[geom1])
    type2 = int(raw_model.geom_type[geom2])
    method = "mujoco_geom_distance"
    if type1 == int(mujoco.mjtGeom.mjGEOM_MESH) or type2 == int(mujoco.mjtGeom.mjGEOM_MESH):
        method = "mujoco_geom_distance_mesh_convex_candidate"
    if _floor_cylinder_support_distance(raw_model, raw_data, geom1, geom2) is not None:
        method = "analytic_cylinder_support_to_containing_floor_top"
    if any(name.startswith(("robot_support_", "fixed_mount")) for name in (name1, name2)):
        lower1, upper1 = _geom_aabb(raw_model, raw_data, geom1)
        lower2, upper2 = _geom_aabb(raw_model, raw_data, geom2)
        separation = np.maximum(np.maximum(lower1 - upper2, lower2 - upper1), 0.0)
        if np.any(separation > 0):
            # Disjoint enclosing boxes prove a conservative positive gap even
            # when the convex-distance solver returns zero for distant boxes.
            distance = float(np.linalg.norm(separation))
            middle = np.maximum(lower1, lower2)
            closest = np.array([np.minimum(middle, upper1), np.minimum(middle, upper2)])
            method = "conservative_aabb_separation_lower_bound"
    return {
        "category": category,
        "geom1": name1,
        "geom2": name2,
        "body1": _geom_body_name(raw_model, geom1),
        "body2": _geom_body_name(raw_model, geom2),
        "frame1": _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom1])),
        "frame2": _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom2])),
        "signed_distance_m": float(distance),
        "closest_points_m": closest.tolist(),
        "method": method,
        "sample_index": int(sample_index),
        "deck_pose_wxyz_m": pose.tolist(),
        "whitelist_reason": _pair_whitelist(name1, name2, distance),
    }


def _minimum_pair_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["category"], record["geom1"], record["geom2"])
        previous = grouped.get(key)
        if previous is None or record["signed_distance_m"] < previous["signed_distance_m"]:
            grouped[key] = record
    return list(grouped.values())


def _active_contacts(raw_model: Any, raw_data: Any) -> tuple[dict[str, Any], ...]:
    records = []
    for contact_id in range(int(raw_data.ncon)):
        contact = raw_data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        records.append(
            {
                "geom1": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "body1": _geom_body_name(raw_model, geom1),
                "body2": _geom_body_name(raw_model, geom2),
                "distance_m": float(contact.dist),
                "include_in_scene_gate": False,
            }
        )
    return tuple(records)


def _stewart_report(
    raw_model: Any, raw_data: Any, config: SceneVisualConfig, poses: Sequence[np.ndarray]
) -> dict[str, Any]:
    stewart = config.section("stewart")
    platen = config.section("platen")
    nominal_center_z = float(platen["nominal_top_z_m"]) - float(platen["size_m"][2]) / 2
    base_points, nominal_points = _stewart_points(config, nominal_center_z)
    rows = []
    min_overlap = float("inf")
    min_length = float("inf")
    max_length = -float("inf")
    passed = True
    for sample_index, pose in enumerate(poses):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, pose[3:])
        platen_points = nominal_points @ rotation.reshape(3, 3).T + pose[:3]
        lengths = np.linalg.norm(platen_points - base_points, axis=1)
        overlap = float(stewart["outer_length_m"]) + float(stewart["rod_length_m"]) - lengths
        min_overlap = min(min_overlap, float(np.min(overlap)))
        min_length = min(min_length, float(np.min(lengths)))
        max_length = max(max_length, float(np.max(lengths)))
        sample_passed = bool(
            np.all(lengths >= float(stewart["leg_min_m"]))
            and np.all(lengths <= float(stewart["leg_max_m"]))
            and np.all(overlap >= float(config.section("clearance")["minimum_stewart_overlap_m"]))
        )
        passed = passed and sample_passed
        rows.append(
            {
                "sample_index": sample_index,
                "lengths_m": lengths.tolist(),
                "overlap_m": overlap.tolist(),
                "passed": sample_passed,
            }
        )
    return {
        "method": "analytic Stewart ellipse endpoints with registered pose samples",
        "sample_count": len(poses),
        "min_length_m": min_length,
        "max_length_m": max_length,
        "min_segment_overlap_m": min_overlap,
        "leg_limits_m": [float(stewart["leg_min_m"]), float(stewart["leg_max_m"])],
        "minimum_required_overlap_m": float(config.section("clearance")["minimum_stewart_overlap_m"]),
        "passed": passed,
        "samples": rows,
    }


def scene_clearance_report(
    sim: Any,
    config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None,
    envelope: Any = None,
) -> ClearanceReport:
    """Check candidate visual overlap and assembly clearance.

    Primitive pairs use MuJoCo's signed ``mj_geomDistance``.  Mesh-involving
    pairs are labelled as convex-candidate checks; this report intentionally
    records finite sampled coverage and does not claim a continuous proof.
    """

    scene_config = _coerce_config(config)
    raw_model, raw_data = _raw_model_and_data(sim)
    nominal_deck_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, DECK_BODY_NAME)
    nominal_pose = np.concatenate(
        (
            np.asarray(raw_data.xpos[nominal_deck_id], dtype=float),
            np.asarray(raw_data.xquat[nominal_deck_id], dtype=float),
        )
    )
    poses = _safe_envelope_poses(scene_config, nominal_pose, envelope)
    if not poses:
        raise SceneConfigError("safe clearance envelope must contain at least one pose")
    candidates = _candidate_pairs(raw_model, raw_data, scene_config)
    nominal_records: list[dict[str, Any]] = []
    safe_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    # Work in a private MjData so a clearance audit never changes the caller's
    # state, mocap targets, warmstart buffers, or policy-facing history.
    audit_data = mujoco.MjData(raw_model)
    np.copyto(audit_data.qpos, raw_data.qpos)
    np.copyto(audit_data.qvel, raw_data.qvel)
    np.copyto(audit_data.act, raw_data.act)
    deck_joint_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, DECK_FREEJOINT_NAME)
    deck_qpos = int(raw_model.jnt_qposadr[deck_joint_id])
    for sample_index, pose in enumerate(poses):
        audit_data.qpos[deck_qpos : deck_qpos + 7] = pose
        mujoco.mj_forward(raw_model, audit_data)
        for category, geom1, name1, geom2, name2 in candidates:
            if sample_index and all(
                _frame_for_body(raw_model, int(raw_model.geom_bodyid[i])) == "world" for i in (geom1, geom2)
            ):
                continue
            record = _distance_record(raw_model, audit_data, category, geom1, name1, geom2, name2, sample_index, pose)
            key = (category, name1, name2)
            if key not in safe_records or record["signed_distance_m"] < safe_records[key]["signed_distance_m"]:
                safe_records[key] = record
            if sample_index == 0:
                nominal_records.append(record)
    nominal_min = _minimum_pair_records(nominal_records)
    safe_min = list(safe_records.values())
    nominal_map = {(record["category"], record["geom1"], record["geom2"]): record for record in nominal_min}
    safe_map = {(record["category"], record["geom1"], record["geom2"]): record for record in safe_min}
    # Put every candidate in the compact report, retaining both nominal and
    # sampled minima. This lets a negative fixture be diagnosed by name/frame.
    visual_visual = []
    visual_collision = []
    unexpected = []
    tolerance = float(scene_config.section("clearance")["required_margin_m"])
    for key, safe_record in safe_map.items():
        category, name1, name2 = key
        record = dict(safe_record)
        record["nominal_signed_distance_m"] = (
            float(nominal_map[key]["signed_distance_m"]) if key in nominal_map else None
        )
        record["safe_envelope_min_signed_distance_m"] = float(safe_record["signed_distance_m"])
        record["safe_envelope_sample_count"] = len(poses)
        if category == "visual_visual":
            visual_visual.append(record)
        else:
            visual_collision.append(record)
        if record["whitelist_reason"] is None and float(record["safe_envelope_min_signed_distance_m"]) < -tolerance:
            unexpected.append(record)
    support = _support_report(raw_model, raw_data, scene_config)
    support_errors = [
        value
        for key, value in (
            ("worktable", support["worktable_assembly_error_m"]),
            ("robot_mount", support["robot_mount_assembly_error_m"]),
        )
        if value is not None and abs(float(value)) > float(scene_config.section("clearance")["assembly_tolerance_m"])
    ]
    stewart_report = _stewart_report(raw_model, raw_data, scene_config, poses)
    active_contacts = _active_contacts(raw_model, raw_data)
    warnings = []
    if support["robot_mount_lowest_support_z_m"] is None:
        warnings.append("robot mount support proxy was not found; mount clearance is incomplete")
    passed = not unexpected and not support_errors and bool(stewart_report["passed"])
    nominal = {
        "deck_pose_wxyz_m": nominal_pose.tolist(),
        "sample_count": 1,
        "visual_visual_minimums": [
            record for record in visual_visual if record["nominal_signed_distance_m"] is not None
        ],
        "visual_collision_minimums": [
            record for record in visual_collision if record["nominal_signed_distance_m"] is not None
        ],
    }
    safe = {
        "sample_count": len(poses),
        "sampling_method": scene_config.section("safe_excitation_envelope")["sampling_method"],
        "translation_abs_m": list(scene_config.section("safe_excitation_envelope")["translation_abs_m"]),
        "rotation_abs_rad": list(scene_config.section("safe_excitation_envelope")["rotation_abs_rad"]),
        "poses_wxyz_m": [pose.tolist() for pose in poses],
        "visual_visual_minimums": visual_visual,
        "visual_collision_minimums": visual_collision,
    }
    return ClearanceReport(
        passed=passed,
        config_sha256=scene_config.config_sha256,
        geometry_variant=scene_config.geometry_variant,
        nominal=nominal,
        safe_envelope=safe,
        visual_visual_candidates=tuple(visual_visual),
        visual_collision_candidates=tuple(visual_collision),
        active_physical_contacts=active_contacts,
        support_interfaces=support,
        stewart=stewart_report,
        unexpected_penetration_pairs=tuple(unexpected),
        warnings=tuple(warnings),
    )
