"""Task-native Phase 06F contact and horizontal-force fixtures.

The fixtures compile raw MuJoCo models with the canonical Can mesh, mass and
inertia and the exact open-worktable / target-bottom support geometry.  They
do not construct a task, controller, reward, or success evaluator.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_diagnose_normal_impact_v8 import (
    CAN_INERTIA_KG_M2,
    CAN_MASS_KG,
    CAN_PLACEMENT_Z_M,
)


OPEN_TABLE_XY_M = (-0.10, -0.13)
TARGET_BOTTOM_XY_M = (-0.10, 0.17)
OPEN_TABLE_TOP_Z_M = 0.80
TARGET_BOTTOM_TOP_Z_M = 0.812
OPEN_TABLE_HALF_SIZE_M = (0.325, 0.30, 0.03)
TARGET_BOTTOM_HALF_SIZE_M = (0.09, 0.08, 0.006)
SETTLE_DURATION_S = 0.50
FINAL_WINDOW_S = 0.10


class ContactFixtureError(ValueError):
    """Raised for malformed fixture inputs."""


def _fmt(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "candidate_id",
        "condim",
        "sliding_mu",
        "torsional_mu",
        "rolling_mu",
        "margin_m",
        "gap_m",
        "solref",
        "solimp",
        "iterations",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ContactFixtureError("contact candidate is missing: " + ", ".join(missing))
    value = copy.deepcopy(dict(candidate))
    if float(value["solref"][0]) < 0.0004:
        raise ContactFixtureError("contact solref time constant must be >= 2*dt at dt=0.0002")
    return value


def _surface_spec(support: str) -> tuple[tuple[float, float], float, tuple[float, float, float], str, str]:
    if support == "open_worktable":
        return OPEN_TABLE_XY_M, OPEN_TABLE_TOP_Z_M, OPEN_TABLE_HALF_SIZE_M, "table_collision", "can_open_worktable"
    if support == "target_bottom":
        return TARGET_BOTTOM_XY_M, TARGET_BOTTOM_TOP_Z_M, TARGET_BOTTOM_HALF_SIZE_M, "target_container_bottom", "can_target_bottom"
    raise ContactFixtureError(f"unknown task-native support: {support}")


def raw_contact_xml(candidate: Mapping[str, Any], *, support: str, timestep_s: float = 0.0002) -> str:
    """Build one exact task-native support fixture."""

    item = _candidate(candidate)
    xy, top_z, half_size, geom_name, _ = _surface_spec(support)
    center_z = top_z - half_size[2]
    can_z = top_z + CAN_PLACEMENT_Z_M
    mesh_path = (Path(models.assets_root) / "objects" / "meshes" / "can.msh").resolve()
    sliding = float(item["sliding_mu"]["table_object"])
    # MuJoCo pair friction is [tangent1, tangent2, torsion, roll1, roll2].
    # Both tangential directions implement the registered isotropic sliding
    # coefficient; this is intentionally more explicit than geom's 3-vector.
    friction = (
        sliding,
        sliding,
        float(item["torsional_mu"]),
        float(item["rolling_mu"]),
        float(item["rolling_mu"]),
    )
    return f'''<mujoco model="shakebench_phase06f_task_native_contact">
  <compiler angle="radian" inertiafromgeom="false" autolimits="true" />
  <option timestep="{float(timestep_s):.17g}" integrator="Euler" solver="Newton" iterations="{int(item['iterations'])}" tolerance="1e-12" gravity="0 0 -9.81" />
  <asset><mesh name="can_mesh" file="{mesh_path}" /></asset>
  <worldbody>
    <body name="support" pos="{xy[0]:.17g} {xy[1]:.17g} {center_z:.17g}">
      <geom name="{geom_name}" type="box" size="{_fmt(half_size)}" />
    </body>
    <body name="can_main" pos="{xy[0]:.17g} {xy[1]:.17g} {can_z:.17g}">
      <freejoint name="can_joint0" />
      <inertial pos="0 0 0" mass="{CAN_MASS_KG:.17g}" diaginertia="{_fmt(CAN_INERTIA_KG_M2)}" />
      <geom name="can_g0" type="mesh" mesh="can_mesh" contype="2" conaffinity="0" />
    </body>
  </worldbody>
  <contact>
    <pair geom1="can_g0" geom2="{geom_name}" friction="{_fmt(friction)}" condim="{int(item['condim'])}" margin="{float(item['margin_m']):.17g}" gap="{float(item['gap_m']):.17g}" solref="{_fmt(item['solref'])}" solimp="{_fmt(item['solimp'])}" />
  </contact>
</mujoco>'''


def _handles(model: Any, support: str) -> tuple[int, int, int, int, str]:
    _, _, _, geom_name, interface = _surface_spec(support)
    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can_main"))
    joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_joint0"))
    can_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "can_g0"))
    support_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    if min(body, joint, can_geom, support_geom) < 0:
        raise ContactFixtureError("compiled fixture is missing a named body/joint/geom")
    return body, joint, can_geom, support_geom, interface


def compiled_pair_audit(candidate: Mapping[str, Any], *, support: str = "open_worktable") -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(raw_contact_xml(candidate, support=support))
    body, _, can_geom, support_geom, interface = _handles(model, support)
    matches = [
        index
        for index in range(int(model.npair))
        if {int(model.pair_geom1[index]), int(model.pair_geom2[index])} == {can_geom, support_geom}
    ]
    if len(matches) != 1:
        raise ContactFixtureError("compiled explicit pair scope is incomplete or duplicated")
    index = matches[0]
    return {
        "interface": interface,
        "pair_count": int(model.npair),
        "matching_pair_count": 1,
        "condim": int(model.pair_dim[index]),
        "friction": np.asarray(model.pair_friction[index], dtype=float).tolist(),
        "margin_m": float(model.pair_margin[index]),
        "gap_m": float(model.pair_gap[index]),
        "solref": np.asarray(model.pair_solref[index], dtype=float).tolist(),
        "solimp": np.asarray(model.pair_solimp[index], dtype=float).tolist(),
        "iterations": int(model.opt.iterations),
        "can_mass_kg": float(model.body_mass[body]),
        "can_inertia_kg_m2": np.asarray(model.body_inertia[body], dtype=float).tolist(),
    }


def _contact_sample(model: Any, data: Any, *, can_geom: int, support_geom: int, interface: str) -> dict[str, Any]:
    contacts = []
    normal_force = 0.0
    horizontal_force = np.zeros(2, dtype=float)
    penetration = 0.0
    finite = True
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} != {can_geom, support_geom}:
            continue
        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, wrench)
        finite = bool(finite and np.all(np.isfinite(wrench)) and np.isfinite(float(contact.dist)))
        normal_force += abs(float(wrench[0]))
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        world_force = frame.T.dot(wrench[:3])
        horizontal_force += world_force[:2]
        penetration = max(penetration, max(0.0, -float(contact.dist)))
        contacts.append(
            {
                "interface": interface,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                "distance_m": float(contact.dist),
                "normal_force_N": abs(float(wrench[0])),
            }
        )
    return {
        "contact_count": len(contacts),
        "named_support_contact": bool(contacts),
        "normal_force_N": normal_force,
        "horizontal_force_N": horizontal_force.tolist(),
        "penetration_m": penetration,
        "finite_contact_forces": finite,
        "contacts": contacts,
    }


def _warnings(data: Any) -> int:
    return int(np.sum(np.asarray(data.warning.number, dtype=int)))


def run_task_native_settling(
    candidate: Mapping[str, Any], *, support: str, timestep_s: float = 0.0002, duration_s: float = SETTLE_DURATION_S
) -> dict[str, Any]:
    """Settle from the exact task support pose and report the final window."""

    model = mujoco.MjModel.from_xml_string(raw_contact_xml(candidate, support=support, timestep_s=timestep_s))
    data = mujoco.MjData(model)
    body, joint, can_geom, support_geom, interface = _handles(model, support)
    qvel_address = int(model.jnt_dofadr[joint])
    mujoco.mj_forward(model, data)
    steps = int(round(float(duration_s) / float(timestep_s)))
    final_start = float(duration_s) - FINAL_WINDOW_S
    final_rows = []
    maximum_penetration = 0.0
    forces_finite = True
    for index in range(steps + 1):
        sample = _contact_sample(model, data, can_geom=can_geom, support_geom=support_geom, interface=interface)
        maximum_penetration = max(maximum_penetration, float(sample["penetration_m"]))
        forces_finite = bool(forces_finite and sample["finite_contact_forces"])
        if float(data.time) >= final_start - 1.0e-12:
            velocity = np.asarray(data.qvel[qvel_address : qvel_address + 6], dtype=float)
            final_rows.append(
                {
                    "time_s": float(data.time),
                    "linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
                    "angular_speed_rad_s": float(np.linalg.norm(velocity[3:])),
                    "contact": sample,
                }
            )
        if index < steps:
            data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
    warning_count = _warnings(data)
    return {
        "support": support,
        "interface": interface,
        "gamma": 0.0,
        "action": "zero",
        "task_native_xy_m": list(_surface_spec(support)[0]),
        "settle_duration_s": float(duration_s),
        "final_window_s": FINAL_WINDOW_S,
        "final_sample_count": len(final_rows),
        "final_window_named_support_continuous": bool(final_rows and all(row["contact"]["named_support_contact"] for row in final_rows)),
        "final_window_max_linear_speed_m_s": max(row["linear_speed_m_s"] for row in final_rows),
        "final_window_max_angular_speed_rad_s": max(row["angular_speed_rad_s"] for row in final_rows),
        "final_window_min_support_force_N": min(row["contact"]["normal_force_N"] for row in final_rows),
        "final_window_mean_support_force_N": float(np.mean([row["contact"]["normal_force_N"] for row in final_rows])),
        "maximum_penetration_m": maximum_penetration,
        "finite_contact_forces": forces_finite,
        "finite_state": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "warning_count": warning_count,
        "terminal_position_m": np.asarray(data.xpos[body], dtype=float).tolist(),
        "terminal_qpos": np.asarray(data.qpos, dtype=float).tolist(),
        "final_window": final_rows,
    }


def run_horizontal_force_threshold(
    candidate: Mapping[str, Any],
    *,
    force_factors: Sequence[float],
    reference_mu: float,
    mass_kg: float,
    gravity_m_s2: float,
    plateau_duration_s: float,
    static_displacement_max_m: float,
    monotonic_tolerance_m: float,
    bracket_relative_tolerance: float,
) -> dict[str, Any]:
    """Measure the level force transition, resetting to one settled state."""

    item = _candidate(candidate)
    levels = tuple(float(value) for value in force_factors)
    required = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5)
    if any(value not in levels for value in required) or tuple(sorted(set(levels))) != levels:
        raise ContactFixtureError("force factors must be unique, sorted, and include 0/0.5/0.8/1.0/1.2/1.5")
    reference_force = float(reference_mu) * float(mass_kg) * float(gravity_m_s2)
    model = mujoco.MjModel.from_xml_string(raw_contact_xml(item, support="open_worktable"))
    data = mujoco.MjData(model)
    body, joint, can_geom, support_geom, interface = _handles(model, "open_worktable")
    qvel_address = int(model.jnt_dofadr[joint])
    mujoco.mj_forward(model, data)
    settle_steps = int(round(SETTLE_DURATION_S / float(model.opt.timestep)))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    settled_qpos = np.asarray(data.qpos, dtype=float).copy()
    settled_position = np.asarray(data.xpos[body], dtype=float).copy()
    records = []
    plateau_steps = int(round(float(plateau_duration_s) / float(model.opt.timestep)))
    for factor in levels:
        data.qpos[:] = settled_qpos
        data.qvel[:] = 0.0
        data.time = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        rows = []
        for index in range(plateau_steps + 1):
            contact = _contact_sample(model, data, can_geom=can_geom, support_geom=support_geom, interface=interface)
            position = np.asarray(data.xpos[body], dtype=float)
            velocity = np.asarray(data.qvel[qvel_address : qvel_address + 6], dtype=float)
            rows.append(
                {
                    "time_s": float(data.time),
                    "along_displacement_m": float(position[0] - settled_position[0]),
                    "along_velocity_m_s": float(velocity[0]),
                    "transverse_drift_m": float(position[1] - settled_position[1]),
                    "support_force_N": float(contact["normal_force_N"]),
                    "penetration_m": float(contact["penetration_m"]),
                    "contact_identity": [row["interface"] for row in contact["contacts"]],
                    "warning_count": _warnings(data),
                }
            )
            if index < plateau_steps:
                data.xfrc_applied[body, :3] = (factor * reference_force, 0.0, 0.0)
                mujoco.mj_step(model, data)
        displacements = np.asarray([row["along_displacement_m"] for row in rows], dtype=float)
        diffs = np.diff(displacements)
        maximum_displacement = float(np.max(np.abs(displacements)))
        moved = bool(maximum_displacement > static_displacement_max_m)
        records.append(
            {
                "force_factor": factor,
                "applied_force_N": factor * reference_force,
                "maximum_along_displacement_m": maximum_displacement,
                "terminal_along_displacement_m": float(displacements[-1]),
                "maximum_abs_along_velocity_m_s": max(abs(row["along_velocity_m_s"]) for row in rows),
                "maximum_abs_transverse_drift_m": max(abs(row["transverse_drift_m"]) for row in rows),
                "minimum_support_force_N": min(row["support_force_N"] for row in rows),
                "maximum_penetration_m": max(row["penetration_m"] for row in rows),
                "contact_identity_valid": all(
                    not row["contact_identity"] or set(row["contact_identity"]) == {interface} for row in rows
                ),
                "warning_count": max(row["warning_count"] for row in rows),
                "moved": moved,
                "monotonic_along_motion": bool(moved and np.all(diffs >= -float(monotonic_tolerance_m))),
                "trace": rows,
            }
        )
    stationary = [row for row in records if not row["moved"]]
    moving = [row for row in records if row["moved"]]
    lower = max((row["applied_force_N"] for row in stationary), default=None)
    upper = min((row["applied_force_N"] for row in moving), default=None)
    bracketed = bool(
        lower is not None
        and upper is not None
        and lower <= reference_force <= upper
        and (upper - lower) / reference_force <= float(bracket_relative_tolerance)
    )
    return {
        "fixture": "raw_mujoco_task_geometry_level_horizontal_body_com_force",
        "support": "open_worktable",
        "reset_to_identical_settled_state_each_plateau": True,
        "settle_duration_s": SETTLE_DURATION_S,
        "plateau_duration_s": float(plateau_duration_s),
        "protocol_reference": {"mu": float(reference_mu), "mass_kg": float(mass_kg), "gravity_m_s2": float(gravity_m_s2), "F_mu_N": reference_force},
        "compiled_sliding_mu": float(item["sliding_mu"]["table_object"]),
        "force_factors": list(levels),
        "static_displacement_max_m": float(static_displacement_max_m),
        "monotonic_tolerance_m": float(monotonic_tolerance_m),
        "bracket_relative_tolerance": float(bracket_relative_tolerance),
        "transition": {"lower_stationary_force_N": lower, "upper_moving_force_N": upper, "brackets_F_mu": bracketed},
        "records": records,
    }


def force_threshold_gates(result: Mapping[str, Any], *, penetration_max_m: float, transverse_drift_max_m: float) -> dict[str, bool]:
    records = list(result.get("records", ()))
    reference = float(result.get("protocol_reference", {}).get("F_mu_N", math.nan))
    zero = next((row for row in records if row.get("force_factor") == 0.0), {})
    subthreshold = [row for row in records if float(row.get("force_factor", math.inf)) < 1.0]
    above = [row for row in records if float(row.get("force_factor", -math.inf)) > 1.0]
    static_limit = float(result.get("static_displacement_max_m", math.nan))
    return {
        "finite_reference": bool(np.isfinite(reference) and reference > 0.0),
        "zero_force_supported_stationary": bool(
            zero
            and zero.get("contact_identity_valid") is True
            and float(zero.get("maximum_along_displacement_m", math.inf)) <= static_limit
            and float(zero.get("minimum_support_force_N", 0.0)) > 0.0
        ),
        "subthreshold_static": bool(
            subthreshold
            and all(float(row.get("maximum_along_displacement_m", math.inf)) <= static_limit for row in subthreshold)
        ),
        "above_threshold_monotonic_motion": any(row.get("monotonic_along_motion") is True for row in above),
        "transition_brackets_reference": result.get("transition", {}).get("brackets_F_mu") is True,
        "transverse_stable": bool(records and all(float(row.get("maximum_abs_transverse_drift_m", math.inf)) <= transverse_drift_max_m for row in records)),
        "penetration_legal": bool(records and all(float(row.get("maximum_penetration_m", math.inf)) <= penetration_max_m for row in records)),
        "contacts_named": bool(records and all(row.get("contact_identity_valid") is True for row in records)),
        "warnings_clear": bool(records and all(int(row.get("warning_count", 1)) == 0 for row in records)),
    }


def run_single_axis_slip(
    candidate: Mapping[str, Any], *, initial_speed_m_s: float = 0.05, duration_s: float = 0.50
) -> dict[str, Any]:
    """Apply the registered level single-axis initial-velocity sanity probe."""

    item = _candidate(candidate)
    model = mujoco.MjModel.from_xml_string(raw_contact_xml(item, support="open_worktable"))
    data = mujoco.MjData(model)
    body, joint, can_geom, support_geom, interface = _handles(model, "open_worktable")
    qvel_address = int(model.jnt_dofadr[joint])
    mujoco.mj_forward(model, data)
    for _ in range(int(round(SETTLE_DURATION_S / float(model.opt.timestep)))):
        mujoco.mj_step(model, data)
    data.qvel[qvel_address : qvel_address + 6] = 0.0
    data.qvel[qvel_address] = float(initial_speed_m_s)
    start = np.asarray(data.xpos[body], dtype=float).copy()
    mujoco.mj_forward(model, data)
    maximum_penetration = 0.0
    named = True
    for _ in range(int(round(float(duration_s) / float(model.opt.timestep)))):
        mujoco.mj_step(model, data)
        contact = _contact_sample(model, data, can_geom=can_geom, support_geom=support_geom, interface=interface)
        maximum_penetration = max(maximum_penetration, float(contact["penetration_m"]))
        named = bool(named and all(row["interface"] == interface for row in contact["contacts"]))
    velocity = np.asarray(data.qvel[qvel_address : qvel_address + 6], dtype=float)
    displacement = np.asarray(data.xpos[body], dtype=float) - start
    return {
        "initial_speed_m_s": float(initial_speed_m_s),
        "duration_s": float(duration_s),
        "final_speed_m_s": float(np.linalg.norm(velocity[:3])),
        "along_displacement_m": float(displacement[0]),
        "transverse_drift_m": float(displacement[1]),
        "maximum_penetration_m": maximum_penetration,
        "contact_identity_valid": named,
        "finite_state": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "warning_count": _warnings(data),
    }


def run_timestep_convergence(candidate: Mapping[str, Any], *, timesteps_s: Sequence[float]) -> dict[str, Any]:
    """Compare task-native support forces at all registered timesteps."""

    records = []
    for timestep in timesteps_s:
        result = run_task_native_settling(candidate, support="open_worktable", timestep_s=float(timestep))
        records.append(
            {
                "timestep_s": float(timestep),
                "mean_support_force_N": result["final_window_mean_support_force_N"],
                "maximum_penetration_m": result["maximum_penetration_m"],
                "warning_count": result["warning_count"],
                "final_window_named_support_continuous": result["final_window_named_support_continuous"],
            }
        )
    reference = records[0]["mean_support_force_N"]
    for row in records:
        row["relative_force_error"] = abs(row["mean_support_force_N"] - reference) / max(abs(reference), 1.0e-12)
    return {
        "timesteps_s": [float(value) for value in timesteps_s],
        "reference_timestep_s": records[0]["timestep_s"],
        "records": records,
        "maximum_relative_force_error": max(row["relative_force_error"] for row in records),
    }


__all__ = [
    "ContactFixtureError",
    "FINAL_WINDOW_S",
    "OPEN_TABLE_XY_M",
    "SETTLE_DURATION_S",
    "TARGET_BOTTOM_XY_M",
    "compiled_pair_audit",
    "force_threshold_gates",
    "raw_contact_xml",
    "run_horizontal_force_threshold",
    "run_single_axis_slip",
    "run_task_native_settling",
    "run_timestep_convergence",
]
