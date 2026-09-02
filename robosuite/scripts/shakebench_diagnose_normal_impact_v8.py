"""Non-scoreable Phase 06R7/V8 normal-impact and incline-fixture diagnostic.

This module compiles a small raw MuJoCo model containing the exact committed
Can collision mesh and ShakeBench table collision geometry.  It deliberately
does not instantiate a robosuite task environment and never calls reward,
success, metrics, controller, or policy-observation code.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_select_physics_v7 import (
    _contact_candidate_mapping,
    _resolve_v7,
    load_v7_protocol,
    validate_v7_protocol,
)
from robosuite.utils.shakebench_artifacts import file_sha256, payload_hash, write_json_atomic


SCHEMA_ID = "shakebench.phase06r7.v8.normal_impact_diagnostic"
SCHEMA_VERSION = 1
OUTPUT_FILENAME = "shakebench_phase_06r7_v8_normal_impact_diagnostic.json"
V7_STATUS_FILENAME = "shakebench_phase_06r6_v7_status.json"
V7_SELECTED_FILENAME = "shakebench_phase_06r6_v7_selected_candidates.json"
V7_RAW_PREFIX = "shakebench_phase_06r6_v7_raw_contact_"
RECOVERY_DURATION_S = 0.50
TAIL_WINDOW_S = 0.05
VELOCITY_MAX_M_S = 0.02
ANGULAR_VELOCITY_MAX_RAD_S = 0.20
PENETRATION_MAX_M = 0.0005
LEVEL_SETTLE_DURATION_S = 0.50
LEVEL_HOLD_DURATION_S = 0.10
INCLINE_HORIZON_S = 0.25
INCLINE_BINARY_ITERATIONS = 5
INCLINE_RESOLUTION_RAD = 0.01
INCLINE_SLIP_DISPLACEMENT_M = 0.0005
CAN_MASS_KG = 0.349
CAN_INERTIA_KG_M2 = (0.00024106572568945823, 0.00024106572568945823, 0.00010986473347417683)
CAN_PLACEMENT_Z_M = 0.040297003330440104
TABLE_BODY_POS_M = (0.0, 0.0, 0.77)
TABLE_HALF_SIZE_M = (0.325, 0.30, 0.03)
TABLE_MASS_KG = 32.0
TABLE_INERTIA_KG_M2 = (0.9696, 1.1363, 2.0867)
DECK_MASS_KG = 400.0
DECK_INERTIA_KG_M2 = (12.12, 14.083333333333334, 26.033333333333332)
DECK_EQ_SOLREF = (0.0004, 0.5)
DECK_EQ_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
ISOLATOR_K = (20212.949813431005, 20212.949813431005, 20212.949813431005, 344.5044633826647, 403.7339333144822, 329.5184560600506)
ISOLATOR_C = (321.6990877275948, 321.6990877275948, 321.6990877275948, 7.310611768609593, 8.567500157457797, 10.488898224393315)
ISOLATOR_SPRINGREF = (0.0, 0.0, 0.015530637680177088, 0.0, 0.0, 0.0)
ISOLATOR_LIMITS = (0.025, 0.025, 0.025, 0.08726646259971647, 0.08726646259971647, 0.08726646259971647)


def _fmt(values: Sequence[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _quat_mul(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = (float(value) for value in first)
    w2, x2, y2, z2 = (float(value) for value in second)
    return np.asarray((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dtype=float)


def _quat_inverse(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=float)
    return np.asarray((value[0], -value[1], -value[2], -value[3]), dtype=float) / float(np.dot(value, value))


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)), dtype=float)


def _raw_xml(candidate: Mapping[str, Any], *, fixed_table: bool = False, table_angle_rad: float = 0.0, can_pos_m: Sequence[float] = (0.0, 0.0, 0.95), can_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> str:
    mesh_path = (Path(models.assets_root) / "objects" / "meshes" / "can.msh").resolve()
    friction_table = (0.30, float(candidate["torsional_mu"]), float(candidate["rolling_mu"]), float(candidate["rolling_mu"]), float(candidate["rolling_mu"]))
    friction = _fmt(friction_table)
    solref = _fmt(candidate["solref"])
    solimp = _fmt(candidate["solimp"])
    if fixed_table:
        table = f'<body name="worktable" pos="{_fmt(TABLE_BODY_POS_M)}" euler="0 {format(float(table_angle_rad), ".17g")} 0"><inertial pos="0 0 0" mass="{TABLE_MASS_KG:.17g}" diaginertia="{_fmt(TABLE_INERTIA_KG_M2)}"/><geom name="table_collision" type="box" size="{_fmt(TABLE_HALF_SIZE_M)}" pos="0 0 0"/></body>'
        deck = ""
    else:
        joints = []
        axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        types = ("slide", "slide", "slide", "hinge", "hinge", "hinge")
        for index, (axis, joint_type, stiffness, damping, springref, limit) in enumerate(zip(axes, types, ISOLATOR_K, ISOLATOR_C, ISOLATOR_SPRINGREF, ISOLATOR_LIMITS)):
            joints.append(f'<joint name="isolator_{index}" type="{joint_type}" axis="{_fmt(axis)}" stiffness="{stiffness:.17g}" damping="{damping:.17g}" springref="{springref:.17g}" limited="true" range="{-limit:.17g} {limit:.17g}"/>')
        table = f'<body name="worktable" pos="{_fmt(TABLE_BODY_POS_M)}">{"".join(joints)}<inertial pos="0 0 0" mass="{TABLE_MASS_KG:.17g}" diaginertia="{_fmt(TABLE_INERTIA_KG_M2)}"/><geom name="table_collision" type="box" size="{_fmt(TABLE_HALF_SIZE_M)}" pos="0 0 0"/></body>'
        deck = f'<body name="deck_driver" mocap="true"><site name="deck_driver_site"/></body><body name="deck" pos="0 0 0"><freejoint name="deck_freejoint"/><inertial pos="0 0 0" mass="{DECK_MASS_KG:.17g}" diaginertia="{_fmt(DECK_INERTIA_KG_M2)}"/><site name="deck_site"/>{table}</body>'
    equality = f'<equality><weld name="deck_weld" body1="deck_driver" body2="deck" solref="{_fmt(DECK_EQ_SOLREF)}" solimp="{_fmt(DECK_EQ_SOLIMP)}"/></equality>' if not fixed_table else ""
    world_table = table if fixed_table else deck
    return f'''<mujoco model="shakebench_v8_raw_contact">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true" />
  <option timestep="0.0002" integrator="Euler" solver="Newton" iterations="100" tolerance="1e-12" gravity="0 0 -9.81" />
  <asset><mesh name="can_mesh" file="{mesh_path}" /></asset>
  <worldbody>
    {world_table}
    <body name="can_main" pos="{_fmt(can_pos_m)}" quat="{_fmt(can_quat_wxyz)}">
      <freejoint name="can_joint0" />
      <inertial pos="0 0 0" mass="{CAN_MASS_KG:.17g}" diaginertia="{_fmt(CAN_INERTIA_KG_M2)}" />
      <geom name="can_g0" type="mesh" mesh="can_mesh" />
    </body>
  </worldbody>
  {equality}
  <contact><pair geom1="table_collision" geom2="can_g0" friction="{friction}" condim="{int(candidate['condim'])}" margin="{float(candidate['margin_m']):.17g}" gap="{float(candidate['gap_m']):.17g}" solref="{solref}" solimp="{solimp}" /></contact>
</mujoco>'''


def _ids(model: Any) -> tuple[int, int, int, int]:
    can_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can_main"))
    can_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_joint0"))
    table_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worktable"))
    table_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision"))
    if min(can_body, can_joint, table_body, table_geom) < 0:
        raise RuntimeError("raw V8 fixture is missing a named Can/table handle")
    return can_body, can_joint, table_body, table_geom


def _contact_rows(model: Any, data: Any, *, table_geom: int, cumulative_impulse: float, dt: float) -> tuple[dict[str, Any], float]:
    count = 0
    normal_force = 0.0
    tangential_force = 0.0
    minimum_distance = math.inf
    penetration = 0.0
    contacts = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if int(contact.geom1) != table_geom and int(contact.geom2) != table_geom:
            continue
        count += 1
        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = abs(float(wrench[0]))
        tangent = float(np.linalg.norm(wrench[1:3]))
        normal_force += normal
        tangential_force += tangent
        minimum_distance = min(minimum_distance, float(contact.dist))
        penetration = max(penetration, max(0.0, -float(contact.dist)))
        cumulative_impulse += normal * dt
        contacts.append({"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)), "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)), "interface": "can_open_worktable", "distance_m": float(contact.dist), "normal_force_N": normal, "tangential_force_N": tangent})
    return {"contact_count": count, "active": count > 0, "normal_force_N": normal_force, "tangential_force_N": tangential_force, "normal_impulse_Ns": cumulative_impulse, "minimum_distance_m": 0.0 if minimum_distance == math.inf else minimum_distance, "penetration_m": penetration, "contacts": contacts}, cumulative_impulse


def _energy(model: Any, data: Any, *, can_body: int, can_qvel: np.ndarray, table_height: float) -> dict[str, float]:
    mass = float(model.body_mass[can_body])
    inertia = np.asarray(model.body_inertia[can_body], dtype=float)
    kinetic_trans = 0.5 * mass * float(np.dot(can_qvel[:3], can_qvel[:3]))
    kinetic_rot = 0.5 * float(np.dot(inertia * can_qvel[3:6], can_qvel[3:6]))
    potential = mass * 9.81 * max(0.0, float(data.xpos[can_body][2]) - table_height)
    return {"kinetic_translational_J": kinetic_trans, "kinetic_rotational_J": kinetic_rot, "potential_J": potential, "mechanical_energy_J": kinetic_trans + kinetic_rot + potential}


def _normal_trace(model: Any, data: Any, *, duration_s: float, start_from_drop: bool = True) -> dict[str, Any]:
    can_body, can_joint, table_body, table_geom = _ids(model)
    can_qpos = int(model.jnt_qposadr[can_joint])
    can_qvel = int(model.jnt_dofadr[can_joint])
    if start_from_drop:
        data.qpos[can_qpos : can_qpos + 3] = (0.0, 0.0, 0.95)
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[can_qvel : can_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    steps = int(math.ceil(duration_s / dt))
    table_height = float(data.xpos[table_body][2]) + TABLE_HALF_SIZE_M[2]
    trace = []
    cumulative_impulse = 0.0
    states = []
    for index in range(steps + 1):
        qvel = np.asarray(data.qvel[can_qvel : can_qvel + 6], dtype=float).copy()
        contacts, cumulative_impulse = _contact_rows(model, data, table_geom=table_geom, cumulative_impulse=cumulative_impulse, dt=dt)
        state = bool(contacts["active"])
        states.append(state)
        trace.append({"time_s": float(data.time), "can_position_m": np.asarray(data.xpos[can_body], dtype=float).tolist(), "can_orientation_wxyz": np.asarray(data.xquat[can_body], dtype=float).tolist(), "linear_velocity_m_s": qvel[:3].tolist(), "angular_velocity_rad_s": qvel[3:6].tolist(), "vertical_velocity_m_s": float(qvel[2]), "linear_speed_m_s": float(np.linalg.norm(qvel[:3])), "angular_speed_rad_s": float(np.linalg.norm(qvel[3:6])), "contacts": contacts, "energy": _energy(model, data, can_body=can_body, can_qvel=qvel, table_height=table_height), "warning_number": np.asarray(data.warning.number, dtype=int).tolist()})
        if index < steps:
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
    transitions = []
    previous = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state != previous:
            transitions.append({"time_s": trace[index]["time_s"], "event": "contact_reentry" if state else "contact_loss", "contact_before": previous, "contact_after": state})
            previous = state
    loss_intervals = []
    loss_start = None
    for row, state in zip(trace, states):
        if not state and loss_start is None:
            loss_start = float(row["time_s"])
        if state and loss_start is not None:
            loss_intervals.append({"start_time_s": loss_start, "end_time_s": float(row["time_s"]), "duration_s": float(row["time_s"]) - loss_start})
            loss_start = None
    if loss_start is not None:
        loss_intervals.append({"start_time_s": loss_start, "end_time_s": float(trace[-1]["time_s"]), "duration_s": float(trace[-1]["time_s"]) - loss_start})
    vz = np.asarray([float(row["vertical_velocity_m_s"]) for row in trace], dtype=float)
    speeds = np.asarray([float(row["linear_speed_m_s"]) for row in trace], dtype=float)
    angular = np.asarray([float(row["angular_speed_rad_s"]) for row in trace], dtype=float)
    tail_start = duration_s - TAIL_WINDOW_S
    tail = np.asarray([index for index, row in enumerate(trace) if float(row["time_s"]) >= tail_start - 1.0e-12], dtype=int)
    terminal = trace[-1]
    impact_indices = [index for index, row in enumerate(trace) if row["contacts"]["active"]]
    impact_energies = []
    for index in impact_indices:
        if index == 0 or not trace[index - 1]["contacts"]["active"]:
            impact_energies.append({"time_s": trace[index]["time_s"], "mechanical_energy_J": trace[index]["energy"]["mechanical_energy_J"], "energy_fraction_of_release": trace[index]["energy"]["mechanical_energy_J"] / max(trace[0]["energy"]["mechanical_energy_J"], 1.0e-12)})
    return {
        "release_time_s": float(trace[0]["time_s"]),
        "first_impact_time_s": next((float(row["time_s"]) for row in trace if row["contacts"]["active"]), None),
        "duration_s": float(trace[-1]["time_s"]),
        "trace_dt_s": dt,
        "trace_sample_count": len(trace),
        "trace": trace,
        "contact_transitions": transitions,
        "contact_loss_intervals": loss_intervals,
        "terminal_contact_state": {"named_table_contact": bool(terminal["contacts"]["active"]), "normal_force_N": float(terminal["contacts"]["normal_force_N"]), "contact_count": int(terminal["contacts"]["contact_count"])},
        "terminal_linear_velocity_m_s": float(terminal["linear_speed_m_s"]),
        "terminal_angular_velocity_rad_s": float(terminal["angular_speed_rad_s"]),
        "terminal_vertical_velocity_m_s": float(terminal["vertical_velocity_m_s"]),
        "terminal_direction": [float(value) for value in terminal["linear_velocity_m_s"]],
        "tail_window_s": TAIL_WINDOW_S,
        "tail_window_max_linear_velocity_m_s": float(np.max(speeds[tail])),
        "tail_window_rms_linear_velocity_m_s": float(np.sqrt(np.mean(speeds[tail] ** 2))),
        "tail_window_max_angular_velocity_rad_s": float(np.max(angular[tail])),
        "tail_window_rms_angular_velocity_rad_s": float(np.sqrt(np.mean(angular[tail] ** 2))),
        "peak_downward_vertical_speed_m_s": float(max(0.0, -np.min(vz))),
        "peak_rebound_vertical_speed_m_s": float(max(0.0, np.max(vz))),
        "maximum_penetration_m": float(max(row["contacts"]["penetration_m"] for row in trace)),
        "normal_energy_at_impact": impact_energies,
        "warning_count": int(np.sum(np.asarray(trace[-1]["warning_number"], dtype=int))),
    }


def run_normal_impact_trace(candidate: Mapping[str, Any], *, duration_s: float = RECOVERY_DURATION_S) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(_raw_xml(candidate))
    data = mujoco.MjData(model)
    result = _normal_trace(model, data, duration_s=duration_s)
    result["fixture"] = {"can_mass_kg": CAN_MASS_KG, "can_inertia_kg_m2": list(CAN_INERTIA_KG_M2), "table_mass_kg": TABLE_MASS_KG, "table_inertia_kg_m2": list(TABLE_INERTIA_KG_M2), "table_collision_size_m": list(TABLE_HALF_SIZE_M), "table_top_z_m": 0.80, "drop_height_m": 0.15, "can_release_position_m": [0.0, 0.0, 0.95], "model_timestep_s": float(model.opt.timestep), "candidate": copy.deepcopy(dict(candidate))}
    return result


def run_level_control(candidate: Mapping[str, Any]) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(_raw_xml(candidate))
    data = mujoco.MjData(model)
    can_body, can_joint, table_body, table_geom = _ids(model)
    can_qpos = int(model.jnt_qposadr[can_joint])
    can_qvel = int(model.jnt_dofadr[can_joint])
    settled_trace = _normal_trace(model, data, duration_s=LEVEL_SETTLE_DURATION_S)
    settled_qpos = np.asarray(data.qpos, dtype=float).copy()
    settled_can_pos = np.asarray(data.xpos[can_body], dtype=float).copy()
    settled_can_quat = np.asarray(data.xquat[can_body], dtype=float).copy()
    settled_table_pos = np.asarray(data.xpos[table_body], dtype=float).copy()
    settled_table_quat = np.asarray(data.xquat[table_body], dtype=float).copy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    hold_rows = []
    hold_steps = int(math.ceil(LEVEL_HOLD_DURATION_S / float(model.opt.timestep)))
    for index in range(hold_steps + 1):
        contacts, _ = _contact_rows(model, data, table_geom=table_geom, cumulative_impulse=0.0, dt=float(model.opt.timestep))
        qvel = np.asarray(data.qvel[can_qvel : can_qvel + 6], dtype=float)
        hold_rows.append({"time_s": float(index * model.opt.timestep), "can_position_m": np.asarray(data.xpos[can_body], dtype=float).tolist(), "linear_speed_m_s": float(np.linalg.norm(qvel[:3])), "horizontal_speed_m_s": float(np.linalg.norm(qvel[:2])), "horizontal_displacement_m": float(np.linalg.norm(np.asarray(data.xpos[can_body], dtype=float)[:2] - settled_can_pos[:2])), "contacts": contacts, "warning_number": np.asarray(data.warning.number, dtype=int).tolist()})
        if index < hold_steps:
            mujoco.mj_step(model, data)
    return {"fixture": "level_zero_angle_support_control", "settle_duration_s": LEVEL_SETTLE_DURATION_S, "observation_horizon_s": LEVEL_HOLD_DURATION_S, "settled_can_position_m": settled_can_pos.tolist(), "settled_can_quat_wxyz": settled_can_quat.tolist(), "settled_table_position_m": settled_table_pos.tolist(), "settled_table_quat_wxyz": settled_table_quat.tolist(), "settled_qpos_digest": payload_hash({"qpos": settled_qpos.tolist()}), "settle_terminal": settled_trace, "hold_trace": hold_rows, "terminal_support": hold_rows[-1]["contacts"], "maximum_hold_horizontal_speed_m_s": max(row["horizontal_speed_m_s"] for row in hold_rows), "maximum_hold_horizontal_displacement_m": max(row["horizontal_displacement_m"] for row in hold_rows), "passed": bool(hold_rows[-1]["contacts"]["contact_count"] > 0 and hold_rows[-1]["contacts"]["normal_force_N"] > 0.001 and max(row["horizontal_speed_m_s"] for row in hold_rows) <= VELOCITY_MAX_M_S and max(row["horizontal_displacement_m"] for row in hold_rows) <= INCLINE_SLIP_DISPLACEMENT_M and sum(int(value) for value in hold_rows[-1]["warning_number"]) == 0), "_settled_can_position": settled_can_pos.tolist(), "_settled_can_quat": settled_can_quat.tolist(), "_settled_table_position": settled_table_pos.tolist(), "_settled_table_quat": settled_table_quat.tolist()}


def _incline_one(candidate: Mapping[str, Any], *, angle_rad: float, level: Mapping[str, Any]) -> dict[str, Any]:
    table_position = np.asarray(level["_settled_table_position"], dtype=float)
    table_quat = np.asarray(level["_settled_table_quat"], dtype=float)
    can_position = np.asarray(level["_settled_can_position"], dtype=float)
    can_quat = np.asarray(level["_settled_can_quat"], dtype=float)
    table_rotation = _rotation_y(angle_rad)
    base_rotation_flat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(base_rotation_flat, table_quat)
    base_rotation = base_rotation_flat.reshape(3, 3)
    relative_position = base_rotation.T.dot(can_position - table_position)
    relative_quat = _quat_mul(_quat_inverse(table_quat), can_quat)
    incline_table_quat = np.asarray((math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0), dtype=float)
    incline_position = table_position + table_rotation.dot(relative_position)
    incline_quat = _quat_mul(incline_table_quat, relative_quat)
    model = mujoco.MjModel.from_xml_string(_raw_xml(candidate, fixed_table=True, table_angle_rad=angle_rad, can_pos_m=incline_position, can_quat_wxyz=incline_quat))
    data = mujoco.MjData(model)
    can_body, can_joint, _, table_geom = _ids(model)
    can_qvel = int(model.jnt_dofadr[can_joint])
    data.qvel[can_qvel : can_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    initial = np.asarray(data.xpos[can_body], dtype=float).copy()
    rows = []
    steps = int(math.ceil(INCLINE_HORIZON_S / float(model.opt.timestep)))
    for index in range(steps + 1):
        contacts, _ = _contact_rows(model, data, table_geom=table_geom, cumulative_impulse=0.0, dt=float(model.opt.timestep))
        qvel = np.asarray(data.qvel[can_qvel : can_qvel + 6], dtype=float)
        rows.append({"time_s": float(index * model.opt.timestep), "can_position_m": np.asarray(data.xpos[can_body], dtype=float).tolist(), "horizontal_displacement_m": float(np.linalg.norm(np.asarray(data.xpos[can_body], dtype=float)[:2] - initial[:2])), "horizontal_speed_m_s": float(np.linalg.norm(qvel[:2])), "contacts": contacts, "warning_number": np.asarray(data.warning.number, dtype=int).tolist()})
        if index < steps:
            mujoco.mj_step(model, data)
    displacement = float(rows[-1]["horizontal_displacement_m"])
    return {"angle_rad": float(angle_rad), "observation_horizon_s": INCLINE_HORIZON_S, "trace": rows, "horizontal_displacement_m": displacement, "terminal_horizontal_speed_m_s": float(rows[-1]["horizontal_speed_m_s"]), "terminal_contact_count": int(rows[-1]["contacts"]["contact_count"]), "maximum_penetration_m": float(max(row["contacts"]["penetration_m"] for row in rows)), "warning_count": int(sum(int(value) for value in rows[-1]["warning_number"])), "sliding": bool(displacement > INCLINE_SLIP_DISPLACEMENT_M)}


def run_incline_calibration(candidate: Mapping[str, Any], level: Mapping[str, Any]) -> dict[str, Any]:
    coarse_angles = (0.0, 0.10, 0.20, 0.28, 0.30, 0.35)
    records = [_incline_one(candidate, angle_rad=angle, level=level) for angle in coarse_angles]
    refinements = []
    for _ in range(INCLINE_BINARY_ITERATIONS):
        stable = [float(row["angle_rad"]) for row in records if not row["sliding"]]
        sliding = [float(row["angle_rad"]) for row in records if row["sliding"]]
        if not stable or not sliding:
            break
        lower = max(stable)
        upper = min(sliding)
        if upper <= lower:
            break
        midpoint = 0.5 * (lower + upper)
        row = _incline_one(candidate, angle_rad=midpoint, level=level)
        records.append(row)
        refinements.append(row)
    stable = [float(row["angle_rad"]) for row in records if not row["sliding"]]
    sliding = [float(row["angle_rad"]) for row in records if row["sliding"]]
    lower = max(stable) if stable else None
    upper = min(sliding) if sliding else None
    width = None if lower is None or upper is None else upper - lower
    return {"measurement": "raw_mujoco_level_first_binary_incline_bracket", "coarse_angle_grid_rad": list(coarse_angles), "binary_refinement_iterations": INCLINE_BINARY_ITERATIONS, "slip_displacement_threshold_m": INCLINE_SLIP_DISPLACEMENT_M, "declared_resolution_rad": INCLINE_RESOLUTION_RAD, "angle_records": sorted(records, key=lambda row: float(row["angle_rad"])), "binary_refinement_records": refinements, "lower_stable_angle_rad": lower, "upper_sliding_angle_rad": upper, "bracket_width_rad": width, "bracket_order_passed": bool(lower is not None and upper is not None and lower < upper and width <= INCLINE_RESOLUTION_RAD), "analytic_coulomb_comparison_angle_rad": math.atan(0.30), "analytic_is_comparison_only": True}


def _v7_facts() -> dict[str, Any]:
    rows = []
    for path in sorted((Path(models.assets_root)).glob(f"{V7_RAW_PREFIX}*.json")):
        if "contact_" not in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        evidence = payload.get("evidence", {})
        physics = evidence.get("physics", {})
        recovery = evidence.get("derived_numeric_metrics", {}).get("recovery", {})
        trace = physics.get("recovery_trace", [])
        terminal_contact = bool(trace and trace[-1].get("contacts", {}).get("contact_count", 0) > 0)
        rows.append({"candidate_id": evidence.get("candidate_id"), "terminal_linear_velocity_m_s": recovery.get("terminal_linear_velocity_m_s"), "terminal_angular_velocity_rad_s": recovery.get("terminal_angular_velocity_rad_s"), "terminal_named_table_contact": terminal_contact, "raw_sha256": file_sha256(path)})
    return {"contact_candidates": rows, "incline_fixture_limitation": "V7 angle grid began at 0.10 rad and had no zero-angle stable control; its lower bracket is invalid and is not reused."}


def diagnose_all(*, protocol_path: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    protocol, protocol_name, protocol_hash = load_v7_protocol(protocol_path)
    validate_v7_protocol(protocol, protocol_bytes_hash=protocol_hash)
    states = [state for state in _resolve_v7(protocol, protocol_bytes_hash=protocol_hash) if state.stage == "contact"]
    candidates = []
    for state in states:
        candidate = _contact_candidate_mapping(state)
        normal = run_normal_impact_trace(candidate)
        candidates.append({"candidate_id": state.candidate_id, "contact_tuple": candidate, "normal_impact": normal})
    default_state = next(state for state in states if state.candidate_id == "c6_rolling_high")
    default_candidate = _contact_candidate_mapping(default_state)
    level = run_level_control(default_candidate)
    incline = run_incline_calibration(default_candidate, level) if level["passed"] else {"bracket_order_passed": False, "blocked_by_level_control": True, "angle_records": [], "analytic_is_comparison_only": True}
    level_public = {key: value for key, value in level.items() if not str(key).startswith("_")}
    payload = {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "status": "DESIGN_EVIDENCE_ONLY", "selection_authority": False, "scoreable": False, "not_v8_candidate_result": True, "no_task_outcome_used": True, "protocol": {"path": protocol_name, "sha256_bytes": protocol_hash}, "fixed_contract": {"recovery_duration_s": RECOVERY_DURATION_S, "tail_window_s": TAIL_WINDOW_S, "velocity_max_m_s": VELOCITY_MAX_M_S, "angular_velocity_max_rad_s": ANGULAR_VELOCITY_MAX_RAD_S, "penetration_max_m": PENETRATION_MAX_M, "terminal_named_table_support_required": True}, "exact_fixture": {"can_mass_kg": CAN_MASS_KG, "can_inertia_kg_m2": list(CAN_INERTIA_KG_M2), "table_mass_kg": TABLE_MASS_KG, "table_inertia_kg_m2": list(TABLE_INERTIA_KG_M2), "table_collision_size_m": list(TABLE_HALF_SIZE_M), "drop_height_m": 0.15, "v7_default_contact": default_candidate, "raw_model_only": True}, "v7_mechanism": _v7_facts(), "normal_impact_candidates": candidates, "level_control": level_public, "incline_calibration": incline, "rationale": "V7 ends with no named table support and downward Can motion. V8 measures terminal support and normal energy directly, and calibrates incline only after a level zero-angle support control."}
    payload["payload_sha256"] = payload_hash(payload)
    destination = Path(output_path) if output_path is not None else Path(models.assets_root) / OUTPUT_FILENAME
    write_json_atomic(destination, payload)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        result = diagnose_all(protocol_path=args.protocol, output_path=args.output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"], "candidate_count": len(result["normal_impact_candidates"]), "path": args.output or str(Path(models.assets_root) / OUTPUT_FILENAME), "payload_sha256": result["payload_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
