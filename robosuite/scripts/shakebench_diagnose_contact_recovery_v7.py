"""Non-scoreable Phase 06R6 diagnostic for the V6 contact failures.

This script uses the V6 resolved physics facts and the real MuJoCo contact
model, but it is not a V7 candidate run.  It never calls environment reward,
success, metrics, or controller outcome APIs and it never writes an official
profile.  Its output is design evidence used to choose a finite V7 contact
family.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_select_physics_v6 import (
    _profile_from_state,
    _resolve,
    load_v6_protocol,
    validate_v6_protocol,
)
from robosuite.utils.shakebench_artifacts import payload_hash, write_json_atomic
from robosuite.utils.shakebench_protocol_v6 import ResolvedProbeState


SCHEMA_ID = "shakebench.phase06r6.v7.contact_recovery_diagnostic"
SCHEMA_VERSION = 1
OUTPUT_FILENAME = "shakebench_phase_06r6_v7_contact_recovery_diagnostic.json"
V6_RAW_PREFIX = "shakebench_phase_06r5_v6_raw_contact_"
RECOVERY_DURATION_S = 0.50
RECOVERY_VELOCITY_MAX_M_S = 0.02
RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S = 0.20
PENETRATION_MAX_M = 0.0005
TAIL_WINDOW_S = 0.05


def _body_name(model: Any, body_id: int) -> str | None:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    return None if value is None else str(value)


def _geom_name(model: Any, geom_id: int) -> str | None:
    value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return None if value is None else str(value)


def _interface(first: str | None, second: str | None, *, can: set[str], table: set[str], target_bottom: set[str], target_wall: set[str], finger: set[str]) -> str:
    pair = {first, second}
    if pair & can and pair & table:
        return "can_open_worktable"
    if pair & can and pair & target_bottom:
        return "can_target_bottom"
    if pair & can and pair & target_wall:
        return "can_target_walls"
    if pair & can and pair & finger:
        return "can_panda_finger_pads"
    return "other"


def _contact_snapshot(model: Any, data: Any, *, can_names: set[str], table_names: set[str], bottom_names: set[str], wall_names: set[str], finger_names: set[str], cumulative_impulse: dict[str, float], dt: float) -> dict[str, Any]:
    interfaces: dict[str, dict[str, float]] = {}
    contacts: list[dict[str, Any]] = []
    penetration = 0.0
    normal_force = 0.0
    tangential_force = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first = _geom_name(model, int(contact.geom1))
        second = _geom_name(model, int(contact.geom2))
        name = _interface(first, second, can=can_names, table=table_names, target_bottom=bottom_names, target_wall=wall_names, finger=finger_names)
        if name == "other":
            continue
        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, wrench)
        current_normal = abs(float(wrench[0]))
        current_tangent = float(np.linalg.norm(wrench[1:3]))
        normal_force += current_normal
        tangential_force += current_tangent
        penetration = max(penetration, max(0.0, -float(contact.dist)))
        cumulative_impulse[name] = cumulative_impulse.get(name, 0.0) + current_normal * dt
        row = interfaces.setdefault(name, {"contact_count": 0.0, "normal_force_N": 0.0, "tangential_force_N": 0.0, "normal_impulse_Ns": 0.0})
        row["contact_count"] += 1.0
        row["normal_force_N"] += current_normal
        row["tangential_force_N"] += current_tangent
        row["normal_impulse_Ns"] = cumulative_impulse[name]
        contacts.append({"geom1": first, "geom2": second, "interface": name, "distance_m": float(contact.dist), "normal_force_N": current_normal, "tangential_force_N": current_tangent})
    return {
        "contact_count": len(contacts),
        "active_interfaces": sorted(interfaces),
        "interfaces": {key: {name: float(value) for name, value in row.items()} for key, row in interfaces.items()},
        "normal_force_N": normal_force,
        "tangential_force_N": tangential_force,
        "penetration_m": penetration,
        "contacts": contacts,
    }


def _mechanical_energy(model: Any, data: Any, *, body_id: int, qvel: np.ndarray, gravity_m_s2: float) -> dict[str, float]:
    mass = float(model.body_mass[body_id])
    inertia = np.asarray(model.body_inertia[body_id], dtype=float)
    kinetic_trans = 0.5 * mass * float(np.sum(qvel[:3] * qvel[:3]))
    kinetic_rot = 0.5 * float(np.sum(inertia * qvel[3:6] * qvel[3:6]))
    potential = mass * gravity_m_s2 * float(data.xpos[body_id][2])
    return {"kinetic_translational_J": kinetic_trans, "kinetic_rotational_J": kinetic_rot, "potential_J": potential, "mechanical_energy_J": kinetic_trans + kinetic_rot + potential}


def _force_envelope_audit(model: Any, *, finger_actuator_names: tuple[str, ...]) -> dict[str, Any]:
    rows = []
    limits = []
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name is None or str(name) not in finger_actuator_names:
            continue
        name = str(name)
        gear = np.asarray(model.actuator_gear[actuator_id], dtype=float)
        forcerange = np.asarray(model.actuator_forcerange[actuator_id], dtype=float)
        ctrlrange = np.asarray(model.actuator_ctrlrange[actuator_id], dtype=float)
        transmission_gain = float(np.max(np.abs(gear[:1])))
        force_limit = float(np.max(np.abs(forcerange)) * transmission_gain)
        limits.append(force_limit)
        rows.append({"actuator_id": actuator_id, "name": name, "ctrlrange": ctrlrange.tolist(), "forcerange": forcerange.tolist(), "gear": gear.tolist(), "transmission_gain": transmission_gain, "joint_id": int(model.actuator_trnid[actuator_id, 0]), "force_limit_N": force_limit})
    if not rows:
        raise RuntimeError("compiled Panda finger actuator set is empty")
    total = float(sum(limits))
    return {
        "actuators": rows,
        "finite_force_envelope_N": total,
        "derivation": "sum over the two compiled Panda finger position actuators of max(abs(actuator_forcerange)) * abs(first transmission gear); both are unit-gain slide transmissions",
        "contact_force_is_not_actuator_limited_when_solver_penetration_is_injected": True,
    }


def _finger_load_trace(env: Any, *, duration_s: float, envelope: Mapping[str, Any]) -> dict[str, Any]:
    model = env.sim.model._model
    data = env.sim.data._data
    can_body_id = int(env.can_body_id)
    can_joint_id = int(model.joint(env.can.joints[0]).id)
    can_qpos = int(model.jnt_qposadr[can_joint_id])
    can_qvel = int(model.jnt_dofadr[can_joint_id])
    pad_positions = [data.geom_xpos[model.geom(name).id].copy() for name in env.finger_pad_geom_names]
    target_position = np.mean(pad_positions, axis=0)
    data.qpos[can_qpos : can_qpos + 3] = target_position
    data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[can_qvel : can_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    can_names = set(env.can.contact_geoms)
    finger_names = set(env.finger_pad_geom_names)
    trace = []
    steps = max(1, int(math.ceil(duration_s / float(model.opt.timestep))))
    maximum_normal = 0.0
    maximum_penetration = 0.0
    named_pairs: set[tuple[str | None, str | None]] = set()
    cumulative_impulse: dict[str, float] = {}
    for index in range(steps + 1):
        contacts = _contact_snapshot(model, data, can_names=can_names, table_names=set(), bottom_names=set(), wall_names=set(), finger_names=finger_names, cumulative_impulse=cumulative_impulse, dt=float(model.opt.timestep))
        maximum_normal = max(maximum_normal, float(contacts["normal_force_N"]))
        maximum_penetration = max(maximum_penetration, float(contacts["penetration_m"]))
        named_pairs.update((pair["geom1"], pair["geom2"]) for pair in contacts["contacts"])
        trace.append({"time_s": float(index * model.opt.timestep), "can_position_m": np.asarray(data.xpos[can_body_id], dtype=float).tolist(), "contacts": contacts})
        if index < steps:
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
    return {
        "trajectory": {"type": "V6-compatible fixed Can pose at compiled Panda finger-pad midpoint", "duration_s": duration_s, "actuator_control": "all controls zero", "initial_velocity": [0.0] * 6, "target_position_m": target_position.tolist()},
        "trace": trace,
        "maximum_normal_force_N": maximum_normal,
        "maximum_penetration_m": maximum_penetration,
        "force_envelope_N": float(envelope["finite_force_envelope_N"]),
        "over_force_rejected": bool(maximum_normal > float(envelope["finite_force_envelope_N"])),
        "named_pair_audit": [list(pair) for pair in sorted(named_pairs, key=lambda item: (str(item[0]), str(item[1])))],
    }


def _declared_finger_load_trace(env: Any, *, duration_s: float, envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Probe each compiled Can--finger pair with a fixed, shallow pose hold.

    The two pad faces are audited in separate holds because the compiled pad
    boxes are not exactly symmetric about the Can mesh.  ``mj_forward`` is
    intentionally used instead of a task step: this is a declared contact
    force/penetration trajectory, not a grasp-control or task-outcome probe.
    """

    model = env.sim.model._model
    data = env.sim.data._data
    can_body_id = int(env.can_body_id)
    can_joint_id = int(model.joint(env.can.joints[0]).id)
    can_qpos = int(model.jnt_qposadr[can_joint_id])
    can_qvel = int(model.jnt_dofadr[can_joint_id])
    finger_joint_ids = [int(model.joint(name).id) for name in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")]
    finger_qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in finger_joint_ids]
    finger_actuator_ids = [int(model.actuator(name).id) for name in ("gripper0_right_gripper_finger_joint1", "gripper0_right_gripper_finger_joint2")]
    pad_ids = [int(model.geom(name).id) for name in env.finger_pad_geom_names]
    q = 0.0256
    pad_offsets_m = (0.02909, -0.02890)
    steps = max(1, int(math.ceil(duration_s / float(model.opt.timestep))))
    can_names = set(env.can.contact_geoms)
    finger_names = set(env.finger_pad_geom_names)
    trace = []
    maximum_normal = 0.0
    maximum_penetration = 0.0
    named_pairs: set[tuple[str | None, str | None]] = set()
    for pad_index, pad_id in enumerate(pad_ids):
        # First open the compiled fingers and obtain their actual world pose.
        data.qpos[finger_qpos[0]] = q
        data.qpos[finger_qpos[1]] = -q
        mujoco.mj_forward(model, data)
        pad_position = np.asarray(data.geom_xpos[pad_id], dtype=float).copy()
        target_position = pad_position + np.asarray((0.0, pad_offsets_m[pad_index], 0.0))
        data.qpos[can_qpos : can_qpos + 3] = target_position
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[can_qvel : can_qvel + 6] = 0.0
        data.ctrl[:] = 0.0
        data.ctrl[finger_actuator_ids] = (q, -q)
        cumulative_impulse: dict[str, float] = {}
        for index in range(steps + 1):
            # Reapply the declared pose at every sample.  This keeps the
            # force/penetration measurement aligned to the named static hold
            # and prevents gravity-driven task motion from becoming an
            # unregistered controller experiment.
            data.qpos[finger_qpos[0]] = q
            data.qpos[finger_qpos[1]] = -q
            data.qpos[can_qpos : can_qpos + 3] = target_position
            data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[can_qvel : can_qvel + 6] = 0.0
            mujoco.mj_forward(model, data)
            contacts = _contact_snapshot(
                model,
                data,
                can_names=can_names,
                table_names=set(),
                bottom_names=set(),
                wall_names=set(),
                finger_names=finger_names,
                cumulative_impulse=cumulative_impulse,
                dt=float(model.opt.timestep),
            )
            maximum_normal = max(maximum_normal, float(contacts["normal_force_N"]))
            maximum_penetration = max(maximum_penetration, float(contacts["penetration_m"]))
            named_pairs.update((pair["geom1"], pair["geom2"]) for pair in contacts["contacts"])
            trace.append({"time_s": float(index * model.opt.timestep), "pad_index": pad_index, "hold_time_s": float(index * model.opt.timestep), "can_position_m": target_position.tolist(), "contacts": contacts})
    required_pairs = {
        tuple(sorted((str(env.can.contact_geoms[0]), str(env.finger_pad_geom_names[0])))),
        tuple(sorted((str(env.can.contact_geoms[0]), str(env.finger_pad_geom_names[1])))),
    }
    observed_pairs = {tuple(sorted((str(first), str(second)))) for first, second in named_pairs}
    return {
        "trajectory": {
            "type": "two declared shallow static Can--finger face holds",
            "duration_s": duration_s,
            "hold_count": len(pad_ids),
            "finger_joint_positions": [q, -q],
            "pad_offset_m_by_pad": list(pad_offsets_m),
            "actuator_control": [q, -q],
            "pose_reapplied_before_each_mj_forward": True,
        },
        "trace": trace,
        "maximum_normal_force_N": maximum_normal,
        "maximum_penetration_m": maximum_penetration,
        "normal_force_N": maximum_normal,
        "penetration_m": maximum_penetration,
        "force_envelope_N": float(envelope["finite_force_envelope_N"]),
        "minimum_force_N": min(
            float(row["contacts"]["normal_force_N"])
            for row in trace
            if row["contacts"]["contact_count"]
        ) if any(row["contacts"]["contact_count"] for row in trace) else 0.0,
        "named_pair_audit": [list(pair) for pair in sorted(named_pairs, key=lambda item: (str(item[0]), str(item[1])))],
        "required_named_pairs": [list(pair) for pair in sorted(required_pairs)],
        "all_required_named_pairs_observed": required_pairs.issubset(observed_pairs),
        "over_force_rejected": bool(maximum_normal > float(envelope["finite_force_envelope_N"])),
        "penetration_gate_passed": bool(maximum_penetration <= PENETRATION_MAX_M),
    }


def diagnose_state(state: ResolvedProbeState, *, finger_mode: str = "v6_midpoint") -> dict[str, Any]:
    if state.stage != "contact" or state.contact is None:
        raise ValueError("diagnostic requires a V6 contact state")
    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    profile = _profile_from_state(state)
    contact = state.contact
    envelope_audit: dict[str, Any] | None = None
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=state.common.physics_timestep_s,
        target_container_friction=(float(contact.sliding_mu["table_object"]), contact.torsional_mu, contact.rolling_mu),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=20,
        seed=17,
    )
    try:
        model = env.sim.model._model
        data = env.sim.data._data
        envelope_audit = _force_envelope_audit(
            model,
            finger_actuator_names=("gripper0_right_gripper_finger_joint1", "gripper0_right_gripper_finger_joint2"),
        )
        env.reset()
        can_body_id = int(env.can_body_id)
        can_joint_id = int(model.joint(env.can.joints[0]).id)
        can_qpos = int(model.jnt_qposadr[can_joint_id])
        can_qvel = int(model.jnt_dofadr[can_joint_id])
        table_top = np.asarray(env.arena.table_top_abs, dtype=float)
        data.qpos[can_qpos : can_qpos + 3] = table_top + np.asarray((0.0, 0.0, 0.15))
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[can_qvel : can_qvel + 6] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        can_names = set(env.can.contact_geoms)
        table_names = set(env.table_contact_geom_names)
        bottom_names = set(env.target_bottom_geom_names)
        wall_names = set(env.target_wall_geom_names)
        finger_names = set(env.finger_pad_geom_names)
        cumulative_impulse: dict[str, float] = {}
        trace = []
        first_impact_time = None
        steps = int(math.ceil(RECOVERY_DURATION_S / float(model.opt.timestep)))
        for index in range(steps + 1):
            contacts = _contact_snapshot(model, data, can_names=can_names, table_names=table_names, bottom_names=bottom_names, wall_names=wall_names, finger_names=finger_names, cumulative_impulse=cumulative_impulse, dt=float(model.opt.timestep))
            if first_impact_time is None and contacts["contact_count"] and contacts["penetration_m"] >= 0.0:
                # A contact with a non-positive distance is an impact; the
                # initial release frame has no Can contact in this fixture.
                if any(float(pair["distance_m"]) <= 0.0 for pair in contacts["contacts"]):
                    first_impact_time = float(index * model.opt.timestep)
            velocity = np.asarray(data.qvel[can_qvel : can_qvel + 6], dtype=float)
            energy = _mechanical_energy(model, data, body_id=can_body_id, qvel=velocity, gravity_m_s2=9.81)
            trace.append({"time_s": float(index * model.opt.timestep), "position_m": np.asarray(data.xpos[can_body_id], dtype=float).tolist(), "orientation_wxyz": np.asarray(data.xquat[can_body_id], dtype=float).tolist(), "linear_velocity_m_s": velocity[:3].tolist(), "angular_velocity_rad_s": velocity[3:6].tolist(), "linear_speed_m_s": float(np.linalg.norm(velocity[:3])), "angular_speed_rad_s": float(np.linalg.norm(velocity[3:6])), "contacts": contacts, "energy": energy, "warning_number": np.asarray(data.warning.number, dtype=int).tolist()})
            if index < steps:
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)
        tail = [row for row in trace if row["time_s"] >= RECOVERY_DURATION_S - TAIL_WINDOW_S - 1.0e-12]
        linear_tail = np.asarray([row["linear_speed_m_s"] for row in tail], dtype=float)
        angular_tail = np.asarray([row["angular_speed_rad_s"] for row in tail], dtype=float)
        terminal = trace[-1]
        sustained_spin = bool(np.mean(angular_tail >= RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S) > 0.8)
        sustained_translation = bool(np.mean(linear_tail >= RECOVERY_VELOCITY_MAX_M_S) > 0.8)
        mechanism = "persistent_rolling_spin" if sustained_spin and sustained_translation else ("persistent_translation" if sustained_translation else "rebound_or_solver_oscillation")
        recovery = {
            "release_time_s": 0.0,
            "first_impact_time_s": first_impact_time,
            "duration_s": RECOVERY_DURATION_S,
            "trace_dt_s": float(model.opt.timestep),
            "trace_sample_count": len(trace),
            "terminal_linear_velocity_m_s": terminal["linear_speed_m_s"],
            "terminal_angular_velocity_rad_s": terminal["angular_speed_rad_s"],
            "tail_window_s": TAIL_WINDOW_S,
            "tail_window_max_linear_velocity_m_s": float(np.max(linear_tail)),
            "tail_window_rms_linear_velocity_m_s": float(np.sqrt(np.mean(linear_tail**2))),
            "tail_window_max_angular_velocity_rad_s": float(np.max(angular_tail)),
            "tail_window_rms_angular_velocity_rad_s": float(np.sqrt(np.mean(angular_tail**2))),
            "maximum_penetration_m": float(max(row["contacts"]["penetration_m"] for row in trace)),
            "maximum_normal_force_N": float(max(row["contacts"]["normal_force_N"] for row in trace)),
            "recovery_velocity_gate_passed": bool(terminal["linear_speed_m_s"] <= RECOVERY_VELOCITY_MAX_M_S and float(np.max(linear_tail)) <= RECOVERY_VELOCITY_MAX_M_S),
            "recovery_angular_velocity_gate_passed": bool(terminal["angular_speed_rad_s"] <= RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S and float(np.max(angular_tail)) <= RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S),
            "penetration_gate_passed": bool(max(row["contacts"]["penetration_m"] for row in trace) <= PENETRATION_MAX_M),
            "mechanism_assessment": mechanism,
            "sustained_translation_above_gate": sustained_translation,
            "sustained_spin_above_gate": sustained_spin,
        }
        if finger_mode == "v6_midpoint":
            finger = _finger_load_trace(env, duration_s=0.10, envelope=envelope_audit)
        elif finger_mode == "v7_declared":
            finger = _declared_finger_load_trace(env, duration_s=0.10, envelope=envelope_audit)
        else:
            raise ValueError(f"unsupported finger_mode: {finger_mode}")
        return {
            "candidate_id": contact.candidate_id,
            "profile": {"condim": contact.condim, "sliding_mu": dict(contact.sliding_mu), "torsional_mu": contact.torsional_mu, "rolling_mu": contact.rolling_mu, "margin_m": contact.margin_m, "gap_m": contact.gap_m, "solref": list(contact.solref), "solimp": list(contact.solimp), "iterations": contact.iterations, "interfaces": list(contact.interfaces)},
            "recovery": recovery,
            "recovery_trace": trace,
            "finger_force_envelope": envelope_audit,
            "finger_load": finger,
            "selection_input": False,
            "scoreable": False,
        }
    finally:
        env.close()


def diagnose_all(*, protocol_path: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    protocol, protocol_name, protocol_hash = load_v6_protocol(protocol_path)
    validate_v6_protocol(protocol, protocol_bytes_hash=protocol_hash)
    states = [state for state in _resolve(protocol, protocol_bytes_hash=protocol_hash) if state.stage == "contact"]
    results = [diagnose_state(state) for state in states]
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "DESIGN_EVIDENCE_ONLY",
        "selection_authority": False,
        "scoreable": False,
        "not_v7_candidate_result": True,
        "protocol": {"path": protocol_name, "sha256_bytes": protocol_hash},
        "fixed_recovery_contract": {"duration_s": RECOVERY_DURATION_S, "tail_window_s": TAIL_WINDOW_S, "velocity_max_m_s": RECOVERY_VELOCITY_MAX_M_S, "angular_velocity_max_rad_s": RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S, "penetration_max_m": PENETRATION_MAX_M},
        "no_task_outcome_used": True,
        "candidates": results,
        "rationale": "V6 candidates exhibit sustained post-impact translation/spin and the historical finger midpoint placement injects solver contact force far above the compiled two-actuator envelope; V7 must activate rolling resistance and reject over-force evidence.",
    }
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
    print(json.dumps({"status": result["status"], "candidates": len(result["candidates"]), "path": args.output or str(Path(models.assets_root) / OUTPUT_FILENAME), "payload_sha256": result["payload_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
