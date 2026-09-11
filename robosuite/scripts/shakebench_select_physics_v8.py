"""Phase 06R7/V8 normal-impact contact selection and verifier.

V8 inherits only the independently checked V6/V7 driver and isolator
evidence.  It varies legal normal contact response parameters over a finite,
pre-registered family and requires raw terminal table support in addition to
the frozen recovery speed, angular-speed, penetration, force, slip, and
convergence gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_diagnose_normal_impact_v8 import (
    ANGULAR_VELOCITY_MAX_RAD_S,
    CAN_INERTIA_KG_M2,
    CAN_MASS_KG,
    INCLINE_BINARY_ITERATIONS,
    INCLINE_RESOLUTION_RAD,
    INCLINE_SLIP_DISPLACEMENT_M,
    LEVEL_HOLD_DURATION_S,
    LEVEL_SETTLE_DURATION_S,
    PENETRATION_MAX_M,
    RECOVERY_DURATION_S,
    TAIL_WINDOW_S,
    VELOCITY_MAX_M_S,
    run_incline_calibration,
    run_level_control,
    run_normal_impact_trace,
)
from robosuite.scripts.shakebench_select_physics_v6 import (
    V6ProtocolError,
    _contact_candidate_mapping,
    _driver_eligibility,
    _isolator_eligible,
    _isolator_score,
    _parity_eligible,
    _profile_from_state,
    _resolve,
    _select_driver,
    _v6_replay_probe,
    load_v6_protocol,
    validate_v6_protocol,
)
from robosuite.scripts.shakebench_select_physics_v7 import (
    FINGER_PAIR_NAMES,
    _compile_contact_audit,
    _compile_force_envelope,
    _compiled_contact_matches,
    _finger_from_trace,
    _verify_v6_inherited_evidence,
    load_v7_protocol,
)
from robosuite.utils.shakebench_artifacts import (
    file_sha256,
    payload_hash,
    sha256_bytes,
    sha256_json,
    verify_payload_hash,
    write_json_atomic,
)
from robosuite.utils.shakebench_phase06_adapters import (
    AdapterContractError,
    adapter_contract_payload,
)
from robosuite.utils.shakebench_physics import PhysicsProfile, physics_profile_hash
from robosuite.utils.shakebench_protocol_v6 import (
    ResolvedProbeState,
    V6ProtocolStateError as V8ProtocolStateError,
    legal_replay_binding_templates,
    resolve_all_protocol_states_v6 as resolve_all_protocol_states_v8,
    resolve_replay_binding_templates,
    sha256_json as sha256_state_json,
)


SCHEMA_ID = "shakebench.phase06r7.v8.physics_selection"
PROTOCOL_SCHEMA_ID = "shakebench.phase06r7.v8.physics_selection_protocol"
SCHEMA_VERSION = 8
PROTOCOL_FILENAME = "shakebench_selection_protocol_v8.yaml"
RAW_PREFIX = "shakebench_phase_06r7_v8_raw_"
SELECTED_FILENAME = "shakebench_phase_06r7_v8_selected_candidates.json"
EXCLUDED_FILENAME = "shakebench_phase_06r7_v8_excluded_candidates.json"
FEASIBILITY_FILENAME = "shakebench_phase_06r7_v8_feasibility.json"
STATUS_FILENAME = "shakebench_phase_06r7_v8_status.json"
PARITY_FILENAME = "shakebench_phase_06r7_v8_gamma_zero_parity.json"
DETERMINISM_FILENAME = "shakebench_phase_06r7_v8_replay_determinism.json"
OFFICIAL_PROFILE_FILENAME = "shakebench_official_physics.yaml"
V7_PROTOCOL_FILENAME = "shakebench_selection_protocol_v7.yaml"
V7_SELECTED_FILENAME = "shakebench_phase_06r6_v7_selected_candidates.json"
V7_STATUS_FILENAME = "shakebench_phase_06r6_v7_status.json"
V7_FEASIBILITY_FILENAME = "shakebench_phase_06r6_v7_feasibility.json"

CONTACT_IDS = {
    "n6_critical_negative_control",
    "n6_overdamped_2",
    "n6_overdamped_4",
    "n6_slow_overdamped",
    "n6_soft_impedance",
}
NONCONTACT_STAGES = {"driver", "isolator"}
REPLAY_GROUPS = {"driver", "isolator", "contact", "gamma_zero_parity"}


class V8ProtocolError(ValueError):
    """V8 protocol or registration-time contract failure."""


class V8EvidenceError(RuntimeError):
    """V8 raw evidence is absent, mutated, or incomplete."""


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    try:
        import yaml

        value = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise V8ProtocolError("V8 protocol root must be a mapping")
    return value


def load_v8_protocol(path: str | Path | None = None) -> tuple[Mapping[str, Any], str, str]:
    protocol_path = Path(models.assets_root) / PROTOCOL_FILENAME if path is None else Path(path)
    raw = protocol_path.read_bytes()
    return _load_yaml(protocol_path), str(protocol_path), sha256_bytes(raw)


def _resolve_v8(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> tuple[ResolvedProbeState, ...]:
    context = dict(selection_context or {})
    if protocol_bytes_hash is not None:
        context["protocol_sha256_bytes"] = protocol_bytes_hash
    context.setdefault("protocol_sha256_normalized", sha256_json(protocol))
    try:
        return resolve_all_protocol_states_v8(protocol, context)
    except V8ProtocolStateError as exc:
        raise V8ProtocolError(str(exc)) from exc


def _mapping(value: Any, label: str, *, required: bool = True) -> Mapping[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise V8ProtocolError(f"{label} must be a mapping")
    return value


def _rows(protocol: Mapping[str, Any], section: str) -> list[Mapping[str, Any]]:
    value = _mapping(protocol.get("components"), "components").get(section)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise V8ProtocolError(f"components.{section} must be a list of mappings")
    return list(value)


def _same_json(left: Any, right: Any) -> bool:
    return sha256_json(left) == sha256_json(right)


def _asset(name: str) -> Path:
    path = Path(models.assets_root) / name
    if not path.is_file():
        raise V8ProtocolError(f"required asset is missing: {name}")
    return path


def _noncontact_view(state: ResolvedProbeState) -> dict[str, Any]:
    value = state.to_dict()
    value["common"].pop("output", None)
    value["common"].pop("protocol_identity", None)
    if state.stage == "replay" and value.get("replay") is not None:
        value["replay"].pop("candidate_fields", None)
    return value


def _validate_v7_noncontact_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    from robosuite.scripts.shakebench_select_physics_v7 import _resolve_v7, load_v7_protocol, validate_v7_protocol

    v7_protocol, v7_name, v7_hash = load_v7_protocol(_asset(V7_PROTOCOL_FILENAME))
    validate_v7_protocol(v7_protocol, protocol_bytes_hash=v7_hash)
    v7_states = _resolve_v7(v7_protocol, protocol_bytes_hash=v7_hash)
    v8_states = _resolve_v8(protocol)
    v8_by_id = {state.state_id: state for state in v8_states}
    counts: dict[str, int] = {}
    for state in v7_states:
        if state.stage not in NONCONTACT_STAGES:
            continue
        other = v8_by_id.get(state.state_id)
        if other is None or other.stage != state.stage or not _same_json(_noncontact_view(state), _noncontact_view(other)):
            raise V8ProtocolError(f"V8 changed inherited resolved state: {state.state_id}")
        counts[state.stage] = counts.get(state.stage, 0) + 1
    if counts != {"driver": 18, "isolator": 3}:
        raise V8ProtocolError(f"V8 inherited state coverage is incomplete: {counts}")
    return {"protocol_path": v7_name, "protocol_sha256": v7_hash, "states": v7_states, "v8_states": v8_states}


def validate_v8_protocol(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None, check_v7_inheritance: bool = True) -> dict[str, Any]:
    """Validate V8 shape, exact V7 inheritance, and all resolved states."""

    if protocol.get("schema_id") != PROTOCOL_SCHEMA_ID or protocol.get("schema_version") != SCHEMA_VERSION:
        raise V8ProtocolError("wrong V8 protocol schema")
    if protocol.get("phase") != "06R7":
        raise V8ProtocolError("V8 protocol phase must be 06R7")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise V8ProtocolError("V8 protocol must be pre_registered and immutable")
    if protocol.get("selection_authority") != "physics_probes_only":
        raise V8ProtocolError("V8 selection authority must be physics_probes_only")
    scope = str(_mapping(protocol.get("scope"), "scope").get("forbidden_selection_inputs", "")).lower()
    for name in ("task_success", "reward", "controller_outcome", "state_tier", "policy_observation", "gamma_star"):
        if name not in scope:
            raise V8ProtocolError(f"scope must forbid {name}")
    for key in ("task_success", "task_sr", "reward", "controller_outcome_used", "state_tier_ranking", "policy_observation", "Gamma_star"):
        if key in protocol and protocol.get(key) not in (False, None, ""):
            raise V8ProtocolError(f"forbidden selection input appears at protocol root: {key}")

    v7_protocol, _, v7_hash = load_v7_protocol(_asset(V7_PROTOCOL_FILENAME))
    for key in ("frozen_facts", "measurement", "driver", "isolator", "parity"):
        if not _same_json(protocol.get(key), v7_protocol.get(key)):
            raise V8ProtocolError(f"V8 changed V7 immutable contract: {key}")
    v7_defaults = _mapping(v7_protocol.get("selection"), "V7 selection").get("default", {})
    defaults = _mapping(protocol.get("selection"), "selection").get("default", {})
    if defaults.get("driver_candidate_id") != v7_defaults.get("driver_candidate_id") or defaults.get("isolator_candidate_id") != v7_defaults.get("isolator_candidate_id"):
        raise V8ProtocolError("V8 changed inherited driver/isolator defaults")
    contacts = _rows(protocol, "contact_candidates")
    ids = {str(row.get("candidate_id", "")) for row in contacts}
    if ids != CONTACT_IDS or len(contacts) != len(CONTACT_IDS):
        raise V8ProtocolError("V8 normal-contact candidate set is incomplete or duplicated")
    v7_contacts = {str(row["candidate_id"]): row for row in _rows(v7_protocol, "contact_candidates")}
    v7_default_id = str(v7_defaults.get("contact_candidate_id"))
    if v7_default_id not in v7_contacts:
        raise V8ProtocolError("V7 default contact tuple is missing")
    base = v7_contacts[v7_default_id]
    for row in contacts:
        candidate_id = str(row["candidate_id"])
        for field in ("condim", "sliding_mu", "torsional_mu", "rolling_mu", "margin_m", "gap_m", "solref", "solimp", "iterations", "interfaces"):
            if field not in row:
                raise V8ProtocolError(f"V8 contact candidate {candidate_id} is missing {field}")
        for field in ("condim", "sliding_mu", "torsional_mu", "rolling_mu", "margin_m", "gap_m", "iterations", "interfaces"):
            if not _same_json(row[field], base[field]):
                raise V8ProtocolError(f"V8 changed tangential/rolling contact tuple for {candidate_id}: {field}")
        if float(row["solref"][0]) < 2.0 * 0.0002 or float(row["solref"][0]) <= 0.0 or float(row["solref"][1]) <= 0.0:
            raise V8ProtocolError(f"V8 normal solref is illegal for {candidate_id}")
        if not _same_json(row["solimp"][2:], base["solimp"][2:]):
            raise V8ProtocolError(f"V8 changed non-normal solimp fields for {candidate_id}")
    critical = next(row for row in contacts if row["candidate_id"] == "n6_critical_negative_control")
    critical_tuple = {key: value for key, value in critical.items() if key != "candidate_id"}
    base_tuple = {key: value for key, value in base.items() if key != "candidate_id"}
    if not _same_json(critical_tuple, base_tuple):
        raise V8ProtocolError("V8 critical negative control does not preserve the V7 default tuple")
    for candidate_id in ("n6_overdamped_2", "n6_overdamped_4", "n6_slow_overdamped"):
        row = next(row for row in contacts if row["candidate_id"] == candidate_id)
        if float(row["solref"][1]) <= 1.0:
            raise V8ProtocolError(f"V8 candidate is not over-damped: {candidate_id}")
    slow = next(row for row in contacts if row["candidate_id"] == "n6_slow_overdamped")
    if not (float(slow["solref"][0]) > float(base["solref"][0]) and float(slow["solref"][1]) > 1.0):
        raise V8ProtocolError("V8 slow compliant candidate is not slower and over-damped")
    soft = next(row for row in contacts if row["candidate_id"] == "n6_soft_impedance")
    if _same_json(soft["solimp"], base["solimp"]):
        raise V8ProtocolError("V8 impedance-variation candidate does not vary solimp")

    contact = _mapping(protocol.get("contact"), "contact")
    for key in ("interfaces", "probes", "hard_gates", "scoring", "normal_impact"):
        if key not in contact:
            raise V8ProtocolError(f"contact.{key} is required")
    if tuple(contact["interfaces"]) != tuple(v7_protocol["contact"]["interfaces"]):
        raise V8ProtocolError("V8 contact interfaces changed")
    probes = _mapping(contact["probes"], "contact.probes")
    for key in ("static_support_settle_s", "slip_initial_speed_m_s", "slip_duration_s", "impact_drop_height_m", "recovery_duration_s", "tail_window_s", "finger_load_duration_s", "timestep_candidates_s", "finger_force_trajectory", "finger_named_pairs"):
        if not _same_json(probes.get(key), v7_protocol["contact"]["probes"].get(key)):
            raise V8ProtocolError(f"V8 changed inherited contact probe convention: {key}")
    if float(probes.get("recovery_duration_s")) != RECOVERY_DURATION_S or float(probes.get("tail_window_s")) != TAIL_WINDOW_S:
        raise V8ProtocolError("V8 recovery horizon/window changed")
    normal_spec = _mapping(contact["normal_impact"], "contact.normal_impact")
    if float(normal_spec.get("duration_s")) != RECOVERY_DURATION_S or float(normal_spec.get("trace_dt_s")) != 0.0002:
        raise V8ProtocolError("V8 normal trace convention changed")
    coarse = tuple(float(value) for value in normal_spec.get("incline_coarse_angles_rad", ()))
    if not coarse or coarse[0] != 0.0 or not any(0.0 < value < math.atan(0.30) for value in coarse) or not any(value > math.atan(0.30) for value in coarse):
        raise V8ProtocolError("V8 incline grid must start at zero and straddle atan(0.30)")
    if int(normal_spec.get("incline_binary_refinement_iterations", -1)) != INCLINE_BINARY_ITERATIONS or float(normal_spec.get("incline_resolution_rad", -1.0)) != INCLINE_RESOLUTION_RAD:
        raise V8ProtocolError("V8 incline binary-refinement convention changed")
    gates = _mapping(contact["hard_gates"], "contact.hard_gates")
    for key in ("static_support_force_min_N", "incline_slip_displacement_m", "slip_final_speed_max_m_s", "maximum_illegal_penetration_m", "recovery_velocity_max_m_s", "recovery_angular_velocity_max_rad_s", "terminal_support_force_min_N", "finger_force_min_N", "finger_force_max_N", "finger_penetration_max_m", "warning_count_max", "timestep_trace_relative_error_max", "level_horizontal_speed_max_m_s", "level_horizontal_displacement_max_m", "incline_bracket_resolution_rad"):
        if key not in gates:
            raise V8ProtocolError(f"V8 contact hard gate is missing {key}")
    for key, expected in (("maximum_illegal_penetration_m", PENETRATION_MAX_M), ("recovery_velocity_max_m_s", VELOCITY_MAX_M_S), ("recovery_angular_velocity_max_rad_s", ANGULAR_VELOCITY_MAX_RAD_S), ("finger_force_max_N", 40.0), ("finger_penetration_max_m", PENETRATION_MAX_M), ("incline_bracket_resolution_rad", INCLINE_RESOLUTION_RAD), ("incline_slip_displacement_m", INCLINE_SLIP_DISPLACEMENT_M)):
        if float(gates[key]) != expected:
            raise V8ProtocolError(f"V8 hard gate changed frozen value: {key}")
    diagnostic = _mapping(contact.get("diagnostic"), "contact.diagnostic", required=False)
    if diagnostic and diagnostic.get("selection_authority") is not False:
        raise V8ProtocolError("V8 diagnostic must not have selection authority")

    replay = _mapping(protocol.get("replay"), "replay")
    if int(replay.get("process_count", 0)) < 3 or not set(REPLAY_GROUPS).issubset(set(replay.get("groups", ()) )):
        raise V8ProtocolError("V8 replay contract is incomplete")
    states = _resolve_v8(protocol, protocol_bytes_hash=protocol_bytes_hash)
    declared_count = int(protocol.get("manifest_state_count", -1))
    if declared_count != len(states):
        raise V8ProtocolError("V8 manifest_state_count does not match resolved states")
    if any(not state.output.startswith(RAW_PREFIX) or not state.output.endswith(".json") for state in states):
        raise V8ProtocolError("V8 raw outputs must use the flat V8 prefix")
    stage_counts = {stage: sum(state.stage == stage for state in states) for stage in ("driver", "isolator", "contact", "parity", "replay")}
    expected_counts = {"driver": 18, "isolator": 3, "contact": len(CONTACT_IDS), "parity": 1, "replay": 12}
    if stage_counts != expected_counts:
        raise V8ProtocolError(f"V8 manifest coverage is wrong: {stage_counts}")
    replay_groups = {state.replay.selected_component for state in states if state.stage == "replay" and state.replay is not None}
    if not REPLAY_GROUPS.issubset(replay_groups):
        raise V8ProtocolError("V8 replay manifest is missing a required group")
    inherited = _validate_v7_noncontact_protocol(protocol) if check_v7_inheritance else None
    return {"state_count": len(states), "stage_counts": stage_counts, "state_ids": [state.state_id for state in states], "outputs": [state.output for state in states], "resolved_state_digest": sha256_state_json([state.to_dict() for state in states]), "legal_replay_binding_count": len(legal_replay_binding_templates(protocol)), "v7_protocol_sha256": v7_hash, "inherited": {"driver": 18, "isolator": 3} if inherited is not None else None}


def dry_run_manifest(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    structure = validate_v8_protocol(protocol, protocol_bytes_hash=protocol_bytes_hash)
    states = _resolve_v8(protocol, protocol_bytes_hash=protocol_bytes_hash)
    bindings = legal_replay_binding_templates(protocol)
    return {"schema_id": SCHEMA_ID + ".resolved_manifest", "schema_version": SCHEMA_VERSION, "state_count": len(states), "stage_counts": structure["stage_counts"], "resolved_state_digest": structure["resolved_state_digest"], "states": [state.to_dict() for state in states], "legal_replay_binding_count": len(bindings), "legal_replay_bindings": [copy.deepcopy(dict(binding)) for binding in bindings], "mujoco_model_created": False}


def _contact_plan_with_normal_fields(plan: Any) -> dict[str, Any]:
    value = plan.to_dict()
    requested = list(value.get("requested_fields", ()))
    requested.extend(("contact.normal_impact.duration_s", "contact.normal_impact.trace_dt_s", "contact.normal_impact.terminal_support_force_min_N", "contact.normal_impact.incline_coarse_angles_rad", "contact.normal_impact.incline_binary_refinement_iterations", "contact.normal_impact.incline_resolution_rad"))
    value["requested_fields"] = requested
    return value


def adapter_contract(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    def encode(state: ResolvedProbeState, plan, kind: str) -> dict[str, Any]:
        if state.stage == "contact" or (
            kind == "legal_replay_component" and state.replay is not None and state.replay.selected_component == "contact"
        ):
            return _contact_plan_with_normal_fields(plan)
        return plan.to_dict()

    return adapter_contract_payload(
        _resolve_v8(protocol, protocol_bytes_hash=protocol_bytes_hash),
        resolve_replay_binding_templates(protocol),
        schema_id=SCHEMA_ID + ".adapter_contract",
        schema_version=SCHEMA_VERSION,
        plan_payload=encode,
    )



def _normal_from_trace(normal: Any, *, timestep_s: float, duration_s: float, tail_window_s: float, penetration_limit_m: float, velocity_limit_m_s: float, angular_limit_rad_s: float, support_force_min_N: float) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(normal, Mapping) or not isinstance(normal.get("trace"), list) or len(normal["trace"]) < 2:
        return {}, ["normal-impact trace is missing"]
    trace = normal["trace"]
    times = []
    speeds = []
    angular = []
    vertical = []
    penetration = []
    warning_total = 0
    for index, row in enumerate(trace):
        try:
            time_s = float(row["time_s"])
            linear = np.asarray(row["linear_velocity_m_s"], dtype=float)
            omega = np.asarray(row["angular_velocity_rad_s"], dtype=float)
            contacts = row["contacts"]
            if linear.shape != (3,) or omega.shape != (3,) or not isinstance(contacts, Mapping):
                raise ValueError("wrong raw vector/contact shape")
            value = float(contacts["penetration_m"])
            if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(omega)) or not np.isfinite(value) or value < 0.0:
                raise ValueError("non-finite raw normal trace value")
            for contact in contacts.get("contacts", ()):
                if not isinstance(contact, Mapping) or contact.get("interface") != "can_open_worktable" or {str(contact.get("geom1")), str(contact.get("geom2"))} != {"can_g0", "table_collision"}:
                    raise ValueError("wrong named normal-contact interface")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"normal trace row {index} is malformed: {exc}")
            continue
        times.append(time_s)
        speeds.append(float(np.linalg.norm(linear)))
        angular.append(float(np.linalg.norm(omega)))
        vertical.append(float(linear[2]))
        penetration.append(value)
        warning_total += int(np.sum(np.asarray(row.get("warning_number", ()), dtype=int)))
    if len(times) != len(trace):
        return {}, errors or ["normal trace rows are incomplete"]
    expected = np.arange(len(trace), dtype=float) * timestep_s
    if not np.allclose(times, expected, rtol=0.0, atol=1.0e-12):
        errors.append("normal trace timestamps are not aligned to the declared timestep")
    if not np.isclose(times[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(times[-1], duration_s, rtol=0.0, atol=1.0e-12):
        errors.append("normal trace does not cover exactly 0.50 s")
    tail = np.asarray([index for index, time_s in enumerate(times) if time_s >= duration_s - tail_window_s - 1.0e-12], dtype=int)
    if not len(tail):
        errors.append("normal final-50-ms window is empty")
        tail = np.asarray([len(trace) - 1], dtype=int)
    terminal = trace[-1]
    terminal_contacts = terminal.get("contacts", {})
    terminal_support = bool(int(terminal_contacts.get("contact_count", 0)) > 0 and float(terminal_contacts.get("normal_force_N", 0.0)) >= support_force_min_N)
    energy = [float(row.get("energy", {}).get("mechanical_energy_J", math.nan)) for row in trace]
    release_energy = max(energy[0], 1.0e-12)
    post_impact_energy = [value / release_energy for index, value in enumerate(energy) if index and trace[index]["contacts"].get("contact_count", 0) > 0]
    result = {
        "release_time_s": times[0],
        "first_impact_time_s": next((time_s for time_s, row in zip(times, trace) if row["contacts"].get("contact_count", 0) > 0), None),
        "duration_s": times[-1],
        "trace_dt_s": timestep_s,
        "trace_sample_count": len(trace),
        "terminal_linear_velocity_m_s": speeds[-1],
        "terminal_angular_velocity_rad_s": angular[-1],
        "terminal_vertical_velocity_m_s": vertical[-1],
        "terminal_contact_count": int(terminal_contacts.get("contact_count", 0)),
        "terminal_normal_force_N": float(terminal_contacts.get("normal_force_N", 0.0)),
        "terminal_named_table_support": terminal_support,
        "tail_window_s": tail_window_s,
        "tail_window_max_linear_velocity_m_s": float(np.max(np.asarray(speeds)[tail])),
        "tail_window_rms_linear_velocity_m_s": float(np.sqrt(np.mean(np.asarray(speeds)[tail] ** 2))),
        "tail_window_max_angular_velocity_rad_s": float(np.max(np.asarray(angular)[tail])),
        "tail_window_rms_angular_velocity_rad_s": float(np.sqrt(np.mean(np.asarray(angular)[tail] ** 2))),
        "peak_downward_vertical_speed_m_s": float(max(0.0, -min(vertical))),
        "peak_rebound_vertical_speed_m_s": float(max(0.0, max(vertical))),
        "maximum_penetration_m": float(max(penetration)),
        "normal_energy_retained_fraction_max": float(max(post_impact_energy, default=0.0)),
        "warning_count": warning_total,
    }
    return result, errors


def _level_from_raw(level: Any, *, speed_limit_m_s: float, displacement_limit_m: float, support_force_min_N: float) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(level, Mapping) or not isinstance(level.get("hold_trace"), list) or not level["hold_trace"]:
        return {}, ["level support control is missing"]
    rows = level["hold_trace"]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("contacts"), Mapping):
            errors.append(f"level hold row {index} is malformed")
            continue
        if any(not np.isfinite(float(row.get(key, math.nan))) for key in ("horizontal_speed_m_s", "horizontal_displacement_m")):
            errors.append(f"level hold row {index} has non-finite motion")
        if int(row["contacts"].get("contact_count", 0)) < 0:
            errors.append(f"level hold row {index} has invalid contact count")
    terminal = rows[-1]["contacts"]
    result = {"terminal_contact_count": int(terminal.get("contact_count", 0)), "terminal_normal_force_N": float(terminal.get("normal_force_N", 0.0)), "maximum_horizontal_speed_m_s": max(float(row["horizontal_speed_m_s"]) for row in rows), "maximum_horizontal_displacement_m": max(float(row["horizontal_displacement_m"]) for row in rows), "warning_count": int(sum(int(value) for value in rows[-1].get("warning_number", ())))}
    result["passed"] = bool(not errors and result["terminal_contact_count"] > 0 and result["terminal_normal_force_N"] >= support_force_min_N and result["maximum_horizontal_speed_m_s"] <= speed_limit_m_s and result["maximum_horizontal_displacement_m"] <= displacement_limit_m and result["warning_count"] == 0)
    return result, errors


def _incline_from_raw(incline: Any, *, resolution_rad: float, displacement_gate_m: float) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(incline, Mapping) or not isinstance(incline.get("angle_records"), list) or not incline["angle_records"]:
        return {}, ["incline calibration is missing"]
    rows = incline["angle_records"]
    angles = [float(row.get("angle_rad", math.nan)) for row in rows if isinstance(row, Mapping)]
    if not angles or not any(np.isclose(value, 0.0, rtol=0.0, atol=1.0e-12) for value in angles):
        errors.append("incline calibration lacks a zero-angle record")
    stable = [float(row["angle_rad"]) for row in rows if isinstance(row, Mapping) and float(row.get("horizontal_displacement_m", math.inf)) <= displacement_gate_m]
    sliding = [float(row["angle_rad"]) for row in rows if isinstance(row, Mapping) and float(row.get("horizontal_displacement_m", -math.inf)) > displacement_gate_m]
    lower = max(stable) if stable else None
    upper = min(sliding) if sliding else None
    width = None if lower is None or upper is None else upper - lower
    if lower is None or not np.isclose(lower if lower == 0.0 else 0.0, 0.0, rtol=0.0, atol=1.0e-12):
        # A non-zero stable lower bound is acceptable, but a level record is
        # mandatory and is checked explicitly above.  Keep this branch only
        # for malformed synthetic traces.
        if not any(np.isclose(value, 0.0, rtol=0.0, atol=1.0e-12) for value in stable):
            errors.append("incline calibration has no stable zero-angle control")
    result = {"lower_stable_angle_rad": lower, "upper_sliding_angle_rad": upper, "bracket_width_rad": width, "declared_resolution_rad": resolution_rad, "slip_displacement_threshold_m": displacement_gate_m, "angle_count": len(rows)}
    result["passed"] = bool(not errors and lower is not None and upper is not None and lower < upper and width <= resolution_rad)
    return result, errors


def _v8_contact_gates(evidence: Mapping[str, Any], state: ResolvedProbeState) -> tuple[dict[str, bool], dict[str, Any], list[str]]:
    if state.contact is None:
        return {}, {}, ["V8 contact child is missing"]
    gates = state.contact.hard_gates
    normal, normal_errors = _normal_from_trace(evidence.get("normal_impact"), timestep_s=state.common.physics_timestep_s, duration_s=float(state.contact.probes["recovery_duration_s"]), tail_window_s=float(state.contact.probes["tail_window_s"]), penetration_limit_m=float(gates["maximum_illegal_penetration_m"]), velocity_limit_m_s=float(gates["recovery_velocity_max_m_s"]), angular_limit_rad_s=float(gates["recovery_angular_velocity_max_rad_s"]), support_force_min_N=float(gates["terminal_support_force_min_N"]))
    level, level_errors = _level_from_raw(evidence.get("level_control"), speed_limit_m_s=float(gates["level_horizontal_speed_max_m_s"]), displacement_limit_m=float(gates["level_horizontal_displacement_max_m"]), support_force_min_N=float(gates["terminal_support_force_min_N"]))
    displacement_gate = float(gates["incline_slip_displacement_m"])
    incline, incline_errors = _incline_from_raw(evidence.get("incline_calibration"), resolution_rad=float(gates["incline_bracket_resolution_rad"]), displacement_gate_m=displacement_gate) if level.get("passed") else ({}, ["incline was not run because level control failed"])
    finger, finger_errors = _finger_from_trace(evidence.get("finger_load"), gates=gates)
    compiled_ok, compiled_errors = _compiled_contact_matches(evidence.get("compiled_contact_profile", {}), state)
    compatibility = evidence.get("v7_compatibility", {})
    static = compatibility.get("static_support", {})
    slip = compatibility.get("single_axis_slip", {})
    convergence = compatibility.get("timestep_convergence", {})
    convergence_error = max((float(row.get("relative_force_error", math.inf)) for row in convergence.get("records", ()) if isinstance(row, Mapping)), default=math.inf)
    envelope = evidence.get("finger_force_envelope", {})
    checks = {
        "compiled_profile": compiled_ok,
        "static_support": float(static.get("normal_force_N", 0.0)) >= float(gates["static_support_force_min_N"]),
        "slip": np.isfinite(float(slip.get("final_speed_m_s", math.nan))) and float(slip.get("final_speed_m_s", math.inf)) <= float(gates["slip_final_speed_max_m_s"]),
        "normal_trace": not normal_errors,
        "normal_terminal_support": bool(normal.get("terminal_named_table_support")) and float(normal.get("terminal_normal_force_N", 0.0)) >= float(gates["terminal_support_force_min_N"]),
        "recovery_velocity": float(normal.get("terminal_linear_velocity_m_s", math.inf)) <= float(gates["recovery_velocity_max_m_s"]) and float(normal.get("tail_window_max_linear_velocity_m_s", math.inf)) <= float(gates["recovery_velocity_max_m_s"]),
        "recovery_angular_velocity": float(normal.get("terminal_angular_velocity_rad_s", math.inf)) <= float(gates["recovery_angular_velocity_max_rad_s"]) and float(normal.get("tail_window_max_angular_velocity_rad_s", math.inf)) <= float(gates["recovery_angular_velocity_max_rad_s"]),
        "penetration": float(normal.get("maximum_penetration_m", math.inf)) <= float(gates["maximum_illegal_penetration_m"]),
        "level_control": not level_errors and level.get("passed") is True,
        "incline_bracket": not incline_errors and incline.get("passed") is True,
        "finger_trace": not finger_errors,
        "finger_force_minimum": float(finger.get("minimum_normal_force_N", 0.0)) >= float(gates["finger_force_min_N"]),
        "finger_force_maximum": float(finger.get("maximum_normal_force_N", math.inf)) <= float(gates["finger_force_max_N"]) and float(envelope.get("finite_force_envelope_N", math.nan)) == float(gates["finger_force_max_N"]),
        "finger_penetration": float(finger.get("maximum_penetration_m", math.inf)) <= float(gates["finger_penetration_max_m"]),
        "warnings": int(normal.get("warning_count", 1)) + int(level.get("warning_count", 1)) <= int(gates["warning_count_max"]),
        "timestep_convergence": convergence_error <= float(gates["timestep_trace_relative_error_max"]),
    }
    derived = {"normal": normal, "normal_errors": normal_errors, "level": level, "level_errors": level_errors, "incline": incline, "incline_errors": incline_errors, "finger": finger, "finger_errors": finger_errors, "compiled_errors": compiled_errors, "convergence_max_relative_force_error": convergence_error}
    return checks, derived, normal_errors + level_errors + incline_errors + finger_errors + compiled_errors


def _v8_contact_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    if state.stage not in {"contact", "replay"} or state.contact is None:
        raise V8ProtocolError("V8 contact adapter received a state without a contact child")
    if state.stage == "replay" and state.replay is not None and state.replay.selected_component != "contact":
        raise V8ProtocolError("V8 contact probe was requested for a non-contact replay binding")
    candidate = _contact_candidate_mapping(state)
    profile = _profile_from_state(state)
    from robosuite.scripts.shakebench_select_physics import _contact_candidate_probe

    compatibility = _contact_candidate_probe(profile, candidate, run_expensive=True, recovery_duration_s=RECOVERY_DURATION_S, reset_drop_velocity=True)
    normal = run_normal_impact_trace(candidate, duration_s=RECOVERY_DURATION_S)
    level = run_level_control(candidate)
    incline = run_incline_calibration(candidate, level) if level.get("passed") is True else {"blocked_by_level_control": True, "angle_records": []}
    from robosuite.scripts.shakebench_diagnose_contact_recovery_v7 import _declared_finger_load_trace, _force_envelope_audit
    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    probe_env = VibrationPickPlaceCan(robots="Panda", physics_profile=profile, model_timestep=state.common.physics_timestep_s, target_container_friction=(0.30, state.contact.torsional_mu, state.contact.rolling_mu), has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False, control_freq=20, horizon=20, seed=17)
    try:
        force_envelope = _force_envelope_audit(probe_env.sim.model._model, finger_actuator_names=("gripper0_right_gripper_finger_joint1", "gripper0_right_gripper_finger_joint2"))
        probe_env.reset()
        finger = _declared_finger_load_trace(probe_env, duration_s=float(state.contact.probes["finger_load_duration_s"]), envelope=force_envelope)
        compiled = _compile_contact_audit(state)
    finally:
        probe_env.close()
    physics = {"candidate_profile": candidate, "compiled_contact_profile": compiled, "v7_compatibility": compatibility, "normal_impact": normal, "level_control": level, "incline_calibration": incline, "finger_force_envelope": force_envelope, "finger_load": finger}
    checks, derived, errors = _v8_contact_gates(physics, state)
    return {"candidate_id": state.contact.candidate_id, "physics": physics, "recomputed_numeric_gates": checks, "derived_numeric_metrics": derived, "passed": bool(all(checks.values()) and not errors), "exclusion_reasons": [] if all(checks.values()) and not errors else ["contact_hard_gate_failed"]}


def _verify_v7_noncontact_evidence() -> dict[str, Any]:
    from robosuite.scripts.shakebench_select_physics_v7 import _verify_v6_inherited_evidence, _verify_v6_reference_rows

    v7_protocol, v7_name, v7_hash = __import__("robosuite.scripts.shakebench_select_physics_v7", fromlist=["load_v7_protocol"]).load_v7_protocol(_asset(V7_PROTOCOL_FILENAME))
    __import__("robosuite.scripts.shakebench_select_physics_v7", fromlist=["validate_v7_protocol"]).validate_v7_protocol(v7_protocol, protocol_bytes_hash=v7_hash)
    v7_selected_path = _asset(V7_SELECTED_FILENAME)
    v7_status_path = _asset(V7_STATUS_FILENAME)
    v7_feasibility_path = _asset(V7_FEASIBILITY_FILENAME)
    selected = json.loads(v7_selected_path.read_text(encoding="utf-8"))
    status = json.loads(v7_status_path.read_text(encoding="utf-8"))
    feasibility = json.loads(v7_feasibility_path.read_text(encoding="utf-8"))
    errors = []
    if status.get("status") != "BLOCKED" or selected.get("status") != "BLOCKED":
        errors.append("V7 completed BLOCKED status is not preserved")
    ids = selected.get("selection", {}).get("selected_candidate_ids", {})
    if ids.get("driver") != "dt_nominal" or ids.get("isolator") != "low_frequency_damped" or ids.get("contact") is not None:
        errors.append("V8 attempted to inherit a V7 contact selection")
    if not verify_payload_hash(selected) or not verify_payload_hash(status) or not verify_payload_hash(feasibility):
        errors.append("V7 selected/status/feasibility payload hash failed")
    if selected.get("protocol", {}).get("bytes_sha256") != v7_hash or status.get("protocol", {}).get("bytes_sha256") != v7_hash or feasibility.get("protocol", {}).get("bytes_sha256") != v7_hash:
        errors.append("V7 protocol hash failed")
    inherited = _verify_v6_inherited_evidence()
    errors.extend(_verify_v6_reference_rows(selected, v7_selected_path, inherited))
    if errors:
        raise V8ProtocolError("V7 non-contact inheritance failed: " + "; ".join(errors))
    return {"protocol_path": v7_name, "protocol_sha256": v7_hash, "selected_path": v7_selected_path.name, "selected_sha256": file_sha256(v7_selected_path), "status_path": v7_status_path.name, "status_sha256": file_sha256(v7_status_path), "v6_inherited_raw_files": inherited["raw_files"], "selected_driver": "dt_nominal", "selected_isolator": "low_frequency_damped"}


def _raw_payload(state: ResolvedProbeState, evidence: Mapping[str, Any], *, attempt: int, retry_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    identity = state.common.protocol_identity
    payload = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": state.stage, "state_id": state.state_id, "protocol_sha256_bytes": identity.get("bytes_sha256"), "protocol_sha256_normalized": identity.get("normalized_sha256"), "resolved_state_digest": state.resolved_state_digest, "attempt": attempt, "retry_ledger": copy.deepcopy(retry_ledger), "evidence": dict(evidence)}
    payload["payload_sha256"] = payload_hash(payload)
    return payload


def _verify_existing_raw(path: Path, state: ResolvedProbeState) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V8EvidenceError(f"cannot read V8 raw artifact: {path.name}") from exc
    identity = state.common.protocol_identity
    if payload.get("state_id") != state.state_id or payload.get("stage") != state.stage or payload.get("resolved_state_digest") != state.resolved_state_digest or payload.get("protocol_sha256_bytes") != identity.get("bytes_sha256") or payload.get("protocol_sha256_normalized") != identity.get("normalized_sha256") or not verify_payload_hash(payload):
        raise V8EvidenceError(f"V8 raw provenance/hash mismatch: {state.state_id}")
    return payload


def run_v8_stage(protocol: Mapping[str, Any], *, stage: str, output_dir: str | Path, probe: Any, protocol_bytes_hash: str, selection_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    states = [state for state in _resolve_v8(protocol, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context) if state.stage == stage]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for state in states:
        path = output / state.output
        if path.is_file():
            records.append(_verify_existing_raw(path, state))
            continue
        ledger = []
        for attempt in (1, 2):
            try:
                evidence = probe(state)
                if not isinstance(evidence, Mapping):
                    raise V8EvidenceError(f"{stage} probe returned non-mapping evidence")
                payload = _raw_payload(state, evidence, attempt=attempt, retry_ledger=ledger)
                write_json_atomic(path, payload)
                records.append(payload)
                break
            except (V8ProtocolError, V8ProtocolStateError, V8EvidenceError, AdapterContractError):
                raise
            except Exception as exc:
                ledger.append({"attempt": attempt, "state_config_identical": True, "exception_type": type(exc).__name__, "exception_message": str(exc), "failure_taxonomy": "infrastructure_failure"})
        else:
            payload = _raw_payload(state, {"status": "infrastructure_failure_repeated", "passed": False}, attempt=2, retry_ledger=ledger)
            write_json_atomic(path, payload)
            raise V8EvidenceError(f"repeated infrastructure failure: {state.state_id}")
    return records


def _v8_replay_worker(protocol_path: str, state_id: str) -> int:
    protocol, _, protocol_hash = load_v8_protocol(protocol_path)
    validate_v8_protocol(protocol, protocol_bytes_hash=protocol_hash)
    state = next(state for state in _resolve_v8(protocol, protocol_bytes_hash=protocol_hash) if state.state_id == state_id)
    print(json.dumps(_v6_replay_probe(state), ensure_ascii=False, sort_keys=True))
    return 0


def _subprocess_replay_probe(protocol_path: str | Path, state: ResolvedProbeState) -> Mapping[str, Any]:
    completed = subprocess.run([sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v8", "--worker", "--protocol", str(Path(protocol_path).resolve()), "--state-id", state.state_id], cwd=str(Path(__file__).resolve().parents[2]), env={**os.environ, "PYTHONUNBUFFERED": "1"}, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise V8EvidenceError(completed.stderr[-4000:] or completed.stdout[-4000:])
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "trace_digest" in value:
            return value
    raise V8EvidenceError("V8 replay worker did not emit a trace record")


def _replay_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("evidence", {}).get("selected_component"))].append(record)
    result = {}
    for group, rows in groups.items():
        evidence = [row.get("evidence", {}) for row in sorted(rows, key=lambda item: int(item.get("evidence", {}).get("process_index", 0)))]
        result[group] = {"process_count": len(evidence), "process_indices": [row.get("process_index") for row in evidence], "trace_digests": [row.get("trace_digest") for row in evidence], "metric_digests": [row.get("metric_digest") for row in evidence], "trace_fields": evidence[0].get("trace_fields", []) if evidence else [], "complete_trace": all(row.get("complete_trace") is True for row in evidence)}
    return result


def _v8_contact_score(record: Mapping[str, Any], state: ResolvedProbeState) -> tuple[float, float, str]:
    if state.contact is None:
        return math.inf, math.inf, state.candidate_id
    _, metrics, _ = _v8_contact_gates(record.get("evidence", {}).get("physics", {}), state)
    normal = metrics.get("normal", {})
    incline = metrics.get("incline", {})
    finger = metrics.get("finger", {})
    values = {"normal_peak_rebound_vertical_speed_m_s": float(normal.get("peak_rebound_vertical_speed_m_s", math.inf)), "normal_energy_retained_fraction": float(normal.get("normal_energy_retained_fraction_max", math.inf)), "recovery_velocity_m_s": float(normal.get("tail_window_max_linear_velocity_m_s", math.inf)), "incline_bracket_width_rad": float(incline.get("bracket_width_rad", math.inf)), "maximum_penetration_m": float(normal.get("maximum_penetration_m", math.inf)), "finger_force_relative_error": abs(float(finger.get("maximum_normal_force_N", math.inf)) / 40.0 - 1.0)}
    limits = {"normal_peak_rebound_vertical_speed_m_s": 1.0, "normal_energy_retained_fraction": 1.0, "recovery_velocity_m_s": float(state.contact.hard_gates["recovery_velocity_max_m_s"]), "incline_bracket_width_rad": float(state.contact.hard_gates["incline_bracket_resolution_rad"]), "maximum_penetration_m": float(state.contact.hard_gates["maximum_illegal_penetration_m"]), "finger_force_relative_error": 1.0}
    weights = state.contact.scoring.get("weights", {})
    distance = sum(float(weights.get(key, 1.0)) * values[key] / max(limits[key], 1.0e-12) for key in values if key in weights)
    margin = min(float(state.contact.hard_gates["recovery_velocity_max_m_s"]) / max(values["recovery_velocity_m_s"], 1.0e-12), float(state.contact.hard_gates["maximum_illegal_penetration_m"]) / max(values["maximum_penetration_m"], 1.0e-12))
    return float(distance), float(-margin), state.contact.candidate_id


def _artifact_rows(states: Mapping[str, ResolvedProbeState], records: Sequence[Mapping[str, Any]], output: Path) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        state = states.get(str(record.get("state_id")))
        if state is None:
            raise V8EvidenceError(f"record has unknown V8 state: {record.get('state_id')}")
        path = output / state.output
        rows.append({"stage": state.stage, "state_id": state.state_id, "path": path.name, "sha256": file_sha256(path), "resolved_state_digest": state.resolved_state_digest})
    return rows


def _profile_from_v8_state(state: ResolvedProbeState, *, protocol_hash: str, evidence_manifest_hash: str) -> dict[str, Any]:
    payload = _profile_from_state(state).to_dict()
    payload.update({"profile_id": "shakebench.official.physics.v1", "status": "official_immutable", "scoreable": True, "selection_basis": "phase06r7_v8_normal_impact_physics_only", "protocol_file": PROTOCOL_FILENAME, "protocol_sha256": protocol_hash, "evidence_manifest_sha256": evidence_manifest_hash, "freeze_commit": "phase_06r7_v8_normal_impact_registration"})
    payload["profile_sha256"] = physics_profile_hash(payload)
    return PhysicsProfile(payload=payload, source="V8 independently verified selection", profile_sha256=payload["profile_sha256"]).assert_valid().to_dict()


def _write_status(output: Path, *, status: str, protocol_hash: str, normalized_hash: str, feasibility_hash: str | None, adapter_digest: str | None, reason: str | None = None, selected_path: Path | None = None, profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    taxonomy = None if status == "PASS" else ("invalid_protocol_configuration" if reason and reason.startswith("invalid_protocol_configuration") else ("evidence_integrity_failure" if reason and reason.startswith("evidence_integrity_failure") else "physics_gate_failure"))
    value: dict[str, Any] = {"schema_id": "shakebench.phase06r7.v8.selection_status", "schema_version": 1, "status": status, "failure_taxonomy": taxonomy, "reason": reason, "blocking_stage": None if status == "PASS" else "contact", "protocol": {"path": PROTOCOL_FILENAME, "bytes_sha256": protocol_hash, "normalized_sha256": normalized_hash}, "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash}, "adapter_contract_digest": adapter_digest, "official_profile_publication": "PASS" if status == "PASS" else "forbidden", "phase07": "authorized_after_independent_verification" if status == "PASS" else "forbidden"}
    if selected_path is not None:
        value["selection_artifact"] = {"path": selected_path.name, "sha256": file_sha256(selected_path)}
    if profile is not None:
        path = output / OFFICIAL_PROFILE_FILENAME
        value["official_profile"] = {"path": path.name, "file_sha256": file_sha256(path), "profile_sha256": profile["profile_sha256"]}
    value["payload_sha256"] = payload_hash(value)
    write_json_atomic(output / STATUS_FILENAME, value)
    return value


def _last_json_line(output: str) -> Mapping[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise V8ProtocolError("pure V8 command did not emit JSON")


def write_feasibility_artifact(protocol_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    protocol, protocol_name, protocol_hash = load_v8_protocol(protocol_path)
    command_specs = (("validate", "--validate-protocol"), ("dry_run", "--dry-run-manifest"), ("adapter_contract", "--adapter-contract"))
    results: dict[str, Mapping[str, Any]] = {}
    exit_codes: dict[str, int] = {}
    for label, flag in command_specs:
        completed = subprocess.run([sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v8", "--protocol", str(protocol_path), flag], cwd=str(Path(__file__).resolve().parents[2]), env={**os.environ, "PYTHONUNBUFFERED": "1"}, capture_output=True, text=True, check=False)
        exit_codes[label] = int(completed.returncode)
        results[label] = _last_json_line(completed.stdout or completed.stderr)
    if any(code != 0 for code in exit_codes.values()):
        raise V8ProtocolError(f"V8 pure registration command failed: {exit_codes}")
    structure = validate_v8_protocol(protocol, protocol_bytes_hash=protocol_hash)
    contract = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
    states = _resolve_v8(protocol, protocol_bytes_hash=protocol_hash)
    measurement = _mapping(protocol.get("measurement"), "measurement")
    capacity_path = _asset(str(measurement["capacity_preflight"]))
    proofs = []
    for state in states:
        ratio = state.common.control_period_s / state.common.physics_timestep_s
        proofs.append({"state_id": state.state_id, "control_period_over_dt": ratio, "integer": state.common.control_steps, "passed": bool(np.isclose(ratio, state.common.control_steps, rtol=0.0, atol=1.0e-12))})
    artifact = {"schema_id": SCHEMA_ID + ".feasibility", "schema_version": 1, "status": "PASS", "protocol": {"path": protocol_name, "bytes_sha256": protocol_hash, "normalized_sha256": sha256_json(protocol), "immutable_after_registration": protocol.get("immutable_after_registration")}, "resolved_state_digest": structure["resolved_state_digest"], "resolved_states": [state.to_dict() for state in states], "state_count": len(states), "stage_counts": structure["stage_counts"], "legal_replay_binding_count": len(resolve_replay_binding_templates(protocol)), "adapter_contract_digest": contract["adapter_contract_digest"], "adapter_contract_state_count": contract["state_count"], "adapter_contract_prepared_plan_count": contract["prepared_plan_count"], "integer_ratio_proofs": proofs, "capacity_evidence": {"path": capacity_path.name, "sha256": file_sha256(capacity_path), "selection_input": False}, "v7_inheritance": {"protocol": V7_PROTOCOL_FILENAME, "protocol_sha256": structure["v7_protocol_sha256"], "selected_driver": "dt_nominal", "selected_isolator": "low_frequency_damped", "v7_contact_selection_inherited": False}, "commands": {label: {"command": f"python -m robosuite.scripts.shakebench_select_physics_v8 --protocol {protocol_path} {flag}", "exit_code": exit_codes[label], "result_digest": sha256_json(results[label])} for label, flag in command_specs}}
    artifact["payload_sha256"] = payload_hash(artifact)
    destination = Path(output_path) if output_path is not None else Path(models.assets_root) / FEASIBILITY_FILENAME
    write_json_atomic(destination, artifact)
    return artifact


def _run_selection_artifacts(output: Path, protocol: Mapping[str, Any], protocol_hash: str, structure: Mapping[str, Any], contract: Mapping[str, Any], inherited: Mapping[str, Any], contact_records: Sequence[Mapping[str, Any]], parity_records: Sequence[Mapping[str, Any]], replay_records: Sequence[Mapping[str, Any]], states: Mapping[str, ResolvedProbeState], default_contact: str, feasibility_hash: str | None) -> dict[str, Any]:
    stage_records = list(contact_records) + list(parity_records) + list(replay_records)
    parity_state = next(state for state in states.values() if state.stage == "parity")
    parity_record = parity_records[0] if parity_records else {}
    parity_ok = bool(parity_record and _parity_eligible(parity_record, parity_state))
    parity_artifact = {"schema_id": SCHEMA_ID + ".parity", "schema_version": SCHEMA_VERSION, "status": "PASS" if parity_ok else "BLOCKED", "protocol_sha256_bytes": protocol_hash, "protocol_sha256_normalized": sha256_json(protocol), "resolved_state_digest": parity_state.resolved_state_digest, "raw_file": {"path": parity_state.output, "state_id": parity_state.state_id}, "evidence": parity_record.get("evidence", {})}
    parity_artifact["payload_sha256"] = payload_hash(parity_artifact)
    parity_path = output / PARITY_FILENAME
    write_json_atomic(parity_path, parity_artifact)
    replay_summary = _replay_groups(replay_records)
    replay_ok = set(replay_summary) >= REPLAY_GROUPS and all(row["process_count"] == 3 and len(set(row["trace_digests"])) == 1 and len(set(row["metric_digests"])) == 1 and row["complete_trace"] for row in replay_summary.values())
    determinism_artifact = {"schema_id": SCHEMA_ID + ".replay_determinism", "schema_version": SCHEMA_VERSION, "status": "PASS" if replay_ok else "BLOCKED", "protocol_sha256_bytes": protocol_hash, "protocol_sha256_normalized": sha256_json(protocol), "resolved_state_digest": structure["resolved_state_digest"], "independent_process_count": 3, "groups": replay_summary, "same_process_reset_used": False}
    determinism_artifact["payload_sha256"] = payload_hash(determinism_artifact)
    determinism_path = output / DETERMINISM_FILENAME
    write_json_atomic(determinism_path, determinism_artifact)
    stage_rows = _artifact_rows(states, stage_records, output)
    contact_details = {}
    eligible = []
    for record in contact_records:
        state = states[str(record["state_id"])]
        checks, derived, errors = _v8_contact_gates(record.get("evidence", {}).get("physics", {}), state)
        contact_details[state.candidate_id] = {"checks": checks, "derived": derived, "errors": errors}
        if all(checks.values()) and not errors:
            eligible.append(state.candidate_id)
    ranked = sorted(eligible, key=lambda candidate_id: _v8_contact_score(next(record for record in contact_records if states[str(record["state_id"])].candidate_id == candidate_id), next(state for state in states.values() if state.stage == "contact" and state.candidate_id == candidate_id)))
    winning = ranked[0] if ranked else None
    overall = winning is not None and winning == default_contact and parity_ok and replay_ok
    selected_contact = winning if overall else None
    excluded = [{"stage": "contact", "candidate_id": candidate_id, "reason": "contact_hard_gate_failed", "recomputed_gates": contact_details[candidate_id]["checks"], "diagnostic_errors": contact_details[candidate_id]["errors"], "metrics": contact_details[candidate_id]["derived"]} for candidate_id in sorted(CONTACT_IDS) if candidate_id not in eligible]
    reason = None if overall else ("physics_gate_failure: no contact candidate passed" if winning is None else ("evidence_integrity_failure: winning contact differs from registered parity/replay binding" if winning != default_contact else "evidence_integrity_failure: final V8 parity/replay gates failed"))
    selected = {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "status": "PASS" if overall else "BLOCKED", "physics_only": True, "protocol": {"path": PROTOCOL_FILENAME, "bytes_sha256": protocol_hash, "normalized_sha256": sha256_json(protocol), "status": protocol.get("status")}, "resolved_state_digest": structure["resolved_state_digest"], "adapter_contract_digest": contract["adapter_contract_digest"], "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash}, "selection": {"selected_candidate_ids": {"driver": inherited["selected_driver"], "isolator": inherited["selected_isolator"], "contact": selected_contact}, "physics_only": True, "task_success_used": False, "reward_used": False, "controller_outcome_used": False}, "candidate_counts": {"driver": 18, "isolator": 3, "contact": len(contact_records)}, "eligible_counts": {"driver": 3, "isolator": 1, "contact": len(eligible)}, "inherited_v7_noncontact": {"selected_driver": inherited["selected_driver"], "selected_isolator": inherited["selected_isolator"], "v7_contact_selection_inherited": False}, "raw_files": stage_rows, "official_profile": None, "official_profile_alignment": False, "replay_groups": sorted(REPLAY_GROUPS), "blocking_reason": reason, "parity_artifact": {"path": parity_path.name, "sha256": file_sha256(parity_path)}, "determinism_artifact": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)}, "excluded": excluded}
    selected_path = output / SELECTED_FILENAME
    selected["payload_sha256"] = payload_hash(selected)
    write_json_atomic(selected_path, selected)
    excluded_payload = {"schema_id": SCHEMA_ID + ".excluded", "schema_version": SCHEMA_VERSION, "status": "PASS", "protocol_sha256_bytes": protocol_hash, "selected_candidate_ids": selected["selection"]["selected_candidate_ids"], "excluded": excluded, "physics_only": True}
    excluded_payload["payload_sha256"] = payload_hash(excluded_payload)
    write_json_atomic(output / EXCLUDED_FILENAME, {**excluded_payload, "payload_sha256": payload_hash(excluded_payload)})
    if not overall:
        status = _write_status(output, status="BLOCKED", protocol_hash=protocol_hash, normalized_hash=sha256_json(protocol), feasibility_hash=feasibility_hash, adapter_digest=contract["adapter_contract_digest"], reason=reason, selected_path=selected_path)
        return {"status": status, "selected": selected, "contact_details": contact_details}
    evidence_manifest_hash = sha256_json({"raw_files": stage_rows, "parity": {"path": parity_path.name, "sha256": file_sha256(parity_path)}, "determinism": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)}})
    selected_state = next(state for state in states.values() if state.stage == "contact" and state.candidate_id == selected_contact)
    profile = _profile_from_v8_state(selected_state, protocol_hash=protocol_hash, evidence_manifest_hash=evidence_manifest_hash)
    profile_path = output / OFFICIAL_PROFILE_FILENAME
    try:
        import yaml

        temporary = profile_path.with_name(profile_path.name + ".tmp")
        temporary.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
        temporary.replace(profile_path)
    except ImportError:
        write_json_atomic(profile_path, profile)
    selected["official_profile"] = OFFICIAL_PROFILE_FILENAME
    selected["official_profile_alignment"] = True
    selected["payload_sha256"] = payload_hash(selected)
    write_json_atomic(selected_path, selected)
    verification = verify_v8_selection_artifact(selected_path, protocol_path=output / PROTOCOL_FILENAME if (output / PROTOCOL_FILENAME).exists() else Path(models.assets_root) / PROTOCOL_FILENAME, require_published_status=False)
    if verification.get("passed") is not True:
        try:
            profile_path.unlink()
        except FileNotFoundError:
            pass
        selected["status"] = "BLOCKED"
        selected["selection"]["selected_candidate_ids"]["contact"] = None
        selected["official_profile"] = None
        selected["official_profile_alignment"] = False
        selected["blocking_reason"] = "evidence_integrity_failure: independent V8 verifier failed"
        selected["payload_sha256"] = payload_hash(selected)
        write_json_atomic(selected_path, selected)
        status = _write_status(output, status="BLOCKED", protocol_hash=protocol_hash, normalized_hash=sha256_json(protocol), feasibility_hash=feasibility_hash, adapter_digest=contract["adapter_contract_digest"], reason=selected["blocking_reason"], selected_path=selected_path)
        return {"status": status, "verification": verification, "selected": selected}
    status = _write_status(output, status="PASS", protocol_hash=protocol_hash, normalized_hash=sha256_json(protocol), feasibility_hash=feasibility_hash, adapter_digest=contract["adapter_contract_digest"], selected_path=selected_path, profile=profile)
    return {"status": status, "verification": verification, "selected": selected}


def run_v8_selection(*, protocol_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    protocol, protocol_name, protocol_hash = load_v8_protocol(protocol_path)
    structure = validate_v8_protocol(protocol, protocol_bytes_hash=protocol_hash)
    contract = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
    inherited = _verify_v7_noncontact_evidence()
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feasibility_path = output / FEASIBILITY_FILENAME
    feasibility_hash = file_sha256(feasibility_path) if feasibility_path.is_file() else None
    default_contact = str(_mapping(protocol.get("selection"), "selection").get("default", {}).get("contact_candidate_id", ""))
    context = {"driver_candidate_id": inherited["selected_driver"], "isolator_candidate_id": inherited["selected_isolator"], "contact_candidate_id": default_contact}
    states = {state.state_id: state for state in _resolve_v8(protocol, protocol_bytes_hash=protocol_hash, selection_context=context)}
    contacts = run_v8_stage(protocol, stage="contact", output_dir=output, probe=_v8_contact_probe, protocol_bytes_hash=protocol_hash, selection_context={"driver_candidate_id": inherited["selected_driver"], "isolator_candidate_id": inherited["selected_isolator"]})
    candidates = {states[str(record["state_id"])].candidate_id: record for record in contacts}
    eligible = []
    for candidate_id, record in candidates.items():
        state = states[str(record["state_id"])]
        checks, derived, errors = _v8_contact_gates(record.get("evidence", {}), state)
        if all(checks.values()) and not errors:
            eligible.append(candidate_id)
    winning = sorted(eligible, key=lambda candidate_id: _v8_contact_score(candidates[candidate_id], states[str(candidates[candidate_id]["state_id"])]))[0] if eligible else None
    bind = default_contact
    context["contact_candidate_id"] = bind
    parity = run_v8_stage(protocol, stage="parity", output_dir=output, probe=__import__("robosuite.scripts.shakebench_select_physics_v6", fromlist=["_v6_parity_probe"])._v6_parity_probe, protocol_bytes_hash=protocol_hash, selection_context=context)
    replay = run_v8_stage(protocol, stage="replay", output_dir=output, probe=lambda state: _subprocess_replay_probe(protocol_name, state), protocol_bytes_hash=protocol_hash, selection_context=context)
    return _run_selection_artifacts(output, protocol, protocol_hash, structure, contract, inherited, contacts, parity, replay, states, default_contact, feasibility_hash)


def _raw_index(selected: Mapping[str, Any], selected_path: Path, states: Mapping[str, ResolvedProbeState]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    errors = []
    raw: dict[str, Mapping[str, Any]] = {}
    entries = selected.get("raw_files")
    if not isinstance(entries, list):
        return {}, ["V8 raw_files is missing"]
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("V8 raw entry is not a mapping")
            continue
        path = selected_path.parent / str(entry.get("path", ""))
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            errors.append("V8 raw file hash mismatch: " + path.name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("V8 raw JSON is unreadable: " + path.name)
            continue
        state_id = str(payload.get("state_id"))
        state = states.get(state_id)
        if state is None:
            errors.append("V8 raw state is not in the resolved manifest: " + state_id)
            continue
        if state.stage in NONCONTACT_STAGES:
            errors.append("V8 raw manifest must not duplicate inherited non-contact evidence: " + state_id)
        identity = state.common.protocol_identity
        if not verify_payload_hash(payload) or payload.get("stage") != state.stage or payload.get("resolved_state_digest") != state.resolved_state_digest or payload.get("protocol_sha256_bytes") != identity.get("bytes_sha256") or payload.get("protocol_sha256_normalized") != identity.get("normalized_sha256"):
            errors.append("V8 raw/resolved-state provenance mismatch: " + state_id)
        if entry.get("path") != state.output or not str(entry.get("path", "")).startswith(RAW_PREFIX):
            errors.append("V8 raw entry path mismatch: " + state_id)
        if state_id in raw:
            errors.append("duplicate V8 raw state: " + state_id)
        raw[state_id] = payload
    expected = {state.state_id for state in states.values() if state.stage not in NONCONTACT_STAGES}
    if set(raw) != expected:
        errors.append("V8 raw state coverage differs from contact/parity/replay manifest")
    return raw, errors


def _verify_profile(profile_path: Path, *, protocol_hash: str, stage_rows: Sequence[Mapping[str, Any]], parity_path: Path, determinism_path: Path, selected_state: ResolvedProbeState) -> list[str]:
    errors = []
    if not profile_path.is_file():
        return ["official V8 profile is missing"]
    try:
        import yaml

        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"official V8 profile is unreadable: {exc}"]
    if not isinstance(profile, Mapping):
        return ["official V8 profile is not a mapping"]
    try:
        PhysicsProfile(payload=profile, source=str(profile_path), profile_sha256=str(profile.get("profile_sha256"))).assert_valid()
    except Exception as exc:
        errors.append(f"official V8 profile validation failed: {exc}")
    if profile.get("protocol_sha256") != protocol_hash:
        errors.append("official V8 profile protocol hash mismatch")
    expected_manifest = sha256_json({"raw_files": list(stage_rows), "parity": {"path": parity_path.name, "sha256": file_sha256(parity_path)}, "determinism": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)}})
    if profile.get("evidence_manifest_sha256") != expected_manifest:
        errors.append("official V8 profile evidence-manifest hash mismatch")
    if selected_state.contact is None:
        errors.append("selected V8 state lacks contact child")
    elif profile.get("physics", {}).get("contact", {}).get("condim") != selected_state.contact.condim or not np.isclose(float(profile.get("physics", {}).get("contact", {}).get("solref", [math.nan])[1]), selected_state.contact.solref[1], rtol=0.0, atol=1.0e-14):
        errors.append("official V8 profile normal contact tuple mismatch")
    if profile.get("physics", {}).get("isolator", {}).get("candidate_id") != "low_frequency_damped":
        errors.append("official V8 profile does not bind inherited isolator")
    return errors


def verify_v8_selection_artifact(path: str | Path, *, protocol_path: str | Path | None = None, require_published_status: bool = True) -> dict[str, Any]:
    selected_path = Path(path)
    errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if not isinstance(selected, Mapping):
            raise V8ProtocolError("V8 selected artifact is not a mapping")
        protocol, protocol_name, protocol_hash = load_v8_protocol(protocol_path)
        structure = validate_v8_protocol(protocol, protocol_bytes_hash=protocol_hash)
        states_tuple = _resolve_v8(protocol, protocol_bytes_hash=protocol_hash)
    except (OSError, json.JSONDecodeError, V8ProtocolError, V8ProtocolStateError) as exc:
        return {"passed": False, "errors": [str(exc)], "checks": checks}
    states = {state.state_id: state for state in states_tuple}
    try:
        inherited = _verify_v7_noncontact_evidence()
        if selected.get("inherited_v7_noncontact", {}).get("selected_driver") != inherited["selected_driver"] or selected.get("inherited_v7_noncontact", {}).get("selected_isolator") != inherited["selected_isolator"] or selected.get("inherited_v7_noncontact", {}).get("v7_contact_selection_inherited") is not False:
            errors.append("V8 inherited V7 non-contact metadata mismatch")
    except V8ProtocolError as exc:
        errors.append(str(exc))
        inherited = {"selected_driver": "dt_nominal", "selected_isolator": "low_frequency_damped", "v6_inherited_raw_files": []}
    checks["protocol_bytes_hash"] = selected.get("protocol", {}).get("bytes_sha256") == protocol_hash
    checks["protocol_normalized_hash"] = selected.get("protocol", {}).get("normalized_sha256") == sha256_json(protocol)
    checks["resolved_state_digest"] = selected.get("resolved_state_digest") == structure["resolved_state_digest"]
    for name, passed in (("protocol bytes", checks["protocol_bytes_hash"]), ("protocol normalized", checks["protocol_normalized_hash"]), ("resolved state", checks["resolved_state_digest"])):
        if not passed:
            errors.append(f"V8 selected {name} hash mismatch")
    if not verify_payload_hash(selected):
        errors.append("V8 selected payload hash mismatch")
    selection = selected.get("selection", {})
    if selected.get("physics_only") is not True or selection.get("task_success_used") is not False or selection.get("reward_used") is not False or selection.get("controller_outcome_used") is not False:
        errors.append("V8 selection contains a forbidden non-physics input")
    feasibility_ref = selected.get("feasibility", {})
    feasibility_path = selected_path.parent / str(feasibility_ref.get("path", FEASIBILITY_FILENAME))
    try:
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        if feasibility_ref.get("sha256") != file_sha256(feasibility_path) or not verify_payload_hash(feasibility) or feasibility.get("protocol", {}).get("bytes_sha256") != protocol_hash or feasibility.get("resolved_state_digest") != structure["resolved_state_digest"] or feasibility.get("adapter_contract_digest") != selected.get("adapter_contract_digest"):
            errors.append("V8 feasibility hash/provenance mismatch")
    except (OSError, json.JSONDecodeError):
        errors.append("V8 feasibility artifact is unreadable")
    raw, raw_errors = _raw_index(selected, selected_path, states)
    errors.extend(raw_errors)
    contacts = {}
    details = {}
    for state in states.values():
        if state.stage != "contact" or state.state_id not in raw:
            continue
        record = raw[state.state_id]
        contacts[state.candidate_id] = record
        checks_for_contact, derived, numeric_errors = _v8_contact_gates(record.get("evidence", {}).get("physics", {}), state)
        details[state.candidate_id] = {"checks": checks_for_contact, "derived": derived, "errors": numeric_errors}
        compiled_ok, compiled_errors = _compiled_contact_matches(record.get("evidence", {}).get("physics", {}).get("compiled_contact_profile", {}), state)
        if not compiled_ok:
            errors.extend(f"{state.candidate_id}: {error}" for error in compiled_errors)
        try:
            actual = _compile_contact_audit(state)
            independent_ok, independent_errors = _compiled_contact_matches(actual, state)
            if not independent_ok:
                errors.extend(f"{state.candidate_id}: independent compiled audit failed: {error}" for error in independent_errors)
            if not _same_json(actual, record.get("evidence", {}).get("physics", {}).get("compiled_contact_profile", {})):
                errors.append(f"{state.candidate_id}: compiled contact audit changed")
            force = _compile_force_envelope(state)
            declared = record.get("evidence", {}).get("physics", {}).get("finger_force_envelope", {})
            if force.get("finite_force_envelope_N") != declared.get("finite_force_envelope_N") or force.get("finite_force_envelope_N") != float(state.contact.hard_gates["finger_force_max_N"]):
                errors.append(f"{state.candidate_id}: force envelope changed")
        except Exception as exc:
            errors.append(f"{state.candidate_id}: independent contact audit failed: {exc}")
    eligible = [candidate_id for candidate_id, item in details.items() if all(item["checks"].values()) and not item["errors"]]
    recomputed_contact = sorted(eligible, key=lambda candidate_id: _v8_contact_score(contacts[candidate_id], next(state for state in states.values() if state.stage == "contact" and state.candidate_id == candidate_id)))[0] if eligible else None
    checks["contact_coverage"] = set(contacts) == CONTACT_IDS
    checks["contact_eligibility"] = recomputed_contact is not None
    if not checks["contact_coverage"]:
        errors.append("V8 contact coverage is incomplete")
    if not checks["contact_eligibility"]:
        errors.append("V8 contact hard-gate eligibility failed")
    registered_contact = str(_mapping(protocol.get("selection"), "selection").get("default", {}).get("contact_candidate_id", ""))
    if recomputed_contact is not None and recomputed_contact != registered_contact:
        errors.append("V8 winning contact differs from the registered parity/replay binding")
    parity_rows = [row for row in raw.values() if row.get("stage") == "parity"]
    parity_state = next((state for state in states.values() if state.stage == "parity"), None)
    parity_ok = len(parity_rows) == 1 and parity_state is not None and _parity_eligible(parity_rows[0], parity_state)
    checks["parity"] = parity_ok
    if not parity_ok:
        errors.append("V8 parity evidence failed")
    parity_path = selected_path.parent / str(selected.get("parity_artifact", {}).get("path", PARITY_FILENAME))
    try:
        parity_artifact = json.loads(parity_path.read_text(encoding="utf-8"))
        if selected.get("parity_artifact", {}).get("sha256") != file_sha256(parity_path) or not verify_payload_hash(parity_artifact) or parity_artifact.get("status") != ("PASS" if parity_ok else "BLOCKED"):
            errors.append("V8 parity artifact mismatch")
    except (OSError, json.JSONDecodeError):
        errors.append("V8 parity artifact is unreadable")
    replay_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw.values():
        if row.get("stage") == "replay":
            replay_groups[str(row.get("evidence", {}).get("selected_component"))].append(row)
    replay_ok = REPLAY_GROUPS.issubset(replay_groups) and all(len(replay_groups[group]) == 3 for group in REPLAY_GROUPS)
    if replay_ok:
        for group in REPLAY_GROUPS:
            rows = replay_groups[group]
            replay_ok = replay_ok and {int(row.get("evidence", {}).get("process_index", 0)) for row in rows} == {1, 2, 3} and len({row.get("evidence", {}).get("trace_digest") for row in rows}) == 1 and len({row.get("evidence", {}).get("metric_digest") for row in rows}) == 1 and all(row.get("evidence", {}).get("complete_trace") is True and row.get("evidence", {}).get("same_process_reset_used") is False for row in rows)
    checks["replay"] = replay_ok
    if not replay_ok:
        errors.append("V8 four-group replay evidence failed")
    determinism_path = selected_path.parent / str(selected.get("determinism_artifact", {}).get("path", DETERMINISM_FILENAME))
    try:
        determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
        if selected.get("determinism_artifact", {}).get("sha256") != file_sha256(determinism_path) or not verify_payload_hash(determinism) or determinism.get("status") != ("PASS" if replay_ok else "BLOCKED"):
            errors.append("V8 determinism artifact mismatch")
    except (OSError, json.JSONDecodeError):
        errors.append("V8 determinism artifact is unreadable")
    excluded_path = selected_path.parent / EXCLUDED_FILENAME
    try:
        excluded = json.loads(excluded_path.read_text(encoding="utf-8"))
        if not verify_payload_hash(excluded) or excluded.get("protocol_sha256_bytes") != protocol_hash or excluded.get("excluded") != selected.get("excluded"):
            errors.append("V8 selected/excluded tables disagree")
    except (OSError, json.JSONDecodeError):
        errors.append("V8 excluded artifact is unreadable")
    declared_contact = selection.get("selected_candidate_ids", {}).get("contact")
    final_ok = recomputed_contact is not None and parity_ok and replay_ok and declared_contact == recomputed_contact
    if final_ok:
        state = next(state for state in states.values() if state.stage == "contact" and state.candidate_id == recomputed_contact)
        if selection.get("selected_candidate_ids", {}).get("driver") != "dt_nominal" or selection.get("selected_candidate_ids", {}).get("isolator") != "low_frequency_damped":
            errors.append("V8 selected inherited component IDs are incorrect")
        if selected.get("status") != "PASS" or selected.get("official_profile") != OFFICIAL_PROFILE_FILENAME or selected.get("official_profile_alignment") is not True:
            errors.append("V8 PASS profile alignment is missing")
        errors.extend(_verify_profile(selected_path.parent / OFFICIAL_PROFILE_FILENAME, protocol_hash=protocol_hash, stage_rows=selected.get("raw_files", []), parity_path=parity_path, determinism_path=determinism_path, selected_state=state))
    else:
        if selected.get("status") != "BLOCKED" or declared_contact is not None or selected.get("official_profile") not in (None, "") or selected.get("official_profile_alignment") is not False:
            errors.append("blocked V8 artifact must have selected_contact=null and no official profile payload")
    try:
        expected_contract = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
        if selected.get("adapter_contract_digest") != expected_contract["adapter_contract_digest"]:
            errors.append("V8 adapter-contract digest mismatch")
    except Exception as exc:
        errors.append(f"V8 adapter-contract recomputation failed: {exc}")
    status_path = selected_path.parent / STATUS_FILENAME
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not verify_payload_hash(status) or status.get("adapter_contract_digest") != selected.get("adapter_contract_digest") or status.get("protocol", {}).get("bytes_sha256") != protocol_hash or status.get("status") != ("PASS" if selected.get("status") == "PASS" else "BLOCKED"):
            errors.append("V8 status artifact mismatch")
    except (OSError, json.JSONDecodeError):
        if require_published_status:
            errors.append("V8 status artifact is unreadable")
    return {"passed": selected.get("status") == "PASS" and not errors, "errors": errors, "checks": checks, "recomputed_selection": {"driver": "dt_nominal", "isolator": "low_frequency_damped", "contact": recomputed_contact}, "contact_gate_details": details}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    parser.add_argument("--adapter-contract", action="store_true")
    parser.add_argument("--write-feasibility", action="store_true")
    parser.add_argument("--stage", choices=("contact", "parity", "replay", "all"))
    parser.add_argument("--verify", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--state-id", default=None)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.protocol is None or args.state_id is None:
                raise V8ProtocolError("--worker requires --protocol and --state-id")
            return _v8_replay_worker(args.protocol, args.state_id)
        protocol, name, protocol_hash = load_v8_protocol(args.protocol)
        if args.validate_protocol:
            result = {"passed": True, "path": name, "protocol_sha256_bytes": protocol_hash, "structure": validate_v8_protocol(protocol, protocol_bytes_hash=protocol_hash)}
        elif args.dry_run_manifest:
            result = dry_run_manifest(protocol, protocol_bytes_hash=protocol_hash)
        elif args.adapter_contract:
            result = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
        elif args.write_feasibility:
            result = write_feasibility_artifact(name)
        elif args.verify:
            result = verify_v8_selection_artifact(args.verify, protocol_path=name)
        elif args.stage:
            inherited = _verify_v7_noncontact_evidence()
            default = str(_mapping(protocol.get("selection"), "selection").get("default", {}).get("contact_candidate_id"))
            context = {"driver_candidate_id": inherited["selected_driver"], "isolator_candidate_id": inherited["selected_isolator"], "contact_candidate_id": default}
            if args.stage == "contact":
                result = {"records": run_v8_stage(protocol, stage="contact", output_dir=Path(models.assets_root), probe=_v8_contact_probe, protocol_bytes_hash=protocol_hash, selection_context={"driver_candidate_id": inherited["selected_driver"], "isolator_candidate_id": inherited["selected_isolator"]})}
            elif args.stage == "parity":
                result = {"records": run_v8_stage(protocol, stage="parity", output_dir=Path(models.assets_root), probe=__import__("robosuite.scripts.shakebench_select_physics_v6", fromlist=["_v6_parity_probe"])._v6_parity_probe, protocol_bytes_hash=protocol_hash, selection_context=context)}
            elif args.stage == "replay":
                result = {"records": run_v8_stage(protocol, stage="replay", output_dir=Path(models.assets_root), probe=lambda state: _subprocess_replay_probe(name, state), protocol_bytes_hash=protocol_hash, selection_context=context)}
            else:
                result = run_v8_selection(protocol_path=name, output_dir=Path(models.assets_root))
        else:
            result = run_v8_selection(protocol_path=name, output_dir=Path(models.assets_root))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
