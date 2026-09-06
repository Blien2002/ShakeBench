"""Phase 06R6/V7 contact-recovery selection and independent verifier.

V7 inherits only the verified V6 driver/isolator evidence.  It registers and
probes a finite contact family with raw recovery traces, a compiled Panda
finger-force envelope, and a MuJoCo incline bracket.  Task reward, success,
controller outcome, policy observations, and State tiers are intentionally
outside this module.

The pure commands are safe before registration::

    --validate-protocol
    --dry-run-manifest
    --adapter-contract
    --write-feasibility

Physics commands write only V7-prefixed evidence.  A profile is written only
after :func:`verify_v7_selection_artifact` returns PASS.
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

import mujoco
import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_diagnose_contact_recovery_v7 import (
    PENETRATION_MAX_M,
    RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S,
    RECOVERY_DURATION_S,
    RECOVERY_VELOCITY_MAX_M_S,
    TAIL_WINDOW_S,
    _force_envelope_audit,
    diagnose_state,
)
from robosuite.scripts.shakebench_select_physics_v6 import (
    V6EvidenceError,
    V6ProtocolError,
    _contact_candidate_mapping,
    _driver_eligibility,
    _isolator_eligible,
    _isolator_score,
    _parity_eligible,
    _profile_from_state,
    _resolve,
    _select_driver,
    _v6_isolator_probe,
    _v6_parity_probe,
    _v6_replay_probe,
    load_v6_protocol,
    validate_v6_protocol,
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
    NoPhysicsBackend,
    prepare_contact_probe,
    prepare_driver_probe,
    prepare_isolator_probe,
    prepare_parity_probe,
    prepare_replay_probe,
)
from robosuite.utils.shakebench_physics import (
    PhysicsProfile,
    make_probe_physics_profile,
    physics_profile_hash,
)
from robosuite.utils.shakebench_protocol_v7 import (
    ResolvedProbeState,
    V7ProtocolStateError,
    legal_replay_binding_templates,
    resolve_all_protocol_states_v7,
    resolve_replay_binding_templates,
)
from robosuite.utils.shakebench_protocol_v7 import sha256_json as sha256_state_json

SCHEMA_ID = "shakebench.phase06r6.v7.physics_selection"
PROTOCOL_SCHEMA_ID = "shakebench.phase06r6.v7.physics_selection_protocol"
SCHEMA_VERSION = 7
PROTOCOL_FILENAME = "shakebench_selection_protocol_v7.yaml"
RAW_PREFIX = "shakebench_phase_06r6_v7_raw_"
SELECTED_FILENAME = "shakebench_phase_06r6_v7_selected_candidates.json"
EXCLUDED_FILENAME = "shakebench_phase_06r6_v7_excluded_candidates.json"
FEASIBILITY_FILENAME = "shakebench_phase_06r6_v7_feasibility.json"
STATUS_FILENAME = "shakebench_phase_06r6_v7_status.json"
PARITY_FILENAME = "shakebench_phase_06r6_v7_gamma_zero_parity.json"
DETERMINISM_FILENAME = "shakebench_phase_06r6_v7_replay_determinism.json"
OFFICIAL_PROFILE_FILENAME = "shakebench_official_physics.yaml"
DIAGNOSTIC_FILENAME = "shakebench_phase_06r6_v7_contact_recovery_diagnostic.json"
V6_PROTOCOL_FILENAME = "shakebench_selection_protocol_v6.yaml"
V6_SELECTED_FILENAME = "shakebench_phase_06r5_v6_selected_candidates.json"
V6_STATUS_FILENAME = "shakebench_phase_06r5_v6_status.json"
V6_RAW_PREFIX = "shakebench_phase_06r5_v6_raw_"

CONTACT_IDS = {
    "c3_nominal",
    "c3_softer",
    "c4_torsional",
    "c6_baseline",
    "c6_rolling_low",
    "c6_rolling_high",
    "c6_damped",
}
NONCONTACT_STAGES = {"driver", "isolator"}
REPLAY_GROUPS = {"driver", "isolator", "contact", "gamma_zero_parity"}
FINGER_PAIR_NAMES = {
    tuple(sorted(("can_g0", "gripper0_right_finger1_pad_collision"))),
    tuple(sorted(("can_g0", "gripper0_right_finger2_pad_collision"))),
}


class V7ProtocolError(ValueError):
    """V7 protocol or registration-time contract failure."""


class V7EvidenceError(RuntimeError):
    """V7 raw evidence is absent, mutated, or incomplete."""


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    try:
        import yaml

        value = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise V7ProtocolError("V7 protocol root must be a mapping")
    return value


def load_v7_protocol(path: str | Path | None = None) -> tuple[Mapping[str, Any], str, str]:
    protocol_path = Path(models.assets_root) / PROTOCOL_FILENAME if path is None else Path(path)
    raw = protocol_path.read_bytes()
    return _load_yaml(protocol_path), str(protocol_path), sha256_bytes(raw)


def _resolve_v7(
    protocol: Mapping[str, Any],
    *,
    protocol_bytes_hash: str | None = None,
    selection_context: Mapping[str, Any] | None = None,
) -> tuple[ResolvedProbeState, ...]:
    context = dict(selection_context or {})
    if protocol_bytes_hash is not None:
        context["protocol_sha256_bytes"] = protocol_bytes_hash
    context.setdefault("protocol_sha256_normalized", sha256_json(protocol))
    try:
        return resolve_all_protocol_states_v7(protocol, context)
    except V7ProtocolStateError as exc:
        raise V7ProtocolError(str(exc)) from exc


def _mapping(value: Any, label: str, *, required: bool = True) -> Mapping[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise V7ProtocolError(f"{label} must be a mapping")
    return value


def _rows(protocol: Mapping[str, Any], section: str) -> list[Mapping[str, Any]]:
    value = _mapping(protocol.get("components"), "components").get(section)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise V7ProtocolError(f"components.{section} must be a list of mappings")
    return list(value)


def _driver_rows(protocol: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _mapping(protocol.get("driver"), "driver").get("convergence_candidates")
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise V7ProtocolError("driver.convergence_candidates must be a list of mappings")
    return list(value)


def _require_ids(rows: Sequence[Mapping[str, Any]], expected: set[str], label: str) -> None:
    ids = [str(row.get("candidate_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise V7ProtocolError(f"{label} candidate set is incomplete or duplicated")


def _same_json(left: Any, right: Any) -> bool:
    return sha256_json(left) == sha256_json(right)


def _v6_asset(name: str, asset_root: str | Path | None = None) -> Path:
    configured_root = os.environ.get("SHAKEBENCH_EVIDENCE_ROOT")
    base = (
        Path(asset_root)
        if asset_root is not None
        else Path(configured_root) if configured_root else Path(models.assets_root)
    )
    path = base / name
    if not path.is_file():
        raise V7ProtocolError(f"required V6 evidence is missing: {name}")
    return path


def _noncontact_child_view(state: ResolvedProbeState) -> dict[str, Any]:
    value = state.to_dict()
    common = value["common"]
    common.pop("output", None)
    common.pop("protocol_identity", None)
    # Replay rows carry the contact binding only for the contact-dependent
    # groups.  Driver/isolator replay rows are inherited verbatim.
    if state.stage == "replay" and value.get("replay") is not None:
        value["replay"].pop("candidate_fields", None)
    return value


def _validate_v6_noncontact_protocol(v7_protocol: Mapping[str, Any]) -> dict[str, Any]:
    v6_protocol, v6_name, v6_hash = load_v6_protocol(_v6_asset(V6_PROTOCOL_FILENAME))
    v6_states = _resolve(v6_protocol, protocol_bytes_hash=v6_hash)
    v7_states = _resolve_v7(v7_protocol)
    v6_by_id = {state.state_id: state for state in v6_states}
    v7_by_id = {state.state_id: state for state in v7_states}
    inherited = {}
    for state in v6_states:
        if state.stage not in NONCONTACT_STAGES:
            continue
        other = v7_by_id.get(state.state_id)
        if other is None or other.stage != state.stage:
            raise V7ProtocolError(f"V7 lost inherited {state.stage} state: {state.state_id}")
        if not _same_json(_noncontact_child_view(state), _noncontact_child_view(other)):
            raise V7ProtocolError(f"V7 changed inherited resolved state: {state.state_id}")
        inherited[state.stage] = inherited.get(state.stage, 0) + 1
    if inherited != {"driver": 18, "isolator": 3}:
        raise V7ProtocolError(f"V7 inherited state coverage is incomplete: {inherited}")
    return {"protocol_path": v6_name, "protocol_sha256": v6_hash, "states": v6_states, "v7_states": v7_states}


def validate_v7_protocol(
    protocol: Mapping[str, Any],
    *,
    protocol_bytes_hash: str | None = None,
    check_v6_inheritance: bool = True,
) -> dict[str, Any]:
    """Validate V7 shape, exact inherited values, and every resolved state."""

    if protocol.get("schema_id") != PROTOCOL_SCHEMA_ID or protocol.get("schema_version") != SCHEMA_VERSION:
        raise V7ProtocolError("wrong V7 protocol schema")
    if protocol.get("phase") != "06R6":
        raise V7ProtocolError("V7 protocol phase must be 06R6")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise V7ProtocolError("V7 protocol must be pre_registered and immutable")
    if protocol.get("selection_authority") != "physics_probes_only":
        raise V7ProtocolError("V7 selection authority must be physics_probes_only")
    scope = _mapping(protocol.get("scope"), "scope")
    forbidden = str(scope.get("forbidden_selection_inputs", "")).lower()
    for name in ("task_success", "reward", "controller_outcome", "state_tier", "policy_observation", "gamma_star"):
        if name in forbidden:
            continue
        raise V7ProtocolError(f"scope must forbid {name}")
    for key in (
        "task_success",
        "task_sr",
        "reward",
        "controller_outcome_used",
        "state_tier_ranking",
        "policy_observation",
        "Gamma_star",
    ):
        if key in protocol and protocol.get(key) not in (False, None, ""):
            raise V7ProtocolError(f"forbidden selection input appears at protocol root: {key}")

    facts = _mapping(protocol.get("frozen_facts"), "frozen_facts")
    required_facts = (
        "control_frequency_hz",
        "f_max_hz",
        "authored_spectrum",
        "safe_gamma_candidates",
        "load_cases",
        "gravity_m_s2",
        "table_mass_kg",
        "table_inertia_kg_m2",
    )
    if any(key not in facts for key in required_facts):
        raise V7ProtocolError("V7 frozen_facts is incomplete")
    measurement = _mapping(protocol.get("measurement"), "measurement")
    if measurement.get("capacity_selection_input") is not False:
        raise V7ProtocolError("V7 capacity preflight must remain non-decisional")
    for key in ("trace_schema_id", "trace_schema_version", "minimum_cadence_hz", "capacity_preflight"):
        if key not in measurement:
            raise V7ProtocolError(f"measurement is missing {key}")

    v6_protocol, _, v6_hash = load_v6_protocol(_v6_asset(V6_PROTOCOL_FILENAME))
    # All frozen facts, measurement settings, driver rows, and isolator rows
    # are immutable V6 inputs.  Candidate-owned contact values are checked
    # separately below.
    if not _same_json(protocol.get("frozen_facts"), v6_protocol.get("frozen_facts")):
        raise V7ProtocolError("V7 changed V6 frozen facts")
    if not _same_json(protocol.get("measurement"), v6_protocol.get("measurement")):
        raise V7ProtocolError("V7 changed V6 measurement contract")
    if not _same_json(protocol.get("driver"), v6_protocol.get("driver")):
        raise V7ProtocolError("V7 changed the V6 driver contract")
    if not _same_json(
        protocol.get("components", {}).get("isolator_candidates"),
        v6_protocol.get("components", {}).get("isolator_candidates"),
    ):
        raise V7ProtocolError("V7 changed V6 isolator candidates")
    if not _same_json(protocol.get("isolator"), v6_protocol.get("isolator")):
        raise V7ProtocolError("V7 changed the V6 isolator contract")
    if not _same_json(protocol.get("parity"), v6_protocol.get("parity")):
        raise V7ProtocolError("V7 changed the V6 parity geometry/action contract")

    contacts = _rows(protocol, "contact_candidates")
    _require_ids(contacts, CONTACT_IDS, "contact")
    v6_contacts = {str(row["candidate_id"]): row for row in _rows(v6_protocol, "contact_candidates")}
    for row in contacts:
        candidate_id = str(row["candidate_id"])
        for field in (
            "condim",
            "sliding_mu",
            "torsional_mu",
            "rolling_mu",
            "margin_m",
            "gap_m",
            "solref",
            "solimp",
            "iterations",
            "interfaces",
        ):
            if field not in row:
                raise V7ProtocolError(f"contact candidate {candidate_id} is missing {field}")
        if candidate_id in {"c3_nominal", "c3_softer", "c4_torsional"} and not _same_json(
            row, v6_contacts[candidate_id]
        ):
            raise V7ProtocolError(f"V7 changed retained V6 negative control: {candidate_id}")
        if row["sliding_mu"] != {"table_object": 0.30, "finger_object": 1.00}:
            raise V7ProtocolError(f"V7 sliding friction changed for {candidate_id}")
        if int(row["iterations"]) != 100 or float(row["margin_m"]) != 0.0 or float(row["gap_m"]) != 0.0:
            raise V7ProtocolError(f"V7 contact tuple changed global margin/gap/iterations: {candidate_id}")
        if float(row["solref"][0]) < 2.0 * 0.0002:
            raise V7ProtocolError(f"V7 contact solref violates 2*dt: {candidate_id}")
    new_rows = {str(row["candidate_id"]): row for row in contacts if str(row["candidate_id"]).startswith("c6_")}
    if not all(int(row["condim"]) == 6 for row in new_rows.values()):
        raise V7ProtocolError("every added V7 contact candidate must use condim=6")
    if set(new_rows) != {"c6_baseline", "c6_rolling_low", "c6_rolling_high", "c6_damped"}:
        raise V7ProtocolError("V7 condim=6 contact family is incomplete")
    if not (
        float(new_rows["c6_rolling_low"]["rolling_mu"])
        < float(new_rows["c6_baseline"]["rolling_mu"])
        < float(new_rows["c6_rolling_high"]["rolling_mu"])
    ):
        raise V7ProtocolError("V7 rolling levels do not bracket the baseline")
    if not np.isclose(
        math.log(float(new_rows["c6_baseline"]["rolling_mu"]) / float(new_rows["c6_rolling_low"]["rolling_mu"])),
        math.log(float(new_rows["c6_rolling_high"]["rolling_mu"]) / float(new_rows["c6_baseline"]["rolling_mu"])),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise V7ProtocolError("V7 rolling levels must bracket baseline logarithmically")
    if not np.isclose(float(new_rows["c6_damped"]["solref"][0]), 0.0004, rtol=0.0, atol=1.0e-15) or np.isclose(
        float(new_rows["c6_damped"]["solref"][1]), 1.0, rtol=0.0, atol=1.0e-15
    ):
        raise V7ProtocolError("V7 damping candidate must change only the solref damping ratio")
    for candidate_id, row in new_rows.items():
        for field in (
            "condim",
            "sliding_mu",
            "torsional_mu",
            "margin_m",
            "gap_m",
            "solref",
            "solimp",
            "iterations",
            "interfaces",
        ):
            if not _same_json(row[field], v6_contacts["c3_nominal"][field]) and field not in {
                "condim",
                "rolling_mu",
                "solref",
            }:
                raise V7ProtocolError(f"V7 changed a non-contact-family field for {candidate_id}: {field}")

    contact_spec = _mapping(protocol.get("contact"), "contact")
    for key in ("interfaces", "probes", "hard_gates", "scoring", "diagnostic"):
        if key not in contact_spec:
            raise V7ProtocolError(f"contact.{key} is required")
    if tuple(contact_spec["interfaces"]) != (
        "can_open_worktable",
        "can_target_bottom",
        "can_target_walls",
        "can_panda_finger_pads",
    ):
        raise V7ProtocolError("V7 contact interface set changed")
    probes = _mapping(contact_spec["probes"], "contact.probes")
    if (
        float(probes.get("recovery_duration_s")) != RECOVERY_DURATION_S
        or float(probes.get("recovery_velocity_max_m_s")) != RECOVERY_VELOCITY_MAX_M_S
        or float(probes.get("recovery_angular_velocity_max_rad_s")) != RECOVERY_ANGULAR_VELOCITY_MAX_RAD_S
        or float(probes.get("maximum_illegal_penetration_m")) != PENETRATION_MAX_M
    ):
        raise V7ProtocolError("V7 recovery convention changed")
    incline_angles = tuple(float(value) for value in probes.get("incline_angles_rad", ()))
    if len(incline_angles) < 4 or not all(left < right for left, right in zip(incline_angles, incline_angles[1:])):
        raise V7ProtocolError("V7 incline angle grid must be finite and increasing")
    if float(probes.get("incline_bracket_resolution_rad", 1.0)) <= 0.0:
        raise V7ProtocolError("V7 incline bracket resolution is invalid")
    gates = _mapping(contact_spec["hard_gates"], "contact.hard_gates")
    for key in (
        "static_support_force_min_N",
        "maximum_illegal_penetration_m",
        "recovery_velocity_max_m_s",
        "recovery_angular_velocity_max_rad_s",
        "finger_force_min_N",
        "finger_force_max_N",
        "finger_penetration_max_m",
        "warning_count_max",
        "timestep_trace_relative_error_max",
        "incline_bracket_resolution_rad",
        "slip_final_speed_max_m_s",
    ):
        if key not in gates:
            raise V7ProtocolError(f"V7 contact hard gate is missing {key}")
    if not np.isclose(float(gates["finger_force_max_N"]), 40.0, rtol=0.0, atol=1.0e-12):
        raise V7ProtocolError("V7 force gate does not equal the diagnostic-derived 40 N envelope")
    diagnostic = _mapping(contact_spec["diagnostic"], "contact.diagnostic")
    diagnostic_path = _v6_asset(str(diagnostic.get("path", DIAGNOSTIC_FILENAME)))
    if str(diagnostic.get("sha256")) != file_sha256(diagnostic_path):
        raise V7ProtocolError("V7 diagnostic artifact hash mismatch")

    states = _resolve_v7(protocol, protocol_bytes_hash=protocol_bytes_hash)
    declared_count = int(protocol.get("manifest_state_count", -1))
    if declared_count != len(states):
        raise V7ProtocolError("manifest_state_count does not match resolved V7 states")
    if any(not state.output.startswith(RAW_PREFIX) or not state.output.endswith(".json") for state in states):
        raise V7ProtocolError("V7 raw outputs must use the flat V7 prefix")
    stage_counts = {
        stage: sum(state.stage == stage for state in states)
        for stage in ("driver", "isolator", "contact", "parity", "replay")
    }
    if stage_counts != {"driver": 18, "isolator": 3, "contact": len(CONTACT_IDS), "parity": 1, "replay": 12}:
        raise V7ProtocolError(f"V7 manifest stage coverage is wrong: {stage_counts}")
    replay_groups = {
        state.replay.selected_component for state in states if state.stage == "replay" and state.replay is not None
    }
    if not REPLAY_GROUPS.issubset(replay_groups):
        raise V7ProtocolError("V7 replay manifest is missing a required group")
    inherited = None
    if check_v6_inheritance:
        inherited = _validate_v6_noncontact_protocol(protocol)
    return {
        "state_count": len(states),
        "stage_counts": stage_counts,
        "state_ids": [state.state_id for state in states],
        "outputs": [state.output for state in states],
        "resolved_state_digest": sha256_state_json([state.to_dict() for state in states]),
        "legal_replay_binding_count": len(legal_replay_binding_templates(protocol)),
        "v6_protocol_sha256": v6_hash,
        "inherited": {"driver": 18, "isolator": 3} if inherited is not None else None,
    }


def dry_run_manifest(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    structure = validate_v7_protocol(protocol, protocol_bytes_hash=protocol_bytes_hash)
    states = _resolve_v7(protocol, protocol_bytes_hash=protocol_bytes_hash)
    bindings = legal_replay_binding_templates(protocol)
    return {
        "schema_id": SCHEMA_ID + ".resolved_manifest",
        "schema_version": SCHEMA_VERSION,
        "state_count": len(states),
        "stage_counts": structure["stage_counts"],
        "resolved_state_digest": structure["resolved_state_digest"],
        "states": [state.to_dict() for state in states],
        "legal_replay_binding_count": len(bindings),
        "legal_replay_bindings": [copy.deepcopy(dict(binding)) for binding in bindings],
        "mujoco_model_created": False,
    }


def adapter_contract(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    """Exercise every V7 state and every legal replay binding without physics."""

    states = _resolve_v7(protocol, protocol_bytes_hash=protocol_bytes_hash)
    backend = NoPhysicsBackend()
    plans: list[dict[str, Any]] = []

    def prepare(state: ResolvedProbeState) -> None:
        if state.stage == "driver":
            plan = prepare_driver_probe(state, backend)
        elif state.stage == "isolator":
            plan = prepare_isolator_probe(state, backend)
        elif state.stage == "contact":
            plan = prepare_contact_probe(state, backend)
        elif state.stage == "parity":
            plan = prepare_parity_probe(state, backend)
        elif state.stage == "replay":
            plan = prepare_replay_probe(state, backend)
        else:
            raise AdapterContractError(f"unsupported V7 stage: {state.stage}")
        if plan.complete is not True or not plan.requested_fields:
            raise AdapterContractError(f"incomplete V7 {state.stage} adapter plan")
        plans.append({"kind": "manifest_state", "state_digest": state.resolved_state_digest, "plan": plan.to_dict()})

    for state in states:
        prepare(state)
    for state in resolve_replay_binding_templates(protocol):
        replay_plan = prepare_replay_probe(state, backend)
        plans.append(
            {"kind": "legal_replay_binding", "state_digest": state.resolved_state_digest, "plan": replay_plan.to_dict()}
        )
        selected = state.replay.selected_component if state.replay is not None else ""
        if selected == "driver":
            component_plan = prepare_driver_probe(state, backend)
        elif selected == "isolator":
            component_plan = prepare_isolator_probe(state, backend)
        elif selected == "contact":
            component_plan = prepare_contact_probe(state, backend)
        else:
            component_plan = prepare_parity_probe(state, backend)
        if component_plan.complete is not True:
            raise AdapterContractError(f"incomplete replay component plan: {selected}")
        plans.append(
            {
                "kind": "legal_replay_component",
                "state_digest": state.resolved_state_digest,
                "plan": component_plan.to_dict(),
            }
        )
    return {
        "schema_id": SCHEMA_ID + ".adapter_contract",
        "schema_version": SCHEMA_VERSION,
        "state_count": len(states),
        "legal_replay_binding_count": len(resolve_replay_binding_templates(protocol)),
        "prepared_plan_count": len(plans),
        "adapter_contract_digest": sha256_json(plans),
        "plans": plans,
        "backend": {
            "type": "NoPhysicsBackend",
            "preparation_event_count": len(backend.preparation_events),
            "physics_calls": 0,
        },
        "mujoco_model_created": False,
    }


def _last_json_line(output: str) -> Mapping[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise V7ProtocolError("pure V7 command did not emit JSON")


def write_feasibility_artifact(protocol_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    protocol, protocol_name, protocol_hash = load_v7_protocol(protocol_path)
    command_specs = (
        ("validate", "--validate-protocol"),
        ("dry_run", "--dry-run-manifest"),
        ("adapter_contract", "--adapter-contract"),
    )
    results: dict[str, Mapping[str, Any]] = {}
    exit_codes: dict[str, int] = {}
    for label, flag in command_specs:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robosuite.scripts.shakebench_select_physics_v7",
                "--protocol",
                str(protocol_path),
                flag,
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        exit_codes[label] = int(completed.returncode)
        results[label] = _last_json_line(completed.stdout or completed.stderr)
    if any(code != 0 for code in exit_codes.values()):
        raise V7ProtocolError(f"cannot register V7 feasibility from failed pure commands: {exit_codes}")
    structure = validate_v7_protocol(protocol, protocol_bytes_hash=protocol_hash)
    contract = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
    states = _resolve_v7(protocol, protocol_bytes_hash=protocol_hash)
    facts = _mapping(protocol.get("frozen_facts"), "frozen_facts")
    measurement = _mapping(protocol.get("measurement"), "measurement")
    proofs = []
    for state in states:
        ratio = state.common.control_period_s / state.common.physics_timestep_s
        proofs.append(
            {
                "state_id": state.state_id,
                "control_period_over_dt": ratio,
                "integer": state.common.control_steps,
                "passed": np.isclose(ratio, state.common.control_steps, rtol=0.0, atol=1.0e-12),
            }
        )
    diagnostic_path = _v6_asset(DIAGNOSTIC_FILENAME)
    artifact = {
        "schema_id": SCHEMA_ID + ".feasibility",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol": {
            "path": protocol_name,
            "bytes_sha256": protocol_hash,
            "normalized_sha256": sha256_json(protocol),
            "immutable_after_registration": protocol.get("immutable_after_registration"),
        },
        "resolved_state_digest": structure["resolved_state_digest"],
        "resolved_states": [state.to_dict() for state in states],
        "state_count": len(states),
        "stage_counts": structure["stage_counts"],
        "legal_replay_binding_count": len(resolve_replay_binding_templates(protocol)),
        "adapter_contract_digest": contract["adapter_contract_digest"],
        "adapter_contract_state_count": contract["state_count"],
        "adapter_contract_prepared_plan_count": contract["prepared_plan_count"],
        "integer_ratio_proofs": proofs,
        "capacity_evidence": {
            "path": str(measurement.get("capacity_preflight")),
            "sha256": file_sha256(_v6_asset(str(measurement.get("capacity_preflight")))),
            "selection_input": measurement_selection_input(protocol),
        },
        "diagnostic_evidence": {
            "path": diagnostic_path.name,
            "sha256": file_sha256(diagnostic_path),
            "selection_authority": False,
        },
        "v6_inheritance": {
            "protocol": V6_PROTOCOL_FILENAME,
            "protocol_sha256": structure["v6_protocol_sha256"],
            "selected_driver": "dt_nominal",
            "selected_isolator": "low_frequency_damped",
        },
        "commands": {
            label: {
                "command": f"python -m robosuite.scripts.shakebench_select_physics_v7 --protocol {protocol_path} {flag}",
                "exit_code": exit_codes[label],
                "result_digest": sha256_json(results[label]),
            }
            for label, flag in command_specs
        },
    }
    artifact["payload_sha256"] = payload_hash(artifact)
    destination = Path(output_path) if output_path is not None else Path(models.assets_root) / FEASIBILITY_FILENAME
    write_json_atomic(destination, artifact)
    return artifact


def measurement_selection_input(protocol: Mapping[str, Any]) -> bool:
    return bool(_mapping(protocol.get("measurement"), "measurement").get("capacity_selection_input"))


def _compile_contact_audit(state: ResolvedProbeState) -> dict[str, Any]:
    """Compile the real task model and return only the contact audit."""

    if state.contact is None:
        raise V7ProtocolError(f"contact child is missing: {state.state_id}")
    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    profile = _profile_from_state(state)
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=state.common.physics_timestep_s,
        target_container_friction=(0.30, state.contact.torsional_mu, state.contact.rolling_mu),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=20,
        seed=17,
    )
    try:
        audit = env.audit_compiled_model()
        contacts = copy.deepcopy(dict(audit["contacts"]))
        contacts["solver_iterations"] = int(env.sim.model._model.opt.iterations)
        contacts["solver_tolerance"] = float(env.sim.model._model.opt.tolerance)
        return contacts
    finally:
        env.close()


def _incline_xml(angle_rad: float, candidate: Mapping[str, Any]) -> str:
    mesh_path = (Path(models.assets_root) / "objects" / "meshes" / "can.msh").resolve()
    friction = " ".join(
        format(float(value), ".17g")
        for value in (
            0.30,
            float(candidate["torsional_mu"]),
            float(candidate["rolling_mu"]),
            float(candidate["rolling_mu"]),
            float(candidate["rolling_mu"]),
        )
    )
    solref = " ".join(format(float(value), ".17g") for value in candidate["solref"])
    solimp = " ".join(format(float(value), ".17g") for value in candidate["solimp"])
    return f"""
<mujoco model="shakebench_v7_incline">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true" />
  <option timestep="0.0002" integrator="Euler" solver="Newton" iterations="100" tolerance="1e-12" gravity="0 0 -9.81" />
  <asset><mesh name="can_mesh" file="{mesh_path}" /></asset>
  <worldbody>
    <geom name="incline_plane" type="plane" size="2 2 0.1" pos="0 0 0" euler="0 {format(float(angle_rad), '.17g')} 0" />
    <body name="can" pos="0 0 0.085">
      <freejoint name="can_free" />
      <geom name="can_g0" type="mesh" mesh="can_mesh" mass="0.349" />
    </body>
  </worldbody>
  <contact>
    <pair geom1="incline_plane" geom2="can_g0" friction="{friction}" condim="{int(candidate['condim'])}" margin="{format(float(candidate['margin_m']), '.17g')}" gap="{format(float(candidate['gap_m']), '.17g')}" solref="{solref}" solimp="{solimp}" />
  </contact>
</mujoco>
"""


def _incline_record(angle_rad: float, candidate: Mapping[str, Any], *, duration_s: float) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(_incline_xml(angle_rad, candidate))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_xy = np.asarray(data.qpos[:2], dtype=float).copy()
    steps = int(math.ceil(duration_s / float(model.opt.timestep)))
    max_penetration = 0.0
    warning_before = np.asarray(data.warning.number, dtype=int).copy()
    for _ in range(steps):
        mujoco.mj_step(model, data)
        for index in range(int(data.ncon)):
            max_penetration = max(max_penetration, max(0.0, -float(data.contact[index].dist)))
    displacement = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - initial_xy))
    speed = float(np.linalg.norm(np.asarray(data.qvel[:2], dtype=float)))
    normal_force = 0.0
    for index in range(int(data.ncon)):
        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal_force += abs(float(wrench[0]))
    warnings = int(np.sum(np.asarray(data.warning.number, dtype=int) - warning_before))
    return {
        "angle_rad": float(angle_rad),
        "duration_s": float(duration_s),
        "step_count": steps,
        "horizontal_displacement_m": displacement,
        "terminal_horizontal_speed_m_s": speed,
        "maximum_penetration_m": max_penetration,
        "terminal_normal_force_N": normal_force,
        "warning_count": warnings,
    }


def _incline_bracket(state: ResolvedProbeState) -> dict[str, Any]:
    if state.contact is None:
        raise V7ProtocolError("incline bracket requires a contact child")
    probes = state.contact.probes
    angles = [float(value) for value in probes["incline_angles_rad"]]
    duration_s = float(probes["incline_duration_s"])
    displacement_gate = float(probes["incline_slip_displacement_m"])
    rows = [_incline_record(angle, _contact_candidate_mapping(state), duration_s=duration_s) for angle in angles]
    stable = [row for row in rows if float(row["horizontal_displacement_m"]) <= displacement_gate]
    sliding = [row for row in rows if float(row["horizontal_displacement_m"]) > displacement_gate]
    lower = max((float(row["angle_rad"]) for row in stable), default=None)
    upper = min((float(row["angle_rad"]) for row in sliding), default=None)
    width = None if lower is None or upper is None else upper - lower
    return {
        "measurement": "raw_mujoco_displacement_bracket",
        "slip_displacement_gate_m": displacement_gate,
        "declared_resolution_rad": float(probes["incline_bracket_resolution_rad"]),
        "angle_records": rows,
        "lower_stable_angle_rad": lower,
        "upper_sliding_angle_rad": upper,
        "bracket_width_rad": width,
        "bracket_order_passed": bool(
            lower is not None
            and upper is not None
            and lower < upper
            and width <= float(probes["incline_bracket_resolution_rad"])
        ),
        "analytic_coulomb_comparison_angle_rad": math.atan(0.30),
        "analytic_is_comparison_only": True,
    }


def _recovery_from_trace(
    trace: Any, *, timestep_s: float, gates: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(trace, list) or len(trace) < 2:
        return {}, ["recovery trace is missing or too short"]
    times = []
    linear = []
    angular = []
    penetration = []
    warning_total = 0
    first_impact = None
    for index, row in enumerate(trace):
        if not isinstance(row, Mapping):
            errors.append(f"recovery trace row {index} is not a mapping")
            continue
        try:
            time_s = float(row["time_s"])
            linear_vector = np.asarray(row["linear_velocity_m_s"], dtype=float)
            angular_vector = np.asarray(row["angular_velocity_rad_s"], dtype=float)
            contacts = row["contacts"]
            if linear_vector.shape != (3,) or angular_vector.shape != (3,) or not isinstance(contacts, Mapping):
                raise ValueError("wrong vector/contact shape")
            if not np.all(np.isfinite(linear_vector)) or not np.all(np.isfinite(angular_vector)):
                raise ValueError("non-finite velocity")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"recovery trace row {index} is malformed: {exc}")
            continue
        times.append(time_s)
        linear.append(float(np.linalg.norm(linear_vector)))
        angular.append(float(np.linalg.norm(angular_vector)))
        row_penetration = float(contacts.get("penetration_m", math.nan))
        if not np.isfinite(row_penetration) or row_penetration < 0.0:
            errors.append(f"recovery trace row {index} has invalid penetration")
            row_penetration = math.inf
        penetration.append(row_penetration)
        warning = np.asarray(row.get("warning_number", ()), dtype=int)
        warning_total += int(np.sum(warning))
        if first_impact is None and any(
            float(contact.get("distance_m", 1.0)) <= 0.0
            for contact in contacts.get("contacts", ())
            if isinstance(contact, Mapping)
        ):
            first_impact = time_s
    if len(times) != len(trace):
        return {}, errors or ["recovery trace has invalid rows"]
    expected_times = np.arange(len(times), dtype=float) * timestep_s
    if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-12):
        errors.append("recovery timestamps are not aligned to the declared model timestep")
    duration = float(gates["recovery_duration_s"])
    if not np.isclose(times[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
        times[-1], duration, rtol=0.0, atol=1.0e-12
    ):
        errors.append("recovery trace does not cover the exact frozen horizon")
    tail_start = duration - float(gates["tail_window_s"])
    tail_indices = [index for index, value in enumerate(times) if value >= tail_start - 1.0e-12]
    if not tail_indices:
        errors.append("recovery trailing window is empty")
        tail_indices = [len(times) - 1]
    linear_array = np.asarray(linear, dtype=float)
    angular_array = np.asarray(angular, dtype=float)
    tail_linear = linear_array[tail_indices]
    tail_angular = angular_array[tail_indices]
    result = {
        "release_time_s": times[0],
        "first_impact_time_s": first_impact,
        "duration_s": times[-1],
        "trace_dt_s": timestep_s,
        "trace_sample_count": len(trace),
        "terminal_linear_velocity_m_s": float(linear_array[-1]),
        "terminal_angular_velocity_rad_s": float(angular_array[-1]),
        "tail_window_s": float(gates["tail_window_s"]),
        "tail_window_max_linear_velocity_m_s": float(np.max(tail_linear)),
        "tail_window_rms_linear_velocity_m_s": float(np.sqrt(np.mean(tail_linear**2))),
        "tail_window_max_angular_velocity_rad_s": float(np.max(tail_angular)),
        "tail_window_rms_angular_velocity_rad_s": float(np.sqrt(np.mean(tail_angular**2))),
        "maximum_penetration_m": float(np.max(np.asarray(penetration, dtype=float))),
        "warning_count": warning_total,
    }
    return result, errors


def _finger_from_trace(finger: Any, *, gates: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(finger, Mapping) or not isinstance(finger.get("trace"), list) or not finger["trace"]:
        return {}, ["finger force/penetration trace is missing"]
    max_force = 0.0
    max_penetration = 0.0
    min_force = math.inf
    observed: set[tuple[str, str]] = set()
    for index, row in enumerate(finger["trace"]):
        if not isinstance(row, Mapping) or not isinstance(row.get("contacts"), Mapping):
            errors.append(f"finger trace row {index} is malformed")
            continue
        contacts = row["contacts"]
        force = float(contacts.get("normal_force_N", math.nan))
        penetration = float(contacts.get("penetration_m", math.nan))
        if not np.isfinite(force) or not np.isfinite(penetration):
            errors.append(f"finger trace row {index} has non-finite force/penetration")
            continue
        max_force = max(max_force, force)
        max_penetration = max(max_penetration, penetration)
        if int(contacts.get("contact_count", 0)) > 0:
            min_force = min(min_force, force)
        for contact in contacts.get("contacts", ()):
            if not isinstance(contact, Mapping) or contact.get("interface") != "can_panda_finger_pads":
                errors.append(f"finger trace row {index} contains a wrong contact interface")
                continue
            first = str(contact.get("geom1"))
            second = str(contact.get("geom2"))
            observed.add(tuple(sorted((first, second))))
    if not FINGER_PAIR_NAMES.issubset(observed):
        errors.append("finger trace does not observe every named Can-finger pair")
    result = {
        "maximum_normal_force_N": max_force,
        "maximum_penetration_m": max_penetration,
        "minimum_normal_force_N": 0.0 if min_force == math.inf else min_force,
        "observed_named_pairs": [list(pair) for pair in sorted(observed)],
        "required_named_pairs": [list(pair) for pair in sorted(FINGER_PAIR_NAMES)],
    }
    return result, errors


def _compiled_contact_matches(compiled: Mapping[str, Any], state: ResolvedProbeState) -> tuple[bool, list[str]]:
    if state.contact is None:
        return False, ["resolved contact child is missing"]
    errors: list[str] = []
    pairs = compiled.get("pairs")
    if not isinstance(pairs, list):
        return False, ["compiled contact pair audit is missing"]
    expected_pairs = {
        tuple(sorted(("can_g0", "table_collision"))): 0.30,
        tuple(sorted(("can_g0", "target_container_bottom"))): 0.30,
        tuple(sorted(("can_g0", "target_container_wall_xneg"))): 0.30,
        tuple(sorted(("can_g0", "target_container_wall_xpos"))): 0.30,
        tuple(sorted(("can_g0", "target_container_wall_yneg"))): 0.30,
        tuple(sorted(("can_g0", "target_container_wall_ypos"))): 0.30,
        tuple(sorted(("can_g0", "gripper0_right_finger1_pad_collision"))): 1.00,
        tuple(sorted(("can_g0", "gripper0_right_finger2_pad_collision"))): 1.00,
    }
    actual_pairs = {}
    for row in pairs:
        if not isinstance(row, Mapping):
            errors.append("compiled contact pair row is not a mapping")
            continue
        key = tuple(sorted((str(row.get("geom1")), str(row.get("geom2")))))
        actual_pairs[key] = row
        expected_sliding = expected_pairs.get(key)
        if expected_sliding is None:
            errors.append(f"unexpected compiled contact pair: {key}")
            continue
        friction = row.get("friction")
        expected_friction = [
            expected_sliding,
            state.contact.torsional_mu,
            state.contact.rolling_mu,
            state.contact.rolling_mu,
            state.contact.rolling_mu,
        ]
        if (
            not isinstance(friction, Sequence)
            or len(friction) != 5
            or not np.allclose(np.asarray(friction, dtype=float), expected_friction, rtol=0.0, atol=1.0e-14)
        ):
            errors.append(f"compiled friction tuple mismatch: {key}")
        for name, observed, expected in (
            ("condim", row.get("condim"), state.contact.condim),
            ("margin_m", row.get("margin_m"), state.contact.margin_m),
            ("gap_m", row.get("gap_m"), state.contact.gap_m),
        ):
            if isinstance(expected, int):
                if int(observed) != expected:
                    errors.append(f"compiled {name} mismatch: {key}")
            elif not np.isclose(float(observed), float(expected), rtol=0.0, atol=1.0e-14):
                errors.append(f"compiled {name} mismatch: {key}")
        if not np.allclose(
            np.asarray(row.get("solref"), dtype=float),
            np.asarray(state.contact.solref, dtype=float),
            rtol=0.0,
            atol=1.0e-14,
        ):
            errors.append(f"compiled solref mismatch: {key}")
        if not np.allclose(
            np.asarray(row.get("solimp"), dtype=float),
            np.asarray(state.contact.solimp, dtype=float),
            rtol=0.0,
            atol=1.0e-14,
        ):
            errors.append(f"compiled solimp mismatch: {key}")
    if set(actual_pairs) != set(expected_pairs):
        errors.append("compiled contact pair coverage differs from the eight named interfaces")
    if int(compiled.get("solver_iterations", -1)) != state.contact.iterations:
        errors.append("compiled contact solver iteration count differs from the resolved candidate")
    return not errors, errors


def _recompute_contact_gates(
    evidence: Mapping[str, Any], state: ResolvedProbeState
) -> tuple[dict[str, bool], dict[str, Any], list[str]]:
    if state.contact is None:
        return {}, {}, ["resolved contact child is missing"]
    physics = evidence.get("physics", evidence)
    gates = state.contact.hard_gates
    recovery, recovery_errors = _recovery_from_trace(
        physics.get("recovery_trace"),
        timestep_s=state.common.physics_timestep_s,
        gates={
            "recovery_duration_s": state.contact.probes["recovery_duration_s"],
            "tail_window_s": state.contact.probes["tail_window_s"],
        },
    )
    finger, finger_errors = _finger_from_trace(physics.get("finger_load"), gates=gates)
    incline = physics.get("incline_threshold", {})
    rows = incline.get("angle_records", []) if isinstance(incline, Mapping) else []
    displacement_gate = float(state.contact.probes["incline_slip_displacement_m"])
    resolution = float(state.contact.probes["incline_bracket_resolution_rad"])
    stable = [
        float(row["angle_rad"])
        for row in rows
        if isinstance(row, Mapping) and float(row.get("horizontal_displacement_m", math.inf)) <= displacement_gate
    ]
    sliding = [
        float(row["angle_rad"])
        for row in rows
        if isinstance(row, Mapping) and float(row.get("horizontal_displacement_m", -math.inf)) > displacement_gate
    ]
    lower = max(stable) if stable else None
    upper = min(sliding) if sliding else None
    width = None if lower is None or upper is None else upper - lower
    compiled_ok, compiled_errors = _compiled_contact_matches(physics.get("compiled_contact_profile", {}), state)
    static = physics.get("v6_compatibility", {}).get("static_support", {})
    slip = physics.get("v6_compatibility", {}).get("single_axis_slip", {})
    convergence = physics.get("v6_compatibility", {}).get("timestep_convergence", {})
    convergence_error = max(
        (
            float(row.get("relative_force_error", math.inf))
            for row in convergence.get("records", ())
            if isinstance(row, Mapping)
        ),
        default=math.inf,
    )
    warning_count = int(physics.get("v6_compatibility", {}).get("warning_count", 1)) + int(
        recovery.get("warning_count", 1)
    )
    checks = {
        "compiled_profile": compiled_ok,
        "static_support": float(static.get("normal_force_N", 0.0)) >= float(gates["static_support_force_min_N"]),
        "incline_bracket": bool(lower is not None and upper is not None and lower < upper and width <= resolution),
        "slip": np.isfinite(float(slip.get("final_speed_m_s", math.nan)))
        and float(slip.get("final_speed_m_s", math.inf)) <= float(gates["slip_final_speed_max_m_s"]),
        "recovery_trace": not recovery_errors,
        "recovery_velocity": float(recovery.get("terminal_linear_velocity_m_s", math.inf))
        <= float(gates["recovery_velocity_max_m_s"])
        and float(recovery.get("tail_window_max_linear_velocity_m_s", math.inf))
        <= float(gates["recovery_velocity_max_m_s"]),
        "recovery_angular_velocity": float(recovery.get("terminal_angular_velocity_rad_s", math.inf))
        <= float(gates["recovery_angular_velocity_max_rad_s"])
        and float(recovery.get("tail_window_max_angular_velocity_rad_s", math.inf))
        <= float(gates["recovery_angular_velocity_max_rad_s"]),
        "penetration": float(recovery.get("maximum_penetration_m", math.inf))
        <= float(gates["maximum_illegal_penetration_m"]),
        "finger_trace": not finger_errors,
        "finger_force_minimum": float(finger.get("minimum_normal_force_N", 0.0)) >= float(gates["finger_force_min_N"]),
        "finger_force_maximum": float(finger.get("maximum_normal_force_N", math.inf))
        <= float(gates["finger_force_max_N"]),
        "finger_penetration": float(finger.get("maximum_penetration_m", math.inf))
        <= float(gates["finger_penetration_max_m"]),
        "warnings": warning_count <= int(gates["warning_count_max"]),
        "timestep_convergence": convergence_error <= float(gates["timestep_trace_relative_error_max"]),
    }
    derived = {
        "recovery": recovery,
        "recovery_errors": recovery_errors,
        "finger": finger,
        "finger_errors": finger_errors,
        "incline_bracket": {
            "lower_stable_angle_rad": lower,
            "upper_sliding_angle_rad": upper,
            "bracket_width_rad": width,
            "displacement_gate_m": displacement_gate,
            "resolution_rad": resolution,
        },
        "compiled_errors": compiled_errors,
        "convergence_max_relative_force_error": convergence_error,
        "warning_count": warning_count,
    }
    return checks, derived, recovery_errors + finger_errors + compiled_errors


def _v7_contact_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    if state.stage not in {"contact", "replay"} or state.contact is None:
        raise V7ProtocolError("V7 contact adapter received a state without a contact child")
    if state.stage == "replay" and state.replay is not None and state.replay.selected_component != "contact":
        raise V7ProtocolError("V7 contact probe was requested for a non-contact replay binding")
    profile = _profile_from_state(state)
    candidate = _contact_candidate_mapping(state)
    from robosuite.scripts.shakebench_select_physics import _contact_candidate_probe

    compatibility = _contact_candidate_probe(
        profile,
        candidate,
        run_expensive=True,
        recovery_duration_s=RECOVERY_DURATION_S,
        reset_drop_velocity=True,
    )
    diagnostic = diagnose_state(state, finger_mode="v7_declared")
    physics = {
        "candidate_profile": candidate,
        "compiled_contact_profile": _compile_contact_audit(state),
        "v6_compatibility": compatibility,
        "recovery_trace": diagnostic["recovery_trace"],
        "recovery": diagnostic["recovery"],
        "finger_force_envelope": diagnostic["finger_force_envelope"],
        "finger_load": diagnostic["finger_load"],
        "incline_threshold": _incline_bracket(state),
        "warning_count": (
            int(diagnostic["recovery_trace"][-1]["warning_number"][0]) if diagnostic["recovery_trace"] else 0
        ),
    }
    checks, derived, errors = _recompute_contact_gates({"physics": physics}, state)
    return {
        "candidate_id": state.contact.candidate_id,
        "physics": physics,
        "recomputed_numeric_gates": checks,
        "derived_numeric_metrics": derived,
        "passed": bool(all(checks.values()) and not errors),
        "exclusion_reasons": [] if all(checks.values()) and not errors else ["contact_hard_gate_failed"],
    }


def _v7_contact_score(evidence: Mapping[str, Any], state: ResolvedProbeState) -> tuple[float, float, str]:
    if state.contact is None:
        return math.inf, math.inf, state.candidate_id
    gates = state.contact.hard_gates
    _, derived, _ = _recompute_contact_gates(evidence, state)
    recovery = derived.get("recovery", {})
    finger = derived.get("finger", {})
    bracket = derived.get("incline_bracket", {})
    convergence = float(derived.get("convergence_max_relative_force_error", math.inf))
    values = {
        "incline_bracket_width_rad": float(bracket.get("bracket_width_rad", math.inf)),
        "maximum_penetration_m": float(recovery.get("maximum_penetration_m", math.inf)),
        "recovery_velocity_m_s": float(recovery.get("tail_window_max_linear_velocity_m_s", math.inf)),
        "recovery_angular_velocity_rad_s": float(recovery.get("tail_window_max_angular_velocity_rad_s", math.inf)),
        "finger_force_relative_error": abs(
            float(finger.get("maximum_normal_force_N", math.inf)) / max(float(gates["finger_force_max_N"]), 1.0e-12)
            - 1.0
        ),
        "timestep_trace_relative_error": convergence,
    }
    limits = {
        "incline_bracket_width_rad": float(gates["incline_bracket_resolution_rad"]),
        "maximum_penetration_m": float(gates["maximum_illegal_penetration_m"]),
        "recovery_velocity_m_s": float(gates["recovery_velocity_max_m_s"]),
        "recovery_angular_velocity_rad_s": float(gates["recovery_angular_velocity_max_rad_s"]),
        "finger_force_relative_error": 1.0,
        "timestep_trace_relative_error": float(gates["timestep_trace_relative_error_max"]),
    }
    weights = state.contact.scoring.get("weights", {})
    distance = sum(
        float(weights.get(key, 1.0)) * values[key] / max(limits[key], 1.0e-12) for key in values if key in weights
    )
    margin = min(
        float(gates["recovery_velocity_max_m_s"]) / max(values["recovery_velocity_m_s"], 1.0e-12),
        float(gates["recovery_angular_velocity_max_rad_s"]) / max(values["recovery_angular_velocity_rad_s"], 1.0e-12),
        float(gates["maximum_illegal_penetration_m"]) / max(values["maximum_penetration_m"], 1.0e-12),
    )
    return float(distance), float(-margin), state.contact.candidate_id


def _raw_payload(
    state: ResolvedProbeState, evidence: Mapping[str, Any], *, attempt: int, retry_ledger: list[dict[str, Any]]
) -> dict[str, Any]:
    identity = state.common.protocol_identity
    payload = {
        "schema_id": SCHEMA_ID + ".raw",
        "schema_version": SCHEMA_VERSION,
        "stage": state.stage,
        "state_id": state.state_id,
        "protocol_sha256_bytes": identity.get("bytes_sha256"),
        "protocol_sha256_normalized": identity.get("normalized_sha256"),
        "resolved_state_digest": state.resolved_state_digest,
        "attempt": attempt,
        "retry_ledger": copy.deepcopy(retry_ledger),
        "evidence": dict(evidence),
    }
    payload["payload_sha256"] = payload_hash(payload)
    return payload


def _verify_existing_raw(path: Path, state: ResolvedProbeState) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V7EvidenceError(f"cannot read V7 raw artifact: {path.name}") from exc
    identity = state.common.protocol_identity
    if (
        payload.get("state_id") != state.state_id
        or payload.get("stage") != state.stage
        or payload.get("resolved_state_digest") != state.resolved_state_digest
    ):
        raise V7EvidenceError(f"V7 raw/resolved-state mismatch: {state.state_id}")
    if (
        payload.get("protocol_sha256_bytes") != identity.get("bytes_sha256")
        or payload.get("protocol_sha256_normalized") != identity.get("normalized_sha256")
        or not verify_payload_hash(payload)
    ):
        raise V7EvidenceError(f"V7 raw hash mismatch: {state.state_id}")
    return payload


def run_v7_stage(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    output_dir: str | Path,
    probe: Any,
    protocol_bytes_hash: str,
    selection_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    states = [
        state
        for state in _resolve_v7(protocol, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)
        if state.stage == stage
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for state in states:
        path = output / state.output
        if path.is_file():
            records.append(_verify_existing_raw(path, state))
            continue
        ledger: list[dict[str, Any]] = []
        for attempt in (1, 2):
            try:
                evidence = probe(state)
                if not isinstance(evidence, Mapping):
                    raise V7EvidenceError(f"{stage} adapter returned non-mapping evidence")
                payload = _raw_payload(state, evidence, attempt=attempt, retry_ledger=ledger)
                write_json_atomic(path, payload)
                records.append(payload)
                break
            except (V7ProtocolError, V7ProtocolStateError, V7EvidenceError, AdapterContractError):
                raise
            except Exception as exc:
                ledger.append(
                    {
                        "attempt": attempt,
                        "state_config_identical": True,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "failure_taxonomy": "infrastructure_failure",
                    }
                )
        else:
            payload = _raw_payload(
                state, {"status": "infrastructure_failure_repeated", "passed": False}, attempt=2, retry_ledger=ledger
            )
            write_json_atomic(path, payload)
            raise V7EvidenceError(f"repeated infrastructure failure: {state.state_id}")
    return records


def _v7_replay_worker(protocol_path: str, state_id: str) -> int:
    protocol, _, protocol_hash = load_v7_protocol(protocol_path)
    validate_v7_protocol(protocol, protocol_bytes_hash=protocol_hash)
    state = next(
        state for state in _resolve_v7(protocol, protocol_bytes_hash=protocol_hash) if state.state_id == state_id
    )
    print(json.dumps(_v6_replay_probe(state), ensure_ascii=False, sort_keys=True))
    return 0


def _subprocess_replay_probe(protocol_path: str | Path, state: ResolvedProbeState) -> Mapping[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robosuite.scripts.shakebench_select_physics_v7",
            "--worker",
            "--protocol",
            str(Path(protocol_path).resolve()),
            "--state-id",
            state.state_id,
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise V7EvidenceError(completed.stderr[-4000:] or completed.stdout[-4000:])
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "trace_digest" in value:
            return value
    raise V7EvidenceError("V7 replay worker did not emit a complete trace record")


def _verify_v6_inherited_evidence(asset_root: str | Path | None = None) -> dict[str, Any]:
    """Recompute the V6 non-contact result without inheriting its contact row."""

    v6_protocol, v6_name, v6_hash = load_v6_protocol(_v6_asset(V6_PROTOCOL_FILENAME, asset_root))
    v6_structure = validate_v6_protocol(v6_protocol, protocol_bytes_hash=v6_hash)
    selected_path = _v6_asset(V6_SELECTED_FILENAME, asset_root)
    status_path = _v6_asset(V6_STATUS_FILENAME, asset_root)
    feasibility_path = _v6_asset("shakebench_phase_06r5_v6_feasibility.json", asset_root)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if (
        status.get("status") != "BLOCKED"
        or status.get("blocking_stage") != "contact"
        or status.get("failure_taxonomy") != "physics_gate_failure"
    ):
        errors.append("V6 global BLOCKED contact status is not authoritative")
    if (
        selected.get("eligible_counts", {}).get("contact") != 0
        or selected.get("selection", {}).get("selected_candidate_ids", {}).get("contact") != "c3_nominal"
    ):
        errors.append("V6 historical contact inconsistency was not preserved for audit")
    if selected.get("official_profile_alignment") is not False:
        errors.append("V6 historical profile alignment marker changed")
    if not verify_payload_hash(selected) or not verify_payload_hash(status) or not verify_payload_hash(feasibility):
        errors.append("V6 inherited artifact payload hash failed")
    if (
        status.get("protocol", {}).get("bytes_sha256") != v6_hash
        or feasibility.get("protocol", {}).get("bytes_sha256") != v6_hash
    ):
        errors.append("V6 inherited protocol hash failed")
    if feasibility.get("resolved_state_digest") != v6_structure["resolved_state_digest"] or status.get(
        "protocol", {}
    ).get("normalized_sha256") != sha256_json(v6_protocol):
        errors.append("V6 inherited resolved/normalized hash failed")
    v6_states = _resolve(v6_protocol, protocol_bytes_hash=v6_hash)
    state_by_id = {state.state_id: state for state in v6_states}
    raw: dict[str, Mapping[str, Any]] = {}
    inherited_files = []
    for entry in selected.get("raw_files", ()):
        if not isinstance(entry, Mapping):
            continue
        path = selected_path.parent / str(entry.get("path", ""))
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            if str(entry.get("stage")) in NONCONTACT_STAGES:
                errors.append(f"V6 inherited raw file hash failed: {path.name}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw[str(payload.get("state_id"))] = payload
        state = state_by_id.get(str(payload.get("state_id")))
        if state is None:
            continue
        if state.stage in NONCONTACT_STAGES:
            identity = state.common.protocol_identity
            if (
                not verify_payload_hash(payload)
                or payload.get("stage") != state.stage
                or payload.get("resolved_state_digest") != state.resolved_state_digest
                or payload.get("protocol_sha256_bytes") != identity.get("bytes_sha256")
                or payload.get("protocol_sha256_normalized") != identity.get("normalized_sha256")
            ):
                errors.append(f"V6 inherited raw provenance failed: {state.state_id}")
            inherited_files.append(
                {
                    "stage": state.stage,
                    "state_id": state.state_id,
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "resolved_state_digest": state.resolved_state_digest,
                }
            )
    if (
        len([item for item in inherited_files if item["stage"] == "driver"]) != 18
        or len([item for item in inherited_files if item["stage"] == "isolator"]) != 3
    ):
        errors.append("V6 inherited non-contact raw coverage is incomplete")
    driver = _driver_eligibility(state_by_id, raw)
    selected_driver = _select_driver(driver, state_by_id)
    if selected_driver != "dt_nominal" or not all(item.get("eligible") is True for item in driver.values()):
        errors.append("V6 driver recomputation did not select dt_nominal")
    isolator_candidates = {}
    for payload in raw.values():
        state = state_by_id.get(str(payload.get("state_id")))
        if state is not None and state.stage == "isolator" and state.isolator is not None:
            isolator_candidates[state.candidate_id] = payload
    isolator_eligible = {
        candidate_id: _isolator_eligible(
            payload,
            next(state for state in v6_states if state.stage == "isolator" and state.candidate_id == candidate_id),
        )
        for candidate_id, payload in isolator_candidates.items()
    }
    eligible_ids = [candidate_id for candidate_id, passed in isolator_eligible.items() if passed]
    selected_isolator = None
    if eligible_ids:
        selected_isolator = sorted(
            eligible_ids,
            key=lambda candidate_id: _isolator_score(
                isolator_candidates[candidate_id],
                next(state for state in v6_states if state.stage == "isolator" and state.candidate_id == candidate_id),
            ),
        )[0]
    if selected_isolator != "low_frequency_damped" or set(isolator_candidates) != {
        "balanced_nominal",
        "low_frequency_damped",
        "high_frequency_light_damping",
    }:
        errors.append("V6 isolator recomputation did not select low_frequency_damped")
    if errors:
        raise V7ProtocolError("V6 inherited evidence failed: " + "; ".join(errors))
    return {
        "protocol_path": v6_name,
        "protocol_sha256": v6_hash,
        "selected_driver": selected_driver,
        "selected_isolator": selected_isolator,
        "raw_files": inherited_files,
    }


def _compile_force_envelope(state: ResolvedProbeState) -> dict[str, Any]:
    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    if state.contact is None:
        raise V7ProtocolError("force-envelope audit requires a contact child")
    profile = _profile_from_state(state)
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=state.common.physics_timestep_s,
        target_container_friction=(0.30, state.contact.torsional_mu, state.contact.rolling_mu),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=20,
        seed=17,
    )
    try:
        return _force_envelope_audit(
            env.sim.model._model,
            finger_actuator_names=("gripper0_right_gripper_finger_joint1", "gripper0_right_gripper_finger_joint2"),
        )
    finally:
        env.close()


def _profile_from_selected_state(
    state: ResolvedProbeState, *, protocol_hash: str, evidence_manifest_hash: str
) -> dict[str, Any]:
    if state.contact is None or state.isolator is None:
        raise V7ProtocolError("selected V7 state is incomplete")
    payload = _profile_from_state(state).to_dict()
    payload.update(
        {
            "profile_id": "shakebench.official.physics.v1",
            "status": "official_immutable",
            "scoreable": True,
            "selection_basis": "phase06r6_v7_physics_only",
            "protocol_file": PROTOCOL_FILENAME,
            "protocol_sha256": protocol_hash,
            "evidence_manifest_sha256": evidence_manifest_hash,
            "freeze_commit": "phase_06r6_v7_contact_recovery_registration",
        }
    )
    payload["profile_sha256"] = physics_profile_hash(payload)
    return (
        PhysicsProfile(
            payload=payload, source="V7 independently verified selection", profile_sha256=payload["profile_sha256"]
        )
        .assert_valid()
        .to_dict()
    )


def _artifact_rows(
    states: Mapping[str, ResolvedProbeState], records: Sequence[Mapping[str, Any]], output: Path
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        state = states.get(str(record.get("state_id")))
        if state is None:
            raise V7EvidenceError(f"record is not a resolved V7 state: {record.get('state_id')}")
        path = output / state.output
        rows.append(
            {
                "stage": state.stage,
                "state_id": state.state_id,
                "path": path.name,
                "sha256": file_sha256(path),
                "resolved_state_digest": state.resolved_state_digest,
            }
        )
    return rows


def _write_status(
    output: Path,
    *,
    status: str,
    protocol_hash: str,
    normalized_hash: str,
    feasibility_hash: str | None,
    adapter_digest: str | None,
    reason: str | None = None,
    selected_path: Path | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_id": "shakebench.phase06r6.v7.selection_status",
        "schema_version": 1,
        "status": status,
        "failure_taxonomy": (
            None
            if status == "PASS"
            else (
                "invalid_protocol_configuration"
                if reason and reason.startswith("invalid_protocol_configuration")
                else "physics_gate_failure"
            )
        ),
        "reason": reason,
        "blocking_stage": None if status == "PASS" else "contact",
        "protocol": {"path": PROTOCOL_FILENAME, "bytes_sha256": protocol_hash, "normalized_sha256": normalized_hash},
        "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash},
        "adapter_contract_digest": adapter_digest,
        "official_profile_publication": "PASS" if status == "PASS" else "forbidden",
        "phase07": "authorized_after_independent_verification" if status == "PASS" else "forbidden",
    }
    if selected_path is not None:
        value["selection_artifact"] = {"path": selected_path.name, "sha256": file_sha256(selected_path)}
    if profile is not None:
        profile_path = output / OFFICIAL_PROFILE_FILENAME
        value["official_profile"] = {
            "path": profile_path.name,
            "file_sha256": file_sha256(profile_path),
            "profile_sha256": profile["profile_sha256"],
        }
    value["payload_sha256"] = payload_hash(value)
    write_json_atomic(output / STATUS_FILENAME, value)
    return value


def _replay_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("evidence", {}).get("selected_component"))].append(record)
    result = {}
    for group, rows in groups.items():
        ordered = sorted(rows, key=lambda item: int(item.get("evidence", {}).get("process_index", 0)))
        evidence = [row.get("evidence", {}) for row in ordered]
        result[group] = {
            "process_count": len(evidence),
            "process_indices": [row.get("process_index") for row in evidence],
            "trace_digests": [row.get("trace_digest") for row in evidence],
            "metric_digests": [row.get("metric_digest") for row in evidence],
            "trace_fields": evidence[0].get("trace_fields", []) if evidence else [],
            "complete_trace": all(row.get("complete_trace") is True for row in evidence),
        }
    return result


def _write_blocked_selection(
    *,
    output: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    normalized_hash: str,
    structure: Mapping[str, Any],
    contract: Mapping[str, Any],
    feasibility_hash: str | None,
    inherited: Mapping[str, Any],
    states: Mapping[str, ResolvedProbeState],
    stage_records: Sequence[Mapping[str, Any]],
    parity_path: Path,
    determinism_path: Path,
    reason: str,
) -> dict[str, Any]:
    excluded = []
    contact_records = [record for record in stage_records if record.get("stage") == "contact"]
    for record in contact_records:
        evidence = record.get("evidence", {})
        state = states.get(str(record.get("state_id")))
        if state is None:
            continue
        checks, derived, errors = _recompute_contact_gates(evidence, state)
        excluded.append(
            {
                "stage": "contact",
                "candidate_id": state.candidate_id,
                "reason": "contact_hard_gate_failed",
                "recomputed_gates": checks,
                "diagnostic_errors": errors,
                "metrics": derived,
            }
        )
    selected = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "physics_only": True,
        "protocol": {
            "path": PROTOCOL_FILENAME,
            "bytes_sha256": protocol_hash,
            "normalized_sha256": normalized_hash,
            "status": protocol.get("status"),
        },
        "resolved_state_digest": structure["resolved_state_digest"],
        "adapter_contract_digest": contract["adapter_contract_digest"],
        "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash},
        "selection": {
            "selected_candidate_ids": {
                "driver": inherited["selected_driver"],
                "isolator": inherited["selected_isolator"],
                "contact": None,
            },
            "physics_only": True,
            "task_success_used": False,
            "reward_used": False,
            "controller_outcome_used": False,
        },
        "candidate_counts": {"driver": 18, "isolator": 3, "contact": len(contact_records)},
        "eligible_counts": {"driver": 3, "isolator": 1, "contact": 0},
        "inherited_v6_raw_files": list(inherited["raw_files"]),
        "raw_files": _artifact_rows(states, list(stage_records), output),
        "official_profile": None,
        "official_profile_alignment": False,
        "replay_groups": sorted(REPLAY_GROUPS),
        "blocking_reason": reason,
        "parity_artifact": {"path": parity_path.name, "sha256": file_sha256(parity_path)},
        "determinism_artifact": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)},
        "excluded": excluded,
    }
    # The parity artifact is not a raw manifest row; append the manifest-state
    # raw rows explicitly from the caller's output files below.
    selected["payload_sha256"] = payload_hash(selected)
    selected_path = output / SELECTED_FILENAME
    write_json_atomic(selected_path, selected)
    excluded_payload = {
        "schema_id": SCHEMA_ID + ".excluded",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_sha256_bytes": protocol_hash,
        "selected_candidate_ids": selected["selection"]["selected_candidate_ids"],
        "excluded": excluded,
        "physics_only": True,
    }
    excluded_payload["payload_sha256"] = payload_hash(excluded_payload)
    write_json_atomic(output / EXCLUDED_FILENAME, excluded_payload)
    status = _write_status(
        output,
        status="BLOCKED",
        protocol_hash=protocol_hash,
        normalized_hash=normalized_hash,
        feasibility_hash=feasibility_hash,
        adapter_digest=contract["adapter_contract_digest"],
        reason=reason,
        selected_path=selected_path,
    )
    return {"status": status, "selected": selected}


def run_v7_selection(
    *, protocol_path: str | Path | None = None, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run all V7 contact/parity/replay stages and publish only on PASS."""

    protocol, protocol_name, protocol_hash = load_v7_protocol(protocol_path)
    structure = validate_v7_protocol(protocol, protocol_bytes_hash=protocol_hash)
    contract = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
    inherited = _verify_v6_inherited_evidence()
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feasibility_path = output / FEASIBILITY_FILENAME
    feasibility_hash = file_sha256(feasibility_path) if feasibility_path.is_file() else None
    default_binding = str(
        _mapping(protocol.get("selection"), "selection").get("default", {}).get("contact_candidate_id", "")
    )
    if default_binding not in CONTACT_IDS:
        raise V7ProtocolError("V7 selection.default contact binding is not registered")
    context = {
        "driver_candidate_id": inherited["selected_driver"],
        "isolator_candidate_id": inherited["selected_isolator"],
    }
    states = {
        state.state_id: state
        for state in _resolve_v7(
            protocol,
            protocol_bytes_hash=protocol_hash,
            selection_context={**context, "contact_candidate_id": default_binding},
        )
    }
    contact_records = run_v7_stage(
        protocol,
        stage="contact",
        output_dir=output,
        probe=_v7_contact_probe,
        protocol_bytes_hash=protocol_hash,
        selection_context=context,
    )
    contact_candidates = {states[str(record["state_id"])].candidate_id: record for record in contact_records}
    eligible_contacts = []
    contact_gate_details = {}
    for candidate_id, record in contact_candidates.items():
        state = states[str(record["state_id"])]
        checks, derived, errors = _recompute_contact_gates(record.get("evidence", {}), state)
        contact_gate_details[candidate_id] = {"checks": checks, "derived": derived, "errors": errors}
        if all(checks.values()) and not errors:
            eligible_contacts.append(candidate_id)
    ranked_contacts = sorted(
        eligible_contacts,
        key=lambda candidate_id: _v7_contact_score(
            contact_candidates[candidate_id]["evidence"], states[str(contact_candidates[candidate_id]["state_id"])]
        ),
    )
    selected_contact = ranked_contacts[0] if ranked_contacts else None
    selected_binding = selected_contact if selected_contact is not None else default_binding
    context["contact_candidate_id"] = selected_binding
    # A pre-registered manifest binds parity/replay to its default contact.
    # If a different contact wins, the registered binding is not silently
    # changed; the run is blocked after all contact evidence is retained.
    binding_mismatch = selected_contact is not None and selected_contact != default_binding
    parity_records: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    stage_records: list[Mapping[str, Any]] = list(contact_records)
    if not binding_mismatch:
        parity_records = run_v7_stage(
            protocol,
            stage="parity",
            output_dir=output,
            probe=_v6_parity_probe,
            protocol_bytes_hash=protocol_hash,
            selection_context=context,
        )
        replay_records = run_v7_stage(
            protocol,
            stage="replay",
            output_dir=output,
            probe=lambda state: _subprocess_replay_probe(protocol_name, state),
            protocol_bytes_hash=protocol_hash,
            selection_context=context,
        )
        stage_records.extend(parity_records)
        stage_records.extend(replay_records)
    else:
        # The manifest binding prevents a candidate from becoming selected
        # without a new registered parity/replay tuple.  No profile is
        # published, and the selected table remains null per the contract.
        reason = "physics_gate_failure: winning contact differs from registered parity/replay binding"
        empty_parity_path = output / PARITY_FILENAME
        parity_payload = {
            "schema_id": SCHEMA_ID + ".parity",
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "protocol_sha256_bytes": protocol_hash,
            "protocol_sha256_normalized": sha256_json(protocol),
            "reason": reason,
        }
        parity_payload["payload_sha256"] = payload_hash(parity_payload)
        write_json_atomic(empty_parity_path, parity_payload)
        empty_determinism_path = output / DETERMINISM_FILENAME
        determinism_payload = {
            "schema_id": SCHEMA_ID + ".replay_determinism",
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "protocol_sha256_bytes": protocol_hash,
            "protocol_sha256_normalized": sha256_json(protocol),
            "groups": {},
            "reason": reason,
        }
        determinism_payload["payload_sha256"] = payload_hash(determinism_payload)
        write_json_atomic(empty_determinism_path, determinism_payload)
        result = _write_blocked_selection(
            output=output,
            protocol=protocol,
            protocol_hash=protocol_hash,
            normalized_hash=sha256_json(protocol),
            structure=structure,
            contract=contract,
            feasibility_hash=feasibility_hash,
            inherited=inherited,
            states=states,
            stage_records=stage_records,
            parity_path=empty_parity_path,
            determinism_path=empty_determinism_path,
            reason=reason,
        )
        return result

    parity_state = next(state for state in states.values() if state.stage == "parity")
    parity_record = parity_records[0] if parity_records else {}
    parity_artifact = {
        "schema_id": SCHEMA_ID + ".parity",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if parity_record and _parity_eligible(parity_record, parity_state) else "BLOCKED",
        "protocol_sha256_bytes": protocol_hash,
        "protocol_sha256_normalized": sha256_json(protocol),
        "resolved_state_digest": parity_state.resolved_state_digest,
        "raw_file": {"path": parity_state.output, "state_id": parity_state.state_id},
        "evidence": parity_record.get("evidence", {}),
    }
    parity_artifact["payload_sha256"] = payload_hash(parity_artifact)
    parity_path = output / PARITY_FILENAME
    write_json_atomic(parity_path, parity_artifact)
    replay_groups = _replay_groups(replay_records)
    determinism_ok = set(replay_groups) >= REPLAY_GROUPS and all(
        row["process_count"] == 3
        and len(set(row["trace_digests"])) == 1
        and len(set(row["metric_digests"])) == 1
        and row["complete_trace"]
        for row in replay_groups.values()
    )
    determinism_artifact = {
        "schema_id": SCHEMA_ID + ".replay_determinism",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if determinism_ok else "BLOCKED",
        "protocol_sha256_bytes": protocol_hash,
        "protocol_sha256_normalized": sha256_json(protocol),
        "resolved_state_digest": structure["resolved_state_digest"],
        "independent_process_count": 3,
        "groups": replay_groups,
        "same_process_reset_used": False,
    }
    determinism_artifact["payload_sha256"] = payload_hash(determinism_artifact)
    determinism_path = output / DETERMINISM_FILENAME
    write_json_atomic(determinism_path, determinism_artifact)
    stage_rows = _artifact_rows(states, stage_records, output)
    excluded = []
    for candidate_id in sorted(CONTACT_IDS):
        if candidate_id not in eligible_contacts:
            excluded.append(
                {
                    "stage": "contact",
                    "candidate_id": candidate_id,
                    "reason": "contact_hard_gate_failed",
                    "recomputed_gates": contact_gate_details[candidate_id]["checks"],
                    "diagnostic_errors": contact_gate_details[candidate_id]["errors"],
                    "metrics": contact_gate_details[candidate_id]["derived"],
                }
            )
    overall_pass = (
        selected_contact is not None
        and selected_contact == default_binding
        and parity_artifact["status"] == "PASS"
        and determinism_ok
    )
    final_selected_contact = selected_contact if overall_pass else None
    reason = (
        None
        if overall_pass
        else (
            "physics_gate_failure: no contact candidate passed"
            if selected_contact is None
            else "evidence_integrity_failure: parity or replay verification failed"
        )
    )
    selected = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if overall_pass else "BLOCKED",
        "physics_only": True,
        "protocol": {
            "path": Path(protocol_name).name,
            "bytes_sha256": protocol_hash,
            "normalized_sha256": sha256_json(protocol),
            "status": protocol.get("status"),
        },
        "resolved_state_digest": structure["resolved_state_digest"],
        "adapter_contract_digest": contract["adapter_contract_digest"],
        "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash},
        "selection": {
            "selected_candidate_ids": {
                "driver": inherited["selected_driver"],
                "isolator": inherited["selected_isolator"],
                "contact": final_selected_contact,
            },
            "physics_only": True,
            "task_success_used": False,
            "reward_used": False,
            "controller_outcome_used": False,
        },
        "candidate_counts": {"driver": 18, "isolator": 3, "contact": len(contact_records)},
        "eligible_counts": {"driver": 3, "isolator": 1, "contact": len(eligible_contacts)},
        "inherited_v6_raw_files": list(inherited["raw_files"]),
        "raw_files": stage_rows,
        "official_profile": None,
        "official_profile_alignment": False,
        "replay_groups": sorted(REPLAY_GROUPS),
        "blocking_reason": reason,
        "parity_artifact": {"path": parity_path.name, "sha256": file_sha256(parity_path)},
        "determinism_artifact": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)},
        "excluded": excluded,
    }
    selected["payload_sha256"] = payload_hash(selected)
    selected_path = output / SELECTED_FILENAME
    write_json_atomic(selected_path, selected)
    excluded_payload = {
        "schema_id": SCHEMA_ID + ".excluded",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_sha256_bytes": protocol_hash,
        "selected_candidate_ids": selected["selection"]["selected_candidate_ids"],
        "excluded": excluded,
        "physics_only": True,
    }
    excluded_payload["payload_sha256"] = payload_hash(excluded_payload)
    write_json_atomic(output / EXCLUDED_FILENAME, excluded_payload)
    if not overall_pass:
        status = _write_status(
            output,
            status="BLOCKED",
            protocol_hash=protocol_hash,
            normalized_hash=sha256_json(protocol),
            feasibility_hash=feasibility_hash,
            adapter_digest=contract["adapter_contract_digest"],
            reason=reason,
            selected_path=selected_path,
        )
        return {"status": status, "selected": selected, "contact_gate_details": contact_gate_details}

    evidence_manifest_hash = sha256_json(
        {
            "raw_files": stage_rows,
            "parity": {"path": parity_path.name, "sha256": file_sha256(parity_path)},
            "determinism": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)},
        }
    )
    selected_state = states[
        next(
            state_id
            for state_id, state in states.items()
            if state.stage == "contact" and state.candidate_id == final_selected_contact
        )
    ]
    profile = _profile_from_selected_state(
        selected_state, protocol_hash=protocol_hash, evidence_manifest_hash=evidence_manifest_hash
    )
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
    verification = verify_v7_selection_artifact(
        selected_path, protocol_path=protocol_name, require_published_status=False
    )
    if verification.get("passed") is not True:
        try:
            profile_path.unlink()
        except FileNotFoundError:
            pass
        selected["status"] = "BLOCKED"
        selected["selection"]["selected_candidate_ids"]["contact"] = None
        selected["official_profile"] = None
        selected["official_profile_alignment"] = False
        selected["blocking_reason"] = "evidence_integrity_failure: independent V7 verifier failed"
        selected["payload_sha256"] = payload_hash(selected)
        write_json_atomic(selected_path, selected)
        status = _write_status(
            output,
            status="BLOCKED",
            protocol_hash=protocol_hash,
            normalized_hash=sha256_json(protocol),
            feasibility_hash=feasibility_hash,
            adapter_digest=contract["adapter_contract_digest"],
            reason=selected["blocking_reason"],
            selected_path=selected_path,
        )
        return {"status": status, "verification": verification, "selected": selected}
    status = _write_status(
        output,
        status="PASS",
        protocol_hash=protocol_hash,
        normalized_hash=sha256_json(protocol),
        feasibility_hash=feasibility_hash,
        adapter_digest=contract["adapter_contract_digest"],
        selected_path=selected_path,
        profile=profile,
    )
    return {"status": status, "verification": verification, "selected": selected}


def _v7_raw_index(
    selected: Mapping[str, Any], selected_path: Path, states: Mapping[str, ResolvedProbeState]
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    errors: list[str] = []
    raw: dict[str, Mapping[str, Any]] = {}
    entries = selected.get("raw_files")
    if not isinstance(entries, list):
        return {}, ["V7 raw_files is missing"]
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("V7 raw manifest entry is not a mapping")
            continue
        path = selected_path.parent / str(entry.get("path", ""))
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            errors.append("V7 raw file hash mismatch: " + path.name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("V7 raw JSON is unreadable: " + path.name)
            continue
        state_id = str(payload.get("state_id"))
        state = states.get(state_id)
        if state is None:
            errors.append("V7 raw state is not in the resolved manifest: " + state_id)
            continue
        if state.stage in NONCONTACT_STAGES:
            errors.append("V7 raw manifest must inherit driver/isolator evidence, not duplicate it: " + state_id)
        identity = state.common.protocol_identity
        if (
            not verify_payload_hash(payload)
            or payload.get("stage") != state.stage
            or payload.get("resolved_state_digest") != state.resolved_state_digest
            or payload.get("protocol_sha256_bytes") != identity.get("bytes_sha256")
            or payload.get("protocol_sha256_normalized") != identity.get("normalized_sha256")
        ):
            errors.append("V7 raw/resolved-state provenance mismatch: " + state_id)
        if entry.get("path") != state.output or not str(entry.get("path", "")).startswith(RAW_PREFIX):
            errors.append("V7 raw entry path disagrees with the resolved V7 output: " + state_id)
        if state_id in raw:
            errors.append("duplicate V7 raw state: " + state_id)
        raw[state_id] = payload
    expected = {state.state_id for state in states.values() if state.stage not in NONCONTACT_STAGES}
    if set(raw) != expected:
        errors.append("V7 raw state coverage differs from contact/parity/replay manifest")
    return raw, errors


def _verify_v6_reference_rows(
    selected: Mapping[str, Any], selected_path: Path, inherited: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    rows = selected.get("inherited_v6_raw_files")
    expected = {(row["stage"], row["state_id"], row["path"], row["sha256"]) for row in inherited["raw_files"]}
    actual = set()
    if not isinstance(rows, list):
        return ["V7 inherited_v6_raw_files is missing"]
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("V7 inherited V6 row is not a mapping")
            continue
        path = selected_path.parent / str(row.get("path", ""))
        if path.is_file() and file_sha256(path) == row.get("sha256"):
            actual.add((row.get("stage"), row.get("state_id"), row.get("path"), row.get("sha256")))
        else:
            errors.append("V7 inherited V6 raw hash mismatch: " + path.name)
    if actual != expected:
        errors.append("V7 inherited V6 raw reference set differs from independently recomputed evidence")
    return errors


def _read_json(path: Path, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, [f"{label} is unreadable"]
    if not isinstance(value, Mapping):
        return None, [f"{label} is not a mapping"]
    return value, []


def _verify_v7_profile(
    profile_path: Path,
    *,
    selected: Mapping[str, Any],
    protocol_hash: str,
    stage_rows: Sequence[Mapping[str, Any]],
    parity_path: Path,
    determinism_path: Path,
    state: ResolvedProbeState,
) -> list[str]:
    errors: list[str] = []
    if not profile_path.is_file():
        return ["official V7 profile is missing"]
    try:
        import yaml

        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"official V7 profile is unreadable: {exc}"]
    if not isinstance(profile, Mapping):
        return ["official V7 profile is not a mapping"]
    try:
        PhysicsProfile(
            payload=profile, source=str(profile_path), profile_sha256=str(profile.get("profile_sha256"))
        ).assert_valid()
    except Exception as exc:
        errors.append(f"official V7 profile validation failed: {exc}")
    if profile.get("protocol_sha256") != protocol_hash:
        errors.append("official V7 profile does not authenticate the V7 protocol")
    evidence_manifest_hash = sha256_json(
        {
            "raw_files": list(stage_rows),
            "parity": {"path": parity_path.name, "sha256": file_sha256(parity_path)},
            "determinism": {"path": determinism_path.name, "sha256": file_sha256(determinism_path)},
        }
    )
    if profile.get("evidence_manifest_sha256") != evidence_manifest_hash:
        errors.append("official V7 profile evidence manifest hash mismatch")
    if profile.get("physics", {}).get("contact", {}).get("condim") != state.contact.condim or not np.isclose(
        float(profile.get("physics", {}).get("contact", {}).get("rolling_mu", math.nan)),
        state.contact.rolling_mu,
        rtol=0.0,
        atol=1.0e-14,
    ):
        errors.append("official V7 profile contact tuple does not match selected state")
    if profile.get("physics", {}).get("isolator", {}).get("candidate_id") != "low_frequency_damped":
        errors.append("official V7 profile does not bind the inherited isolator")
    return errors


def verify_v7_selection_artifact(
    path: str | Path, *, protocol_path: str | Path | None = None, require_published_status: bool = True
) -> dict[str, Any]:
    """Independently recompute V6 inheritance and all V7 numeric gates."""

    selected_path = Path(path)
    errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if not isinstance(selected, Mapping):
            raise V7ProtocolError("V7 selected artifact is not a mapping")
        protocol, protocol_name, protocol_hash = load_v7_protocol(protocol_path)
        structure = validate_v7_protocol(protocol, protocol_bytes_hash=protocol_hash)
        states_tuple = _resolve_v7(protocol, protocol_bytes_hash=protocol_hash)
    except (OSError, json.JSONDecodeError, V7ProtocolError, V7ProtocolStateError) as exc:
        return {"passed": False, "errors": [str(exc)], "checks": checks}
    states = {state.state_id: state for state in states_tuple}
    inherited = None
    try:
        inherited = _verify_v6_inherited_evidence()
        errors.extend(_verify_v6_reference_rows(selected, selected_path, inherited))
    except V7ProtocolError as exc:
        errors.append(str(exc))
    checks["protocol_bytes_hash"] = selected.get("protocol", {}).get("bytes_sha256") == protocol_hash
    checks["protocol_normalized_hash"] = selected.get("protocol", {}).get("normalized_sha256") == sha256_json(protocol)
    checks["resolved_state_digest"] = selected.get("resolved_state_digest") == structure["resolved_state_digest"]
    if not checks["protocol_bytes_hash"]:
        errors.append("V7 selected protocol bytes hash mismatch")
    if not checks["protocol_normalized_hash"]:
        errors.append("V7 selected normalized protocol hash mismatch")
    if not checks["resolved_state_digest"]:
        errors.append("V7 selected resolved-state digest mismatch")
    if not verify_payload_hash(selected):
        errors.append("V7 selected payload hash mismatch")
    selection = selected.get("selection", {})
    if (
        selected.get("physics_only") is not True
        or selection.get("task_success_used") is not False
        or selection.get("reward_used") is not False
        or selection.get("controller_outcome_used") is not False
    ):
        errors.append("V7 selection contains a forbidden non-physics input")

    feasibility_ref = selected.get("feasibility", {})
    feasibility_path = selected_path.parent / str(feasibility_ref.get("path", FEASIBILITY_FILENAME))
    feasibility, feasibility_errors = _read_json(feasibility_path, "V7 feasibility artifact")
    errors.extend(feasibility_errors)
    if feasibility is not None:
        if feasibility_ref.get("sha256") != file_sha256(feasibility_path) or not verify_payload_hash(feasibility):
            errors.append("V7 feasibility hash or payload mismatch")
        if (
            feasibility.get("protocol", {}).get("bytes_sha256") != protocol_hash
            or feasibility.get("resolved_state_digest") != structure["resolved_state_digest"]
            or feasibility.get("adapter_contract_digest") != selected.get("adapter_contract_digest")
        ):
            errors.append("V7 feasibility/protocol/adapter binding mismatch")
        if any(
            int(feasibility.get("commands", {}).get(name, {}).get("exit_code", 1)) != 0
            for name in ("validate", "dry_run", "adapter_contract")
        ):
            errors.append("V7 feasibility records a failed pure command")

    raw, raw_errors = _v7_raw_index(selected, selected_path, states)
    errors.extend(raw_errors)
    contact_candidates = {}
    contact_gate_details = {}
    for state in states.values():
        if state.stage != "contact":
            continue
        record = raw.get(state.state_id)
        if record is None:
            continue
        contact_candidates[state.candidate_id] = record
        evidence = record.get("evidence", {})
        checks_for_contact, derived, numeric_errors = _recompute_contact_gates(evidence, state)
        contact_gate_details[state.candidate_id] = {
            "checks": checks_for_contact,
            "derived": derived,
            "errors": numeric_errors,
        }
        compiled_ok, compiled_errors = _compiled_contact_matches(
            evidence.get("physics", {}).get("compiled_contact_profile", {}), state
        )
        if not compiled_ok:
            errors.extend(f"{state.candidate_id}: {error}" for error in compiled_errors)
        try:
            independently_compiled = _compile_contact_audit(state)
            independent_ok, independent_errors = _compiled_contact_matches(independently_compiled, state)
            if not independent_ok:
                errors.extend(
                    f"{state.candidate_id}: independent compiled audit failed: {error}" for error in independent_errors
                )
            if not _same_json(independently_compiled, evidence.get("physics", {}).get("compiled_contact_profile", {})):
                errors.append(f"{state.candidate_id}: compiled contact audit changed after raw capture")
            force = _compile_force_envelope(state)
            declared_force = evidence.get("physics", {}).get("finger_force_envelope", {})
            if force.get("finite_force_envelope_N") != declared_force.get("finite_force_envelope_N") or force.get(
                "finite_force_envelope_N"
            ) != float(state.contact.hard_gates["finger_force_max_N"]):
                errors.append(f"{state.candidate_id}: force-envelope derivation mismatch")
        except Exception as exc:
            errors.append(f"{state.candidate_id}: independent compiled audit failed: {exc}")
    eligible_contacts = [
        candidate_id
        for candidate_id, details in contact_gate_details.items()
        if all(details["checks"].values()) and not details["errors"]
    ]
    selected_contact = (
        sorted(
            eligible_contacts,
            key=lambda candidate_id: _v7_contact_score(
                contact_candidates[candidate_id]["evidence"],
                next(
                    state
                    for state in states.values()
                    if state.stage == "contact" and state.candidate_id == candidate_id
                ),
            ),
        )[0]
        if eligible_contacts
        else None
    )
    checks["contact_coverage"] = set(contact_candidates) == CONTACT_IDS
    checks["contact_eligibility"] = selected_contact is not None
    if not checks["contact_coverage"]:
        errors.append("V7 contact candidate coverage is incomplete")
    if not checks["contact_eligibility"]:
        errors.append("V7 contact hard-gate eligibility failed")

    parity_rows = [record for record in raw.values() if record.get("stage") == "parity"]
    parity_state = next((state for state in states.values() if state.stage == "parity"), None)
    parity_ok = len(parity_rows) == 1 and parity_state is not None and _parity_eligible(parity_rows[0], parity_state)
    checks["parity"] = parity_ok
    if not parity_ok:
        errors.append("V7 bounded Gamma=0 parity evidence is missing or failed")
    parity_ref = selected.get("parity_artifact", {})
    parity_path = selected_path.parent / str(parity_ref.get("path", PARITY_FILENAME))
    parity_artifact, parity_errors = _read_json(parity_path, "V7 parity artifact")
    errors.extend(parity_errors)
    if parity_artifact is not None:
        if (
            parity_ref.get("sha256") != file_sha256(parity_path)
            or not verify_payload_hash(parity_artifact)
            or parity_artifact.get("protocol_sha256_bytes") != protocol_hash
            or parity_artifact.get("status") != ("PASS" if parity_ok else "BLOCKED")
        ):
            errors.append("V7 parity artifact hash/status mismatch")

    replay_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in raw.values():
        if record.get("stage") == "replay":
            replay_groups[str(record.get("evidence", {}).get("selected_component"))].append(record)
    replay_ok = REPLAY_GROUPS.issubset(replay_groups) and all(len(replay_groups[group]) == 3 for group in REPLAY_GROUPS)
    if replay_ok:
        for group in REPLAY_GROUPS:
            rows = replay_groups[group]
            indices = {int(row.get("evidence", {}).get("process_index", 0)) for row in rows}
            digests = {row.get("evidence", {}).get("trace_digest") for row in rows}
            metrics = {row.get("evidence", {}).get("metric_digest") for row in rows}
            replay_ok = (
                replay_ok
                and indices == {1, 2, 3}
                and len(digests) == 1
                and len(metrics) == 1
                and all(
                    row.get("evidence", {}).get("complete_trace") is True
                    and row.get("evidence", {}).get("same_process_reset_used") is False
                    for row in rows
                )
            )
    checks["replay"] = replay_ok
    if not replay_ok:
        errors.append("V7 four-group replay determinism is incomplete")
    determinism_ref = selected.get("determinism_artifact", {})
    determinism_path = selected_path.parent / str(determinism_ref.get("path", DETERMINISM_FILENAME))
    determinism_artifact, determinism_errors = _read_json(determinism_path, "V7 determinism artifact")
    errors.extend(determinism_errors)
    if determinism_artifact is not None:
        if (
            determinism_ref.get("sha256") != file_sha256(determinism_path)
            or not verify_payload_hash(determinism_artifact)
            or determinism_artifact.get("protocol_sha256_bytes") != protocol_hash
            or determinism_artifact.get("status") != ("PASS" if replay_ok else "BLOCKED")
        ):
            errors.append("V7 determinism artifact hash/status mismatch")

    excluded_path = selected_path.parent / EXCLUDED_FILENAME
    excluded, excluded_errors = _read_json(excluded_path, "V7 excluded artifact")
    errors.extend(excluded_errors)
    if excluded is not None and (
        not verify_payload_hash(excluded)
        or excluded.get("protocol_sha256_bytes") != protocol_hash
        or excluded.get("excluded") != selected.get("excluded")
    ):
        errors.append("V7 selected/excluded candidate tables disagree")
    declared_ids = selection.get("selected_candidate_ids", {}) if isinstance(selection, Mapping) else {}
    declared_contact = declared_ids.get("contact")
    final_ok = selected_contact is not None and parity_ok and replay_ok and declared_contact == selected_contact
    if final_ok:
        if declared_ids.get("driver") != "dt_nominal" or declared_ids.get("isolator") != "low_frequency_damped":
            errors.append("V7 selected inherited component IDs are incorrect")
        if (
            selected.get("official_profile_alignment") is not True
            or selected.get("official_profile") != OFFICIAL_PROFILE_FILENAME
        ):
            errors.append("V7 PASS selection lacks official-profile alignment")
        selected_state = next(
            state for state in states.values() if state.stage == "contact" and state.candidate_id == selected_contact
        )
        errors.extend(
            _verify_v7_profile(
                selected_path.parent / OFFICIAL_PROFILE_FILENAME,
                selected=selected,
                protocol_hash=protocol_hash,
                stage_rows=selected.get("raw_files", []),
                parity_path=parity_path,
                determinism_path=determinism_path,
                state=selected_state,
            )
        )
    else:
        if (
            selected.get("status") != "BLOCKED"
            or declared_contact is not None
            or selected.get("official_profile") not in (None, "")
            or selected.get("official_profile_alignment") is not False
        ):
            errors.append("blocked V7 artifact must have selected_contact=null and no official profile payload")
    if selected.get("status") == "PASS" and not final_ok:
        errors.append("V7 selected artifact claims PASS without all independent gates")
    if selected.get("status") != "PASS" and final_ok:
        errors.append("V7 selected artifact is BLOCKED despite complete independent PASS gates")
    try:
        expected_contract = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
        if selected.get("adapter_contract_digest") != expected_contract["adapter_contract_digest"]:
            errors.append("V7 adapter-contract digest mismatch")
    except Exception as exc:
        errors.append(f"V7 adapter-contract recomputation failed: {exc}")
    status_path = selected_path.parent / STATUS_FILENAME
    status, status_errors = _read_json(status_path, "V7 status artifact")
    errors.extend(status_errors if require_published_status or status_path.exists() else [])
    if status is not None:
        if (
            not verify_payload_hash(status)
            or status.get("adapter_contract_digest") != selected.get("adapter_contract_digest")
            or status.get("protocol", {}).get("bytes_sha256") != protocol_hash
        ):
            errors.append("V7 status binding/hash mismatch")
        expected_status = "PASS" if selected.get("status") == "PASS" else "BLOCKED"
        if status.get("status") != expected_status:
            errors.append("V7 status does not match selected artifact")
    return {
        "passed": selected.get("status") == "PASS" and not errors,
        "errors": errors,
        "checks": checks,
        "recomputed_selection": {
            "driver": "dt_nominal",
            "isolator": "low_frequency_damped",
            "contact": selected_contact,
        },
        "contact_gate_details": contact_gate_details,
    }


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
                raise V7ProtocolError("--worker requires --protocol and --state-id")
            return _v7_replay_worker(args.protocol, args.state_id)
        protocol, name, protocol_hash = load_v7_protocol(args.protocol)
        if args.validate_protocol:
            result = {
                "passed": True,
                "path": name,
                "protocol_sha256_bytes": protocol_hash,
                "structure": validate_v7_protocol(protocol, protocol_bytes_hash=protocol_hash),
            }
        elif args.dry_run_manifest:
            result = dry_run_manifest(protocol, protocol_bytes_hash=protocol_hash)
        elif args.adapter_contract:
            result = adapter_contract(protocol, protocol_bytes_hash=protocol_hash)
        elif args.write_feasibility:
            result = write_feasibility_artifact(name)
        elif args.verify:
            result = verify_v7_selection_artifact(args.verify, protocol_path=name)
        elif args.stage:
            inherited = _verify_v6_inherited_evidence()
            context = {
                "driver_candidate_id": inherited["selected_driver"],
                "isolator_candidate_id": inherited["selected_isolator"],
                "contact_candidate_id": str(
                    _mapping(protocol.get("selection"), "selection").get("default", {}).get("contact_candidate_id")
                ),
            }
            output = Path(models.assets_root)
            if args.stage == "contact":
                result = {
                    "records": run_v7_stage(
                        protocol,
                        stage="contact",
                        output_dir=output,
                        probe=_v7_contact_probe,
                        protocol_bytes_hash=protocol_hash,
                        selection_context=context,
                    )
                }
            elif args.stage == "parity":
                result = {
                    "records": run_v7_stage(
                        protocol,
                        stage="parity",
                        output_dir=output,
                        probe=_v6_parity_probe,
                        protocol_bytes_hash=protocol_hash,
                        selection_context=context,
                    )
                }
            elif args.stage == "replay":
                result = {
                    "records": run_v7_stage(
                        protocol,
                        stage="replay",
                        output_dir=output,
                        probe=lambda state: _subprocess_replay_probe(name, state),
                        protocol_bytes_hash=protocol_hash,
                        selection_context=context,
                    )
                }
            else:
                result = run_v7_selection(protocol_path=name, output_dir=output)
        else:
            result = run_v7_selection(protocol_path=name, output_dir=Path(models.assets_root))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
