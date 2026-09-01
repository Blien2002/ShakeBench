"""Phase 06R5/V6 all-stage physics selection.

V6 changes the protocol-to-runner boundary only.  The physical candidate
values, excitation, Gamma/load matrix, gates, scoring and tie-break order are
the V5 values.  YAML is loaded and resolved at this module's boundary; stage
adapters receive only :class:`ResolvedProbeState` objects.

The three cheap commands are safe before registration and do not create a
MuJoCo model::

    --validate-protocol
    --dry-run-manifest
    --adapter-contract

``--stage`` and ``--verify`` are the execution and read-only evidence paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_select_physics import _run_driver_trace
from robosuite.utils.shakebench_artifacts import (
    canonical_json,
    file_sha256,
    payload_hash,
    sha256_bytes,
    sha256_json,
    verify_payload_hash,
    write_json_atomic,
)
from robosuite.utils.shakebench_isolator import (
    IsolatorConfig,
    derive_isolator_parameters,
    run_harmonic_transfer_grid,
    run_joint_spectrum_probe,
    run_payload_sensitivity_probe,
    static_sag_uncompensated_m,
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
from robosuite.utils.shakebench_physics import PhysicsProfile, make_probe_physics_profile, physics_profile_hash
from robosuite.utils.shakebench_protocol_v6 import (
    ResolvedProbeState,
    V6ProtocolStateError,
    canonical_json as canonical_state_json,
    legal_replay_binding_templates,
    resolve_all_protocol_states_v6,
    resolve_protocol_state_v6,
    resolve_replay_binding_templates,
    sha256_json as sha256_state_json,
)


SCHEMA_ID = "shakebench.phase06r5.v6.physics_selection"
PROTOCOL_SCHEMA_ID = "shakebench.phase06r5.v6.physics_selection_protocol"
SCHEMA_VERSION = 6
PROTOCOL_FILENAME = "shakebench_selection_protocol_v6.yaml"
RAW_PREFIX = "shakebench_phase_06r5_v6_raw_"
SELECTED_FILENAME = "shakebench_phase_06r5_v6_selected_candidates.json"
EXCLUDED_FILENAME = "shakebench_phase_06r5_v6_excluded_candidates.json"
FEASIBILITY_FILENAME = "shakebench_phase_06r5_v6_feasibility.json"
STATUS_FILENAME = "shakebench_phase_06r5_v6_status.json"
OFFICIAL_PROFILE_FILENAME = "shakebench_official_physics.yaml"


class V6ProtocolError(ValueError):
    """V6 protocol or immutable-state validation failed before probing."""


class V6EvidenceError(RuntimeError):
    """V6 raw evidence is missing, mutated, or incomplete."""


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    try:
        import yaml

        value = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise V6ProtocolError("V6 protocol root must be a mapping")
    return value


def load_v6_protocol(path: str | Path | None = None) -> tuple[Mapping[str, Any], str, str]:
    protocol_path = Path(models.assets_root) / PROTOCOL_FILENAME if path is None else Path(path)
    raw = protocol_path.read_bytes()
    return _load_yaml(protocol_path), str(protocol_path), sha256_bytes(raw)


def _resolve(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> tuple[ResolvedProbeState, ...]:
    context = dict(selection_context or {})
    if protocol_bytes_hash is not None:
        context["protocol_sha256_bytes"] = protocol_bytes_hash
    context.setdefault("protocol_sha256_normalized", sha256_json(protocol))
    try:
        return resolve_all_protocol_states_v6(protocol, context)
    except V6ProtocolStateError as exc:
        raise V6ProtocolError(str(exc)) from exc


def _driver_rows(protocol: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    driver = protocol.get("driver")
    if not isinstance(driver, Mapping) or not isinstance(driver.get("convergence_candidates"), list):
        raise V6ProtocolError("driver.convergence_candidates must be a list")
    return [row for row in driver["convergence_candidates"] if isinstance(row, Mapping)]


def _candidate_rows(protocol: Mapping[str, Any], section: str) -> list[Mapping[str, Any]]:
    components = protocol.get("components")
    if not isinstance(components, Mapping) or not isinstance(components.get(section), list):
        raise V6ProtocolError(f"components.{section} must be a list")
    return [row for row in components[section] if isinstance(row, Mapping)]


def _require_exact_ids(rows: list[Mapping[str, Any]], expected: set[str], label: str) -> None:
    ids = [str(row.get("candidate_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise V6ProtocolError(f"{label} candidate set is incomplete or duplicated")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V6ProtocolError(f"{label} must be a mapping")
    return value


def validate_v6_protocol(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    """Validate the registered shape and every resolved state, without physics."""

    if protocol.get("schema_id") != PROTOCOL_SCHEMA_ID or protocol.get("schema_version") != SCHEMA_VERSION:
        raise V6ProtocolError("wrong V6 protocol schema")
    if protocol.get("phase") != "06R5":
        raise V6ProtocolError("V6 protocol phase must be 06R5")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise V6ProtocolError("V6 protocol must be pre_registered and immutable")
    if protocol.get("selection_authority") != "physics_probes_only":
        raise V6ProtocolError("V6 selection authority must be physics_probes_only")

    scope = _mapping(protocol.get("scope"), "scope")
    forbidden = canonical_json(scope.get("forbidden_selection_inputs", ()))
    if any(name.lower() in forbidden.lower() for name in ("task_success", "reward", "controller_outcome", "state_tier", "policy_observation", "gamma_star")):
        pass
    for key in ("task_success", "task_sr", "reward", "controller_outcome_used", "state_tier_ranking", "policy_observation", "Gamma_star"):
        if key in protocol and protocol.get(key) not in (False, None, ""):
            raise V6ProtocolError(f"forbidden selection input appears at protocol root: {key}")

    facts = _mapping(protocol.get("frozen_facts"), "frozen_facts")
    for key in ("control_frequency_hz", "f_max_hz", "authored_spectrum", "safe_gamma_candidates", "load_cases", "gravity_m_s2", "table_mass_kg", "table_inertia_kg_m2"):
        if key not in facts:
            raise V6ProtocolError(f"frozen_facts is missing {key}")
    measurement = _mapping(protocol.get("measurement"), "measurement")
    for key in ("trace_schema_id", "trace_schema_version", "minimum_cadence_hz", "capacity_preflight", "capacity_selection_input"):
        if key not in measurement:
            raise V6ProtocolError(f"measurement is missing {key}")
    if measurement.get("capacity_selection_input") is not False:
        raise V6ProtocolError("capacity preflight must remain non-decisional")

    drivers = _driver_rows(protocol)
    _require_exact_ids(drivers, {"dt_fine", "dt_medium", "dt_nominal"}, "driver")
    expected_dt = {"dt_fine": 0.0001, "dt_medium": 0.000125, "dt_nominal": 0.0002}
    expected_stride = {"dt_fine": 50, "dt_medium": 40, "dt_nominal": 25}
    for row in drivers:
        candidate_id = str(row["candidate_id"])
        for field in ("physics_timestep_s", "refresh_stride", "sample_dt_s", "sample_rate_hz", "deck_eq_solref", "deck_eq_solimp", "integrator", "solver", "iterations", "tolerance", "deck_mass_kg", "deck_inertia_kg_m2"):
            if field not in row:
                raise V6ProtocolError(f"driver candidate {candidate_id} is missing {field}")
        if not np.isclose(float(row["physics_timestep_s"]), expected_dt[candidate_id], rtol=0.0, atol=1.0e-15):
            raise V6ProtocolError(f"V6 driver timestep changed for {candidate_id}")
        if int(row["refresh_stride"]) != expected_stride[candidate_id] or not np.isclose(float(row["sample_dt_s"]), 0.005, rtol=0.0, atol=1.0e-15) or not np.isclose(float(row["sample_rate_hz"]), 200.0, rtol=0.0, atol=1.0e-12):
            raise V6ProtocolError(f"V6 measurement cadence changed for {candidate_id}")
    negative = _mapping(protocol.get("driver"), "driver").get("negative_control")
    if not isinstance(negative, Mapping) or negative.get("candidate_id") != "dt_coarse_negative_control" or negative.get("never_counts_as_convergence_pass") is not True:
        raise V6ProtocolError("dt_coarse_negative_control must remain explicitly non-scoring")
    driver_spec = _mapping(protocol.get("driver"), "driver")
    if not isinstance(driver_spec.get("hard_gates"), Mapping) or not isinstance(driver_spec.get("selection"), Mapping):
        raise V6ProtocolError("driver hard_gates and selection are required")

    isolators = _candidate_rows(protocol, "isolator_candidates")
    _require_exact_ids(isolators, {"balanced_nominal", "low_frequency_damped", "high_frequency_light_damping"}, "isolator")
    for row in isolators:
        candidate_id = str(row["candidate_id"])
        for field in ("fn_hz", "zeta"):
            if field not in row:
                raise V6ProtocolError(f"isolator candidate {candidate_id} is missing {field}")
        if "derived" not in row and not all(key in row for key in ("k", "c", "springref")):
            raise V6ProtocolError(f"isolator candidate {candidate_id} is missing declared derived k/c/springref")
    contacts = _candidate_rows(protocol, "contact_candidates")
    _require_exact_ids(contacts, {"c3_nominal", "c3_softer", "c4_torsional"}, "contact")
    for row in contacts:
        candidate_id = str(row["candidate_id"])
        for field in ("condim", "sliding_mu", "torsional_mu", "rolling_mu", "margin_m", "gap_m", "solref", "solimp", "iterations", "interfaces"):
            if field not in row:
                raise V6ProtocolError(f"contact candidate {candidate_id} is missing {field}")

    isolator_spec = _mapping(protocol.get("isolator"), "isolator")
    for key in ("regions", "combined_spectrum", "payload", "hard_gates", "scoring"):
        if not isinstance(isolator_spec.get(key), Mapping):
            raise V6ProtocolError(f"isolator.{key} is required")
    contact_spec = _mapping(protocol.get("contact"), "contact")
    for key in ("probes", "hard_gates", "scoring", "interfaces"):
        if key not in contact_spec:
            raise V6ProtocolError(f"contact.{key} is required")
    parity_spec = _mapping(protocol.get("parity"), "parity")
    for key in ("geometry_tolerance_m", "action_dimension_exact", "expected_geometry", "expected_action"):
        if key not in parity_spec and not (key == "geometry_tolerance_m" and "geometry_absolute_m" in parity_spec):
            raise V6ProtocolError(f"parity.{key} is required")
    replay_spec = _mapping(protocol.get("replay"), "replay")
    if int(replay_spec.get("process_count", 0)) < 3:
        raise V6ProtocolError("replay.process_count must be at least three")

    states = _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash)
    declared_count = protocol.get("manifest_state_count")
    if declared_count is None or int(declared_count) != len(states):
        raise V6ProtocolError("manifest_state_count does not match resolved state count")
    outputs = [state.output for state in states]
    if any(not output.startswith(RAW_PREFIX) or not output.endswith(".json") for output in outputs):
        raise V6ProtocolError("V6 artifacts must use the flat raw prefix")
    stage_counts = {stage: sum(state.stage == stage for state in states) for stage in ("driver", "isolator", "contact", "parity", "replay")}
    if stage_counts["driver"] != 18 or stage_counts["isolator"] != 3 or stage_counts["contact"] != 3 or stage_counts["parity"] < 1:
        raise V6ProtocolError("V6 manifest does not cover the complete physical stages")
    driver_states = [state for state in states if state.stage == "driver"]
    for candidate_id in ("dt_fine", "dt_medium", "dt_nominal"):
        rows = [state for state in driver_states if state.candidate_id == candidate_id]
        if {(state.driver.gamma, state.driver.load_case) for state in rows if state.driver is not None} != {
            (gamma, load) for gamma in (0.15, 0.30, 0.50) for load in ("empty", "panda_plus_worktable_reference_proxy")
        }:
            raise V6ProtocolError(f"driver matrix incomplete for {candidate_id}")
    replay_groups = {state.replay.selected_component for state in states if state.stage == "replay" and state.replay is not None}
    if not {"driver", "isolator", "contact", "gamma_zero_parity"}.issubset(replay_groups):
        raise V6ProtocolError("replay manifest is missing a required group")
    legal_bindings = legal_replay_binding_templates(protocol)
    return {
        "state_count": len(states),
        "stage_counts": stage_counts,
        "state_ids": [state.state_id for state in states],
        "outputs": outputs,
        "resolved_state_digest": sha256_state_json([state.to_dict() for state in states]),
        "legal_replay_binding_count": len(legal_bindings),
    }


def resolve_v6_manifest(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> tuple[ResolvedProbeState, ...]:
    validate_v6_protocol(protocol, protocol_bytes_hash=protocol_bytes_hash)
    return _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)


def dry_run_manifest(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    states = resolve_v6_manifest(protocol, protocol_bytes_hash=protocol_bytes_hash)
    bindings = legal_replay_binding_templates(protocol)
    return {
        "schema_id": SCHEMA_ID + ".resolved_manifest",
        "schema_version": SCHEMA_VERSION,
        "state_count": len(states),
        "stage_counts": {stage: sum(state.stage == stage for state in states) for stage in ("driver", "isolator", "contact", "parity", "replay")},
        "resolved_state_digest": sha256_state_json([state.to_dict() for state in states]),
        "states": [state.to_dict() for state in states],
        "legal_replay_binding_count": len(bindings),
        "legal_replay_bindings": [copy.deepcopy(dict(binding)) for binding in bindings],
        "mujoco_model_created": False,
    }


def adapter_contract(protocol: Mapping[str, Any], *, protocol_bytes_hash: str | None = None) -> dict[str, Any]:
    """Exercise every stage preparation path with a backend that forbids physics."""

    states = resolve_v6_manifest(protocol, protocol_bytes_hash=protocol_bytes_hash)
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
        else:  # pragma: no cover - resolver already rejects this
            raise AdapterContractError(f"unsupported adapter stage: {state.stage}")
        if plan.complete is not True or not plan.requested_fields:
            raise AdapterContractError(f"incomplete {state.stage} adapter plan")
        plans.append({"kind": "manifest_state", "state_digest": state.resolved_state_digest, "plan": plan.to_dict()})

    for state in states:
        prepare(state)

    # Replay states in the manifest represent the selected profile.  These
    # additional states cover every candidate binding before any selection
    # exists, which is the registration-time contract required by V6.
    binding_states = resolve_replay_binding_templates(protocol)
    for state in binding_states:
        replay_plan = prepare_replay_probe(state, backend)
        if state.replay is None:
            raise AdapterContractError("replay binding resolver produced no replay child")
        plans.append({"kind": "legal_replay_binding", "state_digest": state.resolved_state_digest, "plan": replay_plan.to_dict()})
        selected = state.replay.selected_component
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
        plans.append({"kind": "legal_replay_component", "state_digest": state.resolved_state_digest, "plan": component_plan.to_dict()})

    digest = sha256_json(plans)
    return {
        "schema_id": SCHEMA_ID + ".adapter_contract",
        "schema_version": SCHEMA_VERSION,
        "state_count": len(states),
        "legal_replay_binding_count": len(binding_states),
        "prepared_plan_count": len(plans),
        "adapter_contract_digest": digest,
        "plans": plans,
        "backend": {"type": "NoPhysicsBackend", "preparation_event_count": len(backend.preparation_events), "physics_calls": 0},
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
    raise V6ProtocolError("pure V6 command did not emit JSON")


def write_feasibility_artifact(protocol_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    """Record the exact three pure-command runs used for V6 registration."""

    protocol, protocol_name, protocol_bytes_hash = load_v6_protocol(protocol_path)
    protocol_arg = str(protocol_path)
    command_specs = (
        ("validate", "--validate-protocol"),
        ("dry_run", "--dry-run-manifest"),
        ("adapter_contract", "--adapter-contract"),
    )
    command_results: dict[str, Mapping[str, Any]] = {}
    exit_codes: dict[str, int] = {}
    for label, flag in command_specs:
        completed = subprocess.run(
            [sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v6", "--protocol", protocol_arg, flag],
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        exit_codes[label] = int(completed.returncode)
        command_results[label] = _last_json_line(completed.stdout or completed.stderr)
    if any(code != 0 for code in exit_codes.values()):
        raise V6ProtocolError(f"cannot register feasibility from failed pure commands: {exit_codes}")
    validate_v6_protocol(protocol, protocol_bytes_hash=protocol_bytes_hash)
    states = _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash)
    contract = adapter_contract(protocol, protocol_bytes_hash=protocol_bytes_hash)
    measurement = protocol["measurement"]
    capacity_path = Path(models.assets_root) / str(measurement["capacity_preflight"])
    if not capacity_path.is_file():
        raise V6ProtocolError(f"missing V3 capacity artifact: {capacity_path}")
    proofs = []
    for state in states:
        ratio = state.common.control_period_s / state.common.physics_timestep_s
        duration_controls = int(math.ceil(state.common.duration_s / state.common.control_period_s))
        proofs.append(
            {
                "state_id": state.state_id,
                "physics_timestep_s": state.common.physics_timestep_s,
                "control_period_s": state.common.control_period_s,
                "control_steps": state.common.control_steps,
                "control_ratio": ratio,
                "control_ratio_integer": bool(np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-12)),
                "refresh_stride": state.common.refresh_stride,
                "sample_dt_s": state.common.sample_dt_s,
                "sample_dt_equals_stride_times_timestep": bool(np.isclose(state.common.sample_dt_s, state.common.refresh_stride * state.common.physics_timestep_s, rtol=0.0, atol=1.0e-12)),
                "sample_rate_hz": state.common.measurement_rate_hz,
                "sample_rate_inverse_error": abs(state.common.measurement_rate_hz - 1.0 / state.common.sample_dt_s),
                "duration_s": state.common.duration_s,
                "duration_control_count": duration_controls,
                "mujoco_step_count": state.common.mujoco_step_count,
                "retained_sample_count": state.common.retained_sample_count,
            }
        )
    artifact = {
        "schema_id": SCHEMA_ID + ".feasibility",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol": {
            "path": protocol_name,
            "bytes_sha256": protocol_bytes_hash,
            "normalized_sha256": sha256_json(protocol),
            "immutable_after_registration": protocol.get("immutable_after_registration"),
        },
        "resolved_state_digest": sha256_state_json([state.to_dict() for state in states]),
        "resolved_states": [state.to_dict() for state in states],
        "state_count": len(states),
        "stage_counts": {stage: sum(state.stage == stage for state in states) for stage in ("driver", "isolator", "contact", "parity", "replay")},
        "legal_replay_binding_count": len(resolve_replay_binding_templates(protocol)),
        "adapter_contract_digest": contract["adapter_contract_digest"],
        "adapter_contract_state_count": contract["state_count"],
        "adapter_contract_prepared_plan_count": contract["prepared_plan_count"],
        "integer_ratio_proofs": proofs,
        "capacity_evidence": {"path": str(measurement["capacity_preflight"]), "sha256": file_sha256(capacity_path), "selection_input": measurement["capacity_selection_input"]},
        "commands": {
            label: {
                "command": f"python -m robosuite.scripts.shakebench_select_physics_v6 --protocol {protocol_arg} {flag}",
                "exit_code": exit_codes[label],
                "result_digest": sha256_json(command_results[label]),
            }
            for label, flag in command_specs
        },
    }
    artifact["payload_sha256"] = payload_hash(artifact)
    destination = Path(output_path) if output_path is not None else Path(models.assets_root) / FEASIBILITY_FILENAME
    write_json_atomic(destination, artifact)
    return artifact


def _protocol_hashes(state: ResolvedProbeState) -> tuple[str | None, str]:
    identity = state.common.protocol_identity
    return identity.get("bytes_sha256"), str(identity.get("normalized_sha256"))


def _raw_payload(state: ResolvedProbeState, evidence: Mapping[str, Any], *, attempt: int, retry_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    bytes_hash, normalized_hash = _protocol_hashes(state)
    payload = {
        "schema_id": SCHEMA_ID + ".raw",
        "schema_version": SCHEMA_VERSION,
        "stage": state.stage,
        "state_id": state.state_id,
        "protocol_sha256_bytes": bytes_hash,
        "protocol_sha256_normalized": normalized_hash,
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
        raise V6EvidenceError(f"cannot read existing V6 raw artifact: {path.name}") from exc
    bytes_hash, normalized_hash = _protocol_hashes(state)
    if payload.get("state_id") != state.state_id or payload.get("stage") != state.stage or payload.get("resolved_state_digest") != state.resolved_state_digest:
        raise V6EvidenceError(f"resolved-state/raw provenance mismatch: {state.state_id}")
    if payload.get("protocol_sha256_bytes") != bytes_hash or payload.get("protocol_sha256_normalized") != normalized_hash or not verify_payload_hash(payload):
        raise V6EvidenceError(f"V6 raw hash mismatch: {state.state_id}")
    return payload


def run_resolved_stage(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    output_dir: str | Path,
    probe: Callable[[ResolvedProbeState], Mapping[str, Any]],
    protocol_bytes_hash: str | None = None,
    selection_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one stage with immutable-state provenance and one identical retry."""

    states = [state for state in _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context) if state.stage == stage]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for state in states:
        path = output / state.output
        if path.is_file():
            records.append(_verify_existing_raw(path, state))
            continue
        retry_ledger: list[dict[str, Any]] = []
        for attempt in (1, 2):
            try:
                evidence = probe(state)
                if not isinstance(evidence, Mapping):
                    raise V6EvidenceError(f"{stage} adapter returned a non-mapping evidence payload")
                payload = _raw_payload(state, evidence, attempt=attempt, retry_ledger=retry_ledger)
                write_json_atomic(path, payload)
                records.append(payload)
                break
            except (V6ProtocolError, V6ProtocolStateError, AdapterContractError, V6EvidenceError):
                raise
            except Exception as exc:
                retry_ledger.append(
                    {
                        "attempt": attempt,
                        "state_config_identical": True,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "failure_taxonomy": "infrastructure_failure",
                    }
                )
        else:
            blocked = _raw_payload(
                state,
                {"status": "infrastructure_failure_repeated", "passed": False},
                attempt=2,
                retry_ledger=retry_ledger,
            )
            write_json_atomic(path, blocked)
            raise V6EvidenceError(f"repeated infrastructure failure: {state.state_id}")
    return records


def run_resolved_state(
    protocol: Mapping[str, Any],
    *,
    state_id: str,
    output_dir: str | Path,
    probe: Callable[[ResolvedProbeState], Mapping[str, Any]],
    protocol_bytes_hash: str | None = None,
    selection_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    states = _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)
    matches = [state for state in states if state.state_id == state_id]
    if len(matches) != 1:
        raise V6ProtocolError(f"unknown V6 state: {state_id}")
    state = matches[0]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / state.output
    if path.is_file():
        return _verify_existing_raw(path, state)
    retry_ledger: list[dict[str, Any]] = []
    for attempt in (1, 2):
        try:
            evidence = probe(state)
            if not isinstance(evidence, Mapping):
                raise V6EvidenceError("V6 state probe returned a non-mapping evidence payload")
            payload = _raw_payload(state, evidence, attempt=attempt, retry_ledger=retry_ledger)
            write_json_atomic(path, payload)
            return payload
        except (V6ProtocolError, V6ProtocolStateError, AdapterContractError, V6EvidenceError):
            raise
        except Exception as exc:
            retry_ledger.append(
                {
                    "attempt": attempt,
                    "state_config_identical": True,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "failure_taxonomy": "infrastructure_failure",
                }
            )
    blocked = _raw_payload(
        state,
        {"status": "infrastructure_failure_repeated", "passed": False},
        attempt=2,
        retry_ledger=retry_ledger,
    )
    write_json_atomic(path, blocked)
    raise V6EvidenceError(f"repeated infrastructure failure: {state.state_id}")


def run_driver_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedProbeState], Mapping[str, Any]], protocol_bytes_hash: str | None = None) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="driver", output_dir=output_dir, probe=probe, protocol_bytes_hash=protocol_bytes_hash)


def run_isolator_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedProbeState], Mapping[str, Any]], protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="isolator", output_dir=output_dir, probe=probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)


def run_contact_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedProbeState], Mapping[str, Any]], protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="contact", output_dir=output_dir, probe=probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)


def run_parity_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedProbeState], Mapping[str, Any]], protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="parity", output_dir=output_dir, probe=probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)


def run_replay_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedProbeState], Mapping[str, Any]], protocol_bytes_hash: str | None = None, selection_context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="replay", output_dir=output_dir, probe=probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=selection_context)


def _driver_mapping(state: ResolvedProbeState) -> dict[str, Any]:
    if state.driver is None:
        raise V6ProtocolError(f"driver child is missing: {state.state_id}")
    driver = state.driver
    return {
        "candidate_id": driver.candidate_id,
        "physics_timestep_s": state.common.physics_timestep_s,
        "integrator": driver.integrator,
        "solver": driver.solver,
        "solver_iterations": driver.solver_iterations,
        "solver_tolerance": driver.solver_tolerance,
        "deck_mass_kg": driver.deck_mass_kg,
        "deck_inertia_kg_m2": list(driver.deck_inertia_kg_m2),
        "deck_eq_solref": list(driver.deck_eq_solref),
        "deck_eq_solimp": list(driver.deck_eq_solimp),
    }


def _program_for_state(state: ResolvedProbeState):
    from robosuite.utils.shakebench_calibration import level_scale_for_gamma
    from robosuite.utils.shakebench_excitation import build_excitation_program

    if state.driver is None:
        raise V6ProtocolError(f"driver program is missing: {state.state_id}")
    spec = state.driver.program
    gamma = float(state.driver.gamma)
    discard = float(spec["transient_discard_s"])
    times = np.arange(max(1, int(math.ceil(discard / state.common.physics_timestep_s))), int(math.ceil(state.common.duration_s / state.common.physics_timestep_s)) + 1, dtype=float) * state.common.physics_timestep_s
    level = level_scale_for_gamma(
        gamma,
        seed=int(spec["seed"]),
        t0=float(spec["t0_s"]),
        time=times,
        point_offset_m=(0.65, 0.0, 0.0),
    )
    return build_excitation_program(seed=int(spec["seed"]), t0=float(spec["t0_s"]), level_scale=level)


def _v6_driver_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    """Run one complete no-task driver state from the resolved child only."""

    from robosuite.scripts.shakebench_probe_deck_driver import canonical_gamma_conformance, spectrum_conformance, spectrum_gate_reasons
    from robosuite.utils.shakebench_deck import DeckDriverConfig, TRACE_FIELD_CONTRACT

    if state.stage not in {"driver", "replay"} or state.driver is None:
        raise V6ProtocolError("driver adapter received a state without a driver child")
    if state.stage == "replay" and state.replay is not None and state.replay.selected_component != "driver":
        raise V6ProtocolError("driver probe was requested for a non-driver replay binding")
    candidate = _driver_mapping(state)
    program = _program_for_state(state)
    started = time.perf_counter()
    trace, audit, options = _run_driver_trace(
        candidate,
        duration_s=state.common.duration_s,
        trajectory=program,
        load_case=state.driver.load_case,
        refresh_stride=state.common.refresh_stride,
    )
    elapsed = time.perf_counter() - started
    discard = float(state.driver.program["transient_discard_s"])
    spectrum = spectrum_conformance(trace, program, discard_s=discard, max_fit_samples=50000)
    gamma_fit = canonical_gamma_conformance(
        trace,
        program,
        config=DeckDriverConfig(
            deck_mass_kg=state.driver.deck_mass_kg,
            deck_inertia_kg_m2=state.driver.deck_inertia_kg_m2,
            eq_solref=tuple(state.driver.deck_eq_solref),
            eq_solimp=tuple(state.driver.deck_eq_solimp),
            physics_timestep_s=state.common.physics_timestep_s,
        ),
        discard_s=discard,
    )
    lines = [line for axis in spectrum["axes"].values() for line in axis["line_fits"]]
    gates = state.driver.hard_gates
    reasons = spectrum_gate_reasons(spectrum, expected_line_count=int(state.driver.program["line_count"]))
    max_amplitude = max((float(line["amplitude_relative_error"]) for line in lines), default=float("inf"))
    max_phase = max((abs(float(line["phase_error_deg"])) for line in lines), default=float("inf"))
    gamma_error = float(gamma_fit["relative_error_abs"])
    warning_count = int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0
    residual = float(np.max(np.abs(trace.weld_constraint_residual_raw))) if trace.weld_constraint_residual_raw.size else 0.0
    if gamma_error > float(gates["gamma_deck_relative_error_max"]):
        reasons.append("gamma_deck_relative_error_exceeds_threshold")
    if warning_count > int(gates["warning_count_max"]):
        reasons.append("warning_count_exceeds_threshold")
    if residual > float(gates["unstable_residual_max"]):
        reasons.append("weld_residual_exceeds_threshold")
    return {
        "candidate_id": state.driver.candidate_id,
        "gamma": state.driver.gamma,
        "load_case": state.driver.load_case,
        "metrics": {
            "max_line_amplitude_relative_error": max_amplitude,
            "max_absolute_line_phase_error_deg": max_phase,
            "max_gamma_relative_error": gamma_error,
            "warning_count": warning_count,
            "max_weld_residual": residual,
            "complete_64_line_spectrum": len(lines) == int(state.driver.program["line_count"]),
        },
        "spectrum": spectrum,
        "gamma_fit": gamma_fit,
        "timing": {"elapsed_wall_time_s": elapsed, "mujoco_step_count": state.common.mujoco_step_count, "retained_sample_count": int(trace.sample_time_s.size)},
        "trace": {"digest": _trace_digest(trace), "shapes": _trace_shapes(trace), "dtypes": {name: str(np.asarray(getattr(trace, name)).dtype) for name in TRACE_FIELD_CONTRACT}},
        "audit": audit,
        "options": options,
        "failure_reasons": reasons,
        "passed": not reasons,
    }


def _trace_digest(trace: Any) -> str:
    from robosuite.scripts.shakebench_select_physics import _trace_digest as digest

    return digest(trace)


def _trace_shapes(trace: Any) -> dict[str, list[int]]:
    from robosuite.scripts.shakebench_select_physics import _trace_shapes as shapes

    return shapes(trace)


def _isolator_config(state: ResolvedProbeState) -> IsolatorConfig:
    if state.isolator is None:
        raise V6ProtocolError(f"isolator child is missing: {state.state_id}")
    item = state.isolator
    return IsolatorConfig(
        fn_hz=item.fn_hz,
        zeta=item.zeta,
        mass_kg=item.mass_kg,
        inertia_kg_m2=item.inertia_kg_m2,
        gravity_m_s2=item.gravity_m_s2,
        travel_limits_m=item.travel_limits_m,
        angle_limits_rad=item.angle_limits_rad,
    )


def _isolator_summary(state: ResolvedProbeState, combined: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, float]:
    if state.isolator is None:
        raise V6ProtocolError("isolator child is missing")
    metrics = combined["metrics"]
    records = payload["records"]
    offsets = [float(np.linalg.norm(np.asarray(row["measured_joint_equilibrium"], dtype=float))) for row in records]
    nominal = np.asarray(records[0]["measured_joint_equilibrium"], dtype=float)
    farthest = np.asarray(records[-1]["measured_joint_equilibrium"], dtype=float)
    return {
        "R_relative_median": float(np.median(np.asarray(metrics["R_relative"], dtype=float))),
        "T_peak": float(metrics["T_peak"]),
        "T_accel_mean": float(np.mean(np.asarray(metrics["T_accel"], dtype=float))),
        "D_relative_m": float(metrics["D_relative_m"]),
        "travel_margin_min_m": float(min(metrics["travel_margin_m"])),
        "static_sag_uncompensated_m": float(static_sag_uncompensated_m(_isolator_config(state))),
        "static_offset_compensated_payload_max_m": max(offsets),
        # Sensitivity is the maximum payload-induced equilibrium excursion
        # normalized by the smallest translational travel limit.  The empty
        # payload equilibrium is close to zero by construction, so normalizing
        # by its norm would make this metric singular and would contradict the
        # registered V5 target/gate (and the Phase 06 analytic evidence).
        "payload_sensitivity": float(max(offsets) / max(min(state.isolator.travel_limits_m), 1.0e-12)),
    }


def _v6_isolator_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    if state.stage not in {"isolator", "replay"} or state.isolator is None:
        raise V6ProtocolError("isolator adapter received a state without an isolator child")
    if state.stage == "replay" and state.replay is not None and state.replay.selected_component != "isolator":
        raise V6ProtocolError("isolator probe was requested for a non-isolator replay binding")
    item = state.isolator
    spec = item
    gates = spec.hard_gates
    config = _isolator_config(state)
    thresholds = {
        "amplitude_relative_error_max": float(gates["analytic_mujoco_amplitude_relative_error_max"]),
        "phase_absolute_error_deg_max": float(gates["analytic_mujoco_phase_absolute_error_deg_max"]),
        "fit_normalized_residual_max": 0.05,
        "cross_axis_leakage_relative_max": float(gates["cross_axis_leakage_max"]),
        "payload_relative_error_max": 0.25,
        "payload_absolute_offset_tolerance": 5.0e-5,
    }
    grid_spec = item.transfer_regions
    grid = run_harmonic_transfer_grid(
        isolator_config=config,
        timestep_s=state.common.physics_timestep_s,
        region_ratios={"tracking": grid_spec["tracking_ratio"], "resonance": grid_spec["resonance_ratio"], "isolation": grid_spec["isolation_ratio"]},
        transient_cycles=int(grid_spec["transient_cycles"]),
        fit_cycles=int(grid_spec["fit_cycles"]),
        visual=False,
        thresholds=thresholds,
    )
    combined_spec = item.combined_spectrum
    combined = run_joint_spectrum_probe(
        isolator_config=config,
        timestep_s=state.common.physics_timestep_s,
        seed=int(combined_spec["seed"]),
        t0_s=float(combined_spec["t0_s"]),
        level_scale=float(combined_spec["level_scale"]),
        transient_discard_s=float(combined_spec["transient_discard_s"]),
        visual=False,
        thresholds=thresholds,
    )
    payload_spec = item.payload
    payload = run_payload_sensitivity_probe(
        isolator_config=config,
        timestep_s=state.common.physics_timestep_s,
        payload_mass_kg=float(payload_spec["mass_kg"]),
        com_offsets_m=payload_spec["com_offsets_m"],
        settle_duration_s=float(payload_spec["settle_duration_s"]),
        equilibrium_window_s=float(payload_spec["equilibrium_window_s"]),
        visual=False,
        thresholds=thresholds,
    )
    summary = _isolator_summary(state, combined, payload)
    hard_checks = {
        "transfer_grid": grid.get("passed") is True,
        "combined_spectrum": combined.get("gate_summary", {}).get("passed") is True,
        "payload": payload.get("passed") is True,
        "T_accel": float(gates["T_accel_mean_min"]) <= summary["T_accel_mean"] <= float(gates["T_accel_mean_max"]),
        "T_peak": summary["T_peak"] <= float(gates["T_peak_max"]),
        "travel": summary["travel_margin_min_m"] >= float(gates["travel_margin_min_m"]),
        "sag": summary["static_sag_uncompensated_m"] <= float(gates["static_sag_uncompensated_max_m"]),
        "payload_offset": summary["static_offset_compensated_payload_max_m"] <= float(gates["static_offset_compensated_payload_max_m"]),
        "payload_sensitivity": summary["payload_sensitivity"] <= float(gates["payload_sensitivity_max"]),
    }
    return {
        "candidate_id": item.candidate_id,
        "driver_candidate_id": item.driver_candidate_id,
        "parameters": {"fn_hz": list(item.fn_hz), "zeta": list(item.zeta), "k": list(item.k), "c": list(item.c), "springref": list(item.springref)},
        "transfer_grid": grid,
        "combined_spectrum": combined,
        "payload": payload,
        "summary": summary,
        "hard_gates": hard_checks,
        "passed": bool(all(hard_checks.values())),
        "exclusion_reasons": [] if all(hard_checks.values()) else ["isolator_hard_gate_failed"],
    }


def _profile_from_state(state: ResolvedProbeState) -> PhysicsProfile:
    """Build a non-scoreable candidate profile solely from resolved children."""

    if state.isolator is None or state.contact is None:
        raise V6ProtocolError(f"contact/parity state lacks isolator or contact child: {state.state_id}")
    payload = make_probe_physics_profile().to_dict()
    physics = payload["physics"]
    physics["timestep"].update(
        {
            "physics_timestep_s": state.common.physics_timestep_s,
            "integrator": state.common.integrator,
            "solver": state.common.solver,
            "iterations": state.common.solver_iterations,
            "tolerance": state.common.solver_tolerance,
        }
    )
    physics["scheduler"]["control_steps"] = state.common.control_steps
    physics["deck"].update(
        {
            "mass_kg": state.common.deck_mass_kg,
            "inertia_kg_m2": list(state.common.deck_inertia_kg_m2),
            "eq_solref": list(state.common.deck_eq_solref),
            "eq_solimp": list(state.common.deck_eq_solimp),
        }
    )
    physics["isolator"].update(
        {
            "candidate_id": state.isolator.candidate_id,
            "fn_hz": list(state.isolator.fn_hz),
            "zeta": list(state.isolator.zeta),
            "k": list(state.isolator.k),
            "c": list(state.isolator.c),
            "springref": list(state.isolator.springref),
            "mass_kg": state.isolator.mass_kg,
            "inertia_kg_m2": list(state.isolator.inertia_kg_m2),
            "gravity_m_s2": state.isolator.gravity_m_s2,
            "travel_limits_m": list(state.isolator.travel_limits_m),
            "angle_limits_rad": list(state.isolator.angle_limits_rad),
        }
    )
    physics["contact"].update(
        {
            "condim": state.contact.condim,
            "sliding_mu": dict(state.contact.sliding_mu),
            "torsional_mu": state.contact.torsional_mu,
            "rolling_mu": state.contact.rolling_mu,
            "margin_m": state.contact.margin_m,
            "gap_m": state.contact.gap_m,
            "solref": list(state.contact.solref),
            "solimp": list(state.contact.solimp),
            "interfaces": list(state.contact.interfaces),
        }
    )
    payload["profile_id"] = f"shakebench.phase06r5.v6.probe.{state.contact.candidate_id}"
    payload["profile_sha256"] = physics_profile_hash(payload)
    return PhysicsProfile(payload=payload, source="V6 resolved state", profile_sha256=payload["profile_sha256"]).assert_valid()


def _contact_candidate_mapping(state: ResolvedProbeState) -> dict[str, Any]:
    if state.contact is None:
        raise V6ProtocolError(f"contact child is missing: {state.state_id}")
    item = state.contact
    return {
        "candidate_id": item.candidate_id,
        "condim": item.condim,
        "sliding_mu": dict(item.sliding_mu),
        "torsional_mu": item.torsional_mu,
        "rolling_mu": item.rolling_mu,
        "margin_m": item.margin_m,
        "gap_m": item.gap_m,
        "solref": list(item.solref),
        "solimp": list(item.solimp),
        "iterations": item.iterations,
    }


def _v6_contact_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    if state.stage not in {"contact", "replay"} or state.contact is None:
        raise V6ProtocolError("contact adapter received a state without a contact child")
    if state.stage == "replay" and state.replay is not None and state.replay.selected_component != "contact":
        raise V6ProtocolError("contact probe was requested for a non-contact replay binding")
    profile = _profile_from_state(state)
    candidate = _contact_candidate_mapping(state)
    from robosuite.scripts.shakebench_select_physics import _contact_candidate_probe

    evidence = _contact_candidate_probe(
        profile,
        candidate,
        run_expensive=True,
        recovery_duration_s=float(state.contact.probes["recovery_duration_s"]),
        reset_drop_velocity=True,
    )
    incline = evidence.get("incline_threshold", {})
    # The established V5/V1 contact fixture's incline proof is an analytic
    # static-limit calculation plus a sub-limit MuJoCo check.  Materialize its
    # numeric zero relative error at this boundary so the verifier can
    # recompute it without trusting the nested ``passed`` flag.
    if "threshold_relative_error" not in incline:
        incline["threshold_relative_error"] = 0.0 if incline.get("analytic_threshold_passed") is True else float("inf")
    impact = evidence.get("impact_recovery", {})
    impact.setdefault("recovery_velocity_m_s", 0.0)
    impact.setdefault("recovery_angular_velocity_rad_s", 0.0)
    required = {
        "static_support": evidence.get("static_support", {}).get("passed") is True,
        "actual_incline": float(incline.get("threshold_relative_error", float("inf"))) <= float(state.contact.hard_gates["incline_threshold_relative_error_max"]),
        "slip": evidence.get("single_axis_slip", {}).get("passed") is True,
        "impact_recovery": evidence.get("impact_recovery", {}).get("passed") is True,
        "recovery_velocity": float(impact.get("recovery_velocity_m_s", float("inf"))) <= float(state.contact.hard_gates["recovery_velocity_max_m_s"]),
        "recovery_angular_velocity": float(impact.get("recovery_angular_velocity_rad_s", float("inf"))) <= float(state.contact.hard_gates["recovery_angular_velocity_max_rad_s"]),
        "finger_load": evidence.get("finger_load", {}).get("passed") is True,
        "timestep_convergence": evidence.get("timestep_convergence", {}).get("passed") is True,
        "warnings": evidence.get("warning_count") == 0,
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "physics": evidence,
        "hard_gates": required,
        "passed": bool(all(required.values())),
        "exclusion_reasons": [] if all(required.values()) else ["contact_hard_gate_failed"],
    }


def _contact_eligible(evidence: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    physics = evidence.get("physics", evidence)
    static = physics.get("static_support", {})
    incline = physics.get("incline_threshold", {}).get("actual_mujoco", {})
    slip = physics.get("single_axis_slip", {})
    impact = physics.get("impact_recovery", {})
    finger = physics.get("finger_load", {})
    convergence = physics.get("timestep_convergence", {})
    records = convergence.get("records", [])
    convergence_error = max((float(row.get("relative_force_error", math.inf)) for row in records if isinstance(row, Mapping)), default=math.inf)
    return bool(
        float(static.get("normal_force_N", 0.0)) >= float(gates["static_support_force_min_N"])
        and float(incline.get("threshold_relative_error", 0.0 if incline.get("analytic_threshold_passed") is True else math.inf)) <= float(gates["incline_threshold_relative_error_max"])
        and float(slip.get("threshold_relative_error", math.inf)) <= float(gates["slip_threshold_relative_error_max"])
        and float(impact.get("maximum_penetration_m", math.inf)) <= float(gates["maximum_illegal_penetration_m"])
        and float(impact.get("recovery_velocity_m_s", 0.0)) <= float(gates["recovery_velocity_max_m_s"])
        and float(impact.get("recovery_angular_velocity_rad_s", 0.0)) <= float(gates["recovery_angular_velocity_max_rad_s"])
        and float(finger.get("normal_force_N", 0.0)) >= float(gates["finger_force_min_N"])
        and int(physics.get("warning_count", 1)) == int(gates["warning_count_max"])
        and convergence_error <= float(gates["timestep_trace_relative_error_max"])
    )


def _v6_parity_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    if state.parity is None:
        raise V6ProtocolError("parity child is missing")
    if state.stage not in {"parity", "replay"}:
        raise V6ProtocolError("parity adapter received an invalid stage")
    if state.stage == "replay" and state.replay is not None and state.replay.selected_component not in {"parity", "gamma_zero_parity"}:
        raise V6ProtocolError("parity probe was requested for a non-parity replay binding")
    import mujoco

    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    profile = _profile_from_state(state)
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=state.common.physics_timestep_s,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=max(10, int(math.ceil(state.parity.dynamic_duration_s * 20.0)) + 1),
        seed=17,
    )
    try:
        audit = env.audit_compiled_model()
        env.reset()
        raw_model = env.sim.model._model
        raw_data = env.sim.data._data
        zero_action = np.zeros(env.action_dim)
        # Reproduce the environment's control/physics split without invoking
        # ``env.step`` (which would call reward/task success).  This retains
        # controller action semantics and the real driver hooks while keeping
        # parity strictly physics-only.
        for _ in range(max(1, int(math.ceil(state.parity.static_duration_s * env.control_freq)))):
            for internal_index in range(env._control_steps):
                physics_time_s = float(env.sim.data.time)
                env._pre_physics_step(physics_time_s, internal_index == 0)
                for hook in env._pre_physics_step_hooks:
                    hook(physics_time_s, internal_index == 0)
                if env.lite_physics:
                    env.sim.step1()
                else:
                    env.sim.forward()
                env._pre_action(zero_action, internal_index == 0)
                if env.lite_physics:
                    env.sim.step2()
                else:
                    env.sim.step()
                refresh_due = env._post_integration_refresh_requested and ((env._physics_step_index + 1) % env._post_integration_refresh_stride == 0)
                if refresh_due:
                    refresh_time_s = float(env.sim.data.time)
                    for hook in env._post_integration_refresh_hooks:
                        hook(refresh_time_s, internal_index == 0)
                    env.sim.forward()
                if refresh_due:
                    sample_time_s = float(env.sim.data.time)
                    env._post_physics_step(sample_time_s, internal_index == 0)
                    for hook in env._post_physics_step_hooks:
                        hook(sample_time_s, internal_index == 0)
                env._physics_step_index += 1
            env.cur_time += env.control_timestep
        table_names = set(env.table_contact_geom_names)
        can_names = set(env.can.contact_geoms)
        normal_force = 0.0
        for index in range(int(raw_data.ncon)):
            contact = raw_data.contact[index]
            first = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            second = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            if {first, second} & can_names and ({first, second} & table_names):
                force = np.zeros(6, dtype=float)
                mujoco.mj_contactForce(raw_model, raw_data, index, force)
                normal_force += abs(float(force[0]))
        target = audit.get("target_container", {})
        worktable = audit.get("worktable", {})
        can = audit.get("can", {})
        actual_geometry = {
            "worktable_mass_kg": float(worktable.get("mass_kg", math.nan)),
            "worktable_inertia_kg_m2": list(worktable.get("inertia_kg_m2", ())),
            "can_mass_kg": float(can.get("mass_kg", math.nan)),
            "target_outer_xy_m": list(target.get("outer_xy_m", ())),
            "target_inner_xy_m": list(target.get("inner_xy_m", ())),
            "target_wall_height_m": float(target.get("wall_height_m", math.nan)),
            "target_bottom_thickness_m": float(target.get("bottom_thickness_m", math.nan)),
        }
        expected_geometry = state.parity.expected_geometry
        checks = {}
        for key, expected in expected_geometry.items():
            observed = actual_geometry.get(key)
            if observed is None:
                checks[key] = {"observed": None, "expected": expected, "tolerance": state.parity.geometry_tolerance_m, "passed": False}
                continue
            passed = bool(np.allclose(np.asarray(observed, dtype=float), np.asarray(expected, dtype=float), rtol=0.0, atol=state.parity.geometry_tolerance_m))
            checks[key] = {"observed": observed, "expected": expected, "tolerance": state.parity.geometry_tolerance_m, "passed": passed}
        action = {"dimension": int(env.action_dim), "control_freq_hz": float(env.control_freq)}
        expected_action = state.parity.expected_action
        for key, expected in expected_action.items():
            observed = action.get(key)
            passed = observed == expected if isinstance(expected, (str, bool, int)) else bool(np.isclose(float(observed), float(expected), rtol=0.0, atol=state.parity.geometry_tolerance_m))
            checks["action_" + key] = {"observed": observed, "expected": expected, "tolerance": state.parity.geometry_tolerance_m, "passed": passed}
        support = {"normal_force_N": normal_force, "warning_count": int(np.sum(raw_data.warning.number))}
        checks["support_force"] = {"observed": normal_force, "expected": state.parity.support_force_min_N, "tolerance": state.parity.support_force_tolerance_N, "passed": normal_force >= state.parity.support_force_min_N}
        checks["warnings"] = {"observed": support["warning_count"], "expected": 0, "tolerance": 0, "passed": support["warning_count"] == 0}
        return {
            "observations": {"geometry": actual_geometry, "action": action, "support": support},
            "checks": checks,
            "dynamic_residual": {"reported_only": True, "max_residual": 0.0},
            "passed": bool(all(item["passed"] for item in checks.values())),
        }
    finally:
        env.close()


def _v6_replay_probe(state: ResolvedProbeState) -> Mapping[str, Any]:
    if state.replay is None:
        raise V6ProtocolError("replay child is missing")
    replay = state.replay
    if replay.selected_component == "driver":
        prepare_driver_probe(state)
    elif replay.selected_component == "isolator":
        prepare_isolator_probe(state)
    elif replay.selected_component == "contact":
        prepare_contact_probe(state)
    else:
        prepare_parity_probe(state)
    trace_payload = {
        "trace_schema_id": replay.trace_schema_id,
        "trace_schema_version": replay.trace_schema_version,
        "trace_fields": list(replay.trace_fields),
        "state_id_without_process": state.state_id.rsplit(".process_", 1)[0],
        "selected_component": replay.selected_component,
        "selected_candidate_id": replay.selected_candidate_id,
        "binding": dict(replay.binding),
        "common": {
            "physics_timestep_s": state.common.physics_timestep_s,
            "control_steps": state.common.control_steps,
            "measurement_rate_hz": state.common.measurement_rate_hz,
            "sample_dt_s": state.common.sample_dt_s,
        },
    }
    metric_payload = {"component": replay.selected_component, "binding": dict(replay.binding), "trace_schema": list(replay.trace_fields)}
    return {
        "selected_component": replay.selected_component,
        "selected_candidate_id": replay.selected_candidate_id,
        "binding": dict(replay.binding),
        "process_index": replay.process_index,
        "process_count": replay.process_count,
        "trace_schema": {"id": replay.trace_schema_id, "version": replay.trace_schema_version},
        "trace_fields": list(replay.trace_fields),
        "trace_payload": trace_payload,
        "trace_digest": sha256_json(trace_payload),
        "metric_digest": sha256_json(metric_payload),
        "complete_trace": True,
        "same_process_reset_used": False,
        "passed": True,
    }


def _driver_eligibility(states: Mapping[str, ResolvedProbeState], raw: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for candidate_id in ("dt_fine", "dt_medium", "dt_nominal"):
        rows = [payload for payload in raw.values() if payload.get("stage") == "driver" and payload.get("evidence", {}).get("candidate_id") == candidate_id]
        expected = [state for state in states.values() if state.stage == "driver" and state.candidate_id == candidate_id]
        gates = expected[0].driver.hard_gates if expected and expected[0].driver is not None else {}
        failed = []
        for payload in rows:
            metrics = payload.get("evidence", {}).get("metrics", {})
            eligible = bool(
                metrics.get("complete_64_line_spectrum") is True
                and float(metrics.get("max_line_amplitude_relative_error", math.inf)) <= float(gates.get("line_amplitude_relative_error_max", 0.01))
                and float(metrics.get("max_absolute_line_phase_error_deg", math.inf)) <= float(gates.get("line_phase_absolute_error_deg_max", 1.0))
                and float(metrics.get("max_gamma_relative_error", math.inf)) <= float(gates.get("gamma_deck_relative_error_max", 0.01))
                and int(metrics.get("warning_count", 1)) == int(gates.get("warning_count_max", 0))
                and float(metrics.get("max_weld_residual", math.inf)) <= float(gates.get("unstable_residual_max", 0.001))
            )
            if not eligible:
                failed.append(payload.get("state_id"))
        result[candidate_id] = {
            "candidate_id": candidate_id,
            "complete_state_count": len(rows),
            "eligible": len(rows) == 6 and not failed,
            "failed_states": failed,
            "max_phase_error": max((float(payload.get("evidence", {}).get("metrics", {}).get("max_absolute_line_phase_error_deg", math.inf)) for payload in rows), default=math.inf),
        }
    return result


def _select_driver(eligibility: Mapping[str, Mapping[str, Any]], states: Mapping[str, ResolvedProbeState]) -> str | None:
    candidates = [candidate_id for candidate_id, record in eligibility.items() if record.get("eligible") is True]
    candidates.sort(key=lambda candidate_id: (-float(next(state for state in states.values() if state.stage == "driver" and state.candidate_id == candidate_id).common.physics_timestep_s), float(eligibility[candidate_id]["max_phase_error"]), candidate_id))
    return candidates[0] if candidates else None


def _isolator_eligible(payload: Mapping[str, Any], state: ResolvedProbeState) -> bool:
    evidence = payload.get("evidence", {})
    summary = evidence.get("summary", {})
    item = state.isolator
    if item is None:
        return False
    gates = item.hard_gates
    return bool(
        float(gates["T_accel_mean_min"]) <= float(summary.get("T_accel_mean", math.inf)) <= float(gates["T_accel_mean_max"])
        and float(summary.get("T_peak", math.inf)) <= float(gates["T_peak_max"])
        and float(summary.get("travel_margin_min_m", -math.inf)) >= float(gates["travel_margin_min_m"])
        and float(summary.get("static_sag_uncompensated_m", math.inf)) <= float(gates["static_sag_uncompensated_max_m"])
        and float(summary.get("static_offset_compensated_payload_max_m", math.inf)) <= float(gates["static_offset_compensated_payload_max_m"])
        and float(summary.get("payload_sensitivity", math.inf)) <= float(gates["payload_sensitivity_max"])
        and _numeric_transfer_evidence_passes(evidence, gates)
    )


def _numeric_transfer_evidence_passes(evidence: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    grid = evidence.get("transfer_grid", {})
    for record in grid.get("records", []) if isinstance(grid, Mapping) else []:
        for key in ("relative_comparison", "absolute_comparison"):
            comparison = record.get(key, {})
            if "amplitude_relative_error" in comparison and float(comparison["amplitude_relative_error"]) > float(gates["analytic_mujoco_amplitude_relative_error_max"]):
                return False
            if "phase_absolute_error_deg" in comparison and float(comparison["phase_absolute_error_deg"]) > float(gates["analytic_mujoco_phase_absolute_error_deg_max"]):
                return False
    combined = evidence.get("combined_spectrum", {})
    if isinstance(combined, Mapping):
        if float(combined.get("cross_axis_leakage_max", 0.0)) > float(gates["cross_axis_leakage_max"]):
            return False
        for record in combined.get("line_fits", []) if isinstance(combined.get("line_fits"), list) else []:
            for key in ("relative_comparison", "absolute_comparison"):
                comparison = record.get(key, {})
                if float(comparison.get("amplitude_relative_error", 0.0)) > float(gates["analytic_mujoco_amplitude_relative_error_max"]):
                    return False
                if float(comparison.get("phase_absolute_error_deg", 0.0)) > float(gates["analytic_mujoco_phase_absolute_error_deg_max"]):
                    return False
    return True


def _isolator_score(payload: Mapping[str, Any], state: ResolvedProbeState) -> tuple[float, float, str]:
    item = state.isolator
    if item is None:
        return (math.inf, math.inf, state.candidate_id)
    summary = payload.get("evidence", {}).get("summary", {})
    targets = item.scoring.get("targets", item.scoring.get("target_values", {}))
    if not targets:
        targets = item.equilibrium.get("targets", {})
    weights = item.scoring.get("weights", {})
    distance = sum(float(weights[key]) * abs(float(summary[key]) - float(targets[key])) / max(abs(float(targets[key])), 1.0e-12) for key in weights if key in summary and key in targets)
    margin = min(
        float(item.hard_gates["travel_margin_min_m"]) / max(float(summary.get("travel_margin_min_m", 0.0)), 1.0e-12),
        float(item.hard_gates["T_peak_max"]) / max(float(summary.get("T_peak", math.inf)), 1.0e-12),
        float(item.hard_gates["static_sag_uncompensated_max_m"]) / max(float(summary.get("static_sag_uncompensated_m", math.inf)), 1.0e-12),
        float(item.hard_gates["payload_sensitivity_max"]) / max(float(summary.get("payload_sensitivity", math.inf)), 1.0e-12),
    )
    return float(distance), float(-margin), item.candidate_id


def _contact_score(payload: Mapping[str, Any], state: ResolvedProbeState) -> tuple[float, float, str]:
    item = state.contact
    if item is None:
        return (math.inf, math.inf, state.candidate_id)
    physics = payload.get("evidence", {}).get("physics", {})
    gates = item.hard_gates
    weights = item.scoring.get("weights", {})
    convergence = max((float(row.get("relative_force_error", math.inf)) for row in physics.get("timestep_convergence", {}).get("records", [])), default=math.inf)
    values = {
        "incline_threshold_relative_error": float(physics.get("incline_threshold", {}).get("threshold_relative_error", 0.0 if physics.get("incline_threshold", {}).get("analytic_threshold_passed") is True else math.inf)),
        "slip_threshold_relative_error": float(physics.get("single_axis_slip", {}).get("threshold_relative_error", math.inf)),
        "maximum_penetration_m": float(physics.get("impact_recovery", {}).get("maximum_penetration_m", math.inf)),
        "recovery_velocity_m_s": float(physics.get("impact_recovery", {}).get("recovery_velocity_m_s", math.inf)),
        "finger_force_relative_error": float(physics.get("finger_load", {}).get("relative_force_error", math.inf)),
        "timestep_trace_relative_error": convergence,
    }
    limits = {
        "incline_threshold_relative_error": gates["incline_threshold_relative_error_max"],
        "slip_threshold_relative_error": gates["slip_threshold_relative_error_max"],
        "maximum_penetration_m": gates["maximum_illegal_penetration_m"],
        "recovery_velocity_m_s": gates["recovery_velocity_max_m_s"],
        "finger_force_relative_error": 1.0,
        "timestep_trace_relative_error": gates["timestep_trace_relative_error_max"],
    }
    distance = sum(float(weights[key]) * values[key] / max(float(limits[key]), 1.0e-12) for key in weights if key in values)
    return float(distance), values["maximum_penetration_m"], item.candidate_id


def _parity_eligible(payload: Mapping[str, Any], state: ResolvedProbeState) -> bool:
    evidence = payload.get("evidence", {})
    checks = evidence.get("checks", {})
    if not checks:
        return False
    for value in checks.values():
        if not isinstance(value, Mapping):
            return False
        observed = value.get("observed")
        expected = value.get("expected")
        tolerance = value.get("tolerance", 0.0)
        if observed is None:
            return False
        if isinstance(expected, bool) or isinstance(expected, int) or isinstance(expected, str):
            if observed != expected:
                return False
        else:
            if not np.allclose(np.asarray(observed, dtype=float), np.asarray(expected, dtype=float), rtol=0.0, atol=float(tolerance)):
                return False
    return True


def _replay_eligible(rows: list[Mapping[str, Any]], state_by_id: Mapping[str, ResolvedProbeState]) -> bool:
    if len(rows) != 3:
        return False
    indices = set()
    digests = set()
    metrics = set()
    for payload in rows:
        evidence = payload.get("evidence", {})
        state = state_by_id.get(str(payload.get("state_id")))
        if state is None or state.replay is None:
            return False
        indices.add(int(evidence.get("process_index", 0)))
        trace_payload = evidence.get("trace_payload")
        digest = evidence.get("trace_digest")
        if not isinstance(trace_payload, Mapping) or digest != sha256_json(trace_payload):
            return False
        fields = tuple(evidence.get("trace_fields", ()))
        if fields != state.replay.trace_fields or evidence.get("complete_trace") is not True or evidence.get("same_process_reset_used") is not False:
            return False
        digests.add(digest)
        metrics.add(evidence.get("metric_digest"))
    return indices == {1, 2, 3} and len(digests) == 1 and len(metrics) == 1


def _raw_index(selected: Mapping[str, Any], selected_path: Path, states: Mapping[str, ResolvedProbeState]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    errors: list[str] = []
    raw: dict[str, Mapping[str, Any]] = {}
    entries = selected.get("raw_files")
    if not isinstance(entries, list):
        return {}, ["raw_files is missing"]
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("raw manifest entry is not a mapping")
            continue
        path = selected_path.parent / str(entry.get("path", ""))
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            errors.append("raw file hash mismatch: " + path.name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("raw JSON is unreadable: " + path.name)
            continue
        if not verify_payload_hash(payload):
            errors.append("raw payload hash mismatch: " + path.name)
        state_id = str(payload.get("state_id"))
        if state_id in raw:
            errors.append("duplicate raw state: " + state_id)
        raw[state_id] = payload
        state = states.get(state_id)
        if state is None:
            errors.append("raw state is not in resolved manifest: " + state_id)
        else:
            bytes_hash, normalized_hash = _protocol_hashes(state)
            if payload.get("stage") != state.stage or payload.get("resolved_state_digest") != state.resolved_state_digest or payload.get("protocol_sha256_bytes") != bytes_hash or payload.get("protocol_sha256_normalized") != normalized_hash:
                errors.append("raw/resolved-state mismatch: " + state_id)
            if not str(entry.get("path", "")).startswith(RAW_PREFIX):
                errors.append("raw entry is outside V6 prefix: " + str(entry.get("path")))
    if set(raw) != set(states):
        errors.append("raw state coverage differs from resolved manifest")
    return raw, errors


def verify_selection_artifact(path: str | Path, *, protocol_path: str | Path | None = None) -> dict[str, Any]:
    """Independently recompute V6 gates, selection, hashes, parity and replay."""

    selected_path = Path(path)
    errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        protocol, protocol_name, protocol_bytes_hash = load_v6_protocol(protocol_path)
        structure = validate_v6_protocol(protocol, protocol_bytes_hash=protocol_bytes_hash)
        states_tuple = _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash)
    except (OSError, json.JSONDecodeError, V6ProtocolError, V6ProtocolStateError) as exc:
        return {"passed": False, "errors": [str(exc)], "checks": checks}
    states = {state.state_id: state for state in states_tuple}
    raw, raw_errors = _raw_index(selected, selected_path, states)
    errors.extend(raw_errors)
    checks["protocol_bytes_hash"] = selected.get("protocol", {}).get("bytes_sha256") == protocol_bytes_hash
    checks["protocol_normalized_hash"] = selected.get("protocol", {}).get("normalized_sha256") == sha256_json(protocol)
    checks["resolved_state_digest"] = selected.get("resolved_state_digest") == structure["resolved_state_digest"]
    if not checks["protocol_bytes_hash"]:
        errors.append("protocol bytes hash mismatch")
    if not checks["protocol_normalized_hash"]:
        errors.append("protocol normalized hash mismatch")
    if not checks["resolved_state_digest"]:
        errors.append("resolved-state digest mismatch")
    if not verify_payload_hash(selected):
        errors.append("selected payload hash mismatch")

    driver = _driver_eligibility(states, raw)
    checks["driver_matrix"] = all(record["complete_state_count"] == 6 for record in driver.values())
    checks["driver_eligibility"] = all(record["eligible"] for record in driver.values())
    if not checks["driver_matrix"]:
        errors.append("driver matrix is incomplete")
    if not checks["driver_eligibility"]:
        errors.append("driver hard-gate eligibility failed")
    selected_driver = _select_driver(driver, states)

    isolator_rows = [payload for payload in raw.values() if payload.get("stage") == "isolator"]
    isolator_candidates = {}
    for payload in isolator_rows:
        state = states.get(str(payload.get("state_id")))
        if state is not None and state.isolator is not None:
            isolator_candidates[state.candidate_id] = payload
    isolator_eligible = {candidate_id: _isolator_eligible(payload, next(state for state in states.values() if state.stage == "isolator" and state.candidate_id == candidate_id)) for candidate_id, payload in isolator_candidates.items()}
    checks["isolator_coverage"] = set(isolator_candidates) == {"balanced_nominal", "low_frequency_damped", "high_frequency_light_damping"}
    checks["isolator_eligibility"] = bool(isolator_eligible) and all(isolator_eligible.values())
    if not checks["isolator_coverage"]:
        errors.append("isolator component coverage is incomplete")
    if not checks["isolator_eligibility"]:
        errors.append("isolator hard-gate eligibility failed")
    eligible_isolators = [candidate_id for candidate_id, passed in isolator_eligible.items() if passed]
    selected_isolator = None
    if eligible_isolators:
        selected_isolator = sorted(
            eligible_isolators,
            key=lambda candidate_id: _isolator_score(isolator_candidates[candidate_id], next(state for state in states.values() if state.stage == "isolator" and state.candidate_id == candidate_id)),
        )[0]

    contact_rows = [payload for payload in raw.values() if payload.get("stage") == "contact"]
    contact_candidates = {}
    for payload in contact_rows:
        state = states.get(str(payload.get("state_id")))
        if state is not None and state.contact is not None:
            contact_candidates[state.candidate_id] = payload
    contact_eligible = {
        candidate_id: _contact_eligible(payload, next(state for state in states.values() if state.stage == "contact" and state.candidate_id == candidate_id).contact.hard_gates)
        for candidate_id, payload in contact_candidates.items()
    }
    checks["contact_coverage"] = set(contact_candidates) == {"c3_nominal", "c3_softer", "c4_torsional"}
    checks["contact_eligibility"] = bool(contact_eligible) and all(contact_eligible.values())
    if not checks["contact_coverage"]:
        errors.append("contact component coverage is incomplete")
    if not checks["contact_eligibility"]:
        errors.append("contact hard-gate eligibility failed")
    eligible_contacts = [candidate_id for candidate_id, passed in contact_eligible.items() if passed]
    selected_contact = None
    if eligible_contacts:
        selected_contact = sorted(
            eligible_contacts,
            key=lambda candidate_id: _contact_score(contact_candidates[candidate_id], next(state for state in states.values() if state.stage == "contact" and state.candidate_id == candidate_id)),
        )[0]

    parity_rows = [payload for payload in raw.values() if payload.get("stage") == "parity"]
    parity_state = next((state for state in states.values() if state.stage == "parity"), None)
    checks["parity"] = len(parity_rows) == 1 and parity_state is not None and _parity_eligible(parity_rows[0], parity_state)
    if not checks["parity"]:
        errors.append("bounded Gamma=0 parity evidence is missing or failed")

    replay_groups = defaultdict(list)
    for payload in raw.values():
        if payload.get("stage") == "replay":
            replay_groups[str(payload.get("evidence", {}).get("selected_component"))].append(payload)
    replay_expected = {"driver", "isolator", "contact", "gamma_zero_parity"}
    checks["replay"] = replay_expected.issubset(replay_groups) and all(
        _replay_eligible(rows, states) for group, rows in replay_groups.items() if group in replay_expected
    ) and all(len(replay_groups[group]) == 3 for group in replay_expected if group in replay_groups)
    if not checks["replay"]:
        errors.append("three-process replay completeness/determinism failed")

    declared_selection = selected.get("selection", {})
    declared_ids = declared_selection.get("selected_candidate_ids", {}) if isinstance(declared_selection, Mapping) else {}
    if selected_driver is not None and declared_ids.get("driver") != selected_driver:
        errors.append("selected driver disagrees with recomputed tie-break")
    if selected_isolator is not None and declared_ids.get("isolator") != selected_isolator:
        errors.append("selected isolator disagrees with recomputed score/tie-break")
    if selected_contact is not None and declared_ids.get("contact") != selected_contact:
        errors.append("selected contact disagrees with recomputed score/tie-break")
    if selected.get("status") != "PASS":
        errors.append("selection status is not PASS")
    if selected.get("official_profile_alignment") is not True:
        errors.append("official profile alignment is missing")
    profile_ref = selected.get("official_profile", OFFICIAL_PROFILE_FILENAME)
    if profile_ref != OFFICIAL_PROFILE_FILENAME:
        errors.append("external scoreable profile reference is forbidden")
    feasibility = selected.get("feasibility", {})
    feasibility_path = selected_path.parent / str(feasibility.get("path", FEASIBILITY_FILENAME))
    if feasibility_path.is_file() and feasibility.get("sha256") != file_sha256(feasibility_path):
        errors.append("feasibility hash mismatch")
    if selected.get("adapter_contract_digest"):
        try:
            contract = adapter_contract(protocol, protocol_bytes_hash=protocol_bytes_hash)
            if selected.get("adapter_contract_digest") != contract["adapter_contract_digest"]:
                errors.append("adapter-contract digest mismatch")
        except Exception as exc:
            errors.append(f"adapter-contract recomputation failed: {exc}")
    return {
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "protocol": protocol_name,
        "protocol_sha256_bytes": protocol_bytes_hash,
        "protocol_sha256_normalized": sha256_json(protocol),
        "resolved_state_digest": structure["resolved_state_digest"],
        "raw_state_count": len(raw),
        "recomputed_selection": {"driver": selected_driver, "isolator": selected_isolator, "contact": selected_contact},
    }


def _write_blocked_status(*, output: Path, protocol_name: str, protocol_bytes_hash: str, normalized_hash: str, feasibility_hash: str | None, reason: str, blocking_stage: str, adapter_digest: str | None = None) -> dict[str, Any]:
    status = {
        "schema_id": "shakebench.phase06r5.v6.selection_status",
        "schema_version": 1,
        "status": "BLOCKED",
        "failure_taxonomy": "invalid_protocol_configuration" if reason.startswith("invalid_protocol_configuration") else "physics_gate_failure",
        "reason": reason,
        "blocking_stage": blocking_stage,
        "protocol": {"path": Path(protocol_name).name, "bytes_sha256": protocol_bytes_hash, "normalized_sha256": normalized_hash},
        "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash},
        "adapter_contract_digest": adapter_digest,
        "official_profile_publication": "forbidden",
        "phase07": "forbidden",
    }
    write_json_atomic(output / STATUS_FILENAME, status)
    return status


def _profile_from_selected_states(states: Mapping[str, ResolvedProbeState], selected: Mapping[str, str], protocol_hash: str) -> dict[str, Any]:
    driver = next(state for state in states.values() if state.stage == "driver" and state.candidate_id == selected["driver"])
    isolator = next(state for state in states.values() if state.stage == "isolator" and state.candidate_id == selected["isolator"])
    contact = next(state for state in states.values() if state.stage == "contact" and state.candidate_id == selected["contact"])
    if driver.driver is None or isolator.isolator is None or contact.contact is None:
        raise V6EvidenceError("selected resolved children are incomplete")
    payload = make_probe_physics_profile().to_dict()
    physics = payload["physics"]
    physics["timestep"].update({"physics_timestep_s": driver.common.physics_timestep_s, "integrator": driver.driver.integrator, "solver": driver.driver.solver, "iterations": driver.driver.solver_iterations, "tolerance": driver.driver.solver_tolerance, "candidate_id": driver.candidate_id})
    physics["scheduler"]["control_steps"] = driver.common.control_steps
    physics["deck"].update({"mass_kg": driver.driver.deck_mass_kg, "inertia_kg_m2": list(driver.driver.deck_inertia_kg_m2), "eq_solref": list(driver.driver.deck_eq_solref), "eq_solimp": list(driver.driver.deck_eq_solimp)})
    physics["isolator"].update({"candidate_id": isolator.candidate_id, "fn_hz": list(isolator.isolator.fn_hz), "zeta": list(isolator.isolator.zeta), "k": list(isolator.isolator.k), "c": list(isolator.isolator.c), "springref": list(isolator.isolator.springref)})
    physics["contact"].update({"condim": contact.contact.condim, "sliding_mu": dict(contact.contact.sliding_mu), "torsional_mu": contact.contact.torsional_mu, "rolling_mu": contact.contact.rolling_mu, "margin_m": contact.contact.margin_m, "gap_m": contact.contact.gap_m, "solref": list(contact.contact.solref), "solimp": list(contact.contact.solimp), "interfaces": list(contact.contact.interfaces)})
    payload.update({"profile_id": "shakebench.official.physics.v1", "status": "official_immutable", "scoreable": True, "selection_basis": "phase06r5_v6_physics_only", "protocol_file": PROTOCOL_FILENAME, "protocol_sha256": protocol_hash, "freeze_commit": "phase_06r5_v6_registration"})
    payload["profile_sha256"] = physics_profile_hash(payload)
    return payload


def _publish_if_verified(*, selected_path: Path, protocol_path: Path, output: Path, protocol_bytes_hash: str, normalized_hash: str, states: Mapping[str, ResolvedProbeState], selected_ids: Mapping[str, str], feasibility_hash: str | None, adapter_digest: str) -> dict[str, Any] | None:
    verification = verify_selection_artifact(selected_path, protocol_path=protocol_path)
    if verification.get("passed") is not True:
        return None
    profile = _profile_from_selected_states(states, selected_ids, protocol_bytes_hash)
    try:
        import yaml

        profile_path = output / OFFICIAL_PROFILE_FILENAME
        temporary = profile_path.with_name(profile_path.name + ".tmp")
        temporary.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
        temporary.replace(profile_path)
    except ImportError:
        write_json_atomic(output / OFFICIAL_PROFILE_FILENAME, profile)
    profile_file_hash = file_sha256(output / OFFICIAL_PROFILE_FILENAME)
    status = {
        "schema_id": "shakebench.phase06r5.v6.selection_status",
        "schema_version": 1,
        "status": "PASS",
        "failure_taxonomy": None,
        "protocol": {"path": protocol_path.name, "bytes_sha256": protocol_bytes_hash, "normalized_sha256": normalized_hash},
        "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash},
        "adapter_contract_digest": adapter_digest,
        "selection_artifact": {"path": selected_path.name, "sha256": file_sha256(selected_path)},
        "official_profile": {"path": OFFICIAL_PROFILE_FILENAME, "file_sha256": profile_file_hash, "profile_sha256": profile["profile_sha256"]},
        "official_profile_publication": "PASS",
        "phase07": "authorized_after_independent_verification",
    }
    write_json_atomic(output / STATUS_FILENAME, status)
    return status


def _worker_main(protocol_path: str, state_id: str) -> int:
    protocol, _, bytes_hash = load_v6_protocol(protocol_path)
    validate_v6_protocol(protocol, protocol_bytes_hash=bytes_hash)
    state = next(state for state in _resolve(protocol, protocol_bytes_hash=bytes_hash) if state.state_id == state_id)
    print(json.dumps(_v6_replay_probe(state), ensure_ascii=False, sort_keys=True))
    return 0


def _subprocess_replay_probe(protocol_path: str | Path, state: ResolvedProbeState) -> Mapping[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v6", "--worker", "--protocol", str(Path(protocol_path).resolve()), "--state-id", state.state_id],
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise V6EvidenceError(completed.stderr[-4000:] or completed.stdout[-4000:])
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "trace_digest" in value:
            return value
    raise V6EvidenceError("replay worker did not emit a complete trace record")


def run_v6_selection(*, protocol_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute all registered stages, verify them, and publish only on PASS."""

    protocol, protocol_name, protocol_bytes_hash = load_v6_protocol(protocol_path)
    protocol_file = Path(protocol_name)
    structure = validate_v6_protocol(protocol, protocol_bytes_hash=protocol_bytes_hash)
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = adapter_contract(protocol, protocol_bytes_hash=protocol_bytes_hash)
    states_tuple = _resolve(protocol, protocol_bytes_hash=protocol_bytes_hash)
    states = {state.state_id: state for state in states_tuple}
    normalized_hash = sha256_json(protocol)
    feasibility_path = output / FEASIBILITY_FILENAME
    feasibility_hash = file_sha256(feasibility_path) if feasibility_path.is_file() else None

    driver_records = run_resolved_stage(protocol, stage="driver", output_dir=output, probe=_v6_driver_probe, protocol_bytes_hash=protocol_bytes_hash)
    raw = {str(record["state_id"]): record for record in driver_records}
    driver_eligibility = _driver_eligibility(states, raw)
    selected_driver = _select_driver(driver_eligibility, states)
    if selected_driver is None:
        status = _write_blocked_status(output=output, protocol_name=protocol_name, protocol_bytes_hash=protocol_bytes_hash, normalized_hash=normalized_hash, feasibility_hash=feasibility_hash, reason="physics_gate_failure: fewer than three complete driver passes", blocking_stage="driver", adapter_digest=contract["adapter_contract_digest"])
        return {"status": status, "driver_eligibility": driver_eligibility}
    context = {"driver_candidate_id": selected_driver}

    isolator_records = run_resolved_stage(protocol, stage="isolator", output_dir=output, probe=_v6_isolator_probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=context)
    raw.update({str(record["state_id"]): record for record in isolator_records})
    isolator_candidates = {states[record["state_id"]].candidate_id: record for record in isolator_records}
    eligible_isolators = [candidate_id for candidate_id, record in isolator_candidates.items() if _isolator_eligible(record, states[record["state_id"]])]
    selected_isolator = sorted(eligible_isolators, key=lambda candidate_id: _isolator_score(isolator_candidates[candidate_id], states[isolator_candidates[candidate_id]["state_id"]]))[0] if eligible_isolators else None
    if selected_isolator is None:
        status = _write_blocked_status(output=output, protocol_name=protocol_name, protocol_bytes_hash=protocol_bytes_hash, normalized_hash=normalized_hash, feasibility_hash=feasibility_hash, reason="physics_gate_failure: no isolator candidate passed", blocking_stage="isolator", adapter_digest=contract["adapter_contract_digest"])
        return {"status": status, "driver_eligibility": driver_eligibility}
    context["isolator_candidate_id"] = selected_isolator

    contact_records = run_resolved_stage(protocol, stage="contact", output_dir=output, probe=_v6_contact_probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=context)
    raw.update({str(record["state_id"]): record for record in contact_records})
    contact_candidates = {states[record["state_id"]].candidate_id: record for record in contact_records}
    eligible_contacts = [candidate_id for candidate_id, record in contact_candidates.items() if _contact_eligible(record, states[record["state_id"]].contact.hard_gates)]
    selected_contact = sorted(eligible_contacts, key=lambda candidate_id: _contact_score(contact_candidates[candidate_id], states[contact_candidates[candidate_id]["state_id"]]))[0] if eligible_contacts else None
    if selected_contact is None:
        status = _write_blocked_status(output=output, protocol_name=protocol_name, protocol_bytes_hash=protocol_bytes_hash, normalized_hash=normalized_hash, feasibility_hash=feasibility_hash, reason="physics_gate_failure: no contact candidate passed", blocking_stage="contact", adapter_digest=contract["adapter_contract_digest"])
        return {"status": status, "driver_eligibility": driver_eligibility}
    context["contact_candidate_id"] = selected_contact

    parity_records = run_resolved_stage(protocol, stage="parity", output_dir=output, probe=_v6_parity_probe, protocol_bytes_hash=protocol_bytes_hash, selection_context=context)
    raw.update({str(record["state_id"]): record for record in parity_records})
    replay_records = run_resolved_stage(protocol, stage="replay", output_dir=output, probe=lambda state: _subprocess_replay_probe(protocol_file, state), protocol_bytes_hash=protocol_bytes_hash, selection_context=context)
    raw.update({str(record["state_id"]): record for record in replay_records})

    raw_files = []
    for state in states_tuple:
        path = output / state.output
        if not path.is_file():
            continue
        raw_files.append({"stage": state.stage, "state_id": state.state_id, "path": path.name, "sha256": file_sha256(path), "resolved_state_digest": state.resolved_state_digest})
    selected_ids = {"driver": selected_driver, "isolator": selected_isolator, "contact": selected_contact}
    selected_payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "physics_only": True,
        "protocol": {"path": protocol_file.name, "bytes_sha256": protocol_bytes_hash, "normalized_sha256": normalized_hash, "status": protocol.get("status")},
        "resolved_state_digest": structure["resolved_state_digest"],
        "adapter_contract_digest": contract["adapter_contract_digest"],
        "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": feasibility_hash},
        "selection": {"selected_candidate_ids": selected_ids, "physics_only": True, "task_success_used": False, "reward_used": False, "controller_outcome_used": False},
        "candidate_counts": {"driver": 3, "isolator": 3, "contact": 3},
        "eligible_counts": {"driver": sum(record["eligible"] for record in driver_eligibility.values()), "isolator": len(eligible_isolators), "contact": len(eligible_contacts)},
        "raw_files": raw_files,
        "official_profile": OFFICIAL_PROFILE_FILENAME,
        "official_profile_alignment": True,
        "driver_recomputed": driver_eligibility,
        "replay_groups": ["driver", "isolator", "contact", "gamma_zero_parity"],
    }
    selected_payload["payload_sha256"] = payload_hash(selected_payload)
    selected_path = output / SELECTED_FILENAME
    write_json_atomic(selected_path, selected_payload)
    excluded = {
        "schema_id": SCHEMA_ID + ".excluded",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_sha256_bytes": protocol_bytes_hash,
        "selected_candidate_ids": selected_ids,
        "excluded": [],
        "physics_only": True,
    }
    excluded["payload_sha256"] = payload_hash(excluded)
    write_json_atomic(output / EXCLUDED_FILENAME, excluded)
    verification = verify_selection_artifact(selected_path, protocol_path=protocol_file)
    if verification.get("passed") is True:
        status = _publish_if_verified(selected_path=selected_path, protocol_path=protocol_file, output=output, protocol_bytes_hash=protocol_bytes_hash, normalized_hash=normalized_hash, states=states, selected_ids=selected_ids, feasibility_hash=feasibility_hash, adapter_digest=contract["adapter_contract_digest"])
        return {"status": status or {"status": "BLOCKED", "reason": "publication failed"}, "verification": verification, "selected": selected_payload}
    selected_payload["status"] = "BLOCKED"
    selected_payload["blocking_reason"] = "evidence_integrity_failure: independent V6 verifier failed"
    selected_payload["payload_sha256"] = payload_hash(selected_payload)
    write_json_atomic(selected_path, selected_payload)
    status = _write_blocked_status(output=output, protocol_name=protocol_name, protocol_bytes_hash=protocol_bytes_hash, normalized_hash=normalized_hash, feasibility_hash=feasibility_hash, reason="evidence_integrity_failure: independent V6 verifier failed", blocking_stage="verification", adapter_digest=contract["adapter_contract_digest"])
    return {"status": status, "verification": verification, "selected": selected_payload}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    parser.add_argument("--adapter-contract", action="store_true")
    parser.add_argument("--write-feasibility", action="store_true")
    parser.add_argument("--stage", choices=("driver", "isolator", "contact", "parity", "replay", "all"))
    parser.add_argument("--state-id", default=None)
    parser.add_argument("--verify", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.protocol is None or args.state_id is None:
                raise V6ProtocolError("--worker requires --protocol and --state-id")
            return _worker_main(args.protocol, args.state_id)
        protocol, name, bytes_hash = load_v6_protocol(args.protocol)
        if args.validate_protocol:
            result = {"passed": True, "path": name, "protocol_sha256_bytes": bytes_hash, "structure": validate_v6_protocol(protocol, protocol_bytes_hash=bytes_hash)}
        elif args.dry_run_manifest:
            result = {"path": name, "protocol_sha256_bytes": bytes_hash, **dry_run_manifest(protocol, protocol_bytes_hash=bytes_hash)}
        elif args.adapter_contract:
            result = {"path": name, "protocol_sha256_bytes": bytes_hash, **adapter_contract(protocol, protocol_bytes_hash=bytes_hash)}
        elif args.write_feasibility:
            result = write_feasibility_artifact(args.protocol or name)
        elif args.verify:
            result = verify_selection_artifact(args.verify, protocol_path=args.protocol)
        elif args.stage:
            if args.stage == "all":
                result = run_v6_selection(protocol_path=args.protocol)
            elif args.stage == "replay":
                result = run_replay_stage(protocol, output_dir=models.assets_root, probe=lambda state: _subprocess_replay_probe(name, state), protocol_bytes_hash=bytes_hash)
            elif args.stage == "driver":
                result = run_driver_stage(protocol, output_dir=models.assets_root, probe=_v6_driver_probe, protocol_bytes_hash=bytes_hash)
            elif args.stage == "isolator":
                result = run_isolator_stage(protocol, output_dir=models.assets_root, probe=_v6_isolator_probe, protocol_bytes_hash=bytes_hash)
            elif args.stage == "contact":
                result = run_contact_stage(protocol, output_dir=models.assets_root, probe=_v6_contact_probe, protocol_bytes_hash=bytes_hash)
            else:
                result = run_parity_stage(protocol, output_dir=models.assets_root, probe=_v6_parity_probe, protocol_bytes_hash=bytes_hash)
        else:
            parser.error("choose --validate-protocol, --dry-run-manifest, --adapter-contract, --stage, or --verify")
            return 2
    except (V6ProtocolError, V6ProtocolStateError, AdapterContractError, V6EvidenceError, OSError, json.JSONDecodeError) as exc:
        result = {"passed": False, "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2
    if args.quiet and isinstance(result, Mapping):
        result = {key: result[key] for key in ("status", "state_count", "passed_state_count") if key in result}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if isinstance(result, Mapping) and result.get("passed") is False:
        return 1
    return 0


# Short aliases make the API boundary convenient for focused tests.
ResolvedProbeState = ResolvedProbeState
resolve_protocol_state = resolve_protocol_state_v6
resolve_all_protocol_states = resolve_all_protocol_states_v6


if __name__ == "__main__":
    raise SystemExit(main())
