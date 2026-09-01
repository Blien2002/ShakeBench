"""Phase 06R4/V5 protocol-driven physics selection.

Every execution path resolves a manifest reference through
``resolve_protocol_state``.  The V5 protocol is intentionally not imported
from an official profile and this module can be tested entirely with temporary
synthetic protocols before V5 registration.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robosuite import models
from robosuite.utils.shakebench_artifacts import (
    canonical_json,
    file_sha256,
    payload_hash,
    sha256_bytes,
    sha256_json,
    verify_payload_hash,
    write_json_atomic,
)
from robosuite.utils.shakebench_protocol import (
    ProtocolStateError,
    ResolvedPhysicsProbeState,
    resolve_all_protocol_states,
    resolve_protocol_state,
)


SCHEMA_ID = "shakebench.phase06r4.v5.physics_selection"
SCHEMA_VERSION = 5
PROTOCOL_FILENAME = "shakebench_selection_protocol_v5.yaml"
RAW_PREFIX = "shakebench_phase_06r4_v5_raw_"


class V5ProtocolError(ValueError):
    """A V5 protocol or resolved state is invalid before probing."""


class V5EvidenceError(RuntimeError):
    """V5 raw evidence is missing, mutated, or incomplete."""


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    try:
        import yaml

        value = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise V5ProtocolError("V5 protocol root must be a mapping")
    return value


def load_v5_protocol(path: str | Path | None = None) -> tuple[Mapping[str, Any], str, str]:
    protocol_path = Path(models.assets_root) / PROTOCOL_FILENAME if path is None else Path(path)
    raw = protocol_path.read_bytes()
    return _load_yaml(protocol_path), str(protocol_path), sha256_bytes(raw)


def validate_v5_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("schema_id") != "shakebench.phase06r4.v5.physics_selection_protocol" or protocol.get("schema_version") != 5:
        raise V5ProtocolError("wrong V5 protocol schema")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise V5ProtocolError("V5 protocol must be pre_registered and immutable")
    try:
        states = resolve_all_protocol_states(protocol)
    except ProtocolStateError as exc:
        raise V5ProtocolError(str(exc)) from exc
    if len(states) != int(protocol.get("manifest_state_count", len(states))):
        raise V5ProtocolError("manifest_state_count does not match resolved state count")
    driver_states = [state for state in states if state.stage == "driver"]
    expected_drivers = {"dt_fine", "dt_medium", "dt_nominal"}
    if {state.candidate_id for state in driver_states} != expected_drivers:
        raise V5ProtocolError("V5 driver state set is incomplete")
    expected_gamma = {0.15, 0.30, 0.50}
    expected_load = {"empty", "panda_plus_worktable_reference_proxy"}
    for candidate_id in expected_drivers:
        rows = [state for state in driver_states if state.candidate_id == candidate_id]
        if {(state.gamma, state.load_case) for state in rows} != {(gamma, load) for gamma in expected_gamma for load in expected_load}:
            raise V5ProtocolError(f"driver matrix incomplete for {candidate_id}")
    outputs = [state.output for state in states]
    if any(not output.startswith(RAW_PREFIX) for output in outputs):
        raise V5ProtocolError("V5 output path is outside the flat raw prefix")
    if protocol.get("forbidden_selection_inputs") or "Gamma_star" in canonical_json(protocol.get("scope", {})):
        # The declaration itself is permitted; evidence and selection data
        # are checked separately. This branch only rejects a literal forbidden
        # result field accidentally placed at the root.
        for forbidden in ("task_success_rate", "task_sr", "controller_outcome_used"):
            if forbidden in protocol and protocol.get(forbidden) not in (False, None):
                raise V5ProtocolError(f"forbidden selection input appears as a protocol result: {forbidden}")
    return {
        "state_count": len(states),
        "driver_state_count": len(driver_states),
        "state_ids": [state.state_id for state in states],
        "outputs": outputs,
        "resolved_state_digest": sha256_json([state.to_dict() for state in states]),
    }


def resolve_v5_manifest(protocol: Mapping[str, Any]) -> tuple[ResolvedPhysicsProbeState, ...]:
    validate_v5_protocol(protocol)
    return resolve_all_protocol_states(protocol)


def dry_run_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    states = resolve_v5_manifest(protocol)
    return {
        "schema_id": SCHEMA_ID + ".resolved_manifest",
        "schema_version": SCHEMA_VERSION,
        "state_count": len(states),
        "resolved_state_digest": sha256_json([state.to_dict() for state in states]),
        "states": [state.to_dict() for state in states],
        "mujoco_model_created": False,
    }


def run_resolved_stage(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    output_dir: str | Path,
    probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run/resume a stage using only immutable resolved states."""

    states = [state for state in resolve_v5_manifest(protocol) if state.stage == stage]
    output = Path(output_dir)
    protocol_hash = sha256_json(protocol)
    results = []
    for state in states:
        path = output / state.output
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("state_id") != state.state_id or existing.get("protocol_sha256") != protocol_hash:
                raise V5EvidenceError(f"resolved-state provenance mismatch: {state.state_id}")
            if existing.get("resolved_state_digest") != sha256_json(state.to_dict()) or not verify_payload_hash(existing):
                raise V5EvidenceError(f"resolved-state/raw hash mismatch: {state.state_id}")
            results.append(existing)
            continue
        retry_ledger = []
        for attempt in (1, 2):
            try:
                evidence = dict(probe(state))
                payload = {
                    "schema_id": SCHEMA_ID + ".raw",
                    "schema_version": SCHEMA_VERSION,
                    "stage": state.stage,
                    "state_id": state.state_id,
                    "protocol_sha256": protocol_hash,
                    "resolved_state_digest": sha256_json(state.to_dict()),
                    "attempt": attempt,
                    "retry_ledger": retry_ledger,
                    "evidence": evidence,
                }
                if state.stage == "driver":
                    payload["protocol_driver_gates"] = copy.deepcopy(dict(protocol["driver"]["hard_gates"]))
                payload["payload_sha256"] = payload_hash(payload)
                write_json_atomic(path, payload)
                results.append(payload)
                break
            except ProtocolStateError:
                raise
            except Exception as exc:
                retry_ledger.append({"attempt": attempt, "state_config_identical": True, "exception_type": type(exc).__name__, "exception_message": str(exc), "failure_taxonomy": "infrastructure_failure"})
        else:
            payload = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": state.stage, "state_id": state.state_id, "protocol_sha256": protocol_hash, "resolved_state_digest": sha256_json(state.to_dict()), "status": "infrastructure_failure_repeated", "retry_ledger": retry_ledger, "pass_claim": False}
            payload["payload_sha256"] = payload_hash(payload)
            write_json_atomic(path, payload)
            raise V5EvidenceError(f"repeated infrastructure failure: {state.state_id}")
    return results


def run_resolved_state(
    protocol: Mapping[str, Any],
    *,
    state_id: str,
    output_dir: str | Path,
    probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute one resolved state, preserving the same stage retry contract."""

    state = resolve_protocol_state(protocol, state_id)
    output = Path(output_dir)
    protocol_hash = sha256_json(protocol)
    path = output / state.output
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("state_id") != state.state_id or existing.get("protocol_sha256") != protocol_hash or not verify_payload_hash(existing):
            raise V5EvidenceError(f"resolved-state provenance mismatch: {state.state_id}")
        return existing
    retry_ledger = []
    for attempt in (1, 2):
        try:
            evidence = dict(probe(state))
            payload = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": state.stage, "state_id": state.state_id, "protocol_sha256": protocol_hash, "resolved_state_digest": sha256_json(state.to_dict()), "attempt": attempt, "retry_ledger": retry_ledger, "evidence": evidence}
            if state.stage == "driver":
                payload["protocol_driver_gates"] = copy.deepcopy(dict(protocol["driver"]["hard_gates"]))
            payload["payload_sha256"] = payload_hash(payload)
            write_json_atomic(path, payload)
            return payload
        except ProtocolStateError:
            raise
        except Exception as exc:
            retry_ledger.append({"attempt": attempt, "state_config_identical": True, "exception_type": type(exc).__name__, "exception_message": str(exc), "failure_taxonomy": "infrastructure_failure"})
    payload = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": state.stage, "state_id": state.state_id, "protocol_sha256": protocol_hash, "resolved_state_digest": sha256_json(state.to_dict()), "status": "infrastructure_failure_repeated", "retry_ledger": retry_ledger, "pass_claim": False}
    payload["payload_sha256"] = payload_hash(payload)
    write_json_atomic(path, payload)
    raise V5EvidenceError(f"repeated infrastructure failure: {state.state_id}")


def _v5_driver_probe(state: ResolvedPhysicsProbeState, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute one V5 driver state from the already-resolved immutable state."""

    from robosuite.scripts.shakebench_probe_deck_driver import (
        canonical_gamma_conformance,
        spectrum_conformance,
        spectrum_gate_reasons,
    )
    from robosuite.scripts.shakebench_select_physics import _program_for_gamma, _run_driver_trace, _trace_digest, _trace_shapes
    from robosuite.utils.shakebench_deck import DeckDriverConfig
    from robosuite.utils.shakebench_deck import TRACE_FIELD_CONTRACT
    from robosuite.utils.shakebench_excitation import build_excitation_program

    if state.stage != "driver" or state.gamma is None or state.load_case is None:
        raise ProtocolStateError("driver adapter received a non-driver resolved state")
    candidate = {
        "candidate_id": state.candidate_id,
        "physics_timestep_s": state.physics_timestep_s,
        "integrator": state.integrator,
        "solver": state.solver,
        "solver_iterations": state.solver_iterations,
        "solver_tolerance": state.solver_tolerance,
        "deck_mass_kg": state.deck_mass_kg,
        "deck_inertia_kg_m2": list(state.deck_inertia_kg_m2),
        "deck_eq_solref": list(state.deck_eq_solref),
        "deck_eq_solimp": list(state.deck_eq_solimp),
    }
    level = _program_for_gamma(state.gamma, dt=state.physics_timestep_s, duration_s=state.duration_s)
    facts = protocol["frozen_facts"]
    program = build_excitation_program(
        seed=int(facts["authored_spectrum"]["seed"]),
        t0=float(facts["authored_spectrum"]["t0_s"]),
        level_scale=level,
    )
    started = time.perf_counter()
    trace, audit, options = _run_driver_trace(
        candidate,
        duration_s=state.duration_s,
        trajectory=program,
        load_case=state.load_case,
        refresh_stride=state.refresh_stride,
    )
    elapsed = time.perf_counter() - started
    discard = float(facts["authored_spectrum"]["transient_discard_s"])
    spectrum = spectrum_conformance(trace, program, discard_s=discard, max_fit_samples=50000)
    gamma_fit = canonical_gamma_conformance(
        trace,
        program,
        config=DeckDriverConfig(
            deck_mass_kg=state.deck_mass_kg,
            deck_inertia_kg_m2=state.deck_inertia_kg_m2,
            eq_solref=state.deck_eq_solref,
            eq_solimp=state.deck_eq_solimp,
            physics_timestep_s=state.physics_timestep_s,
        ),
        discard_s=discard,
    )
    lines = [line for axis in spectrum["axes"].values() for line in axis["line_fits"]]
    gates = protocol["driver"]["hard_gates"]
    reasons = spectrum_gate_reasons(spectrum, expected_line_count=int(facts["authored_spectrum"]["line_count"]))
    max_amplitude = max((float(line["amplitude_relative_error"]) for line in lines), default=float("inf"))
    max_phase = max((abs(float(line["phase_error_deg"])) for line in lines), default=float("inf"))
    gamma_error = float(gamma_fit["relative_error_abs"])
    warning_count = int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0
    residual = float(np.max(np.abs(trace.weld_constraint_residual_raw))) if trace.weld_constraint_residual_raw.size else 0.0
    if gamma_error > float(gates["gamma_deck_relative_error_max"]): reasons.append("gamma_deck_relative_error_exceeds_threshold")
    if warning_count > int(gates["warning_count_max"]): reasons.append("warning_count_exceeds_threshold")
    if residual > float(gates["unstable_residual_max"]): reasons.append("weld_residual_exceeds_threshold")
    return {
        "candidate_id": state.candidate_id,
        "gamma": state.gamma,
        "load_case": state.load_case,
        "metrics": {
            "max_line_amplitude_relative_error": max_amplitude,
            "max_absolute_line_phase_error_deg": max_phase,
            "max_gamma_relative_error": gamma_error,
            "warning_count": warning_count,
            "max_weld_residual": residual,
            "complete_64_line_spectrum": len(lines) == int(facts["authored_spectrum"]["line_count"]),
        },
        "spectrum": spectrum,
        "gamma_fit": gamma_fit,
        "timing": {"elapsed_wall_time_s": elapsed, "mujoco_step_count": int(state.mujoco_step_count), "retained_sample_count": int(trace.sample_time_s.size)},
        "trace": {"digest": _trace_digest(trace), "shapes": _trace_shapes(trace), "dtypes": {name: str(np.asarray(getattr(trace, name)).dtype) for name in TRACE_FIELD_CONTRACT}},
        "audit": audit,
        "options": options,
        "failure_reasons": reasons,
        "passed": not reasons,
    }


def run_v5_driver_stage(*, protocol_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    protocol, _, _ = load_v5_protocol(protocol_path)
    validate_v5_protocol(protocol)
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    records = run_driver_stage(protocol, output_dir=output, probe=lambda state: _v5_driver_probe(state, protocol))
    passed = all(record.get("evidence", {}).get("passed") is True for record in records)
    return {"status": "PASS" if passed else "BLOCKED", "state_count": len(records), "passed_state_count": sum(record.get("evidence", {}).get("passed") is True for record in records), "records": records}


def run_driver_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="driver", output_dir=output_dir, probe=probe)


def run_isolator_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="isolator", output_dir=output_dir, probe=probe)


def run_contact_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="contact", output_dir=output_dir, probe=probe)


def run_parity_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="parity", output_dir=output_dir, probe=probe)


def run_replay_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[ResolvedPhysicsProbeState], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_resolved_stage(protocol, stage="replay", output_dir=output_dir, probe=probe)


def _driver_metrics_from_evidence(payload: Mapping[str, Any]) -> tuple[bool, float]:
    metrics = payload.get("evidence", {}).get("metrics", {})
    # Recompute eligibility from raw numeric values. The evidence's passed
    # flag is intentionally ignored.
    gates = payload.get("protocol_driver_gates", {})
    return bool(
        metrics.get("complete_64_line_spectrum") is True
        and float(metrics.get("max_line_amplitude_relative_error", np.inf)) <= float(gates.get("line_amplitude_relative_error_max", 0.01))
        and float(metrics.get("max_absolute_line_phase_error_deg", np.inf)) <= float(gates.get("line_phase_absolute_error_deg_max", 1.0))
        and float(metrics.get("max_gamma_relative_error", np.inf)) <= float(gates.get("gamma_deck_relative_error_max", 0.01))
        and int(metrics.get("warning_count", 1)) == int(gates.get("warning_count_max", 0))
        and float(metrics.get("max_weld_residual", np.inf)) <= float(gates.get("unstable_residual_max", 0.001))
    ), float(metrics.get("max_absolute_line_phase_error_deg", np.inf))


def verify_selection_artifact(path: str | Path, *, protocol_path: str | Path | None = None) -> dict[str, Any]:
    """Verify V5 raw coverage and numeric gates from raw values."""

    selected_path = Path(path)
    errors = []
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        protocol, protocol_name, protocol_bytes_hash = load_v5_protocol(protocol_path)
        structure = validate_v5_protocol(protocol)
    except (OSError, json.JSONDecodeError, V5ProtocolError, ProtocolStateError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    expected = set(structure["state_ids"])
    raw = {}
    for entry in selected.get("raw_files", []):
        raw_path = selected_path.parent / str(entry.get("path", ""))
        if not raw_path.is_file() or file_sha256(raw_path) != entry.get("sha256"):
            errors.append(f"raw file hash mismatch: {raw_path.name}")
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if not verify_payload_hash(payload):
            errors.append(f"raw payload hash mismatch: {raw_path.name}")
        raw[str(payload.get("state_id"))] = payload
    if set(raw) != expected:
        errors.append("raw state coverage differs from resolved manifest")
    for state_id in expected - set(raw):
        errors.append("missing raw state: " + state_id)
    if selected.get("protocol", {}).get("sha256_bytes") != protocol_bytes_hash:
        errors.append("protocol bytes hash mismatch")
    if selected.get("resolved_state_digest") != structure["resolved_state_digest"]:
        errors.append("resolved-state digest mismatch")
    driver_rows = [payload for payload in raw.values() if payload.get("stage") == "driver"]
    for candidate_id in ("dt_fine", "dt_medium", "dt_nominal"):
        candidate_rows = [row for row in driver_rows if row.get("evidence", {}).get("candidate_id") == candidate_id]
        if len(candidate_rows) != 6:
            errors.append(f"driver matrix incomplete: {candidate_id}")
        for row in candidate_rows:
            # V5 raw artifacts carry the protocol gates as provenance, but the
            # result declaration remains ignored.
            eligible, _ = _driver_metrics_from_evidence(row)
            if not eligible:
                errors.append("driver raw hard gate failed: " + str(row.get("state_id")))
    for stage in ("isolator", "contact", "parity", "replay"):
        if not any(row.get("stage") == stage for row in raw.values()):
            errors.append("missing stage group: " + stage)
    if selected.get("status") != "PASS":
        errors.append("selection status is not PASS")
    if not verify_payload_hash(selected):
        errors.append("selected payload hash mismatch")
    if selected.get("official_profile_alignment") is not True:
        errors.append("official profile alignment is missing")
    return {"passed": not errors, "errors": errors, "protocol": protocol_name, "raw_state_count": len(raw), "resolved_state_digest": structure["resolved_state_digest"]}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    parser.add_argument("--stage", choices=("driver", "isolator", "contact", "parity", "replay", "all"))
    parser.add_argument("--state-id", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verify", default=None)
    args = parser.parse_args(argv)
    try:
        if args.validate_protocol:
            protocol, name, digest = load_v5_protocol(args.protocol)
            result = {"passed": True, "path": name, "protocol_sha256_bytes": digest, "structure": validate_v5_protocol(protocol)}
        elif args.dry_run_manifest:
            protocol, name, digest = load_v5_protocol(args.protocol)
            result = {"path": name, "protocol_sha256_bytes": digest, **dry_run_manifest(protocol)}
        elif args.verify:
            result = verify_selection_artifact(args.verify, protocol_path=args.protocol)
        elif args.stage:
            if args.stage != "driver":
                raise V5ProtocolError("real V5 adapters for non-driver stages are not yet enabled")
            protocol, _, _ = load_v5_protocol(args.protocol)
            validate_v5_protocol(protocol)
            if args.state_id is None:
                result = run_v5_driver_stage(protocol_path=args.protocol)
            else:
                result = run_resolved_state(protocol, state_id=args.state_id, output_dir=models.assets_root, probe=lambda state: _v5_driver_probe(state, protocol))
        else:
            parser.error("choose --validate-protocol, --dry-run-manifest, or --verify")
            return 2
    except (V5ProtocolError, ProtocolStateError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    if args.quiet:
        if isinstance(result, Mapping) and "state_id" in result:
            result = {"status": "PASS", "state_id": result["state_id"], "attempt": result.get("attempt")}
        elif isinstance(result, Mapping):
            result = {key: result[key] for key in ("status", "state_count", "passed_state_count") if key in result}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
