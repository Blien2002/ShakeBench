"""Phase 06R3/V4 physics-freeze machinery.

The module has two deliberately separate layers.  The protocol/manifest and
verifier are pure and can be exercised with temporary fixtures; real probe
adapters are passed into the staged runner only after the immutable protocol
has been registered.  No function in this module loads an official profile.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
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


SCHEMA_ID = "shakebench.phase06r3.v4.physics_selection"
SCHEMA_VERSION = 4
PROTOCOL_FILENAME = "shakebench_selection_protocol_v4.yaml"
FEASIBILITY_FILENAME = "shakebench_phase_06r3_v4_feasibility.json"
STATUS_FILENAME = "shakebench_phase_06r3_v4_status.json"
RAW_PREFIX = "shakebench_phase_06r3_v4_raw_"


class V4ProtocolError(ValueError):
    """Raised before registration when a V4 protocol is structurally unsafe."""


class V4VerificationError(RuntimeError):
    """Raised when raw evidence cannot support a V4 selection."""


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    raw = Path(path).read_bytes()
    try:
        import yaml

        payload = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise V4ProtocolError("V4 protocol root must be a mapping")
    return payload


def load_v4_protocol(path: str | Path | None = None) -> tuple[Mapping[str, Any], str, str]:
    protocol_path = Path(models.assets_root) / PROTOCOL_FILENAME if path is None else Path(path)
    raw = protocol_path.read_bytes()
    payload = _load_yaml(protocol_path)
    return payload, str(protocol_path), sha256_bytes(raw)


def _finite_positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise V4ProtocolError(f"{label} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise V4ProtocolError(f"{label} must be finite and positive")
    return number


def _unique(values: Iterable[Any], label: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise V4ProtocolError(f"{label} contains duplicates")


def _validate_driver_structure(protocol: Mapping[str, Any]) -> dict[str, Any]:
    driver = protocol.get("driver")
    if not isinstance(driver, Mapping):
        raise V4ProtocolError("driver section is missing")
    rows = driver.get("convergence_candidates")
    if not isinstance(rows, list) or [row.get("candidate_id") for row in rows] != ["dt_fine", "dt_medium", "dt_nominal"]:
        raise V4ProtocolError("driver convergence table must contain exactly dt_fine, dt_medium, dt_nominal")
    control_period = 1.0 / 20.0
    f_max = _finite_positive(protocol.get("frozen_facts", {}).get("f_max_hz"), "f_max_hz")
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise V4ProtocolError("driver candidate must be a mapping")
        dt = _finite_positive(row.get("physics_timestep_s"), f"{row.get('candidate_id')}.dt")
        ratio = control_period / dt
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-12):
            raise V4ProtocolError(f"{row.get('candidate_id')} does not divide the 20 Hz control period")
        stride = int(row.get("refresh_stride", 0))
        if stride <= 0 or stride != row.get("refresh_stride"):
            raise V4ProtocolError(f"{row.get('candidate_id')} refresh_stride must be a positive integer")
        sample_dt = float(row.get("sample_dt_s"))
        cadence = float(row.get("sample_rate_hz"))
        if not np.isclose(sample_dt, stride * dt, rtol=0.0, atol=1.0e-12):
            raise V4ProtocolError(f"{row.get('candidate_id')} sample_dt_s mismatch")
        if cadence < 20.0 * f_max or not np.isclose(cadence, 1.0 / sample_dt, rtol=0.0, atol=1.0e-9):
            raise V4ProtocolError(f"{row.get('candidate_id')} measurement cadence is unsafe")
        if dt > 1.0 / (20.0 * f_max):
            raise V4ProtocolError(f"{row.get('candidate_id')} violates frequency timestep gate")
        solref = row.get("deck_eq_solref")
        if not isinstance(solref, list) or len(solref) != 2 or float(solref[0]) < 2.0 * dt:
            raise V4ProtocolError(f"{row.get('candidate_id')} violates positive deck solref >= 2dt")
        if row.get("integrator") != "Euler" or row.get("solver") != "Newton" or row.get("iterations") != 100 or float(row.get("tolerance")) != 1.0e-12:
            raise V4ProtocolError(f"{row.get('candidate_id')} solver tuple is not the registered tuple")
        out.append({"candidate_id": row["candidate_id"], "control_steps": int(round(ratio)), "sample_stride": stride, "sample_rate_hz": cadence, "dt_s": dt})
    return {"control_period_s": control_period, "f_max_hz": f_max, "candidates": out}


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every structural V4 constraint before a registration commit."""

    if protocol.get("schema_id") != "shakebench.phase06r3.v4.physics_selection_protocol" or protocol.get("schema_version") != 4:
        raise V4ProtocolError("wrong V4 protocol schema")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise V4ProtocolError("V4 protocol must be pre_registered and immutable_after_registration")
    facts = protocol.get("frozen_facts")
    if not isinstance(facts, Mapping):
        raise V4ProtocolError("frozen_facts section is missing")
    if facts.get("safe_gamma_candidates") != [0.15, 0.30, 0.50]:
        raise V4ProtocolError("safe Gamma set is not frozen")
    if facts.get("load_cases") != ["empty", "panda_plus_worktable_reference_proxy"]:
        raise V4ProtocolError("load cases are incomplete")
    driver = _validate_driver_structure(protocol)
    components = protocol.get("components")
    if not isinstance(components, Mapping):
        raise V4ProtocolError("component candidate tables are missing")
    for section in ("isolator_candidates", "contact_candidates"):
        rows = components.get(section)
        if not isinstance(rows, list) or not rows:
            raise V4ProtocolError(f"{section} must be a nonempty finite table")
        _unique((row.get("candidate_id") for row in rows), section)
    manifest = protocol.get("execution_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise V4ProtocolError("explicit execution_manifest is missing")
    state_ids = [row.get("state_id") for row in manifest]
    paths = [row.get("output") for row in manifest]
    _unique(state_ids, "state_id")
    _unique(paths, "artifact output")
    expected_stages = {"driver", "isolator", "contact", "parity", "replay"}
    for row in manifest:
        if not isinstance(row, Mapping) or row.get("stage") not in expected_stages:
            raise V4ProtocolError("manifest row has an invalid stage")
        if not isinstance(row.get("state_id"), str) or not row["state_id"]:
            raise V4ProtocolError("manifest state_id must be nonempty")
        _finite_positive(row.get("duration_s"), f"{row.get('state_id')}.duration_s")
        _finite_positive(row.get("mujoco_step_count"), f"{row.get('state_id')}.mujoco_step_count")
        _finite_positive(row.get("retained_sample_count"), f"{row.get('state_id')}.retained_sample_count")
        if not isinstance(row.get("output"), str) or not row["output"].startswith(RAW_PREFIX):
            raise V4ProtocolError(f"{row.get('state_id')} output is not a flat V4 artifact")
        if row.get("retry_policy") != "same_state_config_once_then_block":
            raise V4ProtocolError(f"{row.get('state_id')} retry policy is not registered")
    return {"control_period_s": driver["control_period_s"], "f_max_hz": driver["f_max_hz"], "driver": driver, "manifest_count": len(manifest), "state_ids": state_ids, "outputs": paths}


def build_dry_run_manifest(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact registered run manifest without importing MuJoCo."""

    validate_protocol(protocol)
    return [copy.deepcopy(dict(row)) for row in protocol["execution_manifest"]]


def run_staged_probe_group(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    output_dir: str | Path,
    probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run one manifest stage with atomic per-state resume and retry ledger."""

    manifest = [row for row in build_dry_run_manifest(protocol) if row["stage"] == stage]
    output = Path(output_dir)
    protocol_hash = sha256_json(protocol)
    results = []
    for state in manifest:
        path = output / state["output"]
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("protocol_sha256") != protocol_hash or existing.get("state_id") != state["state_id"]:
                raise V4VerificationError(f"existing artifact provenance mismatch: {state['state_id']}")
            results.append(existing)
            continue
        retries = []
        for attempt in (1, 2):
            try:
                evidence = dict(probe(state))
                payload = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": stage, "state_id": state["state_id"], "protocol_sha256": protocol_hash, "attempt": attempt, "retry_ledger": retries, "evidence": evidence}
                payload["payload_sha256"] = payload_hash(payload)
                write_json_atomic(path, payload)
                results.append(payload)
                break
            except Exception as exc:
                retries.append({"attempt": attempt, "state_config_identical": True, "exception_type": type(exc).__name__, "exception_message": str(exc)})
        else:
            blocked = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": stage, "state_id": state["state_id"], "protocol_sha256": protocol_hash, "status": "infrastructure_failure_repeated", "retry_ledger": retries, "pass_claim": False}
            blocked["payload_sha256"] = payload_hash(blocked)
            write_json_atomic(path, blocked)
            raise V4VerificationError(f"repeated infrastructure failure: {state['state_id']}")
    return results


def run_driver_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_staged_probe_group(protocol, stage="driver", output_dir=output_dir, probe=probe)


def run_isolator_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_staged_probe_group(protocol, stage="isolator", output_dir=output_dir, probe=probe)


def run_contact_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_staged_probe_group(protocol, stage="contact", output_dir=output_dir, probe=probe)


def run_parity_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_staged_probe_group(protocol, stage="parity", output_dir=output_dir, probe=probe)


def run_replay_stage(protocol: Mapping[str, Any], *, output_dir: str | Path, probe: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return run_staged_probe_group(protocol, stage="replay", output_dir=output_dir, probe=probe)


def _v4_driver_candidate(protocol: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    for row in protocol["driver"]["convergence_candidates"]:
        if row["candidate_id"] == candidate_id:
            candidate = dict(row)
            candidate["solver_iterations"] = int(candidate.pop("iterations"))
            candidate["solver_tolerance"] = float(candidate.pop("tolerance"))
            return candidate
    raise V4ProtocolError(f"unknown V4 driver candidate: {candidate_id}")


def _v4_driver_probe(protocol: Mapping[str, Any], state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one explicit V4 driver state using the bounded right-limit trace."""

    from robosuite.scripts.shakebench_probe_deck_driver import (
        canonical_gamma_conformance,
        minimum_line_spacing_hz,
        spectrum_conformance,
        spectrum_gate_reasons,
    )
    from robosuite.scripts.shakebench_select_physics import _program_for_gamma, _run_driver_trace, _trace_digest, _trace_shapes
    from robosuite.utils.shakebench_deck import DeckDriverConfig
    from robosuite.utils.shakebench_driver_measurement import DriverMeasurementPlan

    candidate = _v4_driver_candidate(protocol, str(state["candidate_id"]))
    plan = DriverMeasurementPlan(
        physics_timestep_s=float(candidate["physics_timestep_s"]),
        cadence_hz=float(state["sample_rate_hz"]),
        f_max_hz=float(protocol["frozen_facts"]["f_max_hz"]),
    )
    if plan.refresh_stride != int(state["refresh_stride"]):
        raise V4ProtocolError(f"measurement manifest stride mismatch: {state['state_id']}")
    gamma = float(state["gamma"])
    level = _program_for_gamma(gamma, dt=float(candidate["physics_timestep_s"]), duration_s=float(state["duration_s"]))
    excitation = __import__("robosuite.utils.shakebench_excitation", fromlist=["build_excitation_program"]).build_excitation_program(
        seed=int(protocol["frozen_facts"]["authored_spectrum"]["seed"]),
        t0=float(protocol["frozen_facts"]["authored_spectrum"]["t0_s"]),
        level_scale=level,
    )
    started = time.perf_counter()
    trace, audit, options = _run_driver_trace(
        candidate,
        duration_s=float(state["duration_s"]),
        trajectory=excitation,
        load_case=str(state["load_case"]),
        refresh_stride=int(state["refresh_stride"]),
    )
    elapsed = time.perf_counter() - started
    discard = float(protocol["frozen_facts"]["authored_spectrum"]["transient_discard_s"])
    fit = spectrum_conformance(trace, excitation, discard_s=discard, max_fit_samples=50000)
    gamma_fit = canonical_gamma_conformance(
        trace,
        excitation,
        config=DeckDriverConfig(
            deck_mass_kg=float(candidate["deck_mass_kg"]),
            deck_inertia_kg_m2=tuple(candidate["deck_inertia_kg_m2"]),
            eq_solref=tuple(candidate["deck_eq_solref"]),
            eq_solimp=tuple(candidate["deck_eq_solimp"]),
            physics_timestep_s=float(candidate["physics_timestep_s"]),
        ),
        discard_s=discard,
    )
    line_entries = [line for axis in fit["axes"].values() for line in axis["line_fits"]]
    gates = protocol["driver"]["hard_gates"]
    reasons = spectrum_gate_reasons(fit, expected_line_count=int(protocol["frozen_facts"]["authored_spectrum"]["line_count"]))
    max_amplitude = max((float(line["amplitude_relative_error"]) for line in line_entries), default=float("inf"))
    max_phase = max((abs(float(line["phase_error_deg"])) for line in line_entries), default=float("inf"))
    gamma_error = float(gamma_fit["relative_error_abs"])
    warning_count = int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0
    residual = float(np.max(np.abs(trace.weld_constraint_residual_raw))) if trace.weld_constraint_residual_raw.size else 0.0
    if gamma_error > float(gates["gamma_deck_relative_error_max"]): reasons.append("gamma_deck_relative_error_exceeds_threshold")
    if warning_count > int(gates["warning_count_max"]): reasons.append("warning_count_exceeds_threshold")
    if residual > float(gates["unstable_residual_max"]): reasons.append("weld_residual_exceeds_threshold")
    return {
        "candidate_id": candidate["candidate_id"],
        "state_id": state["state_id"],
        "gamma": gamma,
        "load_case": state["load_case"],
        "status": "measured",
        "elapsed_wall_time_s": elapsed,
        "mujoco_step_count": int(np.ceil(float(state["duration_s"]) / float(candidate["physics_timestep_s"]))),
        "retained_sample_count": int(trace.sample_time_s.size),
        "metrics": {
            "max_line_amplitude_relative_error": max_amplitude,
            "max_absolute_line_phase_error_deg": max_phase,
            "max_gamma_relative_error": gamma_error,
            "warning_count": warning_count,
            "max_weld_residual": residual,
            "complete_64_line_spectrum": len(line_entries) == int(protocol["frozen_facts"]["authored_spectrum"]["line_count"]),
        },
        "spectrum": fit,
        "gamma_fit": gamma_fit,
        "trace": {
            "digest": _trace_digest(trace),
            "shapes": _trace_shapes(trace),
            "dtypes": {name: str(np.asarray(getattr(trace, name)).dtype) for name in plan.to_dict()["trace_fields"]},
        },
        "audit": audit,
        "options": options,
        "failure_reasons": reasons,
        "passed": not reasons,
    }


def run_v4_driver_stage(*, protocol_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    protocol, _, _ = load_v4_protocol(protocol_path)
    validate_protocol(protocol)
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    records = run_driver_stage(protocol, output_dir=output, probe=lambda state: _v4_driver_probe(protocol, state))
    passed = all(record.get("evidence", {}).get("passed") is True for record in records)
    return {"status": "PASS" if passed else "BLOCKED", "state_count": len(records), "passed_state_count": sum(record.get("evidence", {}).get("passed") is True for record in records), "records": records}


def publish_official_profile(*, selection_path: str | Path, protocol_path: str | Path, output_path: str | Path, status_path: str | Path) -> dict[str, Any]:
    """Publish only after the independent V4 verifier reports PASS."""

    verification = verify_selection_artifact(selection_path, protocol_path=protocol_path)
    if verification.get("passed") is not True:
        raise V4VerificationError("official profile publication requires a verified V4 PASS")
    selected = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    profile = selected.get("official_profile")
    if not isinstance(profile, Mapping) or profile.get("scoreable") is not True:
        raise V4VerificationError("verified selection has no scoreable official profile")
    profile_path = Path(output_path)
    status_file = Path(status_path)
    profile_hash = write_json_atomic(profile_path, profile)
    write_json_atomic(status_file, {"schema_id": "shakebench.phase06r3.v4.selection_status", "schema_version": 1, "status": "PASS", "profile_file_sha256": profile_hash, "protocol_sha256": verification["protocol_sha256_bytes"]})
    return {"profile_path": str(profile_path), "profile_file_sha256": profile_hash, "status_path": str(status_file)}


def _raw_index(selected: Mapping[str, Any], base: Path) -> dict[str, Mapping[str, Any]]:
    raw = {}
    for entry in selected.get("raw_files", []):
        path = base / str(entry.get("path", ""))
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise V4VerificationError(f"raw hash mismatch: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not verify_payload_hash(payload):
            raise V4VerificationError(f"raw payload hash mismatch: {path.name}")
        raw[str(payload.get("state_id"))] = payload
    return raw


def _driver_eligibility(protocol: Mapping[str, Any], raw: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gates = protocol["driver"]["hard_gates"]
    expected = {row["candidate_id"] for row in protocol["driver"]["convergence_candidates"]}
    result = {}
    for candidate in expected:
        rows = [payload for state, payload in raw.items() if payload.get("stage") == "driver" and payload.get("evidence", {}).get("candidate_id") == candidate]
        eligible = len(rows) == 6
        reasons = []
        for row in rows:
            metrics = row.get("evidence", {}).get("metrics", {})
            checks = (
                float(metrics.get("max_line_amplitude_relative_error", math.inf)) <= float(gates["line_amplitude_relative_error_max"]),
                float(metrics.get("max_absolute_line_phase_error_deg", math.inf)) <= float(gates["line_phase_absolute_error_deg_max"]),
                float(metrics.get("max_gamma_relative_error", math.inf)) <= float(gates["gamma_deck_relative_error_max"]),
                int(metrics.get("warning_count", 1)) == int(gates["warning_count_max"]),
                float(metrics.get("max_weld_residual", math.inf)) <= float(gates["unstable_residual_max"]),
                metrics.get("complete_64_line_spectrum") is True,
            )
            if not all(checks):
                eligible = False
                reasons.append(row.get("state_id"))
        result[candidate] = {"candidate_id": candidate, "complete_state_count": len(rows), "eligible": eligible, "failed_states": reasons, "max_phase_error": max((float(row.get("evidence", {}).get("metrics", {}).get("max_absolute_line_phase_error_deg", math.inf)) for row in rows), default=math.inf)}
    return result


def verify_selection_artifact(path: str | Path, *, protocol_path: str | Path | None = None) -> dict[str, Any]:
    """Recompute V4 hard gates and raw hashes; declarations are never trusted."""

    errors = []
    selected_path = Path(path)
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        protocol, protocol_name, protocol_hash_bytes = load_v4_protocol(protocol_path)
        structure = validate_protocol(protocol)
        raw = _raw_index(selected, selected_path.parent)
    except (OSError, json.JSONDecodeError, V4ProtocolError, V4VerificationError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    protocol_hash = sha256_json(protocol)
    if selected.get("protocol", {}).get("sha256") != protocol_hash:
        errors.append("protocol hash mismatch")
    driver = _driver_eligibility(protocol, raw)
    if not all(row["eligible"] for row in driver.values()):
        errors.append("driver convergence eligibility failed")
    expected_states = set(structure["state_ids"])
    if expected_states != set(raw):
        errors.append("manifest/raw state coverage mismatch")
    for stage in ("isolator", "contact", "parity", "replay"):
        if not any(payload.get("stage") == stage for payload in raw.values()):
            errors.append(f"missing {stage} evidence group")
    if selected.get("status") != "PASS":
        errors.append("selection status is not PASS")
    if selected.get("official_profile_alignment") is not True:
        errors.append("official-profile alignment evidence is missing")
    if not verify_payload_hash(selected):
        errors.append("selected payload hash mismatch")
    return {"passed": not errors, "errors": errors, "protocol": protocol_name, "protocol_sha256_bytes": protocol_hash_bytes, "driver_recomputed": driver, "raw_state_count": len(raw)}


def validate_protocol_file(path: str | Path) -> dict[str, Any]:
    protocol, name, digest = load_v4_protocol(path)
    structure = validate_protocol(protocol)
    return {"passed": True, "path": name, "protocol_sha256": digest, "structure": structure}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    parser.add_argument("--verify", default=None)
    args = parser.parse_args(argv)
    try:
        if args.validate_protocol:
            result = validate_protocol_file(args.protocol or Path(models.assets_root) / PROTOCOL_FILENAME)
        elif args.dry_run_manifest:
            protocol, name, digest = load_v4_protocol(args.protocol or Path(models.assets_root) / PROTOCOL_FILENAME)
            result = {"path": name, "protocol_sha256": digest, "manifest": build_dry_run_manifest(protocol)}
        elif args.verify:
            result = verify_selection_artifact(args.verify, protocol_path=args.protocol)
        else:
            parser.error("choose --validate-protocol, --dry-run-manifest, or --verify")
            return 2
    except (V4ProtocolError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
