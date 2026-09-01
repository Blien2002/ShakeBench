"""Execute the pre-registered Phase 06R physics-only selection.

This implementation deliberately consumes only the V2 protocol and frozen
physical facts.  It never loads the package official profile: the profile is
written only after all candidate evidence, parity, and replay gates pass.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_select_physics import (
    _contact_candidate_probe,
    _program_for_gamma,
    _run_driver_trace,
    _run_gamma_batch,
    _trace_digest,
    _trace_shapes,
)
from robosuite.scripts.shakebench_probe_deck_driver import (
    _program_hash,
    minimum_line_spacing_hz,
    spectrum_conformance,
    spectrum_gate_reasons,
)
from robosuite.utils.shakebench_isolator import (
    IsolatorConfig,
    derive_isolator_parameters,
    run_harmonic_transfer_grid,
    run_joint_spectrum_probe,
    run_payload_sensitivity_probe,
    static_sag_uncompensated_m,
)
from robosuite.utils.shakebench_physics import (
    PHYSICS_PROFILE_SCHEMA_ID,
    PHYSICS_PROFILE_SCHEMA_VERSION,
    PhysicsProfile,
    load_selection_protocol,
    make_probe_physics_profile,
    physics_profile_hash,
)


SCHEMA_ID = "shakebench.phase06r.physics_selection"
SCHEMA_VERSION = 2
PROTOCOL_FILENAME = "shakebench_selection_protocol_v2.yaml"
STATUS_FILENAME = "shakebench_phase_06r_status.json"


class PhysicsSelectionV2Error(RuntimeError):
    """A required V2 probe or integrity check could not complete."""


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_ready(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_protocol() -> tuple[Mapping[str, Any], str, str]:
    path = Path(models.assets_root) / PROTOCOL_FILENAME
    raw = path.read_bytes()
    try:
        import yaml

        payload = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") != "pre_registered":
        raise PhysicsSelectionV2Error("V2 protocol is not a pre-registered mapping")
    return payload, str(path), hashlib.sha256(raw).hexdigest()


def _driver_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "physics_timestep_s": float(row["physics_timestep_s"]),
        "integrator": str(row["integrator"]),
        "solver": str(row["solver"]),
        "solver_iterations": int(row["solver_iterations"]),
        "solver_tolerance": float(row["solver_tolerance"]),
        "deck_mass_kg": float(row["deck_mass_kg"]),
        "deck_inertia_kg_m2": [float(item) for item in row["deck_inertia_kg_m2"]],
        "deck_eq_solref": [float(item) for item in row["deck_eq_solref"]],
        "deck_eq_solimp": [float(item) for item in row["deck_eq_solimp"]],
    }


def _warning_count(audit: Mapping[str, Any]) -> int:
    return int(audit["deck"]["warnings"])


def _driver_screen(candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the complete registered Gamma x load x spectrum matrix."""

    driver = protocol["driver"]
    hard = driver["hard_gates"]
    program = __import__("robosuite.utils.shakebench_excitation", fromlist=["build_excitation_program"]).build_excitation_program(
        seed=int(driver["authored_spectrum"]["seed"]),
        t0=float(driver["authored_spectrum"]["t0_s"]),
        level_scale=0.30,
    )
    discard_s = float(driver["authored_spectrum"]["transient_discard_s"])
    fit_window_s = 2.0 / minimum_line_spacing_hz(program)
    duration_s = discard_s + fit_window_s + float(candidate["physics_timestep_s"])
    records: dict[str, Any] = {}
    crashes: list[dict[str, str]] = []
    for load_case in driver["load_cases"]:
        load = str(load_case)
        try:
            trace, audit, options = _run_driver_trace(
                candidate, duration_s=duration_s, trajectory=program, load_case=load, refresh_stride=16
            )
            fit = spectrum_conformance(trace, program, discard_s=discard_s, max_fit_samples=50000)
            reasons = spectrum_gate_reasons(fit, expected_line_count=int(driver["authored_spectrum"]["line_count"]))
            line_fits = fit.get("line_fits", [])
            max_amplitude = max((float(item["amplitude_relative_error"]) for item in line_fits), default=float("inf"))
            max_phase = max((abs(float(item["phase_error_deg"])) for item in line_fits), default=float("inf"))
            records[load] = {
                "spectrum": {
                    "program_hash": _program_hash(program),
                    "line_count": len(line_fits),
                    "discard_s": discard_s,
                    "fit_window_s": fit_window_s,
                    "fit": fit,
                    "max_line_amplitude_relative_error": max_amplitude,
                    "max_absolute_line_phase_error_deg": max_phase,
                    "failure_reasons": reasons,
                    "trace": audit["trace"],
                    "audit": audit,
                    "options": options,
                    "passed": bool(
                        not reasons
                        and max_amplitude <= float(hard["line_amplitude_relative_error_max"])
                        and max_phase <= float(hard["line_phase_absolute_error_deg_max"])
                        and _warning_count(audit) <= int(hard["warning_count_max"])
                    ),
                },
                "gamma": _run_gamma_batch(candidate, driver["safe_gamma_candidates"], discard_s=discard_s, load_case=load),
            }
        except Exception as exc:  # crash ledger is evidence, never a soft exclusion
            crashes.append({"load_case": load, "exception": f"{type(exc).__name__}: {exc}"})
    gamma_passed = bool(records) and all(
        row["passed"]
        for load_record in records.values()
        for row in load_record["gamma"].values()
    )
    spectrum_passed = len(records) == len(driver["load_cases"]) and all(
        row["spectrum"]["passed"] for row in records.values()
    )
    return {
        "candidate": copy.deepcopy(dict(candidate)),
        "status": "crashed" if crashes else "measured",
        "matrix": records,
        "crash_ledger": crashes,
        "complete_matrix": len(records) == len(driver["load_cases"]),
        "passed": bool(not crashes and spectrum_passed and gamma_passed),
        "exclusion_reasons": [] if not crashes and spectrum_passed and gamma_passed else ["driver_complete_matrix_hard_gate_failed"],
    }


def _select_driver(records: list[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    needed = int(protocol["driver"]["hard_gates"]["minimum_convergence_passes"])
    eligible = [record for record in records if record.get("passed") is True]
    if len(eligible) < needed:
        raise PhysicsSelectionV2Error("blocked_driver_requires_three_complete_dt_passes")
    eligible.sort(
        key=lambda item: (
            -float(item["candidate"]["physics_timestep_s"]),
            max(
                float(load["spectrum"]["max_absolute_line_phase_error_deg"])
                for load in item["matrix"].values()
            ),
            str(item["candidate"]["candidate_id"]),
        )
    )
    return eligible[0]


def _candidate_profile(driver: Mapping[str, Any], isolator: Mapping[str, Any], contact: Mapping[str, Any]) -> PhysicsProfile:
    """Build a non-scoreable probe profile exclusively from V2 candidate values."""

    payload = make_probe_physics_profile().to_dict()
    parameters = derive_isolator_parameters(
        IsolatorConfig(fn_hz=tuple(isolator["fn_hz"]), zeta=tuple(isolator["zeta"]))
    )
    payload.update(
        {
            "profile_id": "shakebench.phase06r.candidate."
            + str(driver["candidate_id"])
            + "."
            + str(isolator["candidate_id"])
            + "."
            + str(contact["candidate_id"]),
            "status": "phase06r_probe_non_scoreable",
            "scoreable": False,
            "not_scoreable_reason": "Phase 06R candidate probe only; not package official physics",
        }
    )
    physics = payload["physics"]
    physics["timestep"] = {
        "physics_timestep_s": float(driver["physics_timestep_s"]),
        "integrator": str(driver["integrator"]),
        "solver": str(driver["solver"]),
        "iterations": int(driver["solver_iterations"]),
        "tolerance": float(driver["solver_tolerance"]),
    }
    physics["scheduler"]["control_steps"] = int(round(1.0 / (20.0 * float(driver["physics_timestep_s"]))))
    physics["deck"]["mass_kg"] = float(driver["deck_mass_kg"])
    physics["deck"]["inertia_kg_m2"] = list(driver["deck_inertia_kg_m2"])
    physics["deck"]["eq_solref"] = list(driver["deck_eq_solref"])
    physics["deck"]["eq_solimp"] = list(driver["deck_eq_solimp"])
    physics["isolator"].update(
        {
            "fn_hz": list(parameters.fn_hz), "zeta": list(parameters.zeta), "k": list(parameters.stiffness),
            "c": list(parameters.damping), "springref": list(parameters.springref),
        }
    )
    physics["contact"].update(
        {
            "condim": int(contact["condim"]), "torsional_mu": float(contact["torsional_mu"]),
            "rolling_mu": float(contact["rolling_mu"]), "margin_m": float(contact["margin_m"]),
            "gap_m": float(contact["gap_m"]), "solref": list(contact["solref"]), "solimp": list(contact["solimp"]),
            "interfaces": ["can_open_worktable", "can_target_bottom", "can_target_walls", "can_panda_finger_pads"],
        }
    )
    payload["profile_sha256"] = physics_profile_hash(payload)
    profile = PhysicsProfile(payload=payload, source="Phase 06R candidate protocol", profile_sha256=payload["profile_sha256"])
    return profile.assert_valid()


def _isolator_summary(candidate: Mapping[str, Any], combined: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = combined["metrics"]
    parameters = derive_isolator_parameters(IsolatorConfig(fn_hz=tuple(candidate["fn_hz"]), zeta=tuple(candidate["zeta"])))
    payload_records = payload["records"]
    payload_offset = max(
        float(np.linalg.norm(np.asarray(row["measured_joint_equilibrium"], dtype=float))) for row in payload_records
    )
    nominal = np.asarray(payload_records[0]["measured_joint_equilibrium"], dtype=float)
    farthest = np.asarray(payload_records[-1]["measured_joint_equilibrium"], dtype=float)
    return {
        "R_relative_median": float(np.median(np.asarray(metrics["R_relative"], dtype=float))),
        "T_peak": float(metrics["T_peak"]),
        "T_accel_mean": float(np.mean(np.asarray(metrics["T_accel"], dtype=float))),
        "D_relative_m": float(metrics["D_relative_m"]),
        "travel_margin_min_m": float(min(metrics["travel_margin_m"])),
        "static_sag_uncompensated_m": float(static_sag_uncompensated_m(parameters)),
        "static_offset_compensated_payload_max_m": payload_offset,
        "payload_sensitivity": float(np.linalg.norm(farthest - nominal) / max(np.linalg.norm(nominal), 1.0e-12)),
    }


def _isolator_screen(candidate: Mapping[str, Any], driver: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    spec = protocol["isolator"]
    gates = spec["hard_gates"]
    config = IsolatorConfig(fn_hz=tuple(candidate["fn_hz"]), zeta=tuple(candidate["zeta"]))
    thresholds = {
        "amplitude_relative_error_max": float(gates["analytic_mujoco_amplitude_relative_error_max"]),
        "phase_absolute_error_deg_max": float(gates["analytic_mujoco_phase_absolute_error_deg_max"]),
        "fit_normalized_residual_max": 0.05,
        "cross_axis_leakage_relative_max": float(gates["combined_cross_axis_leakage_max"]),
        "payload_relative_error_max": 0.25,
        "payload_absolute_offset_tolerance": 5.0e-5,
    }
    try:
        grid = run_harmonic_transfer_grid(
            isolator_config=config, timestep_s=float(driver["physics_timestep_s"]),
            region_ratios={"tracking": spec["probe_regions"]["tracking_ratio"], "resonance": spec["probe_regions"]["resonance_ratio"], "isolation": spec["probe_regions"]["isolation_ratio"]},
            transient_cycles=int(spec["probe_regions"]["transient_cycles"]), fit_cycles=int(spec["probe_regions"]["fit_cycles"]),
            visual=False, thresholds=thresholds,
        )
        program = spec["combined_spectrum"]
        combined = run_joint_spectrum_probe(
            isolator_config=config, timestep_s=float(driver["physics_timestep_s"]), seed=int(program["seed"]),
            t0_s=float(program["t0_s"]), level_scale=float(program["level_scale"]),
            transient_discard_s=float(program["transient_discard_s"]), visual=False, thresholds=thresholds,
        )
        payload = run_payload_sensitivity_probe(
            isolator_config=config, timestep_s=float(driver["physics_timestep_s"]), payload_mass_kg=float(spec["payload_probe"]["mass_kg"]),
            com_offsets_m=spec["payload_probe"]["com_offsets_m"], settle_duration_s=float(spec["payload_probe"]["settle_duration_s"]),
            equilibrium_window_s=float(spec["payload_probe"]["equilibrium_window_s"]), visual=False, thresholds=thresholds,
        )
        summary = _isolator_summary(candidate, combined, payload)
        checks = {
            "transfer_grid": grid["passed"], "combined_spectrum": combined["gate_summary"]["passed"], "payload": payload["passed"],
            "T_accel": float(gates["T_accel_mean_min"]) <= summary["T_accel_mean"] <= float(gates["T_accel_mean_max"]),
            "T_peak": summary["T_peak"] <= float(gates["T_peak_max"]),
            "travel": summary["travel_margin_min_m"] >= float(gates["travel_margin_min_m"]),
            "sag": summary["static_sag_uncompensated_m"] <= float(gates["static_sag_uncompensated_max_m"]),
            "payload_offset": summary["static_offset_compensated_payload_max_m"] <= float(gates["static_offset_compensated_payload_max_m"]),
            "payload_sensitivity": summary["payload_sensitivity"] <= float(gates["payload_sensitivity_max"]),
        }
        return {"candidate": copy.deepcopy(dict(candidate)), "status": "measured", "transfer_grid": grid, "combined_spectrum": combined,
                "payload": payload, "summary": summary, "hard_gates": checks, "passed": bool(all(checks.values())),
                "exclusion_reasons": [] if all(checks.values()) else ["isolator_hard_gate_failed"]}
    except Exception as exc:
        return {"candidate": copy.deepcopy(dict(candidate)), "status": "crashed", "passed": False,
                "crash_ledger": [{"exception": f"{type(exc).__name__}: {exc}"}], "exclusion_reasons": ["isolator_probe_crash"]}


def _normalized_margin(summary: Mapping[str, Any], gates: Mapping[str, Any]) -> float:
    return min(
        float(summary["T_peak"]) and float(gates["T_peak_max"]) / float(summary["T_peak"]),
        float(summary["travel_margin_min_m"]) / float(gates["travel_margin_min_m"]),
        float(gates["static_sag_uncompensated_max_m"]) / max(float(summary["static_sag_uncompensated_m"]), 1e-12),
        float(gates["payload_sensitivity_max"]) / max(float(summary["payload_sensitivity"]), 1e-12),
    )


def _isolator_score(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[float, float, str]:
    targets = protocol["isolator"]["target_values"]
    weights = protocol["isolator"]["scoring"]["weights"]
    summary = record["summary"]
    distance = sum(
        float(weights[key]) * abs(float(summary[key]) - float(targets[key])) / max(abs(float(targets[key])), 1e-12)
        for key in weights
    )
    return float(distance), -_normalized_margin(summary, protocol["isolator"]["hard_gates"]), str(record["candidate"]["candidate_id"])


def _select_isolator(records: list[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    eligible = [record for record in records if record.get("passed") is True]
    if not eligible:
        raise PhysicsSelectionV2Error("blocked_no_isolator_candidate")
    return sorted(eligible, key=lambda item: _isolator_score(item, protocol))[0]


def _contact_screen(candidate: Mapping[str, Any], driver: Mapping[str, Any], isolator: Mapping[str, Any]) -> dict[str, Any]:
    """Run direct-MuJoCo candidate contact evidence; no env.step/reward/success calls."""

    profile = _candidate_profile(driver, isolator, candidate)
    try:
        evidence = _contact_candidate_probe(profile, candidate, run_expensive=True)
    except Exception as exc:
        return {"candidate": copy.deepcopy(dict(candidate)), "status": "crashed", "passed": False,
                "crash_ledger": [{"exception": f"{type(exc).__name__}: {exc}"}], "exclusion_reasons": ["contact_probe_crash"]}
    # The underlying probe is raw MuJoCo only.  It records pair equality, static support, slip, impact/recovery,
    # finger load, and three-dt convergence.  An analytic-only incline entry is intentionally not accepted here.
    actual_incline = evidence.get("incline_threshold", {}).get("actual_mujoco", {})
    required = {
        "static_support": evidence.get("static_support", {}).get("passed") is True,
        "actual_incline": actual_incline.get("passed") is True,
        "slip": evidence.get("single_axis_slip", {}).get("passed") is True,
        "impact_recovery": evidence.get("impact_recovery", {}).get("passed") is True,
        "finger_load": evidence.get("finger_load", {}).get("passed") is True,
        "timestep_convergence": evidence.get("timestep_convergence", {}).get("passed") is True,
        "warnings": evidence.get("warning_count") == 0,
    }
    return {"candidate": copy.deepcopy(dict(candidate)), "status": "measured", "physics": evidence, "hard_gates": required,
            "passed": bool(all(required.values())), "exclusion_reasons": [] if all(required.values()) else ["contact_hard_gate_failed"]}


def _contact_score(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[float, float, str]:
    physics = record["physics"]
    hard = protocol["contact"]["hard_gates"]
    weights = protocol["contact"]["scoring"]["weights"]
    incline = physics["incline_threshold"]["actual_mujoco"]["threshold_relative_error"]
    slip = physics["single_axis_slip"].get("threshold_relative_error", 0.0)
    impact = physics["impact_recovery"]["maximum_penetration_m"]
    finger = physics["finger_load"].get("relative_force_error", 0.0)
    convergence = max(row["relative_force_error"] for row in physics["timestep_convergence"]["records"])
    values = {"incline_threshold_relative_error": incline, "slip_threshold_relative_error": slip,
              "maximum_penetration_m": impact, "recovery_velocity_m_s": physics["impact_recovery"].get("recovery_velocity_m_s", 0.0),
              "finger_force_relative_error": finger, "timestep_trace_relative_error": convergence}
    limits = {"incline_threshold_relative_error": hard["incline_threshold_relative_error_max"], "slip_threshold_relative_error": hard["slip_threshold_relative_error_max"],
              "maximum_penetration_m": hard["maximum_illegal_penetration_m"], "recovery_velocity_m_s": hard["recovery_velocity_max_m_s"],
              "finger_force_relative_error": 1.0, "timestep_trace_relative_error": hard["timestep_trace_relative_error_max"]}
    return (sum(float(weights[key]) * float(values[key]) / max(float(limits[key]), 1e-12) for key in weights),
            float(impact), str(record["candidate"]["candidate_id"]))


def _select_contact(records: list[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    eligible = [record for record in records if record.get("passed") is True]
    if not eligible:
        raise PhysicsSelectionV2Error("blocked_no_contact_candidate")
    return sorted(eligible, key=lambda item: _contact_score(item, protocol))[0]


def _profile_payload(driver: Mapping[str, Any], isolator: Mapping[str, Any], contact: Mapping[str, Any], protocol_hash: str) -> dict[str, Any]:
    profile = _candidate_profile(driver, isolator, contact).to_dict()
    profile.update({"profile_id": "shakebench.official.physics.v1", "status": "official_immutable", "scoreable": True,
                    "selection_basis": "phase06r_physics_only", "protocol_file": PROTOCOL_FILENAME, "protocol_sha256": protocol_hash,
                    "freeze_commit": "phase_06_review_remediation"})
    profile["profile_sha256"] = physics_profile_hash(profile)
    return profile


def _write_raw(output: Path, prefix: str, stage: str, records: list[Mapping[str, Any]], protocol_hash: str) -> list[dict[str, str]]:
    rows = []
    for record in records:
        candidate_id = str(record["candidate"]["candidate_id"])
        path = output / f"{prefix}{stage}_{candidate_id}.json"
        payload = {"schema_id": SCHEMA_ID + ".raw", "schema_version": SCHEMA_VERSION, "stage": stage,
                   "candidate_id": candidate_id, "protocol_sha256": protocol_hash, "physics_only": True, "evidence": record}
        _write_json(path, payload)
        rows.append({"stage": stage, "candidate_id": candidate_id, "path": path.name, "sha256": _file_sha(path)})
    return rows


def run_selection(*, output_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Run the V2 sequence.  Any incomplete group leaves status BLOCKED."""

    protocol, protocol_path, protocol_hash = _load_protocol()
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    prefix = str(protocol["artifact_contract"]["raw_prefix"])
    driver_records = [_driver_screen(_driver_candidate(row), protocol) for row in protocol["driver"]["convergence_candidates"]]
    try:
        selected_driver = _select_driver(driver_records, protocol)
        isolator_records = [_isolator_screen(row, selected_driver["candidate"], protocol) for row in protocol["isolator"]["candidates"]]
        selected_isolator = _select_isolator(isolator_records, protocol)
        contact_records = [_contact_screen(row, selected_driver["candidate"], selected_isolator["candidate"]) for row in protocol["contact"]["candidates"]]
        selected_contact = _select_contact(contact_records, protocol)
    except PhysicsSelectionV2Error:
        raw = _write_raw(output, prefix, "driver", driver_records, protocol_hash)
        status = {"schema_id": "shakebench.phase06r.selection_status", "schema_version": 1, "status": "BLOCKED",
                  "reason": "required V2 candidate group did not pass", "protocol": {"path": protocol_path, "sha256": protocol_hash}, "raw_files": raw}
        _write_json(output / STATUS_FILENAME, status)
        raise
    raw = []
    raw += _write_raw(output, prefix, "driver", driver_records, protocol_hash)
    raw += _write_raw(output, prefix, "isolator", isolator_records, protocol_hash)
    raw += _write_raw(output, prefix, "contact", contact_records, protocol_hash)
    profile = _profile_payload(selected_driver["candidate"], selected_isolator["candidate"], selected_contact["candidate"], protocol_hash)
    selection = {"selected_candidate_count": 1, "physics_only": True, "task_success_used": False, "controller_outcome_used": False,
                 "timestep_solver": selected_driver["candidate"], "isolator": selected_isolator["candidate"], "contact": selected_contact["candidate"]}
    selected = {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "status": "PENDING_PARITY_AND_DETERMINISM",
                "protocol": {"path": protocol_path, "sha256": protocol_hash}, "selection": selection,
                "raw_files": raw, "candidate_counts": {"driver": len(driver_records), "isolator": len(isolator_records), "contact": len(contact_records)},
                "profile": profile}
    selected["payload_sha256"] = _sha(selected)
    _write_json(output / str(protocol["artifact_contract"]["selected_table"]), selected)
    excluded = {"schema_id": SCHEMA_ID + ".excluded", "schema_version": SCHEMA_VERSION, "status": "PENDING_PARITY_AND_DETERMINISM",
                "selected_candidate_ids": {name: selection[name]["candidate_id"] for name in ("timestep_solver", "isolator", "contact")},
                "excluded": [{"stage": stage, "candidate_id": row["candidate"]["candidate_id"], "reasons": row.get("exclusion_reasons", []) or ["lower_priority_after_hard_gates"]}
                             for stage, rows in (("driver", driver_records), ("isolator", isolator_records), ("contact", contact_records)) for row in rows if row.get("passed") is not True],
                "physics_only": True}
    excluded["payload_sha256"] = _sha(excluded)
    _write_json(output / str(protocol["artifact_contract"]["excluded_table"]), excluded)
    return {"selected": selected, "excluded": excluded}


def verify_selection_artifact(path: str | Path) -> dict[str, Any]:
    """Fail closed: hash every raw input and reject non-final / self-asserted PASS artifacts."""
    selected_path = Path(path)
    errors: list[str] = []
    try:
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    content = dict(payload); stored = content.pop("payload_sha256", None)
    if stored != _sha(content): errors.append("selected payload hash mismatch")
    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") != SCHEMA_VERSION: errors.append("wrong schema")
    if payload.get("status") != "PASS": errors.append("selection is not final PASS")
    for raw in payload.get("raw_files", []):
        raw_path = selected_path.parent / str(raw.get("path", ""))
        if not raw_path.is_file() or _file_sha(raw_path) != raw.get("sha256"): errors.append("raw artifact hash mismatch: " + str(raw.get("path")))
    return {"passed": not errors, "errors": errors, "path": str(selected_path)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verify", default=None)
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_selection_artifact(args.verify)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    try:
        result = run_selection(output_dir=args.output_dir)
    except PhysicsSelectionV2Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"status": result["selected"]["status"], "selection": result["selected"]["selection"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
