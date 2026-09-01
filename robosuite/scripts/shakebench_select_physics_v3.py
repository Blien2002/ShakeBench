"""Phase 06R2 V3 physics-only selection runner.

The driver stage is deliberately persisted after every Gamma/load spectrum
record.  A crash is retried only once with the identical state and leaves the
whole group BLOCKED if it recurs.  This runner never resolves an official
physics profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_probe_deck_driver import (
    _program_hash,
    canonical_gamma_conformance,
    minimum_line_spacing_hz,
    spectrum_conformance,
    spectrum_gate_reasons,
)
from robosuite.scripts.shakebench_select_physics import _program_for_gamma, _run_driver_trace, _trace_digest, _trace_shapes
from robosuite.utils.shakebench_driver_measurement import DriverMeasurementPlan
from robosuite.utils.shakebench_physics import physics_profile_hash


SCHEMA_ID = "shakebench.phase06r2.physics_selection"
SCHEMA_VERSION = 3
PROTOCOL_FILENAME = "shakebench_selection_protocol_v3.yaml"


class Phase06R2Blocked(RuntimeError):
    """A required V3 physics group is incomplete or failed its hard gate."""


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(_ready(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def _load_protocol() -> tuple[Mapping[str, Any], str, str, str]:
    path = Path(models.assets_root) / PROTOCOL_FILENAME
    raw = path.read_bytes()
    try:
        import yaml

        payload = yaml.safe_load(raw.decode("utf-8"))
    except ImportError:
        payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") != "pre_registered":
        raise Phase06R2Blocked("V3 protocol is not pre_registered")
    return payload, str(path), hashlib.sha256(raw).hexdigest(), _git_registration_commit()


def _git_registration_commit() -> str:
    # The commit was made before every V3 probe.  Reading HEAD does not feed
    # any physics selection metric and is recorded solely for provenance.
    head = Path(__file__).resolve().parents[2] / ".git" / "HEAD"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        reference = Path(__file__).resolve().parents[2] / ".git" / text[5:]
        return reference.read_text(encoding="utf-8").strip()
    return text


def _driver_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for composed in protocol["composed_candidates"]:
        row = dict(composed["driver"])
        row["composed_candidate_id"] = str(composed["candidate_id"])
        row["solver_iterations"] = int(row.pop("iterations"))
        row["solver_tolerance"] = float(row.pop("tolerance"))
        rows.append(row)
    return rows


def _driver_raw_path(output: Path, candidate_id: str) -> Path:
    return output / f"shakebench_phase_06r2_raw_driver_{candidate_id}.json"


def _persist_driver_record(
    *, output: Path, protocol_path: str, protocol_hash: str, registration_commit: str, candidate: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> None:
    payload = {
        "schema_id": SCHEMA_ID + ".raw_driver",
        "schema_version": SCHEMA_VERSION,
        "stage": "driver",
        "candidate": candidate,
        "protocol": {"path": protocol_path, "sha256": protocol_hash, "registration_commit": registration_commit},
        "physics_only": True,
        "records": records,
        "record_count": len(records),
    }
    payload["payload_sha256"] = _sha(payload)
    _write(_driver_raw_path(output, str(candidate["candidate_id"])), payload)


def _resume_driver_records(
    *, output: Path, candidate: Mapping[str, Any], protocol_hash: str
) -> list[Mapping[str, Any]]:
    """Resume only states already atomically persisted with this V3 protocol.

    A command-host time slice is not a MuJoCo state crash when the previous
    state finished and its raw record has already been committed to disk.
    """

    path = _driver_raw_path(output, str(candidate["candidate_id"]))
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_id") != SCHEMA_ID + ".raw_driver"
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol", {}).get("sha256") != protocol_hash
        or payload.get("candidate", {}).get("candidate_id") != candidate["candidate_id"]
    ):
        raise Phase06R2Blocked("existing driver raw artifact has incompatible provenance")
    content = dict(payload)
    stored = content.pop("payload_sha256", None)
    if stored != _sha(content):
        raise Phase06R2Blocked("existing driver raw artifact hash mismatch")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise Phase06R2Blocked("existing driver raw artifact records are malformed")
    return rows


def _spectrum_record(
    *, candidate: Mapping[str, Any], gamma: float, load_case: str, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    facts = protocol["frozen_facts"]
    measurement = protocol["measurement"]
    stride = int(measurement["refresh_stride_by_driver_candidate"][candidate["candidate_id"]])
    plan = DriverMeasurementPlan(
        physics_timestep_s=float(candidate["physics_timestep_s"]),
        cadence_hz=float(measurement["cadence_by_driver_candidate_hz"][candidate["candidate_id"]]),
        f_max_hz=float(facts["f_max_hz"]),
    )
    if plan.refresh_stride != stride:
        raise Phase06R2Blocked("protocol driver measurement cadence/stride mismatch")
    level_scale = _program_for_gamma(float(gamma), dt=float(candidate["physics_timestep_s"]), duration_s=35.0)
    program = __import__("robosuite.utils.shakebench_excitation", fromlist=["build_excitation_program"]).build_excitation_program(
        seed=int(facts["authored_spectrum"]["seed"]), t0=float(facts["authored_spectrum"]["t0_s"]), level_scale=level_scale
    )
    discard = float(facts["authored_spectrum"]["transient_discard_s"])
    duration = discard + 2.0 / minimum_line_spacing_hz(program) + float(candidate["physics_timestep_s"])
    started = time.perf_counter()
    trace, audit, options = _run_driver_trace(
        candidate, duration_s=duration, trajectory=program, load_case=load_case, refresh_stride=stride
    )
    elapsed = time.perf_counter() - started
    fit = spectrum_conformance(trace, program, discard_s=discard, max_fit_samples=50000)
    gamma_fit = canonical_gamma_conformance(
        trace, program,
        config=__import__("robosuite.utils.shakebench_deck", fromlist=["DeckDriverConfig"]).DeckDriverConfig(
            deck_mass_kg=float(candidate["deck_mass_kg"]), deck_inertia_kg_m2=tuple(candidate["deck_inertia_kg_m2"]),
            eq_solref=tuple(candidate["deck_eq_solref"]), eq_solimp=tuple(candidate["deck_eq_solimp"]),
            physics_timestep_s=float(candidate["physics_timestep_s"]),
        ),
        discard_s=discard,
    )
    gates = protocol["driver"]["hard_gates"]
    reasons = spectrum_gate_reasons(fit, expected_line_count=int(facts["authored_spectrum"]["line_count"]))
    if float(gamma_fit["relative_error_abs"]) > float(gates["gamma_deck_relative_error_max"]):
        reasons.append("deck_gamma_relative_error_exceeds_threshold")
    if int(audit["deck"]["warnings"]) > int(gates["warning_count_max"]):
        reasons.append("warning_count_exceeds_threshold")
    if float(fit["max_weld_constraint_residual_raw"]) > float(gates["unstable_residual_max"]):
        reasons.append("weld_residual_exceeds_threshold")
    return {
        "status": "measured", "gamma": float(gamma), "load_case": load_case, "program_hash": _program_hash(program),
        "level_scale": float(level_scale), "duration_s": float(duration), "elapsed_wall_time_s": float(elapsed),
        "measurement_plan": plan.to_dict(), "mujoco_step_count": int(np.ceil(duration / float(candidate["physics_timestep_s"]))),
        "peak_retained_sample_count": int(trace.sample_time_s.size), "spectrum": fit, "gamma_fit": gamma_fit,
        "trace": {"digest": _trace_digest(trace), "shapes": _trace_shapes(trace),
                  "dtypes": {name: str(np.asarray(getattr(trace, name)).dtype) for name in plan.to_dict()["trace_fields"]}},
        "audit": audit, "options": options, "failure_reasons": reasons, "passed": not reasons,
    }


def run_driver_selection(*, output_dir: Optional[str | Path] = None, candidate_id: Optional[str] = None) -> dict[str, Any]:
    protocol, protocol_path, protocol_hash, registration_commit = _load_protocol()
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    rows = _driver_rows(protocol)
    if candidate_id is not None:
        rows = [row for row in rows if row["candidate_id"] == candidate_id]
        if len(rows) != 1:
            raise ValueError("unknown V3 driver candidate")
    summaries = []
    for candidate in rows:
        records = _resume_driver_records(output=output, candidate=candidate, protocol_hash=protocol_hash)
        completed = {
            (float(row["gamma"]), str(row["load_case"]))
            for row in records
            if row.get("status") == "measured" and row.get("passed") is True
        }
        for gamma in protocol["frozen_facts"]["safe_gamma_candidates"]:
            for load_case in protocol["frozen_facts"]["load_cases"]:
                if (float(gamma), str(load_case)) in completed:
                    continue
                ledger: list[dict[str, Any]] = []
                for attempt in (1, 2):
                    try:
                        record = _spectrum_record(candidate=candidate, gamma=float(gamma), load_case=str(load_case), protocol=protocol)
                        record["retry_ledger"] = ledger
                        records.append(record)
                        _persist_driver_record(output=output, protocol_path=protocol_path, protocol_hash=protocol_hash, registration_commit=registration_commit, candidate=candidate, records=records)
                        break
                    except Exception as exc:
                        ledger.append({"attempt": attempt, "state_config_identical": True, "exception": f"{type(exc).__name__}: {exc}"})
                else:
                    records.append({"status": "incomplete_infrastructure_crash", "gamma": float(gamma), "load_case": str(load_case), "retry_ledger": ledger, "passed": False})
                    _persist_driver_record(output=output, protocol_path=protocol_path, protocol_hash=protocol_hash, registration_commit=registration_commit, candidate=candidate, records=records)
                    raise Phase06R2Blocked("driver_group_incomplete_repeated_same_state_crash")
        summaries.append({"candidate_id": candidate["candidate_id"], "passed": all(row["passed"] for row in records), "record_count": len(records)})
    return {"status": "PASS" if all(item["passed"] for item in summaries) else "BLOCKED", "driver_candidates": summaries,
            "protocol_sha256": protocol_hash, "registration_commit": registration_commit}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--driver-candidate", default=None)
    args = parser.parse_args(argv)
    try:
        result = run_driver_selection(output_dir=args.output_dir, candidate_id=args.driver_candidate)
    except Phase06R2Blocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
