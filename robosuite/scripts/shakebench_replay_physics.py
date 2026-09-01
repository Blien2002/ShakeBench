"""Replay a fixed Phase 06 physics state in independent processes.

The worker emits one digest over every field of the complete ``DeckDriverTrace``.
The parent launches at least three fresh interpreters and compares those
digests, field shapes, dtypes, and compiled option snapshots.  A same-process
reset is intentionally never used as the determinism gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_select_physics import (
    _run_driver_trace,
    _trace_digest,
    _trace_shapes,
)
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.shakebench_physics import load_official_physics_profile


REPLAY_SCHEMA_ID = "shakebench.phase06.physics_replay"
REPLAY_SCHEMA_VERSION = 1
REPLAY_FILENAME = "shakebench_phase_06_replay_determinism.json"
TRACE_FIELDS = (
    "integration_target_time_s",
    "sample_target_time_s",
    "integration_application_time_s",
    "sample_application_time_s",
    "sample_time_s",
    "command_pose",
    "actual_pose",
    "command_twist",
    "actual_twist",
    "command_acceleration",
    "actual_acceleration",
    "sample_target_pose",
    "sample_target_twist",
    "sample_target_acceleration",
    "deck_tracking_pose_error",
    "weld_constraint_residual_raw",
    "weld_constraint_force_raw",
    "solver_iterations",
    "solver_niter",
    "warning_number_delta",
    "warning_lastinfo",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _selected_candidate() -> dict[str, Any]:
    profile = load_official_physics_profile()
    return {
        "candidate_id": "official_replay",
        "physics_timestep_s": profile.model_timestep_s,
        "integrator": str(profile.timestep["integrator"]),
        "solver": str(profile.timestep["solver"]),
        "solver_iterations": int(profile.timestep["iterations"]),
        "solver_tolerance": float(profile.timestep["tolerance"]),
        "deck_eq_solref": list(profile.deck_eq_solref),
        "deck_eq_solimp": list(profile.deck_eq_solimp),
        "deck_mass_kg": profile.deck_mass_kg,
        "deck_inertia_kg_m2": list(profile.deck_inertia_kg_m2),
    }


def _worker_record(*, state_id: str) -> dict[str, Any]:
    if state_id != "phase06_zero_and_six_axis_trace":
        raise ValueError("unknown replay state id")
    profile = load_official_physics_profile()
    candidate = _selected_candidate()
    program = build_excitation_program(seed=17, t0=0.137, level_scale=0.30)
    trace, audit, options = _run_driver_trace(
        candidate,
        duration_s=1.25,
        trajectory=program,
        load_case="empty",
        refresh_stride=1,
    )
    return {
        "state_id": state_id,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "program_hash": hashlib.sha256(
            json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "options": options,
        "trace_digest": _trace_digest(trace),
        "trace_shapes": _trace_shapes(trace),
        "trace_dtypes": {name: str(np.asarray(getattr(trace, name)).dtype) for name in TRACE_FIELDS},
        "trace_fields": list(TRACE_FIELDS),
        "trace_summary": audit["trace"]["summary"],
        "warning_count": int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0,
        "complete_trace_compared": True,
    }


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "trace_digest" in value:
            return value
    raise RuntimeError("replay worker did not emit a trace record")


def run_determinism(
    *,
    processes: int = 3,
    state_id: str = "phase06_zero_and_six_axis_trace",
    output_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    if isinstance(processes, bool) or int(processes) != processes or int(processes) < 3:
        raise ValueError("Phase 06 determinism requires at least three independent processes")
    processes = int(processes)
    profile = load_official_physics_profile()
    command = [sys.executable, "-m", "robosuite.scripts.shakebench_replay_physics", "--worker", "--state-id", state_id]
    workers = []
    failures = []
    for index in range(processes):
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if completed.returncode != 0:
            failures.append(
                {
                    "process_index": index,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                    "stdout": completed.stdout[-4000:],
                }
            )
            continue
        try:
            workers.append(_parse_worker_stdout(completed.stdout))
        except Exception as exc:
            failures.append({"process_index": index, "error": str(exc), "stdout": completed.stdout[-4000:]})

    compared = []
    if workers:
        reference = workers[0]
        for index, worker in enumerate(workers):
            compared.append(
                {
                    "process_index": index,
                    "profile_sha256_equal": worker.get("profile_sha256") == reference.get("profile_sha256"),
                    "program_hash_equal": worker.get("program_hash") == reference.get("program_hash"),
                    "options_equal": worker.get("options") == reference.get("options"),
                    "trace_digest_equal": worker.get("trace_digest") == reference.get("trace_digest"),
                    "trace_shapes_equal": worker.get("trace_shapes") == reference.get("trace_shapes"),
                    "trace_dtypes_equal": worker.get("trace_dtypes") == reference.get("trace_dtypes"),
                    "complete_trace_compared": worker.get("complete_trace_compared") is True,
                    "warning_count": worker.get("warning_count"),
                }
            )
    comparison_passed = bool(
        len(workers) == processes
        and not failures
        and len(compared) == processes
        and all(
            all(value for key, value in row.items() if key not in {"process_index", "warning_count"})
            and row["warning_count"] == 0
            for row in compared
        )
    )
    payload: dict[str, Any] = {
        "schema_id": REPLAY_SCHEMA_ID,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "PASS" if comparison_passed else "BLOCKED",
        "state_id": state_id,
        "independent_process_count": processes,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "trace_fields": list(TRACE_FIELDS),
        "workers": workers,
        "failures": failures,
        "comparison": compared,
        "complete_trace_comparison": comparison_passed,
        "same_process_reset_used": False,
        "crash_handling": {
            "worker_crash": "fail_replay",
            "same_state_only_rerun": True,
            "replacement_state_forbidden": True,
        },
    }
    payload["payload_sha256"] = _sha256_json(payload)
    if output_path is not None:
        path = Path(output_path)
    else:
        path = Path(models.assets_root) / REPLAY_FILENAME
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_replay_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    errors = []
    checks = {}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)], "checks": checks}
    checks["schema"] = payload.get("schema_id") == REPLAY_SCHEMA_ID and payload.get("schema_version") == REPLAY_SCHEMA_VERSION
    content = dict(payload)
    stored_hash = content.pop("payload_sha256", None)
    checks["payload_hash"] = stored_hash == _sha256_json(content)
    checks["process_count"] = int(payload.get("independent_process_count", 0)) >= 3
    checks["complete_trace"] = payload.get("complete_trace_comparison") is True
    checks["no_same_process_reset"] = payload.get("same_process_reset_used") is False
    checks["profile_hash"] = payload.get("profile_sha256") == load_official_physics_profile().profile_sha256
    comparisons = payload.get("comparison", [])
    checks["all_worker_comparisons"] = bool(comparisons) and all(
        row.get("trace_digest_equal") is True
        and row.get("trace_shapes_equal") is True
        and row.get("trace_dtypes_equal") is True
        and row.get("options_equal") is True
        and row.get("complete_trace_compared") is True
        and row.get("warning_count") == 0
        for row in comparisons
        if isinstance(row, Mapping)
    )
    if not all(checks.values()):
        errors.append("one or more replay artifact checks failed")
    return {"passed": not errors, "errors": errors, "checks": checks, "path": str(artifact_path)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--state-id", default="phase06_zero_and_six_axis_trace")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify", default=None)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            print(json.dumps(_worker_record(state_id=args.state_id), ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
            return 2
        return 0
    if args.verify:
        result = verify_replay_artifact(args.verify)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    try:
        result = run_determinism(processes=args.processes, state_id=args.state_id, output_path=args.output)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": result["status"], "profile_sha256": result["profile_sha256"], "processes": result["independent_process_count"]}, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
