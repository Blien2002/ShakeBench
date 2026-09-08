"""Atomic, resumable two-worker CPU batch runner for ShakeBench.

This execution layer never changes science inputs. Classic MuJoCo remains the
only scoreable backend; process identity, affinity, timing and retries are kept
outside the v5 Oracle artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import queue
import resource
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AFFINITIES = {1: (0,), 2: (0, 2)}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
SCHEMA_ID = "shakebench.phase07_5b.cpu_batch"
SCHEMA_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                result[fields[0].rstrip(":")] = int(fields[1])
    except (OSError, ValueError):
        return {}
    return result


def resource_snapshot() -> dict[str, Any]:
    meminfo = _read_key_values(Path("/proc/meminfo"))
    vmstat = _read_key_values(Path("/proc/vmstat"))
    temperatures = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            temperatures.append({"path": str(path), "millideg_c": int(path.read_text().strip())})
        except (OSError, ValueError):
            continue
    frequencies = []
    for cpu in AFFINITIES[2]:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        try:
            frequencies.append({"cpu": cpu, "khz": int(path.read_text().strip())})
        except (OSError, ValueError):
            continue
    return {
        "timestamp_s": time.time(),
        "mem_available_kib": meminfo.get("MemAvailable"),
        "swap_total_kib": meminfo.get("SwapTotal"),
        "swap_free_kib": meminfo.get("SwapFree"),
        "vmstat": {key: vmstat.get(key) for key in ("pswpin", "pswpout", "pgmajfault", "oom_kill")},
        "temperatures": temperatures,
        "frequencies": frequencies,
    }


def _state_digest(state: Mapping[str, Any]) -> str:
    return _hash(state)


def build_jobs(
    states: list[dict[str, Any]], *, workers: int, horizon_steps: int, science_authority_sha256: str
) -> list[dict[str, Any]]:
    jobs = []
    for index, (state, cpu) in enumerate(zip(states[:workers], AFFINITIES[workers], strict=True)):
        science = {
            "state_id": state["state_id"],
            "state_sha256": _state_digest(state),
            "tier": "V0",
            "gamma_commanded": 0.0,
            "horizon_steps": horizon_steps,
            "geometry_profile": "direct_mount_v1",
            "science_authority_sha256": science_authority_sha256,
        }
        jobs.append(
            {
                "job_index": index,
                "job_id": _hash(science),
                "science": science,
                "execution": {"cpu": cpu, "worker_count": workers},
            }
        )
    return jobs


def _resume_record(job: Mapping[str, Any], output: Path, record_path: Path) -> dict[str, Any] | None:
    if not output.is_file() or not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        from robosuite.scripts.shakebench_run_oracle import verify_run_artifact

        verdict = verify_run_artifact(output)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if record.get("job_id") != job["job_id"] or not verdict["passed"]:
        return None
    return {**record, "resumed": True}


def _run_job(job: Mapping[str, Any], output_dir: Path, resume: bool) -> dict[str, Any]:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    cpu = int(job["execution"]["cpu"])
    os.sched_setaffinity(0, {cpu})
    output = output_dir / f"job_{job['job_index']:03d}_{job['science']['state_id']}.json"
    record_path = output.with_suffix(".record.json")
    if resume:
        record = _resume_record(job, output, record_path)
        if record is not None:
            return record
    from robosuite.scripts.shakebench_run_oracle import main as oracle_main
    from robosuite.scripts.shakebench_run_oracle import verify_run_artifact

    retry_ledger = []
    for attempt in (1, 2):
        attempt_path = output.with_name(output.name + f".attempt{attempt}.tmp.json")
        started = time.monotonic()
        try:
            exit_code = oracle_main(
                [
                    "--tier",
                    str(job["science"]["tier"]),
                    "--gamma",
                    str(job["science"]["gamma_commanded"]),
                    "--geometry-profile",
                    str(job["science"]["geometry_profile"]),
                    "--state-id",
                    str(job["science"]["state_id"]),
                    "--horizon-steps",
                    str(job["science"]["horizon_steps"]),
                    "--output",
                    str(attempt_path),
                ]
            )
            verdict = verify_run_artifact(attempt_path)
            if exit_code != 0 or not verdict["passed"]:
                raise RuntimeError(f"oracle artifact failed: exit={exit_code} errors={verdict['errors']}")
            attempt_path.replace(output)
            duration = time.monotonic() - started
            record = {
                "schema_id": SCHEMA_ID + ".job",
                "schema_version": SCHEMA_VERSION,
                "job_id": job["job_id"],
                "job_index": job["job_index"],
                "science": job["science"],
                "execution": {
                    **job["execution"],
                    "pid": os.getpid(),
                    "attempt": attempt,
                    "duration_s": duration,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                },
                "output": str(output),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "semantic_passed": True,
                "retry_ledger": retry_ledger,
                "resumed": False,
            }
            record["payload_sha256"] = _hash(record)
            _write_atomic(record_path, record)
            return record
        except Exception as exc:  # infrastructure failures are recorded and retried once
            retry_ledger.append(
                {
                    "attempt": attempt,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "science_identity_unchanged": True,
                }
            )
    return {
        "schema_id": SCHEMA_ID + ".job",
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "job_index": job["job_index"],
        "science": job["science"],
        "execution": job["execution"],
        "semantic_passed": False,
        "retry_ledger": retry_ledger,
        "resumed": False,
    }


def _worker(job: Mapping[str, Any], output_dir: str, resume: bool, result_queue: Any) -> None:
    result_queue.put(_run_job(job, Path(output_dir), resume))


def run_batch(output_dir: Path, *, workers: int = 2, horizon_steps: int = 1200, resume: bool = False) -> dict[str, Any]:
    if workers not in AFFINITIES:
        raise ValueError("workers must be 1 or 2")
    from robosuite import models
    from robosuite.utils.shakebench_authority import authority_payload_hash, verify_direct_mount_authority

    state_asset = json.loads((Path(models.assets_root) / "shakebench_states_dev.json").read_text(encoding="utf-8"))
    if not isinstance(state_asset, Mapping) or not isinstance(state_asset.get("states"), list):
        raise ValueError("dev-state asset must contain a states list")
    states = state_asset["states"]
    authority = verify_direct_mount_authority()
    jobs = build_jobs(
        states,
        workers=workers,
        horizon_steps=horizon_steps,
        science_authority_sha256=authority_payload_hash(authority),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "schema_id": SCHEMA_ID + ".dispatch",
        "schema_version": SCHEMA_VERSION,
        "worker_count": workers,
        "affinity": list(AFFINITIES[workers]),
        "thread_environment": THREAD_ENV,
        "jobs": jobs,
    }
    dispatch["payload_sha256"] = _hash(dispatch)
    _write_atomic(output_dir / "dispatch.json", dispatch)
    before = resource_snapshot()
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [context.Process(target=_worker, args=(job, str(output_dir), resume, result_queue)) for job in jobs]
    for process in processes:
        process.start()
    records = []
    for _ in jobs:
        try:
            records.append(result_queue.get(timeout=3600))
        except queue.Empty:
            break
    for process in processes:
        process.join()
    elapsed = time.monotonic() - started
    after = resource_snapshot()
    records.sort(key=lambda item: item["job_index"])
    durations = [float(item["execution"]["duration_s"]) for item in records if item.get("semantic_passed")]
    swap_growth = max(0, int(before.get("swap_free_kib") or 0) - int(after.get("swap_free_kib") or 0))
    passed = len(records) == len(jobs) and all(item.get("semantic_passed") for item in records) and not swap_growth
    result = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else ("RESOURCE_LIMITED" if swap_growth else "FAIL"),
        "worker_count": workers,
        "affinity": list(AFFINITIES[workers]),
        "job_count": len(jobs),
        "completed_count": len(records),
        "elapsed_s": elapsed,
        "jobs_s": len(durations) / elapsed if elapsed else 0.0,
        "p50_s": statistics.median(durations) if durations else None,
        "p95_s": sorted(durations)[-1] if durations else None,
        "resource_before": before,
        "resource_after": after,
        "swap_growth_kib": swap_growth,
        "records": records,
    }
    result["payload_sha256"] = _hash(result)
    _write_atomic(output_dir / "aggregate.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=sorted(AFFINITIES), default=2)
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_batch(
        args.output_dir,
        workers=args.workers,
        horizon_steps=args.horizon_steps,
        resume=args.resume,
    )
    print(
        json.dumps({key: result[key] for key in ("status", "worker_count", "jobs_s", "payload_sha256")}, sort_keys=True)
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AFFINITIES", "build_jobs", "main", "resource_snapshot", "run_batch"]
