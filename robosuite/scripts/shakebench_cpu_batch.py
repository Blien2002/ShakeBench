"""Atomic, resumable two-worker CPU batch runner for ShakeBench.

This execution layer never changes science inputs. Classic MuJoCo remains the
only scoreable backend; process identity, affinity, timing and retries are kept
outside the v5 Oracle artifact.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import multiprocessing
import os
import queue
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Phase 08's only scoreable execution shape.  Four and eight workers have
# already produced resource-limited evidence; a one-worker run is not a
# substitute for the paired two-worker protocol validation workload.
AFFINITIES = {2: (0, 2)}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
SCHEMA_ID = "shakebench.phase07_5b.cpu_batch"
SCHEMA_VERSION = 1
RETRY_LEDGER_SCHEMA_ID = SCHEMA_ID + ".retry_ledger"
RECOVERABLE_ERRNOS = frozenset((errno.EAGAIN, errno.EINTR, errno.EIO, errno.ENFILE, errno.EMFILE, errno.ETIMEDOUT))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def job_record_payload_hash(record: Mapping[str, Any]) -> str:
    """Return the canonical hash of a job record without its self-check.

    Args:
        record: Job record containing science and execution provenance.

    Returns:
        SHA-256 of the record after removing ``payload_sha256``.
    """

    content = dict(record)
    content.pop("payload_sha256", None)
    return _hash(content)


def aggregate_payload_hash(aggregate: Mapping[str, Any]) -> str:
    """Return the canonical hash of a batch aggregate without its self-check.

    Args:
        aggregate: Batch aggregate artifact.

    Returns:
        SHA-256 of aggregate content excluding ``payload_sha256``.
    """

    content = dict(aggregate)
    content.pop("payload_sha256", None)
    return _hash(content)


def verify_batch_aggregate(path: str | Path) -> dict[str, Any]:
    """Verify an aggregate before it is admitted to a production scorecard.

    Args:
        path: Aggregate JSON artifact path.

    Returns:
        Verdict with the authenticated aggregate payload when successful.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"aggregate read: {exc}"]}
    errors = []
    if not isinstance(payload, Mapping):
        return {"passed": False, "errors": ["aggregate object"]}
    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("aggregate schema")
    try:
        if payload.get("payload_sha256") != aggregate_payload_hash(payload):
            errors.append("aggregate payload hash")
    except (TypeError, ValueError):
        errors.append("aggregate payload hash")
    for field in (
        "status",
        "worker_count",
        "affinity",
        "job_count",
        "completed_count",
        "records",
        "resource_before",
        "resource_after",
        "incomplete_groups",
    ):
        if field not in payload:
            errors.append(f"aggregate {field}")
    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("aggregate records")
    else:
        try:
            if int(payload.get("job_count")) <= 0 or int(payload.get("completed_count")) != len(records):
                errors.append("aggregate completion counts")
        except (TypeError, ValueError):
            errors.append("aggregate completion counts")
        incomplete = payload.get("incomplete_groups")
        if not isinstance(incomplete, list) or any(not isinstance(item, str) for item in incomplete):
            errors.append("aggregate incomplete groups")
            incomplete_ids = set()
        else:
            incomplete_ids = set(incomplete)
        seen_job_ids = set()
        seen_indexes = set()
        for index, record in enumerate(records):
            prefix = f"aggregate record[{index}]"
            required = {
                "schema_id",
                "schema_version",
                "job_id",
                "job_index",
                "science",
                "execution",
                "semantic_passed",
                "retry_ledger",
                "resumed",
                "payload_sha256",
            }
            if not isinstance(record, Mapping) or not required.issubset(record):
                errors.append(prefix + " schema")
                continue
            if (
                record.get("schema_id") != SCHEMA_ID + ".job"
                or record.get("schema_version") != SCHEMA_VERSION
                or not isinstance(record.get("job_id"), str)
                or not isinstance(record.get("job_index"), int)
                or not isinstance(record.get("science"), Mapping)
                or not isinstance(record.get("execution"), Mapping)
                or not isinstance(record.get("semantic_passed"), bool)
                or record.get("payload_sha256") != job_record_payload_hash(record)
            ):
                errors.append(prefix + " identity")
            if record.get("job_id") in seen_job_ids or record.get("job_index") in seen_indexes:
                errors.append(prefix + " duplicate")
            seen_job_ids.add(record.get("job_id"))
            seen_indexes.add(record.get("job_index"))
            ledger = record.get("retry_ledger")
            if (
                not isinstance(ledger, Mapping)
                or ledger.get("schema_id") != RETRY_LEDGER_SCHEMA_ID
                or ledger.get("schema_version") != SCHEMA_VERSION
                or ledger.get("job_id") != record.get("job_id")
                or ledger.get("science") != record.get("science")
                or not isinstance(ledger.get("events"), list)
                or ledger.get("payload_sha256") != _hash({key: value for key, value in ledger.items() if key != "payload_sha256"})
            ):
                errors.append(prefix + " retry ledger")
            if record.get("semantic_passed") is True:
                output = record.get("output")
                if (
                    not isinstance(output, str)
                    or not Path(output).is_file()
                    or record.get("output_sha256") != hashlib.sha256(Path(output).read_bytes()).hexdigest()
                ):
                    errors.append(prefix + " output")
            else:
                group_status = record.get("group_status")
                if group_status not in {"failed", "incomplete"}:
                    errors.append(prefix + " group status")
                elif group_status == "incomplete" and record.get("job_id") not in incomplete_ids:
                    errors.append(prefix + " incomplete binding")
        if incomplete_ids != {
            record.get("job_id")
            for record in records
            if isinstance(record, Mapping) and record.get("group_status") == "incomplete"
        }:
            errors.append("aggregate incomplete groups")
    return {"passed": not errors, "errors": errors, "payload": dict(payload)}


def mark_job_record_resumed(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return an authenticated record with fresh resume provenance.

    Args:
        record: Previously authenticated job record.

    Returns:
        A copy marked as resumed with a newly computed payload hash.
    """

    resumed = {**record, "resumed": True}
    resumed["payload_sha256"] = job_record_payload_hash(resumed)
    return resumed


def classify_execution_exception(exc: BaseException) -> str:
    """Classify a failure without promoting science/contract errors to retries.

    Args:
        exc: Exception raised while running one immutable job.

    Returns:
        ``"infrastructure"`` only for explicitly recoverable system errors;
        otherwise ``"contract"``.
    """

    if isinstance(exc, (EOFError, BrokenPipeError, MemoryError)):
        return "infrastructure"
    if isinstance(exc, OSError) and exc.errno in RECOVERABLE_ERRNOS:
        return "infrastructure"
    return "contract"


def artifact_requires_exact_retry(payload: Mapping[str, Any]) -> bool:
    """Return whether a semantically valid artifact contains invalid execution.

    Invalid execution is neither a task failure nor a successful completed job:
    the exact immutable job must be attempted once more before the state block
    can be admitted to a scorecard.
    """

    episodes = payload.get("episodes")
    return bool(
        isinstance(episodes, list)
        and any(
            isinstance(episode, Mapping)
            and episode.get("episode_validity") == "invalid"
            and episode.get("termination_cause") == "invalid_execution"
            for episode in episodes
        )
    )


def build_retry_ledger(job: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build an authenticated retry ledger for one immutable job.

    Args:
        job: Dispatch job whose science identity is being retried.
        events: Ordered execution-failure events.

    Returns:
        Ledger with schema, science identity, and payload hash.
    """

    ledger = {
        "schema_id": RETRY_LEDGER_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "science": job["science"],
        "events": [dict(event) for event in events],
    }
    ledger["payload_sha256"] = _hash(ledger)
    return ledger


def _write_retry_ledger(path: Path, job: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    ledger = build_retry_ledger(job, events)
    _write_atomic(path, ledger)
    return ledger


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(values: list[float], quantile: float) -> float | None:
    """Deterministic R-7 linear percentile; this is not a maximum shortcut."""

    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _thread_environment_child(result_queue: Any) -> None:
    """Stdlib-only spawn probe used to guard the pre-import environment rule."""

    result_queue.put({key: os.environ.get(key) for key in THREAD_ENV})


def thread_environment_spawn_probe() -> dict[str, str | None]:
    """Prove a fresh spawn child inherits numerical thread limits at startup.

    This intentionally imports no numerical or physics dependency. It is a
    contract test hook, not a benchmark execution path.
    """

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    previous = {key: os.environ.get(key) for key in THREAD_ENV}
    try:
        os.environ.update(THREAD_ENV)
        child = context.Process(target=_thread_environment_child, args=(result_queue,))
        child.start()
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
    observed = result_queue.get(timeout=10)
    child.join(timeout=10)
    if child.exitcode != 0:
        raise RuntimeError("thread-environment spawn probe failed")
    return observed


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


def telemetry_is_complete(
    before: Mapping[str, Any], after: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> bool:
    """Return whether a batch has all telemetry required for a complete PASS.

    Args:
        before: Resource snapshot collected before worker startup.
        after: Resource snapshot collected after worker completion.
        records: Completed job records with execution provenance.

    Returns:
        True only when memory, VM, frequency, temperature, and per-job RSS are
        all available.
    """

    required_snapshot = ("mem_available_kib", "swap_total_kib", "swap_free_kib", "vmstat")
    required_vmstat = ("pswpin", "pswpout", "pgmajfault", "oom_kill")
    snapshots = (before, after)
    if any(any(snapshot.get(key) is None for key in required_snapshot) for snapshot in snapshots):
        return False
    if any(
        not isinstance(snapshot.get("vmstat"), Mapping)
        or any(snapshot["vmstat"].get(key) is None for key in required_vmstat)
        or not snapshot.get("temperatures")
        or not snapshot.get("frequencies")
        for snapshot in snapshots
    ):
        return False
    return all(
        isinstance(record.get("execution"), Mapping) and record["execution"].get("max_rss_kib") is not None
        for record in records
    )


def _state_digest(state: Mapping[str, Any]) -> str:
    return _hash(state)


def build_jobs(
    states: list[dict[str, Any]],
    *,
    workers: int,
    horizon_steps: int,
    science_authority_sha256: str,
    tier: str = "V0",
    gamma_commanded: float = 0.0,
    state_asset_authority: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build immutable science identities and separate execution assignment.

    Args:
        states: Verified committed state records.
        workers: Required worker count.
        horizon_steps: Episode horizon.
        science_authority_sha256: Frozen Phase 7.5A authority hash.
        tier: Observation tier.
        gamma_commanded: Frozen commanded Gamma.
        state_asset_authority: Verified source-asset binding.

    Returns:
        Deterministic dispatch job records.
    """
    jobs = []
    for index, state in enumerate(states):
        cpu = AFFINITIES[workers][index % workers]
        science = {
            "state_id": state["state_id"],
            "state_sha256": _state_digest(state),
            "tier": tier,
            "gamma_commanded": gamma_commanded,
            "horizon_steps": horizon_steps,
            "geometry_profile": "direct_mount_v1",
            "science_authority_sha256": science_authority_sha256,
            "state_authority_hashes": dict(state.get("authority_hashes", {})),
            "state_canonical_payload_sha256": state.get("canonical_payload_sha256"),
            "state_asset_authority": dict(state_asset_authority or {}),
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
        output_payload = json.loads(output.read_text(encoding="utf-8"))
        from robosuite.scripts.shakebench_run_oracle import verify_run_artifact

        verdict = verify_run_artifact(output)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    supplied_hash = record.get("payload_sha256")
    if (
        record.get("schema_id") != SCHEMA_ID + ".job"
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("job_id") != job["job_id"]
        or record.get("science") != job["science"]
        or supplied_hash != job_record_payload_hash(record)
        or record.get("output_sha256") != hashlib.sha256(output.read_bytes()).hexdigest()
        or record.get("semantic_passed") is not True
        or not verdict["passed"]
        or artifact_requires_exact_retry(output_payload)
    ):
        return None
    resumed = mark_job_record_resumed(record)
    _write_atomic(record_path, resumed)
    return resumed


def _run_job(job: Mapping[str, Any], output_dir: Path, resume: bool, state_asset: str) -> dict[str, Any]:
    cpu = int(job["execution"]["cpu"])
    os.sched_setaffinity(0, {cpu})
    output = output_dir / f"job_{job['job_index']:03d}_{job['science']['state_id']}.json"
    record_path = output.with_suffix(".record.json")
    retry_ledger_path = output.with_suffix(".retry.json")
    if output.exists() or record_path.exists():
        conflict = {
            "schema_id": SCHEMA_ID + ".conflict",
            "schema_version": SCHEMA_VERSION,
            "job_id": job["job_id"],
            "science": job["science"],
            "existing_output": output.exists(),
            "existing_record": record_path.exists(),
        }
        conflict_path = output.with_suffix(".conflict.json")

        def fail_conflict(reason: str) -> dict[str, Any]:
            conflict["reason"] = reason
            conflict["payload_sha256"] = _hash(conflict)
            _write_atomic(conflict_path, conflict)
            failed = {
                "schema_id": SCHEMA_ID + ".job",
                "schema_version": SCHEMA_VERSION,
                "job_id": job["job_id"],
                "job_index": job["job_index"],
                "science": job["science"],
                "execution": job["execution"],
                "semantic_passed": False,
                "retry_ledger": build_retry_ledger(
                    job, [{"classification": "duplicate_conflict", "conflict_ledger": str(conflict_path)}]
                ),
                "resumed": False,
            }
            failed["payload_sha256"] = _hash(failed)
            return failed

        if not resume:
            return fail_conflict("existing result requires authenticated resume")
    if resume:
        record = _resume_record(job, output, record_path)
        if record is not None:
            return record
        if output.exists() or record_path.exists():
            return fail_conflict("existing result failed identity, hash, or semantic resume checks")
    from robosuite.scripts.shakebench_run_oracle import main as oracle_main
    from robosuite.scripts.shakebench_run_oracle import verify_run_artifact

    retry_events: list[Mapping[str, Any]] = []
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
                    "--states",
                    state_asset,
                    "--output",
                    str(attempt_path),
                ]
            )
            verdict = verify_run_artifact(attempt_path)
            if exit_code != 0 or not verdict["passed"]:
                raise RuntimeError(f"oracle artifact failed: exit={exit_code} errors={verdict['errors']}")
            artifact = json.loads(attempt_path.read_text(encoding="utf-8"))
            if artifact_requires_exact_retry(artifact):
                retry_events.append(
                    {
                        "attempt": attempt,
                        "classification": "invalid_execution",
                        "science_identity_unchanged": True,
                    }
                )
                _write_retry_ledger(retry_ledger_path, job, retry_events)
                if attempt == 1:
                    continue
                break
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
                    "simulated_seconds": verdict["trace_count"] / 20.0,
                },
                "output": str(output),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "semantic_passed": True,
                "retry_ledger": _write_retry_ledger(retry_ledger_path, job, retry_events),
                "resumed": False,
            }
            record["payload_sha256"] = job_record_payload_hash(record)
            _write_atomic(record_path, record)
            return record
        except Exception as exc:
            classification = classify_execution_exception(exc)
            event = {
                "attempt": attempt,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "classification": classification,
                "science_identity_unchanged": True,
            }
            retry_events.append(event)
            _write_retry_ledger(retry_ledger_path, job, retry_events)
            if classification == "infrastructure" and attempt == 1:
                continue
            # Task/controller/policy/physics and contract failures are valid
            # denominator outcomes or fail-closed errors, never retry fodder.
            break
    failed = {
        "schema_id": SCHEMA_ID + ".job",
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "job_index": job["job_index"],
        "science": job["science"],
        "execution": job["execution"],
        "semantic_passed": False,
        "retry_ledger": _write_retry_ledger(retry_ledger_path, job, retry_events),
        "group_status": (
            "incomplete"
            if len(retry_events) == 2 and all(event["classification"] == "infrastructure" for event in retry_events)
            else "failed"
        ),
        "resumed": False,
    }
    failed["payload_sha256"] = job_record_payload_hash(failed)
    _write_atomic(record_path, failed)
    return failed


def _worker(jobs: list[Mapping[str, Any]], output_dir: str, resume: bool, state_asset: str, result_queue: Any) -> None:
    """One spawned, CPU-pinned worker consumes only its assigned job stream."""

    if jobs:
        os.sched_setaffinity(0, {int(jobs[0]["execution"]["cpu"])})
    for job in jobs:
        result_queue.put(_run_job(job, Path(output_dir), resume, state_asset))


def run_batch(
    output_dir: Path,
    *,
    workers: int = 2,
    horizon_steps: int = 1200,
    resume: bool = False,
    state_asset: Path | None = None,
    tier: str = "V0",
    gamma_commanded: float = 0.0,
) -> dict[str, Any]:
    """Run an authenticated committed-state batch using exactly two workers.

    Args:
        output_dir: Destination for dispatch, outputs, ledgers, and aggregate.
        workers: Required worker count.
        horizon_steps: Policy horizon for each job.
        resume: Whether authenticated records may be resumed.
        state_asset: Optional official or knee committed-state asset.
        tier: Observation tier.
        gamma_commanded: Frozen commanded Gamma.

    Returns:
        Authenticated aggregate record.

    Raises:
        ValueError: If worker count or state authority is invalid.
    """
    if workers not in AFFINITIES:
        raise ValueError("workers must be exactly 2")
    from robosuite import models
    from robosuite.utils.shakebench_authority import authority_payload_hash, verify_direct_mount_authority

    from robosuite.scripts.shakebench_run_oracle import load_state_asset
    from robosuite.utils.shakebench_committed_states import OFFICIAL_STATE_FILENAME

    asset_path = Path(models.assets_root) / OFFICIAL_STATE_FILENAME if state_asset is None else Path(state_asset)
    resolved_asset = load_state_asset(asset_path)
    if resolved_asset["authority"].get("kind") != "committed":
        raise ValueError("batch runner requires official or knee committed-state authority")
    states = resolved_asset["states"]
    authority = verify_direct_mount_authority()
    jobs = build_jobs(
        states,
        workers=workers,
        horizon_steps=horizon_steps,
        science_authority_sha256=authority_payload_hash(authority),
        tier=tier,
        gamma_commanded=gamma_commanded,
        state_asset_authority=resolved_asset["authority"],
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
    assignments = [[job for job in jobs if job["execution"]["cpu"] == cpu] for cpu in AFFINITIES[workers]]
    processes = [
        context.Process(target=_worker, args=(assigned, str(output_dir), resume, str(asset_path), result_queue))
        for assigned in assignments
    ]
    previous_env = {key: os.environ.get(key) for key in THREAD_ENV}
    try:
        # spawn children inherit this before their interpreter imports the
        # target module (and therefore before any numerical library import).
        os.environ.update(THREAD_ENV)
        for process in processes:
            process.start()
    finally:
        for key, prior in previous_env.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
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
    resource_reasons = []
    telemetry_complete = telemetry_is_complete(before, after, records)
    if not telemetry_complete:
        resource_reasons.append("required_telemetry_missing")
    if swap_growth or any(
        int(after["vmstat"].get(key) or 0) > int(before["vmstat"].get(key) or 0)
        for key in ("pswpin", "pswpout", "pgmajfault", "oom_kill")
    ):
        resource_reasons.append("swap_or_page_fault")
    if any(int(row.get("millideg_c", 0)) >= 95_000 for row in after.get("temperatures", [])):
        resource_reasons.append("thermal_throttle_risk")
    passed = len(records) == len(jobs) and all(item.get("semantic_passed") for item in records) and not resource_reasons
    simulated_seconds = sum(
        float(item.get("execution", {}).get("simulated_seconds", 0.0))
        for item in records
        if item.get("semantic_passed")
    )
    result = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else ("RESOURCE_LIMITED" if resource_reasons else "FAIL"),
        "worker_count": workers,
        "affinity": list(AFFINITIES[workers]),
        "job_count": len(jobs),
        "completed_count": len(records),
        "elapsed_s": elapsed,
        "jobs_s": len(durations) / elapsed if elapsed else 0.0,
        "simulated_seconds": simulated_seconds,
        "simulated_seconds_per_wall_second": simulated_seconds / elapsed if elapsed else 0.0,
        "paired_duration_s": elapsed,
        "scaling_efficiency": sum(durations) / (workers * elapsed) if elapsed else 0.0,
        "percentile_definition": "R-7 linear interpolation: position=(n-1)*q",
        "p50_s": _percentile(durations, 0.50),
        "p95_s": _percentile(durations, 0.95),
        "resource_before": before,
        "resource_after": after,
        "swap_growth_kib": swap_growth,
        "resource_limit_reasons": resource_reasons,
        "telemetry_complete": telemetry_complete,
        "max_rss_kib_by_job": [item.get("execution", {}).get("max_rss_kib") for item in records],
        "vmstat_delta": {
            key: int(after["vmstat"].get(key) or 0) - int(before["vmstat"].get(key) or 0)
            for key in ("pswpin", "pswpout", "pgmajfault", "oom_kill")
        },
        "incomplete_groups": [item["job_id"] for item in records if item.get("group_status") == "incomplete"],
        "records": records,
    }
    result["payload_sha256"] = aggregate_payload_hash(result)
    _write_atomic(output_dir / "aggregate.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=sorted(AFFINITIES), default=2)
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--state-asset", type=Path, default=None, help="frozen official-state authority (defaults to package asset)"
    )
    parser.add_argument("--tier", choices=("V0", "V1", "V2", "V3"), default="V0")
    parser.add_argument("--gamma", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = run_batch(
        args.output_dir,
        workers=args.workers,
        horizon_steps=args.horizon_steps,
        resume=args.resume,
        state_asset=args.state_asset,
        tier=args.tier,
        gamma_commanded=args.gamma,
    )
    print(
        json.dumps({key: result[key] for key in ("status", "worker_count", "jobs_s", "payload_sha256")}, sort_keys=True)
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AFFINITIES",
    "aggregate_payload_hash",
    "build_jobs",
    "main",
    "resource_snapshot",
    "run_batch",
    "telemetry_is_complete",
    "verify_batch_aggregate",
]
