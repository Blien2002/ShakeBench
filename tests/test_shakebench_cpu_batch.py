"""Contract tests for the Phase 7.5B two-worker batch execution layer."""

import pytest
import errno

from robosuite.scripts.shakebench_cpu_batch import (
    AFFINITIES,
    aggregate_payload_hash,
    _percentile,
    _read_key_values,
    build_jobs,
    build_retry_ledger,
    classify_execution_exception,
    job_record_payload_hash,
    mark_job_record_resumed,
    resource_snapshot,
    run_batch,
    telemetry_is_complete,
    verify_batch_aggregate,
    thread_environment_spawn_probe,
)


def _states():
    return [
        {"state_id": "state-000", "seed": 1},
        {"state_id": "state-001", "seed": 2},
    ]


def test_job_identity_is_stable_and_execution_affinity_is_not_science():
    first = build_jobs(_states(), workers=2, horizon_steps=1200, science_authority_sha256="a" * 64)
    second = build_jobs(_states(), workers=2, horizon_steps=1200, science_authority_sha256="a" * 64)
    assert first == second
    assert [job["execution"]["cpu"] for job in first] == list(AFFINITIES[2])
    assert all("cpu" not in job["science"] for job in first)


def test_batch_rejects_unregistered_worker_count(tmp_path):
    with pytest.raises(ValueError, match="exactly 2"):
        run_batch(tmp_path, workers=4)


def test_batch_rejects_a_one_worker_substitute(tmp_path):
    with pytest.raises(ValueError, match="exactly 2"):
        run_batch(tmp_path, workers=1)


def test_resource_snapshot_has_fail_closed_swap_and_vm_fields():
    snapshot = resource_snapshot()
    assert {"mem_available_kib", "swap_total_kib", "swap_free_kib", "vmstat"} <= set(snapshot)
    assert {"pswpin", "pswpout", "pgmajfault", "oom_kill"} == set(snapshot["vmstat"])


def test_proc_parser_accepts_meminfo_units(tmp_path):
    source = tmp_path / "meminfo"
    source.write_text("MemAvailable: 12345 kB\nSwapFree: 678 kB\n", encoding="utf-8")
    assert _read_key_values(source) == {"MemAvailable": 12345, "SwapFree": 678}


def test_percentile_uses_linear_definition_not_unconditional_maximum():
    """P95 is the deterministic R-7 linear percentile used by the aggregate."""

    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.50) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_spawn_child_inherits_all_thread_limits_before_worker_code():
    """Pure stdlib probe; it creates neither an environment nor a rollout."""

    assert thread_environment_spawn_probe() == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def test_resumed_record_remains_authentic_after_repeated_resume_updates():
    record = {
        "schema_id": "shakebench.phase07_5b.cpu_batch.job",
        "schema_version": 1,
        "job_id": "job",
        "science": {"state_id": "official-000"},
        "resumed": False,
    }
    record["payload_sha256"] = job_record_payload_hash(record)

    first = mark_job_record_resumed(record)
    second = mark_job_record_resumed(first)

    assert first["payload_sha256"] == job_record_payload_hash(first)
    assert second["payload_sha256"] == job_record_payload_hash(second)


def test_only_explicit_infrastructure_errors_are_retryable_and_ledger_is_bound():
    job = {"job_id": "job", "science": {"state_id": "official-000"}}
    assert classify_execution_exception(OSError(errno.EIO, "I/O")) == "infrastructure"
    assert classify_execution_exception(OSError(errno.ENOENT, "missing authority")) == "contract"
    assert classify_execution_exception(RuntimeError("bad artifact")) == "contract"

    ledger = build_retry_ledger(job, [{"attempt": 1, "classification": "infrastructure"}])
    assert ledger["job_id"] == "job"
    assert ledger["science"] == job["science"]
    assert len(ledger["payload_sha256"]) == 64


def test_complete_pass_requires_all_resource_telemetry_and_job_rss():
    snapshot = {
        "mem_available_kib": 1,
        "swap_total_kib": 0,
        "swap_free_kib": 0,
        "vmstat": {"pswpin": 0, "pswpout": 0, "pgmajfault": 0, "oom_kill": 0},
        "temperatures": [{"millideg_c": 40_000}],
        "frequencies": [{"cpu": 0, "khz": 2_000_000}],
    }
    assert telemetry_is_complete(snapshot, snapshot, [{"execution": {"max_rss_kib": 1}}])
    assert not telemetry_is_complete({**snapshot, "temperatures": []}, snapshot, [{"execution": {"max_rss_kib": 1}}])


def test_batch_aggregate_verifier_rejects_rehashed_or_tampered_audit_data(tmp_path):
    aggregate = {
        "schema_id": "shakebench.phase07_5b.cpu_batch",
        "schema_version": 1,
        "records": [],
        "resource_before": {},
        "resource_after": {},
        "incomplete_groups": [],
        "payload_sha256": "",
    }
    aggregate["payload_sha256"] = aggregate_payload_hash(aggregate)
    path = tmp_path / "aggregate.json"
    path.write_text(__import__("json").dumps(aggregate), encoding="utf-8")
    assert verify_batch_aggregate(path)["passed"]

    aggregate["incomplete_groups"] = ["tampered"]
    path.write_text(__import__("json").dumps(aggregate), encoding="utf-8")
    assert not verify_batch_aggregate(path)["passed"]
