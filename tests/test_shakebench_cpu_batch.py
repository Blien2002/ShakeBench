"""Contract tests for the Phase 7.5B two-worker batch execution layer."""

import pytest

from robosuite.scripts.shakebench_cpu_batch import (
    AFFINITIES,
    _read_key_values,
    build_jobs,
    resource_snapshot,
    run_batch,
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
    with pytest.raises(ValueError, match="1 or 2"):
        run_batch(tmp_path, workers=4)


def test_resource_snapshot_has_fail_closed_swap_and_vm_fields():
    snapshot = resource_snapshot()
    assert {"mem_available_kib", "swap_total_kib", "swap_free_kib", "vmstat"} <= set(snapshot)
    assert {"pswpin", "pswpout", "pgmajfault", "oom_kill"} == set(snapshot["vmstat"])


def test_proc_parser_accepts_meminfo_units(tmp_path):
    source = tmp_path / "meminfo"
    source.write_text("MemAvailable: 12345 kB\nSwapFree: 678 kB\n", encoding="utf-8")
    assert _read_key_values(source) == {"MemAvailable": 12345, "SwapFree": 678}
