"""Phase 06F-R focused and adversarial evidence tests."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from robosuite.models import assets_root
from robosuite.scripts.shakebench_handoff_remediation import (
    _trace_digest,
    compare_transient_traces,
    load_protocol,
    validate_protocol,
    verify_inherited_raw,
    verify_remediation_bundle,
    verify_trace,
)
from robosuite.utils.shakebench_artifacts import file_sha256
from robosuite.utils.shakebench_physics_finalizer import artifact_hash


ASSETS = Path(assets_root)


def test_remediation_protocol_and_historical_bundle_are_fail_closed():
    protocol, _, protocol_hash = load_protocol(ASSETS / "shakebench_phase_06fr_protocol.yaml")
    validation = validate_protocol(protocol, protocol_hash)
    assert validation["mujoco_calls"] == 0
    result = verify_remediation_bundle(ASSETS, require_pass=True)
    assert result["passed"], result["errors"]
    status = json.loads((ASSETS / "shakebench_phase_06fr_status.json").read_text())
    assert status["status"] == "PASS"


def test_real_environment_trace_negative_cases_reject_middle_sample_dtype_and_forged_complete():
    protocol, _, _ = load_protocol(ASSETS / "shakebench_phase_06fr_protocol.yaml")
    source = json.loads((ASSETS / "shakebench_phase_06fr_raw_real_open_dt_nominal.json").read_text())
    assert verify_trace(source, protocol)["passed"]
    missing = copy.deepcopy(source)
    missing["trace"].pop(len(missing["trace"]) // 2)
    missing["trace_digest"] = _trace_digest(missing["trace"])
    missing["payload_sha256"] = artifact_hash(missing)
    assert verify_trace(missing, protocol)["passed"] is False
    dtype = copy.deepcopy(source)
    dtype["trace"][100]["deck_pose"] = [str(value) for value in dtype["trace"][100]["deck_pose"]]
    dtype["trace_digest"] = _trace_digest(dtype["trace"])
    dtype["payload_sha256"] = artifact_hash(dtype)
    assert verify_trace(dtype, protocol)["passed"] is False
    forged = copy.deepcopy(source)
    forged["complete_trace"] = True
    forged["trace_digest"] = source["trace_digest"]
    forged["payload_sha256"] = artifact_hash(forged)
    assert verify_trace(forged, protocol)["passed"] is False


def test_equal_terminal_summary_with_different_transient_is_rejected():
    protocol, _, _ = load_protocol(ASSETS / "shakebench_phase_06fr_protocol.yaml")
    selected = json.loads((ASSETS / "shakebench_phase_06fr_raw_real_open_dt_nominal.json").read_text())
    finer = json.loads((ASSETS / "shakebench_phase_06fr_raw_real_open_dt_fine.json").read_text())
    mutated = copy.deepcopy(finer)
    mutated["trace"][2499]["penetration_m"] += 0.01
    result = compare_transient_traces(selected, mutated, protocol)
    assert result["passed"] is False
    assert "penetration_m" in result["errors"]


def test_inherited_raw_mutation_and_missing_file_are_detected(tmp_path):
    # Copy only the registered V6 manifest inputs; verifier must not rely on
    # the selection JSON's inherited description alone.
    for name in (
        "shakebench_selection_protocol_v6.yaml",
        "shakebench_phase_06r5_v6_selected_candidates.json",
        "shakebench_phase_06r5_v6_status.json",
        "shakebench_phase_06r5_v6_feasibility.json",
    ):
        shutil.copy2(ASSETS / name, tmp_path / name)
    selected = json.loads((ASSETS / "shakebench_phase_06r5_v6_selected_candidates.json").read_text())
    for row in selected["raw_files"]:
        if row.get("stage") in {"driver", "isolator"}:
            shutil.copy2(ASSETS / row["path"], tmp_path / row["path"])
    baseline = verify_inherited_raw(tmp_path)
    assert baseline["passed"], baseline["errors"]
    mutation_path = tmp_path / "shakebench_phase_06r5_v6_raw_driver_dt_nominal_gamma_0_30_empty.json"
    mutated = json.loads(mutation_path.read_text())
    mutated["evidence"]["metrics"]["max_weld_residual"] = 999.0
    mutation_path.write_text(json.dumps(mutated), encoding="utf-8")
    assert verify_inherited_raw(tmp_path)["passed"] is False
    shutil.copy2(ASSETS / "shakebench_phase_06r5_v6_raw_driver_dt_nominal_gamma_0_30_empty.json", mutation_path)
    mutation_path.unlink()
    assert verify_inherited_raw(tmp_path)["passed"] is False
