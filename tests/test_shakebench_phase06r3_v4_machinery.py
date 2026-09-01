"""Pure V4 protocol/manifest/verifier tests; no V4 MuJoCo artifact is used."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robosuite.scripts.shakebench_select_physics_v4 import (
    RAW_PREFIX,
    V4ProtocolError,
    build_dry_run_manifest,
    run_contact_stage,
    run_driver_stage,
    run_isolator_stage,
    run_parity_stage,
    run_replay_stage,
    run_staged_probe_group,
    sha256_json,
    validate_protocol,
    verify_selection_artifact,
)
from robosuite.utils.shakebench_artifacts import payload_hash, write_json_atomic


def _protocol() -> dict:
    drivers = [
        {"candidate_id": "dt_fine", "physics_timestep_s": 0.0001, "refresh_stride": 50, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_eq_solref": [0.0002, 0.5]},
        {"candidate_id": "dt_medium", "physics_timestep_s": 0.000125, "refresh_stride": 40, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_eq_solref": [0.00025, 0.5]},
        {"candidate_id": "dt_nominal", "physics_timestep_s": 0.0002, "refresh_stride": 25, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_eq_solref": [0.0004, 0.5]},
    ]
    manifest = []
    for driver in drivers:
        for gamma in (0.15, 0.30, 0.50):
            for load in ("empty", "panda_plus_worktable_reference_proxy"):
                state_id = f"driver.{driver['candidate_id']}.gamma_{str(gamma).replace('.', '_')}.{load}"
                manifest.append({"state_id": state_id, "stage": "driver", "candidate_id": driver["candidate_id"], "duration_s": 1.0, "mujoco_step_count": int(round(1.0 / driver["physics_timestep_s"])), "retained_sample_count": 200, "output": RAW_PREFIX + state_id.replace('.', '_') + ".json", "retry_policy": "same_state_config_once_then_block"})
    for stage, state_id in (("isolator", "isolator.balanced_nominal"), ("contact", "contact.c3_nominal"), ("parity", "parity.gamma_zero"), ("replay", "replay.all_groups")):
        manifest.append({"state_id": state_id, "stage": stage, "candidate_id": state_id.split(".")[-1], "duration_s": 1.0, "mujoco_step_count": 1000, "retained_sample_count": 200, "output": RAW_PREFIX + state_id.replace('.', '_') + ".json", "retry_policy": "same_state_config_once_then_block"})
    return {
        "schema_id": "shakebench.phase06r3.v4.physics_selection_protocol",
        "schema_version": 4,
        "status": "pre_registered",
        "immutable_after_registration": True,
        "frozen_facts": {"f_max_hz": 8.87, "safe_gamma_candidates": [0.15, 0.30, 0.50], "load_cases": ["empty", "panda_plus_worktable_reference_proxy"]},
        "driver": {"convergence_candidates": drivers, "hard_gates": {"line_amplitude_relative_error_max": 0.01, "line_phase_absolute_error_deg_max": 1.0, "gamma_deck_relative_error_max": 0.01, "warning_count_max": 0, "unstable_residual_max": 0.001}},
        "components": {"isolator_candidates": [{"candidate_id": "balanced_nominal"}], "contact_candidates": [{"candidate_id": "c3_nominal"}]},
        "execution_manifest": manifest,
    }


def _synthetic_selection(tmp_path: Path, protocol: dict) -> Path:
    (tmp_path / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=True), encoding="utf-8")
    raw_files = []
    protocol_hash = sha256_json(protocol)
    for state in protocol["execution_manifest"]:
        stage = state["stage"]
        if stage == "driver":
            evidence = {"candidate_id": state["candidate_id"], "metrics": {"max_line_amplitude_relative_error": 0.001, "max_absolute_line_phase_error_deg": 0.1, "max_gamma_relative_error": 0.001, "warning_count": 0, "max_weld_residual": 0.0001, "complete_64_line_spectrum": True}}
        else:
            evidence = {"candidate_id": state["candidate_id"], "metrics": {"passed": True}}
        raw = {"schema_id": "shakebench.phase06r3.v4.physics_selection.raw", "schema_version": 4, "stage": stage, "state_id": state["state_id"], "protocol_sha256": protocol_hash, "evidence": evidence}
        raw["payload_sha256"] = payload_hash(raw)
        path = tmp_path / state["output"]
        digest = write_json_atomic(path, raw)
        raw_files.append({"path": path.name, "sha256": digest})
    selected = {"schema_id": "shakebench.phase06r3.v4.physics_selection", "schema_version": 4, "status": "PASS", "protocol": {"sha256": protocol_hash}, "raw_files": raw_files}
    selected["payload_sha256"] = payload_hash(selected)
    path = tmp_path / "selected.json"
    write_json_atomic(path, selected)
    return path


def test_v4_structural_validator_accepts_exact_divisible_manifest():
    result = validate_protocol(_protocol())
    assert result["manifest_count"] == 22
    assert [row["control_steps"] for row in result["driver"]["candidates"]] == [500, 400, 250]
    assert all(row["sample_rate_hz"] >= 177.4 for row in result["driver"]["candidates"])
    assert len(build_dry_run_manifest(_protocol())) == 22


def test_v4_structural_validator_rejects_nonintegral_or_duplicate_manifest():
    bad = _protocol()
    bad["driver"]["convergence_candidates"][1]["physics_timestep_s"] = 0.00015
    with pytest.raises(V4ProtocolError, match="does not divide"):
        validate_protocol(bad)
    duplicate = _protocol()
    duplicate["execution_manifest"].append(copy.deepcopy(duplicate["execution_manifest"][0]))
    with pytest.raises(V4ProtocolError, match="duplicates"):
        validate_protocol(duplicate)


def test_staged_runner_resumes_atomic_synthetic_state(tmp_path):
    protocol = _protocol()
    state = {"state_id": "parity.gamma_zero", "stage": "parity", "duration_s": 1.0, "mujoco_step_count": 1000, "retained_sample_count": 200, "output": RAW_PREFIX + "parity_gamma_zero.json", "retry_policy": "same_state_config_once_then_block"}
    reduced = dict(protocol)
    reduced["execution_manifest"] = [state]
    calls = []
    probe = lambda item: calls.append(item["state_id"]) or {"synthetic": True}
    run_staged_probe_group(reduced, stage="parity", output_dir=tmp_path, probe=probe)
    run_staged_probe_group(reduced, stage="parity", output_dir=tmp_path, probe=probe)
    assert calls == ["parity.gamma_zero"]


def test_all_v4_stages_are_callable_with_temporary_synthetic_fixture(tmp_path):
    protocol = _protocol()
    probe = lambda state: {"synthetic": True, "stage": state["stage"]}
    for stage_runner in (run_driver_stage, run_isolator_stage, run_contact_stage, run_parity_stage, run_replay_stage):
        result = stage_runner(protocol, output_dir=tmp_path, probe=probe)
        assert result
    assert len(list(tmp_path.glob(RAW_PREFIX + "*.json"))) == 22


def test_v4_verifier_recomputes_synthetic_raw_coverage_and_rejects_mutation(tmp_path):
    protocol = _protocol()
    selected_path = _synthetic_selection(tmp_path, protocol)
    # The synthetic artifact intentionally lacks a real official profile, so
    # the verifier must still reject it for missing publication alignment.
    summary = verify_selection_artifact(selected_path, protocol_path=tmp_path / "protocol.yaml")
    assert summary["passed"] is False
    raw_path = next(tmp_path.glob(RAW_PREFIX + "driver_*.json"))
    raw_path.write_text(raw_path.read_text(encoding="utf-8").replace("0.001", "0.002", 1), encoding="utf-8")
    mutated = verify_selection_artifact(selected_path, protocol_path=tmp_path / "protocol.yaml")
    assert mutated["passed"] is False
