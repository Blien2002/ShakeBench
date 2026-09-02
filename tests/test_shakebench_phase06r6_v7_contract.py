"""Phase 06R6 diagnostic and V7 pre-registration contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robosuite.models import assets_root
from robosuite.scripts.shakebench_select_physics_v6 import load_v6_protocol, _resolve
from robosuite.scripts.shakebench_select_physics_v7 import (
    _compile_contact_audit,
    _compile_force_envelope,
    _compiled_contact_matches,
    _finger_from_trace,
    _recovery_from_trace,
    _verify_v6_inherited_evidence,
)


ASSETS = Path(assets_root)
DIAGNOSTIC = ASSETS / "shakebench_phase_06r6_v7_contact_recovery_diagnostic.json"


def test_v7_diagnostic_is_design_only_and_freezes_recovery_convention():
    payload = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    assert payload["status"] == "DESIGN_EVIDENCE_ONLY"
    assert payload["selection_authority"] is False
    assert payload["scoreable"] is False
    assert payload["not_v7_candidate_result"] is True
    assert payload["no_task_outcome_used"] is True
    fixed = payload["fixed_recovery_contract"]
    assert fixed["duration_s"] == pytest.approx(0.50)
    assert fixed["velocity_max_m_s"] == pytest.approx(0.02)
    assert fixed["angular_velocity_max_rad_s"] == pytest.approx(0.20)
    assert fixed["penetration_max_m"] == pytest.approx(0.0005)
    assert len(payload["candidates"]) == 3
    for candidate in payload["candidates"]:
        recovery = candidate["recovery"]
        assert recovery["release_time_s"] == pytest.approx(0.0)
        assert recovery["duration_s"] == pytest.approx(0.50)
        assert recovery["trace_sample_count"] > 1
        assert recovery["first_impact_time_s"] is not None
        assert recovery["tail_window_s"] == pytest.approx(0.05)
        assert len(candidate["recovery_trace"]) == recovery["trace_sample_count"]


def test_v7_diagnostic_derives_finite_force_envelope_and_rejects_v6_midpoint_load():
    payload = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    for candidate in payload["candidates"]:
        envelope = candidate["finger_force_envelope"]
        assert envelope["finite_force_envelope_N"] == pytest.approx(40.0)
        assert "actuator_forcerange" in envelope["derivation"]
        finger = candidate["finger_load"]
        assert finger["maximum_normal_force_N"] > envelope["finite_force_envelope_N"]
        assert finger["over_force_rejected"] is True
        assert len(finger["named_pair_audit"]) == 2


def test_v7_recovery_verifier_rejects_timestamp_mutation_and_wrong_interface():
    payload = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    candidate = payload["candidates"][0]
    gates = {"recovery_duration_s": 0.50, "tail_window_s": 0.05}
    result, errors = _recovery_from_trace(candidate["recovery_trace"], timestep_s=0.0002, gates=gates)
    assert not errors
    assert result["duration_s"] == pytest.approx(0.50)
    mutated = json.loads(json.dumps(candidate["recovery_trace"]))
    mutated[1]["time_s"] += 0.0001
    _, errors = _recovery_from_trace(mutated, timestep_s=0.0002, gates=gates)
    assert any("timestamp" in error for error in errors)
    finger = json.loads(json.dumps(candidate["finger_load"]))
    finger["trace"][0]["contacts"]["contacts"][0]["interface"] = "can_open_worktable"
    _, errors = _finger_from_trace(finger, gates={})
    assert any("interface" in error for error in errors)


def test_v7_compiled_profile_and_force_envelope_are_independently_auditable():
    protocol, _, protocol_hash = load_v6_protocol(ASSETS / "shakebench_selection_protocol_v6.yaml")
    state = next(state for state in _resolve(protocol, protocol_bytes_hash=protocol_hash) if state.stage == "contact" and state.candidate_id == "c3_nominal")
    compiled = _compile_contact_audit(state)
    passed, errors = _compiled_contact_matches(compiled, state)
    assert passed, errors
    envelope = _compile_force_envelope(state)
    assert envelope["finite_force_envelope_N"] == pytest.approx(40.0)
    assert all(row["force_limit_N"] == pytest.approx(20.0) for row in envelope["actuators"])


def test_v7_inheritance_rejects_historical_v6_contact_selection_and_keeps_noncontact():
    inherited = _verify_v6_inherited_evidence()
    assert inherited["selected_driver"] == "dt_nominal"
    assert inherited["selected_isolator"] == "low_frequency_damped"
    assert len(inherited["raw_files"]) == 21
