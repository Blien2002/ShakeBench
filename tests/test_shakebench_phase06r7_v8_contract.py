"""Phase 06R7/V8 normal-impact and fixture contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robosuite.models import assets_root
from robosuite.scripts.shakebench_diagnose_normal_impact_v8 import (
    ANGULAR_VELOCITY_MAX_RAD_S,
    PENETRATION_MAX_M,
    RECOVERY_DURATION_S,
    TAIL_WINDOW_S,
    VELOCITY_MAX_M_S,
)
from robosuite.scripts.shakebench_select_physics_v7 import _profile_from_state, _resolve_v7, load_v7_protocol
from robosuite.scripts.shakebench_select_physics_v8 import (
    _compiled_contact_matches,
    load_v8_protocol,
    _normal_from_trace,
    _incline_from_raw,
    _level_from_raw,
    _compile_contact_audit,
    validate_v8_protocol,
    verify_v8_selection_artifact,
)


ASSETS = Path(assets_root)
DIAGNOSTIC = ASSETS / "shakebench_phase_06r7_v8_normal_impact_diagnostic.json"


def _diagnostic() -> dict:
    return json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))


def test_v8_diagnostic_is_raw_design_evidence_with_exact_normal_trace():
    payload = _diagnostic()
    assert payload["status"] == "DESIGN_EVIDENCE_ONLY"
    assert payload["selection_authority"] is False
    assert payload["scoreable"] is False
    assert payload["not_v8_candidate_result"] is True
    assert payload["no_task_outcome_used"] is True
    assert payload["fixed_contract"] == {
        "recovery_duration_s": RECOVERY_DURATION_S,
        "tail_window_s": TAIL_WINDOW_S,
        "velocity_max_m_s": VELOCITY_MAX_M_S,
        "angular_velocity_max_rad_s": ANGULAR_VELOCITY_MAX_RAD_S,
        "penetration_max_m": PENETRATION_MAX_M,
        "terminal_named_table_support_required": True,
    }
    assert len(payload["normal_impact_candidates"]) == 7
    for candidate in payload["normal_impact_candidates"]:
        normal = candidate["normal_impact"]
        assert normal["release_time_s"] == pytest.approx(0.0)
        assert normal["duration_s"] == pytest.approx(RECOVERY_DURATION_S)
        assert normal["trace_dt_s"] == pytest.approx(0.0002)
        assert len(normal["trace"]) == 2501
        assert normal["first_impact_time_s"] is not None
        assert normal["contact_transitions"]
        assert normal["terminal_contact_state"]["named_table_contact"] is True
        assert normal["tail_window_s"] == pytest.approx(TAIL_WINDOW_S)


def test_v8_level_control_precedes_raw_incline_bracket():
    payload = _diagnostic()
    level = payload["level_control"]
    assert level["passed"] is True
    assert level["terminal_support"]["contact_count"] > 0
    assert level["maximum_hold_horizontal_speed_m_s"] < VELOCITY_MAX_M_S
    assert level["maximum_hold_horizontal_displacement_m"] < 0.0005
    incline = payload["incline_calibration"]
    assert incline["coarse_angle_grid_rad"][0] == pytest.approx(0.0)
    assert incline["binary_refinement_iterations"] == 5
    assert incline["bracket_order_passed"] is True
    assert incline["bracket_width_rad"] <= 0.01
    assert incline["analytic_is_comparison_only"] is True


def test_v8_verifier_rejects_airborne_low_speed_and_trace_mutations():
    payload = _diagnostic()
    normal = payload["normal_impact_candidates"][0]["normal_impact"]
    good, errors = _normal_from_trace(normal, timestep_s=0.0002, duration_s=0.5, tail_window_s=0.05, penetration_limit_m=PENETRATION_MAX_M, velocity_limit_m_s=VELOCITY_MAX_M_S, angular_limit_rad_s=ANGULAR_VELOCITY_MAX_RAD_S, support_force_min_N=0.001)
    assert not errors
    assert good["terminal_named_table_support"] is True
    mutated = copy.deepcopy(normal)
    mutated["trace"][1]["time_s"] += 0.0001
    _, errors = _normal_from_trace(mutated, timestep_s=0.0002, duration_s=0.5, tail_window_s=0.05, penetration_limit_m=PENETRATION_MAX_M, velocity_limit_m_s=VELOCITY_MAX_M_S, angular_limit_rad_s=ANGULAR_VELOCITY_MAX_RAD_S, support_force_min_N=0.001)
    assert any("timestamps" in error for error in errors)
    airborne = copy.deepcopy(normal)
    airborne["trace"][-1]["contacts"] = {"contact_count": 0, "normal_force_N": 0.0, "penetration_m": 0.0, "contacts": []}
    _, errors = _normal_from_trace(airborne, timestep_s=0.0002, duration_s=0.5, tail_window_s=0.05, penetration_limit_m=PENETRATION_MAX_M, velocity_limit_m_s=1.0, angular_limit_rad_s=ANGULAR_VELOCITY_MAX_RAD_S, support_force_min_N=0.001)
    assert not errors
    assert airborne["trace"][-1]["contacts"]["contact_count"] == 0
    assert _normal_from_trace(airborne, timestep_s=0.0002, duration_s=0.5, tail_window_s=0.05, penetration_limit_m=PENETRATION_MAX_M, velocity_limit_m_s=1.0, angular_limit_rad_s=ANGULAR_VELOCITY_MAX_RAD_S, support_force_min_N=0.001)[0]["terminal_named_table_support"] is False
    wrong_interface = copy.deepcopy(normal)
    contact_index = next(index for index, row in enumerate(wrong_interface["trace"]) if row["contacts"]["contacts"])
    wrong_interface["trace"][contact_index]["contacts"]["contacts"][0]["interface"] = "can_target_bottom"
    _, errors = _normal_from_trace(wrong_interface, timestep_s=0.0002, duration_s=0.5, tail_window_s=0.05, penetration_limit_m=PENETRATION_MAX_M, velocity_limit_m_s=VELOCITY_MAX_M_S, angular_limit_rad_s=ANGULAR_VELOCITY_MAX_RAD_S, support_force_min_N=0.001)
    assert any("interface" in error for error in errors)


def test_v8_fake_stable_lower_bracket_and_level_failure_are_rejected():
    payload = _diagnostic()
    incline = copy.deepcopy(payload["incline_calibration"])
    incline["angle_records"] = [row for row in incline["angle_records"] if row["angle_rad"] != 0.0]
    _, errors = _incline_from_raw(incline, resolution_rad=0.01, displacement_gate_m=0.0005)
    assert any("zero-angle" in error for error in errors)
    level = copy.deepcopy(payload["level_control"])
    level["hold_trace"][-1]["contacts"]["contact_count"] = 0
    result, errors = _level_from_raw(level, speed_limit_m_s=0.02, displacement_limit_m=0.0005, support_force_min_N=0.001)
    assert not errors
    assert result["passed"] is False


def test_v8_candidate_contact_tuple_compiles_exactly_against_real_model():
    protocol, _, protocol_hash = load_v7_protocol(ASSETS / "shakebench_selection_protocol_v7.yaml")
    state = next(state for state in _resolve_v7(protocol, protocol_bytes_hash=protocol_hash) if state.stage == "contact" and state.candidate_id == "c6_rolling_high")
    compiled = _compile_contact_audit(state)
    passed, errors = _compiled_contact_matches(compiled, state)
    assert passed, errors


def test_v8_protocol_and_blocked_selection_preserve_the_frozen_binding():
    protocol, _, protocol_hash = load_v8_protocol(ASSETS / "shakebench_selection_protocol_v8.yaml")
    structure = validate_v8_protocol(protocol, protocol_bytes_hash=protocol_hash)
    assert structure["stage_counts"] == {"driver": 18, "isolator": 3, "contact": 5, "parity": 1, "replay": 12}
    assert structure["legal_replay_binding_count"] == 102
    selected = json.loads((ASSETS / "shakebench_phase_06r7_v8_selected_candidates.json").read_text(encoding="utf-8"))
    assert selected["status"] == "BLOCKED"
    assert selected["selection"]["selected_candidate_ids"]["contact"] is None
    result = verify_v8_selection_artifact(ASSETS / "shakebench_phase_06r7_v8_selected_candidates.json", protocol_path=ASSETS / "shakebench_selection_protocol_v8.yaml")
    assert result["passed"] is False
    assert result["recomputed_selection"]["contact"] == "n6_overdamped_4"
    assert any("registered parity/replay binding" in error for error in result["errors"])
