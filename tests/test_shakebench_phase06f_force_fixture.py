"""Raw-MuJoCo Phase 06F level-force fixture tests."""

from __future__ import annotations

import copy

import pytest

from robosuite.scripts.shakebench_diagnose_normal_impact_v8 import CAN_INERTIA_KG_M2, CAN_MASS_KG
from robosuite.utils.shakebench_contact_force import (
    compiled_pair_audit,
    force_threshold_gates,
    run_horizontal_force_threshold,
    run_task_native_settling,
)


def _candidate() -> dict:
    return {
        "candidate_id": "c3_nominal",
        "condim": 3,
        "sliding_mu": {"table_object": 0.30, "finger_object": 1.00},
        "torsional_mu": 0.005,
        "rolling_mu": 0.0001,
        "margin_m": 0.0,
        "gap_m": 0.0,
        "solref": [0.0004, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "iterations": 100,
    }


def _run(candidate: dict) -> dict:
    return run_horizontal_force_threshold(
        candidate,
        force_factors=(0.0, 0.5, 0.8, 1.0, 1.2, 1.5),
        reference_mu=0.30,
        mass_kg=0.349,
        gravity_m_s2=9.81,
        plateau_duration_s=0.25,
        static_displacement_max_m=0.0005,
        monotonic_tolerance_m=1.0e-7,
        bracket_relative_tolerance=0.50,
    )


def test_compiled_fixture_has_exact_mass_inertia_and_isotropic_pair_tuple():
    audit = compiled_pair_audit(_candidate())
    assert audit["can_mass_kg"] == pytest.approx(CAN_MASS_KG, rel=0.0, abs=1.0e-15)
    assert audit["can_inertia_kg_m2"] == pytest.approx(CAN_INERTIA_KG_M2, rel=0.0, abs=1.0e-15)
    assert audit["friction"] == pytest.approx([0.30, 0.30, 0.005, 0.0001, 0.0001])
    assert audit["matching_pair_count"] == 1


@pytest.mark.parametrize("support", ("open_worktable", "target_bottom"))
def test_exact_task_native_support_pose_settles_for_final_window(support):
    result = run_task_native_settling(_candidate(), support=support)
    assert result["settle_duration_s"] == pytest.approx(0.50)
    assert result["final_window_s"] == pytest.approx(0.10)
    assert result["final_window_named_support_continuous"] is True
    assert result["final_window_max_linear_speed_m_s"] <= 0.02
    assert result["final_window_max_angular_speed_rad_s"] <= 0.20
    assert result["maximum_penetration_m"] <= 0.0005
    assert result["finite_contact_forces"] is True
    assert result["warning_count"] == 0


def test_force_plateaus_bracket_analytic_friction_reference():
    result = _run(_candidate())
    assert result["protocol_reference"]["F_mu_N"] == pytest.approx(0.30 * 0.349 * 9.81)
    assert result["force_factors"] == [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]
    assert result["reset_to_identical_settled_state_each_plateau"] is True
    assert all(force_threshold_gates(result, penetration_max_m=0.0005, transverse_drift_max_m=0.0005).values())


def test_broken_compiled_friction_is_detected_by_subthreshold_motion():
    broken = copy.deepcopy(_candidate())
    broken["candidate_id"] = "broken_zero_friction"
    broken["sliding_mu"]["table_object"] = 0.0
    result = _run(broken)
    gates = force_threshold_gates(result, penetration_max_m=0.0005, transverse_drift_max_m=0.0005)
    assert gates["subthreshold_static"] is False
    assert gates["transition_brackets_reference"] is False
    assert not all(gates.values())
