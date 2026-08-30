"""Focused Phase 01 authored Gamma calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.utils.shakebench_calibration import (
    authored_point_vertical_acceleration,
    calibrate_gamma,
    level_scale_for_gamma,
    workpiece_point_acceleration,
)
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.shakebench_safety import SafetyGeometry, check_displacement, check_solver_travel


def test_alpha_cross_r_is_included_in_workpiece_point_acceleration() -> None:
    program = build_excitation_program(seed=6, active_axes=["ry"], level_scale=0.5)
    times = np.array([0.2, 0.7, 1.3])
    motion = program.evaluate(times)
    point = np.array([0.65, 0.0, 0.0])
    point_acceleration = workpiece_point_acceleration(motion, point)
    expected = motion.qdd[..., :3] + np.cross(motion.qdd[..., 3:], point)
    np.testing.assert_allclose(point_acceleration, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        authored_point_vertical_acceleration(motion, point),
        expected[..., 2],
        rtol=0.0,
        atol=0.0,
    )


def test_gamma_calibration_is_replayable_and_linear_in_level_scale() -> None:
    times = np.linspace(0.0, 2.0, 4001)
    unit = calibrate_gamma(seed=9, t0=0.19, level_scale=1.0, time=times)
    scaled = calibrate_gamma(seed=9, t0=0.19, level_scale=0.4, time=times)
    replay = calibrate_gamma(seed=9, t0=0.19, level_scale=0.4, time=times)
    assert scaled.gamma_commanded == pytest.approx(0.4 * unit.unit_peak_factor)
    assert scaled.gamma_commanded == replay.gamma_commanded
    assert scaled.unit_replay == replay.unit_replay
    assert scaled.per_axis_peak == replay.per_axis_peak
    assert scaled.unit_replay["config"]["workpiece_point_offset_m"] == [0.65, 0.0, 0.0]
    assert scaled.unit_replay["time_grid_s"] == times.tolist()
    unit_program = build_excitation_program(
        seed=scaled.unit_replay["seed"],
        t0=scaled.unit_replay["t0"],
        level_scale=scaled.unit_replay["level_scale"],
        active_axes=scaled.unit_replay["active_axes"],
        config=scaled.unit_replay["config"],
    )
    np.testing.assert_array_equal(
        unit_program.line_phase_at_episode_zero,
        np.asarray(scaled.unit_replay["program"]["line_phase_at_episode_zero"]),
    )


def test_level_scale_for_gamma_round_trips_unit_peak_factor() -> None:
    times = np.linspace(0.0, 2.0, 2001)
    unit = calibrate_gamma(seed=13, t0=0.07, time=times)
    requested_gamma = 0.23 * unit.unit_peak_factor
    assert level_scale_for_gamma(
        requested_gamma,
        seed=13,
        t0=0.07,
        time=times,
    ) == pytest.approx(0.23)


def test_gamma_zero_for_no_active_axes() -> None:
    result = calibrate_gamma(seed=1, active_axes=[], time=np.linspace(0.0, 1.0, 101))
    assert result.gamma_commanded == 0.0
    assert result.unit_peak_factor == 0.0
    assert all(value == 0.0 for value in result.per_axis_peak.values())


def test_displacement_gate_checks_combined_translation_norm():
    program = build_excitation_program(seed=3, level_scale=0.5)
    axis_bound = float(np.max(program.axis_displacement_bound[:3]))
    vector_bound = float(np.linalg.norm(program.axis_displacement_bound[:3]))
    limit = (axis_bound + vector_bound) / 2.0
    assert axis_bound < limit < vector_bound
    assert not check_displacement(program, max_displacement_m=limit).passed


def test_rotation_only_feature_displacement_is_not_invisible():
    program = build_excitation_program(seed=4, active_axes=["rz"], level_scale=5.0)
    geometry = SafetyGeometry(feature_offsets_m={"table_corner": (0.5, 0.0, 0.0)})
    report = check_displacement(program, max_displacement_m=0.001, geometry=geometry)
    assert not report.passed
    assert "table_corner" in report.details["feature_bounds_m"]


def test_solver_travel_rejects_rotation_induced_feature_motion():
    program = build_excitation_program(seed=5, active_axes=["rx"], level_scale=5.0)
    geometry = SafetyGeometry(feature_offsets_m={"table_corner": (0.0, 0.5, 0.0)})
    check = check_solver_travel(
        program,
        timestep_s=0.1,
        solver_travel_m=0.008,
        solver_step_fraction=0.25,
        geometry=geometry,
    )
    assert not check.passed
    assert check.details["feature_speed_bound_m_s"] > 0.0
    assert check.details["translation_step_displacement_m"] <= check.limit
    assert check.details["feature_step_displacement_m"] > check.limit
