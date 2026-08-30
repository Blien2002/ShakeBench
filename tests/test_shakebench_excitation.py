"""Phase 01 golden and analytic-contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robosuite.utils.shakebench_calibration import (
    authored_point_vertical_acceleration,
    calibrate_gamma,
)
from robosuite.utils.shakebench_excitation import (
    AXES,
    CONSERVATIVE_MAX_LINE_FREQUENCY_HZ,
    DEFAULT_EXCITATION_CONFIG,
    build_band_table,
    build_excitation_program,
    derive_rotation_accel_rms,
    quintic_ramp,
)
from robosuite.utils.shakebench_safety import (
    SafetyLimits,
    SafetyViolation,
    check_frequency,
    check_safety,
    check_solver_travel,
    validate_safety,
)

FIXTURE_PATH = Path(__file__).with_name("golden_shakebench_excitation_v0.json")
ERROR_SUMMARY_PATH = Path(__file__).with_name("golden_shakebench_excitation_error_summary.json")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_band_table_matches_design_and_rotation_derivation() -> None:
    bands = build_band_table()
    assert tuple(band.axis for band in bands) == AXES
    expected = {
        "tx": (5.0, 0.50, 0.10, 12),
        "ty": (6.5, 0.35, 0.10, 10),
        "tz": (8.0, 1.00, 0.10, 12),
        "rx": (3.0, 0.12, 12),
        "ry": (4.0, 0.12, 10),
        "rz": (2.5, 0.12, 8),
    }
    for band in bands[:3]:
        center, relative_rms, bandwidth, tones = expected[band.axis]
        assert band.center_hz == center
        assert band.relative_accel_rms == relative_rms
        assert band.bandwidth_ratio == bandwidth
        assert band.tones == tones
    by_axis = {band.axis: band for band in bands}
    assert by_axis["rx"].accel_rms == pytest.approx(derive_rotation_accel_rms(by_axis["tz"].accel_rms))
    assert by_axis["ry"].accel_rms == pytest.approx(derive_rotation_accel_rms(by_axis["tz"].accel_rms))
    assert by_axis["rz"].accel_rms == pytest.approx(derive_rotation_accel_rms(by_axis["tx"].accel_rms))
    assert by_axis["rx"].coordinate_unit == "rad"
    assert by_axis["rx"].acceleration_unit == "rad/s^2"


def test_line_and_displacement_amplitude_contract() -> None:
    level = 0.37
    program = build_excitation_program(seed=4, level_scale=level)
    for index, band in enumerate(program.bands):
        active = program.line_mask[index]
        expected_accel_amp = level * band.accel_rms * np.sqrt(2.0 / band.tones)
        assert np.all(program.line_accel_amplitude[index, active] == pytest.approx(expected_accel_amp))
        expected_q_amp = program.line_accel_amplitude[index, active] / program.line_omega_rad_s[index, active] ** 2
        assert np.all(program.line_q_amplitude[index, active] == pytest.approx(expected_q_amp))


def test_same_seed_replays_and_different_seed_changes_program() -> None:
    first = build_excitation_program(seed=101, t0=0.17, level_scale=0.4)
    replay = build_excitation_program(seed=101, t0=0.17, level_scale=0.4)
    different = build_excitation_program(seed=102, t0=0.17, level_scale=0.4)
    np.testing.assert_array_equal(first.line_accel_amplitude, replay.line_accel_amplitude)
    np.testing.assert_array_equal(first.line_omega_rad_s, replay.line_omega_rad_s)
    np.testing.assert_array_equal(first.line_phase_at_episode_zero, replay.line_phase_at_episode_zero)
    np.testing.assert_array_equal(first.evaluate(np.array([0.1, 0.8])).q, replay.evaluate(np.array([0.1, 0.8])).q)
    assert not np.array_equal(first.line_phase_at_episode_zero, different.line_phase_at_episode_zero)


def test_common_t0_time_shift_identity_after_episode_ramp() -> None:
    t0 = 0.137
    shifted = build_excitation_program(seed=8, t0=t0, level_scale=0.3)
    origin = build_excitation_program(seed=8, t0=0.0, level_scale=0.3)
    times = np.array([0.50, 0.73, 1.25, 1.91])
    shifted_motion = shifted.evaluate(times)
    translated_motion = origin.evaluate(times + t0)
    np.testing.assert_allclose(shifted_motion.q, translated_motion.q, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(shifted_motion.qdot, translated_motion.qdot, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(shifted_motion.qdd, translated_motion.qdd, rtol=0.0, atol=2e-11)
    carrier_times = np.array([-0.2, 0.0, 0.25])
    np.testing.assert_allclose(
        shifted.evaluate_carrier(carrier_times).qdd,
        origin.evaluate_carrier(carrier_times + t0).qdd,
        rtol=0.0,
        atol=2e-14,
    )


def test_quintic_ramp_is_c2_and_has_analytic_derivatives() -> None:
    duration = DEFAULT_EXCITATION_CONFIG.ramp_duration_s
    before, before_first, before_second = quintic_ramp(-1e-8, duration)
    at_zero, at_zero_first, at_zero_second = quintic_ramp(0.0, duration)
    at_end, at_end_first, at_end_second = quintic_ramp(duration, duration)
    after, after_first, after_second = quintic_ramp(duration + 1e-8, duration)
    assert float(before) == 0.0
    assert float(at_zero) == 0.0
    assert float(at_end) == 1.0
    assert float(after) == 1.0
    assert float(at_zero_first) == float(at_zero_second) == 0.0
    assert float(at_end_first) == float(at_end_second) == 0.0
    assert float(before_first) == float(before_second) == 0.0
    assert float(after_first) == float(after_second) == 0.0
    midpoint, midpoint_first, midpoint_second = quintic_ramp(duration / 2.0, duration)
    assert float(midpoint) == pytest.approx(0.5)
    assert float(midpoint_first) == pytest.approx(1.875 / duration)
    assert float(midpoint_second) == pytest.approx(0.0)


def test_runtime_derivatives_match_high_precision_numerical_check() -> None:
    program = build_excitation_program(seed=12, t0=0.21, level_scale=0.55)
    times = np.array([0.13, 0.31, 0.73, 1.17])
    step = 1e-6
    center = program.evaluate(times)
    plus = program.evaluate(times + step)
    minus = program.evaluate(times - step)
    numerical_qdot = (plus.q - minus.q) / (2.0 * step)
    numerical_qdd = (plus.qdot - minus.qdot) / (2.0 * step)
    np.testing.assert_allclose(center.qdot, numerical_qdot, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(center.qdd, numerical_qdd, rtol=3e-7, atol=3e-8)


def test_active_axes_mask_is_explicit_and_units_are_serialized() -> None:
    program = build_excitation_program(seed=5, active_axes=["tz"])
    assert program.active_axes == ("tz",)
    assert np.all(program.line_mask[:2] == 0)
    assert np.all(program.line_mask[3:] == 0)
    sample = program.evaluate(np.array([0.2, 0.8]))
    np.testing.assert_array_equal(sample.q[..., [0, 1, 3, 4, 5]], 0.0)
    payload = program.to_dict()
    assert payload["ramp_type"] == "quintic_smoothstep"
    assert payload["program_frame"] == "deck"
    assert payload["bands"][0]["coordinate_unit"] == "m"
    assert payload["bands"][3]["coordinate_unit"] == "rad"


def test_gamma_uses_workpiece_point_alpha_cross_r_and_unit_replay() -> None:
    program = build_excitation_program(seed=17, t0=0.11, level_scale=0.25)
    times = np.linspace(0.0, 2.0, 5001)
    result = calibrate_gamma(program, time=times, point_offset_m=(0.65, 0.0, 0.0))
    motion = program.evaluate(times)
    manual_vertical = authored_point_vertical_acceleration(motion, (0.65, 0.0, 0.0))
    assert result.peak_acceleration_m_s2 == pytest.approx(np.max(np.abs(manual_vertical)))
    assert result.gamma_commanded == pytest.approx(result.peak_acceleration_m_s2 / result.gravity_m_s2)
    unit = calibrate_gamma(
        seed=17,
        t0=0.11,
        level_scale=1.0,
        time=times,
        point_offset_m=(0.65, 0.0, 0.0),
    )
    assert result.unit_peak_factor == pytest.approx(unit.unit_peak_factor)
    assert result.gamma_commanded == pytest.approx(0.25 * unit.unit_peak_factor)
    assert result.unit_replay["level_scale"] == 1.0
    assert result.per_axis_peak["tz"] >= result.per_axis_rms["tz"]


def test_frequency_gate_and_conservative_bound() -> None:
    program = build_excitation_program(seed=31)
    assert program.max_frequency_hz < CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
    assert check_frequency(program, timestep_s=0.001).passed
    assert not check_frequency(program, timestep_s=1.0).passed
    high_frequency = build_excitation_program(
        seed=31,
        config={"frequency_scale": 1.02},
    )
    assert high_frequency.max_frequency_hz > CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
    assert not check_frequency(high_frequency).passed


def test_displacement_non_ballistic_and_solver_travel_fail_closed() -> None:
    normal = build_excitation_program(seed=2, level_scale=0.1)
    assert check_safety(normal, timestep_s=0.001).passed
    high_displacement = build_excitation_program(seed=2, level_scale=20.0)
    report = check_safety(high_displacement, timestep_s=0.001, sample_count=1001)
    assert not report.passed
    assert {check.name for check in report.violations} >= {"displacement", "non_ballistic"}
    high_solver = check_solver_travel(normal, timestep_s=1.0)
    assert not high_solver.passed
    with pytest.raises(SafetyViolation):
        validate_safety(high_displacement, timestep_s=0.001, sample_count=1001)


def test_golden_fixture_replays_and_error_summary_is_zero() -> None:
    fixture = _fixture()
    assert fixture["provenance"]["generator_script"] == "robosuite/scripts/shakebench_generate_excitation_golden.py"
    assert fixture["cases"]
    for case in fixture["cases"]:
        program_payload = case["program"]
        program = build_excitation_program(
            seed=case["seed"],
            t0=case["t0"],
            level_scale=case["level_scale"],
            active_axes=case["active_axes"],
            config=fixture["config"],
        )
        np.testing.assert_allclose(
            program.line_accel_amplitude,
            np.asarray(program_payload["line_accel_amplitude"]),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            program.line_omega_rad_s,
            np.asarray(program_payload["line_omega_rad_s"]),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            program.line_phase_at_episode_zero,
            np.asarray(program_payload["line_phase_at_episode_zero"]),
            rtol=0.0,
            atol=0.0,
        )
        replay = program.evaluate(np.asarray(case["sample_time_s"], dtype=float))
        np.testing.assert_allclose(replay.q, np.asarray(case["motion"]["q"]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(replay.qdot, np.asarray(case["motion"]["qdot"]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(replay.qdd, np.asarray(case["motion"]["qdd"]), rtol=0.0, atol=0.0)
        assert program.max_frequency_hz < 8.87
    errors = json.loads(ERROR_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert np.isfinite(errors["max_error"])
    assert errors["max_error"] <= errors["error_tolerances"]["analytic_derivative_abs"]
    assert all(
        value == 0.0
        for name, value in errors["errors"].items()
        if name.endswith(
            (
                "finite_program",
                "line_accel_amplitude_abs_error",
                "line_q_amplitude_abs_error",
                "max_frequency_excess_hz",
                "safety_failure",
            )
        )
    )


def test_golden_rejection_cases_cover_all_safety_gates() -> None:
    fixture = _fixture()
    for case in fixture["rejection_cases"]:
        assert case["passed"] is False
        assert case["expected_gate"] in case["report"]["violations"]
