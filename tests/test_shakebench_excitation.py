"""Phase 01 golden and analytic-contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robosuite.scripts.shakebench_generate_excitation_golden import validate_fixture
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


def _fixture_content_hash(fixture: dict) -> str:
    payload = json.loads(json.dumps(fixture))
    provenance = payload["provenance"]
    provenance.pop("fixture_sha256", None)
    provenance.pop("update_reason", None)
    provenance.pop("previous_fixture_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reference_motion(program_payload: dict, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently evaluate a serialized line table for golden acceptance."""

    amplitudes = np.asarray(program_payload["line_accel_amplitude"], dtype=float)
    omegas = np.asarray(program_payload["line_omega_rad_s"], dtype=float)
    phases = np.asarray(program_payload["line_phase_at_episode_zero"], dtype=float)
    mask = np.asarray(program_payload["line_mask"], dtype=bool)
    safe_omega = np.where(mask, omegas, 1.0)
    theta = omegas[..., None] * times + phases[..., None]
    carrier_q = np.sum(-amplitudes[..., None] / safe_omega[..., None] ** 2 * np.sin(theta), axis=1).T
    carrier_qdot = np.sum(-amplitudes[..., None] / safe_omega[..., None] * np.cos(theta), axis=1).T
    carrier_qdd = np.sum(amplitudes[..., None] * np.sin(theta), axis=1).T

    duration = float(program_payload["ramp_duration_s"])
    scaled = times / duration
    ramp = np.where(scaled <= 0.0, 0.0, np.where(scaled >= 1.0, 1.0, 10 * scaled**3 - 15 * scaled**4 + 6 * scaled**5))
    ramp_first = np.where(
        (scaled <= 0.0) | (scaled >= 1.0),
        0.0,
        (30 * scaled**2 - 60 * scaled**3 + 30 * scaled**4) / duration,
    )
    ramp_second = np.where(
        (scaled <= 0.0) | (scaled >= 1.0),
        0.0,
        (60 * scaled - 180 * scaled**2 + 120 * scaled**3) / duration**2,
    )
    q = carrier_q * ramp[:, None]
    qdot = carrier_qdot * ramp[:, None] + carrier_q * ramp_first[:, None]
    qdd = carrier_qdd * ramp[:, None] + 2 * carrier_qdot * ramp_first[:, None] + carrier_q * ramp_second[:, None]
    return q, qdot, qdd


def _reference_point_acceleration(
    motion: tuple[np.ndarray, np.ndarray, np.ndarray],
    point: np.ndarray,
    normal: np.ndarray,
    include_centripetal: bool,
) -> np.ndarray:
    """Project independently evaluated rigid-point acceleration onto a normal."""

    q, qdot, qdd = motion
    acceleration = qdd[:, :3] + np.cross(qdd[:, 3:], point)
    if include_centripetal:
        acceleration = acceleration + np.cross(qdot[:, 3:], np.cross(qdot[:, 3:], point))
    return acceleration @ normal


def _reference_cross_bound(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.array(
        [
            first[1] * second[2] + first[2] * second[1],
            first[2] * second[0] + first[0] * second[2],
            first[0] * second[1] + first[1] * second[0],
        ]
    )


def _reference_safety_bounds(program_payload: dict, limits: dict, timestep_s: float) -> tuple[float, float]:
    """Independently calculate displacement and solver-step bounds."""

    amplitude = np.asarray(program_payload["line_accel_amplitude"], dtype=float)
    omega = np.asarray(program_payload["line_omega_rad_s"], dtype=float)
    mask = np.asarray(program_payload["line_mask"], dtype=bool)
    safe_omega = np.where(mask, omega, 1.0)
    position = np.sum(np.abs(np.where(mask, amplitude / safe_omega**2, 0.0)), axis=1)
    carrier_velocity = np.sum(np.abs(np.where(mask, amplitude / safe_omega, 0.0)), axis=1)
    carrier_acceleration = np.sum(np.abs(amplitude), axis=1)
    duration = float(program_payload["config"]["ramp_duration_s"])
    first_bound = 0.0 if duration == 0.0 else 120.0 / duration
    second_bound = 0.0 if duration == 0.0 else 360.0 / duration**2
    velocity = carrier_velocity + position * first_bound
    acceleration = carrier_acceleration + 2 * carrier_velocity * first_bound + position * second_bound
    feature_offsets = limits["geometry"]["feature_offsets_m"]
    displacement_values = [float(np.linalg.norm(position[:3]))]
    step_values = [
        float(np.linalg.norm(velocity[:3])) * timestep_s + 0.5 * float(np.linalg.norm(acceleration[:3])) * timestep_s**2
    ]
    for raw_offset in feature_offsets.values():
        offset = np.asarray(raw_offset, dtype=float)
        displacement_components = position[:3] + _reference_cross_bound(position[3:], np.abs(offset))
        velocity_components = velocity[:3] + _reference_cross_bound(velocity[3:], np.abs(offset))
        centripetal = velocity[3:] * float(np.sum(velocity[3:] * np.abs(offset))) + np.abs(offset) * float(
            np.sum(velocity[3:] ** 2)
        )
        acceleration_components = (
            acceleration[:3] + _reference_cross_bound(acceleration[3:], np.abs(offset)) + centripetal
        )
        displacement_values.append(float(np.linalg.norm(displacement_components)))
        step_values.append(
            float(np.linalg.norm(velocity_components)) * timestep_s
            + 0.5 * float(np.linalg.norm(acceleration_components)) * timestep_s**2
        )
    return max(displacement_values), max(step_values)


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
    assert "episode_time_s" not in payload
    runtime_payload = program.to_runtime_payload(episode_time_s=0.75)
    assert runtime_payload["episode_time_s"] == 0.75
    assert runtime_payload["ramp_type"] == "quintic_smoothstep"
    assert runtime_payload["program_frame"] == "deck"
    with pytest.raises(ValueError):
        program.to_runtime_payload(episode_time_s=-1.0)
    with pytest.raises(ValueError):
        program.to_runtime_payload(episode_time_s=float("nan"))
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
    assert fixture["provenance"]["source_profile_id"] == "shakebench.authored_v0_candidate"
    assert fixture["provenance"]["authored_decision_id"] == "phase-01-remediation-B-20260830"
    assert fixture["config"]["schema_version"] == 2
    assert fixture["provenance"]["fixture_sha256"] == _fixture_content_hash(fixture)
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
        sample_times = np.asarray(case["sample_time_s"], dtype=float)
        reference_motion = _reference_motion(program_payload, sample_times)
        np.testing.assert_allclose(reference_motion[0], np.asarray(case["motion"]["q"]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(reference_motion[1], np.asarray(case["motion"]["qdot"]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(reference_motion[2], np.asarray(case["motion"]["qdd"]), rtol=0.0, atol=0.0)
        production_motion = program.evaluate(sample_times)
        np.testing.assert_allclose(production_motion.q, reference_motion[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(production_motion.qdot, reference_motion[1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(production_motion.qdd, reference_motion[2], rtol=0.0, atol=0.0)
        gamma = case["gamma"]
        gamma_times = np.asarray(gamma["unit_replay"]["time_grid_s"], dtype=float)
        gamma_motion = _reference_motion(program_payload, gamma_times)
        gamma_point = np.asarray(gamma["point_offset_m"], dtype=float)
        gamma_normal = np.asarray(gamma["support_normal"], dtype=float)
        expected_gamma = _reference_point_acceleration(
            gamma_motion,
            gamma_point,
            gamma_normal,
            gamma["include_centripetal"],
        )
        assert gamma["gamma_commanded"] == pytest.approx(
            np.max(np.abs(expected_gamma)) / gamma["gravity_m_s2"],
            rel=0.0,
            abs=1e-14,
        )
        if case["t0"] != 0.0:
            origin_payload = dict(program_payload)
            origin_payload["line_phase_at_episode_zero"] = (
                np.asarray(program_payload["line_phase_at_episode_zero"])
                - np.asarray(program_payload["line_omega_rad_s"]) * case["t0"]
            ).tolist()
            shift_times = np.array([0.60, 0.83, 1.41], dtype=float)
            shifted = _reference_motion(program_payload, shift_times)
            translated = _reference_motion(origin_payload, shift_times + case["t0"])
            np.testing.assert_allclose(shifted[0], translated[0], rtol=0.0, atol=2e-14)
            np.testing.assert_allclose(shifted[1], translated[1], rtol=0.0, atol=2e-13)
            np.testing.assert_allclose(shifted[2], translated[2], rtol=0.0, atol=2e-11)
        assert program.max_frequency_hz < 8.87
    errors = json.loads(ERROR_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert np.isfinite(errors["max_error"])
    assert errors["max_error"] <= errors["error_tolerances"]["analytic_derivative_abs"]
    assert (
        errors["errors"]["six_axis_seed_19_t0.gamma_manual_abs_error"] <= errors["error_tolerances"]["gamma_manual_abs"]
    )
    assert (
        errors["errors"]["six_axis_seed_19_t0.common_t0_shift_abs_error"]
        <= errors["error_tolerances"]["common_t0_shift_abs"]
    )
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
    limits = fixture["safety_profile"]["limits"]
    for case in fixture["rejection_cases"]:
        assert case["passed"] is False
        assert case["expected_gate"] in case["report"]["violations"]
        program_payload = case["program"]
        timestep_s = 1.0 if case["name"] == "reject_solver_travel" else 0.001
        displacement, solver_step = _reference_safety_bounds(program_payload, limits, timestep_s)
        report_by_name = {item["name"]: item for item in case["report"]["checks"]}
        if case["expected_gate"] == "displacement":
            assert displacement > limits["max_displacement_m"]
            assert report_by_name["displacement"]["measured"] == pytest.approx(displacement)
        elif case["expected_gate"] == "frequency":
            frequencies = np.asarray(program_payload["line_omega_rad_s"], dtype=float) / (2.0 * np.pi)
            mask = np.asarray(program_payload["line_mask"], dtype=bool)
            assert np.max(frequencies[mask]) > limits["max_frequency_hz"]
        elif case["expected_gate"] == "non_ballistic":
            times = np.linspace(0.0, 2.0, 2001)
            motion = _reference_motion(program_payload, times)
            details = report_by_name["non_ballistic"]["details"]
            effective_normal = _reference_point_acceleration(
                motion,
                np.asarray(details["point_offset_m"]),
                np.asarray(details["support_normal"]),
                include_centripetal=True,
            )
            gamma = np.max(np.abs(effective_normal)) / 9.81
            assert gamma >= limits["max_gamma"]
            assert report_by_name["non_ballistic"]["measured"] == pytest.approx(gamma)
        elif case["expected_gate"] == "solver_travel":
            assert solver_step > limits["solver_clearance_m"] * limits["solver_step_fraction"]
            assert report_by_name["solver_travel"]["measured"] == pytest.approx(solver_step)


def test_golden_maintenance_default_is_validation_only() -> None:
    before = FIXTURE_PATH.read_bytes()
    result = validate_fixture(FIXTURE_PATH.parent)
    after = FIXTURE_PATH.read_bytes()
    assert result["updated"] is False
    assert before == after
