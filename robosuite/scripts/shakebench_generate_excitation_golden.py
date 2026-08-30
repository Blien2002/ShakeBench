"""Generate self-contained Phase 01 excitation golden fixtures.

Run from the repository root with::

    python -m robosuite.scripts.shakebench_generate_excitation_golden

The output is intentionally flat in ``tests/``.  No simulator is created and
the generated provenance points back to this script rather than to any
external implementation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from robosuite.utils.shakebench_calibration import calibrate_gamma
from robosuite.utils.shakebench_config import ShakeBenchConfig, config_hash
from robosuite.utils.shakebench_excitation import (
    AXES,
    DEFAULT_EXCITATION_CONFIG,
    ExcitationConfig,
    build_excitation_program,
    quintic_ramp,
)
from robosuite.utils.shakebench_safety import SafetyLimits, check_safety

SCRIPT_RELPATH = "robosuite/scripts/shakebench_generate_excitation_golden.py"
FIXTURE_NAME = "golden_shakebench_excitation_v0.json"
ERROR_SUMMARY_NAME = "golden_shakebench_excitation_error_summary.json"


def _config_hash_for(config: ExcitationConfig) -> str:
    envelope = ShakeBenchConfig(options={"excitation": config.to_dict()})
    return config_hash(envelope)


def _json_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float))


def _invariant_errors(program: Any, sample_times: np.ndarray, calibration: Any) -> dict[str, float]:
    """Calculate machine-readable residuals for the authored contracts."""

    line_amplitude_error = 0.0
    q_amplitude_error = 0.0
    for index, band in enumerate(program.bands):
        active = program.line_mask[index]
        expected_accel = program.level_scale * band.accel_rms * np.sqrt(2.0 / band.tones)
        if np.any(active):
            line_amplitude_error = max(
                line_amplitude_error,
                float(np.max(np.abs(program.line_accel_amplitude[index, active] - expected_accel))),
            )
            expected_q = program.line_accel_amplitude[index, active] / program.line_omega_rad_s[index, active] ** 2
            q_amplitude_error = max(
                q_amplitude_error,
                float(np.max(np.abs(program.line_q_amplitude[index, active] - expected_q))),
            )

    derivative_times = np.array([0.13, 0.31, 0.73, 1.17], dtype=float)
    h = 1e-6
    center = program.evaluate(derivative_times)
    plus = program.evaluate(derivative_times + h)
    minus = program.evaluate(derivative_times - h)
    derivative_qdot_error = float(np.max(np.abs(center.qdot - (plus.q - minus.q) / (2.0 * h))))
    derivative_qdd_error = float(np.max(np.abs(center.qdd - (plus.qdot - minus.qdot) / (2.0 * h))))

    origin = build_excitation_program(
        seed=program.seed,
        t0=0.0,
        level_scale=program.level_scale,
        active_axes=program.active_axes,
        config=program.config,
    )
    shifted_identity_error = 0.0
    if program.t0 != 0.0:
        post_ramp_times = np.array(
            [
                max(program.config.ramp_duration_s, 0.5),
                max(program.config.ramp_duration_s, 0.5) + 0.23,
                max(program.config.ramp_duration_s, 0.5) + 0.91,
            ]
        )
        shifted = program.evaluate(post_ramp_times)
        translated = origin.evaluate(post_ramp_times + program.t0)
        shifted_identity_error = max(
            float(np.max(np.abs(shifted.q - translated.q))),
            float(np.max(np.abs(shifted.qdot - translated.qdot))),
            float(np.max(np.abs(shifted.qdd - translated.qdd))),
        )

    ramp_zero = quintic_ramp(0.0, program.config.ramp_duration_s)
    ramp_end = quintic_ramp(program.config.ramp_duration_s, program.config.ramp_duration_s)
    ramp_c2_error = max(
        abs(float(ramp_zero[0])),
        abs(float(ramp_zero[1])),
        abs(float(ramp_zero[2])),
        abs(float(ramp_end[0]) - 1.0),
        abs(float(ramp_end[1])),
        abs(float(ramp_end[2])),
    )
    calibration_times = np.asarray(calibration.unit_replay["time_grid_s"], dtype=float)
    calibration_motion = program.evaluate(calibration_times)
    point = np.asarray(calibration.point_offset_m, dtype=float)
    manual_point_acceleration = calibration_motion.qdd[..., :3] + np.cross(
        calibration_motion.qdd[..., 3:],
        point,
    )
    manual_gamma = float(np.max(np.abs(manual_point_acceleration[..., 2]))) / calibration.gravity_m_s2
    return {
        "line_accel_amplitude_abs_error": line_amplitude_error,
        "line_q_amplitude_abs_error": q_amplitude_error,
        "analytic_qdot_central_difference_abs_error": derivative_qdot_error,
        "analytic_qdd_central_difference_abs_error": derivative_qdd_error,
        "common_t0_shift_abs_error": shifted_identity_error,
        "ramp_c2_boundary_abs_error": ramp_c2_error,
        "gamma_manual_abs_error": abs(calibration.gamma_commanded - manual_gamma),
        "max_frequency_excess_hz": max(0.0, program.max_frequency_hz - 8.87),
        "safety_failure": 0.0,
    }


def _make_case(
    name: str,
    *,
    seed: int,
    t0: float,
    level_scale: float,
    active_axes: list[str],
    config: ExcitationConfig,
    sample_times: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build one replay case and calculate its invariant residuals."""

    program = build_excitation_program(
        seed=seed,
        t0=t0,
        level_scale=level_scale,
        active_axes=active_axes,
        config=config,
    )
    motion = program.evaluate(sample_times)
    calibration_times = np.linspace(0.0, config.episode_duration_s, 4001, dtype=float)
    calibration = calibrate_gamma(program, time=calibration_times)
    safety = check_safety(
        program,
        limits=SafetyLimits(),
        timestep_s=0.001,
        time=calibration_times,
    )
    case = {
        "name": name,
        "seed": seed,
        "t0": t0,
        "level_scale": level_scale,
        "active_axes": active_axes,
        "config_hash": _config_hash_for(config),
        "sample_time_s": sample_times.tolist(),
        "program": program.to_dict(),
        "motion": motion.to_dict(),
        "gamma": calibration.to_dict(),
        "safety": safety.to_dict(),
    }
    invariant_errors = _invariant_errors(program, sample_times, calibration)
    invariant_errors["safety_failure"] = 0.0 if safety.passed else 1.0
    errors = {
        f"{name}.finite_program": (
            0.0
            if all(
                np.all(np.isfinite(array))
                for array in (
                    program.line_accel_amplitude,
                    program.line_omega_rad_s,
                    program.line_phase_at_episode_zero,
                )
            )
            else 1.0
        ),
        **{f"{name}.{key}": value for key, value in invariant_errors.items()},
    }
    case["invariant_errors"] = invariant_errors
    return case, errors


def _make_rejection_case(
    name: str,
    *,
    program: Any,
    config: ExcitationConfig,
    timestep_s: float,
    expected_gate: str,
) -> dict[str, Any]:
    """Build one deliberately unsafe case and record the rejected gate."""

    report = check_safety(program, limits=SafetyLimits(), timestep_s=timestep_s, sample_count=2001)
    return {
        "name": name,
        "config_hash": _config_hash_for(config),
        "expected_gate": expected_gate,
        "passed": report.passed,
        "report": report.to_dict(),
    }


def build_fixture(*, generated_at_utc: str | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    """Build the deterministic Phase 01 fixture and error map."""

    config = DEFAULT_EXCITATION_CONFIG
    sample_times = np.array([-0.10, 0.0, 0.25, 0.50, 1.25, 2.0], dtype=float)
    cases: list[dict[str, Any]] = []
    errors: dict[str, float] = {}
    case_specs = (
        ("six_axis_seed_7", 7, 0.0, 0.20, list(AXES)),
        ("six_axis_seed_19_t0", 19, 0.137, 0.35, list(AXES)),
        ("single_axis_tz", 7, 0.137, 0.20, ["tz"]),
        ("single_axis_ry", 23, 0.271, 0.15, ["ry"]),
    )
    for name, seed, t0, level_scale, active_axes in case_specs:
        case, case_errors = _make_case(
            name,
            seed=seed,
            t0=t0,
            level_scale=level_scale,
            active_axes=active_axes,
            config=config,
            sample_times=sample_times,
        )
        cases.append(case)
        errors.update(case_errors)

    high_displacement = build_excitation_program(seed=3, level_scale=20.0, config=config)
    high_ballistic = build_excitation_program(seed=3, level_scale=5.0, config=config)
    high_solver_travel = build_excitation_program(seed=3, level_scale=20.0, config=config)
    high_frequency_config = ExcitationConfig(frequency_scale=1.02)
    high_frequency = build_excitation_program(seed=3, config=high_frequency_config)
    rejection_cases = [
        _make_rejection_case(
            "reject_displacement",
            program=high_displacement,
            config=config,
            timestep_s=0.001,
            expected_gate="displacement",
        ),
        _make_rejection_case(
            "reject_non_ballistic",
            program=high_ballistic,
            config=config,
            timestep_s=0.001,
            expected_gate="non_ballistic",
        ),
        _make_rejection_case(
            "reject_frequency",
            program=high_frequency,
            config=high_frequency_config,
            timestep_s=0.001,
            expected_gate="frequency",
        ),
        _make_rejection_case(
            "reject_solver_travel",
            program=high_solver_travel,
            config=config,
            timestep_s=1.0,
            expected_gate="solver_travel",
        ),
    ]
    for rejection in rejection_cases:
        errors[f"{rejection['name']}.unexpected_pass"] = 1.0 if rejection["passed"] else 0.0
        errors[f"{rejection['name']}.expected_gate_missing"] = (
            0.0 if rejection["expected_gate"] in rejection["report"]["violations"] else 1.0
        )

    fixture = {
        "schema_id": "shakebench.excitation.golden",
        "schema_version": 1,
        "provenance": {
            "generator_script": SCRIPT_RELPATH,
            "config_hash": _config_hash_for(config),
            "generated_at_utc": generated_at_utc
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
        "config": config.to_dict(),
        "axis_schema": build_excitation_program(config=config).to_dict()["bands"],
        "cases": cases,
        "rejection_cases": rejection_cases,
    }
    return fixture, errors


def write_fixture(output_dir: Path, *, generated_at_utc: str | None = None) -> tuple[Path, Path]:
    """Write the flat golden fixture and machine-readable error summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture, errors = build_fixture(generated_at_utc=generated_at_utc)
    fixture_path = output_dir / FIXTURE_NAME
    error_path = output_dir / ERROR_SUMMARY_NAME
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    error_summary = {
        "schema_id": "shakebench.excitation.golden.errors",
        "schema_version": 1,
        "fixture": FIXTURE_NAME,
        "generator_script": SCRIPT_RELPATH,
        "config_hash": fixture["provenance"]["config_hash"],
        "errors": errors,
        "max_error": max(errors.values()) if errors else 0.0,
        "error_tolerances": {
            "exact_replay_abs": 0.0,
            "analytic_derivative_abs": 1e-7,
            "common_t0_shift_abs": 1e-12,
            "max_frequency_excess_hz": 0.0,
            "safety_failure": 0.0,
        },
    }
    error_path.write_text(
        json.dumps(error_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixture_path, error_path


def main(argv: list[str] | None = None) -> int:
    """Generate golden files from command-line options and return a status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests"),
        help="directory for flat golden files (default: tests)",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="optional fixed ISO-8601 timestamp for reproducible fixture generation",
    )
    args = parser.parse_args(argv)
    fixture_path, error_path = write_fixture(args.output_dir, generated_at_utc=args.generated_at_utc)
    print(json.dumps({"fixture": str(fixture_path), "error_summary": str(error_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
