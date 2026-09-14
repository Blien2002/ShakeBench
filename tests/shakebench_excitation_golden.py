"""Maintain the Phase 01 authored-excitation golden fixture.

The default invocation only validates the checked-in fixture.  Replacing a
fixture requires ``--update`` (or ``--accept-reference-change``) and a human
reason; this prevents a production-generator change from silently rewriting
its own acceptance input.

No simulator is created.  The fixture is a candidate authored profile, not
evidence that the old ShakeBench implementation was exactly reproduced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from robosuite.utils.shakebench_calibration import calibrate_gamma
from robosuite.utils.shakebench_config import (
    ShakeBenchConfig,
    config_hash,
)
from robosuite.utils.shakebench_excitation import (
    AUTHORED_DECISION_ID,
    AUTHORED_PROFILE_ID,
    AUTHORED_SPECTRUM_VERSION,
    AXES,
    DEFAULT_EXCITATION_CONFIG,
    ExcitationConfig,
    axis_schema,
    build_excitation_program,
    excitation_profile_hash,
    quintic_ramp,
)
from robosuite.utils.shakebench_safety import (
    DEFAULT_SAFETY_LIMITS,
    SAFETY_PROFILE_ID,
    SafetyLimits,
    check_safety,
    safety_profile_hash,
)

SCRIPT_RELPATH = "robosuite/scripts/shakebench_generate_excitation_golden.py"
FIXTURE_NAME = "golden_shakebench_excitation_v0.json"
ERROR_SUMMARY_NAME = "golden_shakebench_excitation_error_summary.json"
DEFAULT_GENERATED_AT_UTC = "2026-08-30T00:00:00Z"


def _candidate_provenance(
    config: ExcitationConfig,
    limits: SafetyLimits = DEFAULT_SAFETY_LIMITS,
) -> dict[str, Any]:
    """Build explicit draft provenance for the candidate authored profile."""

    return {
        "excitation_profile_id": AUTHORED_PROFILE_ID,
        "excitation_profile_hash": excitation_profile_hash(config),
        "authored_spectrum_version": AUTHORED_SPECTRUM_VERSION,
        "gravity_m_s2": config.gravity_m_s2,
        "ramp_duration_s": config.ramp_duration_s,
        "frequency_scale": config.frequency_scale,
        "gamma_point_offset_m": list(config.workpiece_point_offset_m),
        "safety_profile_id": SAFETY_PROFILE_ID,
        "safety_profile_hash": safety_profile_hash(limits),
        "safety_profile_limits": limits.to_dict(),
        "official_state_manifest_hash": "UNFROZEN",
        "freeze_status": "candidate",
    }


def _config_hash_for(config: ExcitationConfig, limits: SafetyLimits = DEFAULT_SAFETY_LIMITS) -> str:
    """Hash the explicit draft envelope used by a fixture."""

    envelope = ShakeBenchConfig(
        provenance=_candidate_provenance(config, limits),
        options={},
    )
    return config_hash(envelope)


def _canonical_json(value: Any) -> str:
    """Serialize a fixture value for deterministic content hashing."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fixture_sha256(fixture: dict[str, Any]) -> str:
    """Hash fixture content after removing its self-referential hash field."""

    payload = copy.deepcopy(fixture)
    provenance = payload.get("provenance", {})
    provenance.pop("fixture_sha256", None)
    provenance.pop("update_reason", None)
    provenance.pop("previous_fixture_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _invariant_errors(program: Any, calibration: Any, safety: Any) -> dict[str, float]:
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

    ramp_duration = program.config.ramp_duration_s
    ramp_zero = quintic_ramp(0.0, ramp_duration)
    ramp_end = quintic_ramp(ramp_duration, ramp_duration)
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
    point_acceleration = calibration_motion.qdd[..., :3] + np.cross(calibration_motion.qdd[..., 3:], point)
    angular_velocity = calibration_motion.qdot[..., 3:]
    if calibration.include_centripetal:
        point_acceleration += np.cross(angular_velocity, np.cross(angular_velocity, point))
    normal = np.asarray(calibration.support_normal, dtype=float)
    manual_gamma = (
        float(np.max(np.abs(np.einsum("...i,i->...", point_acceleration, normal)))) / calibration.gravity_m_s2
    )
    return {
        "line_accel_amplitude_abs_error": line_amplitude_error,
        "line_q_amplitude_abs_error": q_amplitude_error,
        "analytic_qdot_central_difference_abs_error": derivative_qdot_error,
        "analytic_qdd_central_difference_abs_error": derivative_qdd_error,
        "common_t0_shift_abs_error": shifted_identity_error,
        "ramp_c2_boundary_abs_error": ramp_c2_error,
        "gamma_manual_abs_error": abs(calibration.gamma_commanded - manual_gamma),
        "max_frequency_excess_hz": max(0.0, program.max_frequency_hz - 8.87),
        "safety_failure": 0.0 if safety.passed else 1.0,
    }


def _make_case(
    name: str,
    *,
    seed: int,
    t0: float,
    level_scale: float,
    active_axes: list[str],
    config: ExcitationConfig,
    limits: SafetyLimits,
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
    safety = check_safety(program, limits=limits, timestep_s=0.001, time=calibration_times)
    case = {
        "name": name,
        "seed": seed,
        "t0": t0,
        "level_scale": level_scale,
        "active_axes": active_axes,
        "config_hash": _config_hash_for(config, limits),
        "sample_time_s": sample_times.tolist(),
        "program": program.to_dict(),
        "motion": motion.to_dict(),
        "gamma": calibration.to_dict(),
        "safety": safety.to_dict(),
    }
    invariant_errors = _invariant_errors(program, calibration, safety)
    case["invariant_errors"] = invariant_errors
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
    return case, errors


def _make_rejection_case(
    name: str,
    *,
    program: Any,
    config: ExcitationConfig,
    limits: SafetyLimits,
    timestep_s: float,
    expected_gate: str,
) -> dict[str, Any]:
    """Build one unsafe case and preserve its authored program for audit."""

    report = check_safety(program, limits=limits, timestep_s=timestep_s, sample_count=2001)
    return {
        "name": name,
        "seed": program.seed,
        "t0": program.t0,
        "level_scale": program.level_scale,
        "active_axes": list(program.active_axes),
        "config_hash": _config_hash_for(config, limits),
        "program": program.to_dict(),
        "expected_gate": expected_gate,
        "passed": report.passed,
        "report": report.to_dict(),
    }


def build_fixture(*, generated_at_utc: str = DEFAULT_GENERATED_AT_UTC) -> tuple[dict[str, Any], dict[str, float]]:
    """Build the candidate fixture and machine-readable invariant errors."""

    config = DEFAULT_EXCITATION_CONFIG
    limits = DEFAULT_SAFETY_LIMITS
    sample_times = np.array([-0.10, 0.0, 0.25, 0.50, 1.25, 2.0], dtype=float)
    cases = []
    errors = {}
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
            limits=limits,
            sample_times=sample_times,
        )
        cases.append(case)
        errors.update(case_errors)

    high_displacement = build_excitation_program(seed=3, level_scale=20.0, config=config)
    high_ballistic = build_excitation_program(seed=3, level_scale=5.0, config=config)
    high_solver_travel = build_excitation_program(seed=3, level_scale=20.0, config=config)
    high_frequency_config = ExcitationConfig(frequency_scale=1.02)
    high_frequency = build_excitation_program(seed=3, config=high_frequency_config)
    rejection_specs = (
        ("reject_displacement", high_displacement, config, 0.001, "displacement"),
        ("reject_non_ballistic", high_ballistic, config, 0.001, "non_ballistic"),
        ("reject_frequency", high_frequency, high_frequency_config, 0.001, "frequency"),
        ("reject_solver_travel", high_solver_travel, config, 1.0, "solver_travel"),
    )
    rejection_cases = []
    for name, program, case_config, timestep_s, expected_gate in rejection_specs:
        rejection = _make_rejection_case(
            name,
            program=program,
            config=case_config,
            limits=limits,
            timestep_s=timestep_s,
            expected_gate=expected_gate,
        )
        rejection_cases.append(rejection)
        errors[f"{name}.unexpected_pass"] = 1.0 if rejection["passed"] else 0.0
        errors[f"{name}.expected_gate_missing"] = 0.0 if expected_gate in rejection["report"]["violations"] else 1.0

    fixture = {
        "schema_id": "shakebench.excitation.golden",
        "schema_version": 2,
        "provenance": {
            "generator_script": SCRIPT_RELPATH,
            "source_profile_id": AUTHORED_PROFILE_ID,
            "authored_spectrum_version": AUTHORED_SPECTRUM_VERSION,
            "authored_decision_id": AUTHORED_DECISION_ID,
            "config_hash": _config_hash_for(config, limits),
            "generated_at_utc": generated_at_utc,
        },
        "config": config.to_dict(),
        "configuration_envelope": ShakeBenchConfig(
            provenance=_candidate_provenance(config, limits),
            options={},
        ).to_dict(),
        "axis_schema": axis_schema(config),
        "safety_profile": {
            "profile_id": SAFETY_PROFILE_ID,
            "profile_hash": safety_profile_hash(limits),
            "limits": limits.to_dict(),
        },
        "cases": cases,
        "rejection_cases": rejection_cases,
    }
    fixture["provenance"]["fixture_sha256"] = _fixture_sha256(fixture)
    return fixture, errors


def _error_summary(fixture: dict[str, Any], errors: dict[str, float]) -> dict[str, Any]:
    """Build the machine-readable error summary for a fixture."""

    return {
        "schema_id": "shakebench.excitation.golden.errors",
        "schema_version": 2,
        "fixture": FIXTURE_NAME,
        "fixture_sha256": fixture["provenance"]["fixture_sha256"],
        "generator_script": SCRIPT_RELPATH,
        "config_hash": fixture["provenance"]["config_hash"],
        "errors": errors,
        "max_error": max(errors.values()) if errors else 0.0,
        "error_tolerances": {
            "exact_replay_abs": 0.0,
            "analytic_derivative_abs": 1e-7,
            "common_t0_shift_abs": 1e-12,
            "gamma_manual_abs": 1e-12,
            "max_frequency_excess_hz": 0.0,
            "safety_failure": 0.0,
        },
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Write stable pretty JSON to an existing output directory."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_fixture(
    output_dir: Path,
    *,
    generated_at_utc: str = DEFAULT_GENERATED_AT_UTC,
    reason: str,
) -> tuple[Path, Path, str | None, str]:
    """Explicitly replace fixture files and return old/new content hashes."""

    if not reason.strip():
        raise ValueError("fixture update requires a non-empty reason")
    fixture_path = output_dir / FIXTURE_NAME
    error_path = output_dir / ERROR_SUMMARY_NAME
    old_hash = None
    if fixture_path.exists():
        old_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        old_hash = old_fixture.get("provenance", {}).get("fixture_sha256") or _fixture_sha256(old_fixture)
    fixture, errors = build_fixture(generated_at_utc=generated_at_utc)
    new_hash = fixture["provenance"]["fixture_sha256"]
    fixture["provenance"]["update_reason"] = reason
    fixture["provenance"]["previous_fixture_sha256"] = old_hash
    # History fields are excluded from the content hash because they describe
    # why a reference was accepted, not the reference payload itself.
    content_hash_payload = copy.deepcopy(fixture)
    content_hash_payload["provenance"].pop("update_reason", None)
    content_hash_payload["provenance"].pop("previous_fixture_sha256", None)
    fixture["provenance"]["fixture_sha256"] = _fixture_sha256(content_hash_payload)
    new_hash = fixture["provenance"]["fixture_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_payload(fixture_path, fixture)
    _write_payload(error_path, _error_summary(fixture, errors))
    return fixture_path, error_path, old_hash, new_hash


def validate_fixture(output_dir: Path) -> dict[str, Any]:
    """Validate an existing fixture without changing either golden file."""

    fixture_path = output_dir / FIXTURE_NAME
    error_path = output_dir / ERROR_SUMMARY_NAME
    if not fixture_path.is_file() or not error_path.is_file():
        raise ValueError(f"missing fixture or error summary in {output_dir}")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    stored_hash = fixture.get("provenance", {}).get("fixture_sha256")
    if stored_hash != _fixture_sha256(fixture):
        raise ValueError("fixture_sha256 does not match canonical fixture content")
    stored_generated_at = fixture.get("provenance", {}).get("generated_at_utc", DEFAULT_GENERATED_AT_UTC)
    expected, errors = build_fixture(generated_at_utc=stored_generated_at)
    expected["provenance"].pop("fixture_sha256", None)
    expected_hash = _fixture_sha256(expected)
    expected["provenance"]["fixture_sha256"] = expected_hash
    actual_for_compare = copy.deepcopy(fixture)
    for payload in (actual_for_compare, expected):
        payload.get("provenance", {}).pop("update_reason", None)
        payload.get("provenance", {}).pop("previous_fixture_sha256", None)
    if actual_for_compare != expected:
        raise ValueError("checked-in fixture differs from the current production candidate; use --update with a reason")
    summary = json.loads(error_path.read_text(encoding="utf-8"))
    if summary.get("fixture_sha256") != stored_hash:
        raise ValueError("error summary fixture_sha256 does not match fixture")
    if summary.get("errors") != errors:
        raise ValueError("error summary does not match current invariant calculations")
    return {
        "fixture": str(fixture_path),
        "fixture_sha256": stored_hash,
        "max_error": summary.get("max_error"),
        "updated": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate or explicitly update the flat golden fixture."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tests"))
    parser.add_argument("--generated-at-utc", default=DEFAULT_GENERATED_AT_UTC)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--update", action="store_true", help="replace fixture; requires --reason")
    action.add_argument(
        "--accept-reference-change",
        action="store_true",
        help="explicitly accept and replace fixture; requires --reason",
    )
    parser.add_argument("--reason", default="", help="human reason required for fixture replacement")
    args = parser.parse_args(argv)
    try:
        if args.update or args.accept_reference_change:
            if not args.reason.strip():
                parser.error("--update/--accept-reference-change requires --reason")
            fixture_path, error_path, old_hash, new_hash = write_fixture(
                args.output_dir,
                generated_at_utc=args.generated_at_utc,
                reason=args.reason,
            )
            result = {
                "fixture": str(fixture_path),
                "error_summary": str(error_path),
                "old_fixture_sha256": old_hash,
                "new_fixture_sha256": new_hash,
                "updated": True,
            }
        else:
            result = validate_fixture(args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
