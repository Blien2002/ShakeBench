"""Deck-probe artifact generation and verification.

Physics and trace primitives live in ``shakebench_probe_deck_driver``; this
module owns batch execution, artifact serialization, and the command-line UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from robosuite.scripts.shakebench_probe_deck_driver import (
    ARTIFACT_SCHEMA_ID,
    ARTIFACT_SCHEMA_VERSION,
    CANONICAL_LOAD_CASES,
    CONTROLLED_ARTIFACT_NAME,
    DEFAULT_THRESHOLDS,
    LOAD_CASE_EMPTY,
    LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY,
    REQUIRED_COVERAGE_FAMILIES,
    Phase02R4Thresholds,
    _actual_coordinates,
    _candidate_screen_reasons,
    _program_hash,
    _sine_command,
    _spectrum_line_entries,
    _zero_record_gate_passed,
    candidate_screen_passed,
    canonical_gamma_conformance,
    compute_gate_summary,
    minimum_line_spacing_hz,
    run_trace,
    select_candidate,
    sine_conformance,
    spectrum_conformance,
    spectrum_gate_reasons,
    synthetic_spectrum_estimator_validation,
)
from robosuite.utils.shakebench_calibration import calibrate_gamma, level_scale_for_gamma
from robosuite.utils.shakebench_deck import DeckDriverConfig, DeckDriverTrace, TRACE_FIELD_CONTRACT, TRACE_SCHEMA_ID, TRACE_SCHEMA_VERSION
from robosuite.utils.shakebench_excitation import AXES, build_excitation_program, excitation_profile_hash
from robosuite.utils.shakebench_safety import check_safety

def _canonical_payload_hash(payload: dict[str, Any]) -> str:
    """Hash artifact content while excluding only self-referential metadata.

    The update reason and previous file hash are part of the authenticated
    payload.  Only ``artifact_update.new_payload_sha256`` is omitted because
    it necessarily points back to this digest; the verifier checks that the
    pointer equals the authenticated digest.
    """

    content = dict(payload)
    content.pop("artifact_integrity", None)
    update = content.get("artifact_update")
    if isinstance(update, Mapping) and "new_payload_sha256" in update:
        update = dict(update)
        update.pop("new_payload_sha256", None)
        content["artifact_update"] = update
    encoded = json.dumps(_json_ready(content), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert mappings, tuples and NumPy scalars into JSON-safe values."""

    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            key = str(key)
            if (
                isinstance(item, (list, tuple, np.ndarray))
                and (key.endswith("_time_grid_s") or key in {"time_grid_s", "fit_time_grid_s"})
            ):
                values = np.asarray(item, dtype=float).reshape(-1)
                if values.size:
                    encoded = json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False)
                    converted[f"{key}_summary"] = {
                        "count": int(values.size),
                        "first_s": float(values[0]),
                        "last_s": float(values[-1]),
                        "median_step_s": float(np.median(np.diff(values))) if values.size > 1 else None,
                        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                    }
                else:
                    converted[f"{key}_summary"] = {"count": 0, "first_s": None, "last_s": None, "median_step_s": None, "sha256": hashlib.sha256(b"[]").hexdigest()}
            else:
                converted[key] = _json_ready(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finalize_artifact(
    result: dict[str, Any],
    *,
    previous_file_sha256: Optional[str] = None,
    update_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Attach deterministic integrity and explicit update provenance."""

    finalized = dict(result)
    finalized.pop("artifact_integrity", None)
    finalized.pop("artifact_update", None)
    if update_reason is not None:
        finalized["artifact_update"] = {
            "mode": "explicit_update",
            "reason": update_reason,
            "previous_file_sha256": previous_file_sha256,
            "new_payload_sha256": None,
        }
    payload_sha256 = _canonical_payload_hash(finalized)
    finalized["artifact_integrity"] = {
        "hash_algorithm": "sha256",
        "hash_scope": "canonical JSON excluding artifact_integrity and artifact_update.new_payload_sha256",
        "payload_sha256": payload_sha256,
    }
    if update_reason is not None:
        finalized["artifact_update"]["new_payload_sha256"] = payload_sha256
    return finalized


def _controlled_artifact_path() -> Path:
    """Return the repository-controlled Phase 02R artifact path."""

    return Path(__file__).resolve().parents[2] / "tests" / CONTROLLED_ARTIFACT_NAME


def _is_controlled_artifact(path: Path) -> bool:
    """Whether ``path`` is the protected committed artifact."""

    return path.resolve() == _controlled_artifact_path()



def _trace_summary(trace: DeckDriverTrace) -> dict[str, Any]:
    return {
        "samples": int(trace.sample_timestamps_s.size),
        "max_deck_tracking_pose_error": (
            float(np.max(np.abs(trace.deck_tracking_pose_error))) if trace.deck_tracking_pose_error.size else 0.0
        ),
        "max_weld_constraint_residual_raw": (
            float(np.max(np.abs(trace.weld_constraint_residual_raw)))
            if trace.weld_constraint_residual_raw.size
            else 0.0
        ),
        "max_weld_constraint_force_raw": (
            float(np.max(np.abs(trace.weld_constraint_force_raw)))
            if trace.weld_constraint_force_raw.size
            else 0.0
        ),
        "max_actual_twist": float(np.max(np.abs(trace.actual_twist))) if trace.actual_twist.size else 0.0,
        "max_actual_acceleration": (
            float(np.max(np.abs(trace.actual_acceleration))) if trace.actual_acceleration.size else 0.0
        ),
        "max_solver_iterations": int(np.max(trace.solver_iterations)) if trace.solver_iterations.size else 0,
        "warning_delta_sum": int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0,
        "warning_delta_max": int(np.max(trace.warning_number_delta)) if trace.warning_number_delta.size else 0,
    }


def _right_limit_input_provenance(trace: DeckDriverTrace) -> dict[str, Any]:
    """Summarize the tested left/right-limit timing fields for one record."""

    if trace.sample_time_s.size == 0:
        return {
            "status": "no_samples",
            "right_limit_target_written_before_refresh": False,
        }
    return {
        "status": "measured",
        "trace_schema_id": TRACE_SCHEMA_ID,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "convention": "integrate under q(t), write q(t+dt) before forward-only refresh, sample at t+dt",
        "right_limit_target_written_before_refresh": True,
        "integration_target_time_first_last_s": [
            float(trace.integration_target_time_s[0]),
            float(trace.integration_target_time_s[-1]),
        ],
        "sample_target_time_first_last_s": [
            float(trace.sample_target_time_s[0]),
            float(trace.sample_target_time_s[-1]),
        ],
        "integration_application_time_first_last_s": [
            float(trace.integration_application_time_s[0]),
            float(trace.integration_application_time_s[-1]),
        ],
        "sample_application_time_first_last_s": [
            float(trace.sample_application_time_s[0]),
            float(trace.sample_application_time_s[-1]),
        ],
        "sample_time_first_last_s": [float(trace.sample_time_s[0]), float(trace.sample_time_s[-1])],
        "sample_target_time_equals_sample_time": bool(
            np.array_equal(trace.sample_target_time_s, trace.sample_time_s)
        ),
        "sample_application_time_equals_sample_time": bool(
            np.array_equal(trace.sample_application_time_s, trace.sample_time_s)
        ),
        "sample_count": int(trace.sample_time_s.size),
    }


def _candidate_profile(
    candidate_id: str,
    *,
    dt_s: float,
    eq_solimp: Iterable[float],
    eq_solref_damping: float,
    frequency_sampling_limit_s: float,
    deck_config: DeckDriverConfig,
    thresholds: Phase02R4Thresholds,
) -> dict[str, Any]:
    eq_solref = [thresholds.positive_eq_solref_time_constant_min_factor_dt * dt_s, eq_solref_damping]
    frequency_passed = dt_s <= frequency_sampling_limit_s
    solref_passed = eq_solref[0] >= thresholds.positive_eq_solref_time_constant_min_factor_dt * dt_s
    return {
        "candidate_id": candidate_id,
        "pre_registered": True,
        "dt_s": float(dt_s),
        "eq_solref": eq_solref,
        "eq_solimp": list(eq_solimp),
        "deck_mass_kg": float(deck_config.deck_mass_kg),
        "deck_inertia_kg_m2": list(deck_config.deck_inertia_kg_m2),
        "frequency_sampling_condition": {
            "rule": "dt <= 1/(20*f_max)",
            "limit_s": float(frequency_sampling_limit_s),
            "passed": bool(frequency_passed),
        },
        "frequency_sampling_limit_s": float(frequency_sampling_limit_s),
        "frequency_sampling_passed": bool(frequency_passed),
        "eq_solref_time_constant_condition": {
            "rule": "eq_solref[0] >= factor * dt",
            "factor": thresholds.positive_eq_solref_time_constant_min_factor_dt,
            "minimum_s": thresholds.positive_eq_solref_time_constant_min_factor_dt * dt_s,
            "actual_s": eq_solref[0],
            "passed": bool(solref_passed),
        },
        "solref_minimum_s": thresholds.positive_eq_solref_time_constant_min_factor_dt * dt_s,
        "solref_relation_passed": bool(solref_passed),
        "selection_basis": "pre-registered physics-only screen; no task/controller success outcome",
        "changed_physics_variables": ["dt_s", "eq_solref[0]"],
        "eq_solref_damping_ratio": float(eq_solref_damping),
    }


def _screen_zero(
    *,
    dt: float,
    duration_s: float,
    config: DeckDriverConfig,
) -> dict[str, Any]:
    trace, audit = run_trace(dt=dt, duration_s=duration_s, config=config)
    metrics = _trace_summary(trace)
    record = {
        **metrics,
        "max_actual_pose": float(np.max(np.abs(_actual_coordinates(trace)))) if trace.actual_pose.size else 0.0,
        "audit": audit,
    }
    record["passed"] = _zero_record_gate_passed(record)
    record["status"] = "measured"
    return record


def _screen_single_axis(
    *,
    dt: float,
    duration_s: float,
    axis_index: int,
    frequency_hz: float,
    config: DeckDriverConfig,
    thresholds: Phase02R4Thresholds,
) -> dict[str, Any]:
    amplitude = 0.001 if axis_index < 3 else 0.0005
    trace, audit = run_trace(
        dt=dt,
        duration_s=duration_s,
        trajectory=_sine_command(axis_index, frequency_hz, amplitude),
        config=config,
    )
    conformance = sine_conformance(
        trace,
        axis_index=axis_index,
        frequency_hz=frequency_hz,
        amplitude=amplitude,
        discard_s=min(0.4, max(0.2, duration_s / 2.0)),
    )
    passed = (
        conformance["amplitude_relative_error"] <= thresholds.amplitude_relative_error
        and abs(conformance["phase_error_deg"]) <= thresholds.absolute_phase_error_deg
    )
    return {
        "axis": AXES[axis_index],
        "frequency_hz": frequency_hz,
        "coordinate_unit": "m" if axis_index < 3 else "rad",
        "audit": audit,
        "conformance": conformance,
        "diagnostics": _trace_summary(trace),
        "passed": bool(passed),
        "status": "measured",
        "failure_reasons": [] if passed else ["single_axis_amplitude_or_phase_gate_failed"],
    }



def run_probe_suite(
    *,
    dt: float = 0.0002,
    duration_s: float = 1.0,
    frequencies_hz: Iterable[float] = (5.0,),
    gammas: Iterable[float] = (0.15, 0.30, 0.50),
    spectrum_duration_s: Optional[float] = None,
    thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Run the R4 pre-registered screen, selection, and confirmatory matrix.

    The candidate profile is materialized before the first MuJoCo probe.  All
    candidates then receive the same physics-only screen.  Only after the
    deterministic selection is recorded does this function run the six
    confirmatory load records, each with a complete 64-line spectrum.
    """

    frequencies_hz = tuple(float(value) for value in frequencies_hz)
    gammas = tuple(float(value) for value in gammas)
    required_gammas = (0.15, 0.30, 0.50)
    if tuple(sorted(gammas)) != required_gammas:
        raise ValueError("R4 confirmatory matrix requires Gamma targets 0.15, 0.30, and 0.50")
    if not frequencies_hz:
        raise ValueError("frequencies_hz must contain at least one value")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not isinstance(thresholds, Phase02R4Thresholds):
        raise TypeError("thresholds must be a Phase02R4Thresholds instance")

    point_offset_m = (0.65, 0.0, 0.0)
    support_normal = (0.0, 0.0, 1.0)
    seed = 17
    t0 = 0.0
    default_solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
    base_config = DeckDriverConfig()
    unit_profile = build_excitation_program(seed=seed, t0=t0, level_scale=1.0)
    frequency_sampling_limit_s = 1.0 / (20.0 * unit_profile.max_frequency_hz)
    minimum_spacing = minimum_line_spacing_hz(unit_profile)
    beat_period_s = 1.0 / minimum_spacing
    required_fit_window_s = 2.0 * beat_period_s
    spectrum_discard_s = max(0.75, float(unit_profile.config.ramp_duration_s) + 0.25)
    if spectrum_duration_s is None:
        requested_spectrum_duration_s = spectrum_discard_s + required_fit_window_s + dt
    else:
        requested_spectrum_duration_s = float(spectrum_duration_s)
    if not np.isfinite(requested_spectrum_duration_s) or requested_spectrum_duration_s <= 0.0:
        raise ValueError("spectrum_duration_s must be finite and positive")
    spectrum_duration = max(requested_spectrum_duration_s, duration_s)
    calibration_steps = int(math.ceil(spectrum_duration / dt))
    calibration_start_index = max(1, int(math.ceil(spectrum_discard_s / dt)))
    calibration_grid = np.arange(calibration_start_index, calibration_steps + 1, dtype=float) * dt
    spectrum_target_gamma = 0.30
    spectrum_level_scale = level_scale_for_gamma(
        spectrum_target_gamma,
        seed=seed,
        t0=t0,
        time=calibration_grid,
        point_offset_m=point_offset_m,
        sample_count=calibration_grid.size,
    )
    spectrum_program = build_excitation_program(seed=seed, t0=t0, level_scale=spectrum_level_scale)
    screen_duration = max(0.5, min(duration_s, 0.5))
    # The conformance spectrum is sampled at a deterministic decimation of
    # the physics trace.  Even the coarsest candidate remains well above the
    # Nyquist requirement for the authored <8.87 Hz program; normal driver
    # tests keep the default stride=1 and therefore retain every physics row.
    spectrum_refresh_stride = 16

    candidate_grid = [
        _candidate_profile(
            candidate_id,
            dt_s=candidate_dt,
            eq_solimp=default_solimp,
            eq_solref_damping=0.5,
            frequency_sampling_limit_s=frequency_sampling_limit_s,
            deck_config=base_config,
            thresholds=thresholds,
        )
        for candidate_id, candidate_dt in (
            ("fine", dt / 2.0),
            ("nominal", dt),
            ("coarse", dt * 2.0),
        )
    ]
    pre_registered_grid = [dict(candidate) for candidate in candidate_grid]
    right_limit_input_convention = {
        "integration": "at t write q(t), then integrate x(t) to x(t+dt)",
        "sample": "at t+dt write q(t+dt), forward-refresh without integration, then sample",
        "actual_state": "qpos/qvel/time at t+dt",
        "actual_acceleration_and_weld": "right-limit quantities for x(t+dt), q(t+dt)",
        "trace_fields": [
            "integration_target_time_s",
            "sample_target_time_s",
            "integration_application_time_s",
            "sample_application_time_s",
            "sample_time_s",
        ],
    }
    result: dict[str, Any] = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "provenance": {
            "phase": "02R4",
            "status": "candidate_grid_pre_registered; screening_pending",
            "mujoco_version": mujoco.__version__,
            "trace_schema_id": TRACE_SCHEMA_ID,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_field_contract": dict(TRACE_FIELD_CONTRACT),
            "physics_profile_freeze": "deferred-to-phase-06",
            "deck_mass_inertia_status": "derived-not-frozen",
            "dt_s": float(dt),
            "seed": seed,
            "t0_s": t0,
            "excitation_profile_hash": excitation_profile_hash(unit_profile.config),
            "excitation_profile_id": unit_profile.config.to_dict()["profile_id"],
            "point_offset_m": list(point_offset_m),
            "support_normal_world": list(support_normal),
            "gamma_definition_include_centripetal": False,
            "post_integration_refresh": True,
            "right_limit_input_convention": right_limit_input_convention,
            "sample_time_grid_s": calibration_grid.tolist(),
            "time_convention": {
                "integration_target": "left-limit q(t)",
                "sample_target": "right-limit q(t+dt)",
                "application_timestamp": "actual mocap write data.time",
                "sample_timestamp": "post-integration data.time",
                "fit": "evaluate authored command at the same post-step sample timestamps; never shift actual trace",
            },
            "fit_convention": {
                "basis": "joint sine/cosine basis plus intercept per authored frequency line",
                "phase": "A*sin(omega*t + phase); atan2(cosine coefficient, sine coefficient)",
                "transient_discard_s": spectrum_discard_s,
                "sample_rate": "median inverse sample-time difference",
            },
            "thresholds": thresholds.to_dict(),
            "not_frozen": ["physics_dt", "deck_mass_kg", "deck_inertia_kg_m2", "eq_solref", "eq_solimp"],
            "candidate_grid_pre_registered": pre_registered_grid,
            "candidate_grid": candidate_grid,
            "selected_provisional_candidate_id": None,
            "candidate_selection": {
                "pre_registered": True,
                "selection_status": "pending_screening",
                "selected_candidate_id": None,
                "screening_completed": False,
                "confirmatory_started": False,
                "selection_rule": select_candidate([], thresholds=thresholds)["selection_rule"],
                "stage_order": ["candidate_grid_pre_registered"],
            },
            "minimum_line_spacing_hz": minimum_spacing,
            "minimum_line_beat_period_s": beat_period_s,
            "required_spectrum_fit_window_s": required_fit_window_s,
            "spectrum_transient_discard_s": spectrum_discard_s,
            "spectrum_refresh_stride": spectrum_refresh_stride,
        },
        "coverage_matrix": [
            {
                "case_family": "zero",
                "case_ids": ["zero"],
                "axes": list(AXES),
                "measurements": ["pose", "twist", "acceleration", "raw_weld_diagnostics", "warnings"],
            },
            {"case_family": "single_axis", "case_ids": list(AXES), "axes": list(AXES), "measurements": ["amplitude", "phase"]},
            {
                "case_family": "authored_spectrum",
                "case_ids": ["screening_target_Gamma_0.30"],
                "axes": list(AXES),
                "measurements": ["every_active_line_amplitude", "every_active_line_phase", "fit_residual", "conditioning"],
            },
            {
                "case_family": "target_gamma",
                "case_ids": [str(value) for value in gammas],
                "axes": list(AXES),
                "measurements": ["Gamma_commanded", "Gamma_deck_actual", "relative_error"],
            },
            {
                "case_family": "load",
                "case_ids": [
                    f"{target}::{load_case}" for target in gammas for load_case in CANONICAL_LOAD_CASES
                ],
                "axes": list(AXES),
                "measurements": ["full_64_line_spectrum", "Gamma_commanded", "Gamma_deck_actual", "right_limit_raw_weld_diagnostics", "load_provenance"],
            },
            {"case_family": "dt_convergence", "case_ids": ["dt_low", "dt_nominal", "dt_high"], "axes": list(AXES), "measurements": ["Gamma", "solver", "warnings"]},
            {"case_family": "solver_sensitivity", "case_ids": ["baseline", "eq_solref_time_constant", "eq_solimp_dmin"], "axes": list(AXES), "measurements": ["Gamma", "solver", "warnings"]},
        ],
        "zero": {},
        "single_axis": {},
        "spectrum": {},
        "gamma": {},
        "load_cases": list(CANONICAL_LOAD_CASES),
        "load_matrix": [],
        "dt_convergence": [],
        "sensitivity": [],
        "confirmatory": {"status": "not_started"},
        "phase03_handoff": "BLOCKED",
    }

    # Screening is deliberately the first executable block after profile
    # registration.  No confirmatory load and no selected ID is written before
    # all three candidates have produced raw screen evidence.
    for candidate in candidate_grid:
        candidate_config = DeckDriverConfig(
            eq_solref=tuple(candidate["eq_solref"]),
            eq_solimp=tuple(candidate["eq_solimp"]),
            physics_timestep_s=float(candidate["dt_s"]),
        )
        if not candidate.get("frequency_sampling_passed") or not candidate.get("solref_relation_passed"):
            candidate["status"] = "screen_rejected_preflight"
            candidate["screening"] = {
                "status": "not_run_preflight_rejected",
                "failure": {"type": "candidate_preflight", "message": "mathematical timing constraint failed"},
            }
            candidate["screening_passed"] = False
            candidate["rejection_reasons"] = _candidate_screen_reasons(candidate, thresholds=thresholds)
            continue
        try:
            zero = _screen_zero(dt=candidate["dt_s"], duration_s=screen_duration, config=candidate_config)
            singles = {
                axis: _screen_single_axis(
                    dt=candidate["dt_s"],
                    duration_s=screen_duration,
                    axis_index=axis_index,
                    frequency_hz=frequencies_hz[axis_index % len(frequencies_hz)],
                    config=candidate_config,
                    thresholds=thresholds,
                )
                for axis_index, axis in enumerate(AXES)
            }
            spectrum_trace, spectrum_audit = run_trace(
                dt=candidate["dt_s"],
                duration_s=spectrum_duration,
                trajectory=spectrum_program,
                refresh_stride=spectrum_refresh_stride,
                config=candidate_config,
            )
            spectrum_fit = spectrum_conformance(
                spectrum_trace,
                spectrum_program,
                discard_s=spectrum_discard_s,
                max_fit_samples=50000,
            )
            spectrum_reasons = spectrum_gate_reasons(spectrum_fit, thresholds=thresholds)
            spectrum_gamma = canonical_gamma_conformance(
                spectrum_trace,
                spectrum_program,
                config=candidate_config,
                point_offset_m=point_offset_m,
                support_normal=support_normal,
                discard_s=spectrum_discard_s,
            )
            if abs(spectrum_gamma["Gamma_commanded"] / spectrum_target_gamma - 1.0) > thresholds.gamma_relative_error:
                spectrum_reasons.append("authored_Gamma_commanded_differs_from_target")
            if spectrum_gamma["relative_error_abs"] > thresholds.gamma_relative_error:
                spectrum_reasons.append("target_gamma_screen_failed")
            spectrum_record = {
                "status": "measured",
                "target_Gamma_commanded": spectrum_target_gamma,
                "level_scale": spectrum_level_scale,
                "program_hash": _program_hash(spectrum_program),
                "profile_hash": excitation_profile_hash(spectrum_program.config),
                "conformance": spectrum_fit,
                "canonical_gamma": spectrum_gamma,
                "diagnostics": _trace_summary(spectrum_trace),
                "audit": spectrum_audit,
                "right_limit_input_provenance": _right_limit_input_provenance(spectrum_trace),
                "passed": not spectrum_reasons,
                "failure_reasons": list(spectrum_reasons),
            }
            candidate["screening"] = {
                "status": "measured",
                "zero": zero,
                "single_axis": singles,
                "target_gamma": spectrum_gamma,
                "authored_spectrum": spectrum_record,
            }
            candidate["status"] = "measured"
            candidate["physics_metrics"] = {
                "zero": zero,
                "single_axis": singles,
                "authored_spectrum": spectrum_record,
                "diagnostics": spectrum_record["diagnostics"],
            }
            candidate["screening_passed"] = candidate_screen_passed(candidate, thresholds=thresholds)
            candidate["rejection_reasons"] = _candidate_screen_reasons(candidate, thresholds=thresholds)
            candidate["screening"]["passed"] = candidate["screening_passed"]
            candidate["screening"]["rejection_reasons"] = list(candidate["rejection_reasons"])
        except Exception as exc:
            candidate["status"] = "screen_failed"
            candidate["screening"] = {
                "status": "failed",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
            candidate["screening_passed"] = False
            candidate["rejection_reasons"] = [f"screen_execution_failed:{type(exc).__name__}:{exc}"]

    decision = select_candidate(candidate_grid, thresholds=thresholds)
    for candidate in candidate_grid:
        candidate["selected_by_rule"] = candidate.get("candidate_id") == decision["selected_candidate_id"]
    selection_record = {
        **decision,
        "pre_registered": True,
        "screening_completed": True,
        "confirmatory_started": False,
        "stage_order": ["candidate_grid_pre_registered", "screening_completed", "selection_decided"],
    }
    result["provenance"]["candidate_selection"] = selection_record
    result["provenance"]["selected_provisional_candidate_id"] = decision["selected_candidate_id"]
    result["provenance"]["status"] = "screening_completed; selection_decided; derived-not-frozen"

    if decision["selected_candidate_id"] is None:
        first_screen = next(
            (candidate.get("screening", {}).get("authored_spectrum") for candidate in candidate_grid if candidate.get("status") == "measured"),
            None,
        )
        if isinstance(first_screen, Mapping):
            result["spectrum"] = dict(first_screen)
            result["spectrum"]["estimator_validation"] = synthetic_spectrum_estimator_validation(
                spectrum_program, sample_dt_s=min(candidate["dt_s"] for candidate in candidate_grid)
            )
        result["confirmatory"] = {
            "status": "not_run_no_candidate",
            "selection": selection_record,
            "raw_screening_preserved": True,
        }
        result["gate_summary"] = compute_gate_summary(result, thresholds=thresholds)
        result["provenance"]["status"] = result["gate_summary"]["overall_phase02r_status"] + "; derived-not-frozen"
        result["phase03_handoff"] = "BLOCKED"
        return result

    selected_candidate = next(
        candidate for candidate in candidate_grid if candidate.get("candidate_id") == decision["selected_candidate_id"]
    )
    selected_dt = float(selected_candidate["dt_s"])
    selected_config = DeckDriverConfig(
        eq_solref=tuple(selected_candidate["eq_solref"]),
        eq_solimp=tuple(selected_candidate["eq_solimp"]),
        physics_timestep_s=selected_dt,
    )
    selection_record["confirmatory_started"] = True
    selection_record["stage_order"].append("confirmatory_started")
    selection_record["selected_by_rule"] = {
        "candidate_id": decision["selected_candidate_id"],
        "rule": decision["selection_rule"],
        "screening_metrics_used": ["frequency_sampling_condition", "eq_solref_time_constant_condition", "zero", "six_single_axis", "64_line_authored_spectrum"],
        "task_success_used": False,
    }

    gamma_duration = max(1.0, duration_s)
    gamma_steps = int(math.ceil(spectrum_duration / selected_dt))
    gamma_start_index = max(1, int(math.ceil(spectrum_discard_s / selected_dt)))
    gamma_time_grid = np.arange(gamma_start_index, gamma_steps + 1, dtype=float) * selected_dt
    gamma_specs: dict[float, dict[str, Any]] = {}
    for target_gamma in required_gammas:
        level = level_scale_for_gamma(
            target_gamma,
            seed=seed,
            t0=t0,
            time=gamma_time_grid,
            point_offset_m=point_offset_m,
            sample_count=gamma_time_grid.size,
        )
        program = build_excitation_program(seed=seed, t0=t0, level_scale=level)
        phase01_calibration = calibrate_gamma(
            program,
            time=gamma_time_grid,
            point_offset_m=point_offset_m,
            support_normal=support_normal,
            include_centripetal=False,
        )
        safety = check_safety(
            program,
            timestep_s=selected_dt,
            time=gamma_time_grid,
            sample_count=gamma_time_grid.size,
            point_offset_m=point_offset_m,
        )
        gamma_specs[target_gamma] = {
            "target_Gamma_commanded": target_gamma,
            "level_scale": float(level),
            "seed": seed,
            "t0_s": t0,
            "profile_hash": excitation_profile_hash(program.config),
            "program_hash": _program_hash(program),
            "profile_id": program.config.to_dict()["profile_id"],
            "point_offset_m": list(point_offset_m),
            "support_normal_world": list(support_normal),
            "gamma_definition_include_centripetal": False,
            "calibration_time_grid_s": gamma_time_grid.tolist(),
            "phase01_calibration": {
                "gamma_commanded_generalized_definition": phase01_calibration.gamma_commanded,
                "unit_peak_factor": phase01_calibration.unit_peak_factor,
                "include_centripetal": phase01_calibration.include_centripetal,
                "unit_replay": dict(phase01_calibration.unit_replay),
            },
            "safety": safety.to_dict(),
            "program": program,
        }

    estimator_validation = synthetic_spectrum_estimator_validation(spectrum_program, sample_dt_s=selected_dt)
    for target_gamma in required_gammas:
        spec = gamma_specs[target_gamma]
        for load_case in CANONICAL_LOAD_CASES:
            case_id = f"{target_gamma}::{load_case}"
            base_record = {
                "case_id": case_id,
                "target_Gamma_commanded": target_gamma,
                "load_case": load_case,
                "seed": seed,
                "t0_s": t0,
                "level_scale": spec["level_scale"],
                "profile_hash": spec["profile_hash"],
                "program_hash": spec["program_hash"],
                "gamma_definition_include_centripetal": False,
                "right_limit_input_convention": right_limit_input_convention,
            }
            if not spec["safety"]["passed"]:
                base_record.update(
                    {
                        "status": "blocked_candidate_safety_rejected",
                        "rejection_evidence": spec["safety"],
                        "failure_reasons": ["candidate_safety_gate_failed"],
                    }
                )
                result["load_matrix"].append(base_record)
                continue
            try:
                trace, audit = run_trace(
                    dt=selected_dt,
                    duration_s=spectrum_duration,
                    trajectory=spec["program"],
                    refresh_stride=spectrum_refresh_stride,
                    config=selected_config,
                    load_case=load_case,
                )
                canonical = canonical_gamma_conformance(
                    trace,
                    spec["program"],
                    config=selected_config,
                    point_offset_m=point_offset_m,
                    support_normal=support_normal,
                    discard_s=spectrum_discard_s,
                )
                fit = spectrum_conformance(
                    trace,
                    spec["program"],
                    discard_s=spectrum_discard_s,
                    max_fit_samples=50000,
                )
                line_reasons = spectrum_gate_reasons(fit, thresholds=thresholds)
                panda_provenance = audit.get("load_provenance", {})
                if load_case == LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY and panda_provenance.get("panda_base_included") is not True:
                    line_reasons.append("panda_base_compiled_audit_failed")
                if target_gamma > 0.0 and abs(canonical["Gamma_commanded"] / target_gamma - 1.0) > thresholds.gamma_relative_error:
                    line_reasons.append("authored_Gamma_commanded_differs_from_target")
                if canonical["relative_error_abs"] > thresholds.gamma_relative_error:
                    line_reasons.append("Gamma_relative_error_exceeds_threshold")
                spectrum_record = {
                    "case_id": case_id,
                    "target_Gamma_commanded": target_gamma,
                    "load_case": load_case,
                    "program_hash": spec["program_hash"],
                    "profile_hash": spec["profile_hash"],
                    "conformance": fit,
                    "canonical_gamma": canonical,
                    "diagnostics": _trace_summary(trace),
                    "audit": audit,
                    "right_limit_input_provenance": _right_limit_input_provenance(trace),
                    "passed": not line_reasons,
                    "failure_reasons": line_reasons,
                }
                base_record.update(
                    {
                        "status": "measured",
                        "Gamma_commanded": canonical["Gamma_commanded"],
                        "Gamma_deck_actual": canonical["Gamma_deck_actual"],
                        "relative_error_abs": canonical["relative_error_abs"],
                        "canonical_gamma": canonical,
                        "spectrum": spectrum_record,
                        "conformance": fit,
                        "audit": audit,
                        "right_limit_input_provenance": spectrum_record["right_limit_input_provenance"],
                        "spectrum_gate": {"passed": not line_reasons, "failure_reasons": line_reasons},
                        "failure_reasons": line_reasons,
                    }
                )
            except Exception as exc:
                base_record.update(
                    {
                        "status": "failed",
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                        "failure_reasons": [f"confirmatory_execution_failed:{type(exc).__name__}:{exc}"],
                    }
                )
                if load_case == LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY:
                    base_record["load_provenance"] = {
                        "load_case": LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY,
                        "panda_base_included": False,
                        "compiled_model_assertions": {"passed": False, "failure": str(exc)},
                        "source": "in-repo Panda fixture compilation/reset failed; no table-only fallback",
                    }
            result["load_matrix"].append(base_record)

    for target_gamma in required_gammas:
        gamma_spec = gamma_specs[target_gamma]
        empty_record = next(
            record
            for record in result["load_matrix"]
            if record.get("target_Gamma_commanded") == target_gamma and record.get("load_case") == LOAD_CASE_EMPTY
        )
        gamma_record = {key: value for key, value in gamma_spec.items() if key != "program"}
        gamma_record["calibration_time_grid_s"] = gamma_spec["calibration_time_grid_s"]
        if empty_record.get("status") == "measured":
            gamma_record.update(
                {
                    "status": "measured",
                    "Gamma_commanded": empty_record["Gamma_commanded"],
                    "Gamma_deck_actual": empty_record["Gamma_deck_actual"],
                    "relative_error_abs": empty_record["relative_error_abs"],
                    "canonical_gamma": empty_record["canonical_gamma"],
                    "right_limit_input_provenance": empty_record["right_limit_input_provenance"],
                }
            )
        else:
            gamma_record.update({"status": "blocked_confirmatory_empty_record", "failure_reasons": empty_record.get("failure_reasons", [])})
        result["gamma"][str(target_gamma)] = gamma_record

    selected_screening = selected_candidate["screening"]
    result["zero"] = selected_screening["zero"]
    result["single_axis"] = selected_screening["single_axis"]
    empty_middle = next(
        record
        for record in result["load_matrix"]
        if record.get("target_Gamma_commanded") == 0.30 and record.get("load_case") == LOAD_CASE_EMPTY
    )
    if empty_middle.get("status") == "measured":
        result["spectrum"] = {
            "case_id": "authored_spectrum",
            "load_case": LOAD_CASE_EMPTY,
            "target_Gamma_commanded": 0.30,
            "seed": seed,
            "t0_s": t0,
            "level_scale": gamma_specs[0.30]["level_scale"],
            "profile_hash": gamma_specs[0.30]["profile_hash"],
            "program_hash": gamma_specs[0.30]["program_hash"],
            "program": gamma_specs[0.30]["program"].to_dict(),
            "conformance": empty_middle["conformance"],
            "canonical_gamma": empty_middle["canonical_gamma"],
            "audit": empty_middle["audit"],
            "diagnostics": empty_middle["spectrum"]["diagnostics"],
            "estimator_validation": estimator_validation,
            "right_limit_input_provenance": empty_middle["right_limit_input_provenance"],
        }

    reference_program = gamma_specs[0.30]["program"]
    dt_values = (selected_dt / 2.0, selected_dt, selected_dt * 2.0)
    fixed_tau = thresholds.positive_eq_solref_time_constant_min_factor_dt * max(dt_values)
    for index, case_dt in enumerate(dt_values):
        case_id = ("dt_low", "dt_nominal", "dt_high")[index]
        fixed_config = DeckDriverConfig(
            eq_solref=(fixed_tau, 1.0),
            eq_solimp=default_solimp,
            physics_timestep_s=case_dt,
        )
        try:
            trace, audit = run_trace(
                dt=case_dt,
                duration_s=min(gamma_duration, 0.5),
                trajectory=reference_program,
                load_case=LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY,
                config=fixed_config,
            )
            canonical = canonical_gamma_conformance(
                trace,
                reference_program,
                config=fixed_config,
                point_offset_m=point_offset_m,
                support_normal=support_normal,
                discard_s=0.0,
            )
            result["dt_convergence"].append(
                {
                    "case_id": case_id,
                    "status": "measured",
                    "dt_s": case_dt,
                    "eq_solref": [fixed_tau, 1.0],
                    "eq_solimp": list(default_solimp),
                    "solref_minimum_check": {"required": 2.0 * case_dt, "passed": fixed_tau >= 2.0 * case_dt},
                    "deck_mass_kg": fixed_config.deck_mass_kg,
                    "deck_inertia_kg_m2": list(fixed_config.deck_inertia_kg_m2),
                    "Gamma_commanded": canonical["Gamma_commanded"],
                    "Gamma_deck_actual": canonical["Gamma_deck_actual"],
                    "relative_error_abs": canonical["relative_error_abs"],
                    "diagnostics": _trace_summary(trace),
                    "audit": audit,
                }
            )
        except Exception as exc:
            result["dt_convergence"].append({"case_id": case_id, "status": "failed", "dt_s": case_dt, "failure": {"type": type(exc).__name__, "message": str(exc)}})

    sensitivity_cases = (
        ("baseline", "none", fixed_tau, default_solimp),
        ("eq_solref_time_constant", "eq_solref_time_constant", 2.0 * fixed_tau, default_solimp),
        ("eq_solimp_dmin", "eq_solimp_dmin", fixed_tau, (0.85, 0.95, 0.001, 0.5, 2.0)),
    )
    for case_id, changed_variable, case_tau, case_solimp in sensitivity_cases:
        sensitivity_config = DeckDriverConfig(
            eq_solref=(case_tau, 1.0),
            eq_solimp=case_solimp,
            physics_timestep_s=selected_dt,
        )
        try:
            trace, audit = run_trace(
                dt=selected_dt,
                duration_s=min(gamma_duration, 0.5),
                trajectory=reference_program,
                load_case=LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY,
                config=sensitivity_config,
            )
            canonical = canonical_gamma_conformance(
                trace,
                reference_program,
                config=sensitivity_config,
                point_offset_m=point_offset_m,
                support_normal=support_normal,
                discard_s=0.0,
            )
            result["sensitivity"].append(
                {
                    "case_id": case_id,
                    "status": "measured",
                    "changed_variable": changed_variable,
                    "dt_s": selected_dt,
                    "eq_solref": [case_tau, 1.0],
                    "eq_solimp": list(case_solimp),
                    "Gamma_commanded": canonical["Gamma_commanded"],
                    "Gamma_deck_actual": canonical["Gamma_deck_actual"],
                    "relative_error_abs": canonical["relative_error_abs"],
                    "diagnostics": _trace_summary(trace),
                    "audit": audit,
                }
            )
        except Exception as exc:
            result["sensitivity"].append({"case_id": case_id, "status": "failed", "changed_variable": changed_variable, "failure": {"type": type(exc).__name__, "message": str(exc)}})

    selection_record["confirmatory_completed"] = True
    selection_record["stage_order"].append("confirmatory_completed")
    result["provenance"]["candidate_selection"] = selection_record
    result["confirmatory"] = {
        "status": "completed",
        "selected_candidate_id": decision["selected_candidate_id"],
        "load_cases": list(CANONICAL_LOAD_CASES),
        "target_gammas": list(required_gammas),
        "matrix_record_count": len(result["load_matrix"]),
        "spectrum_lines_per_record": 64,
        "selection_locked_before_confirmatory": True,
    }
    result["gate_summary"] = compute_gate_summary(result, thresholds=thresholds)
    result["provenance"]["status"] = result["gate_summary"]["overall_phase02r_status"] + "; derived-not-frozen"
    result["phase03_handoff"] = "PASS" if result["gate_summary"]["overall_phase02r_status"] == "complete" else "BLOCKED"
    return result


def verify_artifact(path: str | Path) -> dict[str, Any]:
    """Verify an R4 artifact without changing it.

    ``integrity_valid`` describes schema/hash/provenance consistency.  Physics
    gate failure is reported independently so a blocked raw-evidence artifact
    remains auditable without being mislabeled as conformance success.
    """

    artifact_path = Path(path)
    errors: list[str] = []
    checks: dict[str, bool] = {}
    if not artifact_path.is_file():
        return {"passed": False, "path": str(artifact_path), "errors": [f"artifact does not exist: {artifact_path}"], "checks": checks}
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "path": str(artifact_path),
            "errors": [f"artifact is not valid JSON: {exc}"],
            "checks": checks,
            "file_sha256": _file_sha256(artifact_path),
        }
    if not isinstance(payload, dict):
        return {
            "passed": False,
            "path": str(artifact_path),
            "errors": ["artifact root must be a JSON object"],
            "checks": checks,
            "file_sha256": _file_sha256(artifact_path),
        }

    checks["schema"] = payload.get("schema_id") == ARTIFACT_SCHEMA_ID and payload.get("schema_version") == ARTIFACT_SCHEMA_VERSION
    if not checks["schema"]:
        errors.append("artifact schema_id/schema_version does not match Phase 02R4")
    try:
        computed_payload_hash = _canonical_payload_hash(payload)
    except (TypeError, ValueError) as exc:
        computed_payload_hash = None
        errors.append(f"artifact contains non-canonical JSON values: {exc}")
    integrity = payload.get("artifact_integrity")
    checks["integrity"] = (
        isinstance(integrity, Mapping)
        and integrity.get("hash_algorithm") == "sha256"
        and integrity.get("hash_scope") == "canonical JSON excluding artifact_integrity and artifact_update.new_payload_sha256"
        and integrity.get("payload_sha256") == computed_payload_hash
    )
    if not checks["integrity"]:
        errors.append("artifact payload integrity hash is missing or does not verify")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
        errors.append("provenance must be an object")
    required_provenance = (
        "phase",
        "mujoco_version",
        "trace_schema_id",
        "trace_schema_version",
        "trace_field_contract",
        "seed",
        "t0_s",
        "excitation_profile_hash",
        "point_offset_m",
        "support_normal_world",
        "time_convention",
        "fit_convention",
        "thresholds",
        "not_frozen",
        "minimum_line_spacing_hz",
        "minimum_line_beat_period_s",
        "required_spectrum_fit_window_s",
        "spectrum_transient_discard_s",
        "gamma_definition_include_centripetal",
        "post_integration_refresh",
        "right_limit_input_convention",
        "candidate_grid_pre_registered",
        "candidate_selection",
    )
    checks["command_provenance"] = all(key in provenance for key in required_provenance)
    if not checks["command_provenance"]:
        errors.append("command, selection, or right-limit provenance is incomplete")
    checks["post_integration_refresh"] = provenance.get("post_integration_refresh") is True
    if not checks["post_integration_refresh"]:
        errors.append("artifact does not declare the driver post-integration refresh")
    checks["centripetal_provenance"] = provenance.get("gamma_definition_include_centripetal") is False
    if not checks["centripetal_provenance"]:
        errors.append("Gamma centripetal provenance is not the canonical false boolean")
    checks["right_limit_input_convention"] = (
        isinstance(provenance.get("right_limit_input_convention"), Mapping)
        and "integration" in provenance["right_limit_input_convention"]
        and "sample" in provenance["right_limit_input_convention"]
        and "actual_acceleration_and_weld" in provenance["right_limit_input_convention"]
    )
    if not checks["right_limit_input_convention"]:
        errors.append("right-limit input convention is incomplete")
    checks["threshold_profile"] = provenance.get("thresholds") == DEFAULT_THRESHOLDS.to_dict()
    if not checks["threshold_profile"]:
        errors.append("artifact thresholds do not match the typed pre-registered profile")
    checks["trace_contract"] = (
        provenance.get("trace_schema_id") == TRACE_SCHEMA_ID
        and provenance.get("trace_schema_version") == TRACE_SCHEMA_VERSION
        and provenance.get("trace_field_contract") == dict(TRACE_FIELD_CONTRACT)
    )
    if not checks["trace_contract"]:
        errors.append("artifact trace field contract does not match the runtime contract")

    candidate_grid = provenance.get("candidate_grid")
    preregistered = provenance.get("candidate_grid_pre_registered")
    checks["candidate_grid"] = (
        isinstance(candidate_grid, list)
        and len(candidate_grid) >= 3
        and isinstance(preregistered, list)
        and [row.get("candidate_id") for row in preregistered if isinstance(row, Mapping)]
        == [row.get("candidate_id") for row in candidate_grid if isinstance(row, Mapping)]
        and all(
            isinstance(row, Mapping)
            and row.get("pre_registered") is True
            and row.get("status") in {"measured", "screen_failed", "screen_rejected_preflight"}
            and "screening" in row
            and "rejection_reasons" in row
            for row in candidate_grid
        )
    )
    if not checks["candidate_grid"]:
        errors.append("candidate grid lacks pre-registration or raw screen evidence")
    checks["candidate_profile_lock"] = bool(
        isinstance(candidate_grid, list)
        and isinstance(preregistered, list)
        and all(
            isinstance(row, Mapping)
            and isinstance(next((item for item in preregistered if isinstance(item, Mapping) and item.get("candidate_id") == row.get("candidate_id")), None), Mapping)
            and all(
                row.get(field) == next(
                    item for item in preregistered if isinstance(item, Mapping) and item.get("candidate_id") == row.get("candidate_id")
                ).get(field)
                for field in ("dt_s", "eq_solref", "eq_solimp", "deck_mass_kg", "deck_inertia_kg_m2")
            )
            for row in candidate_grid
        )
    )
    if not checks["candidate_profile_lock"]:
        errors.append("candidate execution records do not match the pre-registered profile")
    selection_record = provenance.get("candidate_selection")
    selected_candidate = provenance.get("selected_provisional_candidate_id")
    recomputed_selection = select_candidate(candidate_grid if isinstance(candidate_grid, list) else [], thresholds=DEFAULT_THRESHOLDS)
    checks["candidate_selection"] = (
        isinstance(selection_record, Mapping)
        and selection_record.get("selection_status") == recomputed_selection["selection_status"]
        and selection_record.get("selected_candidate_id") == recomputed_selection["selected_candidate_id"]
        and selected_candidate == recomputed_selection["selected_candidate_id"]
        and selection_record.get("screening_completed") is True
        and selection_record.get("pre_registered") is True
        and selection_record.get("selection_rule") == recomputed_selection["selection_rule"]
    )
    if not checks["candidate_selection"]:
        errors.append("selected candidate does not agree with the deterministic physics-screen rule")

    checks["candidate_screen_results"] = all(
        isinstance(row, Mapping)
        and row.get("screening_passed") is candidate_screen_passed(row, thresholds=DEFAULT_THRESHOLDS)
        and row.get("selected_by_rule") is (row.get("candidate_id") == selected_candidate)
        for row in candidate_grid
    )
    if not checks["candidate_screen_results"]:
        errors.append("candidate pass/fail or selected-by fields are stale relative to raw screen metrics")

    selection_blocked = recomputed_selection["selected_candidate_id"] is None
    if not selection_blocked:
        checks["selected_candidate_screen"] = any(
            row.get("candidate_id") == selected_candidate and candidate_screen_passed(row, thresholds=DEFAULT_THRESHOLDS)
            for row in candidate_grid
            if isinstance(row, Mapping)
        )
        if not checks["selected_candidate_screen"]:
            errors.append("stored selected candidate does not reproduce a passing screen")
    else:
        checks["selected_candidate_screen"] = False

    matrix = payload.get("coverage_matrix")
    families = {row.get("case_family") for row in matrix if isinstance(row, Mapping)} if isinstance(matrix, list) else set()
    checks["coverage_matrix"] = REQUIRED_COVERAGE_FAMILIES.issubset(families)
    if not checks["coverage_matrix"]:
        errors.append("coverage matrix is missing one or more required case families")

    expected_gammas = {"0.15", "0.3", "0.5"}
    gamma_records = payload.get("gamma")
    checks["gamma_candidates"] = isinstance(gamma_records, Mapping) and expected_gammas.issubset(gamma_records)
    if not checks["gamma_candidates"] and not selection_blocked:
        errors.append("selected artifact is missing required target Gamma records")
    if isinstance(gamma_records, Mapping) and not selection_blocked:
        for key in sorted(expected_gammas):
            record = gamma_records.get(key)
            if not isinstance(record, Mapping) or record.get("status") != "measured":
                errors.append(f"Gamma record {key} is not a measured record")
            elif not all(field in record for field in ("Gamma_commanded", "Gamma_deck_actual", "relative_error_abs", "right_limit_input_provenance")):
                errors.append(f"Gamma record {key} is missing canonical/right-limit metrics")

    load_matrix = payload.get("load_matrix")
    load_matrix = load_matrix if isinstance(load_matrix, list) else []
    expected_pairs = {(gamma, load_case) for gamma in (0.15, 0.3, 0.5) for load_case in CANONICAL_LOAD_CASES}
    actual_pairs = {
        (record.get("target_Gamma_commanded"), record.get("load_case"))
        for record in load_matrix
        if isinstance(record, Mapping)
    }
    checks["canonical_load_cases"] = payload.get("load_cases") == list(CANONICAL_LOAD_CASES)
    if not checks["canonical_load_cases"]:
        errors.append("artifact load cases are not the canonical empty/Panda-inclusive pair")
    checks["gamma_load_matrix"] = expected_pairs == actual_pairs and len(load_matrix) == len(expected_pairs)
    if not checks["gamma_load_matrix"] and not selection_blocked:
        errors.append("selected artifact is missing the complete 3x2 Gamma/load matrix")
    if not selection_blocked:
        for record in load_matrix:
            if not isinstance(record, Mapping):
                errors.append("Gamma/load matrix contains a non-object record")
                continue
            if record.get("status") != "measured":
                errors.append(f"confirmatory record {record.get('case_id')} is not measured")
                continue
            if not all(field in record for field in ("Gamma_commanded", "Gamma_deck_actual", "relative_error_abs", "right_limit_input_provenance", "audit", "spectrum")):
                errors.append(f"confirmatory record {record.get('case_id')} is missing canonical evidence")
            spectrum = record.get("spectrum", {})
            conformance = spectrum.get("conformance") if isinstance(spectrum, Mapping) else None
            if not isinstance(conformance, Mapping) or len(_spectrum_line_entries(conformance)) != 64:
                errors.append(f"confirmatory record {record.get('case_id')} lacks a full 64-line spectrum")
            right_limit = record.get("right_limit_input_provenance", {})
            if not isinstance(right_limit, Mapping) or right_limit.get("right_limit_target_written_before_refresh") is not True:
                errors.append(f"confirmatory record {record.get('case_id')} lacks right-limit input evidence")
            load_provenance = record.get("audit", {}).get("load_provenance", {}) if isinstance(record.get("audit"), Mapping) else {}
            if record.get("load_case") == LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY:
                if load_provenance.get("panda_base_included") is not True or load_provenance.get("compiled_model_assertions", {}).get("passed") is not True:
                    errors.append(f"confirmatory Panda/base audit failed for {record.get('case_id')}")

    top_spectrum = payload.get("spectrum", {})
    top_conformance = top_spectrum.get("conformance") if isinstance(top_spectrum, Mapping) else None
    checks["spectrum_lines"] = isinstance(top_conformance, Mapping) and len(_spectrum_line_entries(top_conformance)) == 64
    if not checks["spectrum_lines"] and not selection_blocked:
        errors.append("selected artifact top-level spectrum lacks 64 lines")
    checks["spectrum_resolution"] = (
        isinstance(top_conformance, Mapping)
        and top_conformance.get("fit_window_meets_resolution") is True
        and float(top_conformance.get("fit_window_s", 0.0)) >= float(top_conformance.get("required_fit_window_s", float("inf")))
    )
    if not checks["spectrum_resolution"] and not selection_blocked:
        errors.append("selected artifact spectrum fit window is shorter than two minimum beat periods")

    computed_gates = compute_gate_summary(payload, thresholds=DEFAULT_THRESHOLDS)
    stored_gates = payload.get("gate_summary")
    gate_names = [name for name, value in computed_gates.items() if isinstance(value, Mapping) and "passed" in value]
    checks["gate_consistency"] = (
        isinstance(stored_gates, Mapping)
        and all(
            isinstance(stored_gates.get(name), Mapping)
            and stored_gates[name].get("passed") is computed_gates[name].get("passed")
            for name in gate_names
        )
        and stored_gates.get("overall_phase02r_status") == computed_gates.get("overall_phase02r_status")
    )
    if not checks["gate_consistency"]:
        errors.append("stored gate summary does not agree with recomputed typed gates")
    handoff = payload.get("phase03_handoff")
    physics_gates_passed = all(computed_gates[name].get("passed") is True for name in gate_names)
    update_record = payload.get("artifact_update")
    if update_record is None:
        checks["update_provenance"] = True
    else:
        checks["update_provenance"] = (
            isinstance(update_record, Mapping)
            and bool(str(update_record.get("reason", "")).strip())
            and "previous_file_sha256" in update_record
            and update_record.get("new_payload_sha256") == (integrity.get("payload_sha256") if isinstance(integrity, Mapping) else None)
        )
        if not checks["update_provenance"]:
            errors.append("artifact update record lacks authenticated reason/previous/new hash")

    integrity_valid = not errors
    expected_handoff = "PASS" if integrity_valid and physics_gates_passed else "BLOCKED"
    checks["phase03_handoff"] = handoff == expected_handoff
    if not checks["phase03_handoff"]:
        errors.append("Phase 03 handoff does not agree with authenticated artifact and recomputed physics gates")
    integrity_valid = not errors
    reported_handoff = handoff if integrity_valid and physics_gates_passed else "BLOCKED"
    return {
        "passed": integrity_valid,
        "integrity_valid": integrity_valid,
        "physics_gates_passed": physics_gates_passed,
        "phase03_handoff": reported_handoff,
        "path": str(artifact_path),
        "schema_id": payload.get("schema_id"),
        "schema_version": payload.get("schema_version"),
        "checks": checks,
        "errors": errors,
        "file_sha256": _file_sha256(artifact_path),
        "payload_sha256": computed_payload_hash,
    }


def _write_artifact_atomic(result: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    """Verify a same-directory temporary artifact before atomic replacement."""

    output_path = output_path.resolve()
    temporary_path: Optional[Path] = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(result), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        verification = verify_artifact(temporary_path)
        if verification.get("integrity_valid") is not True:
            raise RuntimeError("temporary artifact failed verification: " + "; ".join(verification.get("errors", ())))
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_descriptor = os.open(str(output_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return verification
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None, help="optional JSON output path")
    parser.add_argument(
        "--verify-artifact",
        type=str,
        default=None,
        help="verify an existing artifact without running a probe or changing the file",
    )
    parser.add_argument("--update", action="store_true", help="authorize replacing the controlled artifact")
    parser.add_argument(
        "--accept-reference-change",
        action="store_true",
        help="authorize replacing the controlled artifact as an explicit reference change",
    )
    parser.add_argument("--reason", type=str, default="", help="non-empty reason required for artifact updates")
    parser.add_argument("--dt", type=float, default=0.0002)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--spectrum-duration",
        type=float,
        default=None,
        help="authored-spectrum fit duration in seconds (default: transient discard plus two minimum beat periods)",
    )
    args = parser.parse_args(argv)
    if args.verify_artifact is not None:
        if args.output is not None or args.update or args.accept_reference_change:
            print("--verify-artifact cannot be combined with output/update flags", file=sys.stderr)
            return 2
        summary = verify_artifact(args.verify_artifact)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["passed"] else 2

    if (args.update or args.accept_reference_change) and args.output is None:
        print("--update/--accept-reference-change requires --output", file=sys.stderr)
        return 2
    output_path = Path(args.output) if args.output is not None else None
    update_authorized = args.update or args.accept_reference_change
    if output_path is not None and _is_controlled_artifact(output_path) and not update_authorized:
        print(
            "controlled artifact refuses overwrite; use --update or --accept-reference-change with --reason",
            file=sys.stderr,
        )
        return 2
    reason = args.reason.strip()
    if update_authorized and not reason:
        print("artifact update requires a non-empty --reason", file=sys.stderr)
        return 2
    if output_path is not None and not output_path.parent.is_dir():
        print(f"artifact output parent does not exist: {output_path.parent}", file=sys.stderr)
        return 2

    result = run_probe_suite(dt=args.dt, duration_s=args.duration, spectrum_duration_s=args.spectrum_duration)
    previous_file_sha256 = _file_sha256(output_path) if output_path is not None and output_path.is_file() else None
    result = _finalize_artifact(
        result,
        previous_file_sha256=previous_file_sha256 if update_authorized else None,
        update_reason=reason if update_authorized else None,
    )
    if output_path is None:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True, allow_nan=False))
    else:
        try:
            _write_artifact_atomic(result, output_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"artifact write refused: {exc}", file=sys.stderr)
            return 2
    return 0
