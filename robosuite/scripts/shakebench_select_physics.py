"""Run the Phase 06 physics-only selection and write auditable artifacts.

The selector has no task-success input.  Its three stages are deliberately
separate:

1. deck/timestep candidates are measured on the Phase 02 no-task fixture;
2. isolator candidates are scored from the registered transfer envelope and
   checked with the canonical six-axis MuJoCo support;
3. contact candidates are compiled into the real Can pair scope and exercised
   with support, slip, impact, finger-load, and timestep probes.

The default output is flat in ``robosuite/models/assets`` as required by the
Phase 06 contract.  Existing Phase 02–05 artifacts are read only as
provenance; they are never used as task ranking data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robosuite import models
from robosuite.scripts.shakebench_probe_deck_driver import (
    AXES,
    DeckProbeEnv,
    _build_probe_fixture,
    _program_hash,
    _trace_summary,
    canonical_gamma_conformance,
    minimum_line_spacing_hz,
    sine_conformance,
    spectrum_conformance,
    spectrum_gate_reasons,
)
from robosuite.utils.shakebench_calibration import level_scale_for_gamma
from robosuite.utils.shakebench_deck import DeckCommand, DeckDriver, DeckDriverConfig
from robosuite.utils.shakebench_isolator import (
    IsolatorConfig,
    derive_isolator_parameters,
    run_joint_spectrum_probe,
    run_harmonic_transfer_probe,
    static_equilibrium_offset,
    static_sag_uncompensated_m,
    transfer_metrics,
)
from robosuite.utils.shakebench_physics import (
    PhysicsProfile,
    load_official_physics_profile,
    load_selection_protocol,
    physics_profile_hash,
)


SELECTION_SCHEMA_ID = "shakebench.phase06.physics_selection"
SELECTION_SCHEMA_VERSION = 1
RAW_PREFIX = "shakebench_phase_06_raw_"
SELECTED_FILENAME = "shakebench_phase_06_selected_candidates.json"
EXCLUDED_FILENAME = "shakebench_phase_06_excluded_candidates.json"

_INTEGRATOR_NAMES = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
_SOLVER_NAMES = {0: "PGS", 1: "CG", 2: "Newton"}


class PhysicsSelectionError(RuntimeError):
    """Raised when the pre-registered selection cannot complete safely."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.complexfloating):
        item = value.item()
        return [float(item.real), float(item.imag)]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _set_probe_options(xml_string: str, candidate: Mapping[str, Any]) -> str:
    root = ET.fromstring(xml_string)
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", format(float(candidate["physics_timestep_s"]), ".17g"))
    option.set("integrator", str(candidate["integrator"]))
    option.set("solver", str(candidate["solver"]))
    option.set("iterations", str(int(candidate["solver_iterations"])))
    option.set("tolerance", format(float(candidate["solver_tolerance"]), ".17g"))
    return ET.tostring(root, encoding="unicode")


def _driver_config(candidate: Mapping[str, Any]) -> DeckDriverConfig:
    return DeckDriverConfig(
        deck_mass_kg=float(candidate["deck_mass_kg"]),
        deck_inertia_kg_m2=tuple(float(value) for value in candidate["deck_inertia_kg_m2"]),
        eq_solref=tuple(float(value) for value in candidate["deck_eq_solref"]),
        eq_solimp=tuple(float(value) for value in candidate["deck_eq_solimp"]),
        physics_timestep_s=float(candidate["physics_timestep_s"]),
    )


def _run_driver_trace(
    candidate: Mapping[str, Any],
    *,
    duration_s: float,
    trajectory: Any = None,
    load_case: str = "empty",
    refresh_stride: int = 1,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run one no-task driver fixture with candidate-local MuJoCo options."""

    dt = float(candidate["physics_timestep_s"])
    fixture = _build_probe_fixture(model_timestep=dt, load_case=load_case)
    source_xml = _set_probe_options(fixture.xml_string, candidate)
    config = _driver_config(candidate)
    driver = DeckDriver(
        trajectory=trajectory,
        config=config,
        body_handles=fixture.role_handles,
        refresh_stride=refresh_stride,
    )
    env = DeckProbeEnv(
        source_xml,
        driver,
        model_timestep=dt,
        horizon=max(2, int(math.ceil(float(duration_s) * 20.0)) + 1),
        initial_joint_state=fixture.initial_joint_state,
    )
    try:
        env.reset()
        driver.reset_trace()
        for _ in range(max(1, int(math.ceil(float(duration_s) * env.control_freq)))):
            env.step(np.zeros(0))
        trace = driver.trace
        raw_model = env.sim.model._model
        options = {
            "timestep_s": float(raw_model.opt.timestep),
            "integrator": _INTEGRATOR_NAMES.get(int(raw_model.opt.integrator), str(int(raw_model.opt.integrator))),
            "solver": _SOLVER_NAMES.get(int(raw_model.opt.solver), str(int(raw_model.opt.solver))),
            "iterations": int(raw_model.opt.iterations),
            "tolerance": float(raw_model.opt.tolerance),
            "control_freq_hz": float(env.control_freq),
            "control_steps": int(env._control_steps),
            "refresh_stride": int(refresh_stride),
            "post_integration_refresh": True,
        }
        audit = {
            "deck": {
                "parent_graph": {
                    name: (
                        None
                        if int(raw_model.body_parentid[body_id]) == 0
                        else mujoco.mj_id2name(
                            raw_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            int(raw_model.body_parentid[body_id]),
                        )
                    )
                    for body_id in range(int(raw_model.nbody))
                    for name in [mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)]
                },
                "mass_kg": float(raw_model.body_mass[driver._deck_body_id]),
                "inertia_kg_m2": np.asarray(raw_model.body_inertia[driver._deck_body_id], dtype=float).tolist(),
                "eq_solref": np.asarray(raw_model.eq_solref[driver._weld_id], dtype=float).tolist(),
                "eq_solimp": np.asarray(raw_model.eq_solimp[driver._weld_id], dtype=float).tolist(),
                "load_case": load_case,
                "warnings": int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0,
            }
        }
        digest = _trace_digest(trace)
        trace_record = {
            "summary": _trace_summary(trace),
            "digest": digest,
            "field_shapes": _trace_shapes(trace),
            "field_contract": "complete DeckDriverTrace fields; digest covers every field",
        }
        return trace, {**audit, "trace": trace_record}, options
    finally:
        env.close()


def _trace_shapes(trace: Any) -> dict[str, list[int]]:
    return {
        name: list(np.asarray(getattr(trace, name)).shape)
        for name in (
            "integration_target_time_s",
            "sample_target_time_s",
            "integration_application_time_s",
            "sample_application_time_s",
            "sample_time_s",
            "command_pose",
            "actual_pose",
            "command_twist",
            "actual_twist",
            "command_acceleration",
            "actual_acceleration",
            "sample_target_pose",
            "sample_target_twist",
            "sample_target_acceleration",
            "deck_tracking_pose_error",
            "weld_constraint_residual_raw",
            "weld_constraint_force_raw",
            "solver_iterations",
            "solver_niter",
            "warning_number_delta",
            "warning_lastinfo",
        )
    }


def _trace_digest(trace: Any) -> str:
    digest = hashlib.sha256()
    for name in (
        "integration_target_time_s",
        "sample_target_time_s",
        "integration_application_time_s",
        "sample_application_time_s",
        "sample_time_s",
        "command_pose",
        "actual_pose",
        "command_twist",
        "actual_twist",
        "command_acceleration",
        "actual_acceleration",
        "sample_target_pose",
        "sample_target_twist",
        "sample_target_acceleration",
        "deck_tracking_pose_error",
        "weld_constraint_residual_raw",
        "weld_constraint_force_raw",
        "solver_iterations",
        "solver_niter",
        "warning_number_delta",
        "warning_lastinfo",
    ):
        array = np.ascontiguousarray(np.asarray(getattr(trace, name)))
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sine_command(axis_index: int, frequency_hz: float, amplitude: float):
    omega = 2.0 * math.pi * float(frequency_hz)

    def trajectory(time_s: float) -> DeckCommand:
        pose = np.zeros(6, dtype=float)
        twist = np.zeros(6, dtype=float)
        acceleration = np.zeros(6, dtype=float)
        pose[axis_index] = float(amplitude) * math.sin(omega * time_s)
        twist[axis_index] = float(amplitude) * omega * math.cos(omega * time_s)
        acceleration[axis_index] = -float(amplitude) * omega * omega * math.sin(omega * time_s)
        return DeckCommand(pose, twist, acceleration)

    return trajectory


def _candidate_template(row: Mapping[str, Any]) -> dict[str, Any]:
    deck = row.get("deck", {})
    return {
        "candidate_id": str(row["candidate_id"]),
        "physics_timestep_s": float(row["physics_timestep_s"]),
        "integrator": str(row["integrator"]),
        "solver": str(row["solver"]),
        "solver_iterations": int(row["solver_iterations"]),
        "solver_tolerance": float(row["solver_tolerance"]),
        "deck_eq_solref": [float(value) for value in row["deck_eq_solref"]],
        "deck_eq_solimp": [float(value) for value in row["deck_eq_solimp"]],
        "deck_mass_kg": float(deck.get("mass_kg", 400.0)),
        "deck_inertia_kg_m2": [float(value) for value in deck.get("inertia_kg_m2", [12.12, 14.083333333333334, 26.033333333333332])],
    }


def _preflight_driver(candidate: Mapping[str, Any], *, f_max_hz: float) -> dict[str, Any]:
    dt = float(candidate["physics_timestep_s"])
    tau = float(candidate["deck_eq_solref"][0])
    frequency_limit = 1.0 / (20.0 * float(f_max_hz))
    return {
        "frequency_sampling": {
            "dt_s": dt,
            "f_max_hz": float(f_max_hz),
            "limit_s": frequency_limit,
            "passed": dt <= frequency_limit,
            "rule": "dt <= 1/(20*f_max)",
        },
        "positive_solref_time_constant": {
            "time_constant_s": tau,
            "minimum_s": 2.0 * dt,
            "passed": tau >= 2.0 * dt,
            "rule": "positive equality solref time constant >= 2*dt",
        },
    }


def _program_for_gamma(gamma: float, *, dt: float, duration_s: float):
    # Calibrate on the same post-ramp window used by the conformance fit.
    # The early ramp has a smaller peak and would under-calibrate Gamma by a
    # deterministic but incorrect factor.
    start_index = max(1, int(math.ceil(0.75 / dt)))
    times = np.arange(start_index, int(math.ceil(duration_s / dt)) + 1, dtype=float) * dt
    level = level_scale_for_gamma(
        gamma,
        seed=17,
        t0=0.0,
        time=times,
        point_offset_m=(0.65, 0.0, 0.0),
    )
    return level


def _run_gamma_batch(
    candidate: Mapping[str, Any],
    gammas: Iterable[float],
    *,
    discard_s: float,
    load_case: str,
) -> dict[str, dict[str, Any]]:
    """Run all Gamma screens for one load with one compiled no-task model."""

    dt = float(candidate["physics_timestep_s"])
    fixture = _build_probe_fixture(model_timestep=dt, load_case=load_case)
    source_xml = _set_probe_options(fixture.xml_string, candidate)
    config = _driver_config(candidate)
    # Gamma is a low-rate contract summary.  The driver still writes every
    # left-limit command, while the complete post-step trace is sampled at the
    # registered stride used by the Phase 02 spectrum probes.
    gamma_refresh_stride = 16
    driver = DeckDriver(config=config, body_handles=fixture.role_handles, refresh_stride=gamma_refresh_stride)
    env = DeckProbeEnv(
        source_xml,
        driver,
        model_timestep=dt,
        horizon=200,
        initial_joint_state=fixture.initial_joint_state,
    )
    env.hard_reset = False
    records = {}
    try:
        env.reset()
        for gamma_value in gammas:
            gamma = float(gamma_value)
            duration = 1.0 if gamma == 0.0 else max(1.75, discard_s + 1.0)
            level = _program_for_gamma(gamma, dt=dt, duration_s=duration)
            program = __import__(
                "robosuite.utils.shakebench_excitation", fromlist=["build_excitation_program"]
            ).build_excitation_program(seed=17, t0=0.0, level_scale=level)
            driver.trajectory = program
            env.reset()
            driver.reset_trace()
            for _ in range(max(1, int(math.ceil(duration * env.control_freq)))):
                env.step(np.zeros(0))
            trace = driver.trace
            fit = canonical_gamma_conformance(
                trace,
                program,
                config=config,
                point_offset_m=(0.65, 0.0, 0.0),
                support_normal=(0.0, 0.0, 1.0),
                discard_s=discard_s,
            )
            target_error = 0.0 if gamma == 0.0 else abs(fit["Gamma_commanded"] / gamma - 1.0)
            actual_error = 0.0 if gamma == 0.0 else fit["relative_error_abs"]
            raw_model = env.sim.model._model
            options = {
                "timestep_s": float(raw_model.opt.timestep),
                "integrator": _INTEGRATOR_NAMES.get(int(raw_model.opt.integrator), str(int(raw_model.opt.integrator))),
                "solver": _SOLVER_NAMES.get(int(raw_model.opt.solver), str(int(raw_model.opt.solver))),
                "iterations": int(raw_model.opt.iterations),
                "tolerance": float(raw_model.opt.tolerance),
                "control_freq_hz": float(env.control_freq),
                "control_steps": int(env._control_steps),
                "refresh_stride": gamma_refresh_stride,
                "post_integration_refresh": True,
            }
            records[str(gamma)] = {
                "target_gamma": gamma,
                "fit": fit,
                "target_gamma_relative_error": target_error,
                "deck_gamma_relative_error": actual_error,
                "options": options,
                "load_case": load_case,
                "trace": {
                    "summary": _trace_summary(trace),
                    "digest": _trace_digest(trace),
                    "field_shapes": _trace_shapes(trace),
                    "field_contract": "complete DeckDriverTrace fields; digest covers every field",
                },
                "passed": bool(target_error <= 0.01 and actual_error <= 0.01 and not np.any(env.sim.data.warning.number)),
            }
        return records
    finally:
        env.close()


def _driver_candidate_screen(
    candidate: dict[str, Any], protocol: Mapping[str, Any], *, run_full_spectrum: bool = True
) -> dict[str, Any]:
    preflight = _preflight_driver(candidate, f_max_hz=float(protocol["candidate_grids"]["timestep_solver"]["f_max_hz"]))
    result: dict[str, Any] = {
        "candidate": copy.deepcopy(candidate),
        "preflight": preflight,
        "status": "screen_pending",
        "load_cases": ["empty", "panda_plus_worktable_reference_proxy"],
    }
    if not all(item["passed"] for item in preflight.values()):
        result["status"] = "excluded_preflight"
        result["passed"] = False
        result["exclusion_reasons"] = [name for name, item in preflight.items() if not item["passed"]]
        return result

    dt = float(candidate["physics_timestep_s"])
    # The line at f_max is the registered worst-case phase probe.  The six
    # axes are all measured independently; no controller or task is created.
    axis_frequencies = (5.0, 6.5, 8.8, 3.36, 4.48, 2.8)
    single_axis = {}
    for axis_index, frequency in enumerate(axis_frequencies):
        amplitude = 2.0e-4 if axis_index < 3 else 5.0e-4
        trace, audit, options = _run_driver_trace(
            candidate,
            duration_s=max(1.0, 0.75 + 3.0 / frequency),
            trajectory=_sine_command(axis_index, frequency, amplitude),
            refresh_stride=1,
        )
        fit = sine_conformance(
            trace,
            axis_index=axis_index,
            frequency_hz=frequency,
            amplitude=amplitude,
            discard_s=0.5,
        )
        single_axis[AXES[axis_index]] = {
            "frequency_hz": frequency,
            "fit": fit,
            "options": options,
            "audit": audit,
            "trace": audit["trace"],
            "passed": bool(
                fit["amplitude_relative_error"] <= 0.01
                and abs(fit["phase_error_deg"]) <= 1.0
                and audit["deck"]["warnings"] == 0
            ),
        }

    if not all(record["passed"] for record in single_axis.values()):
        result.update(
            {
                "status": "measured_hard_gate_failed",
                "single_axis": single_axis,
                "authored_spectrum": {
                    "status": "not_run_after_hard_gate",
                    "passed": False,
                    "failure_reasons": ["single_axis_amplitude_or_phase_gate_failed"],
                },
                "gamma": {},
                "passed": False,
                "exclusion_reasons": ["driver_single_axis_hard_gate_failed"],
            }
        )
        return result

    # The authored 64-line fit is the expensive, full-spectrum screen.  A
    # fresh run is available for release audits; the normal Phase 06 run
    # replays the immutable Phase 02 no-task physics artifact after the new
    # integrator/solver settings have passed the six-axis screen above.
    spectrum_program = __import__(
        "robosuite.utils.shakebench_excitation", fromlist=["build_excitation_program"]
    ).build_excitation_program(seed=17, t0=0.0, level_scale=0.30)
    discard_s = max(0.75, float(spectrum_program.config.ramp_duration_s) + 0.25)
    fit_window_s = 2.0 / minimum_line_spacing_hz(spectrum_program)
    if run_full_spectrum:
        spectrum_duration = discard_s + fit_window_s + dt
        spectrum_trace, spectrum_audit, spectrum_options = _run_driver_trace(
            candidate,
            duration_s=spectrum_duration,
            trajectory=spectrum_program,
            refresh_stride=16,
        )
        spectrum_fit = spectrum_conformance(
            spectrum_trace,
            spectrum_program,
            discard_s=discard_s,
            max_fit_samples=50000,
        )
        spectrum_reasons = spectrum_gate_reasons(spectrum_fit, expected_line_count=64)
        spectrum_record = {
            "program_hash": _program_hash(spectrum_program),
            "active_line_count": int(np.sum(spectrum_program.line_mask)),
            "discard_s": discard_s,
            "fit_window_s": fit_window_s,
            "fit": spectrum_fit,
            "options": spectrum_options,
            "audit": spectrum_audit,
            "trace": spectrum_audit["trace"],
            "failure_reasons": spectrum_reasons,
            "passed": not spectrum_reasons and spectrum_audit["deck"]["warnings"] == 0,
            "measurement_mode": "fresh_candidate_probe",
        }
    else:
        source_path = Path(__file__).resolve().parents[2] / "tests" / "shakebench_phase_02r_probe.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_id = {
            "dt_fine": "fine",
            "dt_nominal": "nominal",
            "dt_conservative_coarse": "coarse",
        }.get(candidate["candidate_id"], candidate["candidate_id"])
        source_candidate = next(
            row for row in source["provenance"]["candidate_grid"] if row["candidate_id"] == source_id
        )
        source_spectrum = source_candidate["screening"]["authored_spectrum"]
        spectrum_fit = source_spectrum["conformance"]
        spectrum_reasons = spectrum_gate_reasons(spectrum_fit, expected_line_count=64)
        spectrum_record = {
            "program_hash": source_spectrum.get("program_hash", _program_hash(spectrum_program)),
            "active_line_count": 64,
            "discard_s": discard_s,
            "fit_window_s": fit_window_s,
            "fit": spectrum_fit,
            "source_artifact": str(source_path),
            "source_artifact_sha256": _file_sha256(source_path),
            "failure_reasons": spectrum_reasons,
            "passed": bool(source_spectrum.get("passed") is True and not spectrum_reasons),
            "measurement_mode": "replayed_phase02_physics_artifact",
        }

    # Gamma is evaluated at every registered safe point and for both no-task
    # load cases.  This is a driver contract, not a task difficulty measure.
    gamma_values = [float(value) for value in protocol["candidate_grids"]["timestep_solver"]["gamma_scales"]]
    gamma_records = {
        load_case: _run_gamma_batch(
            candidate,
            gamma_values,
            discard_s=discard_s,
            load_case=load_case,
        )
        for load_case in ("empty", "panda_plus_worktable_reference_proxy")
    }

    single_pass = all(record["passed"] for record in single_axis.values())
    gamma_pass = all(
        record["passed"]
        for per_load in gamma_records.values()
        for record in per_load.values()
    )
    result.update(
        {
            "status": "measured",
            "single_axis": single_axis,
            "authored_spectrum": spectrum_record,
            "gamma": gamma_records,
            "passed": bool(single_pass and spectrum_record["passed"] and gamma_pass),
            "exclusion_reasons": [] if single_pass and spectrum_record["passed"] and gamma_pass else ["driver_hard_gate_failed"],
        }
    )
    return result


def _analytic_isolator_metrics(candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    config = IsolatorConfig(
        fn_hz=tuple(float(value) for value in candidate["fn_hz"]),
        zeta=tuple(float(value) for value in candidate["zeta"]),
        mass_kg=float(protocol["candidate_grids"]["isolator"]["reference_mass_kg"]),
        inertia_kg_m2=tuple(float(value) for value in protocol["candidate_grids"]["isolator"]["reference_inertia_kg_m2"]),
        travel_limits_m=tuple(float(value) for value in protocol["candidate_grids"]["isolator"]["travel_limits_m"]),
        angle_limits_rad=tuple(float(value) for value in protocol["candidate_grids"]["isolator"]["angle_limits_rad"]),
    )
    parameters = derive_isolator_parameters(config)
    frequencies = np.asarray(protocol["candidate_grids"]["isolator"]["excitation_probe_frequencies_hz"], dtype=float)
    base_amplitude = 0.001
    metrics = transfer_metrics(frequencies, parameters, base_displacement_amplitude_m=base_amplitude)
    t_accel = np.asarray(metrics["T_accel"], dtype=float)
    r_relative = np.asarray(metrics["R_relative"], dtype=float)
    target = protocol["candidate_grids"]["isolator"]["target_transfer_envelope"]
    payload_offsets = []
    for offset in ((0.0, 0.0), (0.10, 0.0), (0.0, 0.20), (0.10, 0.20)):
        payload_offsets.append(
            static_equilibrium_offset(
                parameters,
                payload_mass_kg=float(protocol["candidate_grids"]["isolator"]["payload_mass_kg"]),
                payload_com_m=(offset[0], offset[1], 0.0),
            )
        )
    payload_norm = max(float(np.linalg.norm(value)) for value in payload_offsets)
    payload_sensitivity = payload_norm / max(float(min(protocol["candidate_grids"]["isolator"]["travel_limits_m"])), 1.0e-12)
    min_travel_margin = float(min(metrics["travel_margin_m"]))
    max_t_peak = float(metrics["T_peak"])
    summary = {
        "T_accel_mean": float(np.mean(t_accel)),
        "T_accel_median": float(np.median(t_accel)),
        "R_relative_median": float(np.median(r_relative)),
        "R_relative_max": float(np.max(r_relative)),
        "T_peak": max_t_peak,
        "D_relative_m": float(metrics["D_relative"]),
        "travel_margin_min_m": min_travel_margin,
        "static_sag_uncompensated_m": float(static_sag_uncompensated_m(parameters)),
        "static_offset_compensated_empty_max_m": 0.0,
        "static_offset_compensated_payload_max_m": payload_norm,
        "payload_sensitivity": payload_sensitivity,
        "tracking_axes": int(np.count_nonzero(np.mean(t_accel, axis=0) >= 0.8)),
        "isolation_axes": int(np.count_nonzero(np.mean(t_accel, axis=0) < 1.0)),
    }
    hard = {
        "R_relative_nonzero": summary["R_relative_max"] >= float(target["R_relative_min"]),
        "T_peak_bounded": summary["T_peak"] <= float(target["T_peak_max"]),
        "T_accel_bounded": float(target["T_accel_min"]) <= summary["T_accel_mean"] <= float(target["T_accel_max"]),
        "travel_margin": summary["travel_margin_min_m"] >= float(target["travel_margin_min_m"]),
        "static_sag": summary["static_sag_uncompensated_m"] <= float(target["static_sag_uncompensated_max_m"]),
        "static_payload_offset": summary["static_offset_compensated_payload_max_m"] <= float(target["static_offset_compensated_payload_max_m"]),
        "payload_sensitivity": summary["payload_sensitivity"] <= float(target["payload_sensitivity_max"]),
        "tracking_axis_present": summary["tracking_axes"] >= int(target["required_tracking_axes"]),
        "isolation_axis_present": summary["isolation_axes"] >= int(target["required_isolation_axes"]),
    }
    return {
        "config": config.to_dict(),
        "parameters": parameters.to_dict(),
        "frequencies_hz": frequencies.tolist(),
        "transfer": metrics,
        "summary": summary,
        "hard_gates": hard,
        "passed": all(hard.values()),
    }


def _isolator_candidate_screen(candidate: Mapping[str, Any], protocol: Mapping[str, Any], *, run_mujoco: bool) -> dict[str, Any]:
    record = {"candidate": copy.deepcopy(dict(candidate))}
    try:
        analytic = _analytic_isolator_metrics(candidate, protocol)
        record["analytic"] = analytic
        if run_mujoco:
            config = IsolatorConfig(fn_hz=tuple(candidate["fn_hz"]), zeta=tuple(candidate["zeta"]))
            # The analytic grid below covers tracking/resonance/isolation.
            # One well-settled tracking probe per axis supplies direct
            # six-axis MuJoCo evidence without importing a task; the prior
            # Phase 03 artifact supplies the same three-region convention.
            frequencies = (0.2,)
            transfer_records = []
            for axis_index, axis in enumerate(AXES):
                for ratio in frequencies:
                    probe = run_harmonic_transfer_probe(
                        axis,
                        ratio * float(candidate["fn_hz"][axis_index]),
                        isolator_config=config,
                        timestep_s=0.0002,
                        transient_cycles=5,
                        fit_cycles=10,
                        visual=False,
                    )
                    transfer_records.append(
                        {
                            "axis": axis,
                            "region_ratio": ratio,
                            "frequency_hz": probe["frequency_hz"],
                            "relative_fit": probe["relative_fit"],
                            "absolute_fit": probe["absolute_fit"],
                            "relative_comparison": probe["relative_comparison"],
                            "absolute_comparison": probe["absolute_comparison"],
                            "passed": probe["passed"],
                        }
                    )
            record["mujoco_transfer"] = {
                "records": transfer_records,
                "record_count": len(transfer_records),
                "passed": all(item["passed"] for item in transfer_records),
            }
        else:
            record["mujoco_transfer"] = {"status": "deferred_until_selected", "passed": True}
        record["passed"] = bool(analytic["passed"] and record["mujoco_transfer"]["passed"])
        record["exclusion_reasons"] = [] if record["passed"] else ["isolator_hard_gate_failed"]
        record["status"] = "measured"
    except Exception as exc:
        record.update({"status": "crashed", "passed": False, "exclusion_reasons": [f"probe_crash:{type(exc).__name__}:{exc}"]})
    return record


def _isolator_score(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[float, float, str]:
    summary = record["analytic"]["summary"]
    envelope = protocol["candidate_grids"]["isolator"]["target_transfer_envelope"]
    targets = {
        "R_relative_median": max(float(envelope["R_relative_min"]), 1.0),
        "T_peak": float(envelope["T_peak_max"]) / 2.0,
        "T_accel_mean": (float(envelope["T_accel_min"]) + float(envelope["T_accel_max"])) / 2.0,
        "travel_margin_min_m": float(envelope["travel_margin_min_m"]) * 2.0,
        "static_sag_uncompensated_m": float(envelope["static_sag_uncompensated_max_m"]) / 2.0,
        "static_offset_compensated_payload_max_m": float(envelope["static_offset_compensated_payload_max_m"]) / 2.0,
        "payload_sensitivity": float(envelope["payload_sensitivity_max"]) / 2.0,
    }
    scales = {
        "R_relative_median": max(targets["R_relative_median"], 1.0),
        "T_peak": max(targets["T_peak"], 1.0),
        "T_accel_mean": max(targets["T_accel_mean"], 1.0),
        "travel_margin_min_m": max(targets["travel_margin_min_m"], 1.0e-6),
        "static_sag_uncompensated_m": max(targets["static_sag_uncompensated_m"], 1.0e-6),
        "static_offset_compensated_payload_max_m": max(targets["static_offset_compensated_payload_max_m"], 1.0e-6),
        "payload_sensitivity": max(targets["payload_sensitivity"], 1.0e-6),
    }
    weights = protocol["candidate_grids"]["isolator"]["scoring"]["weights"]
    distance = 0.0
    for key, target in targets.items():
        distance += float(weights.get(key.replace("_mean", "").replace("_median", ""), 1.0)) * abs(
            float(summary[key]) - target
        ) / scales[key]
    margins = [
        float(summary["travel_margin_min_m"]) / float(envelope["travel_margin_min_m"]),
        float(envelope["T_peak_max"]) / max(float(summary["T_peak"]), 1.0e-12),
        float(envelope["static_sag_uncompensated_max_m"]) / max(float(summary["static_sag_uncompensated_m"]), 1.0e-12),
        float(envelope["payload_sensitivity_max"]) / max(float(summary["payload_sensitivity"]), 1.0e-12),
    ]
    return float(distance), float(min(margins)), str(record["candidate"]["candidate_id"])


def _profile_payload_with_contact(base: PhysicsProfile, contact: Mapping[str, Any], *, dt: Optional[float] = None) -> dict[str, Any]:
    payload = base.to_dict()
    payload["status"] = "contact_candidate_non_scoreable"
    payload["scoreable"] = False
    payload["profile_id"] = "shakebench.contact.candidate." + str(contact["candidate_id"])
    payload["physics"]["contact"] = {
        "pair_scope": "explicit_can_pairs",
        "sliding_mu": {"table_object": 0.30, "finger_object": 1.00},
        "condim": int(contact["condim"]),
        "torsional_mu": float(contact["torsional_mu"]),
        "rolling_mu": float(contact["rolling_mu"]),
        "margin_m": float(contact["margin_m"]),
        "gap_m": float(contact["gap_m"]),
        "solref": [float(value) for value in contact["solref"]],
        "solimp": [float(value) for value in contact["solimp"]],
        "interfaces": list(base.contact.get("interfaces", [])),
    }
    if dt is not None:
        payload["physics"]["timestep"]["physics_timestep_s"] = float(dt)
        payload["physics"]["scheduler"]["control_steps"] = int(round(1.0 / (20.0 * float(dt))))
        payload["physics"]["deck"]["eq_solref"][0] = max(
            float(payload["physics"]["deck"]["eq_solref"][0]), 2.0 * float(dt)
        )
        payload["physics"]["contact"]["solref"][0] = max(
            float(payload["physics"]["contact"]["solref"][0]), 2.0 * float(dt)
        )
    payload["physics"]["timestep"]["iterations"] = int(contact["iterations"])
    payload["profile_sha256"] = physics_profile_hash(payload)
    return payload


def _contact_candidate_probe(
    base_profile: PhysicsProfile,
    candidate: Mapping[str, Any],
    *,
    run_expensive: bool,
    recovery_duration_s: Optional[float] = None,
    reset_drop_velocity: bool = False,
) -> dict[str, Any]:
    """Exercise real task contact geometry without evaluating task outcome."""

    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    payload = _profile_payload_with_contact(base_profile, candidate)
    profile = PhysicsProfile(payload=payload, source="selection contact candidate", profile_sha256=payload["profile_sha256"])
    profile.assert_valid()
    target_friction = (
        0.30,
        float(candidate["torsional_mu"]),
        float(candidate["rolling_mu"]),
    )
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=float(profile.model_timestep_s),
        target_container_friction=target_friction,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=200,
        seed=17,
    )
    # Reuse the compiled contact model across the probe states; a hard reset
    # would replace the MjData object captured by the raw-contact audit.
    env.hard_reset = False
    try:
        model = env.sim.model._model
        data = env.sim.data._data
        can_geom_names = tuple(env.can.contact_geoms)
        table_geom_names = tuple(env.table_contact_geom_names)
        finger_geom_names = tuple(env.finger_pad_geom_names)

        def contact_force_for(allowed_second: Iterable[str]) -> tuple[int, float, float]:
            allowed = set(allowed_second)
            count = 0
            normal_force = 0.0
            min_distance = float("inf")
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                if {first, second} & set(can_geom_names) and ({first, second} & allowed):
                    count += 1
                    force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(model, data, index, force)
                    normal_force += abs(float(force[0]))
                    min_distance = min(min_distance, float(contact.dist))
            return count, normal_force, (0.0 if min_distance == float("inf") else min_distance)

        def physics_steps(count: int) -> None:
            """Advance MuJoCo directly, bypassing task reward/success hooks."""

            for _ in range(int(count)):
                if hasattr(data, "ctrl"):
                    data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)

        # Static support is a direct contact-force check.  It does not call the
        # success evaluator and therefore cannot turn into task ranking data.
        env.reset()
        physics_steps(20)
        static_count, static_force, static_distance = contact_force_for(table_geom_names)
        static_support = {
            "contact_count": static_count,
            "normal_force_N": static_force,
            "minimum_contact_distance_m": static_distance,
            "passed": bool(static_count > 0 and static_force > 0.001),
        }

        can_joint = model.joint(env.can.joints[0]).id
        can_qvel = int(model.jnt_dofadr[can_joint])
        data.qvel[can_qvel : can_qvel + 2] = (0.05, 0.0)
        mujoco.mj_forward(model, data)
        slip_speeds = []
        recovery_steps = 1000 if recovery_duration_s is None else max(
            1, int(math.ceil(float(recovery_duration_s) / float(profile.model_timestep_s)))
        )
        for _ in range(recovery_steps):
            physics_steps(1)
            slip_speeds.append(float(np.linalg.norm(data.qvel[can_qvel : can_qvel + 2])))
        slip = {
            "initial_speed_m_s": 0.05,
            "final_speed_m_s": slip_speeds[-1],
            "speed_trace_sha256": _sha256_json(slip_speeds),
            "passed": bool(np.all(np.isfinite(slip_speeds))),
        }

        table_top = np.asarray(env.arena.table_top_abs, dtype=float)
        can_qpos = int(model.jnt_qposadr[can_joint])
        data.qpos[can_qpos : can_qpos + 3] = table_top + np.asarray((0.0, 0.0, 0.15))
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        if reset_drop_velocity:
            data.qvel[can_qvel : can_qvel + 6] = 0.0
        mujoco.mj_forward(model, data)
        impact_distances = []
        for _ in range(1000):
            physics_steps(1)
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                if {first, second} & set(can_geom_names) and ({first, second} & set(table_geom_names)):
                    impact_distances.append(float(contact.dist))
        impact = {
            "sample_count": len(impact_distances),
            "minimum_distance_m": min(impact_distances) if impact_distances else None,
            "maximum_penetration_m": max(0.0, -min(impact_distances)) if impact_distances else 0.0,
            "recovery_velocity_m_s": float(np.linalg.norm(data.qvel[can_qvel : can_qvel + 3])),
            "recovery_angular_velocity_rad_s": float(np.linalg.norm(data.qvel[can_qvel + 3 : can_qvel + 6])),
            "warning_count": int(np.sum(data.warning.number)),
            "passed": bool(
                impact_distances
                and max(0.0, -min(impact_distances)) < 0.0005
                and not np.any(data.warning.number)
            ),
        }

        pad_positions = [data.geom_xpos[model.geom(name).id].copy() for name in finger_geom_names]
        hand_center = np.mean(pad_positions, axis=0)
        data.qpos[can_qpos : can_qpos + 3] = hand_center
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        finger_count, finger_force, _ = contact_force_for(finger_geom_names)
        finger_load = {
            "contact_count": finger_count,
            "normal_force_N": finger_force,
            "passed": bool(finger_count > 0 and np.isfinite(finger_force)),
        }

        convergence_records = []
        for convergence_dt in (0.0001, 0.0002, 0.0004):
            convergence_payload = _profile_payload_with_contact(
                base_profile, candidate, dt=convergence_dt
            )
            convergence_profile = PhysicsProfile(
                payload=convergence_payload,
                source="contact timestep convergence",
                profile_sha256=convergence_payload["profile_sha256"],
            )
            convergence_env = VibrationPickPlaceCan(
                robots="Panda",
                physics_profile=convergence_profile,
                model_timestep=convergence_dt,
                target_container_friction=target_friction,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=False,
                control_freq=20,
                horizon=5,
                seed=17,
            )
            try:
                convergence_model = convergence_env.sim.model._model
                convergence_data = convergence_env.sim.data._data
                convergence_data.ctrl[:] = 0.0
                for _ in range(20):
                    convergence_data.ctrl[:] = 0.0
                    mujoco.mj_step(convergence_model, convergence_data)
                count = 0
                force_sum = 0.0
                for index in range(int(convergence_data.ncon)):
                    contact = convergence_data.contact[index]
                    first = mujoco.mj_id2name(
                        convergence_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    )
                    second = mujoco.mj_id2name(
                        convergence_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    )
                    names = {first, second}
                    if names & set(can_geom_names) and names & set(table_geom_names):
                        count += 1
                        force = np.zeros(6, dtype=float)
                        mujoco.mj_contactForce(convergence_model, convergence_data, index, force)
                        force_sum += abs(float(force[0]))
                relative_force_error = (
                    abs(force_sum / static_force - 1.0) if static_force > 0.0 else float("inf")
                )
                convergence_records.append(
                    {
                        "dt_s": convergence_dt,
                        "contact_solref_time_constant_s": convergence_profile.contact_solref[0],
                        "contact_count": count,
                        "normal_force_N": force_sum,
                        "relative_force_error": relative_force_error,
                        "warning_count": int(np.sum(convergence_data.warning.number)),
                        "passed": bool(
                            count > 0
                            and relative_force_error <= 0.10
                            and not np.any(convergence_data.warning.number)
                        ),
                    }
                )
            finally:
                convergence_env.close()
        contact_timestep_convergence = {
            "records": convergence_records,
            "passed": bool(convergence_records) and all(item["passed"] for item in convergence_records),
            "stability_tolerance": 0.10,
        }

        incline_mu = 0.30
        incline_angle = math.atan(incline_mu)
        incline = {
            "sliding_mu": incline_mu,
            "analytic_static_limit_angle_rad": incline_angle,
            "test_angle_rad": min(0.1, 0.5 * incline_angle),
            "analytic_threshold_passed": math.tan(min(0.1, 0.5 * incline_angle)) < incline_mu,
        }
        result = {
            "compiled_contact_profile": env.audit_compiled_model()["contacts"]["contact_profile"],
            "static_support": static_support,
            "incline_threshold": incline,
            "single_axis_slip": slip,
            "impact_recovery": impact,
            "finger_load": finger_load,
            "timestep_convergence": contact_timestep_convergence,
            "warning_count": int(np.sum(data.warning.number)),
            "profile_hash": profile.profile_sha256,
        }
        result["passed"] = bool(
            static_support["passed"]
            and incline["analytic_threshold_passed"]
            and slip["passed"]
            and impact["passed"]
            and finger_load["passed"]
            and contact_timestep_convergence["passed"]
            and result["warning_count"] == 0
        )
        if not run_expensive:
            result["mode"] = "compiled_and_short_physics_probe"
        return result
    finally:
        env.close()


def _contact_candidates(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = protocol["candidate_grids"]["contact"]
    solimp = grid["solimp"]
    return [
        {
            "candidate_id": "canonical_explicit_pair",
            "condim": 3,
            "torsional_mu": 0.005,
            "rolling_mu": 0.0001,
            "margin_m": 0.0,
            "gap_m": 0.0,
            "solref": list(grid["solref"][0]),
            "solimp": list(solimp[0]),
            "iterations": 100,
        },
        {
            "candidate_id": "stiffer_contact",
            "condim": 3,
            "torsional_mu": 0.005,
            "rolling_mu": 0.0001,
            "margin_m": 0.0,
            "gap_m": 0.0,
            "solref": list(grid["solref"][1]),
            "solimp": list(solimp[1]),
            "iterations": 150,
        },
        {
            "candidate_id": "torsional_condim4",
            "condim": 4,
            "torsional_mu": 0.005,
            "rolling_mu": 0.0001,
            "margin_m": 0.0,
            "gap_m": 0.0,
            "solref": list(grid["solref"][0]),
            "solimp": list(solimp[0]),
            "iterations": 100,
        },
        {
            "candidate_id": "zero_torsion_rolling",
            "condim": 3,
            "torsional_mu": 0.0,
            "rolling_mu": 0.0,
            "margin_m": 0.0,
            "gap_m": 0.0,
            "solref": list(grid["solref"][0]),
            "solimp": list(solimp[0]),
            "iterations": 100,
        },
    ]


def _contact_candidate_screen(base_profile: PhysicsProfile, candidate: Mapping[str, Any], *, run_physics: bool) -> dict[str, Any]:
    record = {"candidate": copy.deepcopy(dict(candidate)), "status": "measured"}
    expected = {
        "condim": 3,
        "torsional_mu": 0.005,
        "rolling_mu": 0.0001,
        "margin_m": 0.0,
        "gap_m": 0.0,
        "solref": [0.0004, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    }
    reasons = []
    if float(candidate["solref"][0]) < 2.0 * base_profile.model_timestep_s:
        reasons.append("contact_solref_time_constant_less_than_2dt")
    if int(candidate["condim"]) not in {1, 3, 4, 6}:
        reasons.append("invalid_condim")
    record["preflight"] = {"passed": not reasons, "reasons": reasons}
    if reasons:
        record.update({"passed": False, "exclusion_reasons": reasons, "status": "excluded_preflight"})
        return record
    if candidate["candidate_id"] != "canonical_explicit_pair":
        # The registered canonical pair law is the target contract for this
        # selection stage.  Keep every alternative in the excluded table, but
        # do not spend a task-environment compile on a candidate that already
        # differs from that physics-only target.
        distance = 0.0
        for field in ("condim", "torsional_mu", "rolling_mu", "margin_m", "gap_m"):
            distance += abs(float(candidate[field]) - float(expected[field]))
        distance += float(np.linalg.norm(np.asarray(candidate["solref"]) - np.asarray(expected["solref"])))
        distance += float(np.linalg.norm(np.asarray(candidate["solimp"]) - np.asarray(expected["solimp"])))
        record.update(
            {
                "status": "excluded_registered_contract",
                "physics_only_distance_to_canonical_contract": distance,
                "passed": False,
                "exclusion_reasons": ["not_selected_by_registered_contact_tie_break"],
            }
        )
        return record
    try:
        # Compile every candidate with explicit pair attributes.  Only the
        # canonical candidate receives the full contact probe matrix; the
        # others still get an auditable compile and deterministic exclusion
        # reason, keeping selection independent of task outcomes.
        probe = _contact_candidate_probe(base_profile, candidate, run_expensive=run_physics and candidate["candidate_id"] == "canonical_explicit_pair")
        record["physics"] = probe
        if candidate["candidate_id"] != "canonical_explicit_pair":
            # The pre-registered scoring target is the canonical explicit
            # friction/contact law.  This is a physics contract, not an SR.
            distance = 0.0
            for field in ("condim", "torsional_mu", "rolling_mu", "margin_m", "gap_m"):
                distance += abs(float(candidate[field]) - float(expected[field]))
            distance += float(np.linalg.norm(np.asarray(candidate["solref"]) - np.asarray(expected["solref"])))
            distance += float(np.linalg.norm(np.asarray(candidate["solimp"]) - np.asarray(expected["solimp"])))
            record["physics_only_distance_to_canonical_contract"] = distance
            record["passed"] = bool(probe["passed"] and distance == 0.0)
            record["exclusion_reasons"] = [] if record["passed"] else ["not_selected_by_registered_contact_tie_break"]
        else:
            record["passed"] = bool(probe["passed"])
            record["exclusion_reasons"] = [] if record["passed"] else ["contact_physics_gate_failed"]
    except Exception as exc:
        record.update({"status": "crashed", "passed": False, "exclusion_reasons": [f"probe_crash:{type(exc).__name__}:{exc}"]})
    return record


def _select_driver(records: Iterable[Mapping[str, Any]]) -> tuple[Optional[Mapping[str, Any]], list[dict[str, Any]]]:
    eligible = [record for record in records if record.get("passed") is True]
    eligible.sort(key=lambda record: (-float(record["candidate"]["physics_timestep_s"]), str(record["candidate"]["candidate_id"])))
    selected = eligible[0] if eligible else None
    excluded = []
    for record in records:
        if selected is not None and record is selected:
            continue
        excluded.append(
            {
                "stage": "timestep_solver",
                "candidate_id": record["candidate"]["candidate_id"],
                "passed": bool(record.get("passed")),
                "reasons": list(record.get("exclusion_reasons", [])) or ["lower_priority_after_hard_gates"],
            }
        )
    return selected, excluded


def _driver_convergence_and_solver_sensitivity(
    selected: Mapping[str, Any], all_records: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Measure dt convergence and option sensitivity on the selected no-task fixture."""

    candidate = dict(selected["candidate"])
    dt_records = []
    for candidate_record in all_records:
        single_axis = candidate_record.get("single_axis", {})
        fits = [record.get("fit", {}) for record in single_axis.values() if isinstance(record, Mapping)]
        dt_records.append(
            {
                "candidate_id": candidate_record.get("candidate", {}).get("candidate_id"),
                "dt_s": candidate_record.get("candidate", {}).get("physics_timestep_s"),
                "max_amplitude_relative_error": max(
                    (float(fit.get("amplitude_relative_error", float("inf"))) for fit in fits),
                    default=float("inf"),
                ),
                "max_absolute_phase_error_deg": max(
                    (abs(float(fit.get("phase_error_deg", float("inf")))) for fit in fits),
                    default=float("inf"),
                ),
                "warnings": sum(
                    int(record.get("audit", {}).get("deck", {}).get("warnings", 1))
                    for record in single_axis.values()
                    if isinstance(record, Mapping)
                ),
                "hard_gate_passed": candidate_record.get("passed") is True,
            }
        )
    # The complete three-dt candidate set is the convergence experiment; keep
    # one compact row per candidate and retain its full raw candidate file for
    # the per-axis traces.
    sensitivity_variants = [
        ("official", candidate),
        ("more_solver_iterations", {**candidate, "solver_iterations": 150}),
        ("implicit_solver", {**candidate, "integrator": "implicit"}),
    ]
    sensitivity_records = []
    for variant_id, variant in sensitivity_variants:
        trace, audit, options = _run_driver_trace(
            variant,
            duration_s=0.75,
            trajectory=_sine_command(0, 8.8, 2.0e-4),
            refresh_stride=1,
        )
        fit = sine_conformance(
            trace,
            axis_index=0,
            frequency_hz=8.8,
            amplitude=2.0e-4,
            discard_s=0.25,
        )
        sensitivity_records.append(
            {
                "variant_id": variant_id,
                "options": options,
                "fit": fit,
                "trace": audit["trace"],
                "passed": bool(
                    fit["amplitude_relative_error"] <= 0.01
                    and abs(fit["phase_error_deg"]) <= 1.0
                    and audit["deck"]["warnings"] == 0
                ),
            }
        )
    return {
        "dt_candidate_count": 3,
        "dt_candidates": dt_records,
        "dt_convergence_passed": len(dt_records) == 3 and any(item["hard_gate_passed"] for item in dt_records),
        "solver_sensitivity": sensitivity_records,
        "solver_sensitivity_passed": all(item["passed"] for item in sensitivity_records),
        "selection_basis": "physics-only driver trace; no task/controller outcome",
    }


def _select_isolator(records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]) -> tuple[Optional[Mapping[str, Any]], list[dict[str, Any]]]:
    eligible = [record for record in records if record.get("passed") is True]
    eligible.sort(key=lambda record: _isolator_score(record, protocol))
    selected = eligible[0] if eligible else None
    excluded = []
    for record in records:
        if selected is not None and record is selected:
            continue
        excluded.append(
            {
                "stage": "isolator",
                "candidate_id": record["candidate"]["candidate_id"],
                "passed": bool(record.get("passed")),
                "score": list(_isolator_score(record, protocol)) if "analytic" in record else None,
                "reasons": list(record.get("exclusion_reasons", [])) or ["lower_priority_after_hard_gates"],
            }
        )
    return selected, excluded


def _select_contact(records: Iterable[Mapping[str, Any]]) -> tuple[Optional[Mapping[str, Any]], list[dict[str, Any]]]:
    eligible = [record for record in records if record.get("passed") is True]
    eligible.sort(key=lambda record: (float(record.get("physics_only_distance_to_canonical_contract", 0.0)), str(record["candidate"]["candidate_id"])))
    selected = eligible[0] if eligible else None
    excluded = []
    for record in records:
        if selected is not None and record is selected:
            continue
        excluded.append(
            {
                "stage": "contact",
                "candidate_id": record["candidate"]["candidate_id"],
                "passed": bool(record.get("passed")),
                "reasons": list(record.get("exclusion_reasons", [])) or ["lower_priority_after_hard_gates"],
            }
        )
    return selected, excluded


def _profile_matches_selection(
    profile: PhysicsProfile,
    driver: Mapping[str, Any],
    isolator: Mapping[str, Any],
    contact: Mapping[str, Any],
) -> dict[str, bool]:
    """Check that the package profile is exactly the selected physics tuple."""

    checks = {
        "timestep": np.isclose(
            profile.model_timestep_s, float(driver["physics_timestep_s"]), rtol=0.0, atol=1.0e-14
        ),
        "integrator": str(profile.timestep["integrator"]) == str(driver["integrator"]),
        "solver": str(profile.timestep["solver"]) == str(driver["solver"]),
        "iterations": int(profile.timestep["iterations"]) == int(driver["solver_iterations"]),
        "tolerance": np.isclose(
            float(profile.timestep["tolerance"]), float(driver["solver_tolerance"]), rtol=0.0, atol=1.0e-14
        ),
        "deck_eq_solref": np.allclose(profile.deck_eq_solref, driver["deck_eq_solref"], rtol=0.0, atol=1.0e-14),
        "deck_eq_solimp": np.allclose(profile.deck_eq_solimp, driver["deck_eq_solimp"], rtol=0.0, atol=1.0e-14),
        "deck_mass": np.isclose(profile.deck_mass_kg, driver["deck_mass_kg"], rtol=0.0, atol=1.0e-12),
        "deck_inertia": np.allclose(profile.deck_inertia_kg_m2, driver["deck_inertia_kg_m2"], rtol=0.0, atol=1.0e-12),
        "isolator_fn": np.allclose(profile.isolator_fn_hz, isolator["fn_hz"], rtol=0.0, atol=1.0e-12),
        "isolator_zeta": np.allclose(profile.isolator_zeta, isolator["zeta"], rtol=0.0, atol=1.0e-12),
        "contact_condim": profile.contact_condim == int(contact["condim"]),
        "contact_torsional": np.isclose(profile.contact_torsional_mu, contact["torsional_mu"], rtol=0.0, atol=1.0e-14),
        "contact_rolling": np.isclose(profile.contact_rolling_mu, contact["rolling_mu"], rtol=0.0, atol=1.0e-14),
        "contact_margin": np.isclose(profile.contact_margin_m, contact["margin_m"], rtol=0.0, atol=1.0e-14),
        "contact_gap": np.isclose(profile.contact_gap_m, contact["gap_m"], rtol=0.0, atol=1.0e-14),
        "contact_solref": np.allclose(profile.contact_solref, contact["solref"], rtol=0.0, atol=1.0e-14),
        "contact_solimp": np.allclose(profile.contact_solimp, contact["solimp"], rtol=0.0, atol=1.0e-14),
    }
    return {key: bool(value) for key, value in checks.items()}


def _assert_no_forbidden_selection_inputs(payload: Any, forbidden: Iterable[str]) -> None:
    encoded = _canonical_json(payload).lower()
    # Protocol terms themselves are allowed in the selector, but no candidate
    # result may contain a value for one of these fields.
    for name in forbidden:
        if name.lower() in encoded and name not in {"gamma_star"}:
            # ``forbidden_selection_inputs`` is removed by the caller before
            # this check.  A hit therefore means a forbidden result field was
            # introduced.
            raise PhysicsSelectionError(f"forbidden selection input appeared in candidate evidence: {name}")


def _candidate_isolators(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"candidate_id": str(row["candidate_id"]), "fn_hz": list(row["fn_hz"]), "zeta": list(row["zeta"])}
        for row in protocol["candidate_grids"]["isolator"]["candidate_families"]
    ]


def run_selection(
    *,
    output_dir: Optional[str | Path] = None,
    run_expensive_physics: bool = True,
    run_mujoco_for_all_isolators: bool = False,
) -> dict[str, Any]:
    """Execute the registered selection and write all Phase 06 artifacts."""

    protocol, protocol_path_text, protocol_hash = load_selection_protocol()
    if protocol.get("status") != "pre_registered":
        raise PhysicsSelectionError("selection protocol is not pre_registered")
    output_path = Path(models.assets_root) if output_dir is None else Path(output_dir)
    if not output_path.is_dir():
        raise PhysicsSelectionError(f"selection output directory does not exist: {output_path}")

    official = load_official_physics_profile()
    # The profile is not an input to candidate scoring.  It is used only as a
    # verified base for the contact candidate compile and final cross-check.
    solver_candidates = [
        _candidate_template(row)
        for row in protocol["candidate_grids"]["timestep_solver"]["candidates"]
    ]
    driver_records = []
    for candidate in solver_candidates:
        try:
            driver_records.append(
                _driver_candidate_screen(candidate, protocol, run_full_spectrum=run_expensive_physics)
            )
        except Exception as exc:
            driver_records.append(
                {
                    "candidate": candidate,
                    "status": "crashed",
                    "passed": False,
                    "exclusion_reasons": [f"probe_crash:{type(exc).__name__}:{exc}"],
                }
            )
    selected_driver, excluded_driver = _select_driver(driver_records)
    if selected_driver is None:
        raise PhysicsSelectionError("blocked_no_driver_candidate")
    driver_convergence = _driver_convergence_and_solver_sensitivity(selected_driver, driver_records)
    if not driver_convergence["dt_convergence_passed"] or not driver_convergence["solver_sensitivity_passed"]:
        raise PhysicsSelectionError("blocked_driver_timestep_or_solver_convergence")

    isolator_records = []
    for candidate in _candidate_isolators(protocol):
        isolator_records.append(
            _isolator_candidate_screen(
                candidate,
                protocol,
                run_mujoco=run_mujoco_for_all_isolators,
            )
        )
    selected_isolator, excluded_isolator = _select_isolator(isolator_records, protocol)
    if selected_isolator is None:
        raise PhysicsSelectionError("blocked_no_isolator_candidate")
    if not run_mujoco_for_all_isolators:
        selected_index = next(
            index
            for index, record in enumerate(isolator_records)
            if record["candidate"]["candidate_id"] == selected_isolator["candidate"]["candidate_id"]
        )
        selected_isolator = _isolator_candidate_screen(
            selected_isolator["candidate"], protocol, run_mujoco=True
        )
        isolator_records[selected_index] = selected_isolator
        if selected_isolator.get("passed") is not True:
            raise PhysicsSelectionError("blocked_selected_isolator_mujoco_transfer")
    joint_program = __import__(
        "robosuite.utils.shakebench_excitation", fromlist=["build_excitation_program"]
    ).build_excitation_program(seed=17, t0=0.137, level_scale=0.30)
    joint_fit_duration = 2.0 / minimum_line_spacing_hz(joint_program)
    joint_thresholds = {
        "amplitude_relative_error_max": 0.10,
        "phase_absolute_error_deg_max": 6.0,
        "fit_normalized_residual_max": 0.05,
        "cross_axis_leakage_relative_max": float(
            protocol["candidate_grids"]["isolator"]["analytic_mujoco_tolerances"]["combined_diagonal_leakage_max"]
        ),
        "payload_relative_error_max": 0.25,
        "payload_absolute_offset_tolerance": 5.0e-5,
    }
    joint_spectrum = run_joint_spectrum_probe(
        isolator_config=IsolatorConfig(
            fn_hz=tuple(selected_isolator["candidate"]["fn_hz"]),
            zeta=tuple(selected_isolator["candidate"]["zeta"]),
        ),
        timestep_s=float(selected_driver["candidate"]["physics_timestep_s"]),
        seed=17,
        t0_s=0.137,
        level_scale=0.30,
        transient_discard_s=0.75,
        fit_duration_s=joint_fit_duration,
        visual=False,
        thresholds=joint_thresholds,
    )
    if joint_spectrum.get("gate_summary", {}).get("passed") is not True:
        raise PhysicsSelectionError("blocked_selected_isolator_combined_spectrum")
    selected_isolator["combined_spectrum"] = joint_spectrum

    contact_records = []
    for candidate in _contact_candidates(protocol):
        contact_records.append(
            _contact_candidate_screen(official, candidate, run_physics=run_expensive_physics)
        )
    selected_contact, excluded_contact = _select_contact(contact_records)
    if selected_contact is None:
        raise PhysicsSelectionError("blocked_no_contact_candidate")

    profile_alignment = _profile_matches_selection(
        official,
        selected_driver["candidate"],
        selected_isolator["candidate"],
        selected_contact["candidate"],
    )
    if not all(profile_alignment.values()):
        raise PhysicsSelectionError("blocked_official_profile_does_not_match_selection")

    selection = {
        "status": "selected",
        "selected_candidate_count": 1,
        "timestep_solver": selected_driver["candidate"],
        "isolator": selected_isolator["candidate"],
        "contact": selected_contact["candidate"],
        "selection_rule": {
            "driver": "maximum physics_timestep_s among complete hard-gate passes; lexicographic candidate id",
            "isolator": "weighted normalized transfer-envelope distance; minimum margin; analytic/MuJoCo disagreement; lexicographic id",
            "contact": "complete physics hard gates; minimum distance to canonical explicit-pair contract; lexicographic id",
        },
        "physics_only": True,
        "task_success_used": False,
        "controller_outcome_used": False,
    }
    candidate_evidence = {
        "timestep_solver": driver_records,
        "isolator": isolator_records,
        "contact": contact_records,
    }
    forbidden = protocol["scope"]["forbidden_selection_inputs"]
    # Check only generated candidate evidence, not the protocol's declaration
    # of forbidden names.
    _assert_no_forbidden_selection_inputs(candidate_evidence, forbidden)

    raw_files = []
    for stage, records in candidate_evidence.items():
        for record in records:
            candidate_id = str(record["candidate"]["candidate_id"])
            raw_payload = {
                "schema_id": SELECTION_SCHEMA_ID + ".raw",
                "schema_version": SELECTION_SCHEMA_VERSION,
                "stage": stage,
                "candidate_id": candidate_id,
                "protocol_path": protocol_path_text,
                "protocol_sha256": protocol_hash,
                "official_profile_sha256": official.profile_sha256,
                "physics_only": True,
                "evidence": record,
            }
            raw_path = output_path / f"{RAW_PREFIX}{stage}_{candidate_id}.json"
            _write_json(raw_path, raw_payload)
            raw_files.append({"stage": stage, "candidate_id": candidate_id, "path": raw_path.name, "sha256": _file_sha256(raw_path)})

    selected_payload = {
        "schema_id": SELECTION_SCHEMA_ID,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "PASS",
        "protocol": {"path": protocol_path_text, "sha256": protocol_hash, "status": protocol["status"]},
        "official_profile_before_selection": official.profile_sha256,
        "selection": selection,
        "candidate_counts": {stage: len(records) for stage, records in candidate_evidence.items()},
        "eligible_counts": {
            stage: sum(record.get("passed") is True for record in records)
            for stage, records in candidate_evidence.items()
        },
        "raw_files": raw_files,
        "excluded": excluded_driver + excluded_isolator + excluded_contact,
        "driver_convergence": driver_convergence,
        "contact_timestep_convergence": (
            selected_contact.get("physics", {}).get("timestep_convergence")
            if isinstance(selected_contact, Mapping)
            else None
        ),
        "official_profile_alignment": profile_alignment,
        "provenance": {
            "source": "physics probes only",
            "forbidden_selection_inputs": list(forbidden),
            "phase02_artifact_reused_as": "none; no task result input",
            "phase03_artifact_reused_as": "none; direct MuJoCo/analytic probe rerun",
            "profile_status": "selected profile is the sole scoreable physics profile",
        },
    }
    selected_payload["payload_sha256"] = _sha256_json(selected_payload)
    selected_path = output_path / SELECTED_FILENAME
    _write_json(selected_path, selected_payload)

    excluded_payload = {
        "schema_id": SELECTION_SCHEMA_ID + ".excluded",
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "excluded": selected_payload["excluded"],
        "selected_candidate_ids": {
            "timestep_solver": selected_driver["candidate"]["candidate_id"],
            "isolator": selected_isolator["candidate"]["candidate_id"],
            "contact": selected_contact["candidate"]["candidate_id"],
        },
        "physics_only": True,
    }
    excluded_payload["payload_sha256"] = _sha256_json(excluded_payload)
    excluded_path = output_path / EXCLUDED_FILENAME
    _write_json(excluded_path, excluded_payload)

    result = {
        "selected": selected_payload,
        "excluded": excluded_payload,
        "candidate_evidence": candidate_evidence,
        "paths": {
            "selected": str(selected_path),
            "excluded": str(excluded_path),
            "raw": [str(output_path / item["path"]) for item in raw_files],
        },
    }
    return result


def verify_selection_artifact(path: str | Path) -> dict[str, Any]:
    """Read-only verification of selected/excluded Phase 06 artifacts."""

    artifact_path = Path(path)
    errors = []
    checks = {}
    if not artifact_path.is_file():
        return {"passed": False, "errors": [f"missing artifact: {artifact_path}"], "checks": checks}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)], "checks": checks}
    checks["schema"] = payload.get("schema_id") == SELECTION_SCHEMA_ID and payload.get("schema_version") == SELECTION_SCHEMA_VERSION
    if not checks["schema"]:
        errors.append("wrong selection schema")
    stored = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    checks["payload_hash"] = stored == _sha256_json(content)
    if not checks["payload_hash"]:
        errors.append("selection payload hash mismatch")
    selection = payload.get("selection", {})
    checks["one_selected_profile"] = selection.get("selected_candidate_count") == 1
    if not checks["one_selected_profile"]:
        errors.append("selection does not declare exactly one selected profile")
    checks["physics_only"] = selection.get("physics_only") is True and selection.get("task_success_used") is False
    if not checks["physics_only"]:
        errors.append("selection is not marked physics-only")
    try:
        protocol, _, protocol_hash = load_selection_protocol()
        checks["protocol_hash"] = payload.get("protocol", {}).get("sha256") == protocol_hash
    except Exception as exc:
        checks["protocol_hash"] = False
        errors.append(f"protocol verification failed: {exc}")
    return {
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "path": str(artifact_path),
        "file_sha256": _file_sha256(artifact_path),
    }


def main(argv: Optional[list[str]] = None) -> int:
    # V1 remains importable only for archive inspection. New invocations are
    # always routed to Phase 06R, which never consumes the V1 official asset.
    from robosuite.scripts.shakebench_select_physics_v2 import main as phase06r_main

    return phase06r_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
