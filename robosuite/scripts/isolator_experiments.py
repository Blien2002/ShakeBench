"""MuJoCo-based ShakeBench isolator experiments and Phase 03 evidence tools.

This module is intentionally outside ``robosuite.utils``: it is used by
selection and evidence workflows, while runtime environments only need the
analytic model in :mod:`robosuite.utils.shakebench_isolator`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from robosuite.utils.shakebench_isolator import (
    AXES,
    HARMONIC_PROBE_FIT_CYCLES,
    HARMONIC_PROBE_ROTATION_AMPLITUDE_RAD,
    HARMONIC_PROBE_TRANSIENT_CYCLES,
    HARMONIC_PROBE_TRANSLATION_AMPLITUDE_M,
    HARMONIC_REGION_RATIOS,
    PHASE03_PROBE_SAMPLE_STRIDE,
    PHASE03_TRANSFER_SCHEMA_ID,
    PHASE03_TRANSFER_SCHEMA_VERSION,
    PHASE03_TRANSFER_THRESHOLDS,
    HarmonicFit,
    IsolatorConfig,
    IsolatorError,
    IsolatorParameters,
    _axis_index,
    _coerce_config,
    _coerce_scalar,
    _complex_pair,
    _solve_harmonic_coefficients,
    absolute_transfer_function,
    compare_harmonic_fit,
    derive_isolator_parameters,
    fit_harmonic_transfer,
    relative_transfer_function,
    static_equilibrium_offset,
    static_sag_uncompensated_m,
)

def _normalise_quaternion_wxyz(value: Any) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import normalize_wxyz

    return normalize_wxyz(value, error_type=IsolatorError)


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import multiply_wxyz

    return multiply_wxyz(_normalise_quaternion_wxyz(first), _normalise_quaternion_wxyz(second))


def _quaternion_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import inverse_wxyz

    return inverse_wxyz(_normalise_quaternion_wxyz(quaternion))


def _quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import wxyz_to_matrix

    return wxyz_to_matrix(quaternion, error_type=IsolatorError)


def _quaternion_to_rotation_vector_wxyz(quaternion: np.ndarray) -> np.ndarray:
    from robosuite.utils.shakebench_rotations import wxyz_to_rotation_vector

    return wxyz_to_rotation_vector(quaternion, error_type=IsolatorError)



def _body_spatial_state(raw_model: Any, raw_data: Any, body_id: int):
    """Return body pose, world spatial twist and acceleration at its origin."""

    position = np.array(raw_data.xpos[body_id], dtype=float, copy=True)
    quaternion = _normalise_quaternion_wxyz(raw_data.xquat[body_id])
    velocity_rotlin = np.zeros(6, dtype=float)
    mujoco = _lazy_mujoco()
    mujoco.mj_objectVelocity(
        raw_model,
        raw_data,
        mujoco.mjtObj.mjOBJ_BODY,
        int(body_id),
        velocity_rotlin,
        0,
    )
    mujoco.mj_rnePostConstraint(raw_model, raw_data)
    acceleration_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectAcceleration(
        raw_model,
        raw_data,
        mujoco.mjtObj.mjOBJ_BODY,
        int(body_id),
        acceleration_rotlin,
        0,
    )
    twist = np.concatenate((velocity_rotlin[3:], velocity_rotlin[:3]))
    acceleration = np.concatenate((acceleration_rotlin[3:], acceleration_rotlin[:3]))
    return position, quaternion, twist, acceleration


def _body_pose_state(raw_data: Any, body_id: int):
    """Return only a body's pose for high-rate probe sampling."""

    return (
        np.array(raw_data.xpos[body_id], dtype=float, copy=True),
        _normalise_quaternion_wxyz(raw_data.xquat[body_id]),
    )


def _lazy_mujoco():
    """Import MuJoCo only when a runtime physics probe is requested."""

    import mujoco

    return mujoco


def _set_probe_timestep(xml_string: str, timestep_s: float) -> str:
    root = ET.fromstring(xml_string)
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", format(_coerce_scalar("timestep_s", timestep_s, positive=True), ".17g"))
    return ET.tostring(root, encoding="unicode")


def _probe_coordinates(raw_model: Any, raw_data: Any, deck_body_id: int, table_body_id: int, baseline: dict[str, Any]):
    """Evaluate nominal-frame absolute and deck-frame relative worktable coordinates."""

    deck_position, deck_quaternion = _body_pose_state(raw_data, deck_body_id)
    table_position, table_quaternion = _body_pose_state(raw_data, table_body_id)
    deck_initial_rotation = baseline["deck_initial_rotation"]
    table_initial_rotation = baseline["table_initial_rotation"]
    deck_absolute = np.concatenate(
        (
            deck_initial_rotation.T.dot(deck_position - baseline["deck_initial_position"]),
            _quaternion_to_rotation_vector_wxyz(
                _quaternion_multiply_wxyz(
                    _quaternion_inverse_wxyz(baseline["deck_initial_quaternion"]),
                    deck_quaternion,
                )
            ),
        )
    )
    table_absolute = np.concatenate(
        (
            table_initial_rotation.T.dot(table_position - baseline["table_initial_position"]),
            _quaternion_to_rotation_vector_wxyz(
                _quaternion_multiply_wxyz(
                    _quaternion_inverse_wxyz(baseline["table_initial_quaternion"]),
                    table_quaternion,
                )
            ),
        )
    )
    relative_position = _quaternion_to_matrix_wxyz(deck_quaternion).T.dot(table_position - deck_position)
    relative_rotation = _quaternion_to_rotation_vector_wxyz(
        _quaternion_multiply_wxyz(_quaternion_inverse_wxyz(deck_quaternion), table_quaternion)
    )
    relative = np.concatenate(
        (
            relative_position - baseline["relative_initial_position"],
            relative_rotation - baseline["relative_initial_rotation"],
        )
    )
    return {
        "deck_absolute": deck_absolute,
        "table_absolute": table_absolute,
        "relative": relative,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}:{array.shape}".encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _build_probe_simulation(
    isolator_config: Any,
    *,
    timestep_s: float,
    deck_config: Any = None,
    trajectory: Any = None,
    visual: bool = True,
    payload_offset_m: Optional[Iterable[float]] = None,
    payload_mass_kg: float = 0.349,
    gravity_vector: Optional[Iterable[float]] = None,
    neutralize_preload: bool = False,
    payload_geom_type: str = "sphere",
    payload_geom_size: Iterable[float] = (0.01,),
):
    """Build a no-task arena + Phase 02 driver simulation for probes."""

    mujoco = _lazy_mujoco()
    from robosuite.models.arenas.shakebench_arena import ShakeBenchArena
    from robosuite.utils.shakebench_deck import DeckDriver, DeckDriverConfig

    config = _coerce_config(isolator_config)
    timestep = _coerce_scalar("timestep_s", timestep_s, positive=True)
    arena = ShakeBenchArena(isolator_config=config, visual=visual)
    if deck_config is None:
        deck_config = DeckDriverConfig(
            physics_timestep_s=timestep,
            eq_solref=(max(2.0 * timestep, 0.0004), 0.5),
            # The harmonic input reference is the worktable elastic centre.
            # Placing the generated deck origin there prevents a pure base
            # rotation from introducing a lever-arm translation into the
            # six independent relative coordinates.
            deck_pos_m=tuple(float(value) for value in arena.center_pos),
        )
    if not isinstance(deck_config, DeckDriverConfig):
        raise IsolatorError("deck_config must be a DeckDriverConfig")
    driver = DeckDriver(
        trajectory=trajectory,
        config=deck_config,
        body_handles=arena.deck_body_handles,
    )
    source_xml = _set_probe_timestep(arena.get_xml(), timestep)
    if gravity_vector is not None:
        gravity = _vector_from_probe_input("gravity_vector", gravity_vector, 3)
        source_root = ET.fromstring(source_xml)
        option = source_root.find("option")
        if option is None:
            option = ET.Element("option")
            source_root.insert(0, option)
        option.set("gravity", _format_probe_vector(gravity))
        source_xml = ET.tostring(source_root, encoding="unicode")
    if neutralize_preload:
        source_root = ET.fromstring(source_xml)
        worktable = source_root.find("./worldbody/body[@name='worktable']")
        if worktable is None:
            raise IsolatorError("probe XML is missing the worktable body")
        vertical_joint = worktable.find("./joint[@name='isolator_tz']")
        if vertical_joint is None:
            raise IsolatorError("probe XML is missing the vertical isolator joint")
        vertical_joint.set("springref", "0")
        source_xml = ET.tostring(source_root, encoding="unicode")
    processed_xml = driver.processor(source_xml)
    payload_offset = None
    if payload_offset_m is not None:
        payload_offset = np.asarray(tuple(payload_offset_m), dtype=float)
        if payload_offset.shape != (2,) or not np.all(np.isfinite(payload_offset)):
            raise IsolatorError("payload_offset_m must contain two finite x/y values")
        payload_mass = _coerce_scalar("payload_mass_kg", payload_mass_kg, positive=True)
        if payload_geom_type not in {"sphere", "box"}:
            raise IsolatorError("payload_geom_type must be 'sphere' or 'box'")
        geom_size = _vector_from_probe_input(
            "payload_geom_size",
            payload_geom_size,
            1 if payload_geom_type == "sphere" else 3,
        )
        payload_half_height = geom_size[0] if payload_geom_type == "sphere" else geom_size[2]
        root = ET.fromstring(processed_xml)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise IsolatorError("probe XML is missing worldbody")
        table_top = np.asarray(arena.table_top_abs, dtype=float)
        payload_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "phase03_payload",
                "pos": _format_probe_vector(
                    (
                        table_top[0] + payload_offset[0],
                        table_top[1] + payload_offset[1],
                        table_top[2] + payload_half_height,
                    )
                ),
            },
        )
        ET.SubElement(payload_body, "freejoint", {"name": "phase03_payload_free"})
        ET.SubElement(
            payload_body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": format(payload_mass, ".17g"),
                "diaginertia": "0.0001 0.0001 0.0001",
            },
        )
        ET.SubElement(
            payload_body,
            "geom",
            {
                "name": "phase03_payload_geom",
                "type": payload_geom_type,
                "size": _format_probe_vector(geom_size),
                "friction": "1 0.005 0.0001",
            },
        )
        processed_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(processed_xml)
    data = mujoco.MjData(model)
    sim = SimpleNamespace(model=model, data=data)
    driver.bind(sim)
    mujoco.mj_forward(model, data)
    deck_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, deck_config.deck_body_name))
    table_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arena.worktable_body_name))
    if deck_body_id < 0 or table_body_id < 0:
        raise IsolatorError("compiled probe is missing deck or worktable body")
    deck_position, deck_quaternion = _body_pose_state(data, deck_body_id)
    table_position, table_quaternion = _body_pose_state(data, table_body_id)
    baseline = {
        "deck_initial_position": deck_position,
        "deck_initial_quaternion": deck_quaternion,
        "deck_initial_rotation": _quaternion_to_matrix_wxyz(deck_quaternion),
        "table_initial_position": table_position,
        "table_initial_quaternion": table_quaternion,
        "table_initial_rotation": _quaternion_to_matrix_wxyz(table_quaternion),
        "relative_initial_position": _quaternion_to_matrix_wxyz(deck_quaternion).T.dot(table_position - deck_position),
        "relative_initial_rotation": _quaternion_to_rotation_vector_wxyz(
            _quaternion_multiply_wxyz(_quaternion_inverse_wxyz(deck_quaternion), table_quaternion)
        ),
    }
    return arena, config, deck_config, driver, model, data, baseline, deck_body_id, table_body_id, payload_offset


def _format_probe_vector(values: Iterable[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _vector_from_probe_input(name: str, value: Iterable[float], length: int) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise IsolatorError(f"{name} must contain {length} finite values") from exc
    if len(values) != length or not np.all(np.isfinite(values)):
        raise IsolatorError(f"{name} must contain {length} finite values")
    return values


def _run_probe_trajectory(
    isolator_config: Any,
    *,
    timestep_s: float,
    duration_s: float,
    trajectory: Any = None,
    deck_config: Any = None,
    visual: bool = True,
    payload_offset_m: Optional[Iterable[float]] = None,
    payload_mass_kg: float = 0.349,
    record_driver_trace: bool = False,
    gravity_vector: Optional[Iterable[float]] = None,
    neutralize_preload: bool = False,
    sample_stride: int = PHASE03_PROBE_SAMPLE_STRIDE,
    payload_geom_type: str = "sphere",
    payload_geom_size: Iterable[float] = (0.01,),
):
    """Run a right-limit-contract probe and return raw kinematic arrays."""

    mujoco = _lazy_mujoco()
    from robosuite.utils.shakebench_deck import (
        TRACE_SCHEMA_ID,
        TRACE_SCHEMA_VERSION,
        DeckCommand,
    )

    duration = _coerce_scalar("duration_s", duration_s, positive=True)
    if isinstance(sample_stride, (bool, np.bool_)) or int(sample_stride) != sample_stride or int(sample_stride) < 1:
        raise IsolatorError("sample_stride must be a positive integer")
    sample_stride = int(sample_stride)
    if trajectory is None:
        trajectory = lambda _: DeckCommand(np.zeros(6), np.zeros(6), np.zeros(6))
    built = _build_probe_simulation(
        isolator_config,
        timestep_s=timestep_s,
        deck_config=deck_config,
        trajectory=trajectory,
        visual=visual,
        payload_offset_m=payload_offset_m,
        payload_mass_kg=payload_mass_kg,
        gravity_vector=gravity_vector,
        neutralize_preload=neutralize_preload,
        payload_geom_type=payload_geom_type,
        payload_geom_size=payload_geom_size,
    )
    arena, config, deck_config, driver, model, data, baseline, deck_body_id, table_body_id, payload_offset = built
    joint_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"isolator_{axis}")) for axis in AXES]
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    qvel_addresses = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    times = []
    deck_absolute = []
    table_absolute = []
    relative = []
    joint_qpos = []
    joint_qvel = []
    warning_history = []
    solver_history = []
    weld_residual_history = []
    weld_force_history = []
    tracking_error_history = []
    previous_warning = np.zeros_like(np.asarray(data.warning.number, dtype=np.int64))
    steps = int(math.ceil(duration / float(model.opt.timestep)))
    for step_index in range(steps):
        integration_time = float(data.time)
        driver.pre_physics_step(integration_time)
        mujoco.mj_step(model, data)
        sample_time = float(data.time)
        sample_due = (step_index + 1) % sample_stride == 0 or step_index + 1 == steps
        if not sample_due:
            continue
        driver.post_integration_refresh(sample_time)
        mujoco.mj_forward(model, data)
        coordinates = _probe_coordinates(model, data, deck_body_id, table_body_id, baseline)
        times.append(sample_time)
        deck_absolute.append(coordinates["deck_absolute"])
        table_absolute.append(coordinates["table_absolute"])
        relative.append(coordinates["relative"])
        joint_qpos.append([float(data.qpos[address]) for address in qpos_addresses])
        joint_qvel.append([float(data.qvel[address]) for address in qvel_addresses])
        current_warning = np.asarray(data.warning.number, dtype=np.int64)
        warning_history.append(np.maximum(current_warning - previous_warning, 0))
        previous_warning = current_warning.copy()
        solver_niter = np.asarray(getattr(data, "solver_niter", np.zeros(0)), dtype=np.int64)
        solver_history.append(solver_niter.copy())
        weld_rows = np.flatnonzero(
            (np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (np.asarray(data.efc_id) == int(driver._weld_id))
        )
        weld_residual_history.append(np.array(data.efc_pos[weld_rows], dtype=float, copy=True))
        weld_force_history.append(np.array(data.efc_force[weld_rows], dtype=float, copy=True))
        command = driver._evaluate_command(sample_time)
        target_position, target_quaternion, _, _ = driver._command_world_pose(command)
        tracking_error_history.append(
            np.concatenate(
                (
                    np.asarray(data.xpos[deck_body_id], dtype=float) - target_position,
                    _quaternion_to_rotation_vector_wxyz(
                        _quaternion_multiply_wxyz(
                            _normalise_quaternion_wxyz(data.xquat[deck_body_id]),
                            _quaternion_inverse_wxyz(target_quaternion),
                        )
                    ),
                )
            )
        )
    trace = driver.trace if record_driver_trace else None
    time_array = np.asarray(times, dtype=float)
    for position_name in ("deck_absolute", "table_absolute"):
        positions = np.asarray(locals()[position_name], dtype=float)
        velocity = np.gradient(positions, time_array, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, time_array, axis=0, edge_order=2)
        if position_name == "deck_absolute":
            deck_twist = velocity
            deck_acceleration = acceleration
        else:
            table_twist = velocity
            table_acceleration = acceleration
    arrays = {
        "time_s": time_array,
        "deck_absolute": np.asarray(deck_absolute, dtype=float),
        "table_absolute": np.asarray(table_absolute, dtype=float),
        "relative": np.asarray(relative, dtype=float),
        "deck_twist": np.asarray(deck_twist, dtype=float),
        "table_twist": np.asarray(table_twist, dtype=float),
        "deck_acceleration": np.asarray(deck_acceleration, dtype=float),
        "table_acceleration": np.asarray(table_acceleration, dtype=float),
        "joint_qpos": np.asarray(joint_qpos, dtype=float),
        "joint_qvel": np.asarray(joint_qvel, dtype=float),
    }
    warning_delta = np.asarray(warning_history, dtype=np.int64)
    solver_iterations = np.asarray(
        [int(np.max(value)) if value.size else 0 for value in solver_history], dtype=np.int64
    )
    arrays["trace_time_s"] = time_array.copy()
    if weld_residual_history and all(value.size == 6 for value in weld_residual_history):
        arrays["weld_constraint_residual_raw"] = np.asarray(weld_residual_history, dtype=float)
        arrays["weld_constraint_force_raw"] = np.asarray(weld_force_history, dtype=float)
    else:
        arrays["weld_constraint_residual_raw"] = np.empty((time_array.size, 0), dtype=float)
        arrays["weld_constraint_force_raw"] = np.empty((time_array.size, 0), dtype=float)
    summary = {
        "sample_count": int(arrays["time_s"].size),
        "sample_rate_hz": float(1.0 / np.median(np.diff(arrays["time_s"]))),
        "model_timestep_s": float(model.opt.timestep),
        "max_solver_iterations": int(np.max(solver_iterations)) if solver_iterations.size else 0,
        "warning_number_delta_total": warning_delta.sum(axis=0).tolist() if warning_delta.size else [],
        "warning_lastinfo": np.asarray(data.warning.lastinfo, dtype=np.int64).tolist(),
        "max_deck_tracking_pose_error": np.max(np.abs(np.asarray(tracking_error_history)), axis=0).tolist()
        if tracking_error_history
        else [0.0] * 6,
        "max_weld_constraint_residual_raw": np.max(np.abs(arrays["weld_constraint_residual_raw"]), axis=0).tolist()
        if arrays["weld_constraint_residual_raw"].size
        else [],
        "max_weld_constraint_force_raw": np.max(np.abs(arrays["weld_constraint_force_raw"]), axis=0).tolist()
        if arrays["weld_constraint_force_raw"].size
        else [],
        "trace_schema_id": TRACE_SCHEMA_ID,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "driver_trace_recorded": False,
        "driver_trace_equivalent_raw_diagnostics": True,
        "tracking_diagnostics_recorded": True,
    }
    for name, value in arrays.items():
        summary[f"{name}_sha256"] = _array_sha256(value)
    return {
        "arena": arena,
        "config": config,
        "deck_config": deck_config,
        "driver": driver,
        "model": model,
        "data": data,
        "arrays": arrays,
        "summary": summary,
        "payload_offset_m": None if payload_offset is None else payload_offset.copy(),
        "trace": trace,
    }


def _default_harmonic_amplitude(axis_index: int) -> float:
    return (
        HARMONIC_PROBE_TRANSLATION_AMPLITUDE_M
        if axis_index < 3
        else HARMONIC_PROBE_ROTATION_AMPLITUDE_RAD
    )


def _probe_trace_summary(probe: dict[str, Any]) -> dict[str, Any]:
    """Return auditable diagnostics without embedding a full time trace."""

    arrays = probe["arrays"]
    return {
        **probe["summary"],
        "max_relative_coordinate": np.max(np.abs(arrays["relative"]), axis=0).tolist(),
        "max_table_absolute_coordinate": np.max(np.abs(arrays["table_absolute"]), axis=0).tolist(),
        "max_table_twist": np.max(np.abs(arrays["table_twist"]), axis=0).tolist(),
        "max_table_acceleration": np.max(np.abs(arrays["table_acceleration"]), axis=0).tolist(),
        "max_deck_acceleration": np.max(np.abs(arrays["deck_acceleration"]), axis=0).tolist(),
        "trace_sample_count": int(arrays["trace_time_s"].size),
    }


def run_harmonic_transfer_probe(
    axis: str | int,
    frequency_hz: float,
    *,
    isolator_config: Any = None,
    timestep_s: float = 0.0002,
    deck_config: Any = None,
    transient_cycles: int = HARMONIC_PROBE_TRANSIENT_CYCLES,
    fit_cycles: int = HARMONIC_PROBE_FIT_CYCLES,
    amplitude: Optional[float] = None,
    visual: bool = True,
    thresholds: Mapping[str, float] = PHASE03_TRANSFER_THRESHOLDS,
) -> dict[str, Any]:
    """Run one real MuJoCo harmonic transfer measurement.

    The input is the realized deck-origin motion in the nominal frame.  The
    relative output is the worktable COM pose relative to that deck origin,
    expressed in the realized deck frame.  The absolute output is the
    worktable COM pose in its nominal frame.  Both are fit as complex
    coefficients, so ``H_rel`` and ``H_abs`` are checked independently.
    """

    config = _coerce_config(isolator_config)
    axis_index = _axis_index(axis)
    frequency = _coerce_scalar("frequency_hz", frequency_hz, positive=True)
    if isinstance(transient_cycles, (bool, np.bool_)) or int(transient_cycles) != transient_cycles:
        raise IsolatorError("transient_cycles must be a positive integer")
    if isinstance(fit_cycles, (bool, np.bool_)) or int(fit_cycles) != fit_cycles:
        raise IsolatorError("fit_cycles must be a positive integer")
    if int(transient_cycles) < 1 or int(fit_cycles) < 1:
        raise IsolatorError("transient_cycles and fit_cycles must be positive")
    drive_amplitude = (
        _default_harmonic_amplitude(axis_index)
        if amplitude is None
        else _coerce_scalar("amplitude", amplitude, positive=True)
    )
    angular_frequency = 2.0 * math.pi * frequency

    from robosuite.utils.shakebench_deck import DeckCommand

    def trajectory(time_s: float) -> DeckCommand:
        pose = np.zeros(6, dtype=float)
        twist = np.zeros(6, dtype=float)
        acceleration = np.zeros(6, dtype=float)
        pose[axis_index] = drive_amplitude * math.sin(angular_frequency * time_s)
        twist[axis_index] = drive_amplitude * angular_frequency * math.cos(angular_frequency * time_s)
        acceleration[axis_index] = -drive_amplitude * angular_frequency**2 * math.sin(angular_frequency * time_s)
        return DeckCommand(pose, twist, acceleration)

    discard_time = float(transient_cycles) / frequency
    fit_duration = float(fit_cycles) / frequency
    probe = _run_probe_trajectory(
        config,
        timestep_s=timestep_s,
        duration_s=discard_time + fit_duration + float(timestep_s),
        trajectory=trajectory,
        deck_config=deck_config,
        visual=visual,
        gravity_vector=(0.0, 0.0, 0.0),
        neutralize_preload=True,
    )
    arrays = probe["arrays"]
    relative_fits, absolute_fits = _fit_harmonic_output_matrices(
        arrays["time_s"],
        arrays["deck_absolute"][:, axis_index],
        arrays["relative"],
        arrays["table_absolute"],
        frequency,
        discard_time_s=discard_time,
        fit_duration_s=fit_duration,
    )
    relative_fit = relative_fits[axis_index]
    absolute_fit = absolute_fits[axis_index]
    parameters = derive_isolator_parameters(config)
    expected_relative = complex(
        relative_transfer_function(frequency, parameters.fn_hz[axis_index], parameters.zeta[axis_index])
    )
    expected_absolute = complex(
        absolute_transfer_function(frequency, parameters.fn_hz[axis_index], parameters.zeta[axis_index])
    )
    relative_comparison = compare_harmonic_fit(relative_fit, expected_relative, thresholds=thresholds)
    absolute_comparison = compare_harmonic_fit(absolute_fit, expected_absolute, thresholds=thresholds)
    cross_axis = {}
    leakage_values = []
    for output_index, output_axis in enumerate(AXES):
        cross_fit = relative_fits[output_index]
        leakage = float(abs(cross_fit.output_complex_coefficient) / abs(cross_fit.input_complex_coefficient))
        cross_axis[output_axis] = {
            "fit": cross_fit.to_dict(),
            "leakage_relative_to_input": leakage,
            "is_excited_axis": output_index == axis_index,
        }
        if output_index != axis_index:
            leakage_values.append(leakage)
    leakage_max = max(leakage_values) if leakage_values else 0.0
    leakage_passed = leakage_max <= float(thresholds["cross_axis_leakage_relative_max"])
    return {
        "axis": AXES[axis_index],
        "axis_index": axis_index,
        "frequency_hz": frequency,
        "region": None,
        "base_amplitude": drive_amplitude,
        "transient_discard_cycles": int(transient_cycles),
        "transient_discard_s": discard_time,
        "fit_cycles": int(fit_cycles),
        "fit_duration_s": fit_duration,
        "reference": {
            "input_point": "deck body origin / deck site",
            "output_point": "worktable body origin = COM = elastic center",
            "input_frame": "nominal deck frame",
            "relative_output_frame": "realized deck frame",
            "absolute_output_frame": "nominal worktable frame",
            "relative_coordinate": "worktable COM minus deck-origin position and relative SO(3) rotation",
            "linearization_gravity_vector_m_s2": [0.0, 0.0, 0.0],
            "linearization_springref_override": {"isolator_tz": 0.0},
            "left_right_limit_contract": "write q(t), integrate to t+dt, write q(t+dt), forward, sample at t+dt",
            "fit_phase_convention": "phase(output)-phase(input), wrapped to [-pi, pi)",
        },
        "relative_fit": relative_fit.to_dict(),
        "absolute_fit": absolute_fit.to_dict(),
        "analytic_relative_transfer": _complex_pair(expected_relative),
        "analytic_absolute_transfer": _complex_pair(expected_absolute),
        "relative_comparison": relative_comparison,
        "absolute_comparison": absolute_comparison,
        "cross_axis_relative_fits": cross_axis,
        "cross_axis_leakage_max": leakage_max,
        "cross_axis_leakage_gate_passed": leakage_passed,
        "trace_summary": _probe_trace_summary(probe),
        "passed": bool(relative_comparison["passed"] and absolute_comparison["passed"] and leakage_passed),
    }


def run_harmonic_transfer_grid(
    *,
    isolator_config: Any = None,
    timestep_s: float = 0.0002,
    deck_config: Any = None,
    region_ratios: Mapping[str, float] = HARMONIC_REGION_RATIOS,
    transient_cycles: int = HARMONIC_PROBE_TRANSIENT_CYCLES,
    fit_cycles: int = HARMONIC_PROBE_FIT_CYCLES,
    visual: bool = True,
    thresholds: Mapping[str, float] = PHASE03_TRANSFER_THRESHOLDS,
) -> dict[str, Any]:
    """Run tracking/resonance/isolation probes for every axis."""

    config = _coerce_config(isolator_config)
    parameters = derive_isolator_parameters(config)
    ratios = dict(region_ratios)
    if tuple(ratios) != tuple(HARMONIC_REGION_RATIOS):
        raise IsolatorError("region_ratios must contain tracking, resonance and isolation in that order")
    records = []
    for region, ratio in ratios.items():
        ratio_value = _coerce_scalar(f"region_ratios[{region}]", ratio, positive=True)
        for index, axis in enumerate(AXES):
            record = run_harmonic_transfer_probe(
                axis,
                ratio_value * parameters.fn_hz[index],
                isolator_config=config,
                timestep_s=timestep_s,
                deck_config=deck_config,
                transient_cycles=transient_cycles,
                fit_cycles=fit_cycles,
                visual=visual,
                thresholds=thresholds,
            )
            record["region"] = region
            record["frequency_ratio_to_fn"] = ratio_value
            records.append(record)
    return {
        "schema_id": PHASE03_TRANSFER_SCHEMA_ID,
        "schema_version": PHASE03_TRANSFER_SCHEMA_VERSION,
        "axis_order": list(AXES),
        "region_ratios": {key: float(value) for key, value in ratios.items()},
        "thresholds": dict(thresholds),
        "records": records,
        "passed": bool(all(record["passed"] for record in records)),
    }


def _window_values(values: np.ndarray, time: np.ndarray, start_s: float, duration_s: Optional[float] = None):
    end_s = float(time[-1]) if duration_s is None else start_s + duration_s
    mask = (time >= start_s) & (time <= end_s + 1e-12)
    if not np.any(mask):
        raise IsolatorError("requested diagnostics window contains no samples")
    return values[mask]


def _fit_harmonic_output_matrices(
    time: np.ndarray,
    input_signal: np.ndarray,
    relative_outputs: np.ndarray,
    absolute_outputs: np.ndarray,
    frequency_hz: float,
    *,
    discard_time_s: float,
    fit_duration_s: float,
) -> tuple[list[HarmonicFit], list[HarmonicFit]]:
    """Fit one input against twelve outputs with one cached least-squares solve."""

    frequency = _coerce_scalar("frequency_hz", frequency_hz, positive=True)
    times = np.asarray(time, dtype=float)
    inputs = np.asarray(input_signal, dtype=float)
    relative_values = np.asarray(relative_outputs, dtype=float)
    absolute_values = np.asarray(absolute_outputs, dtype=float)
    if (
        times.ndim != 1
        or inputs.ndim != 1
        or relative_values.ndim != 2
        or absolute_values.ndim != 2
        or relative_values.shape[0] != times.size
        or absolute_values.shape[0] != times.size
        or relative_values.shape[1] != len(AXES)
        or absolute_values.shape[1] != len(AXES)
    ):
        raise IsolatorError("harmonic output matrices must have shape (N, 6)")
    discard = _coerce_scalar("discard_time_s", discard_time_s, nonnegative=True)
    duration = _coerce_scalar("fit_duration_s", fit_duration_s, positive=True)
    selected = (times >= discard) & (times <= discard + duration + 1e-12)
    if np.count_nonzero(selected) < 4:
        raise IsolatorError("transient discard / fit duration leaves fewer than four samples")
    fit_time = times[selected]
    right_hand_side = np.column_stack((inputs[selected], relative_values[selected], absolute_values[selected]))
    coefficients, residual, condition = _solve_harmonic_coefficients(fit_time, right_hand_side, frequency)
    input_complex = complex(float(coefficients[1, 0]), -float(coefficients[0, 0]))
    input_amplitude = float(abs(input_complex))
    if input_amplitude <= np.finfo(float).eps:
        raise IsolatorError("harmonic input coefficient is too small to form a transfer")
    input_residual = float(np.sqrt(np.mean(residual[:, 0] ** 2)))
    input_normalized = input_residual / max(input_amplitude, np.finfo(float).tiny)
    relative_fits = []
    absolute_fits = []
    for output_index in range(len(AXES)):
        relative_complex = complex(
            float(coefficients[1, output_index + 1]),
            -float(coefficients[0, output_index + 1]),
        )
        absolute_complex = complex(
            float(coefficients[1, output_index + 1 + len(AXES)]),
            -float(coefficients[0, output_index + 1 + len(AXES)]),
        )
        relative_residual = float(np.sqrt(np.mean(residual[:, output_index + 1] ** 2)))
        absolute_residual = float(np.sqrt(np.mean(residual[:, output_index + 1 + len(AXES)] ** 2)))
        relative_fits.append(
            HarmonicFit(
                frequency_hz=frequency,
                sample_count=int(fit_time.size),
                sample_rate_hz=float(1.0 / np.median(np.diff(fit_time))),
                fit_start_s=float(fit_time[0]),
                fit_end_s=float(fit_time[-1]),
                input_complex_coefficient=input_complex,
                output_complex_coefficient=relative_complex,
                transfer_complex=relative_complex / input_complex,
                input_amplitude=input_amplitude,
                output_amplitude=float(abs(relative_complex)),
                amplitude_ratio=float(abs(relative_complex / input_complex)),
                phase_difference_rad=float(np.angle(relative_complex / input_complex)),
                input_fit_residual_rms=input_residual,
                output_fit_residual_rms=relative_residual,
                normalized_input_residual=input_normalized,
                normalized_output_residual=relative_residual / max(abs(relative_complex), np.finfo(float).tiny),
                condition_number=condition,
            )
        )
        absolute_fits.append(
            HarmonicFit(
                frequency_hz=frequency,
                sample_count=int(fit_time.size),
                sample_rate_hz=float(1.0 / np.median(np.diff(fit_time))),
                fit_start_s=float(fit_time[0]),
                fit_end_s=float(fit_time[-1]),
                input_complex_coefficient=input_complex,
                output_complex_coefficient=absolute_complex,
                transfer_complex=absolute_complex / input_complex,
                input_amplitude=input_amplitude,
                output_amplitude=float(abs(absolute_complex)),
                amplitude_ratio=float(abs(absolute_complex / input_complex)),
                phase_difference_rad=float(np.angle(absolute_complex / input_complex)),
                input_fit_residual_rms=input_residual,
                output_fit_residual_rms=absolute_residual,
                normalized_input_residual=input_normalized,
                normalized_output_residual=absolute_residual / max(abs(absolute_complex), np.finfo(float).tiny),
                condition_number=condition,
            )
        )
    return relative_fits, absolute_fits


def _fit_harmonic_output_matrices_many(
    time: np.ndarray,
    input_signal: np.ndarray,
    relative_outputs: np.ndarray,
    absolute_outputs: np.ndarray,
    frequencies_hz: Iterable[float],
    *,
    discard_time_s: float,
    fit_duration_s: float,
    frequency_chunk_size: int = 8,
) -> list[tuple[list[HarmonicFit], list[HarmonicFit]]]:
    """Fit many frequencies while reusing one RHS and small batched solves."""

    frequencies = np.asarray(tuple(frequencies_hz), dtype=float).reshape(-1)
    if frequencies.size == 0 or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise IsolatorError("frequencies_hz must contain positive finite values")
    if isinstance(frequency_chunk_size, (bool, np.bool_)) or int(frequency_chunk_size) < 1:
        raise IsolatorError("frequency_chunk_size must be a positive integer")
    times = np.asarray(time, dtype=float)
    inputs = np.asarray(input_signal, dtype=float)
    relative_values = np.asarray(relative_outputs, dtype=float)
    absolute_values = np.asarray(absolute_outputs, dtype=float)
    if (
        times.ndim != 1
        or inputs.ndim != 1
        or relative_values.ndim != 2
        or absolute_values.ndim != 2
        or inputs.size != times.size
        or relative_values.shape != (times.size, len(AXES))
        or absolute_values.shape != (times.size, len(AXES))
    ):
        raise IsolatorError("harmonic output matrices must have shape (N, 6)")
    discard = _coerce_scalar("discard_time_s", discard_time_s, nonnegative=True)
    duration = _coerce_scalar("fit_duration_s", fit_duration_s, positive=True)
    selected = (times >= discard) & (times <= discard + duration + 1e-12)
    if np.count_nonzero(selected) < 4:
        raise IsolatorError("transient discard / fit duration leaves fewer than four samples")
    fit_time = times[selected]
    right_hand_side = np.column_stack((inputs[selected], relative_values[selected], absolute_values[selected]))
    output = []
    for start in range(0, frequencies.size, int(frequency_chunk_size)):
        chunk = frequencies[start : start + int(frequency_chunk_size)]
        angular = 2.0 * math.pi * chunk[:, None] * fit_time[None, :]
        sine = np.sin(angular)
        cosine = np.cos(angular)
        basis = np.stack((sine, cosine, np.ones_like(sine)), axis=-1)
        gram = np.einsum("fni,fnj->fij", basis, basis)
        rhs = np.einsum("fni,nk->fik", basis, right_hand_side)
        coefficients = np.linalg.solve(gram, rhs)
        residual = right_hand_side[None, :, :] - np.einsum("fni,fik->fnk", basis, coefficients)
        condition = np.sqrt(np.linalg.cond(gram))
        input_complex = coefficients[:, 1, 0] - 1j * coefficients[:, 0, 0]
        input_amplitude = np.abs(input_complex)
        if np.any(input_amplitude <= np.finfo(float).eps):
            raise IsolatorError("harmonic input coefficient is too small to form a transfer")
        input_residual = np.sqrt(np.mean(residual[:, :, 0] ** 2, axis=1))
        input_normalized = input_residual / np.maximum(input_amplitude, np.finfo(float).tiny)
        sample_rate = float(1.0 / np.median(np.diff(fit_time)))
        for chunk_index, frequency in enumerate(chunk):
            relative_fits = []
            absolute_fits = []
            for output_index in range(len(AXES)):
                relative_complex = (
                    coefficients[chunk_index, 1, output_index + 1]
                    - 1j * coefficients[chunk_index, 0, output_index + 1]
                )
                absolute_complex = (
                    coefficients[chunk_index, 1, output_index + 1 + len(AXES)]
                    - 1j * coefficients[chunk_index, 0, output_index + 1 + len(AXES)]
                )
                relative_residual = float(np.sqrt(np.mean(residual[chunk_index, :, output_index + 1] ** 2)))
                absolute_residual = float(
                    np.sqrt(np.mean(residual[chunk_index, :, output_index + 1 + len(AXES)] ** 2))
                )
                relative_fits.append(
                    HarmonicFit(
                        frequency_hz=float(frequency),
                        sample_count=int(fit_time.size),
                        sample_rate_hz=sample_rate,
                        fit_start_s=float(fit_time[0]),
                        fit_end_s=float(fit_time[-1]),
                        input_complex_coefficient=complex(input_complex[chunk_index]),
                        output_complex_coefficient=complex(relative_complex),
                        transfer_complex=complex(relative_complex / input_complex[chunk_index]),
                        input_amplitude=float(input_amplitude[chunk_index]),
                        output_amplitude=float(abs(relative_complex)),
                        amplitude_ratio=float(abs(relative_complex / input_complex[chunk_index])),
                        phase_difference_rad=float(np.angle(relative_complex / input_complex[chunk_index])),
                        input_fit_residual_rms=float(input_residual[chunk_index]),
                        output_fit_residual_rms=relative_residual,
                        normalized_input_residual=float(input_normalized[chunk_index]),
                        normalized_output_residual=float(
                            relative_residual / max(abs(relative_complex), np.finfo(float).tiny)
                        ),
                        condition_number=float(condition[chunk_index]),
                    )
                )
                absolute_fits.append(
                    HarmonicFit(
                        frequency_hz=float(frequency),
                        sample_count=int(fit_time.size),
                        sample_rate_hz=sample_rate,
                        fit_start_s=float(fit_time[0]),
                        fit_end_s=float(fit_time[-1]),
                        input_complex_coefficient=complex(input_complex[chunk_index]),
                        output_complex_coefficient=complex(absolute_complex),
                        transfer_complex=complex(absolute_complex / input_complex[chunk_index]),
                        input_amplitude=float(input_amplitude[chunk_index]),
                        output_amplitude=float(abs(absolute_complex)),
                        amplitude_ratio=float(abs(absolute_complex / input_complex[chunk_index])),
                        phase_difference_rad=float(np.angle(absolute_complex / input_complex[chunk_index])),
                        input_fit_residual_rms=float(input_residual[chunk_index]),
                        output_fit_residual_rms=absolute_residual,
                        normalized_input_residual=float(input_normalized[chunk_index]),
                        normalized_output_residual=float(
                            absolute_residual / max(abs(absolute_complex), np.finfo(float).tiny)
                        ),
                        condition_number=float(condition[chunk_index]),
                    )
                )
            output.append((relative_fits, absolute_fits))
    return output


def _fit_multifrequency_output_matrices(
    time: np.ndarray,
    input_signal: np.ndarray,
    relative_outputs: np.ndarray,
    absolute_outputs: np.ndarray,
    frequencies_hz: Iterable[float],
    *,
    discard_time_s: float,
    fit_duration_s: float,
) -> list[tuple[list[HarmonicFit], list[HarmonicFit]]]:
    """Fit a complete multi-line spectrum in one shared regression.

    A one-line regression's residual necessarily contains every other
    authored tone.  The joint spectrum contract therefore uses one design
    matrix containing all sine/cosine pairs for an input axis and extracts
    each line's complex coefficient from that shared fit.  This makes the
    residual a genuine model/noise residual instead of an intentional
    omitted-tone residual.
    """

    frequencies = np.asarray(tuple(frequencies_hz), dtype=float).reshape(-1)
    if frequencies.size == 0 or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise IsolatorError("frequencies_hz must contain positive finite values")
    times = np.asarray(time, dtype=float)
    inputs = np.asarray(input_signal, dtype=float)
    relative_values = np.asarray(relative_outputs, dtype=float)
    absolute_values = np.asarray(absolute_outputs, dtype=float)
    if (
        times.ndim != 1
        or inputs.ndim != 1
        or relative_values.shape != (times.size, len(AXES))
        or absolute_values.shape != (times.size, len(AXES))
    ):
        raise IsolatorError("multifrequency output matrices must have shape (N, 6)")
    discard = _coerce_scalar("discard_time_s", discard_time_s, nonnegative=True)
    duration = _coerce_scalar("fit_duration_s", fit_duration_s, positive=True)
    selected = (times >= discard) & (times <= discard + duration + 1e-12)
    if np.count_nonzero(selected) < 4:
        raise IsolatorError("transient discard / fit duration leaves fewer than four samples")
    fit_time = times[selected]
    right_hand_side = np.column_stack((inputs[selected], relative_values[selected], absolute_values[selected]))
    angular = 2.0 * math.pi * frequencies[:, None] * fit_time[None, :]
    basis = np.empty((fit_time.size, frequencies.size * 2 + 1), dtype=float)
    basis[:, :-1:2] = np.sin(angular).T
    basis[:, 1:-1:2] = np.cos(angular).T
    basis[:, -1] = 1.0
    gram = basis.T.dot(basis)
    rhs = basis.T.dot(right_hand_side)
    coefficients = np.linalg.solve(gram, rhs)
    residual = right_hand_side - basis.dot(coefficients)
    condition = float(np.sqrt(np.linalg.cond(gram)))
    sample_rate = float(1.0 / np.median(np.diff(fit_time)))
    input_coefficients = coefficients[:-1:2, 0] - 1j * coefficients[1:-1:2, 0]
    input_amplitudes = np.abs(input_coefficients)
    if np.any(input_amplitudes <= np.finfo(float).eps):
        raise IsolatorError("multifrequency input coefficient is too small to form a transfer")
    input_residual = float(np.sqrt(np.mean(residual[:, 0] ** 2)))
    input_total_rms = float(np.sqrt(np.sum(input_amplitudes**2) / 2.0))
    input_normalized = input_residual / max(input_total_rms, np.finfo(float).tiny)
    output = []
    for line_index, frequency in enumerate(frequencies):
        relative_fits = []
        absolute_fits = []
        for output_index in range(len(AXES)):
            relative_complex = complex(
                coefficients[2 * line_index + 1, output_index + 1],
                -coefficients[2 * line_index, output_index + 1],
            )
            absolute_complex = complex(
                coefficients[2 * line_index + 1, output_index + 1 + len(AXES)],
                -coefficients[2 * line_index, output_index + 1 + len(AXES)],
            )
            relative_coefficients = coefficients[1:-1:2, output_index + 1] - 1j * coefficients[
                0:-1:2, output_index + 1
            ]
            relative_total_rms = float(np.sqrt(np.sum(np.abs(relative_coefficients) ** 2) / 2.0))
            absolute_total_rms = float(
                np.sqrt(
                    np.sum(
                        np.abs(
                            coefficients[1:-1:2, output_index + 1 + len(AXES)]
                            - 1j * coefficients[0:-1:2, output_index + 1 + len(AXES)]
                        )
                        ** 2
                    )
                    / 2.0
                )
            )
            relative_residual = float(np.sqrt(np.mean(residual[:, output_index + 1] ** 2)))
            absolute_residual = float(
                np.sqrt(np.mean(residual[:, output_index + 1 + len(AXES)] ** 2))
            )
            relative_fits.append(
                HarmonicFit(
                    frequency_hz=float(frequency),
                    sample_count=int(fit_time.size),
                    sample_rate_hz=sample_rate,
                    fit_start_s=float(fit_time[0]),
                    fit_end_s=float(fit_time[-1]),
                    input_complex_coefficient=complex(input_coefficients[line_index]),
                    output_complex_coefficient=relative_complex,
                    transfer_complex=complex(relative_complex / input_coefficients[line_index]),
                    input_amplitude=float(input_amplitudes[line_index]),
                    output_amplitude=float(abs(relative_complex)),
                    amplitude_ratio=float(abs(relative_complex / input_coefficients[line_index])),
                    phase_difference_rad=float(np.angle(relative_complex / input_coefficients[line_index])),
                    input_fit_residual_rms=input_residual,
                    output_fit_residual_rms=relative_residual,
                    normalized_input_residual=input_normalized,
                    normalized_output_residual=relative_residual / max(relative_total_rms, np.finfo(float).tiny),
                    condition_number=condition,
                )
            )
            absolute_fits.append(
                HarmonicFit(
                    frequency_hz=float(frequency),
                    sample_count=int(fit_time.size),
                    sample_rate_hz=sample_rate,
                    fit_start_s=float(fit_time[0]),
                    fit_end_s=float(fit_time[-1]),
                    input_complex_coefficient=complex(input_coefficients[line_index]),
                    output_complex_coefficient=absolute_complex,
                    transfer_complex=complex(absolute_complex / input_coefficients[line_index]),
                    input_amplitude=float(input_amplitudes[line_index]),
                    output_amplitude=float(abs(absolute_complex)),
                    amplitude_ratio=float(abs(absolute_complex / input_coefficients[line_index])),
                    phase_difference_rad=float(np.angle(absolute_complex / input_coefficients[line_index])),
                    input_fit_residual_rms=input_residual,
                    output_fit_residual_rms=absolute_residual,
                    normalized_input_residual=input_normalized,
                    normalized_output_residual=absolute_residual / max(absolute_total_rms, np.finfo(float).tiny),
                    condition_number=condition,
                )
            )
        output.append((relative_fits, absolute_fits))
    return output


def _fit_global_spectrum_outputs(
    time: np.ndarray,
    input_signals: np.ndarray,
    relative_outputs: np.ndarray,
    absolute_outputs: np.ndarray,
    line_frequencies_hz: Iterable[float],
    line_input_indices: Iterable[int],
    *,
    discard_time_s: float,
    fit_duration_s: float,
) -> dict[tuple[int, int], tuple[list[HarmonicFit], list[HarmonicFit]]]:
    """Fit all authored lines and all six outputs in one shared regression."""

    frequencies = np.asarray(tuple(line_frequencies_hz), dtype=float).reshape(-1)
    input_indices = np.asarray(tuple(line_input_indices), dtype=int).reshape(-1)
    if frequencies.size == 0 or frequencies.size != input_indices.size:
        raise IsolatorError("line frequencies and input indices must have equal non-zero length")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise IsolatorError("line frequencies must be positive and finite")
    if np.any(input_indices < 0) or np.any(input_indices >= len(AXES)):
        raise IsolatorError("line input indices are outside the six-axis range")
    times = np.asarray(time, dtype=float)
    input_values = np.asarray(input_signals, dtype=float)
    relative_values = np.asarray(relative_outputs, dtype=float)
    absolute_values = np.asarray(absolute_outputs, dtype=float)
    if (
        times.ndim != 1
        or input_values.shape != (times.size, len(AXES))
        or relative_values.shape != (times.size, len(AXES))
        or absolute_values.shape != (times.size, len(AXES))
    ):
        raise IsolatorError("global spectrum signals must have shape (N, 6)")
    discard = _coerce_scalar("discard_time_s", discard_time_s, nonnegative=True)
    duration = _coerce_scalar("fit_duration_s", fit_duration_s, positive=True)
    selected = (times >= discard) & (times <= discard + duration + 1e-12)
    if np.count_nonzero(selected) < 4:
        raise IsolatorError("transient discard / fit duration leaves fewer than four samples")
    fit_time = times[selected]
    angular = 2.0 * math.pi * frequencies[:, None] * fit_time[None, :]
    basis = np.empty((fit_time.size, frequencies.size * 2 + 1), dtype=float)
    basis[:, :-1:2] = np.sin(angular).T
    basis[:, 1:-1:2] = np.cos(angular).T
    basis[:, -1] = 1.0
    right_hand_side = np.column_stack((input_values[selected], relative_values[selected], absolute_values[selected]))
    coefficients, _, _, _ = np.linalg.lstsq(basis, right_hand_side, rcond=None)
    residual = right_hand_side - basis.dot(coefficients)
    condition = float(np.linalg.cond(basis))
    sample_rate = float(1.0 / np.median(np.diff(fit_time)))
    input_complex_all = coefficients[1:-1:2, : len(AXES)] - 1j * coefficients[:-1:2, : len(AXES)]
    input_total_rms = np.sqrt(np.sum(np.abs(input_complex_all) ** 2, axis=0) / 2.0)
    input_residual = np.sqrt(np.mean(residual[:, : len(AXES)] ** 2, axis=0))
    input_normalized = input_residual / np.maximum(input_total_rms, np.finfo(float).tiny)
    output = {}
    for line_index, frequency in enumerate(frequencies):
        relative_complex_all = coefficients[1:-1:2, len(AXES) : 2 * len(AXES)] - 1j * coefficients[
            :-1:2, len(AXES) : 2 * len(AXES)
        ]
        absolute_complex_all = coefficients[1:-1:2, 2 * len(AXES) :] - 1j * coefficients[
            :-1:2, 2 * len(AXES) :
        ]
        relative_total_rms = np.sqrt(np.sum(np.abs(relative_complex_all) ** 2, axis=0) / 2.0)
        absolute_total_rms = np.sqrt(np.sum(np.abs(absolute_complex_all) ** 2, axis=0) / 2.0)
        relative_residual = np.sqrt(np.mean(residual[:, len(AXES) : 2 * len(AXES)] ** 2, axis=0))
        absolute_residual = np.sqrt(np.mean(residual[:, 2 * len(AXES) :] ** 2, axis=0))
        relative_fits = []
        absolute_fits = []
        source_input_index = int(input_indices[line_index])
        input_complex = complex(input_complex_all[line_index, source_input_index])
        for output_index in range(len(AXES)):
            relative_complex = complex(relative_complex_all[line_index, output_index])
            absolute_complex = complex(absolute_complex_all[line_index, output_index])
            relative_fits.append(
                HarmonicFit(
                    frequency_hz=float(frequency),
                    sample_count=int(fit_time.size),
                    sample_rate_hz=sample_rate,
                    fit_start_s=float(fit_time[0]),
                    fit_end_s=float(fit_time[-1]),
                    input_complex_coefficient=input_complex,
                    output_complex_coefficient=relative_complex,
                    transfer_complex=relative_complex / input_complex,
                    input_amplitude=float(abs(input_complex)),
                    output_amplitude=float(abs(relative_complex)),
                    amplitude_ratio=float(abs(relative_complex / input_complex)),
                    phase_difference_rad=float(np.angle(relative_complex / input_complex)),
                    input_fit_residual_rms=float(input_residual[source_input_index]),
                    output_fit_residual_rms=float(relative_residual[output_index]),
                    normalized_input_residual=float(input_normalized[source_input_index]),
                    normalized_output_residual=float(
                        relative_residual[output_index] / max(relative_total_rms[output_index], np.finfo(float).tiny)
                    ),
                    condition_number=condition,
                )
            )
            absolute_fits.append(
                HarmonicFit(
                    frequency_hz=float(frequency),
                    sample_count=int(fit_time.size),
                    sample_rate_hz=sample_rate,
                    fit_start_s=float(fit_time[0]),
                    fit_end_s=float(fit_time[-1]),
                    input_complex_coefficient=input_complex,
                    output_complex_coefficient=absolute_complex,
                    transfer_complex=absolute_complex / input_complex,
                    input_amplitude=float(abs(input_complex)),
                    output_amplitude=float(abs(absolute_complex)),
                    amplitude_ratio=float(abs(absolute_complex / input_complex)),
                    phase_difference_rad=float(np.angle(absolute_complex / input_complex)),
                    input_fit_residual_rms=float(input_residual[source_input_index]),
                    output_fit_residual_rms=float(absolute_residual[output_index]),
                    normalized_input_residual=float(input_normalized[source_input_index]),
                    normalized_output_residual=float(
                        absolute_residual[output_index] / max(absolute_total_rms[output_index], np.finfo(float).tiny)
                    ),
                    condition_number=condition,
                )
            )
        output[(source_input_index, line_index)] = (relative_fits, absolute_fits)
    return output


def _fit_line_record(
    arrays: dict[str, np.ndarray],
    *,
    input_axis_index: int,
    output_axis_index: int,
    frequency_hz: float,
    discard_time_s: float,
    fit_duration_s: float,
    parameters: IsolatorParameters,
) -> dict[str, Any]:
    input_signal = arrays["deck_absolute"][:, input_axis_index]
    relative_signal = arrays["relative"][:, output_axis_index]
    absolute_signal = arrays["table_absolute"][:, output_axis_index]
    relative_fit = fit_harmonic_transfer(
        arrays["time_s"],
        input_signal,
        relative_signal,
        frequency_hz,
        discard_time_s=discard_time_s,
        fit_duration_s=fit_duration_s,
    )
    absolute_fit = fit_harmonic_transfer(
        arrays["time_s"],
        input_signal,
        absolute_signal,
        frequency_hz,
        discard_time_s=discard_time_s,
        fit_duration_s=fit_duration_s,
    )
    expected_relative = complex(
        relative_transfer_function(
            frequency_hz,
            parameters.fn_hz[output_axis_index],
            parameters.zeta[output_axis_index],
        )
    )
    expected_absolute = complex(
        absolute_transfer_function(
            frequency_hz,
            parameters.fn_hz[output_axis_index],
            parameters.zeta[output_axis_index],
        )
    )
    return {
        "input_axis": AXES[input_axis_index],
        "output_axis": AXES[output_axis_index],
        "frequency_hz": float(frequency_hz),
        "relative_fit": relative_fit.to_dict(),
        "absolute_fit": absolute_fit.to_dict(),
        "analytic_relative_transfer": _complex_pair(expected_relative),
        "analytic_absolute_transfer": _complex_pair(expected_absolute),
        "relative_input_output_coefficient": {
            "input": _complex_pair(relative_fit.input_complex_coefficient),
            "output": _complex_pair(relative_fit.output_complex_coefficient),
        },
        "absolute_input_output_coefficient": {
            "input": _complex_pair(absolute_fit.input_complex_coefficient),
            "output": _complex_pair(absolute_fit.output_complex_coefficient),
        },
        "relative_leakage": float(
            abs(relative_fit.output_complex_coefficient) / abs(relative_fit.input_complex_coefficient)
        ),
        "absolute_leakage": float(
            abs(absolute_fit.output_complex_coefficient) / abs(absolute_fit.input_complex_coefficient)
        ),
    }


def _minimum_active_line_spacing_hz(program: Any) -> float:
    frequencies = np.asarray(program.line_frequency_hz[program.line_mask], dtype=float)
    if frequencies.size < 2:
        raise IsolatorError("joint spectrum requires at least two active authored lines")
    differences = np.diff(np.sort(frequencies))
    positive = differences[differences > 0.0]
    if positive.size == 0:
        raise IsolatorError("joint spectrum requires distinct authored line frequencies")
    return float(np.min(positive))


def _spectrum_summary(
    arrays: dict[str, np.ndarray],
    *,
    start_s: float,
    duration_s: float,
    config: IsolatorConfig,
    parameters: IsolatorParameters,
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    time = arrays["time_s"]
    relative = _window_values(arrays["relative"], time, start_s, duration_s)
    table_absolute = _window_values(arrays["table_absolute"], time, start_s, duration_s)
    deck_absolute = _window_values(arrays["deck_absolute"], time, start_s, duration_s)
    deck_acceleration = _window_values(arrays["deck_acceleration"], time, start_s, duration_s)
    table_acceleration = _window_values(arrays["table_acceleration"], time, start_s, duration_s)
    deck_rms = np.sqrt(np.mean(deck_acceleration**2, axis=0))
    table_rms = np.sqrt(np.mean(table_acceleration**2, axis=0))
    relative_acceleration = table_acceleration - deck_acceleration
    relative_rms = np.sqrt(np.mean(relative_acceleration**2, axis=0))
    t_accel = np.divide(table_rms, np.maximum(deck_rms, np.finfo(float).tiny))
    r_relative = np.divide(relative_rms, np.maximum(deck_rms, np.finfo(float).tiny))
    translation_displacement = np.max(np.abs(relative[:, :3]), axis=0)
    rotation_displacement = np.max(np.abs(relative[:, 3:]), axis=0)
    translation_norm = np.linalg.norm(relative[:, :3], axis=1)
    rotation_norm = np.linalg.norm(relative[:, 3:], axis=1)
    return {
        "T_accel": t_accel.tolist(),
        "R_relative": r_relative.tolist(),
        "D_relative_m": float(np.max(translation_norm)),
        "D_relative_rad": float(np.max(rotation_norm)),
        "D_relative_by_axis": np.concatenate((translation_displacement, rotation_displacement)).tolist(),
        "T_peak": None,
        "travel_margin_m": (np.asarray(config.travel_limits_m) - translation_displacement).tolist(),
        "angle_margin_rad": (np.asarray(config.angle_limits_rad) - rotation_displacement).tolist(),
        "static_sag_uncompensated_m": static_sag_uncompensated_m(parameters),
        "static_offset_compensated": [0.0] * 6,
        "window_start_s": float(start_s),
        "window_duration_s": float(duration_s),
        "deck_acceleration_rms": deck_rms.tolist(),
        "table_acceleration_rms": table_rms.tolist(),
        "trace": trace_summary,
    }


def run_joint_spectrum_probe(
    *,
    isolator_config: Any = None,
    timestep_s: float = 0.0002,
    deck_config: Any = None,
    seed: int = 17,
    t0_s: float = 0.137,
    level_scale: float = 0.3,
    excitation_config: Any = None,
    transient_discard_s: float = 2.0,
    fit_duration_s: Optional[float] = None,
    visual: bool = True,
    thresholds: Mapping[str, float] = PHASE03_TRANSFER_THRESHOLDS,
) -> dict[str, Any]:
    """Run one full Phase 01 authored six-axis spectrum in MuJoCo.

    Each active input line is fit against the realized deck-origin input and
    every relative/absolute worktable output axis.  The diagonal records are
    checked against the analytic complex transfer; all off-diagonal records
    are reported as leakage ratios and checked against the independent-axis
    bound.
    """

    from robosuite.utils.shakebench_excitation import (
        AUTHORED_PROFILE_ID,
        AUTHORED_SPECTRUM_VERSION,
        DEFAULT_EXCITATION_CONFIG,
        build_excitation_program,
    )
    from robosuite.utils.shakebench_deck import DeckCommand

    config = _coerce_config(isolator_config)
    parameters = derive_isolator_parameters(config)
    excitation = DEFAULT_EXCITATION_CONFIG if excitation_config is None else excitation_config
    program = build_excitation_program(
        seed=seed,
        t0=t0_s,
        level_scale=level_scale,
        config=excitation,
    )
    spacing = _minimum_active_line_spacing_hz(program)
    minimum_resolution_duration = 2.0 / spacing
    fit_duration = (
        minimum_resolution_duration
        if fit_duration_s is None
        else _coerce_scalar("fit_duration_s", fit_duration_s, positive=True)
    )
    discard = _coerce_scalar("transient_discard_s", transient_discard_s, nonnegative=True)

    def trajectory(time_s: float) -> DeckCommand:
        motion = program.evaluate(time_s)
        return DeckCommand(motion.q, motion.qdot, motion.qdd)

    probe = _run_probe_trajectory(
        config,
        timestep_s=timestep_s,
        duration_s=discard + fit_duration + float(timestep_s),
        trajectory=trajectory,
        deck_config=deck_config,
        visual=visual,
        gravity_vector=(0.0, 0.0, 0.0),
        neutralize_preload=True,
    )
    arrays = probe["arrays"]
    line_records = []
    diagonal_comparisons = []
    leakage_values = []
    line_frequencies = []
    line_input_indices = []
    line_key_map = {}
    global_line_index = 0
    for input_index, input_axis in enumerate(AXES):
        active_lines = np.flatnonzero(program.line_mask[input_index])
        for line_index in active_lines:
            line_frequencies.append(float(program.line_frequency_hz[input_index, line_index]))
            line_input_indices.append(input_index)
            line_key_map[(input_index, int(line_index))] = global_line_index
            global_line_index += 1
    global_line_fit_cache = _fit_global_spectrum_outputs(
        arrays["time_s"],
        arrays["deck_absolute"],
        arrays["relative"],
        arrays["table_absolute"],
        line_frequencies,
        line_input_indices,
        discard_time_s=discard,
        fit_duration_s=fit_duration,
    )
    line_fit_cache = {
        key: global_line_fit_cache[(key[0], global_index)]
        for key, global_index in line_key_map.items()
    }
    for input_index, input_axis in enumerate(AXES):
        active_lines = np.flatnonzero(program.line_mask[input_index])
        for line_index in active_lines:
            frequency = float(program.line_frequency_hz[input_index, line_index])
            relative_fits, absolute_fits = line_fit_cache[(input_index, int(line_index))]
            relative_direct = relative_fits[input_index]
            absolute_direct = absolute_fits[input_index]
            relative_expected = complex(
                relative_transfer_function(frequency, parameters.fn_hz[input_index], parameters.zeta[input_index])
            )
            absolute_expected = complex(
                absolute_transfer_function(frequency, parameters.fn_hz[input_index], parameters.zeta[input_index])
            )
            relative_comparison = compare_harmonic_fit(
                relative_direct,
                relative_expected,
                thresholds=thresholds,
            )
            absolute_comparison = compare_harmonic_fit(
                absolute_direct,
                absolute_expected,
                thresholds=thresholds,
            )
            cross_axis = {}
            for output_index, output_axis in enumerate(AXES):
                cross_relative_fit = relative_fits[output_index]
                cross_absolute_fit = absolute_fits[output_index]
                leakage = float(cross_relative_fit.amplitude_ratio)
                cross_axis[output_axis] = {
                    "input_axis": input_axis,
                    "output_axis": output_axis,
                    "frequency_hz": frequency,
                    "relative_fit": cross_relative_fit.to_dict(),
                    "absolute_fit": cross_absolute_fit.to_dict(),
                    "relative_leakage": leakage,
                    "absolute_leakage": float(cross_absolute_fit.amplitude_ratio),
                    "is_diagonal": output_index == input_index,
                }
                if output_index != input_index:
                    leakage_values.append(leakage)
            line_records.append(
                {
                    "input_axis": input_axis,
                    "input_axis_index": input_index,
                    "line_index": int(line_index),
                    "frequency_hz": frequency,
                    "authored_input_amplitude": float(program.line_accel_amplitude[input_index, line_index]),
                    "relative_fit": relative_direct.to_dict(),
                    "absolute_fit": absolute_direct.to_dict(),
                    "analytic_relative_transfer": _complex_pair(relative_expected),
                    "analytic_absolute_transfer": _complex_pair(absolute_expected),
                    "relative_comparison": relative_comparison,
                    "absolute_comparison": absolute_comparison,
                    "cross_axis_relative_fits": cross_axis,
                }
            )
            diagonal_comparisons.extend((relative_comparison, absolute_comparison))

    leakage_max = max(leakage_values) if leakage_values else 0.0
    leakage_passed = leakage_max <= float(thresholds["cross_axis_leakage_relative_max"])
    summary = _spectrum_summary(
        arrays,
        start_s=discard,
        duration_s=fit_duration,
        config=config,
        parameters=parameters,
        trace_summary=_probe_trace_summary(probe),
    )
    t_peak = max(
        (abs(_pair_to_complex(record["analytic_absolute_transfer"])) for record in line_records),
        default=0.0,
    )
    summary["T_peak"] = float(t_peak)
    safety = {
        "travel_margin_m": summary["travel_margin_m"],
        "angle_margin_rad": summary["angle_margin_rad"],
        "passed": all(value > 0.0 for value in summary["travel_margin_m"] + summary["angle_margin_rad"]),
    }
    diagonal_passed = bool(all(comparison["passed"] for comparison in diagonal_comparisons))
    return {
        "schema_id": PHASE03_TRANSFER_SCHEMA_ID,
        "schema_version": PHASE03_TRANSFER_SCHEMA_VERSION,
        "axis_order": list(AXES),
        "reference": {
            "input_point": "deck body origin / deck site",
            "table_point": "worktable body origin = COM = elastic center",
            "relative_coordinate": "table COM relative to deck-origin, expressed in realized deck frame",
            "absolute_coordinate": "table COM pose in nominal worktable frame",
            "frame": "nominal deck frame for input/absolute output; realized deck frame for relative output",
            "linearization_gravity_vector_m_s2": [0.0, 0.0, 0.0],
            "linearization_springref_override": {"isolator_tz": 0.0},
            "time_convention": "write q(t), integrate to t+dt, write q(t+dt), forward, sample at right limit",
            "fit_convention": "x=a_s*sin(wt)+a_c*cos(wt)+b; phasor=a_c-i*a_s; H=Y/X; phase=arg(H)",
        },
        "program": {
            "profile_id": AUTHORED_PROFILE_ID,
            "authored_spectrum_version": AUTHORED_SPECTRUM_VERSION,
            "seed": int(program.seed),
            "t0_s": float(program.t0),
            "level_scale": float(program.level_scale),
            "program": program.to_dict(),
            "minimum_line_spacing_hz": spacing,
            "minimum_resolution_duration_s": minimum_resolution_duration,
        },
        "transient_discard_s": discard,
        "fit_duration_s": fit_duration,
        "line_fit_count": len(line_records),
        "line_fits": line_records,
        "cross_axis_leakage_max": leakage_max,
        "cross_axis_leakage_gate_passed": leakage_passed,
        "metrics": summary,
        "safety": safety,
        "gate_summary": {
            "diagonal_complex_transfer_passed": diagonal_passed,
            "cross_axis_leakage_passed": leakage_passed,
            "safety_passed": safety["passed"],
            "passed": bool(diagonal_passed and leakage_passed and safety["passed"]),
        },
    }


def _pair_to_complex(value: Iterable[float]) -> complex:
    pair = tuple(float(item) for item in value)
    if len(pair) != 2:
        raise IsolatorError("complex pair must contain two values")
    return complex(pair[0], pair[1])


def _payload_contact_summary(model: Any, data: Any) -> dict[str, Any]:
    """Describe final payload/table contacts without depending on task code."""

    mujoco = _lazy_mujoco()
    names = []
    payload_contacts = 0
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = (None if first is None else str(first), None if second is None else str(second))
        names.append(pair)
        if "phase03_payload_geom" in pair and "table_collision" in pair:
            payload_contacts += 1
    return {
        "ncon": int(data.ncon),
        "pairs": [list(pair) for pair in names],
        "payload_table_contact_count": payload_contacts,
        "payload_supported_by_table": payload_contacts > 0,
    }


def _payload_com_offset_in_table_frame(model: Any, data: Any) -> list[float]:
    """Return the payload COM xy offset from the compiled worktable origin."""

    payload_id = int(model.body("phase03_payload").id)
    table_id = int(model.body("worktable").id)
    table_position, table_quaternion = _body_pose_state(data, table_id)
    table_rotation = _quaternion_to_matrix_wxyz(table_quaternion)
    payload_position = np.asarray(data.xpos[payload_id], dtype=float)
    relative_position = table_rotation.T.dot(payload_position - table_position)
    return [float(relative_position[0]), float(relative_position[1])]


def _payload_equilibrium_gate(
    measured: np.ndarray,
    expected: np.ndarray,
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Compare the force-path coordinates with sign and magnitude gates."""

    checks = {}
    passed = True
    relative_limit = float(thresholds["payload_relative_error_max"])
    absolute_limit = float(thresholds["payload_absolute_offset_tolerance"])
    for index, axis in enumerate(AXES):
        expected_value = float(expected[index])
        measured_value = float(measured[index])
        if abs(expected_value) <= absolute_limit:
            sign_passed = abs(measured_value) <= absolute_limit
            relative_error = None
            amplitude_passed = sign_passed
        else:
            sign_passed = measured_value * expected_value > 0.0
            relative_error = abs(measured_value / expected_value - 1.0)
            amplitude_passed = relative_error <= relative_limit
        axis_passed = bool(sign_passed and amplitude_passed)
        passed = passed and axis_passed
        checks[axis] = {
            "measured": measured_value,
            "expected": expected_value,
            "sign_passed": sign_passed,
            "relative_error": relative_error,
            "amplitude_passed": amplitude_passed,
            "passed": axis_passed,
        }
    return {"checks": checks, "passed": bool(passed)}


def run_payload_sensitivity_probe(
    *,
    isolator_config: Any = None,
    timestep_s: float = 0.0002,
    deck_config: Any = None,
    payload_mass_kg: float = 0.349,
    com_offsets_m: Iterable[Iterable[float]] = ((0.0, 0.0), (0.10, 0.0), (0.0, 0.20), (0.10, 0.20)),
    settle_duration_s: float = 8.0,
    equilibrium_window_s: float = 1.0,
    visual: bool = True,
    thresholds: Mapping[str, float] = PHASE03_TRANSFER_THRESHOLDS,
) -> dict[str, Any]:
    """Measure centered and offset world-child payload equilibrium in MuJoCo."""

    config = _coerce_config(isolator_config)
    parameters = derive_isolator_parameters(config)
    payload_mass = _coerce_scalar("payload_mass_kg", payload_mass_kg, positive=True)
    settle_duration = _coerce_scalar("settle_duration_s", settle_duration_s, positive=True)
    equilibrium_window = _coerce_scalar("equilibrium_window_s", equilibrium_window_s, positive=True)
    offsets = []
    for offset in com_offsets_m:
        values = np.asarray(tuple(offset), dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise IsolatorError("each payload COM offset must contain two finite x/y values")
        offsets.append((float(values[0]), float(values[1])))
    from robosuite.utils.shakebench_deck import DeckCommand

    zero_trajectory = lambda _: DeckCommand(np.zeros(6), np.zeros(6), np.zeros(6))
    records = []
    support_signature = parameters.to_dict()
    for offset in offsets:
        probe = _run_probe_trajectory(
            config,
            timestep_s=timestep_s,
            duration_s=settle_duration,
            trajectory=zero_trajectory,
            deck_config=deck_config,
            visual=visual,
            payload_offset_m=offset,
            payload_mass_kg=payload_mass,
            payload_geom_type="box",
            payload_geom_size=(0.025, 0.025, 0.01),
        )
        arrays = probe["arrays"]
        equilibrium = np.mean(
            _window_values(arrays["joint_qpos"], arrays["time_s"], settle_duration - equilibrium_window),
            axis=0,
        )
        expected = static_equilibrium_offset(
            parameters,
            payload_mass_kg=payload_mass,
            payload_com_m=(offset[0], offset[1], 0.0),
        )
        equilibrium_gate = _payload_equilibrium_gate(equilibrium, expected, thresholds=thresholds)
        travel_margin = np.asarray(config.travel_limits_m) - np.abs(equilibrium[:3])
        angle_margin = np.asarray(config.angle_limits_rad) - np.abs(equilibrium[3:])
        safety_passed = bool(np.all(travel_margin > 0.0) and np.all(angle_margin > 0.0))
        table_body_id = int(probe["model"].body("worktable").id)
        table_mass_unchanged = bool(
            np.isclose(probe["model"].body_mass[table_body_id], parameters.mass_kg, rtol=0.0, atol=1e-12)
        )
        contact = _payload_contact_summary(probe["model"], probe["data"])
        actual_com_offset = _payload_com_offset_in_table_frame(probe["model"], probe["data"])
        com_offset_error = float(np.linalg.norm(np.asarray(actual_com_offset) - np.asarray(offset)))
        com_path_passed = com_offset_error <= 1.0e-3
        compiled_support = probe["arena"].audit_compiled_model(probe["model"])["physics_signature"]
        support_parameters_unchanged = bool(
            np.isclose(compiled_support["mass_kg"], support_signature["mass_kg"], rtol=0.0, atol=1.0e-12)
            and np.allclose(compiled_support["com_m"], (0.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)
            and np.allclose(
                compiled_support["inertia_kg_m2"], support_signature["inertia_kg_m2"], rtol=0.0, atol=1.0e-12
            )
            and np.allclose(compiled_support["stiffness"], support_signature["stiffness"], rtol=0.0, atol=1.0e-12)
            and np.allclose(compiled_support["damping"], support_signature["damping"], rtol=0.0, atol=1.0e-12)
            and np.allclose(compiled_support["springref"], support_signature["springref"], rtol=0.0, atol=1.0e-12)
        )
        record = {
            "payload_mass_kg": payload_mass,
            "com_offset_m": list(offset),
            "payload_topology": "world child + freejoint; contact transfers gravity load",
            "equilibrium_window_s": equilibrium_window,
            "measured_joint_equilibrium": equilibrium.tolist(),
            "analytic_joint_equilibrium": expected.tolist(),
            "actual_com_offset_m": actual_com_offset,
            "com_offset_error_m": com_offset_error,
            "com_path_passed": com_path_passed,
            "equilibrium_gate": equilibrium_gate,
            "travel_margin_m": travel_margin.tolist(),
            "angle_margin_rad": angle_margin.tolist(),
            "safety_passed": safety_passed,
            "table_mass_unchanged": table_mass_unchanged,
            "support_parameters": support_signature,
            "compiled_support_signature": compiled_support,
            "support_parameters_unchanged": support_parameters_unchanged,
            "contact": contact,
            "warnings": probe["summary"]["warning_number_delta_total"],
            "trace_summary": _probe_trace_summary(probe),
        }
        record["passed"] = bool(
            equilibrium_gate["passed"]
            and safety_passed
            and com_path_passed
            and table_mass_unchanged
            and support_parameters_unchanged
            and contact["payload_supported_by_table"]
            and not any(record["warnings"])
        )
        records.append(record)
    return {
        "schema_id": PHASE03_TRANSFER_SCHEMA_ID,
        "schema_version": PHASE03_TRANSFER_SCHEMA_VERSION,
        "axis_order": list(AXES),
        "payload_mass_kg": payload_mass,
        "com_offsets_m": [list(offset) for offset in offsets],
        "thresholds": dict(thresholds),
        "records": records,
        "support_parameters": support_signature,
        "passed": bool(all(record["passed"] for record in records)),
    }


def _phase03_visual_invariance_signature(*, isolator_config: Any = None, timestep_s: float = 0.0002) -> dict[str, Any]:
    """Compile visible/hidden variants and compare physics plus one transfer trace."""

    mujoco = _lazy_mujoco()
    from robosuite.models.arenas.shakebench_arena import ShakeBenchArena
    from robosuite.utils.shakebench_deck import DeckDriverConfig

    config = _coerce_config(isolator_config)
    variants = {}
    for visible in (True, False):
        arena = ShakeBenchArena(isolator_config=config, visual=visible)
        deck_config = DeckDriverConfig(
            physics_timestep_s=timestep_s,
            eq_solref=(max(2.0 * timestep_s, 0.0004), 0.5),
            deck_pos_m=tuple(float(value) for value in arena.center_pos),
        )
        driver_xml = _set_probe_timestep(arena.get_xml(), timestep_s)
        from robosuite.utils.shakebench_deck import DeckDriver

        driver = DeckDriver(config=deck_config, body_handles=arena.deck_body_handles)
        processed_xml = driver.processor(driver_xml)
        model = mujoco.MjModel.from_xml_string(processed_xml)
        audit = arena.audit_compiled_model(model)
        fields = (
            "body_mass",
            "body_ipos",
            "body_inertia",
            "jnt_type",
            "jnt_axis",
            "jnt_stiffness",
            "dof_damping",
            "qpos_spring",
            "jnt_range",
            "jnt_limited",
            "geom_contype",
            "geom_conaffinity",
        )
        hashes = {field: _array_sha256(np.asarray(getattr(model, field))) for field in fields}
        variants["visible" if visible else "hidden"] = {
            "visual": bool(visible),
            "physics_hashes": hashes,
            "physics_signature": audit["physics_signature"],
            "geom_rgba_sha256": _array_sha256(np.asarray(model.geom_rgba)),
        }
    physics_equal = variants["visible"]["physics_hashes"] == variants["hidden"]["physics_hashes"]
    rgba_different = variants["visible"]["geom_rgba_sha256"] != variants["hidden"]["geom_rgba_sha256"]
    visible_transfer = run_harmonic_transfer_probe(
        "tz",
        5.0,
        isolator_config=config,
        timestep_s=timestep_s,
        transient_cycles=2,
        fit_cycles=4,
        visual=True,
    )
    hidden_transfer = run_harmonic_transfer_probe(
        "tz",
        5.0,
        isolator_config=config,
        timestep_s=timestep_s,
        transient_cycles=2,
        fit_cycles=4,
        visual=False,
    )
    transfer_equal = bool(
        np.allclose(
            visible_transfer["relative_fit"]["transfer_complex"],
            hidden_transfer["relative_fit"]["transfer_complex"],
            rtol=0.0,
            atol=1.0e-12,
        )
        and np.allclose(
            visible_transfer["absolute_fit"]["transfer_complex"],
            hidden_transfer["absolute_fit"]["transfer_complex"],
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    return {
        "variants": variants,
        "physics_equal": physics_equal,
        "rgba_different": rgba_different,
        "transfer_trace_equal": transfer_equal,
        "transfer_trace_reference": {
            "axis": "tz",
            "frequency_hz": 5.0,
            "visible": {
                "relative_transfer_complex": visible_transfer["relative_fit"]["transfer_complex"],
                "absolute_transfer_complex": visible_transfer["absolute_fit"]["transfer_complex"],
            },
            "hidden": {
                "relative_transfer_complex": hidden_transfer["relative_fit"]["transfer_complex"],
                "absolute_transfer_complex": hidden_transfer["absolute_fit"]["transfer_complex"],
            },
        },
        "passed": bool(physics_equal and rgba_different and transfer_equal),
    }


def _phase03_default_xml_equality(isolator_config: Any = None) -> dict[str, Any]:
    """Audit the raw arena asset without calling the Python configurator."""

    mujoco = _lazy_mujoco()
    from robosuite.utils.mjcf_utils import xml_path_completion

    config = _coerce_config(isolator_config)
    parameters = derive_isolator_parameters(config)
    model = mujoco.MjModel.from_xml_path(xml_path_completion("arenas/shakebench_arena.xml"))
    body_id = int(model.body("worktable").id)
    checks = {
        "mass": bool(np.isclose(model.body_mass[body_id], config.mass_kg, rtol=0.0, atol=1.0e-12)),
        "com": bool(np.allclose(model.body_ipos[body_id], (0.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)),
        "inertia": bool(np.allclose(model.body_inertia[body_id], config.inertia_kg_m2, rtol=0.0, atol=1.0e-12)),
    }
    joints = {}
    for index, axis in enumerate(AXES):
        joint_id = int(model.joint(f"isolator_{axis}").id)
        dof_id = int(model.jnt_dofadr[joint_id])
        qpos_id = int(model.jnt_qposadr[joint_id])
        limit = config.limits[index]
        joints[axis] = {
            "stiffness": float(model.jnt_stiffness[joint_id]),
            "expected_stiffness": float(parameters.stiffness[index]),
            "damping": float(model.dof_damping[dof_id]),
            "expected_damping": float(parameters.damping[index]),
            "springref": float(model.qpos_spring[qpos_id]),
            "expected_springref": float(parameters.springref[index]),
            "axis": np.asarray(model.jnt_axis[joint_id], dtype=float).tolist(),
            "expected_axis": np.eye(3)[index % 3].tolist(),
            "range": np.asarray(model.jnt_range[joint_id], dtype=float).tolist(),
            "expected_range": [-limit, limit],
            "limited": bool(model.jnt_limited[joint_id]),
            "type": int(model.jnt_type[joint_id]),
            "expected_type": 2 if index < 3 else 3,
        }
        checks[f"{axis}_stiffness"] = bool(
            np.isclose(model.jnt_stiffness[joint_id], parameters.stiffness[index], rtol=0.0, atol=1.0e-12)
        )
        checks[f"{axis}_damping"] = bool(
            np.isclose(model.dof_damping[dof_id], parameters.damping[index], rtol=0.0, atol=1.0e-12)
        )
        checks[f"{axis}_springref"] = bool(
            np.isclose(model.qpos_spring[qpos_id], parameters.springref[index], rtol=0.0, atol=1.0e-12)
        )
        checks[f"{axis}_axis"] = bool(
            np.allclose(model.jnt_axis[joint_id], np.eye(3)[index % 3], rtol=0.0, atol=1.0e-12)
        )
        checks[f"{axis}_range"] = bool(
            model.jnt_limited[joint_id]
            and np.allclose(model.jnt_range[joint_id], (-limit, limit), rtol=0.0, atol=1.0e-12)
        )
    return {
        "asset_path": "robosuite/models/assets/arenas/shakebench_arena.xml",
        "checks": checks,
        "joints": joints,
        "passed": bool(all(checks.values())),
    }


def build_phase03_transfer_artifact(
    *,
    isolator_config: Any = None,
    harmonic_grid: Optional[Mapping[str, Any]] = None,
    joint_spectrum: Optional[Mapping[str, Any]] = None,
    payload_sensitivity: Optional[Mapping[str, Any]] = None,
    visual_invariance: Optional[Mapping[str, Any]] = None,
    timestep_s: float = 0.0002,
) -> dict[str, Any]:
    """Build the complete Phase 03R machine-readable evidence payload."""

    mujoco = _lazy_mujoco()
    from robosuite.utils.mjcf_utils import xml_path_completion
    from robosuite.utils.shakebench_deck import (
        DeckDriverConfig,
        TRACE_FIELD_CONTRACT,
        TRACE_SCHEMA_ID,
        TRACE_SCHEMA_VERSION,
    )
    from robosuite.utils.shakebench_excitation import excitation_profile_hash
    from robosuite.utils.shakebench_safety import SAFETY_PROFILE_ID, safety_profile_hash
    from robosuite.models.arenas.shakebench_arena import ShakeBenchArena

    config = _coerce_config(isolator_config)
    parameters = derive_isolator_parameters(config)
    timestep = _coerce_scalar("timestep_s", timestep_s, positive=True)
    arena = ShakeBenchArena(isolator_config=config)
    deck_config = DeckDriverConfig(
        physics_timestep_s=timestep,
        eq_solref=(max(2.0 * timestep, 0.0004), 0.5),
        deck_pos_m=tuple(float(value) for value in arena.center_pos),
    )
    if harmonic_grid is None:
        harmonic_grid = run_harmonic_transfer_grid(
            isolator_config=config,
            timestep_s=timestep,
            deck_config=deck_config,
        )
    if joint_spectrum is None:
        joint_spectrum = run_joint_spectrum_probe(
            isolator_config=config,
            timestep_s=timestep,
            deck_config=deck_config,
        )
    if payload_sensitivity is None:
        payload_sensitivity = run_payload_sensitivity_probe(
            isolator_config=config,
            timestep_s=timestep,
            deck_config=deck_config,
        )
    if visual_invariance is None:
        visual_invariance = _phase03_visual_invariance_signature(
            isolator_config=config,
            timestep_s=timestep,
        )
    default_xml = _phase03_default_xml_equality(config)
    xml_path = Path(xml_path_completion("arenas/shakebench_arena.xml"))
    jpeg_path = xml_path.parent.parent / "textures" / "shakebench_phenolic_bench_dark_1k.jpg"
    png_path = xml_path.parent.parent / "textures" / "shakebench_phenolic_bench_dark_1k.png"
    asset_hashes = {
        "arena_xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "phenolic_source_jpeg_sha256": hashlib.sha256(jpeg_path.read_bytes()).hexdigest(),
        "phenolic_runtime_png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
    }
    gates = {
        "xml_python_default_equality": bool(default_xml["passed"]),
        "six_axis_harmonic_grid": bool(harmonic_grid.get("passed")),
        "joint_six_axis_spectrum": bool(joint_spectrum.get("gate_summary", {}).get("passed")),
        "payload_mass_com_sensitivity": bool(payload_sensitivity.get("passed")),
        "visual_physics_transfer_invariance": bool(visual_invariance.get("passed")),
    }
    minimum_spacing = float(joint_spectrum["program"]["minimum_line_spacing_hz"])
    required_fit_window = float(joint_spectrum["program"]["minimum_resolution_duration_s"])
    return {
        "schema_id": PHASE03_TRANSFER_SCHEMA_ID,
        "schema_version": PHASE03_TRANSFER_SCHEMA_VERSION,
        "phase": "03R",
        "axis_order": list(AXES),
        "provenance": {
            "phase": "03R",
            "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
            "trace_schema_id": TRACE_SCHEMA_ID,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_field_contract": dict(TRACE_FIELD_CONTRACT),
            "model_timestep_s": timestep,
            "sample_stride": PHASE03_PROBE_SAMPLE_STRIDE,
            "sample_rate_hz": float(1.0 / timestep / PHASE03_PROBE_SAMPLE_STRIDE),
            "harmonic_transient_discard_cycles": HARMONIC_PROBE_TRANSIENT_CYCLES,
            "harmonic_fit_cycles": HARMONIC_PROBE_FIT_CYCLES,
            "joint_spectrum_seed": int(joint_spectrum["program"]["seed"]),
            "joint_spectrum_t0_s": float(joint_spectrum["program"]["t0_s"]),
            "joint_spectrum_level_scale": float(joint_spectrum["program"]["level_scale"]),
            "excitation_profile_id": joint_spectrum["program"]["profile_id"],
            "authored_spectrum_version": joint_spectrum["program"]["authored_spectrum_version"],
            "excitation_profile_hash": excitation_profile_hash(),
            "safety_profile_id": SAFETY_PROFILE_ID,
            "safety_profile_hash": safety_profile_hash(),
            "minimum_line_spacing_hz": minimum_spacing,
            "required_spectrum_fit_window_s": required_fit_window,
            "reference_point_contract": joint_spectrum["reference"],
            "right_limit_time_contract": "write q(t), integrate to t+dt, write q(t+dt), forward, sample at t+dt",
            "fit_phase_convention": "x=a_s*sin(wt)+a_c*cos(wt)+b; phasor=a_c-i*a_s; H=Y/X; phase=arg(H)",
            "harmonic_linearization": {
                "gravity_vector_m_s2": [0.0, 0.0, 0.0],
                "springref_override": {"isolator_tz": 0.0},
                "reason": (
                    "remove static preload from transfer fit; preload is separately tested "
                    "in payload equilibrium"
                ),
            },
            "base_motion_reference": {
                "deck_origin_m": list(deck_config.deck_pos_m),
                "reference_is_worktable_elastic_center": True,
                "frame": "nominal deck frame",
            },
            "isolator_config": config.to_dict(),
            "derived_isolator_parameters": parameters.to_dict(),
            "deck_driver_config": deck_config.to_dict(),
            "not_frozen": [
                "isolator_fn_hz",
                "isolator_zeta",
                "isolator_k",
                "isolator_c",
                "official_physics_profile",
                "task_success_and_task_sr",
            ],
            "official_freeze": "deferred_to_phase_06",
        },
        "asset_hashes": asset_hashes,
        "default_xml": default_xml,
        "thresholds": dict(PHASE03_TRANSFER_THRESHOLDS),
        "harmonic_grid": dict(harmonic_grid),
        "joint_spectrum": dict(joint_spectrum),
        "payload_sensitivity": dict(payload_sensitivity),
        "visual_invariance": dict(visual_invariance),
        "gates": gates,
        "phase04_handoff": "PASS" if all(gates.values()) else "BLOCKED",
    }


def _phase03_canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def phase03_artifact_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash artifact content while excluding only self-referential metadata."""

    content = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    content.pop("artifact_integrity", None)
    update = content.get("artifact_update")
    if isinstance(update, dict):
        update.pop("new_payload_sha256", None)
    return hashlib.sha256(_phase03_canonical_json(content)).hexdigest()


def finalize_phase03_artifact(
    payload: Mapping[str, Any],
    *,
    previous_file_sha256: Optional[str],
    reason: str,
) -> dict[str, Any]:
    """Attach authenticated update provenance and a deterministic payload hash."""

    if not isinstance(reason, str) or not reason.strip():
        raise IsolatorError("Phase 03 artifact update requires a non-empty reason")
    finalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    finalized.pop("artifact_integrity", None)
    finalized.pop("artifact_update", None)
    finalized["artifact_update"] = {
        "mode": "explicit_update",
        "reason": reason.strip(),
        "previous_file_sha256": previous_file_sha256,
        "new_payload_sha256": None,
    }
    payload_hash = phase03_artifact_payload_hash(finalized)
    finalized["artifact_integrity"] = {
        "hash_algorithm": "sha256",
        "hash_scope": "canonical JSON excluding artifact_integrity and artifact_update.new_payload_sha256",
        "payload_sha256": payload_hash,
    }
    finalized["artifact_update"]["new_payload_sha256"] = payload_hash
    return finalized


def write_phase03_transfer_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Atomically write a Phase 03 artifact after an explicit update reason."""

    artifact_path = Path(path)
    previous_hash = None
    if artifact_path.is_file():
        previous_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    finalized = finalize_phase03_artifact(payload, previous_file_sha256=previous_hash, reason=reason)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(finalized, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(artifact_path.parent),
            prefix=f".{artifact_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, artifact_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return finalized


def verify_phase03_transfer_artifact(path: str | Path) -> dict[str, Any]:
    """Read-only verification for the controlled Phase 03 transfer artifact."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        return {"passed": False, "path": str(artifact_path), "errors": ["artifact does not exist"], "checks": {}}
    file_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "path": str(artifact_path),
            "file_sha256": file_hash,
            "errors": [str(exc)],
            "checks": {},
        }
    if not isinstance(payload, dict):
        return {
            "passed": False,
            "path": str(artifact_path),
            "file_sha256": file_hash,
            "errors": ["artifact root must be an object"],
            "checks": {"schema": False},
        }
    errors = []
    default_xml = payload.get("default_xml") if isinstance(payload.get("default_xml"), Mapping) else {}
    harmonic_grid = payload.get("harmonic_grid") if isinstance(payload.get("harmonic_grid"), Mapping) else {}
    joint_spectrum = payload.get("joint_spectrum") if isinstance(payload.get("joint_spectrum"), Mapping) else {}
    joint_gate_summary = (
        joint_spectrum.get("gate_summary")
        if isinstance(joint_spectrum.get("gate_summary"), Mapping)
        else {}
    )
    payload_sensitivity = (
        payload.get("payload_sensitivity")
        if isinstance(payload.get("payload_sensitivity"), Mapping)
        else {}
    )
    visual_invariance = (
        payload.get("visual_invariance")
        if isinstance(payload.get("visual_invariance"), Mapping)
        else {}
    )
    checks = {
        "schema": payload.get("schema_id") == PHASE03_TRANSFER_SCHEMA_ID
        and payload.get("schema_version") == PHASE03_TRANSFER_SCHEMA_VERSION,
        "integrity": False,
        "update_provenance": False,
        "default_xml": bool(default_xml.get("passed")),
        "harmonic_grid": bool(harmonic_grid.get("passed")) and len(harmonic_grid.get("records", [])) == 18,
        "joint_spectrum": bool(joint_gate_summary.get("passed")) and joint_spectrum.get("line_fit_count") == 64,
        "payload_sensitivity": bool(payload_sensitivity.get("passed"))
        and len(payload_sensitivity.get("records", [])) == 4,
        "visual_invariance": bool(visual_invariance.get("passed")),
        "phase04_handoff": payload.get("phase04_handoff") == "PASS",
    }
    integrity = payload.get("artifact_integrity")
    try:
        computed_hash = phase03_artifact_payload_hash(payload)
    except (TypeError, ValueError) as exc:
        computed_hash = None
        errors.append(f"non-canonical artifact values: {exc}")
    checks["integrity"] = bool(
        isinstance(integrity, dict)
        and integrity.get("hash_algorithm") == "sha256"
        and integrity.get("hash_scope")
        == "canonical JSON excluding artifact_integrity and artifact_update.new_payload_sha256"
        and integrity.get("payload_sha256") == computed_hash
    )
    update = payload.get("artifact_update")
    checks["update_provenance"] = bool(
        isinstance(update, dict)
        and str(update.get("reason", "")).strip()
        and "previous_file_sha256" in update
        and update.get("new_payload_sha256")
        == (integrity.get("payload_sha256") if isinstance(integrity, dict) else None)
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(f"{name} gate failed")
    return {
        "passed": bool(all(checks.values())),
        "path": str(artifact_path),
        "file_sha256": file_hash,
        "payload_sha256": computed_hash,
        "phase04_handoff": payload.get("phase04_handoff"),
        "checks": checks,
        "errors": errors,
    }
