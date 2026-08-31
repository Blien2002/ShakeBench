"""Physics-only probes for the Phase 02 dynamic deck driver.

The script intentionally builds a tiny in-memory MJCF fixture.  It exercises
the real :class:`robosuite.environments.base.MujocoEnv` split-step loop and
the role-based XML processor, but does not import a task, arena, Can, or
isolator.  The JSON output keeps provisional solver values explicit; Phase 06
is responsible for freezing an official profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterable, Optional
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from robosuite.environments.base import MujocoEnv
from robosuite.utils.shakebench_deck import (
    DeckCommand,
    DeckDriver,
    DeckDriverConfig,
    DeckDriverTrace,
    TRACE_FIELD_CONTRACT,
    TRACE_SCHEMA_ID,
    TRACE_SCHEMA_VERSION,
    audit_compiled_deck_model,
    audit_deck_xml,
    command_to_world_state,
    rotation_vector_to_quat,
    quat_to_rotation_vector,
    _quat_inverse_wxyz,
    _quat_multiply_wxyz,
)
from robosuite.utils.shakebench_calibration import calibrate_gamma, level_scale_for_gamma
from robosuite.utils.shakebench_excitation import AXES, build_excitation_program, excitation_profile_hash
from robosuite.utils.shakebench_safety import check_safety
from robosuite.utils import transform_utils as T


WORKTABLE_REFERENCE_PROXY_BODY_NAME = "worktable_reference_proxy"
WORKTABLE_REFERENCE_PROXY_GEOM_NAME = "worktable_reference_proxy_geom"
WORKTABLE_REFERENCE_MASS_KG = 32.0
WORKTABLE_REFERENCE_INERTIA_KG_M2 = (0.9696, 1.1363, 2.0867)
LOAD_CASE_EMPTY = "empty"
LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY = "panda_plus_worktable_reference_proxy"
# The old name remains an input boundary alias.  Artifact records use the
# canonical Panda-inclusive name so a table-only lower-bound cannot be
# mistaken for the Phase 02R4 load envelope.
LEGACY_LOAD_CASE_WORKTABLE_REFERENCE_PROXY = "worktable_reference_proxy"
LOAD_CASE_WORKTABLE_REFERENCE_PROXY = LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY
CANONICAL_LOAD_CASES = (LOAD_CASE_EMPTY, LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY)
PANDA_BASE_ROLE = "panda_base"
PANDA_BASE_BODY_NAME = "robot0_base"
ARTIFACT_SCHEMA_ID = "shakebench.phase02r.conformance_matrix"
ARTIFACT_SCHEMA_VERSION = 3
CONTROLLED_ARTIFACT_NAME = "shakebench_phase_02r_probe.json"
REQUIRED_COVERAGE_FAMILIES = {
    "zero",
    "single_axis",
    "authored_spectrum",
    "target_gamma",
    "load",
    "dt_convergence",
    "solver_sensitivity",
}


@dataclass(frozen=True)
class Phase02R4Thresholds:
    """Pre-registered physics-only gates shared by generation and verification."""

    amplitude_relative_error: float = 0.01
    absolute_phase_error_deg: float = 1.0
    gamma_relative_error: float = 0.01
    max_condition_number: float = 10000.0
    max_relative_fit_residual: float = 1e-3
    positive_eq_solref_time_constant_min_factor_dt: float = 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "amplitude_relative_error": self.amplitude_relative_error,
            "absolute_phase_error_deg": self.absolute_phase_error_deg,
            "canonical_gamma_relative_error": self.gamma_relative_error,
            "max_condition_number": self.max_condition_number,
            "max_relative_fit_residual": self.max_relative_fit_residual,
            "positive_eq_solref_time_constant_min_factor_dt": self.positive_eq_solref_time_constant_min_factor_dt,
        }


DEFAULT_THRESHOLDS = Phase02R4Thresholds()


class _ProbeXMLModel:
    """Small model holder implementing the methods used by ``MujocoEnv``."""

    def __init__(self, xml_string: str):
        self._xml_string = xml_string
        self.mujoco_objects = []

    def get_xml(self) -> str:
        return self._xml_string

    def generate_id_mappings(self, sim: Any) -> None:
        return None


class DeckProbeEnv(MujocoEnv):
    """Minimal no-actuator environment used by the command-line probe."""

    _unregistered_env = True

    def __init__(
        self,
        xml_string: str,
        driver: DeckDriver,
        *,
        model_timestep: float,
        control_freq: float = 20.0,
        horizon: int = 1000,
        lite_physics: bool = True,
        initial_joint_state: Optional[Mapping[str, float]] = None,
    ):
        self._probe_xml_string = xml_string
        self._probe_driver = driver
        self._initial_joint_state = dict(initial_joint_state or {})
        self._applied_initial_joint_state = {}
        super().__init__(
            has_renderer=False,
            has_offscreen_renderer=False,
            renderer="mujoco",
            load_model_on_init=False,
            control_freq=control_freq,
            horizon=horizon,
            lite_physics=lite_physics,
            model_timestep=model_timestep,
        )
        driver.install(self)

    def _load_model(self) -> None:
        self.model = _ProbeXMLModel(self._probe_xml_string)

    def _reset_internal(self) -> None:
        super()._reset_internal()
        if not self._initial_joint_state:
            return
        raw_model = self.sim.model._model
        raw_data = self.sim.data._data
        self._applied_initial_joint_state = {}
        for joint_name, value in self._initial_joint_state.items():
            joint_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Panda fixture initial joint {joint_name!r} is missing from compiled model")
            qpos_address = int(raw_model.jnt_qposadr[joint_id])
            qvel_address = int(raw_model.jnt_dofadr[joint_id])
            raw_data.qpos[qpos_address] = float(value)
            raw_data.qvel[qvel_address] = 0.0
            self._applied_initial_joint_state[joint_name] = float(raw_data.qpos[qpos_address])

    def reward(self, action=None) -> float:
        return 0.0

    def _pre_action(self, action, policy_step=False) -> None:
        # The Panda XML carries its real actuators, but this fixture is
        # deliberately no-task and no-controller.  Holding all controls at
        # zero lets MuJoCo compile and integrate the actual subtree without
        # starting a robosuite controller.
        self.sim.data.ctrl[:] = 0.0

    def _check_success(self) -> bool:
        return False


def _normalize_load_case(load_case: str, representative_load: Optional[bool] = None) -> str:
    """Normalize the canonical load case and its legacy boolean boundary alias."""

    if load_case == LEGACY_LOAD_CASE_WORKTABLE_REFERENCE_PROXY:
        load_case = LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY
    if representative_load is not None:
        legacy_case = LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY if representative_load else LOAD_CASE_EMPTY
        if load_case not in (LOAD_CASE_EMPTY, LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY):
            raise ValueError("load_case and representative_load specify different load cases")
        load_case = legacy_case
    if load_case not in CANONICAL_LOAD_CASES:
        raise ValueError(
            "load_case must be 'empty' or 'panda_plus_worktable_reference_proxy'"
        )
    return load_case


@dataclass(frozen=True)
class _ProbeFixture:
    """Compiled-input description for one no-task probe load case."""

    xml_string: str
    role_handles: Mapping[str, str]
    initial_joint_state: Mapping[str, float]
    panda_body_names: tuple[str, ...]
    panda_joint_names: tuple[str, ...]
    panda_source: Optional[str]


def _build_probe_fixture(*, model_timestep: float, load_case: str) -> _ProbeFixture:
    """Build the self-contained driver fixture from the in-repo Panda model."""

    load_case = _normalize_load_case(load_case)
    root = ET.Element("mujoco", {"model": "shakebench_phase02_probe"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": format(float(model_timestep), ".17g"),
            "gravity": "0 0 0",
            "iterations": "100",
            "tolerance": "1e-12",
        },
    )
    sections = {tag: ET.SubElement(root, tag) for tag in ("asset", "actuator", "sensor", "tendon", "equality", "contact")}
    worldbody = ET.SubElement(root, "worldbody")

    if load_case == LOAD_CASE_EMPTY:
        return _ProbeFixture(
            xml_string=ET.tostring(root, encoding="unicode"),
            role_handles={},
            initial_joint_state={},
            panda_body_names=(),
            panda_joint_names=(),
            panda_source=None,
        )

    # This is deliberately the actual shipped robosuite Panda XML model.  We
    # compose its XML sections into the no-task fixture instead of replacing
    # its subtree with a guessed scalar mass or an arm-only inertial block.
    from robosuite.models.robots.manipulators.panda_robot import Panda

    panda = Panda(idn=0)
    panda_root = ET.fromstring(panda.get_xml())
    for tag, target in sections.items():
        source = panda_root.find(tag)
        if source is not None:
            for child in list(source):
                target.append(child)

    proxy = ET.SubElement(worldbody, "body", {"name": WORKTABLE_REFERENCE_PROXY_BODY_NAME, "pos": "0 0 0"})
    ET.SubElement(
        proxy,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": format(WORKTABLE_REFERENCE_MASS_KG, ".17g"),
            "diaginertia": " ".join(format(value, ".17g") for value in WORKTABLE_REFERENCE_INERTIA_KG_M2),
        },
    )
    ET.SubElement(
        proxy,
        "geom",
        {
            "name": WORKTABLE_REFERENCE_PROXY_GEOM_NAME,
            "type": "box",
            "size": "0.30 0.28 0.03",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "rgba": "0.2 0.2 0.2 1",
        },
    )
    panda_worldbody = panda_root.find("worldbody")
    if panda_worldbody is None:
        raise RuntimeError("in-repo Panda model is missing its worldbody")
    for body in list(panda_worldbody):
        worldbody.append(body)

    return _ProbeFixture(
        xml_string=ET.tostring(root, encoding="unicode"),
        role_handles={PANDA_BASE_ROLE: panda.root_body, "worktable_reference": WORKTABLE_REFERENCE_PROXY_BODY_NAME},
        initial_joint_state={name: float(value) for name, value in zip(panda.joints, panda.init_qpos)},
        panda_body_names=tuple(panda.bodies),
        panda_joint_names=tuple(panda.joints),
        panda_source="robosuite.models.assets.robots.panda.robot.xml via Panda(idn=0)",
    )


def build_probe_xml(
    *,
    model_timestep: float,
    load_case: str = LOAD_CASE_EMPTY,
    representative_load: Optional[bool] = None,
) -> str:
    """Return a self-contained empty or Panda-inclusive driver fixture.

    Args:
        model_timestep: Model-local MuJoCo timestep in seconds.
        load_case: Canonical fixture load case.
        representative_load: Legacy boolean alias for the Panda-inclusive
            proxy case. New callers should use ``load_case``.
    """

    load_case = _normalize_load_case(load_case, representative_load)
    return _build_probe_fixture(model_timestep=model_timestep, load_case=load_case).xml_string


def _sine_command(axis_index: int, frequency_hz: float, amplitude: float) -> Any:
    omega = 2.0 * math.pi * float(frequency_hz)

    def trajectory(time_s: float) -> DeckCommand:
        q = np.zeros(6)
        qdot = np.zeros(6)
        qdd = np.zeros(6)
        q[axis_index] = amplitude * math.sin(omega * time_s)
        qdot[axis_index] = amplitude * omega * math.cos(omega * time_s)
        qdd[axis_index] = -amplitude * omega * omega * math.sin(omega * time_s)
        return DeckCommand(q, qdot, qdd)

    return trajectory


def _program_command(program: Any) -> Any:
    return program


def _program_hash(program: Any) -> str:
    """Hash the complete deterministic authored program, including seed/t0."""

    encoded = json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rotation_vector_from_pose(pose: np.ndarray, base_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """Return pose rotations relative to a world-frame base orientation."""

    base_quat = np.asarray(tuple(base_quat_wxyz), dtype=float)
    base_inverse = _quat_inverse_wxyz(base_quat)
    return np.asarray(
        [quat_to_rotation_vector(_quat_multiply_wxyz(base_inverse, quaternion)) for quaternion in pose[:, 3:7]],
        dtype=float,
    )


def _actual_coordinates(
    trace: DeckDriverTrace,
    *,
    base_position_m: Iterable[float] = (0.0, 0.0, 0.0),
    base_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return deck pose coordinates in the authored nominal frame."""

    coordinates = np.array(trace.actual_pose[:, :3], copy=True) - np.asarray(tuple(base_position_m), dtype=float)
    if trace.actual_pose.shape[0]:
        coordinates = np.concatenate(
            (coordinates, _rotation_vector_from_pose(trace.actual_pose, base_quat_wxyz)),
            axis=1,
        )
    else:
        coordinates = np.empty((0, 6), dtype=float)
    return coordinates


def _fit_sine(time_s: np.ndarray, values: np.ndarray, frequency_hz: float) -> tuple[float, float]:
    amplitude, phase, _ = _fit_sine_with_residual(time_s, values, frequency_hz)
    return amplitude, phase


def _fit_sine_with_residual(
    time_s: np.ndarray, values: np.ndarray, frequency_hz: float
) -> tuple[float, float, float]:
    omega = 2.0 * math.pi * float(frequency_hz)
    design = np.column_stack((np.sin(omega * time_s), np.cos(omega * time_s), np.ones(time_s.size)))
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design.dot(coefficient)
    return (
        float(np.hypot(coefficient[0], coefficient[1])),
        float(math.atan2(coefficient[1], coefficient[0])),
        float(np.sqrt(np.mean(residual**2))),
    )


def _fit_sinusoidal_basis(
    time_s: np.ndarray, values: np.ndarray, frequencies_hz: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit a joint sinusoidal basis to prevent nearby-line leakage."""

    frequencies = np.asarray(tuple(frequencies_hz), dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must contain at least one frequency")
    columns = []
    for frequency_hz in frequencies:
        omega = 2.0 * math.pi * float(frequency_hz)
        columns.extend((np.sin(omega * time_s), np.cos(omega * time_s)))
    design = np.column_stack((*columns, np.ones(time_s.size)))
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design.dot(coefficient)
    sine = coefficient[0:-1:2]
    cosine = coefficient[1:-1:2]
    return sine, cosine, float(np.sqrt(np.mean(residual**2))), float(np.linalg.cond(design))


def _phase_difference(first: float, second: float) -> float:
    return float((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def sine_conformance(
    trace: DeckDriverTrace,
    *,
    axis_index: int,
    frequency_hz: float,
    amplitude: float,
    discard_s: float,
    base_position_m: Iterable[float] = (0.0, 0.0, 0.0),
    base_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Fit actual and same-sample authored pose coordinates.

    The fit evaluates the authored command at each post-step sample time. It
    never shifts the actual trace to compensate for a model step.
    """

    if trace.sample_timestamps_s.size == 0:
        raise ValueError("trace is empty")
    mask = trace.sample_timestamps_s >= float(discard_s)
    time_s = trace.sample_timestamps_s[mask]
    actual = _actual_coordinates(
        trace,
        base_position_m=base_position_m,
        base_quat_wxyz=base_quat_wxyz,
    )[mask, axis_index]
    command = amplitude * np.sin(2.0 * math.pi * frequency_hz * time_s)
    actual_amplitude, actual_phase, actual_residual = _fit_sine_with_residual(time_s, actual, frequency_hz)
    command_amplitude, command_phase, command_residual = _fit_sine_with_residual(time_s, command, frequency_hz)
    return {
        "command_amplitude": command_amplitude,
        "actual_amplitude": actual_amplitude,
        "amplitude_ratio": actual_amplitude / command_amplitude if command_amplitude else 0.0,
        "amplitude_relative_error": abs(actual_amplitude - command_amplitude) / command_amplitude
        if command_amplitude
        else 0.0,
        "command_phase_rad": command_phase,
        "actual_phase_rad": actual_phase,
        "phase_error_deg": math.degrees(_phase_difference(actual_phase, command_phase)),
        "command_fit_residual_rms": command_residual,
        "actual_fit_residual_rms": actual_residual,
        "discard_s": float(discard_s),
        "sample_rate_hz": float(1.0 / np.median(np.diff(time_s))) if time_s.size > 1 else 0.0,
        "phase_convention": "A*sin(omega*t + phase); atan2(cos_coefficient, sin_coefficient)",
    }


def gamma_conformance(
    trace: DeckDriverTrace,
    *,
    axis_index: int,
    frequency_hz: float,
    displacement_amplitude: float,
    gravity_m_s2: float = 9.81,
    discard_s: float,
) -> dict[str, Any]:
    """Compare translation-origin acceleration as a lower-bound diagnostic.

    This helper intentionally excludes rotational ``alpha x r`` and
    centripetal terms. Use :func:`canonical_gamma_conformance` for the
    workpiece-point Gamma gate.
    """

    if trace.sample_timestamps_s.size == 0:
        raise ValueError("trace is empty")
    mask = trace.sample_timestamps_s >= float(discard_s)
    time_s = trace.sample_timestamps_s[mask]
    omega = 2.0 * math.pi * float(frequency_hz)
    command = -displacement_amplitude * omega * omega * np.sin(omega * time_s)
    actual = trace.actual_acceleration[mask, axis_index]
    actual_acceleration, actual_phase, actual_residual = _fit_sine_with_residual(time_s, actual, frequency_hz)
    command_acceleration, command_phase, command_residual = _fit_sine_with_residual(time_s, command, frequency_hz)
    command_gamma = command_acceleration / float(gravity_m_s2)
    actual_gamma = actual_acceleration / float(gravity_m_s2)
    return {
        "command_gamma": command_gamma,
        "actual_gamma": actual_gamma,
        "Gamma_commanded": command_gamma,
        "Gamma_deck_actual": actual_gamma,
        "gamma_relative_error": abs(actual_gamma - command_gamma) / command_gamma if command_gamma else 0.0,
        "relative_error_abs": abs(actual_gamma / command_gamma - 1.0) if command_gamma else 0.0,
        "phase_error_deg": math.degrees(_phase_difference(actual_phase, command_phase)),
        "command_phase_rad": command_phase,
        "actual_phase_rad": actual_phase,
        "command_fit_residual_rms": command_residual,
        "actual_fit_residual_rms": actual_residual,
        "scope": "driver_lower_bound_translation_origin; not canonical workpiece-point Gamma",
        "discard_s": float(discard_s),
        "sample_rate_hz": float(1.0 / np.median(np.diff(time_s))) if time_s.size > 1 else 0.0,
    }


def _world_point_acceleration(
    pose: np.ndarray,
    twist: np.ndarray,
    acceleration: np.ndarray,
    point_offset_m: Iterable[float],
    *,
    include_centripetal: bool = False,
) -> np.ndarray:
    """Compute a rigid point acceleration from a world spatial state."""

    point = np.asarray(tuple(point_offset_m), dtype=float)
    rotation = T.quat2mat(np.asarray(pose[3:7], dtype=float)[[1, 2, 3, 0]])
    point_world = rotation.dot(point)
    result = np.asarray(acceleration[:3], dtype=float) + np.cross(acceleration[3:], point_world)
    if include_centripetal:
        result = result + np.cross(twist[3:], np.cross(twist[3:], point_world))
    return result


def _program_world_motion(program: Any, time_s: np.ndarray, config: DeckDriverConfig):
    """Evaluate authored motion and map every sample into world quantities."""

    motion = program.evaluate(time_s)
    pose = np.empty((time_s.size, 7), dtype=float)
    twist = np.empty((time_s.size, 6), dtype=float)
    acceleration = np.empty((time_s.size, 6), dtype=float)
    for index in range(time_s.size):
        command = DeckCommand(motion.q[index], motion.qdot[index], motion.qdd[index])
        world_pose, world_quaternion, world_twist, world_acceleration = command_to_world_state(command, config)
        pose[index] = np.concatenate((world_pose, world_quaternion))
        twist[index] = world_twist
        acceleration[index] = world_acceleration
    return pose, twist, acceleration


def canonical_gamma_conformance(
    trace: DeckDriverTrace,
    program: Any,
    *,
    config: DeckDriverConfig,
    point_offset_m: Iterable[float] = (0.65, 0.0, 0.0),
    support_normal: Iterable[float] = (0.0, 0.0, 1.0),
    gravity_m_s2: float = 9.81,
    discard_s: float = 0.0,
    include_centripetal: bool = False,
) -> dict[str, Any]:
    """Compare commanded and realized Gamma at one physical workpiece point.

    Both quantities are evaluated at the post-step sample timestamps. The
    actual deck state is never shifted to align it with an earlier command.
    Spatial acceleration uses ``a + alpha x r`` and, explicitly, the
    centripetal ``omega x (omega x r)`` term when explicitly enabled. The
    provisional Phase 02R Gamma definition leaves that optional term off so
    the Phase 01 unit-replay scale and the command definition are identical;
    the choice is recorded in every result.
    """

    if trace.sample_timestamps_s.size == 0:
        raise ValueError("trace is empty")
    normal = np.asarray(tuple(support_normal), dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= 0.0:
        raise ValueError("support_normal must be a non-zero finite 3-vector")
    normal /= np.linalg.norm(normal)
    point = np.asarray(tuple(point_offset_m), dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_offset_m must be a finite 3-vector")
    mask = trace.sample_timestamps_s >= float(discard_s)
    time_s = trace.sample_timestamps_s[mask]
    command_pose, command_twist, command_acceleration = _program_world_motion(program, time_s, config)
    command_point_acceleration = np.asarray(
        [
            _world_point_acceleration(
                pose,
                twist,
                acceleration,
                point,
                include_centripetal=include_centripetal,
            )
            for pose, twist, acceleration in zip(command_pose, command_twist, command_acceleration)
        ],
        dtype=float,
    )
    actual_point_acceleration = np.asarray(
        [
            _world_point_acceleration(
                pose,
                twist,
                acceleration,
                point,
                include_centripetal=include_centripetal,
            )
            for pose, twist, acceleration in zip(
                trace.actual_pose[mask], trace.actual_twist[mask], trace.actual_acceleration[mask]
            )
        ],
        dtype=float,
    )
    command_effective = command_point_acceleration.dot(normal)
    actual_effective = actual_point_acceleration.dot(normal)
    command_peak_index = int(np.argmax(np.abs(command_effective)))
    actual_peak_index = int(np.argmax(np.abs(actual_effective)))
    command_gamma = float(np.max(np.abs(command_effective)) / float(gravity_m_s2))
    actual_gamma = float(np.max(np.abs(actual_effective)) / float(gravity_m_s2))
    return {
        "Gamma_commanded": command_gamma,
        "Gamma_deck_actual": actual_gamma,
        "gamma_commanded": command_gamma,
        "gamma_deck_actual": actual_gamma,
        "relative_error_abs": abs(actual_gamma / command_gamma - 1.0) if command_gamma else 0.0,
        "command_peak_time_s": float(time_s[command_peak_index]),
        "actual_peak_time_s": float(time_s[actual_peak_index]),
        "command_peak_acceleration_m_s2": float(np.max(np.abs(command_effective))),
        "actual_peak_acceleration_m_s2": float(np.max(np.abs(actual_effective))),
        "gravity_m_s2": float(gravity_m_s2),
        "point_offset_m": point.tolist(),
        "support_normal_world": normal.tolist(),
        "include_alpha_cross_r": True,
        "include_centripetal": bool(include_centripetal),
        "frame": "world",
        "origin": "deck body origin",
        "sample_count": int(time_s.size),
        "sample_time_grid_s": time_s.tolist(),
        "time_convention": "command evaluated and actual sampled at the same post-step sample timestamp",
    }


def spectrum_conformance(
    trace: DeckDriverTrace,
    program: Any,
    *,
    discard_s: float,
    base_position_m: Iterable[float] = (0.0, 0.0, 0.0),
    base_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
    max_fit_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Report per-line command/actual amplitude and phase for a six-axis program.

    The RMS comparison is intentionally performed on the same physical sample
    times.  It does not hide application latency by shifting the actual trace
    by one model step; command and application timestamps remain available for
    an independent timing audit.
    """

    if trace.sample_timestamps_s.size == 0:
        raise ValueError("trace is empty")
    mask = trace.sample_timestamps_s >= float(discard_s)
    selected_indices = np.flatnonzero(mask)
    if selected_indices.size == 0:
        raise ValueError("discard_s leaves no sample-time data for the spectrum fit")
    if max_fit_samples is not None:
        if isinstance(max_fit_samples, (bool, np.bool_)) or int(max_fit_samples) < 3:
            raise ValueError("max_fit_samples must be an integer >= 3")
        max_fit_samples = int(max_fit_samples)
        if selected_indices.size > max_fit_samples:
            selected_indices = np.linspace(
                selected_indices[0], selected_indices[-1], max_fit_samples, dtype=np.int64
            )
    times = trace.sample_timestamps_s[selected_indices]
    minimum_spacing = minimum_line_spacing_hz(program)
    beat_period = 1.0 / minimum_spacing
    required_fit_window = 2.0 * beat_period
    fit_window_s = float(times[-1] - times[0]) if times.size > 1 else 0.0
    actual = _actual_coordinates(
        trace,
        base_position_m=base_position_m,
        base_quat_wxyz=base_quat_wxyz,
    )[selected_indices]
    motion = program.evaluate(times)
    command = np.asarray(motion.q, dtype=float)
    axis_records = {}
    for axis_index, axis in enumerate(AXES):
        command_rms = float(np.sqrt(np.mean(command[:, axis_index] ** 2)))
        actual_rms = float(np.sqrt(np.mean(actual[:, axis_index] ** 2)))
        line_indices = np.flatnonzero(np.asarray(program.line_mask[axis_index], dtype=bool))
        frequencies = np.asarray(program.line_frequency_hz[axis_index, line_indices], dtype=float)
        command_sine, command_cosine, command_residual, command_condition = _fit_sinusoidal_basis(
            times, command[:, axis_index], frequencies
        )
        actual_sine, actual_cosine, actual_residual, actual_condition = _fit_sinusoidal_basis(
            times, actual[:, axis_index], frequencies
        )
        line_fits = []
        for line_offset, line_index in enumerate(line_indices):
            command_amplitude = float(np.hypot(command_sine[line_offset], command_cosine[line_offset]))
            actual_amplitude = float(np.hypot(actual_sine[line_offset], actual_cosine[line_offset]))
            command_phase = float(math.atan2(command_cosine[line_offset], command_sine[line_offset]))
            actual_phase = float(math.atan2(actual_cosine[line_offset], actual_sine[line_offset]))
            line_fits.append(
                {
                    "line_index": int(line_index),
                    "frequency_hz": float(frequencies[line_offset]),
                    "command_amplitude": command_amplitude,
                    "actual_amplitude": actual_amplitude,
                    "amplitude_ratio": actual_amplitude / command_amplitude if command_amplitude else 0.0,
                    "amplitude_relative_error": (
                        abs(actual_amplitude - command_amplitude) / command_amplitude if command_amplitude else 0.0
                    ),
                    "command_phase_rad": command_phase,
                    "actual_phase_rad": actual_phase,
                    "phase_error_deg": math.degrees(_phase_difference(actual_phase, command_phase)),
                    "fit_residual_rms": actual_residual,
                    "command_fit_residual_rms": command_residual,
                    "condition_number": actual_condition,
                    "phase_convention": "A*sin(omega*t + phase); atan2(cos_coefficient, sin_coefficient)",
                }
            )
        axis_records[axis] = {
            "command_rms": command_rms,
            "actual_rms": actual_rms,
            "rms_relative_error": abs(actual_rms - command_rms) / command_rms if command_rms else 0.0,
            "line_fits": line_fits,
            "active_line_count": len(line_fits),
            "command_fit_residual_rms": command_residual,
            "actual_fit_residual_rms": actual_residual,
            "command_basis_condition_number": command_condition,
            "actual_basis_condition_number": actual_condition,
            "min_active_line_spacing_hz": (
                float(np.min(np.diff(np.sort(frequencies)))) if frequencies.size > 1 else None
            ),
        }
    return {
        "axes": axis_records,
        "fit_quantity": "deck pose coordinates in nominal frame",
        "discard_s": float(discard_s),
        "sample_rate_hz": float(1.0 / np.median(np.diff(times))) if times.size > 1 else 0.0,
        "native_sample_rate_hz": (
            float(1.0 / np.median(np.diff(trace.sample_timestamps_s)))
            if trace.sample_timestamps_s.size > 1
            else 0.0
        ),
        "fit_time_grid_s": times.tolist(),
        "fit_sample_count": int(times.size),
        "fit_sample_stride": (
            int(np.median(np.diff(selected_indices))) if selected_indices.size > 1 else 1
        ),
        "ramp_duration_s": float(program.config.ramp_duration_s),
        "minimum_line_spacing_hz": minimum_spacing,
        "beat_period_s": beat_period,
        "required_fit_window_s": required_fit_window,
        "fit_window_s": fit_window_s,
        "fit_window_meets_resolution": fit_window_s >= required_fit_window,
        "conditioning_rule": {
            "max_condition_number": 10000.0,
            "max_relative_fit_residual": 1e-3,
        },
        "phase_convention": "A*sin(omega*t + phase); joint sinusoidal basis plus intercept",
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
        "max_solver_iterations": int(np.max(trace.solver_iterations)) if trace.solver_iterations.size else 0,
        "warning_count": int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0,
    }


def minimum_line_spacing_hz(program: Any) -> float:
    """Return the smallest within-axis spacing among active authored lines."""

    spacings = []
    for axis_index in range(len(AXES)):
        frequencies = np.sort(np.asarray(program.line_frequency_hz[axis_index][program.line_mask[axis_index]]))
        if frequencies.size > 1:
            spacings.extend(np.diff(frequencies).tolist())
    if not spacings:
        raise ValueError("program must contain at least two active lines")
    return float(np.min(spacings))


def synthetic_spectrum_estimator_validation(
    program: Any,
    *,
    sample_dt_s: float = 0.0005,
    amplitude_scale: float = 1.002,
    phase_perturbation_deg: float = 0.25,
) -> dict[str, Any]:
    """Validate the line estimator on a known perturbed multi-line trace.

    The synthetic trace uses the same sample grid and the same post-ramp fit
    contract as the real driver. It is evaluated before any MuJoCo run so the
    conditioning and residual thresholds cannot be selected from task or
    solver outcomes.
    """

    if not np.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
        raise ValueError("amplitude_scale must be finite and positive")
    if not np.isfinite(phase_perturbation_deg):
        raise ValueError("phase_perturbation_deg must be finite")
    minimum_spacing = minimum_line_spacing_hz(program)
    beat_period = 1.0 / minimum_spacing
    required_window = 2.0 * beat_period
    discard_s = max(0.75, float(program.config.ramp_duration_s) + 0.25)
    duration_s = discard_s + required_window + sample_dt_s
    sample_count = int(math.ceil(duration_s / sample_dt_s))
    time_s = np.arange(1, sample_count + 1, dtype=float) * sample_dt_s
    motion = program.evaluate(time_s)
    phase_perturbation_rad = math.radians(phase_perturbation_deg)
    actual_coordinates = np.zeros_like(motion.q)
    for axis_index in range(len(AXES)):
        active_lines = np.flatnonzero(program.line_mask[axis_index])
        for line_index in active_lines:
            omega = program.line_omega_rad_s[axis_index, line_index]
            amplitude = program.line_accel_amplitude[axis_index, line_index] / omega**2
            phase = program.line_phase_at_episode_zero[axis_index, line_index] + phase_perturbation_rad
            actual_coordinates[:, axis_index] += -amplitude_scale * amplitude * np.sin(omega * time_s + phase)
    actual_pose = np.empty((time_s.size, 7), dtype=float)
    actual_pose[:, :3] = actual_coordinates[:, :3]
    for index, rotation_vector in enumerate(actual_coordinates[:, 3:]):
        actual_pose[index, 3:] = rotation_vector_to_quat(rotation_vector)
    zeros = np.zeros((time_s.size, 6), dtype=float)
    trace = DeckDriverTrace(
        integration_target_time_s=time_s.copy(),
        sample_target_time_s=time_s.copy(),
        integration_application_time_s=time_s.copy(),
        sample_application_time_s=time_s.copy(),
        sample_time_s=time_s.copy(),
        command_pose=actual_pose.copy(),
        actual_pose=actual_pose,
        command_twist=zeros.copy(),
        actual_twist=zeros.copy(),
        command_acceleration=zeros.copy(),
        actual_acceleration=zeros.copy(),
        sample_target_pose=actual_pose.copy(),
        sample_target_twist=zeros.copy(),
        sample_target_acceleration=zeros.copy(),
        deck_tracking_pose_error=zeros.copy(),
        weld_constraint_residual_raw=zeros.copy(),
        weld_constraint_force_raw=zeros.copy(),
        solver_iterations=np.zeros(time_s.size, dtype=np.int64),
        solver_niter=np.zeros((time_s.size, 1), dtype=np.int64),
        warning_number_delta=np.zeros((time_s.size, 1), dtype=np.int64),
        warning_lastinfo=np.zeros((time_s.size, 1), dtype=np.int64),
    )
    fit = spectrum_conformance(trace, program, discard_s=discard_s, max_fit_samples=50000)
    line_results = [
        line
        for axis in fit["axes"].values()
        for line in axis["line_fits"]
    ]
    max_amplitude_error = max(
        abs(line["amplitude_ratio"] - amplitude_scale) for line in line_results
    )
    max_phase_error = max(
        abs(abs(line["phase_error_deg"]) - abs(phase_perturbation_deg)) for line in line_results
    )
    max_relative_residual = max(
        axis["actual_fit_residual_rms"] / axis["actual_rms"]
        if axis["actual_rms"]
        else 0.0
        for axis in fit["axes"].values()
    )
    conditioning_rule = {
        "max_condition_number": 10000.0,
        "max_relative_fit_residual": 1e-3,
        "phase_recovery_tolerance_deg": 0.01,
        "amplitude_recovery_tolerance": 1e-4,
    }
    return {
        "sample_dt_s": float(sample_dt_s),
        "discard_s": float(discard_s),
        "fit_window_s": float(time_s[-1] - time_s[time_s >= discard_s][0]),
        "minimum_line_spacing_hz": minimum_spacing,
        "beat_period_s": beat_period,
        "required_fit_window_s": required_window,
        "line_count": len(line_results),
        "known_amplitude_scale": float(amplitude_scale),
        "known_phase_perturbation_deg": float(phase_perturbation_deg),
        "max_amplitude_recovery_error": max_amplitude_error,
        "max_phase_recovery_error_deg": max_phase_error,
        "max_relative_fit_residual": max_relative_residual,
        "max_condition_number": max(line["condition_number"] for line in line_results),
        "conditioning_rule": conditioning_rule,
        "resolution_passed": fit["fit_window_meets_resolution"],
        "passed": (
            fit["fit_window_meets_resolution"]
            and max_amplitude_error <= conditioning_rule["amplitude_recovery_tolerance"]
            and max_phase_error <= conditioning_rule["phase_recovery_tolerance_deg"]
            and max_relative_residual <= conditioning_rule["max_relative_fit_residual"]
            and max(line["condition_number"] for line in line_results)
            <= conditioning_rule["max_condition_number"]
        ),
        "line_fits": line_results,
        "phase_convention": fit["phase_convention"],
    }


def _spectrum_line_entries(conformance: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Flatten spectrum line records while retaining the axis RMS context."""

    entries = []
    axes = conformance.get("axes", {}) if isinstance(conformance, Mapping) else {}
    if isinstance(axes, Mapping):
        for axis, axis_record in axes.items():
            if not isinstance(axis_record, Mapping):
                continue
            for line in axis_record.get("line_fits", ()):
                if isinstance(line, Mapping):
                    entries.append((str(axis), axis_record, line))
    if not entries and isinstance(conformance, Mapping):
        for line in conformance.get("line_fits", ()):
            if isinstance(line, Mapping):
                entries.append(("unknown", conformance, line))
    return entries


def spectrum_gate_reasons(
    conformance: Mapping[str, Any],
    *,
    thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS,
    expected_line_count: int = 64,
) -> list[str]:
    """Return deterministic rejection reasons for a per-line spectrum fit."""

    reasons: list[str] = []
    if conformance.get("fit_window_meets_resolution") is not True:
        reasons.append("fit_window_resolution_failed")
    entries = _spectrum_line_entries(conformance)
    if len(entries) != expected_line_count:
        reasons.append(f"expected_{expected_line_count}_lines_found_{len(entries)}")
    for axis, axis_record, line in entries:
        label = f"{axis}[{line.get('line_index', '?')}]"
        amplitude_error = line.get("amplitude_relative_error")
        phase_error = line.get("phase_error_deg")
        condition = line.get("condition_number")
        residual = line.get("fit_residual_rms")
        actual_rms = axis_record.get("actual_rms", 0.0)
        if not isinstance(amplitude_error, (int, float)) or not np.isfinite(amplitude_error):
            reasons.append(f"{label}:amplitude_metric_missing")
        elif float(amplitude_error) > thresholds.amplitude_relative_error:
            reasons.append(f"{label}:amplitude_relative_error_exceeds_threshold")
        if not isinstance(phase_error, (int, float)) or not np.isfinite(phase_error):
            reasons.append(f"{label}:phase_metric_missing")
        elif abs(float(phase_error)) > thresholds.absolute_phase_error_deg:
            reasons.append(f"{label}:absolute_phase_error_exceeds_threshold")
        if not isinstance(condition, (int, float)) or not np.isfinite(condition):
            reasons.append(f"{label}:condition_metric_missing")
        elif float(condition) > thresholds.max_condition_number:
            reasons.append(f"{label}:condition_number_exceeds_threshold")
        if not isinstance(residual, (int, float)) or not np.isfinite(residual):
            reasons.append(f"{label}:fit_residual_metric_missing")
        elif "actual_rms" not in axis_record:
            if float(residual) > thresholds.max_relative_fit_residual:
                reasons.append(f"{label}:fit_residual_exceeds_threshold")
        elif isinstance(actual_rms, (int, float)) and float(actual_rms) > 0.0:
            if float(residual) / float(actual_rms) > thresholds.max_relative_fit_residual:
                reasons.append(f"{label}:relative_fit_residual_exceeds_threshold")
        elif float(residual) != 0.0:
            reasons.append(f"{label}:nonzero_residual_for_zero_signal")
    return reasons


def _candidate_screen_reasons(
    candidate: Mapping[str, Any], *, thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS
) -> list[str]:
    reasons: list[str] = []
    if candidate.get("frequency_sampling_passed") is not True and candidate.get(
        "frequency_sampling_condition", {}
    ).get("passed") is not True:
        reasons.append("frequency_sampling_condition_failed")
    if candidate.get("solref_relation_passed") is not True and candidate.get(
        "eq_solref_time_constant_condition", candidate.get("solref_relation_condition", {})
    ).get("passed") is not True:
        reasons.append("eq_solref_ge_2dt_condition_failed")
    screening = candidate.get("screening")
    if not isinstance(screening, Mapping):
        return reasons + ["screening_results_missing"]
    zero = screening.get("zero")
    if not isinstance(zero, Mapping) or zero.get("passed") is not True:
        reasons.append("zero_screen_failed")
    single_axis = screening.get("single_axis")
    if not isinstance(single_axis, Mapping):
        reasons.append("single_axis_screen_results_missing")
    else:
        for axis in AXES:
            record = single_axis.get(axis)
            if not isinstance(record, Mapping) or record.get("passed") is not True:
                reasons.append(f"single_axis_screen_failed:{axis}")
    spectrum = screening.get("authored_spectrum")
    if not isinstance(spectrum, Mapping):
        reasons.append("authored_spectrum_screen_results_missing")
    else:
        conformance = spectrum.get("conformance", spectrum)
        if not isinstance(conformance, Mapping):
            reasons.append("authored_spectrum_screen_results_invalid")
        else:
            reasons.extend(f"authored_spectrum:{reason}" for reason in spectrum_gate_reasons(conformance, thresholds=thresholds))
    target_gamma = screening.get("target_gamma")
    if isinstance(target_gamma, Mapping):
        if target_gamma.get("status", "measured") != "measured":
            reasons.append("target_gamma_screen_not_measured")
        elif target_gamma.get("relative_error_abs", float("inf")) > thresholds.gamma_relative_error:
            reasons.append("target_gamma_screen_failed")
    return reasons


def candidate_screen_passed(
    candidate: Mapping[str, Any], *, thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS
) -> bool:
    """Evaluate the complete pre-registered candidate screen."""

    return not _candidate_screen_reasons(candidate, thresholds=thresholds)


def _candidate_max_phase_error(candidate: Mapping[str, Any]) -> float:
    spectrum = candidate.get("screening", {}).get("authored_spectrum", {})
    conformance = spectrum.get("conformance", spectrum) if isinstance(spectrum, Mapping) else {}
    phase_errors = [abs(float(line.get("phase_error_deg"))) for _, _, line in _spectrum_line_entries(conformance)]
    return max(phase_errors) if phase_errors else float("inf")


def select_candidate(
    candidate_grid: Iterable[Mapping[str, Any]], *, thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS
) -> dict[str, Any]:
    """Select a candidate using only the pre-registered physics screen."""

    candidates = list(candidate_grid)
    rejections = {}
    eligible = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "<missing>")) if isinstance(candidate, Mapping) else "<invalid>"
        reasons = (
            _candidate_screen_reasons(candidate, thresholds=thresholds)
            if isinstance(candidate, Mapping)
            else ["candidate_record_invalid"]
        )
        rejections[candidate_id] = reasons
        if not reasons:
            eligible.append(candidate)
    eligible.sort(
        key=lambda candidate: (
            -float(candidate.get("dt_s", float("nan"))),
            _candidate_max_phase_error(candidate),
            str(candidate.get("candidate_id", "")),
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "selection_status": "selected" if selected is not None else "blocked_no_candidate",
        "selected_candidate_id": selected.get("candidate_id") if selected is not None else None,
        "eligible_candidate_ids": [candidate.get("candidate_id") for candidate in eligible],
        "rejections": rejections,
        "selection_rule": {
            "primary": "maximum_dt_among_screening_passes",
            "secondary": "minimum_max_absolute_line_phase_error_deg_at_equal_dt",
            "tertiary": "lexicographic_candidate_id",
        },
        "thresholds": thresholds.to_dict(),
    }


def _spectrum_record_gate_passed(
    record: Mapping[str, Any], *, thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS
) -> bool:
    conformance = record.get("conformance") if isinstance(record, Mapping) else None
    if conformance is None and isinstance(record, Mapping):
        spectrum = record.get("spectrum")
        conformance = spectrum.get("conformance") if isinstance(spectrum, Mapping) else None
    return isinstance(conformance, Mapping) and not spectrum_gate_reasons(conformance, thresholds=thresholds)


def _zero_record_gate_passed(record: Mapping[str, Any]) -> bool:
    if not isinstance(record, Mapping) or record.get("status", "measured") != "measured":
        return False
    return (
        float(record.get("max_actual_pose", float("inf"))) <= 1e-10
        and float(record.get("max_actual_twist", float("inf"))) <= 1e-10
        and float(record.get("max_actual_acceleration", float("inf"))) <= 1e-10
        and int(record.get("warning_delta_sum", 1)) == 0
    )


def _confirmatory_spectrum_records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = result.get("load_matrix", ())
    return [record for record in records if isinstance(record, Mapping) and record.get("status") == "measured"]


def _gamma_record_gate_passed(
    record: Mapping[str, Any], *, thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS
) -> bool:
    if not isinstance(record, Mapping) or record.get("status") != "measured":
        return False
    target = record.get("target_Gamma_commanded")
    commanded = record.get("Gamma_commanded")
    actual = record.get("Gamma_deck_actual")
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in (target, commanded, actual)):
        return False
    if float(target) <= 0.0:
        return False
    return (
        abs(float(commanded) / float(target) - 1.0) <= thresholds.gamma_relative_error
        and abs(float(actual) / float(commanded) - 1.0) <= thresholds.gamma_relative_error
        and float(record.get("relative_error_abs", float("inf"))) <= thresholds.gamma_relative_error
    )


def compute_gate_summary(
    result: Mapping[str, Any], *, thresholds: Phase02R4Thresholds = DEFAULT_THRESHOLDS
) -> dict[str, Any]:
    """Recompute all R4 physics gates from raw result records.

    This is intentionally the single gate implementation used both by the
    generator and by :func:`verify_artifact`; a changed metric cannot leave a
    stale hand-written gate summary behind.
    """

    provenance = result.get("provenance", {}) if isinstance(result, Mapping) else {}
    candidate_grid = provenance.get("candidate_grid", ()) if isinstance(provenance, Mapping) else ()
    candidate_grid = list(candidate_grid) if isinstance(candidate_grid, list) else []
    selection = select_candidate(candidate_grid, thresholds=thresholds)
    stored_selected = provenance.get("selected_provisional_candidate_id") if isinstance(provenance, Mapping) else None
    candidate_screening_pass = bool(candidate_grid) and all(
        isinstance(candidate, Mapping)
        and candidate.get("status") in {"measured", "screen_rejected_preflight"}
        and isinstance(candidate.get("screening"), Mapping)
        for candidate in candidate_grid
    )
    candidate_selection_record = provenance.get("candidate_selection") if isinstance(provenance, Mapping) else None
    candidate_selection_pass = (
        selection["selected_candidate_id"] is not None
        and stored_selected == selection["selected_candidate_id"]
        and isinstance(candidate_selection_record, Mapping)
        and candidate_selection_record.get("selected_candidate_id") == stored_selected
    )
    selected_candidate = next(
        (
            candidate
            for candidate in candidate_grid
            if isinstance(candidate, Mapping) and candidate.get("candidate_id") == stored_selected
        ),
        None,
    )
    selected_screening = selected_candidate.get("screening", {}) if isinstance(selected_candidate, Mapping) else {}
    zero_pass = _zero_record_gate_passed(selected_screening.get("zero", {}))
    single_axis_records = selected_screening.get("single_axis", {}) if isinstance(selected_screening, Mapping) else {}
    single_axis_pass = (
        isinstance(single_axis_records, Mapping)
        and set(single_axis_records) == set(AXES)
        and all(record.get("passed") is True for record in single_axis_records.values() if isinstance(record, Mapping))
        and all(isinstance(record, Mapping) for record in single_axis_records.values())
    )
    screened_spectrum = selected_screening.get("authored_spectrum", {}) if isinstance(selected_screening, Mapping) else {}
    screened_spectrum_conformance = (
        screened_spectrum.get("conformance", screened_spectrum)
        if isinstance(screened_spectrum, Mapping)
        else {}
    )
    screened_spectrum_pass = (
        isinstance(screened_spectrum_conformance, Mapping)
        and not spectrum_gate_reasons(screened_spectrum_conformance, thresholds=thresholds)
    )

    gamma_records = result.get("gamma", {}) if isinstance(result, Mapping) else {}
    expected_gamma_keys = {"0.15", "0.3", "0.5"}
    canonical_gamma_pass = (
        isinstance(gamma_records, Mapping)
        and expected_gamma_keys.issubset(gamma_records)
        and all(
            _gamma_record_gate_passed(gamma_records[key], thresholds=thresholds)
            for key in expected_gamma_keys
        )
    )
    top_spectrum = result.get("spectrum", {}) if isinstance(result, Mapping) else {}
    estimator = top_spectrum.get("estimator_validation", {}) if isinstance(top_spectrum, Mapping) else {}
    estimator_pass = isinstance(estimator, Mapping) and estimator.get("passed") is True

    load_matrix = result.get("load_matrix", ()) if isinstance(result, Mapping) else ()
    load_records = list(load_matrix) if isinstance(load_matrix, list) else []
    expected_pairs = {
        (gamma, load_case)
        for gamma in (0.15, 0.3, 0.5)
        for load_case in CANONICAL_LOAD_CASES
    }
    actual_pairs = {
        (record.get("target_Gamma_commanded"), record.get("load_case"))
        for record in load_records
        if isinstance(record, Mapping)
    }
    gamma_load_pass = expected_pairs == actual_pairs and len(load_records) == len(expected_pairs) and all(
        _gamma_record_gate_passed(record, thresholds=thresholds)
        for record in load_records
        if isinstance(record, Mapping)
    )
    confirmatory_spectrum_pass = (
        expected_pairs == actual_pairs
        and len(load_records) == len(expected_pairs)
        and all(
            isinstance(record, Mapping)
            and _spectrum_record_gate_passed(record, thresholds=thresholds)
            for record in load_records
        )
    )
    panda_records = [
        record
        for record in load_records
        if isinstance(record, Mapping) and record.get("load_case") == LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY
    ]
    panda_pass = (
        len(panda_records) == 3
        and all(
            record.get("audit", {}).get("load_provenance", {}).get("panda_base_included") is True
            and record.get("audit", {}).get("load_provenance", {}).get("compiled_model_assertions", {}).get("passed") is True
            for record in panda_records
        )
    )
    dt_convergence_pass = len(result.get("dt_convergence", ())) == 3 if isinstance(result, Mapping) else False
    sensitivity_pass = len(result.get("sensitivity", ())) == 3 if isinstance(result, Mapping) else False

    gates = {
        "candidate_screening_gate": {
            "passed": candidate_screening_pass,
            "scope": "all pre-registered candidates complete zero/six-axis/64-line empty-deck screen",
        },
        "candidate_selection_gate": {
            "passed": candidate_selection_pass,
            "scope": "deterministic maximum-dt physics-only selection after screening",
            "selection": selection,
        },
        "zero_fixture_gate": {"passed": zero_pass, "scope": "selected candidate empty zero response"},
        "single_axis_minimal_gate": {
            "passed": single_axis_pass,
            "scope": "selected candidate six single-axis screen fits",
        },
        "authored_spectrum_line_gate": {
            "passed": screened_spectrum_pass,
            "scope": "selected candidate empty-deck 64-line screening fit",
        },
        "spectrum_estimator_gate": {
            "passed": estimator_pass,
            "scope": "pre-registered synthetic recovery with resolution/conditioning rule",
        },
        "canonical_gamma_minimal_gate": {
            "passed": canonical_gamma_pass,
            "scope": "selected candidate target Gamma records",
        },
        "gamma_load_gate": {
            "passed": gamma_load_pass,
            "scope": "confirmatory 3x2 Gamma records",
        },
        "confirmatory_spectrum_line_gate": {
            "passed": confirmatory_spectrum_pass,
            "scope": "every line in every confirmatory Gamma x load spectrum",
        },
        "panda_base_compiled_audit_gate": {
            "passed": panda_pass,
            "scope": "actual robosuite Panda subtree plus 32 kg worktable proxy compiled audit",
        },
        "dt_convergence_gate": {"passed": dt_convergence_pass, "scope": "three declared timestep records"},
        "solver_sensitivity_gate": {"passed": sensitivity_pass, "scope": "three declared solver records"},
    }
    physics_gate_values = [value["passed"] for value in gates.values()]
    overall_pass = bool(physics_gate_values) and all(physics_gate_values)
    gates["overall_phase02r_status"] = "complete" if overall_pass else "partial_or_blocked"
    gates["blocked_reason"] = None if overall_pass else [name for name, value in gates.items() if isinstance(value, Mapping) and value.get("passed") is False]
    # Keep the historical key available to consumers, but derive it from the
    # same Panda-inclusive audit rather than the old table-only proxy.
    gates["worktable_proxy_gate"] = {
        "passed": panda_pass,
        "scope": "canonical Panda/base-inclusive worktable proxy load",
    }
    return gates


def _new_probe(
    *,
    dt: float,
    tau: Optional[float],
    solimp: Iterable[float],
    trajectory: Any,
    load_case: str,
    duration_s: float,
    refresh_stride: int = 1,
    config: Optional[DeckDriverConfig] = None,
) -> tuple[DeckProbeEnv, DeckDriver, DeckDriverConfig]:
    load_case = _normalize_load_case(load_case)
    fixture = _build_probe_fixture(model_timestep=dt, load_case=load_case)
    if config is None:
        solref = (2.0 * dt if tau is None else float(tau), 1.0)
        config = DeckDriverConfig(
            eq_solref=solref,
            eq_solimp=tuple(solimp),
            physics_timestep_s=dt,
        )
    elif config.physics_timestep_s is not None and not math.isclose(
        config.physics_timestep_s, dt, rel_tol=0.0, abs_tol=1e-14
    ):
        raise ValueError("provided DeckDriverConfig.physics_timestep_s must match dt")
    driver = DeckDriver(
        trajectory=trajectory,
        config=config,
        body_handles=fixture.role_handles,
        refresh_stride=refresh_stride,
    )
    env = DeckProbeEnv(
        fixture.xml_string,
        driver,
        model_timestep=dt,
        horizon=max(1, int(math.ceil(duration_s * 20.0)) + 1),
        initial_joint_state=fixture.initial_joint_state,
    )
    env._probe_fixture = fixture
    env.reset()
    driver.reset_trace()
    return env, driver, config


def _compiled_body_subtree(raw_model: Any, root_body_id: int) -> tuple[int, ...]:
    """Return a compiled body subtree in deterministic body-id order."""

    if root_body_id < 0:
        return ()
    descendants = [root_body_id]
    for body_id in range(int(raw_model.nbody)):
        if body_id == root_body_id:
            continue
        ancestor = int(raw_model.body_parentid[body_id])
        while ancestor != 0 and ancestor != root_body_id:
            ancestor = int(raw_model.body_parentid[ancestor])
        if ancestor == root_body_id:
            descendants.append(body_id)
    return tuple(descendants)


def _compiled_panda_load_provenance(
    sim: Any,
    fixture: _ProbeFixture,
    audit: Mapping[str, Any],
    *,
    initial_joint_state: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Audit the actual compiled Panda subtree used by the no-task fixture."""

    raw_model = sim.model._model if hasattr(sim.model, "_model") else sim
    raw_data = sim.data._data if hasattr(sim, "data") and hasattr(sim.data, "_data") else None
    base_name = fixture.role_handles[PANDA_BASE_ROLE]
    base_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, base_name)
    deck_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, DeckDriverConfig().deck_body_name)
    body_ids = _compiled_body_subtree(raw_model, base_id)
    body_names = tuple(
        mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids
    )
    joint_ids = tuple(
        joint_id
        for joint_id in range(int(raw_model.njnt))
        if int(raw_model.jnt_bodyid[joint_id]) in body_ids
    )
    joint_names = tuple(
        mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in joint_ids
    )
    source_body_names = tuple(fixture.panda_body_names)
    source_joint_names = tuple(fixture.panda_joint_names)
    expected_initial_state = dict(fixture.initial_joint_state)
    actual_initial_state = {}
    if initial_joint_state is not None:
        actual_initial_state = {name: float(value) for name, value in initial_joint_state.items()}
    elif raw_data is not None:
        for joint_name in source_joint_names:
            joint_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            qpos_address = int(raw_model.jnt_qposadr[joint_id])
            actual_initial_state[joint_name] = float(raw_data.qpos[qpos_address])

    assertions = {
        "source_model_loaded": fixture.panda_source is not None,
        "source_body_names_match": tuple(sorted(body_names)) == tuple(sorted(source_body_names)),
        "source_joint_names_match": tuple(sorted(joint_names)) == tuple(sorted(source_joint_names)),
        "base_body_present": base_id >= 0,
        "base_role_parent_is_deck": base_id >= 0 and int(raw_model.body_parentid[base_id]) == deck_id,
        "all_attached_bodies_compiled": set(source_body_names).issubset(body_names),
        "all_panda_joints_initialized": (
            set(actual_initial_state) == set(expected_initial_state)
            and all(
                np.isclose(actual_initial_state[name], expected_initial_state[name], rtol=0.0, atol=1e-12)
                for name in expected_initial_state
            )
        ),
        "no_free_joint_in_panda_subtree": not any(
            int(raw_model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE) for joint_id in joint_ids
        ),
        "positive_subtree_mass": bool(body_ids)
        and float(np.sum(np.asarray(raw_model.body_mass[list(body_ids)], dtype=float))) > 0.0,
    }
    body_mass = {name: float(raw_model.body_mass[body_id]) for name, body_id in zip(body_names, body_ids)}
    body_com = {
        name: np.asarray(raw_model.body_ipos[body_id], dtype=float).tolist()
        for name, body_id in zip(body_names, body_ids)
    }
    body_inertia = {
        name: np.asarray(raw_model.body_inertia[body_id], dtype=float).tolist()
        for name, body_id in zip(body_names, body_ids)
    }
    assertions["passed"] = all(assertions.values())
    return {
        "load_case": LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY,
        "body_name": WORKTABLE_REFERENCE_PROXY_BODY_NAME,
        "attachment_role": "worktable_reference",
        "panda_base_role": PANDA_BASE_ROLE,
        "panda_base_body_name": base_name,
        "panda_base_parent_body_name": audit.get("parent_graph", {}).get(base_name),
        "panda_subtree_body_names": list(body_names),
        "attached_body_names": list(body_names),
        "panda_subtree_body_masses_kg": body_mass,
        "panda_subtree_body_com_m": body_com,
        "panda_subtree_body_inertia_kg_m2": body_inertia,
        "panda_subtree_mass_kg": float(np.sum(np.asarray(raw_model.body_mass[list(body_ids)], dtype=float)))
        if body_ids
        else 0.0,
        "initial_joint_state": actual_initial_state,
        "source": fixture.panda_source,
        "compiled_model_assertions": assertions,
        "mass_kg": WORKTABLE_REFERENCE_MASS_KG,
        "inertia_kg_m2": list(WORKTABLE_REFERENCE_INERTIA_KG_M2),
        "com_m": [0.0, 0.0, 0.0],
        "collision": False,
        "spring_damper_or_isolator": False,
        "controller_started": False,
        "task_environment": False,
        "panda_base_included": bool(assertions["passed"]),
    }


def _empty_load_provenance() -> dict[str, Any]:
    return {
        "load_case": LOAD_CASE_EMPTY,
        "mass_kg": 0.0,
        "inertia_kg_m2": [0.0, 0.0, 0.0],
        "com_m": None,
        "collision": False,
        "spring_damper_or_isolator": False,
        "controller_started": False,
        "task_environment": False,
        "panda_base_included": False,
        "source": "no load body",
    }


def run_trace(
    *,
    dt: float,
    duration_s: float,
    trajectory: Any = None,
    tau: Optional[float] = None,
    solimp: Iterable[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    load_case: str = LOAD_CASE_EMPTY,
    representative_load: Optional[bool] = None,
    refresh_stride: int = 1,
    config: Optional[DeckDriverConfig] = None,
) -> tuple[DeckDriverTrace, dict[str, Any]]:
    """Run one probe and return its trace plus compiled XML audit."""

    env, driver, config = _new_probe(
        dt=dt,
        tau=tau,
        solimp=solimp,
        trajectory=trajectory,
        load_case=_normalize_load_case(load_case, representative_load),
        duration_s=duration_s,
        refresh_stride=refresh_stride,
        config=config,
    )
    try:
        steps = int(math.ceil(duration_s * env.control_freq))
        for _ in range(steps):
            env.step(np.zeros(0))
        trace = driver.trace
        audit = audit_compiled_deck_model(env.sim, config, driver.role_handles)
        audit["xml_parent_graph"] = dict(audit_deck_xml(
            driver.processor(
                build_probe_xml(
                    model_timestep=dt,
                    load_case=_normalize_load_case(load_case, representative_load),
                )
            ),
            config,
            driver.role_handles,
        ).parent_graph)
        normalized_load_case = _normalize_load_case(load_case, representative_load)
        audit["load_provenance"] = (
            _compiled_panda_load_provenance(
                env.sim,
                env._probe_fixture,
                audit,
                initial_joint_state=env._applied_initial_joint_state,
            )
            if normalized_load_case != LOAD_CASE_EMPTY
            else _empty_load_provenance()
        )
        return trace, audit
    finally:
        env.close()


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


def _legacy_verify_artifact(path: str | Path) -> dict[str, Any]:
    """Verify a committed Phase 02R artifact without changing it.

    Args:
        path: JSON artifact path to inspect.

    Returns:
        dict: Machine-readable verification result with ``passed``, checks,
            errors, schema identity, and file/payload hashes.
    """

    artifact_path = Path(path)
    errors: list[str] = []
    checks: dict[str, bool] = {}
    if not artifact_path.is_file():
        return {
            "passed": False,
            "path": str(artifact_path),
            "errors": [f"artifact does not exist: {artifact_path}"],
            "checks": checks,
        }
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
        errors.append("artifact root must be a JSON object")
        return {
            "passed": False,
            "path": str(artifact_path),
            "errors": errors,
            "checks": checks,
            "file_sha256": _file_sha256(artifact_path),
        }

    checks["schema"] = payload.get("schema_id") == ARTIFACT_SCHEMA_ID and payload.get("schema_version") == ARTIFACT_SCHEMA_VERSION
    if not checks["schema"]:
        errors.append("artifact schema_id/schema_version does not match Phase 02R")

    integrity = payload.get("artifact_integrity")
    try:
        computed_payload_hash = _canonical_payload_hash(payload)
    except (TypeError, ValueError) as exc:
        computed_payload_hash = None
        errors.append(f"artifact contains non-canonical JSON values: {exc}")
    checks["integrity"] = (
        isinstance(integrity, dict)
        and integrity.get("hash_algorithm") == "sha256"
        and integrity.get("hash_scope") == "canonical JSON excluding artifact_integrity and artifact_update"
        and integrity.get("payload_sha256") == computed_payload_hash
    )
    if not checks["integrity"]:
        errors.append("artifact payload integrity hash is missing or does not verify")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
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
    )
    checks["command_provenance"] = all(key in provenance for key in required_provenance)
    if not checks["command_provenance"]:
        errors.append("command/time/trace provenance is incomplete")
    checks["post_integration_refresh"] = provenance.get("post_integration_refresh") is True
    if not checks["post_integration_refresh"]:
        errors.append("artifact does not declare the driver post-integration refresh")
    checks["centripetal_provenance"] = provenance.get("gamma_definition_include_centripetal") is False
    if not checks["centripetal_provenance"]:
        errors.append("Gamma centripetal provenance is not the canonical false boolean")
    candidate_grid = provenance.get("candidate_grid")
    selected_candidate = provenance.get("selected_provisional_candidate_id")
    checks["candidate_grid"] = (
        isinstance(candidate_grid, list)
        and len(candidate_grid) >= 3
            and isinstance(selected_candidate, str)
            and any(row.get("candidate_id") == selected_candidate for row in candidate_grid if isinstance(row, dict))
        and all(
            isinstance(row, dict)
            and row.get("status") == "measured"
            and isinstance(row.get("physics_metrics"), dict)
            and row.get("frequency_sampling_passed") is True
            and row.get("solref_relation_passed") is True
            for row in candidate_grid
        )
    )
    if not checks["candidate_grid"]:
        errors.append("candidate grid or selected provisional configuration is incomplete/invalid")
    checks["trace_contract"] = (
        provenance.get("trace_schema_id") == TRACE_SCHEMA_ID
        and provenance.get("trace_schema_version") == TRACE_SCHEMA_VERSION
        and provenance.get("trace_field_contract") == dict(TRACE_FIELD_CONTRACT)
    )
    if not checks["trace_contract"]:
        errors.append("artifact trace field contract does not match the runtime contract")

    matrix = payload.get("coverage_matrix")
    families = {row.get("case_family") for row in matrix if isinstance(row, dict)} if isinstance(matrix, list) else set()
    checks["coverage_matrix"] = REQUIRED_COVERAGE_FAMILIES.issubset(families)
    if not checks["coverage_matrix"]:
        errors.append("coverage matrix is missing one or more required case families")

    expected_gammas = {"0.15", "0.3", "0.5"}
    gamma_records = payload.get("gamma")
    checks["gamma_candidates"] = isinstance(gamma_records, dict) and expected_gammas.issubset(gamma_records)
    if not checks["gamma_candidates"]:
        errors.append("required target Gamma records are missing")
    if isinstance(gamma_records, dict):
        for key in sorted(expected_gammas & set(gamma_records)):
            record = gamma_records[key]
            if not isinstance(record, dict):
                errors.append(f"Gamma record {key} must be an object")
                continue
            common = (
                "target_Gamma_commanded",
                "level_scale",
                "seed",
                "t0_s",
                "profile_hash",
                "program_hash",
                "point_offset_m",
                "calibration_time_grid_s",
                "safety",
                "gamma_definition_include_centripetal",
            )
            if not all(field in record for field in common):
                errors.append(f"Gamma record {key} is missing replay provenance")
            elif record.get("gamma_definition_include_centripetal") is not False:
                errors.append(f"Gamma record {key} has inconsistent centripetal provenance")
            if record.get("status") == "measured":
                if not all(field in record for field in ("Gamma_commanded", "Gamma_deck_actual", "relative_error_abs")):
                    errors.append(f"measured Gamma record {key} is missing commanded/actual/error fields")
            elif record.get("status") == "rejected_by_candidate_safety_gate":
                if "rejection_evidence" not in record:
                    errors.append(f"rejected Gamma record {key} is missing rejection evidence")
            else:
                errors.append(f"Gamma record {key} has an unknown status")

    load_matrix = payload.get("load_matrix")
    expected_pairs = {(gamma, load) for gamma in (0.15, 0.3, 0.5) for load in (LOAD_CASE_EMPTY, LOAD_CASE_WORKTABLE_REFERENCE_PROXY)}
    actual_pairs = set()
    if isinstance(load_matrix, list):
        for record in load_matrix:
            if not isinstance(record, dict):
                continue
            actual_pairs.add((record.get("target_Gamma_commanded"), record.get("load_case")))
            if record.get("status", "measured") == "measured":
                if not all(
                    field in record
                    for field in (
                        "Gamma_commanded",
                        "Gamma_deck_actual",
                        "relative_error_abs",
                        "audit",
                        "gamma_definition_include_centripetal",
                    )
                ):
                    errors.append("measured Gamma×load record is missing key metrics")
                elif record.get("gamma_definition_include_centripetal") is not False:
                    errors.append("Gamma×load record has inconsistent centripetal provenance")
                audit = record.get("audit", {})
                load_provenance = audit.get("load_provenance") if isinstance(audit, dict) else None
                if not isinstance(load_provenance, dict):
                    errors.append("Gamma×load record is missing load provenance")
                elif record.get("load_case") == LOAD_CASE_WORKTABLE_REFERENCE_PROXY:
                    if load_provenance.get("mass_kg") != WORKTABLE_REFERENCE_MASS_KG or load_provenance.get("inertia_kg_m2") != list(WORKTABLE_REFERENCE_INERTIA_KG_M2):
                        errors.append("worktable proxy does not carry canonical mass/inertia")
                    if load_provenance.get("panda_base_included") is not False:
                        errors.append("Panda/base proxy inclusion is not explicitly declared")
    checks["gamma_load_matrix"] = expected_pairs.issubset(actual_pairs)
    if not checks["gamma_load_matrix"]:
        errors.append("required Gamma×load records are missing")

    spectrum = payload.get("spectrum", {})
    conformance = spectrum.get("conformance", {}) if isinstance(spectrum, dict) else {}
    axes = conformance.get("axes", {}) if isinstance(conformance, dict) else {}
    line_count = sum(len(value.get("line_fits", [])) for value in axes.values() if isinstance(value, dict))
    checks["spectrum_lines"] = line_count == 64
    if not checks["spectrum_lines"]:
        errors.append(f"spectrum has {line_count} fitted lines; expected 64")
    checks["spectrum_resolution"] = (
        isinstance(conformance, dict)
        and conformance.get("fit_window_meets_resolution") is True
        and float(conformance.get("fit_window_s", 0.0)) >= float(conformance.get("required_fit_window_s", float("inf")))
    )
    if not checks["spectrum_resolution"]:
        errors.append("spectrum fit window is shorter than two minimum beat periods")
    checks["spectrum_line_fields"] = True
    for value in axes.values() if isinstance(axes, dict) else ():
        for line in value.get("line_fits", []) if isinstance(value, dict) else ():
            required_line_fields = ("frequency_hz", "amplitude_relative_error", "phase_error_deg", "fit_residual_rms", "condition_number", "phase_convention")
            if not all(field in line for field in required_line_fields):
                checks["spectrum_line_fields"] = False
    if not checks["spectrum_line_fields"]:
        errors.append("one or more spectrum line fits lack required diagnostics")

    gate_summary = payload.get("gate_summary")
    checks["gate_summary"] = isinstance(gate_summary, dict) and all(
        isinstance(value, dict) and isinstance(value.get("passed"), bool)
        for key, value in gate_summary.items()
        if key not in {"blocked_reason", "overall_phase02r_status"}
    )
    if not checks["gate_summary"]:
        errors.append("gate summary is incomplete")
    overall_status = gate_summary.get("overall_phase02r_status") if isinstance(gate_summary, dict) else None
    single_records = payload.get("single_axis", {})
    computed_single_pass = isinstance(single_records, dict) and len(single_records) == len(AXES) and all(
        isinstance(record, dict)
        and isinstance(record.get("conformance"), dict)
        and record["conformance"].get("amplitude_relative_error", float("inf")) <= 0.01
        and abs(record["conformance"].get("phase_error_deg", float("inf"))) <= 1.0
        for record in single_records.values()
    )
    computed_gamma_pass = (
        isinstance(gamma_records, dict)
        and len(gamma_records) == len(expected_gammas)
        and all(isinstance(record, dict) for record in gamma_records.values())
        and all(
            record.get("status") == "measured" and record.get("relative_error_abs", float("inf")) <= 0.01
            for record in gamma_records.values()
        )
    )
    computed_spectrum_pass = (
        checks["spectrum_lines"]
        and checks["spectrum_resolution"]
        and checks["spectrum_line_fields"]
        and all(
            line.get("amplitude_relative_error", float("inf")) <= 0.01
            and abs(line.get("phase_error_deg", float("inf"))) <= 1.0
            and line.get("condition_number", float("inf")) <= 10000.0
            and (
                line.get("fit_residual_rms", float("inf")) / axis.get("actual_rms", 0.0) <= 1e-3
                if axis.get("actual_rms", 0.0)
                else line.get("fit_residual_rms", float("inf")) == 0.0
            )
            for axis in axes.values()
            if isinstance(axis, dict)
            for line in axis.get("line_fits", [])
        )
    )
    estimator = spectrum.get("estimator_validation", {}) if isinstance(spectrum, dict) else {}
    computed_estimator_pass = isinstance(estimator, dict) and estimator.get("passed") is True
    computed_load_gamma_pass = isinstance(load_matrix, list) and bool(load_matrix) and all(
        record.get("status") == "measured" and record.get("relative_error_abs", float("inf")) <= 0.01
        for record in load_matrix
        if isinstance(record, dict)
    )
    proxy_matrix_records = [
        record
        for record in load_matrix
        if isinstance(record, dict) and record.get("load_case") == LOAD_CASE_WORKTABLE_REFERENCE_PROXY
    ] if isinstance(load_matrix, list) else []
    computed_proxy_pass = bool(proxy_matrix_records) and all(
        record.get("status") == "measured"
        and record.get("load_case") == LOAD_CASE_WORKTABLE_REFERENCE_PROXY
        and record.get("audit", {}).get("load_provenance", {}).get("panda_base_included") is True
        for record in proxy_matrix_records
    )
    expected_gate_values = {
        "provisional_lower_bound_gate": payload.get("provisional_gate", {}).get("passed") is True,
        "single_axis_minimal_gate": computed_single_pass,
        "canonical_gamma_minimal_gate": computed_gamma_pass,
        "authored_spectrum_line_gate": computed_spectrum_pass,
        "spectrum_estimator_gate": computed_estimator_pass,
        "gamma_load_gate": computed_load_gamma_pass,
        "worktable_proxy_gate": computed_proxy_pass,
    }
    checks["gate_consistency"] = isinstance(gate_summary, dict) and all(
        isinstance(gate_summary.get(name), dict) and gate_summary[name].get("passed") is value
        for name, value in expected_gate_values.items()
    ) and ((overall_status == "complete") is all(expected_gate_values.values()))
    if not checks["gate_consistency"]:
        errors.append("gate summary does not agree with recomputed artifact measurements")
    handoff = payload.get("phase03_handoff")
    checks["phase03_handoff"] = (overall_status == "complete" and handoff == "PASS") or (
        overall_status != "complete" and handoff == "BLOCKED"
    )
    if not checks["phase03_handoff"]:
        errors.append("Phase 03 handoff does not agree with artifact gate status")

    update_record = payload.get("artifact_update")
    if update_record is None:
        checks["update_provenance"] = True
    else:
        checks["update_provenance"] = (
            isinstance(update_record, dict)
            and bool(str(update_record.get("reason", "")).strip())
            and "previous_file_sha256" in update_record
            and update_record.get("new_payload_sha256")
            == (integrity.get("payload_sha256") if isinstance(integrity, dict) else None)
        )
        if not checks["update_provenance"]:
            errors.append("artifact update record lacks a non-empty reason or old/new hash")

    integrity_valid = not errors
    physics_gates_passed = isinstance(gate_summary, dict) and all(
        isinstance(value, dict) and value.get("passed") is True
        for key, value in gate_summary.items()
        if key not in {"blocked_reason", "overall_phase02r_status"}
    )
    return {
        "passed": integrity_valid,
        "integrity_valid": integrity_valid,
        "physics_gates_passed": physics_gates_passed,
        "phase03_handoff": handoff,
        "path": str(artifact_path),
        "schema_id": payload.get("schema_id"),
        "schema_version": payload.get("schema_version"),
        "checks": checks,
        "errors": errors,
        "file_sha256": _file_sha256(artifact_path),
        "payload_sha256": computed_payload_hash,
    }


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


def _legacy_run_probe_suite(
    *,
    dt: float = 0.00005,
    duration_s: float = 1.0,
    frequencies_hz: Iterable[float] = (5.0,),
    gammas: Iterable[float] = (0.15, 0.30, 0.50),
    spectrum_duration_s: Optional[float] = None,
) -> dict[str, Any]:
    """Run the machine-readable Phase 02R conformance coverage matrix.

    ``gammas`` are target ``Gamma_commanded`` values. They are converted to
    authored ``level_scale`` values through the Phase 01 unit replay; the
    scale is never renamed to Gamma and is never retuned from realized deck
    data.
    """

    frequencies_hz = tuple(float(value) for value in frequencies_hz)
    gammas = tuple(float(value) for value in gammas)
    if not frequencies_hz:
        raise ValueError("frequencies_hz must contain at least one value")
    if not gammas:
        raise ValueError("gammas must contain at least one target Gamma")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")

    point_offset_m = (0.65, 0.0, 0.0)
    support_normal = (0.0, 0.0, 1.0)
    seed = 17
    t0 = 0.0
    default_solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
    default_tau = 2.0 * dt
    sample_steps = int(math.ceil(duration_s / dt))
    sample_time_grid = np.arange(1, sample_steps + 1, dtype=float) * dt
    profile = build_excitation_program(seed=seed, t0=t0, level_scale=1.0)
    frequency_dt_limit = 1.0 / (20.0 * profile.max_frequency_hz)
    candidate_grid = []
    for candidate_id, candidate_dt in (
        ("fine", dt / 2.0),
        ("selected", dt),
        ("coarse", dt * 2.0),
    ):
        candidate_tau = 2.0 * candidate_dt
        candidate_grid.append(
            {
                "candidate_id": candidate_id,
                "dt_s": candidate_dt,
                "eq_solref": [candidate_tau, 1.0],
                "eq_solimp": list(default_solimp),
                "frequency_sampling_limit_s": frequency_dt_limit,
                "frequency_sampling_passed": candidate_dt <= frequency_dt_limit,
                "solref_minimum_s": 2.0 * candidate_dt,
                "solref_relation_passed": candidate_tau >= 2.0 * candidate_dt,
                "deck_mass_kg": DeckDriverConfig().deck_mass_kg,
                "deck_inertia_kg_m2": list(DeckDriverConfig().deck_inertia_kg_m2),
                "selection_basis": "physics-only frequency and equality time-constant conditions; no task success",
            }
        )

    result: dict[str, Any] = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "provenance": {
            "phase": "02R",
            "status": "provisional; derived-not-frozen",
            "mujoco_version": mujoco.__version__,
            "trace_schema_id": TRACE_SCHEMA_ID,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_field_contract": dict(TRACE_FIELD_CONTRACT),
            "physics_profile_freeze": "deferred-to-phase-06",
            "deck_mass_inertia_status": "derived-not-frozen",
            "dt_s": float(dt),
            "seed": seed,
            "t0_s": t0,
            "excitation_profile_hash": excitation_profile_hash(profile.config),
            "excitation_profile_id": profile.config.to_dict()["profile_id"],
            "point_offset_m": list(point_offset_m),
            "support_normal_world": list(support_normal),
            "gamma_definition_include_centripetal": False,
            "post_integration_refresh": True,
            "sample_time_grid_s": sample_time_grid.tolist(),
            "time_convention": {
                "command_timestamp": "target evaluated time passed to trajectory",
                "application_timestamp": "mocap write time at pre-physics hook",
                "sample_timestamp": "post-step state data.time",
                "fit": "evaluate authored command at the same post-step sample timestamps; never shift actual trace",
            },
            "fit_convention": {
                "basis": "joint sine/cosine basis plus intercept per authored frequency line",
                "phase": "A*sin(omega*t + phase); atan2(cosine coefficient, sine coefficient)",
                "transient_discard_s": "recorded per case",
                "sample_rate": "median inverse sample-time difference",
            },
            "thresholds": {
                "amplitude_relative_error": 0.01,
                "canonical_gamma_relative_error": 0.01,
                "absolute_phase_error_deg": 1.0,
                "positive_eq_solref_time_constant_min_factor_dt": 2.0,
            },
            "not_frozen": ["physics_dt", "deck_mass_kg", "deck_inertia_kg_m2", "eq_solref", "eq_solimp"],
            "candidate_grid": candidate_grid,
            "selected_provisional_candidate_id": "selected",
        },
        "coverage_matrix": [
            {
                "case_family": "provisional_lower_bound",
                "case_ids": ["tx_8.87Hz"],
                "axes": ["tx"],
                "measurements": ["amplitude", "phase", "translation_origin_Gamma"],
                "not_a_substitute_for": "canonical six-axis workpiece-point Gamma gate",
            },
            {
                "case_family": "zero",
                "case_ids": ["zero"],
                "axes": list(AXES),
                "measurements": [
                    "pose",
                    "twist",
                    "acceleration",
                    "deck_tracking_pose_error",
                    "weld_constraint_residual_raw",
                    "weld_constraint_force_raw",
                    "warning_delta",
                ],
            },
            {
                "case_family": "single_axis",
                "case_ids": list(AXES),
                "axes": list(AXES),
                "measurements": ["amplitude", "phase", "Gamma_when_applicable"],
            },
            {
                "case_family": "authored_spectrum",
                "case_ids": ["authored_spectrum"],
                "axes": list(AXES),
                "measurements": ["every_active_line_amplitude", "every_active_line_phase", "fit_residual"],
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
                    f"{target}::{load_case}"
                    for target in gammas
                    for load_case in (LOAD_CASE_EMPTY, LOAD_CASE_WORKTABLE_REFERENCE_PROXY)
                ],
                "axes": list(AXES),
                "measurements": ["tracking", "raw_weld_diagnostics", "Gamma_commanded", "Gamma_deck_actual"],
            },
            {
                "case_family": "dt_convergence",
                "case_ids": ["dt_low", "dt_nominal", "dt_high"],
                "axes": list(AXES),
                "measurements": ["tracking", "raw_weld_diagnostics", "solver", "warning_delta"],
            },
            {
                "case_family": "solver_sensitivity",
                "case_ids": ["baseline", "eq_solref_time_constant", "eq_solimp_dmin"],
                "axes": list(AXES),
                "measurements": ["tracking", "raw_weld_diagnostics", "solver", "warning_delta"],
            },
        ],
        "zero": {},
        "single_axis": {},
        "spectrum": {},
        "gamma": {},
        "load_cases": [LOAD_CASE_EMPTY, LOAD_CASE_WORKTABLE_REFERENCE_PROXY],
        "load_matrix": [],
        "dt_convergence": [],
        "sensitivity": [],
        "phase03_handoff": "BLOCKED",
    }

    def trace_summary(trace: DeckDriverTrace) -> dict[str, Any]:
        return {
            "samples": int(trace.sample_timestamps_s.size),
            "max_deck_tracking_pose_error": (
                float(np.max(np.abs(trace.deck_tracking_pose_error)))
                if trace.deck_tracking_pose_error.size
                else 0.0
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
            "warning_delta_sum": (
                int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0
            ),
            "warning_delta_max": (
                int(np.max(trace.warning_number_delta)) if trace.warning_number_delta.size else 0
            ),
        }

    # Every declared candidate is actually run through the physics-only
    # driver before any candidate is selected. These probes use the same
    # authored line and no task outcome, so selection cannot be task-tuned.
    candidate_duration_s = max(0.25, min(duration_s, 0.25))
    for candidate in candidate_grid:
        candidate_config = DeckDriverConfig(
            eq_solref=tuple(candidate["eq_solref"]),
            eq_solimp=tuple(candidate["eq_solimp"]),
            physics_timestep_s=candidate["dt_s"],
        )
        candidate_trace, candidate_audit = run_trace(
            dt=candidate["dt_s"],
            duration_s=candidate_duration_s,
            trajectory=_sine_command(0, 5.0, 0.001),
            config=candidate_config,
        )
        candidate_pose_fit = sine_conformance(
            candidate_trace,
            axis_index=0,
            frequency_hz=5.0,
            amplitude=0.001,
            discard_s=0.1,
        )
        candidate_gamma_fit = gamma_conformance(
            candidate_trace,
            axis_index=0,
            frequency_hz=5.0,
            displacement_amplitude=0.001,
            discard_s=0.1,
        )
        candidate["status"] = "measured"
        candidate["physics_metrics"] = {
            "pose_conformance": candidate_pose_fit,
            "translation_origin_gamma_conformance": candidate_gamma_fit,
            "diagnostics": trace_summary(candidate_trace),
        }
        candidate["audit"] = candidate_audit
    result["provenance"]["candidate_grid"] = candidate_grid

    # Zero fixture: every state and diagnostic must remain static, including
    # warning deltas rather than just finite array shapes.
    zero_trace, zero_audit = run_trace(dt=dt, duration_s=min(duration_s, 0.25), tau=default_tau)
    result["zero"] = {
        **trace_summary(zero_trace),
        "max_actual_pose": float(np.max(np.abs(_actual_coordinates(zero_trace)))) if zero_trace.actual_pose.size else 0.0,
        "audit": zero_audit,
    }

    # This is the original Phase 02 lower-bound commitment: one translational
    # line at the conservative authored upper frequency. It is recorded
    # separately and is not promoted to six-axis conformance.
    provisional_frequency = 8.87
    provisional_amplitude = 0.001
    provisional_trace, provisional_audit = run_trace(
        dt=dt,
        duration_s=max(duration_s, 0.5),
        trajectory=_sine_command(0, provisional_frequency, provisional_amplitude),
        tau=default_tau,
    )
    provisional_pose_fit = sine_conformance(
        provisional_trace,
        axis_index=0,
        frequency_hz=provisional_frequency,
        amplitude=provisional_amplitude,
        discard_s=min(0.2, max(0.1, duration_s / 2.0)),
    )
    provisional_gamma_fit = gamma_conformance(
        provisional_trace,
        axis_index=0,
        frequency_hz=provisional_frequency,
        displacement_amplitude=provisional_amplitude,
        discard_s=provisional_pose_fit["discard_s"],
    )
    result["provisional_gate"] = {
        "scope": "driver_lower_bound_single_axis_translation; not six_axis_conformance",
        "frequency_hz": provisional_frequency,
        "amplitude_m": provisional_amplitude,
        "pose_fit": provisional_pose_fit,
        "Gamma_fit": provisional_gamma_fit,
        "audit": provisional_audit,
        "passed": (
            provisional_pose_fit["amplitude_relative_error"] <= 0.01
            and abs(provisional_pose_fit["phase_error_deg"]) <= 1.0
            and provisional_gamma_fit["relative_error_abs"] <= 0.01
        ),
        "thresholds": {"amplitude_relative_error": 0.01, "phase_error_deg": 1.0, "Gamma_relative_error": 0.01},
    }

    # Six independent lower-bound single-axis cases. The translation Gamma
    # is intentionally marked as a driver-origin smoke diagnostic; the
    # canonical workpiece-point Gamma gate lives in the target-Gamma block.
    single_gate_values = []
    for axis_index, axis in enumerate(AXES):
        frequency = frequencies_hz[axis_index % len(frequencies_hz)]
        amplitude = 0.001 if axis_index < 3 else 0.0005
        trace, audit = run_trace(
            dt=dt,
            duration_s=max(duration_s, 0.5),
            trajectory=_sine_command(axis_index, frequency, amplitude),
            tau=default_tau,
        )
        conformance = sine_conformance(
            trace,
            axis_index=axis_index,
            frequency_hz=frequency,
            amplitude=amplitude,
            discard_s=min(0.4, max(0.2, duration_s / 2.0)),
        )
        record: dict[str, Any] = {
            "axis": axis,
            "frequency_hz": frequency,
            "coordinate_unit": "m" if axis_index < 3 else "rad",
            "audit": audit,
            "conformance": conformance,
            "diagnostics": trace_summary(trace),
        }
        if axis_index < 3:
            record["Gamma_lower_bound_diagnostic"] = gamma_conformance(
                trace,
                axis_index=axis_index,
                frequency_hz=frequency,
                displacement_amplitude=amplitude,
                discard_s=conformance["discard_s"],
            )
        result["single_axis"][axis] = record
        single_gate_values.append(conformance)

    # Authored six-axis spectrum: derive the fit window from the smallest
    # within-axis line spacing before running the real driver. This makes the
    # resolution requirement a pre-registered property of the authored
    # program, rather than a post-hoc interpretation of a short trace.
    spectrum_target_gamma = 0.30 if 0.30 in gammas else gammas[0]
    spectrum_calibration_duration = max(1.0, duration_s)
    spectrum_calibration_steps = int(math.ceil(spectrum_calibration_duration / dt))
    spectrum_calibration_grid = np.arange(1, spectrum_calibration_steps + 1, dtype=float) * dt
    spectrum_level_scale = level_scale_for_gamma(
        spectrum_target_gamma,
        seed=seed,
        t0=t0,
        time=spectrum_calibration_grid,
        point_offset_m=point_offset_m,
        sample_count=spectrum_calibration_grid.size,
    )
    spectrum_program = build_excitation_program(seed=seed, t0=t0, level_scale=spectrum_level_scale)
    spectrum_minimum_spacing = minimum_line_spacing_hz(spectrum_program)
    spectrum_required_window = 2.0 / spectrum_minimum_spacing
    spectrum_discard = max(0.75, float(spectrum_program.config.ramp_duration_s) + 0.25)
    result["provenance"].update(
        {
            "minimum_line_spacing_hz": spectrum_minimum_spacing,
            "minimum_line_beat_period_s": 1.0 / spectrum_minimum_spacing,
            "required_spectrum_fit_window_s": spectrum_required_window,
            "spectrum_transient_discard_s": spectrum_discard,
        }
    )
    requested_spectrum_duration = (
        spectrum_discard + spectrum_required_window + dt
        if spectrum_duration_s is None
        else float(spectrum_duration_s)
    )
    if not np.isfinite(requested_spectrum_duration) or requested_spectrum_duration <= 0.0:
        raise ValueError("spectrum_duration_s must be finite and positive")
    spectrum_duration = max(requested_spectrum_duration, duration_s)
    spectrum_steps = int(math.ceil(spectrum_duration / dt))
    spectrum_config = DeckDriverConfig(
        physics_timestep_s=dt,
        eq_solref=(default_tau, 1.0),
        eq_solimp=default_solimp,
    )
    spectrum_trace, spectrum_audit = run_trace(
        dt=dt,
        duration_s=spectrum_duration,
        trajectory=spectrum_program,
        config=spectrum_config,
    )
    spectrum_canonical_gamma = canonical_gamma_conformance(
        spectrum_trace,
        spectrum_program,
        config=spectrum_config,
        point_offset_m=point_offset_m,
        support_normal=support_normal,
        discard_s=spectrum_discard,
    )
    result["spectrum"] = {
        "case_id": "authored_spectrum",
        "seed": seed,
        "t0_s": t0,
        "target_Gamma_commanded": spectrum_target_gamma,
        "level_scale": spectrum_level_scale,
        "profile_hash": excitation_profile_hash(spectrum_program.config),
        "program_hash": _program_hash(spectrum_program),
        "active_axes": list(spectrum_program.active_axes),
        "active_line_count": int(np.sum(spectrum_program.line_mask)),
        "max_frequency_hz": spectrum_program.max_frequency_hz,
        "program": spectrum_program.to_dict(),
        "samples": int(spectrum_trace.sample_timestamps_s.size),
        "audit": spectrum_audit,
        "conformance": spectrum_conformance(
            spectrum_trace,
            spectrum_program,
            discard_s=spectrum_discard,
            max_fit_samples=50000,
        ),
        "fit_window_s": float(spectrum_duration - spectrum_discard),
        "estimator_validation": synthetic_spectrum_estimator_validation(
            spectrum_program,
            sample_dt_s=dt,
        ),
        "canonical_gamma": spectrum_canonical_gamma,
        "diagnostics": trace_summary(spectrum_trace),
        "time_grid_s": spectrum_calibration_grid.tolist(),
    }

    # Use the Phase 01 unit replay to turn explicit target Gamma values into
    # authored scales. Safety rejection is retained as evidence and never
    # silently replaced by another candidate.
    gamma_duration = max(1.0, duration_s)
    gamma_steps = int(math.ceil(gamma_duration / dt))
    gamma_time_grid = np.arange(1, gamma_steps + 1, dtype=float) * dt
    gamma_records = []
    for target_gamma in gammas:
        level = level_scale_for_gamma(
            target_gamma,
            seed=seed,
            t0=t0,
            time=gamma_time_grid,
            point_offset_m=point_offset_m,
            sample_count=gamma_time_grid.size,
        )
        gamma_program = build_excitation_program(seed=seed, t0=t0, level_scale=level)
        phase01_calibration = calibrate_gamma(
            gamma_program,
            time=gamma_time_grid,
            point_offset_m=point_offset_m,
            support_normal=support_normal,
            include_centripetal=False,
        )
        safety = check_safety(
            gamma_program,
            timestep_s=dt,
            time=gamma_time_grid,
            sample_count=gamma_time_grid.size,
            point_offset_m=point_offset_m,
        )
        record = {
            "target_Gamma_commanded": float(target_gamma),
            "level_scale": float(level),
            "seed": seed,
            "t0_s": t0,
            "profile_hash": excitation_profile_hash(gamma_program.config),
            "program_hash": _program_hash(gamma_program),
            "profile_id": gamma_program.config.to_dict()["profile_id"],
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
        }
        if not safety.passed:
            record["status"] = "rejected_by_candidate_safety_gate"
            record["rejection_evidence"] = safety.to_dict()
            result["gamma"][str(target_gamma)] = record
            gamma_records.append(record)
            continue
        gamma_trace, gamma_audit = run_trace(
            dt=dt,
            duration_s=gamma_duration,
            trajectory=gamma_program,
            tau=default_tau,
        )
        canonical = canonical_gamma_conformance(
            gamma_trace,
            gamma_program,
            config=DeckDriverConfig(physics_timestep_s=dt, eq_solref=(default_tau, 1.0)),
            point_offset_m=point_offset_m,
            support_normal=support_normal,
            discard_s=0.0,
        )
        record.update(
            {
                "status": "measured",
                "Gamma_commanded": canonical["Gamma_commanded"],
                "Gamma_deck_actual": canonical["Gamma_deck_actual"],
                "relative_error_abs": canonical["relative_error_abs"],
                "canonical_gamma": canonical,
                "audit": gamma_audit,
                "diagnostics": trace_summary(gamma_trace),
            }
        )
        result["gamma"][str(target_gamma)] = record
        gamma_records.append(record)

    # Reuse the middle target candidate for load, dt and solver comparisons.
    reference_target = 0.30 if 0.30 in gammas else gammas[0]
    reference_record = result["gamma"].get(str(reference_target))
    reference_selection: dict[str, Any] = {
        "requested_target_Gamma": float(reference_target),
        "selected_target_Gamma": None,
        "selection_reason": None,
    }
    if reference_record is not None and reference_record.get("status") == "measured":
        selected_target = reference_target
        reference_level = float(reference_record["level_scale"])
        reference_selection["selection_reason"] = "requested_target_measured"
    else:
        measured_records = [
            (float(key), record)
            for key, record in result["gamma"].items()
            if record.get("status") == "measured"
        ]
        if not measured_records:
            selected_target = None
            reference_level = None
            reference_selection["selection_reason"] = "all_requested_targets_rejected"
            reference_selection["rejection_evidence"] = (
                reference_record.get("rejection_evidence") if reference_record is not None else None
            )
        else:
            selected_target, selected_record = measured_records[0]
            reference_level = float(selected_record["level_scale"])
            reference_selection["selection_reason"] = "explicit_fallback_after_requested_target_rejection"
            reference_selection["rejected_target_evidence"] = (
                reference_record.get("rejection_evidence") if reference_record is not None else None
            )
    reference_selection["selected_target_Gamma"] = selected_target
    result["provenance"]["reference_candidate_selection"] = reference_selection
    if selected_target is None:
        reference_program = None
    else:
        reference_program = build_excitation_program(seed=seed, t0=t0, level_scale=reference_level)

    for target_gamma in gammas:
        gamma_record = result["gamma"][str(target_gamma)]
        for load_case in (LOAD_CASE_EMPTY, LOAD_CASE_WORKTABLE_REFERENCE_PROXY):
            case_id = f"{target_gamma}::{load_case}"
            if gamma_record.get("status") != "measured":
                load_record = {
                    "case_id": case_id,
                    "target_Gamma_commanded": float(target_gamma),
                    "load_case": load_case,
                    "status": "blocked_gamma_candidate_rejected",
                    "gamma_definition_include_centripetal": False,
                    "rejection_evidence": gamma_record.get("rejection_evidence"),
                }
                result["load_matrix"].append(load_record)
                continue
            load_level = float(gamma_record["level_scale"])
            load_program = build_excitation_program(seed=seed, t0=t0, level_scale=load_level)
            load_trace, load_audit = run_trace(
                dt=dt,
                duration_s=min(gamma_duration, 0.75),
                trajectory=load_program,
                tau=default_tau,
                load_case=load_case,
            )
            canonical = canonical_gamma_conformance(
                load_trace,
                load_program,
                config=DeckDriverConfig(physics_timestep_s=dt, eq_solref=(default_tau, 1.0)),
                point_offset_m=point_offset_m,
                support_normal=support_normal,
                discard_s=0.0,
            )
            load_record = {
                "case_id": case_id,
                "target_Gamma_commanded": float(target_gamma),
                "load_case": load_case,
                "status": "measured",
                "seed": seed,
                "t0_s": t0,
                "level_scale": load_level,
                "profile_hash": excitation_profile_hash(load_program.config),
                "program_hash": _program_hash(load_program),
                "gamma_definition_include_centripetal": False,
                "Gamma_commanded": canonical["Gamma_commanded"],
                "Gamma_deck_actual": canonical["Gamma_deck_actual"],
                "relative_error_abs": canonical["relative_error_abs"],
                "canonical_gamma": canonical,
                "diagnostics": trace_summary(load_trace),
                "audit": load_audit,
                "comparable_metrics": [
                    "deck_tracking_pose_error",
                    "weld_constraint_residual_raw",
                    "weld_constraint_force_raw",
                    "Gamma_commanded",
                    "Gamma_deck_actual",
                    "relative_error_abs",
                ],
            }
            result["load_matrix"].append(load_record)

    # Three dt cases hold deck mass/inertia, authored excitation, load, weld
    # solimp and the positive solref time constant fixed. Only dt changes.
    dt_values = (dt / 2.0, dt, dt * 2.0)
    fixed_tau = 2.0 * max(dt_values)
    for index, case_dt in enumerate(dt_values):
        fixed_config = DeckDriverConfig(
            eq_solref=(fixed_tau, 1.0),
            eq_solimp=default_solimp,
            physics_timestep_s=case_dt,
        )
        case_id = ("dt_low", "dt_nominal", "dt_high")[index]
        if reference_program is None:
            result["dt_convergence"].append(
                {
                    "case_id": case_id,
                    "status": "blocked_reference_candidate_rejected",
                    "dt_s": case_dt,
                    "eq_solref": [fixed_tau, 1.0],
                    "eq_solimp": list(default_solimp),
                    "reference_candidate_selection": reference_selection,
                }
            )
            continue
        trace, audit = run_trace(
            dt=case_dt,
            duration_s=min(gamma_duration, 0.5),
            trajectory=reference_program,
            load_case=LOAD_CASE_WORKTABLE_REFERENCE_PROXY,
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
                "dt_s": case_dt,
                "eq_solref": [fixed_tau, 1.0],
                "eq_solimp": list(default_solimp),
                "solref_minimum_check": {"required": 2.0 * max(dt_values), "passed": fixed_tau >= 2.0 * max(dt_values)},
                "deck_mass_kg": fixed_config.deck_mass_kg,
                "deck_inertia_kg_m2": list(fixed_config.deck_inertia_kg_m2),
                "excitation": {
                    "seed": seed,
                    "t0_s": t0,
                    "level_scale": reference_level,
                    "profile_hash": excitation_profile_hash(reference_program.config),
                    "program_hash": _program_hash(reference_program),
                },
                "load": LOAD_CASE_WORKTABLE_REFERENCE_PROXY,
                "Gamma_commanded": canonical["Gamma_commanded"],
                "Gamma_deck_actual": canonical["Gamma_deck_actual"],
                "relative_error_abs": canonical["relative_error_abs"],
                "diagnostics": trace_summary(trace),
                "audit": audit,
            }
        )

    # Solver sensitivity changes exactly one declared solver variable per
    # case; dt, excitation, load, mass/inertia and the other solver parameter
    # stay at the same baseline values.
    sensitivity_cases = (
        ("baseline", "none", fixed_tau, default_solimp),
        ("eq_solref_time_constant", "eq_solref_time_constant", 2.0 * fixed_tau, default_solimp),
        ("eq_solimp_dmin", "eq_solimp_dmin", fixed_tau, (0.85, 0.95, 0.001, 0.5, 2.0)),
    )
    for case_id, changed_variable, case_tau, case_solimp in sensitivity_cases:
        sensitivity_config = DeckDriverConfig(
            eq_solref=(case_tau, 1.0),
            eq_solimp=case_solimp,
            physics_timestep_s=dt,
        )
        if reference_program is None:
            result["sensitivity"].append(
                {
                    "case_id": case_id,
                    "status": "blocked_reference_candidate_rejected",
                    "changed_variable": changed_variable,
                    "dt_s": dt,
                    "eq_solref": [case_tau, 1.0],
                    "eq_solimp": list(case_solimp),
                    "reference_candidate_selection": reference_selection,
                }
            )
            continue
        trace, audit = run_trace(
            dt=dt,
            duration_s=min(gamma_duration, 0.5),
            trajectory=reference_program,
            load_case=LOAD_CASE_WORKTABLE_REFERENCE_PROXY,
            config=sensitivity_config,
        )
        result["sensitivity"].append(
            {
                "case_id": case_id,
                "changed_variable": changed_variable,
                "dt_s": dt,
                "eq_solref": [case_tau, 1.0],
                "eq_solimp": list(case_solimp),
                "baseline_eq_solref": [fixed_tau, 1.0],
                "baseline_eq_solimp": list(default_solimp),
                "unchanged": ["dt", "deck_mass_kg", "deck_inertia_kg_m2", "excitation", "load"],
                "diagnostics": trace_summary(trace),
                "audit": audit,
            }
        )

    single_pass = all(
        value["amplitude_relative_error"] <= 0.01 and abs(value["phase_error_deg"]) <= 1.0
        for value in single_gate_values
    )
    gamma_measured = [value for value in gamma_records if value.get("status") == "measured"]
    gamma_pass = len(gamma_measured) == len(gammas) and all(
        value["relative_error_abs"] <= 0.01 for value in gamma_measured
    )
    spectrum_lines = [
        line
        for axis in result["spectrum"].get("conformance", {}).get("axes", {}).values()
        for line in axis.get("line_fits", [])
    ]
    spectrum_conformance_result = result["spectrum"].get("conformance", {})
    spectrum_pass = bool(spectrum_lines) and all(
        line["amplitude_relative_error"] <= 0.01
        and abs(line["phase_error_deg"]) <= 1.0
        and line["condition_number"] <= 10000.0
        and (
            line["fit_residual_rms"] / axis["actual_rms"] <= 1e-3
            if axis["actual_rms"]
            else line["fit_residual_rms"] == 0.0
        )
        for axis in result["spectrum"].get("conformance", {}).get("axes", {}).values()
        for line in axis.get("line_fits", [])
    ) and (
        spectrum_conformance_result.get("fit_window_meets_resolution") is True
        and len(spectrum_lines) == 64
    )
    synthetic_estimator_pass = result["spectrum"].get("estimator_validation", {}).get("passed") is True
    proxy_records = [
        record
        for record in result["load_matrix"]
        if record.get("load_case") == LOAD_CASE_WORKTABLE_REFERENCE_PROXY
    ]
    load_proxy_pass = bool(proxy_records) and all(
        record.get("status") == "measured"
        and record.get("audit", {}).get("load_provenance", {}).get("panda_base_included") is True
        for record in proxy_records
    )
    load_gamma_pass = bool(result["load_matrix"]) and all(
        record.get("status") == "measured" and record.get("relative_error_abs", float("inf")) <= 0.01
        for record in result["load_matrix"]
    )
    overall_pass = (
        result["provisional_gate"]["passed"]
        and single_pass
        and gamma_pass
        and spectrum_pass
        and synthetic_estimator_pass
        and load_proxy_pass
        and load_gamma_pass
    )
    result["gate_summary"] = {
        "provisional_lower_bound_gate": {
            "passed": bool(result["provisional_gate"]["passed"]),
            "scope": result["provisional_gate"]["scope"],
        },
        "single_axis_minimal_gate": {"passed": single_pass, "scope": "six single-axis pose fits"},
        "canonical_gamma_minimal_gate": {"passed": gamma_pass, "scope": "all requested target Gamma candidates"},
        "authored_spectrum_line_gate": {"passed": spectrum_pass, "scope": "every active authored line"},
        "spectrum_estimator_gate": {
            "passed": synthetic_estimator_pass,
            "scope": "pre-driver synthetic recovery with resolution/conditioning rule",
        },
        "gamma_load_gate": {
            "passed": load_gamma_pass,
            "scope": "all target Gamma x required load records",
        },
        "worktable_proxy_gate": {
            "passed": load_proxy_pass,
            "scope": "32 kg/explicit inertia proxy including Panda/base deck-side effect",
        },
        "overall_phase02r_status": "complete" if overall_pass else "partial_or_blocked",
        "blocked_reason": (
            None
            if overall_pass
            else "one or more pre-registered six-axis, estimator, Gamma-load, or deck-side proxy gates are blocked"
        ),
    }
    result["provenance"]["status"] = result["gate_summary"]["overall_phase02r_status"] + "; derived-not-frozen"
    result["phase03_handoff"] = "PASS" if overall_pass else "BLOCKED"
    return result


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


if __name__ == "__main__":
    raise SystemExit(main())
