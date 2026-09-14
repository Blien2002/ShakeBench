"""Physics-only probes for the Phase 02 dynamic deck driver.

The script intentionally builds a tiny in-memory MJCF fixture.  It exercises
the real :class:`robosuite.environments.base.MujocoEnv` split-step loop and
the role-based XML processor, but does not import a task, arena, Can, or
isolator.  The JSON output keeps provisional solver values explicit; Phase 06
is responsible for freezing an official profile.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
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
    audit_compiled_deck_model,
    audit_deck_xml,
    command_to_world_state,
    quat_to_rotation_vector,
    _quat_inverse_wxyz,
    _quat_multiply_wxyz,
)
from robosuite.utils.shakebench_excitation import AXES
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




def _trace_summary(*args, **kwargs):
    def summarize(trace: DeckDriverTrace) -> dict[str, Any]:
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


    return summarize(*args, **kwargs)
