"""Auditable Gamma and authored-spectrum calibration for ShakeBench.

Calibration is performed against the authored command, before any simulator
or isolator response is involved.  A unit-level replay is used to calculate a
dimensionless peak factor, so a requested ``level_scale`` can be reproduced
from the same seed, common ``t0``, active-axis mask, and time window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np

from .config import ShakeBenchConfig, config_hash
from .excitation import (
    AXES,
    ExcitationConfig,
    ExcitationError,
    ExcitationProgram,
    MotionSample,
    build_excitation_program,
    build_mode_program,
)

GAMMA_DEFINITIONS = ("normal_peak_v1", "magnitude_peak_v1")


class CalibrationError(ValueError):
    """Raised when a Gamma calibration request is malformed."""


def _finite_float(name: str, value: Any, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise CalibrationError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        comparator = ">" if strict else ">="
        raise CalibrationError(f"{name} must be {comparator} {minimum}")
    return result


def _time_grid(
    config: ExcitationConfig,
    time: Any | None,
    *,
    duration_s: float | None,
    sample_count: int,
) -> np.ndarray:
    """Validate or construct the monotonically ordered calibration grid."""

    if time is not None:
        values = np.asarray(time, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or values.size == 0:
            raise CalibrationError("time must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise CalibrationError("time must contain only finite values")
        if values.size > 1 and np.any(np.diff(values) < 0.0):
            raise CalibrationError("time must be monotonically non-decreasing")
        return values
    duration = (
        config.episode_duration_s
        if duration_s is None
        else _finite_float("duration_s", duration_s, minimum=0.0, strict=True)
    )
    if isinstance(sample_count, (bool, np.bool_)) or int(sample_count) != sample_count or int(sample_count) < 2:
        raise CalibrationError("sample_count must be an integer >= 2")
    return np.linspace(0.0, duration, int(sample_count), dtype=float)


def _point_offset(config: ExcitationConfig, point_offset_m: Iterable[float] | None) -> np.ndarray:
    if point_offset_m is None:
        point_offset_m = config.workpiece_point_offset_m
    point = np.asarray(tuple(point_offset_m), dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise CalibrationError("point_offset_m must contain three finite values")
    return point


def _support_normal(support_normal: Iterable[float] | None) -> np.ndarray:
    """Validate and normalize a support normal for effective acceleration."""

    if support_normal is None:
        support_normal = (0.0, 0.0, 1.0)
    try:
        normal = np.asarray(tuple(support_normal), dtype=float)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("support_normal must contain three finite values") from exc
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise CalibrationError("support_normal must contain three finite values")
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise CalibrationError("support_normal must not be the zero vector")
    return normal / norm


def workpiece_point_acceleration(
    motion: MotionSample,
    point_offset_m: Iterable[float],
    *,
    include_centripetal: bool = False,
) -> np.ndarray:
    """Calculate rigid-point acceleration from authored six-axis motion.

    The default Gamma path includes the required ``alpha x r`` term.  The
    optional centripetal term is available for diagnostic rigid-body replay;
    leaving it off preserves the linear authored Gamma scaling used by the
    v0 calibration contract.
    """

    if not isinstance(motion, MotionSample):
        raise CalibrationError("motion must be a MotionSample")
    point = np.asarray(tuple(point_offset_m), dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise CalibrationError("point_offset_m must contain three finite values")
    linear_acceleration = motion.qdd[..., :3]
    angular_acceleration = motion.qdd[..., 3:]
    point_acceleration = linear_acceleration + np.cross(angular_acceleration, point)
    if include_centripetal:
        angular_velocity = motion.qdot[..., 3:]
        point_acceleration = point_acceleration + np.cross(
            angular_velocity,
            np.cross(angular_velocity, point),
        )
    return point_acceleration


def authored_point_vertical_acceleration(
    motion: MotionSample,
    point_offset_m: Iterable[float],
    *,
    include_centripetal: bool = False,
) -> np.ndarray:
    """Return the authored workpiece-point vertical acceleration."""

    return workpiece_point_acceleration(
        motion,
        point_offset_m,
        include_centripetal=include_centripetal,
    )[..., 2]


@dataclass(frozen=True)
class GammaCalibration:
    """Serializable result of an authored-command Gamma calibration.

    Attributes:
        seed: Program replay seed.
        t0: Common program phase origin.
        level_scale: Scale used for the authored command.
        gravity_m_s2: Gravity used for Gamma normalization.
        point_offset_m: Configured workpiece point relative to deck origin.
        support_normal: Unit normal used for the effective acceleration peak.
        duration_s: Span of the evaluated time grid.
        sample_count: Number of evaluated time samples.
        peak_time_s: Time at the observed authored vertical peak.
        unit_peak_acceleration_m_s2: Unit-replay vertical peak.
        peak_acceleration_m_s2: Requested-level vertical peak.
        unit_peak_factor: Unit-replay peak divided by gravity.
        gamma_commanded: Requested-level peak divided by gravity.
        per_axis_rms: Sampled ramped acceleration RMS per axis.
        per_axis_peak: Sampled ramped acceleration peak per axis.
        unit_replay: Self-contained replay identity and unit program.
        include_centripetal: Whether the optional centripetal term was used.
    """

    seed: int
    t0: float
    level_scale: float
    gravity_m_s2: float
    point_offset_m: tuple[float, float, float]
    support_normal: tuple[float, float, float]
    duration_s: float
    sample_count: int
    peak_time_s: float
    unit_peak_acceleration_m_s2: float
    peak_acceleration_m_s2: float
    unit_peak_factor: float
    gamma_commanded: float
    per_axis_rms: Mapping[str, float]
    per_axis_peak: Mapping[str, float]
    unit_replay: Mapping[str, Any]
    include_centripetal: bool = False
    gamma_definition: str = "normal_peak_v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration metrics and the complete unit replay."""

        payload = {
            "seed": self.seed,
            "t0": self.t0,
            "level_scale": self.level_scale,
            "gravity_m_s2": self.gravity_m_s2,
            "point_offset_m": list(self.point_offset_m),
            "support_normal": list(self.support_normal),
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "peak_time_s": self.peak_time_s,
            "unit_peak_acceleration_m_s2": self.unit_peak_acceleration_m_s2,
            "peak_acceleration_m_s2": self.peak_acceleration_m_s2,
            "unit_peak_factor": self.unit_peak_factor,
            "gamma_commanded": self.gamma_commanded,
            "gamma_definition": self.gamma_definition,
            "per_axis_rms": dict(self.per_axis_rms),
            "per_axis_peak": dict(self.per_axis_peak),
            "unit_replay": dict(self.unit_replay),
            "include_centripetal": self.include_centripetal,
        }
        return payload


def calibrate_gamma(
    program: ExcitationProgram | None = None,
    *,
    seed: int = 0,
    t0: float = 0.0,
    level_scale: float = 1.0,
    active_axes: Iterable[str | int] | str | np.ndarray | None = None,
    config: ExcitationConfig | Mapping[str, Any] | None = None,
    time: Any | None = None,
    duration_s: float | None = None,
    sample_count: int = 20001,
    point_offset_m: Iterable[float] | None = None,
    support_normal: Iterable[float] | None = None,
    include_centripetal: bool = False,
    gamma_definition: str = "normal_peak_v1",
) -> GammaCalibration:
    """Calibrate ``Gamma_commanded`` from an authored deck command.

    If ``program`` is supplied, all seed/t0/level/active-axis/config values
    come from it.  Otherwise the explicit keyword arguments form the replay
    identity.  The unit replay is generated with the same identity and
    ``level_scale=1``.
    """

    if program is not None and not isinstance(program, ExcitationProgram):
        raise CalibrationError("program must be an ExcitationProgram")
    if gamma_definition not in GAMMA_DEFINITIONS:
        raise CalibrationError(f"unknown gamma_definition {gamma_definition!r}")
    if program is None:
        try:
            authored_program = build_excitation_program(
                seed=seed,
                t0=t0,
                level_scale=level_scale,
                active_axes=active_axes,
                config=config,
            )
        except ExcitationError as exc:
            raise CalibrationError(str(exc)) from exc
    else:
        authored_program = program
        seed = program.seed
        t0 = program.t0
        level_scale = program.level_scale
        config = program.config
        active_axes = program.active_axes

    cfg = authored_program.config
    times = _time_grid(
        cfg,
        time,
        duration_s=duration_s,
        sample_count=sample_count,
    )
    point = _point_offset(cfg, point_offset_m)
    normal = _support_normal(support_normal)
    if authored_program.mode_params or authored_program.mode != "multisine_v1":
        unit_program = build_mode_program(
            authored_program.mode,
            authored_program.mode_params,
            seed=authored_program.seed,
            t0=authored_program.t0,
            level_scale=1.0,
        )
    else:
        unit_program = build_excitation_program(
            seed=authored_program.seed,
            t0=authored_program.t0,
            level_scale=1.0,
            active_axes=authored_program.active_axes,
            config=authored_program.config,
        )
    authored_motion = authored_program.evaluate(times)
    unit_motion = unit_program.evaluate(times)
    authored_point = workpiece_point_acceleration(authored_motion, point, include_centripetal=include_centripetal)
    unit_point = workpiece_point_acceleration(unit_motion, point, include_centripetal=include_centripetal)
    if gamma_definition == "normal_peak_v1":
        authored_measure = np.einsum("...i,i->...", authored_point, normal)
        unit_measure = np.einsum("...i,i->...", unit_point, normal)
    else:
        authored_measure = np.linalg.norm(authored_point, axis=-1)
        unit_measure = np.linalg.norm(unit_point, axis=-1)
    gravity = cfg.gravity_m_s2
    unit_peak_index = int(np.argmax(np.abs(unit_measure)))
    authored_peak_index = int(np.argmax(np.abs(authored_measure)))
    unit_peak = float(np.max(np.abs(unit_measure)))
    authored_peak = float(np.max(np.abs(authored_measure)))
    qdd = authored_motion.qdd
    axis_rms = {axis: float(value) for axis, value in zip(AXES, np.sqrt(np.mean(qdd**2, axis=0)))}
    axis_peak = {axis: float(value) for axis, value in zip(AXES, np.max(np.abs(qdd), axis=0))}
    excitation_config_hash = config_hash(ShakeBenchConfig(options={"excitation": authored_program.config.to_dict()}))
    unit_replay = {
        "schema_id": "shakebench.excitation.unit_replay",
        "schema_version": 1,
        "seed": unit_program.seed,
        "t0": unit_program.t0,
        "level_scale": 1.0,
        "active_axes": list(unit_program.active_axes),
        "point_offset_m": list(point),
        "support_normal": list(normal),
        "sample_count": int(times.size),
        "time_grid_s": times.tolist(),
        "config_hash": excitation_config_hash,
        "config": authored_program.config.to_dict(),
        "include_centripetal": bool(include_centripetal),
        "gamma_definition": gamma_definition,
        "program": unit_program.to_dict(),
        "unit_peak_time_s": float(times[unit_peak_index]),
    }
    return GammaCalibration(
        seed=authored_program.seed,
        t0=authored_program.t0,
        level_scale=authored_program.level_scale,
        gravity_m_s2=gravity,
        point_offset_m=tuple(float(value) for value in point),
        support_normal=tuple(float(value) for value in normal),
        duration_s=float(times[-1] - times[0]),
        sample_count=int(times.size),
        peak_time_s=float(times[authored_peak_index]),
        unit_peak_acceleration_m_s2=unit_peak,
        peak_acceleration_m_s2=authored_peak,
        unit_peak_factor=unit_peak / gravity,
        gamma_commanded=authored_peak / gravity,
        per_axis_rms=axis_rms,
        per_axis_peak=axis_peak,
        unit_replay=unit_replay,
        include_centripetal=bool(include_centripetal),
        gamma_definition=gamma_definition,
    )


def level_scale_for_gamma(
    gamma: float,
    *,
    seed: int = 0,
    t0: float = 0.0,
    active_axes: Iterable[str | int] | str | np.ndarray | None = None,
    config: ExcitationConfig | Mapping[str, Any] | None = None,
    time: Any | None = None,
    duration_s: float | None = None,
    sample_count: int = 20001,
    point_offset_m: Iterable[float] | None = None,
    support_normal: Iterable[float] | None = None,
    gamma_definition: str = "normal_peak_v1",
) -> float:
    """Convert a requested linear authored Gamma to ``level_scale``."""

    requested = _finite_float("gamma", gamma, minimum=0.0)
    unit = calibrate_gamma(
        seed=seed,
        t0=t0,
        level_scale=1.0,
        active_axes=active_axes,
        config=config,
        time=time,
        duration_s=duration_s,
        sample_count=sample_count,
        point_offset_m=point_offset_m,
        support_normal=support_normal,
        gamma_definition=gamma_definition,
    )
    if unit.unit_peak_factor <= 0.0:
        if requested == 0.0:
            return 0.0
        raise CalibrationError("unit authored peak factor is zero")
    return requested / unit.unit_peak_factor


def build_vibration_program(vibration: Mapping[str, Any]) -> ExcitationProgram:
    """Parse the public vibration mapping, calibrate Gamma, and return one program."""

    if not isinstance(vibration, Mapping):
        raise CalibrationError("vibration must be an object")
    allowed = {"mode", "gamma", "seed", "t0_s", "mode_params", "gamma_definition"}
    unknown = sorted(set(vibration) - allowed)
    if unknown:
        raise CalibrationError("unknown vibration field(s): " + ", ".join(repr(item) for item in unknown))
    mode = vibration.get("mode", "multisine_v1")
    gamma = _finite_float("vibration.gamma", vibration.get("gamma", 0.0), minimum=0.0)
    gamma_definition = vibration.get("gamma_definition", "normal_peak_v1")
    if gamma_definition not in GAMMA_DEFINITIONS:
        raise CalibrationError(f"unknown gamma_definition {gamma_definition!r}")
    try:
        unit = build_mode_program(
            mode,
            vibration.get("mode_params", {}),
            seed=vibration.get("seed", 0),
            t0=vibration.get("t0_s", 0.0),
            level_scale=1.0,
        )
    except ExcitationError as exc:
        raise CalibrationError(str(exc)) from exc
    calibration = calibrate_gamma(unit, gamma_definition=gamma_definition)
    if calibration.unit_peak_factor <= 0.0 and gamma > 0.0:
        raise CalibrationError("unit vibration Gamma is zero")
    level_scale = 0.0 if gamma == 0.0 else gamma / calibration.unit_peak_factor
    try:
        program = build_mode_program(
            mode,
            vibration.get("mode_params", {}),
            seed=unit.seed,
            t0=unit.t0,
            level_scale=level_scale,
        )
    except ExcitationError as exc:
        raise CalibrationError(str(exc)) from exc
    window = {"start_s": 0.0, "duration_s": calibration.duration_s, "sample_count": calibration.sample_count}
    return replace(
        program,
        gamma_definition=gamma_definition,
        gamma_requested=gamma,
        calibration_window=window,
    )


def vibration_record(program: ExcitationProgram) -> dict[str, Any]:
    """Return the required replay identity for a configured vibration program."""

    if not isinstance(program, ExcitationProgram) or program.gamma_definition is None:
        raise CalibrationError("program was not built from a vibration config")
    payload = program.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "mode": program.mode,
        "mode_params": dict(program.mode_params),
        "mode_version": payload["mode_version"],
        "gamma_definition": program.gamma_definition,
        "gamma_requested": program.gamma_requested,
        "level_scale": program.level_scale,
        "calibration_window": dict(program.calibration_window or {}),
        "excitation_seed": program.seed,
        "t0_s": program.t0,
        "program_hash": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = [
    "CalibrationError",
    "GAMMA_DEFINITIONS",
    "GammaCalibration",
    "authored_point_vertical_acceleration",
    "build_vibration_program",
    "calibrate_gamma",
    "level_scale_for_gamma",
    "workpiece_point_acceleration",
    "vibration_record",
]
