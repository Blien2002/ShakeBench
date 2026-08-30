"""Auditable Gamma and authored-spectrum calibration for ShakeBench.

Calibration is performed against the authored command, before any simulator
or isolator response is involved.  A unit-level replay is used to calculate a
dimensionless peak factor, so a requested ``level_scale`` can be reproduced
from the same seed, common ``t0``, active-axis mask, and time window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .shakebench_config import ShakeBenchConfig, config_hash
from .shakebench_excitation import (
    AXES,
    ExcitationConfig,
    ExcitationError,
    ExcitationProgram,
    MotionSample,
    build_excitation_program,
)


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


def _axis_statistics(motion: MotionSample) -> tuple[dict[str, float], dict[str, float]]:
    values = motion.qdd
    rms = np.sqrt(np.mean(values**2, axis=0))
    peak = np.max(np.abs(values), axis=0)
    return (
        {axis: float(rms[index]) for index, axis in enumerate(AXES)},
        {axis: float(peak[index]) for index, axis in enumerate(AXES)},
    )


@dataclass(frozen=True)
class GammaCalibration:
    """Serializable result of an authored-command Gamma calibration.

    Attributes:
        seed: Program replay seed.
        t0: Common program phase origin.
        level_scale: Scale used for the authored command.
        gravity_m_s2: Gravity used for Gamma normalization.
        point_offset_m: Configured workpiece point relative to deck origin.
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

    @property
    def peak_factor(self) -> float:
        """Return the unit-replay peak factor in Gamma units."""

        return self.unit_peak_factor

    @property
    def Gamma_commanded(self) -> float:  # noqa: N802 - protocol spelling
        """Return the protocol-spelled authored Gamma scalar."""

        return self.gamma_commanded

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration metrics and the complete unit replay."""

        return {
            "seed": self.seed,
            "t0": self.t0,
            "level_scale": self.level_scale,
            "gravity_m_s2": self.gravity_m_s2,
            "point_offset_m": list(self.point_offset_m),
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "peak_time_s": self.peak_time_s,
            "unit_peak_acceleration_m_s2": self.unit_peak_acceleration_m_s2,
            "peak_acceleration_m_s2": self.peak_acceleration_m_s2,
            "unit_peak_factor": self.unit_peak_factor,
            "peak_factor": self.peak_factor,
            "gamma_commanded": self.gamma_commanded,
            "Gamma_commanded": self.gamma_commanded,
            "per_axis_rms": dict(self.per_axis_rms),
            "per_axis_peak": dict(self.per_axis_peak),
            "unit_replay": dict(self.unit_replay),
            "include_centripetal": self.include_centripetal,
        }


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
    include_centripetal: bool = False,
) -> GammaCalibration:
    """Calibrate ``Gamma_commanded`` from an authored deck command.

    If ``program`` is supplied, all seed/t0/level/active-axis/config values
    come from it.  Otherwise the explicit keyword arguments form the replay
    identity.  The unit replay is generated with the same identity and
    ``level_scale=1``.
    """

    if program is not None and not isinstance(program, ExcitationProgram):
        raise CalibrationError("program must be an ExcitationProgram")
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

    cfg = authored_program.config if config is None or program is not None else config
    if not isinstance(cfg, ExcitationConfig):
        # This branch is only reachable for a mapping passed alongside a
        # program, which is intentionally not allowed to alter the replay.
        cfg = authored_program.config
    times = _time_grid(
        cfg,
        time,
        duration_s=duration_s,
        sample_count=sample_count,
    )
    point = _point_offset(cfg, point_offset_m)
    unit_program = build_excitation_program(
        seed=authored_program.seed,
        t0=authored_program.t0,
        level_scale=1.0,
        active_axes=authored_program.active_axes,
        config=authored_program.config,
    )
    authored_motion = authored_program.evaluate(times)
    unit_motion = unit_program.evaluate(times)
    authored_vertical = authored_point_vertical_acceleration(
        authored_motion,
        point,
        include_centripetal=include_centripetal,
    )
    unit_vertical = authored_point_vertical_acceleration(
        unit_motion,
        point,
        include_centripetal=include_centripetal,
    )
    gravity = cfg.gravity_m_s2
    unit_peak_index = int(np.argmax(np.abs(unit_vertical)))
    authored_peak_index = int(np.argmax(np.abs(authored_vertical)))
    unit_peak = float(np.max(np.abs(unit_vertical)))
    authored_peak = float(np.max(np.abs(authored_vertical)))
    axis_rms, axis_peak = _axis_statistics(authored_motion)
    excitation_config_hash = config_hash(ShakeBenchConfig(options={"excitation": authored_program.config.to_dict()}))
    unit_replay = {
        "schema_id": "shakebench.excitation.unit_replay",
        "schema_version": 1,
        "seed": unit_program.seed,
        "t0": unit_program.t0,
        "level_scale": 1.0,
        "active_axes": list(unit_program.active_axes),
        "point_offset_m": list(point),
        "sample_count": int(times.size),
        "time_grid_s": times.tolist(),
        "config_hash": excitation_config_hash,
        "config": authored_program.config.to_dict(),
        "include_centripetal": bool(include_centripetal),
        "program": unit_program.to_dict(),
        "unit_peak_time_s": float(times[unit_peak_index]),
    }
    return GammaCalibration(
        seed=authored_program.seed,
        t0=authored_program.t0,
        level_scale=authored_program.level_scale,
        gravity_m_s2=gravity,
        point_offset_m=tuple(float(value) for value in point),
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
    )


def gamma_commanded(
    program: ExcitationProgram,
    time: Any | None = None,
    **kwargs: Any,
) -> float:
    """Return the authored ``Gamma_commanded`` scalar for a program."""

    return calibrate_gamma(program, time=time, **kwargs).gamma_commanded


def peak_factor(
    program: ExcitationProgram,
    time: Any | None = None,
    **kwargs: Any,
) -> float:
    """Return the unit-level authored peak factor in Gamma units."""

    return calibrate_gamma(program, time=time, **kwargs).unit_peak_factor


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
    )
    if unit.unit_peak_factor <= 0.0:
        if requested == 0.0:
            return 0.0
        raise CalibrationError("unit authored peak factor is zero")
    return requested / unit.unit_peak_factor


calibrate_gamma_command = calibrate_gamma
compute_gamma = gamma_commanded


__all__ = [
    "CalibrationError",
    "GammaCalibration",
    "authored_point_vertical_acceleration",
    "calibrate_gamma",
    "calibrate_gamma_command",
    "compute_gamma",
    "gamma_commanded",
    "level_scale_for_gamma",
    "peak_factor",
    "workpiece_point_acceleration",
]
