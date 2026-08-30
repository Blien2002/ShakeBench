"""Fail-closed safety gates for the authored ShakeBench excitation.

These checks are simulator-independent input gates.  They intentionally use
conservative analytic displacement and carrier-velocity bounds where
possible, and sampled analytic motion only for the ramped response and
workpiece-point Gamma.  No gate changes a program or silently clips it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .shakebench_calibration import CalibrationError, calibrate_gamma
from .shakebench_excitation import (
    AXES,
    CONSERVATIVE_MAX_LINE_FREQUENCY_HZ,
    ExcitationConfig,
    ExcitationError,
    ExcitationProgram,
)


class SafetyError(ValueError):
    """Base error for invalid safety requests."""


class SafetyViolation(SafetyError):
    """Raised when a fail-closed safety gate rejects a program.

    Attributes:
        report: The complete gate report, when one was available.
    """

    def __init__(self, message: str, report: "SafetyReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SafetyLimits:
    """Default v0 safety limits from the design contract.

    Attributes:
        max_displacement_m: Strict translational displacement limit.
        max_frequency_hz: Strict authored line-frequency limit.
        max_gamma: Strict non-ballistic Gamma limit.
        solver_travel_m: Maximum travel allowed in one solver step.
        min_samples_per_cycle: Required frequency samples per cycle.
        max_angle_rad: Optional rotational coordinate limit.
    """

    max_displacement_m: float = 0.025
    max_frequency_hz: float = CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
    max_gamma: float = 1.0
    solver_travel_m: float = 0.008
    min_samples_per_cycle: int = 20
    max_angle_rad: float | None = None

    def __post_init__(self) -> None:
        values = {
            "max_displacement_m": self.max_displacement_m,
            "max_frequency_hz": self.max_frequency_hz,
            "max_gamma": self.max_gamma,
            "solver_travel_m": self.solver_travel_m,
        }
        for name, value in values.items():
            if isinstance(value, (bool, np.bool_)):
                raise SafetyError(f"{name} must be a non-negative real number")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise SafetyError(f"{name} must be a non-negative real number") from exc
            if not np.isfinite(number) or number < 0.0:
                raise SafetyError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, number)
        if isinstance(self.min_samples_per_cycle, (bool, np.bool_)):
            raise SafetyError("min_samples_per_cycle must be a positive integer")
        if int(self.min_samples_per_cycle) != self.min_samples_per_cycle or int(self.min_samples_per_cycle) < 1:
            raise SafetyError("min_samples_per_cycle must be a positive integer")
        object.__setattr__(self, "min_samples_per_cycle", int(self.min_samples_per_cycle))
        if self.max_angle_rad is not None:
            angle = float(self.max_angle_rad)
            if not np.isfinite(angle) or angle < 0.0:
                raise SafetyError("max_angle_rad must be finite and non-negative")
            object.__setattr__(self, "max_angle_rad", angle)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configured safety limits."""

        return {
            "max_displacement_m": self.max_displacement_m,
            "max_frequency_hz": self.max_frequency_hz,
            "max_gamma": self.max_gamma,
            "solver_travel_m": self.solver_travel_m,
            "min_samples_per_cycle": self.min_samples_per_cycle,
            "max_angle_rad": self.max_angle_rad,
        }


DEFAULT_SAFETY_LIMITS = SafetyLimits()


@dataclass(frozen=True)
class SafetyCheck:
    """Result of one named safety gate.

    Attributes:
        name: Stable gate identifier.
        passed: Whether the gate accepted the program.
        measured: Primary measured value, if applicable.
        limit: Configured comparison limit, if applicable.
        message: Human-readable result description.
        details: Additional JSON-compatible diagnostics.
    """

    name: str
    passed: bool
    measured: float | None
    limit: float | None
    message: str
    details: Mapping[str, Any]

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Serialize this gate result."""

        return {
            "name": self.name,
            "passed": self.passed,
            "measured": self.measured,
            "limit": self.limit,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SafetyReport:
    """Aggregate result for all applicable safety gates.

    Attributes:
        checks: Gate results in deterministic evaluation order.
    """

    checks: tuple[SafetyCheck, ...]

    @property
    def passed(self) -> bool:
        """Return whether every included gate passed."""

        return all(check.passed for check in self.checks)

    @property
    def violations(self) -> tuple[SafetyCheck, ...]:
        """Return only the failed gate results."""

        return tuple(check for check in self.checks if not check.passed)

    @property
    def metrics(self) -> dict[str, float]:
        """Return primary measured values keyed by gate name."""

        return {check.name: check.measured for check in self.checks if check.measured is not None}

    def __bool__(self) -> bool:
        return self.passed

    def raise_if_failed(self) -> "SafetyReport":
        """Raise :class:`SafetyViolation` when any gate failed."""

        if not self.passed:
            message = "; ".join(check.message for check in self.violations)
            raise SafetyViolation(message, self)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete gate report."""

        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "violations": [check.name for check in self.violations],
            "metrics": self.metrics,
        }


def _require_program(program: ExcitationProgram) -> ExcitationProgram:
    if not isinstance(program, ExcitationProgram):
        raise SafetyError("program must be an ExcitationProgram")
    return program


def _sample_times(program: ExcitationProgram, time: Any | None, sample_count: int = 4097) -> np.ndarray:
    """Validate explicit safety times or create the default episode grid."""

    if time is not None:
        values = np.asarray(time, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise SafetyError("time must be a non-empty finite one-dimensional array")
        return values
    if isinstance(sample_count, (bool, np.bool_)) or int(sample_count) != sample_count or int(sample_count) < 2:
        raise SafetyError("sample_count must be an integer >= 2")
    return np.linspace(0.0, program.config.episode_duration_s, int(sample_count), dtype=float)


def check_displacement(
    program: ExcitationProgram,
    *,
    max_displacement_m: float = DEFAULT_SAFETY_LIMITS.max_displacement_m,
) -> SafetyCheck:
    """Reject programs whose conservative translational displacement exceeds 25 mm."""

    _require_program(program)
    limit = float(max_displacement_m)
    if not np.isfinite(limit) or limit < 0.0:
        raise SafetyError("max_displacement_m must be finite and non-negative")
    bounds = program.axis_displacement_bound[:3]
    measured = float(np.max(bounds))
    details = {axis: float(bounds[index]) for index, axis in enumerate(AXES[:3])}
    passed = measured < limit if limit > 0.0 else measured == 0.0
    message = (
        f"displacement bound {measured:.9g} m is within limit {limit:.9g} m"
        if passed
        else f"displacement bound {measured:.9g} m exceeds limit {limit:.9g} m"
    )
    return SafetyCheck("displacement", passed, measured, limit, message, details)


def check_frequency(
    program: ExcitationProgram,
    *,
    max_frequency_hz: float = DEFAULT_SAFETY_LIMITS.max_frequency_hz,
    timestep_s: float | None = None,
    min_samples_per_cycle: int = DEFAULT_SAFETY_LIMITS.min_samples_per_cycle,
) -> SafetyCheck:
    """Check the conservative line-frequency and optional timestep gates."""

    _require_program(program)
    limit = float(max_frequency_hz)
    if not np.isfinite(limit) or limit <= 0.0:
        raise SafetyError("max_frequency_hz must be finite and positive")
    if (
        isinstance(min_samples_per_cycle, (bool, np.bool_))
        or int(min_samples_per_cycle) != min_samples_per_cycle
        or int(min_samples_per_cycle) < 1
    ):
        raise SafetyError("min_samples_per_cycle must be a positive integer")
    samples = int(min_samples_per_cycle)
    measured = program.max_frequency_hz
    messages: list[str] = []
    passed = measured < limit
    if not passed:
        messages.append(f"max line frequency {measured:.9g} Hz must be < {limit:.9g} Hz")
    timestep_value: float | None = None
    timestep_limit: float | None = None
    if timestep_s is not None:
        timestep_value = float(timestep_s)
        if not np.isfinite(timestep_value) or timestep_value <= 0.0:
            raise SafetyError("timestep_s must be finite and positive")
        timestep_limit = 1.0 / (samples * measured) if measured > 0.0 else np.inf
        if timestep_value > timestep_limit:
            passed = False
            messages.append(
                f"timestep {timestep_value:.9g} s exceeds {samples} samples/cycle limit {timestep_limit:.9g} s"
            )
    if not messages:
        messages.append(f"max line frequency {measured:.9g} Hz and timestep sampling pass")
    details = {
        "max_frequency_hz": measured,
        "configured_limit_hz": limit,
        "timestep_s": timestep_value,
        "timestep_limit_s": timestep_limit,
        "min_samples_per_cycle": samples,
    }
    return SafetyCheck("frequency", passed, measured, limit, "; ".join(messages), details)


def check_non_ballistic(
    program: ExcitationProgram,
    *,
    max_gamma: float = DEFAULT_SAFETY_LIMITS.max_gamma,
    time: Any | None = None,
    sample_count: int = 4097,
    point_offset_m: Iterable[float] | None = None,
) -> SafetyCheck:
    """Reject authored workpiece-point vertical peaks at or above one gravity."""

    _require_program(program)
    limit = float(max_gamma)
    if not np.isfinite(limit) or limit < 0.0:
        raise SafetyError("max_gamma must be finite and non-negative")
    try:
        calibration = calibrate_gamma(
            program,
            time=_sample_times(program, time, sample_count),
            point_offset_m=point_offset_m,
        )
    except CalibrationError as exc:
        raise SafetyError(str(exc)) from exc
    measured = calibration.gamma_commanded
    passed = measured < limit if limit > 0.0 else measured == 0.0
    message = (
        f"Gamma_commanded {measured:.9g} is below non-ballistic limit {limit:.9g}"
        if passed
        else f"Gamma_commanded {measured:.9g} reaches/exceeds non-ballistic limit {limit:.9g}"
    )
    return SafetyCheck(
        "non_ballistic",
        passed,
        measured,
        limit,
        message,
        {
            "peak_acceleration_m_s2": calibration.peak_acceleration_m_s2,
            "gravity_m_s2": calibration.gravity_m_s2,
            "peak_time_s": calibration.peak_time_s,
            "point_offset_m": list(calibration.point_offset_m),
        },
    )


def check_solver_travel(
    program: ExcitationProgram,
    *,
    timestep_s: float,
    solver_travel_m: float = DEFAULT_SAFETY_LIMITS.solver_travel_m,
    time: Any | None = None,
    sample_count: int = 4097,
) -> SafetyCheck:
    """Check maximum translational distance travelled during one solver step."""

    _require_program(program)
    dt = float(timestep_s)
    limit = float(solver_travel_m)
    if not np.isfinite(dt) or dt <= 0.0:
        raise SafetyError("timestep_s must be finite and positive")
    if not np.isfinite(limit) or limit < 0.0:
        raise SafetyError("solver_travel_m must be finite and non-negative")
    times = _sample_times(program, time, sample_count)
    motion = program.evaluate(times)
    sampled_velocity = np.max(np.abs(motion.qdot[..., :3]), axis=-1)
    sampled_max_velocity = float(np.max(sampled_velocity))
    carrier_bound = float(np.max(program.carrier_velocity_bound[:3]))
    # The sampled ramped value is the useful measurement; the carrier bound is
    # added as a conservative fallback for unsampled phase extrema.
    max_velocity = max(sampled_max_velocity, carrier_bound)
    travel = max_velocity * dt
    passed = travel < limit if limit > 0.0 else travel == 0.0
    message = (
        f"solver travel bound {travel:.9g} m is within limit {limit:.9g} m"
        if passed
        else f"solver travel bound {travel:.9g} m exceeds limit {limit:.9g} m"
    )
    return SafetyCheck(
        "solver_travel",
        passed,
        travel,
        limit,
        message,
        {
            "timestep_s": dt,
            "max_velocity_m_s": max_velocity,
            "sampled_max_velocity_m_s": sampled_max_velocity,
            "carrier_velocity_bound_m_s": carrier_bound,
        },
    )


def check_angle(
    program: ExcitationProgram,
    *,
    max_angle_rad: float,
    time: Any | None = None,
    sample_count: int = 4097,
) -> SafetyCheck:
    """Optional rotational-angle gate used by later physics phases."""

    _require_program(program)
    limit = float(max_angle_rad)
    if not np.isfinite(limit) or limit < 0.0:
        raise SafetyError("max_angle_rad must be finite and non-negative")
    times = _sample_times(program, time, sample_count)
    measured = float(np.max(np.abs(program.evaluate(times).q[..., 3:])))
    passed = measured < limit if limit > 0.0 else measured == 0.0
    message = (
        f"angle bound {measured:.9g} rad is within limit {limit:.9g} rad"
        if passed
        else f"angle bound {measured:.9g} rad exceeds limit {limit:.9g} rad"
    )
    return SafetyCheck("angle", passed, measured, limit, message, {})


def check_safety(
    program: ExcitationProgram,
    *,
    limits: SafetyLimits = DEFAULT_SAFETY_LIMITS,
    timestep_s: float | None = None,
    time: Any | None = None,
    sample_count: int = 4097,
    point_offset_m: Iterable[float] | None = None,
) -> SafetyReport:
    """Run all applicable fail-closed gates and return an auditable report."""

    _require_program(program)
    if not isinstance(limits, SafetyLimits):
        raise SafetyError("limits must be a SafetyLimits instance")
    checks = [
        check_displacement(program, max_displacement_m=limits.max_displacement_m),
        check_frequency(
            program,
            max_frequency_hz=limits.max_frequency_hz,
            timestep_s=timestep_s,
            min_samples_per_cycle=limits.min_samples_per_cycle,
        ),
        check_non_ballistic(
            program,
            max_gamma=limits.max_gamma,
            time=time,
            sample_count=sample_count,
            point_offset_m=point_offset_m,
        ),
    ]
    if limits.max_angle_rad is not None:
        checks.append(
            check_angle(
                program,
                max_angle_rad=limits.max_angle_rad,
                time=time,
                sample_count=sample_count,
            )
        )
    if timestep_s is not None:
        checks.append(
            check_solver_travel(
                program,
                timestep_s=timestep_s,
                solver_travel_m=limits.solver_travel_m,
                time=time,
                sample_count=sample_count,
            )
        )
    return SafetyReport(tuple(checks))


def validate_safety(
    program: ExcitationProgram,
    *,
    limits: SafetyLimits = DEFAULT_SAFETY_LIMITS,
    timestep_s: float | None = None,
    time: Any | None = None,
    sample_count: int = 4097,
    point_offset_m: Iterable[float] | None = None,
) -> SafetyReport:
    """Run gates and raise :class:`SafetyViolation` on any failure."""

    return check_safety(
        program,
        limits=limits,
        timestep_s=timestep_s,
        time=time,
        sample_count=sample_count,
        point_offset_m=point_offset_m,
    ).raise_if_failed()


assert_safe = validate_safety


__all__ = [
    "DEFAULT_SAFETY_LIMITS",
    "SafetyCheck",
    "SafetyError",
    "SafetyLimits",
    "SafetyReport",
    "SafetyViolation",
    "assert_safe",
    "check_angle",
    "check_displacement",
    "check_frequency",
    "check_non_ballistic",
    "check_safety",
    "check_solver_travel",
    "validate_safety",
]
