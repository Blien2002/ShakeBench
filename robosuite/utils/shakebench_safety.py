"""Fail-closed six-degree-of-freedom safety gates for authored excitation.

The checks in this module are command-level preflight gates.  They do not
claim to know the realized deck/table state that only a later dynamic-deck
phase can measure.  Geometry, units, and conservative formulas are serialized
as a candidate safety profile so a future physics phase can freeze or replace
them with evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .shakebench_calibration import CalibrationError, calibrate_gamma
from .shakebench_excitation import (
    AXES,
    CONSERVATIVE_MAX_LINE_FREQUENCY_HZ,
    ExcitationProgram,
)

SAFETY_PROFILE_ID = "shakebench.safety.candidate.v1"
SAFETY_PROFILE_VERSION = 1


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


def _finite_vector(value: Iterable[float], name: str) -> tuple[float, float, float]:
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"{name} must contain three finite numbers") from exc
    if len(vector) != 3 or not np.all(np.isfinite(vector)):
        raise SafetyError(f"{name} must contain three finite numbers")
    return vector


def _abs_cross_bound(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a componentwise upper bound for ``abs(cross(first, second))``."""

    return np.array(
        [
            first[1] * second[2] + first[2] * second[1],
            first[2] * second[0] + first[0] * second[2],
            first[0] * second[1] + first[1] * second[0],
        ],
        dtype=float,
    )


def _abs_centripetal_bound(angular_velocity: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Bound ``abs(omega x (omega x r))`` from componentwise magnitudes."""

    omega_squared = float(np.sum(angular_velocity**2))
    dot_bound = float(np.sum(angular_velocity * np.abs(offset)))
    return angular_velocity * dot_bound + np.abs(offset) * omega_squared


@dataclass(frozen=True)
class SafetyGeometry:
    """Named contact/clearance features used by six-DoF safety gates.

    Attributes:
        feature_offsets_m: Feature points relative to the authored deck origin.
            Rotation-induced motion is computed as ``alpha x r`` and bounded
            componentwise for preflight.
        workpiece_support_normal: Unit normal used by the non-ballistic gate.
    """

    feature_offsets_m: Mapping[str, Iterable[float]] = field(
        default_factory=lambda: {
            "table_corner_ne": (0.325, 0.30, 0.0),
            "table_corner_nw": (-0.325, 0.30, 0.0),
            "target_wall_top_edge": (-0.01, 0.25, 0.035),
            "workpiece_support_point": (0.65, 0.0, 0.0),
        }
    )
    workpiece_support_normal: Iterable[float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.feature_offsets_m, Mapping) or not self.feature_offsets_m:
            raise SafetyError("feature_offsets_m must be a non-empty mapping")
        frozen_features = {}
        for name, offset in self.feature_offsets_m.items():
            if not isinstance(name, str) or not name.strip():
                raise SafetyError("feature names must be non-empty strings")
            frozen_features[name] = _finite_vector(offset, f"feature_offsets_m[{name!r}]")
        normal = np.asarray(_finite_vector(self.workpiece_support_normal, "workpiece_support_normal"), dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise SafetyError("workpiece_support_normal must not be the zero vector")
        normal = tuple(float(item) for item in normal / norm)
        object.__setattr__(self, "feature_offsets_m", MappingProxyType(frozen_features))
        object.__setattr__(self, "workpiece_support_normal", normal)

    @property
    def workpiece_support_point_m(self) -> tuple[float, float, float]:
        """Return the named workpiece support point."""

        if "workpiece_support_point" not in self.feature_offsets_m:
            raise SafetyError("feature_offsets_m must define workpiece_support_point")
        return self.feature_offsets_m["workpiece_support_point"]

    def to_dict(self) -> dict[str, Any]:
        """Serialize feature offsets and the normalized support normal."""

        return {
            "feature_offsets_m": {name: list(offset) for name, offset in self.feature_offsets_m.items()},
            "workpiece_support_normal": list(self.workpiece_support_normal),
        }


@dataclass(frozen=True)
class SafetyLimits:
    """Candidate safety limits and geometry for authored-command preflight.

    Attributes:
        max_displacement_m: Strict maximum for translation and feature travel.
        max_frequency_hz: Strict authored line-frequency limit.
        max_gamma: Strict non-ballistic effective-normal Gamma limit.
        solver_travel_m: Geometric clearance envelope; it is not automatically
            the allowed per-step displacement.
        solver_step_fraction: Fraction of clearance allowed in one solver step.
        min_samples_per_cycle: Required frequency samples per cycle.
        max_angle_rad: Optional rotational coordinate limit.
        geometry: Named table/wall/workpiece features used by the gates.
    """

    max_displacement_m: float = 0.025
    max_frequency_hz: float = CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
    max_gamma: float = 1.0
    solver_travel_m: float = 0.008
    solver_step_fraction: float = 0.25
    min_samples_per_cycle: int = 20
    max_angle_rad: float | None = None
    geometry: SafetyGeometry = field(default_factory=SafetyGeometry)

    def __post_init__(self) -> None:
        for name in (
            "max_displacement_m",
            "max_frequency_hz",
            "max_gamma",
            "solver_travel_m",
            "solver_step_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise SafetyError(f"{name} must be a finite non-negative number")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise SafetyError(f"{name} must be a finite non-negative number") from exc
            if not np.isfinite(number) or number < 0.0:
                raise SafetyError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, number)
        if self.max_frequency_hz <= 0.0:
            raise SafetyError("max_frequency_hz must be positive")
        if self.solver_travel_m <= 0.0:
            raise SafetyError("solver_travel_m must be positive")
        if not 0.0 < self.solver_step_fraction <= 1.0:
            raise SafetyError("solver_step_fraction must be in (0, 1]")
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
        if not isinstance(self.geometry, SafetyGeometry):
            raise SafetyError("geometry must be a SafetyGeometry instance")

    @property
    def solver_clearance_m(self) -> float:
        """Return the geometric clearance envelope in meters."""

        return self.solver_travel_m

    @property
    def allowed_solver_step_m(self) -> float:
        """Return the fraction of clearance allowed per solver step."""

        return self.solver_clearance_m * self.solver_step_fraction

    def to_dict(self) -> dict[str, Any]:
        """Serialize limits, formulas' inputs, and named geometry."""

        return {
            "max_displacement_m": self.max_displacement_m,
            "max_frequency_hz": self.max_frequency_hz,
            "max_gamma": self.max_gamma,
            "solver_clearance_m": self.solver_clearance_m,
            "solver_step_fraction": self.solver_step_fraction,
            "allowed_solver_step_m": self.allowed_solver_step_m,
            "min_samples_per_cycle": self.min_samples_per_cycle,
            "max_angle_rad": self.max_angle_rad,
            "geometry": self.geometry.to_dict(),
        }


DEFAULT_SAFETY_LIMITS = SafetyLimits()


def safety_profile_payload(limits: SafetyLimits = DEFAULT_SAFETY_LIMITS) -> dict[str, Any]:
    """Return the candidate safety profile payload used for hashing."""

    if not isinstance(limits, SafetyLimits):
        raise SafetyError("limits must be a SafetyLimits instance")
    return {
        "profile_id": SAFETY_PROFILE_ID,
        "profile_version": SAFETY_PROFILE_VERSION,
        "limits": limits.to_dict(),
    }


def safety_profile_hash(limits: SafetyLimits = DEFAULT_SAFETY_LIMITS) -> str:
    """Return the deterministic hash of the candidate safety profile."""

    encoded = json.dumps(
        safety_profile_payload(limits),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _bound_motion_components(program: ExcitationProgram) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative position, velocity, and acceleration component bounds."""

    position_bound = np.asarray(program.axis_displacement_bound, dtype=float)
    carrier_velocity_bound = np.asarray(program.carrier_velocity_bound, dtype=float)
    carrier_acceleration_bound = np.sum(np.abs(program.line_accel_amplitude), axis=1)
    duration = program.config.ramp_duration_s
    if duration == 0.0:
        ramp_first_bound = 0.0
        ramp_second_bound = 0.0
    else:
        # Safe coefficient-sum bounds for the quintic smoothstep derivatives.
        ramp_first_bound = 120.0 / duration
        ramp_second_bound = 360.0 / duration**2
    velocity_bound = carrier_velocity_bound + position_bound * ramp_first_bound
    acceleration_bound = (
        carrier_acceleration_bound
        + 2.0 * carrier_velocity_bound * ramp_first_bound
        + position_bound * ramp_second_bound
    )
    return position_bound, velocity_bound, acceleration_bound, carrier_velocity_bound


def _point_displacement_bound(position_bound: np.ndarray, offset: np.ndarray) -> np.ndarray:
    translation = position_bound[:3]
    rotation = position_bound[3:]
    rotation_term = _abs_cross_bound(rotation, np.abs(offset))
    return translation + rotation_term


def _point_velocity_bound(velocity_bound: np.ndarray, offset: np.ndarray) -> np.ndarray:
    translation = velocity_bound[:3]
    rotation = velocity_bound[3:]
    return translation + _abs_cross_bound(rotation, np.abs(offset))


def _point_acceleration_bound(
    acceleration_bound: np.ndarray, velocity_bound: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    translation = acceleration_bound[:3]
    rotation_acceleration = acceleration_bound[3:]
    rotation_velocity = velocity_bound[3:]
    return (
        translation
        + _abs_cross_bound(rotation_acceleration, np.abs(offset))
        + _abs_centripetal_bound(rotation_velocity, offset)
    )


def _geometry_or_default(geometry: SafetyGeometry | None) -> SafetyGeometry:
    if geometry is None:
        return DEFAULT_SAFETY_LIMITS.geometry
    if not isinstance(geometry, SafetyGeometry):
        raise SafetyError("geometry must be a SafetyGeometry instance")
    return geometry


def check_displacement(
    program: ExcitationProgram,
    *,
    max_displacement_m: float = DEFAULT_SAFETY_LIMITS.max_displacement_m,
    geometry: SafetyGeometry | None = None,
) -> SafetyCheck:
    """Check translation norm and named rotation-induced feature travel."""

    _require_program(program)
    limit = float(max_displacement_m)
    if not np.isfinite(limit) or limit <= 0.0:
        raise SafetyError("max_displacement_m must be finite and positive")
    geometry = _geometry_or_default(geometry)
    position_bound, _, _, _ = _bound_motion_components(program)
    translation_components = position_bound[:3]
    translation_norm = float(np.linalg.norm(translation_components))
    feature_bounds = {}
    for name, raw_offset in geometry.feature_offsets_m.items():
        offset = np.asarray(raw_offset, dtype=float)
        feature_bounds[name] = float(np.linalg.norm(_point_displacement_bound(position_bound, offset)))
    measured = max([translation_norm, *feature_bounds.values()])
    passed = measured < limit
    message = (
        f"six-DoF displacement bound {measured:.9g} m is within limit {limit:.9g} m"
        if passed
        else f"six-DoF displacement bound {measured:.9g} m exceeds limit {limit:.9g} m"
    )
    return SafetyCheck(
        "displacement",
        passed,
        measured,
        limit,
        message,
        {
            "translation_component_bound_m": translation_components.tolist(),
            "translation_vector_norm_bound_m": translation_norm,
            "feature_bounds_m": feature_bounds,
            "feature_offsets_m": geometry.to_dict()["feature_offsets_m"],
            "formula": "||translation||_2 and ||translation + rotation x offset||_2 componentwise bounds",
        },
    )


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
    messages = []
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
    return SafetyCheck(
        "frequency",
        passed,
        measured,
        limit,
        "; ".join(messages),
        {
            "max_frequency_hz": measured,
            "configured_limit_hz": limit,
            "timestep_s": timestep_value,
            "timestep_limit_s": timestep_limit,
            "min_samples_per_cycle": samples,
        },
    )


def check_non_ballistic(
    program: ExcitationProgram,
    *,
    max_gamma: float = DEFAULT_SAFETY_LIMITS.max_gamma,
    time: Any | None = None,
    sample_count: int = 4097,
    point_offset_m: Iterable[float] | None = None,
    support_normal: Iterable[float] | None = None,
    geometry: SafetyGeometry | None = None,
) -> SafetyCheck:
    """Check command-level effective normal acceleration before dynamics exist."""

    _require_program(program)
    limit = float(max_gamma)
    if not np.isfinite(limit) or limit < 0.0:
        raise SafetyError("max_gamma must be finite and non-negative")
    geometry = _geometry_or_default(geometry)
    if point_offset_m is None:
        point_offset_m = geometry.workpiece_support_point_m
    if support_normal is None:
        support_normal = geometry.workpiece_support_normal
    try:
        calibration = calibrate_gamma(
            program,
            time=time,
            sample_count=sample_count,
            point_offset_m=point_offset_m,
            support_normal=support_normal,
            include_centripetal=True,
        )
    except CalibrationError as exc:
        raise SafetyError(str(exc)) from exc
    measured = calibration.gamma_commanded
    passed = measured < limit if limit > 0.0 else measured == 0.0
    message = (
        f"effective-normal Gamma_commanded {measured:.9g} is below limit {limit:.9g}"
        if passed
        else f"effective-normal Gamma_commanded {measured:.9g} reaches/exceeds limit {limit:.9g}"
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
            "support_normal": list(calibration.support_normal),
            "gate_scope": "command_level_preflight; realized-state check deferred to dynamic deck",
            "includes_translation_and_alpha_cross_r": True,
            "includes_centripetal": True,
        },
    )


def check_solver_travel(
    program: ExcitationProgram,
    *,
    timestep_s: float,
    solver_travel_m: float = DEFAULT_SAFETY_LIMITS.solver_travel_m,
    solver_step_fraction: float = DEFAULT_SAFETY_LIMITS.solver_step_fraction,
    geometry: SafetyGeometry | None = None,
    time: Any | None = None,
    sample_count: int = 4097,
) -> SafetyCheck:
    """Check translation and rotation-induced feature travel per solver step."""

    _require_program(program)
    dt = float(timestep_s)
    clearance = float(solver_travel_m)
    fraction = float(solver_step_fraction)
    if not np.isfinite(dt) or dt <= 0.0:
        raise SafetyError("timestep_s must be finite and positive")
    if not np.isfinite(clearance) or clearance <= 0.0:
        raise SafetyError("solver_travel_m/clearance must be finite and positive")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise SafetyError("solver_step_fraction must be in (0, 1]")
    geometry = _geometry_or_default(geometry)
    position_bound, velocity_bound, acceleration_bound, _ = _bound_motion_components(program)
    translation_velocity = velocity_bound[:3]
    translation_acceleration = acceleration_bound[:3]
    translation_speed = float(np.linalg.norm(translation_velocity))
    translation_accel = float(np.linalg.norm(translation_acceleration))
    translation_step = translation_speed * dt + 0.5 * translation_accel * dt**2
    feature_speed_bounds = {}
    feature_accel_bounds = {}
    feature_step_bounds = {}
    for name, raw_offset in geometry.feature_offsets_m.items():
        offset = np.asarray(raw_offset, dtype=float)
        feature_speed = float(np.linalg.norm(_point_velocity_bound(velocity_bound, offset)))
        feature_accel = float(np.linalg.norm(_point_acceleration_bound(acceleration_bound, velocity_bound, offset)))
        feature_speed_bounds[name] = feature_speed
        feature_accel_bounds[name] = feature_accel
        feature_step_bounds[name] = feature_speed * dt + 0.5 * feature_accel * dt**2
    required_step = max([translation_step, *feature_step_bounds.values()])
    allowed_step = clearance * fraction
    passed = required_step < allowed_step
    message = (
        f"required feature travel {required_step:.9g} m is within allowed {allowed_step:.9g} m"
        if passed
        else f"required feature travel {required_step:.9g} m exceeds allowed {allowed_step:.9g} m"
    )
    return SafetyCheck(
        "solver_travel",
        passed,
        required_step,
        allowed_step,
        message,
        {
            "timestep_s": dt,
            "clearance_m": clearance,
            "solver_step_fraction": fraction,
            "allowed_step_displacement_m": allowed_step,
            "translation_speed_bound_m_s": translation_speed,
            "translation_acceleration_bound_m_s2": translation_accel,
            "translation_step_displacement_m": translation_step,
            "feature_speed_bound_m_s": max(feature_speed_bounds.values()),
            "feature_acceleration_bound_m_s2": max(feature_accel_bounds.values()),
            "feature_step_displacement_m": max(feature_step_bounds.values()),
            "feature_speed_bounds_m_s": feature_speed_bounds,
            "feature_acceleration_bounds_m_s2": feature_accel_bounds,
            "feature_step_displacement_bounds_m": feature_step_bounds,
            "formula": "speed_bound*dt + 0.5*acceleration_bound*dt^2; max over translation and named features",
            "time_grid_argument_used": time is not None,
            "sample_count_argument": sample_count,
            "clearance_source": "candidate geometric/contact-feature envelope; not equal to wall thickness",
        },
    )


def check_angle(
    program: ExcitationProgram,
    *,
    max_angle_rad: float,
    time: Any | None = None,
    sample_count: int = 4097,
) -> SafetyCheck:
    """Check sampled authored rotational coordinate magnitude."""

    _require_program(program)
    limit = float(max_angle_rad)
    if not np.isfinite(limit) or limit < 0.0:
        raise SafetyError("max_angle_rad must be finite and non-negative")
    if time is None:
        times = np.linspace(0.0, program.config.episode_duration_s, sample_count)
    else:
        times = np.asarray(time, dtype=float)
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
    """Run all applicable candidate preflight gates and return a report."""

    _require_program(program)
    if not isinstance(limits, SafetyLimits):
        raise SafetyError("limits must be a SafetyLimits instance")
    checks = [
        check_displacement(
            program,
            max_displacement_m=limits.max_displacement_m,
            geometry=limits.geometry,
        ),
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
            geometry=limits.geometry,
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
                solver_step_fraction=limits.solver_step_fraction,
                geometry=limits.geometry,
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
    "SAFETY_PROFILE_ID",
    "SAFETY_PROFILE_VERSION",
    "SafetyCheck",
    "SafetyError",
    "SafetyGeometry",
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
    "safety_profile_hash",
    "safety_profile_payload",
    "validate_safety",
]
