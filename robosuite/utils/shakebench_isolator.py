"""Canonical linear six-degree-of-freedom ShakeBench isolator.

This module owns the small, explicit support model used by Phase 03.  The
model is intentionally a lumped base-excitation model: three translational
and three rotational coordinates share the worktable COM / elastic centre,
and each coordinate has its own natural frequency and damping ratio.

The module does not choose an official operating point.  The defaults are a
provisional probe profile only; callers can sweep ``f_n`` and ``zeta`` and
the resulting ``k`` / ``c`` values are derived afresh for each candidate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
TRANSLATION_AXES = AXES[:3]
ROTATION_AXES = AXES[3:]

# These are canonical geometry / inertial facts, not a selected isolator
# operating point.  The inertia is the explicitly contracted rounded value
# from the design documents.
CANONICAL_WORKTABLE_DIMENSIONS_M = (0.65, 0.60, 0.06)
CANONICAL_WORKTABLE_MASS_KG = 32.0
CANONICAL_WORKTABLE_INERTIA_KG_M2 = (0.9696, 1.1363, 2.0867)

# A useful, reproducible probe point retained from the Phase 02 / spike
# evidence.  Phase 06 must still select or replace it using physics-only
# transfer-envelope evidence.
DEFAULT_NATURAL_FREQUENCY_HZ = (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
DEFAULT_DAMPING_RATIO = (0.10, 0.10, 0.10, 0.10, 0.10, 0.10)
DEFAULT_GRAVITY_M_S2 = 9.81
DEFAULT_TRAVEL_LIMITS_M = (0.025, 0.025, 0.025)
DEFAULT_ANGLE_LIMITS_RAD = tuple(float(value) for value in np.deg2rad((5.0, 5.0, 5.0)))

# Compatibility aliases make the canonical vocabulary easy to discover from
# both the design-tree and worktable terminology.
CANONICAL_TABLETOP_DIMENSIONS_M = CANONICAL_WORKTABLE_DIMENSIONS_M
CANONICAL_TABLE_MASS_KG = CANONICAL_WORKTABLE_MASS_KG
CANONICAL_TABLE_INERTIA_KG_M2 = CANONICAL_WORKTABLE_INERTIA_KG_M2
DEFAULT_FN_HZ = DEFAULT_NATURAL_FREQUENCY_HZ
DEFAULT_ZETA = DEFAULT_DAMPING_RATIO

PHASE03_TRANSFER_SCHEMA_ID = "shakebench.phase03r.transfer"
PHASE03_TRANSFER_SCHEMA_VERSION = 1
HARMONIC_REGION_RATIOS = {
    "tracking": 0.2,
    "resonance": 1.0,
    "isolation": 2.0,
}
# Ten cycles is a registered discard window.  It removes the start-up
# transient in the 2x natural-frequency check while retaining a common
# convention across tracking, resonance, and isolation regions.
HARMONIC_PROBE_TRANSIENT_CYCLES = 10
HARMONIC_PROBE_FIT_CYCLES = 20
HARMONIC_PROBE_TRANSLATION_AMPLITUDE_M = 1.0e-4
HARMONIC_PROBE_ROTATION_AMPLITUDE_RAD = 1.0e-3
PHASE03_PROBE_SAMPLE_STRIDE = 16
PHASE03_TRANSFER_THRESHOLDS = {
    "amplitude_relative_error_max": 0.05,
    "phase_absolute_error_deg_max": 3.0,
    "fit_normalized_residual_max": 0.02,
    "cross_axis_leakage_relative_max": 0.02,
    "payload_relative_error_max": 0.25,
    "payload_absolute_offset_tolerance": 5.0e-5,
}


class IsolatorError(ValueError):
    """Base error for invalid isolator configuration or probe input."""


class IsolatorConfigurationError(IsolatorError):
    """Raised when an isolator parameter cannot be compiled safely."""


class IsolatorSafetyError(IsolatorError):
    """Raised when a relative travel / angle safety gate rejects a candidate."""

    def __init__(self, message: str, report: Optional["IsolatorSafetyReport"] = None):
        super().__init__(message)
        self.report = report


def _coerce_values(
    name: str,
    value: Any,
    length: int,
    *,
    allow_scalar: bool = False,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    """Normalize a scalar or fixed-length numeric vector to a tuple."""

    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise IsolatorConfigurationError(f"{name} must contain finite numeric values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise IsolatorConfigurationError(f"{name} must contain finite numeric values") from exc
    if array.ndim == 0:
        if not allow_scalar:
            raise IsolatorConfigurationError(f"{name} must contain exactly {length} values")
        array = np.full(length, float(array), dtype=float)
    else:
        array = array.reshape(-1)
        if array.size != length:
            raise IsolatorConfigurationError(f"{name} must contain exactly {length} values")
    if not np.all(np.isfinite(array)):
        raise IsolatorConfigurationError(f"{name} must contain finite numeric values")
    if positive and np.any(array <= 0.0):
        raise IsolatorConfigurationError(f"{name} must contain strictly positive values")
    if nonnegative and np.any(array < 0.0):
        raise IsolatorConfigurationError(f"{name} must contain non-negative values")
    return tuple(float(item) for item in array)


def _coerce_scalar(name: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise IsolatorConfigurationError(f"{name} must be a finite numeric scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IsolatorConfigurationError(f"{name} must be a finite numeric scalar") from exc
    if not math.isfinite(number):
        raise IsolatorConfigurationError(f"{name} must be a finite numeric scalar")
    if positive and number <= 0.0:
        raise IsolatorConfigurationError(f"{name} must be strictly positive")
    if nonnegative and number < 0.0:
        raise IsolatorConfigurationError(f"{name} must be non-negative")
    return number


def _coerce_config(config: Any = None) -> "IsolatorConfig":
    if config is None:
        return IsolatorConfig()
    if isinstance(config, IsolatorConfig):
        return config
    if isinstance(config, IsolatorParameters):
        return _config_from_parameters(config)
    if isinstance(config, Mapping):
        payload = dict(config)
        aliases = {
            "natural_frequency_hz": "fn_hz",
            "damping_ratio": "zeta",
            "table_mass_kg": "mass_kg",
            "table_inertia_kg_m2": "inertia_kg_m2",
            "travel_limit_m": "travel_limits_m",
            "angle_limit_rad": "angle_limits_rad",
            "gravity": "gravity_m_s2",
            "f_n_hz": "fn_hz",
            "travel_limits": "travel_limits_m",
            "angle_limits": "angle_limits_rad",
        }
        for source, target in aliases.items():
            if source in payload:
                if target in payload:
                    raise IsolatorConfigurationError(f"{source} and {target} specify the same field")
                payload[target] = payload.pop(source)
        # ``to_dict`` payloads carry an explicit six-vector under ``axes``;
        # axis order is fixed by this module and therefore need not be parsed.
        payload.pop("axes", None)
        payload.pop("axis_order", None)
        allowed = {
            "fn_hz",
            "zeta",
            "mass_kg",
            "inertia_kg_m2",
            "gravity_m_s2",
            "travel_limits_m",
            "angle_limits_rad",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise IsolatorConfigurationError(f"unknown isolator config field(s): {', '.join(unknown)}")
        return IsolatorConfig(**payload)
    raise IsolatorConfigurationError("config must be an IsolatorConfig or mapping")


def _config_from_parameters(parameters: "IsolatorParameters") -> "IsolatorConfig":
    """Rebuild a config view for APIs that need candidate safety limits."""

    return IsolatorConfig(
        fn_hz=parameters.fn_hz,
        zeta=parameters.zeta,
        mass_kg=parameters.mass_kg,
        inertia_kg_m2=parameters.inertia_kg_m2,
        gravity_m_s2=parameters.gravity_m_s2,
    )


@dataclass(frozen=True, init=False)
class IsolatorConfig:
    """Per-axis configuration for the canonical six-DoF support.

    ``fn_hz`` and ``zeta`` accept either one scalar (broadcast to all six
    axes) or a six-vector.  ``travel_limits_m`` covers translation and
    ``angle_limits_rad`` covers rotation.  Limits are strict safety gates and
    are deliberately kept separate from the candidate's natural frequency.
    """

    fn_hz: tuple[float, ...]
    zeta: tuple[float, ...]
    mass_kg: float
    inertia_kg_m2: tuple[float, ...]
    gravity_m_s2: float
    travel_limits_m: tuple[float, ...]
    angle_limits_rad: tuple[float, ...]

    def __init__(
        self,
        fn_hz: Any = DEFAULT_NATURAL_FREQUENCY_HZ,
        zeta: Any = DEFAULT_DAMPING_RATIO,
        mass_kg: Any = CANONICAL_WORKTABLE_MASS_KG,
        inertia_kg_m2: Any = CANONICAL_WORKTABLE_INERTIA_KG_M2,
        gravity_m_s2: Any = DEFAULT_GRAVITY_M_S2,
        travel_limits_m: Any = DEFAULT_TRAVEL_LIMITS_M,
        angle_limits_rad: Any = DEFAULT_ANGLE_LIMITS_RAD,
        *,
        natural_frequency_hz: Any = None,
        damping_ratio: Any = None,
        table_mass_kg: Any = None,
        table_inertia_kg_m2: Any = None,
        travel_limit_m: Any = None,
        angle_limit_rad: Any = None,
        travel_limits: Any = None,
        angle_limits: Any = None,
        gravity: Any = None,
    ) -> None:
        if natural_frequency_hz is not None:
            fn_hz = natural_frequency_hz
        if damping_ratio is not None:
            zeta = damping_ratio
        if table_mass_kg is not None:
            mass_kg = table_mass_kg
        if table_inertia_kg_m2 is not None:
            inertia_kg_m2 = table_inertia_kg_m2
        if travel_limit_m is not None:
            travel_limits_m = travel_limit_m
        if angle_limit_rad is not None:
            angle_limits_rad = angle_limit_rad
        if travel_limits is not None:
            travel_limits_m = travel_limits
        if angle_limits is not None:
            angle_limits_rad = angle_limits
        if gravity is not None:
            gravity_m_s2 = gravity

        object.__setattr__(self, "fn_hz", _coerce_values("fn_hz", fn_hz, 6, allow_scalar=True, positive=True))
        object.__setattr__(self, "zeta", _coerce_values("zeta", zeta, 6, allow_scalar=True, positive=True))
        object.__setattr__(self, "mass_kg", _coerce_scalar("mass_kg", mass_kg, positive=True))
        object.__setattr__(
            self,
            "inertia_kg_m2",
            _coerce_values("inertia_kg_m2", inertia_kg_m2, 3, positive=True),
        )
        object.__setattr__(
            self,
            "gravity_m_s2",
            _coerce_scalar("gravity_m_s2", gravity_m_s2, positive=True),
        )
        object.__setattr__(
            self,
            "travel_limits_m",
            _coerce_values("travel_limits_m", travel_limits_m, 3, positive=True),
        )
        object.__setattr__(
            self,
            "angle_limits_rad",
            _coerce_values("angle_limits_rad", angle_limits_rad, 3, positive=True),
        )

    @property
    def natural_frequency_hz(self) -> tuple[float, ...]:
        """Alias for :attr:`fn_hz`."""

        return self.fn_hz

    @property
    def f_n_hz(self) -> tuple[float, ...]:
        """Short alias for natural frequency."""

        return self.fn_hz

    @property
    def damping_ratio(self) -> tuple[float, ...]:
        """Alias for :attr:`zeta`."""

        return self.zeta

    @property
    def table_mass_kg(self) -> float:
        """Alias for the canonical reference worktable mass."""

        return self.mass_kg

    @property
    def table_inertia_kg_m2(self) -> tuple[float, ...]:
        """Alias for the canonical reference worktable inertia."""

        return self.inertia_kg_m2

    @property
    def limits(self) -> tuple[float, ...]:
        """Return translation limits followed by angular limits."""

        return self.travel_limits_m + self.angle_limits_rad

    @property
    def travel_limit_m(self) -> tuple[float, ...]:
        """Alias for the three translational travel limits."""

        return self.travel_limits_m

    @property
    def angle_limit_rad(self) -> tuple[float, ...]:
        """Alias for the three rotational angle limits."""

        return self.angle_limits_rad

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible immutable-parameter snapshot."""

        return {
            "axis_order": list(AXES),
            "fn_hz": list(self.fn_hz),
            "zeta": list(self.zeta),
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "gravity_m_s2": self.gravity_m_s2,
            "travel_limits_m": list(self.travel_limits_m),
            "angle_limits_rad": list(self.angle_limits_rad),
        }


DEFAULT_ISOLATOR_CONFIG = IsolatorConfig()


@dataclass(frozen=True)
class IsolatorParameters:
    """Derived per-axis support parameters.

    The first three coordinates use kg / N/m / N·s/m.  The last three use
    kg·m² / N·m/rad / N·m·s/rad.  ``springref`` is a generalized coordinate;
    only its ``tz`` element is nonzero because the reference-table preload
    compensates table weight and nothing else.
    """

    fn_hz: tuple[float, ...]
    zeta: tuple[float, ...]
    effective_mass: tuple[float, ...]
    omega_n_rad_s: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    springref: tuple[float, ...]
    mass_kg: float
    inertia_kg_m2: tuple[float, ...]
    gravity_m_s2: float

    def __post_init__(self) -> None:
        values = _coerce_values("fn_hz", self.fn_hz, 6, positive=True)
        object.__setattr__(self, "fn_hz", values)
        values = _coerce_values("zeta", self.zeta, 6, nonnegative=True)
        object.__setattr__(self, "zeta", values)
        for name in ("effective_mass", "omega_n_rad_s", "stiffness"):
            values = _coerce_values(name, getattr(self, name), 6, positive=True)
            object.__setattr__(self, name, values)
        for name in ("damping", "springref"):
            values = _coerce_values(name, getattr(self, name), 6, nonnegative=True)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "mass_kg", _coerce_scalar("mass_kg", self.mass_kg, positive=True))
        object.__setattr__(
            self,
            "inertia_kg_m2",
            _coerce_values("inertia_kg_m2", self.inertia_kg_m2, 3, positive=True),
        )
        object.__setattr__(self, "gravity_m_s2", _coerce_scalar("gravity_m_s2", self.gravity_m_s2, positive=True))

    @property
    def axes(self) -> tuple[str, ...]:
        return AXES

    @property
    def f_n_hz(self) -> tuple[float, ...]:
        """Short alias for natural frequency."""

        return self.fn_hz

    @property
    def k(self) -> tuple[float, ...]:
        """Compatibility alias for derived stiffness."""

        return self.stiffness

    @property
    def m_eff(self) -> tuple[float, ...]:
        """Compatibility alias for generalized effective mass / inertia."""

        return self.effective_mass

    @property
    def omega_n(self) -> tuple[float, ...]:
        """Compatibility alias for natural angular frequency."""

        return self.omega_n_rad_s

    @property
    def c(self) -> tuple[float, ...]:
        """Compatibility alias for derived damping."""

        return self.damping

    @property
    def spring_reference(self) -> tuple[float, ...]:
        """Compatibility alias for ``springref``."""

        return self.springref

    @property
    def springref_z(self) -> float:
        """Reference coordinate that compensates only empty-table gravity."""

        return self.springref[2]

    @property
    def effective_inertia(self) -> tuple[float, ...]:
        """Return the six generalized inertial coefficients."""

        return self.effective_mass

    @property
    def stiffness_N_per_unit(self) -> tuple[float, ...]:
        """Return stiffness with translation/rotation units implied by axis."""

        return self.stiffness

    @property
    def damping_Ns_per_unit(self) -> tuple[float, ...]:
        """Return viscous damping with translation/rotation units implied by axis."""

        return self.damping

    def to_dict(self) -> dict[str, Any]:
        """Return all derived values with units implied by their field names."""

        return {
            "axis_order": list(AXES),
            "fn_hz": list(self.fn_hz),
            "zeta": list(self.zeta),
            "effective_mass": list(self.effective_mass),
            "m_eff": list(self.effective_mass),
            "omega_n_rad_s": list(self.omega_n_rad_s),
            "stiffness": list(self.stiffness),
            "k": list(self.stiffness),
            "damping": list(self.damping),
            "c": list(self.damping),
            "springref": list(self.springref),
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "gravity_m_s2": self.gravity_m_s2,
            "static_sag_uncompensated_m": static_sag_uncompensated_m(self),
        }


def derive_isolator_parameters(
    config: Any = None,
    *,
    fn_hz: Any = None,
    zeta: Any = None,
    mass_kg: Any = None,
    inertia_kg_m2: Any = None,
    gravity_m_s2: Any = None,
    natural_frequency_hz: Any = None,
    damping_ratio: Any = None,
    table_mass_kg: Any = None,
    table_inertia_kg_m2: Any = None,
) -> IsolatorParameters:
    """Derive ``omega_n``, ``k``, ``c`` and preload from a candidate config.

    ``mass_kg`` and ``inertia_kg_m2`` are reference worktable values.  They
    are used once to derive the support; a payload never silently triggers a
    second derivation.
    """

    if natural_frequency_hz is not None:
        if fn_hz is not None:
            raise IsolatorConfigurationError("fn_hz and natural_frequency_hz specify different values")
        fn_hz = natural_frequency_hz
    if damping_ratio is not None:
        if zeta is not None:
            raise IsolatorConfigurationError("zeta and damping_ratio specify different values")
        zeta = damping_ratio
    if table_mass_kg is not None:
        if mass_kg is not None:
            raise IsolatorConfigurationError("mass_kg and table_mass_kg specify different values")
        mass_kg = table_mass_kg
    if table_inertia_kg_m2 is not None:
        if inertia_kg_m2 is not None:
            raise IsolatorConfigurationError("inertia_kg_m2 and table_inertia_kg_m2 specify different values")
        inertia_kg_m2 = table_inertia_kg_m2
    base = _coerce_config(config)
    natural_frequency = (
        base.fn_hz
        if fn_hz is None
        else _coerce_values("fn_hz", fn_hz, 6, allow_scalar=True, positive=True)
    )
    damping_ratio = base.zeta if zeta is None else _coerce_values("zeta", zeta, 6, allow_scalar=True, positive=True)
    reference_mass = base.mass_kg if mass_kg is None else _coerce_scalar("mass_kg", mass_kg, positive=True)
    reference_inertia = (
        base.inertia_kg_m2
        if inertia_kg_m2 is None
        else _coerce_values("inertia_kg_m2", inertia_kg_m2, 3, positive=True)
    )
    gravity = base.gravity_m_s2 if gravity_m_s2 is None else _coerce_scalar("gravity_m_s2", gravity_m_s2, positive=True)

    effective_mass = np.asarray((reference_mass,) * 3 + reference_inertia, dtype=float)
    omega = 2.0 * math.pi * np.asarray(natural_frequency, dtype=float)
    zeta_array = np.asarray(damping_ratio, dtype=float)
    stiffness = effective_mass * omega**2
    damping = 2.0 * zeta_array * effective_mass * omega
    springref = np.zeros(6, dtype=float)
    # Explicitly compensate only the canonical table's own weight.
    springref[2] = reference_mass * gravity / stiffness[2]
    return IsolatorParameters(
        fn_hz=tuple(natural_frequency),
        zeta=tuple(damping_ratio),
        effective_mass=tuple(float(item) for item in effective_mass),
        omega_n_rad_s=tuple(float(item) for item in omega),
        stiffness=tuple(float(item) for item in stiffness),
        damping=tuple(float(item) for item in damping),
        springref=tuple(float(item) for item in springref),
        mass_kg=reference_mass,
        inertia_kg_m2=reference_inertia,
        gravity_m_s2=gravity,
    )


def derive_k_c(config: Any = None, **kwargs: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return only the derived stiffness and damping vectors."""

    parameters = derive_isolator_parameters(config, **kwargs)
    return parameters.stiffness, parameters.damping


def preload_springref(config_or_parameters: Any = None) -> tuple[float, ...]:
    """Return the six-axis spring reference, with only ``tz`` preloaded."""

    if isinstance(config_or_parameters, IsolatorParameters):
        return config_or_parameters.springref
    return derive_isolator_parameters(config_or_parameters).springref


def vertical_preload_springref(config_or_parameters: Any = None) -> float:
    """Return ``M_ref*g/k_z`` in meters."""

    return preload_springref(config_or_parameters)[2]


def static_sag_uncompensated_m(config_or_parameters: Any = None) -> float:
    """Return the vertical sag without the reference-table preload."""

    parameters = (
        config_or_parameters
        if isinstance(config_or_parameters, IsolatorParameters)
        else derive_isolator_parameters(config_or_parameters)
    )
    return parameters.gravity_m_s2 / parameters.omega_n_rad_s[2] ** 2


def static_equilibrium_offset(
    config_or_parameters: Any = None,
    *,
    payload_mass_kg: float = 0.0,
    payload_com_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compute the additional relative equilibrium caused by a payload.

    The returned vector is in ``[tx, ty, tz, rx, ry, rz]`` coordinates.  The
    table's own weight cancels against ``springref_z``.  A payload contributes
    a gravity force at ``payload_com_m`` and therefore, when offset in x/y,
    an explicit small-angle gravity torque as well.
    """

    parameters = (
        config_or_parameters
        if isinstance(config_or_parameters, IsolatorParameters)
        else derive_isolator_parameters(config_or_parameters)
    )
    payload_mass = _coerce_scalar("payload_mass_kg", payload_mass_kg, nonnegative=True)
    com = np.asarray(tuple(payload_com_m), dtype=float)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise IsolatorError("payload_com_m must contain three finite values")
    force = np.array([0.0, 0.0, -payload_mass * parameters.gravity_m_s2], dtype=float)
    torque = np.cross(com, force)
    generalized_load = np.concatenate((force, torque))
    return generalized_load / np.asarray(parameters.stiffness, dtype=float)


def payload_static_offset(
    config_or_parameters: Any = None,
    payload_mass_kg: float = 0.0,
    payload_com_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compatibility alias for :func:`static_equilibrium_offset`."""

    return static_equilibrium_offset(
        config_or_parameters,
        payload_mass_kg=payload_mass_kg,
        payload_com_m=payload_com_m,
    )


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, (int, np.integer)) and not isinstance(axis, (bool, np.bool_)):
        index = int(axis)
        if 0 <= index < len(AXES):
            return index
    elif isinstance(axis, str) and axis in AXES:
        return AXES.index(axis)
    raise IsolatorError(f"axis must be one of {AXES} or an integer in [0, 5]")


def _frequency_array(frequency_hz: Any) -> np.ndarray:
    if isinstance(frequency_hz, (bool, np.bool_)):
        raise IsolatorError("frequency_hz must be finite and non-negative")
    try:
        frequency = np.asarray(frequency_hz, dtype=float)
    except (TypeError, ValueError) as exc:
        raise IsolatorError("frequency_hz must be finite and non-negative") from exc
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise IsolatorError("frequency_hz must be finite and non-negative")
    return frequency


def transfer_function(
    frequency_hz: Any,
    natural_frequency_hz: Any,
    damping_ratio: Any,
    *,
    output: str = "absolute",
) -> np.ndarray | complex:
    """Return the exact base-displacement transfer function.

    For ``r = f/f_n`` and ``d = 2*zeta*r``:

    ``H_abs = (1 + i*d) / (1 - r² + i*d)``
    ``H_rel = r² / (1 - r² + i*d)``

    Absolute displacement and acceleration transmissibility have the same
    magnitude for a nonzero harmonic.  ``output`` may be ``"absolute"`` or
    ``"relative"``.
    """

    frequency = _frequency_array(frequency_hz)
    try:
        natural_input = np.asarray(natural_frequency_hz, dtype=float)
        damping_input = np.asarray(damping_ratio, dtype=float)
    except (TypeError, ValueError) as exc:
        raise IsolatorError("natural_frequency_hz and damping_ratio must be numeric") from exc
    if natural_input.ndim == 0:
        natural = np.asarray(_coerce_scalar("natural_frequency_hz", natural_frequency_hz, positive=True))
    else:
        natural = np.asarray(
            _coerce_values(
                "natural_frequency_hz",
                natural_frequency_hz,
                int(natural_input.size),
                positive=True,
            ),
            dtype=float,
        )
    if damping_input.ndim == 0:
        ratio = np.asarray(_coerce_scalar("damping_ratio", damping_ratio, nonnegative=True))
    else:
        ratio = np.asarray(
            _coerce_values(
                "damping_ratio",
                damping_ratio,
                int(damping_input.size),
                nonnegative=True,
            ),
            dtype=float,
        )
    try:
        natural, ratio = np.broadcast_arrays(natural, ratio)
    except ValueError as exc:
        raise IsolatorError("natural_frequency_hz and damping_ratio are not broadcast-compatible") from exc
    if output not in {"absolute", "relative"}:
        raise IsolatorError("output must be 'absolute' or 'relative'")

    # A vector of axes is interpreted as a trailing axis, producing
    # ``frequency_shape + axis_shape`` for a frequency grid.
    if natural.ndim and frequency.ndim:
        frequency_eval = frequency.reshape(frequency.shape + (1,) * natural.ndim)
        natural_eval = natural.reshape((1,) * frequency.ndim + natural.shape)
        ratio_eval = ratio.reshape((1,) * frequency.ndim + ratio.shape)
    else:
        frequency_eval = frequency
        natural_eval = natural
        ratio_eval = ratio
    r = frequency_eval / natural_eval
    damping_term = 2.0 * ratio_eval * r
    denominator = 1.0 - r**2 + 1j * damping_term
    if output == "relative":
        value = r**2 / denominator
    else:
        value = (1.0 + 1j * damping_term) / denominator
    if np.ndim(value) == 0:
        return complex(value)
    return value


def analytic_transfer(
    frequency_hz: Any,
    natural_frequency_hz: Any,
    damping_ratio: Any,
    *,
    relative: bool = False,
) -> np.ndarray | complex:
    """Convenience wrapper around :func:`transfer_function`."""

    return transfer_function(
        frequency_hz,
        natural_frequency_hz,
        damping_ratio,
        output="relative" if relative else "absolute",
    )


def absolute_transfer_function(frequency_hz: Any, natural_frequency_hz: Any, damping_ratio: Any):
    """Return absolute/base displacement transfer."""

    return transfer_function(frequency_hz, natural_frequency_hz, damping_ratio, output="absolute")


def relative_transfer_function(frequency_hz: Any, natural_frequency_hz: Any, damping_ratio: Any):
    """Return relative displacement/base displacement transfer."""

    return transfer_function(frequency_hz, natural_frequency_hz, damping_ratio, output="relative")


def single_axis_transfer(
    axis: str | int,
    frequency_hz: float,
    config_or_parameters: Any = None,
    *,
    relative: bool = False,
) -> complex:
    """Return one axis' analytic harmonic transfer at one frequency."""

    index = _axis_index(axis)
    parameters = (
        config_or_parameters
        if isinstance(config_or_parameters, IsolatorParameters)
        else derive_isolator_parameters(config_or_parameters)
    )
    value = analytic_transfer(
        frequency_hz,
        parameters.fn_hz[index],
        parameters.zeta[index],
        relative=relative,
    )
    return complex(value)


def six_axis_transfer(
    frequencies_hz: Any,
    config_or_parameters: Any = None,
    *,
    relative: bool = False,
) -> np.ndarray:
    """Evaluate all six independent transfers over a frequency grid."""

    parameters = (
        config_or_parameters
        if isinstance(config_or_parameters, IsolatorParameters)
        else derive_isolator_parameters(config_or_parameters)
    )
    values = analytic_transfer(
        frequencies_hz,
        np.asarray(parameters.fn_hz),
        np.asarray(parameters.zeta),
        relative=relative,
    )
    return np.asarray(values, dtype=complex)


def transmissibility(frequency_hz: Any, natural_frequency_hz: Any, damping_ratio: Any) -> np.ndarray | float:
    """Return absolute acceleration/displacement transmissibility magnitude."""

    value = np.abs(absolute_transfer_function(frequency_hz, natural_frequency_hz, damping_ratio))
    return float(value) if np.ndim(value) == 0 else value


def relative_transmissibility(
    frequency_hz: Any, natural_frequency_hz: Any, damping_ratio: Any
) -> np.ndarray | float:
    """Return relative-displacement transmissibility magnitude."""

    value = np.abs(relative_transfer_function(frequency_hz, natural_frequency_hz, damping_ratio))
    return float(value) if np.ndim(value) == 0 else value


def _wrap_phase_difference(phase: float) -> float:
    """Return a signed phase difference in ``[-pi, pi)``."""

    return float((phase + math.pi) % (2.0 * math.pi) - math.pi)


def _complex_pair(value: complex) -> list[float]:
    """Serialize one complex coefficient without lossy string formatting."""

    value = complex(value)
    return [float(value.real), float(value.imag)]


@dataclass(frozen=True)
class HarmonicFit:
    """Complex sine/cosine fit for one input/output harmonic pair.

    The fit convention is deliberately explicit:

    ``x(t) = a_s sin(wt) + a_c cos(wt) + b``
    ``X = a_c - i*a_s``
    ``H = Y / X``

    Thus ``X`` and ``Y`` are coefficients of ``Re(X exp(iwt))`` and the
    reported phase is ``arg(H)`` (output phase minus input phase).  Including
    an intercept removes the static preload offset without mixing it into the
    harmonic coefficient.
    """

    frequency_hz: float
    sample_count: int
    sample_rate_hz: float
    fit_start_s: float
    fit_end_s: float
    input_complex_coefficient: complex
    output_complex_coefficient: complex
    transfer_complex: complex
    input_amplitude: float
    output_amplitude: float
    amplitude_ratio: float
    phase_difference_rad: float
    input_fit_residual_rms: float
    output_fit_residual_rms: float
    normalized_input_residual: float
    normalized_output_residual: float
    condition_number: float

    def __post_init__(self) -> None:
        frequency = _coerce_scalar("frequency_hz", self.frequency_hz, positive=True)
        if isinstance(self.sample_count, (bool, np.bool_)) or int(self.sample_count) != self.sample_count:
            raise IsolatorError("sample_count must be an integer")
        if int(self.sample_count) < 4:
            raise IsolatorError("sample_count must be at least four")
        sample_rate = _coerce_scalar("sample_rate_hz", self.sample_rate_hz, positive=True)
        fit_start = _coerce_scalar("fit_start_s", self.fit_start_s, nonnegative=True)
        fit_end = _coerce_scalar("fit_end_s", self.fit_end_s, nonnegative=True)
        if fit_end <= fit_start:
            raise IsolatorError("fit_end_s must be greater than fit_start_s")
        for name in (
            "input_complex_coefficient",
            "output_complex_coefficient",
            "transfer_complex",
        ):
            value = complex(getattr(self, name))
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise IsolatorError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "input_amplitude",
            "output_amplitude",
            "amplitude_ratio",
            "input_fit_residual_rms",
            "output_fit_residual_rms",
            "normalized_input_residual",
            "normalized_output_residual",
            "condition_number",
        ):
            value = _coerce_scalar(name, getattr(self, name), nonnegative=True)
            object.__setattr__(self, name, value)
        phase = _coerce_scalar("phase_difference_rad", self.phase_difference_rad)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "fit_start_s", fit_start)
        object.__setattr__(self, "fit_end_s", fit_end)
        object.__setattr__(self, "phase_difference_rad", _wrap_phase_difference(phase))

    @property
    def phase_difference_deg(self) -> float:
        """Return output-minus-input phase in degrees."""

        return float(np.rad2deg(self.phase_difference_rad))

    @property
    def fit_residual_rms(self) -> float:
        """Return the larger absolute fit residual RMS."""

        return max(self.input_fit_residual_rms, self.output_fit_residual_rms)

    @property
    def normalized_fit_residual(self) -> float:
        """Return the larger residual normalized by its fitted amplitude."""

        return max(self.normalized_input_residual, self.normalized_output_residual)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full complex fit and independent quality metrics."""

        return {
            "frequency_hz": self.frequency_hz,
            "sample_count": self.sample_count,
            "sample_rate_hz": self.sample_rate_hz,
            "fit_start_s": self.fit_start_s,
            "fit_end_s": self.fit_end_s,
            "input_complex_coefficient": _complex_pair(self.input_complex_coefficient),
            "output_complex_coefficient": _complex_pair(self.output_complex_coefficient),
            "transfer_complex": _complex_pair(self.transfer_complex),
            "input_amplitude": self.input_amplitude,
            "output_amplitude": self.output_amplitude,
            "amplitude_ratio": self.amplitude_ratio,
            "phase_difference_rad": self.phase_difference_rad,
            "phase_difference_deg": self.phase_difference_deg,
            "input_fit_residual_rms": self.input_fit_residual_rms,
            "output_fit_residual_rms": self.output_fit_residual_rms,
            "fit_residual_rms": self.fit_residual_rms,
            "normalized_input_residual": self.normalized_input_residual,
            "normalized_output_residual": self.normalized_output_residual,
            "normalized_fit_residual": self.normalized_fit_residual,
            "condition_number": self.condition_number,
        }


def _solve_harmonic_coefficients(time: np.ndarray, signals: np.ndarray, frequency_hz: float):
    """Solve sine/cosine/intercept least squares with a small Gram system."""

    omega = 2.0 * math.pi * frequency_hz
    sine = np.sin(omega * time)
    cosine = np.cos(omega * time)
    basis = np.column_stack((sine, cosine, np.ones(time.size)))
    values = np.asarray(signals, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise IsolatorError("harmonic signals must be one- or two-dimensional")
    gram = basis.T.dot(basis)
    right_hand_side = basis.T.dot(values)
    coefficients = np.linalg.solve(gram, right_hand_side)
    residual = values - basis.dot(coefficients)
    condition = float(np.sqrt(np.linalg.cond(gram)))
    if squeeze:
        return coefficients[:, 0], residual[:, 0], condition
    return coefficients, residual, condition


def _fit_one_harmonic(time: np.ndarray, signal: np.ndarray, frequency_hz: float):
    """Fit one real signal and return complex coefficient and diagnostics."""

    coefficients, residual, condition = _solve_harmonic_coefficients(time, signal, frequency_hz)
    complex_coefficient = complex(float(coefficients[1]), -float(coefficients[0]))
    amplitude = float(abs(complex_coefficient))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    normalized_residual = residual_rms / max(amplitude, np.finfo(float).tiny)
    return complex_coefficient, amplitude, residual_rms, normalized_residual, condition


def fit_harmonic_transfer(
    time: Any,
    input_signal: Any,
    output_signal: Any,
    frequency_hz: float,
    *,
    discard_time_s: float = 0.0,
    fit_duration_s: Optional[float] = None,
) -> HarmonicFit:
    """Fit a complex transfer after a registered transient-discard window.

    Args:
        time: Strictly increasing sample times in seconds.
        input_signal: Input coordinate at the same reference point/frame.
        output_signal: Output coordinate at the same harmonic and frame.
        frequency_hz: Positive harmonic frequency.
        discard_time_s: First time included in the fit.
        fit_duration_s: Optional duration after ``discard_time_s``.  If
            omitted, all remaining samples are used.
    """

    frequency = _coerce_scalar("frequency_hz", frequency_hz, positive=True)
    times = np.asarray(time, dtype=float)
    inputs = np.asarray(input_signal, dtype=float)
    outputs = np.asarray(output_signal, dtype=float)
    if times.ndim != 1 or inputs.ndim != 1 or outputs.ndim != 1:
        raise IsolatorError("time, input_signal and output_signal must be one-dimensional")
    if times.size != inputs.size or times.size != outputs.size or times.size < 4:
        raise IsolatorError("harmonic fit inputs must have equal length >= 4")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(outputs)):
        raise IsolatorError("harmonic fit inputs must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise IsolatorError("time must be strictly increasing")
    discard = _coerce_scalar("discard_time_s", discard_time_s, nonnegative=True)
    if fit_duration_s is None:
        fit_end = float(times[-1])
    else:
        duration = _coerce_scalar("fit_duration_s", fit_duration_s, positive=True)
        fit_end = discard + duration
    selected = (times >= discard) & (times <= fit_end + 1e-12)
    if int(np.count_nonzero(selected)) < 4:
        raise IsolatorError("transient discard / fit duration leaves fewer than four samples")
    fit_time = times[selected]
    fit_input = inputs[selected]
    fit_output = outputs[selected]
    input_complex, input_amplitude, input_residual, input_normalized, condition = _fit_one_harmonic(
        fit_time, fit_input, frequency
    )
    output_complex, output_amplitude, output_residual, output_normalized, output_condition = _fit_one_harmonic(
        fit_time, fit_output, frequency
    )
    if input_amplitude <= np.finfo(float).eps:
        raise IsolatorError("harmonic input coefficient is too small to form a transfer")
    transfer = output_complex / input_complex
    sample_rate = float(1.0 / np.median(np.diff(fit_time)))
    return HarmonicFit(
        frequency_hz=frequency,
        sample_count=int(fit_time.size),
        sample_rate_hz=sample_rate,
        fit_start_s=float(fit_time[0]),
        fit_end_s=float(fit_time[-1]),
        input_complex_coefficient=input_complex,
        output_complex_coefficient=output_complex,
        transfer_complex=transfer,
        input_amplitude=input_amplitude,
        output_amplitude=output_amplitude,
        amplitude_ratio=float(abs(transfer)),
        phase_difference_rad=float(np.angle(transfer)),
        input_fit_residual_rms=input_residual,
        output_fit_residual_rms=output_residual,
        normalized_input_residual=input_normalized,
        normalized_output_residual=output_normalized,
        condition_number=max(condition, output_condition),
    )


def compare_harmonic_fit(
    fit: HarmonicFit,
    expected_transfer: complex,
    *,
    thresholds: Mapping[str, float] = PHASE03_TRANSFER_THRESHOLDS,
) -> dict[str, Any]:
    """Compare measured and analytic complex transfer with separate gates."""

    if not isinstance(fit, HarmonicFit):
        raise IsolatorError("fit must be a HarmonicFit")
    expected = complex(expected_transfer)
    if not np.isfinite(expected.real) or not np.isfinite(expected.imag) or abs(expected) <= 0.0:
        raise IsolatorError("expected_transfer must be finite and non-zero")
    amplitude_error = float(abs(fit.transfer_complex) / abs(expected) - 1.0)
    phase_error = abs(_wrap_phase_difference(np.angle(fit.transfer_complex) - np.angle(expected)))
    amplitude_passed = bool(abs(amplitude_error) <= float(thresholds["amplitude_relative_error_max"]))
    phase_passed = bool(np.rad2deg(phase_error) <= float(thresholds["phase_absolute_error_deg_max"]))
    residual_passed = bool(fit.normalized_fit_residual <= float(thresholds["fit_normalized_residual_max"]))
    passed = bool(amplitude_passed and phase_passed and residual_passed)
    return {
        "expected_transfer_complex": _complex_pair(expected),
        "measured_transfer_complex": _complex_pair(fit.transfer_complex),
        "amplitude_relative_error": amplitude_error,
        "phase_absolute_error_rad": float(phase_error),
        "phase_absolute_error_deg": float(np.rad2deg(phase_error)),
        "normalized_fit_residual": fit.normalized_fit_residual,
        "amplitude_gate_passed": amplitude_passed,
        "phase_gate_passed": phase_passed,
        "residual_gate_passed": residual_passed,
        "passed": passed,
    }


def transfer_metrics(
    frequencies_hz: Any,
    config_or_parameters: Any = None,
    *,
    base_displacement_amplitude_m: float = 1.0,
) -> dict[str, Any]:
    """Compute deterministic physics-only transfer-envelope quantities."""

    if isinstance(config_or_parameters, IsolatorParameters):
        parameters = config_or_parameters
        config = _config_from_parameters(parameters)
    else:
        config = _coerce_config(config_or_parameters)
        parameters = derive_isolator_parameters(config)
    frequencies = _frequency_array(frequencies_hz)
    amplitude = _coerce_scalar("base_displacement_amplitude_m", base_displacement_amplitude_m, nonnegative=True)
    absolute = six_axis_transfer(frequencies, parameters, relative=False)
    relative = six_axis_transfer(frequencies, parameters, relative=True)
    absolute_magnitude = np.abs(absolute)
    relative_magnitude = np.abs(relative)
    relative_displacement = relative_magnitude * amplitude
    if relative_displacement.ndim == 1:
        translation_peak = relative_displacement[:3]
        rotation_peak = relative_displacement[3:]
    else:
        reduction_axes = tuple(range(relative_displacement.ndim - 1))
        translation_peak = np.max(relative_displacement[..., :3], axis=reduction_axes)
        rotation_peak = np.max(relative_displacement[..., 3:], axis=reduction_axes)
    travel_margin = np.asarray(config.travel_limits_m, dtype=float) - translation_peak
    angle_margin = np.asarray(config.angle_limits_rad, dtype=float) - rotation_peak
    return {
        "axis_order": list(AXES),
        "frequencies_hz": frequencies.tolist(),
        "absolute_transfer": absolute.tolist(),
        "relative_transfer": relative.tolist(),
        "T_accel": absolute_magnitude.tolist(),
        "R_relative": relative_magnitude.tolist(),
        "T_peak": float(np.max(absolute_magnitude)),
        "D_relative": float(np.max(relative_displacement)),
        "travel_margin_m": travel_margin.tolist(),
        "angle_margin_rad": angle_margin.tolist(),
        "static_sag_uncompensated_m": static_sag_uncompensated_m(parameters),
        "static_offset_compensated": [0.0] * 6,
    }




def parameter_sweep(
    natural_frequency_candidates_hz: Iterable[Any],
    damping_ratio_candidates: Iterable[Any],
    *,
    base_config: Any = None,
) -> tuple[IsolatorParameters, ...]:
    """Build every registered ``f_n``/``zeta`` candidate in stable order."""

    base = _coerce_config(base_config)
    candidates = []
    for natural_frequency in natural_frequency_candidates_hz:
        for damping_ratio in damping_ratio_candidates:
            candidates.append(
                derive_isolator_parameters(
                    base,
                    fn_hz=natural_frequency,
                    zeta=damping_ratio,
                )
            )
    return tuple(candidates)


@dataclass(frozen=True)
class IsolatorSafetyCheck:
    """One strict travel or angle gate."""

    name: str
    passed: bool
    measured: Optional[float]
    limit: Optional[float]
    message: str

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "measured": self.measured,
            "limit": self.limit,
            "message": self.message,
        }


@dataclass(frozen=True)
class IsolatorSafetyReport:
    """Aggregate fail-closed travel / angle report."""

    checks: tuple[IsolatorSafetyCheck, ...]
    max_relative_pose: tuple[float, ...]
    travel_margin_m: tuple[float, ...]
    angle_margin_rad: tuple[float, ...]
    static_sag_uncompensated_m: float
    payload_static_offset: tuple[float, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def violations(self) -> tuple[IsolatorSafetyCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def metrics(self) -> dict[str, float]:
        return {
            check.name: check.measured
            for check in self.checks
            if check.measured is not None
        }

    def __bool__(self) -> bool:
        return self.passed

    def raise_if_failed(self) -> "IsolatorSafetyReport":
        if not self.passed:
            raise IsolatorSafetyError("; ".join(check.message for check in self.violations), self)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "violations": [check.name for check in self.violations],
            "max_relative_pose": list(self.max_relative_pose),
            "travel_margin_m": list(self.travel_margin_m),
            "angle_margin_rad": list(self.angle_margin_rad),
            "static_sag_uncompensated_m": self.static_sag_uncompensated_m,
            "payload_static_offset": list(self.payload_static_offset),
            "metrics": self.metrics,
        }


def _invalid_safety_report(message: str, config: IsolatorConfig) -> IsolatorSafetyReport:
    failed = IsolatorSafetyCheck("input", False, None, None, message)
    return IsolatorSafetyReport(
        checks=(failed,),
        max_relative_pose=(float("nan"),) * 6,
        travel_margin_m=(float("nan"),) * 3,
        angle_margin_rad=(float("nan"),) * 3,
        static_sag_uncompensated_m=static_sag_uncompensated_m(config),
        payload_static_offset=(float("nan"),) * 6,
    )


def check_isolator_safety(
    relative_pose: Any,
    config: Any = None,
    *,
    payload_mass_kg: float = 0.0,
    payload_com_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> IsolatorSafetyReport:
    """Check strict six-axis relative travel and angle limits.

    ``relative_pose`` may be one six-vector or an ``(N, 6)`` trace.  Invalid
    shapes, NaN/Inf values, and invalid payload inputs return a failed report;
    they never pass through as an unchecked candidate.
    """

    if isinstance(config, IsolatorParameters):
        cfg = _config_from_parameters(config)
    else:
        cfg = _coerce_config(config)
    try:
        values = np.asarray(relative_pose, dtype=float)
    except (TypeError, ValueError):
        return _invalid_safety_report("relative_pose must be a finite six-axis vector or trace", cfg)
    if values.ndim == 1:
        if values.shape != (6,):
            return _invalid_safety_report("relative_pose must have six coordinates", cfg)
        samples = values.reshape(1, 6)
    elif values.ndim == 2 and values.shape[1] == 6 and values.shape[0] > 0:
        samples = values
    else:
        return _invalid_safety_report("relative_pose must have shape (6,) or (N, 6)", cfg)
    if not np.all(np.isfinite(samples)):
        return _invalid_safety_report("relative_pose contains non-finite values", cfg)
    try:
        payload_offset = static_equilibrium_offset(
            cfg,
            payload_mass_kg=payload_mass_kg,
            payload_com_m=payload_com_m,
        )
    except (IsolatorError, IsolatorConfigurationError) as exc:
        return _invalid_safety_report(str(exc), cfg)

    maximum = np.max(np.abs(samples), axis=0)
    travel_margin = np.asarray(cfg.travel_limits_m, dtype=float) - maximum[:3]
    angle_margin = np.asarray(cfg.angle_limits_rad, dtype=float) - maximum[3:]
    checks = []
    for index, axis in enumerate(TRANSLATION_AXES):
        passed = bool(maximum[index] < cfg.travel_limits_m[index])
        checks.append(
            IsolatorSafetyCheck(
                f"travel_{axis}",
                passed,
                float(maximum[index]),
                cfg.travel_limits_m[index],
                f"{axis} travel {maximum[index]:.9g} m {'is below' if passed else 'reaches/exceeds'} "
                f"the strict limit {cfg.travel_limits_m[index]:.9g} m",
            )
        )
    for offset, axis in enumerate(ROTATION_AXES):
        index = offset + 3
        passed = bool(maximum[index] < cfg.angle_limits_rad[offset])
        checks.append(
            IsolatorSafetyCheck(
                f"angle_{axis}",
                passed,
                float(maximum[index]),
                cfg.angle_limits_rad[offset],
                f"{axis} angle {maximum[index]:.9g} rad {'is below' if passed else 'reaches/exceeds'} "
                f"the strict limit {cfg.angle_limits_rad[offset]:.9g} rad",
            )
        )
    return IsolatorSafetyReport(
        checks=tuple(checks),
        max_relative_pose=tuple(float(item) for item in maximum),
        travel_margin_m=tuple(float(item) for item in travel_margin),
        angle_margin_rad=tuple(float(item) for item in angle_margin),
        static_sag_uncompensated_m=static_sag_uncompensated_m(cfg),
        payload_static_offset=tuple(float(item) for item in payload_offset),
    )


def check_safety(relative_pose: Any, config: Any = None, **kwargs: Any) -> IsolatorSafetyReport:
    """Short alias for :func:`check_isolator_safety`."""

    return check_isolator_safety(relative_pose, config, **kwargs)


def validate_isolator_safety(relative_pose: Any, config: Any = None, **kwargs: Any) -> IsolatorSafetyReport:
    """Raise :class:`IsolatorSafetyError` unless all strict gates pass."""

    return check_isolator_safety(relative_pose, config, **kwargs).raise_if_failed()


__all__ = [
    "AXES",
    "TRANSLATION_AXES",
    "ROTATION_AXES",
    "CANONICAL_WORKTABLE_DIMENSIONS_M",
    "CANONICAL_WORKTABLE_MASS_KG",
    "CANONICAL_WORKTABLE_INERTIA_KG_M2",
    "CANONICAL_TABLETOP_DIMENSIONS_M",
    "CANONICAL_TABLE_MASS_KG",
    "CANONICAL_TABLE_INERTIA_KG_M2",
    "DEFAULT_NATURAL_FREQUENCY_HZ",
    "DEFAULT_DAMPING_RATIO",
    "DEFAULT_GRAVITY_M_S2",
    "DEFAULT_TRAVEL_LIMITS_M",
    "DEFAULT_ANGLE_LIMITS_RAD",
    "DEFAULT_FN_HZ",
    "DEFAULT_ZETA",
    "DEFAULT_ISOLATOR_CONFIG",
    "PHASE03_TRANSFER_SCHEMA_ID",
    "PHASE03_TRANSFER_SCHEMA_VERSION",
    "HARMONIC_REGION_RATIOS",
    "HARMONIC_PROBE_TRANSIENT_CYCLES",
    "HARMONIC_PROBE_FIT_CYCLES",
    "PHASE03_PROBE_SAMPLE_STRIDE",
    "PHASE03_TRANSFER_THRESHOLDS",
    "IsolatorError",
    "IsolatorConfigurationError",
    "IsolatorSafetyError",
    "IsolatorConfig",
    "IsolatorParameters",
    "derive_isolator_parameters",
    "derive_k_c",
    "preload_springref",
    "vertical_preload_springref",
    "static_sag_uncompensated_m",
    "static_equilibrium_offset",
    "payload_static_offset",
    "transfer_function",
    "analytic_transfer",
    "absolute_transfer_function",
    "relative_transfer_function",
    "single_axis_transfer",
    "six_axis_transfer",
    "transmissibility",
    "relative_transmissibility",
    "transfer_metrics",
    "HarmonicFit",
    "fit_harmonic_transfer",
    "compare_harmonic_fit",
    "parameter_sweep",
    "IsolatorSafetyCheck",
    "IsolatorSafetyReport",
    "check_isolator_safety",
    "check_safety",
    "validate_isolator_safety",
]
