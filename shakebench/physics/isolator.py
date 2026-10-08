"""Canonical linear six-degree-of-freedom ShakeBench isolator.

The model is a lumped base-excitation model: three translational and three
rotational coordinates share the worktable COM / elastic centre, and each
coordinate carries its own natural frequency and damping ratio.  Nothing here
selects an official operating point; the defaults are a provisional probe
profile and callers sweep fn_hz / zeta to derive k and c.

All of the physics is closed form:

    m_eff = (M, M, M, Ixx, Iyy, Izz)        generalized mass / inertia
    k     = m_eff * (2*pi*f_n)^2            per-axis stiffness
    c     = 2*zeta*m_eff*(2*pi*f_n)         per-axis damping
    H_abs = (1 + i*d) / (1 - r^2 + i*d)     base-displacement transfer
    H_rel = r^2 / (1 - r^2 + i*d)           d = 2*zeta*r, r = f/f_n

Measured support motion is fitted to the same convention with a three-term
sine/cosine/intercept least squares, so the analytic transfer is the contract
the compiled MuJoCo scene is checked against.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
TRANSLATION_AXES = AXES[:3]
ROTATION_AXES = AXES[3:]

# Canonical geometry / inertial facts, not a selected isolator operating point.
CANONICAL_WORKTABLE_DIMENSIONS_M = (0.65, 0.60, 0.06)
CANONICAL_WORKTABLE_MASS_KG = 32.0
CANONICAL_WORKTABLE_INERTIA_KG_M2 = (0.9696, 1.1363, 2.0867)

# Provisional probe profile retained from the Phase 02 spike evidence.  Phase
# 06 still selects or replaces it from physics-only transfer-envelope evidence.
DEFAULT_NATURAL_FREQUENCY_HZ = (5.0,) * len(AXES)
DEFAULT_DAMPING_RATIO = (0.10,) * len(AXES)
DEFAULT_GRAVITY_M_S2 = 9.81
DEFAULT_TRAVEL_LIMITS_M = (0.025,) * 3
DEFAULT_ANGLE_LIMITS_RAD = tuple(float(value) for value in np.deg2rad((5.0,) * 3))


class IsolatorError(ValueError):
    """Base error for invalid isolator configuration or probe input."""


class IsolatorConfigurationError(IsolatorError):
    """Raised when an isolator parameter cannot be compiled safely."""


def _checked_array(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    """Coerce a scalar or array to finite floats, rejecting bools and strings."""

    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise IsolatorConfigurationError(f"{name} must contain finite numeric values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise IsolatorConfigurationError(f"{name} must contain finite numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise IsolatorConfigurationError(f"{name} must contain finite numeric values")
    if positive and np.any(array <= 0.0):
        raise IsolatorConfigurationError(f"{name} must contain strictly positive values")
    if nonnegative and np.any(array < 0.0):
        raise IsolatorConfigurationError(f"{name} must contain non-negative values")
    return array


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

    array = _checked_array(name, value, positive=positive, nonnegative=nonnegative)
    if array.ndim == 0:
        if not allow_scalar:
            raise IsolatorConfigurationError(f"{name} must contain exactly {length} values")
        array = np.full(length, float(array), dtype=float)
    else:
        array = array.reshape(-1)
        if array.size != length:
            raise IsolatorConfigurationError(f"{name} must contain exactly {length} values")
    return tuple(float(item) for item in array)


def _coerce_scalar(name: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> float:
    array = _checked_array(name, value, positive=positive, nonnegative=nonnegative)
    if array.ndim != 0:
        raise IsolatorConfigurationError(f"{name} must be a finite numeric scalar")
    return float(array)


@dataclass(frozen=True)
class IsolatorConfig:
    """Per-axis configuration for the canonical six-DoF support.

    fn_hz and zeta accept either one scalar (broadcast to all six axes) or a
    six-vector.  travel_limits_m covers translation and angle_limits_rad
    covers rotation.  Limits are strict safety gates and are kept separate
    from the candidate natural frequency.
    """

    fn_hz: Any = DEFAULT_NATURAL_FREQUENCY_HZ
    zeta: Any = DEFAULT_DAMPING_RATIO
    mass_kg: Any = CANONICAL_WORKTABLE_MASS_KG
    inertia_kg_m2: Any = CANONICAL_WORKTABLE_INERTIA_KG_M2
    gravity_m_s2: Any = DEFAULT_GRAVITY_M_S2
    travel_limits_m: Any = DEFAULT_TRAVEL_LIMITS_M
    angle_limits_rad: Any = DEFAULT_ANGLE_LIMITS_RAD

    def __post_init__(self) -> None:
        object.__setattr__(self, "fn_hz", _coerce_values("fn_hz", self.fn_hz, 6, allow_scalar=True, positive=True))
        object.__setattr__(self, "zeta", _coerce_values("zeta", self.zeta, 6, allow_scalar=True, positive=True))
        object.__setattr__(self, "mass_kg", _coerce_scalar("mass_kg", self.mass_kg, positive=True))
        object.__setattr__(
            self,
            "inertia_kg_m2",
            _coerce_values("inertia_kg_m2", self.inertia_kg_m2, 3, positive=True),
        )
        object.__setattr__(
            self,
            "gravity_m_s2",
            _coerce_scalar("gravity_m_s2", self.gravity_m_s2, positive=True),
        )
        object.__setattr__(
            self,
            "travel_limits_m",
            _coerce_values("travel_limits_m", self.travel_limits_m, 3, positive=True),
        )
        object.__setattr__(
            self,
            "angle_limits_rad",
            _coerce_values("angle_limits_rad", self.angle_limits_rad, 3, positive=True),
        )

    @property
    def limits(self) -> tuple[float, ...]:
        """Return translation limits followed by angular limits."""

        return self.travel_limits_m + self.angle_limits_rad

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
    """Derived per-axis support parameters, produced by derive_isolator_parameters.

    The first three coordinates use kg / N/m / N.s/m.  The last three use
    kg.m^2 / N.m/rad / N.m.s/rad.  springref is a generalized coordinate; only
    its tz element is nonzero because the reference-table preload compensates
    table weight and nothing else.
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

    def to_dict(self) -> dict[str, Any]:
        """Return all derived values with units implied by their field names."""

        return {
            "axis_order": list(AXES),
            "fn_hz": list(self.fn_hz),
            "zeta": list(self.zeta),
            "effective_mass": list(self.effective_mass),
            "omega_n_rad_s": list(self.omega_n_rad_s),
            "stiffness": list(self.stiffness),
            "damping": list(self.damping),
            "springref": list(self.springref),
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "gravity_m_s2": self.gravity_m_s2,
            "static_sag_uncompensated_m": static_sag_uncompensated_m(self),
        }


def _coerce_config(config: Any = None) -> IsolatorConfig:
    if config is None:
        return IsolatorConfig()
    if isinstance(config, IsolatorConfig):
        return config
    if isinstance(config, IsolatorParameters):
        return IsolatorConfig(
            fn_hz=config.fn_hz,
            zeta=config.zeta,
            mass_kg=config.mass_kg,
            inertia_kg_m2=config.inertia_kg_m2,
            gravity_m_s2=config.gravity_m_s2,
        )
    raise IsolatorConfigurationError("config must be an IsolatorConfig or IsolatorParameters")


def _parameters_of(config_or_parameters: Any = None) -> "IsolatorParameters":
    if isinstance(config_or_parameters, IsolatorParameters):
        return config_or_parameters
    return derive_isolator_parameters(config_or_parameters)


def derive_isolator_parameters(
    config: Any = None,
    *,
    fn_hz: Any = None,
    zeta: Any = None,
    mass_kg: Any = None,
    inertia_kg_m2: Any = None,
    gravity_m_s2: Any = None,
) -> IsolatorParameters:
    """Derive omega_n, k, c and preload from a candidate config.

    mass_kg and inertia_kg_m2 are reference worktable values.  They are used
    once to derive the support; a payload never silently triggers a second
    derivation.
    """

    base = _coerce_config(config)
    natural_frequency = (
        base.fn_hz if fn_hz is None else _coerce_values("fn_hz", fn_hz, 6, allow_scalar=True, positive=True)
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
    # Explicitly compensate only the canonical table own weight.
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


def static_sag_uncompensated_m(config_or_parameters: Any = None) -> float:
    """Return the vertical sag without the reference-table preload."""

    parameters = _parameters_of(config_or_parameters)
    return parameters.gravity_m_s2 / parameters.omega_n_rad_s[2] ** 2


def static_equilibrium_offset(
    config_or_parameters: Any = None,
    *,
    payload_mass_kg: float = 0.0,
    payload_com_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compute the additional relative equilibrium caused by a payload.

    The returned vector is in [tx, ty, tz, rx, ry, rz] coordinates.  The table
    own weight cancels against the vertical spring reference.  A payload
    contributes a gravity force at payload_com_m and therefore, when offset in
    x/y, an explicit small-angle gravity torque as well.
    """

    parameters = _parameters_of(config_or_parameters)
    payload_mass = _coerce_scalar("payload_mass_kg", payload_mass_kg, nonnegative=True)
    com = np.asarray(tuple(payload_com_m), dtype=float)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise IsolatorError("payload_com_m must contain three finite values")
    force = np.array([0.0, 0.0, -payload_mass * parameters.gravity_m_s2], dtype=float)
    generalized_load = np.concatenate((force, np.cross(com, force)))
    return generalized_load / np.asarray(parameters.stiffness, dtype=float)


__all__ = [
    "AXES",
    "TRANSLATION_AXES",
    "ROTATION_AXES",
    "CANONICAL_WORKTABLE_DIMENSIONS_M",
    "CANONICAL_WORKTABLE_MASS_KG",
    "CANONICAL_WORKTABLE_INERTIA_KG_M2",
    "DEFAULT_NATURAL_FREQUENCY_HZ",
    "DEFAULT_DAMPING_RATIO",
    "DEFAULT_GRAVITY_M_S2",
    "DEFAULT_TRAVEL_LIMITS_M",
    "DEFAULT_ANGLE_LIMITS_RAD",
    "DEFAULT_ISOLATOR_CONFIG",
    "IsolatorError",
    "IsolatorConfigurationError",
    "IsolatorConfig",
    "IsolatorParameters",
    "derive_isolator_parameters",
    "static_sag_uncompensated_m",
    "static_equilibrium_offset",
]
