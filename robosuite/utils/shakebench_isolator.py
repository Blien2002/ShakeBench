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
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import xml.etree.ElementTree as ET
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


def derive_support_parameters(config: Any = None, **kwargs: Any) -> IsolatorParameters:
    """Alias for :func:`derive_isolator_parameters`."""

    return derive_isolator_parameters(config, **kwargs)


def compute_isolator_parameters(config: Any = None, **kwargs: Any) -> IsolatorParameters:
    """Descriptive alias for :func:`derive_isolator_parameters`."""

    return derive_isolator_parameters(config, **kwargs)


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


def linear_transfer_function(
    frequency_hz: Any,
    natural_frequency_hz: Any,
    damping_ratio: Any,
    *,
    relative: bool = False,
):
    """Descriptive alias for the canonical linear transfer function."""

    return analytic_transfer(
        frequency_hz,
        natural_frequency_hz,
        damping_ratio,
        relative=relative,
    )


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


def _normalise_quaternion_wxyz(quaternion: Any) -> np.ndarray:
    """Normalize one MuJoCo ``wxyz`` quaternion."""

    value = np.asarray(quaternion, dtype=float).reshape(-1)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise IsolatorError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise IsolatorError("quaternion must not be zero")
    return value / norm


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _normalise_quaternion_wxyz(first)
    w2, x2, y2, z2 = _normalise_quaternion_wxyz(second)
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=float,
    )


def _quaternion_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = _normalise_quaternion_wxyz(quaternion)
    return np.array((value[0], -value[1], -value[2], -value[3]), dtype=float)


def _quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalise_quaternion_wxyz(quaternion)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _quaternion_to_rotation_vector_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = _normalise_quaternion_wxyz(quaternion)
    if value[0] < 0.0:
        value = -value
    vector_norm = float(np.linalg.norm(value[1:]))
    if vector_norm <= 1.0e-14:
        return 2.0 * value[1:]
    angle = 2.0 * math.atan2(vector_norm, max(float(value[0]), 0.0))
    return value[1:] * (angle / vector_norm)


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
    "derive_support_parameters",
    "compute_isolator_parameters",
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
    "linear_transfer_function",
    "single_axis_transfer",
    "six_axis_transfer",
    "transmissibility",
    "relative_transmissibility",
    "transfer_metrics",
    "HarmonicFit",
    "fit_harmonic_transfer",
    "compare_harmonic_fit",
    "run_harmonic_transfer_probe",
    "run_harmonic_transfer_grid",
    "run_joint_spectrum_probe",
    "run_payload_sensitivity_probe",
    "build_phase03_transfer_artifact",
    "finalize_phase03_artifact",
    "write_phase03_transfer_artifact",
    "verify_phase03_transfer_artifact",
    "phase03_artifact_payload_hash",
    "parameter_sweep",
    "IsolatorSafetyCheck",
    "IsolatorSafetyReport",
    "check_isolator_safety",
    "check_safety",
    "validate_isolator_safety",
]
