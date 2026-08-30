"""Simulator-independent ShakeBench v0 excitation program.

The benchmark's authored excitation is deliberately kept outside the MuJoCo
environment.  This module owns the six-axis band table, deterministic line
generation, and the analytic motion derivatives consumed by later deck and
sensor phases.  It imports NumPy only; importing it does not create an
environment, change a robosuite macro, or load Torch / Isaac dependencies.

Time conventions
----------------
``time`` is episode-relative and the quintic ramp is therefore shared by all
axes.  ``t0`` is a common carrier phase origin: generated line phases are
stored as ``phase_at_episode_zero = phase + omega * t0``.  Consequently the
unramped waveform (and the full waveform after the ramp) satisfies the useful
time-shift identity ``program(t0).evaluate(t) == program(0).evaluate(t+t0)``.
The ramp remains episode-relative, as required by the protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import pi
from typing import Any, Final

import numpy as np

AXES: Final[tuple[str, ...]] = ("tx", "ty", "tz", "rx", "ry", "rz")
AXIS_NAMES: Final[tuple[str, ...]] = AXES
TRANSLATION_AXES: Final[tuple[str, ...]] = AXES[:3]
ROTATION_AXES: Final[tuple[str, ...]] = AXES[3:]
AXIS_INDEX: Final[dict[str, int]] = {axis: index for index, axis in enumerate(AXES)}

TRANSLATION_COORDINATE_UNIT: Final[str] = "m"
ROTATION_COORDINATE_UNIT: Final[str] = "rad"
TRANSLATION_ACCELERATION_UNIT: Final[str] = "m/s^2"
ROTATION_ACCELERATION_UNIT: Final[str] = "rad/s^2"

DEFAULT_REFERENCE_ACCEL_RMS_M_S2: Final[float] = 1.0
DEFAULT_KAPPA_ROT: Final[float] = 0.30
DEFAULT_REFERENCE_LEVER_M: Final[float] = 0.65
DEFAULT_FREQUENCY_SCALE: Final[float] = 1.0
DEFAULT_JITTER_FRACTION: Final[float] = 0.10
DEFAULT_RAMP_DURATION_S: Final[float] = 0.50
DEFAULT_EPISODE_DURATION_S: Final[float] = 2.0
DEFAULT_GRAVITY_M_S2: Final[float] = 9.81
CONSERVATIVE_MAX_LINE_FREQUENCY_HZ: Final[float] = 8.87
DEFAULT_MAX_LINES: Final[int] = 12

EXCITATION_SCHEMA_ID: Final[str] = "shakebench.excitation"
EXCITATION_SCHEMA_VERSION: Final[int] = 1
PROGRAM_SCHEMA_ID: Final[str] = "shakebench.excitation.program"


class ExcitationError(ValueError):
    """Base error raised for invalid excitation inputs."""


def _finite_float(name: str, value: Any, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ExcitationError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExcitationError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ExcitationError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        comparator = ">" if strict else ">="
        raise ExcitationError(f"{name} must be {comparator} {minimum}")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ExcitationError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ExcitationError(f"{name} must be a positive integer") from exc
    if integer != value or integer <= 0:
        raise ExcitationError(f"{name} must be a positive integer")
    return integer


@dataclass(frozen=True)
class AxisBand:
    """One row of the authored acceleration-PSD band table.

    ``relative_accel_rms`` is relative to the translational reference RMS in
    :class:`ExcitationConfig`.  ``accel_rms`` is exposed as a property so the
    line-amplitude contract can be read directly from a band object.

    Attributes:
        axis: Canonical axis name.
        center_hz: Center frequency after ``frequency_scale`` is applied.
        relative_accel_rms: RMS relative to the translational reference.
        bandwidth_ratio: Fractional half-width around the center frequency.
        tones: Number of spectral lines in this band.
        coordinate_unit: Unit of the integrated coordinate (m or rad).
        acceleration_unit: Unit of the authored acceleration.
        reference_accel_rms_m_s2: Translational reference RMS.
        derived_from_axis: Translation source for a derived rotation band.
        kappa_rot: Rotation-to-translation scale when derived.
        reference_lever_m: Lever used for a derived rotation band.
    """

    axis: str
    center_hz: float
    relative_accel_rms: float
    bandwidth_ratio: float
    tones: int
    coordinate_unit: str
    acceleration_unit: str
    reference_accel_rms_m_s2: float = DEFAULT_REFERENCE_ACCEL_RMS_M_S2
    derived_from_axis: str | None = None
    kappa_rot: float | None = None
    reference_lever_m: float | None = None

    def __post_init__(self) -> None:
        if self.axis not in AXIS_INDEX:
            raise ExcitationError(f"unknown axis {self.axis!r}")
        center_hz = _finite_float("center_hz", self.center_hz, minimum=0.0, strict=True)
        relative_rms = _finite_float("relative_accel_rms", self.relative_accel_rms, minimum=0.0)
        bandwidth_ratio = _finite_float("bandwidth_ratio", self.bandwidth_ratio, minimum=0.0, strict=True)
        tones = _positive_int("tones", self.tones)
        reference = _finite_float("reference_accel_rms_m_s2", self.reference_accel_rms_m_s2, minimum=0.0, strict=True)
        if self.coordinate_unit not in {TRANSLATION_COORDINATE_UNIT, ROTATION_COORDINATE_UNIT}:
            raise ExcitationError(f"unsupported coordinate unit {self.coordinate_unit!r}")
        if self.acceleration_unit not in {
            TRANSLATION_ACCELERATION_UNIT,
            ROTATION_ACCELERATION_UNIT,
        }:
            raise ExcitationError(f"unsupported acceleration unit {self.acceleration_unit!r}")
        if self.derived_from_axis is not None and self.derived_from_axis not in TRANSLATION_AXES:
            raise ExcitationError(f"invalid rotation source axis {self.derived_from_axis!r}")
        object.__setattr__(self, "center_hz", center_hz)
        object.__setattr__(self, "relative_accel_rms", relative_rms)
        object.__setattr__(self, "bandwidth_ratio", bandwidth_ratio)
        object.__setattr__(self, "tones", tones)
        object.__setattr__(self, "reference_accel_rms_m_s2", reference)

    @property
    def accel_rms(self) -> float:
        """Absolute acceleration RMS used by the line-amplitude formula."""

        return self.relative_accel_rms * self.reference_accel_rms_m_s2

    @property
    def relative_rms(self) -> float:
        """Short alias for the design-tree ``relative accel RMS`` column."""

        return self.relative_accel_rms

    @property
    def nominal_band_hz(self) -> tuple[float, float]:
        """Return the lower and upper authored band limits in hertz."""

        half_width = self.center_hz * self.bandwidth_ratio
        return self.center_hz - half_width, self.center_hz + half_width

    @property
    def frequency_band_hz(self) -> tuple[float, float]:
        """Alias for :attr:`nominal_band_hz`."""

        return self.nominal_band_hz

    @property
    def is_rotation(self) -> bool:
        """Whether this row describes an angular coordinate."""

        return self.axis in ROTATION_AXES

    def to_dict(self) -> dict[str, Any]:
        """Serialize the band row using JSON-compatible values."""

        return {
            "axis": self.axis,
            "center_hz": self.center_hz,
            "relative_accel_rms": self.relative_accel_rms,
            "accel_rms": self.accel_rms,
            "bandwidth_ratio": self.bandwidth_ratio,
            "tones": self.tones,
            "nominal_band_hz": list(self.nominal_band_hz),
            "coordinate_unit": self.coordinate_unit,
            "acceleration_unit": self.acceleration_unit,
            "reference_accel_rms_m_s2": self.reference_accel_rms_m_s2,
            "derived_from_axis": self.derived_from_axis,
            "kappa_rot": self.kappa_rot,
            "reference_lever_m": self.reference_lever_m,
        }


@dataclass(frozen=True)
class ExcitationConfig:
    """All authored excitation parameters needed to rebuild a program.

    Attributes:
        frequency_scale: Multiplier applied to every band frequency.
        reference_accel_rms_m_s2: Translation reference acceleration RMS.
        kappa_rot: Rotational RMS multiplier before division by the lever.
        reference_lever_m: Canonical lever for deriving rotation RMS values.
        jitter_fraction: Maximum jitter as a fraction of adjacent line spacing.
        ramp_duration_s: Episode-relative quintic ramp duration.
        episode_duration_s: Default calibration and replay window.
        gravity_m_s2: Gravity used to normalize Gamma.
        conservative_max_line_frequency_hz: Strict frequency safety limit.
        workpiece_point_offset_m: Authored point used for the default Gamma
            calculation, explicitly serialized as part of the config.
    """

    frequency_scale: float = DEFAULT_FREQUENCY_SCALE
    reference_accel_rms_m_s2: float = DEFAULT_REFERENCE_ACCEL_RMS_M_S2
    kappa_rot: float = DEFAULT_KAPPA_ROT
    reference_lever_m: float = DEFAULT_REFERENCE_LEVER_M
    jitter_fraction: float = DEFAULT_JITTER_FRACTION
    ramp_duration_s: float = DEFAULT_RAMP_DURATION_S
    episode_duration_s: float = DEFAULT_EPISODE_DURATION_S
    gravity_m_s2: float = DEFAULT_GRAVITY_M_S2
    conservative_max_line_frequency_hz: float = CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
    workpiece_point_offset_m: tuple[float, float, float] = (
        DEFAULT_REFERENCE_LEVER_M,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        values = {
            "frequency_scale": _finite_float("frequency_scale", self.frequency_scale, minimum=0.0, strict=True),
            "reference_accel_rms_m_s2": _finite_float(
                "reference_accel_rms_m_s2", self.reference_accel_rms_m_s2, minimum=0.0, strict=True
            ),
            "kappa_rot": _finite_float("kappa_rot", self.kappa_rot, minimum=0.0, strict=True),
            "reference_lever_m": _finite_float("reference_lever_m", self.reference_lever_m, minimum=0.0, strict=True),
            "jitter_fraction": _finite_float("jitter_fraction", self.jitter_fraction, minimum=0.0),
            "ramp_duration_s": _finite_float("ramp_duration_s", self.ramp_duration_s, minimum=0.0),
            "episode_duration_s": _finite_float(
                "episode_duration_s", self.episode_duration_s, minimum=0.0, strict=True
            ),
            "gravity_m_s2": _finite_float("gravity_m_s2", self.gravity_m_s2, minimum=0.0, strict=True),
            "conservative_max_line_frequency_hz": _finite_float(
                "conservative_max_line_frequency_hz",
                self.conservative_max_line_frequency_hz,
                minimum=0.0,
                strict=True,
            ),
        }
        try:
            point = tuple(self.workpiece_point_offset_m)
        except TypeError as exc:
            raise ExcitationError("workpiece_point_offset_m must contain three finite values") from exc
        if len(point) != 3:
            raise ExcitationError("workpiece_point_offset_m must contain three finite values")
        point = tuple(_finite_float(f"workpiece_point_offset_m[{index}]", value) for index, value in enumerate(point))
        if values["jitter_fraction"] >= 0.5:
            raise ExcitationError("jitter_fraction must be less than 0.5")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "workpiece_point_offset_m", point)

    @property
    def max_lines(self) -> int:
        """Return the fixed padded line count used by serialized programs."""

        return DEFAULT_MAX_LINES

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration into a JSON-compatible mapping."""

        return {
            "schema_id": EXCITATION_SCHEMA_ID,
            "schema_version": EXCITATION_SCHEMA_VERSION,
            "frequency_scale": self.frequency_scale,
            "reference_accel_rms_m_s2": self.reference_accel_rms_m_s2,
            "kappa_rot": self.kappa_rot,
            "reference_lever_m": self.reference_lever_m,
            "jitter_fraction": self.jitter_fraction,
            "ramp_duration_s": self.ramp_duration_s,
            "episode_duration_s": self.episode_duration_s,
            "gravity_m_s2": self.gravity_m_s2,
            "conservative_max_line_frequency_hz": self.conservative_max_line_frequency_hz,
            "workpiece_point_offset_m": list(self.workpiece_point_offset_m),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExcitationConfig":
        """Parse and strictly validate a serialized excitation config."""

        if not isinstance(payload, Mapping):
            raise ExcitationError("excitation config must be an object")
        allowed = {
            "schema_id",
            "schema_version",
            "frequency_scale",
            "reference_accel_rms_m_s2",
            "kappa_rot",
            "reference_lever_m",
            "jitter_fraction",
            "ramp_duration_s",
            "episode_duration_s",
            "gravity_m_s2",
            "conservative_max_line_frequency_hz",
            "workpiece_point_offset_m",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ExcitationError("Unknown excitation config field(s): " + ", ".join(repr(item) for item in unknown))
        if "schema_id" in payload and payload["schema_id"] != EXCITATION_SCHEMA_ID:
            raise ExcitationError(f"schema_id must be {EXCITATION_SCHEMA_ID!r}")
        if "schema_version" in payload and payload["schema_version"] != EXCITATION_SCHEMA_VERSION:
            raise ExcitationError(f"schema_version must be {EXCITATION_SCHEMA_VERSION}")
        values = dict(payload)
        values.pop("schema_id", None)
        values.pop("schema_version", None)
        return cls(**values)


def _coerce_config(config: ExcitationConfig | Mapping[str, Any] | None) -> ExcitationConfig:
    if config is None:
        return DEFAULT_EXCITATION_CONFIG
    if isinstance(config, ExcitationConfig):
        return config
    if isinstance(config, Mapping):
        return ExcitationConfig.from_dict(config)
    raise ExcitationError("config must be an ExcitationConfig or mapping")


def derive_rotation_accel_rms(
    source_accel_rms: float,
    *,
    kappa_rot: float = DEFAULT_KAPPA_ROT,
    reference_lever_m: float = DEFAULT_REFERENCE_LEVER_M,
) -> float:
    """Derive rotational acceleration RMS from a translational source band."""

    source = _finite_float("source_accel_rms", source_accel_rms, minimum=0.0)
    kappa = _finite_float("kappa_rot", kappa_rot, minimum=0.0, strict=True)
    lever = _finite_float("reference_lever_m", reference_lever_m, minimum=0.0, strict=True)
    return kappa * source / lever


def build_band_table(config: ExcitationConfig | Mapping[str, Any] | None = None) -> tuple[AxisBand, ...]:
    """Build the complete six-axis authored band table.

    Frequency scale applies to both centers and bandwidths.  Rotational RMS
    values are derived at construction time from the documented source
    translation band, ``kappa_rot``, and ``reference_lever_m``.
    """

    cfg = _coerce_config(config)
    base = cfg.reference_accel_rms_m_s2
    translation_rows = (
        ("tx", 5.0, 0.50, 0.10, 12),
        ("ty", 6.5, 0.35, 0.10, 10),
        ("tz", 8.0, 1.00, 0.10, 12),
    )
    rows: list[AxisBand] = []
    for axis, center, relative_rms, bandwidth, tones in translation_rows:
        rows.append(
            AxisBand(
                axis=axis,
                center_hz=center * cfg.frequency_scale,
                relative_accel_rms=relative_rms,
                bandwidth_ratio=bandwidth,
                tones=tones,
                coordinate_unit=TRANSLATION_COORDINATE_UNIT,
                acceleration_unit=TRANSLATION_ACCELERATION_UNIT,
                reference_accel_rms_m_s2=base,
            )
        )

    source_by_axis = {band.axis: band for band in rows}
    rotation_rows = (
        ("rx", 3.0, 0.12, 12, "tz"),
        ("ry", 4.0, 0.12, 10, "tz"),
        ("rz", 2.5, 0.12, 8, "tx"),
    )
    for axis, center, bandwidth, tones, source_axis in rotation_rows:
        absolute_rms = derive_rotation_accel_rms(
            source_by_axis[source_axis].accel_rms,
            kappa_rot=cfg.kappa_rot,
            reference_lever_m=cfg.reference_lever_m,
        )
        rows.append(
            AxisBand(
                axis=axis,
                center_hz=center * cfg.frequency_scale,
                relative_accel_rms=absolute_rms / base,
                bandwidth_ratio=bandwidth,
                tones=tones,
                coordinate_unit=ROTATION_COORDINATE_UNIT,
                acceleration_unit=ROTATION_ACCELERATION_UNIT,
                reference_accel_rms_m_s2=base,
                derived_from_axis=source_axis,
                kappa_rot=cfg.kappa_rot,
                reference_lever_m=cfg.reference_lever_m,
            )
        )
    return tuple(rows)


DEFAULT_EXCITATION_CONFIG: Final[ExcitationConfig] = ExcitationConfig()
BAND_TABLE: Final[tuple[AxisBand, ...]] = build_band_table(DEFAULT_EXCITATION_CONFIG)
DEFAULT_BAND_TABLE: Final[tuple[AxisBand, ...]] = BAND_TABLE
AXIS_BANDS: Final[dict[str, AxisBand]] = {band.axis: band for band in BAND_TABLE}


def band_table_dict(config: ExcitationConfig | Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the authored band table keyed by canonical axis name."""

    return {band.axis: band.to_dict() for band in build_band_table(config)}


def axis_schema(config: ExcitationConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the serializable unit and band schema used by a program."""

    cfg = _coerce_config(config)
    return {
        "schema_id": EXCITATION_SCHEMA_ID,
        "schema_version": EXCITATION_SCHEMA_VERSION,
        "axis_order": list(AXES),
        "translation_axes": list(TRANSLATION_AXES),
        "rotation_axes": list(ROTATION_AXES),
        "coordinate_units": {
            axis: (TRANSLATION_COORDINATE_UNIT if index < 3 else ROTATION_COORDINATE_UNIT)
            for index, axis in enumerate(AXES)
        },
        "acceleration_units": {
            axis: (TRANSLATION_ACCELERATION_UNIT if index < 3 else ROTATION_ACCELERATION_UNIT)
            for index, axis in enumerate(AXES)
        },
        "frequency_scale": cfg.frequency_scale,
        "workpiece_point_offset_m": list(cfg.workpiece_point_offset_m),
        "bands": band_table_dict(cfg),
        "ramp_type": "quintic_smoothstep",
        "ramp_duration_s": cfg.ramp_duration_s,
        "program_frame": "deck",
    }


def _normalise_active_axes(active_axes: Iterable[str | int] | str | np.ndarray | None) -> tuple[str, ...]:
    """Normalize axis names, indices, or a six-entry boolean mask."""

    if active_axes is None:
        return AXES
    if isinstance(active_axes, str):
        if "," in active_axes:
            values: list[str | int] = [item.strip() for item in active_axes.split(",") if item.strip()]
        else:
            values = [active_axes]
    elif isinstance(active_axes, np.ndarray) and active_axes.dtype == bool:
        flat = active_axes.reshape(-1)
        if flat.size != len(AXES):
            raise ExcitationError(f"active_axes boolean mask must have {len(AXES)} entries")
        values = [axis for axis, enabled in zip(AXES, flat) if enabled]
    else:
        try:
            values = list(active_axes)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ExcitationError("active_axes must be axis names, indices, or a boolean mask") from exc
        if values and all(isinstance(value, (bool, np.bool_)) for value in values):
            if len(values) != len(AXES):
                raise ExcitationError(f"active_axes boolean mask must have {len(AXES)} entries")
            values = [axis for axis, enabled in zip(AXES, values) if enabled]
    normalised: list[str] = []
    for value in values:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            index = int(value)
            if index < 0 or index >= len(AXES):
                raise ExcitationError(f"active axis index {index} is out of range")
            axis = AXES[index]
        elif isinstance(value, str):
            axis = value
        else:
            raise ExcitationError(f"invalid active axis {value!r}")
        if axis not in AXIS_INDEX:
            raise ExcitationError(f"unknown active axis {axis!r}")
        if axis in normalised:
            raise ExcitationError(f"active_axes contains duplicate axis {axis!r}")
        normalised.append(axis)
    return tuple(axis for axis in AXES if axis in normalised)


def _phase_wrap(phase: np.ndarray) -> np.ndarray:
    return (phase + pi) % (2.0 * pi) - pi


def quintic_ramp(time: Any, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quintic smoothstep value and analytic first / second derivatives.

    The returned arrays have the same shape as ``time``.  Outside the ramp,
    the value is constant and both derivatives are exactly zero.
    """

    duration = _finite_float("duration_s", duration_s, minimum=0.0)
    times = np.asarray(time, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ExcitationError("time must contain only finite values")
    if duration == 0.0:
        value = np.ones_like(times)
        return value, np.zeros_like(times), np.zeros_like(times)

    scaled = times / duration
    value = np.zeros_like(times)
    first = np.zeros_like(times)
    second = np.zeros_like(times)
    middle = (scaled > 0.0) & (scaled < 1.0)
    inside_or_after = scaled >= 1.0
    value[inside_or_after] = 1.0
    s = scaled[middle]
    value[middle] = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    first[middle] = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    second[middle] = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    return value, first, second


def quintic_smoothstep(time: Any, duration_s: float) -> np.ndarray:
    """Return only the episode-relative quintic ramp value."""

    return quintic_ramp(time, duration_s)[0]


def quintic_smoothstep_derivatives(time: Any, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytic first and second derivatives of the ramp."""

    _, first, second = quintic_ramp(time, duration_s)
    return first, second


@dataclass(frozen=True)
class MotionSample:
    """Analytic six-axis position, velocity and acceleration sample.

    Attributes:
        q: Position array with final dimension ``(tx, ty, tz, rx, ry, rz)``.
        qdot: First time derivative of ``q``.
        qdd: Second time derivative of ``q``.
    """

    q: np.ndarray
    qdot: np.ndarray
    qdd: np.ndarray

    def __post_init__(self) -> None:
        for name in ("q", "qdot", "qdd"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape[-1:] != (len(AXES),):
                raise ExcitationError(f"{name} must have a final dimension of {len(AXES)}")
            if not np.all(np.isfinite(value)):
                raise ExcitationError(f"{name} contains non-finite values")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def position(self) -> np.ndarray:
        """Return the six-axis position array."""

        return self.q

    @property
    def velocity(self) -> np.ndarray:
        """Return the six-axis velocity array."""

        return self.qdot

    @property
    def acceleration(self) -> np.ndarray:
        """Return the six-axis acceleration array."""

        return self.qdd

    def __iter__(self):
        """Yield position, velocity, and acceleration for tuple unpacking."""

        yield self.q
        yield self.qdot
        yield self.qdd

    def to_dict(self) -> dict[str, Any]:
        """Serialize all three analytic motion arrays."""

        return {
            "q": self.q.tolist(),
            "qdot": self.qdot.tolist(),
            "qdd": self.qdd.tolist(),
        }


def _array_with_time_axis(value: np.ndarray, time_ndim: int) -> np.ndarray:
    return value[(slice(None), slice(None)) + (None,) * time_ndim]


@dataclass(frozen=True)
class ExcitationProgram:
    """Deterministic, self-describing authored excitation program.

    Attributes:
        config: Complete authored encoding used to build the line table.
        seed: Non-negative replay seed.
        t0: Common carrier phase origin in seconds.
        level_scale: Scalar applied to all line acceleration amplitudes.
        active_axes: Ordered subset of canonical axes being emitted.
        line_accel_amplitude: Padded ``[6, max_lines]`` line amplitudes.
        line_omega_rad_s: Padded angular frequencies.
        line_phase_at_episode_zero: Padded phases including ``t0``.
        line_mask: Boolean mask for active authored lines.
        bands: Band rows in canonical axis order.
    """

    config: ExcitationConfig
    seed: int
    t0: float
    level_scale: float
    active_axes: tuple[str, ...]
    line_accel_amplitude: np.ndarray
    line_omega_rad_s: np.ndarray
    line_phase_at_episode_zero: np.ndarray
    line_mask: np.ndarray
    bands: tuple[AxisBand, ...] = field(default_factory=lambda: BAND_TABLE)

    def __post_init__(self) -> None:
        if not isinstance(self.config, ExcitationConfig):
            raise ExcitationError("program config must be an ExcitationConfig")
        seed = _positive_int("seed", self.seed) if self.seed != 0 else 0
        t0 = _finite_float("t0", self.t0)
        level_scale = _finite_float("level_scale", self.level_scale, minimum=0.0)
        active_axes = _normalise_active_axes(self.active_axes)
        arrays = {}
        for name in (
            "line_accel_amplitude",
            "line_omega_rad_s",
            "line_phase_at_episode_zero",
            "line_mask",
        ):
            value = np.asarray(getattr(self, name), dtype=bool if name == "line_mask" else float)
            if value.ndim != 2 or value.shape[0] != len(AXES):
                raise ExcitationError(f"{name} must have shape (6, max_lines)")
            if value.shape[1] != self.config.max_lines:
                raise ExcitationError(f"{name} must have {self.config.max_lines} line columns")
            if name != "line_mask" and not np.all(np.isfinite(value)):
                raise ExcitationError(f"{name} contains non-finite values")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            arrays[name] = value
        if not np.all(arrays["line_omega_rad_s"][arrays["line_mask"]] > 0.0):
            raise ExcitationError("active lines must have positive angular frequency")
        if np.any(arrays["line_accel_amplitude"][~arrays["line_mask"]] != 0.0):
            raise ExcitationError("inactive line amplitudes must be zero")
        expected_mask = np.zeros(len(AXES), dtype=bool)
        expected_mask[[AXIS_INDEX[axis] for axis in active_axes]] = True
        actual_axis_mask = np.any(arrays["line_mask"], axis=1)
        if not np.array_equal(expected_mask, actual_axis_mask):
            raise ExcitationError("line_mask does not match active_axes")
        if len(self.bands) != len(AXES) or tuple(band.axis for band in self.bands) != AXES:
            raise ExcitationError("bands must be ordered tx, ty, tz, rx, ry, rz")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "level_scale", level_scale)
        object.__setattr__(self, "active_axes", active_axes)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def max_lines(self) -> int:
        """Return the padded line count in this program."""

        return self.line_mask.shape[1]

    @property
    def line_frequency_hz(self) -> np.ndarray:
        """Return padded line frequencies in hertz."""

        value = self.line_omega_rad_s / (2.0 * pi)
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Alias for :attr:`line_frequency_hz`."""

        return self.line_frequency_hz

    @property
    def max_frequency_hz(self) -> float:
        """Return the largest active line frequency."""

        active = self.line_frequency_hz[self.line_mask]
        return float(np.max(active)) if active.size else 0.0

    @property
    def line_q_amplitude(self) -> np.ndarray:
        """Return per-line integrated displacement/angle amplitudes."""

        omega_safe = np.where(self.line_mask, self.line_omega_rad_s, 1.0)
        value = np.where(self.line_mask, self.line_accel_amplitude / omega_safe**2, 0.0)
        value.setflags(write=False)
        return value

    @property
    def line_displacement_amplitude(self) -> np.ndarray:
        """Alias for :attr:`line_q_amplitude`."""

        return self.line_q_amplitude

    @property
    def axis_accel_rms(self) -> np.ndarray:
        """Return analytic RMS per axis from the line amplitudes."""

        value = np.sqrt(np.sum(self.line_accel_amplitude**2, axis=1) / 2.0)
        value.setflags(write=False)
        return value

    @property
    def authored_accel_rms(self) -> np.ndarray:
        """Alias for :attr:`axis_accel_rms`."""

        return self.axis_accel_rms

    @property
    def axis_displacement_bound(self) -> np.ndarray:
        """Return conservative per-axis displacement/angle bounds."""

        value = np.sum(np.abs(self.line_q_amplitude), axis=1)
        value.setflags(write=False)
        return value

    @property
    def displacement_bound_m(self) -> float:
        """Return the largest translational displacement bound in meters."""

        return float(np.max(self.axis_displacement_bound[:3]))

    @property
    def carrier_velocity_bound(self) -> np.ndarray:
        """Return conservative carrier velocity bounds per axis."""

        omega_safe = np.where(self.line_mask, self.line_omega_rad_s, 1.0)
        value = np.sum(np.abs(self.line_accel_amplitude / omega_safe), axis=1)
        value.setflags(write=False)
        return value

    def _carrier_motion(self, time: Any) -> MotionSample:
        """Evaluate the un-ramped line sum and analytic derivatives."""

        times = np.asarray(time, dtype=float)
        if not np.all(np.isfinite(times)):
            raise ExcitationError("time must contain only finite values")
        time_ndim = times.ndim
        omega = _array_with_time_axis(self.line_omega_rad_s, time_ndim)
        phase = _array_with_time_axis(self.line_phase_at_episode_zero, time_ndim)
        amplitude = _array_with_time_axis(self.line_accel_amplitude, time_ndim)
        theta = omega * times + phase
        omega_safe = np.where(self.line_mask, self.line_omega_rad_s, 1.0)
        omega_safe = _array_with_time_axis(omega_safe, time_ndim)
        q = np.sum(-amplitude / omega_safe**2 * np.sin(theta), axis=1)
        qdot = np.sum(-amplitude / omega_safe * np.cos(theta), axis=1)
        qdd = np.sum(amplitude * np.sin(theta), axis=1)
        return MotionSample(
            q=np.moveaxis(q, 0, -1),
            qdot=np.moveaxis(qdot, 0, -1),
            qdd=np.moveaxis(qdd, 0, -1),
        )

    def evaluate_carrier(self, time: Any) -> MotionSample:
        """Evaluate the un-ramped analytic carrier motion."""

        return self._carrier_motion(time)

    def evaluate(self, time: Any) -> MotionSample:
        """Evaluate ``q``, ``qdot`` and ``qdd`` with analytic ramp derivatives."""

        times = np.asarray(time, dtype=float)
        carrier = self._carrier_motion(times)
        ramp, ramp_first, ramp_second = quintic_ramp(times, self.config.ramp_duration_s)
        q = carrier.q * ramp[..., None]
        qdot = carrier.qdot * ramp[..., None] + carrier.q * ramp_first[..., None]
        qdd = (
            carrier.qdd * ramp[..., None]
            + 2.0 * carrier.qdot * ramp_first[..., None]
            + carrier.q * ramp_second[..., None]
        )
        return MotionSample(q=q, qdot=qdot, qdd=qdd)

    sample = evaluate
    motion = evaluate

    def __call__(self, time: Any) -> MotionSample:
        """Evaluate the program at one or more episode-relative times."""

        return self.evaluate(time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete deterministic program and its band schema."""

        return {
            "schema_id": PROGRAM_SCHEMA_ID,
            "schema_version": EXCITATION_SCHEMA_VERSION,
            "axis_order": list(AXES),
            "seed": self.seed,
            "t0_s": self.t0,
            "level_scale": self.level_scale,
            "active_axes": list(self.active_axes),
            "line_accel_amplitude": self.line_accel_amplitude.tolist(),
            "line_omega_rad_s": self.line_omega_rad_s.tolist(),
            "line_phase_at_episode_zero": self.line_phase_at_episode_zero.tolist(),
            "line_mask": self.line_mask.tolist(),
            "episode_time_s": self.t0,
            "ramp_type": "quintic_smoothstep",
            "ramp_duration_s": self.config.ramp_duration_s,
            "program_frame": "deck",
            "config": self.config.to_dict(),
            "bands": [band.to_dict() for band in self.bands],
        }


def build_excitation_program(
    seed: int = 0,
    t0: float = 0.0,
    level_scale: float = 1.0,
    active_axes: Iterable[str | int] | str | np.ndarray | None = None,
    *,
    config: ExcitationConfig | Mapping[str, Any] | None = None,
) -> ExcitationProgram:
    """Create a deterministic six-axis program from the authored band table."""

    cfg = _coerce_config(config)
    if isinstance(seed, (bool, np.bool_)):
        raise ExcitationError("seed must be an integer")
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise ExcitationError("seed must be an integer") from exc
    if seed_int != seed or seed_int < 0:
        raise ExcitationError("seed must be a non-negative integer")
    t0_float = _finite_float("t0", t0)
    level = _finite_float("level_scale", level_scale, minimum=0.0)
    active = _normalise_active_axes(active_axes)
    bands = build_band_table(cfg)
    rng = np.random.default_rng(seed_int)
    max_lines = cfg.max_lines
    amplitudes = np.zeros((len(AXES), max_lines), dtype=float)
    omegas = np.zeros((len(AXES), max_lines), dtype=float)
    phases = np.zeros((len(AXES), max_lines), dtype=float)
    mask = np.zeros((len(AXES), max_lines), dtype=bool)

    for axis_index, band in enumerate(bands):
        low_hz, high_hz = band.nominal_band_hz
        if band.tones == 1:
            frequencies = np.array([band.center_hz], dtype=float)
        else:
            frequencies = np.linspace(low_hz, high_hz, band.tones, dtype=float)
            spacing = (high_hz - low_hz) / (band.tones - 1)
            jitter = rng.uniform(
                -cfg.jitter_fraction * spacing,
                cfg.jitter_fraction * spacing,
                size=band.tones,
            )
            frequencies = np.clip(frequencies + jitter, low_hz, high_hz)
        omega = 2.0 * pi * frequencies
        phase = rng.uniform(-pi, pi, size=band.tones)
        omegas[axis_index, : band.tones] = omega
        phases[axis_index, : band.tones] = _phase_wrap(phase + omega * t0_float)
        if band.axis in active:
            amplitudes[axis_index, : band.tones] = level * band.accel_rms * np.sqrt(2.0 / band.tones)
            mask[axis_index, : band.tones] = True

    return ExcitationProgram(
        config=cfg,
        seed=seed_int,
        t0=t0_float,
        level_scale=level,
        active_axes=active,
        line_accel_amplitude=amplitudes,
        line_omega_rad_s=omegas,
        line_phase_at_episode_zero=phases,
        line_mask=mask,
        bands=bands,
    )


def generate_excitation_program(*args: Any, **kwargs: Any) -> ExcitationProgram:
    """Named alias used by scripts and later phases."""

    return build_excitation_program(*args, **kwargs)


def evaluate_excitation(
    seed: int = 0,
    t0: float = 0.0,
    time: Any = 0.0,
    level_scale: float = 1.0,
    active_axes: Iterable[str | int] | str | np.ndarray | None = None,
    *,
    config: ExcitationConfig | Mapping[str, Any] | None = None,
) -> MotionSample:
    """Build and evaluate a program with an explicit seed/t0/time contract."""

    return build_excitation_program(
        seed=seed,
        t0=t0,
        level_scale=level_scale,
        active_axes=active_axes,
        config=config,
    ).evaluate(time)


def generate_excitation(*args: Any, **kwargs: Any) -> MotionSample:
    """Compatibility alias for :func:`evaluate_excitation`."""

    return evaluate_excitation(*args, **kwargs)


def synthesize_excitation(*args: Any, **kwargs: Any) -> MotionSample:
    """Evaluate a deterministic excitation using the public functional API."""

    return evaluate_excitation(*args, **kwargs)


def expected_line_accel_amplitude(band: AxisBand, level_scale: float = 1.0) -> float:
    """Return the authored amplitude of every line in ``band``."""

    if not isinstance(band, AxisBand):
        raise ExcitationError("band must be an AxisBand")
    level = _finite_float("level_scale", level_scale, minimum=0.0)
    return level * band.accel_rms * np.sqrt(2.0 / band.tones)


def expected_line_q_amplitude(band: AxisBand, frequency_hz: float, level_scale: float = 1.0) -> float:
    """Return ``accel_amp / omega**2`` for one authored line."""

    frequency = _finite_float("frequency_hz", frequency_hz, minimum=0.0, strict=True)
    return expected_line_accel_amplitude(band, level_scale) / (2.0 * pi * frequency) ** 2


__all__ = [
    "AXES",
    "AXIS_BANDS",
    "AXIS_INDEX",
    "AXIS_NAMES",
    "BAND_TABLE",
    "CONSERVATIVE_MAX_LINE_FREQUENCY_HZ",
    "DEFAULT_BAND_TABLE",
    "DEFAULT_EPISODE_DURATION_S",
    "DEFAULT_EXCITATION_CONFIG",
    "DEFAULT_FREQUENCY_SCALE",
    "DEFAULT_GRAVITY_M_S2",
    "DEFAULT_JITTER_FRACTION",
    "DEFAULT_KAPPA_ROT",
    "DEFAULT_MAX_LINES",
    "DEFAULT_RAMP_DURATION_S",
    "DEFAULT_REFERENCE_ACCEL_RMS_M_S2",
    "DEFAULT_REFERENCE_LEVER_M",
    "EXCITATION_SCHEMA_ID",
    "EXCITATION_SCHEMA_VERSION",
    "ExcitationConfig",
    "ExcitationError",
    "ExcitationProgram",
    "AxisBand",
    "MotionSample",
    "PROGRAM_SCHEMA_ID",
    "ROTATION_AXES",
    "TRANSLATION_AXES",
    "axis_schema",
    "band_table_dict",
    "build_band_table",
    "build_excitation_program",
    "derive_rotation_accel_rms",
    "evaluate_excitation",
    "expected_line_accel_amplitude",
    "expected_line_q_amplitude",
    "generate_excitation",
    "generate_excitation_program",
    "quintic_ramp",
    "quintic_smoothstep",
    "quintic_smoothstep_derivatives",
    "synthesize_excitation",
]
