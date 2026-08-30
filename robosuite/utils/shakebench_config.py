"""Closed configuration and provenance boundary for the ShakeBench extension.

The Phase 01 remediation keeps this module simulator-independent. It owns the
JSON-compatible configuration envelope, recursively freezes payloads at
construction time, validates score-affecting fields, and refuses to declare a
scoreable configuration while the later physics/controller/protocol freeze
authority is absent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Union

JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]

SCHEMA_ID = "shakebench.config"
SCHEMA_VERSION = 2
EXTENSION_ID = "shakebench"
EXTENSION_VERSION = "0"
PYTHON_PACKAGE = "robosuite"
DEFAULT_TRACK = "state_oracle_control"

# A string sentinel is used instead of ``None`` so that an explicitly
# unselected official value cannot be confused with a legitimate nullable
# metadata field. It is only meaningful inside official maps.
UNFROZEN = "UNFROZEN"

# These values are intentionally left UNFROZEN until later physics,
# controller, and state-protocol phases produce evidence.
OFFICIAL_FIELD_NAMES = (
    "gamma_star",
    "physics_timestep_s",
    "deck_eq_solref",
    "deck_eq_solimp",
    "deck_mass_kg",
    "deck_inertia_kg_m2",
    "isolator_fn_hz",
    "isolator_zeta",
    "isolator_k",
    "isolator_c",
    "contact_condim",
    "contact_solref",
    "contact_solimp",
    "gripper_actuator_force",
    "osc_profile_id",
    "official_state_ids",
)
OFFICIAL_FIELDS = frozenset(OFFICIAL_FIELD_NAMES)

# Official replay parameters are deliberately not hidden in the free-form
# ``options`` mapping. Their values remain UNFROZEN in the Phase 01 draft.
OFFICIAL_PROVENANCE_FIELD_NAMES = (
    "excitation_profile_id",
    "excitation_profile_hash",
    "authored_spectrum_version",
    "gravity_m_s2",
    "ramp_duration_s",
    "frequency_scale",
    "gamma_point_offset_m",
    "safety_profile_id",
    "safety_profile_hash",
    "safety_profile_limits",
    "official_state_manifest_hash",
    "freeze_status",
)
OFFICIAL_PROVENANCE_FIELDS = frozenset(OFFICIAL_PROVENANCE_FIELD_NAMES)

# Phase 01 has no authority to freeze physics, controller, or protocol values.
# Later phases may replace this explicit gate with a registered freeze record;
# no user-supplied string can activate it in this phase.
SCOREABLE_FREEZE_AUTHORITY = None


class ShakeBenchConfigError(ValueError):
    """Base error for malformed or non-scoreable configurations."""


class UnknownFieldError(ShakeBenchConfigError):
    """Raised when a mapping contains a key outside the declared schema."""


class InvalidOfficialFieldError(ShakeBenchConfigError):
    """Raised when an official field has invalid structure or value."""


class InvalidProvenanceError(ShakeBenchConfigError):
    """Raised when an official replay provenance value is invalid."""


class UnfrozenOfficialFieldError(ShakeBenchConfigError):
    """Raised when a scoreable configuration still contains ``UNFROZEN``."""


class HiddenOfficialConfigurationError(ShakeBenchConfigError):
    """Raised when score-affecting values are hidden in ``options``."""


class OfficialConfigurationNotReadyError(ShakeBenchConfigError):
    """Raised because later phases have not supplied freeze authority."""


def _unknown_fields(mapping: Mapping[str, Any], allowed: set, path: str) -> None:
    unknown = [name for name in mapping if name not in allowed]
    unknown.sort(key=repr)
    if unknown:
        joined = ", ".join(repr(name) for name in unknown)
        raise UnknownFieldError(f"Unknown field(s) at {path}: {joined}")


def _normalise_json_value(value: Any, path: str) -> JSONValue:
    """Copy a JSON-compatible value while rejecting non-deterministic values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ShakeBenchConfigError(f"Non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        normalised = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ShakeBenchConfigError(f"Object key at {path} must be a string; got {type(key).__name__}")
            normalised[key] = _normalise_json_value(item, f"{path}.{key}")
        return normalised
    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ShakeBenchConfigError(f"Value at {path} is not JSON-compatible: {type(value).__name__}")


def _normalise_mapping(value: Mapping[str, Any], path: str) -> dict:
    if not isinstance(value, Mapping):
        raise ShakeBenchConfigError(f"{path} must be an object")
    normalised = _normalise_json_value(value, path)
    assert isinstance(normalised, dict)
    return normalised


def _freeze_json_value(value: JSONValue) -> Any:
    """Recursively convert JSON values to immutable containers."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> JSONValue:
    """Recursively copy immutable payloads into mutable JSON values."""

    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return copy.deepcopy(value)


def _find_unfrozen(value: Any, path: str) -> list:
    """Return paths of every nested ``UNFROZEN`` sentinel."""

    if isinstance(value, str):
        return [path] if value == UNFROZEN else []
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_find_unfrozen(item, f"{path}[{index}]"))
        return paths
    if isinstance(value, Mapping):
        paths = []
        for key, item in value.items():
            paths.extend(_find_unfrozen(item, f"{path}.{key}"))
        return paths
    return []


def _number(value: Any, path: str, *, minimum: Optional[float] = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidOfficialFieldError(f"{path} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise InvalidOfficialFieldError(f"{path} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        comparator = ">" if strict else ">="
        raise InvalidOfficialFieldError(f"{path} must be {comparator} {minimum}")
    return result


def _vector(value: Any, path: str, length: int, *, positive: bool = False, nonzero: bool = False) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InvalidOfficialFieldError(f"{path} must be a finite vector of length {length}")
    result = tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))
    if positive and any(item <= 0.0 for item in result):
        raise InvalidOfficialFieldError(f"{path} must contain only positive values")
    if nonzero and all(item == 0.0 for item in result):
        raise InvalidOfficialFieldError(f"{path} must not be the zero vector")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidOfficialFieldError(f"{path} must be a non-empty string")
    return value


def _hash_string(value: Any, path: str) -> str:
    result = _nonempty_string(value, path)
    if re.fullmatch(r"[0-9a-fA-F]{64}", result) is None:
        raise InvalidProvenanceError(f"{path} must be a 64-character SHA-256 hex digest")
    return result.lower()


def _validate_solref(value: Any, path: str) -> tuple:
    """Validate MuJoCo's complete two-value positive/direct solref vector."""

    result = _vector(value, path, 2)
    positive_format = result[0] > 0.0 and result[1] > 0.0
    direct_format = result[0] < 0.0 and result[1] < 0.0
    if not (positive_format or direct_format):
        raise InvalidOfficialFieldError(
            f"{path} must use two positive timeconst/dampratio values or two negative direct values"
        )
    return result


def _validate_solimp(value: Any, path: str) -> tuple:
    """Validate MuJoCo's complete ``[dmin, dmax, width, midpoint, power]``."""

    result = _vector(value, path, 5)
    dmin, dmax, width, midpoint, power = result
    if not (0.0 <= dmin < dmax <= 1.0):
        raise InvalidOfficialFieldError(f"{path}[0:2] must satisfy 0 <= dmin < dmax <= 1")
    if width <= 0.0:
        raise InvalidOfficialFieldError(f"{path}[2] width must be positive")
    if not 0.0 <= midpoint <= 1.0:
        raise InvalidOfficialFieldError(f"{path}[3] midpoint must be in [0, 1]")
    if power <= 0.0:
        raise InvalidOfficialFieldError(f"{path}[4] power must be positive")
    return result


def _validate_state_ids(value: Any, path: str) -> dict:
    if not isinstance(value, Mapping):
        raise InvalidOfficialFieldError(f"{path} must contain dev and official arrays")
    _unknown_fields(value, {"dev", "official"}, path)
    if set(value) != {"dev", "official"}:
        raise InvalidOfficialFieldError(f"{path} must contain exactly dev and official arrays")
    result = {}
    expected_lengths = {"dev": 10, "official": 400}
    all_ids = []
    for key, expected_length in expected_lengths.items():
        ids = value[key]
        if not isinstance(ids, (list, tuple)) or len(ids) != expected_length:
            raise InvalidOfficialFieldError(f"{path}.{key} must contain exactly {expected_length} IDs")
        normalised = tuple(_nonempty_string(item, f"{path}.{key}[{index}]") for index, item in enumerate(ids))
        if len(set(normalised)) != len(normalised):
            raise InvalidOfficialFieldError(f"{path}.{key} contains duplicate IDs")
        result[key] = normalised
        all_ids.extend(normalised)
    if len(set(all_ids)) != len(all_ids):
        raise InvalidOfficialFieldError(f"{path} contains IDs shared by dev and official sets")
    return result


def _validate_condim(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 3, 4, 6}:
        raise InvalidOfficialFieldError(f"{path} must be one of MuJoCo condim values 1, 3, 4, 6")
    return value


def _validate_gamma(value: Any, path: str) -> float:
    return _number(value, path, minimum=0.0, strict=True)


def _validate_timestep(value: Any, path: str) -> float:
    return _number(value, path, minimum=0.0, strict=True)


def _validate_id(value: Any, path: str) -> str:
    return _nonempty_string(value, path)


OFFICIAL_VALIDATORS = {
    "gamma_star": _validate_gamma,
    "physics_timestep_s": _validate_timestep,
    "deck_eq_solref": _validate_solref,
    "deck_eq_solimp": _validate_solimp,
    "deck_mass_kg": lambda value, path: _number(value, path, minimum=0.0, strict=True),
    "deck_inertia_kg_m2": lambda value, path: _vector(value, path, 3, positive=True),
    "isolator_fn_hz": lambda value, path: _vector(value, path, 6, positive=True),
    "isolator_zeta": lambda value, path: _vector(value, path, 6, positive=True),
    "isolator_k": lambda value, path: _vector(value, path, 6, positive=True),
    "isolator_c": lambda value, path: _vector(value, path, 6, positive=True),
    "contact_condim": _validate_condim,
    "contact_solref": _validate_solref,
    "contact_solimp": _validate_solimp,
    "gripper_actuator_force": lambda value, path: _number(value, path, minimum=0.0, strict=True),
    "osc_profile_id": _validate_id,
    "official_state_ids": _validate_state_ids,
}


def default_official_fields() -> dict:
    """Return a fresh draft map for fields that later phases must freeze."""

    return {name: UNFROZEN for name in OFFICIAL_FIELD_NAMES}


def default_official_provenance() -> dict:
    """Return a fresh draft map for all replay-affecting provenance values."""

    return {name: UNFROZEN for name in OFFICIAL_PROVENANCE_FIELD_NAMES}


def _validate_safety_limits(value: Any, path: str) -> dict:
    if not isinstance(value, Mapping):
        raise InvalidProvenanceError(f"{path} must be an object")
    allowed = {
        "max_displacement_m",
        "max_frequency_hz",
        "max_gamma",
        "solver_clearance_m",
        "solver_step_fraction",
        "allowed_solver_step_m",
        "min_samples_per_cycle",
        "max_angle_rad",
        "geometry",
    }
    _unknown_fields(value, allowed, path)
    required = {
        "max_displacement_m",
        "max_frequency_hz",
        "max_gamma",
        "solver_clearance_m",
        "solver_step_fraction",
    }
    missing = sorted(required - set(value))
    if missing:
        raise InvalidProvenanceError(f"{path} is missing required limit(s): {', '.join(missing)}")
    result = {
        "max_displacement_m": _number(value["max_displacement_m"], f"{path}.max_displacement_m", minimum=0.0),
        "max_frequency_hz": _number(value["max_frequency_hz"], f"{path}.max_frequency_hz", minimum=0.0, strict=True),
        "max_gamma": _number(value["max_gamma"], f"{path}.max_gamma", minimum=0.0),
        "solver_clearance_m": _number(
            value["solver_clearance_m"], f"{path}.solver_clearance_m", minimum=0.0, strict=True
        ),
        "solver_step_fraction": _number(
            value["solver_step_fraction"], f"{path}.solver_step_fraction", minimum=0.0, strict=True
        ),
    }
    if result["solver_step_fraction"] > 1.0:
        raise InvalidProvenanceError(f"{path}.solver_step_fraction must be <= 1")
    if "allowed_solver_step_m" in value:
        allowed_step = _number(
            value["allowed_solver_step_m"], f"{path}.allowed_solver_step_m", minimum=0.0, strict=True
        )
        expected_step = result["solver_clearance_m"] * result["solver_step_fraction"]
        if allowed_step != expected_step:
            raise InvalidProvenanceError(
                f"{path}.allowed_solver_step_m must equal solver_clearance_m * solver_step_fraction"
            )
        result["allowed_solver_step_m"] = allowed_step
    if "min_samples_per_cycle" in value:
        samples = value["min_samples_per_cycle"]
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise InvalidProvenanceError(f"{path}.min_samples_per_cycle must be a positive integer")
        result["min_samples_per_cycle"] = samples
    if "max_angle_rad" in value and value["max_angle_rad"] is not None:
        result["max_angle_rad"] = _number(value["max_angle_rad"], f"{path}.max_angle_rad", minimum=0.0)
    if "geometry" in value:
        if not isinstance(value["geometry"], Mapping):
            raise InvalidProvenanceError(f"{path}.geometry must be an object")
        result["geometry"] = _normalise_mapping(value["geometry"], f"{path}.geometry")
    return result


def validate_provenance(provenance: Mapping[str, Any], *, scoreable: bool = True) -> Mapping[str, JSONValue]:
    """Validate explicit replay provenance and its safety-profile limits."""

    if not isinstance(scoreable, bool):
        raise ShakeBenchConfigError("scoreable must be a boolean")
    normalised = _normalise_mapping(provenance, "provenance")
    _unknown_fields(normalised, set(OFFICIAL_PROVENANCE_FIELD_NAMES), "provenance")
    if scoreable:
        missing = [name for name in OFFICIAL_PROVENANCE_FIELD_NAMES if name not in normalised]
        if missing:
            raise UnfrozenOfficialFieldError(
                "Scoreable configuration is missing provenance field(s): " + ", ".join(missing)
            )
        unfrozen = []
        for name, value in normalised.items():
            unfrozen.extend(_find_unfrozen(value, f"provenance.{name}"))
        if unfrozen:
            raise UnfrozenOfficialFieldError(
                "Scoreable configuration contains UNFROZEN provenance field(s): " + ", ".join(unfrozen)
            )

    for name, value in normalised.items():
        if value == UNFROZEN:
            continue
        path = f"provenance.{name}"
        if name in {"excitation_profile_id", "authored_spectrum_version", "safety_profile_id", "freeze_status"}:
            if not isinstance(value, str) or not value.strip():
                raise InvalidProvenanceError(f"{path} must be a non-empty string")
            if name == "freeze_status" and value not in {"UNFROZEN", "candidate", "frozen"}:
                raise InvalidProvenanceError(f"{path} must be UNFROZEN, candidate, or frozen")
        elif name in {"excitation_profile_hash", "safety_profile_hash", "official_state_manifest_hash"}:
            _hash_string(value, path)
        elif name == "gravity_m_s2":
            _number(value, path, minimum=0.0, strict=True)
        elif name == "ramp_duration_s":
            _number(value, path, minimum=0.0, strict=True)
        elif name == "frequency_scale":
            _number(value, path, minimum=0.0, strict=True)
        elif name == "gamma_point_offset_m":
            _vector(value, path, 3, nonzero=True)
        elif name == "safety_profile_limits":
            _validate_safety_limits(value, path)
    return _freeze_json_value(normalised)


def validate_official_fields(official_fields: Mapping[str, Any], *, scoreable: bool = True) -> Mapping[str, JSONValue]:
    """Validate every official field and fail closed for scoreable configs."""

    if not isinstance(scoreable, bool):
        raise ShakeBenchConfigError("scoreable must be a boolean")
    normalised = _normalise_mapping(official_fields, "official_fields")
    _unknown_fields(normalised, set(OFFICIAL_FIELD_NAMES), "official_fields")
    if scoreable:
        missing = [name for name in OFFICIAL_FIELD_NAMES if name not in normalised]
        if missing:
            raise UnfrozenOfficialFieldError(
                "Scoreable configuration is missing official field(s): " + ", ".join(missing)
            )
        unfrozen = []
        for name, value in normalised.items():
            unfrozen.extend(_find_unfrozen(value, f"official_fields.{name}"))
        if unfrozen:
            raise UnfrozenOfficialFieldError(
                "Scoreable configuration contains UNFROZEN official field(s): " + ", ".join(unfrozen)
            )

    for name, value in normalised.items():
        if value == UNFROZEN:
            continue
        OFFICIAL_VALIDATORS[name](value, f"official_fields.{name}")
    return _freeze_json_value(normalised)


@dataclass(frozen=True)
class ExtensionIdentity:
    """Identity of the additive extension without changing the Python package.

    Attributes:
        extension_id: Stable extension identifier, always ``shakebench``.
        version: Extension schema/version string.
        python_package: Import package retained for robosuite compatibility.
    """

    extension_id: str = EXTENSION_ID
    version: str = EXTENSION_VERSION
    python_package: str = PYTHON_PACKAGE

    def __post_init__(self) -> None:
        if self.extension_id != EXTENSION_ID:
            raise ShakeBenchConfigError(f"extension_id must be {EXTENSION_ID!r}")
        if not isinstance(self.version, str) or not self.version:
            raise ShakeBenchConfigError("extension version must be a non-empty string")
        if self.python_package != PYTHON_PACKAGE:
            raise ShakeBenchConfigError(f"python_package must remain {PYTHON_PACKAGE!r}")

    def to_dict(self) -> dict:
        """Serialize the identity using the public JSON field names."""

        return {"id": self.extension_id, "version": self.version, "python_package": self.python_package}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExtensionIdentity":
        """Parse and strictly validate an extension identity mapping."""

        if not isinstance(payload, Mapping):
            raise ShakeBenchConfigError("extension must be an object")
        _unknown_fields(payload, {"id", "version", "python_package"}, "extension")
        return cls(
            extension_id=payload.get("id", EXTENSION_ID),
            version=payload.get("version", EXTENSION_VERSION),
            python_package=payload.get("python_package", PYTHON_PACKAGE),
        )


@dataclass(frozen=True)
class ShakeBenchConfig:
    """Closed Phase 01 configuration envelope.

    Attributes:
        schema_version: Configuration envelope version.
        extension: Additive ShakeBench extension identity.
        track: Currently supported benchmark track.
        scoreable: Whether freeze and structural gates must run.
        official_fields: Score-affecting values and their freeze state.
        provenance: Explicit replay-affecting profile and safety metadata.
        options: Draft-only extension payload; scoreable configs must leave it empty.
    """

    schema_version: int = SCHEMA_VERSION
    extension: ExtensionIdentity = field(default_factory=ExtensionIdentity)
    track: str = DEFAULT_TRACK
    scoreable: bool = False
    official_fields: Mapping[str, JSONValue] = field(default_factory=default_official_fields)
    provenance: Mapping[str, JSONValue] = field(default_factory=default_official_provenance)
    options: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ShakeBenchConfigError(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.extension, ExtensionIdentity):
            raise ShakeBenchConfigError("extension must be an ExtensionIdentity")
        if self.track != DEFAULT_TRACK:
            raise ShakeBenchConfigError(f"track must be {DEFAULT_TRACK!r} in Phase 01")
        if not isinstance(self.scoreable, bool):
            raise ShakeBenchConfigError("scoreable must be a boolean")

        official_fields = _normalise_mapping(self.official_fields, "official_fields")
        _unknown_fields(official_fields, set(OFFICIAL_FIELD_NAMES), "official_fields")
        provenance = _normalise_mapping(self.provenance, "provenance")
        _unknown_fields(provenance, set(OFFICIAL_PROVENANCE_FIELD_NAMES), "provenance")
        options = _normalise_mapping(self.options, "options")
        object.__setattr__(self, "official_fields", _freeze_json_value(official_fields))
        object.__setattr__(self, "provenance", _freeze_json_value(provenance))
        object.__setattr__(self, "options", _freeze_json_value(options))

    @property
    def is_scoreable(self) -> bool:
        """Return the declared scoreability flag."""

        return self.scoreable

    @property
    def extension_identity(self) -> ExtensionIdentity:
        """Return the additive extension identity."""

        return self.extension

    def validate(self) -> "ShakeBenchConfig":
        """Validate structure, provenance, hidden options, and freeze authority."""

        validate_official_fields(self.official_fields, scoreable=self.scoreable)
        validate_provenance(self.provenance, scoreable=self.scoreable)
        if self.scoreable and self.options:
            raise HiddenOfficialConfigurationError(
                "scoreable configuration must not hide replay-affecting values in options"
            )
        if self.scoreable and SCOREABLE_FREEZE_AUTHORITY is None:
            raise OfficialConfigurationNotReadyError(
                "Phase 01 has no official freeze authority; later physics/controller/protocol evidence is required"
            )
        return self

    def to_dict(self) -> dict:
        """Serialize the complete configuration into a mutable deep copy."""

        return {
            "schema_version": self.schema_version,
            "extension": self.extension.to_dict(),
            "track": self.track,
            "scoreable": self.scoreable,
            "official_fields": _thaw_json_value(self.official_fields),
            "provenance": _thaw_json_value(self.provenance),
            "options": _thaw_json_value(self.options),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize the envelope as deterministic JSON text."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
        )

    def canonical_json(self) -> str:
        """Return the stable, whitespace-free representation used for hashes."""

        return canonical_json(self)

    def config_hash(self) -> str:
        """Return the SHA-256 hash of :meth:`canonical_json`."""

        return config_hash(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShakeBenchConfig":
        """Parse a configuration mapping while rejecting unknown fields."""

        if not isinstance(payload, Mapping):
            raise ShakeBenchConfigError("configuration must be an object")
        _unknown_fields(
            payload,
            {"schema_version", "extension", "track", "scoreable", "official_fields", "provenance", "options"},
            "config",
        )
        return cls(
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            extension=ExtensionIdentity.from_dict(payload.get("extension", {})),
            track=payload.get("track", DEFAULT_TRACK),
            scoreable=payload.get("scoreable", False),
            official_fields=payload.get("official_fields", default_official_fields()),
            provenance=payload.get("provenance", default_official_provenance()),
            options=payload.get("options", {}),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ShakeBenchConfig":
        """Parse a JSON configuration string."""

        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ShakeBenchConfigError(f"invalid JSON: {exc.msg}") from exc
        return cls.from_dict(value)


def _coerce_config(config: Union[ShakeBenchConfig, Mapping[str, Any]]) -> ShakeBenchConfig:
    if isinstance(config, ShakeBenchConfig):
        return config
    if isinstance(config, Mapping):
        return ShakeBenchConfig.from_dict(config)
    raise ShakeBenchConfigError("expected ShakeBenchConfig or a configuration mapping")


def validate_config(config: Union[ShakeBenchConfig, Mapping[str, Any]]) -> ShakeBenchConfig:
    """Parse and validate a configuration, including scoreability gates."""

    return _coerce_config(config).validate()


def canonical_json(config: Union[ShakeBenchConfig, Mapping[str, Any]]) -> str:
    """Serialize a configuration deterministically for provenance hashing."""

    return _coerce_config(config).to_json(indent=None)


def config_hash(config: Union[ShakeBenchConfig, Mapping[str, Any]]) -> str:
    """Return the lower-case SHA-256 digest of canonical configuration JSON."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _json_value_schema() -> dict:
    return {
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "string"},
            {"type": "array", "items": {"$ref": "#/$defs/jsonValue"}},
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/jsonValue"}},
        ]
    }


def _sentinel_or(schema_definition: dict) -> dict:
    return {"oneOf": [{"const": UNFROZEN}, schema_definition]}


def _vector_schema(length: int) -> dict:
    return {"type": "array", "minItems": length, "maxItems": length, "items": {"type": "number"}}


def schema() -> dict:
    """Return a copy of the strict configuration JSON schema."""

    official_properties = {
        "gamma_star": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "physics_timestep_s": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "deck_eq_solref": _sentinel_or(_vector_schema(2)),
        "deck_eq_solimp": _sentinel_or(_vector_schema(5)),
        "deck_mass_kg": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "deck_inertia_kg_m2": _sentinel_or(_vector_schema(3)),
        "isolator_fn_hz": _sentinel_or(_vector_schema(6)),
        "isolator_zeta": _sentinel_or(_vector_schema(6)),
        "isolator_k": _sentinel_or(_vector_schema(6)),
        "isolator_c": _sentinel_or(_vector_schema(6)),
        "contact_condim": _sentinel_or({"type": "integer", "enum": [1, 3, 4, 6]}),
        "contact_solref": _sentinel_or(_vector_schema(2)),
        "contact_solimp": _sentinel_or(_vector_schema(5)),
        "gripper_actuator_force": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "osc_profile_id": _sentinel_or({"type": "string", "minLength": 1}),
        "official_state_ids": _sentinel_or(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["dev", "official"],
                "properties": {
                    "dev": {"type": "array", "minItems": 10, "maxItems": 10, "items": {"type": "string"}},
                    "official": {
                        "type": "array",
                        "minItems": 400,
                        "maxItems": 400,
                        "items": {"type": "string"},
                    },
                },
            }
        ),
    }
    provenance_properties = {
        "excitation_profile_id": _sentinel_or({"type": "string", "minLength": 1}),
        "excitation_profile_hash": _sentinel_or({"type": "string", "pattern": "^[0-9a-fA-F]{64}$"}),
        "authored_spectrum_version": _sentinel_or({"type": "string", "minLength": 1}),
        "gravity_m_s2": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "ramp_duration_s": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "frequency_scale": _sentinel_or({"type": "number", "exclusiveMinimum": 0}),
        "gamma_point_offset_m": _sentinel_or(_vector_schema(3)),
        "safety_profile_id": _sentinel_or({"type": "string", "minLength": 1}),
        "safety_profile_hash": _sentinel_or({"type": "string", "pattern": "^[0-9a-fA-F]{64}$"}),
        "safety_profile_limits": _sentinel_or(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "max_displacement_m",
                    "max_frequency_hz",
                    "max_gamma",
                    "solver_clearance_m",
                    "solver_step_fraction",
                ],
                "properties": {
                    "max_displacement_m": {"type": "number", "minimum": 0},
                    "max_frequency_hz": {"type": "number", "exclusiveMinimum": 0},
                    "max_gamma": {"type": "number", "minimum": 0},
                    "solver_clearance_m": {"type": "number", "exclusiveMinimum": 0},
                    "solver_step_fraction": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "allowed_solver_step_m": {"type": "number", "exclusiveMinimum": 0},
                    "min_samples_per_cycle": {"type": "integer", "minimum": 1},
                    "max_angle_rad": {"type": ["number", "null"], "minimum": 0},
                    "geometry": {"type": "object"},
                },
            }
        ),
        "official_state_manifest_hash": _sentinel_or({"type": "string", "pattern": "^[0-9a-fA-F]{64}$"}),
        "freeze_status": _sentinel_or({"type": "string", "enum": ["UNFROZEN", "candidate", "frozen"]}),
    }
    return copy.deepcopy(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": SCHEMA_ID,
            "title": "ShakeBench Phase 01 configuration",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "extension",
                "track",
                "scoreable",
                "official_fields",
                "provenance",
                "options",
            ],
            "properties": {
                "schema_version": {"const": SCHEMA_VERSION},
                "extension": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "version", "python_package"],
                    "properties": {
                        "id": {"const": EXTENSION_ID},
                        "version": {"type": "string", "minLength": 1},
                        "python_package": {"const": PYTHON_PACKAGE},
                    },
                },
                "track": {"const": DEFAULT_TRACK},
                "scoreable": {"type": "boolean"},
                "official_fields": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": official_properties,
                },
                "provenance": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(OFFICIAL_PROVENANCE_FIELD_NAMES),
                    "properties": provenance_properties,
                    "description": "Explicit replay provenance; UNFROZEN in Phase 01 draft mode.",
                },
                "options": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/jsonValue"},
                    "description": "Draft-only extension data; scoreable configurations must leave it empty.",
                },
            },
            "$defs": {"jsonValue": _json_value_schema()},
        }
    )


CONFIG_SCHEMA = schema()
Config = ShakeBenchConfig
ConfigError = ShakeBenchConfigError
hash_config = config_hash
serialize_config = canonical_json


__all__ = [
    "CONFIG_SCHEMA",
    "Config",
    "ConfigError",
    "DEFAULT_TRACK",
    "EXTENSION_ID",
    "EXTENSION_VERSION",
    "ExtensionIdentity",
    "HiddenOfficialConfigurationError",
    "InvalidOfficialFieldError",
    "InvalidProvenanceError",
    "JSONValue",
    "OFFICIAL_FIELD_NAMES",
    "OFFICIAL_FIELDS",
    "OFFICIAL_PROVENANCE_FIELD_NAMES",
    "OFFICIAL_PROVENANCE_FIELDS",
    "OfficialConfigurationNotReadyError",
    "PYTHON_PACKAGE",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCOREABLE_FREEZE_AUTHORITY",
    "ShakeBenchConfig",
    "ShakeBenchConfigError",
    "UNFROZEN",
    "UnknownFieldError",
    "UnfrozenOfficialFieldError",
    "canonical_json",
    "config_hash",
    "default_official_fields",
    "default_official_provenance",
    "hash_config",
    "schema",
    "serialize_config",
    "validate_config",
    "validate_official_fields",
    "validate_provenance",
]
