"""Typed configuration and provenance helpers for the ShakeBench extension.

Phase 00 deliberately keeps this module simulator-independent.  It defines the
configuration boundary that later phases can extend without changing the
``robosuite`` package identity or mutating global simulation settings at import
time.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Final, TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


SCHEMA_ID: Final = "shakebench.config"
SCHEMA_VERSION: Final = 1
EXTENSION_ID: Final = "shakebench"
EXTENSION_VERSION: Final = "0"
PYTHON_PACKAGE: Final = "robosuite"
DEFAULT_TRACK: Final = "state_oracle_control"

# A string sentinel is used instead of ``None`` so that an explicitly
# unselected official value cannot be confused with a legitimate nullable
# metadata field.  It is only meaningful inside ``official_fields``.
UNFROZEN: Final = "UNFROZEN"

# These names are the score-affecting boundaries already identified by the
# v0 design.  Their values are intentionally left UNFROZEN until the relevant
# physics, controller, and state-protocol phases produce evidence.
OFFICIAL_FIELD_NAMES: Final[tuple[str, ...]] = (
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
OFFICIAL_FIELDS: Final[frozenset[str]] = frozenset(OFFICIAL_FIELD_NAMES)


class ShakeBenchConfigError(ValueError):
    """Base error for malformed or non-scoreable ShakeBench configurations."""


class UnknownFieldError(ShakeBenchConfigError):
    """Raised when a mapping contains a key outside the declared schema."""


class UnfrozenOfficialFieldError(ShakeBenchConfigError):
    """Raised when a scoreable configuration still contains ``UNFROZEN``."""


def _unknown_fields(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
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
        normalised: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ShakeBenchConfigError(f"Object key at {path} must be a string; got {type(key).__name__}")
            normalised[key] = _normalise_json_value(item, f"{path}.{key}")
        return normalised
    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ShakeBenchConfigError(f"Value at {path} is not JSON-compatible: {type(value).__name__}")


def _normalise_mapping(value: Mapping[str, Any], path: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise ShakeBenchConfigError(f"{path} must be an object")
    normalised = _normalise_json_value(value, path)
    assert isinstance(normalised, dict)
    return normalised


def _find_unfrozen(value: JSONValue, path: str) -> list[str]:
    """Return paths of every nested ``UNFROZEN`` sentinel."""

    if isinstance(value, str):
        return [path] if value == UNFROZEN else []
    if isinstance(value, list):
        paths: list[str] = []
        for index, item in enumerate(value):
            paths.extend(_find_unfrozen(item, f"{path}[{index}]"))
        return paths
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            paths.extend(_find_unfrozen(item, f"{path}.{key}"))
        return paths
    return []


def default_official_fields() -> dict[str, JSONValue]:
    """Return a fresh draft map for fields that later phases must freeze."""

    return {name: UNFROZEN for name in OFFICIAL_FIELD_NAMES}


def validate_official_fields(official_fields: Mapping[str, Any], *, scoreable: bool = True) -> Mapping[str, JSONValue]:
    """Validate the official-field map and fail closed for scoreable configs.

    Draft and development configurations may retain ``UNFROZEN`` values.  A
    configuration marked ``scoreable=True`` may not contain the sentinel,
    including inside nested arrays or objects.
    """

    if not isinstance(scoreable, bool):
        raise ShakeBenchConfigError("scoreable must be a boolean")
    normalised = _normalise_mapping(official_fields, "official_fields")
    _unknown_fields(normalised, set(OFFICIAL_FIELD_NAMES), "official_fields")
    if scoreable:
        missing_fields = [name for name in OFFICIAL_FIELD_NAMES if name not in normalised]
        if missing_fields:
            raise UnfrozenOfficialFieldError(
                "Scoreable configuration is missing official field(s): " + ", ".join(missing_fields)
            )
        unfrozen_paths: list[str] = []
        for name, value in normalised.items():
            unfrozen_paths.extend(_find_unfrozen(value, f"official_fields.{name}"))
        if unfrozen_paths:
            raise UnfrozenOfficialFieldError(
                "Scoreable configuration contains UNFROZEN official field(s): " + ", ".join(unfrozen_paths)
            )
    return normalised


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

    def to_dict(self) -> dict[str, str]:
        """Serialize the identity using the public JSON field names."""

        return {
            "id": self.extension_id,
            "version": self.version,
            "python_package": self.python_package,
        }

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
    """Phase 00 configuration envelope.

    ``official_fields`` intentionally defaults to a complete draft map.  This
    makes an accidental transition to scoreable mode fail closed until every
    score-affecting value has been frozen by a later phase.

    Attributes:
        schema_version: Configuration envelope version.
        extension: Additive ShakeBench extension identity.
        track: Currently supported benchmark track.
        scoreable: Whether all official fields must be frozen.
        official_fields: Score-affecting values and their freeze state.
        options: Extensible non-official configuration payload.
    """

    schema_version: int = SCHEMA_VERSION
    extension: ExtensionIdentity = field(default_factory=ExtensionIdentity)
    track: str = DEFAULT_TRACK
    scoreable: bool = False
    official_fields: Mapping[str, JSONValue] = field(default_factory=default_official_fields)
    options: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ShakeBenchConfigError(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.extension, ExtensionIdentity):
            raise ShakeBenchConfigError("extension must be an ExtensionIdentity")
        if self.track != DEFAULT_TRACK:
            raise ShakeBenchConfigError(f"track must be {DEFAULT_TRACK!r} in Phase 00")
        if not isinstance(self.scoreable, bool):
            raise ShakeBenchConfigError("scoreable must be a boolean")

        official_fields = _normalise_mapping(self.official_fields, "official_fields")
        _unknown_fields(official_fields, set(OFFICIAL_FIELD_NAMES), "official_fields")
        options = _normalise_mapping(self.options, "options")
        object.__setattr__(self, "official_fields", official_fields)
        object.__setattr__(self, "options", options)

    @property
    def is_scoreable(self) -> bool:
        """Alias useful to callers that prefer a predicate-style name."""

        return self.scoreable

    @property
    def extension_identity(self) -> ExtensionIdentity:
        """Explicitly named alias for the extension identity."""

        return self.extension

    def validate(self) -> "ShakeBenchConfig":
        """Validate this envelope for the declared scoreability mode."""

        validate_official_fields(self.official_fields, scoreable=self.scoreable)
        return self

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the complete configuration envelope."""

        return {
            "schema_version": self.schema_version,
            "extension": self.extension.to_dict(),
            "track": self.track,
            "scoreable": self.scoreable,
            "official_fields": copy.deepcopy(dict(self.official_fields)),
            "options": copy.deepcopy(dict(self.options)),
        }

    def to_json(self, *, indent: int | None = None) -> str:
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
            {"schema_version", "extension", "track", "scoreable", "official_fields", "options"},
            "config",
        )
        return cls(
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            extension=ExtensionIdentity.from_dict(payload.get("extension", {})),
            track=payload.get("track", DEFAULT_TRACK),
            scoreable=payload.get("scoreable", False),
            official_fields=payload.get("official_fields", default_official_fields()),
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


def _coerce_config(config: ShakeBenchConfig | Mapping[str, Any]) -> ShakeBenchConfig:
    if isinstance(config, ShakeBenchConfig):
        return config
    if isinstance(config, Mapping):
        return ShakeBenchConfig.from_dict(config)
    raise ShakeBenchConfigError("expected ShakeBenchConfig or a configuration mapping")


def validate_config(config: ShakeBenchConfig | Mapping[str, Any]) -> ShakeBenchConfig:
    """Parse and validate a configuration, including scoreability gates."""

    return _coerce_config(config).validate()


def canonical_json(config: ShakeBenchConfig | Mapping[str, Any]) -> str:
    """Serialize a configuration deterministically for provenance hashing."""

    return _coerce_config(config).to_json(indent=None)


def config_hash(config: ShakeBenchConfig | Mapping[str, Any]) -> str:
    """Return the lower-case SHA-256 digest of the canonical configuration."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _json_value_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "string"},
            {"type": "array", "items": {"$ref": "#/$defs/jsonValue"}},
            {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/jsonValue"},
            },
        ]
    }


def schema() -> dict[str, Any]:
    """Return a copy of the JSON schema advertised by the CLI."""

    return copy.deepcopy(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": SCHEMA_ID,
            "title": "ShakeBench Phase 00 configuration",
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "extension", "track", "scoreable", "official_fields", "options"],
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
                    "properties": {name: {"$ref": "#/$defs/jsonValue"} for name in OFFICIAL_FIELD_NAMES},
                },
                "options": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/jsonValue"},
                },
            },
            "$defs": {"jsonValue": _json_value_schema()},
        }
    )


# Public aliases keep the small API discoverable without introducing a second
# schema or a second package identity.  ``CONFIG_SCHEMA`` is a snapshot for
# callers that expect a constant mapping; ``schema()`` returns a safe copy.
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
    "JSONValue",
    "OFFICIAL_FIELD_NAMES",
    "OFFICIAL_FIELDS",
    "PYTHON_PACKAGE",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ShakeBenchConfig",
    "ShakeBenchConfigError",
    "UNFROZEN",
    "UnknownFieldError",
    "UnfrozenOfficialFieldError",
    "canonical_json",
    "config_hash",
    "default_official_fields",
    "hash_config",
    "schema",
    "serialize_config",
    "validate_config",
    "validate_official_fields",
]
