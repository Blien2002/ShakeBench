"""Bootstrap tests for the additive Phase 00 ShakeBench configuration layer."""

import json

import pytest

import robosuite as suite
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.utils.shakebench_config import (
    OFFICIAL_FIELD_NAMES,
    SCHEMA_VERSION,
    UNFROZEN,
    ExtensionIdentity,
    ShakeBenchConfig,
    UnfrozenOfficialFieldError,
    UnknownFieldError,
    config_hash,
    schema,
    validate_config,
)

EXPECTED_UPSTREAM_ENVIRONMENTS = {
    "Door",
    "Lift",
    "NutAssembly",
    "NutAssemblyRound",
    "NutAssemblySingle",
    "NutAssemblySquare",
    "PickPlace",
    "PickPlaceBread",
    "PickPlaceCan",
    "PickPlaceCereal",
    "PickPlaceMilk",
    "PickPlaceSingle",
    "Stack",
    "ToolHang",
    "TwoArmHandover",
    "TwoArmLift",
    "TwoArmPegInHole",
    "TwoArmTransport",
    "Wipe",
}


def test_robosuite_import_and_registry_unchanged():
    assert suite.__version__ == "1.5.2"
    assert set(REGISTERED_ENVS) == EXPECTED_UPSTREAM_ENVIRONMENTS
    assert set(suite.ALL_ENVIRONMENTS) == EXPECTED_UPSTREAM_ENVIRONMENTS
    assert "VibrationPickPlaceCan" not in REGISTERED_ENVS


def test_extension_identity_keeps_robosuite_package_name():
    identity = ExtensionIdentity()
    assert identity.extension_id == "shakebench"
    assert identity.python_package == "robosuite"
    assert identity.to_dict()["id"] == "shakebench"


def test_schema_round_trip_and_stable_hash():
    first = ShakeBenchConfig(options={"z": 1, "a": {"enabled": True}})
    second = ShakeBenchConfig(options={"a": {"enabled": True}, "z": 1})

    restored = ShakeBenchConfig.from_json(first.to_json())
    assert restored == first
    assert first.canonical_json() == second.canonical_json()
    assert config_hash(first) == config_hash(second)
    assert first.schema_version == SCHEMA_VERSION


def test_unknown_fields_are_rejected():
    payload = ShakeBenchConfig().to_dict()
    payload["unexpected"] = True
    with pytest.raises(UnknownFieldError, match="unexpected"):
        ShakeBenchConfig.from_dict(payload)

    nested_payload = ShakeBenchConfig().to_dict()
    nested_payload["official_fields"]["future_field"] = 1
    with pytest.raises(UnknownFieldError, match="future_field"):
        ShakeBenchConfig.from_dict(nested_payload)


def test_unfrozen_scoreable_config_is_rejected():
    draft = ShakeBenchConfig(scoreable=True)
    with pytest.raises(UnfrozenOfficialFieldError, match="gamma_star"):
        validate_config(draft)


def test_frozen_scoreable_config_is_accepted():
    frozen_fields = {name: 0 for name in OFFICIAL_FIELD_NAMES}
    config = ShakeBenchConfig(scoreable=True, official_fields=frozen_fields)
    assert validate_config(config) == config


def test_non_scoreable_draft_preserves_unfrozen_marker_and_schema():
    config = ShakeBenchConfig()
    assert config.official_fields["gamma_star"] == UNFROZEN
    assert schema()["properties"]["official_fields"]["additionalProperties"] is False
    assert json.loads(config.canonical_json()) == config.to_dict()
