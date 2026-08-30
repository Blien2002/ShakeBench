"""Bootstrap tests for the additive Phase 00 ShakeBench configuration layer."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

import robosuite as suite
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.utils.shakebench_config import (
    OFFICIAL_FIELD_NAMES,
    OFFICIAL_PROVENANCE_FIELD_NAMES,
    SCHEMA_VERSION,
    UNFROZEN,
    ExtensionIdentity,
    OfficialConfigurationNotReadyError,
    ShakeBenchConfig,
    ShakeBenchConfigError,
    UnfrozenOfficialFieldError,
    UnknownFieldError,
    config_hash,
    default_official_provenance,
    schema,
    validate_config,
    validate_official_fields,
    validate_provenance,
)
from robosuite.utils.shakebench_excitation import (
    AUTHORED_PROFILE_ID,
    AUTHORED_SPECTRUM_VERSION,
    DEFAULT_EXCITATION_CONFIG,
    excitation_profile_hash,
)
from robosuite.utils.shakebench_safety import DEFAULT_SAFETY_LIMITS, SAFETY_PROFILE_ID, safety_profile_hash

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


def _structurally_valid_official_fields():
    return {
        "gamma_star": 0.25,
        "physics_timestep_s": 0.001,
        "deck_eq_solref": [0.01, 1.0],
        "deck_eq_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "deck_mass_kg": 32.0,
        "deck_inertia_kg_m2": [0.9696, 1.1363, 2.0867],
        "isolator_fn_hz": [1.0] * 6,
        "isolator_zeta": [0.7] * 6,
        "isolator_k": [100.0] * 6,
        "isolator_c": [10.0] * 6,
        "contact_condim": 3,
        "contact_solref": [0.01, 1.0],
        "contact_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "gripper_actuator_force": 40.0,
        "osc_profile_id": "osc_pose_candidate",
        "official_state_ids": {
            "dev": [f"dev-{index:02d}" for index in range(10)],
            "official": [f"official-{index:03d}" for index in range(400)],
        },
    }


def _structurally_valid_provenance(freeze_status="candidate"):
    return {
        "excitation_profile_id": "shakebench.authored_v0_candidate",
        "excitation_profile_hash": "a" * 64,
        "authored_spectrum_version": "candidate-1",
        "gravity_m_s2": 9.81,
        "ramp_duration_s": 0.5,
        "frequency_scale": 1.0,
        "gamma_point_offset_m": [0.65, 0.0, 0.0],
        "safety_profile_id": "shakebench.safety.candidate.v1",
        "safety_profile_hash": "b" * 64,
        "safety_profile_limits": {
            "max_displacement_m": 0.025,
            "max_frequency_hz": 8.87,
            "max_gamma": 1.0,
            "solver_clearance_m": 0.008,
            "solver_step_fraction": 0.25,
        },
        "official_state_manifest_hash": "c" * 64,
        "freeze_status": freeze_status,
    }


def test_unknown_fields_are_rejected():
    payload = ShakeBenchConfig().to_dict()
    payload["unexpected"] = True
    with pytest.raises(UnknownFieldError, match="unexpected"):
        ShakeBenchConfig.from_dict(payload)

    nested_payload = ShakeBenchConfig().to_dict()
    nested_payload["official_fields"]["future_field"] = 1
    with pytest.raises(UnknownFieldError, match="future_field"):
        ShakeBenchConfig.from_dict(nested_payload)

    provenance_payload = ShakeBenchConfig().to_dict()
    provenance_payload["provenance"]["not_declared"] = True
    with pytest.raises(UnknownFieldError, match="not_declared"):
        ShakeBenchConfig.from_dict(provenance_payload)


def test_config_payload_is_deeply_immutable_and_hash_stable():
    config = ShakeBenchConfig(
        official_fields={"isolator_k": [100.0] * 6},
        options={"nested": {"values": [1, 2, 3]}},
    )
    before = config.config_hash()
    with pytest.raises(TypeError):
        config.official_fields["gamma_star"] = 1.0
    with pytest.raises(TypeError):
        config.options["anything"] = 1
    with pytest.raises(TypeError):
        config.official_fields["isolator_k"][0] = 1.0
    with pytest.raises(TypeError):
        config.options["nested"]["values"][0] = 99
    assert config.config_hash() == before
    mutable_copy = config.to_dict()
    mutable_copy["official_fields"]["isolator_k"][0] = 999.0
    assert config.official_fields["isolator_k"][0] == 100.0


def test_official_field_shapes_ranges_and_solimp_are_validated():
    valid = _structurally_valid_official_fields()
    assert validate_official_fields(valid, scoreable=False)["deck_eq_solimp"] == (0.9, 0.95, 0.001, 0.5, 2.0)
    for name, invalid in (
        ("deck_eq_solref", [0.01]),
        ("deck_eq_solimp", [0.0] * 5),
        ("deck_eq_solimp", [0.9, 0.95, -0.1, 0.5, 2.0]),
        ("isolator_k", [100.0] * 5),
        ("isolator_zeta", [0.0] * 6),
        ("official_state_ids", {"dev": [], "official": []}),
    ):
        invalid_fields = dict(valid)
        invalid_fields[name] = invalid
        with pytest.raises(ShakeBenchConfigError, match=name):
            validate_official_fields(invalid_fields, scoreable=False)


def test_provenance_is_explicit_and_candidate_limits_are_validated():
    provenance = _structurally_valid_provenance()
    assert set(provenance) == set(OFFICIAL_PROVENANCE_FIELD_NAMES)
    assert validate_provenance(provenance, scoreable=False)["frequency_scale"] == 1.0
    invalid = dict(provenance)
    invalid["safety_profile_limits"] = dict(provenance["safety_profile_limits"])
    invalid["safety_profile_limits"]["solver_step_fraction"] = 0.0
    with pytest.raises(ShakeBenchConfigError, match="solver_step_fraction"):
        validate_provenance(invalid, scoreable=False)


def test_candidate_profile_and_safety_hashes_are_in_explicit_provenance():
    fixture = json.loads((Path(__file__).with_name("golden_shakebench_excitation_v0.json")).read_text(encoding="utf-8"))
    provenance = fixture["configuration_envelope"]["provenance"]
    assert provenance["excitation_profile_id"] == AUTHORED_PROFILE_ID
    assert provenance["authored_spectrum_version"] == AUTHORED_SPECTRUM_VERSION
    assert provenance["excitation_profile_hash"] == excitation_profile_hash(DEFAULT_EXCITATION_CONFIG)
    assert provenance["safety_profile_id"] == SAFETY_PROFILE_ID
    assert provenance["safety_profile_hash"] == safety_profile_hash(DEFAULT_SAFETY_LIMITS)
    assert provenance["safety_profile_limits"]["solver_clearance_m"] == 0.008


def test_unfrozen_scoreable_config_is_rejected():
    draft = ShakeBenchConfig(scoreable=True)
    with pytest.raises(UnfrozenOfficialFieldError, match="gamma_star"):
        validate_config(draft)


def test_frozen_scoreable_config_is_accepted():
    zero_fields = {name: 0 for name in OFFICIAL_FIELD_NAMES}
    config = ShakeBenchConfig(
        scoreable=True,
        official_fields=zero_fields,
        provenance=_structurally_valid_provenance(),
    )
    with pytest.raises(ShakeBenchConfigError):
        validate_config(config)


def test_structurally_valid_scoreable_config_still_fails_without_freeze_authority():
    config = ShakeBenchConfig(
        scoreable=True,
        official_fields=_structurally_valid_official_fields(),
        provenance=_structurally_valid_provenance(),
    )
    with pytest.raises(OfficialConfigurationNotReadyError, match="Phase 01"):
        validate_config(config)


def test_scoreable_config_cannot_hide_values_in_options():
    config = ShakeBenchConfig(
        scoreable=True,
        official_fields=_structurally_valid_official_fields(),
        provenance=_structurally_valid_provenance(),
        options={"hidden_gamma": 0.5},
    )
    with pytest.raises(ShakeBenchConfigError, match="options"):
        validate_config(config)


def test_non_scoreable_draft_preserves_unfrozen_marker_and_schema():
    config = ShakeBenchConfig()
    assert config.official_fields["gamma_star"] == UNFROZEN
    assert config.provenance["excitation_profile_hash"] == UNFROZEN
    assert default_official_provenance()["safety_profile_hash"] == UNFROZEN
    assert schema()["properties"]["official_fields"]["additionalProperties"] is False
    assert json.loads(config.canonical_json()) == config.to_dict()


def test_shakebench_texture_is_a_packaged_repo_asset():
    asset = Path(__file__).parents[1] / "robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg"
    assert asset.is_file()
    assert sha256(asset.read_bytes()).hexdigest() == "6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb"
    manifest = (asset.parents[4] / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include robosuite/models/assets/ *" in manifest
