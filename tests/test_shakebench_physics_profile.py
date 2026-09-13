"""Official physics profile and current runtime asset contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from robosuite.models import assets_root
from robosuite.utils.shakebench_physics import (
    OFFICIAL_PHYSICS_PROFILE_FILENAME,
    PhysicsProfileIntegrityError,
    load_official_physics_profile,
    make_probe_physics_profile,
    physics_profile_hash,
)
from robosuite.utils.shakebench_runtime_verifier import (
    RUNTIME_CONTRACT_FILENAME,
    verify_runtime_publication_bundle,
)

ASSETS = Path(assets_root)


def test_packaged_official_profile_loads_and_is_scoreable():
    profile = load_official_physics_profile()
    assert profile.profile_id == "shakebench.official.physics.v2"
    assert profile.status == "official_immutable"
    assert profile.scoreable is True
    assert profile.profile_sha256 == physics_profile_hash(profile.to_dict())


def test_runtime_contract_matches_packaged_assets():
    result = verify_runtime_publication_bundle(ASSETS)
    assert result["passed"] is True, result["errors"]
    assert result["profile_sha256"] == load_official_physics_profile().profile_sha256


def test_runtime_contract_fails_closed_without_its_assets(tmp_path):
    (tmp_path / RUNTIME_CONTRACT_FILENAME).write_text(
        (ASSETS / RUNTIME_CONTRACT_FILENAME).read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = verify_runtime_publication_bundle(tmp_path)
    assert result["passed"] is False
    assert result["errors"]


def test_external_scoreable_profile_path_is_rejected(tmp_path):
    external = tmp_path / OFFICIAL_PHYSICS_PROFILE_FILENAME
    external.write_text((ASSETS / OFFICIAL_PHYSICS_PROFILE_FILENAME).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(PhysicsProfileIntegrityError, match="packaged canonical asset"):
        load_official_physics_profile(external)


def test_probe_profile_remains_explicitly_non_scoreable():
    profile = make_probe_physics_profile()
    assert profile.scoreable is False
    assert profile.status == "probe_non_scoreable"


def test_profile_assets_remain_flat_and_packaged():
    assert (ASSETS / OFFICIAL_PHYSICS_PROFILE_FILENAME).is_file()
    assert not (ASSETS / "shakebench").exists()
    manifest = Path(__file__).parents[1] / "MANIFEST.in"
    assert "recursive-include robosuite/models/assets/ *" in manifest.read_text(encoding="utf-8")
