"""Fail-closed tests for direct-mount authorization independent of geometry flags."""

import json

import pytest

from robosuite import models
from robosuite.utils.shakebench_authority import (
    DirectMountAuthorityError,
    authority_payload_hash,
    verify_direct_mount_authority,
    verify_final_phase08_authority,
)
from robosuite.utils.shakebench_geometry import load_geometry_profile


def test_science_authority_alone_does_not_grant_phase08():
    authority = verify_direct_mount_authority()
    assert authority["scoreable"] is True
    assert authority["phase08_authorized"] is False
    with pytest.raises(DirectMountAuthorityError, match="evidence-manifest binding"):
        verify_final_phase08_authority("0" * 64)


def test_final_authority_accepts_only_the_bound_evidence_core():
    source = models.assets_root + "/shakebench_phase07_5a_final_authority.json"
    evidence_core_sha256 = json.loads(open(source, encoding="utf-8").read())["final_evidence_manifest_sha256"]
    final = verify_final_phase08_authority(evidence_core_sha256)
    assert final["phase08_authorized"] is True


@pytest.mark.parametrize("field", ("status", "phase08_authorized", "scene_sha256", "geometry_payload_sha256"))
def test_authority_mutation_fails_even_if_its_json_is_repacked(tmp_path, field):
    source = models.assets_root + "/shakebench_phase07_5a_requalification.json"
    payload = json.loads(open(source, encoding="utf-8").read())
    payload[field] = "PASS" if field == "status" else (True if field == "phase08_authorized" else "0" * 64)
    target = tmp_path / "authority.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DirectMountAuthorityError):
        verify_direct_mount_authority(target)


def test_geometry_cannot_carry_scoreable_authorization():
    geometry = load_geometry_profile("direct_mount_v1")
    assert "scoreable" not in geometry


def test_final_authority_cannot_authorize_a_different_manifest(tmp_path):
    source = models.assets_root + "/shakebench_phase07_5a_final_authority.json"
    payload = json.loads(open(source, encoding="utf-8").read())
    payload["status"] = "PASS"
    payload["phase08_authorized"] = True
    payload["final_evidence_manifest_sha256"] = "1" * 64
    payload["payload_sha256"] = authority_payload_hash(payload)
    target = tmp_path / "final_authority.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DirectMountAuthorityError, match="evidence-manifest binding"):
        verify_final_phase08_authority("2" * 64, target)


def test_final_authority_unknown_field_fails_closed(tmp_path):
    source = models.assets_root + "/shakebench_phase07_5a_final_authority.json"
    payload = json.loads(open(source, encoding="utf-8").read())
    payload["unexpected"] = True
    target = tmp_path / "final_authority.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DirectMountAuthorityError, match="fields mismatch"):
        verify_final_phase08_authority("0" * 64, target)
