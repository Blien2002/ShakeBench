"""Authenticated Phase 06F publication and Phase 07 handoff tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from robosuite.models import assets_root
from robosuite.scripts.shakebench_finalize_physics import (
    load_protocol,
    verify_final_publication,
)
from robosuite.utils.shakebench_physics_finalizer import (
    artifact_hash,
    validate_finalization_protocol,
    verify_official_publication_bundle,
)
from tests.shakebench_test_helpers import evidence_asset, evidence_root

ASSETS = Path(assets_root)


def test_registered_protocol_has_no_concrete_winner_dependent_contact_id():
    protocol, _, _ = load_protocol()
    structure = validate_finalization_protocol(protocol)
    assert structure["candidate_ids"] == [
        "c3_nominal",
        "n6_critical_negative_control",
        "n6_overdamped_2",
        "n6_overdamped_4",
    ]
    assert structure["dependent_template_count"] == 14
    for row in protocol["dependent_manifest_template"]:
        assert row["contact_binding"] == "selection.contact_winner"
        assert "{contact_candidate_id}" in row["state_id_template"]


def test_published_bundle_and_independent_numeric_verifier_pass():
    evidence_asset("shakebench_phase_06f_raw_contact_c3_nominal.json")
    root = evidence_root()
    bundle = verify_official_publication_bundle(root)
    assert bundle["passed"], bundle["errors"]
    verified = verify_final_publication(root)
    assert verified["passed"], verified["errors"]
    assert verified["contact_winner"] == "c3_nominal"
    assert verified["eligible_contacts"][0] == "c3_nominal"


def test_handoff_tampering_is_detected_without_mujoco(tmp_path):
    names = [
        "shakebench_phase_06_final_status.json",
        "shakebench_phase_06_to_07_handoff.json",
        "shakebench_phase_06f_contact_selection.json",
        "shakebench_phase_06f_dependent_manifest.json",
        "shakebench_phase_06f_protocol.yaml",
        "shakebench_official_physics.yaml",
    ]
    root = evidence_root()
    handoff = json.loads((root / names[1]).read_text(encoding="utf-8"))
    names.extend(row["path"] for row in handoff["contact_evidence"])
    names.extend(row["path"] for row in handoff["dependent_evidence"])
    for name in names:
        (tmp_path / name).write_bytes(evidence_asset(name).read_bytes())
    tampered = copy.deepcopy(handoff)
    tampered["selected_profile"]["sha256"] = "0" * 64
    tampered["payload_sha256"] = artifact_hash(tampered)
    (tmp_path / names[1]).write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    result = verify_official_publication_bundle(tmp_path)
    assert result["passed"] is False
    assert any("profile/status/handoff" in error for error in result["errors"])
