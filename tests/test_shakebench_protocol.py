"""Pure contract tests for Phase 08 committed-state authorities."""

import json

from robosuite import models
from robosuite.utils.shakebench_oracle import OracleControllerProfile
from robosuite.utils.shakebench_outcomes import outcome_contract_sha256
from robosuite.utils.shakebench_committed_states import (
    KNEE_STATE_COUNT,
    KNEE_STATE_FILENAME,
    OFFICIAL_STATE_COUNT,
    OFFICIAL_STATE_FILENAME,
    build_committed_state_artifact,
    verify_committed_state_artifact,
    verify_committed_state_pair,
    verify_phase07_5a_evidence_binding,
)
from robosuite.scripts.shakebench_run_oracle import load_state_asset


def test_frozen_assets_have_exact_counts_and_are_pairwise_disjoint():
    root = models.assets_root
    verdict = verify_committed_state_pair(root + "/" + OFFICIAL_STATE_FILENAME, root + "/" + KNEE_STATE_FILENAME)
    assert verdict["passed"], verdict["errors"]
    assert verdict["official"]["state_count"] == OFFICIAL_STATE_COUNT
    assert verdict["knee"]["state_count"] == KNEE_STATE_COUNT


def test_regeneration_is_exact_and_never_mutates_the_ten_dev_states():
    root = models.assets_root
    official = json.loads(open(root + "/" + OFFICIAL_STATE_FILENAME, encoding="utf-8").read())
    knee = json.loads(open(root + "/" + KNEE_STATE_FILENAME, encoding="utf-8").read())
    assert official == build_committed_state_artifact("official")
    assert knee == build_committed_state_artifact("knee")
    assert all(state["state_id"].startswith("shakebench-official-v1-") for state in official["states"])
    assert all(state["state_id"].startswith("shakebench-knee-v1-") for state in knee["states"])


def test_committed_states_bind_the_current_controller_and_outcome_contract():
    root = models.assets_root
    controller_sha256 = OracleControllerProfile().sha256
    for filename in (OFFICIAL_STATE_FILENAME, KNEE_STATE_FILENAME):
        payload = json.loads(open(root + "/" + filename, encoding="utf-8").read())
        assert payload["authority_hashes"]["controller_profile_sha256"] == controller_sha256
        assert payload["authority_hashes"]["outcome_contract_sha256"] == outcome_contract_sha256()


def test_future_outcomes_and_hash_changes_fail_closed(tmp_path):
    source = json.loads(open(models.assets_root + "/" + KNEE_STATE_FILENAME, encoding="utf-8").read())
    source["states"][0]["success"] = True
    target = tmp_path / "mutated.json"
    target.write_text(json.dumps(source), encoding="utf-8")
    verdict = verify_committed_state_artifact(target, expected_split="knee")
    assert not verdict["passed"]
    assert "forbidden future-outcome field" in verdict["errors"]


def test_committed_state_loader_keeps_authority_and_execution_payload():
    """Runner-facing loader must not silently resolve an official ID through dev."""

    source = models.assets_root + "/" + OFFICIAL_STATE_FILENAME
    loaded = load_state_asset(source)
    state = loaded["states"][0]
    assert loaded["authority"]["kind"] == "committed"
    assert state["state_id"].startswith("shakebench-official-v1-")
    assert {"excitation_seed", "imu_seed", "t0_s", "authority_hashes", "canonical_payload_sha256"} <= set(state)


def test_phase07_5a_evidence_binding_is_package_owned_and_authenticated():
    binding = verify_phase07_5a_evidence_binding()
    assert binding["evidence_core_payload_sha256"] == (
        "9b2c56416de2d88d78062b698004605c551bac37671d10a3600fccec1eb4f2dd"
    )
