"""Phase 06F winner-dependent finalization contracts."""

from __future__ import annotations

import copy

import pytest

from robosuite.utils.shakebench_physics_finalizer import (
    BINDING_RULE,
    BlockedResult,
    PublicationPlan,
    artifact_hash,
    finalize_official_physics,
    materialize_dependent_manifest,
    validate_finalization_protocol,
)


CANDIDATES = (
    "c3_nominal",
    "n6_critical_negative_control",
    "n6_overdamped_2",
    "n6_overdamped_4",
)


def _protocol() -> dict:
    candidate_rows = []
    for index, candidate_id in enumerate(CANDIDATES):
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "condim": 3 if candidate_id == "c3_nominal" else 6,
                "sliding_mu": {"table_object": 0.30, "finger_object": 1.00},
                "torsional_mu": 0.005,
                "rolling_mu": 0.001,
                "margin_m": 0.0,
                "gap_m": 0.0,
                "solref": [0.0004, 1.0 if index < 2 else float(index)],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
                "pair_scope": "explicit_can_pairs",
            }
        )
    templates = []
    for group in (
        "gamma_zero_parity",
        "driver",
        "isolator",
        "contact",
        "gamma_zero_parity_replay",
        "selected_diagnostics",
    ):
        templates.append(
            {
                "group": group,
                "contact_binding": "selection.contact_winner",
                "state_id_template": group
                + ".{contact_candidate_id}.{selection_artifact_sha256}",
                "output_template": "phase06f_"
                + group
                + "_{contact_candidate_id}_{selection_artifact_sha256}.json",
            }
        )
    return {
        "schema_id": "shakebench.phase06f.physics_selection_protocol",
        "schema_version": 1,
        "phase": "06F",
        "status": "pre_registered",
        "immutable_after_registration": True,
        "selection_authority": "physics_probes_only",
        "protocol_filename": "shakebench_phase_06f_protocol.yaml",
        "registration_commit": "test-registration",
        "contact_candidates": candidate_rows,
        "selection": {"minimal_intervention_priority": list(CANDIDATES), "binding_rule": BINDING_RULE},
        "dependent_manifest_template": templates,
        "official_profile": {
            "profile_id": "shakebench.official.physics.v2",
            "physics_template": {"excitation": {}, "timestep": {}, "scheduler": {}, "deck": {}, "isolator": {}},
        },
        "phase07_handoff": {"schema_id": "shakebench.phase06.to_phase07_handoff", "package_hashes": {}},
    }


class SyntheticStore:
    protocol_sha256 = "a" * 64

    def __init__(self, eligible: set[str]):
        self.eligible = eligible
        self.physics_calls = 0
        self.selections = []
        self.manifests = []

    def verify_inherited(self, protocol):
        self.physics_calls += 1
        return {"driver": "dt_nominal", "isolator": "low_frequency_damped", "raw_hashes": ["b" * 64]}

    def execute_contact(self, candidate, protocol):
        self.physics_calls += 1
        return {"candidate_id": candidate["candidate_id"], "passed": candidate["candidate_id"] in self.eligible}

    def verify_contact(self, candidate, evidence, protocol):
        passed = evidence["candidate_id"] in self.eligible
        return passed, () if passed else ("contact_hard_gate_failed",)

    def freeze_selection(self, selection):
        self.selections.append(copy.deepcopy(selection))
        return copy.deepcopy(selection)

    def freeze_dependent_manifest(self, manifest):
        self.manifests.append(copy.deepcopy(manifest))
        return copy.deepcopy(manifest)

    def execute_dependent(self, state, protocol):
        self.physics_calls += 1
        return {
            "state_id": state["state_id"],
            "contact_candidate_id": state["contact_candidate_id"],
            "selection_artifact_sha256": state["selection_artifact_sha256"],
            "passed": True,
        }

    def verify_dependent(self, state, evidence, protocol):
        passed = all(
            (
                evidence.get("passed") is True,
                evidence.get("state_id") == state["state_id"],
                evidence.get("contact_candidate_id") == state["contact_candidate_id"],
                evidence.get("selection_artifact_sha256") == state["selection_artifact_sha256"],
            )
        )
        return passed, () if passed else ("dependent binding mismatch",)


@pytest.mark.parametrize("winner", CANDIDATES)
def test_every_possible_winner_branch_is_materialized_from_selection(winner):
    protocol = _protocol()
    eligible = set(CANDIDATES[CANDIDATES.index(winner) :])
    result = finalize_official_physics(protocol, SyntheticStore(eligible))
    assert isinstance(result, PublicationPlan)
    assert result.contact_winner == winner
    assert result.selection_artifact["contact_winner"] == winner
    for state in result.dependent_manifest["states"]:
        assert state["contact_candidate_id"] == winner
        assert winner in state["state_id"]
        assert result.selection_artifact["payload_sha256"] in state["state_id"]
    assert result.official_profile["physics"]["contact"]["candidate_id"] == winner


def test_v8_regression_binds_all_dependent_states_to_n6_overdamped_4():
    result = finalize_official_physics(_protocol(), SyntheticStore({"n6_overdamped_4"}))
    assert isinstance(result, PublicationPlan)
    assert result.contact_winner == "n6_overdamped_4"
    assert {row["contact_candidate_id"] for row in result.dependent_manifest["states"]} == {
        "n6_overdamped_4"
    }
    assert {row["contact_candidate_id"] for row in result.dependent_evidence} == {"n6_overdamped_4"}


def test_concrete_v8_prebinding_fails_before_any_evidence_or_physics_call():
    protocol = _protocol()
    protocol["dependent_manifest_template"][0]["prebound_contact"] = "n6_overdamped_2"
    store = SyntheticStore({"n6_overdamped_4"})
    result = finalize_official_physics(protocol, store)
    assert isinstance(result, BlockedResult)
    assert result.blocking_stage == "protocol"
    assert "concrete contact ID" in result.reason
    assert store.physics_calls == 0
    assert store.selections == []
    assert store.manifests == []


def test_zero_eligible_contact_emits_no_manifest_or_profile():
    store = SyntheticStore(set())
    result = finalize_official_physics(_protocol(), store)
    assert isinstance(result, BlockedResult)
    assert result.blocking_stage == "contact"
    assert result.selection_artifact["contact_winner"] is None
    assert result.dependent_manifest is None
    assert result.official_profile is None
    assert result.final_status["status"] == "BLOCKED"
    assert result.handoff["status"] == "BLOCKED"
    assert result.handoff["phase07_authorized"] is False
    assert store.manifests == []


def test_manifest_binding_is_deterministic_and_authenticated():
    protocol = _protocol()
    first = materialize_dependent_manifest(protocol, winner="n6_overdamped_4", selection_artifact_sha256="c" * 64)
    second = materialize_dependent_manifest(protocol, winner="n6_overdamped_4", selection_artifact_sha256="c" * 64)
    assert first == second
    assert first["payload_sha256"] == artifact_hash(first)
    assert validate_finalization_protocol(protocol)["dependent_template_count"] == 6
