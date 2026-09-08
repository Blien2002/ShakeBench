"""Synthetic-only scorecard and MDE contract tests."""

import copy
import dataclasses

import pytest

from robosuite.utils.shakebench_scoring import (
    EpisodeResult,
    RunManifest,
    ScorecardError,
    build_scorecard,
    paired_mde,
    verify_scorecard,
    wilson_interval,
)


def _raw(state_id: str, tier: str, success: bool) -> dict:
    return {
        "state_id": state_id,
        "tier": tier,
        "gamma_commanded": 0.25,
        "success": success,
        "failure_reason": None if success else "horizon_exhausted",
        "termination_category": "environment_success" if success else "horizon_exhausted",
        "scene_visual": {"scene_id": "direct", "config_sha256": "a" * 64},
        "geometry_authority": {"kind": "phase07_5a", "scoreable": True, "authority": {"payload_sha256": "b" * 64}},
        "geometry_profile": {"profile_id": "direct_mount_v1", "payload_sha256": "c" * 64},
        "scoreable": True,
        "controller_profile": {"profile_id": "oracle", "payload_sha256": "d" * 64},
        "physics_profile": {"profile_id": "official", "profile_sha256": "e" * 64},
        "task_context": {"frame": "worktable"},
        "task_context_sha256": "f" * 64,
        "actuators": [{"id": 0, "name": "joint0"}],
        "trace_sha256": "1" * 64,
        "metrics": {"success": {"passed": success}},
    }


def _verified(state_id: str, tier: str, success: bool) -> EpisodeResult:
    return EpisodeResult.from_verified_raw(_raw(state_id, tier, success), {"passed": True})


def test_wilson_and_paired_mde_include_failure_denominators():
    interval = wilson_interval(3, 4)
    assert interval["total"] == 4
    assert interval["lower"] < interval["rate"] < interval["upper"]
    mde = paired_mde(discordant_a_only=2, discordant_b_only=1, total=10)
    assert mde["observed_difference"] == pytest.approx(-0.1)
    assert mde["mde_absolute_success_rate"] > 0


def test_scorecard_requires_one_gamma_and_matched_state_blocks():
    rows = [
        _verified("s0", "V0", True),
        _verified("s1", "V0", False),
        _verified("s0", "V1", True),
        _verified("s1", "V1", True),
    ]
    scorecard = build_scorecard(rows)
    assert scorecard["per_tier"]["V0"]["wilson"]["total"] == 2
    comparison = scorecard["paired_comparisons"]["V1_minus_V0"]
    assert comparison["discordant_b_only"] == 1
    assert comparison["observed_difference"] == comparison["delta_success_rate"]
    changed = copy.deepcopy(rows)
    changed[-1] = dataclasses.replace(changed[-1], gamma_commanded=0.30)
    with pytest.raises(ScorecardError, match="mixed Gamma"):
        build_scorecard(changed)
    unmatched = copy.deepcopy(rows)
    unmatched[-1] = dataclasses.replace(unmatched[-1], state_id="s2")
    with pytest.raises(ScorecardError, match="matched state block"):
        build_scorecard(unmatched)


def test_scorecard_rejects_unverified_episode_mappings():
    with pytest.raises(ScorecardError, match="verified EpisodeResult"):
        build_scorecard([_raw("s0", "V0", True)])

    with pytest.raises(ScorecardError, match="semantic verifier"):
        EpisodeResult.from_verified_raw(_raw("s0", "V0", True), {"passed": False})


def test_scorecard_verification_requires_recomputable_raw_provenance():
    scorecard = build_scorecard([_verified("s0", "V0", True)])
    assert not verify_scorecard(scorecard)["passed"]


def test_run_manifest_identity_excludes_execution_provenance():
    first = RunManifest("a" * 64, "V0", 0.25, 1200, {"physics": "b" * 64})
    second = RunManifest("a" * 64, "V0", 0.25, 1200, {"physics": "b" * 64})
    assert first.job_id == second.job_id
    assert "pid" not in first.science_identity()
