"""Fast contract regressions for Phase 08R's failure boundary."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from robosuite.utils.shakebench_oracle import OracleControllerProfile, ShakeBenchOracleController, TaskPhase
from robosuite.utils.shakebench_outcomes import (
    OutcomeContractError,
    outcome_contract_sha256,
    resolve_termination_cause,
    validate_controller_events,
    validate_outcome,
)
from robosuite.utils.shakebench_scoring import EpisodeResult, ScorecardError, build_scorecard
from tests.shakebench_test_helpers import build_tier_observation


def _observation() -> dict:
    return build_tier_observation("V0", goal_frame_pos=(0.0, 0.0, 0.0))


def _result(*, state_id: str, validity: str, outcome: str | None, cause: str) -> EpisodeResult:
    success = outcome == "success"
    raw = {
        "state_id": state_id,
        "tier": "V0",
        "gamma_commanded": 0.0,
        "success": success,
        "failure_reason": None if success else cause,
        "termination_category": cause,
        "episode_validity": validity,
        "score_outcome": outcome,
        "termination_cause": cause,
        "outcome_contract_sha256": outcome_contract_sha256(),
        "scene_visual": {},
        "geometry_authority": {},
        "geometry_profile": None,
        "scoreable": True,
        "controller_profile": {},
        "physics_profile": {},
        "task_context": {},
        "task_context_sha256": "b" * 64,
        "actuators": [],
        "trace_sha256": "c" * 64,
        "metrics": {},
    }
    return EpisodeResult.from_verified_raw(raw, {"passed": True})


def test_outcome_contract_allows_only_orthogonal_terminal_combinations():
    validate_outcome(episode_validity="valid", score_outcome="success", termination_cause="success_latched")
    validate_outcome(episode_validity="valid", score_outcome="unsuccessful", termination_cause="policy_abort")
    validate_outcome(episode_validity="invalid", score_outcome=None, termination_cause="invalid_execution")
    with pytest.raises(OutcomeContractError):
        validate_outcome(episode_validity="valid", score_outcome="success", termination_cause="policy_abort")


def test_terminal_priority_keeps_task_rule_above_success_and_abort():
    assert resolve_termination_cause(
        prior_cause=None,
        task_rule_violation=True,
        success_latched=True,
        policy_abort=True,
        horizon_exhausted=True,
    ) == "task_rule_violation"
    assert resolve_termination_cause(
        prior_cause="policy_error",
        task_rule_violation=True,
        success_latched=True,
        policy_abort=True,
        horizon_exhausted=True,
    ) == "policy_error"


@pytest.mark.parametrize("time_s", (float("inf"), float("nan")))
def test_controller_events_require_finite_time_and_public_observation_summary(time_s):
    event = {"event_type": "edge_risk", "time_s": time_s, "phase": "verify", "decision": "policy_abort"}
    with pytest.raises(OutcomeContractError, match="finite time"):
        validate_controller_events([event])

    event["time_s"] = 0.0
    with pytest.raises(OutcomeContractError, match="observation summary"):
        validate_controller_events([event])


def test_recovery_settle_timeout_stays_in_same_rollout_when_public_state_is_safe():
    controller = ShakeBenchOracleController("V0", OracleControllerProfile(recovery_budget=2))
    observation = _observation()
    controller.executive.phase = TaskPhase.WAIT_PUBLIC_SETTLE
    controller.executive.phase_entered_s = 0.0
    # Finite, table-supported and inside the envelope, but deliberately not
    # stable enough to open yet.
    observation["can_pos_robot_base"] = np.array((0.0, 0.0, 0.070297), dtype=np.float32)
    controller.action(observation, time_s=controller.profile.recovery_settle_s + 0.01)
    assert controller.executive.phase is TaskPhase.WAIT_PUBLIC_SETTLE
    assert controller.executive.abort_reason is None
    assert any(event["event_type"] == "recovery_settle_timeout" for event in controller.executive.controller_events)


def test_edge_risk_is_a_controller_abort_not_a_task_rule_violation():
    controller = ShakeBenchOracleController("V0")
    observation = _observation()
    observation["can_pos_robot_base"] = np.array((0.5, 0.5, 0.04), dtype=np.float32)
    controller.executive.phase = TaskPhase.VERIFY
    controller.action(observation, time_s=controller.profile.verify_s)
    assert controller.executive.phase is TaskPhase.ABORTED
    assert controller.abort_requested
    assert controller.executive.abort_reason == "edge_risk"
    assert controller.executive.failure_reason == "policy_abort"


def test_invalid_execution_marks_a_complete_matrix_ineligible_until_retry():
    valid = _result(state_id="s0", validity="valid", outcome="unsuccessful", cause="horizon_exhausted")
    invalid = _result(state_id="s1", validity="invalid", outcome=None, cause="invalid_execution")
    scorecard = build_scorecard([valid, invalid])
    assert scorecard["per_tier"]["V0"]["invalid_execution_count"] == 1
    with pytest.raises(ScorecardError, match="invalid execution"):
        build_scorecard([valid, invalid], require_complete_tiers=True)


def test_invalid_execution_is_excluded_from_paired_statistics_until_exact_retry():
    rows = [
        dataclasses.replace(_result(state_id="s0", validity="valid", outcome="unsuccessful", cause="horizon_exhausted"), tier="V0"),
        dataclasses.replace(_result(state_id="s1", validity="invalid", outcome=None, cause="invalid_execution"), tier="V0"),
        dataclasses.replace(_result(state_id="s0", validity="valid", outcome="success", cause="success_latched"), tier="V1"),
        dataclasses.replace(_result(state_id="s1", validity="invalid", outcome=None, cause="invalid_execution"), tier="V1"),
    ]

    comparison = build_scorecard(rows)["paired_comparisons"]["V1_minus_V0"]

    assert comparison["total"] == 1
    assert comparison["delta_success_rate"] == pytest.approx(1.0)


def test_v2_scorecard_rejects_a_legacy_episode_without_outcome_authority():
    legacy = dataclasses.replace(
        _result(state_id="s0", validity="valid", outcome="success", cause="success_latched"),
        outcome_contract_sha256=None,
    )

    with pytest.raises(ScorecardError, match="outcome contract"):
        build_scorecard([legacy])
