"""Versioned, orthogonal outcome contract for ShakeBench rollouts.

This module deliberately has no simulator dependency.  It is the single
authority for deciding whether an episode is a scientific observation, what
enters a success-rate denominator, and which terminal combinations are legal.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


OUTCOME_CONTRACT_SCHEMA_ID = "shakebench.outcome_contract"
OUTCOME_CONTRACT_SCHEMA_VERSION = 1
EPISODE_VALIDITIES = frozenset(("valid", "invalid"))
SCORE_OUTCOMES = frozenset(("success", "unsuccessful", None))
TERMINATION_CAUSES = frozenset(
    (
        "success_latched",
        "horizon_exhausted",
        "task_rule_violation",
        "policy_abort",
        "policy_error",
        "invalid_execution",
    )
)
CONTROLLER_EVENT_TYPES = frozenset(
    (
        "phase_deadline",
        "grasp_not_established",
        "grasp_loss",
        "grasp_slip",
        "anchor_drift",
        "tool_clearance_blocked",
        "placement_loss",
        "placement_rebound",
        "recovery_settle_timeout",
        "edge_risk",
        "workspace_risk",
        "evaluator_not_latched_after_public_verify",
    )
)


class OutcomeContractError(ValueError):
    """Raised when an outcome record violates the frozen contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def outcome_contract() -> dict[str, Any]:
    """Return the immutable outcome authority embedded in every new artifact."""

    payload = {
        "schema_id": OUTCOME_CONTRACT_SCHEMA_ID,
        "schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
        "episode_validities": sorted(EPISODE_VALIDITIES),
        "score_outcomes": ["success", "unsuccessful", None],
        "termination_causes": sorted(TERMINATION_CAUSES),
        "controller_event_types": sorted(CONTROLLER_EVENT_TYPES),
        "task_rule": "registered_finite_illegal_penetration_only",
        "success_authority": "environment_strict_success_latch",
    }
    payload["contract_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def outcome_contract_sha256() -> str:
    return str(outcome_contract()["contract_sha256"])


def validate_outcome(
    *, episode_validity: Any, score_outcome: Any, termination_cause: Any
) -> None:
    """Fail closed on contradictory terminal-state combinations."""

    if episode_validity not in EPISODE_VALIDITIES:
        raise OutcomeContractError("unknown episode_validity")
    if score_outcome not in SCORE_OUTCOMES:
        raise OutcomeContractError("unknown score_outcome")
    if termination_cause not in TERMINATION_CAUSES:
        raise OutcomeContractError("unknown termination_cause")
    valid_success = episode_validity == "valid" and score_outcome == "success"
    valid_failure = episode_validity == "valid" and score_outcome == "unsuccessful"
    invalid = episode_validity == "invalid" and score_outcome is None
    if valid_success and termination_cause == "success_latched":
        return
    if valid_failure and termination_cause in {
        "horizon_exhausted",
        "task_rule_violation",
        "policy_abort",
        "policy_error",
    }:
        return
    if invalid and termination_cause == "invalid_execution":
        return
    raise OutcomeContractError("contradictory validity, score outcome, and termination cause")


def resolve_termination_cause(
    *,
    prior_cause: str | None,
    task_rule_violation: bool,
    success_latched: bool,
    policy_abort: bool,
    horizon_exhausted: bool,
) -> str | None:
    """Apply the frozen terminal priority without overwriting earlier evidence."""

    if prior_cause is not None:
        return prior_cause
    if task_rule_violation:
        return "task_rule_violation"
    if success_latched:
        return "success_latched"
    if policy_abort:
        return "policy_abort"
    if horizon_exhausted:
        return "horizon_exhausted"
    return None


def validate_controller_events(events: Any) -> None:
    """Validate that controller diagnostics cannot impersonate task results."""

    if not isinstance(events, list):
        raise OutcomeContractError("controller_events must be a list")
    last_time = -1.0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise OutcomeContractError(f"controller event {index} must be an object")
        if event.get("event_type") not in CONTROLLER_EVENT_TYPES:
            raise OutcomeContractError(f"controller event {index} has an unregistered type")
        try:
            time_s = float(event["time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OutcomeContractError(f"controller event {index} lacks finite time") from exc
        if not math.isfinite(time_s):
            raise OutcomeContractError(f"controller event {index} lacks finite time")
        if not (time_s >= last_time):
            raise OutcomeContractError("controller event timestamps must be monotonic")
        last_time = time_s
        if not isinstance(event.get("phase"), str) or not isinstance(event.get("decision"), str):
            raise OutcomeContractError(f"controller event {index} lacks phase or decision")
        summary = event.get("observation_summary")
        if not isinstance(summary, Mapping) or set(summary) != {
            "can_pos_robot_base",
            "eef_pos_robot_base",
            "recovery_count",
        }:
            raise OutcomeContractError(f"controller event {index} lacks public observation summary")
        try:
            can = [float(value) for value in summary["can_pos_robot_base"]]
            eef = [float(value) for value in summary["eef_pos_robot_base"]]
            recovery_count = int(summary["recovery_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OutcomeContractError(f"controller event {index} lacks public observation summary") from exc
        if len(can) != 3 or len(eef) != 3 or not all(math.isfinite(value) for value in can + eef) or recovery_count < 0:
            raise OutcomeContractError(f"controller event {index} lacks public observation summary")
        if "episode_validity" in event or "score_outcome" in event or "termination_cause" in event:
            raise OutcomeContractError("controller event may not rewrite outcome")


def legacy_projection(*, episode_validity: str, score_outcome: str | None, termination_cause: str) -> tuple[bool, str | None, str]:
    """Compatibility fields for consumers that still display Phase-07 names."""

    validate_outcome(
        episode_validity=episode_validity,
        score_outcome=score_outcome,
        termination_cause=termination_cause,
    )
    return (
        score_outcome == "success",
        None if score_outcome == "success" else termination_cause,
        termination_cause,
    )


__all__ = [
    "CONTROLLER_EVENT_TYPES",
    "EPISODE_VALIDITIES",
    "OUTCOME_CONTRACT_SCHEMA_ID",
    "OUTCOME_CONTRACT_SCHEMA_VERSION",
    "OutcomeContractError",
    "SCORE_OUTCOMES",
    "TERMINATION_CAUSES",
    "legacy_projection",
    "outcome_contract",
    "outcome_contract_sha256",
    "resolve_termination_cause",
    "validate_controller_events",
    "validate_outcome",
]
