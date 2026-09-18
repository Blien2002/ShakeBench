"""Pure Phase 08 scorecard and paired MDE contracts.

All functions consume already-written episode records.  They intentionally do
not import robosuite environments, MuJoCo, GPU runtimes, or the batch runner.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from shakebench.utils.outcomes import OutcomeContractError, validate_outcome

SCORECARD_SCHEMA_ID = "shakebench.phase08.scorecard"
SCORECARD_SCHEMA_VERSION = 2
EPISODE_RESULT_SCHEMA_ID = "shakebench.phase08.episode_result"
_TIERS = ("V0", "V1", "V2", "V3")
_PAIRS = (("V0", "V1"), ("V1", "V2"), ("V2", "V3"), ("V0", "V3"))
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_RESAMPLES = 10_000


class ScorecardError(ValueError):
    """Raised for a malformed or scientifically incomparable scorecard input."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    """Immutable, scoreable projection of one completed raw episode."""

    state_id: str
    tier: str
    gamma_commanded: float
    success: bool
    failure_reason: str | None
    termination_category: str
    scene_visual: Mapping[str, Any]
    geometry_authority: Mapping[str, Any]
    geometry_profile: Mapping[str, Any] | None
    scoreable: bool
    controller_profile: Mapping[str, Any]
    physics_authority: Mapping[str, Any]
    task_context: Mapping[str, Any]
    actuators: Sequence[Mapping[str, Any]]
    metrics: Mapping[str, Any]
    semantic_verifier_verdict: Mapping[str, Any]
    horizon_steps: int
    episode_validity: str = "valid"
    score_outcome: str | None = "unsuccessful"
    termination_cause: str = "horizon_exhausted"

    @classmethod
    def from_verified_raw(cls, raw: Mapping[str, Any], semantic_verifier_verdict: Mapping[str, Any]) -> "EpisodeResult":
        """Build a scoreable result from a semantically verified raw episode.

        Args:
            raw: Episode from a schema-v5 raw run artifact.
            semantic_verifier_verdict: Verdict returned by the shared raw-run
                verifier for the containing artifact.

        Returns:
            Immutable scoreable episode projection.

        Raises:
            ScorecardError: If the verifier did not pass or the episode is
                malformed.
        """

        if not isinstance(semantic_verifier_verdict, Mapping) or semantic_verifier_verdict.get("passed") is not True:
            raise ScorecardError("semantic verifier must pass before EpisodeResult construction")
        required = {
            "state_id",
            "tier",
            "gamma_commanded",
            "success",
            "failure_reason",
            "termination_category",
            "scene_visual",
            "geometry_authority",
            "geometry_profile",
            "scoreable",
            "controller_profile",
            "physics_profile",
            "task_context",
            "actuators",
            "metrics",
            "horizon_steps",
        }
        missing = required.difference(raw)
        if missing:
            raise ScorecardError("episode missing: " + ", ".join(sorted(missing)))
        if raw["tier"] not in _TIERS or not isinstance(raw["success"], bool) or raw["scoreable"] is not True:
            raise ScorecardError("invalid tier, success, or scoreability")
        horizon_steps = raw["horizon_steps"]
        if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, int) or horizon_steps < 1:
            raise ScorecardError("invalid horizon_steps")
        gamma = float(raw["gamma_commanded"])
        if not math.isfinite(gamma) or gamma < 0:
            raise ScorecardError("invalid gamma")
        # Small synthetic callers from Phase 08 can still use the legacy
        # projection; new raw artifacts must provide the orthogonal fields.
        validity = raw.get("episode_validity", "valid")
        outcome = raw.get("score_outcome", "success" if raw["success"] else "unsuccessful")
        cause = raw.get(
            "termination_cause",
            "success_latched" if raw["success"] else str(raw["termination_category"]),
        )
        try:
            validate_outcome(episode_validity=validity, score_outcome=outcome, termination_cause=cause)
        except OutcomeContractError as exc:
            raise ScorecardError("outcome integrity") from exc
        if bool(raw["success"]) != (outcome == "success"):
            raise ScorecardError("success projection integrity")
        return cls(
            state_id=str(raw["state_id"]),
            tier=str(raw["tier"]),
            gamma_commanded=gamma,
            success=raw["success"],
            failure_reason=raw["failure_reason"],
            termination_category=str(raw["termination_category"]),
            scene_visual=dict(raw["scene_visual"]),
            geometry_authority=dict(raw["geometry_authority"]),
            geometry_profile=dict(raw["geometry_profile"]) if isinstance(raw["geometry_profile"], Mapping) else None,
            scoreable=bool(raw["scoreable"]),
            controller_profile=dict(raw["controller_profile"]),
            physics_authority=dict(raw["physics_profile"]),
            task_context=dict(raw["task_context"]),
            actuators=tuple(dict(item) for item in raw["actuators"]),
            metrics=dict(raw["metrics"]),
            semantic_verifier_verdict=dict(semantic_verifier_verdict),
            horizon_steps=horizon_steps,
            episode_validity=str(validity),
            score_outcome=outcome,
            termination_cause=str(cause),
        )

    def science_identity(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "tier": self.tier,
            "gamma_commanded": self.gamma_commanded,
            "horizon_steps": self.horizon_steps,
            "scene_visual": self.scene_visual,
            "geometry_authority": self.geometry_authority,
            "geometry_profile": self.geometry_profile,
            "scoreable": self.scoreable,
            "controller_profile": self.controller_profile,
            "physics_authority": self.physics_authority,
            "task_context": self.task_context,
            "actuators": list(self.actuators),
        }


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float | int]:
    """Compute a two-sided Wilson interval.

    Args:
        successes: Successful episodes.
        total: Episodes in the denominator.
        z: Normal critical value.

    Returns:
        Success rate with confidence bounds.
    """
    if total <= 0 or successes < 0 or successes > total:
        raise ScorecardError("invalid Wilson denominator")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def paired_mde(
    *, discordant_a_only: int, discordant_b_only: int, total: int, alpha: float = 0.05, power: float = 0.80
) -> dict[str, Any]:
    """Report paired effect and preregistered MDE contexts.

    Args:
        discordant_a_only: Left-only successes.
        discordant_b_only: Right-only successes.
        total: Matched state count.
        alpha: Two-sided type-I error rate.
        power: Target power.

    Returns:
        Observed contrast and conservative MDE scenarios.
    """

    if total <= 0 or min(discordant_a_only, discordant_b_only) < 0 or discordant_a_only + discordant_b_only > total:
        raise ScorecardError("invalid paired counts")
    if not (0 < alpha < 1 and 0 < power < 1):
        raise ScorecardError("invalid alpha or power")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    discordance = (discordant_a_only + discordant_b_only) / total
    # Pre-registered conservative discordance scenarios; never infer a zero
    # MDE merely because this run happened to have no discordant pairs.
    scenarios = {str(q): (z_alpha + z_power) * math.sqrt(q / total) for q in (0.2, 0.3, 0.5)}
    return {
        "total": total,
        "discordant_a_only": discordant_a_only,
        "discordant_b_only": discordant_b_only,
        "observed_difference": (discordant_b_only - discordant_a_only) / total,
        "discordance_rate": discordance,
        "alpha": alpha,
        "power": power,
        "mde_absolute_success_rate": scenarios["0.3"],
        "conservative_discordance_scenarios": scenarios,
        "minimum_effect_of_interest": 0.10,
    }


def paired_bootstrap(left: Sequence[EpisodeResult], right: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Bootstrap right-minus-left deltas with a fixed seed.

    Args:
        left: Matched baseline episodes.
        right: Matched comparison episodes.

    Returns:
        Fixed-seed 10,000-resample confidence interval.
    """

    left_by_id = {row.state_id: row for row in left}
    right_by_id = {row.state_id: row for row in right}
    ids = sorted(left_by_id)
    if ids != sorted(right_by_id) or not ids:
        raise ScorecardError("paired bootstrap requires an exact matched state block")
    deltas = [int(right_by_id[key].success) - int(left_by_id[key].success) for key in ids]
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        samples.append(sum(deltas[generator.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()

    # R-7 percentile, matching the runner's explicitly documented definition.
    def percentile(q: float) -> float:
        position = (len(samples) - 1) * q
        low = int(position)
        high = min(low + 1, len(samples) - 1)
        return samples[low] + (position - low) * (samples[high] - samples[low])

    return {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES, "ci_95": [percentile(0.025), percentile(0.975)]}


def _validate_comparable(rows: Sequence[EpisodeResult]) -> None:
    if not rows:
        raise ScorecardError("no episodes")
    for row in rows:
        try:
            validate_outcome(
                episode_validity=row.episode_validity,
                score_outcome=row.score_outcome,
                termination_cause=row.termination_cause,
            )
        except OutcomeContractError as exc:
            raise ScorecardError("outcome integrity") from exc
    identities = {
        (
            row.gamma_commanded,
            row.horizon_steps,
            _canonical(row.scene_visual),
            _canonical(row.geometry_authority),
            _canonical(row.geometry_profile),
            row.scoreable,
            _canonical(row.controller_profile),
            _canonical(row.physics_authority),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ScorecardError("mixed Gamma or runtime authority")
    duplicates = Counter((row.tier, row.state_id) for row in rows)
    if any(count != 1 for count in duplicates.values()):
        raise ScorecardError("duplicate tier/state result")


def build_scorecard(
    rows: Sequence[EpisodeResult | Mapping[str, Any]], *, require_complete_tiers: bool = False
) -> dict[str, Any]:
    """Aggregate verified episodes at one Gamma.

    Args:
        rows: Semantically verified scoreable episodes.
        require_complete_tiers: Require all tiers and 400 matched states.

    Returns:
        Statistical scorecard projection.

    Raises:
        ScorecardError: If input is unverified or incomparable.
    """

    if not all(isinstance(row, EpisodeResult) for row in rows):
        raise ScorecardError("scorecard input must contain verified EpisodeResult values")
    episodes = list(rows)
    _validate_comparable(episodes)
    if require_complete_tiers and any(row.episode_validity != "valid" for row in episodes):
        raise ScorecardError("incomplete tier matrix: invalid execution requires exact-job retry")
    by_tier: dict[str, list[EpisodeResult]] = defaultdict(list)
    for row in episodes:
        by_tier[row.tier].append(row)
    state_sets = {tier: {row.state_id for row in values} for tier, values in by_tier.items()}
    if require_complete_tiers and set(by_tier) != set(_TIERS):
        raise ScorecardError("incomplete tier matrix")
    if len(state_sets) > 1 and len({frozenset(value) for value in state_sets.values()}) != 1:
        raise ScorecardError("tiers do not share a matched state block")
    per_tier = {}
    for tier, values in sorted(by_tier.items()):
        valid_values = [row for row in values if row.episode_validity == "valid"]
        invalid_values = [row for row in values if row.episode_validity == "invalid"]
        successes = sum(row.score_outcome == "success" for row in valid_values)
        failures = Counter(row.termination_cause for row in valid_values if row.score_outcome == "unsuccessful")
        if not valid_values:
            raise ScorecardError("tier has no valid scientific observations")
        rate = successes / len(valid_values)
        per_tier[tier] = {
            "success_count": successes,
            "success_rate": rate,
            "wilson": wilson_interval(successes, len(valid_values)),
            "valid_failure_count": len(valid_values) - successes,
            "invalid_execution_count": len(invalid_values),
            "incomplete": bool(invalid_values),
            "termination_causes": dict(sorted(failures.items())),
            "failure_reasons": dict(sorted(failures.items())),  # compatibility report projection
            "task_rule_violation_count": failures.get("task_rule_violation", 0),
            "policy_abort_count": failures.get("policy_abort", 0),
            "policy_error_count": failures.get("policy_error", 0),
            "horizon_exhausted_count": failures.get("horizon_exhausted", 0),
            "physics_violation_count": failures.get("task_rule_violation", 0),
            "ceiling_limited": rate > 0.90,
            "floor_limited": rate < 0.10,
        }
    if require_complete_tiers and len(next(iter(state_sets.values()))) != 400:
        raise ScorecardError("incomplete official matrix: require 4 x 400")
    comparisons = {}
    for left, right in _PAIRS:
        if left not in by_tier or right not in by_tier:
            continue
        left_by_state = {row.state_id: row for row in by_tier[left]}
        right_by_state = {row.state_id: row for row in by_tier[right]}
        common = sorted(set(left_by_state).intersection(right_by_state))
        valid_common = [
            key
            for key in common
            if left_by_state[key].episode_validity == "valid" and right_by_state[key].episode_validity == "valid"
        ]
        if not valid_common:
            continue
        paired_left = [left_by_state[key] for key in valid_common]
        paired_right = [right_by_state[key] for key in valid_common]
        a_only = sum(
            left_by_state[key].score_outcome == "success" and right_by_state[key].score_outcome != "success"
            for key in valid_common
        )
        b_only = sum(
            right_by_state[key].score_outcome == "success" and left_by_state[key].score_outcome != "success"
            for key in valid_common
        )
        comparison = paired_mde(discordant_a_only=a_only, discordant_b_only=b_only, total=len(valid_common))
        comparison.update(
            {
                "delta_success_rate": (
                    sum(row.score_outcome == "success" for row in paired_right)
                    - sum(row.score_outcome == "success" for row in paired_left)
                )
                / len(valid_common),
                "n_gain": b_only,
                "n_loss": a_only,
                "paired_bootstrap": paired_bootstrap(paired_left, paired_right),
                "incomplete": len(valid_common) != len(common),
                "invalid_execution_count": len(common) - len(valid_common),
            }
        )
        comparisons[f"{right}_minus_{left}"] = comparison
    gamma = episodes[0].gamma_commanded
    result = {
        "schema_id": SCORECARD_SCHEMA_ID,
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "gamma_commanded": gamma,
        "state_count": len(next(iter(state_sets.values()))),
        "per_tier": per_tier,
        "paired_comparisons": comparisons,
    }
    return result


def verify_scorecard(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a production scorecard from its authenticated raw sources.

    Args:
        payload: Scorecard payload emitted by the production CLI.

    Returns:
        Verdict containing every provenance or recomputation error.
    """

    errors = []
    if payload.get("schema_id") != SCORECARD_SCHEMA_ID or payload.get("schema_version") != SCORECARD_SCHEMA_VERSION:
        errors.append("schema")
    comparisons = payload.get("paired_comparisons")
    if not isinstance(comparisons, Mapping) or not set(comparisons).issubset(
        {"V1_minus_V0", "V2_minus_V1", "V3_minus_V2", "V3_minus_V0"}
    ):
        errors.append("paired comparisons")
    sources = payload.get("raw_artifacts")
    if not isinstance(sources, list) or not sources:
        errors.append("raw artifact provenance")
        return {"passed": not errors, "errors": errors}
    try:
        from shakebench.scripts.run_oracle import verify_run_artifact

        episodes = []
        for source in sources:
            if not isinstance(source, Mapping) or set(source) != {"path", "semantic_verifier"}:
                errors.append("raw artifact provenance")
                continue
            path = source["path"]
            if not isinstance(path, str):
                errors.append("raw artifact provenance")
                continue
            verdict = verify_run_artifact(path)
            if verdict.get("passed") is not True or source.get("semantic_verifier") != verdict:
                errors.append("raw artifact semantic verifier")
                continue
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            episodes.extend(EpisodeResult.from_verified_raw(item, verdict) for item in raw["episodes"])
        if not errors:
            recomputed = build_scorecard(episodes, require_complete_tiers=payload.get("state_count") == 400)
            for key in (
                "gamma_commanded",
                "state_count",
                "per_tier",
                "paired_comparisons",
            ):
                if payload.get(key) != recomputed.get(key):
                    errors.append("scorecard recomputation")
                    break
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"raw artifact provenance: {exc}")
    return {"passed": not errors, "errors": errors}


__all__ = [
    "EPISODE_RESULT_SCHEMA_ID",
    "EpisodeResult",
    "SCORECARD_SCHEMA_ID",
    "ScorecardError",
    "build_scorecard",
    "paired_bootstrap",
    "paired_mde",
    "verify_scorecard",
    "wilson_interval",
]
