"""Run the shared State Oracle controller on explicitly supplied dev states.

This runner consumes the Phase 07 pre-registered ten-state dev asset. Phase 08
retains ownership of official/knee state generation and absorbs this frozen
dev subset into the unified committed-state protocol without regenerating it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import shakebench
from robosuite.controllers import load_composite_controller_config
from shakebench import models
from shakebench.utils.artifacts import write_json
from shakebench.utils.calibration import level_scale_for_gamma
from shakebench.utils.dev_states import (
    PHASE07_DEV_STATE_FILENAME,
    verify_phase07_dev_state_artifact,
)
from shakebench.utils.excitation import build_excitation_program
from shakebench.utils.geometry import (
    DEFAULT_GEOMETRY_PROFILE,
    GEOMETRY_PROFILES,
    geometry_scene_path,
    load_geometry_profile,
)
from shakebench.utils.oracle import (
    ORACLE_TASK_KEYS,
    OracleControllerProfile,
    ShakeBenchOracleController,
    ShakeBenchOracleError,
    WorktableTaskContext,
    _public_copy,
)
from shakebench.utils.outcomes import (
    TERMINATION_CAUSES,
    OutcomeContractError,
    legacy_projection,
    outcome_contract,
    resolve_termination_cause,
    validate_controller_events,
    validate_outcome,
)
from shakebench.utils.providers import COMMON_STATE_KEYS, POLICY_FIELD_CONTRACT, TABLE_IMU_POLICY_KEYS
from shakebench.utils.scene import load_scene_visual_config
from shakebench.utils.state_schema import normalize_state


class OracleRunError(RuntimeError):
    """Raised for invalid or incomplete Phase 07 run inputs."""


RUN_SCHEMA_ID = "shakebench.phase07.oracle_run"
RUN_SCHEMA_VERSION = 7
EPISODE_SCHEMA_ID = "shakebench.phase07.oracle_episode"
EPISODE_SCHEMA_VERSION = 7
# The current contract has one observation lane.  Run and episode artifacts
# keep their frozen ``tier`` provenance field so existing readers and run IDs
# stay valid; no caller may select another lane.
CURRENT_TIER_ALIAS = "V0"
# Canonical field-contract lookup for the oracle wire names.
_CONTRACT_KEY_FOR_WIRE = {
    **{wire: canonical for canonical, wire in zip(COMMON_STATE_KEYS, ORACLE_TASK_KEYS)},
}
DEV_STATE_PRE_HISTORY_REWRITE_COMMIT = "dd6fe2edb6384ccdb5116be44f07592b4864e377"
DEV_STATE_REWRITTEN_COMMIT = "08626ea5a5e107df503e266be9065b929d47f882"
DEV_STATE_ANCHOR_REWRITE = {
    DEV_STATE_PRE_HISTORY_REWRITE_COMMIT: DEV_STATE_REWRITTEN_COMMIT,
}
# Compatibility name: new provenance must use the rewritten anchor.
DEV_STATE_ANCHOR_COMMIT = DEV_STATE_REWRITTEN_COMMIT
OFFICIAL_PHYSICS_PROFILE_ID = "shakebench.official.physics.v2"
# Legacy verifier exports retain these names, but the outcome contract is the
# sole registry for v6 termination causes.
TERMINATION_CATEGORIES = TERMINATION_CAUSES
FAILURE_REASONS = TERMINATION_CAUSES


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_dev_states(path: str | Path) -> list[dict[str, Any]]:
    """Load the pre-existing ten-State Phase 07 development list fail-closed."""

    source = Path(path)
    verification = verify_phase07_dev_state_artifact(source)
    if not verification["passed"]:
        failed = [key for key, passed in verification["checks"].items() if not passed]
        raise OracleRunError("Phase 07 dev-state artifact failed: " + ", ".join(failed))
    payload = verification["payload"]
    rows = payload.get("states", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or len(rows) != 10:
        raise OracleRunError("Phase 07 requires exactly the existing 10 dev states")
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise OracleRunError("each dev state must be an object")
        state_id = row.get("state_id")
        xy = np.asarray(row.get("object_xy_m"), dtype=float)
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id in seen
            or xy.shape != (2,)
            or not np.all(np.isfinite(xy))
        ):
            raise OracleRunError("dev states require unique state_id and finite object_xy_m[2]")
        seen.add(state_id)
        result.append(
            {
                "state_id": state_id,
                "object_xy_m": xy.tolist(),
                "seed": int(row.get("excitation_seed", row.get("seed", 0))),
                "imu_seed": int(row.get("imu_seed", row.get("seed", 0))),
                "t0_s": float(row.get("t0_s", 0.0)),
            }
        )
    return sorted(result, key=lambda item: item["state_id"])


def load_state_asset(path: str | Path) -> dict[str, Any]:
    """Load one state asset using its authenticated schema, never its filename.

    Dev remains on its frozen Phase-07 verifier. Official and knee assets use
    the Phase-08 committed-state verifier and retain their complete canonical
    records at the execution boundary.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleRunError(f"state asset read failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise OracleRunError("state asset must be an object")
    if payload.get("schema_id") == "shakebench.phase07.dev_states":
        states = load_dev_states(source)
        return {"states": states, "authority": {"kind": "dev", "dev_state_anchor": _dev_state_anchor(source)}}
    from shakebench.utils.train_states import TRAIN_STATE_SCHEMA, verify_train_state_artifact

    if payload.get("schema_id") == TRAIN_STATE_SCHEMA:
        verdict = verify_train_state_artifact(payload)
        if not verdict["passed"]:
            failed = [key for key, passed in verdict["checks"].items() if not passed]
            raise OracleRunError("train state authority failed: " + ", ".join(failed))
        return {
            "states": verdict["states"],
            "authority": {
                "kind": "train",
                "split": "train",
                "scoreable": False,
            },
        }
    from shakebench.utils.task_states import TASK_STATE_SCHEMA, verify_task_state_artifact

    if payload.get("schema_id") == TASK_STATE_SCHEMA:
        verdict = verify_task_state_artifact(payload)
        if not verdict["passed"]:
            raise OracleRunError("task state authority failed: " + ", ".join(verdict["errors"]))
        return {
            "states": payload["states"],
            "authority": {
                "kind": "task_variants",
                "split": payload["split"],
                "scoreable": False,
            },
        }
    from shakebench.utils.committed_states import verify_committed_state_artifact

    split = payload.get("split")
    if split not in {"official", "knee"}:
        raise OracleRunError("unknown state asset schema/split")
    verdict = verify_committed_state_artifact(payload, expected_split=split)
    if not verdict["passed"]:
        raise OracleRunError("committed state authority failed: " + ", ".join(verdict["errors"]))
    states = payload.get("states")
    if not isinstance(states, list) or not all(isinstance(row, Mapping) for row in states):
        raise OracleRunError("committed state payload malformed")
    return {
        "states": [dict(row) for row in states],
        "authority": {
            "kind": "committed",
            "split": split,
        },
    }


def _default_dev_state_path() -> Path:
    return Path(models.assets_root) / PHASE07_DEV_STATE_FILENAME


def _dev_state_anchor(path: str | Path | None = None) -> dict[str, Any]:
    source = _default_dev_state_path() if path is None else Path(path)
    verification = verify_phase07_dev_state_artifact(source)
    if not verification["passed"]:
        raise OracleRunError(
            "dev-state anchor failed: " + ", ".join(key for key, value in verification["checks"].items() if not value)
        )
    payload = verification["payload"]
    return {
        "commit": DEV_STATE_ANCHOR_COMMIT,
        "pre_history_rewrite_commit": DEV_STATE_PRE_HISTORY_REWRITE_COMMIT,
        "rewritten_commit": DEV_STATE_REWRITTEN_COMMIT,
        "asset": PHASE07_DEV_STATE_FILENAME,
        "generator_id": payload["generator"]["generator_id"],
        "root_seed": payload["generator"]["root_seed"],
        "state_count": len(payload["states"]),
    }


def _dev_state_anchor_match(value: Any, expected: Mapping[str, Any]) -> tuple[bool, str]:
    """Match current provenance or resolve the immutable pre-rewrite form.

    Existing raw R5 artifacts intentionally retain the old commit string.  We
    accept that exact historical shape only through the explicit frozen map;
    newly generated artifacts always carry both anchors and use the rewritten
    SHA as ``commit``.
    """

    if _values_equal(value, expected, atol=0.0):
        return True, "rewritten"
    if not isinstance(value, Mapping):
        return False, "invalid"
    historical = dict(expected)
    historical.pop("pre_history_rewrite_commit", None)
    historical.pop("rewritten_commit", None)
    historical["commit"] = next(iter(DEV_STATE_ANCHOR_REWRITE))
    if DEV_STATE_ANCHOR_REWRITE.get(historical["commit"]) == expected.get("rewritten_commit") and _values_equal(
        value, historical, atol=0.0
    ):
        return True, "pre_history_rewrite_resolved"
    return False, "invalid"


def scene_visual_identity(scene_config) -> dict[str, Any]:
    """Single writer/verifier representation of the scene visual authority."""

    return {
        "scene_id": scene_config.scene_id,
        "geometry_variant": scene_config.geometry_variant,
        "physics_effect": scene_config.physics_effect,
    }


def geometry_authority_identity(geometry_profile: str) -> dict[str, Any]:
    """Return the current world-fixed geometry and outcome authority for a rollout."""

    from shakebench.utils.runtime_verifier import RUNTIME_CONTRACT_FILENAME

    if geometry_profile not in GEOMETRY_PROFILES:
        raise OracleRunError("geometry_profile must be one of " + ", ".join(GEOMETRY_PROFILES))
    if not (Path(models.assets_root) / RUNTIME_CONTRACT_FILENAME).is_file():
        raise OracleRunError("runtime contract asset missing")
    return {
        "kind": geometry_profile,
        # The current topology still requires fresh experimental certification.
        "scoreable": False,
    }


def _actuator_metadata_from_model(raw_model: Any) -> list[dict[str, Any]]:
    result = []
    for actuator_id in range(int(raw_model.nu)):
        result.append(
            {
                "id": actuator_id,
                "name": raw_model.actuator(actuator_id).name,
                "ctrlrange": np.asarray(raw_model.actuator_ctrlrange[actuator_id], dtype=float).copy(),
                "forcerange": np.asarray(raw_model.actuator_forcerange[actuator_id], dtype=float).copy(),
                "ctrllimited": bool(np.asarray(raw_model.actuator_ctrllimited[actuator_id]).item()),
                "forcelimited": bool(np.asarray(raw_model.actuator_forcelimited[actuator_id]).item()),
                "units": "actuator_native",
            }
        )
    return result


def current_contract_observation(env: Any, observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Expert task fields plus the delivered worktable IMU payload."""

    from shakebench.utils.expert import oracle_observation

    raw = env._get_observations() if observation is None else observation
    values = oracle_observation(env, raw)
    values.update({key: raw[key] for key in TABLE_IMU_POLICY_KEYS})
    return values


@lru_cache(maxsize=1)
def _official_actuator_metadata() -> tuple[dict[str, Any], ...]:
    env = shakebench.make(
        "VibrationPickPlace",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        imu_mode="ideal_smoke",
        horizon=1,
        ignore_done=True,
        seed=0,
    )
    try:
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        return tuple(_json_ready(item) for item in _actuator_metadata_from_model(raw_model))
    finally:
        env.close()


def _values_equal(actual: Any, expected: Any, *, atol: float = 1.0e-7) -> bool:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        try:
            actual_array = np.asarray(actual)
            expected_array = np.asarray(expected)
            if actual_array.shape != expected_array.shape:
                return False
            if np.issubdtype(actual_array.dtype, np.number) and np.issubdtype(expected_array.dtype, np.number):
                return bool(np.all(np.isfinite(actual_array)) and np.all(np.isfinite(expected_array))) and bool(
                    np.allclose(actual_array, expected_array, rtol=0.0, atol=atol)
                )
            return bool(np.array_equal(actual_array, expected_array))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_values_equal(actual[key], expected[key], atol=atol) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_values_equal(a, e, atol=atol) for a, e in zip(actual, expected))
    if isinstance(expected, (float, np.floating)):
        try:
            return bool(np.isfinite(float(actual)) and np.isclose(float(actual), float(expected), rtol=0.0, atol=atol))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (int, np.integer, bool, np.bool_)):
        return type(actual) is type(expected) and actual == expected
    return actual == expected


def _finite_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(float(value)))
    if isinstance(value, (int, np.integer, bool, np.bool_)) or value is None or isinstance(value, str):
        return True
    return False


def _array_contract_ok(value: Any, contract: Mapping[str, Any]) -> bool:
    dtype = contract["dtype"]
    if dtype == "str":
        return isinstance(value, str)
    try:
        if dtype == "bool":
            array = np.asarray(value)
            return array.shape == tuple(contract["shape"]) and np.issubdtype(array.dtype, np.bool_)
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.shape != tuple(contract["shape"]):
        return False
    return bool(np.all(np.isfinite(array)))


def _profile_from_payload(value: Any) -> OracleControllerProfile:
    if not isinstance(value, Mapping):
        raise OracleRunError("controller profile must be an object")
    return OracleControllerProfile(**value)


def _compare_actuator_metadata(actual: Any, errors: list[str]) -> None:
    expected = list(_official_actuator_metadata())
    if not isinstance(actual, list) or len(actual) != len(expected):
        errors.append("actuator metadata count/order")
        return
    for index, (observed, wanted) in enumerate(zip(actual, expected)):
        if not isinstance(observed, Mapping):
            errors.append(f"actuator metadata row {index}")
            continue
        for key in ("id", "name", "ctrllimited", "forcelimited", "units"):
            if observed.get(key) != wanted.get(key):
                errors.append(f"actuator {index} {key}")
        for key in ("ctrlrange", "forcerange"):
            try:
                if not np.allclose(
                    np.asarray(observed[key], dtype=float), np.asarray(wanted[key], dtype=float), rtol=0.0, atol=1e-12
                ):
                    errors.append(f"actuator {index} {key}")
            except (KeyError, TypeError, ValueError):
                errors.append(f"actuator {index} {key}")


def run_episode(
    state: Mapping[str, Any],
    *,
    gamma_commanded: float,
    profile: OracleControllerProfile,
    horizon_steps: int = 1200,
    geometry_profile: str = DEFAULT_GEOMETRY_PROFILE,
    hard_reset: bool = True,
    step_observer: Callable[[Any, int, Mapping[str, Any], ShakeBenchOracleController], None] | None = None,
) -> dict[str, Any]:
    """Run a public-observation-only episode and retain action/actuator evidence.

    ``step_observer`` is a qualitative-demo hook. It receives the environment
    after each policy step, but rendered pixels never enter the controller
    observation or the scoreable trace.
    """

    # The verifier binds this episode to the committed record it reloads from the
    # asset, so the alias-normalized execution copy never replaces that record.
    state = normalize_state(state)
    state_id = str(state["state_id"])
    seed = int(state.get("excitation_seed", state.get("seed", 0)))
    imu_seed = int(state.get("imu_seed", seed))
    t0_s = float(state.get("t0_s", 0.0))
    gamma_commanded = float(gamma_commanded)
    if gamma_commanded < 0.0 or not np.isfinite(gamma_commanded):
        raise OracleRunError("gamma_commanded must be finite and non-negative")
    level_scale = level_scale_for_gamma(gamma_commanded, seed=seed, t0=t0_s)
    program = build_excitation_program(seed=seed, t0=t0_s, level_scale=level_scale)
    from shakebench.utils.tasks import task_env_kwargs

    variant = "task" in state
    task_kwargs = task_env_kwargs(state) if variant else {"object_start_xy": tuple(state["object_xy_m"])}
    env = shakebench.make(
        "VibrationPickPlace",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        geometry_profile=geometry_profile,
        imu_mode="canonical_noisy_v1",
        excitation_program=program,
        **task_kwargs,
        imu_seed=imu_seed,
        horizon=horizon_steps,
        ignore_done=True,
        seed=seed,
        # This is execution provenance only. It must not change any science
        # field; Phase 7.5B retains it only after complete episode parity.
        hard_reset=bool(hard_reset),
    )
    try:
        scene_identity = scene_visual_identity(env.scene_config)
        geometry_authority = geometry_authority_identity(geometry_profile)
        environment_scoreable = bool(env.get_policy_task_context()["physics_profile"]["scoreable"])
        if environment_scoreable != (bool(geometry_authority["scoreable"]) and not variant):
            raise OracleRunError("environment and geometry authority scoreability disagree")
        # Train-split states are collected for post-training, never scored.
        scoreable = environment_scoreable and state.get("split") != "train"
        task_context = WorktableTaskContext.from_mapping(env.get_policy_task_context().get("task_context"))
        controller = ShakeBenchOracleController(profile, task_context=task_context)
        observation = current_contract_observation(env, env.reset())
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        actuator_metadata = _actuator_metadata_from_model(raw_model)
        trace = []
        metrics: Mapping[str, Any] | dict[str, Any] | None = None
        episode_validity = "valid"
        score_outcome: str | None = None
        termination_cause: str | None = None
        for step in range(horizon_steps):
            policy_observation = observation
            if not _finite_json(policy_observation):
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                break
            try:
                normalized = controller.action(observation, time_s=step / profile.policy_rate_hz)
            except ShakeBenchOracleError:
                # An explicit controller-contract error on a finite public
                # observation is a policy result, not an infrastructure retry.
                termination_cause = "policy_error"
                score_outcome = "unsuccessful"
                break
            except Exception:
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                break
            if not np.all(np.isfinite(normalized)):
                termination_cause = "policy_error"
                score_outcome = "unsuccessful"
                break
            decoded = normalized.copy()
            decoded[:3] *= profile.position_action_range_m
            decoded[3:6] *= profile.orientation_action_range_rad
            clipped = np.clip(normalized, -1.0, 1.0)
            try:
                observation, _, _, _ = env.step(clipped)
                observation = current_contract_observation(env, observation)
            except Exception:
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                break
            applied = np.asarray(env.sim.data.ctrl, dtype=float).copy()
            actuator_force = np.asarray(env.sim.data.actuator_force, dtype=float).copy()
            if not (_finite_json(observation) and np.all(np.isfinite(applied)) and np.all(np.isfinite(actuator_force))):
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                break
            try:
                metrics_now = env.get_metrics()
            except Exception:
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                break
            metrics = metrics_now
            task_rule_violation = False
            trace.append(
                {
                    "step": step,
                    "policy_time_s": float(step / profile.policy_rate_hz),
                    "measurement_time_s": float(controller.last_trace["measurement_time_s"]),
                    "latency_s": float(controller.last_trace["latency_s"]),
                    "task_state": {key: policy_observation[key].copy() for key in ORACLE_TASK_KEYS},
                    "policy_input": {key: _public_copy(policy_observation[key]) for key in sorted(policy_observation)},
                    "provider_payload": {
                        key: _public_copy(policy_observation[key])
                        for key in controller.last_trace["provider_payload_keys"]
                    },
                    "post_task_state": {key: observation[key].copy() for key in ORACLE_TASK_KEYS},
                    "estimate": controller.last_trace["estimate"],
                    "relative_kinematics": controller.last_trace["relative_kinematics"],
                    "task_desired_action": controller.last_trace["task_desired_action"],
                    "phase_capability": controller.last_trace["phase_capability"],
                    "capability_limits": controller.last_trace["capability_limits"],
                    "post_capability_normalized_action": controller.last_trace["post_capability_normalized_action"],
                    "normalized_action": normalized.copy(),
                    "decoded_action": decoded,
                    "clipped_action": clipped,
                    "applied_actuator_ctrl": applied,
                    "applied_actuator_force": actuator_force,
                    "phase": controller.last_trace["phase"],
                    "controller_events": controller.executive.controller_events,
                    "recovery_count": controller.executive.recovery_count,
                    "environment_success_latched": bool(metrics_now["success"]["passed"]),
                    "task_rule_violation": task_rule_violation,
                }
            )
            if step_observer is not None:
                step_observer(env, step, observation, controller)
            termination_cause = resolve_termination_cause(
                prior_cause=termination_cause,
                task_rule_violation=task_rule_violation,
                success_latched=bool(metrics_now["success"]["passed"]),
                policy_abort=controller.abort_requested,
                horizon_exhausted=False,
            )
            if termination_cause is not None:
                score_outcome = "success" if termination_cause == "success_latched" else "unsuccessful"
                break
        if metrics is None:
            try:
                metrics = env.get_metrics()
            except Exception:
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                metrics = {}
        termination_cause = resolve_termination_cause(
            prior_cause=termination_cause,
            task_rule_violation=False,
            success_latched=bool(metrics.get("success", {}).get("passed")),
            policy_abort=False,
            horizon_exhausted=True,
        )
        if score_outcome is None:
            score_outcome = "success" if termination_cause == "success_latched" else "unsuccessful"
        if episode_validity == "invalid":
            score_outcome = None
        try:
            validate_outcome(
                episode_validity=episode_validity,
                score_outcome=score_outcome,
                termination_cause=termination_cause,
            )
            validate_controller_events(controller.executive.controller_events)
        except OutcomeContractError as exc:
            raise OracleRunError(f"outcome contract violation: {exc}") from exc
        success, failure_reason, termination_category = legacy_projection(
            episode_validity=episode_validity,
            score_outcome=score_outcome,
            termination_cause=termination_cause,
        )
        return {
            "schema_id": EPISODE_SCHEMA_ID,
            "schema_version": EPISODE_SCHEMA_VERSION,
            "state_id": state_id,
            "tier": CURRENT_TIER_ALIAS,
            "gamma_commanded": gamma_commanded,
            "horizon_steps": horizon_steps,
            "program": {"seed": seed, "t0_s": t0_s, "level_scale": level_scale},
            "controller_profile": profile.to_dict(),
            "task_context": task_context.to_dict(),
            "physics_profile": {
                "profile_id": env.physics_profile.profile_id,
            },
            "scene_visual": scene_identity,
            "geometry_profile": env.geometry_profile,
            "geometry_authority": geometry_authority,
            "scoreable": scoreable,
            **({"task_contract": env.task_spec.contract()} if variant else {}),
            "outcome_contract": outcome_contract(),
            "episode_validity": episode_validity,
            "score_outcome": score_outcome,
            "termination_cause": termination_cause,
            "controller_events": controller.executive.controller_events,
            "success": success,
            "failure_reason": failure_reason,
            "termination_category": termination_category,
            "actuators": actuator_metadata,
            "metrics": metrics,
            "trace": trace,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        default=str(Path(models.assets_root) / PHASE07_DEV_STATE_FILENAME),
        help="frozen ten-State dev JSON (defaults to the package asset)",
    )
    parser.add_argument(
        "--tier",
        choices=(CURRENT_TIER_ALIAS,),
        default=CURRENT_TIER_ALIAS,
        help="deprecated alias for the current single-lane expert; not an observation tier",
    )
    parser.add_argument("--gamma", type=float, required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--state-id", default=None, help="run one exact frozen dev state")
    parser.add_argument("--state-ids", default=None, help="comma-separated exact frozen dev state IDs")
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument(
        "--geometry-profile",
        choices=tuple(GEOMETRY_PROFILES),
        default=DEFAULT_GEOMETRY_PROFILE,
        help="explicit assembly profile; direct_mount_v1 remains Phase-07 requalification evidence until authorized",
    )
    args = parser.parse_args(argv)
    if args.gamma is None or args.output is None:
        parser.error("--gamma and --output are required for a normal run")
    state_asset = load_state_asset(args.states)
    states = state_asset["states"]
    if args.state_id is not None and args.state_ids is not None:
        raise OracleRunError("--state-id and --state-ids are mutually exclusive")
    if args.state_id is not None:
        states = [state for state in states if state["state_id"] == args.state_id]
        if len(states) != 1:
            raise OracleRunError("--state-id must name exactly one frozen dev state")
    if args.state_ids is not None:
        requested_ids = tuple(item.strip() for item in args.state_ids.split(",") if item.strip())
        states = [state for state in states if state["state_id"] in requested_ids]
        if len(states) != len(set(requested_ids)):
            raise OracleRunError("--state-ids must name exact, unique frozen dev states")
    if args.limit is not None:
        if args.state_id is not None or args.state_ids is not None:
            raise OracleRunError("--limit cannot be combined with explicit state IDs")
        if args.limit <= 0 or args.limit > len(states):
            raise OracleRunError("--limit must be in [1, 10]")
        states = states[: args.limit]
    states = sorted(states, key=lambda state: state["state_id"])
    profile = OracleControllerProfile()
    geometry_profile = load_geometry_profile(args.geometry_profile)
    scene_config = load_scene_visual_config(geometry_scene_path(args.geometry_profile))
    geometry_payload = geometry_profile if geometry_profile is not None else None
    geometry_authority = geometry_authority_identity(args.geometry_profile)
    scoreable = bool(geometry_authority["scoreable"]) and state_asset["authority"]["kind"] not in {
        "task_variants",
        "train",
    }
    scene_identity = scene_visual_identity(scene_config)
    if args.horizon_steps <= 0:
        raise OracleRunError("--horizon-steps must be positive")
    target = Path(args.output)
    episodes: list[dict[str, Any]] = []
    for state in states:
        episodes.append(
            run_episode(
                state,
                gamma_commanded=args.gamma,
                profile=profile,
                horizon_steps=args.horizon_steps,
                geometry_profile=args.geometry_profile,
            )
        )
    payload = {
        "schema_id": RUN_SCHEMA_ID,
        "schema_version": RUN_SCHEMA_VERSION,
        "tier": CURRENT_TIER_ALIAS,
        "gamma_commanded": args.gamma,
        "controller_profile": profile.to_dict(),
        "evaluator_post_complete_settle_s": profile.completion_evaluator_settle_s,
        "dev_state_anchor": (
            _dev_state_anchor(args.states) if state_asset["authority"]["kind"] == "dev" else _dev_state_anchor()
        ),
        "state_authority": state_asset["authority"],
        "physics_authority": {
            "profile_id": OFFICIAL_PHYSICS_PROFILE_ID,
        },
        "scene_visual": scene_identity,
        "geometry_profile": geometry_payload,
        "geometry_authority": geometry_authority,
        "scoreable": scoreable,
        "outcome_contract": outcome_contract(),
        "episodes": episodes,
    }
    write_json(target, payload)
    print(
        json.dumps(
            {
                "output": str(target),
                "successes": sum(row["success"] for row in episodes),
            },
            sort_keys=True,
        )
    )
    return 0


def _terminal_cause_from_trace(trace: list[Mapping[str, Any]]) -> str | None:
    """Return the runner-priority terminal cause evidenced by the final row."""

    if not trace:
        return None
    final = trace[-1]
    phase = final.get("phase")
    return resolve_termination_cause(
        prior_cause=None,
        task_rule_violation=final.get("task_rule_violation") is True,
        success_latched=final.get("environment_success_latched") is True,
        policy_abort=isinstance(phase, Mapping) and phase.get("phase") == "aborted",
        horizon_exhausted=False,
    )


def verify_run_artifact(path: str | Path) -> dict[str, Any]:
    """Semantically reverify a run without trusting stored summaries or hashes."""

    errors: list[str] = []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"artifact read: {exc}"]}
    if not isinstance(payload, Mapping):
        return {"passed": False, "errors": ["run payload must be an object"]}
    if payload.get("schema_id") != RUN_SCHEMA_ID or payload.get("schema_version") != RUN_SCHEMA_VERSION:
        errors.append("run schema")
    if payload.get("outcome_contract") != outcome_contract():
        errors.append("outcome contract authority")
    required_top_level = {
        "schema_id",
        "schema_version",
        "tier",
        "gamma_commanded",
        "controller_profile",
        "evaluator_post_complete_settle_s",
        "dev_state_anchor",
        "physics_authority",
        "scene_visual",
        "geometry_profile",
        "geometry_authority",
        "scoreable",
        "outcome_contract",
        "episodes",
    }
    if set(payload) - {"process", "state_authority"} != required_top_level:
        errors.append("run required fields")
    if not _finite_json(payload):
        errors.append("nonfinite JSON")
    state_authority = payload.get("state_authority")
    committed_split = None
    if state_authority is None or (isinstance(state_authority, Mapping) and state_authority.get("kind") == "dev"):
        expected_anchor = _dev_state_anchor()
        anchor_matches, anchor_mode = _dev_state_anchor_match(payload.get("dev_state_anchor"), expected_anchor)
        if not anchor_matches:
            errors.append("dev-state anchor")
        states_by_id = {row["state_id"]: row for row in load_dev_states(_default_dev_state_path())}
    elif isinstance(state_authority, Mapping) and state_authority.get("kind") in {"committed", "task_variants"}:
        committed_split = state_authority.get("split")
        from shakebench.utils.committed_states import KNEE_STATE_FILENAME, OFFICIAL_STATE_FILENAME

        if committed_split not in {"official", "knee"}:
            errors.append("committed-state split")
            states_by_id = {}
        else:
            asset = Path(models.assets_root) / (
                OFFICIAL_STATE_FILENAME if committed_split == "official" else KNEE_STATE_FILENAME
            )
            if state_authority.get("kind") == "task_variants":
                from shakebench.utils.task_states import TASK_STATE_FILENAMES

                asset = Path(models.assets_root) / TASK_STATE_FILENAMES[committed_split]
            try:
                resolved = load_state_asset(asset)
                if resolved["authority"] != state_authority:
                    errors.append("committed-state authority")
                states_by_id = {row["state_id"]: row for row in resolved["states"]}
            except OracleRunError as exc:
                errors.append(f"committed-state authority: {exc}")
                states_by_id = {}
        anchor_mode = "committed"
    else:
        errors.append("state authority")
        states_by_id = {}
        anchor_mode = "invalid"
    if not _values_equal(
        payload.get("physics_authority"),
        {"profile_id": OFFICIAL_PHYSICS_PROFILE_ID},
        atol=0.0,
    ):
        errors.append("physics authority")
    expected_scene_visual = None
    expected_geometry_profile = None
    expected_geometry_authority = None
    try:
        geometry_payload = payload.get("geometry_profile")
        if geometry_payload is None:
            geometry_name = "canonical"
        elif isinstance(geometry_payload, Mapping) and isinstance(geometry_payload.get("profile_id"), str):
            geometry_name = str(geometry_payload["profile_id"])
            expected_geometry_profile = load_geometry_profile(geometry_name)
            if not _values_equal(geometry_payload, expected_geometry_profile, atol=0.0):
                errors.append("geometry authority")
        else:
            raise ValueError("geometry_profile must be null or a profile mapping")
        scene_config = load_scene_visual_config(geometry_scene_path(geometry_name))
        expected_scene_visual = scene_visual_identity(scene_config)
        expected_geometry_authority = geometry_authority_identity(geometry_name)
        if not _values_equal(payload.get("scene_visual"), expected_scene_visual, atol=0.0):
            errors.append("scene visual authority")
        if not _values_equal(payload.get("geometry_authority"), expected_geometry_authority, atol=0.0):
            errors.append("geometry authorization")
        variant_run = isinstance(state_authority, Mapping) and state_authority.get("kind") == "task_variants"
        if payload.get("scoreable") is not (bool(expected_geometry_authority["scoreable"]) and not variant_run):
            errors.append("scoreable authority")
    except (OSError, TypeError, ValueError):
        errors.append("scene visual authority unavailable")
    try:
        parsed_profile = _profile_from_payload(payload.get("controller_profile"))
        expected_profile = parsed_profile.to_dict()
    except (OracleRunError, TypeError, ValueError, ShakeBenchOracleError) as exc:
        errors.append(f"controller profile: {exc}")
        parsed_profile = OracleControllerProfile()
        expected_profile = parsed_profile.to_dict()
    if not _values_equal(payload.get("controller_profile"), expected_profile, atol=1.0e-12):
        errors.append("controller profile")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("episodes")
        episodes = []
    seen_state_ids: set[str] = set()
    expected_ids = []
    for episode_index, episode in enumerate(episodes):
        prefix = f"episode[{episode_index}]"
        if not isinstance(episode, Mapping):
            errors.append(prefix + " object")
            continue
        expected_ids.append(episode.get("state_id"))
        if episode.get("schema_id") != EPISODE_SCHEMA_ID or episode.get("schema_version") != EPISODE_SCHEMA_VERSION:
            errors.append(prefix + " schema")
        required_episode_fields = {
            "schema_id",
            "schema_version",
            "state_id",
            "tier",
            "gamma_commanded",
            "horizon_steps",
            "program",
            "controller_profile",
            "task_context",
            "physics_profile",
            "scene_visual",
            "geometry_profile",
            "geometry_authority",
            "scoreable",
            "outcome_contract",
            "episode_validity",
            "score_outcome",
            "termination_cause",
            "controller_events",
            "success",
            "failure_reason",
            "termination_category",
            "actuators",
            "metrics",
            "trace",
        }
        if isinstance(state_authority, Mapping) and state_authority.get("kind") == "task_variants":
            required_episode_fields.add("task_contract")
        if set(episode) != required_episode_fields:
            errors.append(prefix + " required fields")
        if episode.get("outcome_contract") != outcome_contract():
            errors.append(prefix + " outcome contract authority")
        tier = episode.get("tier")
        if tier != payload.get("tier"):
            errors.append(prefix + " tier")
        if float(episode.get("gamma_commanded", float("nan"))) != float(payload.get("gamma_commanded", float("nan"))):
            errors.append(prefix + " gamma")
        state_id = episode.get("state_id")
        if not isinstance(state_id, str) or state_id in seen_state_ids or state_id not in states_by_id:
            errors.append(prefix + " state binding")
            state = None
        else:
            seen_state_ids.add(state_id)
            state = states_by_id[state_id]
            if "task" in state:
                from shakebench.utils.tasks import OBJECT_SUPPORT, TaskSpec

                spec = TaskSpec.from_mapping(state["task"])
                if episode.get("task_contract") != spec.contract():
                    errors.append(prefix + " task contract binding")
                context = episode.get("task_context", {})
                lower, upper, radius = OBJECT_SUPPORT[spec.object_id]
                for key, value in (
                    ("object_collision_lower_support_m", lower),
                    ("object_collision_upper_support_m", upper),
                    ("object_collision_radius_m", radius),
                ):
                    if not _values_equal(context.get(key), value, atol=1e-10):
                        errors.append(prefix + " object geometry binding")
        if not _values_equal(episode.get("controller_profile"), expected_profile, atol=1.0e-12):
            errors.append(prefix + " controller profile")
        try:
            task_context = WorktableTaskContext.from_mapping(episode.get("task_context"))
        except (TypeError, ValueError, ShakeBenchOracleError):
            errors.append(prefix + " task context")
            task_context = WorktableTaskContext()
        physics = episode.get("physics_profile")
        if not _values_equal(
            physics,
            {"profile_id": OFFICIAL_PHYSICS_PROFILE_ID},
            atol=0.0,
        ):
            errors.append(prefix + " physics binding")
        if expected_scene_visual is not None and not _values_equal(
            episode.get("scene_visual"), expected_scene_visual, atol=0.0
        ):
            errors.append(prefix + " scene visual binding")
        if not _values_equal(episode.get("geometry_profile"), expected_geometry_profile, atol=0.0):
            errors.append(prefix + " geometry binding")
        if not _values_equal(episode.get("geometry_authority"), expected_geometry_authority, atol=0.0):
            errors.append(prefix + " geometry authority binding")
        if episode.get("scoreable") is not payload.get("scoreable"):
            errors.append(prefix + " scoreable binding")
        if not isinstance(episode.get("program"), Mapping) or state is None:
            errors.append(prefix + " program")
        else:
            expected_level = level_scale_for_gamma(
                float(episode["gamma_commanded"]),
                seed=state.get("excitation_seed", state.get("seed")),
                t0=state["t0_s"],
            )
            expected_program = {
                "seed": state.get("excitation_seed", state.get("seed")),
                "t0_s": state["t0_s"],
                "level_scale": expected_level,
            }
            if not _values_equal(episode["program"], expected_program, atol=1.0e-12):
                errors.append(prefix + " program binding")
        _compare_actuator_metadata(episode.get("actuators"), errors)
        trace = episode.get("trace")
        if not isinstance(trace, list) or (not trace and episode.get("episode_validity") != "invalid"):
            errors.append(prefix + " trace")
            trace = []
        controller = ShakeBenchOracleController(parsed_profile, task_context=task_context)
        expected_keys = set(ORACLE_TASK_KEYS) | set(TABLE_IMU_POLICY_KEYS)
        contract = {
            **{wire: POLICY_FIELD_CONTRACT[_CONTRACT_KEY_FOR_WIRE[wire]] for wire in ORACLE_TASK_KEYS},
            **{key: POLICY_FIELD_CONTRACT[key] for key in TABLE_IMU_POLICY_KEYS},
        }
        for row_index, row in enumerate(trace):
            row_prefix = f"{prefix}.trace[{row_index}]"
            if not isinstance(row, Mapping):
                errors.append(row_prefix + " object")
                continue
            required_row_keys = {
                "step",
                "policy_time_s",
                "measurement_time_s",
                "latency_s",
                "task_state",
                "policy_input",
                "provider_payload",
                "post_task_state",
                "estimate",
                "relative_kinematics",
                "task_desired_action",
                "phase_capability",
                "capability_limits",
                "post_capability_normalized_action",
                "normalized_action",
                "decoded_action",
                "clipped_action",
                "applied_actuator_ctrl",
                "applied_actuator_force",
                "phase",
                "controller_events",
                "recovery_count",
                "environment_success_latched",
                "task_rule_violation",
            }
            if set(row) != required_row_keys:
                errors.append(row_prefix + " schema")
            if row.get("step") != row_index:
                errors.append(row_prefix + " step sequence")
            if not isinstance(row.get("environment_success_latched"), bool):
                errors.append(row_prefix + " environment success status")
            if not isinstance(row.get("task_rule_violation"), bool):
                errors.append(row_prefix + " task-rule status")
            try:
                policy_time = float(row["policy_time_s"])
                if not np.isclose(policy_time, row_index / expected_profile["policy_rate_hz"], rtol=0.0, atol=1.0e-12):
                    errors.append(row_prefix + " policy time")
            except (KeyError, TypeError, ValueError):
                errors.append(row_prefix + " policy time")
                policy_time = float(row_index / expected_profile["policy_rate_hz"])
            policy_input = row.get("policy_input")
            if (
                not isinstance(policy_input, Mapping)
                or set(policy_input) != expected_keys
                or not _finite_json(policy_input)
            ):
                errors.append(row_prefix + " policy input key set")
                continue
            for key, field_contract in contract.items():
                if not _array_contract_ok(policy_input.get(key), field_contract):
                    errors.append(row_prefix + f" field {key}")
            provider_payload = row.get("provider_payload")
            if not isinstance(provider_payload, Mapping) or set(provider_payload) != set(TABLE_IMU_POLICY_KEYS):
                errors.append(row_prefix + " provider payload key set")
            elif any(not _values_equal(provider_payload[key], policy_input[key], atol=0.0) for key in provider_payload):
                errors.append(row_prefix + " provider payload mismatch")
            task_state = row.get("task_state")
            if not isinstance(task_state, Mapping) or not _values_equal(
                {key: policy_input.get(key) for key in ORACLE_TASK_KEYS}, task_state, atol=0.0
            ):
                errors.append(row_prefix + " task state mismatch")
            post_task_state = row.get("post_task_state")
            try:
                recomputed_action = controller.action(policy_input, time_s=policy_time)
                recomputed_trace = controller.last_trace
                if not _values_equal(row["estimate"], recomputed_trace["estimate"], atol=1.0e-6):
                    errors.append(row_prefix + " estimate recomputation")
                if not _values_equal(row["phase"], recomputed_trace["phase"], atol=1.0e-6):
                    errors.append(row_prefix + " phase recomputation")
                if not _values_equal(row["controller_events"], controller.executive.controller_events, atol=1.0e-6):
                    errors.append(row_prefix + " controller events recomputation")
                if row.get("recovery_count") != controller.executive.recovery_count:
                    errors.append(row_prefix + " recovery count recomputation")
                if not _values_equal(row["relative_kinematics"], recomputed_trace["relative_kinematics"], atol=1.0e-6):
                    errors.append(row_prefix + " relative kinematics recomputation")
                for trace_key in (
                    "task_desired_action",
                    "phase_capability",
                    "capability_limits",
                    "post_capability_normalized_action",
                ):
                    if not _values_equal(row[trace_key], recomputed_trace[trace_key], atol=1.0e-6):
                        errors.append(row_prefix + f" {trace_key} recomputation")
                if not _values_equal(row["normalized_action"], recomputed_action.tolist(), atol=1.0e-6):
                    errors.append(row_prefix + " normalized action recomputation")
                decoded = np.asarray(recomputed_action, dtype=float)
                decoded[:3] *= expected_profile["position_action_range_m"]
                decoded[3:6] *= expected_profile["orientation_action_range_rad"]
                clipped = np.clip(recomputed_action, -1.0, 1.0)
                if not _values_equal(row["decoded_action"], decoded.tolist(), atol=1.0e-6):
                    errors.append(row_prefix + " decoded action recomputation")
                if not _values_equal(row["clipped_action"], clipped.tolist(), atol=1.0e-6):
                    errors.append(row_prefix + " clipped action recomputation")
                if not np.isclose(
                    float(row["measurement_time_s"]), recomputed_trace["measurement_time_s"], atol=1.0e-6
                ):
                    errors.append(row_prefix + " measurement timestamp")
                if not np.isclose(float(row["latency_s"]), recomputed_trace["latency_s"], atol=1.0e-6):
                    errors.append(row_prefix + " latency")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(row_prefix + f" controller replay: {exc}")
            for key in ("task_state", "post_task_state"):
                state_payload = row.get(key)
                if not isinstance(state_payload, Mapping) or set(state_payload) != set(ORACLE_TASK_KEYS):
                    errors.append(row_prefix + f" {key} schema")
                elif not _finite_json(state_payload):
                    errors.append(row_prefix + f" {key} finite")
            for key in ("normalized_action", "decoded_action", "clipped_action"):
                try:
                    array = np.asarray(row.get(key), dtype=float)
                    valid = array.shape == (7,) and np.all(np.isfinite(array))
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(row_prefix + f" {key} shape/finite")
            for key in ("applied_actuator_ctrl", "applied_actuator_force"):
                try:
                    array = np.asarray(row.get(key), dtype=float)
                    valid = array.shape == (len(_official_actuator_metadata()),) and np.all(np.isfinite(array))
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(row_prefix + f" {key} shape/finite")
            estimate_payload = row.get("estimate")
            if not isinstance(estimate_payload, Mapping) or not _values_equal(
                estimate_payload.get("policy_timestamp_s"), row.get("policy_time_s"), atol=1.0e-6
            ):
                errors.append(row_prefix + " estimate policy timestamp")
            if isinstance(estimate_payload, Mapping):
                if estimate_payload.get("schema_id") != "shakebench.relative_support_motion_estimate":
                    errors.append(row_prefix + " estimate schema")
                if estimate_payload.get("schema_version") != 1:
                    errors.append(row_prefix + " estimate schema version")
        success = episode.get("success")
        termination = episode.get("termination_category")
        failure = episode.get("failure_reason")
        try:
            validate_outcome(
                episode_validity=episode.get("episode_validity"),
                score_outcome=episode.get("score_outcome"),
                termination_cause=episode.get("termination_cause"),
            )
            validate_controller_events(episode.get("controller_events"))
            projected = legacy_projection(
                episode_validity=episode["episode_validity"],
                score_outcome=episode["score_outcome"],
                termination_cause=episode["termination_cause"],
            )
            if (success, failure, termination) != projected:
                errors.append(prefix + " legacy outcome projection")
        except (OutcomeContractError, KeyError):
            errors.append(prefix + " outcome integrity")
        try:
            trace_cause = _terminal_cause_from_trace(trace)
            recorded_cause = episode.get("termination_cause")
            horizon_steps = int(episode.get("horizon_steps"))
            if recorded_cause in {"task_rule_violation", "success_latched", "policy_abort"}:
                if trace_cause != recorded_cause:
                    errors.append(prefix + " terminal trace evidence")
            elif recorded_cause == "horizon_exhausted":
                if trace_cause is not None or len(trace) != horizon_steps:
                    errors.append(prefix + " horizon trace evidence")
            elif recorded_cause in {"policy_error", "invalid_execution"}:
                if len(trace) >= horizon_steps:
                    errors.append(prefix + " early terminal trace evidence")
        except (TypeError, ValueError):
            errors.append(prefix + " terminal trace evidence")
        if not _values_equal(episode.get("controller_events"), controller.executive.controller_events, atol=1.0e-6):
            errors.append(prefix + " controller event ledger")
        if not isinstance(episode.get("metrics"), Mapping) or (
            episode.get("episode_validity") != "invalid" and not _finite_json(episode.get("metrics"))
        ):
            errors.append(prefix + " metrics schema/finite")
    if expected_ids != sorted(expected_ids):
        errors.append("state ordering")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "trace_count": sum(len(row.get("trace", ())) for row in episodes if isinstance(row, Mapping)),
        "dev_state_anchor_mode": anchor_mode,
    }


if __name__ == "__main__":
    raise SystemExit(main())
