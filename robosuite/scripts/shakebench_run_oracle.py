"""Run the shared State Oracle controller on explicitly supplied dev states.

This runner consumes the Phase 07 pre-registered ten-state dev asset. Phase 08
retains ownership of official/knee state generation and absorbs this frozen
dev subset into the unified committed-state protocol without regenerating it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import robosuite
from robosuite import models
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.shakebench_authority import DirectMountAuthorityError, verify_direct_mount_authority
from robosuite.utils.shakebench_calibration import level_scale_for_gamma
from robosuite.utils.shakebench_dev_states import (
    PHASE07_DEV_STATE_FILENAME,
    verify_phase07_dev_state_artifact,
)
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.utils.shakebench_geometry import geometry_scene_path, load_geometry_profile
from robosuite.utils.shakebench_oracle import (
    OracleControllerProfile,
    ShakeBenchOracleController,
    ShakeBenchOracleError,
    WorktableTaskContext,
    profile_for_diagnostic_mode,
)
from robosuite.utils.shakebench_outcomes import (
    TERMINATION_CAUSES,
    OutcomeContractError,
    legacy_projection,
    outcome_contract,
    outcome_contract_sha256,
    resolve_termination_cause,
    validate_controller_events,
    validate_outcome,
)
from robosuite.utils.shakebench_providers import COMMON_STATE_KEYS, TIER_POLICY_KEYS, observation_contract_for_tier
from robosuite.utils.shakebench_scene import load_scene_visual_config


class OracleRunError(RuntimeError):
    """Raised for invalid or incomplete Phase 07 run inputs."""


RUN_SCHEMA_ID = "shakebench.phase07.oracle_run"
RUN_SCHEMA_VERSION = 6
EPISODE_SCHEMA_ID = "shakebench.phase07.oracle_episode"
EPISODE_SCHEMA_VERSION = 6
LEGACY_RUN_SCHEMA_VERSION = 5
LEGACY_EPISODE_SCHEMA_VERSION = 5
DETERMINISM_SCHEMA_ID = "shakebench.phase07.determinism_manifest"
DETERMINISM_SCHEMA_VERSION = 5
DEV_STATE_PRE_HISTORY_REWRITE_COMMIT = "dd6fe2edb6384ccdb5116be44f07592b4864e377"
DEV_STATE_REWRITTEN_COMMIT = "08626ea5a5e107df503e266be9065b929d47f882"
DEV_STATE_ANCHOR_REWRITE = {
    DEV_STATE_PRE_HISTORY_REWRITE_COMMIT: DEV_STATE_REWRITTEN_COMMIT,
}
# Compatibility name: new provenance must use the rewritten anchor.
DEV_STATE_ANCHOR_COMMIT = DEV_STATE_REWRITTEN_COMMIT
OFFICIAL_PHYSICS_PROFILE_ID = "shakebench.official.physics.v2"
OFFICIAL_PHYSICS_PROFILE_SHA256 = "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c"
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


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


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
        xy = np.asarray(row.get("can_xy_m"), dtype=float)
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id in seen
            or xy.shape != (2,)
            or not np.all(np.isfinite(xy))
        ):
            raise OracleRunError("dev states require unique state_id and finite can_xy_m[2]")
        seen.add(state_id)
        result.append(
            {
                "state_id": state_id,
                "can_xy_m": xy.tolist(),
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
    from robosuite.utils.shakebench_task_states import TASK_STATE_SCHEMA, verify_task_state_artifact

    if payload.get("schema_id") == TASK_STATE_SCHEMA:
        verdict = verify_task_state_artifact(payload)
        if not verdict["passed"]:
            raise OracleRunError("task state authority failed: " + ", ".join(verdict["errors"]))
        return {"states": payload["states"], "authority": {
            "kind": "task_variants", "split": payload["split"], "scoreable": False,
            "asset_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "asset_payload_sha256": payload["payload_sha256"],
            "authority_hashes": dict(payload["authority_hashes"]),
        }}
    from robosuite.utils.shakebench_committed_states import verify_committed_state_artifact

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
            "asset_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "asset_payload_sha256": payload["artifact_lock"]["payload_sha256"],
            "authority_hashes": dict(payload["authority_hashes"]),
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
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "payload_sha256": payload["artifact_lock"]["payload_sha256"],
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


def verify_phase07_r4_manifest(path: str | Path = "docs/phase_07_r4_manifest.json") -> dict[str, Any]:
    """Verify the compact R4 entry binding without using the mutable R5 profile."""

    manifest_path = Path(path)
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"manifest read: {exc}"], "checks": {}}
    if not isinstance(manifest, Mapping) or manifest.get("phase") != "07R4":
        errors.append("manifest phase")
        manifest = manifest if isinstance(manifest, Mapping) else {}
    artifact_ref = manifest.get("v0_gamma_000") if isinstance(manifest, Mapping) else None
    artifact_path = (
        Path(str(artifact_ref.get("path")))
        if isinstance(artifact_ref, Mapping) and artifact_ref.get("path")
        else Path("out/phase07r4/v0_gamma_000_final.json")
    )
    if not artifact_path.is_file():
        errors.append("R4 final artifact missing")
    else:
        file_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if isinstance(artifact_ref, Mapping) and file_hash != artifact_ref.get("file_sha256"):
            errors.append("R4 final artifact file hash")
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            artifact = {}
            errors.append(f"R4 final artifact read: {exc}")
        if isinstance(artifact, Mapping):
            if artifact.get("payload_sha256") != artifact_ref.get("payload_sha256"):
                errors.append("R4 final artifact payload hash")
            episodes = artifact.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != 10:
                errors.append("R4 dev episode count")
            else:
                if sum(bool(episode.get("success")) for episode in episodes if isinstance(episode, Mapping)) != 10:
                    errors.append("R4 dev success count")
                state002 = next(
                    (
                        episode
                        for episode in episodes
                        if isinstance(episode, Mapping) and episode.get("state_id") == "shakebench-dev-v0-002"
                    ),
                    None,
                )
                if not isinstance(state002, Mapping) or state002.get("termination_category") != "environment_success":
                    errors.append("R4 state-002 success")
            for key in ("controller_profile", "dev_state_anchor", "physics_authority"):
                if key not in artifact:
                    errors.append(f"R4 artifact binding {key}")
            if artifact.get("controller_profile", {}).get("profile_id") != manifest.get("controller_profile_id"):
                errors.append("R4 profile binding")
            if artifact.get("physics_authority", {}).get("profile_id") != manifest.get("physics_profile_id"):
                errors.append("R4 physics binding")
    verifier_paths = {
        "semantic_file_sha256": "out/phase07r4/semantic_verdict_final.json",
        "actuator_file_sha256": "out/phase07r4/actuator_controls_final.json",
        "determinism_file_sha256": "out/phase07r4/determinism_manifest_final.json",
        "package_file_sha256": "out/phase07r4/package_evidence_final.json",
    }
    verifier_bindings = manifest.get("verifiers", {}) if isinstance(manifest, Mapping) else {}
    for key, relative_path in verifier_paths.items():
        verifier_path = Path(relative_path)
        if not verifier_path.is_file():
            errors.append(f"R4 verifier missing {relative_path}")
        elif hashlib.sha256(verifier_path.read_bytes()).hexdigest() != verifier_bindings.get(key):
            errors.append(f"R4 verifier hash {key}")
    checks = {
        "state_002_environment_success": "R4 state-002 success" not in errors,
        "v0_gamma_000_10_of_10": "R4 dev success count" not in errors,
        "final_artifact_hashes": not any("R4 final artifact" in error for error in errors),
        "profile_and_physics_binding": not any("binding" in error for error in errors),
        "verifier_artifacts": not any("R4 verifier" in error for error in errors),
    }
    return {"passed": not errors, "errors": sorted(set(errors)), "checks": checks, "manifest_path": str(manifest_path)}


def scene_visual_identity(scene_config) -> dict[str, Any]:
    """Single writer/verifier representation of the scene visual authority."""

    return {
        "scene_id": scene_config.scene_id,
        "config_sha256": scene_config.config_sha256,
        "geometry_variant": scene_config.geometry_variant,
        "physics_effect": scene_config.physics_effect,
    }


def geometry_authority_identity(geometry_profile: str, *, allow_unverified: bool = False) -> dict[str, Any]:
    """Return the v6 geometry and outcome authority for a rollout."""

    if geometry_profile == "canonical":
        return {
            "kind": "canonical",
            "scoreable": True,
            "controller_profile_sha256": OracleControllerProfile().sha256,
            "outcome_contract_sha256": outcome_contract_sha256(),
        }
    try:
        return {
            "kind": "phase08r",
            "scoreable": True,
            "prior_geometry_authority": dict(verify_direct_mount_authority()),
            "controller_profile_sha256": OracleControllerProfile().sha256,
            "outcome_contract_sha256": outcome_contract_sha256(),
        }
    except DirectMountAuthorityError as exc:
        if allow_unverified:
            return {
                "kind": "phase08r",
                "scoreable": False,
                "verification": "not_authorized_for_scoreable_run",
                "controller_profile_sha256": OracleControllerProfile().sha256,
                "outcome_contract_sha256": outcome_contract_sha256(),
            }
        raise OracleRunError(f"direct-mount authority failed: {exc}") from exc


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


@lru_cache(maxsize=1)
def _official_actuator_metadata() -> tuple[dict[str, Any], ...]:
    env = robosuite.make(
        "VibrationPickPlaceCan",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        observation_tier="V0",
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
    profile_values = dict(value)
    profile_values.pop("profile_sha256", None)
    return OracleControllerProfile(**profile_values)


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
    tier: str,
    gamma_commanded: float,
    profile: OracleControllerProfile,
    horizon_steps: int = 1200,
    geometry_profile: str = "canonical",
    allow_unverified_geometry: bool = False,
    hard_reset: bool = True,
    step_observer: Callable[[Any, int, Mapping[str, Any], ShakeBenchOracleController], None] | None = None,
) -> dict[str, Any]:
    """Run a public-observation-only episode and retain action/actuator evidence.

    ``step_observer`` is a qualitative-demo hook. It receives the environment
    after each policy step, but rendered pixels never enter the controller
    observation or the scoreable trace.
    """

    state_id = str(state["state_id"])
    seed = int(state.get("excitation_seed", state.get("seed", 0)))
    imu_seed = int(state.get("imu_seed", seed))
    t0_s = float(state.get("t0_s", 0.0))
    gamma_commanded = float(gamma_commanded)
    if gamma_commanded < 0.0 or not np.isfinite(gamma_commanded):
        raise OracleRunError("gamma_commanded must be finite and non-negative")
    level_scale = level_scale_for_gamma(gamma_commanded, seed=seed, t0=t0_s)
    program = build_excitation_program(seed=seed, t0=t0_s, level_scale=level_scale)
    from robosuite.utils.shakebench_tasks import legacy_oracle_observation, task_env_kwargs

    variant = "task" in state
    task_kwargs = task_env_kwargs(state) if variant else {"can_start_xy": tuple(state["can_xy_m"])}
    env = robosuite.make(
        "VibrationPickPlace" if variant else "VibrationPickPlaceCan",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        geometry_profile=geometry_profile,
        observation_tier=tier,
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
        geometry_authority = geometry_authority_identity(geometry_profile, allow_unverified=allow_unverified_geometry)
        scoreable = bool(env.get_policy_task_context()["physics_profile"]["scoreable"])
        if scoreable != (bool(geometry_authority["scoreable"]) and not variant):
            raise OracleRunError("environment and geometry authority scoreability disagree")
        task_context = WorktableTaskContext.from_mapping(env.get_policy_task_context().get("task_context"))
        controller = ShakeBenchOracleController(tier, profile, task_context=task_context)
        observation = legacy_oracle_observation(env.reset())
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        actuator_metadata = _actuator_metadata_from_model(raw_model)
        trace = []
        metrics: Mapping[str, Any] | dict[str, Any] | None = None
        episode_validity = "valid"
        score_outcome: str | None = None
        termination_cause: str | None = None
        complete_event_recorded = False
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
            if controller.executive.phase.value == "complete" and not complete_event_recorded:
                controller.executive.record_evaluator_not_latched(
                    observation, step / profile.policy_rate_hz
                )
                complete_event_recorded = True
            decoded = normalized.copy()
            decoded[:3] *= profile.position_action_range_m
            decoded[3:6] *= profile.orientation_action_range_rad
            clipped = np.clip(normalized, -1.0, 1.0)
            try:
                observation, _, _, _ = env.step(clipped)
                observation = legacy_oracle_observation(observation)
            except Exception:
                episode_validity = "invalid"
                termination_cause = "invalid_execution"
                break
            applied = np.asarray(env.sim.data.ctrl, dtype=float).copy()
            actuator_force = np.asarray(env.sim.data.actuator_force, dtype=float).copy()
            if not (
                _finite_json(observation)
                and np.all(np.isfinite(applied))
                and np.all(np.isfinite(actuator_force))
            ):
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
            task_rule_violation = bool(
                metrics_now["max_illegal_penetration_m"]
                >= env.physics_profile.physics["safety"]["maximum_illegal_penetration_m"]
            )
            trace.append(
                {
                    "step": step,
                    "policy_time_s": float(step / profile.policy_rate_hz),
                    "measurement_time_s": float(controller.last_trace["measurement_time_s"]),
                    "latency_s": float(controller.last_trace["latency_s"]),
                    "task_state": {key: policy_observation[key].copy() for key in COMMON_STATE_KEYS},
                    "task_state_sha256": _digest({key: policy_observation[key].copy() for key in COMMON_STATE_KEYS}),
                    "policy_input": {key: policy_observation[key].copy() for key in sorted(policy_observation)},
                    "provider_payload": {
                        key: policy_observation[key].copy() for key in controller.last_trace["provider_payload_keys"]
                    },
                    "post_task_state": {key: observation[key].copy() for key in COMMON_STATE_KEYS},
                    "post_task_state_sha256": _digest({key: observation[key].copy() for key in COMMON_STATE_KEYS}),
                    "estimate": controller.last_trace["estimate"],
                    "control_law": controller.last_trace["control_law"],
                    "relative_kinematics": controller.last_trace["relative_kinematics"],
                    "task_desired_action": controller.last_trace["task_desired_action"],
                    "pre_capability_compensated_action": controller.last_trace["pre_capability_compensated_action"],
                    "phase_capability": controller.last_trace["phase_capability"],
                    "capability_limits": controller.last_trace["capability_limits"],
                    "post_capability_normalized_action": controller.last_trace["post_capability_normalized_action"],
                    "normalized_action": normalized.copy(),
                    "decoded_action": decoded,
                    "clipped_action": clipped,
                    "applied_actuator_ctrl": applied,
                    "applied_actuator_ctrl_sha256": _digest(applied),
                    "applied_actuator_force": actuator_force,
                    "applied_actuator_force_sha256": _digest(actuator_force),
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
            if controller.executive.phase.value == "complete":
                continue
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
            "tier": tier,
            "gamma_commanded": gamma_commanded,
            "horizon_steps": horizon_steps,
            "program": {"seed": seed, "t0_s": t0_s, "level_scale": level_scale},
            "controller_profile": profile.to_dict(),
            "task_context": task_context.to_dict(),
            "task_context_sha256": task_context.sha256,
            "controller_context_hash": controller.controller_context_hash,
            "physics_profile": {
                "profile_id": env.physics_profile.profile_id,
                "profile_sha256": env.physics_profile.profile_sha256,
            },
            "scene_visual": scene_identity,
            "geometry_profile": env.geometry_profile,
            "geometry_authority": geometry_authority,
            "scoreable": scoreable,
            "state_sha256": _digest(state),
            **({"task_contract": env.task_spec.contract()} if variant else {}),
            "outcome_contract": outcome_contract(),
            "outcome_contract_sha256": outcome_contract_sha256(),
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
            "trace_sha256": _digest(trace),
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
    parser.add_argument("--tier", choices=("V0", "V1", "V2", "V3"), required=False)
    parser.add_argument("--gamma", type=float, required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--state-id", default=None, help="run one exact frozen dev state")
    parser.add_argument("--state-ids", default=None, help="comma-separated exact frozen dev state IDs")
    parser.add_argument(
        "--merge-into", default=None, help="replace selected episodes in an existing compatible raw run"
    )
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument(
        "--geometry-profile",
        choices=("canonical", "direct_mount_v1"),
        default="canonical",
        help="explicit assembly profile; direct_mount_v1 remains Phase-07 requalification evidence until authorized",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=(
            "main",
            "task_executive_only",
            "compensation_off",
            "common_public_history",
            "current_state_compensation",
            "preview_off",
            "preview_on",
        ),
        default="main",
        help="explicit R5 diagnostic profile; main is the scoreable reference profile",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a partial per-episode checkpoint next to --output",
    )
    parser.add_argument("--determinism-manifest", default=None)
    parser.add_argument("--process-index", type=int, default=None)
    parser.add_argument("--parent-run-uuid", default=None)
    args = parser.parse_args(argv)
    if args.determinism_manifest is not None:
        if args.state_id is None and args.state_ids is None:
            args.state_id = "shakebench-dev-v0-000"
        return run_determinism_replay(
            args.determinism_manifest,
            state_id=args.state_id,
            state_ids=(None if args.state_ids is None else tuple(item.strip() for item in args.state_ids.split(",") if item.strip())),
            states_path=args.states,
            horizon_steps=args.horizon_steps,
            geometry_profile=args.geometry_profile,
        )["exit_code"]
    if args.tier is None or args.gamma is None or args.output is None:
        parser.error("--tier, --gamma, and --output are required for a normal run")
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
    profile = profile_for_diagnostic_mode(args.diagnostic_mode)
    geometry_profile = load_geometry_profile(args.geometry_profile)
    scene_config = load_scene_visual_config(geometry_scene_path(args.geometry_profile))
    geometry_payload = geometry_profile if geometry_profile is not None else None
    geometry_authority = geometry_authority_identity(args.geometry_profile)
    scoreable = bool(geometry_authority["scoreable"]) and state_asset["authority"]["kind"] != "task_variants"
    scene_identity = scene_visual_identity(scene_config)
    if args.horizon_steps <= 0:
        raise OracleRunError("--horizon-steps must be positive")
    target = Path(args.output)
    partial_target = target.with_name(target.name + ".partial.json")
    episodes: list[dict[str, Any]] = []
    if args.resume and partial_target.exists():
        try:
            partial = json.loads(partial_target.read_text(encoding="utf-8"))
            partial_verdict = verify_run_artifact(partial_target)
            if (
                partial_verdict["passed"]
                and partial.get("schema_id") == RUN_SCHEMA_ID
                and partial.get("schema_version") == RUN_SCHEMA_VERSION
                and partial.get("tier") == args.tier
                and float(partial.get("gamma_commanded")) == float(args.gamma)
                and _values_equal(partial.get("controller_profile"), profile.to_dict(), atol=0.0)
                and partial.get("diagnostic_mode") == profile.diagnostic_mode
                and partial.get("evaluator_post_complete_settle_s") == profile.completion_evaluator_settle_s
                and partial.get("scene_visual") == scene_identity
                and partial.get("geometry_authority") == geometry_authority
                and partial.get("scoreable") == scoreable
                and partial.get("geometry_profile") == geometry_payload
                and partial.get("state_authority") == state_asset["authority"]
                and partial.get("physics_authority")
                == {"profile_id": OFFICIAL_PHYSICS_PROFILE_ID, "profile_sha256": OFFICIAL_PHYSICS_PROFILE_SHA256}
                and isinstance(partial.get("episodes"), list)
            ):
                episodes = list(partial["episodes"])
                completed_ids = {row.get("state_id") for row in episodes if isinstance(row, Mapping)}
                states = [state for state in states if state["state_id"] not in completed_ids]
            else:
                raise OracleRunError("partial checkpoint failed semantic or science-identity verification")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raise OracleRunError("partial checkpoint is unreadable or unauthenticated")
    for state in states:
        episodes.append(
            run_episode(
                state,
                tier=args.tier,
                gamma_commanded=args.gamma,
                profile=profile,
                horizon_steps=args.horizon_steps,
                geometry_profile=args.geometry_profile,
            )
        )
        partial_payload = {
            "schema_id": RUN_SCHEMA_ID,
            "schema_version": RUN_SCHEMA_VERSION,
            "tier": args.tier,
            "gamma_commanded": args.gamma,
            "controller_profile": profile.to_dict(),
            "diagnostic_mode": profile.diagnostic_mode,
            "evaluator_post_complete_settle_s": profile.completion_evaluator_settle_s,
            "dev_state_anchor": (
                _dev_state_anchor(args.states) if state_asset["authority"]["kind"] == "dev" else _dev_state_anchor()
            ),
            "state_authority": state_asset["authority"],
            "physics_authority": {
                "profile_id": OFFICIAL_PHYSICS_PROFILE_ID,
                "profile_sha256": OFFICIAL_PHYSICS_PROFILE_SHA256,
            },
            "scene_visual": scene_identity,
            "geometry_profile": geometry_payload,
            "geometry_authority": geometry_authority,
            "scoreable": scoreable,
            "outcome_contract": outcome_contract(),
            "outcome_contract_sha256": outcome_contract_sha256(),
            "episodes": episodes,
        }
        partial_payload["run_id"] = _digest(
            {
                "tier": args.tier,
                "gamma_commanded": args.gamma,
                "controller_profile": profile.to_dict(),
                "scene_visual": scene_identity,
                "geometry_profile": geometry_payload,
                "geometry_authority": geometry_authority,
                "scoreable": scoreable,
                "outcome_contract_sha256": outcome_contract_sha256(),
                "state_ids": [episode["state_id"] for episode in episodes],
                "state_authority": state_asset["authority"],
            }
        )
        partial_payload["payload_sha256"] = _digest(partial_payload)
        write_json_atomic(partial_target, partial_payload)
    payload = {
        "schema_id": RUN_SCHEMA_ID,
        "schema_version": RUN_SCHEMA_VERSION,
        "tier": args.tier,
        "gamma_commanded": args.gamma,
        "controller_profile": profile.to_dict(),
        "diagnostic_mode": profile.diagnostic_mode,
        "evaluator_post_complete_settle_s": profile.completion_evaluator_settle_s,
        "dev_state_anchor": (
            _dev_state_anchor(args.states) if state_asset["authority"]["kind"] == "dev" else _dev_state_anchor()
        ),
        "state_authority": state_asset["authority"],
        "physics_authority": {
            "profile_id": OFFICIAL_PHYSICS_PROFILE_ID,
            "profile_sha256": OFFICIAL_PHYSICS_PROFILE_SHA256,
        },
        "scene_visual": scene_identity,
        "geometry_profile": geometry_payload,
        "geometry_authority": geometry_authority,
        "scoreable": scoreable,
        "outcome_contract": outcome_contract(),
        "outcome_contract_sha256": outcome_contract_sha256(),
        "episodes": episodes,
    }
    payload["run_id"] = _digest(
        {
            "tier": args.tier,
            "gamma_commanded": args.gamma,
            "controller_profile": profile.to_dict(),
            "scene_visual": payload["scene_visual"],
            "geometry_profile": geometry_payload,
            "geometry_authority": geometry_authority,
            "scoreable": scoreable,
            "outcome_contract_sha256": outcome_contract_sha256(),
            "state_ids": [episode["state_id"] for episode in episodes],
            "state_authority": payload["state_authority"],
        }
    )
    if args.merge_into is not None:
        prior = json.loads(Path(args.merge_into).read_text(encoding="utf-8"))
        if (
            prior.get("schema_id") != payload["schema_id"]
            or prior.get("tier") != payload["tier"]
            or float(prior.get("gamma_commanded")) != float(payload["gamma_commanded"])
            or prior.get("controller_profile") != payload["controller_profile"]
            or prior.get("scene_visual") != payload["scene_visual"]
            or prior.get("geometry_profile") != payload["geometry_profile"]
            or prior.get("geometry_authority") != payload["geometry_authority"]
            or prior.get("scoreable") != payload["scoreable"]
        ):
            raise OracleRunError("--merge-into raw run is not compatible with this controller/tier/Gamma")
        replacements = {episode["state_id"]: episode for episode in episodes}
        prior_episodes = prior.get("episodes")
        if not isinstance(prior_episodes, list) or not all(
            item.get("state_id") in replacements or "state_id" in item for item in prior_episodes
        ):
            raise OracleRunError("--merge-into has invalid episode records")
        payload["episodes"] = [replacements.get(item["state_id"], item) for item in prior_episodes]
    if args.process_index is not None or args.parent_run_uuid is not None:
        if args.process_index is None or not isinstance(args.parent_run_uuid, str) or not args.parent_run_uuid:
            raise OracleRunError("determinism child requires process index and parent run UUID")
        payload["process"] = {
            "process_index": int(args.process_index),
            "process_pid": os.getpid(),
            "process_identity": f"pid:{os.getpid()}",
            "parent_run_uuid": args.parent_run_uuid,
            "start_timestamp_s": time.time(),
        }
    payload["payload_sha256"] = _digest(payload)
    write_json_atomic(target, payload)
    if partial_target.exists():
        partial_target.unlink()
    print(
        json.dumps(
            {
                "output": str(target),
                "payload_sha256": payload["payload_sha256"],
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
        return {"passed": False, "errors": [f"artifact read: {exc}"], "run_id": None}
    if not isinstance(payload, Mapping):
        return {"passed": False, "errors": ["run payload must be an object"], "run_id": None}
    # Historical v5 evidence is intentionally verified as v5, never promoted
    # to the outcome contract or mixed into a v6 score group.
    if payload.get("schema_id") == RUN_SCHEMA_ID and payload.get("schema_version") == LEGACY_RUN_SCHEMA_VERSION:
        return verify_legacy_run_artifact(payload)
    if payload.get("schema_id") != RUN_SCHEMA_ID or payload.get("schema_version") != RUN_SCHEMA_VERSION:
        errors.append("run schema")
    if payload.get("outcome_contract") != outcome_contract() or payload.get("outcome_contract_sha256") != outcome_contract_sha256():
        errors.append("outcome contract authority")
    required_top_level = {
        "schema_id",
        "schema_version",
        "tier",
        "gamma_commanded",
        "controller_profile",
        "diagnostic_mode",
        "evaluator_post_complete_settle_s",
        "dev_state_anchor",
        "physics_authority",
        "scene_visual",
        "geometry_profile",
        "geometry_authority",
        "scoreable",
        "outcome_contract",
        "outcome_contract_sha256",
        "episodes",
        "run_id",
        "payload_sha256",
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
        from robosuite.utils.shakebench_committed_states import KNEE_STATE_FILENAME, OFFICIAL_STATE_FILENAME

        if committed_split not in {"official", "knee"}:
            errors.append("committed-state split")
            states_by_id = {}
        else:
            asset = Path(models.assets_root) / (
                OFFICIAL_STATE_FILENAME if committed_split == "official" else KNEE_STATE_FILENAME
            )
            if state_authority.get("kind") == "task_variants":
                from robosuite.utils.shakebench_task_states import TASK_STATE_FILENAMES

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
        {"profile_id": OFFICIAL_PHYSICS_PROFILE_ID, "profile_sha256": OFFICIAL_PHYSICS_PROFILE_SHA256},
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
    if payload.get("diagnostic_mode", expected_profile.get("diagnostic_mode")) != expected_profile.get(
        "diagnostic_mode"
    ):
        errors.append("diagnostic mode")
    copied = dict(payload)
    expected_payload_hash = copied.pop("payload_sha256", None)
    try:
        if expected_payload_hash != _digest(copied):
            errors.append("payload digest")
    except (TypeError, ValueError):
        errors.append("payload digest")
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
            "task_context_sha256",
            "controller_context_hash",
            "physics_profile",
            "scene_visual",
            "geometry_profile",
            "geometry_authority",
            "scoreable",
            "outcome_contract",
            "outcome_contract_sha256",
            "episode_validity",
            "score_outcome",
            "termination_cause",
            "controller_events",
            "state_sha256",
            "success",
            "failure_reason",
            "termination_category",
            "actuators",
            "metrics",
            "trace",
            "trace_sha256",
        }
        if isinstance(state_authority, Mapping) and state_authority.get("kind") == "task_variants":
            required_episode_fields.add("task_contract")
        if set(episode) != required_episode_fields:
            errors.append(prefix + " required fields")
        if (
            episode.get("outcome_contract") != outcome_contract()
            or episode.get("outcome_contract_sha256") != outcome_contract_sha256()
        ):
            errors.append(prefix + " outcome contract authority")
        tier = episode.get("tier")
        if tier != payload.get("tier") or tier not in TIER_POLICY_KEYS:
            errors.append(prefix + " tier")
            tier = "V0" if tier not in TIER_POLICY_KEYS else tier
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
                from robosuite.utils.shakebench_tasks import TaskSpec, OBJECT_SUPPORT

                spec = TaskSpec.from_mapping(state["task"])
                if episode.get("task_contract") != spec.contract():
                    errors.append(prefix + " task contract binding")
                context = episode.get("task_context", {})
                lower, upper, radius = OBJECT_SUPPORT[spec.object_id]
                for key, value in (("can_collision_lower_support_m", lower),
                                   ("can_collision_upper_support_m", upper), ("can_collision_radius_m", radius)):
                    if not _values_equal(context.get(key), value, atol=1e-10):
                        errors.append(prefix + " object geometry binding")
            if episode.get("state_sha256") != _digest(state):
                errors.append(prefix + " state hash")
            if committed_split is not None:
                bindings = state.get("authority_hashes") if isinstance(state, Mapping) else None
                if not isinstance(bindings, Mapping):
                    errors.append(prefix + " committed outcome authority")
                else:
                    if bindings.get("controller_profile_sha256") != expected_profile["profile_sha256"]:
                        errors.append(prefix + " committed controller authority")
                    if bindings.get("outcome_contract_sha256") != outcome_contract_sha256():
                        errors.append(prefix + " committed outcome authority")
        if not _values_equal(episode.get("controller_profile"), expected_profile, atol=1.0e-12):
            errors.append(prefix + " controller profile")
        try:
            expected_context = WorktableTaskContext.from_mapping(episode.get("task_context"))
            if (
                episode.get("controller_context_hash")
                != hashlib.sha256(
                    f"{expected_profile['profile_sha256']}:{expected_context.sha256}".encode("ascii")
                ).hexdigest()
            ):
                errors.append(prefix + " controller context hash")
        except (TypeError, ValueError, ShakeBenchOracleError):
            errors.append(prefix + " controller context hash")
        physics = episode.get("physics_profile")
        if not _values_equal(
            physics,
            {"profile_id": OFFICIAL_PHYSICS_PROFILE_ID, "profile_sha256": OFFICIAL_PHYSICS_PROFILE_SHA256},
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
        try:
            if episode.get("trace_sha256") != _digest(trace):
                errors.append(prefix + " trace digest")
        except (TypeError, ValueError):
            errors.append(prefix + " trace digest")
        try:
            task_context = WorktableTaskContext.from_mapping(episode.get("task_context"))
            if episode.get("task_context_sha256", task_context.sha256) != task_context.sha256:
                errors.append(prefix + " task context hash")
        except (TypeError, ValueError, ShakeBenchOracleError) as exc:
            errors.append(prefix + f" task context: {exc}")
            task_context = WorktableTaskContext()
        controller = ShakeBenchOracleController(tier, parsed_profile, task_context=task_context)
        expected_keys = set(COMMON_STATE_KEYS) | set(TIER_POLICY_KEYS[tier])
        contract = observation_contract_for_tier(tier)
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
                "task_state_sha256",
                "policy_input",
                "provider_payload",
                "post_task_state",
                "post_task_state_sha256",
                "estimate",
                "control_law",
                "relative_kinematics",
                "task_desired_action",
                "pre_capability_compensated_action",
                "phase_capability",
                "capability_limits",
                "post_capability_normalized_action",
                "normalized_action",
                "decoded_action",
                "clipped_action",
                "applied_actuator_ctrl",
                "applied_actuator_ctrl_sha256",
                "applied_actuator_force",
                "applied_actuator_force_sha256",
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
            if not isinstance(provider_payload, Mapping) or set(provider_payload) != set(TIER_POLICY_KEYS[tier]):
                errors.append(row_prefix + " provider payload key set")
            elif any(not _values_equal(provider_payload[key], policy_input[key], atol=0.0) for key in provider_payload):
                errors.append(row_prefix + " provider payload mismatch")
            task_state = row.get("task_state")
            if not isinstance(task_state, Mapping) or not _values_equal(
                {key: policy_input.get(key) for key in COMMON_STATE_KEYS}, task_state, atol=0.0
            ):
                errors.append(row_prefix + " task state mismatch")
            if row.get("task_state_sha256") != _digest(task_state):
                errors.append(row_prefix + " task state digest")
            post_task_state = row.get("post_task_state")
            if row.get("post_task_state_sha256") != _digest(post_task_state):
                errors.append(row_prefix + " post task state digest")
            try:
                recomputed_action = controller.action(policy_input, time_s=policy_time)
                recomputed_trace = controller.last_trace
                if not _values_equal(row["estimate"], recomputed_trace["estimate"], atol=1.0e-6):
                    errors.append(row_prefix + " estimate recomputation")
                if not _values_equal(row["control_law"], recomputed_trace["control_law"], atol=1.0e-6):
                    errors.append(row_prefix + " control-law recomputation")
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
                    "pre_capability_compensated_action",
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
                if not isinstance(state_payload, Mapping) or set(state_payload) != set(COMMON_STATE_KEYS):
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
                elif row.get(key + "_sha256") != _digest(array):
                    errors.append(row_prefix + f" {key} digest")
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
                prediction_times = estimate_payload.get("prediction_timestamps_s")
                try:
                    if not prediction_times or any(
                        float(value) <= float(row["policy_time_s"]) for value in prediction_times
                    ):
                        errors.append(row_prefix + " prediction causality")
                except (TypeError, ValueError):
                    errors.append(row_prefix + " prediction causality")
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
    run_id_basis = {
        "tier": payload.get("tier"),
        "gamma_commanded": payload.get("gamma_commanded"),
        "controller_profile": expected_profile,
        "state_ids": expected_ids,
        "geometry_authority": payload.get("geometry_authority"),
        "scoreable": payload.get("scoreable"),
        "outcome_contract_sha256": payload.get("outcome_contract_sha256"),
    }
    run_id_basis["scene_visual"] = payload.get("scene_visual")
    run_id_basis["geometry_profile"] = payload.get("geometry_profile")
    if state_authority is not None:
        run_id_basis["state_authority"] = state_authority
    if payload.get("run_id") != _digest(run_id_basis):
        errors.append("run id")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "run_id": payload.get("run_id"),
        "trace_count": sum(len(row.get("trace", ())) for row in episodes if isinstance(row, Mapping)),
        "dev_state_anchor_mode": anchor_mode,
    }


def verify_legacy_run_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only verifier for pre-08R artifacts; it never reinterprets them."""

    errors: list[str] = []
    if payload.get("schema_id") != RUN_SCHEMA_ID or payload.get("schema_version") != LEGACY_RUN_SCHEMA_VERSION:
        errors.append("legacy run schema")
    if "outcome_contract" in payload or "outcome_contract_sha256" in payload:
        errors.append("legacy/new schema mixture")
    copied = dict(payload)
    expected_hash = copied.pop("payload_sha256", None)
    try:
        if expected_hash != _digest(copied):
            errors.append("legacy payload digest")
    except (TypeError, ValueError):
        errors.append("legacy payload digest")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("legacy episodes")
        episodes = []
    for index, episode in enumerate(episodes):
        prefix = f"episode[{index}]"
        if not isinstance(episode, Mapping) or episode.get("schema_version") != LEGACY_EPISODE_SCHEMA_VERSION:
            errors.append(prefix + " legacy schema")
            continue
        if any(key in episode for key in ("episode_validity", "score_outcome", "termination_cause", "controller_events")):
            errors.append(prefix + " legacy/new schema mixture")
        trace = episode.get("trace")
        try:
            if not isinstance(trace, list) or episode.get("trace_sha256") != _digest(trace):
                errors.append(prefix + " legacy trace digest")
        except (TypeError, ValueError):
            errors.append(prefix + " legacy trace digest")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "run_id": payload.get("run_id"),
        "legacy_schema": True,
    }


def _manifest_trace_projection(payload: Mapping[str, Any]) -> list[Any]:
    episodes = payload.get("episodes", [])
    return [
        {
            "state_id": episode.get("state_id"),
            "tier": episode.get("tier"),
            "gamma_commanded": episode.get("gamma_commanded"),
            "success": episode.get("success"),
            "episode_validity": episode.get("episode_validity"),
            "score_outcome": episode.get("score_outcome"),
            "termination_cause": episode.get("termination_cause"),
            "controller_events": episode.get("controller_events"),
            "failure_reason": episode.get("failure_reason"),
            "termination_category": episode.get("termination_category"),
            "scene_visual": episode.get("scene_visual"),
            "geometry_profile": episode.get("geometry_profile"),
            "geometry_authority": episode.get("geometry_authority"),
            "scoreable": episode.get("scoreable"),
            "task_context": episode.get("task_context"),
            "task_context_sha256": episode.get("task_context_sha256"),
            "actuators": episode.get("actuators"),
            "metrics": episode.get("metrics"),
            "trace": episode.get("trace", []),
        }
        for episode in episodes
    ]


def _compare_replay_traces(first: Any, other: Any, *, atol: float = 1.0e-6) -> bool:
    """Compare traces with tolerance only for pre-registered numeric fields."""

    if isinstance(first, Mapping) or isinstance(other, Mapping):
        if not isinstance(first, Mapping) or not isinstance(other, Mapping) or set(first) != set(other):
            return False
        return all(_compare_replay_traces(first[key], other[key], atol=atol) for key in first)
    if isinstance(first, (list, tuple)) or isinstance(other, (list, tuple)):
        if not isinstance(first, (list, tuple)) or not isinstance(other, (list, tuple)) or len(first) != len(other):
            return False
        return all(_compare_replay_traces(a, b, atol=atol) for a, b in zip(first, other))
    if isinstance(first, (float, int, np.floating, np.integer)) and isinstance(
        other, (float, int, np.floating, np.integer)
    ):
        return bool(
            np.isfinite(float(first))
            and np.isfinite(float(other))
            and np.isclose(float(first), float(other), rtol=0.0, atol=atol)
        )
    return first == other


def verify_determinism_manifest(path: str | Path) -> dict[str, Any]:
    """Independently verify three fresh-process raw runs and their full traces."""

    errors: list[str] = []
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"manifest read: {exc}"]}
    if (
        manifest.get("schema_id") != DETERMINISM_SCHEMA_ID
        or manifest.get("schema_version") != DETERMINISM_SCHEMA_VERSION
    ):
        errors.append("manifest schema")
    copied = dict(manifest)
    expected_manifest_hash = copied.pop("manifest_sha256", None)
    try:
        if expected_manifest_hash != _digest(copied):
            errors.append("manifest hash")
    except (TypeError, ValueError):
        errors.append("manifest hash")
    records = manifest.get("records")
    if manifest.get("process_count") != 3 or not isinstance(records, list) or len(records) != 3:
        errors.append("process count")
        records = records if isinstance(records, list) else []
    paths: list[str] = []
    pids: list[int] = []
    process_indexes: list[int] = []
    payloads: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(record, Mapping):
            errors.append(prefix + " object")
            continue
        record_path = record.get("path")
        if not isinstance(record_path, str) or not record_path:
            errors.append(prefix + " path")
            continue
        paths.append(record_path)
        resolved = Path(record_path)
        if not resolved.exists():
            errors.append(prefix + " missing file")
            continue
        try:
            if record.get("file_sha256") != hashlib.sha256(resolved.read_bytes()).hexdigest():
                errors.append(prefix + " file hash")
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(prefix + f" file read: {exc}")
            continue
        payloads.append(payload)
        semantic = verify_run_artifact(resolved)
        if not semantic["passed"]:
            errors.extend(prefix + " semantic: " + error for error in semantic["errors"])
        if record.get("payload_sha256") != payload.get("payload_sha256"):
            errors.append(prefix + " payload hash")
        if record.get("trace_sha256") != _digest(_manifest_trace_projection(payload)):
            errors.append(prefix + " trace hash")
        if record.get("complete") is not True:
            errors.append(prefix + " incomplete")
        process = payload.get("process")
        if not isinstance(process, Mapping):
            errors.append(prefix + " process metadata")
            continue
        pid = process.get("process_pid")
        if not isinstance(pid, int) or pid <= 0 or process.get("process_identity") != f"pid:{pid}":
            errors.append(prefix + " process identity")
        else:
            pids.append(pid)
        process_index = process.get("process_index")
        if not isinstance(process_index, int):
            errors.append(prefix + " process index")
        else:
            process_indexes.append(process_index)
        if process.get("parent_run_uuid") != manifest.get("parent_run_uuid"):
            errors.append(prefix + " parent run binding")
    if len(paths) != len(set(paths)):
        errors.append("duplicate path")
    if len(pids) != len(set(pids)):
        errors.append("duplicate process identity")
    if sorted(process_indexes) != [0, 1, 2]:
        errors.append("process indexes")
    if len(payloads) == 3:
        first = payloads[0]
        for index, payload in enumerate(payloads[1:], start=1):
            for key in (
                "tier",
                "gamma_commanded",
                "controller_profile",
                "physics_authority",
                "dev_state_anchor",
                "scene_visual",
                "geometry_profile",
                "geometry_authority",
                "scoreable",
                "outcome_contract",
                "outcome_contract_sha256",
            ):
                if not _values_equal(payload.get(key), first.get(key), atol=0.0):
                    errors.append(f"binding mismatch {key} record[{index}]")
            if not _compare_replay_traces(_manifest_trace_projection(first), _manifest_trace_projection(payload)):
                errors.append(f"trace mismatch record[{index}]")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "manifest_path": str(manifest_path),
        "process_count": len(records),
    }


def run_determinism_replay(
    manifest_path: str | Path,
    *,
    state_id: str = "shakebench-dev-v0-000",
    state_ids: tuple[str, ...] | None = None,
    states_path: str | Path | None = None,
    horizon_steps: int = 1200,
    geometry_profile: str = "canonical",
) -> dict[str, Any]:
    """Launch three fresh Python children and write a compact replay manifest."""

    manifest_target = Path(manifest_path)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    requested_state_ids = state_ids or (state_id,)
    if not requested_state_ids or len(set(requested_state_ids)) != len(requested_state_ids):
        raise OracleRunError("determinism replay requires unique state IDs")
    parent_run_uuid = str(uuid.uuid4())
    records = []
    for process_index in range(3):
        child_path = manifest_target.parent / f"replay_process_{process_index + 1}.json"
        command = [
            sys.executable,
            "-m",
            "robosuite.scripts.shakebench_run_oracle",
            "--tier",
            "V0",
            "--gamma",
            "0.0",
            "--output",
            str(child_path),
            "--state-ids",
            ",".join(requested_state_ids),
            "--horizon-steps",
            str(horizon_steps),
            "--geometry-profile",
            geometry_profile,
            "--process-index",
            str(process_index),
            "--parent-run-uuid",
            parent_run_uuid,
        ]
        if states_path is not None:
            command.extend(("--states", str(states_path)))
        started = time.time()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_lines = completed.stdout.splitlines()
        stderr_lines = completed.stderr.splitlines()
        record = {
            "process_index": process_index,
            "exit_code": completed.returncode,
            "command": command,
            "stdout_summary": stdout_lines[-5:],
            "stderr_summary": stderr_lines[-5:],
            "path": str(child_path),
            "complete": completed.returncode == 0 and child_path.exists(),
            "start_timestamp_s": started,
        }
        if child_path.exists():
            payload = json.loads(child_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "file_sha256": hashlib.sha256(child_path.read_bytes()).hexdigest(),
                    "payload_sha256": payload.get("payload_sha256"),
                    "trace_sha256": _digest(_manifest_trace_projection(payload)),
                    "process_pid": payload.get("process", {}).get("process_pid"),
                    "process_identity": payload.get("process", {}).get("process_identity"),
                }
            )
        records.append(record)
    manifest = {
        "schema_id": DETERMINISM_SCHEMA_ID,
        "schema_version": DETERMINISM_SCHEMA_VERSION,
        "parent_run_uuid": parent_run_uuid,
        "state_id": requested_state_ids[0],
        "state_ids": list(requested_state_ids),
        "tier": "V0",
        "gamma_commanded": 0.0,
        "geometry_profile": load_geometry_profile(geometry_profile),
        "process_count": 3,
        "records": records,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    manifest_target.write_text(json.dumps(_json_ready(manifest), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verdict = verify_determinism_manifest(manifest_target)
    verdict["manifest_path"] = str(manifest_target)
    verdict["exit_code"] = 0 if verdict["passed"] else 1
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
