"""Phase 08 committed-state authority.

This module deliberately has no environment, MuJoCo, renderer, or rollout
dependency.  It deterministically materializes the two Phase 09 inputs and
validates them fail-closed before a runner is allowed to consume them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robosuite import models
from robosuite.utils.shakebench_artifacts import payload_hash, write_json_atomic
from robosuite.utils.shakebench_dev_states import PHASE07_DEV_STATE_FILENAME, verify_phase07_dev_state_artifact
from robosuite.utils.shakebench_oracle import OracleControllerProfile
from robosuite.utils.shakebench_outcomes import outcome_contract_sha256


PHASE08_STATE_SCHEMA_ID = "shakebench.phase08.committed_states"
PHASE08_STATE_SCHEMA_VERSION = 1
PHASE08_STATE_GENERATOR_ID = "shakebench.sha256_uniform.v1"
PHASE08_STATE_ROOT_SEED = 20260908
OFFICIAL_STATE_COUNT = 400
KNEE_STATE_COUNT = 100
OFFICIAL_STATE_FILENAME = "shakebench_states_official.json"
KNEE_STATE_FILENAME = "shakebench_states_knee.json"
PHASE07_5A_EVIDENCE_BINDING_FILENAME = "shakebench_phase07_5a_evidence_binding.json"
EXPECTED_EVIDENCE_BINDING_PAYLOAD_SHA256 = "4591ec046115ff8cfe17933f0278c3a434a12e20d4f59dbbe84457f1eab55d70"
_NOMINAL_XY_M = (-0.10, -0.13)
_XY_HALF_RANGE_M = 0.02
_FORBIDDEN_STATE_KEYS = frozenset(
    ("action", "actions", "contact", "contacts", "outcome", "success", "failure", "wall_clock", "wall_time")
)


class Phase08StateError(ValueError):
    """Raised when a committed-state authority is malformed or unauthorized."""


def verify_phase07_5a_evidence_binding(path: str | Path | None = None) -> dict[str, Any]:
    """Verify the package-owned Phase 7.5A evidence-core binding.

    Args:
        path: Optional binding asset override used by contract tests.

    Returns:
        Authenticated binding payload.

    Raises:
        Phase08StateError: If the asset schema or digest is not frozen.
    """

    source = Path(models.assets_root) / PHASE07_5A_EVIDENCE_BINDING_FILENAME if path is None else Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase08StateError(f"BLOCKED_BY_PHASE07_5A_HANDOFF: evidence binding read: {exc}") from exc
    required = {
        "schema_id",
        "schema_version",
        "evidence_core_file_sha256",
        "evidence_core_payload_sha256",
        "requalification_manifest_payload_sha256",
        "payload_sha256",
    }
    if not isinstance(payload, Mapping):
        raise Phase08StateError("BLOCKED_BY_PHASE07_5A_HANDOFF: evidence binding must be an object")
    actual_hash = payload_hash(payload)
    if (
        set(payload) != required
        or payload.get("schema_id") != "shakebench.phase07_5a.evidence_binding"
        or payload.get("schema_version") != 1
        or payload.get("payload_sha256") != actual_hash
        or actual_hash != EXPECTED_EVIDENCE_BINDING_PAYLOAD_SHA256
    ):
        raise Phase08StateError("BLOCKED_BY_PHASE07_5A_HANDOFF: evidence binding authentication failed")
    return dict(payload)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _word(split: str, index: int, channel: str, *, size: int = 8) -> int:
    token = f"{PHASE08_STATE_GENERATOR_ID}:{PHASE08_STATE_ROOT_SEED}:{split}:{index}:{channel}"
    return int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:size], "big")


def _uniform(split: str, index: int, channel: str) -> float:
    return _word(split, index, channel) / float(1 << 64)


def _authority_bindings() -> dict[str, str]:
    """Obtain all state-side hashes from the frozen, authenticated authority.

    The inventory-bearing handoff verifier is intentionally run once before
    Phase 08 creates new source files.  Re-running it from this writer would
    make the writer's own additions invalidate that prior handoff.  The
    immutable science and final envelopes remain independently verified here.
    """

    from robosuite.utils.shakebench_authority import (
        authority_payload_hash,
        verify_direct_mount_authority,
        verify_final_phase08_authority,
    )

    authority = verify_direct_mount_authority()
    try:
        # The expected evidence-core digest comes from a separately frozen,
        # package-owned binding, never from the final-authority envelope being
        # verified.
        evidence_hash = verify_phase07_5a_evidence_binding()["evidence_core_payload_sha256"]
        verify_final_phase08_authority(evidence_hash)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise Phase08StateError(f"BLOCKED_BY_PHASE07_5A_HANDOFF: {exc}") from exc
    authority_sha256 = authority_payload_hash(authority)
    runtime_path = Path(models.assets_root) / "shakebench_runtime_contract.json"
    if not runtime_path.is_file():
        raise Phase08StateError("runtime contract asset missing")
    dev_payload = verify_phase07_dev_state_artifact(Path(models.assets_root) / PHASE07_DEV_STATE_FILENAME)["payload"]
    return {
        "science_authority_sha256": authority_sha256,
        "physics_profile_sha256": str(authority["official_physics_profile_sha256"]),
        # Phase 7.5A scene evidence remains a historical binding.  Phase 08R
        # freezes the current controller and outcome contract independently
        # into the committed state authority used by future measurements.
        "controller_profile_sha256": OracleControllerProfile().sha256,
        "outcome_contract_sha256": outcome_contract_sha256(),
        "geometry_profile_sha256": str(authority["geometry_payload_sha256"]),
        "scene_visual_sha256": str(authority["scene_sha256"]),
        "runtime_contract_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        # This is derived from the repository's sole canonical task-contract
        # representation, rather than copied from a report or cached string.
        "task_contract_sha256": payload_hash(dev_payload["task_contract"]),
    }


def _dev_anchor() -> dict[str, str]:
    path = Path(models.assets_root) / PHASE07_DEV_STATE_FILENAME
    verified = verify_phase07_dev_state_artifact(path)
    if not verified["passed"]:
        raise Phase08StateError("frozen dev-state anchor failed")
    return {
        "asset": PHASE07_DEV_STATE_FILENAME,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload_sha256": str(verified["payload"]["artifact_lock"]["payload_sha256"]),
    }


def _state(split: str, index: int, bindings: Mapping[str, str]) -> dict[str, Any]:
    x = _NOMINAL_XY_M[0] + (2.0 * _uniform(split, index, "can_x") - 1.0) * _XY_HALF_RANGE_M
    y = _NOMINAL_XY_M[1] + (2.0 * _uniform(split, index, "can_y") - 1.0) * _XY_HALF_RANGE_M
    state = {
        "schema_id": PHASE08_STATE_SCHEMA_ID + ".record",
        "schema_version": PHASE08_STATE_SCHEMA_VERSION,
        "state_id": f"shakebench-{split}-v1-{index:03d}",
        "split": split,
        "can_pose_worktable": [x, y, 0.07, 1.0, 0.0, 0.0, 0.0],
        "can_xy_m": [x, y],
        "can_yaw_rad": 0.0,
        "can_initial_velocity": [0.0] * 6,
        "target_reference": {
            "frame": "worktable",
            "frame_origin_m": [-0.10, 0.17, 0.042],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "excitation_seed": _word(split, index, "excitation_seed", size=4),
        "imu_seed": _word(split, index, "imu_seed", size=4),
        "t0_s": _uniform(split, index, "t0_s"),
        "Gamma": {"commanded": None, "selection": "unselected; Phase 09 knee calibration only"},
        "authority_hashes": dict(bindings),
    }
    state["canonical_payload_sha256"] = payload_hash(state, field="canonical_payload_sha256")
    return state


def build_committed_state_artifact(split: str) -> dict[str, Any]:
    """Build one authenticated, rollout-independent committed-state artifact.

    Args:
        split: Either ``"official"`` or ``"knee"``.

    Returns:
        Deterministically generated frozen state artifact.

    Raises:
        Phase08StateError: If the split or Phase 7.5A authority is invalid.
    """

    if split not in {"official", "knee"}:
        raise Phase08StateError("split must be official or knee")
    bindings = _authority_bindings()
    count = OFFICIAL_STATE_COUNT if split == "official" else KNEE_STATE_COUNT
    artifact = {
        "schema_id": PHASE08_STATE_SCHEMA_ID,
        "schema_version": PHASE08_STATE_SCHEMA_VERSION,
        "status": "frozen",
        "scoreable": True,
        "split": split,
        "generator": {
            "generator_id": PHASE08_STATE_GENERATOR_ID,
            "root_seed": PHASE08_STATE_ROOT_SEED,
            "mapping": "u64_be(sha256(generator_id:root_seed:split:index:channel)[:8]) / 2**64",
            "selection": "none; deterministic commitment before Phase 09 measurement",
        },
        "authority_hashes": bindings,
        "dev_state_anchor": _dev_anchor(),
        "states": [_state(split, index, bindings) for index in range(count)],
        "artifact_lock": {"payload_sha256": ""},
    }
    artifact["artifact_lock"]["payload_sha256"] = payload_hash(
        {"artifact": {key: value for key, value in artifact.items() if key != "artifact_lock"}}
    )
    return artifact


def _artifact_lock_hash(artifact: Mapping[str, Any]) -> str:
    copied = copy.deepcopy(dict(artifact))
    lock = copied.pop("artifact_lock", None)
    if not isinstance(lock, Mapping):
        return ""
    return payload_hash({"artifact": copied})


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _forbidden_key_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(key in _FORBIDDEN_STATE_KEYS or _forbidden_key_present(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_forbidden_key_present(item) for item in value)
    return False


def _science_payload_hash(state: Mapping[str, Any]) -> str:
    """Compare placement/science payloads independently of split and ID."""

    fields = {
        "can_pose_worktable",
        "can_xy_m",
        "can_yaw_rad",
        "can_initial_velocity",
        "target_reference",
        "excitation_seed",
        "imu_seed",
        "t0_s",
    }
    return payload_hash({key: state.get(key) for key in sorted(fields)})


def verify_committed_state_artifact(
    payload_or_path: Mapping[str, Any] | str | Path, *, expected_split: str | None = None
) -> dict[str, Any]:
    """Fail closed on schema, exact deterministic regeneration, or overlap.

    Args:
        payload_or_path: Committed-state object or JSON artifact path.
        expected_split: Optional required committed split.

    Returns:
        Structured pass/fail verdict.
    """

    try:
        payload = (
            copy.deepcopy(dict(payload_or_path))
            if isinstance(payload_or_path, Mapping)
            else json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"artifact read: {exc}"]}
    split = payload.get("split") if isinstance(payload, Mapping) else None
    errors: list[str] = []
    if expected_split is not None and split != expected_split:
        errors.append("unexpected split")
    if split not in {"official", "knee"}:
        errors.append("split")
        split = "official"
    try:
        expected = build_committed_state_artifact(split)
    except Phase08StateError as exc:
        return {"passed": False, "errors": [str(exc)]}
    if (
        payload.get("schema_id") != PHASE08_STATE_SCHEMA_ID
        or payload.get("schema_version") != PHASE08_STATE_SCHEMA_VERSION
    ):
        errors.append("schema")
    if payload.get("status") != "frozen" or payload.get("scoreable") is not True:
        errors.append("frozen status")
    if not _finite(payload):
        errors.append("non-finite value")
    if _forbidden_key_present(payload):
        errors.append("forbidden future-outcome field")
    if payload.get("artifact_lock", {}).get("payload_sha256") != _artifact_lock_hash(payload):
        errors.append("artifact lock")
    if payload != expected:
        errors.append("deterministic regeneration or authority binding")
    dev = verify_phase07_dev_state_artifact(Path(models.assets_root) / PHASE07_DEV_STATE_FILENAME)
    dev_states = dev.get("payload", {}).get("states", [])
    dev_ids = {state["state_id"] for state in dev_states}
    state_ids = [state.get("state_id") for state in payload.get("states", []) if isinstance(state, Mapping)]
    if len(state_ids) != len(set(state_ids)) or dev_ids.intersection(state_ids):
        errors.append("state ID overlap")
    science_hashes = [_science_payload_hash(state) for state in payload.get("states", []) if isinstance(state, Mapping)]
    dev_science_hashes = {_science_payload_hash(state) for state in dev_states if isinstance(state, Mapping)}
    if len(science_hashes) != len(set(science_hashes)) or dev_science_hashes.intersection(science_hashes):
        errors.append("canonical science-payload overlap")
    return {"passed": not errors, "errors": sorted(set(errors)), "split": split, "state_count": len(state_ids)}


def verify_committed_state_pair(
    official: Mapping[str, Any] | str | Path, knee: Mapping[str, Any] | str | Path
) -> dict[str, Any]:
    """Verify official and knee assets together.

    Args:
        official: Official payload or artifact path.
        knee: Knee-calibration payload or artifact path.

    Returns:
        Combined pass/fail verdict.
    """
    first = verify_committed_state_artifact(official, expected_split="official")
    second = verify_committed_state_artifact(knee, expected_split="knee")
    errors = list(first["errors"]) + list(second["errors"])
    if first["passed"] and second["passed"]:
        official_payload = json.loads(Path(official).read_text()) if not isinstance(official, Mapping) else official
        knee_payload = json.loads(Path(knee).read_text()) if not isinstance(knee, Mapping) else knee
        if {item["state_id"] for item in official_payload["states"]}.intersection(
            item["state_id"] for item in knee_payload["states"]
        ):
            errors.append("official/knee ID overlap")
        official_science = {_science_payload_hash(item) for item in official_payload["states"]}
        knee_science = {_science_payload_hash(item) for item in knee_payload["states"]}
        if official_science.intersection(knee_science):
            errors.append("official/knee canonical science-payload overlap")
    return {"passed": not errors, "errors": sorted(set(errors)), "official": first, "knee": second}


def freeze_committed_state_assets(directory: str | Path | None = None) -> dict[str, Path]:
    """Atomically write Phase 08 assets without executing a rollout.

    Args:
        directory: Optional destination; defaults to package assets.

    Returns:
        Mapping containing official and knee paths.

    Raises:
        Phase08StateError: If post-write validation fails.
    """

    destination = Path(models.assets_root) if directory is None else Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    official = build_committed_state_artifact("official")
    knee = build_committed_state_artifact("knee")
    official_path = destination / OFFICIAL_STATE_FILENAME
    knee_path = destination / KNEE_STATE_FILENAME
    write_json_atomic(official_path, official)
    write_json_atomic(knee_path, knee)
    verdict = verify_committed_state_pair(official_path, knee_path)
    if not verdict["passed"]:
        raise Phase08StateError("post-write verification failed: " + ", ".join(verdict["errors"]))
    return {"official": official_path, "knee": knee_path}


__all__ = [
    "EXPECTED_EVIDENCE_BINDING_PAYLOAD_SHA256",
    "KNEE_STATE_COUNT",
    "KNEE_STATE_FILENAME",
    "OFFICIAL_STATE_COUNT",
    "OFFICIAL_STATE_FILENAME",
    "PHASE07_5A_EVIDENCE_BINDING_FILENAME",
    "PHASE08_STATE_SCHEMA_ID",
    "PHASE08_STATE_SCHEMA_VERSION",
    "Phase08StateError",
    "build_committed_state_artifact",
    "freeze_committed_state_assets",
    "verify_committed_state_artifact",
    "verify_committed_state_pair",
    "verify_phase07_5a_evidence_binding",
]
