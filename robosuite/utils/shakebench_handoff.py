"""Fail-closed verifier for the Phase 7.5A final evidence handoff.

The evidence manifest only binds the immutable science authority. The
package-owned final authority binds this manifest's canonical hash. Keeping
those arrows one-way avoids a hash fixed-point while preserving both bindings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robosuite.scripts.shakebench_run_oracle import verify_determinism_manifest, verify_run_artifact
from robosuite.utils.shakebench_authority import (
    DirectMountAuthorityError,
    authority_payload_hash,
    verify_direct_mount_authority,
    verify_final_phase08_authority,
)


def _digest(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent


def _read_object(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _verify_inventory(root: Path, record: Any, errors: list[str]) -> None:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "payload_sha256"}:
        errors.append("source inventory schema")
        return
    relative = record["path"]
    if not isinstance(relative, str) or Path(relative).is_absolute():
        errors.append("source inventory path")
        return
    target = root / relative
    payload = _read_object(target)
    if payload is None:
        errors.append("source inventory missing")
        return
    if record["sha256"] != hashlib.sha256(target.read_bytes()).hexdigest():
        errors.append("source inventory file hash")
    if payload.get("schema_id") != "shakebench.phase07_5a.source_inventory" or payload.get("schema_version") != 1:
        errors.append("source inventory schema")
    if payload.get("payload_sha256") != _digest(payload) or record["payload_sha256"] != payload.get("payload_sha256"):
        errors.append("source inventory payload hash")
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        errors.append("source inventory files")
        return
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            errors.append(f"source inventory[{index}] schema")
            continue
        item_path = row.get("path")
        if not isinstance(item_path, str) or Path(item_path).is_absolute() or not (root / item_path).is_file():
            errors.append(f"source inventory[{index}] path")
            continue
        source = root / item_path
        if (
            row.get("bytes") != source.stat().st_size
            or row.get("sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            errors.append(f"source inventory[{index}] identity")


def _verify_preflight(path: Path) -> bool:
    payload = _read_object(path)
    if payload is None or payload.get("schema_id") != "shakebench.phase07_5a.scene_preflight":
        return False
    gates = payload.get("gates")
    return isinstance(gates, Mapping) and all(
        gates.get(key) is True
        for key in (
            "scene_ready",
            "clearance_passed",
            "visual_physics_invariant",
            "direct_mount_initial_contact_gate",
            "authority_hashes_match",
        )
    )


def _verify_evidence_core(root: Path, record: Any, science_sha256: str, errors: list[str]) -> str | None:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "payload_sha256"}:
        errors.append("evidence core schema")
        return None
    relative = record.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        errors.append("evidence core path")
        return None
    path = root / relative
    core = _read_object(path)
    if core is None:
        errors.append("evidence core missing")
        return None
    if record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        errors.append("evidence core file hash")
    if core.get("schema_id") != "shakebench.phase07_5a.evidence_core" or core.get("schema_version") != 1:
        errors.append("evidence core identity")
    if core.get("science_authority_sha256") != science_sha256:
        errors.append("evidence core science binding")
    try:
        actual = _digest(core)
    except (TypeError, ValueError):
        errors.append("evidence core payload hash")
        return None
    if core.get("payload_sha256") != actual or record.get("payload_sha256") != actual:
        errors.append("evidence core payload hash")
    return actual


def verify_phase07_5a_handoff(path: str | Path = "docs/phase_07_5a_requalification_manifest.json") -> dict[str, Any]:
    """Verify final raw evidence and both one-way authorization bindings."""

    manifest_path = Path(path)
    errors: list[str] = []
    manifest = _read_object(manifest_path)
    if manifest is None:
        return {"passed": False, "errors": ["manifest read"]}
    required = {
        "schema_id",
        "schema_version",
        "status",
        "phase08_authorized",
        "science_authority",
        "science_authority_sha256",
        "final_authority",
        "final_authority_sha256",
        "evidence_core",
        "artifacts",
        "required_final_artifacts",
        "source_inventory",
        "payload_sha256",
    }
    if set(manifest) != required:
        errors.append("manifest fields")
    if manifest.get("schema_id") != "shakebench.phase07_5a.requalification" or manifest.get("schema_version") != 3:
        errors.append("manifest schema")
    if manifest.get("status") != "PASS" or manifest.get("phase08_authorized") is not True:
        errors.append("manifest not final")
    root = _root(manifest_path)
    try:
        if manifest.get("payload_sha256") != _digest(manifest):
            errors.append("manifest payload hash")
    except (TypeError, ValueError):
        errors.append("manifest payload hash")
    try:
        science = dict(verify_direct_mount_authority())
        if manifest.get("science_authority") != science:
            errors.append("science authority binding")
        if manifest.get("science_authority_sha256") != authority_payload_hash(science):
            errors.append("science authority hash")
        core_hash = _verify_evidence_core(root, manifest.get("evidence_core"), authority_payload_hash(science), errors)
        if core_hash is not None:
            final_authority = dict(verify_final_phase08_authority(core_hash))
            if manifest.get("final_authority") != final_authority:
                errors.append("final authority binding")
            if manifest.get("final_authority_sha256") != authority_payload_hash(final_authority):
                errors.append("final authority hash")
    except DirectMountAuthorityError as exc:
        errors.append(f"final authority: {exc}")
    _verify_inventory(root, manifest.get("source_inventory"), errors)
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        errors.append("artifact list")
        records = []
    seen: set[str] = set()
    required_kinds = {"oracle_run", "determinism", "scene_preflight", "video", "package_evidence", "cpu_manifest"}
    kinds: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"artifact[{index}]"
        if not isinstance(record, Mapping) or set(record) != {"kind", "path", "sha256", "semantic_passed"}:
            errors.append(prefix + " schema")
            continue
        relative = record["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or relative in seen:
            errors.append(prefix + " path")
            continue
        seen.add(relative)
        kinds.add(str(record.get("kind")))
        artifact_path = root / relative
        if not artifact_path.is_file():
            errors.append(prefix + " missing")
            continue
        if record.get("sha256") != hashlib.sha256(artifact_path.read_bytes()).hexdigest():
            errors.append(prefix + " hash")
        if record.get("kind") == "oracle_run":
            verdict = verify_run_artifact(artifact_path)
            if record.get("semantic_passed") is not True or not verdict["passed"]:
                errors.append(prefix + " semantic")
        elif record.get("kind") == "determinism":
            verdict = verify_determinism_manifest(artifact_path)
            if record.get("semantic_passed") is not True or not verdict["passed"]:
                errors.append(prefix + " determinism")
        elif record.get("kind") == "scene_preflight":
            if record.get("semantic_passed") is not True or not _verify_preflight(artifact_path):
                errors.append(prefix + " preflight")
        elif record.get("semantic_passed") is not True:
            errors.append(prefix + " verifier verdict")
    if not required_kinds.issubset(kinds):
        errors.append("required artifact kinds")
    return {"passed": not errors, "errors": sorted(set(errors)), "manifest_path": str(manifest_path)}


__all__ = ["verify_phase07_5a_handoff"]
