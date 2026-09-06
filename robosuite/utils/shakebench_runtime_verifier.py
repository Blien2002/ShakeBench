"""Runtime-only verification for the compact ShakeBench publication contract.

The runtime package deliberately does not open the Phase 06/07 raw evidence
archive.  Full scientific recomputation lives in
``robosuite.scripts.shakebench_audit_evidence`` and requires an explicit
external evidence root or archive.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUNTIME_CONTRACT_FILENAME = "shakebench_runtime_contract.json"


class RuntimePublicationError(ValueError):
    """Raised when a compact package publication contract is invalid."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePublicationError(f"unreadable compact runtime asset {filename}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimePublicationError(f"compact runtime asset {filename} is not an object")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePublicationError(f"unreadable official profile: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimePublicationError("official profile is not an object")
    return value


def _profile_hash(profile: Mapping[str, Any]) -> str:
    payload = dict(profile)
    payload.pop("profile_sha256", None)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def verify_runtime_publication_bundle(asset_root: str | Path) -> dict[str, Any]:
    """Verify only package-owned, compact runtime assets.

    This verifier intentionally never searches for or opens a raw evidence
    member.  Its authority is the immutable profile plus the compact contract
    bindings, not stored full-audit PASS booleans.
    """

    root = Path(asset_root)
    errors: list[str] = []
    try:
        contract = _read_json(root, RUNTIME_CONTRACT_FILENAME)
    except RuntimePublicationError as exc:
        return {"passed": False, "errors": [str(exc)]}

    if contract.get("schema_id") != "shakebench.runtime.publication_contract" or contract.get("schema_version") != 1:
        errors.append("runtime contract schema mismatch")
    if contract.get("payload_sha256") != _payload_hash(contract):
        errors.append("runtime contract payload hash mismatch")
    package_contract = contract.get("package_contract")
    if not isinstance(package_contract, Mapping):
        errors.append("runtime package contract missing")
    else:
        if package_contract.get("raw_archive_required_for_runtime") is not False:
            errors.append("runtime contract requires raw archive")
        if package_contract.get("network_access_required") is not False:
            errors.append("runtime contract requires network")
        if package_contract.get("full_audit_requires_explicit_evidence") is not True:
            errors.append("full-audit explicit-evidence contract missing")

    profile_spec = contract.get("physics_profile")
    protocol_spec = contract.get("protocol")
    selected = contract.get("selected_tuple")
    if (
        not isinstance(profile_spec, Mapping)
        or not isinstance(protocol_spec, Mapping)
        or not isinstance(selected, Mapping)
    ):
        errors.append("runtime selected tuple/profile/protocol binding missing")
        return {"passed": False, "errors": sorted(set(errors))}

    profile_name = str(profile_spec.get("asset_filename", ""))
    protocol_name = str(protocol_spec.get("asset_filename", ""))
    profile_path = root / profile_name
    protocol_path = root / protocol_name
    try:
        profile = _load_yaml(profile_path)
    except RuntimePublicationError as exc:
        errors.append(str(exc))
        profile = {}
    for path, expected, label in (
        (profile_path, profile_spec.get("asset_sha256"), "official profile"),
        (protocol_path, protocol_spec.get("sha256"), "official protocol"),
    ):
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"{label} byte hash binding missing")
        elif not path.is_file() or _file_sha256(path) != expected:
            errors.append(f"{label} byte hash mismatch")

    if profile:
        computed_profile = _profile_hash(profile)
        if computed_profile != profile.get("profile_sha256"):
            errors.append("official profile payload hash mismatch")
        if computed_profile != profile_spec.get("profile_sha256"):
            errors.append("official profile contract hash mismatch")
        if profile.get("profile_id") != "shakebench.official.physics.v2":
            errors.append("official profile id mismatch")
        if profile.get("status") != "official_immutable" or profile.get("scoreable") is not True:
            errors.append("official profile is not immutable scoreable")
        if profile.get("physics", {}).get("contact", {}).get("candidate_id") != selected.get("contact_candidate_id"):
            errors.append("official profile selected contact mismatch")

    if protocol_spec.get("sha256") != contract.get("compact_assets", {}).get(protocol_name):
        errors.append("protocol compact-asset binding mismatch")
    if protocol_spec.get("sha256") != profile.get("protocol_sha256"):
        errors.append("official profile protocol binding mismatch")
    if selected.get("physics_profile_id") != profile_spec.get("profile_id"):
        errors.append("selected tuple profile id mismatch")
    if selected.get("physics_profile_sha256") != profile_spec.get("profile_sha256"):
        errors.append("selected tuple profile hash mismatch")

    compact_assets = contract.get("compact_assets")
    if not isinstance(compact_assets, Mapping):
        errors.append("compact asset hash map missing")
    else:
        for filename, expected in compact_assets.items():
            path = root / str(filename)
            if not isinstance(expected, str) or len(expected) != 64:
                errors.append(f"compact asset hash malformed: {filename}")
            elif not path.is_file() or _file_sha256(path) != expected:
                errors.append(f"compact asset hash mismatch: {filename}")

    rewrite = contract.get("history_rewrite")
    if not isinstance(rewrite, Mapping) or rewrite.get("status") not in {"pending", "complete"}:
        errors.append("history-rewrite binding missing")
    elif rewrite.get("status") == "complete" and (
        not isinstance(rewrite.get("map_sha256"), str) or len(str(rewrite.get("map_sha256"))) != 64
    ):
        errors.append("completed history-rewrite binding has no map hash")

    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "profile_id": profile_spec.get("profile_id"),
        "profile_sha256": profile_spec.get("profile_sha256"),
        "protocol_sha256": protocol_spec.get("sha256"),
        "contact_candidate_id": selected.get("contact_candidate_id"),
        "raw_archive_required_for_runtime": False,
    }


__all__ = ["RUNTIME_CONTRACT_FILENAME", "RuntimePublicationError", "verify_runtime_publication_bundle"]
