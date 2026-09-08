"""Fail-closed Phase 7.5A direct-mount scoreability authority.

Geometry describes an assembly.  This module alone grants the separate
authorization required to score that assembly in Phase 8.  The expected
authority digest is deliberately compiled into this module, so rewriting an
asset and its self-reported hash is insufficient to promote a changed scene.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from robosuite import models
from robosuite.utils.shakebench_geometry import load_geometry_profile
from robosuite.utils.shakebench_scene import load_scene_visual_config

AUTHORITY_FILENAME = "shakebench_phase07_5a_requalification.json"
FINAL_AUTHORITY_FILENAME = "shakebench_phase07_5a_final_authority.json"
AUTHORITY_SCHEMA_ID = "shakebench.phase07_5a.science_authority"
AUTHORITY_SCHEMA_VERSION = 1
FINAL_AUTHORITY_SCHEMA_ID = "shakebench.phase07_5a.final_authority"
FINAL_AUTHORITY_SCHEMA_VERSION = 1
# The science digest is replaced only by a reviewed source change. The final
# envelope is authenticated structurally and by its one-way bindings instead of
# a digest compiled into this inventory-covered module; compiling that digest
# here would create manifest -> inventory -> code -> envelope -> manifest cycle.
EXPECTED_AUTHORITY_SHA256 = "45b354c182d1f06cd7105d00b606649dec005029819bf25b95d31794f41d72f4"
OFFICIAL_PHYSICS_SHA256 = "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c"
CONTROLLER_SHA256 = "60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c"
COMPILED_PHYSICS_SIGNATURE = "a1849e266a9ac705dba9abec5be8ece9e1443ad173c063cb24198a62141f0591"


class DirectMountAuthorityError(ValueError):
    """Raised when a direct-mount authorization cannot be authenticated."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectMountAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DirectMountAuthorityError(f"non-canonical authority JSON: {exc}") from exc


def authority_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash canonical authority content excluding its self-check field."""

    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


def _finite_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return not isinstance(value, (bytes, bytearray))


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _asset_path(path: str | Path | None) -> Path:
    if path is None:
        return Path(models.assets_root) / AUTHORITY_FILENAME
    return Path(path)


def _final_asset_path(path: str | Path | None) -> Path:
    if path is None:
        return Path(models.assets_root) / FINAL_AUTHORITY_FILENAME
    return Path(path)


def _read_authority(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except (OSError, json.JSONDecodeError, DirectMountAuthorityError) as exc:
        raise DirectMountAuthorityError(f"authority read failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DirectMountAuthorityError("authority root must be an object")
    return payload


def verify_direct_mount_authority(path: str | Path | None = None) -> Mapping[str, Any]:
    """Return the immutable science authority used by every scoreable episode.

    This verifier deliberately knows nothing about final evidence.  A raw
    trace therefore binds a stable science identity; final evidence is bound
    later by :func:`verify_final_phase08_authority` without rewriting that
    trace or its runtime authority.
    """

    payload = _read_authority(_asset_path(path))
    required = {
        "schema_id",
        "schema_version",
        "status",
        "scoreable",
        "phase08_authorized",
        "geometry_profile_id",
        "geometry_payload_sha256",
        "scene_sha256",
        "compiled_physics_signature_sha256",
        "controller_profile_sha256",
        "official_physics_profile_sha256",
        "dev_state_asset_sha256",
        "payload_sha256",
    }
    if set(payload) != required:
        raise DirectMountAuthorityError("authority fields mismatch")
    if not _finite_json(payload):
        raise DirectMountAuthorityError("authority contains non-finite or binary value")
    if payload.get("schema_id") != AUTHORITY_SCHEMA_ID or payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise DirectMountAuthorityError("unsupported authority schema")
    if payload.get("status") != "EVIDENCE_COLLECTION_AUTHORIZED" or payload.get("scoreable") is not True:
        raise DirectMountAuthorityError("science authority is not authorized for scoreable evidence")
    if payload.get("phase08_authorized") is not False:
        raise DirectMountAuthorityError("science authority must not authorize Phase 8")
    actual_digest = authority_payload_hash(payload)
    if payload.get("payload_sha256") != actual_digest or actual_digest != EXPECTED_AUTHORITY_SHA256:
        raise DirectMountAuthorityError("authority digest is not the frozen expected digest")
    geometry = load_geometry_profile("direct_mount_v1")
    scene = load_scene_visual_config(geometry["scene_config"])
    dev_path = Path(models.assets_root) / "shakebench_states_dev.json"
    bindings = {
        "geometry_profile_id": geometry["profile_id"],
        "geometry_payload_sha256": geometry["payload_sha256"],
        "scene_sha256": scene.config_sha256,
        "compiled_physics_signature_sha256": COMPILED_PHYSICS_SIGNATURE,
        "controller_profile_sha256": CONTROLLER_SHA256,
        "official_physics_profile_sha256": OFFICIAL_PHYSICS_SHA256,
        "dev_state_asset_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise DirectMountAuthorityError(f"authority binding mismatch: {key}")
    return _frozen(dict(payload))


def verify_final_phase08_authority(evidence_manifest_sha256: str, path: str | Path | None = None) -> Mapping[str, Any]:
    """Verify the final handoff envelope without changing episode identity.

    The envelope binds a frozen science-authority hash to one canonical
    evidence-manifest hash.  The evidence manifest must not contain this
    envelope, which prevents a self-referential hash cycle.
    """

    payload = _read_authority(_final_asset_path(path))
    required = {
        "schema_id",
        "schema_version",
        "status",
        "phase08_authorized",
        "science_authority_sha256",
        "final_evidence_manifest_sha256",
        "payload_sha256",
    }
    if set(payload) != required or not _finite_json(payload):
        raise DirectMountAuthorityError("final authority fields mismatch")
    if (
        payload.get("schema_id") != FINAL_AUTHORITY_SCHEMA_ID
        or payload.get("schema_version") != FINAL_AUTHORITY_SCHEMA_VERSION
    ):
        raise DirectMountAuthorityError("unsupported final authority schema")
    if payload.get("status") != "PASS" or payload.get("phase08_authorized") is not True:
        raise DirectMountAuthorityError("final authority is not authorized")
    actual_digest = authority_payload_hash(payload)
    if payload.get("payload_sha256") != actual_digest:
        raise DirectMountAuthorityError("final authority payload digest mismatch")
    science = verify_direct_mount_authority()
    if payload.get("science_authority_sha256") != authority_payload_hash(science):
        raise DirectMountAuthorityError("final authority science binding mismatch")
    if payload.get("final_evidence_manifest_sha256") != evidence_manifest_sha256:
        raise DirectMountAuthorityError("final authority evidence-manifest binding mismatch")
    return _frozen(dict(payload))


def direct_mount_scoreable() -> bool:
    """Return false rather than propagating a failed authority to an env."""

    try:
        verify_direct_mount_authority()
    except DirectMountAuthorityError:
        return False
    return True


__all__ = [
    "AUTHORITY_FILENAME",
    "AUTHORITY_SCHEMA_ID",
    "AUTHORITY_SCHEMA_VERSION",
    "FINAL_AUTHORITY_FILENAME",
    "FINAL_AUTHORITY_SCHEMA_ID",
    "FINAL_AUTHORITY_SCHEMA_VERSION",
    "DirectMountAuthorityError",
    "authority_payload_hash",
    "direct_mount_scoreable",
    "verify_direct_mount_authority",
    "verify_final_phase08_authority",
]
