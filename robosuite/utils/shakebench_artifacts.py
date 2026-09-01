"""Canonical, atomic artifact primitives for new Phase 06 selection runs.

Archived V1/V2/V3 writers keep their historical canonicalization rules. New
V4 artifacts use these primitives so hashes, raw-file manifests, and writes
have one auditable implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def payload_hash(value: Mapping[str, Any], field: str = "payload_sha256") -> str:
    content = dict(value)
    content.pop(field, None)
    return sha256_json(content)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write one JSON artifact atomically and return its file SHA-256."""

    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return file_sha256(destination)


def verify_payload_hash(payload: Mapping[str, Any], field: str = "payload_sha256") -> bool:
    return payload.get(field) == payload_hash(payload, field=field)
