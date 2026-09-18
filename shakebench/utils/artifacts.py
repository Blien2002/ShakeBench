"""JSON artifact primitives for ShakeBench result files.

Writes are ordinary open/write/close: no temporary file, no rename step, and
no digest of the written bytes.
"""

from __future__ import annotations

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


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
