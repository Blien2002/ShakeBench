"""Execution-boundary state normalization for verified ShakeBench state assets.

Verified Phase 08 committed assets name the object ``can_*``; the environment
boundary reads ``object_*``.  Normalize once here so every runner accepts both
schemas, and keep the original fields so provenance and authority hashes still
refer to the verified record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Execution-boundary name -> verified Phase 08 committed-asset name.
COMMITTED_STATE_ALIASES = {
    "object_xy_m": "can_xy_m",
    "object_pose_worktable": "can_pose_worktable",
    "object_yaw_rad": "can_yaw_rad",
    "object_initial_velocity": "can_initial_velocity",
}


class StateSchemaError(ValueError):
    """Raised when a state record cannot be executed as supplied."""


def normalize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying the execution names for one verified state."""

    if not isinstance(state, Mapping):
        raise StateSchemaError("state must be a mapping loaded from a verified asset")
    normalized = dict(state)
    for canonical, committed in COMMITTED_STATE_ALIASES.items():
        if canonical not in normalized and committed in normalized:
            normalized[canonical] = normalized[committed]
    try:
        xy = [float(value) for value in normalized["object_xy_m"]]
    except (KeyError, TypeError, ValueError):
        raise StateSchemaError("state must include finite object_xy_m[2]") from None
    if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
        raise StateSchemaError("state must include finite object_xy_m[2]")
    return normalized


__all__ = ["COMMITTED_STATE_ALIASES", "StateSchemaError", "normalize_state"]
