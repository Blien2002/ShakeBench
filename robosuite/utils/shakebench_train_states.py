"""Deterministic train-split initial states for post-training demonstrations.

The frozen Phase 07 dev list and the Phase 08 official/knee assets keep their
own verifiers and stay unchanged.  This module owns a separate, configurable
train pool so collection can be scaled to N demonstrations without touching a
frozen evaluation asset.  Train states are never scoreable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from robosuite.utils.shakebench_dev_states import CAN_NOMINAL_XY_M, CAN_XY_HALF_RANGE_M

TRAIN_STATE_SCHEMA = "shakebench.train_states"
TRAIN_STATE_SCHEMA_VERSION = 1
TRAIN_STATE_GENERATOR_ID = "shakebench.sha256_uniform.v1"
TRAIN_STATE_SEED_CEILING = 2**32


class TrainStateError(ValueError):
    """Raised when a train-state artifact or split request is invalid."""


def _uniform_word(seed: int, index: int, channel: str) -> float:
    token = f"{TRAIN_STATE_GENERATOR_ID}:{seed}:{index}:{channel}"
    word = int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:8], "big")
    return word / float(1 << 64)


def _integer_word(seed: int, index: int, channel: str) -> int:
    token = f"{TRAIN_STATE_GENERATOR_ID}:{seed}:{index}:{channel}"
    return int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:4], "big")


def train_state_artifact_hash(payload: Mapping[str, Any]) -> str:
    """Hash the complete artifact except its self-referential hash field."""

    if not isinstance(payload, Mapping):
        raise TrainStateError("train-state artifact must be a mapping")
    normalized = copy.deepcopy(dict(payload))
    lock = normalized.get("artifact_lock")
    if isinstance(lock, dict):
        lock.pop("payload_sha256", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_request(count: Any, seed: Any, half_range_m: Any) -> tuple[int, int, float]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise TrainStateError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < TRAIN_STATE_SEED_CEILING:
        raise TrainStateError("seed must be an integer in [0, 2**32)")
    half_range = float(half_range_m)
    if not math.isfinite(half_range) or not 0.0 < half_range <= 0.05:
        # ponytail: 5 cm keeps the can clear of the target and table edge;
        # raise it only after a clearance audit of the wider region.
        raise TrainStateError("half_range_m must be in (0, 0.05]")
    return count, seed, half_range


def generate_train_states(count: int, *, seed: int, half_range_m: float = CAN_XY_HALF_RANGE_M) -> list[dict[str, Any]]:
    """Return ``count`` deterministic train states, all on the can nominal spot."""

    count, seed, half_range = _validate_request(count, seed, half_range_m)
    states = []
    for index in range(count):
        x = CAN_NOMINAL_XY_M[0] + (2.0 * _uniform_word(seed, index, "can_x") - 1.0) * half_range
        y = CAN_NOMINAL_XY_M[1] + (2.0 * _uniform_word(seed, index, "can_y") - 1.0) * half_range
        excitation_seed = _integer_word(seed, index, "excitation_seed")
        states.append(
            {
                "state_id": f"shakebench-train-v0-{index:04d}",
                "split": "train",
                "object_pose_worktable": [x, y, 0.07, 1.0, 0.0, 0.0, 0.0],
                "object_xy_m": [x, y],
                "object_yaw_rad": 0.0,
                "object_initial_velocity": [0.0] * 6,
                "excitation_seed": excitation_seed,
                "seed": excitation_seed,
                "t0_s": _uniform_word(seed, index, "t0_s"),
                "imu_seed": _integer_word(seed, index, "imu_seed"),
            }
        )
    return states


def build_train_state_artifact(count: int, *, seed: int, half_range_m: float = CAN_XY_HALF_RANGE_M) -> dict[str, Any]:
    """Build the deterministic, rollout-independent train pool payload."""

    count, seed, half_range = _validate_request(count, seed, half_range_m)
    payload = {
        "schema_id": TRAIN_STATE_SCHEMA,
        "schema_version": TRAIN_STATE_SCHEMA_VERSION,
        "status": "generated_train_only",
        "split": "train",
        "scoreable": False,
        "generator": {
            "generator_id": TRAIN_STATE_GENERATOR_ID,
            "seed": seed,
            "count": count,
            "half_range_m": half_range,
            "can_nominal_xy_m": list(CAN_NOMINAL_XY_M),
        },
        "states": generate_train_states(count, seed=seed, half_range_m=half_range),
        "artifact_lock": {"payload_sha256": ""},
    }
    payload["artifact_lock"]["payload_sha256"] = train_state_artifact_hash(payload)
    return payload


def verify_train_state_artifact(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate regeneration, bounds, unique IDs, and the payload hash."""

    if isinstance(payload_or_path, Mapping):
        payload = copy.deepcopy(dict(payload_or_path))
    else:
        payload = json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
    generator = payload.get("generator") if isinstance(payload, Mapping) else None
    states = payload.get("states", ()) if isinstance(payload, Mapping) else ()
    expected = None
    regenerate_error = None
    try:
        expected = build_train_state_artifact(
            generator["count"], seed=generator["seed"], half_range_m=generator["half_range_m"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        regenerate_error = f"{type(exc).__name__}: {exc}"
    rows = [row for row in states if isinstance(row, Mapping)] if isinstance(states, list) else []
    half_range = float(generator["half_range_m"]) if isinstance(generator, Mapping) else float("nan")

    def in_bounds(row: Mapping[str, Any]) -> bool:
        xy = row.get("object_xy_m")
        if not isinstance(xy, list) or len(xy) != 2:
            return False
        offsets = [abs(float(xy[axis]) - CAN_NOMINAL_XY_M[axis]) for axis in (0, 1)]
        return all(math.isfinite(value) for value in offsets) and max(offsets) <= half_range + 1e-12

    checks = {
        "schema": payload.get("schema_id") == TRAIN_STATE_SCHEMA,
        "generator": regenerate_error is None,
        "train_only": payload.get("split") == "train" and payload.get("scoreable") is False,
        "count": isinstance(generator, Mapping) and isinstance(states, list) and len(states) == generator.get("count"),
        "unique_ids": isinstance(states, list)
        and len(rows) == len(states)
        and len({row.get("state_id") for row in rows}) == len(states),
        "bounds": bool(rows) and all(in_bounds(row) for row in rows),
        "integrity": isinstance(payload.get("artifact_lock"), Mapping)
        and payload["artifact_lock"].get("payload_sha256") == train_state_artifact_hash(payload),
        "regeneration": expected is not None and payload == expected,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "error": regenerate_error,
        "payload": payload,
        "states": [dict(row) for row in rows],
    }


def assert_split_disjoint(
    train_ids: Iterable[str], eval_ids: Iterable[str], *, train_label: str = "train", eval_label: str = "evaluation"
) -> None:
    """Fail closed when an evaluation state list reuses training states."""

    overlap = sorted(set(train_ids) & set(eval_ids))
    if overlap:
        raise TrainStateError(
            f"{train_label}/{eval_label} state overlap: {overlap[:5]}"
            + (f" (+{len(overlap) - 5} more)" if len(overlap) > 5 else "")
        )


__all__ = [
    "TRAIN_STATE_SCHEMA",
    "TrainStateError",
    "assert_split_disjoint",
    "build_train_state_artifact",
    "generate_train_states",
    "train_state_artifact_hash",
    "verify_train_state_artifact",
]
