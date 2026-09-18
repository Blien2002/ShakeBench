"""Pre-registered Phase 07 development states.

Only the ten non-scoreable development states live here.  Phase 08 retains
ownership of the 400 official and 100 knee-calibration states.  Values are
derived from SHA-256 words instead of a library RNG so the artifact is stable
across NumPy and Python versions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE07_DEV_STATE_SCHEMA_ID = "shakebench.phase07.dev_states"
PHASE07_DEV_STATE_SCHEMA_VERSION = 1
PHASE07_DEV_STATE_GENERATOR_ID = "shakebench.sha256_uniform.v1"
PHASE07_DEV_STATE_ROOT_SEED = 20260905
PHASE07_DEV_STATE_COUNT = 10
PHASE07_DEV_STATE_FILENAME = "shakebench_states_dev.json"
CAN_NOMINAL_XY_M = (-0.10, -0.13)
CAN_XY_HALF_RANGE_M = 0.02


class Phase07DevStateError(ValueError):
    """Raised when the frozen Phase 07 dev-state artifact is invalid."""


def _uniform_word(index: int, channel: str) -> float:
    token = f"{PHASE07_DEV_STATE_GENERATOR_ID}:{PHASE07_DEV_STATE_ROOT_SEED}:{index}:{channel}"
    word = int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:8], "big")
    return word / float(1 << 64)


def _integer_word(index: int, channel: str) -> int:
    token = f"{PHASE07_DEV_STATE_GENERATOR_ID}:{PHASE07_DEV_STATE_ROOT_SEED}:{index}:{channel}"
    return int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:4], "big")


def build_phase07_dev_state_artifact() -> dict[str, Any]:
    """Build the deterministic, rollout-independent ten-state payload."""

    states = []
    for index in range(PHASE07_DEV_STATE_COUNT):
        x = CAN_NOMINAL_XY_M[0] + (2.0 * _uniform_word(index, "can_x") - 1.0) * CAN_XY_HALF_RANGE_M
        y = CAN_NOMINAL_XY_M[1] + (2.0 * _uniform_word(index, "can_y") - 1.0) * CAN_XY_HALF_RANGE_M
        states.append(
            {
                "state_id": f"shakebench-dev-v0-{index:03d}",
                "split": "dev",
                "object_pose_worktable": [x, y, 0.07, 1.0, 0.0, 0.0, 0.0],
                "object_xy_m": [x, y],
                "object_yaw_rad": 0.0,
                "target_reference": {
                    "frame": "worktable",
                    "frame_origin_m": [-0.10, 0.17, 0.042],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "excitation_seed": _integer_word(index, "excitation_seed"),
                "seed": _integer_word(index, "excitation_seed"),
                "t0_s": _uniform_word(index, "t0_s"),
                "imu_seed": _integer_word(index, "imu_seed"),
                "object_initial_velocity": [0.0] * 6,
            }
        )
    payload = {
        "schema_id": PHASE07_DEV_STATE_SCHEMA_ID,
        "schema_version": PHASE07_DEV_STATE_SCHEMA_VERSION,
        "status": "frozen_dev_only",
        "scoreable": False,
        "split": "dev",
        "generator": {
            "generator_id": PHASE07_DEV_STATE_GENERATOR_ID,
            "root_seed": PHASE07_DEV_STATE_ROOT_SEED,
            "mapping": "u64_be(sha256(generator_id:root_seed:index:channel)[:8]) / 2**64",
            "selection": "none; frozen before Phase 07 dev evaluation",
        },
        "task_contract": {
            "can_nominal_xy_m": list(CAN_NOMINAL_XY_M),
            "can_xy_distribution": "independent_uniform",
            "can_xy_half_range_m": CAN_XY_HALF_RANGE_M,
            "object_yaw_rad": 0.0,
            "development_gamma_commanded": [0.0, 0.15, 0.30],
            "gamma_is_run_condition": True,
        },
        "physics_profile": {
            "profile_id": "shakebench.official.physics.v2",
        },
        "states": states,
    }
    return payload


def verify_phase07_dev_state_artifact(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate provenance, exact regeneration, bounds, IDs, and hash."""

    if isinstance(payload_or_path, Mapping):
        payload = copy.deepcopy(dict(payload_or_path))
    else:
        payload = json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
    expected = build_phase07_dev_state_artifact()
    states = payload.get("states", ()) if isinstance(payload, Mapping) else ()
    checks = {
        "schema": payload.get("schema_id") == PHASE07_DEV_STATE_SCHEMA_ID
        and payload.get("schema_version") == PHASE07_DEV_STATE_SCHEMA_VERSION,
        "dev_only": payload.get("split") == "dev" and payload.get("scoreable") is False,
        "count": isinstance(states, list) and len(states) == PHASE07_DEV_STATE_COUNT,
        "regeneration": payload == expected,
    }
    return {"passed": all(checks.values()), "checks": checks, "payload": payload}


__all__ = [
    "PHASE07_DEV_STATE_FILENAME",
    "PHASE07_DEV_STATE_ROOT_SEED",
    "Phase07DevStateError",
    "build_phase07_dev_state_artifact",
    "verify_phase07_dev_state_artifact",
]
