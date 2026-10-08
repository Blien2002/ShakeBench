"""Load and authenticate ShakeBench state assets of every supported schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class StateAssetError(RuntimeError):
    """Raised for invalid or incomplete Phase 07 run inputs."""


def load_dev_states(path: str | Path) -> list[dict[str, Any]]:
    """Load the pre-existing ten-State Phase 07 development list fail-closed."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateAssetError(f"state asset read failed: {exc}") from exc
    rows = payload.get("states", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise StateAssetError("dev state asset must contain a list of states")
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise StateAssetError("each dev state must be an object")
        state_id = row.get("state_id")
        xy = np.asarray(row.get("object_xy_m"), dtype=float)
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id in seen
            or xy.shape != (2,)
            or not np.all(np.isfinite(xy))
        ):
            raise StateAssetError("dev states require unique state_id and finite object_xy_m[2]")
        seen.add(state_id)
        result.append(
            {
                "state_id": state_id,
                "object_xy_m": xy.tolist(),
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
        raise StateAssetError(f"state asset read failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StateAssetError("state asset must be an object")
    from shakebench.tasks.registry import load_extension_states
    from shakebench.tasks.states.train import TRAIN_STATE_SCHEMA
    from shakebench.tasks.states.variants import TASK_STATE_SCHEMA

    extension = load_extension_states(payload)
    if extension is not None:
        return extension
    schema = payload.get("schema_id")
    if schema == "shakebench.phase07.dev_states":
        return {"states": load_dev_states(source)}
    if schema == TASK_STATE_SCHEMA:
        return {"states": payload["states"]}
    if schema != TRAIN_STATE_SCHEMA and payload.get("split") not in {"official", "knee"}:
        raise StateAssetError("unknown state asset schema/split")
    states = payload.get("states")
    if not isinstance(states, list) or not all(isinstance(row, Mapping) for row in states):
        raise StateAssetError("state asset must contain a list of state objects")
    return {"states": [dict(row) for row in states]}
