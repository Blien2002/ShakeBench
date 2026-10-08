"""Deterministic train-split initial states for post-training demonstrations.

The frozen Phase 07 dev list and the Phase 08 official/knee assets keep their
own verifiers and stay unchanged.  This module owns a separate, configurable
train pool so collection can be scaled to N demonstrations without touching a
frozen evaluation asset.  Train states are never scoreable.

Every train state randomizes four independent channels: the object's planar
position, its start pose (one registered rest pose, including the can's and the
mug's lying poses), the world-frame yaw of that pose, and the excitation/IMU
seeds.  The yaw carries an asymmetric object's facing direction, such as the
mug's cup body, into the state. Apples sample qualified stable rest poses
plus yaw, with the start height measured from the oriented collision mesh.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from shakebench.tasks.catalog import TaskSpec, compose_yaw_wxyz, registered_rest_pose
from shakebench.tasks.states.schema import normalize_state

TRAIN_STATE_SCHEMA = "shakebench.train_states"


class TrainStateError(ValueError):
    """Raised when a train-state artifact or split request is invalid."""


def execution_state_fingerprint(state: Mapping[str, Any]) -> str:
    """Execution identity of one state, independent of its chosen ID and split.

    The comparison covers the inputs that decide an episode: the object start
    position, task spec and orientation, the excitation and IMU seeds, and the
    time offset.  The worktable pose and the initial velocity are derived from
    those by every state producer and are omitted entirely by the frozen dev
    asset, so they are not part of the identity.
    """

    normalized = normalize_state(state)
    try:
        seed = int(normalized.get("excitation_seed", normalized.get("seed", 0)))
        payload = {
            "object_xy_m": [float(value) for value in normalized["object_xy_m"]],
            "task": normalized.get("task"),
            "grasp_region": str(normalized.get("grasp_region", "body")),
            "object_yaw_rad": float(normalized.get("object_yaw_rad", normalized.get("can_yaw_rad", 0.0))),
            "excitation_seed": seed,
            "imu_seed": int(normalized.get("imu_seed", seed)),
            "t0_s": float(normalized.get("t0_s", 0.0)),
        }
    except (TypeError, ValueError):
        raise TrainStateError("state must carry numeric object_xy_m, seeds and t0_s") from None
    if isinstance(normalized.get("task"), Mapping) and normalized["task"].get("object_id") == "apple":
        quat = normalized.get("object_start_quat_wxyz")
        if quat is None:
            registered, _ = registered_rest_pose(TaskSpec(object_id="apple"))
            quat = compose_yaw_wxyz(registered, payload["object_yaw_rad"])
        # q and -q encode the same rotation; the legacy yaw is redundant here.
        sign = -1.0 if next((value for value in quat if value != 0), 1.0) < 0 else 1.0
        payload["object_start_quat_wxyz"] = [sign * float(value) if value else 0.0 for value in quat]
        payload.pop("object_yaw_rad")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _state_identity(value: Any) -> tuple[str, str]:
    """Return the comparison key and the human label of one state or state ID."""

    if isinstance(value, Mapping):
        label = str(value.get("state_id") or "unnamed-state")
        return execution_state_fingerprint(value), label
    return str(value), str(value)


def split_overlap(train_states: Iterable[Any], eval_states: Iterable[Any]) -> list[str]:
    """Labels of the evaluation states that repeat a training execution state.

    Accepts state mappings, compared by :func:`execution_state_fingerprint`, or
    bare state IDs, compared as strings.
    """

    train = {identity: label for identity, label in map(_state_identity, train_states)}
    evaluation = {identity for identity, _ in map(_state_identity, eval_states)}
    return sorted(train[identity] for identity in set(train) & evaluation)


def assert_split_disjoint(
    train_states: Iterable[Any],
    eval_states: Iterable[Any],
    *,
    train_label: str = "train",
    eval_label: str = "evaluation",
) -> None:
    """Fail closed when evaluation reuses a training state.

    Bare state IDs alone cannot see two generators that emit different names for
    the same execution state, so pass the state records to compare them by
    execution fingerprint.
    """

    overlap = split_overlap(train_states, eval_states)
    if overlap:
        raise TrainStateError(
            f"{train_label}/{eval_label} state overlap: {overlap[:5]}"
            + (f" (+{len(overlap) - 5} more)" if len(overlap) > 5 else "")
        )


__all__ = [
    "TRAIN_STATE_SCHEMA",
    "TrainStateError",
    "assert_split_disjoint",
    "execution_state_fingerprint",
    "split_overlap",
]
