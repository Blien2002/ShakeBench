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

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from shakebench.utils.dev_states import CAN_NOMINAL_XY_M
from shakebench.utils.state_schema import normalize_state
from shakebench.utils.tasks import (
    APPLE_STABLE_QUATS_WXYZ,
    OBJECTS,
    TaskSpec,
    apple_pose_support,
    compose_yaw_wxyz,
    registered_rest_pose,
    sample_apple_stable_quat_wxyz,
)

TRAIN_STATE_SCHEMA = "shakebench.train_states"
TRAIN_STATE_SCHEMA_VERSION = 5
# Own namespace: sharing the Phase 07 dev namespace made generate_train_states
# reproduce the frozen dev execution states word for word.
TRAIN_STATE_GENERATOR_ID = "shakebench.sha256_uniform.train.v1"
TRAIN_STATE_SEED_CEILING = 2**32
#: The worktable frame origin sits this far below the tabletop surface
#: (``ShakeBenchArena.table_half_size[2]``), so a posed object origin rests at
#: ``TRAIN_TABLE_SURFACE_Z_M - lower_support_z``.
TRAIN_TABLE_SURFACE_Z_M = 0.03
#: Canonical worktable half extents (``table_full_size`` 0.65 x 0.60 x 0.06).
TRAIN_TABLE_HALF_EXTENTS_XY_M = (0.325, 0.30)
#: Oracle table-edge margin; the sampled pose keeps this much clear of the edge.
TRAIN_TABLE_EDGE_MARGIN_M = 0.008
#: Default square half range of the object start position.
TRAIN_XY_HALF_RANGE_M = 0.08


class TrainStateError(ValueError):
    """Raised when a train-state artifact or split request is invalid."""


def _uniform_word(seed: int, index: int, channel: str) -> float:
    token = f"{TRAIN_STATE_GENERATOR_ID}:{seed}:{index}:{channel}"
    word = int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:8], "big")
    return word / float(1 << 64)


def _integer_word(seed: int, index: int, channel: str) -> int:
    token = f"{TRAIN_STATE_GENERATOR_ID}:{seed}:{index}:{channel}"
    return int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:4], "big")


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


def _validate_request(count: Any, seed: Any, half_range_m: Any) -> tuple[int, int, float]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise TrainStateError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < TRAIN_STATE_SEED_CEILING:
        raise TrainStateError("seed must be an integer in [0, 2**32)")
    half_range = float(half_range_m)
    limit = max_xy_half_range_m()
    if not math.isfinite(half_range) or not 0.0 < half_range <= limit:
        # ponytail: the cap keeps the worst registered rest pose fully on the
        # worktable (nearest edge minus posed radius minus the edge margin).
        # Re-measure those three numbers before widening it.
        raise TrainStateError(f"half_range_m must be in (0, {limit:.4f}]")
    return count, seed, half_range


def _validate_per_variant(per_variant: Any) -> int | None:
    """Return the requested per-variant budget, or None for the random pool."""

    if per_variant is None:
        return None
    if isinstance(per_variant, bool) or not isinstance(per_variant, int) or per_variant < 1:
        raise TrainStateError("per_variant must be a positive integer")
    return per_variant


#: Rest poses the train pool samples per object; an object not listed here is
#: sampled in its registry body pose.  The mug keeps one entry per qualified
#: grip: the double-wall body grip in the registry upright pose, and the
#: double-wall grip of the lying cup. The can has standing and side-lying
#: poses. Single-wall mug pinches are retired.
TRAIN_POSE_REGIONS = {"mug": ("body", "side_double_wall"), "can": ("body", "side")}


def _pose_regions(object_id: str) -> tuple[str, ...]:
    """Return the grasp regions a train pool samples for one object."""

    return TRAIN_POSE_REGIONS.get(object_id, ("body",))


def _posed_radius_m(spec: TaskSpec, region: str) -> float:
    """Table footprint radius of one registered rest pose."""

    entry = OBJECTS[spec.object_id]
    plan = spec.grasp_plan(region)
    return float(plan.get("start_pose_support", entry["support"])[2])


def max_xy_half_range_m() -> float:
    """Largest square half range whose worst rest pose still clears the table edge."""

    room = min(
        TRAIN_TABLE_HALF_EXTENTS_XY_M[axis] - abs(CAN_NOMINAL_XY_M[axis]) for axis in (0, 1)
    )
    radii = [
        _posed_radius_m(TaskSpec(object_id=object_id), region)
        for object_id in OBJECTS
        for region in _pose_regions(object_id)
    ]
    return room - max(radii) - TRAIN_TABLE_EDGE_MARGIN_M


def _variant_pool(object_ids: Iterable[str] | None) -> tuple[tuple[TaskSpec, str], ...]:
    """Return the (variant, grasp region) rest poses a train pool may sample."""

    if object_ids is None:
        requested = tuple(OBJECTS)
    else:
        try:
            requested = tuple(dict.fromkeys(str(value) for value in object_ids))
        except TypeError:
            raise TrainStateError("object_ids must be an iterable of registry object ids") from None
        if not requested:
            raise TrainStateError("object_ids must select at least one registered object")
    pool = []
    for object_id in requested:
        spec = TaskSpec(object_id=object_id)
        for region in _pose_regions(object_id):
            # Fail closed if the registry stops declaring this region's pose.
            registered_rest_pose(spec, region)
            pool.append((spec, region))
    return tuple(pool)


def generate_train_states(
    count: int,
    *,
    seed: int,
    half_range_m: float = TRAIN_XY_HALF_RANGE_M,
    object_ids: Iterable[str] | None = None,
    per_variant: int | None = None,
) -> list[dict[str, Any]]:
    """Return ``count`` deterministic train states around the can nominal spot.

    Position, orientation and the two seed channels are independent. Apples
    sample qualified stable tilts plus yaw; other objects keep a registered rest
    pose plus yaw. With ``per_variant`` set, the pool is filled one rest pose
    at a time instead of sampled at random, so a
    collection can spend an exact episode budget on every object pose and grip.
    """

    count, seed, half_range = _validate_request(count, seed, half_range_m)
    per_variant = _validate_per_variant(per_variant)
    pool = _variant_pool(object_ids)
    states = []
    for index in range(count):
        if per_variant is None:
            spec, region = pool[_integer_word(seed, index, "object_variant") % len(pool)]
        else:
            spec, region = pool[(index // per_variant) % len(pool)]
        yaw = 2.0 * math.pi * _uniform_word(seed, index, "object_yaw")
        registered, lower_support = registered_rest_pose(spec, region)
        quat = compose_yaw_wxyz(registered, yaw)
        if spec.object_id == "apple":
            quat = sample_apple_stable_quat_wxyz(_uniform_word(seed, index, "apple_stable_pose"), yaw)
            lower_support = apple_pose_support(quat)[0]
        x = CAN_NOMINAL_XY_M[0] + (2.0 * _uniform_word(seed, index, "can_x") - 1.0) * half_range
        y = CAN_NOMINAL_XY_M[1] + (2.0 * _uniform_word(seed, index, "can_y") - 1.0) * half_range
        excitation_seed = _integer_word(seed, index, "excitation_seed")
        states.append(
            {
                "state_id": f"shakebench-train-v2-s{seed}-r{half_range:g}-{index:04d}",
                "split": "train",
                "task": spec.to_dict(),
                "grasp_region": region,
                "object_xy_m": [x, y],
                "object_yaw_rad": yaw,
                "object_pose_worktable": [x, y, TRAIN_TABLE_SURFACE_Z_M - lower_support, *quat],
                "object_start_quat_wxyz": list(quat),
                "object_initial_velocity": [0.0] * 6,
                "excitation_seed": excitation_seed,
                "seed": excitation_seed,
                "t0_s": _uniform_word(seed, index, "t0_s"),
                "imu_seed": _integer_word(seed, index, "imu_seed"),
            }
        )
    return states


def build_train_state_artifact(
    count: int,
    *,
    seed: int,
    half_range_m: float = TRAIN_XY_HALF_RANGE_M,
    object_ids: Iterable[str] | None = None,
    per_variant: int | None = None,
) -> dict[str, Any]:
    """Build the deterministic, rollout-independent train pool payload."""

    count, seed, half_range = _validate_request(count, seed, half_range_m)
    per_variant = _validate_per_variant(per_variant)
    pool = _variant_pool(object_ids)
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
            "per_variant": per_variant,
            "variants": [
                {"object_id": spec.object_id, "grasp_region": region} for spec, region in pool
            ],
            "apple_orientation_distribution": "uniform_stable_poses_plus_yaw_v1",
            "yaw_distribution": {
                "channel": "object_yaw",
                "distribution": "independent_uniform",
                "range_rad": [0.0, 2.0 * math.pi],
            },
        },
        "states": generate_train_states(
            count, seed=seed, half_range_m=half_range, object_ids=object_ids, per_variant=per_variant
        ),
    }
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
            generator["count"],
            seed=generator["seed"],
            half_range_m=generator["half_range_m"],
            object_ids=[entry["object_id"] for entry in generator.get("variants", ())] or None,
            per_variant=generator.get("per_variant"),
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

    def orientation_ok(row: Mapping[str, Any]) -> bool:
        """Validate qualified rest poses plus the declared yaw."""

        try:
            spec = TaskSpec.from_mapping(row["task"])
            region = str(row["grasp_region"])
            yaw = float(row["object_yaw_rad"])
            quat = [float(value) for value in row["object_start_quat_wxyz"]]
            pose = [float(value) for value in row["object_pose_worktable"]]
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(yaw) or not 0.0 <= yaw < 2.0 * math.pi:
            return False
        if len(quat) != 4 or len(pose) != 7:
            return False
        norm = math.sqrt(sum(value * value for value in quat))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            return False
        registered, lower_support = registered_rest_pose(spec, region)
        registered = compose_yaw_wxyz(registered, yaw)
        if spec.object_id == "apple":
            lower_support = apple_pose_support(quat)[0]
            if not any(
                max(abs(actual - expected) for actual, expected in zip(quat, compose_yaw_wxyz(rest, yaw))) <= 1e-6
                for rest in APPLE_STABLE_QUATS_WXYZ
            ):
                return False
            registered = quat
        return bool(
            max(abs(quat[axis] - registered[axis]) for axis in range(4)) <= 1e-6
            and max(abs(pose[3 + axis] - quat[axis]) for axis in range(4)) <= 1e-6
            and abs(pose[2] - (TRAIN_TABLE_SURFACE_Z_M - lower_support)) <= 1e-9
        )

    pool = (
        {(str(entry["object_id"]), str(entry["grasp_region"])) for entry in generator.get("variants", ())}
        if isinstance(generator, Mapping)
        else set()
    )

    checks = {
        "schema": payload.get("schema_id") == TRAIN_STATE_SCHEMA,
        "generator": regenerate_error is None,
        "train_only": payload.get("split") == "train" and payload.get("scoreable") is False,
        "count": isinstance(generator, Mapping) and isinstance(states, list) and len(states) == generator.get("count"),
        "unique_ids": isinstance(states, list)
        and len(rows) == len(states)
        and len({row.get("state_id") for row in rows}) == len(states),
        "bounds": bool(rows) and all(in_bounds(row) for row in rows),
        "variants": bool(rows)
        and bool(pool)
        and all(
            isinstance(row.get("task"), Mapping)
            and (str(row["task"].get("object_id")), str(row.get("grasp_region"))) in pool
            for row in rows
        ),
        "orientation": bool(rows) and all(orientation_ok(row) for row in rows),
        "regeneration": expected is not None and payload == expected,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "error": regenerate_error,
        "payload": payload,
        "states": [dict(row) for row in rows],
    }


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
    "build_train_state_artifact",
    "execution_state_fingerprint",
    "generate_train_states",
    "split_overlap",
    "verify_train_state_artifact",
]
