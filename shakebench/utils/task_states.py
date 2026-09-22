"""Deterministic task-state expansion for the current eight-object task set.

Official parents are assigned one variant each, in a balanced round robin.
Knee parents retain all eight variants for paired calibration diagnostics.
No outcome-dependent filtering or Gamma selection occurs here.  Version 2
artifacts describe the retired two-surface task set and are rejected: their
records name objects and a surface dimension that no longer exist.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from shakebench import models
from shakebench.utils.artifacts import write_json
from shakebench.utils.committed_states import build_committed_state_artifact
from shakebench.utils.tasks import OBJECTS, task_variants

TASK_STATE_SCHEMA = "shakebench.phase09.task_states"
TASK_STATE_SCHEMA_VERSION = 3
TASK_STATE_FILENAMES = {split: f"shakebench_task_states_{split}.json" for split in ("official", "knee")}
#: Retired artifacts.  They stay on disk as evidence and are never read.
RETIRED_TASK_STATE_FILENAMES = tuple(f"shakebench_task_states_{split}_v2.json" for split in ("official", "knee"))


def build_task_state_artifact(split: str) -> dict:
    base = build_committed_state_artifact(split)
    variants = task_variants()
    parents = base["states"]
    contracts = {spec.variant_id: spec.contract() for spec in variants}
    records = []
    for index, parent in enumerate(parents):
        selected = (variants[index % len(variants)],) if split == "official" else variants
        for spec in selected:
            # The registered start pose is measured, not recomputed here: the
            # object-frame envelope cannot predict a pose whose thinnest axis
            # is not the object's own +z (the spatula and the potato).
            lower = float(OBJECTS[spec.object_id]["start_pose_support"][0])
            quat = spec.start_quat_wxyz
            record = {
                "schema_id": TASK_STATE_SCHEMA + ".record",
                "schema_version": TASK_STATE_SCHEMA_VERSION,
                "state_id": f"{parent['state_id'].replace('-v1-', '-v3-')}.{spec.variant_id}",
                "parent_state_id": parent["state_id"],
                "split": split,
                "task": spec.to_dict(),
                "object_xy_m": list(parent["can_xy_m"]),
                "object_pose_worktable": [
                    *parent["can_xy_m"],
                    0.03 - lower,
                    *quat,
                ],
                "object_start_quat_wxyz": list(quat),
                "object_initial_velocity": [0.0] * 6,
                "target_reference": copy.deepcopy(parent["target_reference"]),
                "excitation_seed": parent["excitation_seed"],
                "imu_seed": parent["imu_seed"],
                "t0_s": parent["t0_s"],
                "Gamma": copy.deepcopy(parent["Gamma"]),
            }
            records.append(record)
    artifact = {
        "schema_id": TASK_STATE_SCHEMA,
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "split": split,
        "status": "prepared_for_requalification",
        "scoreable": False,
        "qualification": "new task contacts/assets require nominal, vibration and replay requalification before Phase 09 measurement",
        "task_count": 1,
        "variant_count": len(variants),
        "parent_state_count": len(parents),
        "available_parent_state_count": len(base["states"]),
        "variant_assignment": "round_robin_by_parent_index" if split == "official" else "full_cross_product",
        "variant_state_counts": {
            spec.variant_id: sum(row["task"] == spec.to_dict() for row in records) for spec in variants
        },
        "state_count": len(records),
        "aggregation": "one pick_place state pool; equal weight per state; pair tiers by full state_id",
        "task_contracts": contracts,
        "states": records,
    }
    return artifact


def verify_task_state_artifact(payload_or_path, *, expected_split=None):
    if isinstance(payload_or_path, (str, Path)):
        name = Path(payload_or_path).name
        if name in RETIRED_TASK_STATE_FILENAMES:
            return {
                "passed": False,
                "errors": [f"{name} describes the retired two-surface task set; use the v3 artifact"],
            }
    try:
        payload = (
            dict(payload_or_path)
            if isinstance(payload_or_path, Mapping)
            else json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
        )
        if not isinstance(payload, dict):
            raise ValueError("task state artifact must be an object")
        split = payload.get("split")
        if expected_split is not None and split != expected_split:
            raise ValueError("task state split mismatch")
        if payload != build_task_state_artifact(split):
            raise ValueError("task state deterministic regeneration or authority mismatch")
        return {"passed": True, "errors": [], "state_count": len(payload["states"])}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"passed": False, "errors": [str(exc)]}


def freeze_task_state_assets(directory=None):
    destination = Path(models.assets_root) if directory is None else Path(directory)
    paths = {}
    for split, filename in TASK_STATE_FILENAMES.items():
        payload = build_task_state_artifact(split)
        path = destination / filename
        write_json(path, payload)
        verdict = verify_task_state_artifact(path, expected_split=split)
        if not verdict["passed"]:
            raise ValueError(verdict["errors"])
        paths[split] = path
    return paths
