"""Deterministic Phase 09 task-state expansion, separate from historical v1.

Official parents are assigned one variant each, in a balanced round robin.
Knee parents retain all six variants for paired calibration diagnostics.
No outcome-dependent filtering or Gamma selection occurs here.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from robosuite import models
from robosuite.utils.shakebench_artifacts import payload_hash, write_json_atomic
from robosuite.utils.shakebench_committed_states import build_committed_state_artifact
from robosuite.utils.shakebench_tasks import OBJECT_SUPPORT, task_variants

TASK_STATE_SCHEMA = "shakebench.phase09.task_states"
TASK_STATE_FILENAMES = {split: f"shakebench_task_states_{split}_v2.json" for split in ("official", "knee")}


def build_task_state_artifact(split: str) -> dict:
    base = build_committed_state_artifact(split)
    from robosuite.utils.shakebench_oracle import OracleControllerProfile
    from robosuite.utils.shakebench_outcomes import outcome_contract_sha256

    bindings = dict(base["authority_hashes"])
    bindings["controller_profile_sha256"] = OracleControllerProfile().sha256
    bindings["outcome_contract_sha256"] = outcome_contract_sha256()
    variants = task_variants()
    parents = base["states"]
    contracts = {spec.variant_id: spec.contract() for spec in variants}
    records = []
    for index, parent in enumerate(parents):
        selected = (variants[index % len(variants)],) if split == "official" else variants
        for spec in selected:
            lower = OBJECT_SUPPORT[spec.object_id][0]
            record = {
                "schema_id": TASK_STATE_SCHEMA + ".record",
                "schema_version": 2,
                "state_id": f"{parent['state_id'].replace('-v1-', '-v2-')}.{spec.variant_id}",
                "parent_state_id": parent["state_id"],
                "parent_state_sha256": parent["canonical_payload_sha256"],
                "split": split,
                "task": spec.to_dict(),
                "object_xy_m": list(parent["can_xy_m"]),
                "object_pose_worktable": [*parent["can_xy_m"], 0.03 - lower, 1.0, 0.0, 0.0, 0.0],
                "object_yaw_rad": 0.0,
                "object_initial_velocity": [0.0] * 6,
                "target_reference": copy.deepcopy(parent["target_reference"]),
                "excitation_seed": parent["excitation_seed"],
                "imu_seed": parent["imu_seed"],
                "t0_s": parent["t0_s"],
                "Gamma": copy.deepcopy(parent["Gamma"]),
                "authority_hashes": {
                    **bindings,
                    "task_contract_sha256": contracts[spec.variant_id]["task_contract_sha256"],
                },
            }
            record["canonical_payload_sha256"] = payload_hash(record, field="canonical_payload_sha256")
            records.append(record)
    artifact = {
        "schema_id": TASK_STATE_SCHEMA,
        "schema_version": 2,
        "split": split,
        "status": "prepared_for_requalification",
        "scoreable": False,
        "qualification": "new task contacts/assets require nominal, vibration and replay requalification before Phase 09 measurement",
        "parent_asset_payload_sha256": base["artifact_lock"]["payload_sha256"],
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
        "authority_hashes": bindings,
        "states": records,
    }
    artifact["payload_sha256"] = payload_hash(artifact)
    return artifact


def verify_task_state_artifact(payload_or_path, *, expected_split=None):
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
        write_json_atomic(path, payload)
        verdict = verify_task_state_artifact(path, expected_split=split)
        if not verdict["passed"]:
            raise ValueError(verdict["errors"])
        paths[split] = path
    return paths
