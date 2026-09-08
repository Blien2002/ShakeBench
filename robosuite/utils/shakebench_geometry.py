"""Explicit assembly variants, separate from frozen physics and visual profiles."""

import hashlib
import json
from pathlib import Path

import numpy as np

from robosuite import models


def load_geometry_profile(profile="canonical"):
    """Return a validated opt-in assembly; canonical keeps the historical layout."""
    if profile == "canonical":
        return None
    if profile != "direct_mount_v1":
        raise ValueError("geometry_profile must be canonical or direct_mount_v1")
    path = Path(models.assets_root) / "shakebench_geometry_direct_mount_v1.json"
    try:
        payload = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid geometry profile JSON: {exc}") from exc
    required = {
        "schema_id",
        "schema_version",
        "profile_id",
        "geometry_variant",
        "mount_type",
        "initial_joint_qpos_rad",
        "initial_eef_pos_robot_base_m",
        "robot_base_pos_m",
        "table_top_pos_m",
        "scene_config",
        "scene_sha256",
        "physics_effect",
        "retained_table_dimensions_m",
        "mass_inertia_policy",
        "reference",
        "payload_sha256",
    }
    if set(payload) != required:
        raise ValueError("direct-mount geometry fields mismatch")
    digest = payload.pop("payload_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if digest != actual or payload["schema_id"] != "shakebench.geometry" or payload["schema_version"] != 1:
        raise ValueError("invalid geometry profile schema/hash")
    for key in ("robot_base_pos_m", "table_top_pos_m"):
        vector = np.asarray(payload[key], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid geometry profile {key}")
    initial_qpos = np.asarray(payload.get("initial_joint_qpos_rad"), dtype=float)
    if initial_qpos.shape != (7,) or not np.all(np.isfinite(initial_qpos)):
        raise ValueError("direct-mount geometry requires seven finite initial_joint_qpos_rad values")
    initial_eef = np.asarray(payload.get("initial_eef_pos_robot_base_m"), dtype=float)
    if initial_eef.shape != (3,) or not np.all(np.isfinite(initial_eef)):
        raise ValueError("direct-mount geometry requires finite initial_eef_pos_robot_base_m")
    if payload["mount_type"] != "NullMount":
        raise ValueError("direct-mount assembly must declare NullMount")
    if Path(payload["scene_config"]).name != payload["scene_config"]:
        raise ValueError("geometry scene_config must be a packaged filename")
    payload["payload_sha256"] = digest
    return payload


def _reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate geometry key: {key}")
        payload[key] = value
    return payload


def geometry_scene_path(profile):
    """Resolve the assembly's visual authority without opening reference backups."""
    geometry = load_geometry_profile(profile)
    return None if geometry is None else Path(models.assets_root) / geometry["scene_config"]
