"""Current world-fixed assemblies and their scene configuration."""

import json
from pathlib import Path

import numpy as np

from shakebench import models

#: Packaged assemblies: profile id -> (asset filename, schema version, worktable mount).
#: The schema 2 profile predates the explicit mount field and is always the compliant
#: six-axis isolation stage, retained as an extension of the rigid core scene.
GEOMETRY_PROFILES = {
    "world_fixed_arm_v1": ("shakebench_geometry_world_fixed_arm_v1.json", 2, "isolated"),
    "world_fixed_rigid_table_v1": ("shakebench_geometry_world_fixed_rigid_table_v1.json", 3, "rigid"),
}
DEFAULT_GEOMETRY_PROFILE = "world_fixed_arm_v1"


def _profile_spec(profile):
    try:
        return GEOMETRY_PROFILES[profile]
    except (KeyError, TypeError):
        raise ValueError("geometry_profile must be one of " + ", ".join(GEOMETRY_PROFILES)) from None


def worktable_mount(profile=DEFAULT_GEOMETRY_PROFILE):
    """Return the packaged worktable mount: ``isolated`` or ``rigid``."""

    return _profile_spec(profile)[2]


def load_geometry_profile(profile=DEFAULT_GEOMETRY_PROFILE):
    """Load a packaged world-fixed assembly."""
    filename, schema_version, mount = _profile_spec(profile)
    path = Path(models.assets_root) / filename
    try:
        payload = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid geometry profile JSON: {exc}") from exc
    required = {
        "robot_support_mjcf",
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
        "physics_effect",
        "retained_table_dimensions_m",
        "mass_inertia_policy",
        "reference",
    }
    if schema_version >= 3:
        required.add("worktable_mount")
    if set(payload) != required:
        raise ValueError("world-fixed geometry fields mismatch")
    if payload["schema_id"] != "shakebench.geometry" or payload["schema_version"] != schema_version:
        raise ValueError("invalid geometry profile schema")
    if payload.get("worktable_mount", mount) != mount:
        raise ValueError("geometry profile worktable_mount does not match its registered assembly")
    for key in ("robot_base_pos_m", "table_top_pos_m"):
        vector = np.asarray(payload[key], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid geometry profile {key}")
    initial_qpos = np.asarray(payload.get("initial_joint_qpos_rad"), dtype=float)
    if initial_qpos.shape != (7,) or not np.all(np.isfinite(initial_qpos)):
        raise ValueError("world-fixed geometry requires seven finite initial_joint_qpos_rad values")
    initial_eef = np.asarray(payload.get("initial_eef_pos_robot_base_m"), dtype=float)
    if initial_eef.shape != (3,) or not np.all(np.isfinite(initial_eef)):
        raise ValueError("world-fixed geometry requires finite initial_eef_pos_robot_base_m")
    if payload["mount_type"] != "NullMount":
        raise ValueError("world-fixed assembly must declare NullMount")
    if Path(payload["scene_config"]).name != payload["scene_config"]:
        raise ValueError("geometry scene_config must be a packaged filename")
    support_path = Path(payload["robot_support_mjcf"])
    if support_path.is_absolute() or ".." in support_path.parts or support_path.suffix != ".xml":
        raise ValueError("robot_support_mjcf must be a package-relative XML asset")
    if not (Path(models.assets_root) / support_path).is_file():
        raise ValueError("robot support MJCF is missing")
    if payload["profile_id"] != profile:
        raise ValueError("geometry profile identity mismatch")
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
    return Path(models.assets_root) / geometry["scene_config"]
