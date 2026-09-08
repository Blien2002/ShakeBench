"""Authoritative ShakeBench scene visuals, inventory, and clearance audits.

The scene layer is deliberately separate from the benchmark physics contract.
It owns layout, materials, camera poses, and display-only geometry while the
arena and task continue to own the tabletop collision, inertial, contacts, and
success evaluator.  The public surface is intentionally small:

``load_scene_visual_config`` -> validated immutable configuration
``augment_scene_mjcf`` -> deterministic visual MJCF and scene inventory
``audit_compiled_scene`` -> frame and physics-invariance audit
``scene_clearance_report`` -> nominal/sampled signed-distance gate

No visual body created here has a joint or inertial.  Its geoms explicitly use
zero density and zero contact bits so that a display-only scene cannot add a
mass, constraint, contact pair, or task interaction.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from robosuite import models
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_element, new_geom

SCENE_SCHEMA_ID = "shakebench.scene_visual"
SCENE_SCHEMA_VERSION = 1
DEFAULT_SCENE_VISUAL_CONFIG_FILENAME = "shakebench_scene_visual_v1.json"
SCENE_CONFIG_PATH = Path(models.assets_root) / DEFAULT_SCENE_VISUAL_CONFIG_FILENAME
SCENE_GEOM_PREFIX = "shakebench_"
SCENE_BODY_PREFIX = "shakebench_"
TABLE_VISUAL_GEOM_NAME = "table_visual"
DECK_VISUAL_BODY_NAME = "shakebench_platen_visual"
FOUNDATION_BODY_NAME = "shakebench_shaker_foundation"
PIT_BODY_NAME = "shakebench_pit"
GUARDRAIL_BODY_NAME = "shakebench_guardrails"
CONTROL_CABINET_BODY_NAME = "shakebench_control_cabinet"
EMERGENCY_STOP_BODY_NAME = "shakebench_emergency_stop"
WORKTABLE_BODY_NAME = "worktable"
DECK_BODY_NAME = "deck"
DECK_FREEJOINT_NAME = "deck_freejoint"
ROBOT_BASE_BODY_NAME = "robot0_base"


class SceneConfigError(ValueError):
    """Raised when the authoritative scene configuration is invalid."""


class SceneAuditError(SceneConfigError):
    """Raised when a compiled scene violates its structural contract."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_scene_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON payload used by the scene SHA-256."""

    if not isinstance(payload, Mapping):
        raise SceneConfigError("scene configuration must be an object")
    content = copy.deepcopy(_json_ready(payload))
    content.pop("payload_sha256", None)
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SceneConfigError(f"scene configuration is not JSON-canonical: {exc}") from exc


def scene_config_hash(payload: Mapping[str, Any]) -> str:
    """Compute the SHA-256 digest embedded in a scene configuration."""

    return hashlib.sha256(canonical_scene_payload(payload).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _number(name: str, value: Any, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SceneConfigError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneConfigError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise SceneConfigError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        comparator = ">" if strict else ">="
        raise SceneConfigError(f"{name} must be {comparator} {minimum}")
    return result


def _vector(
    name: str, value: Any, length: int, *, minimum: float | None = None, strict: bool = False
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise SceneConfigError(f"{name} must contain {length} finite values")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise SceneConfigError(f"{name} must contain {length} finite values") from exc
    if len(values) != length:
        raise SceneConfigError(f"{name} must contain {length} finite values")
    return tuple(_number(f"{name}[{index}]", item, minimum=minimum, strict=strict) for index, item in enumerate(values))


def _rgb(name: str, value: Any) -> tuple[float, float, float]:
    values = _vector(name, value, 3, minimum=0.0)
    if any(item > 1.0 for item in values):
        raise SceneConfigError(f"{name} values must lie in [0, 1]")
    return values


def _rgba(name: str, value: Any) -> tuple[float, float, float, float]:
    values = _vector(name, value, 4, minimum=0.0)
    if any(item > 1.0 for item in values):
        raise SceneConfigError(f"{name} values must lie in [0, 1]")
    return values


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneConfigError(f"{name} must be an object")
    return value


def _require_unique_strings(name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise SceneConfigError(f"{name} must be a list of non-empty strings")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise SceneConfigError(f"{name} must be a list of non-empty strings") from exc
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise SceneConfigError(f"{name} must be a list of non-empty strings")
    if len(set(values)) != len(values):
        raise SceneConfigError(f"{name} contains duplicate entries")
    return values


def _validate_source(source: Mapping[str, Any]) -> None:
    references = source.get("reference_files")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)) or not references:
        raise SceneConfigError("source.reference_files must be a non-empty list")
    for index, reference in enumerate(references):
        reference = _require_mapping(f"source.reference_files[{index}]", reference)
        path = reference.get("path")
        digest = reference.get("sha256")
        if not isinstance(path, str) or not path.strip() or Path(path).is_absolute():
            raise SceneConfigError("scene reference paths must be relative, non-empty strings")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SceneConfigError(f"source.reference_files[{index}].sha256 must be a lowercase SHA-256 digest")


def _validate_frame_ownership(value: Any) -> Mapping[str, Any]:
    ownership = _require_mapping("frame_ownership", value)
    expected_frames = {"world", "dynamic_deck", "isolated_worktable", "world_free_body"}
    if set(ownership) != expected_frames:
        missing = sorted(expected_frames - set(ownership))
        extra = sorted(set(ownership) - expected_frames)
        raise SceneConfigError(f"frame_ownership frame set mismatch; missing={missing}, extra={extra}")
    seen_bodies: set[str] = set()
    for frame, content in ownership.items():
        content = _require_mapping(f"frame_ownership.{frame}", content)
        bodies = _require_unique_strings(f"frame_ownership.{frame}.bodies", content.get("bodies", ()))
        _require_unique_strings(f"frame_ownership.{frame}.geom_prefixes", content.get("geom_prefixes", ()))
        duplicate = seen_bodies.intersection(bodies)
        if duplicate:
            raise SceneConfigError(f"frame_ownership assigns body role(s) twice: {sorted(duplicate)}")
        seen_bodies.update(bodies)
    return ownership


def _validate_scene_payload(payload: Mapping[str, Any], path: Path) -> None:
    required = {
        "schema_id",
        "schema_version",
        "scene_id",
        "geometry_variant",
        "physics_effect",
        "source",
        "frame_ownership",
        "room",
        "pit",
        "platen",
        "stewart",
        "table_support",
        "clearance",
        "safe_excitation_envelope",
        "render",
        "cameras",
        "materials",
        "role_handles",
        "expected_roles",
        "known_omissions",
        "payload_sha256",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise SceneConfigError(f"scene config fields mismatch; missing={missing}, unknown={unknown}")
    if payload["schema_id"] != SCENE_SCHEMA_ID or int(payload["schema_version"]) != SCENE_SCHEMA_VERSION:
        raise SceneConfigError("unsupported ShakeBench scene schema")
    if not isinstance(payload["scene_id"], str) or not payload["scene_id"].strip():
        raise SceneConfigError("scene_id must be a non-empty string")
    if payload["geometry_variant"] not in {"A", "B", "C", "B-size"}:
        raise SceneConfigError("geometry_variant must be A, B, C, or B-size")
    if payload["physics_effect"] is not False:
        raise SceneConfigError("scene visual configuration must declare physics_effect=false")
    _validate_source(_require_mapping("source", payload["source"]))
    _validate_frame_ownership(payload["frame_ownership"])

    role_handles = _require_mapping("role_handles", payload["role_handles"])
    for role, body in role_handles.items():
        if not isinstance(role, str) or not role.strip() or not isinstance(body, str) or not body.strip():
            raise SceneConfigError("role_handles must map non-empty role names to body names")
    if role_handles.get("deck_visual") != DECK_VISUAL_BODY_NAME:
        raise SceneConfigError("role_handles.deck_visual must bind shakebench_platen_visual")
    if role_handles.get("isolated_worktable") != WORKTABLE_BODY_NAME:
        raise SceneConfigError("role_handles.isolated_worktable must bind worktable")
    _require_unique_strings("expected_roles", payload["expected_roles"])
    _require_unique_strings("known_omissions", payload["known_omissions"])

    room = _require_mapping("room", payload["room"])
    _vector("room.envelope_m", room.get("envelope_m"), 3, minimum=0.0, strict=True)
    _number("room.wall_thickness_m", room.get("wall_thickness_m"), minimum=0.0, strict=True)
    _number("room.wall_height_m", room.get("wall_height_m"), minimum=0.0, strict=True)
    _number("room.floor_slab_thickness_m", room.get("floor_slab_thickness_m"), minimum=0.0, strict=True)
    _validate_laboratory_details(room, payload["frame_ownership"])

    pit = _require_mapping("pit", payload["pit"])
    opening = _vector("pit.opening_xy_m", pit.get("opening_xy_m"), 2, minimum=0.0, strict=True)
    envelope = room["envelope_m"]
    if opening[0] >= envelope[0] or opening[1] >= envelope[1]:
        raise SceneConfigError("pit opening must fit inside the room envelope")
    _number("pit.depth_m", pit.get("depth_m"), minimum=0.0, strict=True)
    _number("pit.border_width_m", pit.get("border_width_m"), minimum=0.0, strict=True)
    _number("pit.safety_line_width_m", pit.get("safety_line_width_m"), minimum=0.0, strict=True)
    _rgb("pit.dark_rgb", pit.get("dark_rgb"))
    _rgb("pit.border_rgb", pit.get("border_rgb"))
    _rgb("pit.safety_rgb", pit.get("safety_rgb"))
    if "guardrail_details" in pit:
        details = _require_mapping("pit.guardrail_details", pit["guardrail_details"])
        for key in (
            "bend_radius_m",
            "stripe_pitch_m",
            "coupler_radius_m",
            "coupler_length_m",
            "anchor_radius_m",
            "anchor_height_m",
            "anchor_xy_offset_m",
        ):
            _number(f"pit.guardrail_details.{key}", details.get(key), minimum=0, strict=True)
        _vector("pit.guardrail_details.foot_plate_size_m", details.get("foot_plate_size_m"), 3, minimum=0, strict=True)
        for key in ("black_rgba", "coupler_rgba"):
            _rgba(f"pit.guardrail_details.{key}", details.get(key))
        fraction = _number("pit.guardrail_details.black_fraction", details.get("black_fraction"))
        if not 0 < fraction < 1:
            raise SceneConfigError("pit.guardrail_details.black_fraction must lie in (0, 1)")
        if not float(pit["guardrail_thickness_m"]) < float(details["bend_radius_m"]) < min(opening) / 2:
            raise SceneConfigError("guardrail bend radius must fit between tube thickness and pit half size")
        for key in ("elbow_arc_segments", "elbow_ring_segments"):
            value = _number(f"pit.guardrail_details.{key}", details.get(key), minimum=4)
            if value != int(value):
                raise SceneConfigError(f"pit.guardrail_details.{key} must be an integer")

    platen = _require_mapping("platen", payload["platen"])
    platen_size = _vector("platen.size_m", platen.get("size_m"), 3, minimum=0.0, strict=True)
    _number("platen.nominal_top_z_m", platen.get("nominal_top_z_m"))
    _rgba("platen.rgba", platen.get("rgba"))
    if platen_size[0] >= opening[0] or platen_size[1] >= opening[1]:
        raise SceneConfigError("platen must fit inside the pit opening")

    stewart = _require_mapping("stewart", payload["stewart"])
    _vector("stewart.base_ellipse_semi_axes_m", stewart.get("base_ellipse_semi_axes_m"), 2, minimum=0.0, strict=True)
    _vector(
        "stewart.platen_ellipse_semi_axes_m", stewart.get("platen_ellipse_semi_axes_m"), 2, minimum=0.0, strict=True
    )
    _number("stewart.joint_pair_spread_deg", stewart.get("joint_pair_spread_deg"), minimum=0.0, strict=True)
    if not 0.0 < float(stewart["joint_pair_spread_deg"]) < 60.0:
        raise SceneConfigError("stewart joint pair spread must be in (0, 60) degrees")
    for key in (
        "leg_nominal_m",
        "leg_min_m",
        "leg_max_m",
        "outer_length_m",
        "rod_length_m",
        "outer_radius_m",
        "rod_radius_m",
        "joint_radius_m",
    ):
        _number(f"stewart.{key}", stewart.get(key), minimum=0.0, strict=True)
    if not float(stewart["leg_min_m"]) < float(stewart["leg_nominal_m"]) < float(stewart["leg_max_m"]):
        raise SceneConfigError("stewart leg nominal length must lie strictly inside its limits")
    _number("stewart.base_joint_z_m", stewart.get("base_joint_z_m"))
    _number("stewart.platen_joint_z_from_center_m", stewart.get("platen_joint_z_from_center_m"))

    table = _require_mapping("table_support", payload["table_support"])
    points = table.get("leg_centers_xy_m")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) != 4:
        raise SceneConfigError("table_support.leg_centers_xy_m must contain four XY points")
    for index, point in enumerate(points):
        _vector(f"table_support.leg_centers_xy_m[{index}]", point, 2)
    for key in (
        "support_plane_z_m",
        "leg_top_clearance_m",
        "frame_half_thickness_m",
        "frame_y_offset_m",
        "frame_x_inset_m",
        "crossbrace_z_local_m",
        "crossbrace_half_length_x_m",
        "crossbrace_half_length_y_m",
        "crossbrace_half_thickness_m",
        "upper_sleeve_radius_m",
        "upper_sleeve_half_height_m",
        "upper_sleeve_center_z_m",
        "lower_sleeve_radius_m",
        "lower_sleeve_half_height_m",
        "lower_sleeve_center_z_m",
    ):
        _number(
            f"table_support.{key}",
            table.get(key),
            minimum=0.0 if "clearance" not in key and "z_local" not in key and key != "frame_y_offset_m" else None,
        )
    for key in ("foot_plate_size_m", "lower_plate_size_m"):
        _vector(f"table_support.{key}", table.get(key), 3, minimum=0.0, strict=True)
    if table.get("layout_variant") not in {"A", "C"} or table["layout_variant"] != payload["geometry_variant"]:
        raise SceneConfigError("table_support.layout_variant must match the supported geometry variant A or C")

    clearance = _require_mapping("clearance", payload["clearance"])
    _number("clearance.candidate_window_m", clearance.get("candidate_window_m"), minimum=0.0, strict=True)
    _number("clearance.assembly_tolerance_m", clearance.get("assembly_tolerance_m"), minimum=0.0, strict=True)
    _number("clearance.required_margin_m", clearance.get("required_margin_m"), minimum=0.0)
    _require_unique_strings("clearance.mount_collision_prefixes", clearance.get("mount_collision_prefixes", ()))

    envelope = _require_mapping("safe_excitation_envelope", payload["safe_excitation_envelope"])
    for key in ("translation_abs_m", "rotation_abs_rad"):
        _vector(f"safe_excitation_envelope.{key}", envelope.get(key), 3, minimum=0.0)
    _number("safe_excitation_envelope.gamma_commanded", envelope.get("gamma_commanded"), minimum=0.0)
    if int(envelope.get("sample_count", 0)) < 1:
        raise SceneConfigError("safe_excitation_envelope.sample_count must be positive")

    render = _require_mapping("render", payload["render"])
    for key in ("default_width", "default_height", "default_fps"):
        value = render.get(key)
        if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
            raise SceneConfigError(f"render.{key} must be a positive integer")
    lights = render.get("lights")
    if not isinstance(lights, Sequence) or not lights:
        raise SceneConfigError("render.lights must be a non-empty list")
    _require_unique_strings("render.lights names", [light.get("name") for light in lights])
    for light in lights:
        if not light["name"].startswith("shakebench_"):
            raise SceneConfigError("render.lights names must start with shakebench_")
        for key in ("pos", "dir"):
            _vector(f"render.lights.{key}", light.get(key), 3)
        if np.linalg.norm(light["dir"]) == 0:
            raise SceneConfigError("render.lights.dir must be non-zero")
        for key in ("ambient", "diffuse", "specular"):
            _rgb(f"render.lights.{key}", light.get(key))
    settings = _require_mapping("render.visual_settings", render.get("visual_settings"))
    if not {"headlight", "quality"} <= set(settings) or set(settings) - {"headlight", "quality", "map"}:
        raise SceneConfigError("render.visual_settings requires headlight/quality and optional map")
    if "map" in settings:
        mapping = _require_mapping("render.visual_settings.map", settings["map"])
        if set(mapping) != {"shadowscale"}:
            raise SceneConfigError("render.visual_settings.map only supports shadowscale")
        _number("render.visual_settings.map.shadowscale", mapping["shadowscale"], minimum=0, strict=True)
    for key in ("ambient", "diffuse", "specular"):
        _rgb(f"render.visual_settings.headlight.{key}", settings["headlight"].get(key))

    cameras = _require_mapping("cameras", payload["cameras"])
    if set(cameras) != {"overview", "assembly", "side"}:
        raise SceneConfigError("cameras must provide overview, assembly, and side")
    camera_names: set[str] = set()
    for key, value in cameras.items():
        camera = _require_mapping(f"cameras.{key}", value)
        name = camera.get("name")
        if not isinstance(name, str) or not name.strip() or name in camera_names:
            raise SceneConfigError("camera names must be unique and non-empty")
        camera_names.add(name)
        _vector(f"cameras.{key}.pos_m", camera.get("pos_m"), 3)
        _vector(f"cameras.{key}.lookat_m", camera.get("lookat_m"), 3)
        _number(f"cameras.{key}.fovy_deg", camera.get("fovy_deg"), minimum=0.0, strict=True)
        if not 1.0 <= float(camera["fovy_deg"]) < 180.0:
            raise SceneConfigError("camera fovy must lie in [1, 180) degrees")

    materials = _require_mapping("materials", payload["materials"])
    for key in ("floor_texture", "wall_texture", "phenolic_texture", "steel_texture", "platen_texture"):
        texture = materials.get(key)
        if (
            not isinstance(texture, str)
            or not texture.strip()
            or Path(texture).is_absolute()
            or ".." in Path(texture).parts
        ):
            raise SceneConfigError(f"materials.{key} must be a package-relative asset path")
        texture_path = Path(models.assets_root) / texture
        if not texture_path.is_file():
            raise SceneConfigError(f"scene texture is missing from the package: {texture_path}")
    for key in (
        "wall_rgb",
        "frame_rgb",
        "rubber_rgb",
        "bolt_rgb",
        "cabinet_rgb",
        "cabinet_light_rgb",
        "emergency_rgb",
        "platen_rgb",
    ):
        _rgb(f"materials.{key}", materials.get(key))

    digest = payload.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SceneConfigError("payload_sha256 must be a lowercase SHA-256 digest")
    actual = scene_config_hash(payload)
    if digest != actual:
        raise SceneConfigError(f"scene payload hash mismatch for {path}: expected {digest}, computed {actual}")


def _validate_laboratory_details(room: Mapping[str, Any], ownership: Mapping[str, Any]) -> None:
    """Reject malformed or physically active authored decoration before MJCF assembly."""

    fixtures = room.get("fixtures")
    equipment = room.get("equipment")
    if not isinstance(fixtures, Sequence) or not isinstance(equipment, Sequence):
        raise SceneConfigError("room.fixtures and room.equipment must be lists")
    names: set[str] = set()
    body_names: set[str] = set()
    records = list(fixtures)
    for item in equipment:
        item = _require_mapping("room.equipment item", item)
        name = item.get("name")
        if name not in ownership["world"]["bodies"] or name in body_names:
            raise SceneConfigError("room.equipment body must be unique and declared in world ownership")
        body_names.add(name)
        _vector("room.equipment.pos_m", item.get("pos_m"), 3)
        _number("room.equipment.yaw_deg", item.get("yaw_deg"))
        geoms = item.get("geoms")
        if not isinstance(geoms, Sequence) or not geoms:
            raise SceneConfigError("room.equipment.geoms must be a non-empty list")
        records.extend(geoms)
    for record in records:
        record = _require_mapping("room visual geom", record)
        if set(record) - {"name", "type", "size", "pos", "material", "rgba", "quat", "bevel_m"}:
            raise SceneConfigError("room visual geom contains unsupported fields")
        name = record.get("name")
        if not isinstance(name, str) or not name.startswith("shakebench_") or name in names:
            raise SceneConfigError("room visual geom names must be unique and start with shakebench_")
        names.add(name)
        size_count = {"box": 3, "cylinder": 2, "capsule": 2, "sphere": 1}.get(record.get("type"))
        if size_count is None:
            raise SceneConfigError("room visual geom type must be box, cylinder, capsule, or sphere")
        _vector("room visual geom size", record.get("size"), size_count, minimum=0.0, strict=True)
        if "bevel_m" in record:
            bevel = _number("room visual geom bevel_m", record["bevel_m"], minimum=0.0, strict=True)
            if record["type"] != "box" or bevel >= min(record["size"]):
                raise SceneConfigError("room visual geom bevel_m must fit strictly within box half sizes")
        _vector("room visual geom pos", record.get("pos"), 3)
        if "rgba" in record:
            _rgba("room visual geom rgba", record["rgba"])
        if "quat" in record:
            quat = _vector("room visual geom quat", record["quat"], 4)
            if np.linalg.norm(quat) == 0:
                raise SceneConfigError("room visual geom quat must be non-zero")


@dataclass(frozen=True)
class SceneVisualConfig:
    """Validated immutable scene visual configuration."""

    payload: Mapping[str, Any]
    source_path: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise SceneConfigError("payload must be a mapping")
        frozen = _freeze(_json_ready(self.payload))
        object.__setattr__(self, "payload", frozen)
        if self.payload_sha256 != self.payload.get("payload_sha256"):
            raise SceneConfigError("SceneVisualConfig payload_sha256 does not match payload")

    @property
    def scene_id(self) -> str:
        return str(self.payload["scene_id"])

    @property
    def schema_id(self) -> str:
        return str(self.payload["schema_id"])

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def geometry_variant(self) -> str:
        return str(self.payload["geometry_variant"])

    @property
    def physics_effect(self) -> bool:
        return bool(self.payload["physics_effect"])

    @property
    def config_sha256(self) -> str:
        return self.payload_sha256

    def section(self, name: str) -> Mapping[str, Any]:
        return _require_mapping(name, self.payload[name])

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]


def load_scene_visual_config(path: str | Path | None = None) -> SceneVisualConfig:
    """Load and authenticate the package-owned scene visual configuration.

    Args:
        path: Package-relative or absolute local path.  ``None`` selects the
            flat asset shipped with robosuite.

    Returns:
        A validated immutable :class:`SceneVisualConfig`.
    """

    if path is None:
        config_path = SCENE_CONFIG_PATH
    else:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = Path(models.assets_root) / config_path
    if not config_path.is_file():
        raise SceneConfigError(f"scene visual config is missing: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneConfigError(f"cannot read scene visual config {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SceneConfigError("scene visual config root must be an object")
    _validate_scene_payload(payload, config_path)
    return SceneVisualConfig(
        payload=payload, source_path=str(config_path), payload_sha256=str(payload["payload_sha256"])
    )


def _coerce_config(config: SceneVisualConfig | Mapping[str, Any] | str | Path | None) -> SceneVisualConfig:
    if isinstance(config, SceneVisualConfig):
        return config
    if config is None or isinstance(config, (str, Path)):
        return load_scene_visual_config(config)
    if isinstance(config, Mapping):
        payload = dict(config)
        _validate_scene_payload(payload, Path("<mapping>"))
        return SceneVisualConfig(
            payload=payload, source_path="<mapping>", payload_sha256=str(payload["payload_sha256"])
        )
    raise SceneConfigError("config must be a SceneVisualConfig, mapping, path, or None")


def _fmt(values: Iterable[float]) -> str:
    return array_to_string(tuple(float(value) for value in values))


def _geom_name_set(root: ET.Element) -> set[str]:
    return {str(name) for element in root.iter("geom") if (name := element.get("name")) is not None}


def _body_name_set(root: ET.Element) -> set[str]:
    return {str(name) for element in root.iter("body") if (name := element.get("name")) is not None}


def _visual_geom(
    name: str,
    geom_type: str,
    size: Iterable[float],
    pos: Iterable[float] = (0.0, 0.0, 0.0),
    *,
    material: str | None = None,
    rgba: Iterable[float] | None = None,
    quat: Iterable[float] | None = None,
) -> ET.Element:
    kwargs: dict[str, Any] = {
        "contype": 0,
        "conaffinity": 0,
        "density": 0.0,
    }
    if material is not None:
        kwargs["material"] = material
    if rgba is not None:
        kwargs["rgba"] = tuple(float(value) for value in rgba)
    if quat is not None:
        kwargs["quat"] = tuple(float(value) for value in quat)
    return new_geom(name=name, type=geom_type, size=tuple(size), pos=tuple(pos), group=1, **kwargs)


def _visual_body(name: str, pos: Iterable[float] = (0.0, 0.0, 0.0)) -> ET.Element:
    return new_body(name=name, pos=tuple(float(value) for value in pos))


def _quat_from_z_axis(direction: Iterable[float]) -> tuple[float, float, float, float]:
    vector = np.asarray(tuple(direction), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise SceneConfigError("visual cylinder direction must be non-zero")
    vector /= norm
    z_axis = np.array((0.0, 0.0, 1.0))
    cross = np.cross(z_axis, vector)
    scalar = 1.0 + float(vector[2])
    quaternion = np.concatenate((cross, (scalar,)))
    q_norm = float(np.linalg.norm(quaternion))
    if q_norm <= 1.0e-12:
        # The only singular case is a 180-degree flip from +Z to -Z.
        quaternion = np.array((1.0, 0.0, 0.0, 0.0))
    else:
        quaternion /= q_norm
    return tuple(float(value) for value in quaternion[[3, 0, 1, 2]])


def _look_at_quat(position: Iterable[float], target: Iterable[float]) -> tuple[float, float, float, float]:
    position = np.asarray(tuple(position), dtype=float)
    target = np.asarray(tuple(target), dtype=float)
    forward = target - position
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-12:
        raise SceneConfigError("camera position and lookat must differ")
    forward /= norm
    right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-12:
        right = np.array((1.0, 0.0, 0.0))
    else:
        right /= right_norm
    camera_z = -forward
    camera_y = np.cross(camera_z, right)
    rotation = np.column_stack((right, camera_y, camera_z))
    xyzw = T.mat2quat(rotation)
    return tuple(float(value) for value in xyzw[[3, 0, 1, 2]])


def _stewart_points(config: SceneVisualConfig, platen_center_z: float) -> tuple[np.ndarray, np.ndarray]:
    stewart = config.section("stewart")
    spread = math.radians(float(stewart["joint_pair_spread_deg"]))
    base_axes = np.asarray(stewart["base_ellipse_semi_axes_m"], dtype=float)
    platen_axes = np.asarray(stewart["platen_ellipse_semi_axes_m"], dtype=float)
    base_angles = []
    platen_angles = []
    for group in range(3):
        centre = group * 2.0 * math.pi / 3.0
        base_angles.extend((centre - spread / 2.0, centre + spread / 2.0))
        centre += math.pi / 3.0
        platen_angles.extend((centre - spread / 2.0, centre + spread / 2.0))
    base = np.column_stack(
        (
            base_axes[0] * np.cos(base_angles),
            base_axes[1] * np.sin(base_angles),
            np.full(6, float(stewart["base_joint_z_m"])),
        )
    )
    platen = np.column_stack(
        (
            platen_axes[0] * np.cos(platen_angles),
            platen_axes[1] * np.sin(platen_angles),
            np.full(6, float(platen_center_z) + float(stewart["platen_joint_z_from_center_m"])),
        )
    )
    return base, platen


def _table_support_geometry(arena: Any, config: SceneVisualConfig) -> dict[str, Any]:
    table = config.section("table_support")
    table_world_z = float(np.asarray(arena.center_pos, dtype=float)[2])
    table_half = np.asarray(arena.table_half_size, dtype=float)
    support_z = float(table["support_plane_z_m"])
    top_world_z = table_world_z - float(table["leg_top_clearance_m"]) - table_half[2]
    points = [tuple(float(value) for value in point) for point in table["leg_centers_xy_m"]]
    leg_bottom_world_z = support_z + float(table["foot_plate_size_m"][-1]) / 2.0
    leg_half_height = (top_world_z - leg_bottom_world_z) / 2.0
    if leg_half_height <= 0.0:
        raise SceneConfigError("table upper legs have no positive height")
    return {
        "table_world_z": table_world_z,
        "support_z": support_z,
        "top_world_z": top_world_z,
        "leg_centers_xy": points,
        "leg_center_world_z": (top_world_z + leg_bottom_world_z) / 2.0,
        "leg_half_height": leg_half_height,
        "table_half": table_half,
    }


def _append_cylinder_between(
    parent: ET.Element,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    *,
    material: str,
) -> None:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        raise SceneConfigError(f"visual cylinder {name!r} has zero length")
    centre = 0.5 * (start + end)
    parent.append(
        _visual_geom(
            name,
            "cylinder",
            (radius, length / 2.0),
            centre,
            material=material,
            quat=_quat_from_z_axis(vector),
        )
    )


def _add_camera(worldbody: ET.Element, camera: Mapping[str, Any]) -> None:
    worldbody.append(
        new_element(
            "camera",
            camera["name"],
            mode="fixed",
            pos=_fmt(camera["pos_m"]),
            quat=_fmt(_look_at_quat(camera["pos_m"], camera["lookat_m"])),
            fovy=float(camera["fovy_deg"]),
        )
    )


def _append_unique_body(worldbody: ET.Element, body: ET.Element) -> None:
    names = _body_name_set(worldbody)
    if body.get("name") in names:
        raise SceneConfigError(f"scene augmentation would duplicate body {body.get('name')!r}")
    worldbody.append(body)


def _append_unique_geom(parent: ET.Element, geom: ET.Element, names: set[str]) -> None:
    name = geom.get("name")
    if name is None or name in names:
        raise SceneConfigError(f"scene augmentation would duplicate geom {name!r}")
    names.add(name)
    parent.append(geom)


def configure_scene_rendering(
    root: ET.Element, config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None
) -> None:
    """Apply the authority's native MuJoCo offscreen framebuffer settings."""

    scene_config = _coerce_config(config)
    render = scene_config.section("render")
    visual = root.find("./visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("./global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(int(render["default_width"])))
    global_visual.set("offheight", str(int(render["default_height"])))
    for tag, attributes in render["visual_settings"].items():
        element = visual.find(f"./{tag}")
        if element is None:
            element = ET.SubElement(visual, tag)
        for key, value in attributes.items():
            element.set(key, _fmt(value) if isinstance(value, (list, tuple)) else str(value))


def _configure_laboratory_lights(arena: Any, config: SceneVisualConfig) -> None:
    """Use neutral overhead and oblique fill lighting in every render path."""

    for light in list(arena.worldbody.findall("./light")):
        if light.get("name", "").startswith("shakebench_"):
            arena.worldbody.remove(light)
    for light in config.section("render")["lights"]:
        ET.SubElement(
            arena.worldbody,
            "light",
            {key: _fmt(value) if isinstance(value, (list, tuple)) else str(value) for key, value in light.items()},
        )


def _configure_materials(arena: Any, config: SceneVisualConfig) -> None:
    """Apply package-relative texture and color choices from the authority."""

    materials = config.section("materials")
    texture_names = {
        "shakebench_texplane": "floor_texture",
        "shakebench_phenolic": "phenolic_texture",
        "shakebench_steel": "steel_texture",
        "shakebench_platen_tex": "platen_texture",
        "shakebench_wall_tex": "wall_texture",
    }
    for texture_name, config_key in texture_names.items():
        texture = arena.asset.find(f"./texture[@name='{texture_name}']")
        if texture is None:
            texture = ET.SubElement(arena.asset, "texture", name=texture_name, type="2d")
        texture.set("file", str((Path(models.assets_root) / materials[config_key]).resolve()))
    material_colors = {
        "shakebench_wall": "wall_rgb",
        "shakebench_foundation": "frame_rgb",
        "shakebench_actuator_outer": "frame_rgb",
        "shakebench_actuator_rod": "platen_rgb",
        "shakebench_joint": "frame_rgb",
        "shakebench_mount_rubber": "rubber_rgb",
        "shakebench_bolt_metal": "bolt_rgb",
        "shakebench_platen_edge": "frame_rgb",
        "shakebench_cabinet": "cabinet_rgb",
        "shakebench_cabinet_light": "cabinet_light_rgb",
        "shakebench_emergency": "emergency_rgb",
    }
    for material_name, config_key in material_colors.items():
        material = arena.asset.find(f"./material[@name='{material_name}']")
        if material is not None:
            material.set("rgba", _fmt((*materials[config_key], 1.0)))
    platen_material = arena.asset.find("./material[@name='shakebench_platen_surface']")
    if platen_material is not None:
        platen_material.set("rgba", _fmt((*materials["platen_rgb"], 1.0)))
    for material_name, settings in materials["surface_settings"].items():
        material = arena.asset.find(f"./material[@name='{material_name}']")
        if material is None:
            material = ET.SubElement(arena.asset, "material", name=material_name)
        for key, value in settings.items():
            material.set(key, _fmt(value) if isinstance(value, (list, tuple)) else str(value))
    configure_scene_rendering(arena.root, config)
    _configure_laboratory_lights(arena, config)


def _append_authored_visuals(arena: Any, parent: ET.Element, records: Sequence, geom_names: set[str]) -> list[str]:
    """Compile config-owned, non-contact equipment and room details."""

    names = []
    for record in records:
        geom = _visual_geom(
            record["name"],
            record["type"],
            record["size"],
            record["pos"],
            material=record.get("material"),
            rgba=record.get("rgba"),
            quat=record.get("quat"),
        )
        if "bevel_m" in record:
            # A true chamfered shell: 24 vertices, flat face normals, no mesh
            # download or hidden collision proxy. All geometry is still massless.
            half = np.asarray(record["size"], dtype=float)
            inset = half - float(record["bevel_m"])
            vertices = []
            for signs in itertools.product((-1, 1), repeat=3):
                for axis in range(3):
                    vertex = inset.copy()
                    vertex[axis] = half[axis]
                    vertices.append(vertex * signs)
            vertices = np.asarray(vertices)
            faces = ConvexHull(vertices).simplices.copy()
            for face in faces:
                a, b, c = vertices[face]
                if np.dot(np.cross(b - a, c - a), a) < 0:
                    face[1], face[2] = face[2], face[1]
            mesh_name = record["name"] + "_chamfer_mesh"
            ET.SubElement(
                arena.asset,
                "mesh",
                name=mesh_name,
                vertex=_fmt(vertices.ravel()),
                face=" ".join(str(int(index)) for index in faces.ravel()),
                smoothnormal="false",
            )
            geom.set("type", "mesh")
            geom.set("mesh", mesh_name)
            geom.attrib.pop("size", None)
        _append_unique_geom(parent, geom, geom_names)
        names.append(record["name"])
    return names


def _get_or_append_body(worldbody: ET.Element, name: str, pos: Iterable[float] = (0.0, 0.0, 0.0)) -> ET.Element:
    """Return an explicit source body or add a new named visual body."""

    existing = worldbody.find(f"./body[@name='{name}']")
    if existing is not None:
        return existing
    body = _visual_body(name, pos)
    worldbody.append(body)
    return body


def _build_detailed_guardrails(arena, body, pit, geom_names):
    """Continuous tubes, true curved elbows, post sockets and grounded anchors."""
    details = pit["guardrail_details"]
    radius = float(pit["guardrail_thickness_m"]) / 2
    bend = float(details["bend_radius_m"])
    height = float(pit["guardrail_height_m"])
    rx, ry = np.asarray(pit["opening_xy_m"], dtype=float) / 2 + float(pit["border_width_m"])
    yellow = (*pit["safety_rgb"], 1.0)
    black = details["black_rgba"]
    coupling = details["coupler_rgba"]
    names = []

    def emit(name, kind, size, pos, rgba, quat=None):
        name = "shakebench_guardrail_" + name
        geom = _visual_geom(name, kind, size, pos, rgba=rgba, quat=quat)
        _append_unique_geom(body, geom, geom_names)
        names.append(name)
        return geom

    def tube(name, a, b, r, rgba):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        delta = b - a
        return emit(name, "cylinder", (r, np.linalg.norm(delta) / 2), (a + b) / 2, rgba, _quat_from_z_axis(delta))

    def bands(name, a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        pitch = float(details["stripe_pitch_m"])
        # The structural tube is continuous under the thin black paint bands.
        for i, start in enumerate(np.arange(pitch * 0.5, length, pitch)):
            end = min(start + pitch * float(details["black_fraction"]), length)
            tube(f"{name}_band_{i}", a + start * direction, a + end * direction, radius + 0.00015, black)

    runs = [
        ((-rx + bend, -ry, height), (rx - bend, -ry, height)),
        ((-rx + bend, ry, height), (rx - bend, ry, height)),
        ((-rx, -ry + bend, height), (-rx, ry - bend, height)),
        ((rx, -ry + bend, height), (rx, ry - bend, height)),
    ]
    sleeve_half = float(details["coupler_length_m"]) / 2
    for i, (a, b) in enumerate(runs):
        tube(f"top_{i}", a, b, radius, yellow)
        bands(f"top_{i}", a, b)
        direction = (np.asarray(b) - a) / np.linalg.norm(np.asarray(b) - a)
        for end, point in enumerate((np.asarray(a), np.asarray(b))):
            tube(
                f"end_socket_{i}_{end}",
                point - sleeve_half * direction,
                point + sleeve_half * direction,
                float(details["coupler_radius_m"]),
                coupling,
            )
            emit(
                f"end_socket_screw_{i}_{end}",
                "cylinder",
                (0.0035, 0.002),
                point + np.array([0, 0, float(details["coupler_radius_m"])]),
                (0.45, 0.48, 0.5, 1),
            )

    # Rounded corners meet the straight centre lines exactly at their endpoints.
    corners = [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]
    arc_steps, ring_steps = int(details["elbow_arc_segments"]), int(details["elbow_ring_segments"])
    for index, (sx, sy, start_angle) in enumerate(corners):
        centre = np.array([sx * (rx - bend), sy * (ry - bend), height])
        vertices = []
        for angle in np.linspace(math.radians(start_angle), math.radians(start_angle + 90), arc_steps + 1):
            radial = np.array([math.cos(angle), math.sin(angle), 0])
            axis_centre = centre + bend * radial
            for phase in np.arange(ring_steps) * 2 * math.pi / ring_steps:
                vertices.append(axis_centre + radius * (math.cos(phase) * radial + np.array([0, 0, math.sin(phase)])))
        faces = []
        for ring in range(arc_steps):
            for k in range(ring_steps):
                a = ring * ring_steps + k
                b = ring * ring_steps + (k + 1) % ring_steps
                faces.extend(((a, a + ring_steps, b), (b, a + ring_steps, b + ring_steps)))
        mesh_name = f"shakebench_guardrail_elbow_mesh_{index}"
        ET.SubElement(
            arena.asset,
            "mesh",
            name=mesh_name,
            vertex=_fmt(np.asarray(vertices).ravel()),
            face=" ".join(str(v) for face in faces for v in face),
            smoothnormal="true",
        )
        geom = emit(f"elbow_{index}", "mesh", (1, 1, 1), (0, 0, 0), black)
        geom.set("mesh", mesh_name)
        geom.attrib.pop("size")

    corner_offset = bend * (1 - 1 / math.sqrt(2))
    posts = [
        (-rx + corner_offset, -ry + corner_offset),
        (-rx, 0),
        (-rx + corner_offset, ry - corner_offset),
        (rx - corner_offset, -ry + corner_offset),
        (rx, 0),
        (rx - corner_offset, ry - corner_offset),
        (0, -ry),
        (0, ry),
    ]
    plate = np.asarray(details["foot_plate_size_m"], dtype=float)
    for i, (x, y) in enumerate(posts):
        emit(f"foot_plate_{i}", "box", plate / 2, (x, y, plate[2] / 2), coupling)
        tube(f"post_{i}", (x, y, plate[2]), (x, y, height), radius, yellow)
        bands(f"post_{i}", (x, y, plate[2]), (x, y, height - 0.035))
        emit(f"foot_socket_{i}", "cylinder", (radius * 1.55, 0.015), (x, y, plate[2] + 0.015), black)
        emit(
            f"post_socket_{i}",
            "cylinder",
            (float(details["coupler_radius_m"]), 0.022),
            (x, y, height - 0.012),
            coupling,
        )
        for bolt, (dx, dy) in enumerate(itertools.product((-1, 1), repeat=2)):
            offset = float(details["anchor_xy_offset_m"])
            bolt_height = float(details["anchor_height_m"])
            pos = np.array([x + dx * offset, y + dy * offset, plate[2] + bolt_height / 2])
            emit(
                f"anchor_{i}_{bolt}",
                "cylinder",
                (float(details["anchor_radius_m"]), bolt_height / 2),
                pos,
                (0.45, 0.48, 0.50, 1),
            )
            emit(
                f"anchor_slot_{i}_{bolt}",
                "box",
                (0.0035, 0.0008, 0.0003),
                pos + np.array([0, 0, bolt_height / 2]),
                black,
            )
        # Seam clamps make the T junction on each straight span explicit.
        if i in (1, 4, 6, 7):
            direction = np.array([0, 1, 0]) if i in (1, 4) else np.array([1, 0, 0])
            centre = np.array([x, y, height])
            tube(
                f"tee_socket_{i}",
                centre - 0.027 * direction,
                centre + 0.027 * direction,
                float(details["coupler_radius_m"]),
                coupling,
            )
    return names


def _build_scene_visuals(
    arena: Any, config: SceneVisualConfig
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Append the configured scene and return body/geom inventory primitives."""

    worldbody = arena.worldbody
    geom_names = _geom_name_set(arena.root)
    materials = config.section("materials")
    room = config.section("room")
    pit = config.section("pit")
    platen = config.section("platen")
    table = config.section("table_support")
    _configure_materials(arena, config)
    envelope = _table_support_geometry(arena, config)
    frame_rgb = _rgba("materials.frame_rgba", (*materials["frame_rgb"], 1.0))
    rubber_rgb = _rgba("materials.rubber_rgba", (*materials["rubber_rgb"], 1.0))
    bolt_rgb = _rgba("materials.bolt_rgba", (*materials["bolt_rgb"], 1.0))

    visual_geoms: list[str] = [TABLE_VISUAL_GEOM_NAME]
    visual_bodies: list[str] = []
    frame_bodies: dict[str, str] = {}

    # The physical plane remains in the model and keeps its contact semantics;
    # its alpha is zero so the four display-only slabs can show a pit opening.
    physical_floor = worldbody.find("./geom[@name='floor']")
    if physical_floor is not None:
        physical_floor.set("rgba", "1 1 1 0")
        physical_floor.set("group", "0")

    room_body = _visual_body("shakebench_laboratory_shell")
    visual_bodies.append(room_body.get("name"))
    frame_bodies[room_body.get("name")] = "world"
    room_x, room_y, room_z = (float(value) / 2.0 for value in room["envelope_m"])
    wall_half = float(room["wall_thickness_m"]) / 2.0
    floor_half_z = float(room["floor_slab_thickness_m"]) / 2.0
    slab_z = -floor_half_z
    pit_half_x, pit_half_y = (float(value) / 2.0 for value in pit["opening_xy_m"])
    # Outside-pit slabs are visual only and stop at the opening edges.
    slabs = (
        ("shakebench_floor_slab_left", (room_x - pit_half_x) / 2.0, room_y, (-room_x - pit_half_x) / 2.0, slab_z),
        ("shakebench_floor_slab_right", (room_x - pit_half_x) / 2.0, room_y, (room_x + pit_half_x) / 2.0, slab_z),
        (
            "shakebench_floor_slab_front",
            pit_half_x,
            (room_y - pit_half_y) / 2.0,
            0.0,
            (-room_y - pit_half_y) / 2.0,
            slab_z,
        ),
        (
            "shakebench_floor_slab_back",
            pit_half_x,
            (room_y - pit_half_y) / 2.0,
            0.0,
            (room_y + pit_half_y) / 2.0,
            slab_z,
        ),
    )
    # The first two records have (half_x, half_y, x, z); the latter two have
    # (half_x, half_y, x, y, z).  Keeping the expansion local avoids a second
    # geometry registry just for these four room slabs.
    for record in slabs:
        if len(record) == 5:
            name, half_x, half_y, x, z = record
            pos = (x, 0.0, z)
        else:
            name, half_x, half_y, x, y, z = record
            pos = (x, y, z)
        _append_unique_geom(
            room_body,
            _visual_geom(name, "box", (half_x, half_y, floor_half_z), pos, material="shakebench_floor_slab"),
            geom_names,
        )
        visual_geoms.append(name)
    walls = (
        ("shakebench_wall_left", (wall_half, room_y, room_z), (-room_x + wall_half, 0.0, room_z), "shakebench_wall"),
        ("shakebench_wall_right", (wall_half, room_y, room_z), (room_x - wall_half, 0.0, room_z), "shakebench_wall"),
        ("shakebench_wall_back", (room_x, wall_half, room_z), (0.0, room_y - wall_half, room_z), "shakebench_wall"),
    )
    for name, size, pos, material in walls:
        _append_unique_geom(room_body, _visual_geom(name, "box", size, pos, material=material), geom_names)
        visual_geoms.append(name)
    visual_geoms.extend(_append_authored_visuals(arena, room_body, room["fixtures"], geom_names))
    _append_unique_body(worldbody, room_body)

    pit_body = _visual_body(PIT_BODY_NAME)
    visual_bodies.append(PIT_BODY_NAME)
    frame_bodies[PIT_BODY_NAME] = "world"
    depth = float(pit["depth_m"])
    bottom_z = -depth + float(room["floor_slab_thickness_m"])
    _append_unique_geom(
        pit_body,
        _visual_geom(
            "shakebench_pit_floor",
            "box",
            (pit_half_x, pit_half_y, floor_half_z),
            (0.0, 0.0, bottom_z),
            rgba=(*pit["dark_rgb"], 1.0),
        ),
        geom_names,
    )
    visual_geoms.append("shakebench_pit_floor")
    pit_wall_half = float(pit["border_width_m"]) / 2.0
    pit_wall_height = depth / 2.0
    for name, size, pos in (
        ("shakebench_pit_wall_xneg", (pit_wall_half, pit_half_y, pit_wall_height), (-pit_half_x, 0.0, -depth / 2.0)),
        ("shakebench_pit_wall_xpos", (pit_wall_half, pit_half_y, pit_wall_height), (pit_half_x, 0.0, -depth / 2.0)),
        ("shakebench_pit_wall_yneg", (pit_half_x, pit_wall_half, pit_wall_height), (0.0, -pit_half_y, -depth / 2.0)),
        ("shakebench_pit_wall_ypos", (pit_half_x, pit_wall_half, pit_wall_height), (0.0, pit_half_y, -depth / 2.0)),
    ):
        _append_unique_geom(pit_body, _visual_geom(name, "box", size, pos, rgba=(*pit["dark_rgb"], 1.0)), geom_names)
        visual_geoms.append(name)
    border = float(pit["border_width_m"])
    for name, size, pos in (
        (
            "shakebench_pit_border_xneg",
            (border / 2.0, pit_half_y + border, floor_half_z),
            (-pit_half_x - border / 2.0, 0.0, 0.0),
        ),
        (
            "shakebench_pit_border_xpos",
            (border / 2.0, pit_half_y + border, floor_half_z),
            (pit_half_x + border / 2.0, 0.0, 0.0),
        ),
        (
            "shakebench_pit_border_yneg",
            (pit_half_x, border / 2.0, floor_half_z),
            (0.0, -pit_half_y - border / 2.0, 0.0),
        ),
        ("shakebench_pit_border_ypos", (pit_half_x, border / 2.0, floor_half_z), (0.0, pit_half_y + border / 2.0, 0.0)),
    ):
        _append_unique_geom(pit_body, _visual_geom(name, "box", size, pos, rgba=(*pit["border_rgb"], 1.0)), geom_names)
        visual_geoms.append(name)
    _append_unique_body(worldbody, pit_body)

    guardrail_body = _visual_body(GUARDRAIL_BODY_NAME)
    visual_bodies.append(GUARDRAIL_BODY_NAME)
    frame_bodies[GUARDRAIL_BODY_NAME] = "world"
    if pit.get("guardrail_details"):
        visual_geoms.extend(_build_detailed_guardrails(arena, guardrail_body, pit, geom_names))
    else:
        rail_x = pit_half_x + border
        rail_y = pit_half_y + border
        rail_height = float(pit["guardrail_height_m"])
        rail_half_t = float(pit["guardrail_thickness_m"]) / 2.0
        post_radius = rail_half_t
        post_points = [
            (-rail_x, -rail_y),
            (-rail_x, 0.0),
            (-rail_x, rail_y),
            (rail_x, -rail_y),
            (rail_x, 0.0),
            (rail_x, rail_y),
            (0.0, -rail_y),
            (0.0, rail_y),
        ]
        for index, (x, y) in enumerate(post_points):
            name = f"shakebench_guardrail_post_{index}"
            _append_unique_geom(
                guardrail_body,
                _visual_geom(
                    name,
                    "cylinder",
                    (post_radius, rail_height / 2.0),
                    (x, y, rail_height / 2.0),
                    rgba=(*pit["safety_rgb"], 1.0),
                ),
                geom_names,
            )
            visual_geoms.append(name)
        rail_z = rail_height
        for index, (size, pos) in enumerate(
            (
                ((rail_x, rail_half_t, rail_half_t), (0.0, -rail_y, rail_z)),
                ((rail_x, rail_half_t, rail_half_t), (0.0, rail_y, rail_z)),
                ((rail_half_t, rail_y, rail_half_t), (-rail_x, 0.0, rail_z)),
                ((rail_half_t, rail_y, rail_half_t), (rail_x, 0.0, rail_z)),
            )
        ):
            name = f"shakebench_guardrail_top_{index}"
            _append_unique_geom(
                guardrail_body, _visual_geom(name, "box", size, pos, rgba=(*pit["safety_rgb"], 1.0)), geom_names
            )
            visual_geoms.append(name)
    _append_unique_body(worldbody, guardrail_body)

    foundation_body = _visual_body(FOUNDATION_BODY_NAME)
    visual_bodies.append(FOUNDATION_BODY_NAME)
    frame_bodies[FOUNDATION_BODY_NAME] = "world"
    stewart = config.section("stewart")
    foundation_half = np.asarray(stewart["foundation_size_m"], dtype=float) / 2.0
    _append_unique_geom(
        foundation_body,
        _visual_geom(
            "shakebench_shaker_foundation_inertia",
            "box",
            foundation_half,
            (0.0, 0.0, float(stewart["foundation_center_z_m"])),
            material="shakebench_foundation",
        ),
        geom_names,
    )
    visual_geoms.append("shakebench_shaker_foundation_inertia")
    _append_unique_geom(
        foundation_body,
        _visual_geom(
            "shakebench_shaker_foundation_base",
            "cylinder",
            (float(stewart["base_plate_radius_m"]), float(stewart["base_plate_height_m"]) / 2.0),
            (0.0, 0.0, float(stewart["base_joint_z_m"])),
            material="shakebench_foundation",
        ),
        geom_names,
    )
    visual_geoms.append("shakebench_shaker_foundation_base")
    platen_center_z = float(platen["nominal_top_z_m"]) - float(platen["size_m"][-1]) / 2.0
    base_points, platen_points = _stewart_points(config, platen_center_z)
    stewart_min = float(stewart["leg_min_m"])
    stewart_max = float(stewart["leg_max_m"])
    lengths = np.linalg.norm(platen_points - base_points, axis=1)
    if np.any(lengths < stewart_min) or np.any(lengths > stewart_max):
        raise SceneConfigError(f"nominal Stewart lengths are outside configured limits: {lengths.tolist()}")
    outer_length = float(stewart["outer_length_m"])
    rod_length = float(stewart["rod_length_m"])
    for index, point in enumerate(base_points):
        flange_name = f"shakebench_shaker_foundation_flange_{index}"
        _append_unique_geom(
            foundation_body,
            _visual_geom(
                flange_name,
                "cylinder",
                (float(stewart["joint_radius_m"]) * 1.3, 0.0175),
                point + (0.0, 0.0, 0.018),
                material="shakebench_bolt_metal",
            ),
            geom_names,
        )
        visual_geoms.append(flange_name)
        direction = (platen_points[index] - point) / lengths[index]
        outer_end = point + outer_length * direction
        outer_name = f"shakebench_stewart_outer_{index}"
        _append_cylinder_between(
            foundation_body,
            outer_name,
            point,
            outer_end,
            float(stewart["outer_radius_m"]),
            material="shakebench_actuator_outer",
        )
        visual_geoms.append(outer_name)
        joint_name = f"shakebench_stewart_outer_joint_{index}"
        _append_unique_geom(
            foundation_body,
            _visual_geom(
                joint_name, "sphere", (float(stewart["joint_radius_m"]),), outer_end, material="shakebench_joint"
            ),
            geom_names,
        )
        visual_geoms.append(joint_name)
        rod_start = platen_points[index] - rod_length * direction
        rod_name = f"shakebench_stewart_rod_{index}"
        # Upper rods are appended to the dynamic deck visual below, after the
        # source-frame world position has been converted to deck-local values.
        del rod_start, rod_name
    for index, (lhs, rhs) in enumerate(((0, 3), (1, 4), (2, 5))):
        start = base_points[lhs].copy()
        end = base_points[rhs].copy()
        start[2] += 0.065
        end[2] += 0.065
        brace_name = f"shakebench_shaker_foundation_crossbrace_{index}"
        _append_cylinder_between(foundation_body, brace_name, start, end, 0.022, material="shakebench_foundation")
        visual_geoms.append(brace_name)
    _append_unique_body(worldbody, foundation_body)

    # The platen and all deck-side lower mounts are one explicit role body.
    # Reparenting this body by the existing deck processor makes every visual
    # follow the ordinary dynamic deck without creating a kinematic parent.
    platen_body = _get_or_append_body(worldbody, DECK_VISUAL_BODY_NAME, (0.0, 0.0, platen_center_z))
    platen_body.set("pos", _fmt((0.0, 0.0, platen_center_z)))
    visual_bodies.append(DECK_VISUAL_BODY_NAME)
    frame_bodies[DECK_VISUAL_BODY_NAME] = "dynamic_deck"
    platen_half = np.asarray(platen["size_m"], dtype=float) / 2.0
    for name, size, pos, material in (
        ("shakebench_platen_surface", platen_half, (0.0, 0.0, 0.0), "shakebench_platen_surface"),
        (
            "shakebench_platen_edge",
            (platen_half[0], platen_half[1], 0.008),
            (0.0, 0.0, -platen_half[2] - 0.008),
            "shakebench_platen_edge",
        ),
    ):
        _append_unique_geom(
            platen_body, _visual_geom(name, "box", size, pos, material=material, rgba=platen["rgba"]), geom_names
        )
        visual_geoms.append(name)
    # Small top fasteners provide the visual cue of a laboratory shaker plate.
    for index, (x, y) in enumerate(((-0.65, -0.42), (-0.65, 0.42), (0.65, -0.42), (0.65, 0.42))):
        name = f"shakebench_platen_bolt_{index}"
        _append_unique_geom(
            platen_body,
            _visual_geom(
                name,
                "cylinder",
                (0.012, 0.003),
                (x, y, platen_half[2] + 0.003),
                material="shakebench_bolt_metal",
                rgba=bolt_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(name)
    lower_plate_size = np.asarray(table["lower_plate_size_m"], dtype=float)
    upper_plate_size = np.asarray(table["foot_plate_size_m"], dtype=float)
    for index, (x, y) in enumerate(envelope["leg_centers_xy"]):
        # Deck-side plates use world XY; upper supports remain worktable-local.
        x += float(arena.center_pos[0])
        y += float(arena.center_pos[1])
        lower_plate_name = f"shakebench_table_lower_mount_plate_{index}"
        lower_plate_z = float(table["support_plane_z_m"]) + lower_plate_size[2] / 2.0 - platen_center_z
        _append_unique_geom(
            platen_body,
            _visual_geom(
                lower_plate_name,
                "box",
                lower_plate_size / 2.0,
                (x, y, lower_plate_z),
                material="shakebench_frame_metal",
            ),
            geom_names,
        )
        visual_geoms.append(lower_plate_name)
        lower_sleeve_name = f"shakebench_table_lower_mount_sleeve_{index}"
        lower_sleeve_z = float(table["lower_sleeve_center_z_m"]) - platen_center_z
        _append_unique_geom(
            platen_body,
            _visual_geom(
                lower_sleeve_name,
                "cylinder",
                (float(table["lower_sleeve_radius_m"]), float(table["lower_sleeve_half_height_m"])),
                (x, y, lower_sleeve_z),
                material="shakebench_mount_rubber",
                rgba=rubber_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(lower_sleeve_name)
    for index, (point, platen_point) in enumerate(zip(base_points, platen_points)):
        direction = (platen_point - point) / lengths[index]
        rod_start = platen_point - rod_length * direction
        rod_name = f"shakebench_stewart_rod_{index}"
        rod_world_centre = 0.5 * (rod_start + platen_point)
        _append_unique_geom(
            platen_body,
            _visual_geom(
                rod_name,
                "cylinder",
                (float(stewart["rod_radius_m"]), rod_length / 2.0),
                rod_world_centre - np.array((0.0, 0.0, platen_center_z)),
                material="shakebench_actuator_rod",
                quat=_quat_from_z_axis(direction),
            ),
            geom_names,
        )
        visual_geoms.append(rod_name)
        joint_name = f"shakebench_stewart_rod_joint_{index}"
        _append_unique_geom(
            platen_body,
            _visual_geom(
                joint_name,
                "sphere",
                (float(stewart["joint_radius_m"]),),
                platen_point - np.array((0.0, 0.0, platen_center_z)),
                material="shakebench_joint",
            ),
            geom_names,
        )
        visual_geoms.append(joint_name)
    # The isolated worktable owns only its upper frame, legs and the upper
    # half of the sleeves.  No geom spans from this body into the deck body.
    frame_half = float(table["frame_half_thickness_m"])
    frame_y = float(table["frame_y_offset_m"])
    frame_specs = (
        (
            "shakebench_table_upper_frame_front",
            (envelope["table_half"][0], frame_half, frame_half),
            (0.0, -envelope["table_half"][1] - frame_y, -0.055),
        ),
        (
            "shakebench_table_upper_frame_back",
            (envelope["table_half"][0], frame_half, frame_half),
            (0.0, envelope["table_half"][1] + frame_y, -0.055),
        ),
        (
            "shakebench_table_upper_frame_left",
            (frame_half, envelope["table_half"][1] - frame_y, frame_half),
            (-envelope["table_half"][0] + float(table["frame_x_inset_m"]), 0.0, -0.055),
        ),
        (
            "shakebench_table_upper_frame_right",
            (frame_half, envelope["table_half"][1] - frame_y, frame_half),
            (envelope["table_half"][0] - float(table["frame_x_inset_m"]), 0.0, -0.055),
        ),
    )
    for name, size, pos in frame_specs:
        _append_unique_geom(
            arena.table_body,
            _visual_geom(name, "box", size, pos, material="shakebench_frame_metal", rgba=frame_rgb),
            geom_names,
        )
        visual_geoms.append(name)
    crossbrace_z = float(table["crossbrace_z_local_m"])
    for name, size, pos in (
        (
            "shakebench_table_upper_crossbrace_front",
            (
                float(table["crossbrace_half_length_x_m"]),
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (0.0, -float(table["crossbrace_half_length_y_m"]), crossbrace_z),
        ),
        (
            "shakebench_table_upper_crossbrace_back",
            (
                float(table["crossbrace_half_length_x_m"]),
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (0.0, float(table["crossbrace_half_length_y_m"]), crossbrace_z),
        ),
        (
            "shakebench_table_upper_crossbrace_left",
            (
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_length_y_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (-float(table["crossbrace_half_length_x_m"]), 0.0, crossbrace_z),
        ),
        (
            "shakebench_table_upper_crossbrace_right",
            (
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_length_y_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (float(table["crossbrace_half_length_x_m"]), 0.0, crossbrace_z),
        ),
    ):
        _append_unique_geom(
            arena.table_body,
            _visual_geom(name, "box", size, pos, material="shakebench_frame_metal", rgba=frame_rgb),
            geom_names,
        )
        visual_geoms.append(name)
    for index, (x, y) in enumerate(envelope["leg_centers_xy"]):
        leg_name = f"shakebench_table_upper_leg_{index}"
        local_z = envelope["leg_center_world_z"] - envelope["table_world_z"]
        _append_unique_geom(
            arena.table_body,
            _visual_geom(
                leg_name,
                "box",
                (0.025, 0.025, envelope["leg_half_height"]),
                (x, y, local_z),
                material="shakebench_frame_metal",
                rgba=frame_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(leg_name)
        foot_name = f"shakebench_table_upper_foot_{index}"
        foot_z = float(table["support_plane_z_m"]) + upper_plate_size[2] / 2.0 - envelope["table_world_z"]
        _append_unique_geom(
            arena.table_body,
            _visual_geom(
                foot_name,
                "box",
                upper_plate_size / 2.0,
                (x, y, foot_z),
                material="shakebench_frame_metal",
                rgba=frame_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(foot_name)
        sleeve_name = f"shakebench_table_upper_sleeve_{index}"
        sleeve_z = float(table["upper_sleeve_center_z_m"]) - envelope["table_world_z"]
        _append_unique_geom(
            arena.table_body,
            _visual_geom(
                sleeve_name,
                "cylinder",
                (float(table["upper_sleeve_radius_m"]), float(table["upper_sleeve_half_height_m"])),
                (x, y, sleeve_z),
                material="shakebench_mount_rubber",
                rgba=rubber_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(sleeve_name)

    for equipment in room["equipment"]:
        body = _visual_body(equipment["name"], equipment["pos_m"])
        angle = math.radians(float(equipment["yaw_deg"])) / 2.0
        body.set("quat", _fmt((math.cos(angle), 0.0, 0.0, math.sin(angle))))
        visual_geoms.extend(_append_authored_visuals(arena, body, equipment["geoms"], geom_names))
        visual_bodies.append(equipment["name"])
        frame_bodies[equipment["name"]] = "world"
        _append_unique_body(worldbody, body)

    for camera in config.section("cameras").values():
        _add_camera(worldbody, camera)
    for camera in config.section("render").get("detail_cameras", {}).values():
        _add_camera(worldbody, camera)

    return frame_bodies, tuple(dict.fromkeys(visual_geoms)), tuple(dict.fromkeys(visual_bodies))


@dataclass(frozen=True)
class SceneInventory:
    """Machine-readable source-scene inventory returned by augmentation."""

    config_sha256: str
    scene_id: str
    geometry_variant: str
    frame_ownership: Mapping[str, Any]
    role_handles: Mapping[str, str]
    visual_geom_names: tuple[str, ...]
    visual_body_names: tuple[str, ...]
    derived_support_plane: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_ownership", MappingProxyType(_thaw(self.frame_ownership)))
        object.__setattr__(self, "role_handles", MappingProxyType(dict(self.role_handles)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "scene_id": self.scene_id,
            "geometry_variant": self.geometry_variant,
            "frame_ownership": _thaw(self.frame_ownership),
            "role_handles": dict(self.role_handles),
            "visual_geom_names": list(self.visual_geom_names),
            "visual_body_names": list(self.visual_body_names),
            "derived_support_plane": _thaw(self.derived_support_plane),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _set_scene_visual_alpha(arena: Any, enabled: bool) -> None:
    inventory: SceneInventory | None = getattr(arena, "scene_inventory", None)
    if inventory is None:
        return
    originals = getattr(arena, "_shakebench_scene_rgba", {})
    for name in inventory.visual_geom_names:
        geom = arena.root.find(f".//geom[@name='{name}']")
        if geom is None:
            continue
        if name not in originals:
            originals[name] = geom.get("rgba")
        original = originals[name]
        if enabled:
            if original is None:
                geom.attrib.pop("rgba", None)
            else:
                geom.set("rgba", original)
        else:
            if original is None:
                geom.set("rgba", "1 1 1 0")
            else:
                rgba = np.asarray(tuple(float(value) for value in original.split()), dtype=float)
                if rgba.size == 4:
                    rgba[3] = 0.0
                    geom.set("rgba", _fmt(rgba))
                else:
                    geom.set("rgba", "1 1 1 0")
    arena._shakebench_scene_rgba = originals


def augment_scene_mjcf(
    arena: Any, config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None
) -> SceneInventory:
    """Add the deterministic visual scene to an existing ShakeBench arena.

    Args:
        arena: A :class:`ShakeBenchArena` instance whose canonical worktable
            body and physical floor already exist.
        config: The authenticated scene configuration or a path/mapping that
            can be loaded into one.

    Returns:
        A :class:`SceneInventory` describing source-frame ownership and the
        display-only names.
    """

    scene_config = _coerce_config(config)
    existing = getattr(arena, "scene_inventory", None)
    if existing is not None:
        if existing.config_sha256 != scene_config.config_sha256:
            raise SceneConfigError("arena was already augmented with a different scene configuration")
        return existing
    frame_bodies, visual_geoms, visual_bodies = _build_scene_visuals(arena, scene_config)
    table_support = _table_support_geometry(arena, scene_config)
    support = {
        "method": "source arena table support plus compiled mount audit",
        "configured_platen_nominal_top_z_m": float(scene_config.section("platen")["nominal_top_z_m"]),
        "worktable_visual_support_bottom_z_m": float(scene_config.section("table_support")["support_plane_z_m"]),
        "table_world_z_m": table_support["table_world_z"],
        "assembly_error_before_robot_mount_m": 0.0,
    }
    body_frames = {
        **{body: frame for body, frame in frame_bodies.items()},
        WORKTABLE_BODY_NAME: "isolated_worktable",
    }
    inventory = SceneInventory(
        config_sha256=scene_config.config_sha256,
        scene_id=scene_config.scene_id,
        geometry_variant=scene_config.geometry_variant,
        frame_ownership=body_frames,
        role_handles=dict(scene_config.section("role_handles")),
        visual_geom_names=visual_geoms,
        visual_body_names=visual_bodies,
        derived_support_plane=support,
    )
    arena.scene_config = scene_config
    arena.scene_inventory = inventory
    arena._shakebench_scene_rgba = {}
    _set_scene_visual_alpha(arena, True)
    return inventory


def scene_visual_geom_names(
    raw_model: Any, config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None
) -> tuple[str, ...]:
    """Return compiled display-only geom names recognized by the scene authority."""

    scene_config = _coerce_config(config)
    prefixes = tuple(
        prefix for frame in scene_config.section("frame_ownership").values() for prefix in frame["geom_prefixes"]
    )
    names = []
    for geom_id in range(int(raw_model.ngeom)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is not None and (name == TABLE_VISUAL_GEOM_NAME or str(name).startswith(prefixes)):
            names.append(str(name))
    return tuple(names)


def _raw_model_and_data(sim_or_model: Any) -> tuple[Any, Any]:
    model = getattr(sim_or_model, "model", sim_or_model)
    raw_model = getattr(model, "_model", model)
    data = getattr(sim_or_model, "data", None)
    raw_data = getattr(data, "_data", data)
    if raw_data is None:
        raw_data = mujoco.MjData(raw_model)
        mujoco.mj_forward(raw_model, raw_data)
    return raw_model, raw_data


def _mujoco_id(raw_model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(raw_model, object_type, name))
    if value < 0:
        raise SceneAuditError(f"compiled scene is missing {object_type} {name!r}")
    return value


def _body_ancestry(raw_model: Any, body_id: int) -> tuple[str, ...]:
    names: list[str] = []
    current = int(body_id)
    while current != 0:
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, current)
        if name is None:
            break
        names.append(str(name))
        current = int(raw_model.body_parentid[current])
    return tuple(reversed(names))


def _geom_body_name(raw_model: Any, geom_id: int) -> str:
    body_id = int(raw_model.geom_bodyid[geom_id])
    name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return str(name)


def _frame_for_body(raw_model: Any, body_id: int) -> str:
    ancestry = _body_ancestry(raw_model, body_id)
    if WORKTABLE_BODY_NAME in ancestry:
        return "isolated_worktable"
    if DECK_BODY_NAME in ancestry:
        return "dynamic_deck"
    if any(name.startswith("shakebench_") for name in ancestry):
        return "world"
    if mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id) == "can_main":
        return "world_free_body"
    return "world"


def _canonical_array(value: Any) -> list[float | int]:
    array = np.asarray(value)
    if array.dtype.kind in "iu":
        return [int(item) for item in array.reshape(-1)]
    return [float(item) for item in array.reshape(-1)]


def _physics_signature_payload(raw_model: Any) -> dict[str, Any]:
    def is_scene_body(name: str) -> bool:
        return name.startswith(SCENE_BODY_PREFIX)

    physical_bodies: dict[str, Any] = {}
    for body_id in range(int(raw_model.nbody)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None or is_scene_body(str(name)):
            continue
        parent_id = int(raw_model.body_parentid[body_id])
        physical_bodies[str(name)] = {
            "parent": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
            "pos": _canonical_array(raw_model.body_pos[body_id]),
            "quat": _canonical_array(raw_model.body_quat[body_id]),
            "mass": float(raw_model.body_mass[body_id]),
            "ipos": _canonical_array(raw_model.body_ipos[body_id]),
            "inertia": _canonical_array(raw_model.body_inertia[body_id]),
        }
    physical_geoms: dict[str, Any] = {}
    for geom_id in range(int(raw_model.ngeom)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            continue
        if not int(raw_model.geom_contype[geom_id]) and not int(raw_model.geom_conaffinity[geom_id]):
            continue
        physical_geoms[str(name)] = {
            "body": _geom_body_name(raw_model, geom_id),
            "type": int(raw_model.geom_type[geom_id]),
            "pos": _canonical_array(raw_model.geom_pos[geom_id]),
            "quat": _canonical_array(raw_model.geom_quat[geom_id]),
            "size": _canonical_array(raw_model.geom_size[geom_id]),
            "friction": _canonical_array(raw_model.geom_friction[geom_id]),
            "solref": _canonical_array(raw_model.geom_solref[geom_id]),
            "solimp": _canonical_array(raw_model.geom_solimp[geom_id]),
            "contype": int(raw_model.geom_contype[geom_id]),
            "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
            "condim": int(raw_model.geom_condim[geom_id]),
            "margin": float(raw_model.geom_margin[geom_id]),
            "gap": float(raw_model.geom_gap[geom_id]),
        }
    joints: dict[str, Any] = {}
    for joint_id in range(int(raw_model.njnt)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        dof_id = int(raw_model.jnt_dofadr[joint_id])
        qpos_id = int(raw_model.jnt_qposadr[joint_id])
        joints[str(name)] = {
            "body": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, int(raw_model.jnt_bodyid[joint_id])),
            "type": int(raw_model.jnt_type[joint_id]),
            "pos": _canonical_array(raw_model.jnt_pos[joint_id]),
            "axis": _canonical_array(raw_model.jnt_axis[joint_id]),
            "range": _canonical_array(raw_model.jnt_range[joint_id]),
            "limited": bool(raw_model.jnt_limited[joint_id]),
            "stiffness": float(raw_model.jnt_stiffness[joint_id]),
            "damping": float(raw_model.dof_damping[dof_id]),
            "springref": float(raw_model.qpos_spring[qpos_id]),
        }
    equalities: dict[str, Any] = {}
    for equality_id in range(int(raw_model.neq)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id)
        if name is None:
            continue
        equalities[str(name)] = {
            "type": int(raw_model.eq_type[equality_id]),
            "body1": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, int(raw_model.eq_obj1id[equality_id])),
            "body2": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, int(raw_model.eq_obj2id[equality_id])),
            "joint1": (
                mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, int(raw_model.eq_obj1id[equality_id]))
                if int(raw_model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT)
                else None
            ),
            "joint2": (
                mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, int(raw_model.eq_obj2id[equality_id]))
                if int(raw_model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT)
                else None
            ),
            "solref": _canonical_array(raw_model.eq_solref[equality_id]),
            "solimp": _canonical_array(raw_model.eq_solimp[equality_id]),
        }
    actuators: dict[str, Any] = {}
    for actuator_id in range(int(raw_model.nu)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name is None:
            continue
        trnid = _canonical_array(raw_model.actuator_trnid[actuator_id])
        target_name = None
        if int(raw_model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            target_name = mujoco.mj_id2name(
                raw_model, mujoco.mjtObj.mjOBJ_JOINT, int(raw_model.actuator_trnid[actuator_id, 0])
            )
        actuators[str(name)] = {
            "trntype": int(raw_model.actuator_trntype[actuator_id]),
            "trnid": trnid,
            "target": target_name,
            "ctrlrange": _canonical_array(raw_model.actuator_ctrlrange[actuator_id]),
            "ctrllimited": bool(raw_model.actuator_ctrllimited[actuator_id]),
        }
    contact_pairs = []
    for pair_id in range(int(raw_model.npair)):
        contact_pairs.append(
            {
                "geom1": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom1[pair_id])),
                "geom2": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom2[pair_id])),
                "friction": _canonical_array(raw_model.pair_friction[pair_id]),
                "solref": _canonical_array(raw_model.pair_solref[pair_id]),
                "solimp": _canonical_array(raw_model.pair_solimp[pair_id]),
                "margin": float(raw_model.pair_margin[pair_id]),
                "gap": float(raw_model.pair_gap[pair_id]),
                "dim": int(raw_model.pair_dim[pair_id]),
            }
        )
    options = {
        "timestep": float(raw_model.opt.timestep),
        "integrator": int(raw_model.opt.integrator),
        "solver": int(raw_model.opt.solver),
        "iterations": int(raw_model.opt.iterations),
        "tolerance": float(raw_model.opt.tolerance),
        "ls_iterations": int(raw_model.opt.ls_iterations),
        "noslip_iterations": int(raw_model.opt.noslip_iterations),
        "ccd_iterations": int(getattr(raw_model.opt, "ccd_iterations", 0)),
        "ccd_tolerance": float(getattr(raw_model.opt, "ccd_tolerance", 0.0)),
        "impratio": float(raw_model.opt.impratio),
        "cone": int(raw_model.opt.cone),
        "jacobian": int(raw_model.opt.jacobian),
        "disableflags": int(raw_model.opt.disableflags),
        "enableflags": int(raw_model.opt.enableflags),
    }
    return {
        "options": options,
        "bodies": physical_bodies,
        "geoms": physical_geoms,
        "joints": joints,
        "equalities": equalities,
        "actuators": actuators,
        "contact_pairs": contact_pairs,
    }


def compiled_physics_signature(sim_or_model: Any) -> dict[str, Any]:
    """Return a stable name-based signature excluding display-only geometry."""

    raw_model, _ = _raw_model_and_data(sim_or_model)
    payload = _physics_signature_payload(raw_model)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "payload": payload}


@dataclass(frozen=True)
class SceneAudit:
    """Compiled scene audit with stable names rather than raw MuJoCo IDs."""

    passed: bool
    config_sha256: str
    scene_id: str
    geometry_variant: str
    frame_ownership: Mapping[str, Any]
    role_audit: Mapping[str, Any]
    visual_geoms: Mapping[str, Any]
    visual_bodies: Mapping[str, Any]
    camera_audit: Mapping[str, Any]
    physics_signature: Mapping[str, Any]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "frame_ownership",
            "role_audit",
            "visual_geoms",
            "visual_bodies",
            "camera_audit",
            "physics_signature",
        ):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, MappingProxyType(_thaw(value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "config_sha256": self.config_sha256,
            "scene_id": self.scene_id,
            "geometry_variant": self.geometry_variant,
            "frame_ownership": _thaw(self.frame_ownership),
            "compiled_frame_ownership": {name: item["frame"] for name, item in self.visual_geoms.items()},
            "role_audit": _thaw(self.role_audit),
            "visual_geoms": _thaw(self.visual_geoms),
            "visual_bodies": _thaw(self.visual_bodies),
            "camera_audit": _thaw(self.camera_audit),
            "physics_signature": _thaw(self.physics_signature),
            "errors": list(self.errors),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def audit_compiled_scene(
    sim: Any, config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None
) -> SceneAudit:
    """Audit frame ownership, visual physics isolation, and cameras.

    Structural violations raise :class:`SceneAuditError`; a successful audit
    is also returned as a dict-like :class:`SceneAudit` for manifests.
    """

    scene_config = _coerce_config(config)
    raw_model, raw_data = _raw_model_and_data(sim)
    visual_names = scene_visual_geom_names(raw_model, scene_config)
    expected_roles = scene_config.section("role_handles")
    role_audit: dict[str, Any] = {}
    for role, body_name in expected_roles.items():
        body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        ancestry = _body_ancestry(raw_model, body_id)
        frame = _frame_for_body(raw_model, body_id)
        if role == "deck_visual" and frame != "dynamic_deck":
            raise SceneAuditError("deck_visual is not a descendant of dynamic deck")
        if role == "isolated_worktable" and WORKTABLE_BODY_NAME not in ancestry:
            raise SceneAuditError("isolated_worktable role is not attached to worktable")
        role_audit[role] = {"body_name": body_name, "ancestry": list(ancestry), "frame": frame, "body_id": body_id}
    for frame, content in scene_config.section("frame_ownership").items():
        if frame == "world_free_body":
            continue
        for body_name in content["bodies"]:
            body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            actual_frame = _frame_for_body(raw_model, body_id)
            if actual_frame != frame:
                raise SceneAuditError(f"frame ownership for body {body_name!r} is {actual_frame!r}, expected {frame!r}")

    visual_geoms: dict[str, Any] = {}
    for name in visual_names:
        geom_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        body_name = _geom_body_name(raw_model, geom_id)
        if int(raw_model.geom_contype[geom_id]) != 0 or int(raw_model.geom_conaffinity[geom_id]) != 0:
            raise SceneAuditError(f"scene visual geom {name!r} contributes to contact")
        visual_geoms[name] = {
            "geom_id": geom_id,
            "body_name": body_name,
            "frame": _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom_id])),
            "type": int(raw_model.geom_type[geom_id]),
            "size_m": _canonical_array(raw_model.geom_size[geom_id]),
            "contype": int(raw_model.geom_contype[geom_id]),
            "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
        }
    if not any(name.startswith("shakebench_platen_") for name in visual_names):
        raise SceneAuditError("compiled scene is missing the visible vibration platen")
    for prefix, expected_frame in (
        ("shakebench_table_upper_", "isolated_worktable"),
        ("shakebench_table_lower_mount_", "dynamic_deck"),
        ("shakebench_stewart_outer_", "world"),
        ("shakebench_stewart_rod_", "dynamic_deck"),
    ):
        names = [name for name in visual_names if name.startswith(prefix)]
        if not names:
            raise SceneAuditError(f"compiled scene is missing visual role prefix {prefix!r}")
        wrong = [name for name in names if visual_geoms[name]["frame"] != expected_frame]
        if wrong:
            raise SceneAuditError(f"visual role {prefix!r} has wrong frame: {wrong}")

    visual_bodies: dict[str, Any] = {}
    for body_name in _body_name_set_from_model(raw_model):
        if not body_name.startswith(SCENE_BODY_PREFIX):
            continue
        body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_names = [
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(int(raw_model.njnt))
            if int(raw_model.jnt_bodyid[joint_id]) == body_id
        ]
        if joint_names:
            raise SceneAuditError(f"visual body {body_name!r} owns a joint: {joint_names}")
        if float(raw_model.body_subtreemass[body_id]) != 0.0:
            raise SceneAuditError(f"visual body {body_name!r} adds mass {raw_model.body_subtreemass[body_id]}")
        visual_bodies[body_name] = {
            "body_id": body_id,
            "frame": _frame_for_body(raw_model, body_id),
            "mass_kg": float(raw_model.body_mass[body_id]),
            "subtree_mass_kg": float(raw_model.body_subtreemass[body_id]),
            "inertia_kg_m2": _canonical_array(raw_model.body_inertia[body_id]),
            "joint_names": [name for name in joint_names if name is not None],
        }

    contact_pairs_with_scene_visuals = []
    visual_set = set(visual_names)
    for pair_id in range(int(raw_model.npair)):
        pair_names = (
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom1[pair_id])),
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_model.pair_geom2[pair_id])),
        )
        if any(name in visual_set for name in pair_names):
            contact_pairs_with_scene_visuals.append(pair_names)
    if contact_pairs_with_scene_visuals:
        raise SceneAuditError(f"scene visuals appear in explicit contact pairs: {contact_pairs_with_scene_visuals}")
    active_visual_contacts = []
    for contact_id in range(int(raw_data.ncon)):
        pair_names = (
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_data.contact[contact_id].geom1)),
            mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(raw_data.contact[contact_id].geom2)),
        )
        if any(name in visual_set for name in pair_names):
            active_visual_contacts.append(pair_names)
    if active_visual_contacts:
        raise SceneAuditError(f"scene visuals have active contacts: {active_visual_contacts}")

    camera_audit: dict[str, Any] = {}
    camera_names = {
        str(mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id))
        for camera_id in range(int(raw_model.ncam))
    }
    for key, camera in scene_config.section("cameras").items():
        name = str(camera["name"])
        if name not in camera_names:
            raise SceneAuditError(f"compiled scene is missing camera {name!r}")
        camera_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        camera_audit[key] = {
            "name": name,
            "camera_id": camera_id,
            "pos_m": _canonical_array(raw_model.cam_pos[camera_id]),
            "fovy_deg": float(raw_model.cam_fovy[camera_id]),
        }
    physics_signature = compiled_physics_signature(sim)
    return SceneAudit(
        passed=True,
        config_sha256=scene_config.config_sha256,
        scene_id=scene_config.scene_id,
        geometry_variant=scene_config.geometry_variant,
        frame_ownership=scene_config.section("frame_ownership"),
        role_audit=role_audit,
        visual_geoms=visual_geoms,
        visual_bodies=visual_bodies,
        camera_audit=camera_audit,
        physics_signature=physics_signature,
    )


def _body_name_set_from_model(raw_model: Any) -> tuple[str, ...]:
    return tuple(
        str(name)
        for body_id in range(int(raw_model.nbody))
        if (name := mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)) is not None
    )


def _geom_aabb(raw_model: Any, raw_data: Any, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    centre = np.asarray(raw_data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(raw_data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    geom_type = int(raw_model.geom_type[geom_id])
    size = np.asarray(raw_model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        local_half = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local_half = np.repeat(size[0], 3)
    elif geom_type in {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}:
        local_half = np.array(
            (size[0], size[0], size[1] + (size[0] if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0))
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        local_half = size[:3]
    else:
        radius = float(raw_model.geom_rbound[geom_id])
        local_half = np.repeat(radius, 3)
    extent = np.abs(rotation).dot(local_half)
    return centre - extent, centre + extent


def _geom_distance(raw_model: Any, raw_data: Any, geom1: int, geom2: int) -> tuple[float, np.ndarray]:
    support = _floor_cylinder_support_distance(raw_model, raw_data, geom1, geom2)
    if support is not None:
        return support
    fromto = np.zeros(6, dtype=np.float64)
    distance = float(mujoco.mj_geomDistance(raw_model, raw_data, int(geom1), int(geom2), 100.0, fromto))
    if not np.isfinite(distance):
        distance = -float("inf") if distance < 0.0 else float("inf")
    return distance, fromto.reshape(2, 3)


def _floor_cylinder_support_distance(model: Any, data: Any, geom1: int, geom2: int) -> tuple[float, np.ndarray] | None:
    """Exact top-face support distance for a cylinder contained over a floor slab.

    MuJoCo's convex distance can return millimetre-scale negative values for
    a tangent wheel on a broad, thin slab. Use the cylinder support function
    only when its entire XY bound lies inside a horizontal slab and its centre
    is above the slab. Penetration remains negative; no tolerance whitelist.
    """

    for slab, wheel in ((geom1, geom2), (geom2, geom1)):
        if not model.geom(slab).name.startswith("shakebench_floor_slab_"):
            continue
        if (
            model.geom_type[slab] != mujoco.mjtGeom.mjGEOM_BOX
            or model.geom_type[wheel] != mujoco.mjtGeom.mjGEOM_CYLINDER
        ):
            continue
        if not np.allclose(data.geom_xmat[slab].reshape(3, 3), np.eye(3), atol=1e-12, rtol=0):
            continue
        lower, upper = _geom_aabb(model, data, slab)
        wheel_lower, wheel_upper = _geom_aabb(model, data, wheel)
        centre = data.geom_xpos[wheel]
        if np.any(wheel_lower[:2] < lower[:2]) or np.any(wheel_upper[:2] > upper[:2]) or centre[2] < upper[2]:
            continue
        axis = data.geom_xmat[wheel].reshape(3, 3)[:, 2]
        radius, half_length = model.geom_size[wheel, :2]
        vertical = np.array([0.0, 0.0, 1.0])
        radial = vertical - axis[2] * axis
        norm = np.linalg.norm(radial)
        point = centre - half_length * np.sign(axis[2]) * axis
        if norm > 1e-12:
            point = point - radius * radial / norm
        floor_point = np.array([point[0], point[1], upper[2]])
        points = np.array([floor_point, point])
        return float(point[2] - upper[2]), points if slab == geom1 else points[::-1]
    return None


def _pair_whitelist(name1: str, name2: str, signed_distance_m: float | None = None) -> str | None:
    names = {name1, name2}
    if names == {"shakebench_platen_surface", "robot0_link0_collision"}:
        # The stock link0 convex mesh extends ~0.033 mm below its authored
        # mounting datum. Accept only this named, sub-0.1 mm seating interface.
        # Larger penetration remains an error, including in negative fixtures.
        if signed_distance_m is not None and signed_distance_m >= -1.0e-4:
            return "direct_panda_flange_to_platen_seating_interface"
    if any(name.startswith("shakebench_pit_") for name in names) and any(
        name.startswith("shakebench_floor_slab_") for name in names
    ):
        return "pit_visual_boundary"
    if any(name.startswith("shakebench_pit_") for name in names) and any(
        name.startswith("shakebench_guardrail_") for name in names
    ):
        return "pit_safety_boundary"
    if any(name.startswith("shakebench_table_upper_") for name in names) and any(
        name.startswith("shakebench_table_lower_mount_") for name in names
    ):
        return "two_stage_isolator_mount_interface"
    if any(name.startswith("shakebench_stewart_rod") for name in names) and any(
        name.startswith("shakebench_platen_") for name in names
    ):
        return "stewart_rod_to_platen_joint_interface"
    if any(name.startswith("shakebench_stewart_outer") for name in names) and any(
        name.startswith("shakebench_stewart_rod") for name in names
    ):
        return "two_segment_stewart_overlap"
    if any(name.startswith("shakebench_shaker_foundation_") for name in names) and any(
        name.startswith("shakebench_pit_") for name in names
    ):
        return "shaker_foundation_inside_pit"
    if any(name.startswith("shakebench_platen_") for name in names) and any(
        name.startswith("fixed_mount", 0) for name in names
    ):
        return "robot_mount_to_platen_expected_support"
    if "table_visual" in names and any(name.startswith("fixed_mount") for name in names):
        # The canonical tabletop edge and the stock Rethink proxy are tangent
        # in AABB space. The primitive distance has a sub-millimetre numerical
        # round-off on this exact edge; a deeper overlap stays a red gate.
        if signed_distance_m is not None and signed_distance_m >= -1.0e-3:
            return "canonical_tabletop_edge_adjacency_with_mount_proxy"
    return None


def _scene_collision_proxy_ids(raw_model: Any, config: SceneVisualConfig) -> tuple[int, ...]:
    prefixes = tuple(config.section("clearance")["mount_collision_prefixes"])
    ids = []
    for geom_id in range(int(raw_model.ngeom)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not (int(raw_model.geom_contype[geom_id]) or int(raw_model.geom_conaffinity[geom_id])):
            continue
        if str(name).startswith(prefixes):
            ids.append(geom_id)
    return tuple(ids)


def _candidate_pairs(
    raw_model: Any, raw_data: Any, config: SceneVisualConfig
) -> tuple[tuple[str, int, str, int, str | None], ...]:
    scene_names = scene_visual_geom_names(raw_model, config)
    scene_ids = {name: _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in scene_names}
    window = float(config.section("clearance")["candidate_window_m"])
    candidates: list[tuple[str, int, str, int, str | None]] = []
    for index, name1 in enumerate(scene_names):
        body1 = int(raw_model.geom_bodyid[scene_ids[name1]])
        for name2 in scene_names[index + 1 :]:
            geom2 = scene_ids[name2]
            if body1 == int(raw_model.geom_bodyid[geom2]):
                continue
            distance, _ = _geom_distance(raw_model, raw_data, scene_ids[name1], geom2)
            whitelist = _pair_whitelist(name1, name2)
            if whitelist is not None or distance <= window:
                candidates.append(("visual_visual", scene_ids[name1], name1, geom2, name2))
    proxy_ids = _scene_collision_proxy_ids(raw_model, config)
    for visual_name, visual_id in scene_ids.items():
        for proxy_id in proxy_ids:
            proxy_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, proxy_id)
            if proxy_name is None:
                continue
            distance, _ = _geom_distance(raw_model, raw_data, visual_id, proxy_id)
            whitelist = _pair_whitelist(visual_name, str(proxy_name))
            if whitelist is not None or distance <= window:
                candidates.append(("visual_collision_proxy", visual_id, visual_name, proxy_id, str(proxy_name)))
    return tuple(candidates)


def _normalise_pose(value: Any, nominal_pose: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 7:
        pose = array.copy()
    elif array.size == 6:
        position = nominal_pose[:3] + array[:3]
        pose = np.concatenate((position, rotation_vector_to_quat(array[3:])))
    else:
        raise SceneConfigError("safe envelope poses must contain six relative or seven world-pose values")
    if not np.all(np.isfinite(pose)):
        raise SceneConfigError("safe envelope poses must be finite")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise SceneConfigError("safe envelope pose quaternion must be non-zero")
    pose[3:] = quaternion / norm
    return pose


def rotation_vector_to_quat(rotation_vector: Iterable[float]) -> np.ndarray:
    """Return a normalized MuJoCo ``wxyz`` quaternion for a rotation vector."""

    vector = np.asarray(tuple(rotation_vector), dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SceneConfigError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-14:
        return np.array((1.0, 0.0, 0.0, 0.0))
    axis = vector / angle
    return np.concatenate(((math.cos(angle / 2.0),), axis * math.sin(angle / 2.0)))


def _safe_envelope_poses(config: SceneVisualConfig, nominal_pose: np.ndarray, envelope: Any) -> tuple[np.ndarray, ...]:
    if envelope is not None:
        if isinstance(envelope, Mapping):
            values = envelope.get("poses", envelope.get("deck_pose_samples"))
            if values is not None:
                return tuple(_normalise_pose(value, nominal_pose) for value in values)
            translations = envelope.get("translation_abs_m")
            rotations = envelope.get("rotation_abs_rad")
            if translations is not None or rotations is not None:
                source = dict(config.to_dict()["safe_excitation_envelope"])
                source.update(envelope)
                envelope = source
        elif hasattr(envelope, "evaluate"):
            times = np.linspace(
                0.0,
                float(config.section("safe_excitation_envelope")["duration_s"]),
                int(config.section("safe_excitation_envelope")["sample_count"]),
            )
            values = []
            for time_s in times:
                result = envelope.evaluate(float(time_s))
                q = np.asarray(getattr(result, "q", result), dtype=float).reshape(-1)
                values.append(_normalise_pose(q, nominal_pose))
            return tuple(values)
        else:
            return tuple(_normalise_pose(value, nominal_pose) for value in envelope)
    source = config.section("safe_excitation_envelope") if envelope is None else _require_mapping("envelope", envelope)
    translation_abs = np.asarray(source["translation_abs_m"], dtype=float)
    rotation_abs = np.asarray(source["rotation_abs_rad"], dtype=float)
    relative: list[np.ndarray] = [np.zeros(6)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            value = np.zeros(6)
            value[axis] = sign * translation_abs[axis]
            relative.append(value)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            value = np.zeros(6)
            value[3 + axis] = sign * rotation_abs[axis]
            relative.append(value)
    # Eight simultaneous translation extrema cover the corners of the
    # registered translation box without pretending to prove a continuum.
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        value = np.zeros(6)
        value[:3] = np.asarray(signs) * translation_abs
        relative.append(value)
    return tuple(_normalise_pose(value, nominal_pose) for value in relative)


def _support_report(raw_model: Any, raw_data: Any, config: SceneVisualConfig) -> dict[str, Any]:
    visual_names = scene_visual_geom_names(raw_model, config)

    def extrema(prefix: str, *, upper: bool = False) -> float | None:
        values = []
        for name in visual_names:
            if name.startswith(prefix):
                geom_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name)
                lower, upper_bound = _geom_aabb(raw_model, raw_data, geom_id)
                values.append(float(upper_bound[2] if upper else lower[2]))
        return (max(values) if upper else min(values)) if values else None

    table_bottom = extrema("shakebench_table_upper_foot_")
    platen_top = extrema("shakebench_platen_surface", upper=True)
    mount_bottom = None
    mount_names = []
    for geom_id in range(int(raw_model.ngeom)):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not str(name).startswith("fixed_mount"):
            continue
        if "pedestal_feet" not in str(name) and "pedestal_col" not in str(name):
            continue
        lower, _ = _geom_aabb(raw_model, raw_data, geom_id)
        mount_names.append(str(name))
        mount_bottom = float(lower[2]) if mount_bottom is None else min(mount_bottom, float(lower[2]))
    configured_top = float(config.section("platen")["nominal_top_z_m"])
    direct_body = config.section("clearance").get("direct_robot_support_body")
    if direct_body is not None:
        body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, direct_body)
        # Panda link0's authored base plane is its installation datum. Convex
        # bounding spheres are not a valid support-plane estimate for its mesh.
        mount_bottom = float(raw_data.xpos[body_id, 2])
        mount_names = [direct_body + ":authored_mount_datum"]
    derivation_inputs = {
        "worktable_foot_lowest_support_z_m": table_bottom,
        "robot_mount_lowest_support_z_m": mount_bottom,
    }
    finite_inputs = [value for value in derivation_inputs.values() if value is not None]
    derived_top = max(finite_inputs) if finite_inputs else None
    return {
        "method": "compiled geom AABB support extrema; direct Panda uses its authored link0 installation datum",
        "platen_nominal_top_z_m": platen_top,
        "configured_platen_nominal_top_z_m": configured_top,
        "derived_platen_nominal_top_z_m": derived_top,
        "derivation_rule": "max of compiled worktable-foot and robot-mount lowest support points",
        "derivation_inputs_z_m": derivation_inputs,
        "derived_vs_compiled_platen_top_error_m": (
            None if derived_top is None or platen_top is None else float(platen_top - derived_top)
        ),
        "worktable_foot_lowest_support_z_m": table_bottom,
        "robot_mount_lowest_support_z_m": mount_bottom,
        "worktable_assembly_error_m": None if table_bottom is None or platen_top is None else table_bottom - platen_top,
        "robot_mount_assembly_error_m": (
            None if mount_bottom is None or platen_top is None else mount_bottom - platen_top
        ),
        "robot_mount_support_geoms": mount_names,
    }


@dataclass(frozen=True)
class ClearanceReport:
    """Signed-distance report for nominal and registered safe poses."""

    passed: bool
    config_sha256: str
    geometry_variant: str
    nominal: Mapping[str, Any]
    safe_envelope: Mapping[str, Any]
    visual_visual_candidates: tuple[Mapping[str, Any], ...]
    visual_collision_candidates: tuple[Mapping[str, Any], ...]
    active_physical_contacts: tuple[Mapping[str, Any], ...]
    support_interfaces: Mapping[str, Any]
    stewart: Mapping[str, Any]
    unexpected_penetration_pairs: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...] = ()

    @property
    def clearance_passed(self) -> bool:
        """Compatibility alias for the scene-clearance gate result."""

        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "clearance_passed": self.passed,
            "config_sha256": self.config_sha256,
            "geometry_variant": self.geometry_variant,
            "nominal": _thaw(self.nominal),
            "safe_envelope": _thaw(self.safe_envelope),
            "visual_visual_candidates": [_thaw(item) for item in self.visual_visual_candidates],
            "visual_collision_candidates": [_thaw(item) for item in self.visual_collision_candidates],
            "active_physical_contacts": [_thaw(item) for item in self.active_physical_contacts],
            "support_interfaces": _thaw(self.support_interfaces),
            "stewart": _thaw(self.stewart),
            "unexpected_penetration_pairs": [_thaw(item) for item in self.unexpected_penetration_pairs],
            "warnings": list(self.warnings),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _distance_record(
    raw_model: Any,
    raw_data: Any,
    category: str,
    geom1: int,
    name1: str,
    geom2: int,
    name2: str,
    sample_index: int,
    pose: np.ndarray,
) -> dict[str, Any]:
    distance, closest = _geom_distance(raw_model, raw_data, geom1, geom2)
    type1 = int(raw_model.geom_type[geom1])
    type2 = int(raw_model.geom_type[geom2])
    method = "mujoco_geom_distance"
    if type1 == int(mujoco.mjtGeom.mjGEOM_MESH) or type2 == int(mujoco.mjtGeom.mjGEOM_MESH):
        method = "mujoco_geom_distance_mesh_convex_candidate"
    if _floor_cylinder_support_distance(raw_model, raw_data, geom1, geom2) is not None:
        method = "analytic_cylinder_support_to_containing_floor_top"
    return {
        "category": category,
        "geom1": name1,
        "geom2": name2,
        "body1": _geom_body_name(raw_model, geom1),
        "body2": _geom_body_name(raw_model, geom2),
        "frame1": _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom1])),
        "frame2": _frame_for_body(raw_model, int(raw_model.geom_bodyid[geom2])),
        "signed_distance_m": float(distance),
        "closest_points_m": closest.tolist(),
        "method": method,
        "sample_index": int(sample_index),
        "deck_pose_wxyz_m": pose.tolist(),
        "whitelist_reason": _pair_whitelist(name1, name2, distance),
    }


def _minimum_pair_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["category"], record["geom1"], record["geom2"])
        previous = grouped.get(key)
        if previous is None or record["signed_distance_m"] < previous["signed_distance_m"]:
            grouped[key] = record
    return list(grouped.values())


def _active_contacts(raw_model: Any, raw_data: Any) -> tuple[dict[str, Any], ...]:
    records = []
    for contact_id in range(int(raw_data.ncon)):
        contact = raw_data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        records.append(
            {
                "geom1": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "body1": _geom_body_name(raw_model, geom1),
                "body2": _geom_body_name(raw_model, geom2),
                "distance_m": float(contact.dist),
                "include_in_scene_gate": False,
            }
        )
    return tuple(records)


def _stewart_report(
    raw_model: Any, raw_data: Any, config: SceneVisualConfig, poses: Sequence[np.ndarray]
) -> dict[str, Any]:
    stewart = config.section("stewart")
    platen_body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, DECK_VISUAL_BODY_NAME)
    nominal_center_z = float(raw_data.xpos[platen_body_id][2])
    base_points, _ = _stewart_points(config, nominal_center_z)
    rows = []
    min_overlap = float("inf")
    min_length = float("inf")
    max_length = -float("inf")
    passed = True
    for sample_index, pose in enumerate(poses):
        _, platen_points = _stewart_points(config, float(pose[2]) + nominal_center_z)
        lengths = np.linalg.norm(platen_points - base_points, axis=1)
        overlap = float(stewart["outer_length_m"]) + float(stewart["rod_length_m"]) - lengths
        min_overlap = min(min_overlap, float(np.min(overlap)))
        min_length = min(min_length, float(np.min(lengths)))
        max_length = max(max_length, float(np.max(lengths)))
        sample_passed = bool(
            np.all(lengths >= float(stewart["leg_min_m"]))
            and np.all(lengths <= float(stewart["leg_max_m"]))
            and np.all(overlap >= float(config.section("clearance")["minimum_stewart_overlap_m"]))
        )
        passed = passed and sample_passed
        rows.append(
            {
                "sample_index": sample_index,
                "lengths_m": lengths.tolist(),
                "overlap_m": overlap.tolist(),
                "passed": sample_passed,
            }
        )
    return {
        "method": "analytic Stewart ellipse endpoints with registered pose samples",
        "sample_count": len(poses),
        "min_length_m": min_length,
        "max_length_m": max_length,
        "min_segment_overlap_m": min_overlap,
        "leg_limits_m": [float(stewart["leg_min_m"]), float(stewart["leg_max_m"])],
        "minimum_required_overlap_m": float(config.section("clearance")["minimum_stewart_overlap_m"]),
        "passed": passed,
        "samples": rows,
    }


def scene_clearance_report(
    sim: Any,
    config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None,
    envelope: Any = None,
) -> ClearanceReport:
    """Check candidate visual overlap and assembly clearance.

    Primitive pairs use MuJoCo's signed ``mj_geomDistance``.  Mesh-involving
    pairs are labelled as convex-candidate checks; this report intentionally
    records finite sampled coverage and does not claim a continuous proof.
    """

    scene_config = _coerce_config(config)
    raw_model, raw_data = _raw_model_and_data(sim)
    nominal_deck_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, DECK_BODY_NAME)
    nominal_pose = np.concatenate(
        (
            np.asarray(raw_data.xpos[nominal_deck_id], dtype=float),
            np.asarray(raw_data.xquat[nominal_deck_id], dtype=float),
        )
    )
    poses = _safe_envelope_poses(scene_config, nominal_pose, envelope)
    if not poses:
        raise SceneConfigError("safe clearance envelope must contain at least one pose")
    candidates = _candidate_pairs(raw_model, raw_data, scene_config)
    nominal_records: list[dict[str, Any]] = []
    safe_records: list[dict[str, Any]] = []
    # Work in a private MjData so a clearance audit never changes the caller's
    # state, mocap targets, warmstart buffers, or policy-facing history.
    audit_data = mujoco.MjData(raw_model)
    np.copyto(audit_data.qpos, raw_data.qpos)
    np.copyto(audit_data.qvel, raw_data.qvel)
    np.copyto(audit_data.act, raw_data.act)
    deck_joint_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, DECK_FREEJOINT_NAME)
    deck_qpos = int(raw_model.jnt_qposadr[deck_joint_id])
    for sample_index, pose in enumerate(poses):
        audit_data.qpos[deck_qpos : deck_qpos + 7] = pose
        mujoco.mj_forward(raw_model, audit_data)
        for category, geom1, name1, geom2, name2 in candidates:
            record = _distance_record(raw_model, audit_data, category, geom1, name1, geom2, name2, sample_index, pose)
            safe_records.append(record)
            if sample_index == 0:
                nominal_records.append(record)
    nominal_min = _minimum_pair_records(nominal_records)
    safe_min = _minimum_pair_records(safe_records)
    nominal_map = {(record["category"], record["geom1"], record["geom2"]): record for record in nominal_min}
    safe_map = {(record["category"], record["geom1"], record["geom2"]): record for record in safe_min}
    # Put every candidate in the compact report, retaining both nominal and
    # sampled minima. This lets a negative fixture be diagnosed by name/frame.
    visual_visual = []
    visual_collision = []
    unexpected = []
    tolerance = float(scene_config.section("clearance")["required_margin_m"])
    for key, safe_record in safe_map.items():
        category, name1, name2 = key
        record = dict(safe_record)
        record["nominal_signed_distance_m"] = (
            float(nominal_map[key]["signed_distance_m"]) if key in nominal_map else None
        )
        record["safe_envelope_min_signed_distance_m"] = float(safe_record["signed_distance_m"])
        record["safe_envelope_sample_count"] = len(poses)
        if category == "visual_visual":
            visual_visual.append(record)
        else:
            visual_collision.append(record)
        if record["whitelist_reason"] is None and float(record["safe_envelope_min_signed_distance_m"]) < -tolerance:
            unexpected.append(record)
    support = _support_report(raw_model, raw_data, scene_config)
    support_errors = [
        value
        for key, value in (
            ("worktable", support["worktable_assembly_error_m"]),
            ("robot_mount", support["robot_mount_assembly_error_m"]),
        )
        if value is not None and abs(float(value)) > float(scene_config.section("clearance")["assembly_tolerance_m"])
    ]
    stewart_report = _stewart_report(raw_model, raw_data, scene_config, poses)
    active_contacts = _active_contacts(raw_model, raw_data)
    warnings = []
    if support["robot_mount_lowest_support_z_m"] is None:
        warnings.append("robot mount support proxy was not found; mount clearance is incomplete")
    passed = not unexpected and not support_errors and bool(stewart_report["passed"])
    nominal = {
        "deck_pose_wxyz_m": nominal_pose.tolist(),
        "sample_count": 1,
        "visual_visual_minimums": [
            record for record in visual_visual if record["nominal_signed_distance_m"] is not None
        ],
        "visual_collision_minimums": [
            record for record in visual_collision if record["nominal_signed_distance_m"] is not None
        ],
    }
    safe = {
        "sample_count": len(poses),
        "sampling_method": scene_config.section("safe_excitation_envelope")["sampling_method"],
        "translation_abs_m": list(scene_config.section("safe_excitation_envelope")["translation_abs_m"]),
        "rotation_abs_rad": list(scene_config.section("safe_excitation_envelope")["rotation_abs_rad"]),
        "poses_wxyz_m": [pose.tolist() for pose in poses],
        "visual_visual_minimums": visual_visual,
        "visual_collision_minimums": visual_collision,
    }
    return ClearanceReport(
        passed=passed,
        config_sha256=scene_config.config_sha256,
        geometry_variant=scene_config.geometry_variant,
        nominal=nominal,
        safe_envelope=safe,
        visual_visual_candidates=tuple(visual_visual),
        visual_collision_candidates=tuple(visual_collision),
        active_physical_contacts=active_contacts,
        support_interfaces=support,
        stewart=stewart_report,
        unexpected_penetration_pairs=tuple(unexpected),
        warnings=tuple(warnings),
    )


__all__ = [
    "SCENE_SCHEMA_ID",
    "SCENE_SCHEMA_VERSION",
    "DEFAULT_SCENE_VISUAL_CONFIG_FILENAME",
    "SCENE_CONFIG_PATH",
    "SceneConfigError",
    "SceneAuditError",
    "SceneVisualConfig",
    "SceneInventory",
    "SceneAudit",
    "ClearanceReport",
    "canonical_scene_payload",
    "scene_config_hash",
    "load_scene_visual_config",
    "augment_scene_mjcf",
    "configure_scene_rendering",
    "compiled_physics_signature",
    "audit_compiled_scene",
    "scene_visual_geom_names",
    "scene_clearance_report",
    "rotation_vector_to_quat",
]
