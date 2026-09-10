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

from robosuite.utils.shakebench_artifacts import json_ready as _json_ready

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
    # Seat the moving foot on the upper isolator sleeve, above the deck-side
    # base plate. Seating both plates on support_z makes them coincide at rest
    # and sends the upper plate through the deck during downward relative travel.
    foot_bottom_world_z = float(table["upper_sleeve_center_z_m"]) + float(table["upper_sleeve_half_height_m"])
    foot_height = float(table["foot_plate_size_m"][-1])
    leg_bottom_world_z = foot_bottom_world_z + foot_height
    leg_half_height = (top_world_z - leg_bottom_world_z) / 2.0
    if leg_half_height <= 0.0:
        raise SceneConfigError("table upper legs have no positive height")
    return {
        "table_world_z": table_world_z,
        "support_z": support_z,
        "foot_center_world_z": foot_bottom_world_z + foot_height / 2.0,
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
    from robosuite.models.arenas.scene_visuals import build_scene_visuals

    frame_bodies, visual_geoms, visual_bodies = build_scene_visuals(arena, scene_config)
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


def compiled_physics_signature(sim_or_model: Any) -> dict[str, Any]:
    from robosuite.models.arenas.scene_audit import compiled_physics_signature as audit

    return audit(sim_or_model)


def audit_compiled_scene(sim_or_model: Any, config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None):
    from robosuite.models.arenas.scene_audit import audit_compiled_scene as audit

    return audit(sim_or_model, config)


def scene_clearance_report(
    sim_or_model: Any,
    config: SceneVisualConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    envelope: Any = None,
):
    from robosuite.models.arenas.scene_audit import scene_clearance_report as audit

    return audit(sim_or_model, config, envelope=envelope)


def _distance_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from robosuite.models.arenas.scene_audit import _distance_record as record

    return record(*args, **kwargs)




__all__ = [
    "SCENE_SCHEMA_ID",
    "SCENE_SCHEMA_VERSION",
    "DEFAULT_SCENE_VISUAL_CONFIG_FILENAME",
    "SCENE_CONFIG_PATH",
    "SceneConfigError",
    "SceneAuditError",
    "SceneVisualConfig",
    "SceneInventory",
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
