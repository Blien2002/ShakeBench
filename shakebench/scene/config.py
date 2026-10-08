"""Authoritative ShakeBench scene visuals, inventory, and clearance audits.

The scene layer is deliberately separate from the benchmark physics contract.
It owns layout, materials, camera poses, and display-only geometry while the
arena and task continue to own the tabletop collision, inertial, contacts, and
success evaluator.  The public surface is intentionally small:

``load_scene_visual_config`` -> validated immutable configuration
``augment_scene_mjcf`` -> deterministic visual MJCF and scene inventory

No visual body created here has a joint or inertial.  Its geoms explicitly use
zero density and zero contact bits so that a display-only scene cannot add a
mass, constraint, contact pair, or task interaction.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_element, new_geom

from shakebench import models
from shakebench.utils.artifacts import json_ready as _json_ready

DEFAULT_SCENE_VISUAL_CONFIG_FILENAME = "shakebench_scene_visual.json"
SCENE_CONFIG_PATH = Path(models.assets_root) / DEFAULT_SCENE_VISUAL_CONFIG_FILENAME
#: Scene textures owned by the upstream robosuite package.  They are resolved
#: from robosuite by name instead of being copied, and every other texture name
#: stays inside the ShakeBench asset root.
SHARED_SCENE_TEXTURES = frozenset({"textures/steel-brushed.png"})


def _texture_path(texture: str) -> Path:
    """Resolve a scene texture from the package that owns it."""

    root = models.robosuite_assets_root if texture in SHARED_SCENE_TEXTURES else models.assets_root
    return Path(root) / texture


TABLE_VISUAL_GEOM_NAME = "table_visual"
DECK_VISUAL_BODY_NAME = "shakebench_platen_visual"
WORKTABLE_BODY_NAME = "worktable"


class SceneConfigError(ValueError):
    """Raised when the authoritative scene configuration is invalid."""


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


def _rgba(name: str, value: Any) -> tuple[float, float, float, float]:
    values = _vector(name, value, 4, minimum=0.0)
    if any(item > 1.0 for item in values):
        raise SceneConfigError(f"{name} values must lie in [0, 1]")
    return values


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SceneConfigError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class SceneVisualConfig:
    """Validated immutable scene visual configuration."""

    payload: Mapping[str, Any]
    source_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise SceneConfigError("payload must be a mapping")
        frozen = _freeze(_json_ready(self.payload))
        object.__setattr__(self, "payload", frozen)

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

    def section(self, name: str) -> Mapping[str, Any]:
        return _require_mapping(name, self.payload[name])

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)


def load_scene_visual_config(path: str | Path | None = None) -> SceneVisualConfig:
    """Load and authenticate the package-owned scene visual configuration.

    Args:
        path: Package-relative or absolute local path.  ``None`` selects the
            flat asset shipped with ShakeBench.

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
    return SceneVisualConfig(payload=payload, source_path=str(config_path))


def _coerce_config(config: SceneVisualConfig | Mapping[str, Any] | str | Path | None) -> SceneVisualConfig:
    if isinstance(config, SceneVisualConfig):
        return config
    if config is None or isinstance(config, (str, Path)):
        return load_scene_visual_config(config)
    if isinstance(config, Mapping):
        payload = dict(config)
        return SceneVisualConfig(payload=payload, source_path="<mapping>")
    raise SceneConfigError("config must be a SceneVisualConfig, mapping, path, or None")


def _fmt(values: Iterable[float]) -> str:
    return array_to_string(tuple(float(value) for value in values))


def _geom_name_set(root: ET.Element) -> set[str]:
    return {str(name) for element in root.iter("geom") if (name := element.get("name")) is not None}


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
    centre = np.asarray(config.section("platen").get("center_xy_m", (0.0, 0.0)), dtype=float)
    base[:, :2] += centre
    platen[:, :2] += centre
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
        texture.set("file", str(_texture_path(materials[config_key]).resolve()))
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
            "scene_id": self.scene_id,
            "geometry_variant": self.geometry_variant,
            "frame_ownership": _thaw(self.frame_ownership),
            "role_handles": dict(self.role_handles),
            "visual_geom_names": list(self.visual_geom_names),
            "visual_body_names": list(self.visual_body_names),
            "derived_support_plane": _thaw(self.derived_support_plane),
        }


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
        if existing.scene_id != scene_config.scene_id:
            raise SceneConfigError("arena was already augmented with a different scene configuration")
        return existing
    from shakebench.models.arenas.scene_visuals import build_scene_visuals

    frame_bodies, visual_geoms, visual_bodies = build_scene_visuals(arena, scene_config)
    table_support = _table_support_geometry(arena, scene_config)
    support = {
        "configured_platen_nominal_top_z_m": float(scene_config.section("platen")["nominal_top_z_m"]),
        "worktable_visual_support_bottom_z_m": float(scene_config.section("table_support")["support_plane_z_m"]),
        "table_world_z_m": table_support["table_world_z"],
    }
    body_frames = {
        **{body: frame for body, frame in frame_bodies.items()},
        WORKTABLE_BODY_NAME: "isolated_worktable",
    }
    inventory = SceneInventory(
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


__all__ = [
    "DEFAULT_SCENE_VISUAL_CONFIG_FILENAME",
    "SCENE_CONFIG_PATH",
    "SceneConfigError",
    "SceneVisualConfig",
    "SceneInventory",
    "load_scene_visual_config",
    "augment_scene_mjcf",
    "configure_scene_rendering",
    "rotation_vector_to_quat",
]
