"""Task selection and public task identity, independent of rollout machinery.

Friction classes are experimental contact coefficients, not measured material
properties. The surface/object pair is the authority (MuJoCo geom defaults are
deliberately not used). All variants retain a common 349 g payload mass.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from robosuite.utils.shakebench_artifacts import payload_hash

OBJECTS = {
    "food_can": {
        "asset": "objects/food_can.xml",
        "source": "RoboCasa Objaverse canned_food_18",
        "source_url": "https://huggingface.co/datasets/robocasa/robocasa-assets",
        "source_license": "CC-BY-4.0",
        "friction_class": "low",
        "metal_mu": 0.15,
        "mat_mu": 0.60,
        "mass_kg": 0.349,
    },
    "light_wood_block": {
        "asset": "robosuite.models.objects.BoxObject:WoodLight:half_size=0.030,0.025,0.010",
        "friction_class": "low",
        "metal_mu": 0.15,
        "mat_mu": 0.60,
        "mass_kg": 0.10,
    },
    "cookie_box": {
        "asset": "objects/cookie_box.xml",
        "source": "RoboCasa Objaverse boxed_food_0",
        "source_url": "https://huggingface.co/datasets/robocasa/robocasa-assets",
        "source_license": "CC-BY-4.0",
        "friction_class": "medium",
        "metal_mu": 0.30,
        "mat_mu": 0.90,
        "mass_kg": 0.349,
    },
    "bread": {
        "asset": "objects/bread.xml",
        "friction_class": "high",
        "metal_mu": 0.50,
        "mat_mu": 1.20,
        "mass_kg": 0.349,
    },
}
SURFACES = ("metal", "mat")
TASK_SCHEMA_ID = "shakebench.task.v1"
MAT_VISUAL_RGBA = (0.90, 1.0, 0.95, 1.0)
MAT_TEXTURE_PATH = "textures/gray-felt.png"
TASK_VISUAL_REVISION = "felt_mat_wire_basket.v2"
# Compiled support in object coordinates, measured from the package meshes.
OBJECT_SUPPORT = {
    "food_can": (-0.0325, 0.0325, 0.025),
    "light_wood_block": (-0.01, 0.01, 0.03905124837953328),
    "cookie_box": (-0.0362, 0.0362, 0.06415052610852073),
    "bread": (-0.023251370186775307, 0.024748632093102355, 0.0312410001023236),
}


def object_asset_hashes(object_id):
    import xml.etree.ElementTree as ET

    from robosuite import models

    root = Path(models.assets_root)
    if object_id == "light_wood_block":
        sources = {
            "objects/generated_objects.py": root.parent / "objects/generated_objects.py",
            "objects/primitive/box.py": root.parent / "objects/primitive/box.py",
            "assets/textures/light-wood.png": root / "textures/light-wood.png",
        }
        return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    source = root / OBJECTS[object_id]["asset"]
    paths = [source] + [
        (source.parent / node.attrib["file"]).resolve()
        for node in ET.parse(source).getroot().findall("./asset/*")
        if "file" in node.attrib
    ]
    if object_id in {"food_can", "cookie_box"}:
        source_dir = {"food_can": "canned_food_18", "cookie_box": "cookie_box_0"}[object_id]
        paths += [
            source.parent / "meshes/robocasa_selected/LICENSE.md",
            source.parent / "meshes/robocasa_selected" / source_dir / "material.mtl",
        ]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@dataclass(frozen=True)
class TaskSpec:
    """Serializable task selector; invalid combinations fail before compilation."""

    task_type: str = "pick_place"
    object_id: str = "food_can"
    surface_id: str = "metal"

    def __post_init__(self):
        if not isinstance(self.task_type, str) or self.task_type != "pick_place":
            raise ValueError(f"unsupported task_type: {self.task_type!r}")
        if not isinstance(self.object_id, str) or self.object_id not in OBJECTS:
            raise ValueError(f"unsupported object_id: {self.object_id!r}")
        if not isinstance(self.surface_id, str) or self.surface_id not in SURFACES:
            raise ValueError(f"unsupported surface_id: {self.surface_id!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | TaskSpec | None) -> TaskSpec:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"task_type", "object_id", "surface_id"}:
            raise ValueError("task must contain exactly task_type, object_id and surface_id")
        return cls(**dict(value))

    def to_dict(self):
        return asdict(self)

    @property
    def variant_id(self):
        return f"{self.task_type}.{self.surface_id}.{self.object_id}"

    @property
    def table_sliding_mu(self):
        return OBJECTS[self.object_id][f"{self.surface_id}_mu"]

    @property
    def target_sliding_mu(self):
        # The target remains bare metal when a mat covers the source tabletop.
        return OBJECTS[self.object_id]["metal_mu"]

    @property
    def object_mass_kg(self):
        return float(OBJECTS[self.object_id]["mass_kg"])

    def contract(self):
        from robosuite import models

        record = {
            "schema_id": TASK_SCHEMA_ID,
            **self.to_dict(),
            "object_asset": OBJECTS[self.object_id]["asset"],
            "object_asset_sha256": object_asset_hashes(self.object_id),
            "object_support_lower_upper_radius_m": list(OBJECT_SUPPORT[self.object_id]),
            "object_friction_class": OBJECTS[self.object_id]["friction_class"],
            "table_object_sliding_mu": self.table_sliding_mu,
            "target_object_sliding_mu": self.target_sliding_mu,
            "finger_object_sliding_mu": 1.0,
            "object_mass_kg": self.object_mass_kg,
            "mat_thickness_m": 0.003 if self.surface_id == "mat" else 0.0,
            "task_visual_revision": TASK_VISUAL_REVISION,
            "target_visual_style": "light_metal_wire_basket",
            "mat_visual_rgba": list(MAT_VISUAL_RGBA) if self.surface_id == "mat" else None,
            "mat_texture": MAT_TEXTURE_PATH if self.surface_id == "mat" else None,
            "mat_texture_repeat": [3, 3] if self.surface_id == "mat" else None,
            "mat_texture_sha256": (
                hashlib.sha256(Path(models.assets_root, MAT_TEXTURE_PATH).read_bytes()).hexdigest()
                if self.surface_id == "mat"
                else None
            ),
            "surface_model": "flush rigid layer; fixed total worktable mass and top height",
            "inertia_model": "mass-normalized compiled geometry",
            "stability_constraint": "upright cosine >= 0.95; displacement after tipping is not classified as sliding",
            "success_semantics": "phase04_vibration_success_evaluator",
            "qualification": "pending_task_variant_requalification",
        }
        for key in ("source", "source_url", "source_license"):
            if key in OBJECTS[self.object_id]:
                record[key] = OBJECTS[self.object_id][key]
        record["task_contract_sha256"] = payload_hash(record, field="task_contract_sha256")
        return record


def make_task_object(spec: TaskSpec | None, *, name="can"):
    """Use existing robosuite meshes and its native textured box primitive."""
    from robosuite.models.objects import BoxObject, BreadObject, CanObject, CookieBoxObject, FoodCanObject
    from robosuite.utils.mjcf_utils import CustomMaterial

    if spec is None:
        return CanObject(name=name)
    if spec.object_id == "food_can":
        return FoodCanObject(name=name)
    if spec.object_id == "cookie_box":
        return CookieBoxObject(name=name)
    if spec.object_id == "light_wood_block":
        material = CustomMaterial(
            texture="WoodLight",
            tex_name=f"{spec.object_id}_texture",
            mat_name=f"{spec.object_id}_material",
            tex_attrib={"type": "2d"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.15", "shininess": "0.1"},
        )
        return BoxObject(
            name=name,
            size=(0.030, 0.025, 0.010),
            material=material,
            joints=[dict(type="free", damping="0.0005")],
        )
    if spec.object_id == "bread":
        return BreadObject(name=name)
    raise ValueError(f"unsupported task object {spec.object_id!r}")


def task_variants() -> tuple[TaskSpec, ...]:
    """Return frozen state variants; experimental objects remain opt-in."""
    return tuple(
        TaskSpec(object_id=obj, surface_id=surface)
        for surface in SURFACES
        for obj in ("food_can", "cookie_box", "bread")
    )


@runtime_checkable
class ShakeBenchTask(Protocol):
    """Common task interface. Poses use robot-base coordinates and xyzw quats.

    reset/step return public observations; get_task_context returns static
    geometry and identity; get_metrics is evaluator-only, never policy input.
    """

    def reset(self) -> Mapping[str, Any]: ...
    def step(self, action) -> tuple[Mapping[str, Any], float, bool, dict]: ...
    @property
    def action_spec(self): ...
    @property
    def policy_observation_keys(self) -> tuple[str, ...]: ...
    def observation_contract(self) -> dict[str, Any]: ...
    def get_task_context(self) -> dict[str, Any]: ...
    def get_metrics(self, *, update=False) -> dict[str, Any]: ...
    def close(self) -> None: ...


def make_task_env(task: Mapping[str, Any] | TaskSpec | None = None, **kwargs) -> ShakeBenchTask:
    """Construct the selected task; task-specific assets stay behind this seam."""
    from robosuite.environments.manipulation.vibration_pick_place import VibrationPickPlace

    return VibrationPickPlace(task=TaskSpec.from_mapping(task), **kwargs)


def task_initial_yaw_rad(spec: TaskSpec) -> float:
    return math.pi / 2.0 if spec.object_id == "cookie_box" else 0.0


def task_env_kwargs(state: Mapping[str, Any]) -> dict:
    """Resolve state initialization; reject unsupported poses rather than ignore them."""
    import numpy as np

    spec = TaskSpec.from_mapping(state["task"])
    xy = np.asarray(state.get("object_xy_m"), dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("object_xy_m must be a finite two-vector")
    yaw = task_initial_yaw_rad(spec)
    expected_pose = [
        *xy,
        0.03 - OBJECT_SUPPORT[spec.object_id][0],
        0.0,
        0.0,
        math.sin(yaw / 2.0),
        math.cos(yaw / 2.0),
    ]
    pose = np.asarray(state.get("object_pose_worktable"), dtype=float)
    velocity = np.asarray(state.get("object_initial_velocity"), dtype=float)
    if (
        pose.shape != (7,)
        or not np.allclose(pose, expected_pose, atol=1e-12, rtol=0)
        or velocity.shape != (6,)
        or np.any(velocity != 0)
        or not math.isclose(float(state.get("object_yaw_rad", float("nan"))), yaw, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("task states require their registered upright pose, yaw and zero velocity")
    return {"task": spec, "object_start_xy": tuple(xy), "object_start_yaw_rad": yaw}


def legacy_oracle_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the existing Can-named oracle wire format without extra truth."""
    return {("can_" + key[7:] if key.startswith("object_") else key): value for key, value in observation.items()}
