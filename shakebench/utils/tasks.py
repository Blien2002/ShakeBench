"""Task selection and public task identity, independent of rollout machinery.

The task set is a small set of objects on the bare metal worktable, one
object per episode.  Every entry records three kinds of number, and they are never
interchanged:

* **official** - the RoboCasa instance path, registry scale and license;
* **geometry** - the contact-geometry envelope measured from the compiled
  asset in the object frame (``tools/pickv2_design.py``);
* **design** - the task's own mass, friction pair and grasp plan.

Masses are per-object design values (a real mug, a real potato), not the
MuJoCo density-derived compile values.  Friction pairs are experimental
contact coefficients for the metal tabletop, not measured material data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

ROBOCASA_SOURCE_URL = "https://huggingface.co/datasets/robocasa/robocasa-assets"
ROBOCASA_LICENSE = "CC-BY-4.0"
ROBOCASA_ASSET_ROOT = "objects/robocasa"

#: object_id -> task entry.  ``support`` is the measured contact envelope in
#: the object frame as (lower_z, upper_z, support_radius) in metres.
OBJECTS = {
    "mug": {
        "asset": f"{ROBOCASA_ASSET_ROOT}/mug/mug_1/model.xml",
        "source": "RoboCasa Objaverse mug_1",
        "source_instance": "objaverse/mug/mug_1",
        "source_url": ROBOCASA_SOURCE_URL,
        "source_license": ROBOCASA_LICENSE,
        "asset_scale": (0.80, 0.80, 0.80),
        "official_asset_scale": (1.0, 1.0, 1.0),
        "object_class": "receptacle_with_handle",
        "friction_class": "glazed_ceramic",
        "table_mu": 0.25,
        "mass_kg": 0.30,
        "support": (-0.03615559794718225, 0.03615560148446003, 0.04600000287744883),
        "start_pose_support": (-0.03615749614935689, 0.03615771107858959, 0.046000650247714145),
        # Measured collision-only centre of mass, start pose, above the lowest point.
        "com_height_m": 0.028128,
        "start_quat_wxyz": (0.708068, -0.000035, 0.000020, 0.706144),
        "instruction": "mug",
        # The mesh origin is the full mug bbox centre; the cup body is 14.4 mm left of it.
        "grasp": {"hold_width_m": 0.0629, "pad_height_m": 0.035, "offset_xy_m": (-0.0144, 0.0)},
    },
    "apple": {
        "asset": f"{ROBOCASA_ASSET_ROOT}/apple/apple_0/model.xml",
        "source": "RoboCasa Objaverse apple_0",
        "source_instance": "objaverse/apple/apple_0",
        "source_url": ROBOCASA_SOURCE_URL,
        "source_license": ROBOCASA_LICENSE,
        "asset_scale": (0.85, 0.85, 0.85),
        "official_asset_scale": (0.9, 0.9, 0.9),
        "object_class": "fruit",
        "friction_class": "fruit_skin",
        "table_mu": 0.30,
        "mass_kg": 0.16,
        "support": (-0.03211376857691277, 0.03207727074735714, 0.033469874213988324),
        "start_pose_support": (-0.031194843322826345, 0.03227772018364363, 0.033279964464956874),
        # Measured collision-only centre of mass, start pose, above the lowest point.
        "com_height_m": 0.027855,
        "start_quat_wxyz": (0.998657, -0.020171, 0.047723, 0.000156),
        "instruction": "apple",
        "grasp": {"hold_width_m": 0.0644, "pad_height_m": 0.028, "offset_xy_m": (0.0, 0.0)},
    },
    # The same can in two declared start poses: standing, and lying on its side.
    "can1": {
        "asset": "robosuite.models.objects.CanObject",
        "source": "robosuite CanObject (can.stl)",
        "source_instance": "robosuite/models/assets/objects/meshes/can.stl",
        "source_url": "https://github.com/ARISE-Initiative/robosuite",
        "source_license": "MIT",
        "asset_scale": (1.0, 1.0, 1.0),
        "official_asset_scale": (1.0, 1.0, 1.0),
        "object_class": "receptacle",
        "friction_class": "tinplate",
        "table_mu": 0.20,
        "mass_kg": 0.40,
        "support": (-0.040297003330440104, 0.03970300217508332, 0.02509177806572465),
        "start_pose_support": (-0.0402970033304401, 0.03970300217508331, 0.02509177806572464),
        # Measured collision-only centre of mass, start pose, above the lowest point.
        "com_height_m": 0.040659,
        "start_quat_wxyz": (0.707107, 0.0, 0.0, 0.707107),
        "instruction": "food can",
        "grasp": {"hold_width_m": 0.0502, "pad_height_m": 0.0407, "offset_xy_m": (0.0, 0.0)},
    },
    "can2": {
        "asset": "robosuite.models.objects.CanObject",
        "source": "robosuite CanObject (can.stl)",
        "source_instance": "robosuite/models/assets/objects/meshes/can.stl",
        "source_url": "https://github.com/ARISE-Initiative/robosuite",
        "source_license": "MIT",
        "asset_scale": (1.0, 1.0, 1.0),
        "official_asset_scale": (1.0, 1.0, 1.0),
        "object_class": "receptacle",
        "friction_class": "tinplate",
        "table_mu": 0.20,
        "mass_kg": 0.40,
        "support": (-0.040297003330440104, 0.03970300217508332, 0.02509177806572465),
        # Lying on its side: the axis is horizontal, so the pads close above the
        # axis height where the centre of mass sits.
        "start_pose_support": (-0.024984, 0.025016001, 0.04628817),
        "com_height_m": 0.025043,
        "start_quat_wxyz": (0.707107, 0.0, -0.707107, 0.0),
        "instruction": "food can lying on its side",
        "grasp": {"hold_width_m": 0.0502, "pad_height_m": 0.0251, "offset_xy_m": (0.0, 0.0)},
    },
}

DEFAULT_OBJECT_ID = "mug"
#: The target container is a single fixed crate shared by every variant.
SURFACES = ("metal",)
TASK_SCHEMA_ID = "shakebench.task.v3"
TASK_VISUAL_REVISION = "three_objects_bare_metal_crate.v1"
GRASP_OPENING_ALLOWANCE_M = 0.010
#: Compiled Panda jaw travel (both fingers) in metres.
PANDA_JAW_LIMIT_M = 0.080
#: A hold needs this much unused jaw travel beyond the object's widest point.
GRASP_JAW_MARGIN_MIN_M = 0.005
UPRIGHT_AXIS_COSINE_MIN = 0.95


@dataclass(frozen=True)
class TaskSpec:
    """Serializable task selector; invalid combinations fail before compilation."""

    task_type: str = "pick_place"
    object_id: str = DEFAULT_OBJECT_ID

    def __post_init__(self):
        if not isinstance(self.task_type, str) or self.task_type != "pick_place":
            raise ValueError(f"unsupported task_type: {self.task_type!r}")
        if not isinstance(self.object_id, str) or self.object_id not in OBJECTS:
            raise ValueError(f"unsupported object_id: {self.object_id!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | TaskSpec | None) -> TaskSpec:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"task_type", "object_id"}:
            raise ValueError(
                "task must contain exactly task_type and object_id; "
                "surface_id belongs to the retired two-surface schema "
                f"({TASK_SCHEMA_ID})"
            )
        return cls(**dict(value))

    def to_dict(self):
        return asdict(self)

    @property
    def variant_id(self):
        return f"{self.task_type}.{self.object_id}"

    @property
    def surface_id(self):
        """Every variant uses the bare metal worktable."""

        return SURFACES[0]

    @property
    def table_sliding_mu(self):
        return float(OBJECTS[self.object_id]["table_mu"])

    @property
    def target_sliding_mu(self):
        # Experimental target coefficients are independent of the container's
        # visual material and match the tabletop pair for this object.
        return self.table_sliding_mu

    @property
    def object_mass_kg(self):
        return float(OBJECTS[self.object_id]["mass_kg"])

    @property
    def start_quat_wxyz(self):
        quat = tuple(float(value) for value in OBJECTS[self.object_id]["start_quat_wxyz"])
        norm = math.sqrt(sum(value * value for value in quat))
        if norm <= 0.0:
            raise ValueError(f"{self.object_id}: start quaternion must be non-zero")
        return tuple(value / norm for value in quat)

    @property
    def grasp(self):
        return self.grasp_plan()

    def grasp_plan(self, region="body"):
        """Select a grip; side_* modes declare a sideways initial pose as well."""
        if region == "side_handle" and self.object_id == "mug":
            # Collision-mesh rest pose: the cup lies on its side with the handle up.
            return {
                "hold_width_m": 0.010,
                "pad_height_m": 0.07936649368,
                "offset_xy_m": (-0.005, 0.02054313279),
                "point_object_m": (0.0, -0.041, 0.005),
                "preopening_m": 0.026,
                "pitch_rad": 0.0,
                "insertion_offset_object_m": (0.0, 0.0, 0.0),
                "start_quat_wxyz": (0.35318140201, -0.61258719399, -0.35318113371, 0.61258706633),
                "start_pose_support": (-0.04388438843, 0.04083958647, 0.05251948046),
                "com_height_m": 0.03446942146,
            }
        if region == "side_single_wall" and self.object_id == "mug":
            # The mug lies on its side with the mouth reachable; the pads straddle
            # the +x wall (one inside the cavity, one outside) after a sideways insert.
            return {
                "hold_width_m": 0.014,
                "pad_height_m": 0.04450628261,
                "offset_xy_m": (-0.01791543853, -0.01410397523),
                "point_object_m": (0.024, -0.003, 0.018),
                "preopening_m": 0.032,
                "pitch_rad": -math.radians(75),
                "insertion_offset_object_m": (0.0, 0.0, 0.050),
                "start_quat_wxyz": (0.68636894226, 0.17440077662, -0.68494945765, -0.17124991119),
                "start_pose_support": (-0.02478028593, 0.03842944253, 0.05657461336),
                "com_height_m": 0.02966551020,
            }
        if region == "body":
            return dict(OBJECTS[self.object_id]["grasp"])
        if region == "handle" and self.object_id == "mug":
            # Pinch the upper crossbar; the outer vertical segment slips under the mug torque.
            return {"hold_width_m": 0.013, "pad_height_m": 0.058, "offset_xy_m": (0.033, 0.0)}
        if region == "single_wall" and self.object_id == "mug":
            # Pinch the +x wall of the object: one pad in the cavity, one outside.
            return {
                "hold_width_m": 0.006,
                "pad_height_m": 0.055,
                "offset_xy_m": (-0.0144, 0.029),
                "preopening_m": 0.032,
            }
        raise ValueError(f"unsupported grasp region {region!r} for {self.object_id}")

    def contract(self, region="body"):
        entry = OBJECTS[self.object_id]
        grasp = self.grasp_plan(region)
        return {
            "schema_id": TASK_SCHEMA_ID,
            **self.to_dict(),
            "object_asset": entry["asset"],
            "object_source": entry["source"],
            "object_source_instance": entry["source_instance"],
            "object_source_url": entry["source_url"],
            "object_source_license": entry["source_license"],
            "object_asset_scale": list(entry["asset_scale"]),
            "object_official_registry_scale": list(entry["official_asset_scale"]),
            "object_class": entry["object_class"],
            "object_support_lower_upper_radius_m": [float(value) for value in entry["support"]],
            "object_friction_class": entry["friction_class"],
            "table_object_sliding_mu": self.table_sliding_mu,
            "target_object_sliding_mu": self.target_sliding_mu,
            "finger_object_sliding_mu": 1.0,
            "object_mass_kg": self.object_mass_kg,
            "object_start_quat_wxyz": list(grasp.get("start_quat_wxyz", self.start_quat_wxyz)),
            "object_com_height_m": float(grasp.get("com_height_m", entry["com_height_m"])),
            "object_grasp": {
                "hold_width_m": float(grasp["hold_width_m"]),
                "pad_height_m": float(grasp["pad_height_m"]),
                "offset_xy_m": [float(value) for value in grasp["offset_xy_m"]],
                **({"point_object_m": list(grasp["point_object_m"])} if "point_object_m" in grasp else {}),
                "jaw_limit_m": PANDA_JAW_LIMIT_M,
                "opening_allowance_m": GRASP_OPENING_ALLOWANCE_M,
                "pads_above_com": float(grasp["pad_height_m"])
                >= float(grasp.get("com_height_m", entry["com_height_m"])),
            },
            "task_visual_revision": TASK_VISUAL_REVISION,
            "table_surface": "bare brushed-steel worktable; the felt mat task was retired",
            "target_container": "single fixed crate shared by every variant",
            "surface_model": "flush rigid layer; fixed total worktable mass and top height",
            "inertia_model": "collision-geometry compile scaled to the design mass; visual and region geoms excluded",
            "mass_values_are": "per-object design masses, not density-derived compile values",
            "friction_values_are": "experimental contact coefficients for the metal tabletop, not measured material data",
            "stability_constraint": (
                "start pose is the measured free-settled rest pose; the grasp pads never close below "
                "the object centre of mass"
            ),
            "success_rule": (
                "majority of the collision geometry inside the crate footprint, resting on the crate "
                "floor, released; orientation and residual speed are reported but do not gate success"
            ),
            "success_semantics": "shakebench.task.v3 positional success evaluator",
            "qualification": "pending_task_variant_requalification",
        }


def make_task_object(spec: TaskSpec | None, *, name="task_object"):
    """Build the selected task object from its recorded asset."""

    from robosuite.models.objects import CanObject, MujocoXMLObject
    from shakebench.models import xml_path_completion

    if spec is None:
        return CanObject(name=name)
    entry = OBJECTS[spec.object_id]
    asset = entry["asset"]
    if asset.startswith("robosuite.models.objects."):
        # robosuite's native Can keeps its own mesh and inertial seam.
        return CanObject(name=name)
    return MujocoXMLObject(
        xml_path_completion(asset),
        name=name,
        joints=[dict(type="free", damping="0.0005")],
        obj_type="all",
        duplicate_collision_geoms=False,
    )


def task_variants() -> tuple[TaskSpec, ...]:
    """Return the frozen variants of the current task set."""

    return tuple(TaskSpec(object_id=object_id) for object_id in OBJECTS)


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

    from shakebench.environments.vibration_pick_place import VibrationPickPlace

    return VibrationPickPlace(task=TaskSpec.from_mapping(task), **kwargs)


def task_start_quat_wxyz(spec: TaskSpec) -> tuple[float, float, float, float]:
    return spec.start_quat_wxyz


def task_env_kwargs(state: Mapping[str, Any]) -> dict:
    """Resolve state initialization; reject unsupported poses rather than ignore them."""

    import numpy as np

    spec = TaskSpec.from_mapping(state["task"])
    xy = np.asarray(state.get("object_xy_m"), dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("object_xy_m must be a finite two-vector")
    expected_quat = np.asarray(state.get("object_start_quat_wxyz", spec.start_quat_wxyz), dtype=float)
    pose = np.asarray(state.get("object_pose_worktable"), dtype=float)
    velocity = np.asarray(state.get("object_initial_velocity"), dtype=float)
    if (
        expected_quat.shape != (4,)
        or not np.allclose(expected_quat, spec.start_quat_wxyz, atol=1e-6, rtol=0)
        or pose.shape != (7,)
        or not np.allclose(pose[:2], xy, atol=1e-12, rtol=0)
        or not np.allclose(pose[3:], expected_quat, atol=1e-6, rtol=0)
        or not np.isfinite(pose[2])
        or velocity.shape != (6,)
        or np.any(velocity != 0)
    ):
        raise ValueError("task states require their registered start pose and zero velocity")
    return {"task": spec, "object_start_xy": tuple(xy), "object_start_quat_wxyz": tuple(expected_quat)}


def grasp_opening_gate_m(spec: TaskSpec, region="body") -> float:
    """Return the jaw opening the oracle expects after a bilateral capture."""

    hold_width = float(spec.grasp_plan(region)["hold_width_m"])
    return min(hold_width + GRASP_OPENING_ALLOWANCE_M, PANDA_JAW_LIMIT_M - 0.002)


def validate_task_registry() -> None:
    """Fail closed on a registry that cannot be grasped or is mis-recorded."""

    for object_id, entry in OBJECTS.items():
        lower, upper, radius = (float(value) for value in entry["support"])
        if not lower < upper:
            raise ValueError(f"{object_id}: support bounds must be increasing")
        if radius <= 0.0:
            raise ValueError(f"{object_id}: support radius must be positive")
        if float(entry["mass_kg"]) <= 0.0:
            raise ValueError(f"{object_id}: design mass must be positive")
        if not 0.0 < float(entry["table_mu"]) <= 1.5:
            raise ValueError(f"{object_id}: table friction must be a plausible coefficient")
        quat = entry["start_quat_wxyz"]
        if len(quat) != 4 or not math.isclose(sum(value * value for value in quat), 1.0, abs_tol=1e-5):
            raise ValueError(f"{object_id}: start quaternion must be a unit quaternion")
        grasp = entry["grasp"]
        pad_height = float(grasp["pad_height_m"])
        if pad_height <= 0.0:
            raise ValueError(f"{object_id}: grasp pad height must be positive")
        if pad_height > (upper - lower):
            raise ValueError(f"{object_id}: grasp pad height must lie inside the object")
        hold_width = float(grasp["hold_width_m"])
        if hold_width + GRASP_JAW_MARGIN_MIN_M > PANDA_JAW_LIMIT_M:
            raise ValueError(
                f"{object_id}: hold width {hold_width:.4f} m leaves less than "
                f"{GRASP_JAW_MARGIN_MIN_M:.3f} m of Panda jaw margin"
            )
        if float(grasp["pad_height_m"]) < float(entry["com_height_m"]):
            raise ValueError(
                f"{object_id}: grasp pads at {grasp['pad_height_m']:.4f} m sit below the "
                f"{entry['com_height_m']:.4f} m centre of mass and would spin the object in the jaws"
            )
        if entry["asset"] is None:
            raise ValueError(f"{object_id}: asset is required")


validate_task_registry()

#: Compatibility name for callers that read the object support triple.
OBJECT_SUPPORT = {object_id: tuple(float(v) for v in entry["support"]) for object_id, entry in OBJECTS.items()}


__all__ = [
    "DEFAULT_OBJECT_ID",
    "GRASP_OPENING_ALLOWANCE_M",
    "OBJECTS",
    "OBJECT_SUPPORT",
    "PANDA_JAW_LIMIT_M",
    "SURFACES",
    "ShakeBenchTask",
    "TASK_SCHEMA_ID",
    "TASK_VISUAL_REVISION",
    "TaskSpec",
    "UPRIGHT_AXIS_COSINE_MIN",
    "grasp_opening_gate_m",
    "make_task_env",
    "make_task_object",
    "task_env_kwargs",
    "task_start_quat_wxyz",
    "task_variants",
    "validate_task_registry",
]
