"""Industrial ShakeBench arena and canonical isolated worktable model."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from robosuite.models.arenas.arena import Arena
from robosuite.utils.mjcf_utils import array_to_string, new_geom, string_to_array, xml_path_completion
from robosuite.utils.shakebench_isolator import (
    AXES,
    CANONICAL_WORKTABLE_DIMENSIONS_M,
    DEFAULT_ISOLATOR_CONFIG,
    IsolatorConfig,
    IsolatorParameters,
    derive_isolator_parameters,
)
from robosuite.utils.shakebench_scene import (
    SceneVisualConfig,
    _set_scene_visual_alpha,
    augment_scene_mjcf,
    load_scene_visual_config,
)

WORKTABLE_BODY_NAME = "worktable"
WORKTABLE_COLLISION_GEOM_NAME = "table_collision"
WORKTABLE_VISUAL_GEOM_NAME = "table_visual"
WORKTABLE_TOP_SITE_NAME = "table_top"
ISOLATED_WORKTABLE_ROLE = "isolated_worktable"
ISOLATOR_JOINT_NAMES = {axis: f"isolator_{axis}" for axis in AXES}

TARGET_CONTAINER_CENTER_XY_M = (-0.10, 0.17)
TARGET_CONTAINER_OUTER_XY_M = (0.18, 0.16)
TARGET_CONTAINER_WALL_THICKNESS_M = 0.008
TARGET_CONTAINER_INNER_XY_M = (0.164, 0.144)
TARGET_CONTAINER_WALL_HEIGHT_M = 0.035
TARGET_CONTAINER_BOTTOM_THICKNESS_M = 0.012

CANONICAL_TARGET_CONTAINER = {
    "center_xy_m": TARGET_CONTAINER_CENTER_XY_M,
    "outer_xy_m": TARGET_CONTAINER_OUTER_XY_M,
    "wall_thickness_m": TARGET_CONTAINER_WALL_THICKNESS_M,
    "inner_xy_m": TARGET_CONTAINER_INNER_XY_M,
    "wall_height_m": TARGET_CONTAINER_WALL_HEIGHT_M,
    "bottom_thickness_m": TARGET_CONTAINER_BOTTOM_THICKNESS_M,
}


class ShakeBenchArenaError(ValueError):
    """Raised when a ShakeBench arena cannot satisfy its explicit contract."""


def _vector(name: str, value: Iterable[float], length: int, *, nonnegative: bool = False) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ShakeBenchArenaError(f"{name} must contain {length} finite values")
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ShakeBenchArenaError(f"{name} must contain {length} finite values") from exc
    if array.size != length or not np.all(np.isfinite(array)):
        raise ShakeBenchArenaError(f"{name} must contain {length} finite values")
    if nonnegative and np.any(array < 0.0):
        raise ShakeBenchArenaError(f"{name} must contain non-negative values")
    return tuple(float(item) for item in array)


def _fmt(values: Iterable[float]) -> str:
    return array_to_string(tuple(float(value) for value in values))


class ShakeBenchArena(Arena):
    """Industrial arena with an explicit, isolated canonical worktable.

    The worktable body origin is simultaneously its COM, principal-inertia
    frame, and isolator elastic centre.  The body is a direct world child in
    the source XML so :class:`ShakeBenchDeckXMLProcessor` can move it under a
    generated dynamic deck by the explicit ``isolated_worktable`` role.

    Args:
        table_full_size: Canonical tabletop dimensions.  Non-canonical sizes
            are rejected so visual resizing cannot silently change the Phase 03
            inertial contract.
        table_friction: MuJoCo friction triple for the tabletop collision
            geometry.  This is kept as an arena input for later contact
            probes; it does not affect the explicit inertial.
        table_offset: World position of the tabletop centre's top surface.
        isolator_config: Candidate six-axis natural frequencies, damping
            ratios, reference mass/inertia, and strict limits.
        visual: Whether the industrial display layer starts visible.  The
            geoms remain in the compiled model either way, so this switch
            cannot alter physics topology or traces.
        include_target_container: If true, add the Phase 04 target geometry
            to the isolated worktable.  It is false by default in Phase 03;
            :meth:`add_target_container` is the explicit task-assembly seam.
        xml: Arena asset path relative to ``robosuite/models/assets``.
    """

    def __init__(
        self,
        table_full_size=CANONICAL_WORKTABLE_DIMENSIONS_M,
        table_friction=(1.0, 0.005, 0.0001),
        table_offset=(0.0, 0.0, 0.8),
        isolator_config=None,
        visual=True,
        include_target_container=False,
        xml="arenas/shakebench_arena.xml",
        *,
        visual_layer=None,
        visuals_enabled=None,
        isolation_config=None,
        scene_config=None,
    ):
        if isolation_config is not None:
            if isolator_config is not None:
                raise ShakeBenchArenaError("isolator_config and isolation_config specify different inputs")
            isolator_config = isolation_config
        if visual_layer is not None:
            visual = visual_layer
        if visuals_enabled is not None:
            visual = visuals_enabled
        if not isinstance(visual, (bool, np.bool_)):
            raise ShakeBenchArenaError("visual must be boolean")

        dimensions = np.asarray(_vector("table_full_size", table_full_size, 3), dtype=float)
        canonical_dimensions = np.asarray(CANONICAL_WORKTABLE_DIMENSIONS_M, dtype=float)
        if not np.allclose(dimensions, canonical_dimensions, rtol=0.0, atol=1e-12):
            raise ShakeBenchArenaError(
                "ShakeBenchArena requires canonical table_full_size "
                f"{CANONICAL_WORKTABLE_DIMENSIONS_M}, got {tuple(dimensions)}"
            )
        self.table_full_size = dimensions
        self.table_half_size = dimensions / 2.0
        self.table_friction = _vector("table_friction", table_friction, 3, nonnegative=True)
        self.table_offset = _vector("table_offset", table_offset, 3)
        self.center_pos = np.asarray(self.table_offset, dtype=float) - np.asarray([0.0, 0.0, self.table_half_size[2]])
        self.isolator_config = self._coerce_isolator_config(isolator_config)
        self.isolator_parameters = derive_isolator_parameters(self.isolator_config)
        self.visual_layer_enabled = bool(visual)
        self._visual_rgba = {}
        self._visual_geom_names = []
        self._target_container_added = False
        self.target_container_geom_names = {}
        self.scene_config: SceneVisualConfig = (
            scene_config if isinstance(scene_config, SceneVisualConfig) else load_scene_visual_config(scene_config)
        )

        super().__init__(xml_path_completion(xml))

        self.table_body = self.worldbody.find(f"./body[@name='{WORKTABLE_BODY_NAME}']")
        if self.table_body is None:
            raise ShakeBenchArenaError(f"arena XML is missing body {WORKTABLE_BODY_NAME!r}")
        self.worktable_body = self.table_body
        self.worktable_body_name = WORKTABLE_BODY_NAME
        self.table_collision = self.table_body.find(f"./geom[@name='{WORKTABLE_COLLISION_GEOM_NAME}']")
        self.table_visual = self.table_body.find(f"./geom[@name='{WORKTABLE_VISUAL_GEOM_NAME}']")
        self.table_top = self.table_body.find(f"./site[@name='{WORKTABLE_TOP_SITE_NAME}']")
        if self.table_collision is None or self.table_visual is None or self.table_top is None:
            raise ShakeBenchArenaError("arena XML is missing the canonical tabletop collision/visual/site handles")
        self.isolator_joints = {}
        for axis, joint_name in ISOLATOR_JOINT_NAMES.items():
            joint = self.table_body.find(f"./joint[@name='{joint_name}']")
            if joint is None:
                raise ShakeBenchArenaError(f"arena XML is missing isolator joint {joint_name!r}")
            self.isolator_joints[axis] = joint

        self._refresh_visual_geom_names()
        self.configure_location()
        self.configure_isolator(self.isolator_config)
        self.scene_inventory = augment_scene_mjcf(self, self.scene_config)
        self._refresh_visual_geom_names()
        if include_target_container:
            self.add_target_container()
        self.set_visual_layer(self.visual_layer_enabled)

    @staticmethod
    def _coerce_isolator_config(config) -> IsolatorConfig:
        if config is None:
            return DEFAULT_ISOLATOR_CONFIG
        if isinstance(config, IsolatorConfig):
            return config
        if isinstance(config, Mapping):
            payload = dict(config)
            payload.pop("axis_order", None)
            payload.pop("axes", None)
            return IsolatorConfig(**payload)
        raise ShakeBenchArenaError("isolator_config must be an IsolatorConfig or mapping")

    def _refresh_visual_geom_names(self) -> None:
        self._visual_geom_names = [
            geom.get("name")
            for geom in self.table_body.findall("./geom")
            if geom.get("group") == "1" and geom.get("name") is not None
        ]

    def configure_location(self) -> None:
        """Place the floor, COM-origin worktable and tabletop site."""

        self.floor.set("pos", _fmt(self.bottom_pos))
        self.table_body.set("pos", _fmt(self.center_pos))
        self.table_collision.set("size", _fmt(self.table_half_size))
        self.table_collision.set("friction", _fmt(self.table_friction))
        self.table_visual.set("size", _fmt(self.table_half_size))
        self.table_top.set("pos", _fmt((0.0, 0.0, self.table_half_size[2])))

    def add_table_mat(self) -> None:
        """Install a flush 3 mm rigid rubber layer with an explicit contact geom.

        The total supported mass and tabletop height stay fixed. Only the mat
        is paired with the manipulated object, avoiding duplicate constraints
        from the underlying metal. It moves rigidly with the isolated table.
        """
        from robosuite.utils.shakebench_tasks import MAT_TEXTURE_PATH, MAT_VISUAL_RGBA

        if self.table_body.find("./geom[@name='table_mat_collision']") is not None:
            raise ShakeBenchArenaError("table mat already installed")
        half = self.table_half_size
        position = (0.0, 0.0, half[2] - 0.0015)
        self.object_support_geom = ET.SubElement(
            self.table_body,
            "geom",
            {
                "name": "table_mat_collision",
                "type": "box",
                "group": "0",
                "size": _fmt((half[0], half[1], 0.0015)),
                "pos": _fmt(position),
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
                "rgba": "0 0 0 0",
            },
        )
        ET.SubElement(
            self.asset,
            "texture",
            {
                "name": "shakebench_task_mat_felt",
                "type": "2d",
                "file": xml_path_completion(MAT_TEXTURE_PATH),
            },
        )
        ET.SubElement(
            self.asset,
            "material",
            {
                "name": "shakebench_task_mat",
                "rgba": _fmt(MAT_VISUAL_RGBA),
                "texture": "shakebench_task_mat_felt",
                "texrepeat": "3 3",
                "texuniform": "false",
                "specular": "0.12",
                "shininess": "0.08",
                "reflectance": "0",
            },
        )
        ET.SubElement(
            self.table_body,
            "geom",
            {
                "name": "table_mat_visual",
                "type": "box",
                "group": "1",
                "size": _fmt((half[0], half[1], 0.0015)),
                "pos": _fmt((0.0, 0.0, position[2] + 0.00015)),
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
                "material": "shakebench_task_mat",
                "rgba": _fmt(MAT_VISUAL_RGBA),
            },
        )
        # A fine bound edge makes the textile layer readable at oblique views.
        for axis in (0, 1):
            for sign in (-1, 1):
                start = [-half[0] + 0.006, -half[1] + 0.006, half[2] + 0.0004]
                end = [half[0] - 0.006, half[1] - 0.006, half[2] + 0.0004]
                start[axis] = end[axis] = sign * (half[axis] - 0.006)
                ET.SubElement(
                    self.table_body,
                    "geom",
                    {
                        "name": f"table_mat_binding_{axis}_{sign}_visual",
                        "type": "capsule",
                        "group": "1",
                        "fromto": _fmt((*start, *end)),
                        "size": "0.00065",
                        "rgba": "0.50 0.59 0.55 1",
                        "contype": "0",
                        "conaffinity": "0",
                        "mass": "0",
                    },
                )
        self._refresh_visual_geom_names()
        self.set_visual_layer(self.visual_layer_enabled)

    def configure_isolator(self, config=None) -> IsolatorParameters:
        """Apply a candidate's derived ``k/c/springref`` to the six joints."""

        self.isolator_config = self._coerce_isolator_config(config)
        parameters = derive_isolator_parameters(self.isolator_config)
        self.isolator_parameters = parameters
        inertial = self.table_body.find("./inertial")
        if inertial is None or len(self.table_body.findall("./inertial")) != 1:
            raise ShakeBenchArenaError("worktable must have exactly one explicit direct inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", format(parameters.mass_kg, ".17g"))
        inertial.set("diaginertia", _fmt(parameters.inertia_kg_m2))
        limits = self.isolator_config.limits
        for index, axis in enumerate(AXES):
            joint = self.isolator_joints[axis]
            joint.set("pos", "0 0 0")
            joint.set("axis", _fmt(np.eye(3)[index % 3]))
            joint.set("stiffness", format(parameters.stiffness[index], ".17g"))
            joint.set("damping", format(parameters.damping[index], ".17g"))
            joint.set("springref", format(parameters.springref[index], ".17g"))
            joint.set("limited", "true")
            joint.set("range", _fmt((-limits[index], limits[index])))
            expected_type = "slide" if index < 3 else "hinge"
            if joint.get("type") != expected_type:
                raise ShakeBenchArenaError(f"isolator joint {joint.get('name')!r} must be type {expected_type!r}")
        return parameters

    @property
    def table_top_abs(self) -> np.ndarray:
        """Absolute world position of the tabletop surface site."""

        return np.asarray(self.bottom_pos, dtype=float) + np.asarray(self.table_offset, dtype=float)

    @property
    def isolated_worktable_role(self) -> str:
        """Stable role name consumed by the Phase 02 deck processor."""

        return ISOLATED_WORKTABLE_ROLE

    @property
    def deck_body_handles(self) -> dict[str, str]:
        """Return the explicit role-to-body mapping for deck assembly."""

        handles = dict(self.scene_inventory.role_handles)
        handles[ISOLATED_WORKTABLE_ROLE] = self.worktable_body_name
        return handles

    @property
    def role_handles(self) -> dict[str, str]:
        """Alias for :attr:`deck_body_handles`."""

        return self.deck_body_handles

    def make_deck_processor(self, config=None, *, body_handles=None, required_roles=None):
        """Build the Phase 02 processor with this arena's worktable handle.

        Additional roles (for example an explicit Panda base) may be supplied
        by the task assembly.  No body name is inferred by this helper.
        """

        from robosuite.utils.shakebench_deck import make_deck_xml_processor

        handles = self.deck_body_handles
        if body_handles is not None:
            if not isinstance(body_handles, Mapping):
                raise ShakeBenchArenaError("body_handles must be a role-to-body mapping")
            for role, body_name in body_handles.items():
                if role in handles and handles[role] != body_name:
                    raise ShakeBenchArenaError(f"body role {role!r} is already bound to {handles[role]!r}")
                if body_name in handles.values() and role not in handles:
                    raise ShakeBenchArenaError(f"body {body_name!r} is already assigned to an arena role")
                handles[role] = body_name
        if required_roles is None:
            required_roles = (ISOLATED_WORKTABLE_ROLE,)
        return make_deck_xml_processor(
            config=config,
            body_handles=handles,
            required_roles=required_roles,
        )

    def process_deck_xml(self, xml_string: str | None = None, config=None, **kwargs) -> str:
        """Apply :meth:`make_deck_processor` to this arena or a merged task XML."""

        source = self.get_xml() if xml_string is None else xml_string
        return self.make_deck_processor(config=config, **kwargs)(source)

    @property
    def target_container_spec(self) -> dict[str, Any]:
        """Return the frozen Phase 04 target-container geometry contract."""

        return {
            "center_xy_m": list(TARGET_CONTAINER_CENTER_XY_M),
            "outer_xy_m": list(TARGET_CONTAINER_OUTER_XY_M),
            "wall_thickness_m": TARGET_CONTAINER_WALL_THICKNESS_M,
            "inner_xy_m": list(TARGET_CONTAINER_INNER_XY_M),
            "wall_height_m": TARGET_CONTAINER_WALL_HEIGHT_M,
            "bottom_thickness_m": TARGET_CONTAINER_BOTTOM_THICKNESS_M,
            "body_name": self.worktable_body_name,
            "rigid_assembly": "isolated_worktable",
        }

    def add_target_container(
        self,
        center_xy_m: Iterable[float] = TARGET_CONTAINER_CENTER_XY_M,
        *,
        friction=(0.30, 0.005, 0.0001),
        add_visual=True,
        visual_style="tray",
    ) -> dict[str, str]:
        """Add the optional bottom and four walls to the isolated assembly.

        The geoms are direct children of the explicit-inertial worktable
        body.  Consequently they add no free joint and no hidden mass.  This
        is an assembly seam for Phase 04, not a task success evaluator.
        """

        if visual_style not in {"tray", "basket"}:
            raise ShakeBenchArenaError("target visual_style must be tray or basket")
        if self._target_container_added:
            raise ShakeBenchArenaError("target container has already been added")
        center = _vector("center_xy_m", center_xy_m, 2)
        friction = _vector("friction", friction, 3, nonnegative=True)
        outer_x, outer_y = TARGET_CONTAINER_OUTER_XY_M
        inner_x, inner_y = TARGET_CONTAINER_INNER_XY_M
        wall = TARGET_CONTAINER_WALL_THICKNESS_M
        wall_height = TARGET_CONTAINER_WALL_HEIGHT_M
        bottom_thickness = TARGET_CONTAINER_BOTTOM_THICKNESS_M
        table_top_z = float(self.table_half_size[2])
        bottom_z = table_top_z + bottom_thickness / 2.0
        wall_z = table_top_z + bottom_thickness + wall_height / 2.0
        x_half = outer_x / 2.0
        y_half = outer_y / 2.0
        wall_half = wall / 2.0
        x_wall_offset = x_half - wall_half
        y_wall_offset = y_half - wall_half
        elements = {
            "bottom": new_geom(
                name="target_container_bottom",
                type="box",
                size=(x_half, y_half, bottom_thickness / 2.0),
                pos=(center[0], center[1], bottom_z),
                group=0,
                friction=friction,
            ),
            "wall_xneg": new_geom(
                name="target_container_wall_xneg",
                type="box",
                size=(wall_half, y_half, wall_height / 2.0),
                pos=(center[0] - x_wall_offset, center[1], wall_z),
                group=0,
                friction=friction,
            ),
            "wall_xpos": new_geom(
                name="target_container_wall_xpos",
                type="box",
                size=(wall_half, y_half, wall_height / 2.0),
                pos=(center[0] + x_wall_offset, center[1], wall_z),
                group=0,
                friction=friction,
            ),
            "wall_yneg": new_geom(
                name="target_container_wall_yneg",
                type="box",
                size=(inner_x / 2.0, wall_half, wall_height / 2.0),
                pos=(center[0], center[1] - y_wall_offset, wall_z),
                group=0,
                friction=friction,
            ),
            "wall_ypos": new_geom(
                name="target_container_wall_ypos",
                type="box",
                size=(inner_x / 2.0, wall_half, wall_height / 2.0),
                pos=(center[0], center[1] + y_wall_offset, wall_z),
                group=0,
                friction=friction,
            ),
        }
        for element in elements.values():
            self.table_body.append(element)
        self.target_container_geom_names = {key: element.get("name") for key, element in elements.items()}
        if add_visual and visual_style == "basket":
            self._add_basket_visuals(center)
        elif add_visual:
            for key, element in elements.items():
                visual_name = f"{element.get('name')}_visual"
                visual = new_geom(
                    name=visual_name,
                    type="box",
                    size=string_to_array(element.get("size")),
                    pos=string_to_array(element.get("pos")),
                    group=1,
                    conaffinity=0,
                    contype=0,
                    rgba=(0.12, 0.14, 0.16, 0.78),
                )
                self.table_body.append(visual)
                self.target_container_geom_names[f"{key}_visual"] = visual_name
        self._target_container_added = True
        self._refresh_visual_geom_names()
        self.set_visual_layer(self.visual_layer_enabled)
        return dict(self.target_container_geom_names)

    def _add_basket_visuals(self, center) -> None:
        """Rounded rim and wire sides inside the existing shallow-box envelope.

        The five original solid geoms remain the collision approximation. All
        basket details are massless visuals and do not alter target tolerances.
        """
        cx, cy = center
        hx, hy = (value / 2 for value in TARGET_CONTAINER_OUTER_XY_M)
        floor = float(self.table_half_size[2]) + TARGET_CONTAINER_BOTTOM_THICKNESS_M
        top = floor + TARGET_CONTAINER_WALL_HEIGHT_M
        rgba = "0.82 0.84 0.80 1"
        ET.SubElement(
            self.asset,
            "material",
            {
                "name": "shakebench_basket_coated_metal",
                "rgba": rgba,
                "specular": "0.3",
                "shininess": "0.22",
                "reflectance": "0.05",
            },
        )

        def add(name, attributes):
            name = f"target_basket_{name}_visual"
            ET.SubElement(
                self.table_body,
                "geom",
                {
                    "name": name,
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                    "mass": "0",
                    "material": "shakebench_basket_coated_metal",
                    "rgba": rgba,
                    **attributes,
                },
            )
            self.target_container_geom_names[name] = name

        def wire(name, start, end, radius):
            add(name, {"type": "capsule", "fromto": _fmt((*start, *end)), "size": str(radius)})

        add(
            "base", {"type": "box", "size": _fmt((hx - 0.002, hy - 0.002, 0.006)), "pos": _fmt((cx, cy, floor - 0.006))}
        )
        # Two rounded horizontal bands and evenly spaced vertical wires.
        for axis, (half, along) in enumerate(((hx, hy), (hy, hx))):
            for sign in (-1, 1):
                fixed = (cx, cy)[axis] + sign * (half - 0.004)
                for band, z, radius in (("rim", top - 0.004, 0.004), ("lower", floor + 0.002, 0.002)):
                    start = [cx - hx + 0.004, cy - hy + 0.004, z]
                    end = [cx + hx - 0.004, cy + hy - 0.004, z]
                    start[axis] = end[axis] = fixed
                    wire(f"{axis}_{sign}_{band}", start, end, radius)
                for index, offset in enumerate(np.linspace(-along + 0.01, along - 0.01, 12)):
                    point = [cx, cy, floor + 0.002]
                    point[axis] = fixed
                    point[1 - axis] += offset
                    wire(f"{axis}_{sign}_upright_{index}", point, [*point[:2], top - 0.004], 0.0014)
        # Quiet grip sleeves distinguish the basket from a plain target frame.
        for sign in (-1, 1):
            add(
                f"grip_{sign}",
                {
                    "type": "capsule",
                    "size": "0.0044",
                    "rgba": "0.35 0.43 0.40 1",
                    "fromto": _fmt(
                        (
                            cx - 0.026,
                            cy + sign * (hy - 0.004),
                            top - 0.004,
                            cx + 0.026,
                            cy + sign * (hy - 0.004),
                            top - 0.004,
                        )
                    ),
                },
            )

    @property
    def target_container_added(self) -> bool:
        return self._target_container_added

    def set_visual_layer(self, enabled: bool) -> None:
        """Toggle display alpha while retaining the exact physics model."""

        if not isinstance(enabled, (bool, np.bool_)):
            raise ShakeBenchArenaError("enabled must be boolean")
        for name in self._visual_geom_names:
            geom = self.table_body.find(f"./geom[@name='{name}']")
            if geom is None:
                continue
            if name not in self._visual_rgba:
                self._visual_rgba[name] = geom.get("rgba")
            original = self._visual_rgba[name]
            if enabled:
                if original is None:
                    geom.attrib.pop("rgba", None)
                else:
                    geom.set("rgba", original)
            else:
                geom.set("rgba", "1 1 1 0")
        self.visual_layer_enabled = bool(enabled)
        _set_scene_visual_alpha(self, bool(enabled))

    @property
    def visual_geom_names(self) -> tuple[str, ...]:
        """Names of all display-only worktable geoms."""

        return tuple(self._visual_geom_names)

    def audit_compiled_model(self, sim_or_model: Any, *, tolerance: float = 1e-10) -> dict[str, Any]:
        """Assert the compiled worktable mass, joints, limits and visual layer."""

        import mujoco

        model = getattr(sim_or_model, "model", sim_or_model)
        raw_model = getattr(model, "_model", model)
        body_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, self.worktable_body_name))
        if body_id < 0:
            raise ShakeBenchArenaError(f"compiled model is missing body {self.worktable_body_name!r}")
        if not np.isclose(raw_model.body_mass[body_id], self.isolator_parameters.mass_kg, rtol=0.0, atol=tolerance):
            raise ShakeBenchArenaError("compiled worktable mass differs from explicit inertial")
        if not np.allclose(
            raw_model.body_inertia[body_id], self.isolator_parameters.inertia_kg_m2, rtol=0.0, atol=tolerance
        ):
            raise ShakeBenchArenaError("compiled worktable inertia differs from explicit inertial")
        if not np.allclose(raw_model.body_ipos[body_id], (0.0, 0.0, 0.0), rtol=0.0, atol=tolerance):
            raise ShakeBenchArenaError("compiled worktable COM is not at its body origin")

        joint_audit = {}
        for index, axis in enumerate(AXES):
            name = ISOLATOR_JOINT_NAMES[axis]
            joint_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise ShakeBenchArenaError(f"compiled model is missing joint {name!r}")
            expected_type = mujoco.mjtJoint.mjJNT_SLIDE if index < 3 else mujoco.mjtJoint.mjJNT_HINGE
            if int(raw_model.jnt_type[joint_id]) != int(expected_type):
                raise ShakeBenchArenaError(f"compiled joint {name!r} has the wrong type")
            if not np.allclose(raw_model.jnt_axis[joint_id], np.eye(3)[index % 3], rtol=0.0, atol=tolerance):
                raise ShakeBenchArenaError(f"compiled joint {name!r} has the wrong axis")
            if not np.allclose(raw_model.jnt_pos[joint_id], (0.0, 0.0, 0.0), rtol=0.0, atol=tolerance):
                raise ShakeBenchArenaError(f"compiled joint {name!r} does not pass through the worktable origin")
            if not np.isclose(
                raw_model.jnt_stiffness[joint_id],
                self.isolator_parameters.stiffness[index],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ShakeBenchArenaError(f"compiled joint {name!r} stiffness differs from derived k")
            dof_id = int(raw_model.jnt_dofadr[joint_id])
            if not np.isclose(
                raw_model.dof_damping[dof_id],
                self.isolator_parameters.damping[index],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ShakeBenchArenaError(f"compiled joint {name!r} damping differs from derived c")
            qpos_id = int(raw_model.jnt_qposadr[joint_id])
            if not np.isclose(
                raw_model.qpos_spring[qpos_id],
                self.isolator_parameters.springref[index],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ShakeBenchArenaError(f"compiled joint {name!r} springref differs from preload")
            limit = self.isolator_config.limits[index]
            if not bool(raw_model.jnt_limited[joint_id]):
                raise ShakeBenchArenaError(f"compiled joint {name!r} is missing its fail-closed limit")
            if not np.allclose(raw_model.jnt_range[joint_id], (-limit, limit), rtol=0.0, atol=tolerance):
                raise ShakeBenchArenaError(f"compiled joint {name!r} range differs from its safety limit")
            joint_audit[axis] = {
                "name": name,
                "id": joint_id,
                "type": "slide" if index < 3 else "hinge",
                "axis": np.asarray(raw_model.jnt_axis[joint_id], dtype=float).tolist(),
                "stiffness": float(raw_model.jnt_stiffness[joint_id]),
                "damping": float(raw_model.dof_damping[dof_id]),
                "springref": float(raw_model.qpos_spring[qpos_id]),
                "range": np.asarray(raw_model.jnt_range[joint_id], dtype=float).tolist(),
                "limited": bool(raw_model.jnt_limited[joint_id]),
            }
        visual_audit = {}
        for name in self.visual_geom_names:
            geom_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if geom_id < 0:
                raise ShakeBenchArenaError(f"compiled model is missing visual geom {name!r}")
            if int(raw_model.geom_contype[geom_id]) != 0 or int(raw_model.geom_conaffinity[geom_id]) != 0:
                raise ShakeBenchArenaError(f"visual geom {name!r} contributes to contact")
            visual_audit[name] = {
                "id": geom_id,
                "contype": int(raw_model.geom_contype[geom_id]),
                "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
            }
        collision_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, WORKTABLE_COLLISION_GEOM_NAME))
        if collision_id < 0:
            raise ShakeBenchArenaError("compiled model is missing tabletop collision geometry")
        result = {
            "body_name": self.worktable_body_name,
            "body_id": body_id,
            "mass_kg": float(raw_model.body_mass[body_id]),
            "com_m": np.asarray(raw_model.body_ipos[body_id], dtype=float).tolist(),
            "inertia_kg_m2": np.asarray(raw_model.body_inertia[body_id], dtype=float).tolist(),
            "joints": joint_audit,
            "visual_geoms": visual_audit,
            "collision_geom": {
                "name": WORKTABLE_COLLISION_GEOM_NAME,
                "id": collision_id,
                "contype": int(raw_model.geom_contype[collision_id]),
                "conaffinity": int(raw_model.geom_conaffinity[collision_id]),
            },
            "physics_signature": {
                "mass_kg": float(raw_model.body_mass[body_id]),
                "com_m": np.asarray(raw_model.body_ipos[body_id], dtype=float).tolist(),
                "inertia_kg_m2": np.asarray(raw_model.body_inertia[body_id], dtype=float).tolist(),
                "joint_names": [ISOLATOR_JOINT_NAMES[axis] for axis in AXES],
                "stiffness": list(self.isolator_parameters.stiffness),
                "damping": list(self.isolator_parameters.damping),
                "springref": list(self.isolator_parameters.springref),
            },
        }
        if int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, "deck")) >= 0:
            from robosuite.utils.shakebench_scene import audit_compiled_scene, scene_clearance_report

            result["scene"] = audit_compiled_scene(sim_or_model, self.scene_config).to_dict()
            if hasattr(sim_or_model, "data") or hasattr(sim_or_model, "_data"):
                result["scene_clearance"] = scene_clearance_report(sim_or_model, self.scene_config).to_dict()
        return result


__all__ = [
    "WORKTABLE_BODY_NAME",
    "WORKTABLE_COLLISION_GEOM_NAME",
    "WORKTABLE_VISUAL_GEOM_NAME",
    "WORKTABLE_TOP_SITE_NAME",
    "ISOLATED_WORKTABLE_ROLE",
    "ISOLATOR_JOINT_NAMES",
    "TARGET_CONTAINER_CENTER_XY_M",
    "TARGET_CONTAINER_OUTER_XY_M",
    "TARGET_CONTAINER_WALL_THICKNESS_M",
    "TARGET_CONTAINER_INNER_XY_M",
    "TARGET_CONTAINER_WALL_HEIGHT_M",
    "TARGET_CONTAINER_BOTTOM_THICKNESS_M",
    "CANONICAL_TARGET_CONTAINER",
    "ShakeBenchArenaError",
    "ShakeBenchArena",
]
