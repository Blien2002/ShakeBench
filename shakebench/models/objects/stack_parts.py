"""Visual meshes, texture atlases and collision shapes for the stack_blocks task family.

Three free 50 mm blocks stack from bottom to top as blue, green and yellow. With a ``tenon`` joint the blue
block carries a square pyramid frustum and the green block has its matching tapered blind pocket underneath.
The green block carries a round frustum and the yellow block has a circular tapered blind pocket. Every
joint stays inside the block outline, so an assembled tower shows the same outer faces as ``plain``, where the
blocks are rounded cubes. The circular pocket rejects the corners of the blue frustum, preserving the order.

Visual meshes are unions of closed shells drawn in group 1 without contacts or mass. Each part has its own
1024 px atlas at 5 px/mm, baked by ``shakebench/scripts/generate_stack_textures.py`` from a solid wood model
through the same ``PART_RECTS`` mapping, so the grain stays continuous across edges and between shells.
``collision_shapes`` returns the matching convex decomposition.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string

from shakebench.models import xml_path_completion

PARTS = ("blue", "green", "yellow")
BLOCK_HALF_M = 0.025
BEVEL_RADIUS_M = 0.002
BEVEL_SEGMENTS = 4
FACE_SUBDIVISIONS = 12
MIN_CELL_M = 0.001
TENON_LENGTH_M = 0.012
JOINT_DEPTH_M = 0.015
PYRAMID_BASE_HALF_M = 0.012
PYRAMID_TOP_HALF_M = 0.005
PYRAMID_SUBDIVISIONS = (6, 4)  # along each edge, up each face
ROUND_SEGMENTS = 64  # maximum radial chord error is 0.018 mm at the pocket mouth
JOINT_GEOMETRY_VERSION = 3

# Both pockets continue their male taper to the blind depth. Clearances are per side for the square and radial
# for the round joint. A round clearance above 2.4 mm would compromise rejection of yellow on blue.
JOINT_SPECS = {
    "plain": {"type": "none"},
    "tenon": {"type": "tenon", "square_clearance_m": 0.0020, "round_clearance_m": 0.0024},
}
WRONG_ORDER_MIN_REST_M = 0.003  # the pocket must hold a yellow block on the blue tenon at least this far up

ATLAS_PX = 1024
PX_PER_M = 5000.0
PAD_PX = 8
TAPER_EXTENT_M = 0.0145
FINISH = {"specular": "0.28", "shininess": "0.42", "emission": "0.10"}

H, R = BLOCK_HALF_M, BEVEL_RADIUS_M
L, D = TENON_LENGTH_M, JOINT_DEPTH_M
S0, S1 = PYRAMID_BASE_HALF_M, PYRAMID_TOP_HALF_M
TAPER = (S0 - S1) / L  # horizontal inset per unit height, shared by the pyramid and the pocket
ALL_SIDES = frozenset((axis, sign) for axis in range(3) for sign in (-1, 1))
FACE_RECTS = {(0, 1): "px", (0, -1): "nx", (1, 1): "py", (1, -1): "ny", (2, 1): "pz", (2, -1): "nz"}
SIDES = ((0, 1, "px"), (0, -1, "nx"), (1, 1, "py"), (1, -1, "ny"))


def joint_geometry(spec):
    """Derived joint dimensions in metres for a JOINT_SPECS name or entry, or None for plain stacking."""
    spec = JOINT_SPECS[spec] if isinstance(spec, str) else spec
    if spec["type"] == "none":
        return None
    if spec["type"] != "tenon":
        raise ValueError(f"unknown joint type {spec['type']!r}")
    square, radial = (float(spec[k]) for k in ("square_clearance_m", "round_clearance_m"))
    for clearance in (square, radial):
        if not (0.0 < clearance and S0 + clearance <= TAPER_EXTENT_M and S0 + clearance - TAPER * D > 0.001):
            raise ValueError("the clearance lies outside the range the atlas and the pocket support")
    # The four square corners cannot enter a centred circular mouth until they are above this height.
    # Translating or rotating the square cannot reduce its minimum enclosing radius.
    rest = (S0 - (S0 + radial) / np.sqrt(2.0)) / TAPER
    if rest < WRONG_ORDER_MIN_REST_M:
        raise ValueError("the tapered pocket would no longer hold the yellow block clear of the blue seat")
    tolerance = np.arcsin(min(1.0, (S0 + square) / (S0 * np.sqrt(2.0)))) - np.pi / 4
    return {
        "version": JOINT_GEOMETRY_VERSION,
        "blue_green_shape": "square_frustum",
        "green_yellow_shape": "round_frustum",
        "tenon_length_m": L,
        "joint_depth_m": D,
        "square_clearance_m": square,
        "pyramid_base_half_m": S0,
        "pyramid_top_half_m": S1,
        "pyramid_height_m": L,
        "pyramid_face_angle_deg": float(np.degrees(np.arctan(1.0 / TAPER))),
        "taper_mouth_half_m": S0 + square,
        "taper_ceiling_half_m": S0 + square - TAPER * D,
        "pyramid_yaw_tolerance_deg": float(np.degrees(tolerance)),
        "round_base_radius_m": S0,
        "round_top_radius_m": S1,
        "round_clearance_m": radial,
        "round_mouth_radius_m": S0 + radial,
        "round_ceiling_radius_m": S0 + radial - TAPER * D,
        "round_yaw_unrestricted": True,
        "round_segments": ROUND_SEGMENTS,
        "yellow_on_blue_rest_height_m": rest,
    }


def _size(s_range, t_range):
    width = int(round((s_range[1] - s_range[0]) * PX_PER_M)) + 2 * PAD_PX
    height = int(round((t_range[1] - t_range[0]) * PX_PER_M)) + 2 * PAD_PX
    return width, height


def _rect(origin, normal_axis, sign, s_axis, t_axis, s_range, t_range, plane, kind):
    """Axis-aligned rect, sampled on the plane where the normal axis equals ``plane``."""
    return {
        "origin": origin,
        "size": _size(s_range, t_range),
        "normal_axis": normal_axis,
        "sign": sign,
        "s_axis": s_axis,
        "t_axis": t_axis,
        "s_range": s_range,
        "t_range": t_range,
        "plane": plane,
        "kind": kind,
    }


def _slope_rect(origin, axis, sign, base_half, base_z, inset, rise, s_range, t_range, kind):
    """Rect on a tapered face of side (axis, sign), with s along the base edge and t up the slope."""
    frame_origin, u, v = np.zeros(3), np.zeros(3), np.zeros(3)
    frame_origin[axis], frame_origin[2] = sign * base_half, base_z
    u[1 - axis] = 1.0
    v[axis], v[2] = -sign * inset, rise
    return {
        "origin": origin,
        "size": _size(s_range, t_range),
        "frame": (frame_origin, u, v / np.linalg.norm(v)),
        "s_range": s_range,
        "t_range": t_range,
        "kind": kind,
    }


def _round_rect(origin, radius, base_z, height, kind):
    """Unwrap a round tapered wall by base arc length and distance along its slope."""
    s_range, t_range = (0.0, 2 * np.pi * radius), (0.0, height * np.hypot(TAPER, 1.0))
    return {
        "origin": origin,
        "size": _size(s_range, t_range),
        "round_frame": (radius, base_z),
        "s_range": s_range,
        "t_range": t_range,
        "kind": kind,
    }


def _layouts():
    full, face = (-H, H), int(round(2 * H * PX_PER_M)) + 2 * PAD_PX
    row = 2 * face
    faces = {
        "px": _rect((0, 0), 0, 1, 1, 2, full, full, H, "side"),
        "nx": _rect((face, 0), 0, -1, 1, 2, full, full, -H, "side"),
        "py": _rect((2 * face, 0), 1, 1, 0, 2, full, full, H, "side"),
        "ny": _rect((0, face), 1, -1, 0, 2, full, full, -H, "side"),
        "pz": _rect((face, face), 2, 1, 0, 1, full, full, H, "end"),
        "nz": _rect((2 * face, face), 2, -1, 0, 1, full, full, -H, "end"),
    }

    blue = dict(faces)
    base, slope = (-S0, S0), (0.0, float(np.hypot(S0 - S1, L)))
    step = _size(base, slope)[0]
    for i, (axis, sign, tag) in enumerate(SIDES):
        blue[f"y_{tag}"] = _slope_rect((i * step, row), axis, sign, S0, H, S0 - S1, L, base, slope, "pyramid_face")
    top = (-S1 - 0.001, S1 + 0.001)
    blue["y_pz"] = _rect((4 * step + 8, row), 2, 1, 0, 1, top, top, H + L, "pyramid_top")

    green, mouth = dict(faces), S0 + JOINT_SPECS["tenon"]["square_clearance_m"]
    span, wall = (-TAPER_EXTENT_M, TAPER_EXTENT_M), (0.0, float(D * np.hypot(TAPER, 1.0)))
    step = _size(span, wall)[0]
    for i, (axis, sign, tag) in enumerate(SIDES):
        green[f"q_{tag}"] = _slope_rect(
            (i * step, row), axis, sign, mouth, -H, TAPER * D, D, span, wall, "tapered_wall"
        )
    ceiling = TAPER_EXTENT_M - TAPER * D + 0.0005
    green["q_nz"] = _rect(
        (4 * step + 8, row), 2, -1, 0, 1, (-ceiling, ceiling), (-ceiling, ceiling), -H + D, "tapered_ceiling"
    )
    green["round_side"] = _round_rect((0, row + 120), S0, H, L, "round_face")
    green["round_top"] = _rect((420, row + 120), 2, 1, 0, 1, top, top, H + L, "round_top")

    yellow = dict(faces)
    radius = S0 + JOINT_SPECS["tenon"]["round_clearance_m"]
    yellow["round_wall"] = _round_rect((0, row), radius, -H, D, "tapered_wall")
    yellow["round_ceiling"] = _rect(
        (500, row), 2, -1, 0, 1, (-ceiling, ceiling), (-ceiling, ceiling), -H + D, "tapered_ceiling"
    )
    return {"blue": blue, "green": green, "yellow": yellow}


def _check_layout(rects):
    boxes = [(*r["origin"], r["origin"][0] + r["size"][0], r["origin"][1] + r["size"][1]) for r in rects.values()]
    for i, a in enumerate(boxes):
        if min(a[:2]) < 0 or max(a[2:]) > ATLAS_PX:
            raise ValueError("atlas rect outside the texture")
        for b in boxes[i + 1 :]:
            if not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]):
                raise ValueError("atlas rects overlap")


PART_RECTS = _layouts()
for _part_rects in PART_RECTS.values():
    _check_layout(_part_rects)


def rect_coordinates(rect, points):
    """(s, t) in metres of part-frame points on a rect, for one point (3,) or many (N, 3)."""
    points = np.asarray(points, float)
    if "round_frame" in rect:
        radius, z = rect["round_frame"]
        return (
            np.mod(np.arctan2(points[..., 1], points[..., 0]), 2 * np.pi) * radius,
            (points[..., 2] - z) * np.hypot(TAPER, 1.0),
        )
    if "frame" in rect:
        origin, u, v = rect["frame"]
        offset = points - origin
        return offset @ u, offset @ v
    return points[..., rect["s_axis"]], points[..., rect["t_axis"]]


def atlas_uv(part, rect_name, point):
    """Atlas texcoord of a part-frame point projected onto a rect, with v measured from the image top."""
    rect = PART_RECTS[part][rect_name]
    s, t = rect_coordinates(rect, point)
    col = rect["origin"][0] + PAD_PX + (s - rect["s_range"][0]) * PX_PER_M
    row = rect["origin"][1] + PAD_PX + (rect["t_range"][1] - t) * PX_PER_M
    return np.array([col / ATLAS_PX, row / ATLAS_PX])


class _Mesh:
    def __init__(self):
        self.vertices, self.normals, self.uvs, self.faces = [], [], [], []

    def quad(self, points, normals, uvs):
        points, normals, uvs = np.asarray(points, float), np.asarray(normals, float), np.asarray(uvs, float)
        order = [0, 1, 2, 3]
        geometric = np.cross(points[1] - points[0], points[2] - points[0]) + np.cross(
            points[2] - points[0], points[3] - points[0]
        )
        if np.dot(geometric, normals.mean(axis=0)) < 0:
            order = [0, 3, 2, 1]
        base = len(self.vertices)
        for index in order:
            self.vertices.append(points[index])
            self.normals.append(normals[index] / np.linalg.norm(normals[index]))
            self.uvs.append(uvs[index])
        self.faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    def triangle(self, points, normals, uvs):
        points, normals = np.asarray(points, float), np.asarray(normals, float)
        order = [0, 1, 2]
        if np.dot(np.cross(points[1] - points[0], points[2] - points[0]), normals.mean(axis=0)) < 0:
            order = [0, 2, 1]
        base = len(self.vertices)
        for index in order:
            self.vertices.append(points[index])
            self.normals.append(normals[index] / np.linalg.norm(normals[index]))
            self.uvs.append(np.asarray(uvs[index], float))
        self.faces.append([base, base + 1, base + 2])

    def attributes(self):
        return {
            "vertex": array_to_string(np.asarray(self.vertices).ravel()),
            "normal": array_to_string(np.asarray(self.normals).ravel()),
            "texcoord": array_to_string(np.asarray(self.uvs).ravel()),
            "face": array_to_string(np.asarray(self.faces, dtype=int).ravel()),
        }


def _q(values):
    return np.round(np.asarray(values, float), 9)


def _axis_samples(required=()):
    """Grid lines along one axis. Bevel strips are tan spaced so fillet segments subtend equal angles."""
    strip = (H - R) + R * np.tan(np.linspace(0.0, np.pi / 4, BEVEL_SEGMENTS + 1))
    keep = np.unique(_q(np.concatenate([strip, -strip, np.asarray(required, float)])))
    uniform = np.linspace(-(H - R), H - R, FACE_SUBDIVISIONS + 1)
    extra = [u for u in uniform if np.min(np.abs(keep - u)) > MIN_CELL_M - 1e-9]
    return np.unique(_q(np.concatenate([keep, extra])))


def _span(samples, lo, hi):
    lo, hi = float(_q(lo)), float(_q(hi))
    return np.concatenate([[lo], samples[(samples > lo + 1e-9) & (samples < hi - 1e-9)], [hi]])


def _box(mesh, part, lo, hi, rounded, rects, samples, skip=None):
    """Closed box shell whose edges between two rounded sides get BEVEL_RADIUS_M fillets.

    A sharp side stays planar. Where the fillet of two rounded neighbours ends on it, its outline follows the
    fillet arc, so shells built with the same samples meet without gaps or overlapping faces.
    """
    lo, hi = _q(lo), _q(hi)
    inner_lo = lo + np.array([R if (j, -1) in rounded else 0.0 for j in range(3)])
    inner_hi = hi - np.array([R if (j, 1) in rounded else 0.0 for j in range(3)])
    for (k, sign), rect_name in rects.items():
        si, tj = [j for j in range(3) if j != k]
        us, vs = _span(samples[si], lo[si], hi[si]), _span(samples[tj], lo[tj], hi[tj])
        plane = hi[k] if sign > 0 else lo[k]
        face_normal = np.eye(3)[k] * sign
        face_rounded = (k, sign) in rounded
        for a in range(len(us) - 1):
            for b in range(len(vs) - 1):
                if skip is not None:
                    center = np.empty(3)
                    center[k], center[si], center[tj] = plane, 0.5 * (us[a] + us[a + 1]), 0.5 * (vs[b] + vs[b + 1])
                    if skip(k, sign, center):
                        continue
                points, normals = [], []
                for u, v in ((us[a], vs[b]), (us[a + 1], vs[b]), (us[a + 1], vs[b + 1]), (us[a], vs[b + 1])):
                    p = np.empty(3)
                    p[k], p[si], p[tj] = plane, u, v
                    q = np.clip(p, inner_lo, inner_hi)
                    d = p - q
                    if not face_rounded:
                        q[k], d[k] = p[k], 0.0
                    norm = np.linalg.norm(d)
                    if norm < 1e-12:
                        points.append(p)
                        normals.append(face_normal)
                    else:
                        # Squares of side R map onto quarter discs, which is the usual fillet on rounded faces
                        # and the fillet outline on sharp ones.
                        points.append(q + d * (np.max(np.abs(d)) / norm))
                        normals.append(d / norm if face_rounded else face_normal)
                mesh.quad(points, normals, [atlas_uv(part, rect_name, point) for point in points])


def _bottom_hole(half):
    """Skip rule that leaves a square opening of half width ``half`` in the bottom face of a block shell."""

    def skip(axis, sign, center):
        return axis == 2 and sign < 0 and abs(center[0]) < half and abs(center[1]) < half

    return skip


def _polygon(mesh, part, points, normal, rect_name):
    """Quad or, when two corners coincide, triangle with one flat normal."""
    unique = []
    for q in points:
        q = np.asarray(q, float)
        if not unique or np.linalg.norm(q - unique[-1]) > 1e-9:
            unique.append(q)
    if len(unique) > 3 and np.linalg.norm(unique[0] - unique[-1]) < 1e-9:
        unique.pop()
    uvs = [atlas_uv(part, rect_name, q) for q in unique]
    if len(unique) == 4:
        mesh.quad(unique, [normal] * 4, uvs)
    elif len(unique) == 3:
        mesh.triangle(unique, [normal] * 3, uvs)


def _pyramid(mesh, part):
    """Square pyramid frustum on the block top, from half width S0 at the base to S1 at height L."""
    nu, nv = PYRAMID_SUBDIVISIONS
    z0, z1 = H, H + L
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    for axis, sign, tag in SIDES:
        normal = np.zeros(3)
        normal[axis], normal[2] = sign * L, S0 - S1

        def point(i, j, axis=axis, sign=sign):
            half = S0 - (S0 - S1) * j / nv
            q = np.zeros(3)
            q[axis], q[1 - axis], q[2] = sign * half, half * (-1.0 + 2.0 * i / nu), z0 + L * j / nv
            return q

        for i in range(nu):
            for j in range(nv):
                _polygon(
                    mesh, part, [point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)], normal, f"y_{tag}"
                )
    grid = np.linspace(-S1, S1, nu + 1)
    for i in range(nu):
        for j in range(nu):
            x0, x1, y0, y1 = grid[i], grid[i + 1], grid[j], grid[j + 1]
            _polygon(mesh, part, [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], up, "y_pz")
    _polygon(mesh, part, [(-S0, -S0, z0), (S0, -S0, z0), (S0, S0, z0), (-S0, S0, z0)], down, "y_pz")


def _tapered_pocket(mesh, part, mouth, samples):
    """Blind pocket in the block bottom that narrows upwards at the pyramid slope, closed by a flat ceiling."""
    ceiling = mouth - TAPER * D
    outer = _span(samples, -mouth, mouth)
    zs = np.linspace(-H, -H + D, 5)
    for axis, sign, tag in SIDES:
        normal = np.zeros(3)
        normal[axis], normal[2] = -sign, -TAPER

        def point(k, z, axis=axis, sign=sign):
            half = mouth - TAPER * (z + H)
            q = np.zeros(3)
            q[axis], q[1 - axis], q[2] = sign * half, outer[k] * half / mouth, z
            return q

        for k in range(len(outer) - 1):
            for j in range(len(zs) - 1):
                wall = [point(k, zs[j]), point(k + 1, zs[j]), point(k + 1, zs[j + 1]), point(k, zs[j + 1])]
                _polygon(mesh, part, wall, normal, f"q_{tag}")
    inner, zc = outer * (ceiling / mouth), -H + D
    down = np.array([0.0, 0.0, -1.0])
    for i in range(len(inner) - 1):
        for j in range(len(inner) - 1):
            x0, x1, y0, y1 = inner[i], inner[i + 1], inner[j], inner[j + 1]
            _polygon(mesh, part, [(x0, y0, zc), (x1, y0, zc), (x1, y1, zc), (x0, y1, zc)], down, "q_nz")


def _round_rings(radius, z, height):
    angles = np.linspace(0.0, 2 * np.pi, ROUND_SEGMENTS + 1)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    return [
        np.column_stack((directions * r, np.full(len(angles), zz)))
        for r, zz in ((radius, z), (radius - TAPER * height, z + height))
    ]


def _round_joint(mesh, part, radius, pocket=False):
    """Round frustum or blind funnel, including the square-to-circle underside rim."""
    z, height = (-H, D) if pocket else (H, L)
    lower, upper = _round_rings(radius, z, height)
    wall, cap = ("round_wall", "round_ceiling") if pocket else ("round_side", "round_top")
    normal_sign = -1 if pocket else 1
    for i in range(ROUND_SEGMENTS):
        points = [lower[i], lower[i + 1], upper[i + 1], upper[i]]
        normals = [
            normal_sign * np.array([q[0] / np.linalg.norm(q[:2]), q[1] / np.linalg.norm(q[:2]), TAPER]) for q in points
        ]
        mesh.quad(points, normals, [atlas_uv(part, wall, q) for q in points])
        center = [0.0, 0.0, z + height]
        _polygon(mesh, part, [center, upper[i], upper[i + 1]], [0, 0, normal_sign], cap)
        if pocket:
            a, b = lower[i], lower[i + 1]
            outer = [np.r_[q[:2] * (radius / np.max(np.abs(q[:2]))), z] for q in (a, b)]
            _polygon(mesh, part, [a, b, outer[1], outer[0]], [0, 0, -1], "nz")


def part_mesh(part, spec):
    """Visual mesh of one part for a JOINT_SPECS name or entry, in the part body frame."""
    geometry = joint_geometry(spec)
    mesh = _Mesh()
    cube = ([-H] * 3, [H] * 3, ALL_SIDES, FACE_RECTS)
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}")
    if geometry is None or part == "blue":
        samples = _axis_samples()
        _box(mesh, part, *cube, (samples,) * 3)
        if geometry is not None:
            _pyramid(mesh, part)
    elif part == "green":
        mouth = geometry["taper_mouth_half_m"]
        samples = _axis_samples((-mouth, mouth))
        _box(mesh, part, *cube, (samples, samples, _axis_samples()), skip=_bottom_hole(mouth))
        _tapered_pocket(mesh, part, mouth, samples)
        _round_joint(mesh, part, S0)
    elif part == "yellow":
        mouth = geometry["round_mouth_radius_m"]
        samples = _axis_samples((-mouth, mouth))
        _box(mesh, part, *cube, (samples, samples, _axis_samples()), skip=_bottom_hole(mouth))
        _round_joint(mesh, part, mouth, pocket=True)
    return mesh


@lru_cache(maxsize=32)
def _mesh_attributes(part, spec_items):
    return part_mesh(part, dict(spec_items)).attributes()


def add_stack_assets(asset, spec):
    """Add the three wood textures, materials and visual meshes for one joint spec. Returns mesh names by part."""
    names = {}
    for part in PARTS:
        texture = f"stack_wood_{part}_tex"
        ET.SubElement(
            asset, "texture", name=texture, type="2d", file=xml_path_completion(f"textures/stack_wood_{part}_1k.png")
        )
        ET.SubElement(
            asset,
            "material",
            name=f"stack_wood_{part}",
            texture=texture,
            texrepeat="1 1",
            texuniform="false",
            rgba="1 1 1 1",
            **FINISH,
        )
        names[part] = f"stack_{part}_mesh"
        items = tuple(sorted((JOINT_SPECS[spec] if isinstance(spec, str) else spec).items()))
        ET.SubElement(asset, "mesh", name=names[part], **_mesh_attributes(part, items))
    return names


def visual_geom(parent, name, mesh, material, pos):
    """A visual-only mesh geom in group 1."""
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="mesh",
        mesh=mesh,
        material=material,
        pos=array_to_string(pos),
        group="1",
        contype="0",
        conaffinity="0",
        mass="0",
    )


def collision_shapes(part, spec):
    """Non-overlapping convex collision pieces of one part in its body frame.

    Boxes carry ``size`` and ``pos``, meshes carry part-frame ``vertices`` whose convex hull MuJoCo uses.
    The square pocket uses four diagonal wedges; the circular pocket uses ROUND_SEGMENTS radial wedges.
    Pocket walls stay mesh-shaped to avoid spurious deep box-box contacts on flush block faces in MuJoCo 3.9.
    """
    geometry = joint_geometry(spec)
    cube = {"name": "collision", "type": "box", "size": [H] * 3, "pos": [0.0, 0.0, 0.0]}
    if geometry is None:
        return [cube]
    if part == "blue":
        corners = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]
        vertices = [(sx * r, sy * r, z) for r, z in ((S0, H), (S1, H + L)) for sx, sy in corners]
        return [cube, {"name": "tenon_collision", "type": "mesh", "vertices": np.asarray(vertices)}]
    upper = {"name": "upper_collision", "type": "box", "size": [H, H, H - D / 2], "pos": [0.0, 0.0, D / 2]}
    if part == "green":
        mouth, ceiling = geometry["taper_mouth_half_m"], geometry["taper_ceiling_half_m"]
        pieces = []
        for axis, sign, tag in SIDES:
            vertices = []
            for inner, z in ((mouth, -H), (ceiling, -H + D)):
                for across in (-1, 1):
                    for half in (inner, H):
                        q = np.zeros(3)
                        q[axis], q[1 - axis], q[2] = sign * half, across * half, z
                        vertices.append(q)
            pieces.append({"name": f"wall_{tag}_collision", "type": "mesh", "vertices": np.asarray(vertices)})
        lower, top = _round_rings(S0, H, L)
        return [
            upper,
            *pieces,
            {"name": "round_tenon_collision", "type": "mesh", "vertices": np.vstack((lower[:-1], top[:-1]))},
        ]
    if part == "yellow":
        lower, top = _round_rings(geometry["round_mouth_radius_m"], -H, D)
        directions = lower[:, :2] / np.linalg.norm(lower[:, :2], axis=1)[:, None]
        outer = H * directions / np.max(np.abs(directions), axis=1)[:, None]
        pieces = []
        for i in range(ROUND_SEGMENTS):
            vertices = []
            for ring in (lower, top):
                for j in (i, i + 1):
                    vertices.extend((ring[j], [*outer[j], ring[j, 2]]))
            pieces.append({"name": f"wall_{i:02d}_collision", "type": "mesh", "vertices": np.asarray(vertices)})
        return [upper, *pieces]
    raise ValueError(f"unknown part {part!r}")


def add_collision_geoms(asset, body, part, spec, prefix, **attributes):
    """Write one part's collision pieces into ``body``, with mesh assets for the convex pieces. Returns geom names."""
    names = []
    for shape in collision_shapes(part, spec):
        name = f"{prefix}{shape['name']}"
        if shape["type"] == "box":
            size, pos = array_to_string(shape["size"]), array_to_string(shape["pos"])
            ET.SubElement(body, "geom", name=name, type="box", size=size, pos=pos, **attributes)
        else:
            ET.SubElement(
                asset, "mesh", name=f"{name}_mesh", vertex=array_to_string(np.asarray(shape["vertices"]).ravel())
            )
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=f"{name}_mesh", **attributes)
        names.append(name)
    return names
