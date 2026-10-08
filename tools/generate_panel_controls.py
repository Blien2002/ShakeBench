"""Independent waist-shaped plate handle and one-ended directional paint.

All geometry is authored analytically in metres. Official product photography
informed the silhouette; no manufacturer CAD, mesh, logo or texture is used.
Only the upper grip, paint, fitted upper collisions and knob inertia are updated.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from fit_panel_collisions import hull_piece, inertia_parts, resource

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "shakebench/models/assets/objects/panel/panel.xml"
N = np.array([-0.603857688, 0.0, 0.7970921482])
B = np.stack([[N[2], 0.0, -N[0]], [0.0, 1.0, 0.0], N], axis=1)
CX = -0.0006866251721
CY = 0.000101668743
HALF_LENGTH = 0.0137557942104
HALF_WIDTH = 0.00435286349074673
WAIST_HALF_WIDTH = 0.0034
END_RADIUS = 0.001
BASE = 0.0148
SHOULDER_TOP = 0.01495
HEIGHT_SCALE = 0.85
SIDE_TOP = SHOULDER_TOP + 0.0105 * HEIGHT_SCALE
CROWN_RADIUS = 0.00065 * HEIGHT_SCALE
TOP = SIDE_TOP + CROWN_RADIUS
PAINT_WIDTH = 0.0022
PAINT_START = 0.00015
PAINT_WALL_BOTTOM = TOP - 0.0036 * HEIGHT_SCALE
PAINT_OFFSET = 0.000012


def outline():
    """CCW plan contour: modest flat waist, broad ends, rounded corners."""
    a = HALF_LENGTH - END_RADIUS
    x = np.linspace(-a, a, 137)
    u = np.clip((np.abs(x) - 0.0018) / (0.0102 - 0.0018), 0, 1)
    width = WAIST_HALF_WIDTH + (HALF_WIDTH - WAIST_HALF_WIDTH) * (3 * u**2 - 2 * u**3)
    p = [[CX + t, CY - w] for t, w in zip(x, width)]
    for angle in np.linspace(-np.pi / 2, 0, 17)[1:]:
        p.append([CX + a + END_RADIUS * np.cos(angle), CY - HALF_WIDTH + END_RADIUS + END_RADIUS * np.sin(angle)])
    p.append([CX + HALF_LENGTH, CY + HALF_WIDTH - END_RADIUS])
    for angle in np.linspace(0, np.pi / 2, 17)[1:]:
        p.append([CX + a + END_RADIUS * np.cos(angle), CY + HALF_WIDTH - END_RADIUS + END_RADIUS * np.sin(angle)])
    p.extend([[CX + t, CY + w] for t, w in zip(x[::-1][1:], width[::-1][1:])])
    for angle in np.linspace(np.pi / 2, np.pi, 17)[1:]:
        p.append([CX - a + END_RADIUS * np.cos(angle), CY + HALF_WIDTH - END_RADIUS + END_RADIUS * np.sin(angle)])
    p.append([CX - HALF_LENGTH, CY - HALF_WIDTH + END_RADIUS])
    for angle in np.linspace(np.pi, 3 * np.pi / 2, 17)[1:-1]:
        p.append([CX - a + END_RADIUS * np.cos(angle), CY - HALF_WIDTH + END_RADIUS + END_RADIUS * np.sin(angle)])
    return np.array(p)


def explicit_mesh(name, points, normals, faces):
    values = {
        "name": name,
        "vertex": " ".join(f"{v:.12g}" for v in np.asarray(points).ravel()),
        "normal": " ".join(f"{v:.12g}" for v in np.asarray(normals).ravel()),
        "face": " ".join(str(int(v)) for v in np.asarray(faces).ravel()),
    }
    return ET.tostring(ET.Element("mesh", values), encoding="unicode")


def grip_mesh():
    plan = outline()
    count = len(plan)
    tangent = np.roll(plan, -1, axis=0) - np.roll(plan, 1, axis=0)
    outward = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    pts, normals = [], []
    for z in (BASE, SIDE_TOP):
        pts.extend(np.c_[plan, np.full(count, z)])
        normals.extend(np.c_[outward, np.zeros(count)])
    for angle in np.linspace(0, np.pi / 2, 17)[1:]:
        inset = CROWN_RADIUS * (1 - np.cos(angle))
        z = SIDE_TOP + CROWN_RADIUS * np.sin(angle)
        pts.extend(np.c_[plan - inset * outward, np.full(count, z)])
        normals.extend(np.c_[outward * np.cos(angle), np.full(count, np.sin(angle))])
    rings = len(pts) // count
    faces = []
    for j in range(rings - 1):
        for i in range(count):
            a, b = j * count + i, j * count + (i + 1) % count
            faces.extend([[a, b, b + count], [a, b + count, a + count]])
    # A centre fan is valid for this star-shaped, concave waist contour.
    for j, sign in ((0, -1), (rings - 1, 1)):
        start = len(pts)
        ring = np.array(pts[j * count : (j + 1) * count])
        pts.extend(ring)
        pts.append([CX, CY, BASE if sign < 0 else TOP])
        normals.extend([[0, 0, sign]] * (count + 1))
        for i in range(count):
            f = [start + count, start + i, start + (i + 1) % count]
            faces.append(f if sign > 0 else f[::-1])
    points = np.array(pts) @ B.T
    normal = np.array(normals) @ B.T
    return explicit_mesh("panel_refined_grip", points, normal, faces)


def paint_mesh():
    """Single strip on +t half, rounded inner cap, continuous end wrap."""
    end = CX + HALF_LENGTH
    flat_end = end - CROWN_RADIUS
    r = PAINT_WIDTH / 2
    stations = []
    # Rounded start cap, followed by constant-width paint on the planar crown.
    for x in np.linspace(PAINT_START, PAINT_START + r, 17):
        width = np.sqrt(max(r * r - (x - PAINT_START - r) ** 2, 1e-16))
        stations.append((x, TOP, width, np.array([0.0, 0.0, 1.0])))
    for x in np.linspace(PAINT_START + r, flat_end, 121)[1:]:
        stations.append((x, TOP, r, np.array([0.0, 0.0, 1.0])))
    for angle in np.linspace(0, np.pi / 2, 33)[1:]:
        stations.append(
            (
                flat_end + CROWN_RADIUS * np.sin(angle),
                SIDE_TOP + CROWN_RADIUS * np.cos(angle),
                r,
                np.array([np.sin(angle), 0.0, np.cos(angle)]),
            )
        )
    for z in np.linspace(SIDE_TOP, PAINT_WALL_BOTTOM, 31)[1:]:
        stations.append((end, z, r, np.array([1.0, 0.0, 0.0])))
    points, normals, faces = [], [], []
    for i, (x, z, width, normal) in enumerate(stations):
        for y in (CY - width, CY + width):
            points.append((np.array([x, y, z]) + PAINT_OFFSET * normal) @ B.T)
            normals.append(normal @ B.T)
        if i:
            a = 2 * (i - 1)
            faces.extend([[a, a + 2, a + 3], [a, a + 3, a + 1]])
    return explicit_mesh("panel_refined_mark", points, normals, faces)


def replace_mesh(text, name, node):
    pattern = r'<mesh\b(?=[^>]*\bname="' + re.escape(name) + r'")[^>]*/>'
    text, count = re.subn(pattern, lambda _: node, text)
    assert count == 1, (name, count)
    return text


def generate():
    text = PANEL.read_text()
    text = replace_mesh(text, "panel_refined_grip", grip_mesh())
    text = replace_mesh(text, "panel_refined_mark", paint_mesh())
    tree = ET.fromstring(text)
    grip = tree.find('.//mesh[@name="panel_refined_grip"]')
    v = np.fromstring(grip.get("vertex"), sep=" ").reshape(-1, 3)
    f = np.fromstring(grip.get("face"), sep=" ", dtype=int).reshape(-1, 3)
    # Retain names and official contact attributes; replace just the 12 upper hulls.
    cuts = np.array(
        [
            -0.0144424193825,
            -0.012,
            -0.009,
            -0.006,
            -0.003,
            -0.0014317547355,
            0.0015,
            0.0045,
            0.0065,
            0.0085,
            0.0105,
            0.012,
            0.0130691690383,
        ]
    )
    pieces = []
    for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        p, face = hull_piece(v, f, [(0, lo, hi)])
        name = f"panel_contact_grip_{i:02d}"
        text = replace_mesh(text, name, resource(name, p, face))
        pieces.append({"mesh": name, "t_range_mm": [lo * 1000, hi * 1000], "vertices": len(p)})
    inertial, inertia = inertia_parts(tree, ["panel_refined_grip", "panel_refined_root", "panel_refined_dial"], 0.045)
    pattern = r'(<body name="panel_knob"[^>]*>)(.*?)(</body>)'

    def replace_inertia(match):
        body, count = re.subn(r"<inertial\b[^>]*/>", lambda _: inertial, match[2])
        assert count == 1
        return match[1] + body + match[3]

    text = re.sub(pattern, replace_inertia, text, flags=re.S)
    PANEL.write_text(text)
    config = {
        "design": "waist-shaped flat upper handle",
        "units": "m",
        "length": 2 * HALF_LENGTH,
        "max_width": 2 * HALF_WIDTH,
        "waist_width": 2 * WAIST_HALF_WIDTH,
        "base": BASE,
        "shoulder_top": SHOULDER_TOP,
        "parallel_face_top": SIDE_TOP,
        "exposed_parallel_height": SIDE_TOP - SHOULDER_TOP,
        "top": TOP,
        "user_height_scale_above_shoulder": HEIGHT_SCALE,
        "plan_corner_radius": END_RADIUS,
        "crown_radius": CROWN_RADIUS,
        "paint_width": PAINT_WIDTH,
        "paint_start_t": PAINT_START,
        "paint_end_t": CX + HALF_LENGTH + PAINT_OFFSET,
        "paint_wall_bottom": PAINT_WALL_BOTTOM,
        "paint_surface_offset": PAINT_OFFSET,
        "upper_contact_pieces": pieces,
        "knob_inertia": inertia,
        "grasp_validation": "Geometry and compilation only; parent owns physical handle-only validation.",
        "reference": "Private Schneider official product photo XB5_633_CPMFS17192; no manufacturer CAD or branding.",
    }
    (ROOT / "panel_flat_handle_geometry.json").write_text(json.dumps(config, indent=2))
    contacts_path = ROOT / "panel_contact_generation.json"
    contacts = json.loads(contacts_path.read_text())
    contacts["knob_inertia"] = inertia
    contacts["upper_handle_design"] = config["design"]
    contacts["upper_handle_geometry_validation"] = config["grasp_validation"]
    for entry in contacts["knob_pieces"]:
        if entry["source"] == "panel_refined_grip":
            entry["vertices"] = next(p["vertices"] for p in pieces if p["t_range_mm"] == entry["t_range_mm"])
    contacts_path.write_text(json.dumps(contacts, indent=2))
    print(
        json.dumps(
            {
                k: config[k]
                for k in ["length", "max_width", "waist_width", "exposed_parallel_height", "top", "paint_width"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    generate()
