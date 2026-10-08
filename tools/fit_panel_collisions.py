"""Fit contact hulls to the frozen, independently generated control surfaces.

Only collision resources and explicit moving-control inertia are written. Visual mesh
attributes remain byte-identical. No external geometry or physics-profile override.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "shakebench/models/assets/objects/panel/panel.xml"
N = np.array([-0.603857688, 0, 0.7970921482])
B = np.stack([[0.7970921482, 0, 0.603857688], [0, 1, 0], N], 1)


def source_mesh(tree, name):
    node = tree.find(f".//mesh[@name='{name}']")
    v = np.fromstring(node.get("vertex"), sep=" ").reshape(-1, 3)
    f = np.fromstring(node.get("face"), sep=" ", dtype=int).reshape(-1, 3)
    return v, f


def clip(poly, axis, cut, sign):
    out = []
    for i, q in enumerate(poly):
        p = poly[i - 1]
        a = sign * (p[axis] - cut)
        b = sign * (q[axis] - cut)
        if (a >= 0) != (b >= 0):
            u = (cut - p[axis]) / (q[axis] - p[axis])
            out.append(p + u * (q - p))
        if b >= 0:
            out.append(q)
    return out


def hull_piece(vertices, faces, limits):
    tri = (vertices @ B)[faces]
    mask = np.ones(len(tri), dtype=bool)
    for ax, lo, hi in limits:
        mask &= (tri[:, :, ax].max(1) >= lo - 1e-12) & (tri[:, :, ax].min(1) <= hi + 1e-12)
    points = []
    for t in tri[mask]:
        poly = list(t)
        for ax, lo, hi in limits:
            poly = clip(poly, ax, lo, 1)
            if poly:
                poly = clip(poly, ax, hi, -1)
            if not poly:
                break
        points.extend(poly)
    if len(points) < 4:
        return None
    p = np.unique(np.round(points, 12), axis=0) @ B.T
    h = ConvexHull(p)
    used = h.vertices
    lookup = {old: i for i, old in enumerate(used)}
    face = []
    for t, eq in zip(h.simplices, h.equations):
        if np.dot(np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]]), eq[:3]) < 0:
            t = t[[0, 2, 1]]
        face.append([lookup[i] for i in t])
    return p[used], np.array(face)


def resource(name, points, faces):
    return (
        '<mesh name="'
        + name
        + '" vertex="'
        + " ".join(f"{x:.12g}" for x in points.ravel())
        + '" face="'
        + " ".join(map(str, faces.ravel()))
        + '" />'
    )


def inertia_parts(tree, names, mass):
    volume = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    for name in names:
        v, f = source_mesh(tree, name)
        t = v[f]
        vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
        summ = t.sum(1)
        volume += vol.sum()
        first += (vol[:, None] * summ / 4).sum(0)
        second += np.einsum("i,ij,ik->jk", vol / 20, summ, summ)
        second += np.einsum("i,ilj,ilk->jk", vol / 20, t, t)
    assert volume > 0
    com = first / volume
    inertia = np.trace(second) * np.eye(3) - second - volume * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    inertia *= mass / volume
    assert np.linalg.eigvalsh(inertia).min() > 0
    attrs = (
        'pos="'
        + " ".join(f"{x:.12g}" for x in com)
        + '" mass="'
        + str(mass)
        + '" fullinertia="'
        + " ".join(
            f"{x:.12g}"
            for x in [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        )
        + '"'
    )
    return "<inertial " + attrs + " />", {
        "mass_kg": mass,
        "com_m": com.tolist(),
        "inertia_kg_m2": inertia.tolist(),
        "visual_component_volume_cm3": volume * 1e6,
        "assumption": "Uniform volume-weighted visual components, normalized to existing control mass. Components slightly overlap; includes no material-density or measured-mechanism claim.",
    }


def geom(name, mesh, contype=5):
    return f'<geom name="{name}" type="mesh" mesh="{mesh}" mass="0" contype="{contype}" conaffinity="0" group="0" rgba=".5 .5 .5 0" friction=".8 .005 .0001" condim="4" />'


def generate():
    text = PANEL.read_text()
    tree = ET.fromstring(text)
    # Remove only this generator's previous contacts/inertia to allow regeneration.
    text = re.sub(r'<mesh name="panel_contact_[^"]+"[^>]*/>\s*', "", text)
    text = re.sub(r'<geom name="panel_knob_(?:grip_\d+|root_\d+|dial|inset)_collision"[^>]*/>\s*', "", text)
    text = re.sub(r'<geom name="panel_switch_shoulder_collision"[^>]*/>\s*', "", text)
    for body in ("knob", "lever"):
        pattern = rf'(<body name="panel_{body}"[^>]*>)(.*?)(</body>)'
        text = re.sub(pattern, lambda m: m[1] + re.sub(r"<inertial[^>]*/>\s*", "", m[2]) + m[3], text, flags=re.S)
    assets = []
    geoms = []
    summary = {"visual_revision": "original_controls", "knob_pieces": []}
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
    v, f = source_mesh(tree, "panel_refined_grip")
    for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        p, face = hull_piece(v, f, [(0, lo, hi)])
        mesh = f"panel_contact_grip_{i:02d}"
        name = "panel_knob_collision" if lo < 0 < hi else f"panel_knob_grip_{i:02d}_collision"
        assets.append(resource(mesh, p, face))
        geoms.append(geom(name, mesh))
        summary["knob_pieces"].append(
            {"geom": name, "source": "panel_refined_grip", "t_range_mm": [lo * 1000, hi * 1000], "vertices": len(p)}
        )
    v, f = source_mesh(tree, "panel_refined_root")
    xcuts = [-0.0144424193825, -0.009, -0.004, 0.002, 0.0065, 0.010, 0.0130691690383]
    zcuts = [0.0125, 0.0133, 0.0141, 0.0148, 0.01495]
    count = 0
    for lo, hi in zip(xcuts[:-1], xcuts[1:]):
        for bottom, top in zip(zcuts[:-1], zcuts[1:]):
            piece = hull_piece(v, f, [(0, lo, hi), (2, bottom, top)])
            if piece is None:
                continue
            p, face = piece
            mesh = f"panel_contact_root_{count:02d}"
            name = f"panel_knob_root_{count:02d}_collision"
            assets.append(resource(mesh, p, face))
            geoms.append(geom(name, mesh))
            count += 1
            summary["knob_pieces"].append(
                {
                    "geom": name,
                    "source": "panel_refined_root",
                    "t_range_mm": [lo * 1000, hi * 1000],
                    "n_range_mm": [bottom * 1000, top * 1000],
                    "vertices": len(p),
                }
            )
    for label in ("dial", "inset"):
        geoms.append(geom(f"panel_knob_{label}_collision", f"panel_refined_{label}"))
    inertial, summary["knob_inertia"] = inertia_parts(
        tree, ["panel_refined_grip", "panel_refined_root", "panel_refined_dial"], 0.045
    )
    text = re.sub(
        r'<geom\b(?=[^>]*\bname="panel_knob_collision")[^>]*/>', lambda _: inertial + "\n" + "\n".join(geoms), text
    )
    inertial, summary["lever_inertia"] = inertia_parts(tree, ["panel_refined_lever"], 0.025)
    text = re.sub(
        r'<geom\b(?=[^>]*\bname="panel_lever_collision")[^>]*/>',
        lambda _: inertial + "\n" + geom("panel_lever_collision", "panel_refined_lever"),
        text,
    )
    text = re.sub(
        r'<geom\b(?=[^>]*\bname="panel_switch_base_collision")[^>]*/>',
        lambda _: geom("panel_switch_base_collision", "panel_refined_nut", 3)
        + "\n"
        + geom("panel_switch_shoulder_collision", "panel_refined_shoulder", 3),
        text,
    )
    text = text.replace("</asset>", "\n" + "\n".join(assets) + "\n</asset>")
    PANEL.write_text(text)
    (ROOT / "panel_contact_generation.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "knob_grip_pieces": len(cuts) - 1,
                "knob_root_pieces": count,
                "knob_total": len(geoms),
                "knob_inertia": summary["knob_inertia"],
                "lever_inertia": summary["lever_inertia"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    generate()
