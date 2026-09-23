"""Copy the eight-object task assets from the official RoboCasa instance tree.

The new task set reuses the official `objaverse` instance *model.xml* files
verbatim except for one documented transform: the category scale from
RoboCasa's own registry is baked into the mesh/geom/body attributes, so the
compiled geometry in ShakeBench is exactly the geometry recorded in
`SOURCES.json` and checked by `tests/test_shakebench_pick_place_v2_assets.py`.

Usage:
    python tools/build_pick_assets.py --source <extracted objaverse dir> \
        --dest shakebench/models/assets/objects/robocasa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_URL = "https://huggingface.co/datasets/robocasa/robocasa-assets"
SOURCE_REVISION = "objaverse.zip downloaded 2026-09-18 (RoboCasa `main` 4f8a2980def75a55dff96b990745b83540425f09)"
SOURCE_LICENSE = "CC-BY-4.0"

#: object_id -> (official instance path, task scale)
#:
#: Task scales are the RoboCasa registry values except for mug and apple: the
#: compiled Panda jaw opens to 80 mm, so the task keeps at least 10 mm of
#: allowance around the closed pads instead of pinching at the jaw limit.
INSTANCES = {
    "cereal": ("cereal/cereal_0", (1.0, 1.0, 1.0)),
    "mug": ("mug/mug_1", (0.80, 0.80, 0.80)),
    "apple": ("apple/apple_0", (0.85, 0.85, 0.85)),
    "spatula": ("spatula/spatula_0", (1.10, 1.10, 1.10)),
    "bar_soap": ("bar_soap/bar_soap_0", (0.95, 0.95, 1.05)),
    "rolling_pin": ("rolling_pin/rolling_pin_0", (1.25, 1.25, 1.25)),
    "potato": ("potato/potato_1", (0.85, 0.85, 0.85)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _scaled(text: str | None, scale) -> str | None:
    if text is None:
        return None
    values = [float(item) for item in text.split()]
    return " ".join(f"{value * factor:.17g}" for value, factor in zip(values, scale))


def tree_referenced_files(source_xml: Path):
    """Yield every ``file=`` asset element of one instance XML."""

    for element in ET.parse(source_xml).getroot().iter():
        if element.get("file") is not None:
            yield element


def transform(source_xml: Path, scale) -> str:
    """Return the instance MJCF with the category scale baked in."""

    tree = ET.parse(source_xml)
    root = tree.getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("scale", _scaled(mesh.get("scale", "1 1 1"), scale))
    for body in root.iter("body"):
        if body.get("pos") is not None:
            body.set("pos", _scaled(body.get("pos"), scale))
    for geom in root.iter("geom"):
        if geom.get("pos") is not None:
            geom.set("pos", _scaled(geom.get("pos"), scale))
        if geom.get("size") is not None:
            geom.set("size", _scaled(geom.get("size"), scale))
    return ET.tostring(root, encoding="unicode")


def build(source_root: Path, dest_root: Path) -> dict:
    dest_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for object_id, (instance, scale) in sorted(INSTANCES.items()):
        source_dir = source_root / instance
        model_xml = source_dir / "model.xml"
        if not model_xml.is_file():
            raise FileNotFoundError(model_xml)
        dest_dir = dest_root / instance
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        # Copy only the meshes/textures the instance XML actually references:
        # the archive also ships per-chunk convex pieces that this XML does not
        # use, and they would otherwise double the asset size for no fidelity.
        referenced = sorted({element.get("file") for element in tree_referenced_files(model_xml)})
        for relative in referenced:
            source_asset = source_dir / relative
            if not source_asset.is_file():
                raise FileNotFoundError(source_asset)
            destination_asset = dest_dir / relative
            destination_asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_asset, destination_asset)
        generated = dest_dir / "model.xml"
        generated.write_text(transform(model_xml, scale), encoding="utf-8")
        sites = measure_sites(generated)
        add_sites(generated, sites)
        records[object_id] = {
            "instance": instance,
            "source_model_xml": str(model_xml),
            "source_sha256": _sha256(model_xml),
            "generated_sha256": _sha256(generated),
            "applied_scale": list(scale),
            "support_lower_upper_radius_m": list(sites),
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "source_license": SOURCE_LICENSE,
        }
    payload = {
        "schema_id": "shakebench.pick_place_v2.assets",
        "note": "official RoboCasa objaverse instances; only the registry scale is baked in",
        "objects": records,
    }
    (dest_root / "SOURCES.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def measure_sites(path: Path) -> tuple[float, float, float]:
    """Measure (lower_z, upper_z, radius) of the contact geoms after scaling."""

    import mujoco
    import numpy as np
    from pick_measure import geom_points

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cloud = np.concatenate(
        [
            geom_points(model, data, geom_id)
            for geom_id in range(model.ngeom)
            if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
        ]
    )
    return (
        float(cloud[:, 2].min()),
        float(cloud[:, 2].max()),
        float(np.max(np.linalg.norm(cloud[:, :2], axis=1))),
    )


def add_sites(path: Path, sites: tuple[float, float, float]) -> None:
    """Add the robosuite object sites that the placement sampler needs."""

    lower, upper, radius = sites
    tree = ET.parse(path)
    wrapper = tree.getroot().find("./worldbody/body")
    for element in list(wrapper.findall("./site")):
        wrapper.remove(element)
    for name, position in (
        ("bottom_site", (0.0, 0.0, lower)),
        ("top_site", (0.0, 0.0, upper)),
        ("horizontal_radius_site", (radius, 0.0, 0.0)),
    ):
        wrapper.append(
            ET.Element(
                "site",
                {
                    "name": name,
                    "pos": " ".join(f"{value:.17g}" for value in position),
                    "size": "0.005",
                    "rgba": "0 0 0 0",
                },
            )
        )
    tree.write(path, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="extracted objaverse/ directory")
    parser.add_argument("--dest", required=True, type=Path, help="destination asset directory")
    args = parser.parse_args()
    payload = build(args.source, args.dest)
    for object_id, record in payload["objects"].items():
        print(f"{object_id:12s} {record['instance']:26s} scale={record['applied_scale']}")


if __name__ == "__main__":
    main()
