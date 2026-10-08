"""Flat proxy registration, body mass/inertia and mesh-envelope geometry on CPU."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "shakebench/models/objects/water_support.py"
spec = importlib.util.spec_from_file_location("water_support_under_test", path)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def fake_bottle():
    asset = ET.Element("asset")
    body = ET.Element("body", name="wine_bottle_main")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", pos="0 0 -.0385521", mass=".7", diaginertia=".0027926 .0027925 .000349")
    return SimpleNamespace(asset=asset, naming_prefix="wine_bottle_", _contact_geoms=[], get_obj=lambda: body)


def test_registered_mesh_preserves_mass_and_fits_expected_envelope():
    bottle = fake_bottle()
    name = support.add_flat_water_sole(bottle, -0.1298401173734259)
    assert name == "wine_bottle_flat_sole" and bottle._contact_geoms == ["flat_sole"]
    root = ET.Element("mujoco")
    root.append(bottle.asset)
    ET.SubElement(root, "worldbody").append(bottle.get_obj())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    bid, gid = model.body("wine_bottle_main").id, model.geom(name).id
    assert model.body_mass[bid] == 0.7
    np.testing.assert_allclose(model.body_ipos[bid], [0, 0, -0.0385521], atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[bid], [0.0027926, 0.0027925, 0.000349], atol=1e-12)
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[gid] == 0
    mesh = model.geom_dataid[gid]
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    q = model.geom_quat[gid]
    from scipy.spatial.transform import Rotation

    vertices = (
        model.mesh_vert[first : first + count] @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T + model.geom_pos[gid]
    )
    np.testing.assert_allclose(np.linalg.norm(vertices[:, :2], axis=1), 0.034, atol=1e-8)
    assert np.isclose(vertices[:, 2].min(), -0.1303401173734259, atol=1e-8)
    assert np.isclose(np.ptp(vertices[:, 2]), 0.002, atol=1e-8)


def test_duplicate_registration_is_rejected():
    bottle = fake_bottle()
    support.add_flat_water_sole(bottle, -0.12984)
    with pytest.raises(ValueError):
        support.add_flat_water_sole(bottle, -0.12984)


def test_invalid_bottom_does_not_mutate_object():
    bottle = fake_bottle()
    with pytest.raises(ValueError):
        support.add_flat_water_sole(bottle, np.nan)
    assert not bottle._contact_geoms and not list(bottle.asset)
