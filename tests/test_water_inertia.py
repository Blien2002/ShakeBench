"""CPU checks for a realizable fixed-mass content redistribution, including invalid inputs."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "shakebench/models/objects/water_inertia.py"
spec = importlib.util.spec_from_file_location("water_inertia_under_test", path)
water = importlib.util.module_from_spec(spec)
spec.loader.exec_module(water)
COM = np.array([3.60794e-7, -4.00533e-5, -0.0150773])
INERTIA = np.array([0.00319607, 0.00319598, 0.000371694])
QUAT = np.array([0.6929650945, -0.0001611370, 0.0001648270, 0.7209710983])
BOTTOM = -0.12776431561529336


def test_lower_COM_preserves_mass_and_horizontal_center_with_realizable_tensor():
    attrs = water.bottom_weighted_inertial(COM, INERTIA, QUAT, 0.7, BOTTOM)
    new = np.fromstring(attrs["pos"], sep=" ")
    assert float(attrs["mass"]) == 0.7
    np.testing.assert_allclose(new[:2], COM[:2], rtol=0, atol=1e-18)
    assert BOTTOM < new[2] < COM[2]
    assert 0.020 < COM[2] - new[2] < 0.030
    import xml.etree.ElementTree as ET

    root = ET.Element("mujoco")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="bottle")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", **attrs)
    ET.SubElement(body, "geom", type="cylinder", size=".033 .128")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    i = model.body("bottle").id
    assert model.body_mass[i] == 0.7
    assert min(model.body_inertia[i]) > 0
    assert max(model.body_inertia[i]) < sum(model.body_inertia[i]) - max(model.body_inertia[i])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, np.nan])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        water.bottom_weighted_inertial(COM, INERTIA, QUAT, 0.7, BOTTOM, lower_fraction=fraction)


def test_bad_source_inertia_and_upward_content_are_rejected():
    with pytest.raises(ValueError):
        water.bottom_weighted_inertial(COM, [1, 0.1, 0.1], QUAT, 0.7, BOTTOM)
    with pytest.raises(ValueError):
        water.bottom_weighted_inertial(COM, INERTIA, QUAT, 0.7, 0)


def test_lower_content_volume_has_reasonable_water_density():
    volume = np.pi * water.LOWER_COLUMN_RADIUS_M**2 * water.LOWER_COLUMN_HEIGHT_M
    density = 0.7 * water.LOWER_MASS_FRACTION / volume
    assert 950 < density < 1050
