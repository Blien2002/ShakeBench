"""Two-box wooden T and exact projected-union coverage, in metres."""

import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from robosuite.models.objects import CompositeObject
from shakebench.models import xml_path_completion

HALF_HEIGHT_M = 0.01
# Non-overlapping crossbar and stem: 150 x 150 x 20 mm, 30 mm stroke.
RECTANGLES = (((0.0, 0.06), (0.075, 0.015)), ((0.0, -0.015), (0.015, 0.06)))
TARGET_AREA_M2 = sum(4 * np.prod(size) for _, size in RECTANGLES)
CORNERS = np.array(list(product((-1, 1), repeat=3)))
# Counterclockwise outline in the T body's xy frame.
OUTLINE = np.array(
    [
        [-0.015, -0.075],
        [0.015, -0.075],
        [0.015, 0.045],
        [0.075, 0.045],
        [0.075, 0.075],
        [-0.075, 0.075],
        [-0.075, 0.045],
        [-0.015, 0.045],
    ]
)


def make_tee():
    """A single free body; collision boxes carry mass, visuals carry wood texture."""
    tee = CompositeObject(
        "push_t_block",
        total_size=[0.075, 0.075, HALF_HEIGHT_M],
        geom_types=["box", "box"],
        geom_names=["bar", "stem"],
        geom_sizes=[[*size, HALF_HEIGHT_M] for _, size in RECTANGLES],
        geom_locations=[[*center, 0] for center, _ in RECTANGLES],
        locations_relative_to_center=True,
        density=600.0,
        rgba=[0.92, 0.82, 0.65, 1],
    )
    ET.SubElement(
        tee.asset,
        "texture",
        name="push_t_light_wood",
        type="2d",
        file=xml_path_completion("textures/ambientcg_wood095_color_1k.png"),
    )
    # Physical-scale mapping keeps the grain size consistent across both boxes.
    ET.SubElement(
        tee.asset,
        "material",
        name="push_t_wood",
        texture="push_t_light_wood",
        texrepeat="6.25 12.5",
        texuniform="true",
        rgba="0.94 0.98 1 1",
        emission="0",
        specular="0.08",
        shininess="0.12",
    )
    for geom in tee.get_obj().iter("geom"):
        if geom.get("group") == "1":
            geom.set("material", "push_t_wood")
            geom.set("mass", "0")
            geom.attrib.pop("rgba", None)
    return tee


def projected_geometry(position, rotation):
    """Project all 16 box vertices into the target frame; retain the lowest z.

    Convex hulls include the side faces when tilted. Union overlap is removed by
    inclusion-exclusion in coverage(), so tilt cannot double-count shared area.
    """
    polygons, lowest = [], float("inf")
    for center, size in RECTANGLES:
        vertices = (CORNERS * [*size, HALF_HEIGHT_M] + [*center, 0]) @ rotation.T + position
        lowest = min(lowest, float(vertices[:, 2].min()))
        points = vertices[:, :2]
        polygons.append(points[ConvexHull(points).vertices])
    return polygons, lowest


def intersection(subject, clip):
    """Sutherland-Hodgman intersection of two counterclockwise convex polygons."""
    output = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            break
        source, output = output, []
        edge = b - a

        def side(p):
            delta = p - a
            return edge[0] * delta[1] - edge[1] * delta[0]

        previous = source[-1]
        previous_side = side(previous)
        for current in source:
            current_side = side(current)
            if (current_side >= 0) != (previous_side >= 0):
                output.append(previous + (current - previous) * previous_side / (previous_side - current_side))
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
    return np.asarray(output).reshape(-1, 2)


def area(polygon):
    if len(polygon) < 3:
        return 0.0
    x, y = polygon.T
    return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2)


TARGET_POLYGONS = tuple(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size + center for center, size in RECTANGLES)


def coverage(polygons):
    """Intersection of projected solid T and target, divided by target area."""
    shared = intersection(polygons[0], polygons[1])
    covered = sum(
        sum(area(intersection(p, target)) for p in polygons) - area(intersection(shared, target))
        for target in TARGET_POLYGONS
    )
    return float(np.clip(covered / TARGET_AREA_M2, 0, 1))
