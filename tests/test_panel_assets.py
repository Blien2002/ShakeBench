"""Frozen visual surfaces, fitted contacts and explicit control inertia."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from shakebench.tasks.runtime import make_environment


@pytest.fixture(scope="module")
def panel(initial_state):
    env, _ = make_environment(initial_state("panel_operation"), gamma=0, horizon=3)
    yield env
    env.close()


def test_display_geometry_does_not_change_control_inertia(panel):
    root = ET.fromstring(panel.model.get_xml())
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for name in (
        "panel_knob_dial_visual",
        "panel_knob_grip_visual",
        "panel_switch_base_visual",
        "panel_switch_handle_visual",
        "panel_switch_shoulder_visual",
    ):
        geom = root.find(f".//geom[@name='{name}']")
        assert geom is not None
        assert geom.get("mass") == "0"
        assert geom.get("contype") == geom.get("conaffinity") == "0"
        parent = next(node for node in root.iter() if geom in list(node))
        parent.remove(geom)
    changed = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_axis", "jnt_range", "dof_damping"):
        np.testing.assert_array_equal(getattr(changed, field), getattr(original, field), err_msg=field)


def test_collar_is_a_fixed_contact_and_revision_is_reported(panel):
    model = panel.sim.model._model
    collar = model.geom("panel_switch_base_collision")
    assert collar.type == mujoco.mjtGeom.mjGEOM_MESH
    assert model.body_mass[collar.bodyid] == 0
    assert panel.get_policy_task_context()["control_asset_revision"] == "original_controls"


def test_flat_grip_has_approved_exposed_height_and_opposed_end_faces(panel):
    root = ET.fromstring(panel.model.get_xml())
    mesh = root.find(".//mesh[@name='panel_refined_grip']")
    basis = np.stack([[0.7970921482, 0, 0.603857688], [0, 1, 0], [-0.603857688, 0, 0.7970921482]], axis=1)
    points = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3) @ basis
    faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
    shoulder = root.find(".//mesh[@name='panel_refined_root']")
    shoulder_points = np.fromstring(shoulder.get("vertex"), sep=" ").reshape(-1, 3) @ basis
    assert shoulder_points[:, 2].max() == pytest.approx(0.01495, abs=1e-10)
    assert points[:, 2].max() == pytest.approx(0.0244275, abs=1e-10)
    assert np.ptp(points[:, 0]) == pytest.approx(0.0275115884208, abs=1e-10)
    assert np.ptp(points[:, 1]) == pytest.approx(0.00870572698149346, abs=1e-10)
    triangles = points[faces]

    def section(fixed_axes, values, variable_axis):
        hits = []
        for triangle in triangles:
            axes = list(fixed_axes)
            matrix = np.stack([triangle[1, axes] - triangle[0, axes], triangle[2, axes] - triangle[0, axes]], axis=1)
            if abs(np.linalg.det(matrix)) < 1e-18:
                continue
            u, v = np.linalg.solve(matrix, np.asarray(values) - triangle[0, axes])
            if u >= -1e-8 and v >= -1e-8 and u + v <= 1 + 1e-8:
                hits.append(
                    triangle[0, variable_axis]
                    + u * (triangle[1, variable_axis] - triangle[0, variable_axis])
                    + v * (triangle[2, variable_axis] - triangle[0, variable_axis])
                )
        return min(hits), max(hits)

    for height in (0.0151, 0.0193, 0.0231, 0.0238749):
        rear, front = section((1, 2), (0.000101668743, height), 0)
        assert rear == pytest.approx(-0.0144424193825, abs=1e-10)
        assert front == pytest.approx(0.0130691690383, abs=1e-10)
        lo, hi = section((0, 2), (-0.0006866251721, height), 1)
        assert hi - lo == pytest.approx(0.0068, abs=1e-10)
    assert 0.023875 - shoulder_points[:, 2].max() == pytest.approx(0.008925, abs=1e-10)


def test_direction_mark_is_one_ended_and_follows_joint(panel):
    root = ET.fromstring(panel.model.get_xml())
    mesh = root.find(".//mesh[@name='panel_refined_mark']")
    basis = np.stack([[0.7970921482, 0, 0.603857688], [0, 1, 0], [-0.603857688, 0, 0.7970921482]], axis=1)
    points = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3) @ basis
    assert points[:, 0].min() == pytest.approx(0.00015, abs=1e-10)
    assert points[:, 0].max() == pytest.approx(0.0130811690383, abs=1e-10)
    assert np.ptp(points[:, 1]) == pytest.approx(0.0022, abs=1e-10)
    model = panel.sim.model._model
    marker = model.geom("panel_knob_direction_visual")
    body_id = model.body("panel_knob").id
    assert marker.bodyid == body_id
    joint_id = model.joint("panel_knob_joint").id
    data = mujoco.MjData(model)
    vectors = []
    for angle in (0, 60, 120):
        data.qpos[model.jnt_qposadr[joint_id]] = np.deg2rad(angle)
        mujoco.mj_forward(model, data)
        vectors.append(data.xmat[body_id].reshape(3, 3) @ basis[:, 0])
    assert np.dot(vectors[0], vectors[1]) == pytest.approx(0.5, abs=1e-8)
    assert np.dot(vectors[0], vectors[2]) == pytest.approx(-0.5, abs=1e-8)


def test_painted_crown_and_end_wrap_have_outward_continuous_faces(panel):
    root = ET.fromstring(panel.model.get_xml())
    mesh = root.find(".//mesh[@name='panel_refined_mark']")
    vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
    normals = np.fromstring(mesh.get("normal"), sep=" ").reshape(-1, 3)
    faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
    basis = np.stack([[0.7970921482, 0, 0.603857688], [0, 1, 0], [-0.603857688, 0, 0.7970921482]], axis=1)
    stations = (vertices @ basis).reshape(-1, 2, 3).mean(axis=1)
    planar = stations[stations[:, 0] < 0.0125166690383]
    np.testing.assert_allclose(planar[:, 2], 0.0244395, atol=1e-10)
    wall = stations[stations[:, 2] < 0.023875]
    assert len(wall) > 20
    np.testing.assert_allclose(wall[:, 0], 0.0130811690383, atol=1e-10)
    assert np.max(np.linalg.norm(np.diff(stations, axis=0), axis=1)) < 0.00015
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert np.min(np.sum(cross * normals[faces].mean(axis=1), axis=1)) > 0


def test_flat_grip_mesh_is_closed_and_waist_is_concave(panel):
    from collections import Counter

    root = ET.fromstring(panel.model.get_xml())
    mesh = root.find(".//mesh[@name='panel_refined_grip']")
    vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
    faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
    _, lookup = np.unique(np.round(vertices, 11), axis=0, return_inverse=True)
    edges = Counter()
    for triangle in lookup[faces]:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edges[tuple(sorted((int(a), int(b))))] += 1
    assert set(edges.values()) == {2}
    basis = np.stack([[0.7970921482, 0, 0.603857688], [0, 1, 0], [-0.603857688, 0, 0.7970921482]], axis=1)
    points = vertices @ basis
    flat_waist = points[(np.abs(points[:, 0] + 0.0006866251721) < 0.0018) & (np.abs(points[:, 2] - 0.023875) < 1e-10)]
    assert np.ptp(flat_waist[:, 1]) == pytest.approx(0.0068, abs=1e-10)
    broad_ends = points[(np.abs(points[:, 0] + 0.0006866251721) > 0.0102) & (np.abs(points[:, 2] - 0.023875) < 1e-10)]
    assert np.ptp(broad_ends[:, 1]) > np.ptp(flat_waist[:, 1]) + 0.0018


def test_compound_control_contacts_have_no_missing_finger_pairs(panel):
    model = panel.sim.model._model
    assert len(panel.panel_control_geom_groups["knob"]) == 38
    assert len(panel.panel_control_geom_groups["lever"]) == 1
    robot = panel.robots[0]
    pads = {model.geom(f"gripper0_right_finger{i}_pad_collision").id for i in (1, 2)}
    pairs = {frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    for kind, ids in panel.panel_control_geom_groups.items():
        assert panel.panel_control_geom_ids[kind] in ids
        for gid in ids:
            geom = model.geom(gid)
            assert geom.bodyid == model.body(f"panel_{kind}").id
            assert geom.contype == 5 and geom.conaffinity == 0
            for pad in pads:
                assert frozenset((gid, pad)) in pairs
    for kind, mass in (("knob", 0.045), ("lever", 0.025)):
        body = model.body(f"panel_{kind}")
        assert body.mass == pytest.approx(mass)
        assert np.all(body.inertia > 0)
        assert max(body.inertia) < sum(body.inertia) - max(body.inertia)


def test_secondary_knob_contact_prevents_release():
    from types import SimpleNamespace

    from shakebench.environments.panel_operation import PanelOperation

    samples = []
    contact = SimpleNamespace(geom1=102, geom2=200, dist=0.0)
    env = SimpleNamespace(
        task_state={"steps": ["turn_on"]},
        _success=False,
        _violation=None,
        _completed_steps=0,
        panel_control_geom_groups={"knob": frozenset((101, 102))},
        robot_geom_ids={200},
        panel_qpos_indexes=[0, 1, 2],
        sim=SimpleNamespace(data=SimpleNamespace(_data=SimpleNamespace(contact=[contact], ncon=1), qpos=np.zeros(3))),
        _discrete_controls=lambda: {},
        _record_control_metrics=lambda *args: samples.append(args),
    )
    PanelOperation._record_post_physics_metrics(env, 1.0)
    assert samples[-1][-1] is False
    contact.geom1, contact.geom2 = contact.geom2, contact.geom1
    PanelOperation._record_post_physics_metrics(env, 1.1)
    assert samples[-1][-1] is False
    contact.dist = 0.00101
    PanelOperation._record_post_physics_metrics(env, 1.2)
    assert samples[-1][-1] is True
