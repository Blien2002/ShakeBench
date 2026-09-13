"""Appearance acceptance: cabinet orientation and continuous safety rail joints."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from robosuite import models
from robosuite.environments.manipulation.vibration_pick_place import VibrationPickPlace


@pytest.fixture(scope="module")
def env():
    env = VibrationPickPlace(
        robots="Panda",
        geometry_profile="world_fixed_arm_v1",
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=1,
    )
    yield env
    env.close()


def test_cabinet_operator_face_points_towards_shaker_and_monitor_is_blank(env):
    model, data = env.sim.model._model, env.sim.data._data
    cabinet = data.body("shakebench_control_cabinet")
    inward = -cabinet.xpos.copy()
    inward[2] = 0
    inward /= np.linalg.norm(inward)
    normal = cabinet.xmat.reshape(3, 3) @ np.array([0, -1, 0])
    assert np.dot(inward, normal) > 0.99999
    names = [model.geom(i).name or "" for i in range(model.ngeom)]
    assert not any(name.startswith("shakebench_instrument_bench_display_bar_") for name in names)
    screen = model.geom("shakebench_instrument_bench_monitor_screen")
    assert model.mat_emission[screen.matid] == 0
    # The static direct-mount world must retain the last authored furniture layout.
    np.testing.assert_allclose(data.body("shakebench_tool_cart").xpos, [-0.78, 2.12, 0], atol=1e-12)
    np.testing.assert_allclose(data.body("shakebench_storage_cabinet").xpos, [-2.2, 0.72, 0], atol=1e-12)
    assert model.geom("shakebench_storage_cabinet_sensor_tray_base").contype == 0


def test_curved_elbow_centres_meet_straight_tube_ends(env):
    model, data = env.sim.model._model, env.sim.data._data
    ends = []
    for suffix in ("0", "1", "2_south", "2_north", "3"):
        geom = model.geom(f"shakebench_guardrail_top_{suffix}")
        pose = data.geom(geom.id)
        axis = pose.xmat.reshape(3, 3)[:, 2]
        ends.extend([pose.xpos - axis * geom.size[1], pose.xpos + axis * geom.size[1]])
    elbow_ends = []
    ring_size = env.scene_config.section("pit")["guardrail_details"]["elbow_ring_segments"]
    for i in range(4):
        geom = model.geom(f"shakebench_guardrail_elbow_{i}")
        mesh_id = model.geom_dataid[geom.id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[start : start + count]
        pose = data.geom(geom.id)
        world = verts @ pose.xmat.reshape(3, 3).T + pose.xpos
        elbow_ends.extend([world[:ring_size].mean(axis=0), world[-ring_size:].mean(axis=0)])
    distances = np.linalg.norm(np.asarray(ends)[:, None, :] - np.asarray(elbow_ends)[None, :, :], axis=2)
    assert np.all(distances.min(axis=0) < 1e-6)
    opening_ends = np.asarray(ends)[distances.min(axis=1) > 1e-6]
    np.testing.assert_allclose(np.sort(opening_ends[:, 1]), [-0.46, 0.46], atol=1e-6)
    assert env._scene_clearance.passed


def test_only_one_explicit_light_casts_shadows(env):
    model = env.sim.model._model
    assert np.count_nonzero(model.light_castshadow) == 1
    assert model.light("shakebench_key_light").castshadow
    assert not model.light("shakebench_fill_light").castshadow


def test_robot_is_seated_on_ground_anchored_metal_plate(env):
    model, data = env.sim.model._model, env.sim.data._data
    foundation = model.geom("robot_support_foundation")
    plate = model.geom("robot_support_mount_plate")
    assert model.body("robot_support").parentid == 0
    assert model.body("robot_support").jntnum == 0
    assert plate.bodyid == foundation.bodyid
    assert plate.size[0] == plate.size[1]
    assert model.material(int(plate.matid[0])).name == "shakebench_frame_metal"
    lower = data.geom(plate.id).xpos - plate.size
    upper = data.geom(plate.id).xpos + plate.size
    np.testing.assert_allclose(lower[2], data.geom(foundation.id).xpos[2] + foundation.size[2], atol=1e-12)
    np.testing.assert_allclose(data.body("robot0_link0").xpos[2], upper[2], atol=1e-12)
    # Check actual mesh vertices at the installation plane, not its bounding sphere.
    base = model.geom("robot0_link0_collision")
    mesh = model.mesh(int(model.geom_dataid[base.id]))
    vertices = model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]]
    world = vertices @ data.geom(base.id).xmat.reshape(3, 3).T + data.geom(base.id).xpos
    footprint = world[world[:, 2] < upper[2] + 0.002, :2]
    assert footprint.size
    assert np.all(footprint.min(0) >= lower[:2]) and np.all(footprint.max(0) <= upper[:2])
    for index in range(4):
        washer = model.geom(f"robot_support_plate_anchor_{index}_washer")
        np.testing.assert_allclose(data.geom(washer.id).xpos[2] - washer.size[1], upper[2], atol=1e-12)


def test_opening_rail_ends_have_anchored_vertical_posts(env):
    model, data = env.sim.model._model, env.sim.data._data
    for side, sign in (("south", 1), ("north", -1)):
        rail = model.geom(f"shakebench_guardrail_top_2_{side}")
        end = data.geom(rail.id).xpos + sign * rail.size[1] * data.geom(rail.id).xmat.reshape(3, 3)[:, 2]
        post = model.geom(f"shakebench_guardrail_opening_{side}_post")
        foot = model.geom(f"shakebench_guardrail_opening_{side}_foot")
        np.testing.assert_allclose(data.geom(post.id).xpos + [0, 0, post.size[1]], end, atol=1e-12)
        np.testing.assert_allclose(
            data.geom(post.id).xpos[2] - post.size[1], data.geom(foot.id).xpos[2] + foot.size[2], atol=1e-12
        )
        np.testing.assert_allclose(data.geom(foot.id).xpos[2] - foot.size[2], 0, atol=1e-12)
        np.testing.assert_allclose(post.rgba, model.geom("shakebench_guardrail_post_0").rgba)
        assert env._scene_audit.visual_geoms[post.name]["frame"] == "world"


def test_support_mjcf_compiles_natively_and_matches_environment(env):
    source = Path(models.assets_root, env.geometry_profile["robot_support_mjcf"])
    # Resolve the same material names without any Python scene builder.
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><material name="shakebench_floor_slab"/>'
        '<material name="shakebench_frame_metal"/></asset>'
        '<include file="support.xml"/></mujoco>',
        assets={"support.xml": source.read_bytes()},
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    runtime_model, runtime_data = env.sim.model._model, env.sim.data._data
    source_names = {model.geom(i).name for i in range(model.ngeom)}
    runtime_names = {
        runtime_model.geom(i).name
        for i in range(runtime_model.ngeom)
        if int(runtime_model.geom_bodyid[i]) == runtime_model.body("robot_support").id
    }
    assert source_names == runtime_names
    assert model.body("robot_support").jntnum == 0
    for name in source_names:
        source_geom, runtime_geom = model.geom(name), runtime_model.geom(name)
        np.testing.assert_array_equal(source_geom.size, runtime_geom.size)
        np.testing.assert_array_equal(data.geom(source_geom.id).xpos, runtime_data.geom(runtime_geom.id).xpos)
        np.testing.assert_array_equal(source_geom.contype, runtime_geom.contype)
        np.testing.assert_array_equal(source_geom.conaffinity, runtime_geom.conaffinity)


def test_geometry_rejects_modified_support_mjcf(tmp_path, monkeypatch):
    from robosuite.utils.shakebench_geometry import load_geometry_profile

    assets = Path(models.assets_root)
    profile = load_geometry_profile()
    support = tmp_path / profile["robot_support_mjcf"]
    support.parent.mkdir(parents=True)
    support.write_bytes((assets / profile["robot_support_mjcf"]).read_bytes() + b"\n<!-- changed -->\n")
    (tmp_path / "shakebench_geometry_world_fixed_arm_v1.json").write_bytes(
        (assets / "shakebench_geometry_world_fixed_arm_v1.json").read_bytes()
    )
    monkeypatch.setattr(models, "assets_root", str(tmp_path))
    with pytest.raises(ValueError, match="robot support MJCF hash mismatch"):
        load_geometry_profile()
