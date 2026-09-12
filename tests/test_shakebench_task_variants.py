"""Task state identity, contact behavior, and object-independent environment gates."""

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from robosuite.utils.shakebench_metrics import extract_can_collision_envelope
from robosuite.utils.shakebench_physics import resolve_physics_profile
from robosuite.utils.shakebench_task_states import build_task_state_artifact, verify_task_state_artifact
from robosuite.utils.shakebench_tasks import (
    OBJECT_SUPPORT,
    ShakeBenchTask,
    TaskSpec,
    legacy_oracle_observation,
    make_task_env,
    make_task_object,
    task_env_kwargs,
    task_variants,
)


@pytest.fixture(scope="module")
def official_states():
    return build_task_state_artifact("official")


def test_phase09_official_is_balanced_without_sixfold_expansion(official_states):
    payload = official_states
    assert (payload["task_count"], payload["variant_count"], payload["state_count"]) == (1, 6, 400)
    assert payload["parent_state_count"] == payload["available_parent_state_count"] == 400
    assert payload["variant_assignment"] == "round_robin_by_parent_index"
    assert list(payload["variant_state_counts"].values()) == [67, 67, 67, 67, 66, 66]
    assert payload["scoreable"] is False
    assert verify_task_state_artifact(payload)["passed"]
    assert len({row["state_id"] for row in payload["states"]}) == 400
    assert len({row["parent_state_id"] for row in payload["states"]}) == 400
    for index, row in enumerate(payload["states"]):
        assert row["task"] == task_variants()[index % 6].to_dict()
        kwargs = task_env_kwargs(row)
        assert kwargs["task"].to_dict() == row["task"]
        assert kwargs["object_start_yaw_rad"] == row["object_yaw_rad"]
        if row["task"]["object_id"] == "cookie_box":
            np.testing.assert_allclose(row["object_pose_worktable"][3:], [0.0, 0.0, 2**-0.5, 2**-0.5])
    knee = build_task_state_artifact("knee")
    assert knee["state_count"] == 600
    assert knee["variant_assignment"] == "full_cross_product"
    for start in range(0, 600, 6):
        group = knee["states"][start : start + 6]
        assert {TaskSpec.from_mapping(row["task"]).variant_id for row in group} == {
            s.variant_id for s in task_variants()
        }
        for key in ("parent_state_id", "excitation_seed", "imu_seed", "t0_s"):
            assert len({row[key] for row in group}) == 1
    assert not {r["parent_state_id"] for r in knee["states"]} & {r["parent_state_id"] for r in payload["states"]}


@pytest.mark.parametrize("mutation", ["object", "friction", "placement", "missing_variant"])
def test_task_state_tampering_fails_closed(official_states, mutation):
    payload = copy.deepcopy(official_states)
    if mutation == "object":
        payload["states"][0]["task"]["object_id"] = "wood_cube"
    elif mutation == "friction":
        payload["task_contracts"]["pick_place.metal.food_can"]["table_object_sliding_mu"] = 0.9
    elif mutation == "placement":
        payload["states"][0]["object_pose_worktable"][2] += 0.02
    else:
        payload["states"].pop()
    assert not verify_task_state_artifact(payload)["passed"]


def test_task_selection_and_initial_state_reject_unsupported_inputs(official_states):
    for kwargs in ({"object_id": "unknown"}, {"surface_id": "wood"}, {"task_type": "stack"}, {"object_id": []}):
        with pytest.raises(ValueError):
            TaskSpec(**kwargs)
    row = copy.deepcopy(official_states["states"][0])
    row["object_initial_velocity"][0] = 0.1
    with pytest.raises(ValueError, match="zero velocity"):
        task_env_kwargs(row)


def test_cookie_box_state_registers_its_narrow_side_for_grasp(official_states):
    row = next(state for state in official_states["states"] if state["task"]["object_id"] == "cookie_box")
    kwargs = task_env_kwargs(row)
    assert kwargs["object_start_yaw_rad"] == pytest.approx(np.pi / 2.0)
    np.testing.assert_allclose(row["object_pose_worktable"][3:], [0.0, 0.0, 2**-0.5, 2**-0.5])


def _sliding_probe(spec, *, vibration=False):
    """Known lateral impulse on a real stock collision mesh and pair friction."""
    obj = make_task_object(spec)
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", {"timestep": "0.001"})
    root.append(copy.deepcopy(obj.asset))
    world = ET.SubElement(root, "worldbody")
    support = ET.SubElement(world, "body", {"name": "moving_table"}) if vibration else world
    if vibration:
        ET.SubElement(support, "joint", {"name": "table_x", "type": "slide", "axis": "1 0 0"})
    ET.SubElement(
        support,
        "geom",
        {
            "name": "support",
            "type": "box",
            "size": "2 2 0.05",
            "pos": "0 0 -0.05",
            "mass": "1000000",
            "contype": "0",
            "conaffinity": "2",
        },
    )
    body = copy.deepcopy(obj.get_obj())
    body.set("pos", f"0 0 {-OBJECT_SUPPORT[spec.object_id][0]}")
    for geom in body.findall(".//geom"):
        if geom.get("name") in obj.contact_geoms:
            geom.set("contype", "2")
            geom.set("conaffinity", "0")
            geom.set("mass", str(spec.object_mass_kg))
    world.append(body)
    contact = ET.SubElement(root, "contact")
    for name in obj.contact_geoms:
        ET.SubElement(
            contact,
            "pair",
            {
                "geom1": name,
                "geom2": "support",
                **resolve_physics_profile("official").pair_attributes(spec.table_sliding_mu),
            },
        )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    envelope = extract_can_collision_envelope(model, obj.root_body, obj.contact_geoms)
    np.testing.assert_allclose(
        (envelope.lower_support_z_m, envelope.upper_support_z_m, envelope.support_radius_m),
        OBJECT_SUPPORT[spec.object_id],
        atol=1e-10,
    )
    data = mujoco.MjData(model)
    for _ in range(300):
        mujoco.mj_step(model, data)
    object_qpos = model.jnt_qposadr[model.joint(obj.joints[0]).id]
    object_dof = model.jnt_dofadr[model.joint(obj.joints[0]).id]
    start = data.qpos[object_qpos]
    peak = 0.0
    if not vibration:
        data.qvel[object_dof] = 0.15
    for index in range(1000 if vibration else 300):
        if vibration:
            t = (index + 1) * 0.001
            u = min(t / 0.3, 1.0)
            ramp = u**3 * (10 - 15 * u + 6 * u**2)
            ramp_dot = 30 * u**2 * (1 - u) ** 2 / 0.3 if t < 0.3 else 0.0
            omega = 2 * np.pi * 4
            # Set both position and velocity. Teleporting a mocap plane has no
            # tangential surface velocity and cannot test frictional entrainment.
            data.qpos[0] = 0.01 * ramp * np.sin(omega * t)
            data.qvel[0] = 0.01 * (ramp_dot * np.sin(omega * t) + ramp * omega * np.cos(omega * t))
        mujoco.mj_step(model, data)
        shift = data.qpos[0] if vibration else 0.0
        peak = max(peak, abs(data.qpos[object_qpos] - shift - start))
    assert np.all(np.isfinite(data.qpos))
    return peak


@pytest.mark.parametrize("object_id", ["food_can", "cookie_box", "bread"])
def test_mat_reduces_sliding_of_each_stock_object(object_id):
    metal = _sliding_probe(TaskSpec(object_id=object_id, surface_id="metal"))
    mat = _sliding_probe(TaskSpec(object_id=object_id, surface_id="mat"))
    assert metal > mat * 1.2


@pytest.mark.parametrize("object_id", ["food_can", "cookie_box", "bread"])
def test_mat_reduces_table_relative_slip_during_horizontal_vibration(object_id):
    metal = _sliding_probe(TaskSpec(object_id=object_id, surface_id="metal"), vibration=True)
    mat = _sliding_probe(TaskSpec(object_id=object_id, surface_id="mat"), vibration=True)
    assert metal > mat * 1.2


@pytest.mark.parametrize("spec", task_variants(), ids=lambda s: s.variant_id)
def test_compiled_task_reset_step_contact_roles_and_public_interface(spec):
    env = make_task_env(spec, observation_tier="V0", hard_reset=False, horizon=3, seed=17)
    try:
        assert isinstance(env, ShakeBenchTask)
        observation = env.reset()
        assert set(observation) == set(env.policy_observation_keys) == set(env.observation_contract())
        assert "object_pos_robot_base" in observation and not any(key.startswith("can_") for key in observation)
        assert "can_pos_robot_base" in legacy_oracle_observation(observation)
        assert env.get_task_context()["task"] == spec.contract()
        assert env.get_task_context()["object"]["mass_kg"] == spec.object_mass_kg
        model = env.sim.model._model
        assert model.body("worktable").mass == 32.0
        audit = env._compiled_contract
        for pair in audit["contacts"]["pairs"]:
            expected = {
                "table_object": spec.table_sliding_mu,
                "target_object": spec.target_sliding_mu,
                "finger_object": 1.0,
            }[pair["interface"]]
            np.testing.assert_allclose(pair["friction"][:2], expected)
        support = "table_mat_collision" if spec.surface_id == "mat" else "table_collision"
        assert env.table_contact_geom_names == (support,)
        start = observation["object_pos_robot_base"].copy()
        for _ in range(2):
            obs, _, _, _ = env.step(np.zeros(env.action_dim))
            assert np.all(np.isfinite(obs["object_pos_robot_base"]))
        reset = env.reset()
        np.testing.assert_allclose(reset["object_pos_robot_base"], start, atol=1e-10)
        assert env.get_policy_task_context()["physics_profile"]["scoreable"] is False
    finally:
        env.close()
