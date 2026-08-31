"""Integration gates for the Phase 04 VibrationPickPlaceCan environment."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

import robosuite
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.utils.shakebench_metrics import (
    CANONICAL_CAN_COLLISION_ENVELOPE,
    CANONICAL_CAN_INERTIA_KG_M2,
    CANONICAL_CAN_MASS_KG,
    CONTACT_INTERFACE_FINGER_OBJECT,
    CONTACT_INTERFACE_TABLE_OBJECT,
    CONTACT_INTERFACE_TARGET_OBJECT,
    equivalent_cylinder_inertia,
    extract_can_collision_envelope,
)


def _make_env(**kwargs):
    options = {
        "robots": "Panda",
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": False,
        "control_freq": 20,
        "model_timestep": 0.0002,
        "horizon": 20,
        "seed": 7,
    }
    options.update(kwargs)
    return robosuite.make("VibrationPickPlaceCan", **options)


def test_environment_is_registered_and_has_headless_lifecycle():
    assert "VibrationPickPlaceCan" in REGISTERED_ENVS
    env = _make_env()
    try:
        assert env.sim is not None
        assert env.reset() is not None
        action = np.zeros(env.action_dim)
        _, _, _, _ = env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize("robots", ["Sawyer", ["Panda", "Panda"]])
def test_environment_fails_closed_to_single_panda(robots):
    with pytest.raises((AssertionError, ValueError), match="Panda|single"):
        _make_env(robots=robots)


def test_compiled_topology_inertia_and_target_geometry_are_auditable():
    env = _make_env()
    try:
        audit = env.audit_compiled_model()
        model = env.sim.model

        assert audit["can"]["mass_kg"] == pytest.approx(CANONICAL_CAN_MASS_KG, rel=0, abs=1e-12)
        assert audit["can"]["inertia_kg_m2"] == pytest.approx(CANONICAL_CAN_INERTIA_KG_M2, rel=0, abs=1e-12)
        assert audit["topology"]["can_parent"] is None
        assert audit["topology"]["robot_base_parent"] == audit["topology"]["deck_body"]
        assert audit["topology"]["worktable_parent"] == audit["topology"]["deck_body"]
        assert audit["topology"]["target_body"] == audit["topology"]["worktable_body"]
        assert audit["worktable"]["mass_kg"] == pytest.approx(32.0)
        assert len(audit["worktable"]["joints"]) == 6
        assert model.njnt == 6 + 1 + 7 + 2 + 1  # isolator, deck, Panda, gripper, Can

        target = audit["target_container"]
        assert target["outer_xy_m"] == pytest.approx([0.18, 0.16])
        assert target["inner_xy_m"] == pytest.approx([0.164, 0.144])
        assert target["wall_height_m"] == pytest.approx(0.035)
        assert target["bottom_thickness_m"] == pytest.approx(0.012)
        assert len(target["collision_geom_names"]) == 5
        assert target["collision_geometry"]["target_container_bottom"]["size_m"] == pytest.approx(
            [0.09, 0.08, 0.006]
        )
        assert target["collision_geometry"]["target_container_wall_xpos"]["size_m"] == pytest.approx(
            [0.004, 0.08, 0.0175]
        )
    finally:
        env.close()


def test_compiled_can_envelope_is_the_single_inertia_and_placement_authority():
    env = _make_env()
    try:
        envelope = env.can_collision_envelope
        assert envelope.source_geom_names == tuple(env.can.contact_geoms)
        assert envelope.source_model_hash == CANONICAL_CAN_COLLISION_ENVELOPE.source_model_hash
        assert extract_can_collision_envelope(
            env.sim, env.can.root_body, env.can.contact_geoms
        ) == envelope
        expected_inertia = equivalent_cylinder_inertia(
            CANONICAL_CAN_MASS_KG,
            envelope.support_radius_m,
            envelope.height_m,
        )
        assert env.can_inertia == pytest.approx(expected_inertia, rel=0.0, abs=1e-15)
        assert env.audit_compiled_model()["can"]["inertia_kg_m2"] == pytest.approx(
            expected_inertia, rel=0.0, abs=1e-15
        )
        assert env.can_placement_z_offset_m == pytest.approx(-envelope.lower_support_z_m)
        assert envelope.height_m == pytest.approx(0.08000000550552341, abs=1e-12)
        assert envelope.support_radius_m == pytest.approx(0.02509177806572465, abs=1e-12)
    finally:
        env.close()


def test_compiled_contact_roles_are_scoped_and_do_not_rewrite_geom_friction():
    env = _make_env()
    try:
        audit = env.audit_compiled_model()
        roles = audit["contacts"]["roles"]
        assert len(roles[CONTACT_INTERFACE_TABLE_OBJECT]) == 1
        assert len(roles[CONTACT_INTERFACE_TARGET_OBJECT]) == 5
        assert len(roles[CONTACT_INTERFACE_FINGER_OBJECT]) == 2
        assert audit["contacts"]["pair_count"] == 8
        assert all(item["sliding_mu"] == pytest.approx(0.30) for item in roles[CONTACT_INTERFACE_TABLE_OBJECT])
        assert all(item["sliding_mu"] == pytest.approx(0.30) for item in roles[CONTACT_INTERFACE_TARGET_OBJECT])
        assert all(item["sliding_mu"] == pytest.approx(1.00) for item in roles[CONTACT_INTERFACE_FINGER_OBJECT])

        geometry_friction = audit["contacts"]["geometry_friction"]
        geometry_bits = audit["contacts"]["geometry_contact_bits"]
        assert geometry_friction[env.can.contact_geoms[0]][0] == pytest.approx(0.95)
        assert all(geometry_friction[name][0] == pytest.approx(2.0) for name in env.finger_pad_geom_names)
        assert geometry_friction[env.arena.table_collision.get("name")][0] == pytest.approx(1.0)
        assert geometry_bits[env.can.contact_geoms[0]] == [2, 0]
        assert all(geometry_bits[name][1] == 2 for name in env.finger_pad_geom_names)
    finally:
        env.close()


def test_gamma_zero_passive_scene_has_bounded_can_contact_and_no_warnings():
    env = _make_env(horizon=60)
    try:
        env.reset()
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        report = env.get_metrics()
        assert report["success"]["subconditions"]["finger_can_contact_absent"]
        assert report["max_illegal_penetration_m"] < 0.0005
        assert np.all(np.isfinite(report["can"]["worktable"]["pose"]))
        assert report["contacts"]["total_force_by_interface_N"][CONTACT_INTERFACE_TABLE_OBJECT][2] > 0.0
        assert np.all(env.sim.data.warning.number == 0)
    finally:
        env.close()


def test_target_bottom_contact_and_wall_straddling_are_distinguishable():
    env = _make_env()
    try:
        env.reset()
        target_origin = env.target_frame_world_position()
        qpos = np.concatenate((target_origin + np.array([0.0, 0.0, 0.04]), [1.0, 0.0, 0.0, 0.0]))
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        report = env.get_metrics(update=True)
        assert report["contacts"]["by_interface"][CONTACT_INTERFACE_TARGET_OBJECT]
        assert report["contacts"]["target_bottom_contact_present"]
        assert report["success"]["subconditions"]["supported_by_target_bottom"]

        qpos[:2] += np.array([0.082 + 0.025, 0.0])
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        report = env.get_metrics(update=True)
        assert report["contacts"]["target_wall_contact_present"]
        assert not report["success"]["subconditions"]["containment"]
    finally:
        env.close()


def test_target_bottom_contact_is_not_support_without_force_or_valid_height():
    env = _make_env()
    try:
        env.reset()
        target_origin = env.target_frame_world_position()
        qpos = np.concatenate((target_origin + np.array([0.0, 0.0, 0.04]), [1.0, 0.0, 0.0, 0.0]))
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        supported = env.get_metrics(update=True)
        assert supported["success"]["subconditions"]["supported_by_target_bottom"]
        assert supported["contacts"]["target_bottom_support_force_N"] > 0.0

        qpos[2] -= 0.002
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        below = env.get_metrics(update=True)
        assert below["contacts"]["target_bottom_contact_present"]
        assert not below["success"]["subconditions"]["supported_by_target_bottom"]

        qpos[:2] = target_origin[:2] + np.array([0.107, 0.0])
        qpos[2] = target_origin[2] + 0.065
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        side = env.get_metrics(update=True)
        assert side["contacts"]["target_wall_contact_present"]
        assert not side["success"]["subconditions"]["supported_by_target_bottom"]
    finally:
        env.close()


def test_stateful_table_and_in_hand_slip_metrics_record_first_slip_and_loss():
    env = _make_env()
    try:
        env.reset()
        never_grasped = env.get_metrics(update=True)
        assert not never_grasped["contacts"]["finger_can_contact_present"]
        assert not never_grasped["finger_contact_loss_after_grasp"]
        env.step(np.zeros(env.action_dim))
        baseline = env.get_metrics()
        assert baseline["contacts"]["table_contact_present"]

        can_joint_id = env.sim.model.joint_name2id(env.can.joints[0])
        can_dof = env.sim.model.jnt_dofadr[can_joint_id]
        env.sim.data.qvel[can_dof : can_dof + 2] = [0.05, 0.0]
        env.sim.forward()
        slipping = env.get_metrics(update=True)
        assert slipping["table_slip_speed_m_s"] > 0.001
        assert slipping["first_slip_time_s"] is not None

        pad_positions = [
            env.sim.data.geom_xpos[env.sim.model.geom_name2id(name)] for name in env.finger_pad_geom_names
        ]
        hand_center = np.mean(pad_positions, axis=0)
        qpos = np.concatenate((hand_center, [1.0, 0.0, 0.0, 0.0]))
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        env.get_metrics(update=True)
        qpos[0] += 0.003
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        hand_slip = env.get_metrics(update=True)
        assert hand_slip["contacts"]["finger_can_contact_present"]
        assert hand_slip["in_hand_translation_slip_m"] == pytest.approx(0.003)

        qpos[0] += 0.2
        env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
        env.sim.forward()
        released = env.get_metrics(update=True)
        assert released["finger_contact_loss_after_grasp"]
        assert not released["contacts"]["finger_can_contact_present"]
    finally:
        env.close()


def test_object_observables_use_an_explicit_robot_base_frame_seam():
    env = _make_env(use_object_obs=True)
    try:
        observations = env.reset()
        assert env.policy_task_state_frame == "robot_base"
        assert "can_pos_robot_base" in observations
        assert "can_quat_robot_base" in observations
        assert "can_pos" not in observations
        assert "can_quat" not in observations
        expected = env.get_metrics(update=True)["can"]["robot_base"]
        np.testing.assert_allclose(observations["can_pos_robot_base"], expected["position_m"])
    finally:
        env.close()


def test_runtime_png_is_git_tracked_and_clean_source_archive_compiles():
    repo_root = Path(__file__).resolve().parents[2]
    relative_asset = "robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_asset],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, tracked.stderr

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        dist_dir = temporary_root / "dist"
        dist_dir.mkdir()
        subprocess.run(
            [sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        archives = sorted(dist_dir.glob("*.tar.gz"))
        assert len(archives) == 1
        extract_root = temporary_root / "extract"
        extract_root.mkdir()
        with tarfile.open(archives[0], "r:gz") as archive:
            names = archive.getnames()
            assert any(name.endswith(relative_asset) for name in names)
            archive.extractall(extract_root, filter="data")
        package_root = next(extract_root.iterdir())
        extracted_asset = package_root / relative_asset
        extracted_arena = package_root / "robosuite/models/assets/arenas/shakebench_arena.xml"
        assert extracted_asset.is_file()
        assert "shakebench_phenolic_bench_dark_1k.png" in extracted_arena.read_text(encoding="utf-8")
        mujoco.MjModel.from_xml_path(str(extracted_arena))
