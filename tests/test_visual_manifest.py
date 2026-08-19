from __future__ import annotations

from pathlib import Path
import math

import pytest
import torch

from vibench.config import BenchmarkConfig, VibrationConfig
from vibench.controller import (
    grasp_feasibility,
    grasp_feasibility_table,
    collision_safe_descend_clearance,
    latch_finger_contact_targets,
    projected_half_height,
    rate_limit_joint_target,
    rate_limit_translation,
    short_axis_yaw,
)
from vibench.diagnostics import classify_penetration_pair
from vibench.supports import support_group_geometries
from vibench.vibration import offline_support_travel_report
from vibench.visual_manifest import (
    load_visual_manifest,
    prim_anchor_audit,
    visual_feature_facts,
)


def test_default_profile_is_official_and_unassisted() -> None:
    cfg = BenchmarkConfig()
    assert cfg.physics_hz == 1000
    assert cfg.solver_substeps == 5
    assert cfg.effective_substep_hz == 5000
    assert cfg.grasp_assist is False
    assert cfg.contact_solref == (0.00060, 1.0)
    assert cfg.resolved_robot_base[2] - cfg.robot_base[2] == pytest.approx(cfg.assembly_clearance_m)
    assert cfg.resolved_worktable_center == cfg.worktable_center
    assert cfg.resolved_target_center == cfg.target_center
    assert cfg.max_substep_displacement_m == pytest.approx(0.00040)
    assert cfg.score_penetration_threshold_mm == pytest.approx(0.3360225, abs=1e-6)


def test_penetration_pair_classifier_covers_v0_pairs() -> None:
    pairs = {
        classify_penetration_pair("/World/envs/env_0/Workpiece/mesh", "/World/envs/env_0/WorkTableTop/mesh"),
        classify_penetration_pair("/World/envs/env_0/Workpiece/mesh", "/World/envs/env_0/Robot/panda_leftfinger/mesh"),
        classify_penetration_pair("/World/envs/env_0/Workpiece/mesh", "/World/envs/env_0/Robot/panda_rightfinger/mesh"),
        classify_penetration_pair("/World/envs/env_0/Workpiece/mesh", "/World/envs/env_0/TargetBin/Walls/mesh"),
        classify_penetration_pair("/World/envs/env_0/WorkTableLegFL/mesh", "/World/envs/env_0/VibrationFloor/mesh"),
        classify_penetration_pair("/World/envs/env_0/Robot/panda_link3/mesh", "/World/envs/env_0/VibrationFloor/mesh"),
    }
    assert None not in pairs
    assert len(pairs) == 6


def test_official_rate_keeps_replayed_substep_travel_below_geometry_gate() -> None:
    cfg = BenchmarkConfig()
    groups = support_group_geometries(cfg)
    official = offline_support_travel_report(cfg, groups, 1000, 5)
    training = offline_support_travel_report(cfg, groups, 240, 4)
    assert official.max_substep_travel_m < cfg.max_substep_displacement_m
    assert training.max_substep_travel_m > cfg.max_substep_displacement_m


def test_training_spectral_profile_is_rejected_before_scene_build() -> None:
    cfg = BenchmarkConfig()
    groups = support_group_geometries(cfg)
    training = offline_support_travel_report(cfg, groups, 240, 4)
    assert training.max_substep_travel_m > cfg.max_substep_displacement_m
    safe = BenchmarkConfig(vibration=VibrationConfig(mode="spectral", spectral_scale=0.15))
    safe_report = offline_support_travel_report(safe, support_group_geometries(safe), 240, 4)
    assert safe_report.max_substep_travel_m < cfg.max_substep_displacement_m


def test_deterministic_sine_startup_gate_rejects_unsafe_motion() -> None:
    unsafe = BenchmarkConfig(
        vibration=VibrationConfig(
            mode="sine", sine_axis="tz", sine_amplitude=0.01, sine_frequency_hz=50.0
        )
    )
    report = offline_support_travel_report(unsafe, support_group_geometries(unsafe), 240, 4)
    assert report.max_substep_travel_m > unsafe.max_substep_displacement_m


def test_grasp_feasibility_table_and_short_axis_alignment() -> None:
    rows = grasp_feasibility_table()
    assert len(rows) == 12
    feasible, minimum = grasp_feasibility("sugar_box", 0.75)
    assert feasible
    assert minimum == pytest.approx(0.06950775, abs=1.0e-8)
    assert grasp_feasibility("cracker_box", 0.75)[0] is False
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(short_axis_yaw(identity, (0.06675, 0.13125, 0.0285)), torch.zeros(1))
    torch.testing.assert_close(projected_half_height(identity, (0.06675, 0.13125, 0.0285)), torch.tensor([0.01425]))


def test_controller_targets_are_rate_limited_without_overshoot() -> None:
    current_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    target_xyz = torch.tensor([[0.0, 0.0, -0.20], [1.0, 2.0, 3.00001]])
    limited_xyz = rate_limit_translation(current_xyz, target_xyz, 0.001)
    torch.testing.assert_close(limited_xyz[0], torch.tensor([0.0, 0.0, -0.001]))
    torch.testing.assert_close(limited_xyz[1], target_xyz[1])

    current_joint = torch.tensor([[0.040, 0.040]])
    target_joint = torch.tensor([[0.012, 0.012]])
    limited_joint = rate_limit_joint_target(current_joint, target_joint, 0.00002)
    torch.testing.assert_close(limited_joint, torch.tensor([[0.03998, 0.03998]]))


def test_each_finger_latches_independently_at_first_contact() -> None:
    current = torch.tensor([[0.0350, 0.0340]])
    commanded = torch.tensor([[0.0348, 0.0338]])
    desired = torch.tensor([[0.0120, 0.0120]])
    latched = torch.zeros((1, 2), dtype=torch.bool)
    first, latched, targets = latch_finger_contact_targets(
        current,
        commanded,
        desired,
        torch.tensor([[True, False]]),
        latched,
        None,
        0.0001,
    )
    torch.testing.assert_close(first, torch.tensor([[0.0347, 0.0120]]))
    assert latched.tolist() == [[True, False]]
    second, latched, targets = latch_finger_contact_targets(
        torch.tensor([[0.0349, 0.0325]]),
        torch.tensor([[0.0347, 0.0323]]),
        desired,
        torch.tensor([[False, True]]),
        latched,
        targets,
        0.0001,
    )
    torch.testing.assert_close(second, torch.tensor([[0.0347, 0.0322]]))
    assert latched.all()


def test_controller_motion_limits_are_validated_in_config() -> None:
    cfg = BenchmarkConfig()
    assert cfg.approach_clearance_m == pytest.approx(0.080)
    assert cfg.descend_linear_speed_m_s * cfg.descend_timeout_s >= (
        cfg.approach_clearance_m - cfg.descend_clearance_m
    )
    assert cfg.gripper_contact_preload_m == pytest.approx(0.0003)
    assert cfg.lift_takeoff_speed_m_s < cfg.arm_linear_speed_m_s
    assert cfg.gripper_closing_speed_m_s == pytest.approx(0.003)
    assert cfg.gripper_contact_recovery_speed_m_s == pytest.approx(0.001)
    assert cfg.grasp_contact_loss_timeout_s < cfg.grasp_timeout_s
    assert cfg.grasp_slip_tolerance_m == pytest.approx(0.008)


def test_descend_clearance_accounts_for_full_finger_collider_reach() -> None:
    object_height = torch.tensor([0.033602, 0.080])
    clearance = collision_safe_descend_clearance(0.004, 0.05385, object_height, 0.012)
    torch.testing.assert_close(clearance, torch.tensor([0.032248, 0.004000]), atol=1.0e-6, rtol=0.0)


def test_visual_manifest_matches_scene_authoring_facts() -> None:
    manifest = load_visual_manifest()
    assert manifest["schema_version"] == 1
    facts = visual_feature_facts()
    features = manifest["features"]
    assert len(features) >= 28
    for feature in features:
        assert feature["prim_path"] and feature["parent_prim"] and feature["material_binding"]
        actual = facts[feature["metric"]]
        assert abs(actual - float(feature["expected"])) <= float(feature["tolerance"]), feature


def test_all_visual_attachments_are_within_five_mm_of_anchor() -> None:
    audit = prim_anchor_audit()
    assert len(audit) >= 30
    assert max(entry.error_m for entry in audit) < 0.005
    bolt_entries = [entry for entry in audit if entry.name.startswith("robot_mount_bolt_")]
    assert len(bolt_entries) == 8
    assert all(entry.parent_prim.endswith("RobotMountFlange") for entry in bolt_entries)


def test_parent_transform_fixes_remain_local_in_source() -> None:
    root = Path(__file__).resolve().parents[1]
    visuals = (root / "src" / "vibench" / "visual_assets.py").read_text(encoding="utf-8")
    arena = (root / "src" / "vibench" / "arena.py").read_text(encoding="utf-8")
    assert "(0.092 * math.cos(angle), 0.092 * math.sin(angle), 0.010)" in visuals
    assert "Gf.Vec3d(dx, dy, 0.008)" in visuals
    assert 'create_prim(sensor_root, "Xform"' in visuals
    assert 'create_prim(post_root, "Xform"' in arena


def test_settle_phase_holds_reset_pose_instead_of_sweeping_through_table() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = (root / "src" / "vibench" / "controller.py").read_text(encoding="utf-8")
    assert 'if phase.name == "settle"' in controller
    assert 'self.settle_pose_b = obs["ee_pose_b"].clone()' in controller
    assert 'self.orientation_b = self.settle_pose_b[:, 3:7]' in controller
    assert "arm = self.settle_joint_position" in controller
    assert "latch_finger_contact_targets(" in controller


def test_run_script_serializes_usd_build_by_default_but_allows_override() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run.sh").read_text(encoding="utf-8")
    assert 'export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-1}"' in runner


def test_support_authoring_has_single_anchor_and_no_legacy_pitch_sign() -> None:
    root = Path(__file__).resolve().parents[1]
    supports = (root / "src" / "vibench" / "supports.py").read_text(encoding="utf-8")
    task = (root / "src" / "vibench" / "task.py").read_text(encoding="utf-8")
    assert "quat_apply(quat, local - anchor)" in supports
    assert "quat_from_euler_xyz(q[:, 3], q[:, 4], q[:, 5])" in supports
    assert "quat_from_euler_xyz(motion[:, 3], -motion[:, 4], motion[:, 5])" not in task
    assert "write_support_groups(" in task
