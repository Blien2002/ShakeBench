"""Public-seam tests for the Phase 04 task metrics and success latch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robosuite.utils.shakebench_metrics import (
    DEFAULT_SUCCESS_THRESHOLDS,
    SuccessSnapshot,
    VibrationSuccessEvaluator,
    relative_pose_twist,
    verify_phase04_environment_artifact,
)


def _snapshot(**overrides):
    values = {
        "support_points_target_xy": np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]], dtype=float),
        "target_inner_half_extents_m": (0.082, 0.072),
        "target_bottom_contact_present": True,
        "target_bottom_support_force_N": DEFAULT_SUCCESS_THRESHOLDS.target_bottom_support_force_threshold_N + 1.0,
        "lower_support_z_m": 0.0,
        "finger_can_contact_present": False,
        "relative_linear_speed_m_s": 0.0,
        "relative_angular_speed_rad_s": 0.0,
        "illegal_penetration_m": 0.0,
    }
    values.update(overrides)
    return SuccessSnapshot(**values)


def test_success_latch_requires_the_full_half_second_and_is_continuous():
    evaluator = VibrationSuccessEvaluator()

    assert not evaluator.evaluate(_snapshot(), time_s=0.0).passed
    assert not evaluator.evaluate(_snapshot(), time_s=0.499999).passed
    assert evaluator.evaluate(_snapshot(), time_s=0.5).passed
    assert evaluator.evaluate(_snapshot(finger_can_contact_present=True), time_s=0.6).passed

    evaluator.reset()
    assert not evaluator.evaluate(_snapshot(), time_s=0.0).passed
    assert not evaluator.evaluate(_snapshot(target_bottom_contact_present=False), time_s=0.2).passed
    assert not evaluator.evaluate(_snapshot(), time_s=0.5).passed


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_bottom_contact_present", False),
        ("finger_can_contact_present", True),
        ("relative_linear_speed_m_s", DEFAULT_SUCCESS_THRESHOLDS.max_relative_linear_speed_m_s),
        ("relative_angular_speed_rad_s", DEFAULT_SUCCESS_THRESHOLDS.max_relative_angular_speed_rad_s),
        ("illegal_penetration_m", DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m),
    ],
)
def test_each_success_subcondition_has_a_strict_negative_boundary(field, value):
    evaluator = VibrationSuccessEvaluator()
    result = evaluator.evaluate(_snapshot(**{field: value}), time_s=0.5)

    assert not result.passed
    assert not all(result.subconditions.values())
    assert not result.subconditions[
        {
            "target_bottom_contact_present": "supported_by_target_bottom",
            "finger_can_contact_present": "finger_can_contact_absent",
            "relative_linear_speed_m_s": "relative_linear_speed",
            "relative_angular_speed_rad_s": "relative_angular_speed",
            "illegal_penetration_m": "illegal_penetration",
        }[field]
    ]


def test_target_bottom_support_requires_strict_force_and_height_boundaries():
    threshold = DEFAULT_SUCCESS_THRESHOLDS.target_bottom_support_force_threshold_N
    evaluator = VibrationSuccessEvaluator()

    force_boundary = _snapshot(target_bottom_support_force_N=threshold)
    result = evaluator.evaluate(force_boundary, time_s=0.0)
    assert not result.subconditions["supported_by_target_bottom"]

    evaluator.reset()
    force_positive = _snapshot(target_bottom_support_force_N=threshold + 1e-9)
    assert not evaluator.evaluate(force_positive, time_s=0.0).passed
    assert evaluator.evaluate(force_positive, time_s=0.5).subconditions["supported_by_target_bottom"]

    evaluator.reset()
    height_boundary = _snapshot(
        lower_support_z_m=-DEFAULT_SUCCESS_THRESHOLDS.target_bottom_support_z_tolerance_m
    )
    assert evaluator.evaluate(height_boundary, time_s=0.0).subconditions["supported_by_target_bottom"]
    height_below = _snapshot(
        lower_support_z_m=-DEFAULT_SUCCESS_THRESHOLDS.target_bottom_support_z_tolerance_m - 1e-9
    )
    assert not VibrationSuccessEvaluator().evaluate(height_below, time_s=0.0).subconditions[
        "supported_by_target_bottom"
    ]


def test_containment_uses_collision_support_points_and_allows_height_above_wall():
    inside = _snapshot()
    assert inside.containment

    wall_straddling = _snapshot(
        support_points_target_xy=np.array([[-0.082, -0.02], [0.083, -0.02], [0.083, 0.02], [-0.082, 0.02]], dtype=float)
    )
    assert not wall_straddling.containment

    high_can = _snapshot()
    assert high_can.containment


def test_snapshot_rejects_invalid_support_shape_and_nonfinite_values():
    with pytest.raises(ValueError):
        _snapshot(support_points_target_xy=np.zeros((4, 3)))
    with pytest.raises(ValueError):
        _snapshot(relative_linear_speed_m_s=np.nan)


def test_relative_pose_and_twist_are_invariant_under_rigid_deck_motion():
    angle = 0.43
    frame_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    frame_position = np.array([1.2, -0.4, 0.8])
    frame_twist = np.array([0.4, -0.2, 0.1, 0.0, 0.0, 1.3])
    relative_position = np.array([0.2, 0.1, 0.3])
    child_position = frame_position + frame_rotation.dot(relative_position)
    child_twist = frame_twist.copy()
    child_twist[:3] += np.cross(frame_twist[3:], child_position - frame_position)

    result = relative_pose_twist(
        child_position,
        frame_rotation,
        child_twist,
        frame_position,
        frame_rotation,
        frame_twist,
    )

    np.testing.assert_allclose(result.position_m, relative_position, atol=1e-12)
    np.testing.assert_allclose(result.linear_velocity_m_s, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.angular_velocity_rad_s, np.zeros(3), atol=1e-12)


def test_phase04_evidence_artifact_is_machine_readable_and_scope_locked():
    artifact = json.loads(
        (Path(__file__).with_name("shakebench_phase_04_environment.json")).read_text(encoding="utf-8")
    )
    assert artifact["schema_id"] == "shakebench.phase04.environment"
    assert artifact["status"] == "PASS"
    assert artifact["environment"]["robots"] == ["Panda"]
    assert artifact["compiled"]["contact_pair_count"] == 8
    assert artifact["compiled"]["contact_roles"] == {
        "table_object": 1,
        "target_object": 5,
        "finger_object": 2,
    }
    assert all(probe["passed"] for probe in artifact["probes"].values())


def test_phase04_artifact_verifier_is_read_only_and_hash_locked(tmp_path):
    artifact_path = Path(__file__).with_name("shakebench_phase_04_environment.json")
    before = artifact_path.read_bytes()
    summary = verify_phase04_environment_artifact(artifact_path)
    assert summary["passed"]
    assert summary["integrity_valid"]
    assert artifact_path.read_bytes() == before

    mutated = json.loads(before.decode("utf-8"))
    mutated["compiled"]["can"]["mass_kg"] = 0.350
    mutated_path = tmp_path / "mutated_phase04.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    mutation_summary = verify_phase04_environment_artifact(mutated_path)
    assert not mutation_summary["passed"]
    assert not mutation_summary["integrity_valid"]
