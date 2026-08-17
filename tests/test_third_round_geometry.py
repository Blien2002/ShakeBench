from __future__ import annotations

import math

import torch

from vibration_benchmark_v2.arena import load_room_arena_cfg, static_equipment_geometry_report
from vibration_benchmark_v2.config import BenchmarkConfig
from vibration_benchmark_v2.shaker import (
    ShakerGeometryCfg,
    actuator_platen_overlap_violations,
    platen_joint_clearance_report,
)
from vibration_benchmark_v2.visual_assets import platform_shadow_layout
from vibration_benchmark_v2.vibration import SpectralVibration


def _platform_pose(benchmark: BenchmarkConfig, displacement: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = displacement[3:]
    cr, sr = torch.cos(rx / 2.0), torch.sin(rx / 2.0)
    cp, sp = torch.cos(ry / 2.0), torch.sin(ry / 2.0)
    cy, sy = torch.cos(rz / 2.0), torch.sin(rz / 2.0)
    quaternion = torch.stack(
        (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    )
    return torch.cat((torch.tensor(benchmark.platform_center) + displacement[:3], quaternion))


def test_platen_joint_xy_clearance() -> None:
    report = platen_joint_clearance_report(ShakerGeometryCfg())
    assert report["x_clearance_m"] >= 0.0, report
    assert report["y_clearance_m"] >= 0.0, report


def test_platen_joint_is_fully_below_skirt() -> None:
    report = platen_joint_clearance_report(ShakerGeometryCfg())
    assert report["z_clearance_m"] >= 0.0, report


def test_all_actuator_segments_clear_platen_for_complete_episode() -> None:
    benchmark = BenchmarkConfig()
    source = SpectralVibration(benchmark.vibration, 1, "cpu")
    for step in range(1000):
        displacement, _, _ = source.sample(step * benchmark.dt)
        pose = _platform_pose(benchmark, displacement[0]).unsqueeze(0)
        violations = actuator_platen_overlap_violations(pose, benchmark.shaker, benchmark.platform_size)
        assert not violations, {"step": step, "time_s": step * benchmark.dt, "first": violations[0]}


def test_static_equipment_is_grounded_without_post_shelf_overlap() -> None:
    report = static_equipment_geometry_report(load_room_arena_cfg())
    ground_errors = {name: value for name, value in report.items() if name.endswith("ground_error_m")}
    assert all(abs(value) <= 0.002 for value in ground_errors.values()), report
    assert report["tool_cart_post_below_shelf_m"] <= 0.040, report


def test_shadow_layout_is_complete_and_aligned() -> None:
    actual = platform_shadow_layout()
    expected = {
        "table_foot_0": (-0.09, -0.245),
        "table_foot_1": (-0.09, 0.245),
        "table_foot_2": (0.45, -0.245),
        "table_foot_3": (0.45, 0.245),
        "robot_base": (-0.47, 0.0),
        "target_bin": (0.08, 0.17),
        "workpiece": (0.08, -0.13),
        "platen": (0.0, 0.0),
    }
    missing = sorted(set(expected) - set(actual))
    errors = {
        name: math.dist(actual[name], point)
        for name, point in expected.items()
        if name in actual
    }
    assert not missing, {"missing": missing, "actual": actual}
    assert all(value < 0.003 for value in errors.values()), errors
