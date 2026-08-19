from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

from vibench.arena import load_room_arena_cfg
from vibench.config import BenchmarkConfig, VibrationConfig
from vibench.shaker import ShakerGeometryCfg, joint_points, solve_leg_transforms
from vibench.supports import (
    angular_velocity_from_euler_rates,
    support_group_geometries,
    support_pose_velocity,
)
from vibench.vibration import (
    SpectralVibration,
    _euler_rotation_matrices,
    _synthesize_episode,
)
from vibench.wrist_camera import (
    WRIST_CAMERA_AIM_H,
    WRIST_CAMERA_EYE_H,
    WRIST_CAMERA_FORWARD_H,
    WRIST_CAMERA_UP_H,
    wrist_camera_frame_from_hand,
    wrist_camera_pose_from_hand,
)


def test_seed_is_reproducible() -> None:
    cfg = VibrationConfig(mode="spectral", seed=31)
    a = SpectralVibration(cfg, 2, "cpu")
    b = SpectralVibration(cfg, 2, "cpu")
    for time_s in (0.0, 0.4, 1.0, 2.3):
        for lhs, rhs in zip(a.sample(time_s), b.sample(time_s)):
            torch.testing.assert_close(lhs, rhs)


def test_support_group_members_form_one_rigid_body() -> None:
    cfg = BenchmarkConfig()
    groups = support_group_geometries(cfg)
    q = torch.tensor([[0.003, -0.002, 0.004, 0.011, 0.0077, 0.0072]], dtype=torch.float64)
    qd = torch.tensor([[0.020, 0.030, 0.040, 0.200, 0.130, 0.045]], dtype=torch.float64)
    env_origins = torch.zeros((1, 3), dtype=torch.float64)
    quat = quat_from_euler_xyz(q[:, 3], q[:, 4], q[:, 5])
    rotation = matrix_from_quat(quat)

    for group in groups:
        anchor = torch.tensor(group.rotation_anchor, dtype=torch.float64)
        positions = []
        quats = []
        velocities = []
        locals_ = []
        for member in group.members:
            local = torch.tensor(member.local, dtype=torch.float64).unsqueeze(0)
            position, member_quat, velocity = support_pose_velocity(
                local,
                q,
                qd,
                group.rotation_anchor,
                env_origins,
            )
            positions.append(position)
            quats.append(member_quat)
            velocities.append(velocity)
            locals_.append(local)
            offset = local - anchor
            expected = env_origins + q[:, :3] + anchor + (rotation @ offset.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(position, expected, atol=1e-12, rtol=0.0)
            torch.testing.assert_close(member_quat, quat, atol=1e-12, rtol=0.0)
        for lhs in range(len(positions)):
            for rhs in range(lhs + 1, len(positions)):
                distance = float((positions[lhs] - positions[rhs]).norm())
                local_distance = float((locals_[lhs] - locals_[rhs]).norm())
                assert math.isclose(distance, local_distance, abs_tol=1.0e-9)

    # The replay rotation helper must be the same Rz·Ry·Rx matrix the writer
    # derives from quat_from_euler_xyz.  A second matrix convention here is
    # precisely the P0 double-truth bug.
    torch.testing.assert_close(
        torch.as_tensor(_euler_rotation_matrices(q.numpy()), dtype=torch.float64),
        rotation,
        atol=1.0e-15,
        rtol=0.0,
    )

    # Angular velocity must be the exact derivative of quat_from_euler_xyz's
    # Rz·Ry·Rx rotation; finite-difference Rdot and recover omega_hat.
    h = 1.0e-6
    rdot = torch.zeros((1, 3, 3), dtype=torch.float64)
    for axis in (3, 4, 5):
        plus = q.clone()
        minus = q.clone()
        plus[:, axis] += h
        minus[:, axis] -= h
        rplus = matrix_from_quat(quat_from_euler_xyz(plus[:, 3], plus[:, 4], plus[:, 5]))
        rminus = matrix_from_quat(quat_from_euler_xyz(minus[:, 3], minus[:, 4], minus[:, 5]))
        rdot += qd[:, axis] * (rplus - rminus) / (2.0 * h)
    omega_hat = rdot @ rotation.transpose(1, 2)
    omega_true = torch.stack(
        (omega_hat[:, 2, 1], omega_hat[:, 0, 2], omega_hat[:, 1, 0]),
        dim=1,
    )
    torch.testing.assert_close(
        angular_velocity_from_euler_rates(q, qd),
        omega_true,
        atol=1.0e-9,
        rtol=0.0,
    )


def test_replay_waveform_matches_simulation_sample() -> None:
    cfg = VibrationConfig(mode="spectral", seed=17)
    source = SpectralVibration(cfg, 1, "cpu")
    time_s, q, qd, qdd = _synthesize_episode(cfg, 1000, 2.0, 0)
    for step in (0, 1, 137, 500, 999):
        simulated = source.sample(float(time_s[step]))
        torch.testing.assert_close(
            simulated[0][0],
            torch.as_tensor(q[step], dtype=torch.float32),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            simulated[1][0],
            torch.as_tensor(qd[step], dtype=torch.float32),
            atol=1.0e-4,
            rtol=1.0e-4,
        )
        torch.testing.assert_close(
            simulated[2][0],
            torch.as_tensor(qdd[step], dtype=torch.float32),
            atol=1.0e-3,
            rtol=1.0e-3,
        )


def test_active_axes_use_independent_random_streams() -> None:
    source = SpectralVibration(VibrationConfig(mode="spectral"), 2, "cpu")
    # Regression for the pre-fix bug where every axis reused the same RNG
    # stream: same-tone axes had byte-identical phase vectors.
    assert not torch.equal(source._phase["tx"], source._phase["tz"])
    assert not torch.equal(source._phase["ty"], source._phase["ry"])
    assert not torch.equal(source._phase["tx"][0], source._phase["tx"][1])


def test_all_six_axes_are_present_after_ramp() -> None:
    source = SpectralVibration(VibrationConfig(mode="spectral"), 1, "cpu")
    displacement, _, _ = source.sample(1.37)
    assert torch.all(displacement.abs() > 0.0)


def test_spectral_scale_changes_all_six_axes_and_is_reproducible() -> None:
    full = SpectralVibration(VibrationConfig(mode="spectral", spectral_scale=1.0), 1, "cpu")
    scaled = SpectralVibration(VibrationConfig(mode="spectral", spectral_scale=0.15), 1, "cpu")
    for full_value, scaled_value in zip(full.sample(1.37), scaled.sample(1.37)):
        torch.testing.assert_close(scaled_value, 0.15 * full_value)


def test_spectral_active_axes_zero_unselected_channels() -> None:
    vibration = SpectralVibration(
        VibrationConfig(mode="spectral", spectral_scale=0.25, active_axes=("tx", "tz")),
        1,
        "cpu",
    )
    q, qd, qdd = vibration.sample(1.25)
    assert torch.all(q[:, [0, 2]] != 0.0)
    assert torch.all(qd[:, [0, 2]] != 0.0)
    assert torch.all(qdd[:, [0, 2]] != 0.0)
    zeros = torch.zeros((1, 4))
    torch.testing.assert_close(q[:, [1, 3, 4, 5]], zeros)
    torch.testing.assert_close(qd[:, [1, 3, 4, 5]], zeros)
    torch.testing.assert_close(qdd[:, [1, 3, 4, 5]], zeros)


def test_vehicle_three_dof_axes_include_pitch_and_zero_other_channels() -> None:
    vibration = SpectralVibration(
        VibrationConfig(mode="spectral", active_axes=("tx", "tz", "ry")),
        1,
        "cpu",
    )
    q, qd, qdd = vibration.sample(1.25)
    assert torch.all(q[:, [0, 2, 4]] != 0.0)
    assert torch.all(qd[:, [0, 2, 4]] != 0.0)
    assert torch.all(qdd[:, [0, 2, 4]] != 0.0)
    zeros = torch.zeros((1, 3))
    torch.testing.assert_close(q[:, [1, 3, 5]], zeros)
    torch.testing.assert_close(qd[:, [1, 3, 5]], zeros)
    torch.testing.assert_close(qdd[:, [1, 3, 5]], zeros)


def test_spectral_active_axes_validation() -> None:
    for active_axes in ((), ("tx", "bad"), ("tx", "tx")):
        try:
            VibrationConfig(mode="spectral", active_axes=active_axes)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid active axes accepted: {active_axes}")


def test_off_mode_is_zero() -> None:
    source = SpectralVibration(VibrationConfig(mode="off"), 3, "cpu")
    for value in source.sample(2.0):
        assert torch.count_nonzero(value) == 0


def test_stewart_leg_lengths_match_direct_analytic_geometry() -> None:
    cfg = ShakerGeometryCfg()
    angle = 0.012
    pose = torch.tensor(
        [[0.003, -0.002, 0.045, math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)]],
        dtype=torch.float64,
    )
    solved = solve_leg_transforms(pose, cfg)
    base, platen = joint_points(cfg, dtype=torch.float64)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    platen_world = pose[0, :3] + platen @ rotation.T
    expected = torch.linalg.norm(platen_world - base, dim=-1)
    torch.testing.assert_close(solved.lengths_m[0], expected, atol=1.0e-10, rtol=0.0)
    assert float(solved.lengths_m[0].max() - solved.lengths_m[0].min()) > 1.0e-3


def test_clite_support_config_is_available_but_not_for_panel() -> None:
    import pytest

    clite = BenchmarkConfig(support_config="C2_CLITE")
    assert clite.use_clite_support is True
    with pytest.raises(ValueError):
        BenchmarkConfig(task="panel_operation", support_config="C2_CLITE")


def test_default_spectral_motion_keeps_all_stewart_legs_in_stroke() -> None:
    benchmark = BenchmarkConfig()
    source = SpectralVibration(benchmark.vibration, 1, "cpu")
    for step in range(1000):
        displacement, _, _ = source.sample(step * benchmark.dt)
        rx, ry, rz = displacement[0, 3:]
        cr, sr = torch.cos(rx / 2.0), torch.sin(rx / 2.0)
        cp, sp = torch.cos(ry / 2.0), torch.sin(ry / 2.0)
        cy, sy = torch.cos(rz / 2.0), torch.sin(rz / 2.0)
        quaternion = torch.stack(
            (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
        )
        position = torch.tensor(benchmark.platform_center) + displacement[0, :3]
        lengths = solve_leg_transforms(torch.cat((position, quaternion)).unsqueeze(0), benchmark.shaker).lengths_m
        assert torch.all(lengths >= benchmark.shaker.leg_stroke_min)
        assert torch.all(lengths <= benchmark.shaker.leg_stroke_max)


def test_stewart_segments_are_continuous_and_joint_scale_is_physical() -> None:
    cfg = ShakerGeometryCfg()
    pose = torch.tensor([[(0.0), 0.0, 0.04, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    solved = solve_leg_transforms(pose, cfg)
    overlap = cfg.cylinder_length + cfg.rod_length - solved.lengths_m
    assert torch.all(overlap > 0.05)
    assert torch.all(overlap < min(cfg.cylinder_length, cfg.rod_length))
    assert 2.0 <= (2.0 * cfg.joint_radius) / cfg.rod_radius <= 2.5


def test_room_style_registry_loads_deterministically() -> None:
    cfg = load_room_arena_cfg()
    assert cfg.plank_rows == 15
    assert cfg.plank_columns == 5
    assert cfg.size_m == (6.0, 5.0, 3.0)
    assert cfg.pit_size_m == (2.05, 1.55)
    assert cfg.pit_depth_m == 0.78
    assert len(cfg.wallpaper_rgb) >= 3
    assert len(cfg.wood_rgb) >= 4
    assert Path(cfg.floor_texture_path).is_file()
    assert Path(cfg.wallpaper_texture_path).is_file()


def test_vendored_texture_hashes_are_stable() -> None:
    root = Path(__file__).resolve().parents[1] / "assets" / "textures"
    expected = {
        "dark_wooden_planks_diff_1k.jpg": "e8216baa6b2d701b5523fcb904d45570acf7b96d160ef815e96e2dbfa82bdd9b",
        "floral_jacquard_diff_1k.jpg": "4534bf28534ebd8f24544c389a76f42630bca7621ae55feabbb8da32f98e6bf4",
        "marble_01_diff_1k.jpg": "b10dabea976d68baa976a0d4e8dac58789df8d26a1b50697f320106f1b00a229",
        "natural_wood_planks_1k.jpg": "b89dda0016e92931fadd1516eb2dac1c40481da8f7a0a9683e7b040613de898a",
        "floral_jacquard_beige_1k.jpg": "f4ddcba3093c85528012c86da59e0e16d1c68ac3b81d8349d8e4efa61eb7f0e3",
        "marble_01_light_gray_1k.jpg": "193ef5fb471a82fb196f076e992d497c96e519e8fe637ea79bd25ebe470e39f9",
        "platen_threaded_holes_1k.jpg": "f686cdc8166275e3ea54a763b70a6aa65adee0ea1a22c4caa7a8cbf9a763f3a0",
        "epoxy_floor_cool_gray_1k.jpg": "7da2141fb9b7a38966114d5876d9616899f749222c9299360f341fa4b4c2a474",
        "industrial_wall_light_gray_1k.jpg": "9910939ec5a98cfb3863ba33b752266126756103422b1a54c384a39d68f564f4",
        "phenolic_bench_dark_1k.jpg": "6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest

    manifest = (Path(__file__).resolve().parents[1] / "configs" / "assets.yaml").read_text(encoding="utf-8")
    assert "file: assets/textures/phenolic_bench_dark_1k.jpg" in manifest
    assert expected["phenolic_bench_dark_1k.jpg"] in manifest
    assert "file: assets/textures/platen_threaded_holes_1k.jpg" in manifest
    assert expected["platen_threaded_holes_1k.jpg"] in manifest


def test_wrist_camera_extrinsic_uses_physical_optical_frame() -> None:
    identity_hand = torch.tensor([[0.2, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]])
    eye, target = wrist_camera_pose_from_hand(identity_hand)
    expected_eye = identity_hand[:, :3] + torch.tensor([WRIST_CAMERA_EYE_H])
    torch.testing.assert_close(eye, expected_eye)
    torch.testing.assert_close(
        target - eye,
        0.55 * torch.tensor([WRIST_CAMERA_FORWARD_H]),
    )
    # The camera is outside the wrist-flange silhouette, on the wrist side,
    # and its invariant inward pitch intersects a fixed gripper-frame point.
    assert WRIST_CAMERA_EYE_H[0] > 0.05
    assert WRIST_CAMERA_EYE_H[2] < 0.0
    forward = torch.tensor(WRIST_CAMERA_FORWARD_H)
    aim_ray = torch.tensor(WRIST_CAMERA_AIM_H) - torch.tensor(WRIST_CAMERA_EYE_H)
    torch.testing.assert_close(forward.norm(), torch.tensor(1.0))
    torch.testing.assert_close(forward, aim_ray / aim_ray.norm())
    hand_plane_t = -WRIST_CAMERA_EYE_H[2] / WRIST_CAMERA_FORWARD_H[2]
    hand_plane_x = WRIST_CAMERA_EYE_H[0] + hand_plane_t * WRIST_CAMERA_FORWARD_H[0]
    assert hand_plane_x > 0.07
    frame_eye, frame_target, frame_up = wrist_camera_frame_from_hand(identity_hand)
    torch.testing.assert_close(frame_eye, eye)
    torch.testing.assert_close(frame_target, target)
    torch.testing.assert_close(frame_up, torch.tensor([WRIST_CAMERA_UP_H]))
    torch.testing.assert_close(
        torch.sum((frame_target - frame_eye) * frame_up, dim=1),
        torch.zeros(1),
        atol=1e-6,
        rtol=0.0,
    )
