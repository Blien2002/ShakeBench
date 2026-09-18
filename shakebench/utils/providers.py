"""Current ShakeBench observation contract for the single worktable IMU.

The provider exposes only the fields the current policy contract allows.  It
never returns MuJoCo qpos/qvel arrays and never retains support-state history;
evaluation truth is collected separately by shakebench_privilege.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Optional

import mujoco
import numpy as np

from shakebench.utils.sensors import (
    CANONICAL_IMU_PROFILE,
    GRAVITY_WORLD_M_S2,
    CanonicalIMU,
    CanonicalIMUProfile,
    ShakeBenchSensorError,
    _normalise_quaternion_wxyz,
)

COMMON_STATE_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos_robot_base",
    "robot0_eef_quat_robot_base",
    "robot0_gripper_state",
    "robot0_wrist_force",
    "robot0_wrist_torque",
    "robot0_fingertip_pos_robot_base",
    "object_pos_robot_base",
    "object_quat_robot_base",
    "goal_frame_pos_robot_base",
    "goal_frame_quat_robot_base",
    "goal_inner_half_extents_target",
    "goal_z_bounds_target",
    "goal_orientation_mask",
)
TABLE_IMU_POLICY_KEYS = ("table_imu_window", "table_imu_dt_s", "table_imu_timestamps_s")
POLICY_FIELD_CONTRACT = MappingProxyType(
    {
        "robot0_joint_pos": {"shape": (7,), "dtype": "float64", "units": "rad", "frame": "robot_base"},
        "robot0_joint_vel": {"shape": (7,), "dtype": "float64", "units": "rad/s", "frame": "robot_base"},
        "robot0_eef_pos_robot_base": {"shape": (3,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "robot0_eef_quat_robot_base": {"shape": (4,), "dtype": "float32", "units": "unitless", "frame": "robot_base"},
        "robot0_gripper_state": {"shape": (4,), "dtype": "float32", "units": "rad,rad/s", "frame": "robot_base"},
        "robot0_wrist_force": {"shape": (3,), "dtype": "float32", "units": "N", "frame": "wrist_sensor"},
        "robot0_wrist_torque": {"shape": (3,), "dtype": "float32", "units": "N*m", "frame": "wrist_sensor"},
        "robot0_fingertip_pos_robot_base": {"shape": (6,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "object_pos_robot_base": {"shape": (3,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "object_quat_robot_base": {"shape": (4,), "dtype": "float32", "units": "unitless", "frame": "robot_base"},
        "goal_frame_pos_robot_base": {"shape": (3,), "dtype": "float32", "units": "m", "frame": "robot_base"},
        "goal_frame_quat_robot_base": {
            "shape": (4,),
            "dtype": "float32",
            "units": "unitless",
            "frame": "robot_base_xyzw",
        },
        "goal_inner_half_extents_target": {"shape": (2,), "dtype": "float32", "units": "m", "frame": "target"},
        "goal_z_bounds_target": {"shape": (2,), "dtype": "float32", "units": "m", "frame": "target"},
        "goal_orientation_mask": {"shape": (3,), "dtype": "bool", "units": "unitless", "frame": "target"},
        "table_imu_window": {"shape": (10, 6), "dtype": "float32", "units": "m/s2,rad/s", "frame": "table_imu"},
        "table_imu_dt_s": {"shape": (), "dtype": "float32", "units": "s", "frame": "acquisition_time"},
        "table_imu_timestamps_s": {"shape": (10,), "dtype": "float64", "units": "s", "frame": "acquisition_time"},
    }
)


class ShakeBenchProviderError(ValueError):
    """Raised when the worktable IMU provider cannot satisfy its contract."""


class TableIMUProvider:
    """Delayed/noisy IMU provider rigidly mounted below the worktable."""

    def __init__(
        self,
        *,
        seed: int = 0,
        imu_mode: str = "canonical_noisy_v1",
        imu_profile: CanonicalIMUProfile = CANONICAL_IMU_PROFILE,
        sensor_site_name: str = "table_imu_site",
        sensor_position_body_m: Iterable[float] = (0.0, 0.0, -0.03),
        sensor_quat_body_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        gravity_world_m_s2: Iterable[float] = GRAVITY_WORLD_M_S2,
    ) -> None:
        try:
            self.imu = CanonicalIMU(seed=seed, mode=imu_mode, profile=imu_profile)
        except (ShakeBenchSensorError, TypeError, ValueError) as exc:
            raise ShakeBenchProviderError(str(exc)) from exc
        self.imu_body_name = "worktable"
        self.sensor_body_name = self.imu_body_name
        self.sensor_site_name = str(sensor_site_name)
        self.sensor_position_body_m = np.array(sensor_position_body_m, dtype=float, copy=True)
        self.sensor_quat_body_wxyz = np.array(sensor_quat_body_wxyz, dtype=float, copy=True)
        self.gravity_world_m_s2 = np.array(gravity_world_m_s2, dtype=float, copy=True)
        if self.sensor_position_body_m.shape != (3,) or not np.all(np.isfinite(self.sensor_position_body_m)):
            raise ShakeBenchProviderError("sensor_position_body_m must have three finite values")
        if self.sensor_quat_body_wxyz.shape != (4,) or not np.all(np.isfinite(self.sensor_quat_body_wxyz)):
            raise ShakeBenchProviderError("sensor_quat_body_wxyz must have four finite values")
        if self.gravity_world_m_s2.shape != (3,) or not np.all(np.isfinite(self.gravity_world_m_s2)):
            raise ShakeBenchProviderError("gravity_world_m_s2 must have three finite values")
        self.sensor_position_body_m.setflags(write=False)
        self.sensor_quat_body_wxyz.setflags(write=False)
        self.gravity_world_m_s2.setflags(write=False)
        self._next_acquisition_time_s = self.imu.profile.dt_s
        self._last_physics_time_s = 0.0
        self._compiled_imu_mount = None
        self.reset()

    @property
    def policy_keys(self) -> tuple[str, ...]:
        return TABLE_IMU_POLICY_KEYS

    def audit_compiled_mount(self, sim: Any) -> dict[str, Any]:
        """Audit the massless IMU site and its calibrated worktable extrinsics."""

        model = getattr(sim, "model", sim)
        model = getattr(model, "_model", model)
        sensor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.sensor_site_name))
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.imu_body_name))
        if sensor_id < 0:
            raise ShakeBenchProviderError(f"compiled model is missing IMU site {self.sensor_site_name!r}")
        if body_id < 0:
            raise ShakeBenchProviderError(f"compiled model is missing IMU body {self.imu_body_name!r}")
        parent_id = int(model.site_bodyid[sensor_id])
        if parent_id != body_id:
            raise ShakeBenchProviderError(f"IMU site {self.sensor_site_name!r} must belong to {self.imu_body_name!r}")
        local_position = np.asarray(model.site_pos[sensor_id], dtype=float).copy()
        local_quaternion = _normalise_quaternion_wxyz(model.site_quat[sensor_id], "compiled IMU site quaternion")
        metadata_position = np.asarray(self.sensor_position_body_m, dtype=float)
        metadata_quaternion = _normalise_quaternion_wxyz(self.sensor_quat_body_wxyz, "sensor_quat_body_wxyz")
        if not np.allclose(metadata_position, local_position, rtol=0.0, atol=1e-12) or not np.allclose(
            metadata_quaternion, local_quaternion, rtol=0.0, atol=1e-12
        ):
            raise ShakeBenchProviderError("configured IMU extrinsics do not match table_imu_site")
        mount = {
            "sensor_site_name": self.sensor_site_name,
            "sensor_parent": "worktable",
            "parent_body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
            "sensor_position_m_in_worktable": metadata_position.copy(),
            "sensor_quaternion_wxyz_in_worktable": metadata_quaternion.copy(),
            "sensor_profile_id": self.imu.profile.profile_id,
        }
        self._compiled_imu_mount = mount
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in mount.items()}

    def reset(self, sim: Any = None, *, timestamp_s: float = 0.0) -> None:
        if not np.isfinite(float(timestamp_s)) or float(timestamp_s) < 0.0:
            raise ShakeBenchProviderError("timestamp_s must be finite and non-negative")
        if sim is None:
            self.imu.reset(timestamp_s=timestamp_s)
        else:
            try:
                self.audit_compiled_mount(sim)
                from shakebench.utils.sensors import clean_imu_measurement_from_sim

                clean, kinematics = clean_imu_measurement_from_sim(
                    sim,
                    self.imu_body_name,
                    sensor_position_body_m=self.sensor_position_body_m,
                    sensor_quat_body_wxyz=self.sensor_quat_body_wxyz,
                    gravity_world_m_s2=self.gravity_world_m_s2,
                )
                # Initialize the filter from the actual reset state; the
                # environment settles the loaded support before this call.
                self.imu.reset(initial_clean_measurement=clean, timestamp_s=timestamp_s)
                self.imu._last_kinematics = kinematics
            except (ShakeBenchSensorError, TypeError, ValueError) as exc:
                raise ShakeBenchProviderError(str(exc)) from exc
        self._next_acquisition_time_s = float(timestamp_s) + self.imu.profile.dt_s
        self._last_physics_time_s = float(timestamp_s)

    def on_physics_sample(self, sim: Any, sample_time_s: float, policy_step: bool = False) -> None:
        del policy_step
        if not np.isfinite(float(sample_time_s)) or float(sample_time_s) < 0.0:
            raise ShakeBenchProviderError("sample_time_s must be finite and non-negative")
        sample_time = float(sample_time_s)
        if sample_time < self._last_physics_time_s - 1e-12:
            self.reset(sim, timestamp_s=sample_time)
        while sample_time + 1e-12 >= self._next_acquisition_time_s:
            self.imu.acquire_from_sim(
                sim,
                timestamp_s=self._next_acquisition_time_s,
                body_name=self.imu_body_name,
                sensor_position_body_m=self.sensor_position_body_m,
                sensor_quat_body_wxyz=self.sensor_quat_body_wxyz,
                gravity_world_m_s2=self.gravity_world_m_s2,
            )
            self._next_acquisition_time_s += self.imu.profile.dt_s
        self._last_physics_time_s = sample_time

    def observation(self, sim: Any = None) -> dict[str, np.ndarray]:
        del sim
        return self.imu.to_policy_observation()

    def privileged_snapshot(self, sim: Any = None, *, time_s: Optional[float] = None) -> dict[str, Any]:
        del sim, time_s
        trace = self.imu.trace
        result = {f"imu_{key}": value for key, value in trace.items()}
        if trace["clean_measurement"].ndim == 2:
            result["imu_clean_specific_force_m_s2"] = trace["clean_measurement"][:, :3]
            result["imu_clean_angular_velocity_rad_s"] = trace["clean_measurement"][:, 3:]
            result["imu_clean_measurement_mixed_channels"] = trace["clean_measurement"]
            result["imu_filtered_measurement_mixed_channels"] = trace["filtered_measurement"]
        result["imu_window_acquisition_time_s"] = self.imu.window_acquisition_timestamps_s
        result["imu_initial_bias"] = self.imu.initial_bias
        result["imu_current_bias"] = self.imu.bias
        result["imu_last_kinematics"] = self.imu.last_kinematics
        result["imu_body_name"] = self.imu_body_name
        result["imu_compiled_mount"] = (
            None
            if self._compiled_imu_mount is None
            else {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._compiled_imu_mount.items()
            }
        )
        return result


__all__ = [
    "COMMON_STATE_KEYS",
    "POLICY_FIELD_CONTRACT",
    "TABLE_IMU_POLICY_KEYS",
    "ShakeBenchProviderError",
    "TableIMUProvider",
]
