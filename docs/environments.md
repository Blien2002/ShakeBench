# Environments

`shakebench.make("PickPlace", ...)` and `shakebench.make("PanelOperation", ...)` return Gymnasium environments. `reset()` returns `(observation, info)` and `step()` returns `(observation, reward, terminated, truncated, info)`.

`episode_s` is the physical time budget and `horizon=round(episode_s*control_freq)` is derived from it. `physics_hz` must be an integer multiple of `control_freq`. Each policy action is executed for `physics_hz/control_freq` inner steps using zero-order hold, interpolation, or variable-impedance feedforward.

## Actions

| Controller | Shape | Meaning |
|---|---:|---|
| `JOINT_POSITION` | 8 | seven arm joints and one gripper channel |
| `OSC_POSE` | 7 | normalized Cartesian delta position/rotation and gripper |
| `VARIABLE_IMPEDANCE` | 13 | equilibrium pose, six diagonal stiffnesses, grip force |

Actions are normalized to `[-1,1]`. Values outside that range are clipped and reported as `info["action_clipped"]`. Variable impedance maps translation stiffness to 10–3000 N/m, rotation stiffness to 1–300 Nm/rad, and grip force to 0–70 N.

## Observations

Robot observations are always available as `robot0_joint_pos`, `robot0_joint_vel`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `robot0_wrist_force`, and `robot0_wrist_torque`.

- `use_camera_obs`: `wrist_camera_image`.
- `use_imu_obs`: delayed/noisy/quantized six-channel `deck_imu`.
- `use_object_obs` (privileged): object/target truth, penetration, mounting delta, mass, friction, and COM offset.
- `use_vibration_obs` (privileged): true `q`, `qd`, `qdd`, calibrated level, and `t0`.

No world-frame `_w` key is exposed when privileged groups are disabled. The shaped PickPlace reward is negative object-to-target distance; sparse mode reports task completion. The episode is truncated at the derived horizon.
