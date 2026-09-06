# Phase 05：State V0–V3、Canonical IMU 与权限隔离

状态：**SUPERSEDED by Phase 05R**

本阶段在 Phase 04R 的任务和接触契约上实现 State Oracle-Control Track 的
V0–V3 observation ladder、canonical deck IMU 和独立 privileged recorder。
没有实现 Oracle controller，也没有新建 ShakeBench 子目录。

Phase 05R 发现并修复了本报告版本中的 robot-base mounting、exact returned
key-set、prefill timeline、signed-int16 endpoint 和 defensive-copy 边界问题；
当前 handoff 结论以
`docs/phase_05_observation_imu_remediation_report.md` 为准。

## 1. 实现落点

| 路径 | 内容 |
| --- | --- |
| `robosuite/utils/shakebench_sensors.py` | canonical 200 Hz IMU、specific-force 刚体方程、40 Hz 二阶 Butterworth、seeded residual bias/Brownian diffusion、noise density、clip、16-bit deterministic quantization、delivery queue、timestamp/history trace |
| `robosuite/utils/shakebench_providers.py` | V0–V3 cumulative policy providers、V2 pose/twist/acceleration transforms、V3 authored-program payload |
| `robosuite/utils/shakebench_privilege.py` | explicit recorder/evaluator adapters、`privileged_` namespace validation、policy key audit |
| `robosuite/environments/manipulation/vibration_pick_place_can.py` | tier API、common State observables、provider hooks、static-history reset、privileged snapshot integration |
| `tests/test_shakebench_sensors.py` | IMU physics/filter/noise/delay/quantization/replay tests |
| `tests/test_shakebench_providers.py` | exact tier key additions、V2 moving-frame transform、V3 reconstruction tests |
| `tests/test_shakebench_privilege.py` | recorder namespace, callback, wrapper and fail-closed tests |

## 2. Observation contract

`observation_tier` accepts exactly `V0`, `V1`, `V2` or `V3`. Explicit State tiers
require `use_camera_obs=False`, `control_freq=20`, and a model timestep that
divides the 5 ms acquisition interval. `None` remains a legacy compatibility
mode so existing camera and stock robosuite calls retain their previous API.

Every explicit tier shares robot joint position/velocity, robot-base EEF pose,
gripper state, wrist force/torque, fingertip positions, Can pose in robot-base,
and the complete target frame: `goal_frame_pos_robot_base`,
`goal_frame_quat_robot_base` (xyzw), target-local half-extents/z bounds, and
an orientation mask. This Phase 07 semantic remediation replaces the former
center-only goal fields so policies can transform into a rotated target frame.
Raw deck/table state is absent from V0 and V1.

The dedicated additions are exactly:

```text
V0: none
V1: deck_imu_window[10,6], deck_imu_dt_s
V2: deck_pose/twist/accel_in_nominal_frame,
    table_pose/twist/accel_in_deck_frame
V3: line_accel_amplitude[6,max_lines], line_omega_rad_s[6,max_lines],
    line_phase_at_episode_zero[6,max_lines], line_mask[6,max_lines],
    episode_time_s, ramp_type, ramp_duration_s, program_frame
```

The public V1 window is `float32`, oldest-to-newest, and contains only the
delayed noisy signal. V2 uses one current support snapshot and no support
history, online frequency/program estimation, or future extrapolation. V3's
phase array is the authored phase at episode zero and does not expose `seed` or
`t0`; it can reconstruct authored future command only.

## 3. Canonical IMU

The frozen profile is `canonical_midgrade_v1`:

```text
ODR/window       200 Hz; policy window [10,6]; policy rate 20 Hz
filter           causal order-2 Butterworth, 40 Hz, ENBW 40.7618155662 Hz
delivery         one acquisition-sample delay (5 ms authored delivery delay)
accelerometer    ±16 g, 16 bit, 150 µg/sqrt(Hz)
gyroscope       ±2000 deg/s, 16 bit, 0.005 deg/s/sqrt(Hz)
bias            residual Gaussian reset + independent Brownian diffusion
```

The physical sensor equation computes point acceleration with both
`alpha × r` and `omega × (omega × r)`, subtracts world gravity, rotates into
the sensor frame, and uses realized spatial angular velocity for the gyro.
MuJoCo's gravity-inclusive object-acceleration convention is normalized at the
adapter boundary before this equation. Reset prefill uses a physical static
reading rather than synthetic zero frames. Clean signal, bias, white noise,
filter state, clipping flags, quantization codes and acquisition/delivery
timestamps are recorder-only.

`ideal_smoke` uses the same topology, equations, filter, delay and window while
zeroing bias/noise and bypassing quantization. It is explicitly non-scoreable.

## 4. Privilege boundary

`PrivilegedRecorder` is an independent interface, not an `Observable`. It
accepts only namespace-qualified fields and stores defensive copies. The
environment records full Can/goal truth, realized and commanded support state,
IMU decomposition, contacts, static parameters, actions and success
subconditions under direct `privileged_` keys (with a nested provider snapshot
for convenience). The regular observation path audits and rejects any
`privileged_` key; renderer/debug flags and the GymWrapper do not add recorder
truth to policy observations.

## 5. Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_shakebench_sensors.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py
```

Result: **16 passed, 1 skipped**. The skipped test requires the optional
`gymnasium`/`gym` dependency, which is not installed in this workspace.

The combined Phase 01–04 plus Phase 05 targeted suite passed **86 tests** with
one optional GymWrapper skip. Deck-driver, calibration, action-playback and
Phase 04 environment regressions passed **79 tests**. Formatting and static
checks (`black`, `isort`, `flake8`, `pyflakes`, `py_compile`, and
`git diff --check`) passed.

The existing renderer-dependent all-environment/camera smoke remains subject
to the workspace's previously documented unavailable EGL/swrast runtime; the
legacy no-renderer environment path is covered by the Phase 04 regression
suite.

## 6. Historical scope decision

The original Phase 05 implementation stopped before the remediation gates.
Oracle controller work, committed states, Gamma knee selection, official
scorecard, and physics/controller freeze remain deferred to their later phases.
