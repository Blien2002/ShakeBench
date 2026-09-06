# Phase 05R：Observation / IMU Remediation

状态：**PASS；Phase 06 handoff = PASS**

本 remediation 只修复 Phase 05 的 policy/privileged、mounting、时间线、
编码和 copy 边界。没有重置工作树，没有修改 Phase 02 driver time contract、
Phase 03 isolator physics 或 Phase 04 success semantics，没有实现 Oracle
controller，也没有新建目录。

## 1. 修复内容

### 1.1 Robot-base IMU

V1/V2/V3 的 IMU provider 现在直接读取 compiled `robot0_base` body 的
world pose、spatial twist 和 spatial acceleration。canonical sensor metadata
保持：

```text
sensor frame parent       robot_base
sensor position in base   [0, 0, 0] m
sensor quaternion         [1, 0, 0, 0] wxyz
compiled body parent      deck
compiled robot-base pose  [-0.485, 0, 0.912, 1, 0, 0, 0] in deck
```

`V1Provider.audit_compiled_mount()` 会在编译和 reset 时拒绝错误 body、
错误 parent 或错误 canonical extrinsics。真实旋转 deck 回归在非零
`alpha/omega` 下用独立刚体公式重建 robot-base specific force，并证明它与
deck-origin zero-lever 结果不同；gyro 使用 realized spatial angular
velocity。

### 1.2 Exact policy boundary

显式 State tier 的最终 `reset()` / `step()` 返回值由 allowlist 重新构造，
不是对 stock observations 做 subset 判断。实际 key-set 严格为：

```text
V0 = COMMON_STATE_KEYS
V1 = COMMON_STATE_KEYS + (deck_imu_window, deck_imu_dt_s)
V2 = V1 + (deck_pose_in_nominal_frame, deck_twist_in_nominal_frame,
           deck_accel_in_nominal_frame, table_pose_in_deck_frame,
           table_twist_in_deck_frame, table_accel_in_deck_frame)
V3 = V2 + (line_accel_amplitude, line_omega_rad_s,
           line_phase_at_episode_zero, line_mask, episode_time_s,
           ramp_type, ramp_duration_s, program_frame)
```

Phase 07 replaced the ambiguous public goal center/extents/z fields with the
single scoreable target-frame schema shared verbatim by V0--V3:
`goal_frame_pos_robot_base`, `goal_frame_quat_robot_base` (xyzw),
`goal_inner_half_extents_target`, `goal_z_bounds_target`, and
`goal_orientation_mask`. The current target frame is task geometry, not a V2
support-state channel; V0 still has no dedicated vibration signal.

Stock joint acceleration, joint sin/cos, gripper qpos/qvel aliases, modality
aggregates、world-frame EEF/object fields 均不会越过 policy boundary。legacy
`observation_tier=None` 仍保留上游行为。每个最终 policy value 都是
defensive copy。

### 1.3 Continuous acquisition / delivery timeline

reset timeline 已冻结为：

```text
window acquisitions: [-0.050, -0.045, ..., -0.005] s
pending queue:       [0.000] s
first policy step acquisitions: [0.005, 0.010, ..., 0.050] s
first delivered window:         [0.000, 0.005, ..., 0.045] s
pending after first step:       [0.050] s
```

每个 live 20 Hz step 正好产生 10 个 200 Hz acquisitions，相邻 timestamp
严格为 0.005 s。hard reset、non-hard reset 和 timestamp rollback 都会重新
建立同一连续时间链；policy 只收到 delayed window 和 dt，recorder 保存
acquisition/delivery/delivered-acquisition timestamps。

### 1.4 Signed 16-bit encoding

编码现在严格使用 two's-complement range：

```text
code_min = -32768
code_max = +32767
LSB = 2 * range / 65536
```

`+range` 饱和到 `+32767`，`-range` 为 `-32768`。physical clipping flag
与 quantizer endpoint saturation 是两个独立字段；codes 输出为 `np.int16`
并通过 int16 round-trip 验证。

### 1.5 Immutability and privilege isolation

gravity/filter process-global constants 已改成 immutable tuples；profile、
provider config、`IMUSample`、delivery items、filter trace、`RigidBodyState`
和 `SupportState` 均在最终 copy 后设为 read-only 或返回独立 copy。

`PrivilegedRecorder` 的 `deepcopy` 失败会抛 `ShakeBenchPrivilegeError`，不会
退回原对象。records、latest、return value 和 callback payload 互相独立；
所有 truth 字段使用 direct `privileged_` keys。普通 observation、debug/
renderer 选项以及 GymWrapper 均没有 privileged truth。

### 1.6 Independent V3 reconstruction and wrapper

`reconstruct_authored_motion()` 只读取公开的 line arrays、mask、ramp metadata
和 query time，独立计算 authored `q/qdot/qdd`；测试会临时禁用原始
`ExcitationProgram.evaluate()` 以证明没有借 provider 内部 program 自证。

GymWrapper 现在对显式 State tier 默认使用 exact policy allowlist，拒绝
`privileged_*`、不可用 aggregate、debug 和 renderer keys。由于当前运行环境
没有安装 gymnasium/gym，production wrapper 不提供 fake-Gym fallback；测试在
`tmp_path` 中创建仅覆盖实际调用 API 的最小 Gymnasium package，并通过
`PYTHONPATH` 优先级在隔离子进程执行仓库真实 `GymWrapper` 代码路径。另有无
shim 与 `gym==0.25.2` 子进程验证缺失/过旧依赖都会明确失败，没有 skip。

## 2. Controlled artifact

机器可读 evidence 位于
[`tests/shakebench_phase_05_observation.json`](../tests/shakebench_phase_05_observation.json)。
`verify_phase05_observation_artifact()` 只读验证 schema、exact key-set、
robot-base mount、profile hash、连续 timeline、int16 endpoint、noise/filter
统计、V2/V3 reconstruction 和 privilege evidence。artifact lock 使用
`update_reason`、previous payload SHA-256 和 current payload SHA-256；默认
不会写回 artifact。

## 3. Verification

Phase 05R 定向测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_shakebench_sensors.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py
```

结果：**28 passed**，无 skip。

Phase 05R 加 Phase 04 environment/metrics/deck-driver 回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_shakebench_sensors.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py \
  tests/test_shakebench_metrics.py \
  tests/test_shakebench_deck_driver.py
```

结果：**111 passed**。

`black --check`、`isort --check-only`、`flake8`、`pyflakes`、`py_compile`、
`git diff --check` 和 controlled-artifact verifier 均通过。renderer-dependent
的既有全环境/camera smoke 仍受工作区此前记录的 EGL/swrast 主机条件限制；
Phase 05R 的真实 GymWrapper、renderer-flag 和 adversarial isolation tests
使用 test-only isolated dependency harness，不依赖该图形路径并已通过。

## 4. Handoff decision

以下 Phase 06 handoff gates 全部通过：robot-base IMU physical mounting、
V0–V3 exact returned key-set、连续 200 Hz timeline、合法 signed int16、
sensor/recorder truth immutability、noise/filter RMS、independent V3
reconstruction 和真实 wrapper leakage tests。

后续仍按阶段处理 physics freeze、Oracle controller、committed states、
Gamma knee 与 official scorecard。
