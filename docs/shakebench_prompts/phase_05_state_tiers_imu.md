# Phase 05 Prompt：State V0–V3、Canonical IMU 与权限隔离（直接扩展 Observables）

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接扩展 `VibrationPickPlaceCan._setup_observables()`、环境参数和 `robosuite/utils/` 模块，实现 State task observation、V0–V3、200 Hz IMU 及 privileged 隔离；不实现 Oracle controller，不新建子文件夹。

## 前置与必读

- Phase 04 环境/metrics/evaluator 测试通过。
- 读取 `docs/robosuite_benchmark_design_tree.md` Observation 部分和 `docs/canonical_imu_profile_validation.md`。
- 审计 robosuite `Observable` update cadence、cache、sampling rate、delay/corrupter API 和 wrappers。

## 修改落点（不新建目录）

```text
robosuite/utils/shakebench_sensors.py    # IMU 传感器模型
robosuite/utils/shakebench_providers.py  # V0–V3 vibration providers
robosuite/utils/shakebench_privilege.py  # recorder/evaluator 隔离与审计
robosuite/environments/manipulation/vibration_pick_place_can.py  # observables 接入
tests/test_shakebench_sensors.py
tests/test_shakebench_providers.py
tests/test_shakebench_privilege.py
```

## Environment API

为新环境增加明确参数，例如：

```text
observation_tier = V0|V1|V2|V3
use_camera_obs = False for current benchmark
privileged_recorder = explicit wrapper/callback, not observation flag
```

保持现有 robosuite 环境 API 不变。unknown tier 或组合 fail closed。

## Common State observables

所有 tier 完全一致：robot joint pos/vel、EEF pose in robot-base、gripper state、公开 wrist/fingertip channels、Can pose in robot-base、goal region in robot-base。goal 是 region，不伪造唯一 target pose。

raw table/deck state 不进入 common fields。

## V1 IMU

从 Phase 02 realized deck motion 生成 sensor-frame specific force 和 gyro：包含 gravity、`alpha×r`、`omega×(omega×r)`，gyro 使用 spatial angular velocity。

实现并冻结：200 Hz acquisition、二阶 40 Hz Butterworth、seeded residual bias/Brownian diffusion、noise-density 驱动的 pre-filter noise、clip、16-bit deterministic quantization、1 sample delivery delay、20 Hz 时 oldest-to-newest `[10,6]` window、acquisition timestamps 和 static-history prefill。

Policy 只见 noisy delayed window；clean/bias/noise/filter/clipping/timestamps 仅 privileged recorder。

## V2/V3

V2：

```text
deck pose/twist/accel in nominal frame
table pose/twist/accel in deck frame
```

后端无关，current-only，不保存 support history、不估计 frequency/program、不预测未来。

V3：self-describing line amplitude/omega/phase-at-episode-zero/mask、episode time、ramp type/duration、program frame。可重建 future authored command，但不含 future realized state/contact/outcome。

## Privileged isolation

使用独立 recorder 接口和 `privileged_` namespace 记录 full object/goal、command/actual supports、IMU decomposition、contacts、parameters、actions 和 success subconditions。普通 obs dict、GymWrapper、debug/renderer flags 均不可访问。

## 必须测试

- `V0 ⊂ V1 ⊂ V2 ⊂ V3` exact key sets；
- shapes/dtypes/units/frames/timestamps；
- stationary +g/free-fall/lever-arm；
- filter −3 dB/ENBW/noise RMS/bias diffusion；
- quantization/delay/10 contiguous samples；
- V2 transforms/current-only；
- V3 analytic reconstruction；
- V0–V2 absence of program fields；
- wrapper/debug privilege leakage；
- deterministic sensor replay；
- existing observable/camera/wrapper tests 无回归。

## 完成条件

1. 新环境每个 tier 可 reset/step 并通过 space/value validation；
2. IMU 所有定量测试通过；
3. V2 current-only、V3 authored-future-only；
4. privileged truth 在 policy API 不可达；
5. 输出 `docs/phase_05_report.md`；
6. 未实现或调优 Oracle controller；未新建目录。

完成后停止。
