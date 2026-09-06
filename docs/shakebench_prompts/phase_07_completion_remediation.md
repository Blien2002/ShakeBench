# Phase 07 Completion Remediation：修正 tier 语义、六轴控制与证据链

你在 `/home/miracle04/Desktop/ShakeBench` 中继续完成 Phase 07。当前工作树已经有一版
Phase 07 实现和 `docs/phase_07_report.md`，但审查发现其 PASS 结论不成立。直接修复现有
实现，持续工作到本提示词的完成条件全部满足；不要创建 Phase 07R、Phase 06 新版本或
另一条 handoff/verifier 链。

本轮只修 Phase 07 controller、公共任务状态语义和 Phase 07 evidence。保持
`shakebench.official.physics.v2` 的 timestep、solver、deck、isolator、contact、task
geometry 和 success 数值阈值不变。Phase 06F-R2 handoff 继续作为唯一 physics authority。

## 1. 开始状态与失效声明

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_07_oracle_controller.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_07_report.md
```

检查 `git status`，保留当前全部修改和未跟踪 Phase 07 artifacts。记录起始 commit，运行：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
```

只要该命令通过，就继续使用现有 official physics。先把 `docs/phase_07_report.md` 的当前
PASS 标为 `INVALIDATED_BY_REVIEW`，列出旧 Phase 07 run payload hashes，并记录以下失效
原因：目标四元数转换、V2/V3 重力语义、V3 future program 未参与控制、六轴/recovery
不完整、failure/determinism/actuator evidence 不完整。旧 rollout 可用于诊断，但不能再
用于任何 Phase 07 gate、Phase 08 handoff 或论文结果。

完成标准：报告在代码修复前已明确当前 Phase 07 不具备 PASS authority；Phase 06
profile ID/hash 未变化。

## 2. 先建立四个红灯反馈

修改 production code 前，把以下问题写成独立、快速、确定性的测试，并实际运行确认旧
实现失败。expected value 必须来自数学不变量或独立构造，不能复制 production 中的同一
转换链。

### 2.1 Target quaternion

证明 `T.mat2quat(I)` 已经是 `xyzw=[0,0,0,1]`，当前环境对它再次
`convert_quat(..., to="xyzw")` 会错误得到 `[0,0,1,0]`。测试至少覆盖 identity、绕 x/y/z
的已知 90° 旋转，并通过“公开 quaternion 重建的 rotation matrix 等于
`R_robot_base_world.T @ R_target_world`”判断；允许 `q` 与 `-q` 等价。

### 2.2 Typed acceleration semantics

证明 V2/V3 的 `table_accel_in_deck_frame=[0,0,0,0,0,0]` 经过当前 control law 会产生
`+0.01 m` Z correction。冻结以下 typed contract：

```text
V1 input      = sensor-frame specific force + angular velocity, noisy/delayed
V1 estimate   = gravity-compensated motion acceleration in the declared control frame
V2/V3 input   = gravity-free inertial support acceleration from SupportState
law input     = gravity-free motion acceleration; control law itself不再统一减 9.81
```

测试至少要求：V0 neutral、nominal stationary V1、zero-acceleration V2 和 V3 都产生零
vibration correction。由于 benchmark 含旋转激励，V1 的重力补偿必须使用 frame-aware
attitude/gravity estimate；增加一个 tilted-but-stationary IMU case，不能把倾斜产生的重力
投影当作水平振动。

### 2.3 V3 future causality

在完全相同的 current task/support state 下，只改变 public V3 future program 的 amplitude
或 phase。当前实现的 typed estimate/action 不变，测试必须先失败。另加 neutral program
control：future program 为零时 V3 与相同 current estimate 的 V2 行为一致。

### 2.4 Failure integrity

构造 horizon 用尽和 `TaskPhase.FAILED` 两种 run，证明当前 runner 会产生
`success=false, failure_reason=null` 或继续运行。测试要求每个 unsuccessful episode 都有
非空、枚举化 termination/failure reason，并在 controller FAILED 后立即停止。

四个红灯全部在旧实现上失败后再进入修复；把命令和失败摘要写入 Phase 07 report。

## 3. 修正公共 target frame

在 `VibrationPickPlaceCan` 中直接输出正确的 `xyzw` quaternion。`T.mat2quat()` 的结果不再
做第二次格式转换。保留已经冻结的 scoreable schema：

```text
goal_frame_pos_robot_base
goal_frame_quat_robot_base
goal_inner_half_extents_target
goal_z_bounds_target
goal_orientation_mask
```

更新 environment/provider/privilege/Phase 05 observation artifact 的相关测试。公共 target
frame 与 evaluator frame 使用同一个几何定义，但测试通过独立 rotation-matrix 不变量比较，
避免同源验证。

完成标准：identity observation 为 `[0,0,0,1]`；三组已知旋转均能从公开 quaternion 重建
正确矩阵；使用公开字段的 target-local ↔ robot-base round trip 在容差内闭合。

同时保留逐 physics-step success 语义，但消除 `ShakeBenchMetrics.success_snapshot()` 与
`update()` 的重复几何/contact 提取：抽出一个共享 evaluator primitive，缓存编译后不变的
Can local collision support points，每个 substep 只变换必要点并读取成功子条件，不构造完整
diagnostic report。加入同一 sim state 下 lightweight snapshot 与 full update 的逐字段等价
测试，并记录修复前后的单 episode wall time；性能结果是诊断，不改变 success 语义或阈值。

## 4. 分离 V1 与 V2/V3 的加速度语义

让 `VibrationEstimate` 明确携带 frame、units、timestamp/latency 和 gravity-free motion
acceleration。V1 estimator 消费完整 10×6 IMU window，而不是只取最后一个样本；用公开
gyro、初始 nominal attitude 和明确的滤波/姿态更新把 specific force 转为 control-frame
motion acceleration。不要读取 simulator pose、clean IMU、bias truth 或 privileged state。

V2/V3 直接消费 `SupportState` 已提供的 gravity-free inertial acceleration；不得再次扣除
gravity。所有 frame 变换显式写出并测试。相同 typed neutral estimate 进入 shared law 时，
各 tier 得到相同 correction。

完成标准：第 2.2 节红灯变绿；Gamma=0 的 V2/V3 不再出现固定 `+0.01 m` Z correction；
tilted stationary V1 不产生系统水平补偿；噪声只产生与 canonical IMU profile 一致的有限
残差。

## 5. 让 V3 真正使用固定 horizon 的 future program

在唯一 `OracleControllerProfile` 中冻结并哈希 `future_query_horizon_s`。V3 只从 public
program fields 和 `episode_time_s` 重建：

```text
t_query = episode_time_s + future_query_horizon_s
future authored deck q/qdot/qdd at t_query
```

将重建结果作为 typed `VibrationEstimate` 的 future component。用公开的 isolator/static
context 和一条有单位、frame、符号说明的 feed-forward 关系将它用于同一个
`SharedVibrationControlLaw`。V3 不能查询 future realized table/contact/task state。V0–V3
仍共用一个 control-law 类和一个参数 profile；tier 分支只负责把允许的信息转换成 typed
estimate。

必须证明：

1. 相同 current state、不同 future program 会产生不同 V3 estimate 和 action；
2. neutral future program 不改变相同 current estimate 的 V2/V3 action；
3. V3 query time 始终是记录的固定 horizon，越界/NaN/malformed payload fail closed；
4. positive-Gamma 的 V2/V3 raw action 不再因“future 未使用”而逐元素完全相同。

完成标准：第 2.3 节红灯变绿，并有公式、frame 和数值 positive control 证据；不得用任意
tier-specific bias 人工制造 V2/V3 差异。

## 6. 完成六轴 action 与公开信号 recovery

为 TaskExecutive 定义完整 6D EEF target：position 加 orientation。approach/grasp/lift 可
使用冻结的 tool-to-Can grasp orientation；transport/place 根据当前 public target frame
保持 Can 轴与 target local z 的合理对齐，yaw 在无任务约束时保持连续。用 quaternion
error → rotation vector 生成 OSC_POSE 的后三轴 action，处理 `q/-q`、归一化、最短旋转和
裁剪。

将 angular IMU/support/future estimate 接入同一个 vibration law。六个 OSC channel 和
gripper channel 都要有正负向 unit test，并至少有一个 environment-backed positive-control
test 证明 decoded action 和实际 arm/gripper control 响应方向正确。

recovery 只使用公共字段和 controller 自己的 episode history：

- grasp estimate 同时使用 gripper aperture、Can–EEF/fingertip 相对几何和可用 wrist wrench；
- grasp 时保存 Can-in-EEF reference，用当前公开 Can/EEF pose 检测 translation/rotation slip；
- 用 target-local fingertip/EEF geometry 检测 finger-table/container risk；
- grasp loss、可恢复 slip 或 public placement rebound 转入有上限的 safe-lift/re-approach/
  regrasp/re-place；
- slip/contact loss 保持诊断，不直接改变 environment success；
- policy 不读取 evaluator pass、native contacts 或 penetration 来决定动作。

VERIFY 不得仅凭 Can 中心 XY 进入不可恢复 COMPLETE。利用公开 Can collision envelope、
target frame 和有限差分 motion 做保守 public verification；若公开状态显示 rebound/outside/
unstable 且 recovery budget 尚在，应重新放置。runner 仍以 environment evaluator 作为唯一
success authority。

完成标准：六轴/gripper positive controls、public grasp loss、slip、risk、placement rebound、
recovery budget exhaustion 和 evaluator isolation 均有真实调用 seam 测试；删除未使用的
`_eef_quaternion_xyzw`，或使其成为上述 orientation controller 的明确状态。

## 7. 修正 runner 与可审计 evidence

Phase 07 run schema 升级一次并固定，至少记录：

```text
schema_version / run_id / state_id / tier / Gamma / horizon_steps
complete lossless policy input consumed by controller
provider payload / typed estimate / future query
task phase and recovery transition
normalized / decoded / clipped action
actuator names, ids, ctrl, force, ctrlrange, forcerange and units
termination category / failure reason
trace digest / payload digest / controller+physics+state hashes
```

runner 必须在 evaluator success、controller FAILED、nonfinite/physics violation、明确的
post-verify timeout 或 episode horizon 上终止。所有 `success=false` 都有非空 reason；
`horizon_exhausted` 不得留作 `null`。`completion_evaluator_settle_s` 只有一个权威来源，
纳入 controller/run profile hash；删除 runner 中冲突的独立常量。

实际 actuator validation 不止检查 finite：保存 name/id mapping，重算 action scaling，检查
control/force bounds 和 gripper open/close 符号，并用正向控制证明六轴 action 到 OSC/arm
control 的有效路径。为 run artifact 增加只读 verifier；mutation、缺字段、换序、截断 trace、
错误 digest 和 `failure_reason=null` 都必须被拒绝。

三进程 determinism 必须由同一命令启动三个新进程，保存三份完整原始结果或可重新打开的
lossless trace 文件。determinism manifest 记录每个 process identity、path、file/payload/
trace hash、complete flag 和比较容差；独立 verifier 重新打开三份证据后计算 verdict。
只有 `process_count=3` 加一个摘要 hash 不能通过。

完成标准：任一 raw episode 可从保存的完整 policy input 重放 controller action；所有失败
reason 完整；actuator 和 determinism verdict 均可从 raw evidence 独立重算。

## 8. Dev-state anchor、打包与最终重跑

当前 10 个 dev states 是合理的 Phase 07 依赖修复，但此前没有独立 Git anchor。保持现有
generator、root seed、state IDs 和 state payload 字节不变；先单独提交/锚定 dev-state
generator 与 `shakebench_states_dev.json`，记录 commit 和 payload hash。Phase 08 必须复用
该 asset。不要生成或读取 100 knee / 400 official states。

修正 runtime packaging：clean wheel 和 clean sdist 安装都必须能加载 package-owned
controller profile、dev-state asset 和只读 verifier。记录安装位置、profile hash 和 import/
load 命令。完整 raw traces 与 runtime package 分开；package 只保留运行所需资产和紧凑的
manifest/digests，报告完整 evidence 的路径、总大小和 hashes。

由于 target frame、控制律和 recovery 都会改变 action，旧 Phase 07 rollout evidence 全部
失效。保留其 top-level hashes 作为历史记录，然后用修复后的唯一 controller profile 重新
运行：

```text
Gamma=0:    V0 全部 10 dev states
matched:    V0/V1/V2/V3，同一个固定 dev state
smoke:      V0/V1/V2/V3 × Gamma {0.15,0.30} × 排序前 3 个 dev states
determinism:同一冻结 controller/state 的 3 个独立进程
```

positive-Gamma success 和 tier 排序继续只是诊断。每个失败 episode 必须保留，不能替换
state/seed 或只发布 valid rows。

完成标准：新 artifacts 全部绑定 dev-state anchor、修复后 controller profile 和 unchanged
official physics hash；不存在旧 trace 与新 summary 混用。

## 9. 验证与完成条件

至少运行：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
python -m pytest -q tests/test_shakebench_oracle.py tests/test_shakebench_metrics.py \
  tests/test_shakebench_providers.py tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
python -m black --check -W 1 <本轮修改的 Python 文件>
python -m isort --check-only <本轮修改的 Python 文件>
git diff --check
```

并运行新的 run-artifact、actuator、package-install 和 3-process determinism verifiers。
renderer/EGL 仍按独立 infrastructure limitation 记录，不阻止当前 State lane。

只有以下全部成立才把 `docs/phase_07_report.md` 改回 **PASS**：

1. 四个红灯全部变绿且没有用同源 expected 掩盖错误；
2. target quaternion、V1/V2/V3 acceleration frame/semantics 正确；
3. V3 在固定 horizon 真正使用 authored future program；
4. 6D OSC + gripper positive controls 和公开信号 recovery 完整；
5. Gamma=0 State-V0 仍为 10/10；
6. fixed positive-Gamma smoke 全部运行并保留完整 failure reasons；
7. actuator、三进程 determinism、privilege 和 raw evidence 可独立验证；
8. clean wheel 与 sdist 安装能加载同一 controller profile/hash；
9. Black、isort、focused tests 和相关 non-renderer regression 通过；
10. physics/contact/task geometry/success thresholds 未改变，knee/official states 未访问。

最终报告按 `correctness fixes / hard gates / diagnostics / infrastructure limitations` 分节，
列出复现命令、原始证据路径、旧新 hashes 和仍存在的限制。完成后停止，不进入 Phase 08。
