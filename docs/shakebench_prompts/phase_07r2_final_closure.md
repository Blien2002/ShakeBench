# Phase 07R2 Prompt：State Oracle 最终语义与证据闭环

你在 `/home/miracle04/Desktop/ShakeBench` 中继续完成 Phase 07R2。当前
`docs/phase_07_report.md` 声称 completion remediation PASS，但独立审查确认仍存在
V1 姿态/时间语义、Can-in-EEF recovery、V3 物理映射、run verifier、三进程证据、
actuator positive-control、package evidence 和 Git freeze 缺口。

直接修复当前工作树，持续执行到本提示词的 Phase 8 handoff 条件全部满足，或得到一个
有最小复现的真实 scientific blocker。不要停在计划、helper 单元测试或“文件哈希一致”。

## 0. 范围与权威边界

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_07_oracle_controller.md
docs/shakebench_prompts/phase_07_completion_remediation.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_07_report.md
```

当前 dev-state anchor 是：

```text
commit = dd6fe2edb6384ccdb5116be44f07592b4864e377
asset  = robosuite/models/assets/shakebench_states_dev.json
```

保持该 commit 中 generator、root seed、10 个 state IDs 和 state payload 字节不变。
Phase 07R2 不生成、不读取 100 knee states 或 400 official states。

唯一 physics authority 仍是：

```text
profile = shakebench.official.physics.v2
hash    = c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c
```

不修改 timestep、solver、deck、isolator、contact tuple、task geometry、success evaluator
语义或成功阈值。开始时运行并记录：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
```

先将当前 Phase 07 report 状态改为 `INVALIDATED_BY_PHASE_07R2_REVIEW`。现有 remediation
raw traces 仅作诊断，不能混入 R2 hard gates。记录其 payload hashes 后再开始改代码。

完成标准：Phase 6 verifier 通过；当前 Phase 07 PASS authority 已撤销；dev-state asset
与 `dd6fe2ed` 完全一致。

## 1. 建立五个生产路径红灯

先在 `tests/test_shakebench_phase07_completion_red.py` 或现有 Phase 07 tests 中加入以下
回归，并实际运行确认当前 production path 失败。expected value 使用独立数学不变量，
mutation test 在篡改后重新计算所有普通 hashes，避免只测试 digest。

### 1.1 V1 跨 policy-step 姿态

构造连续两个或更多 20 Hz policy windows：第一组 gyro samples 将 sensor 从 nominal
姿态转到已知倾角；后续窗口保持该倾角、gyro=0、specific force 只包含该姿态下的重力。
通过真实 `ShakeBenchOracleController("V1").action(...)` 顺序调用，而不是直接给 helper
传入正确 attitude。

当前实现每次在
`vibration_estimate_from_public_observation()` 中使用 identity attitude，第二个窗口会把
重力投影误认为 motion acceleration。红灯必须同时断言：

```text
gravity-free motion acceleration ≈ 0
controller attitude persists across policy calls
estimate.timestamp_s increases with policy time
latency_s equals public canonical end-to-end measurement latency, not 0.005 s sample interval
```

### 1.2 完整 Can-in-EEF SE(3) slip

建立 grasp reference 后测试两个对照：

1. EEF 与 Can 做相同刚体旋转/平移，Can-in-EEF transform 不变，slip 必须为零；
2. EEF 固定、Can 相对 EEF 旋转或平移，必须报告对应 rotation/translation slip。

当前实现忽略 `can_quat_robot_base`，第二种情况错误返回 rotation slip=0。

### 1.3 Public VERIFY 与 rebound

构造 Can 中心仍在目标 XY 内、但其 collision envelope 越界或 public finite-difference
速度超限的序列。当前 VERIFY 只检查中心 XY 并进入不可恢复 COMPLETE；红灯要求保持
VERIFY 或进入 bounded re-place recovery。再覆盖连续稳定窗口后保持 open/verify 的情况。

policy 不得读取 evaluator pass、native contact、support force 或 penetration 来决定动作。

### 1.4 Resealed semantic mutation

复制一个合法 Phase 07 run artifact，分别执行以下 mutation；每次同时重算 trace digest
和 payload digest：

- 删除任一 trace row 的 `policy_input` 或 `applied_actuator_force`；
- 修改 actuator name/id/ctrlrange/forcerange；
- 修改 normalized/decoded/clipped action 中一个值；
- 交换或删除中间 trace row；
- 修改 controller profile、state hash、physics hash、termination 或 failure reason；
- 插入 NaN/Inf、错误 shape/dtype 或 tier 不允许的 policy key。

当前 `verify_run_artifact()` 会接受其中多种 resealed mutation。每一种都必须成为红灯。

### 1.5 真三进程证据

为 determinism manifest 加入 verifier red test。以下情况必须失败：

```text
三个相同 process_identity
缺失任一 raw process file
同一个 path 被引用三次
file/payload/trace hash 任一不匹配
controller/profile/state/physics binding 不一致
trace 不完整或 sample sequence 不一致
manifest 自身 hash 不匹配
```

完成标准：五组红灯在当前实现上产生预期失败，测试记录明确指向用户可观察的错误，
然后才修改 production code。

## 2. 实现有状态 V1 IMU estimator

把 IMU 估计从纯函数中的“每窗口 identity 初始化”移到 controller-owned episode state。
实现一个明确的 V1 estimator，至少维护：

```text
attitude_control_from_sensor
last_policy_time_s
last_window_end_time_s
gravity estimate / filter state
reset generation
```

每次 policy call 消费该次公开的 10×6 oldest-to-newest window，按 200 Hz 顺序积分 gyro，
并在保持跨窗口 attitude 的前提下把 sensor-frame specific force 转为声明的 control frame
gravity-free motion acceleration。处理 quaternion/rotation normalization 和时间倒退 reset。

必须区分：

```text
deck_imu_dt_s              = 0.005 s sample interval
canonical filter delay     ≈ 4.87 ms
delivery delay             = 5 ms
nominal end-to-end latency ≈ 9.87 ms
policy estimate timestamp  = 当前 policy time 对应的 measurement time
```

end-to-end latency 从公开 canonical IMU static context/profile 获取或在唯一 controller
profile 中显式冻结并哈希。不要读取 clean IMU、true attitude、bias/noise decomposition、
sim pose 或其他 privileged truth。

V0、V2、V3 的 estimator 保持无历史或 current-only 合同；V2 不借用 V1 estimator history
辨识 excitation。`VibrationEstimate` 明确记录 frame、units、measurement timestamp、policy
timestamp 和 latency。

完成标准：1.1 红灯变绿；持续倾斜静止 case 不产生系统运动补偿；真实 V1 raw trace 的
timestamp 单调递增；reset 后 estimator 回到 nominal state；同一 state/process 重放一致。

## 3. 修正 Can-in-EEF recovery 与 public VERIFY

用完整刚体变换记录 grasp reference：

```text
T_eef_can = inverse(T_base_eef) @ T_base_can
```

其中 Can 和 EEF 的 position/quaternion 都来自公共 observation。translation slip 比较
`T_eef_can.translation`，rotation slip 使用最短旋转向量比较 `T_eef_can.rotation`，正确处理
`q/-q`。不要在 robot-base frame 直接比较 `Can−EEF`，也不要把 EEF 自身转动当成 Can
相对滑移。

grasp confidence 至少组合：

```text
gripper aperture
Can-to-fingertip / Can-to-EEF geometry
wrist wrench consistency or explicitly recorded confidence contribution
```

若 wrench 只作诊断，不能在 docstring/report 中声称它参与 hold verdict；若参与动作，阈值、
单位和选择依据必须进入唯一 controller profile/hash。默认
`grasp_geometry_tolerance_m=0.150`、`grasp_slip_tolerance_m=0.250` 等明显宽阈值需要用 Can
尺寸、gripper geometry 和 10 个 dev states 给出依据；不可把接近 transport distance 的阈值
称为有效 slip detector。

Public VERIFY 使用：

- 完整 Can collision envelope 在 target-local XY 的保守 containment；
- Can/target public pose history 的 finite-difference linear/angular speed；
- 连续若干 policy samples 的 public stability；
- release/open-gripper public state。

它不复刻 privileged success evaluator，也不自报 task success。若 public state显示
outside、rebound、unstable 或 recoverable grasp/placement loss，进入有上限的 safe-lift →
re-approach → regrasp/re-place。若 public state稳定，保持 VERIFY/open action，runner 只以
environment evaluator 判定 success。删除仅凭中心 XY 进入不可逆 COMPLETE 的路径，或让
COMPLETE 只表示“public verification candidate”，且仍可因公开 rebound 恢复。

完成标准：1.2/1.3 红灯变绿；grasp loss、rigid carry、relative slip、container risk、
placement rebound、recovery exhaustion 均有真实 TaskExecutive seam 测试；所有动作决策
不依赖 `env.sim` 或 privileged keys。

## 4. 使 V3 future feed-forward 具有明确物理语义

保留已经实现的固定：

```text
t_query = episode_time_s + future_query_horizon_s
```

但不得直接把 authored deck `qdd` 当作 robot-base relative-support acceleration。使用所有
tier 都可读取的 public static context：isolator `fn/zeta/k/c`、support topology、deck↔robot
base fixed transform 和 frame conventions。对于 canonical linear base-excited isolator，
逐轴用公开、测试覆盖的频域关系预测 table/deck 相对响应；推荐使用：

```text
H_table/deck(jω)
  = (ω_n² + j 2ζω_nω) / (ω_n² - ω² + j 2ζω_nω)

H_relative(jω) = H_table/deck(jω) - 1
```

将 public line amplitude/frequency/phase 在 `t_query` 处组合成 predicted future relative
linear/angular acceleration，再经显式 deck-frame → robot-base-frame transform 进入 typed
estimate。若采用等价时域方法，必须证明与上述解析 transfer 在冻结容差内一致。

把不同量纲的增益分开并在名称中标明单位：

```text
current_linear_accel_gain_s2
current_angular_velocity_gain_s
current_angular_accel_gain_s2（若使用）
future_linear_accel_gain_s2
future_angular_accel_gain_s2
```

禁止用 `support_accel_compensation` 同时乘 `m/s²` 和 `rad/s`。公式、符号、frame、单位和
clipping 顺序进入 profile documentation、profile hash 和 tests。

测试必须证明：

1. 低频、共振附近、高频单轴 line 的 predicted transfer 与解析式一致；
2. 六轴 program 组合不串轴，frame transform 正确；
3. neutral future program 使相同 current estimate 的 V2/V3 action 相同；
4. 只改变 future amplitude/phase 会按公式改变 V3 action；
5. V3 只使用 authored future command，不读取 future realized state/contact/task outcome。

完成标准：V3 future 差异来自公开 isolator model 与固定 horizon，而不是 arbitrary tier bias；
positive-Gamma V3 结果无 NaN/越界，成功率和 tier 排序仍仅作诊断。

## 5. 完成 7-channel environment-backed positive controls

对 robosuite `OSC_POSE + Panda gripper` 的 7 个 channel 分别执行正、负小动作：

```text
translation x/y/z
rotation rx/ry/rz
gripper open/close
```

unit tests 检查 quaternion error、最短旋转、q/-q、action normalization、per-axis clipping；
environment-backed test 在 official profile 环境中检查：

- decoded action 与 profile scale 一致；
- 对应 OSC goal / arm control 对该 channel 有有限、可解释的响应；
- 非目标 channel 没有由 action mapping 引起的交换或符号错误；
- gripper `-1=open`、`+1=closed` 与 compiled actuator control方向一致；
- ctrl/force 对 `ctrllimited/forcelimited` actuator 满足 compiled bounds；
- actuator name/id/order 与 compiled Panda model 相同。

当前只检查 `rz` 正负的测试不算六轴验证。测试不得要求动力学完全解耦；验证的是 action
mapping、符号、边界和实际 actuator path。

完成标准：14 个 OSC 正负 controls 与 gripper 两方向都有 machine-readable result；至少一
个真实 environment step 覆盖每个 channel；mutation 任一 mapping/name/sign/bound 会失败。

## 6. 把 run verifier 升级为语义 verifier

Phase 07 run schema 只再升级一次。每个 trace row 保存准确的：

```text
policy_time_s / measurement_time_s / latency_s
complete policy input consumed by controller
typed estimate including V3 future query
TaskExecutive state and recovery transition
normalized / decoded / clipped action
compiled actuator metadata + applied ctrl/force
pre/post public task state
```

`verify_run_artifact()` 必须在不信任 stored summaries/hashes 的前提下重新计算：

1. exact schema、dtype、shape、finite、timestamp 单调性和 tier key set；
2. physics/controller/dev-state/profile hashes及 `dd6fe2ed` dev anchor binding；
3. 按 trace 顺序重新实例化/reset controller，从完整 policy input 重算 typed estimate、
   future query、phase transition、normalized/decoded/clipped action；
4. action scaling、clipping 和 7-channel mapping；
5. actuator metadata 与 official compiled Panda model 的 name/id/order/ranges；
6. applied ctrl/force finite 和已声明 bounds；
7. trace sample count、step/time continuity、termination category、success/failure integrity；
8. trace/payload/run IDs 和所有 digests。

stored Boolean、stored estimate、stored action、stored actuator metadata 和重新封装后的 hash
都不能成为 authority。第 1.4 节每个 resealed mutation 必须被 semantic recomputation 拒绝。

将测试 fixtures 共用的 tier observation builder 提取成一个 test helper，避免
`test_shakebench_oracle.py` 与 completion-red tests 继续复制 schema。

完成标准：合法 current artifacts 通过；所有 resealed adversarial mutations 失败；任一
artifact 能在没有原 rollout process 的情况下重算 controller/action verdict。

## 7. 生成并验证真实三进程 replay

在现有 runner 中提供一个单一、可复现的 determinism 命令，由 parent process 使用
`subprocess` 启动三个新的 Python processes。每个 child raw artifact 记录：

```text
process_index
real os.getpid()
parent run UUID
start timestamp
state/controller/physics/dev-anchor bindings
complete trace and semantic-verifier verdict
```

三个 child 的 PID/process_index 必须不同；payload 因 process metadata 可以不同，比较
authority 是绑定相同后的 state/action/metric trace。manifest verifier 必须：

- 重新打开三个不同 path；
- 校验 file/payload/trace hash 和 manifest 自身 hash；
- 验证三个独立 process identities；
- 调用 Phase 07 semantic run verifier；
- 比较完整 trace schema/sample count/timestamps/state/action/metrics；
- 对预注册 float fields 使用明确 tolerance，其余字段 exact；
- 拒绝重复 path、重复 PID、缺文件、截断、换绑和 mutation。

不要把相同文件 hash 的前缀写成 `process_identity`。保存生成命令、stdout/stderr 摘要和
三个 child exit codes。

完成标准：同一 frozen V0/Gamma=0/dev state 的三个真实 processes 全部通过，manifest
verifier 可从磁盘独立重算；对任一记录的 identity/path/hash/trace 做 mutation 都失败。

## 8. Package、仓库体积与最终 evidence

构建 clean wheel 和 sdist，并分别安装到两个新临时目录，使用 `--no-deps` 从安装位置执行：

```text
import robosuite.utils.shakebench_oracle
load OracleControllerProfile and verify its hash
load/verify shakebench_states_dev.json
run Phase 07 semantic artifact verifier on a compact fixture
compile/create VibrationPickPlaceCan without renderer
```

保存 machine-readable package evidence：artifact path/hash、安装路径、imported module path、
profile hash、dev-state hash、verifier result、environment smoke result 和 exit code。报告只写
“wheel/sdist contains files”不能通过。

避免把约 345 MB 的新旧未压缩 trace 永久写进 Git history：

- 完整旧 traces 移到现有 gitignored `out/phase07_invalidated/`，保留且不删除；
- 完整 R2 traces 放入 `out/phase07r2/`，作为 release-data 输入；
- Git 中保存 compact manifests、summary、必要的小型 adversarial fixtures 和每个外部 raw
  file 的 size/SHA-256；
- Phase 07 report 记录如何从 manifests 定位并验证完整 evidence。

若仓库当前没有合适的 ignored `out/` 规则，只添加精确规则；不要删除原始证据。

完成标准：wheel/sdist 两个隔离安装均通过；Git 新增证据保持紧凑；完整 traces 可通过
manifest hashes 验证且未丢失。

## 9. 重新运行整个 Phase 07R2 矩阵

V1 estimator、recovery 和 V3 law 都会改变轨迹，因此当前 remediation traces 全部失效。
保留其 top-level hashes 后，使用一个最终 controller profile 重新运行：

```text
Gamma=0
  V0 × 全部 10 dev states

matched Gamma=0
  V0/V1/V2/V3 × shakebench-dev-v0-000

positive-Gamma diagnostics
  V0/V1/V2/V3 × Gamma {0.15,0.30}
  × 按 state_id 排序前 3 个 dev states

determinism
  V0 × Gamma=0 × shakebench-dev-v0-000 × 3 fresh processes
```

所有被比较的 tier 使用完全相同的 TaskExecutive、gains、rate、gripper、horizon、recovery
budget 和 action injection；只通过 provider/typed estimate 得到不同信息。保留全部失败，
每个 unsuccessful episode 有非空 failure reason 和 termination category。不得根据结果修改
physics/contact/success threshold，不要求 `V3>V2>V1>V0`。

完成标准：V0 Gamma=0 仍为 10/10；matched/smoke 矩阵无 missing/duplicate/replacement；
所有 run 通过 semantic verifier；新 summary 只引用 R2 raw hashes。

## 10. 验证、冻结与 Phase 8 handoff

至少运行：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
python -m pytest -q tests/test_shakebench_phase07_completion_red.py
python -m pytest -q tests/test_shakebench_oracle.py tests/test_shakebench_metrics.py \
  tests/test_shakebench_providers.py tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
python -m isort --check-only <全部修改的 Python 文件>
python -m black --check -W 1 <全部修改的 Python 文件>
git diff --check
```

再运行：

- 完整 Phase 07 semantic run verifier；
- actuator 7-channel positive-control verifier；
- determinism manifest verifier；
- clean wheel/sdist package verifier；
- 仓库可运行的完整 non-renderer regression。renderer/EGL 缺失单独记录为 infrastructure
  limitation，不冒充通过，也不阻止 State lane。

最终更新 `docs/phase_07_report.md`，分别列出：

```text
correctness fixes
hard gates
positive-Gamma diagnostics
adversarial verifier results
package/determinism evidence
infrastructure limitations
known limitations
```

Phase 07R2 只有同时满足以下条件才能 PASS：

1. V1 attitude、gravity、timestamp 和 latency 在真实 controller path 上正确；
2. Can-in-EEF SE(3) slip 与 public rebound recovery 正确；
3. V3 future response 使用公开 isolator/static context，frame/单位/增益明确；
4. 全部 7 action channels 的正负 mapping 和实际 actuator path 通过；
5. resealed semantic mutations 全部被拒绝；
6. 三个不同 Python processes 的完整 replay 可独立验证；
7. wheel/sdist clean-install evidence 可重放；
8. V0 Gamma=0 10/10，matched/smoke/R2 evidence 完整；
9. physics/contact/task geometry/success threshold 和 dev states 字节未改变；
10. isort、Black、focused tests、non-renderer regression 和 `git diff --check` 通过；
11. 最终 controller/profile/report/compact manifests 已提交并绑定 final commit；
12. `git status --short` 只允许显示明确记录的 ignored raw-output 目录，否则必须为空。

最后生成一个紧凑的 Phase 07→08 handoff record，绑定 final commit、controller profile hash、
dev-state anchor/hash、official physics hash、semantic/determinism/package manifests 和验证命令。
这不是新的 verifier chain；Phase 08 只需重新打开这些现有 R2 manifests 并确认 hashes。

任一条件未满足时保持 Phase 07R2 BLOCKED，并指出最小失败证据。全部满足后才允许进入
Phase 08；完成后停止，不生成 official/knee states，不运行 Phase 08 implementation。
