# Phase 07R5 Prompt：Oracle Motion Semantics 与 Preview Closure

你在 `/home/miracle04/Desktop/ShakeBench` 中继续 Phase 07。本提示词在 R4 controller core
完成后执行。R4 可以保持 `BLOCKED_BY_PHASE_07R4_EVIDENCE_REGENERATION`，但必须由
`docs/phase_07_r4_manifest.json` 证明 state 002 成功、V0/Gamma=0 为 10/10、grasp/safety
语义通过，以及 semantic/matched/actuator/determinism/package 基础证据存在。缺少
Gamma=0.30 diagnostic 和 final commit 是本轮接受并最终关闭的 deferred evidence，不应先在
R4 重跑。

R4 若因 nominal solvability、grasp trajectory、clearance、worktable bounds、PRELIFT、safe
recovery 或 semantic verifier 失败而 BLOCKED，则继续执行 R4，不能进入本提示词。

本轮修复共享 State reference controller 的运动语义、阶段动作约束与 V3 preview 证据，
然后重新闭合 Phase 07。目标是让 V0–V3 provider 估计同一个物理对象，使共享控制律的
比较具有可解释性；同时关闭已确认的新旧 phase 限幅遗漏、robot-base 速度误判和未接入的
anchor drift。不要把 reference controller 称为数学性能上界，也不要进入 Phase 08、knee
scan 或 official evaluation。

## 1. R4 后置入口与权威输入

主路径只读取：

```text
docs/shakebench_prompts/README.md
docs/phase_07_r4_manifest.json
docs/phase_07_report.md 顶部 R4 current section
docs/oracle_benchmark_comparison_research.md
docs/robosuite_benchmark_design_tree.md 的 State tier / action / physics gate / scoring sections
docs/robosuite_benchmark_v0_spec.md 的 controller / tier / failure / release-gate sections
```

R2/R3/R4 prompt 全文只作为 disclosed provenance：仅当 R4 manifest verifier 失败、当前报告
无法解释某个 binding，或回归测试明确指向历史合同冲突时读取对应文件。不要在正常 R5
路径中同时加载三份已执行的长提示词。

开始前检查 `git status`、记录当前 commit 和已有修改，不覆盖其他人的工作。验证 R4 core：

1. state 002 的 after artifact 是 environment success；
2. 同一 R4 profile 下 V0/Gamma=0 frozen dev matrix 为 10/10；
3. before/after artifact、最终 10-state artifact、路径、大小、SHA-256 和 semantic verdict 完整；
4. R4 final controller profile、task-context、dev-state anchor 和 physics hashes 完整；
5. anchor/clearance/worktable bounds/PRELIFT/safe recovery tests 通过；
6. matched Gamma=0、actuator 14/14、真实三进程、clean wheel/sdist 和 Gamma=0.15 smoke 已完成；
7. Gamma=0.30、final evidence regeneration 与 final commit 明确列为 deferred-to-R5。

第 1–6 项任一缺失时，将 R5 标记为 `BLOCKED_BY_MISSING_PHASE_07R4_CORE`，列出缺失证据并
停止。第 7 项缺失不是入口 blocker，由 R5 第 15–17 节一次性关闭。入口成立后，将报告顶部
状态更新为 `BLOCKED_BY_PHASE_07R5_MOTION_SEMANTICS`，并把 R4 core 结果保留为不可改写的
provenance。

冻结输入保持不变：

```text
dev-state anchor = dd6fe2edb6384ccdb5116be44f07592b4864e377
official physics = shakebench.official.physics.v2
physics SHA-256  = c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c
robot            = Panda
action           = 6D OSC_POSE + 1D gripper
policy rate      = 20 Hz
```

保存 R4 输入 artifact 的 byte hashes。本轮不得修改 physics/contact、task geometry、success
thresholds、dev states、excitation states 或 simulator timestep；不得以改变 Gamma、摩擦、
Can/target 尺寸、horizon 或状态集合来改善控制器结果。

完成标准：R4 后置条件可由文件和 verifier 独立确认；所有冻结 hash 与 R4 入口一致。

### 1.1 执行顺序

R5 必须按以下顺序执行：

```text
R4 compact-manifest verification
→ motion semantics / capability / common estimate implementation
→ analytic and focused regression
→ V0/Gamma=0 10/10 on the new final profile
→ one final matched + Gamma {0.15,0.30} + ablation run
→ one final determinism + actuator + package closure
→ final commit and Phase 8 handoff
```

在 motion-semantics 修改完成前，不运行 R4 缺失的 Gamma=0.30，不重新生成 package、actuator
或 determinism evidence。它们只对最终 R5 profile 有效。

## 2. 本轮保护的研究命题

R5 只保护以下命题：

1. V0–V3 的差异来自合法信息通道，而不是同名 estimate 字段装入不同物理对象；
2. 同一个 tier-invariant control law 消费同一 frame、reference point、units 和 time semantics；
3. 物体与移动目标保持相对静止时，controller 与 evaluator 对“稳定”的判断一致；
4. V3 只使用当前时刻已公开的 authored program 预测未来，不读取 future realized state；
5. Phase 07 的 nominal-solvability 与可重复证据来自固定状态上的真实 physics execution。

最终论文可表述为“冻结的 reference controller 在不同合法信息 tier 下的表现”。除非另有
严格证明，不得表述为信息 tier 的最优性能、任务可达上界或未来信息的理论价值。

每个新增硬门按 `README.md` 的门控治理记录 `protected_claim`、`threshold_basis`、
`failure_scope` 和 `reproduction`。已有 gate 能保护同一命题时扩展已有 verifier，不新建重复门。

## 3. 固定修改前基线

在任何 production 修改前运行并保存：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_oracle.py \
  tests/test_shakebench_phase07_completion_red.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_metrics.py
```

重新验证 R4 manifest、成功的 state-002 after artifact 与 R4 profile，再以同一 profile 重放
一次 state 002，输出到 `out/phase07r5/r4_entry_replay.json`。当前
`BLOCKED_BY_PHASE_07R4_EVIDENCE_REGENERATION` 不改变该行为，因为它的 controller core 已
成功；其他 R4 BLOCKED 状态不满足 R5 入口，应返回 R4，而不是在此重跑历史失败。不得覆盖
R4 原文件。

从真实 trace 和纯函数探针分别记录：

```text
phase and phase transition
desired / compensated / normalized / clipped action
Can, target, worktable and EEF poses
controller public linear/angular speed
evaluator relative linear/angular speed
anchor error and action-limit state
V0–V3 estimate source/frame/reference point/timestamps
V3 predicted versus reconstructed authored motion
```

完成标准：基线 artifact 可重算，当前问题按 action、kinematics、estimator、preview 四类分开，
不把历史 R2/R3 失败当作当前 R4 树的新实测结果。

## 4. 先建立可证伪回归

测试应穿过真实 public interfaces。对 R4 已修复并已有覆盖的行为先证明现有测试会通过；只为
当前仍存在的缺陷建立红灯，不制造与实现细节绑定的假失败。

### 4.1 Phase capability 与动作限幅

已知静态缺陷：`ShakeBenchOracleController.action()` 的下降限幅只枚举旧
`DESCEND/GRASP`，R4 当前执行链使用 `VERTICAL_DESCEND/GRASP_CLOSE`。构造相同期望 6D delta，
证明新旧等价阶段必须获得相同的 translation/orientation rate、saturation 和 gripper 约束。

测试覆盖所有 `TaskPhase`，并要求每个 phase 恰好属于一个显式 motion capability：

```text
HOLD
FREE_SPACE
CLEARANCE_TRANSLATE
CONTACT_APPROACH
GRIPPER_CLOSE
OBJECT_HELD
RECOVERY_LOWER
RECOVERY_RETREAT
TERMINAL
```

phase 名称是执行状态，capability 是动作约束来源。新增或兼容旧 phase 时，遗漏映射必须在
profile 构建或测试收集时失败。

### 4.2 目标相对稳定语义

至少构造以下公开状态序列：

1. Can 和 target/worktable 在 50 ms 内刚体共同平移 5 mm，相对位姿恒定；
2. Can 和 target/worktable 共同旋转，相对位姿恒定；
3. target 静止、Can 单独平移；
4. Can 静止、target 单独平移；
5. 四元数经过 `q`/`-q` 等价表示和 ±π wrap；
6. timestamp 重复、倒退或 episode reset。

前两种的相对线/角速度应接近零，后三类产生正确、有限、frame-aware 的结果。controller 的
VERIFY、ALIGN_SETTLE 和 recovery safety 应与 evaluator 的目标相对运动定义一致；时间异常
必须 reset history 或 fail closed，不能产生无穷速度。

### 4.3 R4 路径与 anchor 不变量

重新运行 R4 的初始斜向 approach、完整 tool-envelope swept clearance、worktable bounds 和
safe recovery 测试。额外证明：

- 初始动作先建立足够 clearance，再做 lateral alignment；
- 证书覆盖 swept segment / sampled path，而不只检查终点 EEF 或两个瞬时 tip；
- `_anchor_error_m()` 与 `anchor_drift_tolerance_m` 实际控制阶段转移；
- anchor drift 超限进入保守撤回/重新对齐，未超限继续 descend；
- 重新锚定只发生在 Can 相对 worktable 稳定且完整 collision envelope 在 bounds 内时。

若 R4 已完整实现这些保证，本节应直接为绿灯；若有缺口，将其作为 R4 carry-over 在本轮关闭，
不得删除 R4 测试或降低 clearance/bounds。

### 4.4 Tier estimate 的共同物理量

构造解析运动 fixture，使 deck、robot base、worktable 和 target 分别出现：刚体共同运动、
isolator 相对运动、纯平移、纯旋转、带 lever arm 的旋转、静止和瞬态 ramp。对 V0–V3 的 typed
estimate 断言：

```text
semantic quantity
coordinate frame
reference point
linear/angular units
measurement timestamp
policy timestamp
prediction timestamps
latency and validity/confidence
```

所有 tier 字段含义完全相同。信息不足时用明确的 validity/confidence 或零阶估计表达，不能把
另一物理量塞入同一字段。共享控制律不得通过 `tier` 字符串猜测语义。

### 4.5 V3 causal preview

至少覆盖：零 program、ramp 前、ramp 中、ramp 后、三个频率区间、纯旋转、多个谱线和当前
realized state 相同但未来 program 不同。验证：

- 当前时刻之前相同的 program/history 产生相同当前估计；
- 未来 authored program 变化只影响 prediction horizon，不改写 measurement；
- quintic ramp 的位移、速度、加速度包含 envelope 的一阶/二阶导数项；
- prediction query 全部晚于 policy timestamp，且不会读取 future simulator state；
- 预测从当前公开状态连续出发，在 horizon 起点没有不合理跳变；
- frame rotation、角量和 reference-point translation 均可独立复算。

完成标准：每个问题都有当前失败或已有正确行为的明确证据；测试名字描述物理不变量，不描述
拟采用的私有函数结构。

## 5. 用 capability 统一 phase 动作约束

为 `TaskPhase` 建立唯一、可枚举的 capability 映射，或者用等价的 typed phase policy。至少集中
定义：translation/orientation 最大 normalized action、接触方向限制、gripper action、是否允许
vibration feed-forward、超时与安全退出。

将 `VERTICAL_DESCEND` 和 `GRASP_CLOSE` 纳入与旧 `DESCEND/GRASP` 等价或更保守的接触动作
限制。兼容 alias 可以保留用于旧 artifact 解析，但 production phase machine 只使用一套规范
名称。所有限制在 vibration compensation 后、normalized action 输出前执行，使前馈不能绕过
contact safety；同时记录 pre-limit 与 post-limit action。

完成标准：4.1 全绿；每个 phase 的 capability 在 trace 中可见；没有依赖散落的 phase 集合来
重复决定同一约束。

## 6. 统一 public relative kinematics

实现 controller-owned、public-only 的 SE(3) relative-motion tracker。对于抓取、桌边和
recovery，跟踪 `T_worktable_can`；对于 placement success proxy，跟踪 `T_target_can`。target 与
worktable 在当前任务中属于同一刚性 assembly，但 origin 不同，必须通过已公开静态 context
显式转换，不能把二者数值直接互换。

tracker 至少满足：

```text
relative translation before finite difference
shortest rotation-vector angular difference
declared timestamp and dt
q/-q invariant
reset on new episode or non-monotonic time
no simulator/contact/evaluator truth
```

ALIGN_SETTLE、anchor drift、PRELIFT follow、VERIFY 和 recovery 根据各自物理问题消费明确的
relative quantity。最终 VERIFY 的 speed semantics 必须与
`ShakeBenchMetrics.success_snapshot()` 一致；策略仍只形成 public proxy，环境 evaluator 保持
最终成功 authority。

完成标准：4.2 全绿；共同运动不再被误判；真实相对滑移仍能触发抓取或恢复保护。

## 7. 定义一个共同的 disturbance estimate contract

先写清设计再改代码。将当前泛化过度的 `VibrationEstimate` 重构或版本化为 typed
`RelativeSupportMotionEstimate`（名称可调整），其唯一语义为：

> 在声明的 robot-control frame 和 reference point 上，worktable/target 相对 robot base 的
> 当前运动与未来短时运动估计。

contract 至少包含：

```text
schema/version
frame and reference point
current relative pose/twist/acceleration as actually used
measurement and policy timestamps
prediction time grid
predicted relative pose/twist/acceleration
validity/confidence or uncertainty
source/provenance
units
```

只保留控制律实际消费且能验证语义的字段。若控制律只需要 twist 和预测 displacement，不为了
形式完整而发布未使用的 acceleration。所有 arrays immutable、finite、shape-checked；序列化后
可由 semantic verifier 重算。

各 tier 使用同一目标量：

- **共同 task-state history**：所有 V0–V3 都可使用公共 goal/worktable pose history；这是
  State track 已允许的信息，不能只给高 tier 使用。
- **V0**：没有专用振动 channel；从共同公开状态做 causal baseline，或者显式给出 neutral/
  low-confidence estimate。两者择一并在修改 positive-Gamma 结果前冻结。
- **V1**：只增加 delayed/noisy deck IMU。若从 deck inertial motion 推断 table-relative response，
  所需 isolator model 和固定 transforms 必须作为所有 tier 可见的静态 contract；使用 causal
  state estimator，不能读取 V2 realized support state。
- **V2**：将当前 realized table-in-deck pose/twist/acceleration 转换为共同 frame 和 reference
  point；同时保留 V1 字段的存在不代表要把不等价信号相加两次。
- **V3**：在 V2 当前状态上增加 authored future excitation，输出同一相对运动量的预测轨迹。

保持 tier key sets 与信息预算不扩大。若必须新增公共静态 transform/model context，证明它不含
episode outcome 或 runtime realized vibration，并让 V0–V3 完全相同；纳入 context hash 和文档。

完成标准：4.4 全绿；不再出现 V1“deck 自身惯性加速度”与 V2“table 相对 deck 加速度”共享
同一字段/增益的情况；neutral estimate 在所有 tier 上 action-equivalent。

## 8. 让共享控制律消费共同语义

`TaskExecutive` 继续产生无补偿的 6D task-space error。共享 control law 只消费第 7 节的共同
estimate，输出明确单位的 task-space correction，再由 capability 执行最终安全限幅。

控制律满足：

1. 不读取 `tier` 名称，不按 V0/V1/V2/V3 分支；
2. estimate invalid 时采取已冻结的 neutral/fallback 行为；
3. pose、twist、acceleration 与 prediction 不重复补偿同一运动；
4. rotation、translation 和 lever arm 的 frame 变换完整；
5. phase-dependent 使用方式由 capability/公开 phase 状态决定，对所有 tier 相同；
6. correction、clipping 和 saturation 全部记录，可独立重算；
7. gripper action 不受 vibration correction 影响。

参数单位写入 profile，不以“V3 必须优于 V2”或任何预期 tier 排名调参。参数选择先保护
V0/Gamma=0 nominal solvability，再依据解析 tracking/prediction error、动作饱和和安全余量；
positive-Gamma SR 只作诊断，不能驱动物理或成功阈值变化。

完成标准：同一 estimate 导致相同 correction；不同 provider 的差异可追溯到合法输入导致的
共同目标量估计差异。

## 9. 闭合 V1 estimator 的因果与时延

保留 R2 已建立的 stateful attitude、gravity separation、canonical IMU sample rate 和完整
end-to-end latency。进一步明确：

- IMU specific force、deck inertial acceleration、robot-base motion 与 table-relative response
  是不同量；每一步转换都有 frame、单位与公式；
- measurement timestamp 对应 delayed window 的有效时刻，policy timestamp 对应 action；
- overlap window 不重复积分相同 gyro interval；reset 后状态确定；
- 估计 worktable relative motion 时使用 causal history 和已公开模型；无法观测的状态通过
  uncertainty/validity 表达；
- V1 不访问 `table_*_in_deck_frame`、future authored program、raw qpos/qvel 或 evaluator truth。

用 synthetic IMU、静止倾斜、共同加速、旋转和 ramp fixture 验证符号、相位和时延。不要以
某个 episode 成功来替代 estimator 本身的误差验证。

完成标准：V1 contract 与第 7 节一致，privilege test 对直接和缓存/属性/trace 路径均 fail
closed。

## 10. 闭合 V2 当前状态转换

V2 使用公开 `table_pose/twist/accel_in_deck_frame`，将其变换到第 7 节共同 frame/reference
point。转换至少覆盖：

```text
translation and rotation composition
angular velocity and acceleration
reference-point lever arm
Coriolis/centripetal terms where required
deck-to-robot-base fixed transform
timestamp alignment
```

不要同时把 deck inertial acceleration 与 table-in-deck relative acceleration作为同一 correction
重复加入。若控制律需要两者，分别命名、分别建模，并证明共同目标量的组合公式。

完成标准：解析 rigid-body fixture 可复算 V2 estimate；V2 current-only 不因 V3 program 字段或
future query 改变。

## 11. 将 V3 preview 变成可验证的短时预测

以 V2 的当前共同状态作为 prediction 初值，以公开 authored program 作为未来 forcing，在一组
显式时间点上预测共同的 table/target-relative motion。prediction horizon 和时间网格写入
profile；若最终控制律只消费一个汇总量，也必须从完整短时轨迹通过记录在 profile 的确定性
operator 得到。

当前 `future_relative_acceleration_from_public_program()` 使用稳态 transfer 和固定 0.1 s 点的
方式可作为对照 baseline。最终实现必须：

1. 与 `reconstruct_authored_motion()` 的 quintic ramp 位移/速度/加速度定义一致，包含 ramp
   derivative terms；
2. 从当前公开 realized relative state 连续初始化，说明隔振瞬态如何处理；
3. 处理六轴、frame rotation、reference point 和各轴自然频率/阻尼；
4. 只查询未来 authored forcing，不查询未来 realized support/object/robot state；
5. 输出 prediction validity，并在模型失配或数值异常时退回 V2-equivalent 行为；
6. 零 program 与 V2 action-equivalent。

建立与真实任务 episode 解耦的预测验证矩阵：ramp 三个区间、频率低于/接近/高于隔振固有
频率、单轴与混合轴、不同 phase。误差阈值从数值积分误差、policy period 和控制 correction
预算推导，记录 threshold basis；不以 tier success ranking 选择预测模型。

不强制采用 MPC。状态空间积分、解析传播或其他因果方法均可，只要第 4.5 节和上述契约可
独立重算。

完成标准：4.5 全绿；V3 的 improvement/failure 可解释为预测和控制结果，不是当前状态语义被
替换。

## 12. 诊断基线与实验边界

在不改变 main-track 定义的前提下增加或复用以下 diagnostic switches，全部写入 artifact 和
profile hash：

```text
task executive only / compensation off
common public-history estimator
current-state compensation
preview off / preview on
```

这些开关用于回答补偿贡献，不形成新的 official tier。所有比较使用相同 dev state、episode
budget、action space、rate、physics 和 success evaluator。报告：

```text
success and termination category
first-grasp success
recovery count and final success after recovery
phase durations
relative-motion estimate/prediction error
action correction and saturation fraction
minimum clearance/table-edge margin
in-hand slip/contact loss
```

不要跨 reset 丢弃失败直到成功；如果以后生成成功 demonstrations，另行同时报告 raw attempt
success、accepted-demo count、reset count 和 planner/controller failure。

本轮只运行 dev 和已预注册的 positive-Gamma smoke diagnostics。禁止读取、生成、扫描或评估
knee/official states。

## 13. Tight execution loop

按以下顺序迭代：

1. 纯函数/状态机回归：4.1–4.5；
2. R4 state-002 路径和 action trace；
3. 全部 10 个 V0/Gamma=0 dev states；
4. matched Gamma=0 的 V0–V3/dev-000；
5. positive-Gamma diagnostic matrix；
6. determinism、actuator、semantic verifier 和 package closure。

state-002 命令沿用 R4，输出到新目录：

```bash
python -m robosuite.scripts.shakebench_run_oracle \
  --tier V0 --gamma 0.0 \
  --state-id shakebench-dev-v0-002 \
  --horizon-steps 1200 \
  --output out/phase07r5/dev002_candidate.json
```

每轮 candidate 保存 profile/context hash 和完整 semantic trace。state 002 必须由 environment
evaluator 判定成功；`horizon_exhausted`、仅 controller 自报完成或删除失败 trace 均不算通过。

通过后运行其余九个冻结 dev states。新失败按 phase/capability/estimator/prediction taxonomy 修
共享逻辑；禁止 state ID、seed、tier 或 outcome-specific 分支。R4 若已达到 10/10，R5 改动后
仍须重跑，因为 action/estimator/profile hashes 已改变。

完成标准：修改循环有 before/after 可比 artifact；失败均保留在分母和 failure histogram。

## 14. Artifact 与 semantic verifier 升级

controller/estimate schema 变化时版本化 run schema，不让旧 verifier 静默接受新语义。每个
policy step 至少记录：

```text
public policy input and exact key set
relative-kinematics tracker input/output/timestamps/reset generation
estimate schema/frame/reference point/units/validity/source
measurement, policy and prediction timestamps
predicted trajectory or deterministic compact representation
task desired action
pre-capability compensated action
post-capability normalized/clipped action
gripper action, applied ctrl and realized actuator force
phase/capability/anchor/recovery diagnostics
```

verifier 从 public input、static context 和 profile 独立重算 tracker、estimate、correction、limits、
phase transitions 与 failure verdict。增加 resealed mutations，至少覆盖：frame、reference point、
timestamp、prediction value、validity、phase capability、anchor drift、normalized action 和 final
termination。改变 JSON 后重算外层 hash 仍必须被 semantic mismatch 拒绝。

raw evidence 写入 ignored `out/phase07r5/`。Git 只保存 compact manifest/summary、必要 fixtures、
raw 路径/大小/hash 和 reproduction commands。不要把完整 trace 加入 runtime package assets。

完成标准：旧 schema 明确迁移或拒绝；新 artifact 可以在独立进程和 clean package 中验证。

## 15. 重新生成 Phase 07 authority

R5 修改 controller/profile/provider/runner 后，R4 及之前的 rollout、determinism、actuator 和
package evidence只保留 provenance，不能作为最终 PASS authority。用最终 profile 重新生成：

```text
V0 Gamma=0                  10 frozen dev states
matched Gamma=0             V0/V1/V2/V3 × dev-000
positive-Gamma diagnostics  V0/V1/V2/V3 × {0.15,0.30} × first 3 dev states
diagnostic ablations        compensation/history/preview switches on fixed dev subset
determinism                  V0/Gamma=0/dev-000 × 3 fresh processes
actuator                     all 7 channels × positive/negative
package                      clean wheel + clean sdist isolated installs
```

positive-Gamma diagnostics 不设置 V0<V1<V2<V3 单调硬门。更高 tier 理论上可忽略新增信息，但
当前固定算法的实测性能不保证单调；任何反常结果保存并解释，不围绕期望排名反调 physics、
success 或 state selection。

final profile 选择顺序必须在候选结果之前写入报告，至少包含：

1. 所有 R4/R5 safety 和 semantic tests；
2. V0/Gamma=0 nominal solvability；
3. analytic estimator/prediction error；
4. action saturation 与安全余量；
5. recovery attempts；
6. deterministic tie-break。

不得把 positive-Gamma tier 排名、knee 或 official 结果放入选择规则。

## 16. 最低验证集合

至少运行：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_phase07_completion_red.py \
  tests/test_shakebench_oracle.py \
  tests/test_shakebench_actuators.py \
  tests/test_shakebench_metrics.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
python -m isort --check-only <全部修改 Python 文件>
python -m black --check -W 1 <全部修改 Python 文件>
git diff --check
```

随后运行完整可用的 non-renderer regression。若环境中的第三方 pytest plugin 干扰收集，记录
原始失败，并用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ...` 隔离外部 plugin；不能
用该变量隐藏仓库自身 plugin、fixture 或测试失败。

## 17. PASS 与 Phase 08 handoff

只有以下全部满足才把 Phase 07 标为 PASS 并授权 Phase 08：

1. R4 已实际执行且其 anchor、clearance、worktable bounds、PRELIFT 和 safe recovery 门保持通过；
2. 所有 current production phases 映射到唯一 capability，新阶段不能绕过接触动作限幅；
3. controller stability 使用 Can 相对 target/worktable 的 SE(3) 运动，与 evaluator 语义一致；
4. anchor drift threshold 已接入阶段转移并有正负测试；
5. V0–V3 typed estimate 表示同一 frame/reference point/time 下的共同物理量；
6. V1 causal/latency/privilege、V2 transform 和 V3 causal preview 的解析验证全部通过；
7. shared law 不读取 tier 名称，neutral estimates action-equivalent，capability 在补偿后执行；
8. state 002 成功，全部 10 个 V0/Gamma=0 dev states 为 10/10；
9. matched/smoke/ablation failures 完整保留，未以成功筛选替代原始 attempt；
10. semantic mutation、三进程 determinism、actuator 和 clean package evidence 全部由最终 hashes
    新生成并通过；
11. physics/contact/task geometry/success thresholds/dev states 字节保持冻结；
12. 未读取或运行 knee/official states；
13. `docs/phase_07_report.md`、compact manifests、规范术语与 Phase 07→08 handoff 绑定 final
    commit，工作树除明确 ignored raw output 外为空。

最终重写 `docs/phase_07_report.md` 顶部当前结论。报告必须区分：

```text
reference controller capability
information-tier comparison
diagnostic expert / ablation（若存在）
environment success authority
```

给出 R4 entry 状态、R5 before/after 最小证据、共同 estimate contract、frame/time diagram、
state-002 timeline、10/10 summary、预测误差矩阵、final hashes 和全部未关闭限制。旧 R2–R4 内容
压缩为 provenance 索引，不删除原始 hashes。

任一条件不满足时保持 `BLOCKED_BY_PHASE_07R5_MOTION_SEMANTICS`，保存最小复现，明确被阻止的
研究命题并停止在 Phase 07。全部通过后才授权 Phase 08，随后停止。
