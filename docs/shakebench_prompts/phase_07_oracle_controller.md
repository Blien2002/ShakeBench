# Phase 07 Prompt：任务语义闭环与共享 State Oracle Controller

你在 `/home/miracle04/Desktop/ShakeBench` 中工作。完成 `VibrationPickPlaceCan`
的任务语义闭环，随后实现可复现的 reference State Oracle controller。持续工作到本
提示词的完成条件满足，或出现本提示词定义的真实 blocker；不要停在计划、代码骨架
或仅单元测试通过的状态。

本阶段允许修复已经确认的任务语义实现缺口，但保持 Phase 06F 冻结的 physics、
contact tuple、task geometry 和 success 数值阈值不变。修复实现缺口后只重验受影响
的合同，不启动新的 physics candidate selection，也不创建新的 Phase 06 协议版本。

## 1. 唯一入口

完整读取：

```text
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/shakebench_prompts/README.md
docs/phase_06fr2_semantic_handoff_report.md
docs/spike_implementation_formal_review.md
```

记录起始 commit 和 `git status`，保留工作树中已有修改。随后在 import controller、
创建环境或读取 Phase 07 state 之前运行一次：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
```

该命令退出 0 且返回 `passed=true` 才能使用 official physics。历史 V1–V8 status
只是审计记录，不能替代此结果。把命令、退出码、profile ID 和 profile hash 写入
Phase 07 报告；后续 Phase 07 代码变化不通过增加 handoff 版本来重新冻结物理。

## 2. 先闭合两个任务语义缺口

在编写 controller 前，用回归测试证明并修复以下两个缺口。它们是既有设计合同的
实现修复，不是新的 benchmark 难度或物理选择。

### 2.1 完整的公共目标区域

State V0–V3 的公共 observation 必须足以在 robot-base frame 与旋转、倾斜的
target-container local frame 之间变换点和位姿。当前只有 center、half-extents、
z bounds 和 orientation mask，无法区分中心相同但姿态不同的目标箱。

采用一个明确且唯一的 scoreable schema：

```text
goal_frame_pos_robot_base       float32[3], m
goal_frame_quat_robot_base      float32[4], unit quaternion, xyzw
goal_inner_half_extents_target  float32[2], m, target-frame x/y axes
goal_z_bounds_target            float32[2], m, target-frame z axis
goal_orientation_mask           bool[3]
```

迁移现有含糊命名的 goal fields，更新 `COMMON_STATE_KEYS`、field contract、环境
observables、privilege audit、Phase 05 observation artifact、相关文档和测试。正式
scoreable observation 只保留上面的单一 schema；不要同时保留两套含义相同的字段。

回归测试至少构造两个 center 和 dimensions 相同、target orientation 不同的状态，
并证明 policy 只使用公共字段就能：

1. 区分两个目标区域；
2. 将 robot-base 点变换到正确的 target-local 坐标；
3. 得到与 privileged evaluator 相同的目标 frame 约定；
4. 在 V0–V3 中获得完全相同的公共 task fields。

目标箱的当前位姿属于完成任务所需的 task geometry。报告中明确说明：V0 仍然没有
专用振动 channel；V2 的增量是显式 support pose/twist/acceleration，不是补齐缺失
的任务几何。

### 2.2 真正连续的 0.50 s success window

离散仿真中的“连续”定义为：每个 internal physics step 都满足 success 子条件。
policy-rate observation 和报告可以保持 20 Hz，但两个 policy boundary 之间的瞬时
失去 containment、bottom support、finger-contact absence、速度或 penetration 合同，
必须重置 success candidate window。

实现最小的 physics-step success accumulator，复用已经编译的几何与命名接触信息；
避免在每个 substep 构造完整报告或重复序列化。现有 `0.50 s`、速度、支撑力和
`0.50 mm` penetration 数值保持不变。

先加入一个红灯回归：在两个 20 Hz boundary 之间注入恰好一个 physics-step 的失败
条件，旧实现会错误锁存 success；修复后该事件必须重置窗口。再覆盖持续满足、持续
失败、时间倒退/reset 和锁存后的行为。

修复后重跑受影响的 success/environment/observation/privilege 测试，以及 official
profile 下 open-worktable 和 target-bottom 的 Gamma=0 task-native settling。若冻结
物理在新采样语义下仍满足既有阈值，记录 targeted revalidation 即可。若它真实失败，
保存最小 trace 并标记 scientific blocker；不要调整阈值、contact 或 solver 使其通过。

完成本节的标准：两个回归测试在修复前能针对旧行为失败，修复后通过；公共 schema
和 privilege boundary 一致；受影响的 Gamma=0 settling 仍通过。之后才开始 controller。

## 3. 实现共享 Oracle controller

直接在现有目录中修改或新增文件，主要落点为：

```text
robosuite/utils/shakebench_oracle.py
robosuite/utils/shakebench_providers.py
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/scripts/shakebench_run_oracle.py
tests/test_shakebench_oracle.py
```

保持一条执行链：

```text
public State task observation
→ VibrationProvider(V0/V1/V2/V3)
→ typed VibrationEstimate
→ one SharedVibrationControlLaw
→ one TaskExecutive
→ robosuite 7D action: 6D OSC_POSE + 1D gripper
```

V0 使用零专用振动估计；V1 只使用 noisy/delayed IMU；V2 使用当前 realized support
state；V3 额外查询固定 latency horizon 的 authored future program。V2 controller
不保存 support history、不辨识谱线、不外推 excitation program。

`TaskExecutive` 至少包含：

```text
settle → approach → descend → grasp → lift → transport → place → release → verify
```

每个 phase 有明确进入条件、deadline 和 failure reason。target 与避障 waypoint 每步
由公共 target frame 计算。success 只由 environment evaluator 返回，policy 不自报。

controller 只能读取公开 observation、公开 static context 和标准 action API。任何用于
决策的 bilateral grasp、grasp loss、finger-table risk 或 slip estimate，都必须由公开
的 gripper state、EEF/Can pose、fingertip position、wrist wrench等字段导出，并在测试中
证明移除 privileged truth 后仍工作。native contact、penetration 和 evaluator truth 只进
recorder。

slip、短暂 contact loss 和 placement rebound 是诊断事件，不自动成为 reference policy
的终止条件。只要 episode horizon 和安全合同允许，TaskExecutive 应尝试稳定、重抓或
重新放置；最终 task success 仍由相同 evaluator 判断。不可恢复的超时、动作异常、
physics violation 和 controller crash 才产生终止 failure reason。

使用 robosuite 标准 Panda `OSC_POSE` 和默认 1D gripper action。验证并记录 normalized、
raw、decoded、clipped、applied action，以及实际 actuator force/torque；不能用配置值代替
实际输出证据。不要复制 spike 的 8D 双指 action、force switch 或 `move_action_gain`。

## 4. 开发集、调参与冻结

只使用既有 10 个 dev state IDs。Phase 07 可以在以下预先固定的开发难度上调试同一
controller，三个值均已在 Phase 06 driver matrix 中使用：

```text
Gamma_commanded ∈ {0.00, 0.15, 0.30}
```

Gamma=0 用于证明 nominal task 可解；0.15/0.30 用于暴露移动目标、延迟补偿、抓持和
恢复路径中的实现问题。不得读取 100 knee states、400 official states 或扫描新的 Gamma
来寻找漂亮的 tier 排序。

所有被比较的配置必须同时应用于 V0–V3。TaskExecutive、phase thresholds、OSC gains、
rate、gripper semantics、horizon、recovery budget 和 action injection 只有一份参数；tier
只通过 `VibrationProvider` 输入影响 action。记录每次实际参与选择的 controller profile
和 dev 结果，最终只发布一个冻结 profile。

开发期间可用固定 dev state 子集缩短反馈；最终候选必须完成：

1. Gamma=0 的 10 个 State-V0 dev states 全部成功；
2. Gamma=0 至少一个 matched state 在 V0–V3 记录 task state、provider payload、typed
   estimate、control-law output 和最终 action，证明 controller profile 与 action injection
   路径相同；数值 action 可以因真实 provider 输入不同而变化，但必须可逐项归因；
3. 按 state ID 排序取前 3 个 dev states，在 Gamma=0.15 和 0.30 上运行全部 V0–V3；
4. positive-Gamma smoke 中 action/state 全部 finite、无 privilege leak、无 infrastructure
   error，并完整记录 success 与 failure reason。

positive-Gamma success rate、tier 单调排序、slip 数量和恢复次数均为诊断结果，不是本
阶段硬门。不要为了得到 `V3 > V2 > V1 > V0` 修改 controller 或 benchmark。

## 5. 必须形成的验证证据

实现与测试必须覆盖真实调用 seam，而不只是孤立 helper：

- 完整 target-frame public schema 与 privilege isolation；
- physics-step success-window 瞬时失败回归；
- action 六轴与 gripper channel 的正向控制；
- phase、deadline、recovery 和 failure-reason transitions；
- moving/rotating target-frame command；
- public-signal grasp/slip/risk estimation，且无 `env.sim` 依赖；
- V2 current-only 与 V3 fixed-latency query；
- tier 间 controller profile/hash 相同；
- Gamma=0 matched pipeline attribution；相同 typed neutral estimate 必须产生相同 action；
- applied actuator force/torque；
- 同一冻结 controller trace 的 3-process deterministic replay；
- 受影响的 environment/controller/non-renderer 回归。

renderer/EGL 缺失若只影响画面生成，记录为独立 infrastructure limitation；它不阻止
State controller 和非渲染证据完成。真实 simulation crash、NaN、physics violation、
privilege leak 或不确定性重放是 blocker。

## 6. 门控纪律

执行 `docs/shakebench_prompts/README.md` 的门控治理。新增硬门前，先在 Phase 07 报告中
写清它保护的具体研究结论、阈值来源、失败影响范围和可复现信号。普通实现错误直接修复
并重跑受影响测试；诊断指标如实报告；本阶段不增加 verifier 链、handoff 层或 physics
profile 版本。

## 7. 完成条件

只有以下条件全部满足才报告 Phase 07 PASS：

1. Phase 06F-R2 单一入口验证通过；
2. 完整目标 frame schema 和 physics-step success window 两个语义缺口已由回归关闭；
3. targeted Gamma=0 settling revalidation 通过且未修改冻结物理或 success 数值；
4. Gamma=0 State-V0 dev 为 10/10；
5. V0–V3 共享一个 controller profile，Gamma=0 matched pipeline attribution 通过；
6. 固定 positive-Gamma smoke 完成，所有结果均保留，不以 tier 排序决定 PASS；
7. 3-process deterministic trace、privilege isolation、applied-actuator 和相关回归通过；
8. controller profile 可从 clean wheel/sdist 安装读取；
9. 输出 raw dev episodes、action/force evidence 和 `docs/phase_07_report.md`。

报告分别列出 hard-gate results、diagnostics 和 infrastructure limitations，并给出从 raw
证据复现每个 hard gate 的命令。完成后停止；不生成 knee/official states，不运行 knee
scan 或 official evaluation。
