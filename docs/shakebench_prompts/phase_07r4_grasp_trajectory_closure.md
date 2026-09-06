# Phase 07R4 Prompt：Grasp Trajectory 与 Edge-Safe Recovery Closure

你在 `/home/miracle04/Desktop/ShakeBench` 中继续 Phase 07。R3 已正确实现 Panda
open/close 符号、严格 public grasp、PRELIFT_VERIFY 和早期 slip detection，但最新
environment replay 仍不能通过 V0/Gamma=0 nominal solvability。

本轮只修通用抓取轨迹锚定、真实 worktable bounds 和 recovery release 时序。保持 R2/R3
已经通过的 observation、IMU、V3、semantic verifier、actuator mapper、physics-step success
与 dev states 不变。持续工作到 state 002 和全部 10 个 dev states 成功，再重新闭合 Phase 07
evidence；不要实现 Phase 08。

## R4 → R5 交接补充（后续审查决定）

R4 的完成边界是 controller execution core，而不是最终 Phase 07 authority。若
`docs/phase_07_r4_manifest.json` 可以独立证明以下全部成立：

```text
state 002 environment success
V0/Gamma=0 frozen dev states = 10/10
frozen grasp anchor / swept clearance / actual worktable bounds / PRELIFT / safe recovery tests pass
R4 final profile and WorktableTaskContext hashes recorded
semantic verifier pass
matched Gamma=0, actuator 14/14, real 3-process determinism and clean package smoke completed
Gamma=0.15 fixed smoke matrix completed with explicit failures
```

则 R4 core 视为 **COMPLETE_FOR_R5_ENTRY**。如果剩余缺口只有 Gamma=0.30 diagnostic、最终
evidence regeneration 和 final Git commit，不再在 R4 中补跑：R5 会修改 controller/provider
motion semantics，这些运行即使现在完成也会失效。报告可以保持
`BLOCKED_BY_PHASE_07R4_EVIDENCE_REGENERATION`，但该状态只阻止 Phase 8，不阻止 R5。

R4 若仍有 state 002/10×Gamma=0、clearance、bounds、PRELIFT、recovery、semantic 或安全门
失败，则不满足本交接，必须继续 R4。R5 不得用 motion-semantics 重构掩盖未关闭的 grasp
execution blocker。

本补充优先于下文要求 R4 自己重新生成 Gamma=0.30/final authority 的旧完成表述。R4 保存
compact manifest 和未完成清单后停止，最终 matched/smoke/determinism/actuator/package/commit
只在 R5 最终 profile 上生成一次。

## 1. 当前确定性 blocker

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_07r2_final_closure.md
docs/shakebench_prompts/phase_07r3_nominal_solvability.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_07_report.md
```

冻结输入：

```text
dev-state anchor = dd6fe2edb6384ccdb5116be44f07592b4864e377
official physics = shakebench.official.physics.v2
physics hash     = c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c
worktable        = 0.65 × 0.60 m
Can radius       = 0.02509177806572465 m
```

先验证 Phase 6 handoff、dev asset bytes 和当前 R3 semantic verifier。将报告状态改为
`BLOCKED_BY_PHASE_07R4_GRASP_TRAJECTORY`，保留所有 R3 candidate hashes。

使用当前代码重新生成基线：

```bash
python -m robosuite.scripts.shakebench_run_oracle \
  --tier V0 --gamma 0.0 \
  --state-id shakebench-dev-v0-002 \
  --horizon-steps 1200 \
  --output out/phase07r4/blocker_before.json
```

当前已观察到的通用失败机制：

```text
initial/early approach worktable-local Can x ≈ -0.113 m
first grasp end                    x ≈ -0.212 m
second descend/grasp               x ≈ -0.286 m
PRELIFT_VERIFY                     x ≈ -0.311 m
Can radius included: -0.311 - 0.025 < -0.325 table edge
```

R3 recovery 使用以 target frame 为中心的对称 `±0.4 m` proxy，因此错误地把悬出真实桌边的
Can 标为 recoverable，随后开爪/retreat，物体坠落并触发
`public_object_out_of_workspace`。

完成标准：基线 artifact 通过 semantic verifier，phase timeline 和 worktable-local轨迹可从
raw policy input 独立重算；确认问题不是 physics、state corruption 或证据错误。

## 2. 先建立四个红灯

### 2.1 Grasp target 不得追逐被接触后的 Can

构造 TaskExecutive 进入 DESCEND 后的状态，保持 target/worktable frame 不变，只移动 public
Can pose。当前 `_phase_goal()` 使用实时 `can + offset`，EEF grasp goal 会随 Can 移动。

红灯要求：

- APPROACH 可跟踪尚未接触的稳定 Can；
- 进入 ALIGN/DESCEND 时冻结 Can 的 worktable/target-local grasp anchor；
- DESCEND、GRASP、PRELIFT_VERIFY 使用该 anchor；
- Can 被指尖推动后，EEF target 不继续同向追赶；
- worktable frame 随振动移动时，冻结 anchor 仍正确变换到当前 robot-base frame。

### 2.2 Collision-free approach path

当前 EEF 从初始位置直接斜向移动到 Can 上方，finger pads 可能在完成水平对齐前扫到 Can。
构造 finger-pad/tool clearance test，要求执行链为：

```text
CLEARANCE_LIFT
→ LATERAL_ALIGN_ABOVE_CAN
→ ALIGN_SETTLE
→ VERTICAL_DESCEND
```

在 `LATERAL_ALIGN_ABOVE_CAN` 完成前，所有 finger-pad collision support points 相对 Can top 和
open worktable 保持预注册 clearance。不能只检查 EEF origin。

### 2.3 真实 worktable-local bounds

使用公开 static context 构造 worktable frame、半尺寸 `[0.325,0.30]`、target origin offset 和
Can collision radius。以下必须区分：

- Can collision envelope 完整在桌面内；
- Can center 在桌面内但 collision envelope 已悬边；
- target-local 坐标绝对值小于 `0.4`，但换算到 worktable 后已越界；
- target frame 随 worktable 旋转/平移后的相同物理状态。

第二、第三种不得标为 recoverable。删除 `recovery_table_half_extent_m=0.40` 这种以 target 为
中心的正方形 proxy。

### 2.4 Recovery 不得在未确认支撑时开爪

从 PRELIFT_VERIFY/slip failure 构造：Can 可能仍被夹住、可能落桌、可能悬边。当前 held=false
会直接进入 RECOVERY_OPEN。

红灯要求 recovery 首先保持 close 并反向执行最近一次预抬/抬升轨迹；只有连续 public
samples 同时证明 Can 接近真实桌面、速度稳定、collision envelope 在 bounds 内后才能开爪。
悬边、桌下、快速下落和不可达状态必须 fail closed，不能进入普通 APPROACH。

完成标准：四组测试在当前实现上失败，且每组分别锁定 target chasing、swept collision、
frame/bounds 和 unsafe release，不用放宽成功条件制造红灯。

## 3. 引入公开静态 WorktableTaskContext

把设计树已经允许所有 tier 公开的静态任务几何变成一个 typed、只读 context，至少包含：

```text
worktable_size_xy_m = [0.65, 0.60]
worktable_half_extents_xy_m
target_frame_origin_in_worktable_m
table_surface_z_in_worktable_m
Can collision radius / lower support / upper support
finger-pad/tool support offsets
support topology ID
```

这些值从环境/compiled geometry audit 和 package-owned task contract加载并交给所有 V0–V3；
不能从 state ID、当前 outcome 或 privileged contact读取。构建时断言 context 与环境 compiled
geometry 一致，并纳入 controller/task context hash。

提供明确转换：

```text
robot-base point ↔ current target frame ↔ worktable frame
```

target 与 worktable 属于同一 rigid assembly，但 origin 不同。所有 table-edge、source recovery
和 table-height 判断必须在 worktable frame 中完成；target containment 仍在 target frame。

完成标准：2.3 红灯变绿；context 在 V0–V3 完全相同；没有新增 runtime vibration privilege。

## 4. 冻结 grasp anchor，停止接触后追逐

将 acquisition 细化为：

```text
CLEARANCE_LIFT
→ LATERAL_ALIGN_ABOVE_CAN
→ ALIGN_SETTLE
→ VERTICAL_DESCEND
→ GRASP_CLOSE
→ PRELIFT_VERIFY
```

在 `ALIGN_SETTLE` 中要求 public Can pose/speed 连续若干 samples 稳定且 collision envelope 有
足够 table-edge margin。满足后保存：

```text
T_worktable_can_anchor
T_worktable_eef_grasp_goal
tool / finger-pad clearance certificate
anchor timestamp
```

后续 grasp goal 每步通过当前 public worktable/target frame 转到 robot base，但其 worktable-local
x/y 和 grasp orientation 固定。不能继续使用实时 Can pose更新 DESCEND/GRASP target。

如果在闭合期间 Can 相对 anchor 移动超过基于 Can radius 的小阈值：停止闭合推进，保持安全
高度或沿进入路径撤回；待 Can 稳定后重新采样 anchor。禁止沿 Can 被推动方向继续追逐。

所有 waypoint clearance 使用完整 finger-pad/tool几何，确保水平运动发生在 Can top 与浅箱壁
之上；vertical descend 才降低高度。

完成标准：2.1/2.2 红灯变绿；state 002 第一次与第二次尝试中的 Can worktable-local drift
显著低于旧轨迹，且在 PRELIFT 前始终保留 `Can radius + safety margin` 的桌边余量。

## 5. 保留严格 grasp，并使 PRELIFT 只验证随动

保留 R3 已修正的：

- `open=-1`、`close=+1`；
- wrist wrench 无独立 hold authority；
- full Can-in-EEF SE(3)；
- sub-Can-radius slip threshold；
- PRELIFT_VERIFY。

不要为了让 state 002 通过而简单降低 aperture/corridor/expected-transform要求。先修轨迹对齐；
只有独立 geometry/noise negative controls证明误拒绝时，才允许统一调整阈值，并记录所有候选
profile 和选择依据。

PRELIFT 从冻结 grasp anchor 出发、保持 close，只做小幅 table-normal lift。进入正式 LIFT
要求：Can 相对 worktable 上升、Can-in-EEF 稳定、tilt 稳定、aperture/corridor 持续有效且
table-edge margin 仍安全。未通过时进入第 6 节的 reverse-path recovery。

完成标准：closed-empty、single-side push、Can-stays-on-table 均被拒绝；真实 rigid follow 被
接受；grasp validator 不因放宽到无物理意义的范围而通过。

## 6. 实现 reverse-path、edge-safe recovery

每次进入 DESCEND/PRELIFT/LIFT 时保存可逆路径状态。发生 grasp miss/slip 时：

```text
RECOVERY_HOLD_CLOSE
→ REVERSE_TO_PRELIFT_START
→ LOWER_TO_VERIFIED_TABLE_HEIGHT（如 Can 随 EEF）
→ WAIT_PUBLIC_SETTLE
→ OPEN_IF_FULLY_SUPPORTED_AND_IN_BOUNDS
→ CLEARANCE_RETREAT
→ RE-ALIGN
```

public “可安全开爪”证据至少要求：

```text
Can collision envelope inside actual worktable bounds with margin
Can lower support near actual table surface
public linear/angular speed below recovery limits for N samples
EEF/finger pads not driving Can toward an edge
Can not below table / not ballistic / inside reachable workspace
```

公共状态不能证明支撑时，保持 closed 并执行保守 lowering；若仍无法建立安全状态，明确失败。
不要把 held=false 等同于“已经安全落桌”，也不要在 recovery 后立即追踪正在移动的 Can。

对 near-edge case：若 Can envelope 已悬边，禁止开爪和普通重抓；如果共享 controller 没有公开、
可验证的向内恢复 primitive，则以 `public_object_edge_unrecoverable` 失败。nominal 10/10 应通过
前面的 collision-free anchor避免进入该状态，而不是扩大 table bounds。

完成标准：2.4 红灯变绿；R3 trace 在 step 106/112 对应状态不会被错误标为可安全开爪；
recovery transitions 与实际动作相符。

## 7. Tight feedback loop

每次修改后只先运行 state 002：

```bash
python -m robosuite.scripts.shakebench_run_oracle \
  --tier V0 --gamma 0.0 \
  --state-id shakebench-dev-v0-002 \
  --horizon-steps 1200 \
  --output out/phase07r4/dev002_candidate.json
```

输出必须通过 semantic verifier，并生成 before/after timeline：

```text
phase
Can/EEF/fingertip worktable-local pose
frozen anchor and error
gripper action/ctrl/force/aperture
prelift follow
table-edge margin
recovery preconditions/transitions
success/failure
```

state 002 的完成条件是 environment evaluator success，不接受“没有掉下去但 horizon exhausted”。
通过后运行另外 9 个 dev states。任何新失败按 phase taxonomy 修共享逻辑，不按 state ID 分支，
不修改 state、seed、horizon、physics、task geometry 或 success threshold。

最终 controller profile 选择顺序：

1. V0/Gamma=0 10/10；
2. 较少 grasp/recovery attempts；
3. 较大最小 table-edge margin；
4. 较少 action saturation；
5. profile hash 字典序 tie-break。

## 8. 生成 R5 entry evidence

controller phase/profile 改变后，R2/R3 rollout 不再证明当前 R4 core。保留旧 hashes，并用
R4 profile 生成足以建立 R5 baseline 的一次性入口证据：

```text
V0 Gamma=0                  10 dev states
matched Gamma=0             V0/V1/V2/V3 × dev-000
positive-Gamma diagnostics  V0/V1/V2/V3 × {0.15} × first 3 dev states
determinism                  V0/Gamma=0/dev-000 × 3 fresh processes
actuator                     7 channels × positive/negative
package                      clean wheel + clean sdist isolated installs
```

每个 run 使用 R2 semantic verifier重算；三进程、actuator 和 package 使用已有专用 verifier。
完整 raw 写入 ignored `out/phase07r4/`；Git 只保存 compact manifest/summary、必要 fixtures 和
raw path/size/hash。保留全部 diagnostic failures，不要求 tier 单调排序。Gamma=0.30 和所有
final-profile evidence 只在 R5 语义修改后运行。

## 9. R4 core completion 与 Phase 07R5 handoff

至少运行：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
python -m pytest -q tests/test_shakebench_phase07_completion_red.py
python -m pytest -q tests/test_shakebench_oracle.py tests/test_shakebench_actuators.py \
  tests/test_shakebench_metrics.py tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
python -m isort --check-only <全部修改 Python 文件>
python -m black --check -W 1 <全部修改 Python 文件>
git diff --check
```

再运行完整可用的 non-renderer regression。以下全部满足时，将 R4 记录为
`COMPLETE_FOR_R5_ENTRY`：

1. grasp anchor 不追逐 contact-induced Can motion；
2. clearance/lateral-align/vertical-descend 路径通过 swept-geometry tests；
3. recovery 使用 actual asymmetric worktable bounds 和 Can envelope；
4. PRELIFT/recovery 在确认安全落桌前保持 gripper closed；
5. state 002 成功，全部 10 个 V0/Gamma=0 dev states 为 10/10；
6. R2 的 V1、V2/V3、target frame、success、semantic、actuator、determinism 回归保持通过；
7. matched Gamma=0、Gamma=0.15 smoke、actuator、determinism 和 package entry evidence 可独立重算；
8. physics/contact/task geometry/success thresholds/dev states 字节未改变；
9. R4 controller/profile、task context、report、compact manifest 和 deferred-to-R5 清单完整。

更新 `docs/phase_07_report.md` 顶部 R4 section，给出 state 002 before/after worktable-local
timeline、10/10 summary、R4 profile 与 entry-evidence hashes；旧 R2/R3 过程压缩为 provenance
索引。若第 1–8 项失败则继续 R4 BLOCKED。若它们通过且只剩 Gamma=0.30/final regeneration/
final commit，则保留 `BLOCKED_BY_PHASE_07R4_EVIDENCE_REGENERATION` 作为“不可进 Phase 8”状态，
同时明确 `R5_entry=true`，随后执行 `phase_07r5_oracle_motion_semantics_closure.md`。
