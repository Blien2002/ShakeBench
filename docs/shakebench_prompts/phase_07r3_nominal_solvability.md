# Phase 07R3 Prompt：Oracle Nominal Solvability Closure

你在 `/home/miracle04/Desktop/ShakeBench` 中继续 Phase 07。Phase 07R2 已完成 target-frame、
V1 stateful IMU、Can-in-EEF SE(3)、V3 transfer、semantic verifier、actuator 和三进程工具，
但被正确阻塞在 V0 / Gamma=0 nominal solvability。

本轮只修共享 reference controller 的抓取、持有和恢复逻辑，持续工作到 10 个冻结 dev
states 全部成功并重新闭合 Phase 07 evidence。不要创建 state-specific 分支，不修改物理、
任务、成功阈值或 dev states，不进入 Phase 08 implementation。

## 1. 权威输入与当前最小失败

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_07_oracle_controller.md
docs/shakebench_prompts/phase_07r2_final_closure.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_07_report.md
```

保持以下冻结输入不变：

```text
dev-state anchor = dd6fe2edb6384ccdb5116be44f07592b4864e377
official physics = shakebench.official.physics.v2
physics SHA-256  = c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c
```

开始时运行 Phase 6 verifier，并验证 dev asset 与 `dd6fe2ed` byte-identical。将当前报告状态
更新为 `BLOCKED_BY_PHASE_07R3_NOMINAL_SOLVABILITY`，保留 R2 blocker 和 profile hashes。

当前确定性最小复现为：

```bash
python -m robosuite.scripts.shakebench_run_oracle \
  --tier V0 \
  --gamma 0.0 \
  --state-id shakebench-dev-v0-002 \
  --horizon-steps 120 \
  --output out/phase07r3/blocker_before.json
```

当前预期：

```text
success=false
termination_category=controller_failed
failure_reason=public_object_out_of_workspace
trace length=117
```

已知失败链：第一次 grasp 未建立后重试；第二次被 public grasp heuristic 接受；LIFT 后 Can
脱落；translation slip 到 `0.25 m` 才触发恢复；Can 已在桌下高速下落，最终超过 public
workspace radius。

完成标准：上述命令在修改前稳定复现相同失败，且 current blocker 通过 R2 semantic
verifier；这证明问题位于 controller execution，不在 artifact、physics 或 state corruption。

## 2. 先增加三个红灯回归

### 2.1 Phase-to-gripper semantics

Panda 单通道动作的公开约定为：

```text
-1 = open
+1 = closed
```

当前 profile 的 `gripper_hold_action=-1` 与该约定冲突。增加通过真实
`TaskExecutive.command()` / `ShakeBenchOracleController.action()` 的测试：

| Phase | required gripper action |
|---|---:|
| SETTLE / APPROACH / DESCEND | `-1` open |
| GRASP / PRELIFT_VERIFY / LIFT / TRANSPORT / PLACE | `+1` close/hold |
| RELEASE / VERIFY | `-1` open |
| FAILED | `-1` open only after recoverability policy allows safe release |

不要只测试底层 Panda mapper；必须测试 phase-to-action wiring。当前 LIFT/TRANSPORT/PLACE
测试应先失败。

### 2.2 False grasp establishment

构造以下 public-only negative controls：

- gripper aperture 看似被物体阻挡，但 Can 不在两 finger pads 的合理 grasp corridor；
- wrist wrench 较大但 Can 已远离 EEF；
- Can 只被单侧推碰或仍留在桌面，没有随 EEF 小幅预抬；
- `Can-to-EEF > 0.15 m`，但旧 `0.5 m` wrench fallback 仍会报告 held。

这些情况不得进入正式 LIFT。wrist wrench 不得单独或与 `0.5 m` 宽几何门组合成为 hold
authority。

### 2.3 Early slip and recoverability

从合法 Can-in-EEF reference 构造逐步增加的 translation slip。在 Can 位移达到一个 Can
radius 之前，controller 必须已经进入 controlled recovery；当前 `0.25 m` threshold 应先
失败。另构造：

- Can 仍在桌面附近且速度有限：可恢复；
- Can 已低于桌面、正在快速下落或超出可达范围：明确 `public_object_unrecoverable`，不再
  用普通 APPROACH 追赶。

完成标准：三个测试在当前实现上变红，并分别捕获错误 hold sign、false-positive grasp 和
过晚/不安全 recovery；之后才修改 production code。

## 3. 固定 gripper action 语义

用无歧义字段替代 `gripper_hold_action`，例如：

```text
gripper_open_action  = -1.0
gripper_close_action = +1.0
```

若保留现有字段，必须固定为 `+1.0` 并在 profile validation 中断言 Panda action contract。
GRASP 开始后至安全 release/recovery-lower 完成前持续输出 close。不能在 LIFT、TRANSPORT、
PLACE 或悬空 recovery 的第一步输出 open。

运行一个 environment-backed phase trace，记录 normalized gripper action、compiled finger
ctrl、finger actuator force 和 qpos，证明：

```text
GRASP closes
PRELIFT_VERIFY remains closed
LIFT/TRANSPORT/PLACE remain closed
RELEASE opens
```

仅将 `-1` 改为 `+1` 不是本阶段完成条件：此前单变量探针已经表明 sign 修正本身不足以让
state 002 成功，必须继续闭合 grasp establishment。

## 4. 增加 public-only PRELIFT_VERIFY

将抓取阶段细化为：

```text
GRASP
→ PRELIFT_VERIFY
→ LIFT
```

GRASP 结束时先保存完整 public `T_eef_can` reference，但只把它标为 candidate grasp。
PRELIFT_VERIFY 保持 gripper closed，沿当前公开 worktable/target local z 方向执行小幅、低速、
有界预抬；位移及持续时间作为 controller profile 参数冻结并记录单位。

只有同时满足以下 public evidence 才进入正式 LIFT：

1. Can 相对工作台/target frame 的高度随 EEF 增加；
2. `T_eef_can` translation 在基于 Can/gripper geometry 的小容差内稳定；
3. Can tilt 相对 EEF 稳定；轴对称 Can 的纯 yaw 只作诊断；
4. finger aperture 保持在“被物体阻挡”的合理区间；
5. Can 位于双指形成的 grasp corridor，而非单侧或远距离状态；
6. public values finite，Can 未离开桌面可达区域。

PRELIFT_VERIFY 失败时，如果 Can 仍在桌面附近，先把 EEF/Can 降回安全高度再开爪并重试；
若 Can 已不可恢复，则明确失败。不要在未验证随动前直接执行 full lift。

测试至少覆盖 successful bilateral prelift、closed-empty gripper、single-side contact proxy、
Can stays on table、Can follows EEF rigidly、Can begins slipping 六种情况。

完成标准：只有能在多个连续 policy samples 中随 EEF 稳定运动的 Can 才进入 LIFT；
state 002 不再因 false grasp 进入 full lift。

## 5. 用几何尺度重定义 grasp/hold/slip

所有阈值必须从已公开的 Can collision radius/height、finger-pad positions、expected grasp
transform、policy rate 和 measurement noise 推导。删除或替换以下宽门：

```text
grasp_geometry_tolerance_m        = 0.150
grasp_wrench_geometry_tolerance_m = 0.500
grasp_slip_tolerance_m            = 0.250
```

实现建议：

- 在 EEF frame 中计算两 fingertip points 和 Can center/axis；
- 验证 Can 位于两指闭合轴的有效 segment/corridor 内；
- 分离沿 closing axis、垂直 grasp plane 和 tool z 的误差；
- grasp reference 建立后直接比较当前 `T_eef_can`；
- translation slip threshold 应显著小于 Can radius `0.02509 m`，并由噪声 negative control
  给出下界；
- rotation slip 分解为 Can symmetry-axis yaw 与 tilt，yaw 记录但不触发 v0 nominal failure；
- wrist wrench 只能增加有几何支撑的 confidence，不能让远离 gripper 的 Can 被判 held；
- 连续 2–3 个 policy samples 证实 loss/slip，避免单点噪声触发，同时保证在 Can 下落到
  桌面以下前响应。

不要根据 state ID、seed 或固定 XY 写分支。允许使用 10 个 dev states 调试共享阈值，但
报告必须列出每个参与选择的 profile、失败阶段和最终统一选择理由。

完成标准：false-positive tests 通过；rigid carry 不误报；在 blocker trace 的对应尺度上，
loss/slip 至少比原 step 110 提前到 Can 尚在桌面可恢复范围时触发。

## 6. 实现有安全前置条件的 recovery

将当前“任何 recovery 都跳回 APPROACH”改成明确子状态，例如：

```text
RECOVERY_HOLD
→ RECOVERY_LOWER_IF_NEEDED
→ RECOVERY_OPEN
→ RETREAT
→ APPROACH
```

public recoverability 至少检查：

- Can 相对 worktable 的高度和 XY 是否仍在桌面范围；
- public finite-difference vertical speed 是否允许安全接近；
- Can 是否仍在 gripper 中、悬空、已落桌或已低于桌面；
- EEF/Can 是否在 Panda 的公开 workspace envelope 内；
- remaining recovery budget。

悬空但仍夹持时先保持 closed 并安全降低；已稳定落桌时才 open/re-approach；明显自由落体、
桌下或不可达时立即给出具体 failure reason。不能让 EEF 追赶高速下落物体，也不能用增大
workspace radius 掩盖失败。

每次 recovery 记录 trigger、precondition、selected transition、开始/结束时间、Can/EEF
public pose/speed 和 budget。任务 failure 仍进入分母。

完成标准：可恢复的桌面 grasp miss 能重新抓取；悬空 slip 不会立即开爪；不可恢复 case
快速、明确终止；不存在宽松阈值导致 Can 已下落 `0.25 m` 后才开始恢复。

## 7. Tight loop，然后全 dev 验证

每次 controller/profile 修改后先运行：

```bash
python -m robosuite.scripts.shakebench_run_oracle \
  --tier V0 --gamma 0.0 \
  --state-id shakebench-dev-v0-002 \
  --horizon-steps 1200 \
  --output out/phase07r3/dev002_candidate.json
```

必须同时满足：

```text
semantic verifier PASS
environment success=true
failure_reason=null
no physics violation / NaN / workspace escape
grasp established through PRELIFT_VERIFY
release and 0.50 s evaluator success completed
```

state 002 转绿后立即运行全部 10 个冻结 dev states。任何其他 state 失败时保留 trace，按同一
phase/failure taxonomy 修共享逻辑；不得为单个 state 调位置、seed、horizon、physics 或
success threshold。最终必须是同一 controller profile 下 10/10。

建议将 controller profile 选择规则固定为：

1. 先满足 10/10 nominal success；
2. 再最小化 recovery 总次数；
3. 再最小化 action saturation 和 task duration；
4. tie-break 使用 profile hash 字典序。

该规则只选择共享 controller 参数，不改变 benchmark physics 或 task。

## 8. 重新生成受影响的 Phase 07 证据

gripper/grasp/recovery/profile 任一变化都会改变 action trace，因此旧 R2 rollout、actuator、
determinism 和 package evidence 都不是最终 authority。保留其 hashes 后重新生成：

```text
V0 Gamma=0                  10 dev states
matched Gamma=0             V0/V1/V2/V3 × dev-000
positive-Gamma diagnostics  V0/V1/V2/V3 × {0.15,0.30} × first 3 dev states
determinism                  V0/Gamma=0/dev-000 × 3 fresh processes
actuator                     all 7 channels, positive and negative
package                      fresh clean wheel + fresh clean sdist installs
```

所有 artifacts 绑定：

```text
final controller profile/hash
dd6fe2ed dev-state anchor + asset hashes
official physics profile/hash
task/observation schema hashes
source/final commit
```

完整 raw traces 写入已忽略的 `out/phase07r3/`；Git 只保存 compact manifest/summary、必要
adversarial fixtures 和外部 raw 文件的 path/size/SHA-256。所有 unsuccessful diagnostic
episodes 保留非空 failure reason，不要求 positive-Gamma tier 单调排序。

R2 semantic run verifier、determinism verifier、actuator verifier 和 package verifier 必须
重新打开新证据并重算，不接受 stored PASS Boolean。

## 9. 回归、冻结与 Phase 8 handoff

至少运行：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
python -m pytest -q tests/test_shakebench_phase07_completion_red.py
python -m pytest -q tests/test_shakebench_oracle.py tests/test_shakebench_actuators.py \
  tests/test_shakebench_metrics.py tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
python -m isort --check-only <全部修改的 Python 文件>
python -m black --check -W 1 <全部修改的 Python 文件>
git diff --check
```

再运行仓库可用的完整 non-renderer regression，并将 renderer/EGL 限制独立记录。

Phase 07R3 只有以下全部成立才 PASS：

1. phase-to-gripper semantics test 证明 hold=`+1`、release=`-1`；
2. PRELIFT_VERIFY 能拒绝 false grasp 并接受真实 rigid carry；
3. grasp/slip thresholds 有几何与噪声依据，不含 `0.5 m` wrench fallback 或 `0.25 m` late slip；
4. recovery 在开爪/追踪前验证可恢复性；
5. state 002 最小复现转绿；
6. V0/Gamma=0 十个冻结 dev states 为 10/10；
7. V1/V2/V3、target frame、physics-step success 和 V3 transfer 回归保持通过；
8. semantic/actuator/determinism/package evidence 全部由最终 profile 新生成并可独立复算；
9. physics/contact/task geometry/success threshold/dev states 未改变；
10. final controller、report、compact manifests 和 Phase 07→08 handoff 已提交并绑定 final
    commit；
11. 工作树除明确 ignored raw output 外为空。

最终将 `docs/phase_07_report.md` 更新为单一当前结论，旧 PASS/BLOCKED 过程压缩为 provenance
索引，避免正文继续混排失效结果。报告给出 state 002 before/after phase timeline、10/10
summary、最终 profile、所有 verifier 命令和 evidence hashes。

任一条件失败则继续保持 Phase 07 BLOCKED，并给出最小复现。全部通过后才授权 Phase 08；
完成后停止，不实现 Phase 08，不生成 knee/official evaluation results。
