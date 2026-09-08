# Phase 7.5C 执行提示词：本地科学闭环、CPU 吞吐优化与 Phase 8 最终交接

你在以下工作副本中执行：

```text
/home/miracle04/Desktop/ShakeBench/.phase07_5a_clean
```

本任务只处理本地代码、测试、证据和提示词。不要提交、推送、发布 release、创建 PR，
也不要运行 Phase 8/9 的 knee scan、100 knee rollouts、400×4 official rollouts。

目标按顺序分为三段：

1. 修复 `direct_mount_v1` 当前 Phase 7 requalification 的本地硬门缺口；
2. 在科学等价条件下完成可保留的 CPU 加速和 batch 基础；
3. 用最终 authority 更新 `phase_08_protocol_scorecard.md`，使下一智能体可以直接执行。

任一科学或授权硬门失败时，输出具体 `BLOCKED_BY_*` 状态并停止后续 state authority 工作。
诊断失败保留原始证据，不更换 dev state、物理参数、任务几何、成功阈值或控制器语义。

## 1. 必读材料与当前事实

完整读取：

```text
docs/phase_07_5a_requalification_report.md
docs/phase_07_5a_requalification_manifest.json
docs/phase_07_5a_direct_mount_report.md
docs/phase_07_5a_direct_mount_manifest.json
docs/phase_07_5a_scene_finish_report.md
docs/phase_07_5a_storage_layout_report.md
docs/shakebench_prompts/phase_07_5a_scene_restoration_and_clearance.md
docs/shakebench_prompts/phase_07_5b_cpu_rollout_throughput.md（位于旧 workspace 时读取其绝对路径）
docs/shakebench_prompts/phase_08_protocol_scorecard.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/agents/issue-tracker.md
CONTRIBUTING.md
```

重点代码：

```text
robosuite/models/assets/shakebench_geometry_direct_mount_v1.json
robosuite/models/assets/shakebench_scene_direct_mount_v1.json
robosuite/utils/shakebench_geometry.py
robosuite/utils/shakebench_scene.py
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/scripts/shakebench_scene_preflight.py
robosuite/scripts/shakebench_run_oracle.py
robosuite/demos/demo_shakebench_oracle_video.py
tests/test_shakebench_direct_mount.py
tests/test_shakebench_scene.py
tests/test_shakebench_scene_finishes.py
tests/test_shakebench_oracle.py
```

已知科学基线：

```text
Phase 7R6.1 baseline commit:
4238e37cc3b57d626a37f7950df974fc10b0d1e8

controller SHA-256:
60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c

official physics SHA-256:
c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c

dev-state file SHA-256:
07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6

scene SHA-256:
8a23e7c39b1e182b4e70171eb6631a866000e9163f4ff2db445fa457034a7dde

compiled physics signature:
a1849e266a9ac705dba9abec5be8ece9e1443ad173c063cb24198a62141f0591
```

当前 `direct_mount_v1` 已修复默认 Panda 姿态穿入新桌面的问题。其初始关节姿态必须保持
collision-free，并使末端位于 robot-base frame `[0.45, 0.0, 0.55] m`，朝向与 canonical
Oracle grasp reference 一致。已有行为证据显示 V0/Gamma=0 的 10 个 dev states 为 10/10，
state000 的 V0--V3/Gamma=0.15 smoke 全部成功。

这些结果尚不能直接形成最终 authority，原因如下：

- 完整 10/10、determinism 和 smoke 绑定候选 geometry hash `e763c...`、`scoreable=false`；
- 当前 `scoreable=true` hash `1c7169...` 只有 state000 一条完整成功回合；
- 当前 verifier 会以 `geometry authority` 拒绝候选 smoke；
- `scoreable` 由 geometry JSON 自行声明，没有独立 handoff authority；
- run verifier 没有强制 episode scene/geometry/scoreable 字段；
- determinism projection 缺 final metrics、task context 和 actuator metadata；
- scene preflight 退出码没有纳入 `visual_physics_invariant`；
- 修复后没有新的成功 direct-mount MP4、关键帧和人工视觉验收；
- 最终 wheel/sdist 没有保存在稳定的 requalification evidence 目录；
- 尚未运行完整可用测试集。

## 2. 先冻结本地工作输入

开始时只读记录：

```text
pwd
git rev-parse HEAD
git status --short
Python / MuJoCo / NumPy versions
CPU topology / allowed cpuset / RAM / swap
```

生成一个 sorted source inventory，覆盖本任务所有 science/runtime 代码、profile、scene、state、
tests 和 prompt 文件。每项保存相对路径、字节数和 SHA-256，再计算 inventory payload hash。
该 inventory 是本地 dirty tree 的可复现身份，不要求 Git commit。

完成标准：任何后续实现文件变化都会改变 inventory hash；报告中不得只写旧 HEAD 作为当前源码身份。

## 3. 修复 scoreability authority

### 3.1 分离 geometry 与 authorization

`shakebench_geometry_direct_mount_v1.json` 只描述几何和初始化，不自行授予 scoreability。
优先移除其中的 `scoreable`，或确保 loader 永远不直接采信该字段。

新增 package-owned runtime authority，例如：

```text
robosuite/models/assets/shakebench_phase07_5a_requalification.json
```

至少包含：

```text
schema/version/status
phase08_authorized
authorized geometry profile ID + payload hash
scene hash
compiled physics signature
controller/physics/dev-state hashes
final evidence-manifest hash
payload_sha256
```

在 `shakebench_geometry.py` 或独立深模块中提供一个唯一入口：

```text
verify_direct_mount_authority() -> verified immutable authority
```

验证器必须：

- 先验证 runtime authority schema 与 canonical payload hash；
- 与代码中冻结的 expected authority digest 比较，不能只相信文件自报 hash；
- 验证 geometry、scene、controller、physics、dev-state 和 evidence manifest 绑定；
- mutation、删除、unknown schema、重复字段、非有限数全部 fail closed；
- 只有验证成功才返回 `scoreable=true` / `phase08_authorized=true`；
- geometry loader 只负责几何合法性，不决定 scoreability。

避免循环自引用：runtime authority 不包含冻结其 digest 的代码文件 hash；完整 source inventory
放在 evidence manifest，由报告解释两者关系。

### 3.2 环境 scoreability

`VibrationPickPlaceCan.get_policy_task_context()` 的 scoreable 计算必须同时要求：

```text
official physics is scoreable
AND geometry profile is authorized by verify_direct_mount_authority()
AND actual scene/geometry hashes equal the authority
```

禁止从 geometry JSON 的布尔值直接传播 scoreable。

新增负例测试：

- geometry 数值变化后重算其自哈希，仍不得 scoreable；
- geometry 中加入或切换 `scoreable=true` 不得授权；
- authority status/phase08 flag/hash 任一变化后重算自哈希仍失败；
- scene、physics、controller、dev state 任一 mismatch 失败；
- canonical 历史路径保持原行为。

完成标准：没有经过独立 authority verifier 的 direct-mount 环境永远报告 `scoreable=false` 或拒绝创建 scoreable run。

## 4. 加固 run artifact 与 determinism verifier

若必需字段合同发生变化，升级当前 run/episode/determinism schema；旧 v4 只作为明确 legacy
输入验证，不能用于新 Phase 8 authority。

为当前 schema 强制要求顶层：

```text
scene_visual
geometry_profile
geometry_authority
scoreable
controller_profile
physics_authority
dev_state_anchor
```

每个 episode 强制要求：

```text
scene_visual
geometry_profile or geometry identity
geometry_authority
scoreable
task_context + hash
actuator metadata
trace + hash
final metrics
```

顶层、episode、实际 loader authority 必须三方一致。删除字段、交换 scene/geometry、修改
scoreable、使用候选 hash、重新封装 payload hash均应失败。

把重复的四字段 `scene_visual` 映射提取为单一 helper/value object，使 writer 与 verifier 使用同一
实现。geometry authority 同样只有一个 canonical serialization helper。

扩展 determinism projection，至少比较：

```text
state/tier/Gamma
scene/geometry/runtime authorities
controller/physics/task context
actuator metadata
完整 policy trace
final metrics
success/failure/termination
```

只剥离 PID、进程索引、UUID、墙钟、临时路径等 execution provenance。不得剥离科学字段。

完成标准：新增 deletion/mutation tests 全部通过；三进程中任一 final metric、actuator 或 geometry 字段变化都会使 determinism verifier 失败。

## 5. 修复 scene preflight gate

CLI 成功退出必须至少同时满足：

```text
scene_ready
clearance_passed
visual_physics_invariant
direct-mount initial robot/table active-contact gate
authority hashes match（当请求 scoreable direct_mount 时）
```

增加测试，分别把每个 gate 置 false，断言 exit code 非零。检查初始 `table_collision` 不得与
Panda link5/link6/link7/hand 产生接触；Panda link0/platen 的已登记坐合界面保持原白名单上限。

完成标准：任一 gate 失败时 CLI 返回非零，输出报告仍完整落盘并明确失败原因。

## 6. 重新生成最终 Phase 7 行为证据

先完成以上代码与测试，再冻结新的最终 geometry/runtime authority hash。所有下列证据必须使用
最终 `scoreable=true` authority 和当前 run schema，不能复用候选 `e763c...` raw：

1. V0、Gamma=0、全部 10 个原始 dev states，要求 10/10；
2. state000、V0、Gamma=0，三个独立 spawn process，完整 determinism PASS；
3. state000、Gamma=0.15、V0--V3 matched smoke，四个 artifact 均语义验证 PASS；
4. final scoreable state000 probe，作为 runtime authority smoke；
5. scene preflight final artifact。

使用 `--resume` 和每 episode 原子 checkpoint。任一 dev failure 原样保留并阻塞，不替换 state、
seed、Can pose 或 horizon。positive-Gamma smoke 只证明接口和控制可运行，不用于选择 Gamma 或
声称 tier 排序。

新 evidence manifest 必须：

- 引用最终 raw 的文件 SHA-256、payload/trace hashes 和语义 verifier verdict；
- 有 canonical `payload_sha256`；
- 绑定 runtime authority、source inventory、scene/geometry/physics/controller/dev-state hashes；
- 记录实际命令、软件版本、horizon 和 completion conditions；
- 有独立 `verify_phase07_5a_handoff()` CLI/module；
- mutation、missing file、hash mismatch、candidate geometry、false scoreable 全部失败。

完成标准：在当前本地工作树和独立 package install 中，handoff verifier 都返回 PASS。

## 7. 生成并人工检查成功视频

使用最终 authority 运行新的 direct-mount 视频，不覆盖历史失败录像：

```text
out/demo/phase07_5a_direct_mount_authorized_v1_v0_gamma015.mp4
out/demo/phase07_5a_direct_mount_authorized_v1_v0_gamma015.json
```

要求：

- V0、Gamma=0.15、state000；
- native MuJoCo EGL + FFmpeg；
- sidecar 绑定最终 geometry/runtime authority、scene/controller/physics/state/video hashes；
- evaluator `success=true`；
- 导出 approach、grasp/prelift、transport、place、success-hold 关键帧；
- 实际查看关键帧和视频，确认 Can、指尖、桌面、机器人、platen 无视觉穿透或明显跳变；
- 报告 camera、分辨率、fps、steps、success window 和人工检查结论。

完成标准：新 MP4、sidecar、关键帧 manifest 和人工视觉验收全部存在且通过；历史失败视频继续标记为 provenance。

## 8. 保留可复核 package evidence

最终代码完成后重新构建 wheel/sdist。使用唯一临时 build/install 目录，随后把最终产物和紧凑
package evidence 复制到稳定的本地证据目录：

```text
out/phase07_5a_requalification/package/
```

package evidence 至少保存文件名、字节数、SHA-256、安装根、Python/MuJoCo 版本和以下 PASS：

- 从非源码目录导入安装包；
- geometry/runtime authority 可验证；
- direct scene 与三张纹理来自安装根；
- headless direct-mount environment 创建/reset；
- scoreable=true；
- scene preflight/handoff verifier 可在安装包中运行；
- 不访问 backup、源码 checkout、网络或旧 raw archive。

完成标准：manifest 引用的 wheel/sdist 确实存在于稳定路径，重新计算 hash 与记录一致。

## 9. CPU 加速：先低风险、再热路径

当前实测：

```text
hard_reset=True:  construct + explicit reset = 14.59 s
hard_reset=False: construct + explicit reset = 7.15 s
second soft reset = 0.0076 s

10-step cProfile:
30.57 s total / 79.4M calls
_pair_whitelist: 1.73M calls / 16.57 s cumulative
environment step: 6.74 s
success_snapshot/_success_primitive: about 2.5/2.3 s
```

### 9.1 去除重复 hard reset

让 headless Oracle runner 避免构造后立即进行第二次 hard compile。优先使用
`hard_reset=False` + 显式 deterministic reset，或直接消费初始化后的等价 observation。

用同一 state/tier/Gamma 比较优化前后：reset state、observation keys/values、actions、actuator
ctrl/force、trace、metrics、success window 和 termination 必须完全一致。只有 parity 通过才保留。

### 9.2 摊薄 scene clearance

完整 21-pose scene/clearance audit 对同一 compiled model identity 每 worker 只执行一次。后续
episode 通过 scene/geometry/physics hash、compiled identity 和初始 active-contact 小检查复用已验证
结果。缓存 key 必须覆盖所有会改变几何、物理、scene 或 mount 的输入；key mismatch 重新执行完整
audit，不能跳过。

优先缓存 geom/body role、prefix classification 和 pair whitelist 结果，消除每次百万级字符串
`startswith/any`。缓存内容 immutable，不修改共享 `mjModel`。

### 9.3 success physics-step fast path

在保持每 0.2 ms physics step 检查的前提下，为 success evaluator 提供只计算所需标量的内部
fast path：containment、target-bottom support、finger contact、relative speed、illegal penetration。
缓存 body/geom IDs、contact role membership 和 Can support points；完整 metrics 仍在 policy rate
生成。

优化前注册逐字段比较规则。动作、接触类别、success subcondition、candidate-window reset/latch、
failure 和 termination 必须完全一致。短暂一子步失稳必须仍然重置 0.5 s 连续窗口。

### 9.4 可选后续热点

只有新 profile 仍证明值得时，评估冻结 excitation time grid 的预计算和 deck transform 常量缓存。
保持 left/right-limit sample、mocap refresh、`sim.forward()`、OSC 和 IMU/provider 时序不变。

### 9.5 多进程吞吐

实现或完成 `spawn` batch runner、atomic per-job output、resume、retry ledger、stable aggregation。
本机预注册 P-core 首 sibling：

```text
1=[0]
2=[0,2]
4=[0,2,4,6]
8=[0,2,4,6,8,10,12,14]
```

在数值库 import 前设置 OMP/MKL/OpenBLAS/NumExpr 每 worker 1 thread。正式比较 1/2/4/8 workers，
记录 RSS、available RAM、swap、频率和温度。当前 swap 已满、available RAM 约 12 GiB；出现换页、
OOM 或明显 thermal throttling 时，8-worker 结果标 resource-limited，默认采用稳定的 4 workers。

至少输出：single-episode latency、jobs/s、sim-seconds/wall-second、P50/P95、paired per-job time、
scaling efficiency。profile run 不作为正式 timing。

GPU 只用于 renderer。当前 classic MuJoCo、5 kHz Python callbacks、OSC/provider/evaluator 闭环不接
MJX；不安装或使用 MJX/JAX/CUDA physics。A100/MJX 属独立后端研究阶段。

完成标准：至少保留一项通过 full scientific parity 的端到端优化；如果没有优化通过，保留可靠 batch runner、负结果和瓶颈报告，不降低科学合同。

## 10. 全量测试与格式

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
black --check <所有修改的 Python 文件>
isort --check-only <所有修改的 Python 文件>
git diff --check
```

renderer-only 测试若因当前宿主缺少显示后端而 skip，逐项记录；真实失败不能改成 skip。
新增测试必须覆盖 authority mutation/deletion、preflight 每个 gate、final-schema run verifier、完整
determinism projection、soft-reset/reuse contamination 和 package isolation。

完成标准：完整可用测试集通过；focused tests 不能替代全量测试结论。

## 11. 更新 Phase 8 提示词

最后修改 `docs/shakebench_prompts/phase_08_protocol_scorecard.md`：

- 删除会过期的 geometry hash 缓存，改为读取并执行 Phase 7.5A handoff verifier；
- 第一个可执行步骤必须认证 runtime authority、evidence manifest、最终 raw 和 package evidence；
- 认证失败时禁止生成 official/knee state authority；
- 明确 10 个 dev states 原样复用；
- 写入最终 run/episode schema 与必须字段；
- 写入已保留 CPU 优化、环境复用条件、batch runner、1/2/4/8 worker 和资源保护规则；
- 保持 Phase 8 只生成 400 official + 100 knee state artifacts、实现 runner/scorecard/MDE，
  不执行 knee scan 或正式 rollouts；
- 保持 classic MuJoCo 为唯一 scoreable backend，GPU renderer 与科学 runner 隔离。

Phase 8 prompt 不应再声称“候选证据已完成授权”；它应引用最终 verifier 的 PASS 结果和最终
artifact paths。提示词中若必须展示 hash，从 verified authority 自动读取或要求执行者记录，避免
复制当前 `a70750...`、`e763c...`、`1c7169...` 等将来可能再次变化的缓存。

完成标准：一个新智能体只读 Phase 8 prompt 就能执行正确的本地入口验证、协议实现和 CPU batch，不会误用候选 evidence。

## 12. 最终交付与 PASS 条件

输出并更新：

```text
robosuite/models/assets/shakebench_phase07_5a_requalification.json
docs/phase_07_5a_requalification_manifest.json
docs/phase_07_5a_requalification_report.md
docs/phase_07_5b_cpu_throughput_report.md
docs/phase_07_5b_manifest.json
docs/shakebench_prompts/phase_08_protocol_scorecard.md
```

最终 Phase 7.5C 只有在以下全部成立时为 PASS：

1. direct-mount scoreability 只能由独立 authority verifier 授予；
2. preflight 所有 gate fail closed；
3. final authority 下 10/10 dev、三进程完整 determinism、V0--V3 smoke 均通过；
4. final run artifacts 删除或篡改 scene/geometry/scoreable 时失败；
5. 成功 direct-mount MP4、sidecar、关键帧和人工验收通过；
6. wheel/sdist 存在于稳定证据目录且 clean install 验证通过；
7. 至少一项 CPU 优化通过端到端科学等价，或给出完整负结果；
8. 全量可用测试、Black、isort、diff check 通过；
9. 新 Phase 8 prompt 引用最终 verifier，不含过期 authority 假设；
10. 未运行 knee scan、knee rollouts、official rollouts，也未修改或替换 dev states。

最终报告必须把“科学行为通过”“授权闭环通过”“CPU 性能通过”“Phase 8 entry authorized”分开
列出，不能用其中一项代替其他项。全部通过后停止，等待用户启动 Phase 8。
