# Phase 08 执行提示词：认证入口、Committed States、CPU Batch 与 Scorecard

在 `/home/miracle04/Desktop/ShakeBench/.phase07_5a_clean` 工作。Phase 08 的交付是协议、runner、
scorecard 和 MDE；**不执行** knee scan、100 个 knee rollout 或 400×4 official rollout。
classic MuJoCo 是唯一 scoreable physics backend。GPU 仅可用于 demo renderer，不能接入科学 runner。
MJWarp/MJX/Warp/CUDA physics 仅允许未来独立的 `scoreable=false` feasibility，不得替代本阶段 raw 或 authority。

## 1. 唯一入口：先认证 Phase 7.5A

第一条可执行命令必须为：

```sh
python -m robosuite.scripts.shakebench_verify_phase07_5a_handoff \
  --manifest docs/phase_07_5a_requalification_manifest.json
```

它会验证稳定的 package-owned science authority
`robosuite/models/assets/shakebench_phase07_5a_requalification.json`、独立绑定最终 manifest 的
`robosuite/models/assets/shakebench_phase07_5a_final_authority.json`、raw artifact hash、语义 verifier 和
source inventory。两层 identity 禁止 manifest/authority 哈希循环，也禁止 finalization 改写既有 raw。
随后在非源码目录做 wheel/sdist clean-install
smoke，确认 authority、direct scene 和纹理均来自安装根。不得手抄或缓存 geometry hash；应从
`verify_direct_mount_authority()` 的返回值记录实际值。

认证失败时输出 `BLOCKED_BY_PHASE07_5A_HANDOFF` 并停止创建 official/knee state authority，停止
scoreable batch 和 state authority 写入。不要用候选 evidence、旧 geometry JSON 的 `scoreable`
字段、旧 scene hash 或历史 report 替代该 verifier。

## 2. 冻结不变量

- 十个 `shakebench_states_dev.json` dev state 原样复用；不重排、不替换、不按结果筛选。
- 固定 physics timestep 0.0002 s、20 Hz policy、每 policy step 250 internal steps、既有 controller、
  task geometry、成功阈值、provider 语义和 failure denominator。
- 不把 renderer pixel、privileged state 或 future realized outcome 送入 State policy。
- 任何 scene/geometry/controller/physics/dev-state mismatch、非有限数、unknown schema 或 hash mismatch
  必须 fail closed。

## 3. 当前 Oracle artifact 合同

新 run/episode 使用 Phase-07 schema v5。顶层必须有：

```text
scene_visual, geometry_profile, geometry_authority, scoreable,
controller_profile, physics_authority, dev_state_anchor
```

每个 episode 必须重复 scene/geometry/authority/scoreable，另有 task_context + hash、actuator metadata、
完整 trace + hash、final metrics、success/failure/termination。writer 和 verifier 共用
`scene_visual_identity()` 与 `geometry_authority_identity()`。不得接受 v4 作为新的 Phase 8 authority。
determinism projection 必须比较 task/tier/Gamma、scene/geometry/runtime authority、controller/physics、
task context、actuator metadata、完整 trace、final metrics 和 termination；仅剥离 PID、worker index、
UUID、wall clock、临时路径等 execution provenance。

## 4. Committed states（只生成，不运行）

生成并验证 400 official task states 与 100 knee-calibration states 的冻结 artifact；它们和已有 10 dev
state 的 ID/payload 必须无交集。每条记录固定 schema/version、state ID、split、Can worktable-frame pose、
target reference、excitation/IMU seed、Gamma、task/physics/controller/geometry/runtime hashes 与 canonical
payload hash。禁止把动作、未来 contact、episode outcome 或 wall-clock 信息写入 state。

## 5. CPU batch 合同

使用 `spawn`，每 worker 独立 env/controller/RNG，并在数值库 import 前把 OMP/MKL/OpenBLAS/NumExpr
限制为每 worker 一线程。正式 runner 固定为 2 workers，只用 P-core 首 sibling：

```text
2=[0,2]
```

每 job 是不可变 science identity（state/tier/Gamma/horizon 与所有 authority hash）；PID、CPU、attempt、
hostname、路径、完成顺序和 wall time 只属于 execution provenance。dispatch 前原子落盘 manifest；每 job
原子 publish，resume/retry 只接收同 job identity，task failure 不重跑，infra failure 记录 ledger 后才可重试。

Phase 7.5B 已实测 1/2/4/8。4 workers 出现 swap 增长，8 workers 两次增长到约 1.6--2.0 GiB，均为
`resource_limited`；不得自动重试或用于 Phase 8/9。正式 2-worker batch 在禁用 swap 的受控窗口通过，
使用 `robosuite.scripts.shakebench_cpu_batch` 的 spawn、atomic publish、resume/retry ledger 和稳定聚合。

环境复用仅在 fresh→fresh、fresh→reuse、reuse→reuse（含顺序反转）全部逐字段 parity 和三进程
determinism 后启用。`hard_reset=False` 已通过一次 full-horizon parity，但仍是待验证候选，当前未保留；
只有 reset、observation、action、actuator、trace、metrics、success window 和 termination 完全一致才可启用。

只运行 2-worker Phase 8 protocol validation workload，不能替代或偷偷执行 Phase 9 rollout。
记录 RSS、available RAM、swap、frequency、temperature、P50/P95、jobs/s、sim-seconds/wall-second、
paired duration 和 scaling efficiency。出现 swap/page fault、OOM 或 thermal throttling 时把该档标记
`resource_limited`；默认和上限均为稳定的 2 workers。4/8-worker 已有负证据，不得作为默认值。

## 6. Scorecard 与停止条件

实现 EpisodeResult、RunManifest、paired single-Gamma scorecard 与 MDE 的纯合成/contract 测试。保留
raw result、verdict、retry/duplicate conflict、resource telemetry 和 semantic verifier。不得运行 knee scan
或正式 rollout；Phase 08 完成后等待用户明确启动 Phase 9。
