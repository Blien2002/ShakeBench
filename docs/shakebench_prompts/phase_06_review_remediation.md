# Phase 06R Prompt：重新预注册并修复 Official Physics Freeze

你在 `/home/miracle04/Desktop/ShakeBench` 工作。当前 Phase 06 结论无效，必须先将其视为 BLOCKED；不得进入 Phase 07，不得修改 controller，不得使用 task success、V0–V3 ranking、policy observation 或任何 task SR。

当前已知审查缺陷：

1. contact selection 写死 canonical 结果，未 probe 非 canonical candidates；
2. dt convergence 只要求任一 dt pass，但 protocol 要求 3 个 convergence passes；
3. driver 的 amplitude/phase 未覆盖每个 safe Gamma × empty/representative load；
4. isolator 未对全部候选运行 MuJoCo transfer / combined spectrum，且 scoring target/margin 排序不符合预注册；
5. Gamma=0 bounded parity 没有实际 artifact；
6. determinism 只覆盖 deck trace，没有覆盖 driver/isolator/contact/parity 关键 probes；
7. scoreable profile 可由外部路径绕过 packaged official profile；
8. selection 在开始时读取预先存在的 official profile，形成循环选择；
9. Phase 06 不应改变已冻结的 Phase 05 observation/IMU semantics；
10. verifier 没有从 raw artifacts 重新计算 hard gates。

先完整阅读：

- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/robosuite_benchmark_design_tree.md` 的 physics gates、transfer-envelope、failure integrity
- 当前 `robosuite/models/assets/shakebench_selection_protocol.yaml`
- 当前 `robosuite/scripts/shakebench_select_physics.py`
- 当前 `robosuite/scripts/shakebench_replay_physics.py`
- 当前 `docs/phase_06_report.md`

## 0. 作废与重新预注册

不要删除现有 raw evidence。保留为 archive/provenance，但明确标记为 `invalidated_phase06r`，不得再被 environment 当作 scoreable official physics 使用。

在现有 `robosuite/models/assets/` 根目录新建：

- `shakebench_selection_protocol_v2.yaml`

该文件必须在任何 Phase 06R probe 前创建并单独提交。提交后不得修改。

V2 protocol 必须包含：

- 一个明确、有限、可实际跑完的 composed candidate table；不要只给可组合 grid 后在代码中私自挑 4 个；
- 每个 candidate 的完整 dt / integrator / solver / iterations / deck / isolator / contact tuple；
- 明确数值化 target、hard exclusions、scoring、secondary ranking、deterministic tie-break；
- contact 的真实 physics scoring，而非“距离 canonical”的硬编码选择；
- 每个 probe 的 sample window、settle duration、trace tolerance、crash/retry ledger；
- 3 个能够通过硬 gate 的 dt convergence candidates；若原 coarse 是负例，可保留为明确 negative control，但不能冒充第三个 convergence pass；
- protocol hash、commit hash 和 immutable-after-run 声明。

## 1. 消除 official profile 循环依赖

在 V2 selection 期间，禁止加载或使用现有 official profile 作为 candidate/probe 输入。

候选 fixture 只能来自：

- V2 protocol；
- 已冻结的 upstream physical facts（32 kg table、惯量、Can 质量、已冻结 sliding μ）；
- Phase 02–05 的只读 provenance。

只有 V2 selection 全部 hard gates 通过后，才生成新的 canonical：

- `robosuite/models/assets/shakebench_official_physics.yaml`

旧 profile 必须保存为 flat archived artifact，且 loader 在 V2 完成前 fail-closed。

scoreable environment 只能接受 packaged canonical official profile；禁止通过任意外部 YAML path 或自重算 hash 的 profile 获得 `scoreable=True`。外部 path / mapping 只能是明确的 `non_scoreable` probe/training profile。

所有 score-affecting environment inputs 都必须锁到 profile，包括至少：

- timestep / integrator / solver / iterations / tolerance；
- deck inertial / weld / scheduler；
- isolator；
- table/target/contact friction 与所有 pair parameters；
- 任何会改变 scoreable contact 或 stepping semantics 的 physics option。

## 2. Driver / timestep / weld

对每个可选 dt candidate、每个 safe Gamma、每个 load case：

```text
Gamma ∈ pre-registered safe set
load ∈ {empty, representative}
axis / line ∈ 完整六轴 64-line authored spectrum
```

直接测量并保存：

- line amplitude error；
- phase error；
- Gamma deck error；
- raw weld residual / force；
- warnings；
- solver diagnostics；
- complete trace hash。

不得只在 empty load 跑 spectrum、只在 representative load 跑 Gamma 标量。

必须有 3 个 dt complete hard-gate passes，再按 V2 protocol 的明确 cost/tie-break 选择唯一 dt。

solver/integrator/iterations sensitivity candidates 也必须在 protocol 中预注册，而不是 selection 后临时构造。

## 3. Isolator

每个 isolator candidate 都必须在选择前完成：

- six-axis analytic transfer；
- six-axis MuJoCo transfer，至少 tracking / resonance / isolation 三段；
- combined authored six-axis spectrum；
- `T_accel`、`R_relative`、P99 `D_relative`、`T_peak`、travel margin、static sag、preload offset、payload sensitivity；
- empty 与 payload equilibrium；
- timestep / solver stability；
- warnings / safety gates。

不得先只用 analytic 选择，再只对 selected candidate 跑 MuJoCo。

V2 scoring 的所有 target center、normalization、权重和排序方向必须直接来自 protocol。特别是：

```text
minimum_normalized_margin 必须 descending
```

不要在 Python 中临时推导未预注册 target。

## 4. Contact

对 V2 protocol 中的每一个 non-preflight-excluded contact candidate，运行真实 MuJoCo physics probes：

- static support；
- 实际 MuJoCo incline threshold，并与 Coulomb analytic threshold 比较；
- single-axis slip threshold / deceleration，与 analytic 误差比较；
- impact 后至少 `0.50 s` recovery/settle；
- finger load，使用预注册 force/penetration/warning criterion；
- 3 dt contact trace convergence；
- explicit pair scope audit；
- condim、torsional/rolling、margin/gap、solref/solimp、iterations 的 compiled equality audit。

不得调用 `env.step()`、reward、`_check_success()` 或 success evaluator；contact probes 必须直接从 MuJoCo raw state / contact / force / trace 读取。

禁止因 candidate “不是 canonical”就在 probe 前排除。选择必须来自 V2 protocol 的 hard gates 与 scoring。

## 5. Gamma=0 bounded parity

新增真实、machine-readable parity artifact 与 test。

比较 dynamic Gamma=0 scene 与 stock static scene 的：

- task geometry；
- action shape / semantics；
- passive support；
- nominal task poses；
- Panda reachability / solvability prerequisites；
- pre-registered dynamic residual。

不要要求 trajectory equality；不要只在 report 中声明已经比较。

## 6. Determinism 与 failure integrity

`shakebench_replay_physics.py` 必须在至少 3 个独立进程中分别覆盖：

- driver/timestep/weld；
- selected isolator transfer + combined spectrum；
- selected contact probe；
- Gamma=0 bounded parity。

每项比较完整 trace，而非只有 summary；可以使用 digest，但 digest 必须覆盖所有原始字段并记录 schema/shapes/dtypes。

实现 protocol 的 crash policy：

- infrastructure crash 只能重跑同一 state/config；
- 记录 retry；
- 同 state 重复 infrastructure crash => group incomplete / selection BLOCKED；
- 不得把 crash 简单当成 candidate exclusion 后继续 PASS。

## 7. Integrity、tests、scope

selection verifier 必须从 raw artifacts 重新计算：

- hard gates；
- candidate eligibility；
- scoring；
- tie-break；
- profile alignment；
- raw file hashes；
- protocol hash；
- selected/excluded table consistency。

不能只验证 artifact 自己声明的 `passed=true` 和自算 payload hash。

恢复 `robosuite/utils/shakebench_sensors.py` 中仅为 Phase 06 引入、会改变 Phase 05 observation semantics 的改动；如确有必要修改，必须证明 Phase 05 profile/hash/schema/trace 完全不变，并新增相应 regression test。

保留并扩展：

- `tests/test_shakebench_physics_profile.py`
- 现有 driver / isolator / environment / provider tests

新增明确的 negative tests：

- external scoreable profile path 被拒绝；
- 无 3 个 dt convergence passes => BLOCKED；
- 未 probe 的 contact candidate => verifier 失败；
- inverted margin tie-break => verifier 失败；
- representative load 缺 spectrum => verifier 失败；
- 缺 parity artifact / 缺关键 probe 3-process trace => verifier 失败；
- raw artifact 变更即 verifier 失败。

## 8. 最终产出与停止条件

保持所有新增文件 flat，不新建目录。

输出：

- 新 protocol V2；
- 新 official profile 与 hash；
- V2 raw artifacts；
- selected/excluded tables；
- parity artifact；
- complete multi-probe determinism artifact；
- 更新后的 `docs/phase_06_report.md`；
- 明确的 V2 protocol commit。

只有同时满足以下条件时才可写 PASS：

1. 一个且仅一个 official profile 通过；
2. 3 个 dt convergence candidates 通过完整 hard gates；
3. 每个 selected component 的所有 required probes 已完成；
4. 每个 contact candidate 已按 protocol 完成或被合法 preflight exclusion；
5. selection 不读取 task SR / success / controller outcome；
6. parity 与关键 probe determinism 均通过；
7. verifier 可从 raw evidence 重算结果；
8. wheel 安装后 profile/hash 一致；
9. 所有可运行 physics/env tests 无回归；
10. renderer/EGL 限制单独报告，不能伪称 full suite PASS。

任何一项失败：写 BLOCKED，保留 evidence，停止，不进入 Phase 07。
