# Phase 09 Prompt：Gamma Knee 与 400×4 Official Measurement

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。本阶段只运行冻结 measurement，不开发功能。

## 前置硬门

- Phase 00–08 完成；
- Phase 7.5A 的 `scene_ready`、`clearance_passed`、`demo_verified` 和选定
  `scene_visual`/`geometry_variant` 已冻结；每个 run 必须绑定该 authority。
- git working tree clean；
- dependency/package/physics/controller/task/state/config hashes 冻结；
- 3-process determinism 通过；
- 100 knee 与 400 official states 完整无交集；
- scorecard synthetic tests 通过；
- scoreable config 无 `UNFROZEN` 字段。

任一失败即停止。

Phase 9 只接受本阶段选定的 scene/geometry profile。visual-only scene hash
可以随 run metadata 认证，但不得将不同 geometry profile 的结果放在同一
scorecard 中；旧 R6/R6.1 release evidence 仍是历史基线，不是本轮测量输入。

## Measurement纪律

- raw outputs 写入仓库根 gitignored `out/`（若 .gitignore 未包含 `out/` 则先添加该条目），不复用旧 episode；
- 每 episode 前写 manifest；
- failure 保留；infrastructure crash 只重跑同 state 并记录；
- 本阶段不修改代码、threshold、controller、physics、states；
- 发现 bug 后整轮作废，回责任 phase 修复，以新 commit/run ID 重跑。

## A. Knee calibration

只运行 State-V0 和 100 knee episodes。

1. Gamma=0 要求 SR>=0.95；失败停止。
2. 在 Phase 06 safe `Gamma_commanded` 范围 coarse scan，再按预注册规则 fine scan。
3. 每 Gamma 复用同 100 episodes。
4. 对 `SR_V0(Gamma_commanded)` 做 non-increasing isotonic fit。
5. `Gamma_star` 为首次达到 absolute SR=0.5 的插值点，按 0.05 舍入。
6. safe 范围无 crossing 则 `no_valid_decision_point` 并停止。
7. freeze artifact 保存 raw grid/fit/rounding/hashes 和 deck actual conformance。

`Gamma_deck_actual` 必须满足 1%；`Gamma_table_actual` 只作为 response。

## B. Official evaluation

同一 Gamma_star 运行 State-V0/V1/V2/V3 各 400 states，共 1600 episodes。预注册 state-block 内 tier 顺序，避免 tier 与热状态/时间混淆；matched block 复用 object/excitation/sensor seeds。

## C. Aggregate

从 raw episodes 生成 per-tier SR/Wilson/failures/violations、four paired comparisons、ceiling/floor flags、diagnostics、rerun/incomplete ledger 和 response summaries。

唯一主指标是 `SR@Gamma_star`。不增加 AUC、复合分或临时第二 Gamma。

## 完成条件

1. knee 可从 raw 独立重算；
2. 4×400 matrix 完整，否则 group 明确 incomplete 且不排名；
3. paired counts 逐 state 可追溯；
4. 输出 1600 raw results、manifests、ledger、scorecard JSON/CSV/Markdown、plots 和 `docs/phase_09_report.md`；
5. measurement 期间 git diff 为空（仅 `out/` 为 ignored 运行产物）。

完成后停止，不做 release 修补或 Vision 实验。
