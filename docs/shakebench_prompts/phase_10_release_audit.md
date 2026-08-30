# Phase 10 Prompt：ShakeBench Fork 的 Clean-room Release Audit

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。只做发布审计、clean-room reproduction 和证据打包；发现问题回责任 phase，不在本阶段修功能。

## 前置与目标

- Phase 09 official group 完整。
- 读取本目录 README、`docs/phase_00_report.md` 至 `docs/phase_09_report.md` 和 `docs/robosuite_benchmark_design_tree.md` release/claim 边界。
- 证明干净 clone 当前 ShakeBench fork（内部 package 仍为 robosuite）可安装、创建 `VibrationPickPlaceCan`、重建 physics gates/Gamma selection/scorecard，且不破坏上游环境。

## 1. Clean-room install

在新临时目录/环境 clone 当前 fork，按公开 README editable/regular install；不引用本机 cache、未提交 assets 或任何外部目录。运行全部 upstream+ShakeBench tests，记录 OS/Python/MuJoCo/driver 版本。

## 2. Upstream兼容审计

- 原有 `robosuite.make` 环境列表与默认行为只增加新环境，不改变旧 env action/observation/timestep；
- existing demos/controllers/renderers/robots tests 通过；
- ShakeBench feature 未启用时无 global macro/callback side effect；
- wheel/sdist 包含新增 arena、texture、平铺在 assets 根的 config/profile/state 文件和 scripts。

## 3. Artifact/reproduction

验证 licenses/hashes、official profiles、10 dev/400 official/100 knee states、raw Phase 09 results、rerun ledger 和 schema versions。

重跑全部 physics gates 与 3-process determinism；从 raw 重算 knee fit 和 1600-episode scorecard；在 clean environment 重跑预注册 matched subset 并比较 trace。如资源允许再全量重跑 1600，并清楚区分 raw-score recomputation 与 full-rollout reproduction。

## 4. Permission与scientific audit

- 枚举每 tier 实际 keys；wrapper/debug/renderer 不能泄漏 privileged truth；
- state artifacts 无 future realized outcomes；
- evaluator 只计分不回注 policy；
- 核对 Gamma semantics、driver 1%/1°、COM/preload、six-axis transfer、friction、bounded parity、MDE/n=400、failure denominator 和 ceiling flags；
- claims 不包含 Vision/SOTA/real-world/multi-task。

## 5. Release packaging

生成 evidence index，链接每个 claim 的 raw artifact/script/config/test；更新 ShakeBench extension 用户文档、changelog、citation、license/attribution 和 known limitations。除非项目明确决定 fork 版本号，本阶段不伪装成上游 robosuite 官方 release。

## 完成条件

1. clean install、全 upstream+extension tests 通过；
2. package artifacts 完整（含平铺在 `robosuite/models/assets/` 的 profile/state 文件）；
3. physics/permission/determinism gates 通过；
4. knee/scorecard 可重算；
5. 无 silent exclusion 或未解释 rerun；
6. claims 逐条有证据；
7. blocker list 为空。

产出 `docs/phase_10_release_audit.md`、clean-room log、artifact manifest、evidence index、reproducibility statement 和 limitations。任一门失败则标 blocked 并停止。
