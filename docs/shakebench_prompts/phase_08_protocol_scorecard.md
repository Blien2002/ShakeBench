# Phase 08 Prompt：Committed States、运行协议与配对 Scorecard

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接在 `robosuite/utils/` 中实现 state 生成/回放、result manifests、MDE calculator 和单 Gamma scorecard；不运行 knee 或 official evaluation，不新建子文件夹。

## 前置与必读

- Phase 07 controller 冻结且 Gamma=0 dev 通过。
- 读取 `docs/robosuite_benchmark_design_tree.md` committed states、MDE、failure integrity、Gamma knee 和 paired statistics。

## 数据合同

```text
10 dev task states
400 official task states
100 knee-calibration episodes
  = 10 object placements × 10 excitation seed/t0 pairs
```

三组无交集；Can yaw=0；task XY 在 nominal 周围独立 uniform ±0.02m。不得根据 rollout 成败筛选或替换 state。

## 直接修改落点（不新建目录）

```text
robosuite/utils/shakebench_protocol.py
robosuite/utils/shakebench_scoring.py
robosuite/scripts/shakebench_generate_states.py
robosuite/scripts/shakebench_replay_state.py
robosuite/scripts/shakebench_scorecard.py
tests/test_shakebench_protocol.py
tests/test_shakebench_scoring.py
```

committed state 数据作为平铺 package-data 文件直接放 `robosuite/models/assets/` 根：

```text
robosuite/models/assets/shakebench_states_dev.json
robosuite/models/assets/shakebench_states_official.json
robosuite/models/assets/shakebench_states_knee.json
```

该目录随 wheel/sdist 发布（MANIFEST.in 已递归打包），不新建子目录。

## CommittedState

至少保存 state_id/split/schema、Can worktable-frame pose、target reference、excitation seed/t0/level scale、Gamma commanded、IMU seed/bias seed、task/physics/controller/config hashes 和 provenance。不得保存 future realized contacts/outcomes。

reset/replay 精确恢复 object/support/sensor/program；V0–V3 共享 matched fields 和 seeds。

## EpisodeResult/RunManifest

保存 success subconditions、failure/infrastructure status、three Gamma values、diagnostics、actions、rerun ledger、versions/hashes 和 scoreable flag。

## Scorecard

每 tier：success count/400、SR、95% Wilson、failure histogram、physics violations、missing/reruns。

paired `V1−V0/V2−V1/V3−V2/V3−V0`：Delta SR、fixed-seed 10,000 paired-bootstrap CI、n_gain/n_loss。SR>0.90 或 <0.10 标 ceiling/floor limited；不自动第二主 Gamma。

Integrity：task/controller/physics/policy crash 进入分母；infrastructure crash 只重跑同 state；重复失败使 group incomplete；不生成 valid-only 主 SR。

## Power contract

```text
minimum effect = 0.10 Delta SR
alpha = 0.05 two-sided
target power >=0.80
official n=400 paired episodes per tier
```

实现可复现 MDE/power calculator 并输出 assumptions。

## 完成条件

1. 10/400/100 artifacts 尺寸正确且无交集；
2. deterministic IDs/hashes/replay 通过；
3. state corruption 与 unknown schema fail closed；
4. Wilson/bootstrap/gain-loss/saturation/incomplete synthetic tests 通过；
5. 400 denominator 在 task failures 下保持；
6. package artifact 包含平铺在 assets 根的 state 文件；
7. 输出 `docs/phase_08_report.md`；
8. 未运行 knee scan 或 1600 official rollouts；未新建目录。

完成后停止。
