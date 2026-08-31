# Phase 02R2 修复提示词：闭合 post-step 同一性、谱线分辨率与载荷矩阵

你在 `/home/miracle04/Desktop/ShakeBench` 工作。当前 Phase 02/02R 的代码和 artifact 已在工作树中。本阶段只修复仍会污染 Phase 03 isolator transfer/preload 的 Dynamic Deck 证据问题；保留现有修改，不重置工作树，不创建子目录，不实现 isolator、arena、Can、任务、IMU provider、oracle 或正式评测。

leading word 是 **同时**：每条 trace 的 `time`、`qpos`、derived kinematics、constraint diagnostics、command reference 和 Gamma 必须描述同一个 MuJoCo state。任何一拍旧数据都不能进入 driver conformance、传递率或 preload 结论。

## 必读与权威顺序

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_02_dynamic_deck_driver.md
docs/shakebench_prompts/phase_02_review_remediation.md
docs/robosuite_benchmark_design_tree.md       # 尤其 3.3、8.2、9.1
docs/robosuite_benchmark_v0_spec.md
docs/phase_02_report.md
docs/phase_02_remediation_report.md
robosuite/environments/base.py
robosuite/utils/shakebench_deck.py
robosuite/scripts/shakebench_probe_deck_driver.py
tests/test_shakebench_deck_driver.py
tests/shakebench_phase_02r_probe.json
```

冲突优先级：最新设计树 > v0 spec > Phase report > earlier prompt。当前报告中 `partial/blocked` 的 artifact 不是可接受通过证据；不要依据其数字进入 Phase 03。

## 允许修改范围

```text
robosuite/environments/base.py
robosuite/environments/robot_env.py
robosuite/environments/manipulation/*.py
robosuite/utils/shakebench_deck.py
robosuite/scripts/shakebench_probe_deck_driver.py
tests/test_shakebench_deck_driver.py
tests/test_shakebench_*.py
tests/shakebench_phase_02r_probe.json
docs/integration_map.md
docs/phase_02_report.md
docs/phase_02_remediation_report.md
docs/phase_02_final_remediation_report.md
```

先写 red tests，后修改 production code。新行为必须是 opt-in：没有安装 ShakeBench driver 的上游环境不得增加 `forward()`、改变 action/observation 时序或改变 global macro。

## 1. 使 post-step state 真正同时

当前错误：`mj_step2` / `mj_step` 推进了 `qpos` 与 `data.time`，但 `xpos`、`xquat`、velocity/acceleration cache 与 equality buffers 仍可能是 integration 前状态。把旧 derived state 标成 `t + dt` 是错误测量。

实现一个最小、driver-opt-in 的 post-integration refresh seam：

1. 仅当已安装 driver 显式请求时，在每个 physics integration 完成后、`_update_observables()` 与 post-physics hooks 前，运行 MuJoCo forward/等价的完整 derived-state refresh；
2. refresh 后 trace 的 `sample_timestamps_s == data.time`，而 `actual_pose/twist/acceleration`、`efc_pos`、`efc_force` 和 solver diagnostics 全部来自该时刻；
3. 默认 robosuite 环境保持旧路径，不多一次 forward；
4. non-lite `step()` 与 lite `step1/step2` 都覆盖；hard-reset 与 non-hard-reset 都覆盖；
5. 不通过把 sample timestamp 回退为 `t`、也不通过将 command/actual 人为平移一拍来“修复” phase。

必须写回归，直接断言：

```text
trace.sample_time == sim.data.time
trace.actual deck world pose == 从当前 qpos forward 后得到的 deck pose
trace raw weld residual/force == 当前 post-refresh constraint buffers
```

再加一个故意禁用 refresh 的局部 test fixture，证明它会出现 `qpos` 与 `xpos` 不一致；这防止未来删除 seam 后测试仍绿。要避免该 fixture 改变默认生产路径。

更新 `DeckDriverTrace.field_contract`、report 和 artifact，明确 tracking pose error 的唯一符号约定。选择 `actual - command` 或 `command - actual` 之一，并让代码、schema、tests、文档完全一致。

## 2. 重新建立可分辨的逐谱线 conformance

所有此前由 stale post-step state 产生的 phase/Gamma/artifact 数字作废，必须在 refresh 修复后重跑。

1. 由 authored active-line frequencies 计算 `minimum_line_spacing_hz` 与 beat period `1 / minimum_line_spacing_hz`；
2. fit window 从 ramp 后开始，预注册为至少 **两个最小 beat periods**，即 `fit_window_s >= 2 / minimum_line_spacing_hz`。本 profile 最小间隔约 0.05888 Hz，故 9 s 不可接受；
3. 在运行真实 driver 前，用合成的已知 line program 验证 estimator 能在该窗口、同一 sample grid 与一个预注册的小 perturbation 下恢复每条 amplitude/phase；据此冻结 condition-number/residual rejection rule。不能只打印 condition number；
4. full spectrum 对 64 条 active line 的每条输出 amplitude error、phase error、fit residual、condition number、window、sample rate 和 phase convention；
5. 只可选择满足 `dt <= 1/(20 f_max)`、positive `eq_solref >= 2*dt` 的 provisional configuration。可从预注册的小型 physics-only candidate grid 中选一个通过组合，但不得看 task success，也不得在 artifact 后静默替换；
6. designated provisional driver 在 **六个单轴、全六轴逐线谱、canonical Gamma** 上都必须满足 amplitude relative error `<=1%`、absolute phase error `<=1°`、Gamma relative error `<=1%`。任何一项未通过，写 blocked evidence 并停在 Phase 02R2。

对 full-spectrum fit，报告与 tests 必须区分 `driver pose coordinate conformance` 和 future `isolator transfer`；这里仍不实现 isolator。

## 3. 用可辩护的 Gamma × load 矩阵关闭 Phase 03 preload 前提

每个 target `Gamma_commanded ∈ {0.15, 0.30, 0.50}`（或被 candidate safety gate 明确拒绝的同一 target）都必须分别在 every pre-registered load case 下输出：

```text
Gamma_commanded
Gamma_deck_actual
abs(Gamma_deck_actual / Gamma_commanded - 1)
per-line/single-axis tracking metrics as applicable
same-time raw weld diagnostics
```

load case 至少包括：

1. empty deck；
2. 一个**有来源的机械 proxy**，而不是任意 1 kg block。该 proxy 至少显式覆盖未来 32 kg worktable reference mass/inertia 的 deck-side loading effect，并说明 Panda/base effect 如何纳入或为何该 envelope 保守；
3. 如果当前阶段无法在不引入 Phase 03 runtime topology 的前提下构造足以覆盖 Panda/base + table 的 proxy，则将此声明为 blocked，并禁止用 “1 kg representative” 通过 gate。

proxy 是 driver conformance load，不是隔振器模型：不得添加 spring/damper、isolator joints 或任务接触。所有 proxy 的 mass/inertia、COM、attachment role、collision setting 和 provenance 必须进入 artifact；也不得根据 load case 的任务成功率调参数。

## 4. 锁定审计对象与正式 artifact

- `DeckBodyHandles`、`DeckXMLAudit` 的 public mapping 递归不可变；`to_dict()` 仍返回可安全修改副本。为 mutation rejection 写测试。
- 正式 artifact 是测试输入而非报告附件。测试读取 `tests/shakebench_phase_02r_probe.json`，校验 schema/version、SHA-256、command provenance、trace field contract、coverage matrix、所有 pass gates 与每个 required Gamma×load record。
- artifact 更新需使用显式 `--update` / `--accept-reference-change` 与非空 reason，记录旧/新 hash；默认 probe invocation 只能验证 artifact 或写到临时路径，不能覆盖受控 artifact。
- artifact 锁定的 pass criteria 必须与报告一致。若任何 gate blocked，artifact 和报告必须如实为 blocked，且 Phase 03 gate 不得被设置为 pass。
- 统一 centripetal provenance：`level_scale_for_gamma`、Phase 01 calibration record、actual Gamma 和报告使用同一个明确 boolean；不允许一个字段写 true、另一个定义写 false。

## 5. 报告与停止条件

更新 `docs/phase_02_report.md`，并写 `docs/phase_02_final_remediation_report.md`。必须列出：

- post-refresh state contract 与默认路径未改变的证据；
- full spectrum fit 的预注册 resolution/conditioning rule；
- candidate grid、最终 provisional combination、全 64 line 及 Gamma×load gate；
- artifact command、SHA、更新 reason、完整 provenance；
- 哪些值仍 derived-not-frozen；
- 唯一 Phase 03 handoff 结论：`PASS` 或 `BLOCKED`。

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_deck_driver.py --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_config.py \
  tests/test_shakebench_excitation.py \
  tests/test_shakebench_calibration.py \
  tests/test_shakebench_deck_driver.py --tb=short
python -m robosuite.scripts.shakebench_probe_deck_driver \
  --verify-artifact tests/shakebench_phase_02r_probe.json
```

完成的定义：

1. 同一 `sample_time` 的 qpos、derived kinematics、constraint buffers、trace 和 Gamma 已由回归证明一致；
2. full-spectrum estimator 具有足够频率分辨率并通过预注册 conditioning/residual gate；
3. 通过的 provisional configuration 在全部六轴逐线、canonical Gamma 和每个 required load case 满足 driver threshold；
4. 正式 artifact 不可静默漂移，审计 mapping 不可事后篡改；
5. 没有实现 Phase 03 runtime physics。

若 1–4 中任何一项不满足，保留 raw evidence，报告 `BLOCKED`，完成后停止。只有全部满足时，报告明确允许开始 Phase 03。
