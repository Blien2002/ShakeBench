# Phase 02R3 修复提示词：闭合 mocap 左右极限与 acceleration/Gamma 时间契约

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 02R2 已正确发现 qpos/derived-cache 的 stale-state 问题，但其 refresh 仍拿旧 mocap target `q(t)` 重算 `t+dt` acceleration，再与 `qdd(t+dt)` 比较。本阶段只闭合这一离散时间契约并重做受影响证据；保留工作树，不重置，不创建子目录，不实现 Phase 03 isolator/arena/task/IMU/oracle。

leading word 是 **极限**：每个记录量必须明确是 interval 内的左极限 `t+dt−`，还是在 `t+dt` 更新 input 后的右极限 `t+dt+`。禁止把两者混成“same-time”。

## 必读与范围

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_02_dynamic_deck_driver.md
docs/shakebench_prompts/phase_02_review_remediation.md
docs/shakebench_prompts/phase_02_final_conformance_remediation.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_02_final_remediation_report.md
robosuite/environments/base.py
robosuite/utils/shakebench_deck.py
robosuite/scripts/shakebench_probe_deck_driver.py
tests/test_shakebench_deck_driver.py
tests/shakebench_phase_02r_probe.json
```

允许修改：`robosuite/environments/base.py`、`robosuite/utils/shakebench_deck.py`、`robosuite/scripts/shakebench_probe_deck_driver.py`、现有 Phase 02 test/artifact/report/integration-map 文件。新测试或 artifact 仅平铺在 `tests/` 根。先写 red tests，再修生产代码。

## 1. 冻结右极限测量契约

采用且只采用下列契约：

```text
at t:
  write integration target q(t)
  integrate state from t to t+dt under q(t)

at t+dt:
  qpos/twist are the integrated state x(t+dt)
  write sample/right-limit target q(t+dt)
  refresh forward-derived quantities without integrating again
  sample acceleration, equality residual/force as right-limit quantities
```

因此：

- pose / twist sample 是 `x(t+dt)`；
- `actual_acceleration`、`weld_constraint_residual_raw`、`weld_constraint_force_raw` 是对状态 `x(t+dt)`、输入 `q(t+dt)` 的右极限；
- `Gamma_deck_actual(t+dt+)` 只能与 authored `Gamma_commanded(t+dt)` / `qdd(t+dt)` 比较；
- 每个 trace row 必须同时记录 `integration_target_time_s=t`、`sample_target_time_s=t+dt`、实际两次 mocap-write timestamp，以及 `sample_time_s=t+dt`；
- refresh 不得改变 qpos、qvel、time 或执行第二次 integration；下一 physics step 的 pre-hook 重写同一 `q(t+dt)` 是幂等操作。

实现建议：让 `DeckDriver` 作为 opt-in post-integration refresh provider，在 `sim.forward()` 前显式写入 sample/right-limit target。默认 robosuite env 不请求该 seam，不增加 forward，不改变行为。

不要采用下列伪修复：回退 sample timestamp、把 actual trace 平移一个 step、用 finite difference 替代 MuJoCo spatial acceleration、或只在报告中改时间文字。

## 2. 必须先变红的精确回归

至少加入这些独立测试：

1. **旧 target 负例：** 5 Hz translation fixture，在 refresh 时故意保留 `q(t)`。断言 right-limit acceleration 与 `qdd(t+dt)` 明显不一致；保留数值 evidence。
2. **正确 right-limit 正例：** 同一 fixture，在 refresh 前写入 `q(t+dt)`。断言实际 acceleration amplitude 与 authored `qdd(t+dt)` 相符（使用预注册 tolerance），并断言 mocap target 已等于 `q(t+dt)`。
3. **状态不被 refresh 改写：** 保存 refresh 前后 qpos、qvel、time；全部 bitwise / 严格等值。derived `xpos/xquat` 在 refresh 后必须与当前 qpos forward result 一致。
4. **trace contract：** 每一 row 的 integration target、sample target、sample state、Gamma time grid 与 raw weld diagnostics 都通过时间/shape/value test；不得仅测试数组有限。
5. lite 和 non-lite、hard-reset 和 non-hard-reset 都覆盖；无 driver 的 upstream env 明确断言不请求 refresh。

测试必须能复现此前受控证据：旧 target refresh 得约 `35.24 m/s²`，sample target refresh 得约 `0.98670 m/s²`，authored 值约 `0.98696 m/s²`（容许版本/精度造成的窄容差变化），从而防止未来只修 qpos/xpos 而再次遗漏 target 时刻。

## 3. 重建 measurement、Gamma 与 conformance artifact

此前 R2 artifact 的实际 acceleration、Gamma、phase、weld force/residual 数字全部作废。修复后：

- 更新 trace schema/version、field contract 和 artifact schema/provenance，记录 `right_limit_input_convention`；
- canonical Gamma 使用同一 `include_centripetal` boolean，command 与 actual 均按 sample/right-limit convention计算；
- 对每个 target Gamma、每个 load case、六个单轴和 64 条谱线重新运行；
- 保留 R2 的两-beat-period resolution、synthetic estimator、actual physics conformance threshold和 Gamma×load matrix；
- candidate grid 必须实际运行并报告各候选的 physics metrics，不能仅记录 `dt/tau` 元数据再硬编码 selected；选择只看预注册 physics metrics，不看 task SR；
- 32 kg table proxy 继续保留，但未包含 Panda/base 时 Phase 03 handoff 仍必须是 BLOCKED；不得把它改称完整 preload envelope。

artifact verify 需要同时区分：

```text
integrity_valid   # artifact schema/hash/provenance 自洽
physics_gates_passed
phase03_handoff   # PASS only when both are true and Panda/base envelope is present
```

不得将 `integrity_valid=true` 显示或报告为 conformance passed。

## 4. 报告与停止条件

更新 `docs/phase_02_final_remediation_report.md`，并新增 `docs/phase_02_time_contract_report.md`。报告必须直接回答：

1. 左/右极限 contract 是什么；
2. 旧 target 负例、right-limit 正例的数值；
3. 重新计算后全部 phase/Gamma/line/load gate；
4. artifact hash、更新 reason、schema version；
5. Phase 03 handoff 是 `PASS` 还是 `BLOCKED`，及具体原因。

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

完成定义：每条 acceleration/Gamma/weld row 有明确且测试覆盖的右极限 input；旧 target 的 23× 假 Gamma 不可再出现；全矩阵在新 contract 下重新测量；只有所有 driver gates 加上 Panda/base-inclusive load envelope 均通过时才能设置 Phase 03 handoff `PASS`。否则记录 raw evidence、写 `BLOCKED` 并停止。
