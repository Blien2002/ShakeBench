# Phase 02R4 修复提示词：预注册 driver 选择与 Panda/base-inclusive load envelope

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 02R3 已通过右极限时间契约，Gamma measurement 可信；当前只剩两个 Phase 03 前置物理门：driver 参数尚未按结果选择、deck-side load envelope 没有 Panda/base。只修复这两项及其 artifact lock；保留工作树，不重置，不创建目录，不实现 isolator、arena、任务、接触、IMU、oracle 或正式评测。

leading word 是 **选择**：candidate 必须在运行前定义，依据预注册 physics metrics 选择，选择后才运行 confirmatory matrix。不得先写 selected、后跑 smoke，或根据 task success 选择任何参数。

## 必读与范围

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_02_dynamic_deck_driver.md
docs/shakebench_prompts/phase_02_time_contract_remediation.md
docs/robosuite_benchmark_design_tree.md       # 尤其 3.3、8.2、9.1
docs/robosuite_benchmark_v0_spec.md
docs/phase_02_time_contract_report.md
robosuite/environments/base.py
robosuite/utils/shakebench_deck.py
robosuite/scripts/shakebench_probe_deck_driver.py
tests/test_shakebench_deck_driver.py
tests/shakebench_phase_02r_probe.json
```

可修改：Phase 02 driver/probe/XML fixture/tests/artifact/docs；新增文件只平铺在现有 `tests/` 或 `docs/`。不得把 Panda/base load fixture 变成 task environment，也不得加入 spring、damper、isolator joint、Can、container、controller或 task reward。

## 1. 预注册并真正执行 candidate selection

在首次运行前，将 candidate grid 作为机器可读 profile 写入 artifact provenance。每行明确 `candidate_id`、`dt_s`、`eq_solref`、`eq_solimp`、deck mass/inertia、frequency-sampling condition 和 `eq_solref >= 2*dt` condition。

grid 至少包含当前 fine / selected / coarse 的三个不同 dt 组合；可以包含预注册的 `eq_solref` / `eq_solimp` 变体，但每次只改变明确的物理变量。所有 candidate 先通过数学时序约束，再进入真实 MuJoCo probe。

采用下列两阶段、确定性的 selection：

1. **screening：** 每个 candidate 运行 zero、六个单轴，以及 target `Gamma_commanded=0.30` 的完整 64-line authored spectrum（empty deck，使用两-beat fit window和 right-limit contract）。逐线要求 amplitude relative error `<=1%`、absolute phase error `<=1°`、fit conditioning/residual 通过；
2. **选择：** 只在 screening 全部通过的 candidate 中选择 `dt` 最大者；若相同 dt 有多个，则选择最大逐线 absolute phase error 最小者；若仍相同，按 artifact 中预注册 ID 的字典序 tie-break；
3. **无候选通过：** 保存每个 candidate 的 raw results / rejection reason，`phase03_handoff=BLOCKED` 并停止；
4. **confirmatory：** 仅对已选 candidate 运行第 2 节的完整 Γ×load matrix。不得在 confirmatory failure 后换 candidate；应回到新的、显式 revision 的 candidate grid。

candidate grid 的 metadata 不算执行证据。artifact 必须保存每个 candidate 的真实 screen metrics、pass/fail 和 selected-by rule；测试应能篡改一个 metric 后证明 selected ID / handoff 发生相应变化。

## 2. 构造真实 Panda/base-inclusive、无任务的 driver load fixture

Phase 02 的 load fixture 必须成为 driver conformance fixture，而不是 1 kg 占位物或最终 task。

构造最小 MJCF/robosuite fixture，包含：

```text
dynamic deck
├── actual robosuite Panda robot model
│   └── explicit base body role, mounted rigidly under deck
└── worktable_reference_proxy
    └── M=32.0 kg, I=[0.9696,1.1363,2.0867] kg m², COM=[0,0,0]
```

要求：

- 使用随仓 robosuite Panda robot model/robot XML，而不是猜测 Panda 质量或单个 scalar block；编译后自动记录 Panda subtree/body masses、COM、inertia information、base role、initial joint state与全部 attached body names；
- Panda joints 可以固定在明确的 nominal configuration 或使用无控制的 deterministic reset，但不启动 task/controller；
- table proxy 在本阶段是刚性 deck-side **conservative driver load**，不是隔振工作台，不含 spring/damper/joint；明确 collision disabled，不能产生 task contact；
- 若实际 Panda 模型不能在 no-task fixture 中稳定编译/重置，保留 raw failure；不得退回到 table-only proxy 并称 Panda-inclusive；
- load provenance 必须为 `panda_base_included=true` 且包含可审计 compiled-model assertions。否则 worktable proxy gate 必须 fail closed。

## 3. Confirmatory Γ×load conformance matrix

仅对 selection 后的 candidate，运行：

```text
Gamma targets = [0.15, 0.30, 0.50]
load cases    = [empty, panda_plus_worktable_reference_proxy]
```

每个 3×2 record 必须有 canonical `Gamma_commanded`、`Gamma_deck_actual`、relative error、right-limit input provenance、same-time raw weld diagnostics、warnings/iterations和 load provenance。每一项要求 Gamma relative error `<=1%`。

对每个 load case、每个 Gamma target，运行完整 64-line spectrum 并逐条执行 amplitude/phase/fit gate；不能以只跑 Gamma scalar 或单轴 smoke 代替。保存每条 line 的实际指标和失败 reason。

只有 screening、Panda/base compiled audit 和 confirmatory 3×2 全部通过，才能设置：

```text
physics_gates_passed = true
phase03_handoff = PASS
```

这仍只是 provisional driver profile；`dt/solref/solimp/deck mass/inertia` 保持 `derived-not-frozen`，正式 freeze 仍属于 Phase 06。

## 4. Artifact 与证据锁定

- artifact provenance 新增并验证 `right_limit_input_convention`；
- 对受控 artifact 使用：候选 payload 写入临时同目录文件 → `verify_artifact(temp)` → fsync / atomic replace。任何失败保留原 artifact；
- artifact integrity 必须覆盖 update reason、previous hash 与 selected-candidate provenance，不能将整个 update record 排除在 payload hash 外；
- generator 和 verifier 共用同一个 typed threshold/profile 和 `compute_gate_summary`，避免重复的 1% / 1° / conditioning 常量；
- tests 从独立 expected manifest / 固定 test constant 校验受控 artifact file hash，并验证 `integrity_valid` 与 `physics_gates_passed` / `phase03_handoff` 是不同概念。

## 5. 报告与停止

更新 Phase 02 reports，新增 `docs/phase_02_driver_selection_report.md`。报告必须列出完整 candidate screen 表、deterministic selected-by decision、Panda/base compiled audit、3×2×64 conformance summary、artifact hash/update reason，以及唯一 handoff 结论。

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

完成定义：driver 已由真实 pre-registered screen 选择；selected candidate 在 Panda/base-inclusive 3×2×64 matrix 通过所有 physics gate；artifact 可原子更新且被独立锁定；报告明确 `phase03_handoff=PASS`。否则保留 raw evidence，报告 `BLOCKED`，停止在 Phase 02R4。
