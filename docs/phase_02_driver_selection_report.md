# Phase 02R4 driver selection and load remediation report

状态：**PASS（provisional，derived-not-frozen）**。R4 已在首次 physics probe 前注册 candidate profile，完成 zero/六单轴/目标 `Gamma=0.30` 的 64-line empty-deck screening，按预注册规则选出 driver，并仅对该 candidate 完成 Panda/base-inclusive 的 `3×2×64` confirmatory matrix。没有实现 Phase 03 isolator/arena、task、接触、IMU、oracle 或正式评测。

## 1. Pre-registered selection

候选在任何 MuJoCo probe 前写入 artifact provenance。所有候选均满足频率采样条件 `dt <= 1/(20*f_max)` 和 `eq_solref[0] >= 2×dt`，并使用相同 deck inertial 与 `eq_solimp`；本轮只改变 `dt` 与相应的 equality time constant。

| candidate | `dt_s` | `eq_solref` | screen | max line amplitude relative error | max absolute phase | decision |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `fine` | `0.0001` | `[0.0002, 0.5]` | PASS | `2.523e-6` | `0.334400°` | eligible |
| `nominal` | `0.0002` | `[0.0004, 0.5]` | PASS | `1.009e-5` | `0.668800°` | **selected** |
| `coarse` | `0.0004` | `[0.0008, 0.5]` | FAIL | `4.035e-5` | `1.337599°` | rejected: phase gate |

确定性选择规则为：先保留 screening 全部通过者；再取最大 `dt`；同 `dt` 取最大逐线 absolute phase 最小者；仍相同按预注册 `candidate_id` 字典序。最终 `selected_candidate_id=nominal`。screening、selection 和 confirmatory 的 artifact stage 顺序分别保存；没有使用 task success 或 controller outcome。

## 2. Panda/base-inclusive no-task fixture

`panda_plus_worktable_reference_proxy` 由仓内 `Panda(idn=0)` / `robosuite/models/assets/robots/panda/robot.xml` 直接组成，不使用猜测质量或单个 scalar robot block：

- explicit role `panda_base -> robot0_base -> deck`；compiled Panda subtree 为 10 个 body、7 个 joint，总 body mass `18.5 kg`；
- provenance 保存每个 attached body 的 compiled mass、COM (`body_ipos`) 和 diagonal inertia、全部 attached body names，以及 deterministic nominal initial joint state；compiled source body/joint name assertions 全部通过；
- `worktable_reference_proxy` 为 `32.0 kg`、COM `[0,0,0]`、`I=[0.9696,1.1363,2.0867] kg·m²`，collision disabled；不含 spring/damper/isolator；
- fixture 明确 `controller_started=false`、`task_environment=false`，只把 Panda XML 的真实 actuators 留给 MuJoCo，controls 保持 zero；
- 每个 Panda load record 的 `panda_base_included=true` 且 `compiled_model_assertions.passed=true`。

若 Panda 编译或 reset 失败，代码会保存 raw failure 并 fail closed；不会退回 table-only proxy。当前实际 compiled audit 通过。

## 3. Confirmatory `Gamma×load` matrix

选中 `nominal` 后才运行以下 6 条记录。每条都包含 right-limit input provenance、same-time raw `efc_pos/efc_force`、warnings/iterations、load provenance 和完整 64-line spectrum。

| target `Gamma` | load | `Gamma_commanded` | `Gamma_deck_actual` | relative error | max line phase | max line amplitude error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.15 | empty | 0.1499355370 | 0.1499712694 | 0.0002383182 | 0.668803° | 1.009e-5 |
| 0.15 | Panda + proxy | 0.1499355370 | 0.1499713537 | 0.0002388810 | 0.668803° | 9.768e-6 |
| 0.30 | empty | 0.2998715775 | 0.2999430129 | 0.0002382198 | 0.668800° | 1.009e-5 |
| 0.30 | Panda + proxy | 0.2998715775 | 0.2999431830 | 0.0002387870 | 0.668800° | 9.748e-6 |
| 0.50 | empty | 0.4997869467 | 0.4999059470 | 0.0002381021 | 0.668793° | 1.009e-5 |
| 0.50 | Panda + proxy | 0.4997869467 | 0.4999062315 | 0.0002386714 | 0.668793° | 9.740e-6 |

所有矩阵记录的 64 条 line 均满足 amplitude `<=1%`、absolute phase `<=1°`、conditioning/residual gate；Gamma relative error 均 `<=1%`。Gamma 使用统一 `include_centripetal=false`、同一 right-limit sample convention；`Gamma_commanded` 与 level scale 分开保存。

## 4. Artifact lock and gates

正式 artifact：[tests/shakebench_phase_02r_probe.json](../tests/shakebench_phase_02r_probe.json)

```text
schema_id/version       = shakebench.phase02r.conformance_matrix / 3
trace schema/version     = shakebench.deck_driver.trace / 3
file SHA-256             = 4f8cebf474d6bd24ccaa736df7180e112c8656ebda6264972835fda37622f6c6
payload SHA-256          = df52713a7c0fc3db18a200ccfbfe22f8613cfbf156a30dbdb06b8f24d0b4bfaf
previous file SHA-256    = 15abbbe65fb961ab3e9798b5adcefd3ceae924ac9c1645133181fbd5e1b4d89f
update reason            = Phase 02R4 full-window Gamma calibration correction and Panda/base-inclusive confirmatory re-measurement
```

artifact integrity 的 hash scope 包含 update reason、previous file hash 和 selected-candidate provenance；只排除 self-referential `artifact_update.new_payload_sha256`。controlled artifact 使用同目录 temporary payload → verifier → fsync → atomic replace；只读 verification 不改文件。

主要 physics gates：candidate screening、deterministic selection、zero、six single-axis、64-line authored spectrum、synthetic estimator、Gamma、3×2 Gamma/load、3×2×64 spectrum、Panda compiled audit、dt convergence 和 solver sensitivity 全部 PASS。`integrity_valid=true` 与 `physics_gates_passed=true` 是独立字段。

## 5. Verification and handoff

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_shakebench_deck_driver.py --tb=short
python -m robosuite.scripts.shakebench_probe_deck_driver \
  --verify-artifact tests/shakebench_phase_02r_probe.json
```

实际结果为 driver `59 passed`，Phase 01+02 combined `93 passed`，上游 non-EGL subset `255 passed, 58 skipped`。verifier 当前返回 `integrity_valid=true`、`physics_gates_passed=true`、`phase03_handoff=PASS`。该 PASS 只表示 Phase 02R4 的 provisional driver/load conformance；`dt/eq_solref/eq_solimp/deck mass/inertia` 仍由 Phase 06 freeze authority 冻结。

唯一 handoff 结论：**`phase03_handoff=PASS`**。可以进入 Phase 03；不得把本报告解释为已实现 Phase 03 runtime physics 或 official benchmark freeze。
