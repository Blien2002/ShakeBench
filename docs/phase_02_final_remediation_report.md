# Phase 02 final conformance remediation report

状态：**PASS（Phase 02R4 provisional）**。当前 driver 已完成 right-limit time contract、pre-registered result-driven selection，以及 Panda/base-inclusive `3×2×64` physics conformance。`dt/eq_solref/eq_solimp/deck mass/inertia` 仍保持 **derived-not-frozen**，正式 freeze 归 Phase 06。

## Runtime contract

```text
at t:       write integration target q(t); integrate x(t) → x(t+dt)
at t+dt:    write sample target q(t+dt); forward-refresh without integration; sample
```

trace schema 是 `shakebench.deck_driver.trace` version `3`；artifact schema 是 `shakebench.phase02r.conformance_matrix` version `3`。`command_*` 是左极限，`sample_target_*`、actual acceleration、raw `efc_pos/efc_force` 和 solver data 是右极限。无 driver 的现有环境不请求 post-integration refresh。

## R4 result

筛选在 confirmatory 前完成，且不读取 task/controller success：

| candidate | `dt_s` | `eq_solref` | screen result | max phase |
| --- | ---: | --- | --- | ---: |
| fine | `0.0001` | `[0.0002,0.5]` | PASS | `0.334400°` |
| nominal | `0.0002` | `[0.0004,0.5]` | **SELECTED** | `0.668800°` |
| coarse | `0.0004` | `[0.0008,0.5]` | FAIL | `1.337599°` |

screen 对每个 candidate 实际运行 zero、六单轴和目标 `Gamma=0.30` 的 64-line authored spectrum；line amplitude `<=1%`、absolute phase `<=1°`、conditioning/residual 作为筛选门。选择规则是最大通过 `dt`，同 `dt` 再按最大逐线 phase，最后按 ID 字典序。

选中 candidate 后，confirmatory matrix 只运行 nominal，覆盖：

```text
Gamma targets = [0.15, 0.30, 0.50]
load cases    = [empty, panda_plus_worktable_reference_proxy]
spectra       = 6 records × 64 active lines
```

每条 record 的 Gamma relative error 小于 `0.024%`，最大 line phase `0.668803°`，最大 line amplitude relative error `1.010e-5`；六条记录均通过 full line gate。

## Panda/base load audit

`panda_plus_worktable_reference_proxy` 直接使用仓内 `Panda(idn=0)` 和 `robots/panda/robot.xml`。`panda_base` role 为 `robot0_base`，compiled subtree 有 10 bodies、7 joints、18.5 kg；artifact 保存每个 body 的 mass/COM/inertia、所有 attached names、nominal initial joint state 与 source/joint/body compiled assertions。32 kg worktable proxy 的惯量为 `[0.9696,1.1363,2.0867] kg·m²`、COM zero、collision disabled、无 spring/damper/isolator。fixture 明确 `controller_started=false`、`task_environment=false`。

## Artifact lock

正式 artifact：[tests/shakebench_phase_02r_probe.json](../tests/shakebench_phase_02r_probe.json)

```text
file SHA-256          = 4f8cebf474d6bd24ccaa736df7180e112c8656ebda6264972835fda37622f6c6
payload SHA-256       = df52713a7c0fc3db18a200ccfbfe22f8613cfbf156a30dbdb06b8f24d0b4bfaf
update reason         = Phase 02R4 full-window Gamma calibration correction and Panda/base-inclusive confirmatory re-measurement
previous file SHA-256 = 15abbbe65fb961ab3e9798b5adcefd3ceae924ac9c1645133181fbd5e1b4d89f
```

payload hash 覆盖 update reason、previous hash 和 selected-candidate provenance；只排除 self-referential `artifact_update.new_payload_sha256`。controlled update 采用 same-directory temporary payload、verify、fsync 和 atomic replace；失败不覆盖原文件。

## Verification and handoff

测试结果：driver `59 passed`；Phase 01+02 combined `93 passed`；上游 non-EGL subset `255 passed, 58 skipped`。静态 compile、`git diff --check` 和独立 artifact manifest/hash 校验均通过。

verifier 当前返回：

```text
integrity_valid      = true
physics_gates_passed = true
phase03_handoff      = PASS
```

唯一结论：**`phase03_handoff=PASS`**。这只授权进入 Phase 03；不表示 Phase 03 已实现或 Phase 02 physics profile 已正式冻结。
