# Phase 02 report：Dynamic Deck Driver

状态：Phase 02R4 **PASS（provisional）**。driver 已由 pre-registered physics screen 选择，并在实际 robosuite Panda/base-inclusive no-task fixture 上完成 `3×2×64` conformance。所有 physics profile 参数仍为 **derived-not-frozen**，正式 freeze 属于 Phase 06。

当前没有加入 isolator、arena、Can、task、接触、IMU、oracle 或正式评测，也没有新建 `shakebench` 子目录。

## Contract

每条记录遵循同一离散时间契约：

```text
t:       write q(t) → integrate x(t) to t+dt
t+dt:    write q(t+dt) → forward-only refresh → sample x(t+dt), acceleration and raw weld data
```

`command_*` 是左极限 integration input；`sample_target_*`、actual acceleration、raw `efc_pos/efc_force` 和 solver diagnostics 是右极限量。每条 trace row 记录两个 target time、两个 mocap application time 和 `sample_time_s`。默认 robosuite 环境未安装 driver 时不请求 refresh；显式 spectrum probe 使用 deterministic stride 16 的实际 post-step samples，sample rate 仍高于 authored program Nyquist 要求。

## R4 selection and load evidence

candidate grid 为 `dt={0.0001,0.0002,0.0004}`，`eq_solref[0]=2×dt`、damping ratio `0.5`，default `eq_solimp`，deck mass/inertia `400 kg/[12.12,14.0833333333,26.0333333333]`。fine 与 nominal 的完整 zero/六单轴/64-line empty screen 通过，coarse 因最大 line phase `1.337599°` 拒绝；按最大可行 dt 选择 nominal `dt=0.0002`、`eq_solref=[0.0004,0.5]`。

confirmatory load case 是 canonical `empty` 与 `panda_plus_worktable_reference_proxy`。后者直接编译仓内 Panda model，包含 `robot0_base` 角色及其 10-body/7-joint subtree（compiled subtree mass `18.5 kg`），另有显式 `32 kg`、`I=[0.9696,1.1363,2.0867]` proxy；proxy collision disabled、无 spring/damper/isolator、无 task/controller。

完整表、逐线指标、compiled assertions、selection rule 和唯一 handoff 见 [`docs/phase_02_driver_selection_report.md`](phase_02_driver_selection_report.md)。

## Artifact and handoff

正式 artifact：[tests/shakebench_phase_02r_probe.json](../tests/shakebench_phase_02r_probe.json)

```text
schema_id/version    = shakebench.phase02r.conformance_matrix / 3
trace schema/version  = shakebench.deck_driver.trace / 3
file SHA-256          = 4f8cebf474d6bd24ccaa736df7180e112c8656ebda6264972835fda37622f6c6
payload SHA-256       = df52713a7c0fc3db18a200ccfbfe22f8613cfbf156a30dbdb06b8f24d0b4bfaf
update reason         = Phase 02R4 full-window Gamma calibration correction and Panda/base-inclusive confirmatory re-measurement
```

验证结果：`integrity_valid=true`、`physics_gates_passed=true`、`phase03_handoff=PASS`。artifact lock 对 update reason、previous file hash 和 selected-candidate provenance 做 authenticated hashing；受控文件走 temporary verify/fsync/atomic replace。

R4 driver tests `59 passed`；Phase 01+02 combined `93 passed`；上游 non-EGL subset `255 passed, 58 skipped`。

Phase 03 handoff：**PASS**。这是 provisional driver/load 证据，不是 official physics freeze。
