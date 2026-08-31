# Phase 02R remediation report

本报告的 R3 measurement 结论已由 Phase 02R4 supersede。R3 的 right-limit implementation 仍是当前 runtime contract；R4 在此基础上完成了数据驱动 candidate selection 和 Panda/base-inclusive load remediation。

当前正式报告：[`docs/phase_02_driver_selection_report.md`](phase_02_driver_selection_report.md)。

## Preserved R3 contract

```text
t:       write q(t) → integrate x(t) to t+dt
t+dt:    write q(t+dt) → forward-only refresh → sample right-limit state/input
```

受控旧 target evidence 仍为 `35.2417467926 m/s²`，right-limit target evidence 为 `0.9867014160 m/s²`，authored `qdd` 为 `0.9869604401 m/s²`。这部分由 `DeckDriverTrace` schema 3 和 regression tests 锁定。

## R4 superseding evidence

- candidate grid 在运行前预注册；fine/nominal/coarse 实际完成 zero、六单轴和目标 `Gamma=0.30` 的完整 64-line empty-deck screen；最终选中最大可行 `dt=0.0002 s` 的 `nominal`；
- confirmatory 只运行 selection 后的 candidate，覆盖 `empty` 与 `panda_plus_worktable_reference_proxy` 的三个 Gamma target，每条 load record 具备完整 64-line spectrum；
- Panda 使用随仓 `Panda(idn=0)` XML，compiled subtree 10 bodies/7 joints、18.5 kg，保存 per-body mass/COM/inertia、attached names、base role 和 initial joint state；
- 所有 R4 physics gates、artifact integrity 和 Phase 03 handoff 均为 PASS；参数仍为 derived-not-frozen。

正式 artifact：[tests/shakebench_phase_02r_probe.json](../tests/shakebench_phase_02r_probe.json)

```text
file SHA-256     = 4f8cebf474d6bd24ccaa736df7180e112c8656ebda6264972835fda37622f6c6
payload SHA-256  = df52713a7c0fc3db18a200ccfbfe22f8613cfbf156a30dbdb06b8f24d0b4bfaf
schema version   = 3
handoff          = PASS
```

R4 不实现 Phase 03 isolator/arena/task 或任何 official evaluation。
