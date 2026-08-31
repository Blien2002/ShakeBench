# Phase 02 time-contract report

R3 的 right-limit contract 已保留并在 R4 artifact 中重新验证；R4 同时关闭了 driver selection 与 Panda/base-inclusive load 前置门，因此当前 handoff 为 **PASS**。

## Left/right-limit contract

```text
at t:
  write integration target q(t)
  integrate state x(t) → x(t+dt) under q(t)

at t+dt:
  qpos/qvel/time are x(t+dt)
  write sample/right-limit target q(t+dt)
  forward-derived quantities without a second integration
  sample acceleration, equality residual/force and solver diagnostics
```

每一条完整 trace row 记录 `integration_target_time_s=t`、`sample_target_time_s=t+dt`、两次 mocap write timestamp 和 `sample_time_s=t+dt`。`DeckDriverTrace` version 为 `3`；spectrum probe 的 artifact sample stride 为 deterministic 16，但每个被记录 row 仍遵循该 contract，且最粗 sample rate 仍满足 authored-line Nyquist 条件。

## Controlled regression evidence

5 Hz translation fixture（`dt=0.0002 s`、amplitude `0.001 m`）：

| refresh input | measured acceleration amplitude |
| --- | ---: |
| stale old target `q(t)` | `35.2417467926 m/s²` |
| correct right-limit target `q(t+dt)` | `0.9867014160 m/s²` |
| authored `qdd(t+dt)` | `0.9869604401 m/s²` |

right-limit 正例在 1% tolerance 内通过；refresh 前后 qpos/qvel/time 严格不变，derived pose 与 current qpos 一致，raw buffers 与 current `efc_pos/efc_force` 一致。lite/non-lite、hard/non-hard reset、disabled-refresh negative fixture 和无 driver upstream env 均有回归覆盖。

## R4 measurement result

candidate screen 使用 two-beat resolution（minimum line spacing `0.05887579362532458 Hz`、beat period `16.984909050463557 s`、required fit window `33.969818100927114 s`；actual fit window `34.2464 s`），逐 candidate 保存 64 条 line metrics。`nominal dt=0.0002 s`、`eq_solref=[0.0004,0.5]` 按最大通过 dt 规则被选中；`coarse` 因 `1.337599°` max phase 被拒绝。

selected candidate 的 empty/Panda-inclusive full `3×2×64` matrix 全部通过 amplitude、phase、conditioning/residual 和 Gamma `<=1%` gate。Panda fixture 的 compiled source/body/joint audit、right-limit provenance 和 raw weld diagnostics 均进入同一 artifact。

## Artifact

[tests/shakebench_phase_02r_probe.json](../tests/shakebench_phase_02r_probe.json)

```text
schema version   = 3
file SHA-256     = 4f8cebf474d6bd24ccaa736df7180e112c8656ebda6264972835fda37622f6c6
payload SHA-256  = df52713a7c0fc3db18a200ccfbfe22f8613cfbf156a30dbdb06b8f24d0b4bfaf
integrity_valid  = true
physics_gates    = true
phase03_handoff  = PASS
```

所有旧 target 的 23× 假 Gamma 证据只作为历史 negative evidence，不进入当前 conformance artifact。当前 artifact 的 update record 保存 non-empty reason、previous file hash 和 new payload hash，hash scope 不排除 update reason/previous hash。
