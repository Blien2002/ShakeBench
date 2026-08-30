# Phase 01 report：纯 NumPy authored excitation core

状态：Phase 01 additive excitation core 完成；经 Phase 01R remediation 降级为 `new authored v0 candidate`，不是已证明的旧 ShakeBench exact reuse。实现仍位于原 robosuite package 的现有目录中；没有创建 MuJoCo model/environment，也没有创建新的子目录。

## 1. 范围与基线

- implementation repo：`/home/miracle04/Desktop/ShakeBench`；该目录是原 robosuite。
- Phase 00 基线 commit：`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`；Phase 01 初始实现 commit：`d1db196e061e8795cbdde8064ddb9cf71b1d492c`。
- 只依据随仓的 design tree、v0 spec、phase prompts 和 Phase 00 integration map；没有读取或复制外部目录/实现。
- 本阶段 runtime path 只依赖 NumPy 与本仓库的配置 envelope；没有导入 Torch、Isaac、MuJoCo 或环境注册路径。
- Python 支持策略：新增代码要求 Python `>=3.7`，因为使用 `from __future__ import annotations` 和 dataclasses；`setup.py` 与 Black target 已同步，不再声明 Python `>=3`。

## 2. 实际产出

| 路径 | 内容 |
| --- | --- |
| `robosuite/utils/shakebench_excitation.py` | 六轴 candidate band table、确定性 line frequency jitter/phase、`seed/t0/time/level_scale/active_axes` API、五阶 ramp、解析 `q/qdot/qdd` 和可重放的 serialized line program |
| `robosuite/utils/shakebench_calibration.py` | workpiece-point authored vertical peak、`alpha × r`、self-contained unit replay、level scale、peak factor、per-axis RMS/peak 和 Gamma result |
| `robosuite/utils/shakebench_safety.py` | candidate displacement、non-ballistic、frequency/timestep、六自由度 feature solver-travel 及可选 angle gate；失败不裁剪输入而是 fail closed |
| `robosuite/scripts/shakebench_generate_excitation_golden.py` | 自包含 golden fixture 和误差摘要生成器 |
| `tests/test_shakebench_excitation.py` | band、line/q amplitude、replay、t0、ramp、analytic derivative、active-axis、Gamma、safety 和 golden replay 测试 |
| `tests/test_shakebench_calibration.py` | `alpha × r`、unit replay、Gamma scaling、level-scale round trip 测试 |
| `tests/golden_shakebench_excitation_v0.json` | 已生成的平铺 golden fixture |
| `tests/golden_shakebench_excitation_error_summary.json` | 机器可读 invariant residual 与 tolerance |

## 3. Candidate authored encoding for this phase

由于当前仓库缺少可审计的旧 ShakeBench 激励算法、source commit 或 reference dataset，Phase 01R 选择 B；以下是依据设计树写入的 candidate 表，不宣称 exact reuse：

| axis | center Hz | relative accel RMS | bandwidth ratio | tones | unit |
| --- | ---: | ---: | ---: | ---: | --- |
| `tx` | 5.0 | 0.50 | 0.10 | 12 | m / m/s² |
| `ty` | 6.5 | 0.35 | 0.10 | 10 | m / m/s² |
| `tz` | 8.0 | 1.00 | 0.10 | 12 | m / m/s² |
| `rx` | 3.0 | `0.30 × tz_RMS / 0.65 m` | 0.12 | 12 | rad / rad/s² |
| `ry` | 4.0 | `0.30 × tz_RMS / 0.65 m` | 0.12 | 10 | rad / rad/s² |
| `rz` | 2.5 | `0.30 × tx_RMS / 0.65 m` | 0.12 | 8 | rad / rad/s² |

The explicit candidate defaults are `reference_accel_rms_m_s2=1.0`,
`kappa_rot=0.30`, `reference_lever_m=0.65`, uniform bounded jitter of 10% of
the adjacent line spacing, `ramp_duration_s=0.50`, and a 2.0 s authored
episode window.  The workpiece point is explicitly serialized as
`workpiece_point_offset_m=[0.65, 0, 0]`; it is a Phase 01 provisional authored
choice, not a claim that the later physical workpiece geometry is frozen.
These values are serialized in `ExcitationConfig` and in each program, so
later phases can replace a provisional value only with a new provenance
record.

`t0` is a common carrier phase origin and is absorbed into
`line_phase_at_episode_zero`; the quintic ramp remains episode-relative.  The
time-shift identity is therefore exact for the carrier and for corresponding
post-ramp samples.  Official program payloads expose no per-axis phase
override.

For Gamma, the candidate workpiece point is `[0.65, 0, 0] m` from the deck
origin and is explicit in the config. The authored point acceleration includes
`a + alpha × r`; the command-level safety preflight additionally evaluates the
effective support normal with the centripetal term. The commanded Gamma is the
authored point normal peak divided by `gravity_m_s2`; realized deck/table Gamma
remains deferred to Phase 02+.

## 4. Golden provenance

Generation command:

```text
python -m robosuite.scripts.shakebench_generate_excitation_golden \
  --generated-at-utc 2026-08-30T00:00:00Z
```

The fixture records:

- generator: `robosuite/scripts/shakebench_generate_excitation_golden.py`;
- generation time: `2026-08-30T00:00:00Z` (metadata only; not the reproducibility anchor);
- fixture source profile: `shakebench.authored_v0_candidate`, decision `phase-01-remediation-B-20260830`;
- fixture config-envelope hash: `91c0f38575eccb429d18948c0d8f47050407e837ccd2b65c803cf7560eef7661`;
- canonical fixture payload SHA-256 (excluding its self-hash and update history fields): `3111107a8db4e91ba38b684a354c2826b3dc197b901df9c35718c183add753d5`;
- machine-readable error summary `max_error`: `1.7642198812950483e-10`, below the recorded analytic-derivative tolerance `1e-7`;
- line amplitude, line displacement amplitude, frequency, ramp C2 and safety residuals are zero; central-difference residuals are numerical-check evidence only.

Each Gamma result's `unit_replay` contains the complete excitation config,
its config hash, the actual time grid, and a serialized level-one program; a
custom calibration time grid is therefore replayable without hidden state.

The fixture is a read-only acceptance input. It includes multiple seeds, nonzero `t0`, six-axis and single-axis
programs, samples before/during/after the ramp, Gamma records, and explicit
displacement/non-ballistic/frequency/solver-travel rejection cases.

## 5. Verification

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_excitation.py tests/test_shakebench_calibration.py \
  tests/test_shakebench_config.py --tb=short
34 passed in 0.97s

python -m compileall -q robosuite/utils/shakebench_*.py \
  robosuite/scripts/shakebench_*.py
```

The Phase 00 upstream non-EGL regression remains `255 passed, 58 skipped`;
the existing headless renderer/EGL blocker is recorded in
`docs/phase_00_report.md`.  Phase 01 adds no environment or rendering path.

## 6. Boundary audit

- No `robosuite/utils/shakebench/`, `tests/test_shakebench/`,
  `docs/shakebench/` or `robosuite/models/assets/shakebench/` directory was
  created.
- No existing environment, task, controller, physics model, registry,
  timestep macro or package identity was modified.
- No official evaluation, deck driver, isolator, task, IMU, provider or
  controller work was pulled forward from later phases.
- `scoreable=True` remains fail-closed: all default official physics/controller/
  protocol fields are `UNFROZEN`, and Phase 01 has no freeze authority.

Phase 01 complete; stop here until `phase_02_dynamic_deck_driver.md` is
explicitly requested.
