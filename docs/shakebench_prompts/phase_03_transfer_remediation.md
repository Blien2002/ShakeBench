# Phase 03R 修复提示词：闭合六轴复数传递、联合谱与偏载平衡证据

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 03 已实现 arena 和 isolator 基础，但其当前 MuJoCo evidence 只验证 six axes 的单一 `4 Hz` 相对振幅，不能作为 Phase 04 的物理前提。本阶段只补全 isolator physics validation 和 XML-default 一致性；保留工作树，不重置，不新建目录，不实现任务、Can、target success、IMU、controller 或 official scoring。

leading word 是 **复数**：传递函数不是一个振幅数字。每个 MuJoCo measurement 必须和同一 reference point/frame/time convention 下的解析复数 transfer 对比 amplitude 与 phase；联合谱必须证明轴间不会被实现错误耦合。

## 必读与范围

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_03_arena_and_isolator.md
docs/robosuite_benchmark_design_tree.md       # 尤其 canonical isolator、8.2 gates
docs/robosuite_benchmark_v0_spec.md
docs/phase_03_report.md
robosuite/models/arenas/shakebench_arena.py
robosuite/models/assets/arenas/shakebench_arena.xml
robosuite/utils/shakebench_isolator.py
tests/test_shakebench_arena.py
tests/test_shakebench_isolator.py
```

可修改：Phase 03 arena/isolator/tests/docs/现有 assets 文件。新增 artifact 或 test 只能平铺在 `tests/`；不得实现 Phase 04 runtime task，也不得改变 Phase 02 已通过的 driver time contract。

## 1. 单一物理定义与 XML 默认一致性

1. 预注册并记录 harmonic probe 的：base motion reference point、worktable COM reference point、relative coordinate定义、world/nominal frame、left/right-limit target convention、fit phase convention和 transient discard window。
2. 解析 `H_abs` / `H_rel` 与 MuJoCo 各自比较同一个量。若 MuJoCo 记录 `z = worktable_COM - deck_reference`，则使用 `H_rel`；若记录 absolute table response，则使用 `H_abs`。不能把 `qpos` 与错误时间的 mocap command 拟合后只比较幅值。
3. 原始 `robosuite/models/assets/arenas/shakebench_arena.xml` 直接用 MuJoCo compile 时的 default joints、stiffness、damping、springref 必须与 `ShakeBenchArena()` / `DEFAULT_ISOLATOR_CONFIG` 完全一致；或 asset 明确没有 physical default、并 fail closed 要求 Python configurator。二者只能选一。当前 XML 的 rotational default 与 Python 5 Hz derivation 不一致，必须消除。
4. XML/Python default equality 的 test 必须绕开 `ShakeBenchArena.configure_isolator()`：直接编译 XML，再逐项比较 `jnt_stiffness`、damping、springref、range、axis、inertial 与 default derived parameters。

## 2. 预注册 six-axis harmonic transfer grid

用 six-axis independent relative coordinates，在下列三个区域对每个轴运行真实 MuJoCo harmonic probe：

```text
tracking:  f = 0.2 * f_n
resonance: f = 1.0 * f_n
isolation: f = 2.0 * f_n
```

对 default candidate `f_n=5 Hz`，即 `[1, 5, 10] Hz`。每个 axis×frequency sample：

- 使用 Phase 02R4 driver 或等价的、明确记录相同左右极限 contract 的 deck input；
- 等待预注册的 transient discard，再在至少 20 个周期或足以稳定 fit 的时间窗上做 sine/cosine complex fit；
- 输出 input/output complex coefficient、amplitude ratio、phase difference、fit residual、sample rate、warning/solver diagnostics；
- 分别比较 `H_rel` 和必要时 `H_abs`，而不是只比较 relative magnitude；
- 使用在运行前写入 artifact 的 tolerance。最低限度要求 amplitude relative error 与 absolute phase error分别有单独 gate；不可只保留 5% amplitude envelope。

若 resonance 点因 small-angle/constraint discretization 达不到预注册容差，保留 raw evidence、标记 Phase 03R BLOCKED；不得调任务或默默放宽门槛。参数化 sweep 可以继续，但不选择或冻结最终 `f_n/zeta`。

## 3. 真实六轴联合谱 probe

在同一已验证 worktable COM frame 运行一次完整 authored six-axis excitation program。该 probe 必须：

- 使用 Phase 01 profile identity、seed/t0、right-limit driver contract；
- 对每个轴的每条 active line 进行 joint sine/cosine fit，输出 amplitude、phase、residual与 conditioning；
- 记录 off-axis output：当分析模型声称 axes 独立时，未激励/非对应线的 cross-axis response 必须被量化并受预注册 leakage bound 约束；
- 记录 deck input、table absolute motion、table-relative-to-deck motion、travel/angle margin、solver/warning diagnostics；
- 用同一 reference point 定义 `T_accel/R_relative/D_relative/T_peak/travel_margin/static sag`。

这不是 Phase 02 deck conformance 的重复，而是 deck→isolated-worktable 的 six-axis response。不运行任务、接触或 success。

## 4. MuJoCo payload mass / COM sensitivity

保留当前 centered world-child `0.349 kg` payload 物理 test，并新增实际偏载物理 probe：

```text
mass = 0.349 kg
COM offsets = [(0,0,0), (0.10,0,0), (0,0.20,0), (0.10,0.20,0)] m
```

payload 必须仍是 world child + freejoint，经明确、非任务的桌面接触传递重量；不可运动学挂到 worktable。对每一个 offset：

- 等到静态平衡，测工作台 six joint equilibrium；
- 比较解析 `static_equilibrium_offset` 的 `tz/rx/ry` 符号、幅值和 reference frame；
- 断言 `k/c` / springref 不随 payload 或 COM 改变；
- 记录 travel/angle margins 和 warnings。

如果 world-child payload 接触模型本身无法稳定提供已知 COM force path，则保留 failure evidence，不能只用解析函数替代 MuJoCo sensitivity claim。

## 5. Artifact、报告与 handoff

新增平铺的机器可读 `tests/shakebench_phase_03_transfer.json`，记录 grid、thresholds、raw fits、payload matrix、visual signature 与 config/profile hashes。默认验证只读；更新必须显式 reason、旧/新 hash 和完整 rerun。

更新 `docs/phase_03_report.md`，新增 `docs/phase_03_transfer_remediation_report.md`。报告明确区分：

```text
analytic model contract
MuJoCo measured transfer
provisional sweep evidence
official freeze (deferred to Phase 06)
Phase 04 handoff = PASS or BLOCKED
```

Phase 04 handoff 仅当以下全部通过时为 PASS：

1. XML direct compile 与 Python default isolator config 一致；
2. 全部 six-axis × three-region probes 在 amplitude/phase/residual gate 内；
3. joint six-axis spectrum 与 independent-axis/leakage gate 一致；
4. centered/offset payload equilibrium 在 MuJoCo 中与解析相符，且 support parameters 未改变；
5. visual layer does not alter physical/transfer trace。

否则 Phase 03R 报告 BLOCKED 并停止；不得进入 Phase 04。

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_isolator.py \
  tests/test_shakebench_arena.py --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_deck_driver.py \
  tests/test_shakebench_isolator.py \
  tests/test_shakebench_arena.py --tb=short
```

完成后停止。
