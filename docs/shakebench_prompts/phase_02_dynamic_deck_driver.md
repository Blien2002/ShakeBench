# Phase 02 Prompt：直接在 robosuite step/model 中实现 Dynamic Deck Driver

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接修改现有 robosuite 文件（重点 `robosuite/environments/base.py`），使 ShakeBench 新环境可拥有 environment-owned timestep、XML reparent processor 和 dynamic deck driver；现有环境默认行为必须保持不变，不新建子文件夹。

## 前置与必读

- Phase 01 通过；读取 `docs/integration_map.md`。
- 读取 `docs/robosuite_benchmark_design_tree.md` 的 deck driver、weld gates 和 spike 复用边界。
- 参考随仓证据：`docs/robosuite_benchmark_design_tree.md` 的 driver/weld 合同、`docs/spike_implementation_formal_review.md`、`docs/repro_weld_sag_20260828.md`；不得引用任何外部目录。

## 目标拓扑

```text
world
├── deck_driver  # mocap, non-contact
└── deck         # ordinary body + freejoint + explicit inertial
    └── named mount roles / probe load

equality weld(deck_driver, deck)
```

dynamic descendants 通过 deck 获得真实 base-excitation inertia。直接 mocap parent 方案作为 negative regression 保留，不作为实现。

## 修改要求

### 1. Environment-owned timestep seam（直接修改 `robosuite/environments/base.py`）

按 Phase 00 审计结果实现最小、向后兼容 seam：ShakeBench env 可指定 model timestep，而未指定的所有现有 env 继续使用当前 macro/default 路径。

- import/构造时不修改全局 `macros.SIMULATION_TIMESTEP`；
- multi-env sequential construction 不串扰；
- `sim.model.opt.timestep`、`model_timestep`、controller timing 一致；
- 为所有受影响父类签名和现有 env 加回归测试。

### 2. Role-based XML processor（新模块 `robosuite/utils/shakebench_deck.py`）

- 创建 driver/deck/freejoint/weld/sites；
- 按显式 body role/handle 重挂 robot base 或 probe body；
- 不硬编码 Lift 的 `table/cube`；
- duplicate application 与 missing role fail closed；
- 编译后可审计 parent graph、inertial、equality 参数。

### 3. Physics-step scheduler（在现有 `MujocoEnv.step` 路径上最小侵入）

在对应 `mj_step1` 位置计算前写入正确时刻的 mocap target。记录 command/application/sample timestamps；不用不可审计的一拍超前常量掩盖时序。

输出 command/actual pose、spatial twist/accel、weld residual/wrench、solver warnings/iterations。

### 4. Probe

新增 `robosuite/scripts/shakebench_probe_deck_driver.py` 和 `tests/test_shakebench_deck_driver.py`，覆盖 zero、六个单轴、六轴 spectrum、多个 Gamma、empty/representative load、多档 dt/eq_solref/eq_solimp。

Phase 02 只找到可行 provisional 组合；Phase 06 才冻结 official profile。

### 5. Negative regression

最小 5 Hz resonance fixture 证明 `mocap parent + compliant child` 得到 relative amplitude 0 / transmissibility 1，而非解析约 5.099，防止未来删除 dynamic deck+weld。fixture 平铺进 `tests/`。

## 完成条件

1. 至少一个 provisional 配置在 minimal probe 中达到 amplitude/Gamma 误差<=1%、phase<=1°；
2. ShakeBench timestep 不通过 global mutation 实现；
3. existing env timestep/action/determinism tests 无回归；
4. six-axis instrumentation 和 negative regression 自动化；
5. 报告 `docs/phase_02_report.md` 标记参数 derived-not-frozen；
6. 未加入 isolator、Can 或 task；未新建目录。

完成后停止。
