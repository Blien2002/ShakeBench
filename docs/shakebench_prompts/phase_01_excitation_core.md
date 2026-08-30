# Phase 01 Prompt：在 robosuite utils 中实现纯 NumPy 激励模块

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接在现有 `robosuite/utils/` 中新增模块文件，实现 simulator-independent excitation/config/tests；不创建 environment，不新建任何子文件夹或 distribution。

## 前置与必读

- Phase 00 全部完成，读取 `docs/integration_map.md`。
- 完整读取本目录 `README.md`、`docs/robosuite_benchmark_design_tree.md` 第 3 节和 `docs/robosuite_benchmark_v0_spec.md` Excitation 部分。
- 实现只依据随仓权威文档；不得引用任何外部目录或外部代码。

## 修改落点（均不新建目录）

```text
robosuite/utils/shakebench_excitation.py
robosuite/utils/shakebench_config.py       # Phase 00 已有，按需扩展
robosuite/utils/shakebench_calibration.py
robosuite/utils/shakebench_safety.py
tests/test_shakebench_excitation.py
tests/test_shakebench_calibration.py
robosuite/scripts/shakebench_generate_excitation_golden.py
```

golden fixtures 作为平铺文件放在 `tests/`（文件名 `golden_shakebench_excitation_*.json`），不建子目录。不得让普通 robosuite import 加载 Torch 或 Isaac。

## 实现合同

### Authored spectrum

完整编码设计树六轴 band table、tones、relative RMS、bandwidth、rotation rule、frequency scale 和 deterministic jitter/phase。API 显式接收 `seed/t0/time/level_scale/active_axes`。

同一 `t0` 是所有轴共同时间平移；official schema 不支持逐轴 phase override。

### Analytic motion

- 每 line 解析计算 `q/qdot/qdd`；
- 五阶 ramp 包含解析一阶/二阶导数；
- translation/rotation units 写入 schema；
- runtime derivatives 不用 finite difference。

### Gamma 与安全门

- `Gamma_commanded` 基于 authored deck command 的 workpiece-point vertical peak；
- 考虑 `alpha × r`；
- 保存 unit replay、level scale、peak factor、per-axis RMS/peak；
- 实现 displacement、non-ballistic、frequency 和 solver-travel 输入检查；
- 保守最高 line frequency `<8.87 Hz`。

## Golden fixtures（自包含，不依赖外部参考代码）

1. 实现完成后，用 `robosuite/scripts/shakebench_generate_excitation_golden.py` 生成一组小型 fixtures 并提交；每个 fixture 内记录生成脚本路径、config hash 和生成时间。
2. 每个 fixture 必须通过以下解析不变量，测试自动校验：
   - 每 line 加速度幅值关系 `accel_amp = level_scale * band.accel_rms * sqrt(2 / tones)`；
   - 位移幅值关系 `q_amp = accel_amp / omega^2`；
   - 转动带由 `kappa_rot` 与 `reference_lever_m` 派生；
   - 同 seed exact replay；不同 seed 不同 program；
   - 共享 `t0` 时间平移恒等式；
   - 五阶 ramp 及其一、二阶导数连续；
   - 任意采样点的解析导数与高精度数值差分一致；
   - 保守最高 line frequency `<8.87 Hz`。
3. 覆盖多 seed、非零 `t0`、ramp 前/中/后、六轴与单轴、Gamma 标定、displacement/ballistic rejection。
4. 后续阶段只读已提交 fixture；fixture 更新必须重新跑全部不变量并记录 provenance。

## 必须测试

- golden fixtures 与解析不变量；
- analytic derivative numerical check；
- same/different seed；
- common `t0` time-shift identity；
- Gamma accuracy；
- ramp C2 continuity；
- all safety failures；
- normal robosuite import/environment smoke 不变。

## 产出与完成条件

产出 `docs/phase_01_report.md`、golden provenance 和机器可读误差摘要。

完成条件：

1. runtime path 只依赖 NumPy/SciPy 与 robosuite 自身；
2. 所有 golden 解析不变量与安全门通过；
3. 现有 robosuite 目标测试无回归；
4. 未创建 MuJoCo model/environment；
5. 未新建任何目录，未引用任何外部目录。

完成后停止。
