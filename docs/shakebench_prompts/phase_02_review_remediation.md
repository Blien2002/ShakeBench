# Phase 02R 修复提示词：封闭 Dynamic Deck 的测量语义与 conformance 证据

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 02 的实现已经在工作树中；本阶段只修复其审查发现的 driver、测量和兼容性问题。保留其他人的既有修改，不重置工作树，不创建任何 `shakebench` 子目录，也不推进 isolator、arena、Can、任务、IMU provider、oracle 或正式评测。

本阶段的 leading word 是 **同一性**：每个被比较的 command、actual 和诊断量必须具有同一时刻、同一坐标系、同一物理定义。能运行、数组有限或 RMS 接近都不能替代同一性。

## 必读与权威顺序

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_02_dynamic_deck_driver.md
docs/robosuite_benchmark_design_tree.md       # 尤其 3.3、8.2、9.1
docs/robosuite_benchmark_v0_spec.md
docs/phase_02_report.md
docs/integration_map.md
robosuite/utils/shakebench_deck.py
robosuite/scripts/shakebench_probe_deck_driver.py
tests/test_shakebench_deck_driver.py
robosuite/utils/shakebench_calibration.py
robosuite/utils/shakebench_excitation.py
```

设计冲突按：最新设计树 > v0 spec > Phase report > 旧 prompt。MuJoCo 的 frame/constraint 语义必须以当前已安装 MuJoCo 的官方 API 文档与本地 headers 为准；不要由“单位姿态测试恰好通过”推断坐标系正确。

## 允许修改的落点

```text
robosuite/environments/base.py
robosuite/environments/robot_env.py
robosuite/environments/manipulation/*.py
robosuite/controllers/parts/controller.py
robosuite/controllers/parts/gripper/gripper_controller.py
robosuite/controllers/parts/mobile_base/mobile_base_controller.py
robosuite/utils/shakebench_deck.py
robosuite/scripts/shakebench_probe_deck_driver.py
tests/test_shakebench_deck_driver.py
tests/test_shakebench_*.py
docs/integration_map.md
docs/phase_02_report.md
docs/phase_02_remediation_report.md
```

可以新增平铺在 `tests/` 根的 fixture 或测试文件；不得新建目录。先写每个问题的 red test，再修生产代码。不要通过放宽阈值、移除测试、时间平移 actual trace、或改写 Gamma 定义使结果变绿。

## 1. 关闭 driver 安装的半启用生命周期

当前危险状态是：XML processor 尚未作用于已编译 model，但 pre/post hook 已经开始运行。必须让 driver 的 XML、compiled audit 和 hooks 原子地启用。

选择并明确实现一种安全 API：

1. **pre-init install：** `DeckDriver.install()` 只允许在 `env.sim is None` / model 未初始化时调用；对已初始化环境 fail closed 并给出如何在新环境构造路径中安装的错误信息；或
2. **atomic rebuild install：** 已初始化环境使用单独显式 API，在同一调用中完成 hard reset、processor 应用、compiled audit、bind，最后才注册 hooks。失败时恢复为没有 hooks 的旧状态。

无论选哪种，普通 `install()` 不得留下“下一次 reset 才可能正确”的中间状态。添加回归：late install 后直接 `step()` 要么明确拒绝，要么已经拥有 audited deck，绝不出现 missing-deck bind error；hard-reset 与 non-hard-reset 的行为也要明确。

## 2. 修复 timestep seam 与 controller fallback

- 所有原生环境未显式传 `model_timestep` 时继续走原有 macro/default 行为；显式 ShakeBench env 的 `sim.model.opt.timestep`、`env.model_timestep`、physics step count 和 controller timestep 必须一致，且不会改写 global macro。
- 处理 lightweight mock simulator 时，不能在 fallback 前先解引用 `sim.model.opt`。将读取逻辑集中为一个小型 helper，或用安全的嵌套 `getattr`；Controller、GripperController、MobileBaseController 共用它而不是复制三份。
- 为 default env、两个不同 explicit-timestep env 顺序构造、不可整除 fail-closed，以及没有 `model` / `opt` 的 mock 分别写测试。
- 更新新增公开参数的 Google-style docstrings。删除或收束 Phase 02 尚未需要的重复 aliases / 多套同义 configuration names；保留一组 canonical 名称，必要兼容 alias 必须只在边界处转换并测试。

## 3. 修复 six-axis actual state 的 frame 与时间定义

`DeckDriverTrace` 的 contract 必须逐字段写清：frame、origin、component order、采样时刻与单位。

1. 若 trace 称 `actual_twist` / `actual_acceleration` 为 world spatial quantity，就必须返回 world quantity。MuJoCo free-joint 的 rotational `qvel/qacc` 是 body-local tangent quantity，不能直接标为 world angular velocity/acceleration。
2. 推荐在 post-constraint 阶段调用官方 body/object spatial API，明确选择 global frame；若使用 generalized `qacc`，必须以实际姿态转换 angular component，并处理所需 transport terms。不要用 finite difference 伪造 acceleration。
3. 对 command 也要保持同一物理定义：rotation-vector coordinate derivative 不可在有限旋转下无条件标作 spatial angular twist/acceleration。实现正确 SO(3) mapping，或将 trace 字段明确降级为 generalized rotational-coordinate derivatives；后者不得与 spatial actual state 混比。
4. 添加非单位 base orientation（例如 world Z 轴 90°）的单轴 rotation regression。它必须断言 command 与 actual 的同一 world axis 匹配；不能只检查范数或有限性。
5. 明确 command timestamp 为 target evaluated time、application timestamp 为写入 mocap time、sample timestamp 为 post-step state time。任何 fit 只能使用预注册的同一时间约定，不得针对 actual 追补一个 model-step shift。

## 4. 正确记录 weld residual 与 wrench

当前 pose tracking error、MuJoCo equality residual 和 constraint multiplier 是三个不同东西，必须分开命名和记录：

```text
deck_tracking_pose_error             # command pose vs actual deck pose；world frame
weld_constraint_residual_raw         # MuJoCo equality residual / efc_pos；constraint-coordinate order
weld_constraint_force_raw            # MuJoCo efc_force；constraint-coordinate order
weld_wrench_world_at_deck_site       # 只有明确推导后才能提供；world frame、力+力矩、指定作用点与符号
```

- 若本阶段不能严谨地从 equality Jacobian / body interaction force 推导最后一项，就不要把 raw `efc_force` 叫 wrench；保留 raw diagnostics，并在报告中说明 physical wrench 的实现被延后且不作为当前 gate。
- 如果提供 world wrench，写出转换公式、frame、site origin、sign convention，并用虚功/单轴力矩或等价的解析 regression 验证，而不是只检查 shape。
- XML/compiled audit 继续验证 `eq_solref/eq_solimp`；runtime trace 必须能审计追踪误差、raw constraint residual/force 和 solver warning/iterations，不得混用名称。

## 5. 按 canonical Gamma 定义重建 probe

`Gamma_commanded` 是实验自变量，不等于 `level_scale`。使用 Phase 01 的 `calibrate_gamma` / `level_scale_for_gamma`、同一 workpiece point、support normal、`alpha × r`（及明确的 centripetal 选择）定义 candidate Gamma。

对每个 candidate `Gamma_commanded`：

1. 从 unit replay 求出对应 `level_scale`，记录 seed、`t0`、program/profile hash、Gamma point 与 time grid；
2. command Gamma 用 authored command 同一时间定义计算；
3. `Gamma_deck_actual` 用 actual deck 的正确 world spatial acceleration、actual angular velocity/acceleration 和同一物理 workpiece point计算；
4. 记录 `abs(Gamma_deck_actual / Gamma_commanded - 1)`，而不以 actual 结果重新归一化 input；
5. 空载与 representative load 均输出这些量。

至少用三个明确的 target Gamma values（可以继续使用 0.15/0.30/0.50 作为**目标 Gamma**，而非 scale）；若 candidate safety gate 拒绝其中值，保存 rejection evidence，不能静默替换。

单轴 translation smoke Gamma 也必须注明它只是 driver lower-bound diagnostic；不能替代 canonical workpiece-point Gamma gate。

## 6. 把 conformance 变成覆盖矩阵，而非 smoke checks

Phase 02R 不冻结参数，但必须诚实验证当前 provisional candidate。构建机器可读 probe matrix，并在 tests 中断言其内容而不是只断言数组有限。

矩阵至少包括：

| 轴 / 情形 | 必须度量 |
| --- | --- |
| zero | pose/twist/accel 与 weld diagnostics 静止，warning delta 为零 |
| 六个单轴 | command→actual amplitude、phase、Gamma（适用时） |
| 六轴 authored spectrum | 每条 active line 的 amplitude 与 phase；不可只比较 RMS |
| 多个 target Gamma | `Gamma_commanded`、`Gamma_deck_actual`、relative error |
| empty / representative load | 上述关键 tracking/Gamma 指标均可比较 |
| 三档 dt | **固定** deck mass/inertia、`eq_solref`、`eq_solimp`、激励、load，只改变 dt；同时满足 `solref >= 2*max(dt)` |
| solref / solimp sensitivity | 每次仅改变预先声明的一个变量，记录而非据 task SR 调参 |

对 designated provisional candidate，在第 02 原 prompt 所承诺的 minimal gate 至少断言：amplitude relative error `<=1%`、canonical Gamma relative error `<=1%`、absolute phase error `<=1°`。如果六轴/载荷/收敛矩阵的更严格 design-tree conformance 尚未达到，应将 Phase 02R 标记 blocked/partial，保存全部 raw results；不得把单个 tx 空载通过写成 six-axis conformance 已完成。

phase fit 必须逐线对实际 sample-time trace 做已记录的 sinusoidal basis fit；记录 transient discard window、采样率、fit residual 和 phase convention。联合谱中频率近似、line spacing、ramp 时间窗须避免频谱泄漏误判。

## 7. 报告、验证与停止

更新 `docs/phase_02_report.md`，并新增 `docs/phase_02_remediation_report.md`。报告必须：

- 区分实际通过的 gate、未通过/blocked 的 gate、以及仅 smoke coverage；
- 不再把 `level_scale` 称为 Gamma；
- 声明所有 provisional `dt/solref/solimp/deck mass/inertia` 仍为 derived-not-frozen；
- 给出 raw machine-readable artifact 的路径、hash、命令、配置、seed/t0、time convention 和每项阈值；
- 说明 Phase 03 可以开始的条件：所有 P1 measurements 的同一性测试通过；否则停在 Phase 02R。

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_deck_driver.py --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_config.py \
  tests/test_shakebench_excitation.py \
  tests/test_shakebench_calibration.py \
  tests/test_shakebench_deck_driver.py --tb=short
python -m robosuite.scripts.shakebench_probe_deck_driver --output <committed-flat-artifact-path>
```

并运行与当前环境兼容的 non-EGL upstream regression subset；若完整 suite 被既有 argparse/EGL 问题阻塞，保留命令、首个阻断与已通过数量。

完成定义：

1. deck driver 没有半启用生命周期；
2. `Gamma_commanded`、`Gamma_deck_actual`、坐标系和 sample-time 具有可审计的同一性；
3. world spatial actual state 在非单位姿态下通过轴级 regression；
4. weld tracking/error/raw constraint data 不再被混称为空间 wrench；
5. conformance matrix 能证明它真正测了哪些轴、谱线、Gamma、载荷与 dt，而不是只证明程序能跑；
6. 没有引入 Phase 03+ runtime physics。

完成后停止。若完整矩阵证明当前 provisional driver 不满足 pre-registered gate，报告 blocked evidence，而不是继续调 task 或隔振器。
