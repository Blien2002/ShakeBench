# Phase 00 Prompt：改名 ShakeBench、上游集成审计与直接集成骨架

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。本阶段先完成 robosuite→ShakeBench 改名，再在现有 robosuite package 内建立直接集成骨架；不得初始化新项目、复制整个 package、实现后续 physics，也不得新建任何 shakebench 子文件夹。

## 步骤 0：robosuite → ShakeBench 改名

前提（已由用户/前置操作完成，本阶段只验证，不引用任何外部目录）：

- 权威设计文档已随仓：`docs/robosuite_benchmark_design_tree.md`、`docs/robosuite_benchmark_v0_spec.md`、`docs/spike_implementation_formal_review.md`、`docs/canonical_imu_profile_validation.md`、`docs/repro_weld_sag_20260828.md`；
- 导入记录：`docs/IMPORT_PROVENANCE.md`。

1. 确认 Desktop 上存在 `robosuite`（ARISE fork）且没有其他名为 ShakeBench 的目录。
2. 执行改名：

```bash
mv /home/miracle04/Desktop/robosuite /home/miracle04/Desktop/ShakeBench
```

3. 保留原 .git 与 upstream remote（ARISE-Initiative/robosuite fork）。如需对外发布，新增用户自己的 ShakeBench remote；不得重写历史或伪装为上游官方 release。
4. 内部 Python package 名保持 `robosuite`：`import robosuite`、环境注册表、`robosuite/models/...` 资产路径和 tests 布局全部不变；只有仓库目录名与项目名称为 ShakeBench。
5. 验证：`git status` 正常、从新路径 editable install/import 成功、`docs/` 权威文件与 `docs/IMPORT_PROVENANCE.md` 哈希一致、`docs/shakebench_prompts/` 随仓库保留。

## 先读与基线

1. 完整读取 `docs/robosuite_benchmark_design_tree.md`、`docs/robosuite_benchmark_v0_spec.md` 和本目录 `README.md`。
2. 检查 `git status`、当前 commit、Python/MuJoCo/robosuite version。
3. 运行并记录当前上游测试基线。
4. 审计以下真实入口，不凭经验猜路径：
   - environment registration / `robosuite.make`；
   - `MujocoEnv` XML processors、reset 和 lite-physics step loop；
   - `ManipulationEnv` / `Lift` model assembly；
   - `TableArena`、CanObject、Panda base/gripper handles；
   - Observable、wrapper、asset packaging 和 tests conventions。

## 目标

形成最小直接集成骨架，使后续阶段能直接在现有目录与文件上实现而不污染现有环境默认行为。

## 实施步骤

### 1. 写集成决策记录

在 `docs/integration_map.md` 记录：

- 每个新增模块文件的最终路径与职责（只允许现有目录内新文件）；
- 必须修改的上游核心文件及原因；
- environment registration/import 路径；
- timestep/step-loop/XML reparent seams；
- package data 和 texture 路径（平铺进现有 assets 目录）；
- test 文件放法与运行命令。

每个 seam 指向实际源码 symbol。

### 2. 建立扁平扩展模块（不建子文件夹）

新增 `robosuite/utils/shakebench_config.py`，只放：

- typed config/schema skeleton；
- stable serialization/config hash；
- `UNFROZEN` official-field validation；
- extension identity。

不要新增 distribution，也不要修改 package name/version。

### 3. 建立测试与脚本（只新增文件，不新建目录）

新增：

```text
tests/test_shakebench_config.py
robosuite/scripts/shakebench_cli.py
```

CLI 提供 `version` / `validate-config` / `print-schema`。测试 robosuite import、现有环境 registry 不变、ShakeBench schema round-trip、unknown-field rejection、unfrozen scoreable config rejection。

### 4. 建立兼容性护栏

- 新 extension 未显式使用时，现有环境的 observation/action/model timestep/XML 保持不变。
- 不在 import 时修改 `robosuite.macros.SIMULATION_TIMESTEP` 或注册全局 callbacks。
- 如后续需要公共核心 seam，先以测试锁定现有行为，并在 integration map 中写最小向后兼容签名；本阶段不实现 seam。

## 必须产出

- 改名验证记录（写入 `docs/phase_00_report.md`）；
- `robosuite/utils/shakebench_config.py`；
- `tests/test_shakebench_config.py`；
- `robosuite/scripts/shakebench_cli.py`；
- `docs/integration_map.md`；
- `docs/phase_00_report.md`；
- 上游全测/目标测试基线记录。

## 完成条件

1. 仓库已由 robosuite 改名 ShakeBench，Desktop 上不再有名为 ShakeBench 的其他目录；
2. `docs/` 权威文件与 `docs/IMPORT_PROVENANCE.md` 哈希一致；
3. 当前 repo 原地 editable install/import 成功；
4. 现有 robosuite environment registry、action/observation smoke 没有变化；
5. 新 bootstrap tests 和受影响上游 tests 通过；
6. integration map 对每个后续模块都有明确落点；
7. 未新建任何目录（无 `robosuite/utils/shakebench/`、`tests/test_shakebench/`、`docs/shakebench/` 等），未实现 excitation、deck、arena、task 或 controller，未引用任何外部目录。

完成后停止。
