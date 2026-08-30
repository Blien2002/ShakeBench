# Phase 00 report：改名、上游集成审计与直接集成骨架

日期：2026-08-29  
仓库：`/home/miracle04/Desktop/ShakeBench`  
起始 commit：`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`  
状态：Phase 00 additive bootstrap 已完成；本报告中的历史 baseline 与当前仓库状态分开记录。

## 1. 改名验证

目录关系已由用户完成并经本阶段只读核对：

- `/home/miracle04/Desktop/ShakeBench` 是原 `robosuite` ARISE fork 的当前实现仓库；
- `/home/miracle04/Desktop/robosuite` 不存在；
- `/home/miracle04/Desktop/ShakeBench_backyp` 是原来的 ShakeBench 备份目录；
- Desktop 上没有第二个精确名称为 `ShakeBench` 的目录；本阶段没有再次执行 `mv`，也没有读取或引用备份目录；
- 当前仓库 `.git` 保留；Phase 00 审计开始时 `origin` 为 `https://github.com/ARISE-Initiative/robosuite.git`，当前发布 remote 为 `https://github.com/Blien2002/ShakeBench.git`；没有重写历史或 force push。

初始 git 状态为 `master...origin/master`，既有权威资料均为工作树未跟踪文件；本阶段保留这些用户已有修改。

## 2. 权威文件与 asset provenance

`docs/IMPORT_PROVENANCE.md` 中登记的 SHA-256 与 `docs/` 根权威副本一致：

| 文件 | 实际 SHA-256 | 结果 |
| --- | --- | --- |
| `docs/robosuite_benchmark_design_tree.md` | `47ad5a9200dbca708ca1577756435a866f8b7d340c7ac747fcfc17bf7c49af10` | current match |
| `docs/robosuite_benchmark_v0_spec.md` | `a8eb3e568851199386c1e0c89120d3811bbb08029c57a1209dbffc675904f346` | current match |
| `docs/spike_implementation_formal_review.md` | `fe59f27b0a35d3fc96276292a586114fa352de426544962d6185e67ad2e8ec9f` | match |
| `docs/canonical_imu_profile_validation.md` | `a81a0bd5e27bdea4f4363c8475edd0faa4be3256c89d600cad24ac68e2d3f833` | match |
| `docs/repro_weld_sag_20260828.md` | `7af3fe4963470c98079d5e1ba6f2fbfc7b01d4701bca63404207767c8684b0de` | match |

提示词目录 `docs/shakebench_prompts/` 保留在当前仓库。已验证工业工作台 texture 位于现有 assets 路径 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg`，大小 `163720` bytes，SHA-256 为 `6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb`，与 provenance 记录一致。

## 3. 安装、版本与注册表基线

成功执行：

```bash
python -m pip install -e .
```

本次验证环境：

```text
Python    3.13.5
robosuite 1.5.2   (/home/miracle04/Desktop/ShakeBench/robosuite/__init__.py)
MuJoCo    3.9.0
NumPy     1.26.4
```

`mink==0.0.5` 作为可选依赖被安装，以覆盖 GR1 composite-controller 上游测试；它将 NumPy 降为其声明的 `<2.0.0` 范围。`robosuite_models` 未安装，因此保留上游已有 warning；Phase 00 不需要它。

editable import smoke：`import robosuite, mujoco, numpy` 成功；`robosuite.__version__ == "1.5.2"`；当前注册环境数量为 19，名称集合由 bootstrap test 锁定。新增 config module 不 import `robosuite`，不会触发 registry mutation。

## 4. 上游入口审计

完整符号级落点见 [`integration_map.md`](integration_map.md)。本阶段实际确认了：

- `register_env` / `EnvMeta.__new__` / `make` 位于 `robosuite/environments/base.py`；根导入链在 `robosuite/__init__.py`；
- XML processor 位于 `MujocoEnv._xml_processors`、`set_xml_processor`、`_initialize_sim` 和 `edit_model_xml`；
- reset 位于 `MujocoEnv.reset` / `_reset_internal`；lite-physics loop 位于 `MujocoEnv.step`；
- `RobotEnv._pre_action` 是现有控制分发 hook；`initialize_time` 和 `MujocoWorldBase.__init__` 是 timestep seam；
- `ManipulationEnv`、`Lift`、`Task`、`TableArena`、`CanObject`、Panda 的 base/gripper attachment 和 `Observable` 生命周期均已定位到实际 symbol；
- `MANIFEST.in` 已递归包含 `robosuite/models/assets/` 和 `robosuite/scripts/`，`setup.py` 使用 `include_package_data=True`。

## 5. Phase 00 实施内容

新增：

- [`robosuite/utils/shakebench_config.py`](../robosuite/utils/shakebench_config.py)：`ShakeBenchConfig`、`ExtensionIdentity`、JSON schema、canonical JSON、SHA-256、`UNFROZEN` official-field gate；默认 `scoreable=False`；
- [`robosuite/scripts/shakebench_cli.py`](../robosuite/scripts/shakebench_cli.py)：`version`、`validate-config`、`print-schema`；
- [`tests/test_shakebench_config.py`](../tests/test_shakebench_config.py)：import/registry、round-trip/hash、unknown-field、scoreability 和 schema tests；
- [`docs/integration_map.md`](integration_map.md)：当前文件职责、真实上游 seam、后续模块和 package-data/test 落点；
- 本报告。

没有修改：package name/version、`robosuite.macros.SIMULATION_TIMESTEP`、global callbacks、environment registration、MujocoEnv/RobotEnv/Lift/TableArena/Can/Panda/wrapper 的既有代码；没有实现 excitation、deck、isolator、arena、task、contact、IMU、provider、controller 或 scorecard；没有新建任何目录。

## 6. 测试证据

### 6.1 原始上游命令的已知工具链阻断

直接执行：

```bash
python -m pytest -q
```

在收集阶段退出 `3`：仓库中两个旧测试模块在 import 时调用 `argparse.parse_args()`，拒绝 pytest 的 `-q` 参数；同时当前 Python 3.13 环境加载 ROS `launch_testing` entrypoint 时出现未知 hook `pytest_launch_collect_makemodule`。这不是 Phase 00 文件引入的失败。

### 6.2 去除外部 runner 干扰后的上游基线

安装可选 `mink` 后执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  --ignore=tests/test_controllers/test_linear_interpolator.py \
  --ignore=tests/test_controllers/test_variable_impedance.py \
  -x -vv --tb=short
```

收集到 320 items；首个运行时失败前为 `49 passed, 6 skipped`。失败在既有 `tests/test_environments/test_all_environments.py::test_all_environments` 的 Door headless render 初始化：当前主机缺少可用 EGL device display / `swrast_dri.so`。失败栈位于既有 `robosuite/utils/binding_utils.py` 与 EGL context，不涉及 Phase 00 新文件。`MjRenderContext.__del__` 随后出现既有的未初始化 `con` 清理 warning。

### 6.3 Phase 00 目标测试与 CLI

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_shakebench_config.py
```

结果：`7 passed`。

CLI smoke 已验证：

```bash
python -m robosuite.scripts.shakebench_cli version
python -m robosuite.scripts.shakebench_cli print-schema
```

`version` 报告 extension `shakebench`、schema `1`（Phase 00 historical result；当前 config schema 为 `2`）、Python package `robosuite`、package version `1.5.2`；`print-schema` 输出严格 top-level schema。draft config（`scoreable=false`）验证成功；将 `scoreable=true` 且保留默认 `UNFROZEN` 字段时以 exit status `2` 拒绝。

### 6.4 不依赖主机 EGL 的上游回归集合

为隔离主机图形驱动条件，执行了不创建 offscreen/on-screen renderer 的 controller、action playback、gripper、robot 集合：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_controllers/test_composite_controllers.py \
  tests/test_environments/test_action_playback.py \
  tests/test_grippers tests/test_robots --tb=short
```

结果：`255 passed, 58 skipped in 160.05s`。Phase 00 没有修改这些上游实现；该结果作为当前 additive 骨架的回归证据。

## 7. 完成条件核对

| 条件 | 结果 |
| --- | --- |
| 仓库目录改名并保留 `.git` / upstream | PASS |
| 权威 docs 与 provenance hash 一致 | PASS |
| 新路径 editable install/import | PASS |
| 既有 registry、action/observation 默认路径无修改 | PASS；registry/config bootstrap 通过，未修改公共核心 |
| bootstrap 与受影响 smoke tests | PASS；完整渲染基线受主机 EGL 条件阻断 |
| 后续模块 integration map 明确落点 | PASS |
| 不新建 shakebench 子目录、不实现后续 physics/task/controller、不引用外部目录 | PASS |

## 8. Current checkpoint after Phase 01/01R

- Phase 01R 开始前的当前基线为 `5e5829ee`；remediation implementation checkpoint 为 `b4ee4808`。
- Phase 01 authored spectrum 已明确为 new authored candidate；不能称为旧 ShakeBench exact reuse 或 official-ready。
- 当前配置 schema 已升级到 version `2`，scoreable gate 仍因后续 freeze authority 缺失而 fail closed。

Phase 00 的历史测试结果仍有效；Phase 01/01R 的最新测试和修复证据见 `docs/phase_01_remediation_report.md`。
