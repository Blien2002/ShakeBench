# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目简介

这是 **ViBench**：一个面向具身智能操作的独立 Isaac Lab + Newton/MJWarp benchmark。Franka Panda 在其基座和工作台受到六轴振动激励时执行拾取与放置任务。本项目特意**不**导入旧版 RM75/MuJoCo 项目，也**不**依赖桌面上的任何其他项目目录——所有配置、资产、工具和测试都位于本仓库内；唯一外部运行时是 `ISAACLAB_ROOT` 指定的 Isaac Lab 环境。

`docs/reports/current_implementation.md` 是权威且最新的实现参考文档（中文）。进行非简单更改前请先阅读；行为发生变化时请同步更新。`docs/fourth_round_validation.md` 记录了当前的验证边界以及尚未得到证实的内容。

## 解释器与环境

此代码仓库中的任何内容都不能在系统 Python 下运行——`isaaclab` 只安装在单独的 venv 中。即使单元测试也会导入 `arena`/`shaker`/`visual_assets`，而这些模块又会导入 `isaaclab`，因此不存在“纯”测试子集。

```bash
# Demo / any simulation entry point (sets PYTHONPATH + interpreter):
./run.sh [args]                       # honors ISAACLAB_ROOT; default is $HOME/IsaacLab-3.0

# Tests (always disables the pytest plugin autoload that breaks the Isaac venv):
./run_tests.sh
./run_tests.sh tests/test_vibration.py::test_seed_is_reproducible

# Generic runner for tools/probes that need the same Isaac venv:
./run_python.sh tools/visual_audit.py <mp4|png> \
  --time-s 6.0 --regions-config configs/visual_regions.yaml \
  --compare docs/visual_baseline.json --json-output out/audit.json
./run_python.sh tools/generate_lab_textures.py   # regenerates assets/textures (SHA-256 pinned by tests)
./run_python.sh scripts/probe_newtongl_capabilities.py
```

复现标准 Demo：

```bash
./run.sh --record --physics-profile official --episode-s 16 \
  --vibration spectral --seed 17 --workpiece sugar_box --workpiece-scale 0.75 \
  --output out/benchmark_v2_wrist_camera_fixed.mp4 \
  --metrics-output out/benchmark_v2_wrist_camera_fixed.json
```

CLI 完整实现位于 `src/vibench/cli.py`；当 `success=true` 时以状态码 0 退出，否则以状态码 2 退出——非零退出是合理的结果，不一定表示存在错误。

## 每步调用链

整个系统就是一个循环；理解以下顺序即可弄清大部分代码：

1. `SpectralVibration.sample()` (`vibration.py`)——生成*地板中心*处的六轴位移/速度/加速度。各谱线的导数均通过解析法计算，绝不使用有限差分。
2. `write_support_groups()` (`supports.py`)——把 deck 六轴运动经单一刚体变换 `p = q[:3] + c + R(l-c)` 写入所有支撑成员，而非采用小角度近似。
3. `VibrationBenchmarkTask._write_supports()` (`task.py`)——将运动学状态写入振动地板、Panda 的**浮动**根节点、工作台及桌腿、目标箱，以及 12 个通过解析法求解的 Stewart 杆件。
4. `ScriptedPickPlaceController.command()` (`controller.py`)——DLS 微分 IK + 阶段状态机。
5. Newton/MJWarp 推进动力学与接触计算。
6. `task.observation()`——位姿、滤波后的接触、腕部力/力矩、穿透量、相机外参。
7. `BenchmarkRecorder` (`recording.py`)——合成主视图、腕部画中画及遥测叠加层。

工件从不通过脚本控制——它只响应重力和接触。

## 无法从文件名推断的架构事实

**C2 支撑布局（单坐标系硬装）。** 机械臂和工作台是固定在同一个可见振动地板上的*同级节点*，所有被驱动资产属于同一个 deck `SupportGroup`，位姿统一由 `q + c + R(l-c)` 生成。`arm_mount_xy_m / table_mount_xy_m` 与两个 7 mm dynamic clearance 已删除；机器人根/桌腿底只用显式 `assembly_clearance_m=0.5 mm` 装配公差。`src/vibench/supports.py` 是支撑位姿的**唯一写入点**；Stewart 平台、房间、桌面饰边等视觉件不在此列。Panda 使用 `fix_root_link=False`，重力已开启。

**仅用于视觉呈现的 Newton 形状与结构 pair filter。** Stewart 平台、房间、桌面饰边、阴影和腕部相机外壳会被渲染，但都带有 `UsdPhysics.CollisionAPI`，且 `collisionEnabled=false`。同一刚性支撑组内的结构连接（`panda_link0↔VibrationFloor`、`WorkTableLeg↔VibrationFloor`、`WorkTableLeg↔WorkTableTop`）通过 Newton Builder 的 `add_shape_collision_filter_pair` 在模型构建时排除。当前拓扑为 386 个 Newton 形状 / 29 个 MJWarp 几何体 / **339 个候选配对**。

**两种物理配置。** `official` = 1000 Hz × 5 个子步，`solref=(0.00060, 1.0)`，可计分。`training` = 240 Hz × 4 个子步，`solref=(0.0025, 1.0)`，明确**不可**计分。`official` 在低于 1000 Hz 时会拒绝启动。启动门是**离线全回合重放** `offline_support_travel_report()`：同一份 SupportGroup 几何表、真实 seed 波形，取真实最大子步行程；几何门限固定为 `0.05 × 8 mm = 0.40 mm`，穿透评分门槛固定为工件最小尺寸的 1%（sugar_box@0.75 约 0.336 mm）。`BenchmarkConfig.__post_init__` 还会拒绝短于两个求解器子步的 `solref` 时间常数。`SpectralVibration.reseed()` 为每个环境**和每个轴**派生独立随机流。

**碰撞体的真实信息来自运行时，而非表格。** `config.py` 中的 `YCB_DIMENSIONS_M` 仅用于仿真前的可行性检查。运行时，任务会读取实际转换后的 Newton 碰撞体 AABB，以放置工件并计算抓取几何信息。此前一个 21.6 mm 的穿透错误正是由于信任标称尺寸所致（物体自由下落了约 0.10 m）。同样，下降目标根据*实时*手指碰撞体的伸展范围计算——`panda_leftfinger` 碰撞体在指节原点下方延伸 53.85 mm，远高于缩放后的 sugar box，因此简单采用“物体顶面上方 4 mm”的目标会先撞到桌面。

**渲染约束（NewtonGL/ViewerGL）。** Albedo/roughness/metallic 有效。Normal maps、opacity 和 emissive 不会传入着色器；请将浮雕效果和 AO 烘焙进 RGB。USD point/area lights 不会被导入。Directional shadow mapping 虽然存在，但已特意禁用（在移动特写镜头中不稳定）。尝试修复视觉问题前，请先查看 `docs/newtongl_capabilities.md`。

**腕部相机。** 这是位于 `panda_hand/WristCamera` 下方的真实建模 D415-style 组件，具有固定的手眼外参（`WRIST_CAMERA_EYE_H/FORWARD_H/UP_H`）。视点、光轴和图像向上方向均由同一个 `panda_hand` quaternion 变换——`benchmark_rendering.py` 扩展了 NewtonGL，使其可接受完整的滚转自由度。它不会跟踪工件，不会相对世界坐标系保持稳定，也没有数字防抖。不要通过重新瞄准来“修复”抖动。

**视觉回归通过断言验证，而非目测。** `configs/visual_manifest.yaml` 声明预期的 prim 路径、父 prim、材质绑定和数量；`visual_manifest.py` 会生成特征事实和父级锚点审计（阈值：5 mm），`tests/test_visual_manifest.py` 同时断言数值和源码层面的父子模式。`configs/visual_regions.yaml` 为 `tools/visual_audit.py` 提供固定的像素 ROI。移动视觉 prim 或更改其父级时，必须更新 manifest 和 `docs/prim_anchor_audit.md`。

## 计分完整性规则

这些是项目的核心约束——违反这些约束会在不易察觉的情况下使结果失效。

- `grasp_assist` 默认**关闭**，正式计分不得依赖它（`configs/scenarios.yaml: official_score_requires_grasp_assist: false`）。`--no-grasp-assist` 只是为兼容旧版而保留的写法。
- 启用辅助后，要求双侧指尖接触力 >0.05 N 且持续 ≥4 个连续物理帧、手部与物体距离 <0.15 m，并且瞬时穿透量 <0.5 mm；穿透量超过 1.0 mm 时立即释放。所有这些信息都会记录在 metrics JSON 中。
- 高难度回合可以如实返回 `success=false`。**不得**为了复现较早的通过帧而重新启用辅助、放宽穿透阈值或延长超时。
- 失败原因是一级指标，不是需要隐藏的错误：`descend_table_contact`、`descend_contact_timeout`、`grasp_contact_timeout`、`grasp_contact_lost`、`grasp_z_guard_triggered`。
- 必须保留、不得掩盖的后端限制：当 `use_mujoco_contacts=True` 时，NativeCCD 会将所设定的 1 mm 接触裕量清零。JSON 会记录 `nativeccd_margin_honored=false`，而不是声称该裕量已经生效。
- 降低激励频率并提高位移，并不会让基准测试变得“更难但等价”——在加速度 RMS 固定时，`v = a/ω`，因此这会增加每个子步的移动距离和穿透量。`validate_impulsive_timestep()` 会对此进行防护。

## 已知的上游不稳定问题

Newton/Isaac 在模型构建过程中可能因 `malloc(): unaligned tcache chunk detected`、double-free 或没有任何 Python traceback 的纯 `SIGSEGV` 而中止——通常发生在仿真循环开始*之前*。这是上游问题，并非代码仓库自身的问题。请重试；如果某次运行未生成 JSON/MP4，则应将该产物视为未验证，而不能复用旧产物作为证据。

## 目录结构

- `src/vibench/`——库与 CLI。`cli.py`（命令行入口）、`paths.py`（项目根解析）、`config.py`（所有数值/资产 dataclass + 验证）、`vibration.py`、`supports.py`、`shaker.py`、`scene.py`（Newton cfg + 场景组装）、`arena.py`（房间）、`visual_assets.py`（USD UV 材质和细节）、`wrist_camera.py`、`benchmark_rendering.py`、`task.py`（仿真循环、观测值、指标）、`panel_task.py`/`panel*.py`、`controller.py`、`recording.py`、`diagnostics.py`、`visual_manifest.py`。
- `configs/`——`scenarios.yaml`（场景矩阵 + 评估/接触/控制器策略）、`assets.yaml`（资产来源、许可证、纹理 SHA-256）、`room.yaml`、`visual_manifest.yaml`、`visual_regions.yaml`。
- `docs/`——验证日志、视觉基线、锚点审计；`docs/reports/` 为权威实现说明与报告，`docs/prompts/` 为历史重构提示词。`out/`——生成的 MP4/PNG/JSON 产物。
