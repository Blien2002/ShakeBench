# ViBench

**ViBench** 是一个面向具身智能操作的独立 benchmark：Franka Panda 机械臂需要在基座与工作台同时受到六自由度振动激励的条件下，完成拾取—搬运—放置任务。项目基于 Isaac Lab + Newton/MJWarp，使用资产化的 Franka Panda、YCB 工件、参数化 Stewart 振动台与工业实验室房间，输出可复现的仿真轨迹、多视角视频和结构化评估 JSON。

项目完全自包含：配置、纹理、生成器、测试与评估代码都在本仓库内；它不导入、不依赖桌面上的任何其他项目目录。唯一的外部运行时是另行安装的 Isaac Lab（含 Newton）环境，通过 `ISAACLAB_ROOT` 指定。

## 核心场景

- **C2 振动支撑布局**：Panda 与工作台是同一块可见振动地板上的同级节点，分别安装在测点 `(0.75, -0.45)` 与 `(-0.75, 0.45)`；六轴激励经完整 SE(3) 映射，旋转会在两侧产生真实不同的局部平动。
- **Stewart 6-3 振动台**：参数化几何、6 腿解析解与 12 段伸缩缸视觉模型；纯视觉硬件不进入 MJWarp 接触表。
- **资产化工作场景**：官方 Franka Panda USD、YCB `cracker_box / sugar_box / tomato_soup_can / mustard_bottle`、带真实碰撞壁的目标盒、带 UV 纹理的酚醛工作台。
- **实验室房间**：`6.00 x 5.00 x 3.00 m` 冷灰工业实验室、环氧地坪、下沉地坑与后侧防护栏，作为全局视觉上下文。
- **腕部相机**：固定手眼外参的 D415 风格物理模型，光心/光轴/画面上方向统一由 `panda_hand` 姿态变换，不做数字防抖或工件追踪。
- **两种物理档位**：`official`（1000 Hz × 4 子步，可评分）与 `training`（240 Hz × 4 子步，明确不可评分）。
- **诚实评分**：`grasp_assist` 默认关闭；接触、穿透、失败原因与辅助使用情况全部写入 metrics JSON。

## 快速开始

```bash
cd /path/to/ViBench

# 若 Isaac Lab 不在 ~/IsaacLab-3.0，先指定其路径
export ISAACLAB_ROOT=/path/to/IsaacLab-3.0

# 运行默认 16 s official 档基准回合（不录像，输出 metrics）
./run.sh

# 录像并输出结构化指标
./run.sh --record --vibration spectral --seed 17 --workpiece sugar_box \
  --output out/vibench_wrist_camera.mp4 \
  --metrics-output out/vibench_wrist_camera.json

# 运行测试
./run_tests.sh
```

`run.sh` 会解析 Isaac Lab 环境、设置 `PYTHONPATH`，并默认导出 `PXR_WORK_THREAD_LIMIT=1` 以保证 OpenUSD 场景构建的确定性。其他仓库工具可用 `./run_python.sh` 在同一个 Isaac Lab venv 中执行：

```bash
./run_python.sh tools/visual_audit.py <mp4|png> \
  --time-s 6.0 --regions-config configs/visual_regions.yaml \
  --compare docs/visual_baseline.json --json-output out/audit.json
```

可选地把本包以 editable 方式安装进 Isaac Lab venv，获得 `vibench` 控制台命令与 `python -m vibench`：

```bash
"$ISAACLAB_ROOT/.venv/bin/python" -m pip install -e .
vibench --help
```

## 常用 CLI 参数

- `--scenario NAME`：加载 `configs/scenarios.yaml` 中的场景，例如 `clean_baseline`、`z_sine_control`、`safe_full_6dof_demo`。
- `--vibration off|sine|spectral`（默认 `spectral`）
- `--workpiece cracker_box|sugar_box|soup_can|mustard_bottle`
- `--workpiece-scale 0.75`、`--seed 17`
- `--physics-profile official|training`：默认 `official`；低于 1000 Hz 的 official 请求会被拒绝启动。
- `--grasp-assist`：显式启用演示辅助；正式评分与默认录制均不启用。
- `--record`、`--output PATH`、`--metrics-output PATH`、`--camera-preset main|stewart_side`、`--no-overlays`。

## 项目结构

```
ViBench/
├── run.sh / run_tests.sh / run_python.sh   # Isaac Lab venv 启动器
├── pyproject.toml                          # vibench 包与 vibench CLI
├── configs/                                # 场景矩阵、资产/房间/视觉回归清单
├── assets/textures/                        # 仓库内生成的确定性纹理
├── src/vibench/                            # benchmark 库与 CLI
│   ├── cli.py                              # 命令行入口
│   ├── scene.py / task.py / controller.py  # 仿真场景、任务循环、脚本控制器
│   ├── vibration.py / mounting.py / shaker.py
│   └── arena.py / visual_assets.py / wrist_camera.py / recording.py ...
├── scripts/                                # run_demo 兼容入口与 NewtonGL 探针
├── tools/                                  # 视觉审计、纹理生成器
├── tests/                                  # pytest 单元与回归测试
├── docs/                                   # 验证记录、视觉基线、锚点审计
│   ├── reports/                            # 当前实现说明、基准报告
│   └── prompts/                            # 历史重构提示词
└── out/                                    # 生成的 MP4/PNG/JSON（git 忽略）
```

## 输出与评分

- 最终演示：[out/benchmark_v2_wrist_camera_fixed.mp4](out/benchmark_v2_wrist_camera_fixed.mp4)
- 指标 JSON：[out/benchmark_v2_wrist_camera_fixed.json](out/benchmark_v2_wrist_camera_fixed.json)
- 场景矩阵：[configs/scenarios.yaml](configs/scenarios.yaml)
- 资产清单：[configs/assets.yaml](configs/assets.yaml)
- 视觉回归清单：[configs/visual_manifest.yaml](configs/visual_manifest.yaml)
- 权威实现说明：[docs/reports/current_implementation.md](docs/reports/current_implementation.md)
- 实现报告：[docs/reports/benchmark_v2_report.md](docs/reports/benchmark_v2_report.md)
- 第四轮验证与穿透基线：[docs/fourth_round_validation.md](docs/fourth_round_validation.md)、[docs/penetration_baseline.json](docs/penetration_baseline.json)、[docs/penetration_official_final.json](docs/penetration_official_final.json)
- 父变换审计：[docs/prim_anchor_audit.md](docs/prim_anchor_audit.md)

## 评分边界

默认录屏和正式评分都不启用 `grasp_assist`。控制器按工件实际 collider 的短轴调整俯视偏航，下降由指尖接触触发停止，并持续执行 z 保护。若显式使用 `--grasp-assist`，只有双指连续接触且当前穿透小于 0.5 mm 才允许捕获相对位姿；保持期间穿透超过 1.0 mm 会立即解除。JSON 公开记录双指确认、辅助使用/拒绝/解除、最大穿透深度、接触对和时刻。困难回合可能诚实返回 `success=false`，不应通过恢复辅助或放宽穿透门槛改写结果。

YCB 数据集遵循 CC BY 4.0，使用时请引用 Calli et al., *The YCB Object and Model Set*, 2015。Isaac Sim/Isaac Lab 资产遵循 NVIDIA 对应资产条款。当前活动纹理由仓库内确定性生成器产生；旧 Poly Haven 纹理仅保留用于来源追溯。逐项校验值见 [assets/textures/README.md](assets/textures/README.md)。
