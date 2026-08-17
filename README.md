# Vibration Benchmark v2

独立的 Isaac Lab + Newton/MJWarp 振动操作项目。它不导入旧 RM75 工程，旧工程可继续作为基线。当前场景包含视觉无碰撞的 Stewart 六自由度振动台、下沉式实验室地坑和防护栏；C2 支撑使用完整 SE(3) 刚体映射。

## 场景资产

- 机械臂与夹爪：Isaac Lab 官方 `FRANKA_PANDA_HIGH_PD_CFG` 对应的 Franka Panda USD。
- 工作台：带明确厚边、深色方管框架、下拉杆和带螺栓地脚板的工业实验台；桌面覆盖深色酚醛树脂 UV 纹理，碰撞仍只使用稳定的原始桌面/四腿凸体。
- 振动台：参数化 6-3 Stewart 几何、12 段伸缩缸、万向节、惯性块与空气弹簧；纯视觉硬件不进入 MJWarp 接触表。
- 房间：`6.00 x 5.00 x 3.00 m` 冷灰工业实验室，采用环氧地坪、浅灰墙板、深色踢脚线和可读地坑；安全黄护栏只保留在后侧，主相机侧保持开放。全局外观不随环境复制且不进入 MJWarp。
- 目标区：浅蓝白收纳盒，由底板和四面真实碰撞壁组成，不是无碰撞视觉标记。
- 打光：参考 ManiSkill 的中性 `0.3` 环境光与双白色方向光结构，并沿用 robosuite TableArena 默认关闭硬投影阴影的设置；主相机与腕部相机共享同一渲染标定。
- 工件：YCB `cracker_box`、`sugar_box`、`tomato_soup_can`、`mustard_bottle` 物理 USD。
- 接触求解：官方档 1000 Hz × 4 子步（默认），训练档 240 Hz × 4；左右指尖另有抓取接触与下降保护传感器。
- 腕部相机：带碰撞外壳和安装支架的 D415 风格物理模型，固定在 `panda_hand` 的夹爪上侧。光心、光轴和画面上方向均由同一个手眼外参变换，不做工件追踪或世界坐标防抖。

场景采用旧版 C2 布置：Panda 和工作台彼此独立，二者的底座均直接安装在同一块可见振动地板上，不存在“机械臂放在振动台上、振动台再放到桌上”的层级。六自由度地板激励按 C2 的机械臂测点 `(0.75, -0.45)` 与工作台测点 `(-0.75, 0.45)` 映射为两个局部支撑运动。默认激励是带固定种子的六轴窄带随机谱，而不是单一正弦。

## 运行

```bash
cd /home/miracle04/Desktop/vibration_benchmark_v2
./run.sh --record --vibration spectral --workpiece sugar_box \
  --output out/benchmark_v2_wrist_camera_fixed.mp4
```

常用参数：

- `--vibration off|sine|spectral`
- `--seed 17`
- `--workpiece cracker_box|sugar_box|soup_can|mustard_bottle`
- `--workpiece-scale 0.75`（当前默认值）
- `--physics-profile official|training`：默认 `official`；训练档不用于正式评分。
- `--physics-hz`：显式覆盖频率；official 低于 1000 Hz 会拒绝启动。
- `--grasp-assist`：显式启用演示辅助；默认、正式评分和默认录制均关闭。
- `--no-grasp-assist`：旧命令兼容写法，当前默认本来就是关闭。
- `--metrics-output out/result.json`

首次运行会从 NVIDIA 官方 S3 资产根加载 USD，之后由 Omniverse 客户端缓存。默认根固定为已实际验证可用的 Isaac 5.0 快照，避免本机 Isaac Lab 3.0 beta 指向尚未发布的 6.0 路径。

## 输出与测试

- 最终演示：[out/benchmark_v2_wrist_camera_fixed.mp4](out/benchmark_v2_wrist_camera_fixed.mp4)
- 腕部视角关键帧：[out/wrist_fixed_approach.png](out/wrist_fixed_approach.png)、[out/wrist_fixed_grasp.png](out/wrist_fixed_grasp.png)、[out/wrist_fixed_transfer.png](out/wrist_fixed_transfer.png)、[out/wrist_fixed_release.png](out/wrist_fixed_release.png)
- 指标 JSON：[out/benchmark_v2_wrist_camera_fixed.json](out/benchmark_v2_wrist_camera_fixed.json)
- 场景矩阵：[configs/scenarios.yaml](configs/scenarios.yaml)
- 资产清单：[configs/assets.yaml](configs/assets.yaml)
- 视觉回归清单：[configs/visual_manifest.yaml](configs/visual_manifest.yaml)
- 穿透基线与第四轮验证：[docs/penetration_baseline.json](docs/penetration_baseline.json)、[docs/penetration_official_final.json](docs/penetration_official_final.json)、[docs/fourth_round_validation.md](docs/fourth_round_validation.md)
- 父变换审计：[docs/prim_anchor_audit.md](docs/prim_anchor_audit.md)
- 当前完整实现说明：[CURRENT_IMPLEMENTATION.md](CURRENT_IMPLEMENTATION.md)
- 第二轮视觉基线/最终结果：[docs/visual_baseline.json](docs/visual_baseline.json)、[docs/visual_final.json](docs/visual_final.json)
- 第三轮视觉审计：[docs/third_round_final_visual.json](docs/third_round_final_visual.json)、[docs/third_round_final_frame.png](docs/third_round_final_frame.png)
- 实现报告：[BENCHMARK_V2_REPORT.md](BENCHMARK_V2_REPORT.md)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=src:/home/miracle04/IsaacLab-3.0/source/isaaclab_assets \
  /home/miracle04/IsaacLab-3.0/.venv/bin/python -m pytest -q tests
```

## 评分边界

默认录屏和正式评分都不启用 `grasp_assist`。控制器按工件实际 collider 的短轴调整俯视偏航，下降由指尖接触触发停止，并持续执行 z 保护。若显式使用 `--grasp-assist`，只有双指连续接触且当前穿透小于 0.5 mm 才允许捕获相对位姿；保持期间穿透超过 1.0 mm 会立即解除。JSON 公开记录双指确认、辅助使用/拒绝/解除以及最大穿透深度、接触对和时刻。困难回合可能诚实返回 `success=false`，不应通过恢复辅助或放宽穿透门槛改写结果。

YCB 数据集遵循 CC BY 4.0；使用时请引用 Calli et al., *The YCB Object and Model Set*, 2015。Isaac Sim/Isaac Lab 资产遵循 NVIDIA 对应资产条款。当前活动的环氧地坪、工业墙板、酚醛桌面和台面孔阵列均由仓库内确定性生成器产生；旧 Poly Haven 纹理仅保留用于来源追溯。逐项校验值见 [assets/textures/README.md](assets/textures/README.md)。
