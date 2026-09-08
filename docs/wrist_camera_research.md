# ShakeBench 腕部相机参考与接入建议

调研日期：2026-09-08。范围：robosuite 上游、LIBERO、robomimic 官方文档与源码。本文件记录设计依据；实际运行与测试结果由实现说明提供。

当前改造场景采用 `geometry_profile="direct_mount_v1"`。腕部相机已在该场景中验证：
它不改变 State 观测协议、场景物理参数或机械臂关节配置。

## 上游已经提供腕部安装

本仓库的 Panda 模型 `robosuite/models/assets/robots/panda/robot.xml` 已保留上游的 `eye_in_hand` 相机：它位于 `right_hand` body 下，采用 `mode="fixed"`、局部位置 `0.05 0 0`、四元数（wxyz）`0 0.707108 0.707108 0`、垂直视场角 `75` 度。上游源码明确将该相机用于沿末端方向观察。[robosuite Panda 模型](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/robots/panda/robot.xml)

MuJoCo 的 `fixed` 模式将相机固定在所属 body 的局部坐标系，因此相机会随腕部运动，而不是固定在世界中。相机沿自身负 Z 轴看向场景，正 Y 轴为图像上方。[MuJoCo camera 定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)

robosuite 在构造环境时选择相机，并将图像自动加入观测字典；标准键名使用 `<camera>_image`、`<camera>_depth`。Panda 单臂场景使用带实例前缀的 `robot0_eye_in_hand`，其 RGB 键为 `robot0_eye_in_hand_image`，这也与下述两个 benchmark 的实际调用一致。[robosuite Sensors](https://robosuite.ai/docs/modules/sensors.html)

## 操作 benchmark 约定

| 参考 | 官方实现 | 对 ShakeBench 的意义 |
| --- | --- | --- |
| LIBERO | `ControlEnv` 默认同时选择 `agentview` 和 `robot0_eye_in_hand`，两者默认 128×128，开启相机观测和离屏渲染。[环境封装源码](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py) | 外部视角与腕部视角并存，分别覆盖全局布局和末端交互。 |
| LIBERO 训练数据 | 默认 `use_eye_in_hand: true`；训练侧 `eye_in_hand_rgb` 映射到环境侧 `robot0_eye_in_hand_image`。[数据配置源码](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/configs/data/default.yaml) | 环境保留标准键名，训练适配层可以另设别名。 |
| robomimic | 官方图像提取示例使用 `--camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84`；添加 `--depth` 可同时提取深度。[数据处理文档](https://robomimic.github.io/docs/v0.4/datasets/robosuite.html) | 相机名称与分辨率、RGB/深度配置分离，可沿用现有数据接口。 |
| robomimic 元数据 | 数据转换器保存相机内参与外参，并专门处理 `eye_in_hand` 名称的相机；源码意图是将腕部外参表达到末端控制坐标系。[转换器源码](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py) | 若后续保存标定信息，应明确坐标系及变换方向；这里不直接照搬其矩阵计算。 |

## 本仓库接入建议

以下是结合上述来源和本地结构得出的工程建议，不是对其它 benchmark 的额外事实陈述。

1. 复用现有 `robot0_eye_in_hand` 及其上游安装位姿，避免再添加带质量或碰撞几何的相机刚体，保持已完成阶段的动力学配置稳定。
2. 在支持视觉的环境入口中提供 `agentview` 与腕部相机的默认组合，保留用户显式设置 `camera_names` 的能力。原始 RGB 键继续使用 `robot0_eye_in_hand_image`。
3. 保持现有纯状态观测 tier 的契约及无渲染执行路径；腕部视角可先在原生相机观测和演示中启用，不应偷偷将像素混入固定维数的状态向量。
4. 演示应能看到腕部画面，并支持与外部画面同时显示，以检查抓取时的目标可见性。展示画面转换与策略原始观测应分开处理，避免重复翻转像素。
5. 验证相机所属 body、局部安装位姿恒定、腕部运动后世界位姿变化，以及 RGB 内容确实更新；另外实际渲染检查初始化、接近物体和抓取阶段的遮挡与朝向。仅检查字典中存在键名不足以确认腕部视角可用。

## 实施与验证

- `VibrationPickPlaceCan` 的默认视觉相机列表已改为 `agentview`、`robot0_eye_in_hand`；保留显式单相机选择、分辨率和可选深度接口。默认仍不启用渲染，V0–V3 State 协议保持原样。
- Oracle 视频脚本支持 `--wrist-inset` 同步小窗及 `--camera robot0_eye_in_hand` 独立腕部视角；原生离屏缓冲区按请求尺寸扩展，支持 512×512 输出。使用示例见 [demos.md](demos.md#shakebench-wrist-camera)。
- 复现命令在修改前报 `KeyError: robot0_eye_in_hand_image`；接入后相机、演示、provider、privilege 四组测试共 24 项通过。随后增加 512×512 缓冲区处理，重新运行受影响的相机和演示组，5 项通过。测试使用真实 MuJoCo/EGL，覆盖 RGB/深度、随腕运动、图像更新、硬重置、显式单相机以及渲染不修改仿真状态或 State 观测。

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pytest \
  tests/test_shakebench_wrist_camera.py \
  tests/test_shakebench_oracle_demo_video.py \
  tests/test_shakebench_providers.py tests/test_shakebench_privilege.py -q
```

实录文件位于 `out/demo/shakebench_wrist_camera.mp4`，配套 JSON 保存运行参数与哈希，关键帧拼图为 `out/demo/shakebench_wrist_camera_contact_sheet.png`。V0、Gamma=0、`shakebench-dev-v0-000` 在 206 个控制步后成功；这是定性演示，不构成新的评分证据。检查了 0/2/4/6/8/10 秒关键帧：腕部画面随动作同步变化，罐体位于画面下部，抓取近距离存在夹爪遮挡。保留上游安装位姿与 75° 视场角，未增加物理质量、接触或关节。
