> **ShakeBench 当前场景（2026-09-23）**：`VibrationPickPlace` 提供抓放任务，`RingOnPeg` 提供从大到小的双环木质套柱任务；当前默认场景为 `world_fixed_rigid_table_v1`。Panda 不带移动底座，直接固定安装在岸边实体地基上；地基与岸边齐平，临坑三边的顶部和竖直壁均带黑色包边。
> IMU 世界模型计划的 B/C/D 阶段已完成，新场景尚未取得实验认证。Phase 06–08 的选型、发布、handoff、authority 与 evidence 链已删除，当前完整性校验只绑定 physics profile 与 geometry/scene/arena/support 资产（`python -m shakebench.utils.runtime_verifier --write` 可重建清单）。桌下 IMU、双相机数据采集和策略接入接口已提供，视频历史/预测器训练仍由独立模块负责。
> 运行与验证见 [世界固定机械臂场景](docs/world_fixed_arm_scene.md)。
> ShakeBench 专用代码与资源位于 shakebench/ 包，导入入口为 shakebench.make；robosuite/ 保留通用框架。策略接入见 shakebench/utils/websocket_policy.py（WebSocket 策略协议，与具体模型无关），Gamma=0 数据采集见 [LeRobot 采集说明](docs/lerobot_collection.md)。

## 套环任务（CPU 开发版）

机械臂从台面随机位置依次抓取蓝色大环、黄色小环，按从大到小的顺序套到中央圆头木杆上。两环环壁厚度为 25/21 mm，支持 Panda 一指入孔、一指贴外壁的夹取方式。每一阶段要求已放置的环位置正确、松手并稳定 0.5 秒，且后续环还未放在杆上；仅两个环全部完成后奖励 1。错误顺序可以通过移走提前放置的环纠正。阶段真值只在 metrics 中报告。

环内/外半径从大到小为 30/55、24/45 mm，均厚 16 mm。木杆为陡圆台，底端/顶端半径为 18/9 mm，总高 120 mm，球形头半径 12.5 mm，小于最小环的有效孔径。浅木纹底盘为无槽实心圆盘，半径 62 mm、厚 12 mm，比蓝环外缘宽 7 mm。两个环随机分布在桌面可操作区域，互不重叠，可接触底盘边缘。底板和杆固定在移动工作台上，复用振动、Panda、IMU 和共享 CPU rollout。

参考 [MetaWorld 的外观/碰撞分离](https://github.com/Farama-Foundation/Metaworld/blob/59fc34d7768af9785e4688c3e1db671424f4a6c3/metaworld/assets/objects/assets/assembly_peg.xml)，环保留 16 段碰撞体，采用独立圆角外观网格和仓库已有的浅木纹理。木杆使用 [ambientCG Wood 095](https://ambientcg.com/view?id=Wood095) 的 CC0 细木纹，并按圆台表面等比例展开；素材来源及校验值随贴图保存。底盘仅使用一个圆柱碰撞体；视觉网格无质量、无碰撞。无新增依赖。

```python
import numpy as np
from shakebench.environments.ring_on_peg import default_state
from shakebench.utils.rollout import ShakeBenchTaskEnv

task = ShakeBenchTaskEnv(default_state(), gamma=0.0, observation_source="contract")
try:
    observation, info = task.reset()
    observation, reward, terminated, truncated, info = task.step(np.zeros(7))
finally:
    task.close()
```

导入 `shakebench.environments.ring_on_peg` 后也可使用 `shakebench.make("RingOnPeg")`。共享策略评测入口：

```bash
python -m shakebench.scripts.evaluate \
  --task-module shakebench.environments.ring_on_peg \
  --states shakebench/models/assets/shakebench_ring_on_peg_states.json \
  --policy my_policy:make_policy --observation-source contract
```

套环状态文件显式记录桌面坐标系内双环/杆的位置、各环初始 yaw、激励和 IMU 种子；底盘以杆为中心。无显式 `ring_state` 的环境每次 reset 随机放置两个环，传入 `seed` 可重现序列；显式状态保持固定，便于重试和回放。旧状态会明确报错，避免以旧状态运行不同任务。开发状态不用于认证分数；GPU 批量采集尚不支持套环；CPU Oracle 支持按蓝、黄顺序自动套环。

随附状态池包含 20 个随机布局。生成更多布局（同种子可复现）：

```bash
python -m shakebench.scripts.generate_ring_states --output out/ring_states.json --count 100 --seed 42
```

采集时将 `--states` 指向生成的文件，每条状态对应一个布局，R 重试保持当前布局。

SpaceMouse＋键盘静态采集（LeRobot v2.1，双相机＋8D 本体状态＋7D 动作，不采集 IMU）。所有任务必须显式指定 `--task`，没有默认任务：

```bash
source /home/miracle04/.venvs/shakebench-lerobot/bin/activate
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m shakebench.scripts.collect_lerobot_spacemouse \
  --device spacemouse --task ring_on_peg \
  --output out/spacemouse_ring_stack --limit 20

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m shakebench.scripts.collect_lerobot_spacemouse \
  --device spacemouse --task pick_place \
  --output out/spacemouse_pick_place_static --limit 1
```

各任务独立读取 `shakebench/models/assets/collection_<task>.json`，其中定义状态池、注册模块、时程、操作灵敏度、图像尺寸和主相机。命令行同名参数优先，例如 `--horizon-steps 3000`；`--config /path/to/collection.json` 可替换整份配置，配置中的相对状态路径相对于该 JSON 所在目录。套环默认 2400 步、位移/旋转灵敏度 0.2/0.3；pick-place 默认 1200 步、灵敏度 1.0/1.0。`--limit` 是整个采集计划的 episode 数量，中断续采仍使用原计划。

Enter 开始；WASD 平移，Q/E 降低/抬高，SpaceMouse 旋转，空格切换夹爪；R 丢弃当前尝试并重试，Esc 退出。窗口显示已完成环数，双环成功后自动保存；超时也保存但不进入成功训练子集。每条轨迹都从选定状态重新开始，`--episodes-per-state` 控制重复次数。默认 `teleop` 物理配置、仿真 20 Hz。SpaceMouse 采集被中断后，使用相同参数再次运行会从 manifest 中的下一个状态继续；第一条轨迹尚未保存时只清理临时图像，不覆盖已保存 episode。

```bash
python -m shakebench.scripts.export_sft_subset \
  --dataset out/spacemouse_ring_stack --output out/spacemouse_ring_stack_sft
python -m shakebench.scripts.verify_collection \
  --dataset out/spacemouse_ring_stack_sft \
  --task-module shakebench.environments.ring_on_peg \
  --eval-assets shakebench/models/assets/shakebench_ring_on_peg_states.json
```

上述校验检查来源状态、格式、动作、图像和成功终止；若有独立评测状态池，须一并加入 `--eval-assets` 才能检查训练/评测隔离。套环自动采集使用独立的特权状态 Oracle，以 OSC 动作完成抓取、抬升和释放；失败尝试会标记为失败，成功子集仍由上述导出器筛选。

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m shakebench.scripts.collect_lerobot \
  --device oracle --task ring_on_peg --physics-profile teleop \
  --output out/oracle_ring_stack --limit 20
```

`teleop` 是较快、不可认证评分的物理配置；Oracle 读取仿真真值作为控制输入，但数据集图像仍为任务配置指定的 256×256。初始位置可能不可抓取或发生滑落，请以采集清单中的成功数量为准。

所有 IL 训练数据（CPU/GPU Oracle、SpaceMouse、SFT 子集）不包含 IMU 观测、统计或安装审计。底层 IMU 传感器、仿真接口和非 IL 实验能力保留，供后续阶段使用。

## 葡萄酒入架任务（CPU 开发版）

`PlaceWineAtRackLocation` 参考 [RLBench / PerAct 的原任务](https://github.com/MohitShridhar/RLBench/blob/peract/rlbench/tasks/place_wine_at_rack_location.py)，保留 `middle / left / right` 三个变体及到达目标后松手的语义。酒瓶初始竖立在桌面，机械臂需要将其横放进指定槽位。MuJoCo 酒架由前后两根弧形托梁和两根低矮侧脚组成，固定在移动工作台上；酒瓶复用仓库已有的 RoboCasa `wine_3` 网格及许可。此版本适配 ShakeBench 场景，不逐尺寸复刻 CoppeliaSim 的原模型。

酒架造型参考 [Classico 实木酒架](https://www.classico.co.nz/products/wooden-modular-wine-rack?variant=42478967324854) 的开放骨架与弧形承托结构，分别为瓶身和瓶颈设置托槽，配合圆润边缘、木榫及沿构件方向铺设的自然橡木纹。使用 [ambientCG Wood049](https://ambientcg.com/view?id=Wood049) 的 2K CC0 贴图；来源和许可记录在 `shakebench/models/assets/wine_rack_visual_sources.json`。

成功要求酒瓶的完整碰撞几何进入指定位置、瓶颈朝酒架 +x 方向且瓶轴偏差不超过 15°、前后两根托梁同时承重、与机械臂脱离接触，并连续保持 0.5 秒。错误槽位、悬空、竖放、反向及越界均不算成功；成功在 episode 内锁存。碰撞托槽与可视曲线一致，采用分段凸网格近似，木架与酒瓶接触的两个切向摩擦系数均为 0.30。所有位置检查在移动酒架坐标系中进行。环境复用 Panda、20 Hz、7D OSC 动作、振动、IMU 和公共 CPU rollout；开发状态不具备认证评分资格。

当前任务与状态格式均为版本 2，随附状态已重新生成；旧版箱式酒架状态会被拒绝，避免混用不同物理任务。

```python
import shakebench
from shakebench.environments.place_wine_at_rack_location import default_state

env = shakebench.make("PlaceWineAtRackLocation", task_state=default_state("left"))
try:
    observation = env.reset()
finally:
    env.close()
```

随附 30 条确定性状态，三个位置各 10 条。公共评测和 SpaceMouse 采集入口：

```bash
python -m shakebench.scripts.evaluate \
  --task-module shakebench.environments.place_wine_at_rack_location \
  --states shakebench/models/assets/shakebench_place_wine_at_rack_location_states.json \
  --policy my_policy:make_policy --observation-source contract

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m shakebench.scripts.collect_lerobot_spacemouse \
  --device spacemouse --task place_wine_at_rack_location \
  --output out/spacemouse_wine_rack --limit 30

python -m shakebench.scripts.generate_wine_rack_states \
  --output out/wine_rack_train_states.json --count 300 --seed 43 --split train
```

手动采集默认 2400 步，平移/旋转灵敏度 0.2/0.3，256×256 双相机。该任务尚未提供专用 Oracle 或 MJWarp 采集适配。

# robosuite

![gallery of_environments](docs/images/gallery.png)

[**[Homepage]**](https://robosuite.ai/) &ensp; [**[White Paper]**](https://arxiv.org/abs/2009.12293) &ensp; [**[Documentations]**](https://robosuite.ai/docs/overview.html) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Latest Updates

- [10/28/2024] **v1.5**: Added support for diverse robot embodiments (including humanoids), custom robot composition, composite controllers (including whole body controllers), more teleoperation devices, photo-realistic rendering. [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.0) [[documentation]](http://robosuite.ai/docs/overview.html)

- [11/15/2022] **v1.4**: Backend migration to DeepMind's official [MuJoCo Python binding](https://github.com/deepmind/mujoco), robot textures, and bug fixes :robot: [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.4.0) [[documentation]](http://robosuite.ai/docs/v1.4/)

- [10/19/2021] **v1.3**: Ray tracing and physically based rendering tools :sparkles: and access to additional vision modalities 🎥 [[video spotlight]](https://www.youtube.com/watch?v=2xesly6JrQ8) [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.3) [[documentation]](http://robosuite.ai/docs/v1.3/)

- [02/17/2021] **v1.2**: Added observable sensor models :eyes: and dynamics randomization :game_die: [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.2)

- [12/17/2020] **v1.1**: Refactored infrastructure and standardized model classes for much easier environment prototyping :wrench: [[release notes]](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.1)

-------

**robosuite** is a simulation framework powered by the [MuJoCo](http://mujoco.org/) physics engine for robot learning. It also offers a suite of benchmark environments for reproducible research. The current release (v1.5) features support for diverse robot embodiments (including humanoids), custom robot composition, composite controllers (including whole body controllers), more teleoperation devices, photo-realistic rendering. This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.

Data-driven algorithms, such as reinforcement learning and imitation learning, provide a powerful and generic tool in robotics. These learning paradigms, fueled by new advances in deep learning, have achieved some exciting successes in a variety of robot control problems. However, the challenges of reproducibility and the limited accessibility of robot hardware (especially during a pandemic) have impaired research progress. The overarching goal of **robosuite** is to provide researchers with:

* a standardized set of benchmarking tasks for rigorous evaluation and algorithm development;
* a modular design that offers great flexibility in designing new robot simulation environments;
* a high-quality implementation of robot controllers and off-the-shelf learning algorithms to lower the barriers to entry.

This framework was originally developed in late 2017 by researchers in [Stanford Vision and Learning Lab](http://svl.stanford.edu) (SVL) as an internal tool for robot learning research. Now, it is actively maintained and used for robotics research projects in SVL, the [UT Robot Perception and Learning Lab](http://rpl.cs.utexas.edu) (RPL) and NVIDIA [Generalist Embodied Agent Research Group](https://research.nvidia.com/labs/gear/) (GEAR). We welcome community contributions to this project. For details, please check out our [contributing guidelines](CONTRIBUTING.md).

**Robosuite** offers a modular design of APIs for building new environments, robot embodiments, and robot controllers with procedural generation. We highlight these primary features below:

* **standardized tasks**: a set of standardized manipulation tasks of large diversity and varying complexity and RL benchmarking results for reproducible research;
* **procedural generation**: modular APIs for programmatically creating new environments and new tasks as combinations of robot models, arenas, and parameterized 3D objects. Check out our repo [robosuite_models](https://github.com/ARISE-Initiative/robosuite_models) for extra robot models tailored to robosuite.
* **robot controllers**: a selection of controller types to command the robots, such as joint-space velocity control, inverse kinematics control, operational space control, and whole body control;
* **teleoperation devices**: a selection of teleoperation devices including keyboard, spacemouse and MuJoCo viewer drag-drop;
* **multi-modal sensors**: heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception;
* **human demonstrations**: utilities for collecting human demonstrations, replaying demonstration datasets, and leveraging demonstration data for learning. Check out our sister project [robomimic](https://arise-initiative.github.io/robomimic-web/);
* **photorealistic rendering**: integration with advanced graphics tools that provide real-time photorealistic renderings of simulated scenes, including support for NVIDIA Isaac Sim rendering.

## Citation
Please cite [**robosuite**](https://robosuite.ai) if you use this framework in your publications:
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
