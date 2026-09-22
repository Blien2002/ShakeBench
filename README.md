> **ShakeBench 当前场景（2026-09-22）**：`VibrationPickPlace` 提供抓放任务，`RingOnPeg` 提供单环套柱任务；当前默认场景为 `world_fixed_rigid_table_v1`。Panda 不带移动底座，直接固定安装在岸边实体地基上；地基与岸边齐平，临坑三边的顶部和竖直壁均带黑色包边。
> IMU 世界模型计划的 B/C/D 阶段已完成，新场景尚未取得实验认证。Phase 06–08 的选型、发布、handoff、authority 与 evidence 链已删除，当前完整性校验只绑定 physics profile 与 geometry/scene/arena/support 资产（`python -m shakebench.utils.runtime_verifier --write` 可重建清单）。桌下 IMU、双相机数据采集和策略接入接口已提供，视频历史/预测器训练仍由独立模块负责。
> 运行与验证见 [世界固定机械臂场景](docs/world_fixed_arm_scene.md)。
> ShakeBench 专用代码与资源位于 shakebench/ 包，导入入口为 shakebench.make；robosuite/ 保留通用框架。策略接入见 shakebench/utils/websocket_policy.py（WebSocket 策略协议，与具体模型无关），Gamma=0 数据采集见 [LeRobot 采集说明](docs/lerobot_collection.md)。

## 套环任务（CPU 开发版）

机械臂抓起橙色环，套到蓝色竖直柱上，并松手放在桌面。复用工作台、振动驱动、Panda 和桌下 IMU；环使用原生分段中空碰撞体。成功要求真实孔径内穿柱、近水平落桌、无机器人接触、相对桌面稳定，连续满足 0.5 秒后锁存；奖励为成功时 1，否则 0。

环内/外半径为 25/45 mm，厚 16 mm；柱半径 12 mm，高 80 mm。孔内判定逐面检查分段环的内壁，允许 0.5 mm 接触容差。参考 [RLBench 的目标检测](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/tasks/insert_onto_square_peg.py)、[MetaWorld 的圆环装配](https://github.com/Farama-Foundation/Metaworld/blob/59fc34d7768af9785e4688c3e1db671424f4a6c3/metaworld/envs/sawyer_assembly_peg_v3.py) 和本仓库 robosuite 的 `HollowCylinderObject`；ShakeBench 额外按移动桌面坐标检查松手后的稳定停留。

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
  --states shakebench/models/assets/shakebench_ring_on_peg_states_v1.json \
  --policy my_policy:make_policy --observation-source contract
```

状态文件显式记录桌面坐标系内的环/柱位置、环初始 yaw、激励和 IMU 种子。开发状态不用于认证分数；GPU 批量采集和现有 pick-place oracle 尚不支持套环。

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
