# 世界固定机械臂场景

本次实施已覆盖 `imu_world_model_migration_plan.md` 的 B 阶段和 C 桌下 IMU。统一振动模式、预测数据和 policy 接入尚未实施；完整迁移与全仓历史清理仍待后续阶段。

`VibrationPickPlace` 默认且仅接受 `world_fixed_arm_v1` 安装配置。`robot0_base` 与 `robot_support` 都是没有关节、没有 mocap 的 world 直接子节点；机械臂的七个操作关节保持正常。`worktable` 保留甲板下的六自由度隔振机构，目标容器跟随工作台，任务物体保留 freejoint。

机械臂不带移动底座，直接安装在与岸边连通、下接坑底的实体地基上。地基和机械臂基座都是 world 的无关节刚性子树，因此不会随甲板振动。地基范围为 X `[-1.225, -0.47]`、Y `[-0.42, 0.42]`、Z `[-0.765, 0]` m；顶面与岸边同高并使用同一环氧地坪材质。临坑三边增加与坑口相同颜色和高度的黑色 U 形顶部包边，凸台三面内壁从地坪一直到坑底全部使用原坑壁的深黑色。岸边原包边向开口内延长一个包边宽度，与凸台两侧包边完整搭接。对应岸边栏杆在 Y `[-0.46, 0.46]` m 留出开口，坑壁也按地基宽度切开，两端栏杆保留。

机械臂底部增加 `260 × 260 × 16` mm 金属方盘，四角带垫圈、螺栓头和内孔细节，表现为锚固于凸台地面的安装盘。方盘中心为 `[-0.605, 0, 0.008]` m，底面 z=0、顶面 z=0.016 m；机械臂基座安装高度同步为 z=0.016 m，方盘完整位于凸台范围内。盘体与凸台共享世界固定刚体。栏杆开口两端 X=-1.385、Y=±0.46 m 各补一根黄黑立杆，通过底脚板和四颗锚栓落地，上端与原横杆端点接合。

甲板保持 `1.25 × 1.40` m，中心为 `[0.35, 0]` m，厚度保持 0.08 m；范围为 X `[-0.275, 0.975]` m。工作台保持 `0.65 × 0.60 × 0.06` m，桌面中心向甲板内侧移动至 `[0.11, 0, 0.299]` m，左侧留边由 30 mm 增至 60 mm。

甲板下方的 Stewart 安装点、外筒、基座盘和底座同步移位并缩窄，避免与实体地基相交；腿长包络重新检查。隔振器、接触、时间步及现有集中质量驱动参数保持不变；驱动的有效质量和惯量不是由可视甲板体积自动推导的，新尺寸尚未取得物理重标定认证。

凸台、金属方盘、锚栓与黑色内壁的几何源文件是 [shakebench_robot_support.xml](../robosuite/models/assets/arenas/shakebench_robot_support.xml)。这些固定几何直接用 MJCF 的 `body/geom` 定义；Python 只读取文件和切换显示，不再生成或重设它们的尺寸与位置。坑壁、原栏杆和开口立杆直接保存在 [场景 MJCF](../robosuite/models/assets/arenas/shakebench_world_additions.xml) 中。

[当前安装配置](../robosuite/models/assets/shakebench_geometry_world_fixed_arm_v1.json)（schema 2）通过 `robot_support_mjcf` 与 `robot_support_mjcf_sha256` 绑定支撑 MJCF，不再重复存储支撑几何。修改该 XML 后，需要更新它的 SHA-256 和安装配置的 `payload_sha256`。支撑 MJCF 可原生 `<include>`；其材质 `shakebench_floor_slab` 与 `shakebench_frame_metal` 由主 arena 定义。

机械臂和工作台的放置、甲板／隔振器参数仍由安装与场景配置控制；既有参数化场景部件仍通过环境组装成最终 MJCF。本次迁移范围是新增的固定凸台组件，并未将整个 robosuite 环境重写为纯静态 XML。

普通 `reset()` 在物体落位后，以零激励把六个隔振器坐标直接置为带载静平衡：
`springref` 只补偿 32 kg 工作台自重，物体重量与力臂扭矩由
`static_equilibrium_offset` 解析给出，物体随后随工作台做同一刚体变换，避免跌落。
剩余的甲板软约束和接触残差由有界物理收敛吸收，保持机器人初始关节位姿：
甲板及隔振器广义速度须小于 `1e-5`（m/s 或 rad/s）、广义加速度小于 `0.05`
（m/s² 或 rad/s²）；物体线速度小于 `0.001 m/s`、角速度小于 `0.02 rad/s`，
上述条件须连续满足 0.1 秒，最多计算 5 秒，未收敛时报错。当前场景约 1 秒收敛。
接触求解存在微小残差，这里的稳定是有限容差内的稳定，不代表所有加速度逐位为零。
随后速度清零、仿真时钟归零并重置 IMU、驱动记录与任务指标；收敛过程不进入 episode
及 IMU 采样记录。`reset_settle_duration_s` 记录初始化模拟时长。显式确定性状态恢复
保留所指定的状态，不执行这一步收敛。IMU 初始滤波值来自实际物理状态。

构造与运行：

```python
import numpy as np
from robosuite.environments.manipulation.vibration_pick_place import VibrationPickPlace
from robosuite.utils.shakebench_calibration import level_scale_for_gamma
from robosuite.utils.shakebench_excitation import build_excitation_program

program = build_excitation_program(
    seed=42, level_scale=level_scale_for_gamma(0.15, seed=42)
)
env = VibrationPickPlace(
    robots="Panda", task={"object_id": "food_can", "surface_id": "metal"},
    deck_trajectory=program, observation_tier=None,
    use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
    initialization_noise=None,
)
try:
    env.reset()
    for _ in range(20):
        env.step(np.zeros(env.action_dim))
finally:
    env.close()
```

这里复用现有激励构造器；`mode × Gamma` 接口属于后续 D 阶段。`canonical`、`direct_mount_v1` 安装选项及被替代的直接安装配置已删除。旧 V0–V3 观测依赖甲板上的机械臂和 IMU，当前场景明确拒绝这些选项。当前环境始终从 `worktable/table_imu_site` 采集唯一桌下 IMU，公开 `table_imu_window`、`table_imu_dt_s` 和 `table_imu_timestamps_s`；XYZ 与安装四元数可由环境参数校准。当前上下文只记录固定的世界到基座关系和工作台到 IMU 外参，不再发布固定 `deck_to_robot_base` 外参或虚假的基座 IMU 安装信息。

运行场景审计并生成预览：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m robosuite.scripts.shakebench_scene_preflight \
  --output out/world_fixed_arm/scene_preflight.json \
  --preview out/world_fixed_arm/scene.png
```

无图形环境时去掉 `--preview` 即可。审计入口使用当前任务，不再依赖旧直接安装 authority。报告包含安装配置身份、编译拓扑、支撑面误差、运动间隙、初始接触与显隐物理不变性。

验证命令：

```bash
python -m pytest tests/test_shakebench_world_fixed_arm.py \
  tests/test_shakebench_scene.py tests/test_shakebench_scene_finishes.py -q
```

当前验收覆盖：三物体 × 两表面；Gamma=0 的静态运行和 seed=42、Gamma=0.15 的一秒六轴振动；基座世界位姿不变；工作台非零响应；甲板位置跟踪误差小于 2 mm；reset；带关节限位的抓取点和释放点逆运动学（位置误差小于 2 mm、姿态误差小于 0.01 rad）；六轴隔振传递与静态预载；支撑穿透负例；显隐物理一致性。

间隙扫描包含名义姿态、12 个单轴极值与 64 个平移／转角组合极值，共 77 个姿态，范围为各轴 ±1.5 mm、±0.003 rad。支撑与运动部件的配对不受名义距离筛选限制。分离的包围盒给出保守正间距；其余配对使用 MuJoCo 距离查询。该检查是有限采样，不代表任意 Gamma 或连续运动空间的证明。逆运动学验证为端点可达性，不等同于完整抓放策略成功率。

2026-09-12 实体地基布局预检：场景结构、间隙、初始接触及显隐物理一致性四项通过。77 个姿态中，地基与运动部件的最小保守间隙为 **191.14 mm**（甲板边缘到地基）；Stewart 腿长范围为 0.747–0.816 m。机器可读报告位于 `out/world_fixed_arm/scene_preflight.json`，预览位于 `out/world_fixed_arm/scene.png`。此数值仅适用于当前实体地基布局。当前实体地基布局的 39 项定向测试全部通过：安装／任务／隔振与外观 20 项，场景 17 项，以及新增的甲板偏移边界、倾转腿长回归 2 项。

2026-09-13 金属方盘与开口立杆补充验收：本轮 7 项定向测试通过，覆盖方盘承托实际机器人安装轮廓、方盘落地、锚栓装配、立杆上下衔接、原有外观、代表任务的静态／振动与抓放可达性，以及错误装配负例。更新后的场景预检四项通过；这轮定向验收不代表重跑前述全套测试。

2026-09-13 固定凸台组件改为 MJCF 源文件后，8 项针对性验证及场景预检四项通过，包括原生 `<include>` 编译、运行环境与 XML 一致、资产变更 hash 校验、外观和显隐物理一致性。迁移前后 11 组编译数组（刚体位姿、质量和惯量，几何位姿、尺寸、颜色、材质索引及碰撞标志）逐字节一致，命名物理签名一致。

场景审计通过仍不授予实验资格：`scoreable=False`，`experiment_certification=pending`。旧状态文件、oracle/GPU 采集入口和实验结果未在本阶段迁移或重新认证，不能用于声明新场景的性能。
