# ShakeBench 当前实现说明

更新日期：2026-08-18
项目目录：`ShakeBench`
当前后端：Isaac Lab + Newton/MJWarp（`use_mujoco_contacts=True`）

## 1. 文档范围与当前结论

本文只描述**当前代码中已落地并可运行**的实现。项目不导入旧 RM75/MuJoCo 项目，也不依赖仓库外项目目录；唯一外部运行时是 `ISAACLAB_ROOT` 指定的 Isaac Lab 环境。

当前模型状态：

- 支撑模型为**单坐标系硬装甲板**：Panda、工作台、桌腿、目标盒同属一个 deck `SupportGroup`，虚拟测点 `arm_mount_xy_m / table_mount_xy_m` 和两个 7 mm `dynamic_clearance` 已删除。
- **official CLI 默认 `C2_CLITE`：1000 Hz × 4 子步、solver iterations=50**，平台/工作台由 mocap-weld 按子步驱动；`--support-config C2` 保留为 1000 Hz × 5 子步的 kinematic 外层写入模式。
- 重力已开启，Panda 为 `fix_root_link=False` 的浮动根，根位姿由 deck 组写入。
- 安全门为**离线全回合重放**，几何门限固定 `0.40 mm`；穿透评分门槛固定为工件最小尺寸的 1%。
- 旧模型产物已作废，状态见 [docs/baseline_status.md](../baseline_status.md)。

当前验证结论：

- 38 项测试全部通过。
- official 1 s 谱探针不再出现 `robot_link<->platen` 伪接触；1 s settle 最大穿透 0.242 mm（`workpiece<->worktable`）。
- official 完整 16 s 回合目前为诚实失败（`grasp_z_guard_triggered` / `grasp_contact_timeout`，取决于相位与录制开销），最大穿透约 0.88–0.96 mm；`support_geometry_valid=true`。
- 数值地板未满足计分资格（≤ 0.112 mm）。**official 当前不可计分**，直到支撑子步注入和重力后的控制器整定完成。

## 2. 总体架构

```text
VibrationConfig ──> SpectralVibration ──> deck 六轴 q / qd / qdd
                                              │
                                              v
                                      SupportGroup("deck")
                              platform, robot, worktable,
                              table_legs, target bin
                                              │
                              ScriptedPickPlaceController
                                              │
                         Newton/MJWarp 推进动力学与接触
                                              │
                        关节/腕力、指尖接触、腕部 RGB、任务指标
                                              │
                                      NewtonGL 视频合成
```

每个外层步的调用链：

1. `SpectralVibration.sample()` 产生 deck 中心六轴位移/速度/加速度；
2. `write_support_groups()` 用单一刚体变换把 deck 组所有成员写入仿真；
3. `ScriptedPickPlaceController.command()` 生成 7 轴机械臂和 2 轴夹爪目标；
4. Newton/MJWarp 推进动力学与接触；
5. `observation()` 汇总位姿、接触、腕部力/力矩、相机外参；
6. `BenchmarkRecorder` 合成主视图、腕部画中画和遥测叠加。

`offline_support_travel_report()` 在场景构建前使用同一份 `SupportGroup` 几何表重放完整回合，执行启动安全门。

主要入口：[src/shakebench/cli.py](../../src/shakebench/cli.py)（`scripts/run_demo.py` 为兼容启动器）。

## 3. 物理后端

| 参数 | 值 |
|---|---|
| 官方档 | 1000 Hz × 4 子步 + C2_CLITE（iterations=50），`solref=(0.00060 s, 1.0)` |
| 训练档 | 240 Hz × 4 子步，`solref=(0.0025 s, 1.0)`，不可评分 |
| 积分器 | `implicitfast` |
| 主迭代 / 线搜索迭代 | 80 / 24（可 `--solver-iterations` 覆盖做实验） |
| CCD 迭代 | 50 |
| 摩擦锥 | `elliptic` |
| 默认摩擦 | `material_mu=1.5` |
| 恢复系数 | 0 |
| NativeCCD margin | 请求 1.0 mm，实际清零并写入 JSON |

结构性 pair filter 在模型构建时安装，排除同组机械连接对：

```text
panda_link0  ↔ VibrationFloor
WorkTableLeg ↔ VibrationFloor
WorkTableLeg ↔ WorkTableTop
```

当前接触拓扑：**386 Newton shapes / 29 MJWarp geometries / 339 candidate pairs**。

## 4. 六轴振动算法

`VibrationConfig.mode`：`off` / `sine` / `spectral`（默认）。

`spectral` 模式每轴使用窄带随机相位谱线：

```math
q_a(t)=\sum_k A_{a,k}\sin(\omega_{a,k}t+\phi_{a,k}),
\qquad A_{a,k}=RMS_a\sqrt{2/N_a}
```

- 频率在带宽内均匀铺设并加确定性抖动；
- 随机流由 `[seed, env_id, axis_index]` 派生，seed 可复现、环境/轴独立；
- 速度、加速度为解析导数，不使用有限差分；
- 前 `0.75 s` 使用五次 smoothstep，位置/速度/加速度两端 C² 连续。

默认频带：

| 轴 | 中心频率 | RMS | 谱线数 |
|---|---:|---:|---:|
| `tx` | 18 Hz | 0.50 mm | 12 |
| `ty` | 13 Hz | 0.25 mm | 10 |
| `tz` | 32 Hz | 1.50 mm | 12 |
| `rx` | 8 Hz | 4.00 mrad | 12 |
| `ry` | 11 Hz | 2.00 mrad | 10 |
| `rz` | 6 Hz | 1.20 mrad | 8 |

### 4.1 启动安全门（离线重放）

`offline_support_travel_report()` 在启动时按当前 seed 重放完整回合，对每个 `SupportGroup` 成员计算：

- `max_substep_travel_mm`：子步网格最大真实行程，参与门限判定；
- `max_v_dt_mm`：若速度写入被忽略时的 teleport 界；
- `max_half_a_dt2_mm`：一阶保持积分的二阶残差。

几何门限固定：

```text
max_substep_displacement_m = 0.05 × 8 mm = 0.40 mm
```

default seed=17 official 实测约 0.262 mm（含成员 `bound_radius_m` 包围半径，最坏成员为 platform 角点）；training 约 1.203 mm 被拒绝。任何官方 seed 超限时，治理动作是整体提高子步数，不允许删除 seed 或放宽门限。

## 5. 支撑模型

### 5.1 硬装 deck（当前唯一默认模型）

所有被驱动资产属于 `SupportGroup("deck")`，成员位姿由组唯一变换生成：

```text
p = q[:3] + c + R(l - c)，  c = platform_center = (0, 0, 0.04)
```

- `l` 只能是可见布局坐标；
- 组的 `rotation_anchor` 是组属性，不是成员属性；
- 组内成员速度使用精确 Euler-rate → world angular velocity 映射；
- `assembly_clearance_m=0.0005` 只用于避免同组结构面 `dist≈0` 的浮点抖动，不是动态间隙。

实现位于 [src/shakebench/supports.py](../../src/shakebench/supports.py)。`task.py` 与 `panel_task.py` 都调用同一个 `write_support_groups()`；Stewart 平台视觉件继续由 deck 组 platform 成员的位姿驱动。

### 5.2 C2_CLITE（official 默认）

`--support-config C2_CLITE` 把振动地板和工作台改为动态刚体，由 fixed-root mocap driver + WELD equality constraint 驱动；mocap 轨迹在 solver 子步网格上直接写 `mjw_data.mocap_*`，默认 `clite_mocap_update_decimation=2`。桌腿、目标盒与 Panda 根仍为 kinematic 轨迹写入。

五 seed 数值地板标定（1000 Hz × 4 子步、iterations=50）为 0.067–0.102 mm，全部低于 0.112 mm 计分资格线。1 s 官方回合墙钟约 50 s，16 s 约 13 分钟，仍偏慢但作为 official 候选配置使用。

### 5.3 legacy 测点模型

旧的虚拟测点运动映射已随 `mounting.py` 一起删除。其数值问题记录在 [docs/c2_mount_inconsistency.md](../c2_mount_inconsistency.md)。

## 6. 场景资产

### 6.1 Franka Panda

- Isaac Lab `FRANKA_PANDA_HIGH_PD_CFG`，Isaac 5.0 `panda_instanceable.usd`；
- `fix_root_link=False`，根节点由 deck 组驱动；
- **重力已开启**；当前控制器在重力开启后的完整回合仍需重新整定。

### 6.2 振动地板与 Stewart 平台

- 可见地板：`1.60 × 1.10 × 0.08 m` kinematic cuboid，参与碰撞；
- 参数化 6-3 Stewart 平台为纯视觉：12 段伸缩缸 + 工件动态阴影，`collision_enabled=False`；
- 每帧解析求解 6 腿长度与 12 段位姿。

### 6.3 工作台

- 桌面 `0.65 × 0.60 × 0.06 m`，四条 `0.055 × 0.055 m` 物理桌腿；
- 桌腿高度按 `table_bottom − floor_top` 精确计算，底部与地板保持 `assembly_clearance_m`；
- 深色酚醛树脂 UV 桌面与外观层为确定性生成。

### 6.4 目标盒与工件

- 目标盒：浅蓝白色四墙收纳盒，底板与墙均有碰撞；
- 工件：四种 YCB `Axis_Aligned_Physics` 资产，默认 `sugar_box@0.75`，只受重力与接触；
- 工件初始高度、抓取几何和下降目标使用运行时读取的 Newton collider AABB，不信任标称尺寸表。

### 6.5 面板操作任务

`--task panel_operation` 在工作台上挂载固定斜面控制台：knob / lever / button 三控件，支持随机或显式操作序列。该任务复用同一 deck 支撑模型；`C2_CLITE` 与面板任务不组合。

### 6.6 腕部相机

- 位于 `panda_hand/WristCamera`，固定手眼外参：
  `eye=(0.097766738, 0, -0.045291247)`，`forward=(-0.466662058, 0, 0.884435709)`，`up=(0.884435709, 0, 0.466662058)`；
- 光心、光轴和画面向上方向都由 `panda_hand` 姿态变换，不做工件追踪或数字防抖；
- 渲染为 384 × 240，75° 垂直 FOV，延迟创建。

## 7. 控制器

`ScriptedPickPlaceController`：

- DLS 差分 IK，命令坐标系为振动机械臂根坐标系；
- 工件短轴偏航抓取；
- 阶段机：settle → approach → descend → grasp → lift → transfer → place → release → retreat；
- 所有笛卡尔/夹爪目标逐步限速；
- 下降碰桌、抓取超时、双侧接触丢失、z-guard 触发均返回明确失败原因；
- `grasp_assist` 默认关闭；显式开启时要求双侧指尖接触 >0.05 N、持续 ≥4 物理帧、手物距离 <0.15 m、穿透 <0.5 mm，保持中穿透 >1.0 mm 立即释放。
- 重力开启后的整定：`grasp_z_guard_margin_m=0.002`；双侧接触计数在单帧丢失时衰减而不是清零；夹爪闭合速度 0.003→0.006 m/s、grasp timeout 2.5→4.0 s；descend/grasp 阶段 XY 跟踪使用 `arm_linear_speed_m_s`，只有 Z 轴保持慢速下降。
- 当前完整回合仍未稳定建立双侧接触；实验参数 `--solver-iterations 120` 可把 settle 地板从 0.242 降到约 0.178 mm，但 grasp 阶段接触仍会单侧丢失。

控制器当前是**状态型参考策略**，直接读取工件/目标真值位姿，不是视觉策略。

## 8. 传感器与观测

### 8.1 传感器

- `JointWrenchSensorCfg`：腕部 `panda_link7` 六维力/力矩；
- 工件接触传感器：过滤桌面、目标盒、双指；
- 左右指尖接触传感器：分别过滤工件；
- 左右指尖下降保护接触传感器：过滤工件或桌面；
- 腕部 RGB：Newton ViewerGL，固定手眼外参。

### 8.2 观测键

| 键 | 内容 |
|---|---|
| `joint_pos` | 全部关节位置 |
| `ee_pose_w / ee_pose_b` | 末端世界/根坐标位姿 |
| `left/right_finger_pose_w` | 双指世界位姿 |
| `finger_center_b` | 指尖中心根坐标 |
| `root_pose_w` | 机器人根位姿 |
| `workpiece_pose_w/b`、`target_pose_w/b` | 工件/目标位姿 |
| `vibration_q/qd/qdd` | deck 六轴运动 |
| `mount_delta_z` | 机器人基座与工作台在甲板表面的 Z 向差 |
| `shaker_leg_lengths_m` | Stewart 六腿长度 |
| `wrist_force_b / wrist_torque_b` | 腕部力/力矩 |
| `left/right_finger_contact_n` | 双指接触力 |
| `left/right_finger_descent_contact_n` | 下降保护接触力 |
| `bilateral_contact_streak` | 连续双侧接触帧数 |
| `grasped` | grasp-assist 保持状态 |
| `penetration_mm` | 当前最深穿透 |
| `wrist_camera_eye/target/up_w` | 腕部相机外参 |

## 9. 任务判定与指标

`EpisodeMetrics` 主要字段：

- `lifted`：工件高度超过目标高度 0.10 m；
- `placed`：已抬升、水平误差 <0.07 m、已释放；
- `success = lifted and placed`；
- `max_penetration_mm/pair/shapes/t`：全回合最深穿透；
- `penetration_frames_over_0p5mm`：穿透 >0.5 mm 的帧占比；
- `max_wrist_force_n / max_wrist_torque_nm`；
- `bilateral_contact_confirmed`、`grasp_assist_used`；
- `ee_tracking_error_rms_m / max_ee_tracking_error_m`：deck 系下控制器指令与实际 EE 的偏差；
- `support_geometry_valid`：每步扫描全部 active contacts，任一结构性禁止接触对出现穿透即置 false；
- 诚实失败原因：`descend_table_contact`、`grasp_contact_timeout`、`grasp_contact_lost`、`grasp_z_guard_triggered` 等。

评分门槛规则（固定，不随 seed/谱型调整）：

```text
score_penetration_threshold_mm = 0.01 × 工件最小 collider 尺寸(mm)
```

sugar_box@0.75 时约 `0.336 mm`。official 只有在“只振动不操作”的数值地板 ≤ 门槛/3 时才有计分资格；当前尚未满足。

## 10. 视觉与录制

- 1280 × 720 主视图 + 384 × 240 腕部画中画；
- 遥测：控制阶段、seed、振动曲线、`mount_delta_z`、双指接触、穿透、`support_geometry_valid`；
- 输出 H.264 MP4 与结构化 metrics JSON；
- 最新录制：`out/shakebench_stage_a_latest.mp4`（6.1 s，183 帧，诚实失败）。

## 11. 配置与 CLI

### 11.1 配置文件

| 文件 | 用途 |
|---|---|
| `src/shakebench/config.py` | 数值、资产、振动与任务 dataclass |
| `configs/scenarios.yaml` | 场景矩阵、物理档、接触策略、评分策略 |
| `configs/assets.yaml` | 资产来源、许可、纹理 SHA-256 |
| `configs/room.yaml` | 房间布局与视觉样式 |
| `configs/visual_manifest.yaml` | 可见特征、父 prim、材质断言 |

### 11.2 CLI 参数

```text
--scenario NAME
--task pick_place|panel_operation
--panel-sequence KNOB,LEVER,BUTTON
--panel-seed INT
--record / --output PATH / --camera-preset main|stewart_side|panel_review
--no-overlays
--workpiece cracker_box|sugar_box|soup_can|mustard_bottle
--workpiece-scale FLOAT
--vibration off|sine|spectral
--sine-axis / --sine-amplitude / --sine-frequency-hz
--spectral-scale FLOAT
--vibration-axes tx,ty,tz,rx,ry,rz
--seed INT / --episode-s FLOAT / --num-envs INT / --device DEVICE
--physics-profile official|training / --physics-hz INT
--solver-substeps INT
--contact-solref-timeconst FLOAT
--solver-iterations INT
--support-config C2|C2_CLITE
--gripper-closing-speed FLOAT
--grasp-timeout-s FLOAT
--grasp-assist / --no-grasp-assist
--metrics-output PATH
```

`run.sh` / `run_tests.sh` / `run_python.sh` 负责解析 Isaac Lab venv 与项目路径。

## 12. 测试与验证

- 测试入口：`./run_tests.sh`，当前 **38 passed**。
- 覆盖：seed 复现与轴独立随机流、ramp 与谱、**支持组刚体一致性（成员间距 / 共享 quat / 精确 ω 与四元数数值微分对拍）**、replay 与仿真波形一致性、Stewart 几何/行程、纹理哈希、相机外参、父锚点审计、视觉 manifest、重放安全门、旧几何回归。

当前验证状态与产物：

| 项目 | 结果 |
|---|---|
| official 1 s 谱探针 | 最大穿透 0.242 mm，`workpiece<->worktable`，`robot_link<->platen` 消失 |
| official 16 s 回合 | `support_geometry_valid=true`；`lifted=true` 后 `grasp_z_guard_triggered` / `grasp_contact_timeout` |
| 数值地板（普通 C2，5 seed） | 最大 0.259 mm，未达计分资格 |
| 数值地板（C2_CLITE 官方默认 4sub/50it，5 seed） | 0.067–0.102 mm，全部 ≤ 0.112 mm 资格线 |

详见 [docs/baseline_status.md](../baseline_status.md)。

## 13. 复现当前版本

```bash
cd ShakeBench
./run.sh --record \
  --physics-profile official \
  --episode-s 16 \
  --vibration spectral \
  --seed 17 \
  --workpiece sugar_box \
  --workpiece-scale 0.75 \
  --output out/shakebench_stage_a_latest.mp4 \
  --metrics-output out/shakebench_stage_a_latest.json
```

困难回合允许并应保留 `success=false`；不得用 `--grasp-assist` 或放宽门槛制造通过结果。

## 14. 源码模块索引

| 模块 | 主要职责 |
|---|---|
| `config.py` | 全局数值、振动频带、资产、面板与几何门限 |
| `vibration.py` | 六轴正弦/随机谱、解析导数、离线重放安全门 |
| `supports.py` | SupportGroup、支撑位姿唯一写入点、结构 pair filter |
| `shaker.py` | Stewart 参数化几何、六腿解析解与视觉刚体 |
| `scene.py` | Newton 配置、Panda/YCB/Stewart/桌面/盒子/面板/传感器装配 |
| `task.py` | pick-place 仿真循环、支撑驱动、观测、接触门控与指标 |
| `panel_task.py` / `panel.py` / `panel_controls.py` / `panel_controller.py` | 面板操作任务 |
| `controller.py` | DLS 差分 IK、状态机与夹爪接触闭合 |
| `arena.py` / `visual_assets.py` | 房间、纹理、外观层与阴影 |
| `wrist_camera.py` | 腕部相机物理模型与 RGB 传感器 |
| `benchmark_rendering.py` / `recording.py` | 打光、相机滚转、MP4 与遥测 |
| `diagnostics.py` / `visual_manifest.py` | 接触拓扑、穿透探针、视觉事实审计 |
| `cli.py` | CLI、回合执行、启动门、JSON 输出 |

## 15. 已知边界与尚未实现内容

- official 当前**不可计分**：普通 C2 数值地板未满足 D2 资格，完整抓取回合在重力开启后失败。
- C2_CLITE 官方默认配置五 seed 地板全部达标，但 16 s 墙钟约 13 分钟；计时显示约 75% 时间在 `NewtonManager._step_solver`，后续需要上游/底层优化。
- 控制器是状态型参考策略，不是视觉端到端策略。
- 没有 RL/模仿学习训练代码、Gym 注册或数据集导出器。
- 腕部相机只有 RGB，没有深度/分割/噪声模型。
- 视频为滚动振动时域曲线，尚无实时 FFT/PSD。
- `use_mujoco_contacts=True` 的 NativeCCD 会把 1 mm authored margin 清零，JSON 公开记录。
- Newton 原生初始化存在进程级 `SIGSEGV`/double-free/无 traceback 退出，通常发生在模型构建前；无 JSON/MP4 产物的运行不得复用旧产物作为证据。
- 腕部力/力矩未与真实 Panda 传感器标定，不能用于实机载荷结论。
