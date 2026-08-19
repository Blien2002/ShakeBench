# ViBench 当前实现说明

更新日期：2026-08-18  
项目目录：`ViBench`
当前后端：Isaac Lab + Newton/MJWarp（启用 MuJoCo 接触生成）

## 1. 文档范围与当前结论

本文说明项目中**已经落地并可运行**的资产、算法、传感器、任务功能、渲染录制与验证代码。项目是独立工程，不导入旧 RM75/MuJoCo 项目的 Python 模块；旧版本的六轴随机谱思想已在本项目内重新实现。

第四轮已经加入可回归的穿透探针、官方/训练双物理档、真实 collider 尺寸初始化、短轴偏航抓取、接触触发下降、assist 穿透门控、父变换审计和视觉资产清单。当前默认是 1000 Hz × 5 子步且 `grasp_assist=False`；困难回合允许诚实返回 `success=false`。

下面的 14.334 s `success=true` 结果是第三轮历史视觉演示，用于证明当时的录制链路与资产渲染可工作；它启用了辅助保持，**不能当作第四轮纯物理成绩**：

| 项目 | 当前结果 |
|---|---:|
| 视频 | 1280 × 720，30 FPS，430 帧，14.334 s |
| 振动 | 六轴窄带随机谱，seed=17 |
| `lifted` | `true` |
| `placed` | `true` |
| `success` | `true` |
| 双指接触确认 | `true` |
| 最终水平误差 | 0.05092 m |
| 回合起始接触基线 | 385 Newton shapes / 29 MJWarp geometries / 348 candidate pairs |
| 6.0 s 中央 ROI 视觉审计 | IQR 127 / std 59.81 / 窄带占比 43.90% |
| 第三轮区域色温跨度 | 45.20（要求不高于 50） |
| 相邻大表面亮度差 | 27.65 / 63.94（要求不低于 25） |

演示文件：[out/benchmark_v2_wrist_camera_fixed.mp4](out/benchmark_v2_wrist_camera_fixed.mp4)  
机器可读结果：[out/benchmark_v2_wrist_camera_fixed.json](out/benchmark_v2_wrist_camera_fixed.json)

阶段 A 硬装重构后接触拓扑实测为 386 Newton shapes / 29 MJWarp geometries / **339 candidate pairs**（结构性 pair filter 排除 9 对）。V0 改造前基线为 21.633 mm（自由落体撞桌面），详见 [docs/penetration_baseline.json](docs/penetration_baseline.json)；当前验证边界见 [docs/fourth_round_validation.md](docs/fourth_round_validation.md)。

## 2. 总体架构

```text
VibrationConfig ──> SpectralVibration ──> 中心六轴 q / qd / qdd
                                             │
                                             v
                                     C2 多测点运动映射
                                      │               │
                                      v               v
                               Panda 浮动根       工作台/目标盒
                                      │               │
                                      └──── 物理接触 ──┘
                                             │
                              Differential IK + 双指控制
                                             │
                       关节力/力矩、指尖接触、腕部 RGB、任务指标
                                             │
                                      NewtonGL 视频合成
```

每个仿真步的实际调用链为：

1. `SpectralVibration.sample()` 产生地板中心的位移、速度和加速度；
2. `c2_support_motions()` 计算机械臂侧和工作台侧的局部运动；
3. `VibrationBenchmarkTask._write_supports()` 写入地板、Panda 根、工作台、桌腿和目标盒状态；
4. `ScriptedPickPlaceController.command()` 生成 7 轴机械臂和 2 轴夹爪目标；
5. Newton/MJWarp 推进动力学与接触；
6. `observation()` 汇总状态、接触、六维力和相机外参；
7. `BenchmarkRecorder` 合成主视角、腕部 RGB、振动曲线和任务状态。

主要入口是 [src/vibench/cli.py](src/vibench/cli.py)（兼容启动器为 [scripts/run_demo.py](scripts/run_demo.py)），公共配置位于 [config.py](src/vibench/config.py)，场景装配位于 [scene.py](src/vibench/scene.py)。

新增 `support_config="C2_CLITE"` 可选模式（CLI：`--support-config C2_CLITE`）：振动平台与工作台改为动态刚体，由 fixed-root mocap driver + WELD equality constraint 驱动，mocap 按 solver 子步更新；桌腿、目标盒与 Panda 浮动根暂保留原 kinematic 轨迹写入。该模式为实验性质，不计入 official 评分。

## 3. 已实现资产

### 3.1 Franka Panda 机械臂与夹爪

| 属性 | 实现 |
|---|---|
| 资产来源 | Isaac Lab 官方 `FRANKA_PANDA_HIGH_PD_CFG` |
| USD | NVIDIA Isaac 5.0 `panda_instanceable.usd` |
| 机械臂 | 7 个 Panda 旋转关节 |
| 夹爪 | `panda_finger_joint1/2` 双指夹爪 |
| 碰撞与关节限制 | 沿用官方 USD/Isaac Lab 配置 |
| 基座形式 | 从固定根改为 floating root，由 C2 振动状态驱动 |

代码实现：

- [scene.py 的 `_franka_cfg()`](src/vibench/scene.py#L67) 复制官方配置、切换 USD 根并开启接触传感器；
- `fix_root_link=False` 暴露根部六自由度，`VibrationBenchmarkTask` 每步写入其根位姿和速度；
- [task.py 的关节与刚体索引初始化](src/vibench/task.py#L36) 自动解析机械臂、双指、手部和末端刚体索引。

### 3.2 公共振动地板与 C2 布置

地板是尺寸 `1.60 × 1.10 × 0.08 m`、质量配置值 300 kg 的运动学刚体。Panda 和工作台彼此独立，不存在“机械臂放在工作台上”的父子层级。两者共享同一个 deck `SupportGroup`：组内所有成员的位姿由甲板中心六轴运动经单一刚体变换生成：

```text
p = q[:3] + c + R(l - c)，  c = (0, 0, 0.04)
```

虚拟测点 `arm_mount_xy_m / table_mount_xy_m` 已删除；机器人基座、工作台、桌腿和目标盒都使用可见布局坐标，0.5 mm `assembly_clearance_m` 只作为结构装配公差。

代码实现：

- [scene.py 的 `platform`](src/vibench/scene.py#L98) 定义可见、带碰撞的运动学地板；
- [config.py 的 `BenchmarkConfig`](src/vibench/config.py#L87) 定义平台和任务布局；
- [supports.py 的 `write_support_groups()`](src/vibench/supports.py) 是支撑位姿的唯一写入点。

振动地板下方增加了参数化 6-3 Stewart 平台。第三轮把基座铰点改为 `0.85 × 0.60 m` 半轴椭圆、动平台铰点改为 `0.62 × 0.40 m` 半轴椭圆；缸筒半径为 50 mm，动平台铰点位于台面底面以下并保留至少 10 mm 余量。`ShakerGeometryCfg` 配置椭圆铰点、成对角间距、缸筒与活塞杆、行程、空气弹簧及惯性块外观；每帧解析求解 6 条腿的长度和 12 段刚体位姿。第 13 个跟随刚体承载工件动态接触阴影，与 Stewart 视觉集合一起更新。所有这些硬件与阴影均为 `collisionEnabled=false` 的纯视觉 Newton shape，不进入 MJWarp 接触表。代码见 [shaker.py](src/vibench/shaker.py)。
- [task.py 的 `_write_supports()`](src/vibench/task.py#L133) 把同一中心运动映射并写入各资产。

### 3.3 独立深色工业工作台

工作台由一块 `0.65 × 0.60 × 0.06 m` 的桌面和四条 `0.055 × 0.055 m` 桌腿组成。桌面、桌腿均具有独立 box 碰撞体；桌面上附加确定性生成的深色酚醛树脂 UV 表面。碰撞禁用的外观层包括深色四边包边、方管、前后下拉杆、交叉杆、四块方形地脚板、16 个螺栓头和接地阴影。桌腿的物理 box 仍按 `table_bottom - floor_top` 精确计算高度，顶面与桌面底面重合。

代码实现：

- [scene.py 的工作台与桌腿定义](src/vibench/scene.py#L130)；
- [make_scene_cfg()](src/vibench/scene.py#L247) 根据桌面尺寸自动计算桌腿高度和位置；
- [spawn_textured_table_surface()](src/vibench/visual_assets.py#L151) 在桌面上生成带 UV 的纹理薄层。

### 3.4 浅色目标收纳盒

目标区不是无碰撞标记，而是浅蓝白色收纳盒：

- 外尺寸：`0.18 × 0.16 m`；
- 底板厚度：`0.012 m`；
- 四面墙厚度：`0.008 m`；
- 墙高：`0.035 m`；
- 底板和四面墙均有碰撞几何。

代码实现：

- [scene.py 的 `target`](src/vibench/scene.py#L202) 定义运动学底板；
- [spawn_shallow_bin_walls()](src/vibench/visual_assets.py#L190) 参数化生成四面实体碰撞墙；
- 目标盒随工作台侧 C2 支撑运动，不固定在世界坐标中。

### 3.5 YCB 工件

当前支持四种 NVIDIA/Isaac Sim `Axis_Aligned_Physics` YCB 资产：

| CLI 名称 | YCB 资产 |
|---|---|
| `cracker_box` | `003_cracker_box.usd` |
| `sugar_box` | `004_sugar_box.usd` |
| `soup_can` | `005_tomato_soup_can.usd` |
| `mustard_bottle` | `006_mustard_bottle.usd` |

工件保留重力、刚体和物理碰撞，默认统一缩放为 `0.75`，也可用 `--workpiece-scale` 调整。

代码实现：

- [config.py 的 `YCB_ASSETS`](src/vibench/config.py#L62) 提供名称映射；
- [scene.py 的 `workpiece`](src/vibench/scene.py#L182) 定义刚体、摩擦、接触间隙和求解迭代；
- [make_scene_cfg() 的资产替换](src/vibench/scene.py#L303) 按命令行选择 USD 与缩放。

### 3.6 冷灰工业实验室、环氧地坪与地坑

房间采用 Arena 风格的“布局与样式分离”：三面浅灰工业墙板、冷灰环氧地坪和深色踢脚线由配置生成。房间尺寸为 `6.00 × 5.00 × 3.00 m`。振动台位于 `2.05 × 1.55 × 0.78 m` 下沉式地坑内；地坑使用非纯黑层级与格栅保持可读性，后侧保留安全黄护栏，主相机侧完全开放。

实现包含两层视觉路径：

1. 参数化大尺度墙板、踢脚线、地坑和设备几何，作为渲染器无关的确定性后备；
2. 确定性 UV 纹理表面，提供无木板接缝的环氧树脂变化与工业墙板细节。

代码实现：

- [configs/room.yaml](configs/room.yaml) 保存房间布局、冷灰色阶和 UV 重复参数；
- [load_room_arena_cfg()](src/vibench/arena.py#L62) 读取样式配置；
- [spawn_room_arena()](src/vibench/arena.py#L115) 生成四片式地板、地坑、后侧护栏、三墙、工业 UV 表面、踢脚线及中景设备；
- [author_textured_quad()](src/vibench/visual_assets.py#L55) 生成 0.2 mm 封闭 UV 薄网格，并显式关闭碰撞，使纹理表面只参与 NewtonGL 渲染。
- 房间固定生成在 `/World/RoomArena`，不随 `num_envs` 复制。

### 3.7 本地纹理资产

| 用途 | 当前文件 | 来源/许可 |
|---|---|---|
| 冷灰环氧地坪 | `epoxy_floor_cool_gray_1k.jpg` | 项目确定性生成器，无木板接缝 |
| 浅灰工业墙板 | `industrial_wall_light_gray_1k.jpg` | 项目确定性生成器，大尺度板缝与细微漆面颗粒 |
| 深色酚醛桌面 | `phenolic_bench_dark_1k.jpg` | 项目确定性生成器，中性低重复树脂纹理 |
| platen 孔阵列 | `platen_threaded_holes_1k.jpg` | 项目确定性生成器，7 × 11 倒角孔且无分区接缝线 |

纹理已固化到 [assets/textures](assets/textures)，离线运行不需要重新下载。来源、处理方法和哈希见 [assets/textures/README.md](assets/textures/README.md) 与 [configs/assets.yaml](configs/assets.yaml)。

### 3.8 腕部相机物理模型

腕部相机不只是一个虚拟视角。项目在 `panda_hand/WristCamera` 下生成了：

- 刚性安装支架；
- D415 风格相机外壳；
- 前面板；
- RGB 镜头和红外镜头；
- 物理碰撞几何；
- 独立 `OpticalFrame`。

相机安装在夹爪上侧并位于腕部法兰轮廓外，固定内倾 27.8°。当前手眼外参为：

```text
eye_H     = ( 0.097766738, 0, -0.045291247 ) m
forward_H = (-0.466662058, 0,  0.884435709 )
up_H      = ( 0.884435709, 0,  0.466662058 )
```

代码实现：

- [wrist_camera.py 的外参常量](src/vibench/wrist_camera.py#L19)；
- [spawn_wrist_camera_assembly()](src/vibench/wrist_camera.py#L80) 生成物理模型；
- [wrist_camera_frame_from_hand()](src/vibench/wrist_camera.py#L130) 用 `panda_hand` 四元数同时变换光心、光轴和画面上方向；
- [BenchmarkNewtonGlPerspectiveVideo.update_camera_frame()](src/vibench/benchmark_rendering.py#L61) 为 NewtonGL 补齐完整相机滚转自由度；
- 相机不追踪工件、不固定在世界坐标中，也不使用数字防抖。

## 4. 物理后端与接触求解

当前并非“零 MuJoCo 依赖”配置。物理框架是 Isaac Lab + Newton，求解器配置为 `MJWarpSolverCfg`，并设置 `use_mujoco_contacts=True` 使用 MuJoCo 接触生成路径。

主要参数：

| 参数 | 值 |
|---|---:|
| 物理频率 | 1000 Hz 官方档；240 Hz 训练档 |
| Newton 子步 | 官方 5；训练 4 |
| 积分器 | `implicitfast` |
| 主迭代 | 80 |
| 线搜索迭代 | 24 |
| 摩擦锥 | `elliptic` |
| CCD 迭代 | 50 |
| 默认摩擦系数 | 1.5（任务配置） |
| 恢复系数 | 0 |
| 接触 `solref` | 官方 `(0.00060 s, 1.0)`；训练 `(0.0025 s, 1.0)`，JSON 公开记录 |
| 请求 contact margin | 1.0 mm；NativeCCD 路径实际清零并公开记录 |

对应代码：[make_sim_cfg()](src/vibench/scene.py#L34)。这一配置优先保证高频基座运动下的接触稳定性，而不是宣称与某台真实机器人完成动力学标定。

## 5. 六轴复杂振动算法

### 5.1 支持的激励模式

`VibrationConfig.mode` 支持：

- `off`：六轴严格为零；
- `sine`：指定单轴正弦，用于基准检查；
- `spectral`：带种子的六轴窄带随机谱，作为默认复杂振动。

配置入口：[VibrationConfig](src/vibench/config.py#L41)。

### 5.2 随机谱合成

每个轴由若干随机相位谱线组成：

```math
q_a(t)=\sum_{k=1}^{N_a} A_{a,k}\sin(\omega_{a,k}t+\phi_{a,k})
```

其中频率在配置带宽内均匀铺设并加入小幅随机抖动，相位由 episode seed 决定。谱线幅值使用：

```math
A_{a,k}=RMS_a\sqrt{2/N_a}
```

位移、速度和加速度不是数值差分，而是对谱线解析求导。实现位于 [SpectralVibration](src/vibench/vibration.py#L13)。随机流使用 `[seed, env_id, axis_index]` 派生，因此同一 seed 可复现，不同并行环境、不同活动轴彼此独立。

默认频带：

| 轴 | 中心频率 | RMS | 谱线数 |
|---|---:|---:|---:|
| `tx` | 18 Hz | 0.50 mm | 12 |
| `ty` | 13 Hz | 0.25 mm | 10 |
| `tz` | 32 Hz | 1.50 mm | 12 |
| `rx` | 8 Hz | 4.00 mrad | 12 |
| `ry` | 11 Hz | 2.00 mrad | 10 |
| `rz` | 6 Hz | 1.20 mrad | 8 |

启动门 `offline_support_travel_report()` 按当前 seed 的解析波形离线重放完整回合，对 `SupportGroup` 每个成员计算真实最大速度，给出子步行程、外层 `v·dt` 与 `0.5·a·dt²` 三项。几何门限固定为 `0.05 × 8 mm = 0.40 mm`（与激励无关）；default seed=17 的 official 1000 Hz × 5 重放子步行程约 0.232 mm，training 240 Hz × 4 约 1.203 mm，仍被拒绝。评分穿透门槛固定为工件最小 collider 尺寸的 1%（sugar_box@0.75 约 0.336 mm）。

### 5.3 平滑启动包络

前 `0.75 s` 使用五次 smoothstep：

```math
r(u)=10u^3-15u^4+6u^5
```

代码同时计算 `r`、`r'` 和 `r''`，按乘积法则作用于 `q/qd/qdd`，从而避免零时刻位移或加速度突变。实现位于 [SpectralVibration._ramp()](src/vibench/vibration.py#L63)。

### 5.4 C2 多测点映射

地板中心运动记为 `[tx, ty, tz, rx, ry, rz]`。位于安装向量 `r` 的测点使用完整刚体变换：

```math
p_i=t+Rr-r
```

其中 `R` 由完整欧拉角构造，并保留该 benchmark 既有的 pitch 符号约定。因此旋转激励会在机械臂侧和工作台侧产生不同的局部平动；两者不是复制同一条 Z 轴曲线。4 mrad 时新旧近似的垂向差小于 1 um，50 mrad 时旧近似的全位置误差超过 900 um。实现位于 [mounting.py](src/vibench/mounting.py)。

### 5.5 支撑状态写入

`_support_state()` 将局部欧拉角转为四元数，并计算刚体安装点的线速度：

```math
v_i=v_0+\omega\times (Rr_i)
```

随后 `_write_supports()` 分别驱动：公共振动地板、Panda 浮动根、桌面、四条桌腿、目标盒，以及由当前动平台姿态解析得到的 12 段 Stewart 伸缩缸。工件不被直接写入振动轨迹，而是通过重力及与桌面的接触自然响应。对应代码：[task.py](src/vibench/task.py#L122)。

## 6. Pick-and-place 控制算法

### 6.1 差分逆运动学

参考控制器使用 Isaac Lab `DifferentialIKController`：

- 命令类型：绝对末端 pose；
- IK 方法：DLS（damped least squares）；
- 输出：Panda 7 个关节的位置目标；
- 命令坐标系：振动机械臂根坐标系；
- 末端姿态：从实时工件姿态提取较短水平主轴并组合俯视四元数（wxyz），确保手指沿可夹短边闭合。

对应代码：[ScriptedPickPlaceController](src/vibench/controller.py#L23)。

### 6.2 状态机

| 阶段 | 时长 | 夹爪目标 | 功能 |
|---|---:|---:|---|
| `settle` | 1.2 s | 0.040 m | 初始化与稳定 |
| `approach` | 1.8 s | 0.040 m | 以 0.15 m/s 上限到工件上方约 0.08 m |
| `descend` | 最长 2.0 s | 0.040 m | 以 0.06 m/s 上限跟踪安全预抓高度；碰工件或到位后停止，碰桌失败 |
| `grasp` | 最长 2.0 s | 0.012 m | 每指 0.005 m/s 闭合并等待双侧接触；超时失败 |
| `lift` | 1.8 s | 0.012 m | 抬升 |
| `transfer` | 2.2 s | 0.012 m | 移动至目标盒上方 |
| `place` | 1.8 s | 0.012 m | 下降放置 |
| `release` | 1.0 s | 0.040 m | 张开夹爪 |
| `retreat` | 1.5 s | 0.040 m | 末端退回 |

接近、下降和抓取阶段使用仿真真值工件位姿与实际 Newton collider 尺寸更新目标，因此这是**状态型参考策略**，不是仅依赖腕部相机的视觉策略。启动时会检查最小水平尺寸是否超过 75 mm 可用开口；下降高度同时读取 Newton 指尖 collider 的完整向下伸出量，确保指尖底部与桌面至少保留 1 mm。所有笛卡尔目标和夹爪目标均有逐步速度限制，消除了阶段切换时的位姿跳变。下降碰桌、抓取超时或搬运时双指接触持续丢失都会返回明确失败原因。相关逻辑见 [controller.py 的 `command()`](src/vibench/controller.py#L65)。

### 6.3 双指接触后的防穿模闭合

夹爪不会始终向任意深的闭合目标推进。左右指尖均检测到大于 `0.05 N` 的工件接触后：

1. 读取当前指关节位置；
2. 仅增加可配置的 `0.1 mm` 预紧，并且不会比已经下发的闭合目标更张开；
3. 捕获接触时的指关节目标并保持；
4. 避免继续闭合造成明显穿模。

该逻辑位于 [controller.py](src/vibench/controller.py#L117)。

## 7. 抓取辅助及其边界

默认演示和官方评分均为 `grasp_assist=False`。只有显式 `--grasp-assist` 时才考虑辅助保持，并必须同时满足：

- 控制器正在请求闭合；
- 左指接触力 > 0.05 N；
- 右指接触力 > 0.05 N；
- 手—物距离 < 0.15 m；
- 条件连续保持至少 4 个物理帧。
- 接触瞬间最大穿透 < 0.5 mm。

满足后才记录**接触瞬间**的手—物相对位姿，并在持有期间随手部刚体同步工件位姿和速度。释放命令或穿透超过 1.0 mm 会立即解除保持。实现位于 [VibrationBenchmarkTask._update_grasp_assist()](src/vibench/task.py)。

必须注意：

- 第三轮历史录屏成功结果包含此辅助逻辑，但第四轮默认录屏不包含；
- JSON 会公开记录 `bilateral_contact_confirmed`、`grasp_assist_used`、`grasp_assist_rejected_penetration` 和 `grasp_assist_released_penetration`；
- `--no-grasp-assist` 仅作为旧命令兼容写法；默认已经关闭；
- [configs/scenarios.yaml](configs/scenarios.yaml) 明确规定官方评分不应要求 grasp assist。

## 8. 已实现传感器与观测

### 8.1 腕部六自由度力感知

`JointWrenchSensorCfg` 挂载到整个 Panda articulation，每个物理步更新。任务读取 `panda_link7` 通道，输出：

- `wrist_force_b`：三维力；
- `wrist_torque_b`：三维力矩。

配置位于 [scene.py](src/vibench/scene.py#L221)，读取位于 [task.py 的 `observation()`](src/vibench/task.py#L269)。运行指标记录整回合的力范数和力矩范数峰值。

这些数值尚未与真实 Panda 腕部传感器标定，不应直接用于实机载荷结论。

### 8.2 指尖与工件接触传感器

当前配置三组接触观测：

- 工件相对桌面、目标盒和双指的过滤接触；
- 左指尖只过滤工件；
- 右指尖只过滤工件。

`_filtered_contact_force()` 对过滤力矩阵取向量范数和最大值，得到每个环境的左右指尖接触力。配置见 [scene.py](src/vibench/scene.py#L222)，处理见 [task.py](src/vibench/task.py#L196)。

### 8.3 腕部 RGB

| 属性 | 值 |
|---|---:|
| 分辨率 | 384 × 240 |
| 垂直视场 | 75° |
| 渲染器 | Newton ViewerGL |
| 安装方式 | 固定手眼外参，完整位置/光轴/滚转 |

`NewtonGlWristCameraSensor` 延迟创建渲染器：只进行状态仿真时不会支付 RGB 渲染开销。代码见 [wrist_camera.py](src/vibench/wrist_camera.py#L155)。

### 8.4 观测字典

`VibrationBenchmarkTask.observation()` 当前提供：

| 键 | 内容 |
|---|---|
| `joint_pos` | 全部关节位置 |
| `ee_pose_w`, `ee_pose_b` | 末端世界/机器人根坐标位姿 |
| `left_finger_pose_w`, `right_finger_pose_w` | 双指世界位姿 |
| `finger_center_b` | 指尖中心在机器人根坐标中的位置 |
| `root_pose_w` | 机器人根位姿 |
| `workpiece_pose_w/b` | 工件世界/根坐标位姿 |
| `target_pose_w/b` | 目标世界/根坐标位姿 |
| `vibration_q/qd/qdd` | 六轴位移、速度、加速度 |
| `mount_delta_z` | C2 机械臂与工作台测点的 Z 向差值 |
| `shaker_leg_lengths_m` | Stewart 六条支腿的瞬时长度 |
| `wrist_force_b`, `wrist_torque_b` | 腕部三维力/三维力矩 |
| `left/right_finger_contact_n` | 双指过滤接触力 |
| `left/right_finger_descent_contact_n` | 与工件或桌面的下降保护接触力 |
| `bilateral_contact_streak` | 连续双侧接触帧数 |
| `grasped` | 当前辅助保持状态 |
| `penetration_mm` | 当前受监控接触对的最深穿透 |
| `wrist_camera_eye/target/up_w` | 完整腕部相机世界外参 |

## 9. 任务判定与指标

`EpisodeMetrics` 实现以下指标：

- `lifted`：工件高度超过目标高度 0.10 m；
- `placed`：已经抬升、水平目标误差 < 0.07 m，且已释放；
- `success = lifted and placed`；
- `final_xy_error_m`：工件与目标中心的最终水平距离；
- `max_wrist_force_n`：腕部力范数峰值；
- `max_wrist_torque_nm`：腕部力矩范数峰值；
- `bilateral_contact_confirmed`：是否经过真实双侧接触门控；
- `grasp_assist_used`：本回合是否实际使用辅助保持。
- `max_penetration_mm/pair/t`：全回合最深穿透、接触对与发生时刻；
- `penetration_frames_over_0p5mm`：穿透超过 0.5 mm 的物理帧占比；
- `grasp_assist_rejected_penetration/released_penetration`：assist 门控事件；
- `grasp_z_guard_triggered/descend_contact_timeout/descend_table_contact`：下降阶段的诚实失败原因；
- `grasp_contact_timeout/grasp_contact_lost`：未建立双指接触或搬运中丢失接触。

定义和更新分别位于 [EpisodeMetrics](src/vibench/task.py#L22) 与 [_update_metrics()](src/vibench/task.py#L353)。

## 10. 材质、纹理和打光

### 10.1 USD 纹理管线

项目使用便携的：

```text
UsdPrimvarReader_float2 -> UsdUVTexture -> UsdPreviewSurface -> Material
```

纹理颜色空间为 sRGB，S/T 均为 repeat，并显式指定白色基础色，避免 Newton 对未解析颜色的形状套用调试色。代码见 [_textured_material()](src/vibench/visual_assets.py#L18)。

### 10.2 Benchmark 打光

NewtonGL 录制打光采用：

- 中性天空环境光 `(0.36, 0.36, 0.36)`；
- 中性地面环境光 `(0.14, 0.14, 0.14)`；
- 白色斜上主光；
- 相机方向白色补光；
- 关闭摄影环境贴图和彩色反射；
- 关闭硬投影阴影；
- 曝光 `0.88`，镜面强度 `0.30`。

实现位于 [benchmark_rendering.py](src/vibench/benchmark_rendering.py#L12)。主相机与腕部相机复用同一渲染标定，减少色偏。

## 11. 视频录制与可视化功能

`BenchmarkRecorder` 输出 H.264 MP4，并叠加：

- 1280 × 720 主场景视角；
- 当前控制阶段和仿真时间；
- C2 两测点的 `Delta-z`；
- 左右指尖接触力；
- 当前保持状态；
- 六轴随机谱模式和 seed；
- 按谱配置解析计算的平动合成 `a_rms_g`（默认 6.22 g）；
- 384 × 240 腕部 RGB 画中画；
- 最近 4 s 的 `tx/tz/rx` 归一化振动曲线；
- 成功后显示 `TASK SUCCESS`。
- 当前/历史最大穿透与接触对；超过 0.5 mm 时标红。

代码位于 [recording.py](src/vibench/recording.py)。当前录屏显示的是滚动振动时域曲线；尚未实现实时 FFT/PSD 图。

## 12. 配置和命令行功能

### 12.1 配置文件

| 文件 | 用途 |
|---|---|
| [config.py](src/vibench/config.py) | 数值、资产和振动 dataclass 配置 |
| [configs/assets.yaml](configs/assets.yaml) | 资产来源、路径、许可和纹理哈希 |
| [configs/room.yaml](configs/room.yaml) | 房间布局与视觉样式 |
| [configs/scenarios.yaml](configs/scenarios.yaml) | baseline、单轴正弦、六轴随机谱场景矩阵 |
| [configs/visual_manifest.yaml](configs/visual_manifest.yaml) | V6 可见特征、父 prim、材质和几何断言 |

### 12.2 CLI 参数

入口 [scripts/run_demo.py](scripts/run_demo.py) 当前支持：

```text
--record
--output PATH
--camera-preset main|stewart_side
--no-overlays
--workpiece cracker_box|sugar_box|soup_can|mustard_bottle
--workpiece-scale FLOAT
--vibration off|sine|spectral
--sine-axis tx|ty|tz|rx|ry|rz
--sine-amplitude FLOAT
--sine-frequency-hz FLOAT
--seed INT
--episode-s FLOAT
--num-envs INT
--device DEVICE
--physics-profile official|training
--physics-hz INT
--grasp-assist
--no-grasp-assist
--metrics-output PATH
```

`run.sh` 通过 `scripts/isaac_env.sh` 解析 `ISAACLAB_ROOT`（默认 `$HOME/IsaacLab-3.0`）、设置项目源码与 Isaac Lab 资产路径，并使用该环境中的 Python 启动。测试可用 `./run_tests.sh`，其他仓库工具可用 `./run_python.sh`。

## 13. 测试与验证

[tests/test_vibration.py](tests/test_vibration.py)、[tests/test_third_round_geometry.py](tests/test_third_round_geometry.py) 与 [tests/test_visual_manifest.py](tests/test_visual_manifest.py) 当前合计包含 38 项自动测试，覆盖核心动力学、第三轮几何与第四轮物理/资产回归。

1. 相同随机 seed 的谱完全可复现，且不同活动轴使用独立随机流；
2. ramp 后六个轴均存在非零激励；
3. `off` 模式严格为零；
4. C2 支撑 Z 向差值与解析式一致；
5. 房间样式配置可确定性加载；
6. 活动环氧地坪、工业墙板、酚醛桌面和 platen 纹理 SHA-256 固定，且条目出现在 manifest；
7. 腕部相机光心、光轴、上方向、归一化和法兰避障约束正确。
8. 4 mrad/50 mrad 边界下完整 SE(3) 与旧小角度近似满足预期误差界；
9. Stewart 倾斜位姿的六腿长度与独立解析距离一致；
10. 默认随机谱 1000 步内六腿均不越过行程限制。
11. Stewart 外筒/活塞杆保持有效重叠，球铰直径为杆半径的 2.0–2.5 倍；
12. 动平台椭圆铰点在 X/Y 方向均位于台面内并满足安全余量；
13. 铰点整体位于 80 mm 台裙下方；
14. 完整 1000 步谱激励中所有作动器段均避开台面 AABB；
15. 机柜与工具车接地，且工具车立柱不与层板错误重叠；
16. 工件、台面、桌腿、机械臂、目标盒及房间设备阴影布局完整且对齐。

运行测试：

```bash
cd ViBench
./run_tests.sh
```

当前结果：`38 passed`。阶段 A 硬装重构后 official 1 s × 5 谱探针（seed=17，`out/stage_a_spectral_probe.json`）最大穿透 **0.242 mm**，接触对为 `workpiece<->worktable`，腕力峰值 0.030 N，`robot_link<->platen` 伪接触已消失；完整 16 s 回合（`out/stage_a_16s_seed17.json`）为 `lifted=true` 后 `grasp_z_guard_triggered` 的诚实失败，最大穿透 0.955 mm `workpiece<->worktable`、`support_geometry_valid=true`。数值地板标定见 `docs/baseline_status.md`：普通 C2 五 seed 最大 0.259 mm，未达 D2 计分资格（≤0.112 mm）；C2_CLITE 支撑按子步驱动可降到 0.064–0.123 mm，但 seed47 不稳定且 16 s 吞吐不足。下一修复项是**支撑子步注入的工程化**（不依赖 C2_CLITE 的私有手动循环）与 **grasp_z_guard/控制器在重力开启后的重新整定**。旧 `out/p0_5substep_spectral_probe.json` 的 0.665 mm `robot_link<->platen` 结果属于虚拟测点模型，已作废。

## 14. 复现当前完整演示

```bash
cd ViBench
./run.sh --record \
  --physics-profile official \
  --episode-s 16 \
  --vibration spectral \
  --seed 17 \
  --workpiece sugar_box \
  --workpiece-scale 0.75 \
  --output out/benchmark_v2_wrist_camera_fixed.mp4 \
  --metrics-output out/benchmark_v2_wrist_camera_fixed.json
```

该命令默认就是纯物理无辅助抓取。若困难回合失败，保留 `success=false`；不要为复现旧画面而偷偷追加 `--grasp-assist`。

## 15. 源码模块索引

| 模块 | 主要职责 |
|---|---|
| [config.py](src/vibench/config.py) | 全局数值、振动频带、资产选择 |
| [vibration.py](src/vibench/vibration.py) | 六轴正弦/随机谱和解析导数 |
| [mounting.py](src/vibench/mounting.py) | C2 多安装点完整 SE(3) 运动映射 |
| [shaker.py](src/vibench/shaker.py) | Stewart 参数化几何、六腿解析解与视觉刚体生成 |
| [scene.py](src/vibench/scene.py) | Newton 配置、Panda/YCB/Stewart/桌面/盒子/传感器场景装配 |
| [arena.py](src/vibench/arena.py) | 工业房间布局、下沉地坑、后侧防护栏、环氧地坪、中景设备和阴影 |
| [visual_assets.py](src/vibench/visual_assets.py) | USD UV 材质、台面/桌面细节、传感器、线缆、警示带和视觉阴影 |
| [wrist_camera.py](src/vibench/wrist_camera.py) | 腕部相机物理建模、固定手眼外参和 RGB 传感器 |
| [benchmark_rendering.py](src/vibench/benchmark_rendering.py) | 中性打光和完整相机滚转支持 |
| [task.py](src/vibench/task.py) | 仿真循环、支撑驱动、观测、接触门控和指标 |
| [controller.py](src/vibench/controller.py) | DLS 差分 IK、状态机与夹爪接触闭合 |
| [recording.py](src/vibench/recording.py) | 视频、腕部画中画、遥测和振动曲线 |
| [cli.py](src/vibench/cli.py) | CLI、回合执行、日志和 JSON 输出；[scripts/run_demo.py](scripts/run_demo.py) 是兼容启动器 |
| [diagnostics.py](src/vibench/diagnostics.py) | Newton/MJWarp 形状、接触几何和候选对快照 |
| [tests/test_vibration.py](tests/test_vibration.py) | 确定性、振动、SE(3)、Stewart、纹理和相机外参测试 |
| [tests/test_third_round_geometry.py](tests/test_third_round_geometry.py) | 第三轮台面净空、作动器避障、设备接地和阴影布局测试 |
| [visual_manifest.py](src/vibench/visual_manifest.py) | 可见特征事实与附件世界锚点审计 |
| [tests/test_visual_manifest.py](tests/test_visual_manifest.py) | V0–V6 配置、抓取几何、父变换和视觉清单测试 |

## 16. 已知边界与尚未实现内容

为避免误解，当前版本存在以下明确边界：

- 第三轮历史成功演示使用真实双侧接触门控后的相对位姿保持；第四轮默认与官方评分均关闭；
- 参考控制器使用工件和目标的仿真真值位姿，不是视觉端到端策略；
- 尚未提供 RL/模仿学习训练代码、Gym 注册或标准数据集导出器；
- 腕部相机当前仅输出 RGB，尚未输出深度、分割图或相机噪声模型；
- 已有末端三维力与三维力矩，但没有触觉阵列或实机标定模型；
- 视频已有滚动振动曲线，尚未加入实时 FFT/PSD 频谱；
- 房间纹理已本地固化，但 Panda/YCB 首次运行仍依赖 NVIDIA 资产根或本机 Omniverse 缓存；
- Newton/MJWarp 当前启用了 MuJoCo 接触生成，因此不是完全零 MuJoCo 依赖；
- Isaac 5.0 Franka USD 在当前 Newton 路径会给出质量/惯量提示，当前完整任务未出现数值失稳，但力峰值不应被当作实机标定值。
- 当前 Newton 原生初始化存在进程级 `SIGSEGV`/double-free/无 traceback 退出，通常发生在模型构建前；第三轮正式 14.334 s 回合和第四轮多次短探针均曾完整成功，但本轮连续验证时该上游问题不再只是低概率。
- `use_mujoco_contacts=True` 的 NativeCCD/MULTICCD 会把 1 mm authored margin 清零；当前实现公开记录这一限制，并按档位使用满足子步稳定下限的 `solref` 调整接触柔度。
- `C2_CLITE` 模式目前只把振动平台和工作台改为动态约束支撑；桌腿、目标盒和 Panda 浮动根仍为 kinematic 轨迹写入，且该模式使用 NewtonManager 私有 substep 接口，标记为实验性质、不计分。

以上边界不影响当前演示与软件功能复现，但在把工程升级为正式公开 benchmark 前，应把纯物理无辅助基线、视觉策略接口、数据记录协议和经过校核的动力学参数作为后续工作。
