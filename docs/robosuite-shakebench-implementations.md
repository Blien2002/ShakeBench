# 当前仓库的 robosuite ShakeBench 实现清单

本文只盘点当前工作树仍在使用的可执行代码与其加载的包内资产。Phase 06–08 的选型、发布、handoff 与 evidence archive 链已在 2026-09-13 的清理中删除，不再列在此处；需要历史证据请回到 git 历史。

## 1. 任务环境与场景

| 实现 | 功能 | 关键源码 |
| --- | --- | --- |
| `VibrationPickPlace` | 唯一公开任务入口：单 Panda 世界固定安装、隔振工作台、甲板驱动、桌下 IMU、三物体 × 两表面 `TaskSpec`、metrics 与成功判定。 | `robosuite/environments/manipulation/vibration_pick_place.py` |
| `ShakeBenchArena` 与场景审计 | 工业场景、隔振工作台、目标容器装配缝，以及 frame ownership／间隙／相机审计。 | `robosuite/models/arenas/shakebench_arena.py`、`scene_visuals.py`、`scene_audit.py` |
| `TaskSpec` / `ShakeBenchTask` / `make_task_env` | 物体与表面选择、物体生成、统一 `reset/step/close` 协议与当前环境选择。 | `robosuite/utils/shakebench_tasks.py` |
| task-state 扩展 | 把已认证的 parent states 扩展为 v2 任务状态（official 轮换变体、knee 全组合），并校验与冻结资产。 | `robosuite/utils/shakebench_task_states.py` |
| state authority | Phase 08 official/knee committed states 与 Phase 07 dev states 的构建、哈希、验证和冻结。 | `robosuite/utils/shakebench_committed_states.py`、`shakebench_dev_states.py` |

## 2. 运行时子系统

| 模块 | 功能 | 关键源码 |
| --- | --- | --- |
| deck driver | 动态六自由度甲板驱动、可审计 XML 变换与 trace。 | `robosuite/utils/shakebench_deck.py` |
| excitation / calibration | 六轴频带、确定性谱线、quintic ramp、Gamma 标定与 mode × Gamma 运行身份。 | `robosuite/utils/shakebench_excitation.py`、`shakebench_calibration.py` |
| isolator / safety | 六自由度隔振器参数、静态偏移与传递函数；六轴激励的 fail-closed 安全界。 | `robosuite/utils/shakebench_isolator.py`、`shakebench_safety.py` |
| sensors / providers | 桌下工作台 IMU 的比力与角速度、滤波、噪声、量化与单采样交付；policy-safe 单 IMU provider。 | `robosuite/utils/shakebench_sensors.py`、`shakebench_providers.py` |
| privilege | privileged recorder 与 policy observation 的隔离审计。 | `robosuite/utils/shakebench_privilege.py` |
| metrics / outcomes / scoring | 接触、相对位姿与速度、连续稳定成功窗口；版本化 outcome contract；Wilson 区间与 paired 统计。 | `robosuite/utils/shakebench_metrics.py`、`shakebench_outcomes.py`、`shakebench_scoring.py` |
| oracle | 只读取 public observation 与静态上下文的 FSM reference controller（任务执行器，不做振动补偿）。 | `robosuite/utils/shakebench_oracle.py`、`oracle_executive.py` |
| physics / geometry / scene | 不可变 official/probe physics profile、当前几何 profile、场景视觉与净空审计。 | `robosuite/utils/shakebench_physics.py`、`shakebench_geometry.py`、`shakebench_scene.py` |
| runtime verifier / artifacts | 当前资产清单（physics profile 与 geometry/scene/arena）的字节与 payload 校验，以及 canonical JSON/hash/原子写入。 | `robosuite/utils/shakebench_runtime_verifier.py`、`shakebench_artifacts.py` |
| mjwarp / starvla | GPU 采集后端与 StarVLA 观测、动作与客户端边界。 | `robosuite/utils/shakebench_mjwarp.py`、`shakebench_starvla.py` |
| config / rotations | Phase 00 配置与来源信封；wxyz 精度四元数工具。 | `robosuite/utils/shakebench_config.py`、`shakebench_rotations.py` |

## 3. 入口与演示

| 入口 | 功能 |
| --- | --- |
| `python -m robosuite.scripts.shakebench_run_oracle` | 在 dev/task states 上运行 reference controller，生成 rollout/run artifact 并 fail-closed 校验。 |
| `python -m robosuite.scripts.shakebench_collect_lerobot` | 采集 LeRobot v2.1 gamma=0 专家数据集。 |
| `python -m robosuite.scripts.shakebench_run_starvla` | StarVLA 任务接口的采集与回放入口。 |
| `python -m robosuite.scripts.shakebench_gpu_batch` | MJWarp GPU 批量采集（NPZ 与 JSON 元数据）。 |
| `python -m robosuite.scripts.shakebench_scene_preflight` | scene／frame／clearance 预检。 |
| `python -m robosuite.scripts.shakebench_verify_actuators` | Panda OSC_POSE 与 gripper 七通道动作路径验证。 |
| `python -m robosuite.utils.shakebench_runtime_verifier --write` | 依据当前资产重写 `shakebench_runtime_contract.json`；不带 --write 时校验。 |
| `python -m robosuite.demos.demo_shakebench_task_variants` | 渲染六个 pick/place 变体的场景预览。 |
| `python -m robosuite.demos.demo_shakebench_task_slip` | 记录物体在振动条件下的相对滑移视频（非策略能力展示）。 |
| `python -m robosuite.demos.demo_shakebench_oracle_video` | 录制 oracle 定性视频；相机像素不进入 policy、metrics 或科学证据链。 |

## 4. 被运行时加载的当前资产

| 资产 | 说明 |
| --- | --- |
| `shakebench_official_physics.yaml` | scoreable、immutable 的 official physics profile。 |
| `shakebench_runtime_contract.json` | 当前资产清单：绑定点是 physics profile、geometry/scene JSON 与 arena/support MJCF 的字节哈希。 |
| `shakebench_geometry_world_fixed_arm_v1.json`、`shakebench_scene_world_fixed_arm_v1.json`、`arenas/shakebench_arena.xml`、`arenas/shakebench_robot_support.xml` | 当前世界固定安装、场景视觉与物理拓扑。 |
| `shakebench_states_dev.json` | 10 个 dev 状态（object 命名），当前采集入口的默认状态表。 |
| `shakebench_states_official.json`、`shakebench_states_knee.json`、`shakebench_task_states_*_v2.json` | Phase 08 committed states 与由其扩展的任务状态；已按当前 runtime contract、geometry 与 outcome 绑定重新冻结，新实验前仍需完成资格认证。 |

## 5. 最小使用示例

```bash
# 程序内构造当前唯一任务入口
python -c "import robosuite; env = robosuite.make('VibrationPickPlace', robots='Panda'); env.reset()"

# 在显式 task-state 上运行 reference controller
python -m robosuite.scripts.shakebench_run_oracle --states robosuite/models/assets/shakebench_task_states_official_v2.json --state-id <state-id>

# 校验或重写当前运行时资产清单
python -m robosuite.utils.shakebench_runtime_verifier
python -m robosuite.utils.shakebench_runtime_verifier --write
```
