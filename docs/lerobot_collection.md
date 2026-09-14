# Gamma=0 oracle 数据采集

```bash
python -m pip install -r requirements-collection.txt
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m robosuite.scripts.shakebench_collect_lerobot \
  --output out/lerobot_gamma_zero
```

建议独立 Python 3.10–3.12 环境，先安装本仓库 `requirements.txt`，再安装采集依赖。
固定使用官方 `lerobot==0.3.3`，其 `CODEBASE_VERSION` 为 `v2.1`；代码拒绝使用 v3 写入器。
参考：[官方写入器源码](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py)。

默认对冻结的十个 dev states 各执行一次当前 oracle，最多 1200 步。可用 `--limit 1`、
可重复的 `--state-id` 或 `--horizon-steps 2` 缩小采集范围。

`--limit` 是 episode 预算，不得超过所选状态数；超出时报错而不是静默少采，因为重复运行同一状态池
不会增加场景覆盖。要采集 N 条不同初始状态的示范，先生成独立的 train 状态池，再把它作为 `--states`：

```bash
python -m robosuite.scripts.shakebench_generate_train_states \
  --output robosuite/models/assets/shakebench_states_train_pool.json --count 100 --seed 20260914
python -m robosuite.scripts.shakebench_collect_lerobot --states robosuite/models/assets/shakebench_states_train_pool.json \
  --output out/lerobot_gamma_zero_train
```

train 状态池按 seed 确定性重建，state ID 含 seed 与 half-range（`shakebench-train-v0-s<seed>-r<range>-<index>`），
因此不同池不会重名、按 ID 合并不会替换错 episode；生成命名空间与 dev 生成器分开，避免与冻结 dev 状态
逐字重复。`scoreable=false`，与冻结的 dev/official/knee 资产各自独立校验。
评测必须使用与训练无交集的状态：`shakebench_evaluate --dataset` 按执行状态指纹（物体起始位置、
excitation/IMU seed、t0、task 规格）比较，而不只是比 state ID（见 `shakebench_evaluate` 的 `--dataset` 检查）。
输出目录必须不存在，不覆盖、不上传。
这是实时 MuJoCo CPU 物理 rollout，EGL 渲染双相机；不是合成动作，也不是实物机器人采集。
使用当前 `world_fixed_arm_v1` 场景、`official` 物理配置及隔离的当前状态专家。
Gamma 固定为零，没有命令行改写入口；零外部激励不等于 IMU 为零，接触、重力和传感器噪声仍保留。

每行对应动作执行前的观测 `o_t`、实际执行的 `a_t`，以及执行后的 `next.*`：

| 字段 | 内容 |
| --- | --- |
| `observation.images.main` | 主视角 `task_close`，即 LIBERO `agentview` 姿态（见下），RGB 256×256；也可用 `--main-camera` 直接传编译场景中的相机名 |
| `observation.images.wrist` | `robot0_eye_in_hand` 腕部相机，与主相机同一仿真时刻 |
| `observation.state` | float32[8]，base 系末端位置/轴角及左右夹爪关节位置 |
| `action` | float32[7]，Panda OSC_POSE，机器人 base 系的归一化位姿增量及夹爪控制，范围 [-1, 1] |
| `observation.table_imu_window` | float32[10,6]，200 Hz 因果窗口；前三列为加速度 m/s²，后三列为角速度 rad/s，IMU 局部坐标系 |
| `observation.table_imu_timestamps_s` | float64[10]，相对于 episode 的采样时间，保留初始负时间历史及传感器延迟 |
| `observation.table_imu_dt_s` | float32[1]，采样间隔 0.005 s |
| `task_index` | 指向 `meta/tasks.jsonl` 中的英文指令，官方读取器将其解析为 `task` |
| `next.reward / next.done / next.success` | 当前动作后的奖励、episode 结束标记、成功标记；超时的 done=true 不代表成功 |

任务指令：默认 can 任务为 **Pick up the can from the table and place it in the target tray.**；
bread/cookie-box 等任务变体按当前状态的 `task_description` 写入（例如 **Pick up the bread from the table
and place it in the target basket.**），manifest 用 `tasks` 列出本次采集写入的所有指令，
每个 episode 记录自己的 `instruction`。
动作前三维控制器缩放为每步 ±0.05 m，中三维为 ±0.5 rad，夹爪 -1 打开、+1 闭合。
oracle 自身的阶段限幅仍生效。采样率固定 20 Hz，`timestamp=t/20`；不补帧、不添加视频展示用的停留帧。

主视角 `task_close` 用的就是 LIBERO 的 `agentview` 姿态：LIBERO 在 `libero/libero/envs/bddl_base_domain.py` 的
`BenchmarkEnv._setup_camera` 中把观测相机设为 pos `[0.5886, 0, 1.4904]`、quat wxyz
`[0.6380, 0.3049, 0.3049, 0.6380]`（`env_wrapper.py` 的 `camera_names[0]`，另一台
`canonical_agentview` 只是同一姿态后退 0.05 m），对应台面中心 `(0, 0, 0.8)` 的 0.8 m 方桌。
这里把同一相机-台面偏移按台面尺寸比 `0.65/0.80` 缩放后搬到 ShakeBench 台面中心，姿态不变
（MuJoCo 自由相机，45° fovy，与 LIBERO 一致），因此 0.65 m×0.60 m 台面在画面中的占比与
LIBERO 的 0.8 m 方桌相当（近端台角同样超出画面），物体比 LIBERO 近 `1/0.8125 ≈ 1.23` 倍、
像素尺度相应放大。相机实现见
`robosuite/demos/demo_shakebench_oracle_video.py` 的 `_task_close_camera`（oracle 视频与数据集共用同一台相机）。

图像以无损 PNG 字节嵌入 `data/chunk-000/episode_XXXXXX.parquet`，这是 LeRobot v2.1 的 image 模式，
无需外部 MP4。采集器另生成 StarVLA 所需的 `meta/modality.json`，接入见 [StarVLA 文档](starvla.md)。官方写入器同时生成 `meta/info.json`、`tasks.jsonl`、`episodes.jsonl`、`episodes_stats.jsonl`。
`meta/shakebench_collection.json` 额外保留初始状态、IMU 绑定、控制器配置、动作语义及终止原因。
失败和超时 rollout 也会保存，可按该文件的 `success` 筛选；程序返回 0 表示采集完成，不表示任务全部成功。
同一文件还记录 `state_authority`（dev/train/official/knee 身份）与 `sft_subset` 统计
（选中 episode、成功/失败计数、帧数），后者与导出器共用同一成功规则。
训练应使用导出的成功子集，而不是未筛选的采集目录：

```bash
python -m robosuite.scripts.shakebench_export_sft_subset \
  --dataset out/lerobot_gamma_zero_train --output out/lerobot_gamma_zero_train_sft
```

导出器只复制 `success_latched` episode，并把它们**从 0 连续重编号**：parquet 文件名、`episodes.jsonl`、
`episodes_stats.jsonl`、parquet 的 `episode_index`/`index` 列以及 `info.json` 计数都描述导出编号
（官方读取器会枚举 `range(total_episodes)`，稀疏目录会直接读失败）。元数据只复制格式文件
（`tasks.jsonl`、`modality.json` 以及存在时的 `stats.json` 聚合副本），不继承 StarVLA 生成的
`stats_gr00t.json`、`steps_data_index.pkl` 等派生训练缓存——它们描述源 episode，会让子集报告错的
步数、索引和 min-max 归一化统计。
导出的 `meta/shakebench_collection.json` 只描述导出数据（episode 重编号并另存 `source_episode_index`，
`requested_states` 只列保留状态，`sft_subset.selection_rule` 标明本次选择规则）；源 manifest 原样
保存在 `meta/shakebench_source_collection.json`。`meta/shakebench_sft_subset.json` 记录来源 manifest
哈希、源 episode 索引、导出索引与帧数，`shakebench_evaluate --dataset` 会读取它作为溯源。
失败 episode 仍留在源目录供审计；确实要用失败示范时需显式传 `--include-failures`。
中断时 `complete=false`，已保存 episode 保留，当前未保存 episode 不视为完整数据；不自动续跑。
这些训练采集数据的 `scoreable=false`，不能替代正式 benchmark 评分。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("shakebench/oracle-gamma-zero", root="out/lerobot_gamma_zero")
sample = ds[0]  # image: float32 CHW [0,1]; action: float32[7]; task: English string
```

可运行验证会实际采集两条短 episode，用官方读取器回读，并独立重放相同初始状态，
比较 oracle 动作、IMU 及时间对齐，同时检查双图像和禁止覆盖：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m pytest -q tests/test_shakebench_collect_lerobot.py
```
