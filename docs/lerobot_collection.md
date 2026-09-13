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
可重复的 `--state-id` 或 `--horizon-steps 2` 缩小采集范围。输出目录必须不存在，不覆盖、不上传。
这是实时 MuJoCo CPU 物理 rollout，EGL 渲染双相机；不是合成动作，也不是实物机器人采集。
使用当前 `world_fixed_arm_v1` 场景、`official` 物理配置及隔离的当前状态专家。
Gamma 固定为零，没有命令行改写入口；零外部激励不等于 IMU 为零，接触、重力和传感器噪声仍保留。

每行对应动作执行前的观测 `o_t`、实际执行的 `a_t`，以及执行后的 `next.*`：

| 字段 | 内容 |
| --- | --- |
| `observation.images.main` | 默认复用 oracle 视频的 `task_close` 任务主视角，RGB 256×256，可用 `--main-camera` 切换编译场景中的相机 |
| `observation.images.wrist` | `robot0_eye_in_hand` 腕部相机，与主相机同一仿真时刻 |
| `observation.state` | float32[8]，base 系末端位置/轴角及左右夹爪关节位置 |
| `action` | float32[7]，Panda OSC_POSE，机器人 base 系的归一化位姿增量及夹爪控制，范围 [-1, 1] |
| `observation.table_imu_window` | float32[10,6]，200 Hz 因果窗口；前三列为加速度 m/s²，后三列为角速度 rad/s，IMU 局部坐标系 |
| `observation.table_imu_timestamps_s` | float64[10]，相对于 episode 的采样时间，保留初始负时间历史及传感器延迟 |
| `observation.table_imu_dt_s` | float32[1]，采样间隔 0.005 s |
| `task_index` | 指向 `meta/tasks.jsonl` 中的英文指令，官方读取器将其解析为 `task` |
| `next.reward / next.done / next.success` | 当前动作后的奖励、episode 结束标记、成功标记；超时的 done=true 不代表成功 |

任务指令：**Pick up the can from the table and place it in the target tray.**
动作前三维控制器缩放为每步 ±0.05 m，中三维为 ±0.5 rad，夹爪 -1 打开、+1 闭合。
oracle 自身的阶段限幅仍生效。采样率固定 20 Hz，`timestamp=t/20`；不补帧、不添加视频展示用的停留帧。

图像以无损 PNG 字节嵌入 `data/chunk-000/episode_XXXXXX.parquet`，这是 LeRobot v2.1 的 image 模式，
无需外部 MP4。采集器另生成 StarVLA 所需的 `meta/modality.json`，接入见 [StarVLA 文档](starvla.md)。官方写入器同时生成 `meta/info.json`、`tasks.jsonl`、`episodes.jsonl`、`episodes_stats.jsonl`。
`meta/shakebench_collection.json` 额外保留初始状态、IMU 绑定、控制器配置、动作语义及终止原因。
失败和超时 rollout 也会保存，可按该文件的 `success` 筛选；程序返回 0 表示采集完成，不表示任务全部成功。
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
