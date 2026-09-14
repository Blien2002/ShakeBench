# 迁移场景的 GPU oracle rollout

入口 `robosuite.scripts.shakebench_gpu_batch` 使用当前 `world_fixed_arm_v1` MJCF 场景。
MuJoCo Warp 在 GPU 上执行物理、Panda OSC、夹爪、接触和连续成功判定；Python 执行现有 oracle 的任务操作逻辑。
没有图像采集，输出始终 `scoreable=false`，不能据此替代正式 CPU 评测。

## 安装和 Gamma 扫描

依赖安装方式参见 [MuJoCo Warp 官方文档](https://mujoco.readthedocs.io/en/latest/mjwarp/)。
本仓库固定 `requirements-gpu.txt` 中的版本，因为已有 GPU 内核访问版本相关的数组布局和接触力函数。
可在已有 robosuite 环境上创建隔离环境：

```bash
python -m venv --system-site-packages /tmp/shakebench-gpu-venv
/tmp/shakebench-gpu-venv/bin/python -m pip install -r requirements-gpu.txt
/tmp/shakebench-gpu-venv/bin/python -m robosuite.scripts.shakebench_gpu_batch \
  --states robosuite/models/assets/shakebench_states_dev.json \
  --output out/gpu_world_fixed_gamma_sweep \
  --gammas 0 0.15 0.30 0.45 0.60 0.75 0.95 \
  --physics-profile probe --limit 1 --num-worlds 7 --horizon-steps 1200 --device cuda:0
```

`--limit` 限制输入 state 数；每个 state 都执行所有 Gamma。
`--gamma 0.15` 可单独指定一个值；`--mode` 支持 `multisine_v1` 和 `single_sine_v1` 的默认参数。
使用当前 `vibration` 配置及 `normal_peak_v1` 标定，Gamma=0 的位置、速度和加速度命令严格为零。
示例使用仍可验证的冻结 dev states。历史 official/task variants 资产绑定旧几何身份，
这些资产已改绑当前 runtime contract；正式评测前仍需完成资格验证，不能直接当成已认证的评测状态。
首次检查可使用 `--horizon-steps 3`，完整任务应保留默认 1200 步。

CUDA 不可用时直接失败，不会退回 CPU；`--device cpu --no-capture` 是显式的 Warp 调试模式。
默认捕获 25 个物理子步的 CUDA Graph，每个 20 Hz 动作重放十次。
`--nconmax`、`--njmax` 控制容量；溢出记录为 `invalid_execution`。

默认 `official` 保留 MJCF 的 0.4 ms 硬接触设定；GPU 后端会对罐体的手指抓取、桌面和托盘支撑接触
做设备侧标定，但不改变 CPU 物理或成功阈值。当前 MJWarp float32 后端中，桌面静止接触的加速度仍会
与 CPU double 后端产生可测差异，尚未通过 IMU 一致性检查。
`probe` 使用仓库已有的 20 ms 非正式接触配置，并通过下方的完整 CPU/GPU 状态与 IMU 对照；
因此示例使用它。两种输出均为非评分采集数据。

## 观测和文件

环境不再接收旧 `observation_tier`。独立的 `shakebench_expert.oracle_observation` 从当前编译状态读取
机械臂、物体和目标的 base 系状态；视频入口共用该函数。专家继续复用控制器的 V0 操作分支，
但这不代表恢复 V0–V3 学习观测合同。`--tier V0` 仅作为旧命令的别名保留；V1–V3 不支持。

IMU 始终采样：从编译的 `table_imu_site` 解析父 body、位置和方向，并通过现有 provider 校验绑定。
200 Hz GPU 样本在控制周期结束时下载，复用 CPU `CanonicalIMU` 的噪声、滤波、量化和单采样延迟。
reset 保留 CPU 初始化得到的加载平衡状态和 IMU 历史。

输出使用 collection schema v2：

- `observations/table_imu_*`：当前桌下 IMU 合同，T+1 帧。
- `privileged_oracle/*`：独立专家真值，T+1 帧，不是视觉学习模型的输入。
- `qpos`、`qvel`、`time_s`：T+1 帧；`actions`、`clipped_actions`、`ctrl`、`actuator_force`、`contacts`、`phase`：T 帧。
- `metadata_json`：状态、当前几何及物理身份、振动模式/Gamma 定义/程序 hash、IMU 绑定、终止原因和控制器事件。
- `manifest.json`：完整性状态、每个 episode 的 Gamma、hash、有效性和结果。
- `model_<sha256>.xml`：当前编译模型；资产路径仍引用仓库，搬动输出时须保留对应资产。

NPZ 使用 `np.load(path, allow_pickle=False)` 读取。输出必须是新目录或空目录；已完成 episode 原子发布且不覆盖。
中断保留已有 NPZ，manifest 的 `complete=false`；当前不自动续跑。

## 验证

```bash
python -m pytest -q tests/test_shakebench_gpu_migration.py
SHAKEBENCH_TEST_DEVICE=cuda:0 SHAKEBENCH_TEST_PHYSICS_PROFILE=probe /tmp/shakebench-gpu-venv/bin/python -m pytest -q tests/test_shakebench_gpu_migration.py
```

第一条检查当前场景、隔离专家字段、Gamma=0 和闭环动作；第二条必须实际使用指定设备，
追加相同动作下的 CPU/GPU qpos、base 系位姿及桌下 IMU 窗口/时间戳对照。
物理为 float32，OSC 小矩阵使用 float64。GPU 模型上传后恢复 MJCF 中的 solver tolerance，
避免 MJWarp 默认放宽到 1e-6 后过早终止；实际值写入每个 episode。
一致性检查不要求逐位相同，也不能用短程检查推断完整任务成功率。

## GPU 数据采集（LeRobot v2.1）

```bash
/tmp/shakebench-lerobot-venv/bin/python -m robosuite.scripts.shakebench_collect_lerobot_gpu \
  --output out/lerobot_task_close_gpu \
  --states robosuite/models/assets/shakebench_task_states_official_v2.json \
  --num-worlds 8 --image-writer-threads 8 --device cuda:0
```

与 CPU 采集器 `shakebench_collect_lerobot` 共用同一套 LeRobot v2.1 schema、动作空间、
8 维状态和 IMU 字段，区别只在后端：物理走 MJWarp，图像走 mujoco_warp 的 CUDA 光追
（`MJWarpBatch.enable_rendering` / `render_rgb`，不需要 EGL）。宿主只为播种设备模型而编译 MJCF。
主视角 `task_close` 姿态由 `demo_shakebench_oracle_video.task_close_camera_pose` 提供，
运行时盖写到模型相机 `frontview`，因此 GPU 渲染只吃模型相机也能复现 CPU 的取景。
输出 `scoreable=false`；CPU 路径仍是评分与参考通道。

采集器会按任务变体分组，在同一批次中执行 `--num-worlds` 个 state；不同物体或台面配置不会混进同一
MJCF batch。每个 world 仍保存为独立 LeRobot episode，帧的 pre-step 图像、动作、状态和 next outcome
保持原有对齐。默认使用 4 个 world 和 4 个异步图像写线程；服务器可根据显存和 CPU 写盘能力调大这两个值。

多进程服务器不要并发写同一个 LeRobot 根目录。用 state-list 分片，并为每个分片指定独立输出目录：
下面示例假设每个进程独占一张 GPU。

```bash
for shard in 0 1 2 3; do
  /tmp/shakebench-lerobot-venv/bin/python -m robosuite.scripts.shakebench_collect_lerobot_gpu \
    --states robosuite/models/assets/shakebench_task_states_official_v2.json \
    --output "out/lerobot_gpu/shard_${shard}" \
    --num-shards 4 --shard-index "${shard}" --num-worlds 8 --device "cuda:${shard}" &
done
wait
```

每个分片的 `meta/shakebench_collection.json` 记录 state 列表、分片编号、batch 大小和图像写线程数，
便于之后审计或合并。`--limit` 在分片之后生效，适合做每个服务器进程的预算限制。

当前状态（2026-09-14）：GPU official rollout 已能完成一次成功判定；GPU 输出仍为 `scoreable=false`。

- 根因是 float32 设备求解器下 4e-4 s 接触产生微小弹跳：托盘支撑接触断续，连续 0.5 s
  成功窗口无法累积。`MJWarpBatch` 现在按接触类别标定手指抓取和桌面/托盘支撑接触。
- 同一 dev state、同一动作序列下，CPU 在 policy step 173 latch success，GPU 在 step 174 latch
  success；GPU 最大非法穿透约 `2.4e-4 m`，低于 `5e-4 m` 阈值。
- 这只修复 rollout 无法完成的问题；GPU 与 CPU 的逐位物理/IMU 等价性仍需单独验证，CPU 继续作为评分参考。
