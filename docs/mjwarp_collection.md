# MJWarp oracle 采集

第一版复用 CPU oracle，GPU 批量执行 250 个物理子步中的 Panda OSC、夹爪、振动台驱动、接触归约及连续成功判定。
200 Hz 原始 IMU 样本在每个 20 Hz 控制周期结束后整批下载，由原 `CanonicalIMU` 处理噪声、滤波、量化及一采样延迟。
CPU/GPU 只在控制周期边界交换数据。CUDA 默认预热并捕获 25 子步的图，每个动作重放 10 次。

## 安装与运行

建议在独立采集环境安装当前项目及可选依赖，不修改正在运行 CPU 实验的环境：

```bash
uv venv .venv-gpu --python 3.11
uv pip install --python .venv-gpu/bin/python -e . -r requirements-gpu.txt
.venv-gpu/bin/python -m robosuite.scripts.shakebench_gpu_batch \
  --states robosuite/models/assets/shakebench_task_states_official_v2.json \
  --output out/gpu_collection_v0_gamma060 \
  --tier V0 --gamma 0.60 --num-worlds 16 --device cuda:0
```

首次验证可增加 `--limit 2 --horizon-steps 2`。`--state-id` 可重复指定。
按任务物体/表面分组，相同模型内批量运行不同位置和种子；数量不足的一批使用其实际 world 数。
`--nconmax 128 --njmax 512` 是起始容量，溢出会使受影响轨迹标记为 `invalid_execution`，不能视为成功数据。
`--device cpu --no-capture` 使用真正的 MJWarp CPU 调试设备，不是 classic MuJoCo 回退，速度很慢。

## 文件格式

- `manifest.json`：版本、输入 authority、完整性状态、episode 文件 SHA-256、终止原因及端到端有效 episode/h。
- `model_<sha256>.xml`：当前模型的 MuJoCo XML，资产路径引用当前仓库；搬动仓库时需一并保留模型资产并修正路径。
- `episode_XXXXXX.npz`：可通过 `np.load(path, allow_pickle=False)` 读取。

每个 episode 有 T 行 `actions`（归一化、截断前）、`clipped_actions`（实际高层输入）、`ctrl`、`actuator_force`、`contacts` 和 `phase`。
`qpos`、`qvel`、`time_s` 及 `observations/<key>` 有 T+1 行：第 t 行观测生成第 t 行动作，第 t+1 行是动作后的状态。
`metadata_json` 是一个 Unicode JSON 字符串，包含完整 state、激励程序、物理配置 hash、模型 hash、终止原因和控制器事件。
`contacts` 的四列为物体最大穿透量、容器底面支撑力、底面接触标志、手指接触标志，均取控制周期末。
成功 latch 保持逐物理子步判断，不从这四个低频诊断列重算。

输出必须是新目录或空目录。episode 原子发布且拒绝覆盖；中断时已有完整 NPZ 保留，manifest 的 `complete` 为 false。
当前不自动续跑。每个 world 的 terminal 状态已在记录中保存，即使其他 world 继续执行也不会继续追加该轨迹。

## 范围与数值边界

只支持当前 direct-mount Panda、默认固定阻抗 OSC_POSE、20 Hz 策略/5 kHz 物理以及 V0–V3；不提供图像采集或局部自动 reset。
没有修改原 CPU runner、oracle 或正式评测流程。输出使用独立 schema，始终 `scoreable=false`。
固定 `mujoco==3.9.0`、`mujoco-warp==3.9.0`、`warp-lang==1.13.0`，因为内核访问固定版本的 MJWarp 数组布局与接触力 helper。
OSC 小矩阵使用 float64 和与参考实现相同的 1e-15 伪逆相对截断，物理状态仍是 MJWarp float32。
MJWarp 将求解 tolerance 下限设为 1e-6，不能将 CPU 的 1e-12 配置视为实际 GPU tolerance；文件记录实际值。
同种子保证输入可追溯，不承诺跨后端或 GPU 重复执行逐位相同。

已在可用的 NVIDIA A100-SXM4-80GB 上实际验证 CUDA Graph；其他设备及完整任务成功率仍需按采集配置验证。

## 验证

```bash
uv pip install --python .venv-gpu/bin/python pytest
.venv-gpu/bin/python -m pytest -q tests/test_shakebench_mjwarp.py
SHAKEBENCH_TEST_DEVICE=cuda:0 .venv-gpu/bin/python -m pytest -q tests/test_shakebench_mjwarp.py
```

测试覆盖控制器矩阵求解（包含秩亏）、移动基座下 CPU/Warp 力矩对照、公开观测、reset/IMU 时序及不可覆盖的数据写入。
首轮完整采集应以相同 states/Gamma 对照 CPU 成功率与失败类型，并查看 manifest 中的无效 episode；不能用单个短程 smoke 推断完整任务成功率。

2026-09-12 验证结果：新增 4 项测试在 Warp CPU 和 A100 CUDA 两种设备上均通过；CUDA 测试启用图捕获。
原 oracle/outcome 的 35 项回归通过。真实 `food_can` task state 的 V3、Gamma=0.1、2-step CLI 采集在 Warp CPU 上通过，
生成的 NPZ 可无 pickle 读取，T+1 帧对齐、模型/文件 hash、实际 tolerance 和终止原因检查通过；保存的 XML 与末态可重新加载。

同日 A100 微基准使用 V0、Gamma=0、direct-mount、20 个动作，同配置重复 world，排除构建和图编译，仅测预热后的 rollout：

| 执行方式 | 总动作数 | 耗时 | 总动作/s |
|---|---:|---:|---:|
| classic MuJoCo，单 CPU 环境 | 20 | 9.89 s | 2.02 |
| MJWarp，1 world | 20 | 10.62 s | 1.88 |
| MJWarp，16 worlds | 320 | 12.94 s | 24.72 |

16 worlds 总吞吐约为单 CPU 环境的 12.2 倍，单 world 没有提速。
CPU 对照包含 oracle、env.step 和 metrics；GPU 计时还包含采集记录构建，两者均不包含磁盘写入。
这是短程、同配置的执行微基准，不是完整 episode/h、任务成功率或相对多进程 CPU runner 的加速比。
