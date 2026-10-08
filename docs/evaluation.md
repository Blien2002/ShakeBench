# 评测指南

本文说明如何用 ShakeBench 评测一个操作策略，以及如何解读结果文件。

## 评测设置

一次评测由三部分决定。

**任务与初始状态。**
- 每个任务随附一个或多个状态资产（JSON）。每条状态固定了物体布局、目标、振动激励种子和 IMU 噪声种子，所以同一条状态的回放可以复现。
- 默认评测资产中的全部状态，`--state-ids` 可以只选其中一部分。
- `pick_place` 是内置任务；其他任务需要 `--task-module shakebench.environments.<模块>`。
- 资产列表见 [tasks.md](tasks.md)。

**振动设置。**
- `--gamma` 是振动强度，`0` 表示台面静止。
- `--mode` 选择激励类型：
  - `multisine_v1`：默认的多正弦激励；
  - `single_sine_v1`：单正弦激励；
  - `custom_multisine_v1`：自定义多正弦，参数用 `--mode-params` 以 JSON 传入。
- `--sway-v1` 使用低频摇摆预设。
- 比较不同振动强度下的表现时，用同一组状态、分别以几个 `--gamma` 值运行。

**观测方式。**
- `--observation-source cameras`（默认）：提供主视角和腕部相机图像、8 维本体状态和台面 IMU 窗口。
- `--observation-source contract`：只提供状态观测，不渲染图像，适合在没有 GPU 的机器上快速检查。

## 策略接口

`--policy module:factory` 指向一个工厂函数，评测器按以下规则调用它：

- 只把函数签名中声明过的 `--policy-arg NAME=VALUE` 传进去。值先按 Python 字面量解析，解析失败就当作字符串。
- 如果函数声明了 `inference_timeout_s` 参数，还会传入 `--inference-timeout-s` 的值。
- 传入未声明的参数会直接报错。

工厂返回的策略对象需要满足：

| 成员 | 要求 |
| --- | --- |
| `chunk_size` | 正整数：一次预测的动作步数 |
| `predict(observation)` | 返回单个动作 `[7]` 或动作块 `[N, 7]`；数值必须有限且在 `[-1, 1]` 内；可以是 numpy 数组或 torch 张量 |
| `deadline_s`（可选） | 适配器自己保证的单次请求时限（秒）。只有声明了它才能使用 `--inference-timeout-s` |

**动作。** 7 维归一化 OSC 位姿增量 `delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper`，控制频率 20 Hz。`--action-horizon` 设定每次预测后执行几步动作再重新调用策略。

**观测（`cameras` 模式）。**

| 键 | 内容 |
| --- | --- |
| `observation.images.main` | 主视角 RGB，默认相机 `task_close`，默认 256×256（`--height`/`--width`/`--main-camera` 可改） |
| `observation.images.wrist` | 腕部相机 RGB（`robot0_eye_in_hand`） |
| `observation.state` | 8 维本体状态：末端位置（机器人基座系，米）、末端姿态（旋转向量，弧度）、左右指关节位置（米） |
| `observation.table_imu_window`<br>`observation.table_imu_timestamps_s`<br>`observation.table_imu_dt_s` | 台面 IMU 最近一段时间的测量窗口，以及对应的时间戳和采样间隔 |

策略拿不到仿真真值：评测器会拒绝带有特权命名空间的观测键。

## 运行方式

**同步评测**（最常用）：`python -m shakebench.scripts.evaluate`。参数见 `--help`，常用参数如下：

- `--horizon-steps`：回合时长上限（控制步数）；
- `--policy-id`：写进结果的策略标签；
- `--output`：结果路径；不传则打印到终端。

**异步评测**：`python -m shakebench.scripts.evaluate_async`。

- 策略推理和仿真时间并行推进，用来考察推理延迟的影响。
- `--time-scale` 设定每个仿真秒对应的墙钟秒数，测得的推理延迟按同一比例缩放。
- 超时会如实报告。
- 配套适配器：`shakebench.policies.async_policy:make_chunk_policy`。

**WebSocket 策略服务**：通过同步入口的 `--policy shakebench.policies.websocket:make_policy` 连接；`python -m shakebench.scripts.eval_websocket_policy --host <地址> --port <端口>` 是单回合 pick-place 简化入口。

- 策略在独立环境中以服务形式运行，评测器通过 `shakebench.policies.websocket:make_policy` 连接。
- 需要能导入策略服务方提供的客户端包（`deployment`）。

**本地检查点**：`shakebench.policies.pretrained:make_policy` 加载本地 LeRobot 风格的检查点，需要安装 `.[policies]`。

**录制回放视频**：`python -m shakebench.demos.demo_policy_video --policy ... --state-id ... --output out/demo.mp4`。

- 需要 ffmpeg 和可用的渲染后端（如 `MUJOCO_GL=egl`）。
- 视频只用于定性观察，不替代评测结果。

## 训练/评测隔离

如果策略是用某个数据集训练的，用 `--dataset` 指向带有已完成 `meta/shakebench_collection.json` 和 `meta/info.json` 的数据集。评测器会读取其中记录的训练状态，拒绝评测与训练状态重合的状态；`--allow-train-states` 可以显式放开这项检查，并写入结果。只有标准 LeRobot 导出、缺少该采集清单的数据集不被此选项接受，需另外核对状态来源。

## 结果文件

结果使用 `shakebench.policy_evaluation` 格式（版本 1）。每个回合记录以下内容：

- 状态和振动设置；
- 时长上限；
- 观测约定；
- 策略标识；
- 实际执行的动作数量（不保存完整动作轨迹）；
- 终止原因；
- 有类型的策略错误。

终止原因取以下值之一：

| 值 | 含义 |
| --- | --- |
| `success_latched` | 任务成功（成功一经判定即锁存） |
| `horizon_exhausted` | 达到时长上限仍未成功 |
| `task_rule_violation` | 违反任务规则 |
| `policy_error` | 策略报错、超时或输出不合法 |
| `policy_abort` | 策略中止了回合 |
| `invalid_execution` | 评测器判定该回合的执行无效 |

汇总时，所有回合都计入分母，包括策略出错的回合。这些结果是策略评测的记录，不是官方成绩单。

## 可复现性

固定状态与种子有助于复现；仿真器版本、硬件、接触几何和相机资产变化仍可能改变结果。`tools/compare_rollouts.py` 可以检查给定动作序列下的数值轨迹，不能保证所有策略或接触场景都等价。
