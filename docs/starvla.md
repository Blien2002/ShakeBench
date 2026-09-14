# StarVLA 接口

ShakeBench 负责仿真、oracle 数据采集与 StarVLA 客户端。模型、训练器和 IMU 融合研究均在外部 StarVLA 仓库实现。
适配基准为官方稳定分支 `starVLA`，commit `3422b9f2387b6f682cf02802904a77b23ab13afd`。
旧 `shakebench_vla.py`、`VLATask`、通用 policy dispatch 和对应文档/测试已废除。

## 数据与注册

按 [采集说明](lerobot_collection.md) 安装采集依赖后，生成新数据目录：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m robosuite.scripts.shakebench_collect_lerobot \
  --output /data/shakebench/oracle-gamma-zero
```

输出为 LeRobot v2.1，无损 PNG 嵌入 parquet；包括双相机、归一化 OSC 动作、8 维机器人状态、
桌面 IMU 窗口/时间戳/采样间隔、英文 task 和终止信息。额外的 `meta/modality.json` 将
`main/wrist`、`action.osc/gripper`、`state.proprio` 与 `task_index` 映射给 StarVLA。
先前生成的缺少 state 的样例不能补造真实 state，需重新采集。

注册指向的训练数据必须是成功子集：先按 [采集说明](lerobot_collection.md) 导出
`meta/shakebench_sft_subset.json` 标记的成功 episode 目录，再让 `data_root_dir`/mixture 指向该目录。
未筛选的采集目录包含 timeout、`task_rule_violation` 等失败 rollout，直接训练会把这些示范当作正常行为。

在外部 StarVLA 仓库注册本接口（将路径替换为实际路径）：

```bash
ln -s /path/to/ShakeBench/integrations/starvla /path/to/starVLA/examples/ShakeBench
```

使用 StarVLA 自身的 QwenOFT 训练配置，覆盖这些字段：

```yaml
framework:
  name: QwenOFT
  action_model:
    action_dim: 7
    action_horizon: 8
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: /data/shakebench
    data_mix: shakebench
    lerobot_version: v2.0
    action_mode: abs
    include_state: false
    obs_image_size: [224, 224]
    video_backend: torchvision_av
```

这里 `v2.0` 是所适配 StarVLA loader 对 v2 系列目录的内部开关，文件格式仍为 v2.1。
`action_mode: abs` 表示原样加载已经是 delta OSC 的命令，不再对相邻动作做差分。
action window 固定 t…t+7；位置/旋转命令在 episode 尾部补零，夹爪保持最后命令。
训练前在 StarVLA 环境运行其官方 loader，使用你的完整训练 YAML：

```bash
python starVLA/dataloader/lerobot_datasets.py --config_yaml /path/to/shakebench_train.yaml
```

提供的注册为双图像加语言 baseline。机器人状态和 IMU 保存在原始数据中，默认不输入模型。
若要使用它们，应在 StarVLA 侧同步修改训练与部署的输入处理、归一化和模型配置；本仓库不设计融合模型。

## 部署

StarVLA 模型服务运行在独立环境，使用上述注册训练得到的 checkpoint 和同次训练统计量：

```bash
python deployment/model_server/server_policy.py --ckpt_path /path/to/checkpoints/steps_XXX_pytorch_model.pt --port 10093
```

ShakeBench 环境只需官方 client 的依赖 `websockets>=14`、`msgpack`、`typing_extensions`，
并将外部 StarVLA 仓库放入 PYTHONPATH：

```bash
PYTHONPATH=/path/to/starVLA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m robosuite.scripts.shakebench_run_starvla --host 127.0.0.1 --port 10093 --action-horizon 1
```

`shakebench_run_starvla` 只是该 WebSocket 客户端的薄封装，转发给模型无关入口
`python -m robosuite.scripts.shakebench_evaluate --policy module:factory`。任何满足 `chunk_size` 与
`predict(observation)` 的对象都能用同一 runner 评测，结果记录 state_id、Gamma、horizon、观测合同、
实际执行动作摘要、policy 身份与分类后的 policy 错误。

用训练过的数据集评测时传 `--dataset <采集目录>`：runner 从中读取相机与分辨率，取代默认 256x256 `task_close`，
并检查评测状态与训练状态无交集（确实要复现训练状态时才用 `--allow-train-states`）。
`--inference-timeout-s` 给每次请求设置硬 deadline；超时记为 `policy_timeout`，不会无限等待。

客户端发送 `examples=[{image: [main, wrist], lang: instruction}]` 和 `unnorm_key=new_embodiment`，
图像使用与固定版本 loader 一致的 PIL resize 到 224×224，采集原图仍为 256×256。
接收 `data.actions[0]`，检查形状、有限性及 [-1,1] 范围后执行。服务端执行训练统计量的反归一化。
恢复后的单位已经是数据中的 OSC 命令，不再除以 0.05/0.5；夹爪 -1 打开、+1 闭合，
不要套用 LIBERO 的夹爪翻转，也不要对当前 MuJoCo Renderer 图像旋转 180 度。
StarVLA 按 embodiment tag 保存统计，因此单独训练 ShakeBench 时统计键为 `new_embodiment`，
数据 mixture/robot_type 名仍为 `shakebench`。客户端同时校验 `action.osc/action.gripper` 合同，
现成 LIBERO checkpoint 不匹配。本注册用于单一 ShakeBench mixture，不混合其他 NEW_EMBODIMENT 数据。

`action-horizon` 可设 1…8。仿真在推理期间暂停，报告墙钟推理耗时；结果 `scoreable=false`。
接口测试不加载模型权重；真实 checkpoint 成功率需在 StarVLA 训练完成后评测。

可重复验证命令（外部 StarVLA 及其依赖可导入时）：

```bash
PYTHONPATH=/path/to/starVLA SHAKEBENCH_TEST_DATASET=/data/shakebench/oracle-gamma-zero \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_shakebench_starvla.py
```

测试覆盖官方 WebSocket client 通信和官方 loader 读取真实数据。未设置数据路径时，loader 测试会跳过。

本次验证：官方稳定分支 loader 回读、动作归一化/还原、WebSocket 通信及动作执行协议共 4 项通过；
两条真实 MuJoCo 短 episode 的 LeRobot 回读与独立重放测试通过，真实环境 reset/step 超时终止检查通过。
未提供训练后的 StarVLA checkpoint，尚未验证学习策略成功率。

官方依据：[loader](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/starVLA/dataloader/gr00t_lerobot/datasets.py)、
[server wrapper](https://github.com/starVLA/starVLA/blob/3422b9f2387b6f682cf02802904a77b23ab13afd/deployment/model_server/policy_wrapper.py)。
