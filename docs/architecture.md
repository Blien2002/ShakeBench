# 架构

## 包结构

| 包 | 职责 |
| --- | --- |
| `environments/` | 任务环境，一个模块对应一个 `task_type`；`base.py` 中的 `ShakeBenchEnv`（robosuite 扩展点）和 `ShakeBenchTask`（振动台任务的公共初始化）是任务的基类 |
| `tasks/` | 任务注册（`registry`）、pick-place 物体目录（`catalog`）、成功指标、观测权限、共享运行时（`runtime`）、初始状态资产（`states/`） |
| `physics/` | 振动激励程序、六自由度隔振平台、动态台面驱动、物理配置与标定 |
| `sensors/` | 台面 IMU 模型与面向策略的观测约定 |
| `scene/` | 场景外观配置、世界固定几何、相机、视频写入 |
| `models/` | MJCF 场景与物体、资产文件、资产完整性校验 |
| `rollout/` | 面向策略的评测环境封装（`task_env`）与回合结果（`outcomes`） |
| `policies/` | 策略适配器 |
| `evaluation/` | 评测流程与结果格式 |
| `wrappers/` | Gym 接口 |
| `scripts/`、`demos/` | 命令行入口 |

`tests/test_layout.py` 检查两条依赖规则：
- 仿真核心（`environments`、`tasks`、`physics`、`sensors`、`scene`、`models`、`utils`）不导入 `rollout`、`policies`、`evaluation`、`wrappers`；
- 任何库模块都不导入 `scripts` 或 `demos`。

## ShakeBenchEnv：不修改 robosuite 的扩展方式

robosuite 只支持全局固定的物理步长，物理循环里也没有回调。ShakeBench 需要三项能力：

1. **环境自己的物理步长。** 官方物理配置为 0.2 ms（5 kHz）；控制频率 20 Hz，因此每个控制步包含 250 个物理步。
2. **每个物理步前后的回调。** 动态台面（`physics.deck.DeckDriver`）在每一步之前施加振动；台面 IMU 在积分之后采样。
3. **用 alpha=0 隐藏调试站点。** robosuite 用负 alpha 隐藏站点，但 MuJoCo 仍会渲染这些站点的阴影，从而影响相机图像。

`shakebench/environments/base.py` 中的 `ShakeBenchEnv` 继承 robosuite 的 `ManipulationEnv`，复刻其 `step()` 与 `_initialize_sim()` 并加入上述能力。没有注册任何回调的环境，行为与 robosuite 原生完全相同。

一个控制步内，每个物理步按如下顺序执行：

1. pre-physics 回调（台面驱动）。
2. `sim.step1()`，然后 `_pre_action()`（控制器计算力矩），然后 `sim.step2()`。
3. 如果请求了积分后刷新（按 stride 计）：先运行 refresh 回调，再执行 `sim.forward()`。
4. 更新 observables。
5. 如果本步做了刷新：运行 post-physics 回调（IMU 采样、指标记录）。

**robosuite 版本。** robosuite 固定在上游提交 `5ce6643`，`ShakeBenchEnv.step()` 与 `_initialize_sim()` 复刻的就是该版本的实现。升级 robosuite 时需要：
- 对照上游这两个方法的变化，同步修改；
- 用 `tools/compare_rollouts.py` 确认所有任务的回放不变。

## 任务由哪些部分组成

一个任务模块 `shakebench/environments/<task>.py` 包含：

- 环境类：继承 `ShakeBenchTask`（`pick_place` 直接继承 `ShakeBenchEnv`），负责搭建场景、判定成功、输出 metrics。构造函数先校验状态，再调用 `_init_worktable()` 建立物理配置、振动台和 IMU，然后设置任务字段，最后调用 `_init_robosuite()`；
- `default_state()`、状态校验函数和状态资产加载函数；
- 模块末尾的注册语句：`register_task(task_type, TaskDefinition(...))` 和 `register_state_loader(schema_id, loader)`。

`TaskDefinition` 提供以下内容：
- 构建环境的工厂；
- 状态规范化；
- 环境参数；
- 任务描述（task_id 与语言指令）；
- 执行身份指纹（用于检查训练和评测状态是否重叠）。

`shakebench.tasks.runtime.make_environment` 的工作流程：根据状态找到对应的任务定义，统一配置 Panda OSC 控制器、20 Hz 控制、振动激励和 IMU 种子，然后构建环境并 reset。`pick_place` 是内置任务。

## 状态资产

状态资产是带 schema 标识的 JSON。`shakebench.tasks.states.assets.load_state_asset` 统一读取各种 schema 并返回状态列表，不做来源认证（不校验哈希锚定，也不重新生成状态来比对）。支持的 schema 包括：
- 开发状态；
- 保留评测集；
- 物体位姿变体；
- 训练集；
- 各扩展任务注册的 schema。

扩展任务的 schema 需要先导入对应的任务模块才能解析。各任务在构建环境时校验自己的状态，包括版本号。

## 版本与可复现性

资产随仓库发布。运行时不做资产或场景审计，也不对状态资产做来源认证，这与 ManiSkill、LIBERO、RLBench 等主流 benchmark 的做法一致。可比性由版本号保证：任务和状态资产都带版本号，与当前版本不符的状态会被拒绝。改动是否影响行为，用 `tools/compare_rollouts.py` 对拍确认。

## 新增一个任务

1. 在 `shakebench/environments/` 中新建模块。环境类继承 `ShakeBenchTask`，构造函数按 `push_t.py` 的写法调用 `_init_worktable()` 和 `_init_robosuite()`，并实现 `_record_post_physics_metrics()`。
2. 实现 `default_state`、状态校验和状态加载，并在模块末尾完成注册。
3. 把状态资产放进 `shakebench/models/assets/`。
4. 把新任务加入 `tests/conftest.py` 的任务列表，并在 `docs/tasks.md` 中补充说明。
