# 操作 benchmark 的 oracle 设计对比

调研日期：2026-09-06。范围：RLBench、Meta-World v3、ManiSkill 3、CALVIN 的官方文档与当前公开源码，以及 ShakeBench 当前工作树（含已有未提交改动）的设计分析和轻量验证。外部源码链接指向可变分支，因此下列结论描述此次读取的实现，正式实验应另行固定 commit。

## 先区分两个接口

“Oracle”可能指输出动作的 privileged/scripted expert，也可能只是读取仿真状态的成功判定器。CALVIN 的 `task_oracle` 属于后者；RLBench/ManiSkill 的自动演示生成器属于前者。二者需要分别评价：专家策略是否可靠、使用哪些信息，以及成功定义是否独立、符合任务语义。这个区分来自下述各仓库调用链，而非统一行业命名标准。

## 可比事实

| Benchmark | 动作生成方式 | 真值信息与闭环程度 | 成功判定、失败处理 |
| --- | --- | --- | --- |
| RLBench | 作者指定 waypoint，规划器连接路径，执行夹爪指令 | task 可按物体真值位置重设 waypoint；支持按现场结果重复动作 | 独立 task conditions；未成功的 demo 抛错，采集层可 reset 重试 |
| Meta-World v3 | 每任务 scripted policy；PickPlace 是位置阈值分段和 P 控制 | `get_action(obs)` 每步读取手、物体、目标位置；依赖状态观测 | 环境输出 `info.success`；正式评测统计 episode 内是否曾成功 |
| ManiSkill 3 | task solution 构造抓取与目标位姿，再调用运动规划和关节控制 | PickCube 直接读取仿真物体 OBB/pose、TCP/goal；读取真值不等于每步重规划 | `evaluate()` 定义成功；可只保存成功 demo，同时跟踪全部尝试成功率 |
| CALVIN | 此处 `task_oracle` 不输出动作；动作由受测 model 生成 | checker 读取子任务开始/当前 scene info，model 读取 obs 和语言 | 状态变化谓词判定，长序列中失败就停止计数；数据来自 human play |

表格依据见以下逐项源码。不能将这里读取的一个 task policy 的实现细节推断为该 benchmark 所有任务的保证。

### RLBench：waypoint expert 可以包含任务层反馈

`Scene.get_demo()` 获取 waypoints，逐段获取路径并通过物理 step 执行；夹爪关闭时还调用 `gripper.grasp()`。最后独立检查 `task.success()`，不成功则抛出 `DemoError`。因此“到达最后 waypoint”并不自动等于任务成功；同时其 grasp/碰撞处理语义不能未经验证就等同其他仿真器的纯接触抓取。[官方 scene.py](https://raw.githubusercontent.com/stepjam/RLBench/master/rlbench/backend/scene.py)

官方复杂任务教程在每步检查哪些物体已进入容器，并通过 `register_waypoint_ability_start` 和 `register_waypoints_should_repeat` 更新抓取 waypoint、重复尝试。它明确考虑失败抓取，并展示条件集合和顺序条件的表达方式。[官方 complex_task 教程](https://github.com/stepjam/RLBench/blob/master/tutorials/complex_task.md)

采集器 `_get_live_demos()` 默认每条 demo 最多尝试 10 次，每次先 reset；只把成功返回的 demo 加入数据集。保存了随机状态并提供恢复到 demo 初始状态的接口。这是生成器的接受策略，不能把返回数据集的成功率当作一次尝试的成功率。[官方 task_environment.py](https://raw.githubusercontent.com/stepjam/RLBench/master/rlbench/task_environment.py)

### Meta-World：简单规则与实时状态反馈是成立的 expert 设计

`SawyerPickPlaceV3Policy` 从 obs 解析手位置、夹爪开度、物体位置和目标位置；按 XY 对齐、下降、等待夹爪闭合、移向目标几段决策；以 `p=10` 产生位置增量。每次调用重新根据当前 obs 决策，且无需完整路径规划器。[官方 PickPlace policy](https://raw.githubusercontent.com/Farama-Foundation/Metaworld/master/metaworld/policies/sawyer_pick_place_v3_policy.py)

官方专家使用示例直接执行 `policy.get_action(obs)` 和 `env.step(a)`。这提示“读取物体真值”是否构成特权要相对受测 policy 的观测协议判断：在同样开放状态的 MT 设置里不能一概称为额外特权；相对 RGB policy 则是明显的信息优势。[官方 Expert Trajectories](https://metaworld.farama.org/benchmark/expert_trajectories/)

PickPlace 环境的 `evaluate_state()` 用物体到目标距离不超过 0.07 m 判成功，另行输出 grasp_success 等诊断字段。成功不依赖 scripted policy 正处于哪个阶段。[官方 PickPlace env](https://raw.githubusercontent.com/Farama-Foundation/Metaworld/master/metaworld/envs/sawyer_pick_place_v3.py)

官方评测协议使用成功率而非累计奖励，episode 内任何一步成功即可计成功。[官方 Evaluation](https://metaworld.farama.org/evaluation/evaluation/)

### ManiSkill：将演示规划、物理执行和质量筛选分开

PickCube solution 直接读取物体 OBB、物体 pose、当前 TCP 朝向与目标 pose，构建 grasp pose，依次执行 reach、grasp、close、move-to-goal。因此它是以特权几何构造规划目标的 expert，而不是图像感知策略。[官方 PickCube solution](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/examples/motionplanning/panda/solutions/pick_cube.py)

两指夹爪控制器通过 `env.step()` 执行规划关节位置（或位置与速度），开闭夹爪也是保持当前关节位置执行多个 step。路径跟踪并不意味着每一步都重新估计物体并重规划。[官方 two_finger_gripper motionplanner](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py)

生成脚本支持 `--only-count-success`：失败时递增 seed、丢弃该轨迹，再继续直到达到成功数量；同时维护所有尝试的 success 列表、规划失败计数，并报告成功率、规划失败率、轨迹长度。轨迹元数据标记 `source_type="motionplanning"`。这说明成功筛选是显式数据生产操作，而不是 benchmark episode 被允许无限重开。[官方 run.py](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/examples/motionplanning/panda/run.py)

PickCube `evaluate()` 同时要求物体到目标的距离达标以及机器人静止；`is_grasped` 单独返回。不要把这个静止条件误读成“物体在震动下持续稳定若干秒”。[官方 PickCube env](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/envs/tasks/tabletop/pick_cube.py)

学习文档明确区分 `success_once` 与 `success_at_end`，并指出不同演示来源和仿真后端会影响学习结果；一般模仿学习报告 success_once。[官方 Learning from Demonstrations setup](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html)

### CALVIN：oracle 是状态变化裁判，不是操纵策略

评测脚本中动作由 `model.step(obs, lang_annotation)` 产生；`task_oracle.get_task_info_for_set(start_info, current_info, {subtask})` 仅判断目标是否达成。默认 1000 条序列，每子任务最多 360 步；序列开始 reset 环境，成功进入下一子任务而不重置场景，失败则返回已完成数量。[官方 evaluate_policy.py](https://raw.githubusercontent.com/mees/calvin/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py)

`Tasks` 将任务定义为开始与结束 scene/robot info 的变化。例如旋转任务还约束非目标轴旋转、平移距离和接触关系。因此 checker 可以表达“做了某种操作”，而不仅是“当前物体恰好在某坐标”。[官方 tasks.py](https://raw.githubusercontent.com/mees/calvin_env/main/calvin_env/envs/tasks.py)

官方数据集每个环境包含 6 小时 teleoperated play，共 4 个环境。不能把 CALVIN task_oracle 误认为这些动作演示的生成器。[官方 dataset README](https://github.com/mees/calvin/blob/main/dataset/README.md)

## 对摇晃/外扰操作 benchmark 的设计推论

以下是根据上述接口差异提出的建议，不是其他 benchmark 已验证适用于 ShakeBench 的实验结论。

1. 特权状态 + 手写阶段策略是合理的实现类型。可靠性应由完整评测分布上的表现证明，不由“oracle”命名或是否采用规划器决定。
2. 明确 oracle 用途：可解性参考、可部署控制基线、成功演示生成器需要不同承诺。一个可失败的 expert 不构成数学性能上界。
3. 固定 seed 集上报告一次尝试成功率、允许同 episode 恢复后的成功率、采集接受率及失败原因。不要以成功筛选后的数据取代真实难度分布。
4. 分离 policy 的阶段完成信号和环境的任务成功裁判；裁判最好能给出距离、接触、保持时长等独立证据。
5. 对外扰任务保留 success_once 便于诊断，但核心指标应按任务语义定义扰动后 success_at_end 或持续保持成功；普通 tabletop benchmark 的“碰到目标即成功”不自动适用。
6. 除物体 pose 外，还要明确专家是否看到基座/平台速度、加速度、相位和未来轨迹。当前真值反馈与预知未来扰动是两种信息预算，应分开命名和报告。
7. 若提高 expert 的恢复能力，应在同一 episode 的时间/动作预算内检验重抓取、重新定位、重规划；另行披露用于数据生产的跨 reset/seed 重试。
8. 比较 learned policy 与 expert 时固定动作空间、控制频率、物理后端、初始状态和终止协议，并记录演示来源。若不同就说明差异，避免把 controller/physics 优势归因于策略能力。

本笔记未运行上述外部 benchmarks，不声称某一 expert 在全任务或外扰条件下达到特定成功率。

## ShakeBench 当前设计判断

结论：公共观测、任务执行、振动估计、共享控制律、独立评分器的分层合理；当前控制器适合作为受限信息的参考基线，但尚无充分证据支持将其称为可靠 expert 或性能上界。更迫切的改进是控制语义与真实任务验证，而不是增加更多审核门或直接引入复杂规划器。

本地依据：

- [控制器](../robosuite/utils/shakebench_oracle.py)：`WorktableTaskContext`、`TaskExecutive`、`VibrationEstimate`、`SharedVibrationControlLaw`、`ShakeBenchOracleController`。
- [观测协议](../robosuite/utils/shakebench_providers.py)：V0–V3 共享物体/目标/机器人状态，新增通道累积；V0 仍可由公开目标位姿历史推断部分运动。
- [评分器](../robosuite/utils/shakebench_metrics.py)：目标系接触、包络、相对速度与连续保持条件。
- [runner](../robosuite/scripts/shakebench_run_oracle.py)：独立读取环境成功，记录完整动作和执行器结果，失败保留在 episode 分母。
- [当前报告](phase_07_report.md)：首部状态为 `BLOCKED_BY_PHASE_07R4_GRASP_TRAJECTORY`，早期 10/10 和 smoke 证据已撤销；R4 完整 rollout 尚未形成最终证据。本次没有重新运行长时 MuJoCo episodes，不能把历史 R2/R3 失败当成新 R4 实测结果。

### 应保留的设计

1. 控制器只接受公开 State、静态任务上下文和分层通道，不直接持有 `env.sim`；这些真值状态在 State track 内属于合法输入，相对 Vision track 才是额外信息优势。
2. V0–V3 共用执行器、阶段逻辑、动作空间、20 Hz 频率与参数，对“固定算法下新增信息的收益”有明确解释。
3. 策略判断抓取/放置是否可信，环境独立判断任务是否成功。连续 0.5 s 支撑和相对稳定适合振动任务，不应因其他 benchmark 使用更简单的 success_once 就放松。
4. worktable-local anchor、试抬后确认跟随、滑移检测、有限恢复预算，均与振动接触任务有关；无需为了追求简单而全部删除。
5. 固定 dev/knee/official 划分、配对 state IDs 和完整失败记录，优于在评测时重采直到成功。

### 具体实现问题与设计风险

| 优先级 | 证据 | 影响与建议 |
| --- | --- | --- |
| P0 | `ShakeBenchOracleController.action()` 的下降动作限幅只枚举 `DESCEND/GRASP`，当前阶段机使用 `VERTICAL_DESCEND/GRASP_CLOSE` | 新阶段绕过 `descend_position_action_limit=0.30`。按阶段能力统一约束，避免新枚举漏继承。轻量探针固定 desired delta 为 0.04 m，旧阶段输出 0.30、新阶段输出 0.80；这是 action 映射验证，不是完整轨迹实验。 |
| P0 | `_update_public_kinematics()` 对 Can 在 robot-base 中的位姿做差分；`_public_verify_ok()`、对齐/恢复稳定判断复用该速度 | 评分器使用 Can 相对 target 的速度，两者语义不同。探针让 Can 和 goal 在 50 ms 内共同平移 5 mm，相对位姿不变，控制器仍报告 0.10 m/s 并清零稳定计数。改为先计算 Can-in-target/worktable 位姿，再差分；可仅用 V0 合法观测实现。该不一致可能影响进入抓取和恢复，并非只影响最终评分。 |
| P0 | `_phase_goal(APPROACH)` 仍直接设为实时 Can+高度，抵达后才进入 `CLEARANCE_LIFT`；`public_tool_clearance_certificate()` 仅在 `ALIGN_SETTLE` 检查当前两指点 | 新的阶段名称尚不能证明初始斜向 approach 或水平运动过程无碰撞。先建立安全高度，再水平对齐；验证完整工具包络和路径。此项是代码风险，未以新 rollout 证明碰撞必然发生。 |
| P1 | V1 将去重力 deck IMU 运动放入 `current_linear_accel_m_s2/current_angular_velocity_rad_s`；V2 则放入 table 相对 deck 的加速度/角速度；共享律对二者施加相同增益 | 统一类型和单位未统一被估计的物理对象。V1→V2 变化同时包含信息和信号定义变化，难以解释为同一个状态估计变准。应定义明确的共同预测量，例如目标相对 robot-base 的运动，并使各 tier 从合法输入估计它；或把基座惯性扰动与目标相对运动分成独立通道。坐标变换、作用点、时延也应成为契约。 |
| P1 | V3 以稳态线性隔振传递函数预测固定 0.1 s 后加速度，再以固定增益叠加到位姿误差 | 属于启发式 preview，不是规划最优解或完整未来状态。当前预测在 ramp 中仅乘包络，而公开 authored-motion 重建包含包络一阶/二阶导数项；也未以当前 realized state 初始化隔振瞬态。应检验预测误差、相位/频率响应，并考虑短时轨迹预测、阶段相关前馈和不确定性门控。无需预设必须采用 MPC。 |
| P2 | `_anchor_error_m()` 与 `anchor_drift_tolerance_m` 存在但未被阶段逻辑调用；新旧阶段并存 | 冻结 anchor 有助于阻止推着物体追赶，但物体自然滑移后仍需明确定义何时安全退回、重新锚定。建议以运动/抓取/恢复 primitive 封装阶段入口、退出条件和安全不变量。 |

这些问题不能推出整个 benchmark 的物理设计无效，也不能证明控制器失败对应任务不可解。尤其应避免用加大摩擦、减小振动或放宽成功条件掩盖动作映射和轨迹问题。

### 实验结论应如何限定

当前共享律近似是 `位姿误差 - K_a·当前加速度 - K_w·当前角速度 - K_f·未来加速度`，包含逐通道限幅。相同参数有助于隔离软件差异，却不保证合理利用所有 tier 的信息。

可支持的命题是“这套冻结参考控制器在不同合法信息通道下的表现”。它不能直接支持“V3 信息不如 V2”或“获得未来激励也无法完成任务”。更高 tier 的最优策略理论上可以忽略新增信息，但固定启发式控制器没有性能单调保证。V0 的公共目标位姿历史也含有运动信息；增设 V0-history 基线可以检验专用振动传感器相对已有状态反馈的增益。

建议保留主协议的共享控制器配对比较，额外将第二种控制器、history estimator、preview 关闭等作为明确标注的消融，使用相同 dev 调参预算与冻结规则。如果需要可解性诊断，可另建使用真值几何和规划的 diagnostic expert，保持物理、动作、力矩/夹爪约束不变，单独披露权限，不混入受限信息主结果，也不称其为数学上界。

[规范第 12 节](robosuite_benchmark_v0_spec.md) 用 V0 在 100 knee states 上校准 `Gamma_star`，要求无振动成功率至少 0.95，并在独立 400 official states 上评分。这种标定可以成立，但 `Gamma_star` 是相对冻结参考控制器的难度点，不是环境固有常数。先修好并冻结控制器，再完成 knee 标定；改变控制器后的新标定应作为新版本记录，不能基于 official 表现反调基线。

### 建议执行次序

1. 补齐新阶段动作限幅、相对稳定判断和安全 approach，先复现 state-002，再完成冻结十状态的 V0/Gamma=0 验证。十状态全成功是回归门，不能替代更广分布可靠性评估。
2. 统一各 tier 估计对象/坐标/作用点/时序，在已知平移、旋转、共同运动、目标相对运动场景检验估计和控制方向。
3. 在相同 dev states 比较关闭补偿、当前状态补偿和 preview；报告阶段耗时、抓取成立率、滑移/恢复、动作饱和及预测误差，不以 tier 排名为调参目标。
4. 冻结模型和参数后再按已有协议进行 knee/official 实验。每个 episode 保留初始失败与恢复记录，可报告首抓成功率、恢复后最终成功率和恢复次数；跨 reset 的 demo 采集接受率单列。

### 本次验证边界

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_shakebench_oracle.py`：23 passed，0.58 s。直接 `pytest` 首次被系统 ROS 插件/模块路径影响而未完成收集；使用同一 Python 并隔离自动插件后通过。
- 执行了两个不读写模拟器的最小探针，分别验证新旧阶段限幅差异与 Can/goal 共同运动被记作非零速度。
- 未运行完整 MuJoCo 成功率矩阵，也未实测外部 benchmark；本次仅新增研究文档，没有修改控制器、评分器、物理配置或用户已有改动。
