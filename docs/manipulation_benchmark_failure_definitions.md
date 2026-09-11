# 具身操作 benchmark 的任务失败边界调研

调研日期：2026-09-08。核对六套系统的官方协议、任务谓词、环境返回值、评测循环和演示采集逻辑。源码固定到 commit；结论针对列明的协议，不自动推广到使用同一环境的第三方 wrapper。

## 六套系统共同揭示的边界

这次调研支持的结论是：在所查官方评测协议中，**尚未达成目标、违反任务约束、预算用尽，以及控制器内部阶段失败，是不同事件**。Meta-World、LIBERO 和 CALVIN 的所查评测循环没有把一次抓空或一次滑移统一设成立即失败；ManiSkill 的推荐评测还会忽略成功/失败产生的提前 termination，另行记录完整 episode 的结果。RLBench 则展示了任务确实可以显式定义不可违反的条件。各条事实的来源与版本见下文逐项说明。

| 系统／已核查协议 | 未成功时何时结束 | 一次普通抓空／滑移 | 明确例外与限制 |
|---|---|---|---|
| Meta-World v3 官方评测 | 成功时提前结束，否则到 horizon；默认基础上限 500 步 | 所查通用路径不因此终止，可继续行动 | 不能把 `grasp_success=False` 当 episode 失败 |
| ManiSkill 3 官方 RL／IL 评测建议 | 忽略任务 termination，至时间限制统一统计 | 取决于任务状态定义，但评测包装不会仅因该 termination 重置 | 分别记录 success_once、success_at_end、fail_once、fail_at_end；基础 API 默认仍可提前终止 |
| LIBERO 官方评测 | 在配置的 max_steps 内累计是否曾成功，默认配置 600 步 | 没有该通用立即终止门 | 其任务环境把 done 定义为成功；不能套用 Gym 中 done=失败的想当然解释 |
| CALVIN 官方 LH-MTLC | 每条语言子任务最多 360 步；超时截断后续任务链 | 子任务预算内仍可纠错 | 语言子任务失败会停止五任务链；不是一次内部抓取 primitive 失败就截断 |
| RLBench 核心 API／任务定义 | 成功，或任务显式注册的失败约束触发；外部评测另设预算 | 普通 success=False 本身不会要求终止 | `beat_the_buzz` 等任务有显式禁止检测区；本次未找到全仓统一官方计分 harness |
| robosuite 1.5 系列基础框架 | 默认 done 仅由 horizon 决定 | 不由通用阶段机判终止 | 下游 rollout 决定 success_once 或 final success；框架本身不是统一排行榜协议 |

不同系统的“一步”可能是控制步、策略查询或完整规划动作，频率与动作持续时间也不同。上述 50／360／500／600 等计数不能直接换成 ShakeBench 的控制时间预算或相互比较。


## Meta-World v3：未完成目标通常继续运行，首次成功即可计成功

本节源码固定在 `6e01ad7e2ffb2302e4dca04f796fcd8837df8540`（提交日期 2026-08-10）；官方网页按 2026-09-08 读取。以下结论适用于所查官方评测和 v3 Sawyer 环境。

- **评测定义**：官方 MT/ML 协议以 episode 内任意时刻达到任务成功条件计成功。官方 `evaluation()` 会打开成功提前终止 wrapper，再从结束 episode 的 `final_info.success` 汇总；不是用单步 reward 或 `grasp_success` 判整轮输赢。[官方评测说明](https://metaworld.farama.org/evaluation/evaluation/)、[evaluation.py](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/evaluation.py#L53-L99)
- **环境终止**：基础 `SawyerXYZEnv.step()` 正常路径的 `terminated` 始终为 False；达到 `max_path_length` 才置 `truncated=True`，该基类默认 500 步。`AutoTerminateOnSuccessWrapper` 只额外使成功成为 termination。所查路径没有“抓空一次/抓持丢失一次就终止”的通用条件。[step](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/sawyer_xyz_env.py#L580-L642)、[成功 wrapper](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/wrappers.py#L207-L230)
- **PickPlace 实例**：`success` 由物体到目标距离不超过 0.07 m 得到；`grasp_success` 作为另外的诊断字段返回。一次抓取没有成立，本身不会永久否决后续成功。[PickPlace evaluate_state](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/envs/sawyer_pick_place_v3.py#L86-L115)
- **异常边界**：源码保留 `_did_see_sim_exception` 分支，触发时返回最后稳定 observation、零 reward、success=False 和两个 False 结束标记。这是被读取到的特殊异常路径，不足以证明当前所有 MuJoCo 异常都会被捕获或评测一定能正常结束；也不能宣称 Meta-World 会统一删除无效 episode 并免费重试。[异常分支](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/sawyer_xyz_env.py#L603-L619)

**对恢复的含义（推论）**：在任务尚未成功且预算未耗尽时，策略可以继续接近、重抓、重新放置。框架提供继续行动的机会，不保证策略一定具有恢复能力，也没有在本次所查协议中统一限制只能重抓几次。

## ManiSkill 3：失败状态、提前终止和整轮指标可以分开

本节源码固定在 `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`（提交日期 2026-08-02）；在线文档标为 ManiSkill 3.0.1。区分基础环境 API 和官方推荐的评测包装。

- **基础环境 API**：任务的 `evaluate()` 可分别返回 `success` 和 `fail`；`BaseEnv.step()` 以二者逻辑 OR 计算 `terminated`。没有 success/fail 字段时不以它们终止。因此 `terminated=True` 本身并不代表失败。[BaseEnv.step](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/sapien_env.py#L1050-L1080)
- **明确失败条件不是必需的**：官方教程中的 PushCube 只定义成功条件；教程把“物体掉下桌面”作为作者可以另行加入的失败条件示例，不能写成 PushCube 已经实现该终止门。[官方任务教程](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/docs/source/user_guide/tutorials/custom_tasks/intro.md#L223-L246)
- **PickCube 实例**：成功要求物体到目标的距离达到阈值且机器人静止；`is_grasped` 另行输出，这个 evaluate 函数没有返回 `fail`。默认注册的 episode 上限为 50 步，但实验可以显式覆盖，不能把 50 推广到全部 ManiSkill 任务。[PickCube](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/tasks/tabletop/pick_cube.py#L147-L159)
- **官方 RL 和模仿学习评测**：均建议 `ignore_terminations=True`、关闭 success/fail 导致的提前重置，跑至时间限制，并同时记录 `success_once`（曾成功）、`success_at_end`（最后一步成功）、`fail_once`、`fail_at_end`。模仿学习说明明确指出通常报告 success_once。[RL 评测](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#evaluation)、[模仿学习评测](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html#evaluation)
- **源码证实其独立性**：vector wrapper 分别 OR 累积 success/fail；忽略 termination 时把结束标记置 False，同时保留当前与历史指标。这个通用机制不强制“曾失败就永远不能成功”；特定任务的 fail 是否不可逆，要看任务自身实现。[vector wrapper](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/vector/wrappers/gymnasium.py#L127-L162)
- **专家采集另外处理**：官方运动规划采集脚本把规划异常或返回 -1 记成一次不成功尝试；开启 `only_count_success` 时丢弃轨迹、换 seed 继续收集，同时维护尝试成功率。这是演示生产的筛选规则，不是模型评测允许失败后换 seed。[规划采集脚本](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/examples/motionplanning/panda/run.py#L85-L120)

**对恢复的含义（推论）**：中途抓持状态变化可作为诊断，episode 内继续操作与结果质量可以分别测量。若 ShakeBench 采用持续稳定成功条件，仍可允许此前重抓和反弹；不必为了允许恢复而照搬更宽松的瞬时成功条件。

## robosuite：环境默认运行至 horizon，阶段机不参与通用终止

上游源码固定在 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`（提交日期 2026-07-11），官方在线文档为 1.5 系列。robosuite 是框架，具体论文或算法还可以包一层评测逻辑。

- **默认 done**：`MujocoEnv._post_action()` 只按 `timestep >= horizon` 且未开启 `ignore_done` 设置 done。官方文档也说明达到成功条件后环境仍继续运行至固定 horizon。[上游 _post_action](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/base.py#L532-L548)、[官方 Rewards and Termination](https://robosuite.ai/docs/modules/environments.html#rewards-and-termination)
- **PickPlace 实例**：每次检查物体是否处于对应箱体、末端是否足够远，按 single-object 或 multi-object 模式返回成功。检查当前任务状态，没有引用外部 expert 的 approach/grasp/transport 阶段或其局部超时。[上游 PickPlace success](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/manipulation/pick_place.py#L737-L762)
- **评测范围限制**：仅凭 robosuite 的基础 API，不能断定某个下游论文最终采用 success_once 还是 success_at_end；需要再读其 rollout 脚本。本次不把框架示例当作所有下游研究的统一规则。

**对 ShakeBench 的直接关系**：ShakeBench 的 `controller_failed` 提前停止由自身 runner 增加，并非继承 robosuite 就必须采用的规则。该判断来自本地 [runner](../robosuite/scripts/shakebench_run_oracle.py#L610)，以及上述上游 done 实现。

## LIBERO：官方评测按预算内曾经成功计分

版本：官方 `Lifelong-Robot-Learning/LIBERO` commit [`8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/commit/8f1084e3132a39270c3a13ebe37270a43ece2a01)，commit 日期 2025-03-15。这里研究的是仓库自带 lifelong 评测，不是 OpenVLA 等下游项目另设步数的评测协议。官方文档明确将 `lifelong/main.py` 和 `lifelong/evaluate.py` 作为复现实验入口。[官方评测入口说明](https://lifelong-robot-learning.github.io/LIBERO/html/research_topics/overview.html)

### 任务判定与环境返回值

- `BDDLBaseDomain.step()` 先调用底层环境，再用 `done = self._check_success()` 覆盖对外返回的 `done`。因此这里的 `done` 在该接口上表示任务成功，不能按通常 Gym 语义把任意 `done` 理解成成功或失败都有可能的 episode 结束。[`bddl_base_domain.py:800–809`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py#L800-L809)
- 基类 `_check_success()` 自身是占位方法；必须继续跟踪任务子类。核对的 `Libero_Tabletop_Manipulation._check_success()` 对 BDDL 的全部 goal predicates 求合取。[`libero_tabletop_manipulation.py:135–159`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/problems/libero_tabletop_manipulation.py#L135-L159)
- 具体例子“把桌面中央的黑碗放到盘子上”的 BDDL goal 是 `On(akita_black_bowl_1, plate_1)`；没有把“第一次抓取必须成功”列为目标。[任务 BDDL:131–133](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl#L131-L133)
- `ControlEnv.step()` 原样转发底层结果，`check_success()` 直接查询底层成功方法，没有在这一层增加抓取失败计数。[`env_wrapper.py:87–104`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py#L87-L104)

### 官方 evaluator 怎样判定失败

`metric.py` 的 `evaluate_one_task_success()` 每次 reset 后加载预定义初态，先执行 5 个零动作，再执行最多 `cfg.eval.max_steps` 个策略动作。每一步做 `dones[k] = dones[k] or done[k]`，所有环境都曾成功才提前结束这批评测，最后以累计布尔值统计成功率。到预算末仍从未成功的 trial 贡献 0。默认配置是 `max_steps: 600`、`n_eval: 20`。[`metric.py:102–158`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/lifelong/metric.py#L102-L158)、[`configs/eval/default.yaml:1–10`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/configs/eval/default.yaml#L1-L10)

独立评测入口 `evaluate.py` 同样累计历史 `done`，没有改成只看最后一步，也没有局部阶段失败分支。[`evaluate.py:250–287`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/lifelong/evaluate.py#L250-L287)

**可直接确认的规则**：在这些评测循环中，首次抓空后只要还没耗尽预算、环境仍正常运行，策略仍可继续行动；成功之后暂时失去成功状态不会撤销已累计的成功。

**解释边界**：这不保证物体跌落后的状态一定可恢复，也不证明全部任务谓词都宽松。这里核实的是成功结果如何由环境传入 evaluator，以及 evaluator 没有因抓空/滑移事件立即判负；没有对所有物体几何与所有 task 的可恢复域进行穷举。

### 演示采集不是相同规则

`scripts/collect_demonstration.py` 的人类采集 loop 由输入设备 reset 或成功保持计数结束。首次检测到成功将计数设为 10，持续成功时递减，失去成功时重置；人工 reset 则将该轨迹加入不保存列表。这是演示采集的筛选逻辑，不能用它声称官方测试要求同样的连续成功保持窗口。[`collect_demonstration.py:47–101`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/collect_demonstration.py#L47-L101)

数值/基础设施异常也不等同于普通任务失败：例如 `metric.py` 对环境创建失败重试最多 5 次后抛异常，没有在这里把创建失败简单记作一个成功率为 0 的 trial；策略执行中其他异常的完整处理不在本次已验证范围内。[`metric.py:81–100`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/lifelong/metric.py#L81-L100)

## CALVIN：允许指令内部纠错，但指令链遇到超时就停止

版本：官方 `mees/calvin` commit [`fa03f01f19c65920e18cf37398a9ce859274af76`](https://github.com/mees/calvin/commit/fa03f01f19c65920e18cf37398a9ce859274af76)，commit 日期 2025-09-08；环境使用该仓库 gitlink 指向的 `mees/calvin_env` commit [`1431a46bd36bde5903fb6345e68b5ccc30def666`](https://github.com/mees/calvin_env/commit/1431a46bd36bde5903fb6345e68b5ccc30def666)，没有用环境仓库 HEAD 替代依赖版本。协议是 README 指定的 Long-horizon Multi-task Language Control / CALVIN Challenge。[官方协议说明](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/README.md#-evaluation-the-calvin-challenge)

### 环境层与任务 checker

`PlayTableSimEnv.step()` 在施加动作、推进物理并获取观测后返回 `(obs, 0, False, info)`，环境本体的 `done` 恒为 False。任务是否完成由外部 `Tasks` checker 判断；它比较当前自然语言子任务开始时的状态与当前状态，返回已完成的任务集合。[`play_table_env.py:225–246`](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/envs/play_table_env.py#L225-L246)、[`tasks.py:9–47`](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/envs/tasks.py#L9-L47)

checker 的“False”是当前未满足条件。例如 `lift_object` 检查相对初态的高度变化、与机器人的接触及起始支撑表面；`stack_objects` 还检查目标接触关系及速度阈值。这些方法返回 False，并不会在环境中锁存永久失败。[`tasks.py:122–155`](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/envs/tasks.py#L122-L155)、[`tasks.py:265–284`](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/envs/tasks.py#L265-L284)

### 官方评测 loop

官方脚本设 `EP_LEN = 360`、`NUM_SEQUENCES = 1000`。每条指令开始时调用 `model.reset()` 并保存 `start_info`，然后最多执行 360 步：忽略环境 reward/done，只检查该指令的目标条件；第一次满足立即返回 True，否则循环耗尽返回 False。五任务链仅在链首 reset 环境，之后沿用真实结果状态；某条指令返回 False 时结束整条链并返回已经完成的指令数。[`evaluate_policy.py:38–39`](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/evaluate_policy.py#L38-L39)、[`evaluate_policy.py:124–179`](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/evaluate_policy.py#L124-L179)

结果同时报告平均成功链长度，以及至少连续完成 1、2、3、4、5 条指令的比例。例如完成前 3 条、第 4 条超时的试验，链长度是 3，并非所有统计都记为 0。[`evaluation/utils.py:77–95`](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/utils.py#L77-L95)

**与 ShakeBench 讨论的对应关系**：CALVIN 的 360 步预算约束的是 benchmark 对外定义的一条自然语言任务，例如“拿起红色方块”，不是策略内部的 approach、close_gripper、wait_stable 等 FSM 阶段。不能援引 CALVIN 的“子任务超时终止任务链”，来证明内部恢复等待 1.2 秒未满足条件应当立即判定整个 pick-and-place 不可恢复。前半句是已验证协议事实；后半句是本调研对语义层级的比较。

### 数据采集与未知项

官方 VR 采集代码持续取得人类动作，按 start 按钮开始记录，按 reset 结束并重置环境，长按 reset 可删除当前 episode；`DataRecorder` 保存该人为结束标记。该采集 loop 没有自动专家 FSM 的“某次抓取失败即拒收”条件。[`vrdatacollector.py:31–48`](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/vrdatacollector.py#L31-L48)、[`data_recorder.py:41–94`](https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/io_utils/data_recorder.py#L41-L94)

这不表示最终发布数据从未经过人工质检或后处理筛选；本次未核实每段历史 play 数据的采集后决策。LH-MTLC 之外的 single-task 模式、第三方 wrappers 和 action chunking 预算也不混入以上结论。

## RLBench：显式任务约束可以提前失败，但未成功本身不构成失败终止

版本：官方 `stepjam/RLBench` commit [`02720bba4c73fe02eb75df946b8791b806028a9d`](https://github.com/stepjam/RLBench/commit/02720bba4c73fe02eb75df946b8791b806028a9d)，commit 日期 2025-01-25。

### 核心 API 明确分开 success 和 terminate

`Task.success()` 首先检查注册的非空失败条件列表：当列表内所有条件成立，返回 `(False, True)`；否则检查成功条件，成功则 `(True, True)`，尚未成功则 `(False, False)`。`TaskEnvironment.step()` 将该结果作为奖励与终止返回值交给调用者，未包含控制器阶段 deadline 判定。[`backend/task.py:287–300`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/backend/task.py#L287-L300)、[`task_environment.py:95–109`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/task_environment.py#L95-L109)

`register_fail_conditions()` 的官方说明将失败约束定位为可选条件，给出的例子是脆弱物体跌落或触碰禁止对象。它定义的是任务约束，和普通时刻尚未到达成功目标不同。[`backend/task.py:189–199`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/backend/task.py#L189-L199)

实际例子 `beat_the_buzz` 注册 `DetectedCondition(wand, middle_sensor)` 为失败条件，并要求到达另一端且松开夹爪才成功。任务文字要求环沿杆移动且保持间隙。可以据此确认“进入任务规定的禁触检测区”是专门的提前失败条件；本次没有检查模型文件中的传感器几何，因此不把这个布尔值说成精确测得的物理接触。[`tasks/beat_the_buzz.py:10–33`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/tasks/beat_the_buzz.py#L10-L33)

补充源码搜索：已下载该 commit 的 107 个 `rlbench/tasks/*.py` 文件，按 `register_fail_conditions`、`_fail_conditions`、`def success` 搜索，显式注册上述失败 API 的实例仅命中 `beat_the_buzz.py`。这是可复查的静态搜索结果，不能据此断言其他任务绝不会发生不可达状态、动作异常或任务自身副作用。[固定版本任务目录](https://github.com/stepjam/RLBench/tree/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/tasks)

### 评测预算必须由具体协议补足

本次检索了官方仓库文件树与 README，未找到一个可与 CALVIN `evaluate_policy.py` 或 LIBERO `metric.py` 同等定位、规定所有 RLBench 使用者统一 horizon 的官方测试 runner。官方 `examples/single_task_rl.py` 每 40 步 reset、总计 120 个训练步，并没有按 `terminate` 中断。这只是随机策略使用示例，不能把 40 步写成 RLBench 正式评测预算，也不能据它推导正式指标。[`examples/single_task_rl.py:32–45`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/examples/single_task_rl.py#L32-L45)、[官方仓库](https://github.com/stepjam/RLBench/tree/02720bba4c73fe02eb75df946b8791b806028a9d)

因此已验证的是任务 API 层的失败/终止语义；若要声称“某 RLBench 实验中 IK 失败会计失败”“最多 25 次动作”“超时后怎样计分”，还必须指定那篇工作及其 evaluator。本调研不把第三方 RLBench 使用惯例冒充统一官方规则。

### 自动演示失败不是策略测试失败

自动演示由 waypoint 驱动。路径规划失败会抛 `DemoError`；走完路径后可额外模拟 10 步等待任务达到成功，最终仍未成功也抛 `DemoError`。这些判定针对参考演示生成器。[`backend/scene.py:335–365`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/backend/scene.py#L335-L365)、[`backend/scene.py:433–453`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/backend/scene.py#L433-L453)

`TaskEnvironment._get_live_demos()` 在一次失败后 reset，再尝试生成演示；默认允许最多 10 次尝试，全部耗尽才报“无法收集演示”。这 10 次包含重置世界，不能解释为测试时在同一 rollout 中允许 10 次纠错，更不能把一次坏演示定义成所有策略都无法完成任务。[`task_environment.py:20–22`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/task_environment.py#L20-L22)、[`task_environment.py:141–163`](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/task_environment.py#L141-L163)

## 对 ShakeBench 的解释与建议（本项目推论）

1. **允许过程纠错有直接先例。** 在 episode 内保留相同物体、物理状态和剩余预算，让策略重抓、等待或重新放置，与以上几类协议的正常继续执行逻辑相容。“恢复机会”不意味着跨 reset、替换 seed 或无限时间。
2. **最终成功可以保持严格。** 把 Can 正确放置并连续稳定 0.5 秒，与允许此前出现滑移、抓空、反弹没有逻辑冲突。success_once 在 ShakeBench 中可以表示“曾通过完整 0.5 秒成功窗口”，而非“瞬时落入目标”；是否还需坚持至 episode 末尾，是另一个必须明确的指标选择。
3. **控制器放弃仍是该策略的一次失败，但不是任务不可解的证据。** 当前 runner 检查 executive.phase == failed 并提前结束。若继续把它作为 reference policy 的显式 abort，应命名为 controller_abort／recovery_exhausted／recovery_settle_timeout 等，保留分母，不把它提升为所有策略通用的环境失败条件。[本地 runner](/home/miracle04/Desktop/ShakeBench/robosuite/scripts/shakebench_run_oracle.py:610)
4. **真正的提前任务失败需要独立且可核查的定义。** 禁止区域、任务要求保护的物体、明确不可回收区域等，可以像 RLBench 的任务失败条件一样事先定义。8 mm 桌沿余量、单次失去抓持、局部 1.2 秒未稳定，都不能仅凭名称自动获得“不可恢复”的语义。
5. **要修复的是恢复／放弃决策，而非只删 runner 的 break。** 当前 controller 进入 FAILED 后输出停止任务动作及松爪，继续调用并不会凭空产生恢复策略。[本地 terminal command](/home/miracle04/Desktop/ShakeBench/robosuite/utils/shakebench_oracle.py:2652)
6. **仿真异常处理没有从本次调研得到统一标准。** Meta-World 保留特殊异常返回；RLBench 有 action/demo 异常；ManiSkill 专家采集会捕获规划异常。将 infrastructure failure 与策略导致的违规分别记录，是对 ShakeBench 的建议，不是“所有 benchmark 都不计异常”的事实。无效 episode 的重跑、计分与分母规则必须预先确定，不能观察结果后决定排除。
7. **先定义边界，再校准 Γ。** 后续 Γ crossing 应是在明确恢复预算、任务失败条件和 controller abort 规则之后测得；否则它同时反映任务受扰难度和控制器提前放弃倾向。

当前实现的局部等待分支即使 `_public_recoverability()` 为 True，也可能在 `recovery_settle_s` 耗尽后发出 `public_object_unrecoverable`。[恢复等待分支](/home/miracle04/Desktop/ShakeBench/robosuite/utils/shakebench_oracle.py:2624) 这是前一轮本地合成观测探针已复现的代码行为；本次外部调研未重新运行该探针，也没有据此断言某条真实 demo 必能恢复。

## 本次验证范围

- 调研日期：2026-09-08。只使用官方文档、官方任务定义、环境 API、评测脚本和演示采集源码；未采用第三方算法仓库替代 benchmark 官方协议。
- 所有固定 commit 链接对应已读取的源码；在线文档仍可能随 latest 更新，具体读取日期已记录。部分网站搜索缓存与下载源码行数不同，精确源码行号以固定 commit 下载文件为准。
- 未安装或运行上述外部仿真 benchmark，未做恢复成功率或物理可达性实验；“策略可继续行动”是从控制流得到的结论，不是“继续一定能成功”的实测证明。
- 本次只新增研究文档。ShakeBench 的 controller、evaluator、runner、states 与 Γ 协议均未修改。
