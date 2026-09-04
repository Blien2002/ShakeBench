# 成熟具身操作 benchmark 的接触物理实现调研

## 问题与结论

本调研检查成熟 manipulation benchmark 是否像 ShakeBench 当前 Phase 06
一样，用独立的 `15 cm` 落下、`0.50 s` 恢复速度和斜面阈值扫描来选择并
冻结 contact 参数。

结论：**没有发现这种做法是主流 benchmark 的发布硬门。** 更常见的
实现是：固定仿真器与资产参数，使用合法无碰撞的初始化或固定 simulator
state，必要时运行若干 settle steps，然后用接触、位置、静止或 symbolic
predicate 定义任务成功，并用 demonstration / task validator 做端到端回归。

ShakeBench 因为要研究振动诱发滑移，确实需要比普通 benchmark 更强的
接触证据；但 `15 cm drop + 0.50 s 内低于固定速度` 更适合作为压力测试和
诊断，不宜继续作为 official physics 发布的必要条件。

## 官方仓库对照

| Benchmark | 物理/初始化实现 | 发布或任务验证方式 | 独立 drop/incline 硬门 |
|---|---|---|---|
| robosuite | MJCF 对象直接携带 `friction/solref/solimp/condim`；placement sampler 使用对象边界站点放置并避免初始重叠 | PickPlace reset 直接写 free-joint pose；成功由对象是否在 bin 及末端距离判断 | 未发现 |
| LIBERO | 建立在 robosuite 上；评测可直接恢复保存的扁平 MuJoCo state，随后 `forward()` 并重建 observation | 调用环境 `_check_success()`；初始状态作为 benchmark 数据发布 | 未发现 |
| ManiSkill | 明确提供 `SimConfig`、solver iterations、仿真/控制频率和默认材料；默认静/动摩擦 `0.3`、restitution `0` | PickCube 把 cube 直接初始化在桌面高度；成功要求对象到目标且机器人静止，并使用接触力判断 grasp | 未发现通用发布硬门 |
| Meta-World | 任务 reset 直接设置对象初始位置与目标随机化 | 按对象/末端位置和任务状态计算 success/reward | 未发现 |
| RLBench | 接触与动态属性主要保存在 CoppeliaSim task model；Python task 注册 success conditions | task validator 对不同 variation 运行并要求大多数 demonstration 成功 | 未发现；官方还明确提醒物体可能滑落出工作区 |
| RoboCasa | 继承 robosuite；placement config 选择 fixture/region，任务 success 常检查对象—fixture 接触、容器包含关系和 gripper far | 接触/几何 predicate 与任务 demonstrations/evaluation | 未发现 |
| BEHAVIOR-1K | 采样 task scene 后显式运行大量 physics steps，再 `keep_still()` 并保存初始 state；动作 primitive 放置后调用 settle | BDDL/object-state predicate；settle 后重新检查目标 predicate | 有 settle，但未发现统一 `15 cm/0.5 s` drop 参数选择门 |

## 一手实现依据

### robosuite / LIBERO / RoboCasa

- robosuite 官方对象文档展示资产 MJCF 直接定义
  `solimp="0.998 0.998 0.001"`、`solref="0.001 1"`、friction 和
  `condim`；placement sampler 使用 bottom/top/horizontal-radius sites
  保证放置范围与无重叠：
  [robosuite Objects](https://robosuite.ai/docs/_sources/modules/objects.html)。
- PickPlace 的 `_reset_internal()` 从 placement sampler 取得 pose 并直接写
  free joint；`_check_success()` 按 bin 几何和末端距离判断：
  [robosuite pick_place.py](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/pick_place.py)。
- LIBERO wrapper 的 `set_init_state()` 恢复保存的 MuJoCo state，执行
  `sim.forward()`、`check_success()` 和 observation 更新；它没有先运行一套
  drop calibration：
  [LIBERO env_wrapper.py](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py)。
- RoboCasa 的 counter pick-place success 明确使用
  `check_obj_fixture_contact()` 加 `gripper_obj_far()`；对象通过 fixture
  placement config 初始化：
  [RoboCasa kitchen_pick_place.py](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/atomic/kitchen_pick_place.py)。

### ManiSkill

- 官方高级任务文档冻结了默认材料与仿真参数：静/动摩擦均为 `0.3`、
  restitution 为 `0`，默认 `sim_freq=100`、`control_freq=20`，并公开
  contact-force query：
  [ManiSkill advanced task docs](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/tutorials/custom_tasks/advanced.md)。
- PickCube 把 cube 的 z 位置直接设为 half-size，即贴桌初始化；其
  `evaluate()` 要求 cube 进入目标并且 robot joint velocity 足够小：
  [ManiSkill PickCube source](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/pick_cube.html)。

这说明 ManiSkill 会显式冻结 simulator/material defaults，并在任务成功里
加入静止条件；但没有把任意高度 drop recovery 变成所有任务的发布门。

### RLBench

- RLBench 任务由 CoppeliaSim model 和 Python success wiring 组成；官方任务
  教程要求 task validator 在不同 variation 上运行 demonstrations，并以
  demonstration 成功作为验收：
  [RLBench simple task tutorial](https://github.com/stepjam/RLBench/blob/master/tutorials/simple_task.md)。
- 官方 README 的 Gotchas 还明确指出，物体可能从夹爪滑落并掉出工作区，
  benchmark 并不自动 safeguard 这类状态：
  [RLBench repository](https://github.com/stepjam/RLBench)。

### BEHAVIOR-1K

- scene/task sampling 脚本在 reset 后运行 `300` 个 physics steps，并在保存
  initial state 前对对象调用 `keep_still()`：
  [sample_b1k_tasks.py](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/scripts/sampling/sample_b1k_tasks.py)。
- symbolic action primitive 在放置对象后运行 settle，再检查对象 predicate；
  若 predicate 不成立则报告放置失败：
  [symbolic_semantic_action_primitives.py](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/action_primitives/symbolic_semantic_action_primitives.py)。

BEHAVIOR 是所查项目中最接近“显式 settle”的实现，但它使用场景/动作级
settle 与 predicate 校验，而不是从统一高度落下并反向调 solver 参数。

## 对 ShakeBench 的判断

### 值得保留

ShakeBench 的核心科学命题是振动下的接触、滑移和操作，因此以下证据比
普通 benchmark 更有必要：

1. 固定 MuJoCo 版本、timestep、solver、摩擦与 contact tuple；
2. Γ=0 时对象合法初始化、支撑接触、穿透和数值稳定；
3. `mu=0.30` 的静态/低速滑移 sanity check，且 analytic Coulomb 结果只作
   对照；
4. scoreable Gamma 下的 actual contact/slip/relative-motion 记录；
5. task evaluator 使用 simulator truth，policy observation 保持隔离；
6. 固定初始状态、seed、demonstration/controller regression 和 failure 分母。

### 应降级为诊断

- `15 cm` drop 不是当前无-bin PickPlaceCan 的典型 reset 或 release 条件；
- `0.50 s` 内速度必须低于 `0.02 m/s` 是 ShakeBench 自定义的压力门，不是
  上述 benchmark 的共同规范；
- 用该压力门反复扩大 `solref/solimp/condim` 候选，会使 contact 参数逐渐
  针对一个人工 fixture 过拟合；
- 它可以保留为 appendix stress test，报告 rebound、penetration、contact
  loss 和能量，但不应决定 official profile 是否可发布。

## 建议的最小 Phase 06 contact freeze

1. 从 robosuite/Can 的合法 contact profile 出发，固定完整 MJCF tuple 与
   MuJoCo 版本，而不是继续按 drop 结果无限搜索。
2. 使用真实任务初始 pose，在 Γ=0 下运行一个预注册 settle 窗口；要求末端
   有命名支撑接触、无 warning/NaN、穿透在界内、连续若干采样速度稳定。
3. 修复 incline fixture：必须先通过 `0 rad` 水平 control，位移从 post-settle
   pose 计算，再测有限角度 bracket。它是摩擦 sanity check，不是精密材料
   反演。
4. 保留一次 15 cm drop 作为非阻塞 diagnostic，公开完整 trace；不要用它
   继续调 official contact。
5. 在最终 profile 上运行 Gamma=0 passive support、目标 Gamma 的振动接触
   response、三进程 replay 和后续 Phase 07 oracle/demo regression。

这套门比 robosuite/LIBERO/RLBench 更严格，因为它直接服务于 ShakeBench
的振动物理主张；同时避免让 benchmark 发布被一个非任务分布的 drop stress
fixture 无限阻塞。

## Phase 06F 规范归并

Phase 06F 将接触证据分成发布硬门与公开诊断，且不修改已经冻结的任务成功
判据。official contact 发布硬门只有：完整编译后 contact tuple 与显式 Can pair
作用域；Can 在 open worktable 与 target bottom 上按任务原生 pose 初始化；
`Gamma=0`、zero action 下被动 settle `0.50 s`；最后连续 `0.10 s` 有命名支撑
接触，线速度 `<=0.02 m/s`、角速度 `<=0.20 rad/s`；非法穿透
`<=0.50 mm`、接触力有限且无 warning/NaN；冻结 `mu=0.30` 的水平面力阈值
与单轴滑移 sanity；finger force/penetration envelope；以及三档 timestep 接触
收敛和跨进程确定性重放。

`0.15 m` free-drop/recovery、斜面阈值/bracket、placement、free-table slip、
in-hand slip 与 contact-loss 指标仍完整公开，但只作为诊断，不参与候选资格或
排序。斜面 fixture 的既有大偏差也必须保留并标注，不能代替水平力阈值硬门。

这一划分与成熟具身 benchmark 的实际发布方式一致：冻结 simulator/material
配置，在任务原生初始化上 settle 并验证任务 predicate。`0.15 m` 高落体不属于
无 source bin 的 PickPlaceCan reset/release 分布，不能继续驱动 contact tuple
过拟合。降级 standalone drop 不会削弱任务成功：policy release 后仍必须连续
`0.50 s` 满足既有 containment、相对线/角速度、无 finger contact、target-bottom
支撑和穿透条件。
