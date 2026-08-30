# ShakeBench robosuite `spike/` 实现合理性正式审计

> 审计日期：2026-08-28  
> 审计范围：`spike/` 的振动甲板、六自由度隔振工作台、robosuite / MuJoCo 接入、接触与指标、现有验证证据。  
> 目标拓扑：振动甲板同时刚性承载机械臂基座，并通过隔振器承载工作台与工件。

## 1. 结论

`spike/` 的**核心物理方案合理，当前产品化实现不合格**。

更精确地说：

- 它足以证明 robosuite 1.5.2 + MuJoCo 3.3.7 能承载目标拓扑和规定性六轴基座激励；
- 它足以作为新仓库的参考实现、对照 oracle 和回归数据来源；
- 它不应被整体复制后直接宣布为正式 benchmark backend；
- 新项目应保留其物理模式，重写其工程边界、测量协议和发布级验证。

因此，本审计不支持两个极端结论：

1. “`spike/` 已经成熟，只需升格目录”——证据不足；
2. “`spike/` 路线不可信，应该全部推翻”——与源码、官方建模建议和实测结果不符。

推荐决策是：**复用模型，不继承原型债务。**

## 2. 已确认的目标拓扑

用户澄清的拓扑是：

```text
prescribed shaker excitation
→ dynamic vibrating deck
   ├── robot base（刚性安装）
   └── 6-DoF isolator
       └── worktable + object
```

这与当前代码一致：

- `spike/isolator.py:1-5` 明确写明机器人硬装在 `deck`，工作台是同一 `deck` 的子体，但通过六个柔顺关节恢复支撑动力学；
- `spike/env_shakedeck.py:114-121` 将 `robot0_base` 和 `table` 都放入 `deck`，仅对 `table` 插入隔振关节；
- 本地编译模型检查得到 `robot0_base.parent == deck`、`table.parent == deck`、`deck.parent == world`。

所以拓扑项判定为 **PASS**。

## 3. 分项判定

| 分项 | 判定 | 可否直接用于正式 benchmark |
|---|---|---|
| 目标机械拓扑 | PASS | 可以保留 |
| mocap + weld 驱动动态甲板的原理 | PASS | 可以保留 |
| 当前 `_pre_action` + 一步超前调度 | CONDITIONAL | 需重构和补验证 |
| 六自由度线性隔振器的基础方程 | PASS | 可作为 canonical lumped model |
| 当前隔振器的物理覆盖 | CONDITIONAL | 不能声称等价于真实隔振硬件 |
| 六轴随机谱与解析导数 | PASS | 需抽成独立 NumPy 公共 API |
| Γ 作为甲板输入标度 | CONDITIONAL | 必须同时记录实际甲板/桌面响应 |
| 显式 cube-table 接触 pair | PASS | 设计纪律值得保留 |
| 摩擦变化的物理归因 | FAIL | 尚未做解析/台架校准 |
| 滑移、placement、success 指标 | FAIL | 当前定义不能作为正式 evaluator |
| robosuite 多任务泛化 | FAIL | 当前硬编码 `Lift/Panda/cube` |
| 发布级可复现性 | FAIL | 自检入口损坏且没有自动化测试 |

## 4. 振动甲板驱动方式

### 4.1 当前方式为什么合理

当前实现不是让 mocap 几何直接参与接触，而是：

1. 创建无自由度的 `deck_drv` mocap body；
2. 创建带 freejoint 和 400 kg ballast 的普通动态 `deck`；
3. 用 weld equality constraint 把二者连接；
4. 把机器人基座和工作台支撑链挂在动态 `deck` 下。

代码见 `spike/env_shakedeck.py:95-121,151-161`。

MuJoCo 3.3.7 官方建模文档指出，mocap body 本身在物理上被视为静止，直接用移动 mocap body 接触会缺失其相对速度；官方给出的更稳健方案正是增加普通动态 body，并通过 weld equality constraint 跟随 mocap body。[MuJoCo 3.3.7：MoCap bodies](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#mocap-bodies)

因此，在“输入是规定的甲板位移时程，而不是研究 shaker 执行器自身”的前提下，当前方案优于：

- 每步直接覆盖动态 body 的 `qpos`；
- 让 mocap body 自己承担接触；
- 用未经辨识的巨大外力强迫甲板跟踪。

### 4.2 当前实现仍有四个风险

#### A. 调度依赖 robosuite 的 split-step 内部顺序

robosuite 1.5.2 的 `lite_physics` 顺序是：

```text
step1 → _pre_action → step2 → _update_observables
```

官方源码见 [robosuite v1.5.2 `base.py`](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.2/robosuite/environments/base.py)。

`spike/env_shakedeck.py:316-333` 在 `_pre_action` 中写 mocap 目标，并人为提前一个 physics step。该补偿在当前固定的 `lite_physics=True`、`dt=0.2 ms` 下有实验依据，但它是对内部时序的适配，不是稳定的公共抽象。

正式实现应显式拥有 physics-step loop，保证 mocap target 在对应 `mj_step1` 前已经写入；若继续沿用一步超前法，必须把 `lite_physics`、时间步和调用序列写入不可变协议并做版本门。

#### B. weld 不是无限刚的运动学约束

MuJoCo equality weld 是软约束；官方文档明确指出，当 weld tracking 与其他接触约束竞争时，结果取决于 weld 与 contact 的相对软硬度。[MuJoCo 3.3.7：MoCap bodies](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#mocap-bodies)

当前 `WELD_SOLREF=(0.0004, 1.0)` 恰好是 `2 × dt`。MuJoCo 建议正值 `solref` 的 time constant 至少为两倍时间步，以避免积分不稳定；当前配置位于允许边界，而非拥有充足裕量。[MuJoCo 3.3.7：Solver parameters](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#solver-parameters)

因此必须报告：

- 六轴 command-vs-actual 位姿、速度和加速度误差；
- 空载、工件接触、机械臂运动三种载荷下的跟踪误差；
- weld constraint force / torque；
- 时间步和 `solref` 敏感性。

#### C. 400 kg ballast 是数值设计参数，不是已校准硬件参数

`spike/env_shakedeck.py:28,98-110` 给动态 deck 添加 400 kg 隐形 ballast。它有利于形成稳定的动态代理，但当前没有硬件质量、惯量或敏感性论证。

如果论文只把 deck 当作理想位移输入端，ballast 应被标为数值实现参数，并证明结果对合理范围不敏感；如果声称模拟具体振动台，它就必须来自硬件辨识。

#### D. 当前 tracking gate 主要覆盖平移

现有 `run_exp1.py` 的 weld tracking 只比较 deck site 的平移位置，未比较旋转、角速度和角加速度。六轴 benchmark 不能用三轴 tracking gate 代替六轴 gate。

## 5. 六自由度隔振器

### 5.1 基础方程正确

`spike/isolator.py:21-101` 使用：

```text
k = M ω_n²
c = 2 ζ M ω_n
```

并对转动自由度以惯量 `I` 代替质量 `M`。`spike/isolator.py:119-161` 在工作台 body 上插入三个 slide 和三个 hinge joint，分别设置 stiffness、damping 和竖直 gravity-compensating `springref`。

本地编译检查显示：

- table mass：`32.00000000000001 kg`，设计值 `32 kg`；
- table inertia：`[1.71333, 1.71333, 3.41333] kg·m²`，与设计值基本一致；
- 5 Hz、ζ=0.1 时，编译后的平移刚度 `31582.734 N/m`、阻尼 `201.062 N·s/m`，与解析式一致。

因此基础方程与 MuJoCo 参数落地判定为 **PASS**。

### 5.2 本地复跑证据

运行：

```text
spike/.venv/bin/python spike/run_iso_stage1.py \
  --self-check --output /tmp/shakebench_spike_audit/iso
```

结果：

- `all_passed=true`；
- 5 Hz 单轴平移激励下，7 组 `(f_n, ζ)` 的解析/仿真 transmissibility 最大相对误差约 `0.441%`；
- 两次相同运行的最大 `qpos` 差为 `0`；
- 静态工作台高度相对刚性配置的最大误差约 `2.97 μm`；
- MuJoCo warning 总数为 `0`。

这是真正支持 `spike` 的强证据：至少在单轴线性小振幅条件下，MuJoCo 模型实现了作者声称的二阶基座激励响应。

### 5.3 目前不能推出什么

当前自检只用 5 Hz 的 x 方向谐波。它没有验证：

- y/z 平移；
- rx/ry/rz 转动；
- 跨越共振点的完整频响曲线；
- 六轴同时激励时的耦合；
- 大位移、行程限位、碰撞止挡；
- 工作台质心与支撑中心偏置；
- payload 质量、质心和接触载荷变化；
- 非线性隔振器或真实 Stewart / air-spring 几何。

所以当前模型可以称为 **canonical linear 6-DoF lumped isolator**，不能称为某个真实隔振系统的数字孪生。

## 6. 激励与难度定义

### 6.1 可复用内容

`spike/vibration.py:30-67` 以解析式输出六轴 `q/qd/qdd`，保持了原仓库每条谱线从加速度到速度、位移的解析积分。`seed/t0`、频带和 Γ 标定思想值得复用。

### 6.2 当前工程接口不应继承

`spike/vibration.py:18-27` 通过伪造 `torch` module，并调用主仓库私有函数 `_synthesize_axis_lines()`，才能绕过 Isaac/Torch 依赖。这适合一次性 spike，不适合独立发布。

新仓库应把频谱合成、校准和安全门重写为正式的纯 NumPy 公共 API，并用固定 golden cases 对旧实现做数值等价验证。

### 6.3 Γ 只标定输入端，不代表隔振后桌面实际难度

当前 `calibrated_vibration()` 标定的是 deck command 对名义工件点的垂向峰值加速度；经过隔振器后，工作台可能在共振区放大，也可能在高频区衰减。

因此正式 episode 至少要同时记录：

```text
Gamma_commanded     # excitation program 定义值
Gamma_deck_actual   # weld 后实际甲板响应
Gamma_table_actual  # 隔振后工作台目标点响应
```

摩擦/滑移分析还需要实际桌面切向加速度、法向加速度和姿态，而不能只使用垂向 `Γ`。

原仓库的 Γ 使用小角度 `α × r`，未在标定式中加入有限转动下的 `ω × (ω × r)`。现有角度很小时该项可能很小，但“exact Γ”声明必须以实际 MuJoCo body kinematics 复核，而不能只引用命令解析式。

## 7. 接触与摩擦

### 7.1 显式 contact pair 是正确设计

`spike/env_shakedeck.py:172-191` 只对 `cube_g0 ↔ table_collision` 设置接触参数，避免把 cube-table 摩擦和 `solref` 传播到 Panda 指垫。该做法符合 MuJoCo 对显式 contact pair 的建模接口，应保留。

编译检查得到：

- `condim=3`，即法向 + 两个切向摩擦方向；
- `margin=gap=1 mm`；
- `solref=(0.6 ms, 1.0)`；
- `μx=μy=1.5`（或实验指定值）；
- 静态接触实际 penetration 约 `0.098 μm`。

MuJoCo 3.3.7 中，正 `gap` 会创建不施力的 inactive contact 区间；当前 `margin==gap` 使实际力从几何表面附近开始，而 1 mm 主要用于接触预检测，不应被描述成 1 mm 的物理接触厚度。[MuJoCo 3.3.7 XML reference：`contact/pair`](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#contact-pair)

### 7.2 现有摩擦 sweep 只说明“产生了合理趋势”

本地已有 `STRONG, Γ=0.5, seed=17` 结果：

| cube-table μ | 最大 table-frame 位移 |
|---:|---:|
| 0.2 | 3.172 mm |
| 0.3 | 1.404 mm |
| 0.5 | 0.00028 mm |
| 0.8 | 0.00028 mm |
| 1.5 | 0.00028 mm |

趋势符合“低摩擦更易滑移”的预期，但它不是摩擦模型校准：

- 只有单 seed；
- `obj_slip_on_table` 使用三维位移范数，混入竖直弹跳/沉降；
- 没有记录首次滑移时刻和相对切向速度；
- 没有与单轴理论阈值或斜面试验对照；
- 没有时间步、contact softness 和接触点数量敏感性；
- MuJoCo 的 `μ` 是接触 pair 参数，不天然等于某种真实材料的测量值。

如果 Pick&Place 使用“不同摩擦工件”，必须分开两个界面：

```text
mu_table_object     # acquisition 前与 release 后的自由体滑移
mu_finger_object    # grasp / transport 阶段的在手滑移
```

只修改前者不能支撑“工件整体更难抓持”的结论；同时修改二者又会破坏归因。二者应成为独立实验因子。

## 8. 指标与任务语义

当前 `spike/metrics.py` 不能直接成为正式 evaluator，原因包括：

1. `max_penetration_m` 对所有场景 contact 取最大值，而不是按 table-object、finger-object、finger-table 分类；
2. table slip 使用三维位移范数，不是桌面切平面内的累计/净滑移；
3. success 判定没有要求 `placement.within_tolerance`；
4. placement target 被硬编码为初始 object base-frame 位置加 `(0, 0.15, 0)`；
5. 在隔振工作台任务中，正式 target 应明确属于 table frame、deck frame 还是 robot-base frame；当前实现未把这一语义做成任务契约；
6. `70 mm` placement tolerance 对精密放置过宽，且没有按物体尺度归一化；
7. contact loss 阈值 `0.05 N` 和 slip failure `10 mm` 是经验值，尚无标定或敏感性报告。

建议正式指标至少包括：

- table tangent-plane 净滑移与累计滑移；
- 首次滑移时间、滑移持续时间、峰值相对速度；
- table-object 接触丢失比例和最长连续丢失；
- in-hand 相对平移、相对旋转和双指接触状态；
- target-frame 最终位置/姿态误差；
- task success；
- 按接口分类的 penetration、normal/tangential impulse；
- command/deck/table 三层实际响应。

## 9. robosuite 工程接入

### 9.1 可保留的机制

- robosuite 官方支持在编译前使用 XML processor；`spike/env_shakedeck.py:279-290` 使用的是框架提供的能力，而非 monkey patch；
- robosuite 1.5.2 的 `_pre_action` 和 `_update_observables` 确实在每个 physics substep 调用，因此当前驱动和高频 instrumentation 不是只在 20 Hz policy step 更新；
- `hard_reset` 后重新编译模型，使 XML 变换生效，机制成立。

### 9.2 必须重写的部分

当前代码硬编码：

- robosuite `Lift`；
- `table`、`robot0_base`、`cube_g0`、`table_collision`；
- Panda 8-D 双指动作；
- 全局 `robosuite.macros.SIMULATION_TIMESTEP`；
- 单 robot、单 object、单 table body。

这证明“Lift spike 能运行”，不证明“多任务 benchmark adapter 已存在”。独立新仓库应建立自有 arena/support builder，把以下名称通过显式 handle/schema 传入，而不是在最终 XML 中猜名字：

- deck body；
- robot mount roots；
- isolated payload root；
- task surface geoms；
- task object geoms；
- measurement sites。

## 10. 可复现性实测

### 10.1 通过项

- 本地环境仍可导入 `robosuite 1.5.2`、`mujoco 3.3.7`、`numpy 1.26.4`；
- 隔振自检可从当前源码重跑并通过；
- 相同条件的 MuJoCo `qpos` 逐值一致；
- 隔振自检没有产生 MuJoCo warning。

### 10.2 阻塞项

当前通用自检命令：

```text
spike/.venv/bin/python spike/run_exp1.py --self-check
```

实际失败：`run_selfcheck()` 在构造 `checks` 后没有返回值，随后 CLI 访问 `result["all_passed"]` 触发 `TypeError: 'NoneType' object is not subscriptable`。对应源码见 `spike/run_exp1.py:144-202,243-277,520-526`。

此外：

- `spike/` 没有 pytest 覆盖；
- `requirements.txt` 只固定 robosuite 和 MuJoCo，没有固定 NumPy、SciPy、Python 或 lockfile；
- 大量历史 rollout 被忽略，只有少数汇总 JSON 被 Git 跟踪；
- 当前脚本把 experiment orchestration、cache resume、统计和环境实现混在一起。

因此当前 `spike/` 的发布级可复现性判定为 **FAIL**。

## 11. 与候选方案的正式比较

### 11.1 甲板驱动

| 方案 | 接触速度物理 | 规定轨迹复现 | 参数负担 | 判定 |
|---|---:|---:|---:|---|
| 每步覆盖动态 body `qpos` | 差 | 高 | 低 | 拒绝作为正式接触 benchmark |
| mocap body 直接接触 | 差，官方指出缺少相对速度 | 高 | 低 | 拒绝 |
| **mocap + weld + dynamic deck（当前）** | 较好 | 高 | 中 | **首版推荐** |
| 力/执行器驱动真实 shaker | 最好 | 输入不再是精确位移轨迹 | 高 | 硬件建模阶段再做 |

结论：如果首版把 shaker 视为理想位移源，当前 mocap+weld 原理是合理首选。

### 11.2 隔振器

| 方案 | payload/contact 反作用 | 可解释性 | 硬件真实性 | 判定 |
|---|---:|---:|---:|---|
| 预先用传递函数过滤 table pose | 无 | 高 | 取决于拟合 | 只能做 replay/control |
| **六自由度 lumped spring-damper（当前）** | 有 | 高 | 中低 | **canonical benchmark 推荐** |
| 显式 Stewart/air-spring/linkage | 有 | 中 | 高 | 校准扩展 |
| 有限元/柔性台面 | 有 | 低 | 更高 | 不属于 v1 必需范围 |

结论：当前 lumped isolator 是研究操作任务的合理第一层物理抽象，但论文应准确命名其抽象层级。

### 11.3 robosuite 接入

| 方案 | 原型速度 | 多任务稳定性 | 判定 |
|---|---:|---:|---|
| 事后 XML processor + 硬编码 body 名（当前） | 高 | 低 | 保留作 conformance oracle |
| 参数化 XML processor + 显式 handles | 中 | 中高 | 可接受 |
| 自定义 Arena / task composition + support builder | 中低 | 高 | 正式仓库推荐 |

## 12. 从 spike 升级为正式 backend 的硬门

只有以下门全部通过，才能称为 benchmark backend：

### G0 — 独立核心

- 纯 NumPy excitation/calibration；
- 不伪造 Torch module，不调用旧仓库私有函数；
- fixed seed golden trajectory 全等/容差测试。

### G1 — 拓扑与模型编译

- robot base、deck、isolator、table 的 parent graph 自动断言；
- mass/inertia、joint axis、stiffness、damping、springref 自动断言；
- 多任务 body/geom handle 不依赖字符串猜测。

### G2 — 六轴驱动

- 六个单轴的位姿/速度/加速度 command-vs-actual；
- 六轴联合激励；
- 空载、静态物体、机械臂动作三种载荷；
- weld force/torque、时间步和 solref 收敛性。

### G3 — 隔振器

- 六个自由度的频率 sweep；
- 共振区、隔振区、准刚性区解析对照；
- payload 质量/质心变化；
- 行程限位和极端状态拒绝；
- 被动性/能量与数值稳定检查。

### G4 — 接触与摩擦

- 静态承载、斜面阈值、单轴滑移阈值；
- drop/impact 与接触恢复；
- timestep、solver、solref/solimp 敏感性；
- `mu_table_object` 与 `mu_finger_object` 独立。

### G5 — 任务与指标

- target frame 明确定义；
- success 必须包含真正的 placement 条件；
- tangent slip、in-hand slip、contact loss 和 placement 分解；
- Γ command/deck/table 三层记录；
- matched initial states 与 paired evaluation。

### G6 — 发布可复现性

- 修复自检；
- pytest 覆盖；
- lockfile / container；
- 固定配置、seed 和 evaluator；
- 全新 clone 一条命令重现物理 gate。

## 13. 对新独立仓库的复用边界

### 应复用

- mocap + weld + dynamic deck 这一物理模式；
- 机器人刚接 deck、工作台六自由度隔振的 support graph；
- `k=Mω²`、`c=2ζMω` 的 canonical isolator 参数化；
- 显式 task contact pair 的作用域纪律；
- 六轴解析谱、seed/t0、Γ 标定与安全门；
- command/actual 分离、paired episodes 和连续诊断指标的思想。

### 应重写

- `ShakeDeckLift` 继承和 XML 字符串硬编码；
- `_pre_action` 一步超前调度；
- vibration 的 Torch stub / 私有函数访问；
- 当前 metrics 和 success；
- run scripts、cache 和 experiment orchestration；
- 配置与依赖发布方式。

### 不应作为证据继承

- 当前 scripted policy 的成功率；
- 当前排行榜；
- 仅靠单 seed 摩擦 sweep 得出的“滑移规律”；
- 仅靠无 warning 得出的物理真实性结论。

## 14. 与 RoboMME 先例的关系

RoboMME 确实证明同一工作可以同时发布 benchmark 与针对性方法；但它先把中心问题限定为“当前观测不足、历史信息必需”，再以 temporal/spatial/object/procedural taxonomy 建立任务，最后用统一骨干和预算的 memory 方法矩阵验证 benchmark。[RoboMME 论文](https://arxiv.org/abs/2603.04639)、[官方 benchmark 仓库](https://github.com/RoboMME/robomme_benchmark)、[官方 policy-learning 仓库](https://github.com/RoboMME/robomme_policy_learning)

对 ShakeBench 的直接启示是：

- benchmark 与未来方法可以出现在同一篇最终论文；
- benchmark 工程和 evaluator 必须先独立成立；
- 方法不能参与环境定义、test seed 或 scorer；
- 摩擦不同的工件可以是 Pick&Place 的一个 factor，但不能替代隔振传递和接触物理验证；
- 二元成功率之外必须有连续物理指标。

更完整的 RoboMME 对照见 `docs/research/robomm_benchmark_positioning.md`。

## 15. 最终建议

在已经选择“新建独立 robosuite 仓库”的前提下：

1. 不要从零重新发明 deck/isolator 物理模式；
2. 不要把 `spike/` 目录直接复制为 `src/`；
3. 把 `spike` 当作 conformance oracle：新实现必须复现其已通过的单轴 transmissibility、确定性和静态高度结果；
4. 新实现随后必须通过 G0–G6 中 `spike` 尚未覆盖的门；
5. 在这些门通过以前，不开始构建大任务矩阵，也不把不同摩擦工件的趋势写成物理规律结论。

一句话决策：**接受 spike 的物理路线，拒绝 spike 的现成产品化资格。**
