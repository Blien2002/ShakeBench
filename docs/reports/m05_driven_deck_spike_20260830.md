# M0.5 驱动甲板提示词审查与隔离验证报告

日期：2026-08-30  
仓库提交：`d1db196e061e8795cbdde8064ddb9cf71b1d492c`  
运行时：MuJoCo `3.9.0`，robosuite `1.5.2`

## 1. 结论摘要

Claude 提示词提出的核心物理问题**值得验证且确实存在**：直接逐步改写 mocap 位姿不会向其后代提供甲板的速度与加速度历史，因此不适合表示需要基座惯性激励的机械臂。

但提示词本身也有多处事实错误或无效判据，不能按原文直接得到一个可靠的 go/no-go：

1. 断言 D 错误。本版本的 `MujocoXML.merge()` 已有 `merge_body=<body name>` 参数，机器人可以直接并入 `deck`，无需事后 XML 手术。
2. Stage B 仅以位置 RMS 判断“轨迹一致”不充分。原文的无阻尼 position actuator 可以位置 PASS、速度和加速度却严重失真。
3. “加大 platform 质量”不会改善固定 `kp` 下的跟踪，实测方向相反。
4. `data.cacc` 不是可直接与解析甲板加速度逐项比较的普通运动学数组；其值采用 RNE/质心空间约定，并包含重力项。
5. C2 中“持续发送零 delta 动作”等价于“恒定基座系目标”的说法不成立。默认 OSC 的 `goal_update_mode="achieved"` 会把零 delta 重设到当前已实现位姿。
6. C2 的二选一峰峰值表把“参考系是否缓存”和“有限带宽控制误差”混在了一起。
7. C3 要求 robosuite 内部 `J_full.shape[1] == model.nv` 不符合现有控制器设计。控制器有意按名称派生的 `qvel_index` 取机械臂列；这正是增加前置 deck DOF 后仍然安全的原因。

最终判定分成两层：

- **robosuite 重挂能力：GO。** C1、索引完整性和跨进程确定性均通过，控制器基座原点也确实逐控制步更新。
- **原提示词给出的“无阻尼位置执行器 + 只看位置 RMS”驱动配方：NO-GO。** 36 个预定扫描点没有一个达到峰值或 RMS `<10 µm`，且高增益产生未被位置 gate 捕获的速度/加速度振荡。
- **DeckBench 总体路线：CONDITIONAL GO。** 保留 robosuite，但必须先修正驱动器、轨迹 gate、C2 实验和观测换算；现有证据不支持因为重挂问题转向 dm_control。

## 2. 验证隔离与证据边界

所有临时脚本和结果均位于 `/tmp/shakebench_m05_spike`，未向仓库加入验证代码；仓库中只增加本报告。

本机安装的 MuJoCo wheel 只含公开头文件、Python 绑定和 `libmujoco.so.3.9.0`，不含引擎 C 实现文件。因此“读取已安装包内 `mj_comVel` / `mj_kinematics` / `mj_rnePostConstraint` 的 C 源码”这一要求无法按字面执行。本报告使用：

- 本机 3.9.0 ABI、头文件与实际运行结果；
- MuJoCo 3.9.0 官方文档及同版本官方源码标签；
- 当前仓库内 robosuite 1.5.2 源码。

官方依据：

- [MuJoCo 3.9.0 Modeling — Kinematic tree / MoCap bodies](https://mujoco.readthedocs.io/en/3.9.0/modeling.html)
- [MuJoCo 3.9.0 XML Reference — body / joint / mocap](https://mujoco.readthedocs.io/en/3.9.0/XMLreference.html)
- [MuJoCo 3.9.0 Computation — pipeline and RNE](https://mujoco.readthedocs.io/en/3.9.0/computation/index.html)
- [MuJoCo 3.9.0 engine_core_smooth.c](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_core_smooth.c)
- [MuJoCo 3.9.0 engine_core_constraint.c](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_core_constraint.c)

## 3. Stage A：四条断言判定

| 断言 | 判定 | 说明 |
|---|---|---|
| A：mocap 不向子体传递自身运动速度，缺失基座惯性激励 | **成立，但表述需收窄** | mocap 在动力学中被视为固定；改写 `mocap_pos/quat` 只改变正向运动学位姿。实验中 platform `cvel_x` 严格为 0，铰接单摆不响应。重力没有“全部缺失”，`cacc_z` 仍按 RNE 约定为 `+9.81`。|
| B：无关节子体与父体树焊接，动力学刚性传播 | **成立** | 官方 Kinematic tree 明确规定无关节 body welded to parent。对动态父体，这种树连接没有 equality weld 的柔度。|
| C：正确架构必然是驱动关节 + 执行器 | **部分成立** | 动态关节能提供正确的广义速度/加速度通道，也避免 weld 柔度；但“position actuator 有限增益误差可单靠加质量/加 kp 压小”不成立。执行器阻尼、前馈、采样和积分器必须共同设计。|
| D：`world.merge(mujoco_robot)` 硬编码到 worldbody，必须 XML 手术 | **驳回** | 默认调用确实并到 worldbody，但 API 明确支持 `merge_body`。`MujocoXML.merge` 在 `robosuite/models/base.py:83-112` 查找指定 body 并把对方 worldbody 子体直接 append。|

### 提出者具体错在哪里

断言 D 把“调用点使用默认参数”误当成“底层 API 硬编码”。当前 `Task.merge_robot()` 调用默认 merge 是一个可覆写的装配选择，不是框架能力限制。

断言 C 还包含方向错误：对近似单自由度平台，闭环固有频率约为 `sqrt(kp / M)`。固定 `kp` 时增加质量会降低带宽并增大正弦跟踪误差。Stage B 扫描也验证了这一点。

### 第五条路

`mjSpec` / `MjSpec.attach` 可以更方便地在 spec 层挂接 body，但只改变模型装配，不会赋予 mocap 速度，因此不是 A 的物理解法。可考虑的真正替代控制方式是：动态甲板关节 + 解析速度/加速度前馈（由 inverse dynamics 或通用 actuator 施力）+ 低增益反馈，而不是纯 P position actuator。它仍属于动态关节架构，但避免为压缩位置误差而使用极端刚度。

## 4. Stage B：MuJoCo 最小实验

### 4.1 可一键复现的反馈回路

隔离命令：

```bash
python /tmp/shakebench_m05_spike/mocap_vs_driven.py --kp 1e9 --timestep 2e-5
```

同一命令独立运行 3 次，JSON 输出逐位相同。

参数：`A=0.01 m`、`f=5 Hz`、platform `100 kg`、position actuator `kp=1e9`、无执行器阻尼、仿真 `2 s`。

| 量 | mocap | driven |
|---|---:|---:|
| 位置跟踪 RMS | `0` | `70.263 µm` |
| 位置跟踪峰值 | `0` | `100.360 µm` |
| platform `cvel_x` 峰值 | `0` | `0.627933 m/s` |
| drive `qacc` 峰值 | 不存在该 DOF | `1000.817 m/s²` |
| 单摆角峰峰值 | `0` | `0.114502 rad` |
| `cacc_z` 均值 | `9.81` | `9.81` |

目标速度峰值只有 `2πfA = 0.314159 m/s`，目标加速度峰值只有 `(2πf)^2 A = 9.8696 m/s²`。因此 driven 的位置 RMS 虽已通过原提示词 `<1% A = 100 µm` 的 gate，速度峰值约为目标的 2 倍、加速度峰值约为目标的 101 倍。此结果直接证伪“位置 RMS 通过后后续动力学对比就有意义”。

单摆比值因 mocap 分母严格为 0，在浮点实现中趋于无穷；`>100` 只能作为定性 gate，不能作为稳健的连续指标。应同时给 mocap 噪声绝对上限，例如 `<1e-10 rad`。

### 4.2 mocap 下有自由度子体

模型成功编译：mocap 模型 `nq=1, nv=1`，该唯一 DOF 就是后代单摆铰链。因此官方“dof-less descendants”并不表示 mocap 不能有带关节后代；它只描述哪些后代属于 mocap 自己的 weld group。

### 4.3 原配方扫描结果

扫描严格使用无阻尼 position actuator；`kp ∈ {1e8,1e9,1e10}`，仿真步长 `1e-5 s`。表中的速度、加速度误差揭示了位置 gate 看不到的高频自由振荡。

| f Hz | mass kg | kp | peak µm | RMS µm | cond(M) | v RMS err m/s | a RMS err m/s² |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 100 | 1e8 | 324.6 | 222.7 | 3.27e3 | 0.222 | 222 |
| 5 | 100 | 1e9 | 100.4 | 70.3 | 3.27e3 | 0.222 | 701 |
| 5 | 100 | 1e10 | 31.6 | 22.3 | 3.27e3 | 0.222 | 2220 |
| 5 | 1000 | 1e8 | 1103.1 | 714.7 | 3.22e4 | 0.224 | 71.1 |
| 5 | 1000 | 1e9 | 324.4 | 222.5 | 3.22e4 | 0.222 | 222 |
| 5 | 1000 | 1e10 | 100.4 | 70.3 | 3.22e4 | 0.222 | 702 |
| 5 | 5000 | 1e8 | 2848.7 | 1692.5 | 1.61e5 | 0.234 | 33.0 |
| 5 | 5000 | 1e9 | 755.5 | 499.9 | 1.61e5 | 0.223 | 99.7 |
| 5 | 5000 | 1e10 | 227.2 | 157.2 | 1.61e5 | 0.222 | 314 |
| 20 | 100 | 1e8 | 1439.2 | 909.4 | 3.27e3 | 0.905 | 900 |
| 20 | 100 | 1e9 | 414.4 | 281.9 | 3.27e3 | 0.890 | 2810 |
| 20 | 100 | 1e10 | 127.6 | 89.1 | 3.27e3 | 0.890 | 8880 |
| 20 | 1000 | 1e8 | 6578.1 | 3600.7 | 3.22e4 | 1.08 | 334 |
| 20 | 1000 | 1e9 | 1437.2 | 909.4 | 3.22e4 | 0.904 | 902 |
| 20 | 1000 | 1e10 | 413.9 | 281.7 | 3.22e4 | 0.890 | 2810 |
| 20 | 5000 | 1e8 | 79462.2 | 51554.5 | 1.61e5 | 7.14 | 931 |
| 20 | 5000 | 1e9 | 3896.4 | 2261.2 | 1.61e5 | 0.966 | 434 |
| 20 | 5000 | 1e10 | 975.2 | 636.0 | 1.61e5 | 0.895 | 1270 |
| 50 | 100 | 1e8 | 4587.9 | 2578.5 | 3.27e3 | 2.48 | 2460 |
| 50 | 100 | 1e9 | 1104.7 | 715.5 | 3.27e3 | 2.24 | 7100 |
| 50 | 100 | 1e10 | 325.2 | 223.0 | 3.27e3 | 2.23 | 22200 |
| 50 | 1000 | 1e8 | 187691.6 | 102124.2 | 3.22e4 | 32.0 | 10200 |
| 50 | 1000 | 1e9 | 4578.1 | 2572.7 | 3.22e4 | 2.47 | 2460 |
| 50 | 1000 | 1e10 | 1103.3 | 714.8 | 3.22e4 | 2.24 | 7110 |
| 50 | 5000 | 1e8 | 16950.4 | 8749.1 | 1.61e5 | 2.79 | 860 |
| 50 | 5000 | 1e9 | 23317.0 | 12639.1 | 1.61e5 | 5.26 | 2140 |
| 50 | 5000 | 1e10 | 2849.1 | 1692.1 | 1.61e5 | 2.34 | 3300 |
| 100 | 100 | 1e8 | 16871.1 | 8767.5 | 3.27e3 | 8.14 | 7550 |
| 100 | 100 | 1e9 | 2472.0 | 1497.4 | 3.27e3 | 4.62 | 14600 |
| 100 | 100 | 1e10 | 672.1 | 448.6 | 3.27e3 | 4.46 | 44600 |
| 100 | 1000 | 1e8 | 19217.4 | 10844.4 | 3.22e4 | 6.19 | 3800 |
| 100 | 1000 | 1e9 | 16874.0 | 8674.5 | 3.22e4 | 8.09 | 7500 |
| 100 | 1000 | 1e10 | 2475.5 | 1494.2 | 3.22e4 | 4.62 | 14600 |
| 100 | 5000 | 1e8 | 12763.1 | 7421.3 | 1.61e5 | 4.68 | 2940 |
| 100 | 5000 | 1e9 | 32068.8 | 15898.4 | 1.61e5 | 9.64 | 5650 |
| 100 | 5000 | 1e10 | 7969.2 | 4319.3 | 1.61e5 | 5.71 | 7870 |

36 个点中，位置峰值 `<10 µm`：`0/36`；位置 RMS `<10 µm`：`0/36`。质量矩阵条件数随质量从约 `3.27e3` 增到 `1.61e5`，与 `kp` 无关。

附加对照中加入提示词未列出的 `dampratio=1` 后，`f=5 Hz, M=100 kg, kp=1e12, dt=1e-6 s` 可把位置峰值压到 `6.315 µm`、速度峰值压到正确的 `0.314159 m/s`。这证明问题不是“动态关节路线必然不可行”，而是原配方和 gate 不完整；该工作点的极端刚度和微秒步长也不适合作为未经进一步评估的推荐点。

## 5. Stage C：robosuite spike

### C1 重挂和物理字段

直接调用 `world.merge(robot, merge_body="deck")`。机器人全部列举字段逐位相同：

`geom_friction/solref/solimp/condim/margin/priority`、`body_mass/inertia/ipos`、`dof_damping/armature/frictionloss`、`jnt_range/stiffness`、`actuator_gainprm/biasprm/ctrlrange/forcerange`。

site、sensor 名称集合不变。robosuite 在加载 XML 时已经把 default class 内联，重挂不会引入新的 `childclass` 继承差异。

判定：**PASS**。

### C2 基座参考系

源码显示 `CompositeController.update_state()` 每次从 `{prefix}{arm}_center` site 读取世界系 base pose，并调用 arm controller 的 `update_origin()`。OSC 在 `input_ref_frame="base"` 时，每次运行用当前 origin 把保存的基座系目标转回世界系，并显式减去基座线/角速度。

隔离动态实验也确认 controller origin 的峰峰值与实际 deck 位移峰峰值一致到 `1e-15 m` 量级，不存在缓存。

但原 C2 数值表不能作为硬 gate：在有限 `kp=150` 的默认 OSC 和 5 Hz 惯性激励下，基座系末端仍会有真实跟踪误差；而每个策略步反复发送零 delta 会在默认 achieved 模式下移动目标。正确实验必须：

1. 只设置一次固定基座系绝对目标，或把 goal update mode 固定为 `desired`；
2. 单独记录 origin 更新误差；
3. 用“更新 origin”与“冻结 origin”的差分对照识别缓存；
4. 把控制误差另设带宽相关阈值，不能要求数学上的 `≈0`。

结构判定：**PASS**；原预注册数值判据：**无效，需重写**。

### C3 Jacobian 和质量矩阵

- 固定模型 `nv=7`，重挂模型 `nv=8`；raw Jacobian 分别为 `6×7`、`6×8`。
- 机械臂 DOF 从 `[0..6]` 移到 `[1..7]`。
- 按名称取机械臂列后，Jacobian 最大差异 `0`。
- 机械臂质量子块最大差异 `0`。
- Robot 和 Controller 均通过 joint name 派生 qpos/qvel 地址；controller 从 raw Jacobian 和 full mass matrix 中按 `qvel_index` 取机械臂子块。

因此没有发现会被前置 deck DOF 破坏的常量索引。提示词要求控制器内部的 `J_full` 保持全 `nv` 列反而会破坏其现有 arm-only OSC 维度约定。

判定：**PASS（需修正提示词的 shape 断言）**。

### C4 观测

robosuite 1.5.2 默认 `robot0_eef_pos` 直接返回 `site_xpos`，`robot0_eef_quat` / `robot0_eef_quat_site` 也明确是世界系。DeckBench 必须在 State track 的观测组装点用当前机器人 base pose 做 `T_base_world @ T_world_eef` 换算。

另有既有兼容性问题：`eef_pos` 来自 site，而旧兼容键 `eef_quat` 来自 body；新代码应优先使用 `eef_quat_site`。

判定：**需要 DeckBench 自行换算，不是框架阻塞项**。

### C5 确定性

同一 seed 的完整 robosuite 隔离 episode 在 3 个独立 Python 进程中输出逐位相同；MuJoCo 最小实验也在 3 个独立进程中逐位相同。

判定：**PASS（仅覆盖本 spike；未来完整任务仍需保留跨进程 gate）**。

## 6. 建议的修订 gate

1. 轨迹一致性必须同时检查 `qpos`、`qvel`、`qacc`；建议相对目标幅值分别设阈值，并检查驱动频率以外的频谱能量。
2. position actuator 至少显式指定 damping/velocity feedback；优先比较“解析前馈 + 有限反馈”而不是继续无限增大 `kp`。
3. 不使用增大质量作为跟踪调优手段；质量应来自物理模型，控制带宽单独设计。
4. `cacc` 只作为理解 RNE 空间量的诊断值；甲板解析跟踪使用关节 `qvel/qacc` 或 frame velocity/acceleration sensor，并注明坐标系。
5. C2 使用一次性绝对基座系目标，并添加 origin 更新的直接断言；不要用零 delta 连续刷新代替固定目标。
6. C3 对 raw MuJoCo Jacobian 检查 `nv`，对 robosuite OSC 则检查其列是否严格等于名称派生的 arm DOF 集合。
7. C4 明确使用 `eef_quat_site`，在 DeckBench 观测层统一转换为基座系。

## 7. 最终决策

核心问题成立：**mocap 直接驱动不能作为 DeckBench 的惯性基座。**

但“robosuite 必须靠 XML 手术才能重挂”不成立，实验证明 robosuite 可保持参数、Jacobian、质量矩阵和名称完整性。当前真正的阻塞是驱动器与验证 gate，而不是 robosuite 模型装配。

因此建议：**保留 robosuite，暂停采纳原 Stage B 配方；先完成带速度/加速度 gate 的动态关节驱动器重新设计，再进入 DeckBench 业务实现。**
