# robosuite 振动操作 Benchmark：设计树与未决问题

> 状态日期：2026-08-29  
> 文档用途：记录已经确认的设计合同、明确拒绝的路线、尚未决定的问题及其依赖关系。  
> 当前阶段：只设计和构建 benchmark；训练策略与作者方法暂不进入实现范围。

## 0. 状态标记

- **DECIDED**：用户已经明确确认，除非出现新证据，不应静默修改。
- **PROPOSED**：已有推荐方案，但用户尚未确认。
- **OPEN**：仍需讨论或依赖前置实验。
- **REJECTED**：已经明确不采用。

## 1. 总体目标

### 1.1 研究产物 — DECIDED

- 最终目标是可投稿、可复现实验的研究 benchmark。
- 最终论文可以像 RoboMME 一样同时包含 benchmark、分析和针对性方法。
- 当前工作范围只包含 benchmark；策略、后训练和 SOTA 方法以后再进入。
- 新实现放入独立 robosuite 仓库；当前 ShakeBench 仓库作为设计、协议和验证资产来源。
- 当前只实现和评测 `State Oracle-Control Track`；Vision Track 延后为一节消融实验。

### 1.2 研究覆盖 — DECIDED

- 长期目标是覆盖振动操作中的主要问题，而不是只做一个演示场景。
- 近期只实现一个 PickPlace 任务以快速验证完整链路。
- 首个任务通过不代表 benchmark 完成；后续任务范围另行讨论。

### 1.3 完成门 — DECIDED

冻结 v1 的最低条件：

1. 确定性通过：同一 state ID 至少在 3 个独立进程中重放，状态、动作和指标 trace 在预注册容差内一致；
2. 无振动静态任务与接触物理通过；
3. 扰动阶梯同时具有非饱和成功区和退化区；
4. matched episodes 上能产生可解释、可重复、带不确定性的差异；
5. 本文第 8 节列出的物理验证全部通过。

## 2. 机械拓扑与隔振模型

### 2.1 支撑拓扑 — DECIDED

```text
prescribed shaker excitation
→ dynamic vibrating deck
   ├── robot base（刚性安装）
   └── canonical linear 6-DoF isolator
       └── open worktable + task target + object contact
```

- 机器人基座与振动甲板刚接。
- 工作台通过隔振器连接到振动甲板。
- 工作台与机器人不完全共模运动。
- 自由工件不能成为工作台的运动学子体；工件运动只能由重力、接触和抓取产生。

### 2.2 隔振器抽象 — DECIDED

- 使用可解释的 canonical 六自由度线性模型。
- 六个相对自由度为 `tx/ty/tz/rx/ry/rz`。
- 每轴使用线性弹簧和粘性阻尼。
- 不声称模拟某个具体品牌或硬件的数字孪生。
- 首版只使用一组固定隔振参数；隔振器的作用是避免完全共模，而不是成为 official suite 的实验轴。

基本关系：

```text
k_i = m_eff_i (2π f_n,i)^2
c_i = 2 ζ_i m_eff_i (2π f_n,i)
```

转动自由度使用有效转动惯量代替质量。

### 2.3 canonical 惯性和参数不变量 — PARTIALLY DECIDED（Q23）

需要冻结以下物理合同：

- 空载 open worktable 的参考质量 `M_ref`；
- worktable 质心/弹性中心处的参考惯量 `I_ref`；
- 六轴 nominal `f_n` 和 `ζ`；
- 由上述参数一次性派生并冻结的 `k` 和 `c`；
- 竖直静态平衡位置与预压缩补偿。

已接受的惯性合同：

```text
tabletop dimensions = 0.65 × 0.60 × 0.06 m
M_ref = 32 kg
I_ref = [0.9696, 1.1363, 2.0867] kg·m²
```

- `I_ref` 按上述尺寸和 32 kg 等效长方体计算；
- isolator elastic center、worktable body-frame origin、worktable COM 与 principal-inertia frame 明确定义为同一点/同一组轴，因此该 `I_ref` 是正确的转动有效惯量；
- 工业隔振支座可以视觉上位于桌面下方，但不改变 canonical elastic center；若未来把弹性中心移到支座平面，必须按平行轴定理重算惯量并显式建模平移—转动耦合；
- 工业框架、横撑、脚板、螺栓和隔振器外观不贡献隐藏质量；
- 所有惯性通过 isolated worktable root 的显式 inertial 定义；
- Can 等 payload 不触发重新调参。

已接受的派生与冻结原则：

1. benchmark 配置公开 `M_ref/I_ref/f_n/ζ`；
2. 构建时一次性派生 `k/c`；
3. 编译后断言实际质量、惯量、刚度和阻尼；
4. 此后更换工件时不重新调 `k/c`；
5. 工件质量变化应自然改变载荷、静态下沉和实际传递特性。

竖直支撑使用 nominal-table preload：

```text
springref_z = M_ref * g / k_z
```

- 只补偿 32 kg worktable 自重，使空载 nominal pose 不因重力下沉；
- 不补偿 0.349 kg Can 或未来 payload，payload 引起的额外下沉和传递变化属于真实任务响应；
- 未补偿参考量 `g/(2π f_n,z)^2` 与补偿后的空载/载荷静态偏移都必须进入 provenance。

明确不推荐：为了让每种工件拥有完全相同响应而逐工件重新调隔振器。这会消除物体质量这一真实物理因素。

Q23 尚未完全关闭：仍需经第 2.4 节的 physics-only transfer-envelope pilot 选定 nominal `f_n/ζ`，随后一次性派生并冻结 `k/c`。

### 2.4 固定隔振参数如何选 — DECIDED（Q24）/ 阈值 OPEN

旧提案是“physics pilot 后再用 scripted task 成功率选一组区分度适中的参数”。用户询问是否有更好的办法。

已接受的方案：**按预注册传递包络选择，不按任务成功率调隔振器。**

流程：

1. 固定 ShakeBench v0 激励谱；
2. 在物理层扫描候选 `(f_n, ζ)`；
3. 对每个候选计算解析和 MuJoCo 实测的频率响应；
4. 计算以下无策略指标：

```text
T_accel       = RMS(table acceleration) / RMS(deck acceleration)
R_relative    = RMS(table acceleration - deck acceleration) / RMS(deck acceleration)
D_relative    = P99 ||table pose - deck pose||
T_peak        = max spectral-line transmissibility
travel_margin = isolator travel cap - realized travel
static_sag_uncompensated = g / (2π f_n,z)^2
static_offset_compensated = realized empty/payload equilibrium offset
```

5. 用预注册规则选择最接近目标传递包络、同时满足安全约束的候选；
6. deterministic tie-break 后冻结参数；
7. task pilot 只用于发现任务是否整体退化，不用于回调隔振器；
8. 若任务全成功或全失败，应扩展/细化预注册 Γ knee-calibration 网格，而不是重新调隔振器。

已接受的目标性质；具体阈值仍未确定：

- `R_relative` 明显大于零，避免完全共模；
- `T_peak` 有限，避免共振主宰整个任务；
- 部分谱线表现为跟随/放大，部分谱线表现为隔振；
- 全部 knee-calibration 候选 Γ 安全范围内满足行程、角度、求解器步长和非弹道约束；
- preload 后的空载桌面保持 nominal pose；payload 引起的额外静态偏移不得破坏可达性、浅箱几何或隔振行程；
- 结果对小幅 timestep / solver 改变稳定。

仍需决定：上述目标区间、评分函数和 tie-break 的具体数值。

## 3. 振动激励

### 3.1 v0 — DECIDED

快速验证原样复用当前 ShakeBench 数值：

- 六轴解析多频谱；
- 确定性 `seed/t0`；
- 五阶 episode-relative ramp；
- 以下完整 acceleration-PSD band table（`frequency_scale=1`）：

| axis | center Hz | relative accel RMS | bandwidth ratio | tones | nominal band Hz |
|---|---:|---:|---:|---:|---:|
| `tx` | 5.0 | 0.50 | 0.10 | 12 | 4.50–5.50 |
| `ty` | 6.5 | 0.35 | 0.10 | 10 | 5.85–7.15 |
| `tz` | 8.0 | 1.00 | 0.10 | 12 | 7.20–8.80 |
| `rx` | 3.0 | `0.30 × tz_RMS / 0.65 m` | 0.12 | 12 | 2.64–3.36 |
| `ry` | 4.0 | `0.30 × tz_RMS / 0.65 m` | 0.12 | 10 | 3.52–4.48 |
| `rz` | 2.5 | `0.30 × tx_RMS / 0.65 m` | 0.12 | 8 | 2.20–2.80 |

- 每条 line 在 nominal band 内等间距后加入确定性 bounded jitter；当前配置的保守最高频率上界为 `f_max < 8.87 Hz`；
- 现有 Γ ladder 仅作为 knee-calibration 的初始候选网格，不作为跨 Γ 主结果；
- 位移与求解器行程安全门。

physics timestep 必须同时满足：

```text
dt <= 1 / (20 * f_max)
positive solref time constant >= 2 * dt
```

这只是必要条件；最终 `dt` 仍由 weld/contact/isolator 的三档收敛实验决定，而不是只按激励采样率决定。

### 3.2 v1 — DECIDED / `Gamma_star` 数值 OPEN

- 保留 ShakeBench 的谱生成算法、确定性和校准协议。
- 根据固定隔振器和 State-V0 knee calibration 冻结单一主计分 `Gamma_star`。
- 不默认 `0.15/0.30/0.50/0.75/0.95` 在新后端仍具有合理难度；这些值只作为初始扫描候选。
- `Gamma_star` 必须在 official states 运行前冻结；完整跨 Γ 曲线只属于后续消融。

### 3.3 必须记录的响应层 — DECIDED

每个 episode 至少记录：

```text
Gamma_commanded
Gamma_deck_actual
Gamma_table_actual
commanded deck q/qdot/qdd
actual deck pose/twist/acceleration
actual table pose/twist/acceleration
table-relative-to-robot support state
```

- `Gamma_commanded` 是实验自变量，也是 `Gamma_star` 的定义域；
- `Gamma_deck_actual` 用于 driver conformance gate，不重新归一化主难度；
- `Gamma_table_actual` 是隔振器产生的任务响应，禁止把它反过来当作每回合难度校准目标。

## 4. 首个任务：开放起始区 + 浅目标容器 PickPlaceCan

### 4.1 原生 BinsArena — REJECTED

- 不使用原生 `PickPlaceCan` 的 `bin1/bin2` 场景。
- 不使用有墙 source bin；Can 初始位置必须位于开放桌面。
- 允许目标区域使用一个有墙浅容器。拒绝的是原生“双 bin + source wall confinement”任务，而不是所有目标容器。
- 原因：source bin 墙会约束抓取前自由体滑移；target walls 则是放置任务本身的一部分。

### 4.2 自定义任务 — DECIDED

工作名：`VibrationPickPlaceCan`。

建议实现基础 — PROPOSED：

- 以 robosuite `TableArena` / `Lift` 任务结构为起点，而不是 `BinsArena PickPlaceCan`；
- 使用 robosuite `CanObject` 资产替换 Lift cube；
- Can 初始位于开放工作台表面；
- 在工作台上添加一个有墙浅目标容器，容器随隔振工作台运动；
- Panda base 属于 deck；
- open table 和目标区域属于 isolated worktable；
- Can 保持 world child + freejoint。

该路线比改造原生 PickPlaceCan 更接近现有 spike，但仍必须把 Lift/cube/table 字符串硬编码改成正式 task handles。

### 4.2.1 工业工作台视觉语言 — DECIDED（Q29）

视觉上继承 ShakeBench 工业工作台，而不是使用裸 robosuite 默认桌面：

- 深灰 phenolic-resin 桌面纹理；
- 深色方管框架、桌面包边和下部横撑；
- 工业脚板、螺栓与结构细节；
- 使用仓库确定性生成并有 SHA-256 记录的 `phenolic_bench_dark_1k.jpg`；
- 在脚板与振动甲板之间增加清晰可见的隔振支座，避免视觉上仍表现为刚性螺栓直连；
- 视觉 frame 与螺栓可以是 collision-free MJCF primitives，不得暗中改变 canonical 惯性。

物理 tabletop 尺寸参考 ShakeBench，冻结为 `0.65 × 0.60 × 0.06 m`。原实现顶部质量 45 kg、四条腿各 3 kg，但这些质量来自 kinematic 场景，不作为已验证真值；canonical 总质量和惯量重新定义。

### 4.3 成功判据 — DECIDED（Q25/Q37/Q43）

用户接受以下原则：

```text
Can 碰撞几何的水平投影完整位于有墙目标容器内边界
+ 已与夹爪释放
+ 连续稳定一段时间
+ 线速度和角速度低于阈值
+ 无非法穿透
```

- 全部判据在当前 worktable/target-container local frame 中计算；
- containment 使用 Can collision geometry 的水平支撑点，不使用中心点、visual mesh 或 world AABB；
- Can 底部必须由箱底支撑且不得跨坐在墙上；目标箱是浅箱，因此不要求整个 Can 低于箱壁；
- 不使用静态 world AABB；
- v0 暂不要求特定最终朝向；
- 目标容器冻结为 ShakeBench `shallow_storage_bin`：外平面 `0.18 × 0.16 m`、壁厚 `0.008 m`、内平面 `0.164 × 0.144 m`、壁高 `0.035 m`、底厚 `0.012 m`；由一个底板和四个 MJCF box collider 构成；
- 目标容器与工作台属于同一个 isolated rigid assembly，不新增自由度或隐藏质量；
- 成功稳定门冻结为：连续 `0.50 s` 保持水平 containment、Can 相对目标箱线速度 `<0.02 m/s`、相对角速度 `<0.20 rad/s`、无 finger–Can contact、由目标底板支撑、非法穿透 `<0.50 mm`；
- 成功只在完整连续窗口通过后锁存；释放后的瞬时弹跳不得立即计为成功。

### 4.4 nominal task layout — DECIDED（Q42）

全部使用 worktable-local 坐标：

```text
Can start XY       = (-0.10, -0.13) m
target center XY   = (-0.10, +0.17) m
transport distance = 0.30 m
```

- Can 从开放桌面开始，source 区域无墙；
- 目标浅箱完整位于 `0.65 × 0.60 m` 桌面内；
- 构建后必须通过 Panda 可达性、无碰撞抓取和箱壁避障 gate；
- 若 nominal layout 失败，只允许按预注册规则整体平移，不得根据振动成绩调位置。

### 4.5 committed initial states — DECIDED（Q44）

```text
410 committed task states
├── 10 dev states
└── 400 official states

Can nominal XY = (-0.10, -0.13) m in worktable frame
XY perturbation = independent uniform ±0.02 m
Can yaw = 0 for every v0 State-track episode
target = fixed
```

- Can 近似轴对称，yaw 随机化对当前 State Track 几乎不产生可审计难度，因此 v0 固定为零；不以初始倾角或初速度替代，避免引入新的初始能量和失败机制；
- 每条状态同时保存 object pose、excitation seed、`t0`、`level_scale` 和 commanded/deck/table Γ；
- 所有 State V0–V3 使用相同 state IDs；正式计分不在线重新采样，也不替换失败 seed；
- 另有 100 个 knee-calibration episodes，与 410 个 committed task states 完全不重叠。

## 5. 观测设计：两个正交轴

### 5.1 当前阶段的排名与开发优先级 — DECIDED（Q21）

```text
task perception axis ∈ {Vision, State}
vibration information axis ∈ {V0, V1, V2, V3}
```

Vision/State 回答“任务几何如何获得”；V0–V3 回答“增加多少专用振动信息”。二者不能混成一条 privilege ladder。

成熟 benchmark 外部调研原先支持“Vision 主榜、State 诊断上界”。用户在了解该证据后决定当前阶段采用：

```text
Current benchmark development and oracle-control evaluation
└── State-{V0,V1,V2,V3}

Deferred ablation section
└── Vision Track（具体 tier 和实验矩阵以后冻结）

Privileged evaluator / recorder
└── full simulator and contact truth
```

当前 State 主线回答：在任务状态真值可用时，机械振动如何影响控制、抓取、接触和放置。它暂时不回答视觉感知退化；腕部视觉只在后续消融中讨论。在 Vision 实验真正完成以前，论文和 README 不得宣称视觉鲁棒性已经被主 benchmark 覆盖。

完整外部证据见 `docs/research/oracle_design_in_mature_manipulation_benchmarks.md`。

### 5.2 Vision Track — DEFERRED（后续消融）

未来消融的候选观测：

```text
agentview RGB（lab/world fixed）
robot0_eye_in_hand RGB
robot proprioception
公开腕部/指尖传感器（具体字段待定）
task instruction / active object id
```

不进入 Vision 主策略：

```text
Can pose truth
target region pose truth
raw table/deck pose or twist
segmentation truth
native contact truth
penetration
mass / friction / COM truth
```

`robot0_robotview` 可作为诊断相机。Vision 消融的 tier、数据量、公平比较和表格位置尚未冻结，当前不实现。

### 5.3 State Oracle-Control Track — DECIDED（Q21）

当前唯一实现和评测轨道的公共任务状态：

```text
robot joint position / velocity
EEF pose in robot-base frame
gripper state
公开腕部/指尖传感器
Can pose in robot-base frame
goal region in robot-base frame
```

目标应表示为 region，而不是伪造唯一 target pose：

```text
goal_center_b
goal_half_extents
goal_z_bounds
orientation_constraint / mask
```

raw table pose/twist 不进入 State 公共字段，因为：

- table pose 会泄漏 V2 的 `q`；
- table twist 会泄漏 V2 的 `qdot`；
- acceleration 会直接替代 V1/V2；
- 只给 State 又会让 Vision/State 差异同时包含任务感知和振动感知。

实时 Can 与 goal region 的 robot-base 表达已经提供完成任务所需的相对几何。完整 support state 应进入 V2 或单独 diagnostic lane。

该轨道使用 task-state truth，因此在论文中应明确称为 `State Oracle-Control Track`，不能暗示已经解决视觉感知。未来 Vision 消融不得与 State 数字无标记混排或求平均。

### 5.4 State V0–V3 振动信息台阶 — DECIDED

所有 tier 共享相同 task state、controller、action schema、policy rate 和执行器；只增加专用振动信息：

| Tier | 共同 task perception | 专用振动信息 |
|---|---|---|
| V0 | State 公共任务字段 | 无 |
| V1 | State 公共任务字段 | 实际 IMU measurement |
| V2 | V1 | 当前 realized support `q/qdot/qdd` |
| V3 | V2 | 自描述的 future authored excitation program |

重要限制：

- V0 仍可能通过物体运动、视觉、触觉间接感知振动；它只是没有专用振动通道。
- V1 必须来自实际刚体运动和明确传感器模型，不能直接把 commanded `qdd/qd` 拼成“IMU”。
- V2 不能包含 seed、`t0`、line phase 或未来轨迹。
- V3 可以看到 authored future command program，不能看到未来接触或未来 realized plant state。
- 当前 Oracle ladder 中，V2 provider 对 support state 是 current-only：不保存 support-state history、不在线辨识谱线、不估计频率、不外推未来；TaskExecutive 自身的 phase memory 不得用于重建激励程序。
- 因此当前 `V3−V2` 的解释是“显式未来程序相对于当前状态、无在线谱辨识控制器的增益”，不是信息论意义上的绝对不可推断性。未来若允许 recurrent learned V2 policy，必须改写为“显式 program 相对于在线辨识的增益”，不得沿用当前解释。
- 原 ShakeBench 当前代码并未正确实现该累积台阶；新仓库只能迁移设计，不能复制旧 privilege groups。

已决定：

- V1 IMU 安装在振动甲板/机器人基座，而不是隔振工作台；
- V2 同时提供 realized deck state 与 realized table-relative-to-deck state；
- V0–V3 完全共享 TaskExecutive、phase machine、target generator、OSC、action schema、20 Hz rate、gripper semantics、initial states 和 horizon。

V1 Policy 只接收带噪声的实际 IMU measurement；clean IMU 与噪声分解只进入 privileged recorder。

IMU 以 200 Hz 固定采样；20 Hz Policy 每步接收最近 10 个带噪样本：

```text
deck_imu_window: float32[10, 6]
deck_imu_dt_s: 0.005
```

传感器方程必须包含 sensor-frame specific force、重力、角加速度杠杆项和向心项。旧 `SyntheticDeckIMU` 的噪声数量级可参考，但其运动学方程不能照搬。

厂商 datasheet 专项调研已经完成，完整证据见 `docs/research/canonical_imu_profile_validation.md`。调研结论为：原拟议参数数量级总体合理，并已增加明确的数字低通和延迟语义。

已冻结的 canonical profile — DECIDED（Q38）：

```text
ODR                         200 Hz
policy window               10 × 6, oldest-to-newest
low-pass                    2nd-order 40 Hz Butterworth
ENBW                        40.7618155662 Hz
filter group delay          ~4.87 ms at low frequency
additional delivery delay   1 sample = 5 ms
nominal end-to-end delay    ~9.87 ms at low frequency

accelerometer               ±16 g, 16-bit, 150 ug/sqrt(Hz)
initial residual bias std   0.02 m/s²
bias diffusion              1e-4 m/s²/sqrt(s), benchmark-authored

gyroscope                   ±2000 deg/s, 16-bit, 0.005 deg/s/sqrt(Hz)
initial residual bias std   0.05 deg/s
bias diffusion              1e-4 deg/s/sqrt(s), benchmark-authored
```

关键 provenance：noise density 的量级由 Bosch BMI088、TDK ICM-42688-P 和 ADIS16470 官方资料支持；精确滤波器、5 ms delivery delay、bias residual 和 Brownian diffusion 是 benchmark authored choices，不能伪称某款器件的 datasheet 参数。

V2 使用后端无关的 pose + spatial twist + spatial acceleration：

```text
deck_pose/twist/accel_in_nominal_frame
table_pose/twist/accel_in_deck_frame
```

pose 使用 position + quaternion；twist/acceleration 使用 linear + angular 6-vector，不直接暴露 MuJoCo `qpos/qvel`，也不把 Euler 导数误当角速度。

V3 program schema 冻结为：

```text
line_accel_amplitude[6, max_lines]
line_omega_rad_s[6, max_lines]
line_phase_at_episode_zero[6, max_lines]
line_mask[6, max_lines]
episode_time_s
ramp_type = quintic_smoothstep
ramp_duration_s
program_frame
```

`phase_at_episode_zero` 已吸收 episode 时间窗口偏移；V3 不暴露未来 realized plant/contact/task state。

### 5.5 Privileged evaluator 与额外 diagnostic lanes — DECIDED / 字段细节 OPEN

不进入主榜：

```text
D_task_truth
D_support_state
D_full_dynamics
D_future_realized
```

完整 recorder-only truth 使用显式 `privileged_` 前缀，包括世界系物体/目标、deck/table 状态、接触、穿透、质量、惯量、摩擦、COM、原始/裁剪/实际动作和 controller provenance。

### 5.6 所有 State V0–V3 共享的静态物理合同 — DECIDED（Q39）

以下 episode-static / benchmark-static metadata 向所有 tier 公开，不构成振动信息 privilege：

```text
worktable dimensions / M_ref / I_ref
Can dimensions / mass / inertia
isolator fn / zeta / k / c
table-object and finger-object nominal friction
IMU extrinsics / sample rate / canonical sensor profile id
control rate
target-container geometry and success semantics
```

运行时 realized deck/table state 仍严格按 V1–V3 权限提供。公开模型参数不等于公开实际轨迹。

## 6. 动作与控制

### 6.1 v0 — DECIDED

- Panda；
- 原生 robosuite 7D action；
- `OSC_POSE + 1D gripper`；
- policy/control frequency 固定 20 Hz；
- 不沿用 spike 的自定义 8D 双指目标、force switch 和 `move_action_gain`。
- State V0–V3 共享完全相同的 TaskExecutive、phase machine、target generator、OSC 参数、gripper semantics、initial states 和 episode horizon；只替换 VibrationProvider。

### 6.2 v1 — OPEN

- 是否加入 policy bandwidth suite；
- official physics timestep；
- gripper actuator force、接触预载和关闭语义；
- OSC 增益与输出缩放；
- 控制器参数是否成为 task context metadata。

## 7. 摩擦与工件因子

### 7.1 必须分开的接触接口 — DECIDED

```text
mu_table_object
mu_finger_object
```

- 前者影响抓取前与释放后的自由体滑移；
- 后者影响抓取和搬运中的在手滑移；
- 二者不能用一个“workpiece friction”同时改变后再声称完成归因。

### 7.2 OPEN

- v0 Can 使用哪个 nominal `mu_table_object`；
- v0 Can 使用哪个 nominal `mu_finger_object`；
- friction 是否通过同几何、同质量的 material variants 实现；
- 是否以及何时加入不同质量、质心、几何的工件；
- friction 因子进入正式 suite 还是仅作为 diagnostic slice。

### 7.2.1 canonical v0 sliding friction — DECIDED（Q45）

使用显式 contact pair：

```text
Can ↔ open worktable / target bottom / target walls:
  sliding_mu = 0.30

Can ↔ Panda finger pads:
  sliding_mu = 1.00
```

- 两个接口不得用一个 `workpiece_mu` 联动；
- 未来摩擦消融必须一次只改变一个接口；
- torsional/rolling friction、`condim` 和 contact `solref/solimp` 仍通过独立接触验证冻结，不根据任务成功率反调。

### 7.3 v0 Can 惯性 — DECIDED（Q35）

- 复用 robosuite Can 的视觉/碰撞几何；
- 不使用原生 mesh density 编译出的约 16 g 质量；
- canonical mass 固定为 `0.349 kg`，参考 ShakeBench/YCB soup-can 元数据；
- 以显式等效圆柱惯量覆盖 mesh-derived inertia；
- 质量、惯量和 COM 编译后必须自动断言。

## 8. 纯仿真物理验证

### 8.1 Claim 边界 — DECIDED

可以声称：

> 在公开、可解释的 canonical 六自由度线性隔振模型和 MuJoCo 接触模型中，观察到激励、摩擦和任务失效之间的系统关系。

不能直接声称发现现实世界物体的普适滑移规律。

### 8.2 必须通过的 gate — DECIDED

1. **Deck driver contract**：冻结 equality weld 的 `eq_solref/eq_solimp`、deck mass/inertia、physics `dt` 和写入时序；正值 `eq_solref` time constant 不得低于 `2×dt`；不得根据 task SR 调整这些值；
2. **Six-axis driver conformance**：在全 knee-calibration 候选 Γ 安全范围内，逐轴/逐谱线验证 deck command→actual amplitude 与 phase；预注册目标为 line amplitude relative error `<=1%`、phase error `<=1°`、`|Gamma_deck_actual/Gamma_commanded−1|<=1%`。若 pilot 无法在稳定求解下达到，driver 设计视为 blocked，不得仅记录误差后继续计分；
3. **Six-axis isolator transfer**：六个自由度各自的单频传递函数与解析解对比，并验证六轴联合激励的 command/deck/table 实际响应；
4. **Static equilibrium**：验证 32 kg table preload 后的空载 nominal pose、0.349 kg Can 引起的额外静态偏移、target geometry 和 Panda 可达性；
5. **Timestep/solver convergence**：至少三档 physics timestep，且同时满足频率采样与 constraint time-constant 条件；
6. **Contact/friction validation**：摩擦斜面或单轴加速度滑移阈值与解析结果对比；
7. **Gamma=0 bounded parity**：不要求动态隔振桌与原生静态 robosuite 逐轨迹“等价”；要求相同 robot/controller/task geometry 下的可解性、nominal task poses、成功判据和动作语义在预注册容差内一致，并单独报告动态桌静态响应；
8. **Safety rejection**：极端频率、Γ、隔振行程、角度和 solver travel 触发 fail-closed 拒绝门；
9. **Interface-resolved metrics**：按接触接口记录力、冲量、滑移和穿透；
10. **Cross-process determinism**：同一 state ID 至少由 3 个独立进程重放，比较完整 state/action/metric traces；只在同一进程 reset 重复不算通过。

## 9. spike 与旧 ShakeBench 的复用边界

### 9.1 接受的物理模式

- mocap driver + weld + dynamic deck；
- robot hard-mounted to deck；
- table connected through linear 6-DoF support；
- 显式 task contact pair；
- 解析六轴谱、seed/t0、Γ 和安全门；
- matched episodes、连续指标和 recorder-only truth。

明确拒绝把 dynamic deck 直接替换成带动态后代的 mocap parent：MuJoCo 将 mocap motion 视为逐步静态重定位，隔振 joint 看不到正确的基座惯性输入。最小 5 Hz 共振探针中该方案得到 relative-joint amplitude `0`、transmissibility `1`，而解析值约为 `5.099`。mocap subtree 只能精确搬动运动学/视觉层级，不能替代本 benchmark 的动态 base excitation。

### 9.2 必须重写

- `ShakeDeckLift` 的 Lift/cube/table 字符串硬编码；
- `_pre_action` 一步超前调度；
- Torch stub 和私有振动函数访问；
- 当前 success 和 slip metrics；
- experiment scripts、cache 和统计混合；
- 旧 privilege group / Oracle 实现。

正式结论仍是：接受 spike 的物理路线，不接受 spike 的现成产品化资格。

## 10. 未决问题索引与依赖

### 10.1 当前明确未决

| ID | 问题 | 依赖 | 下一步证据/决策 |
|---|---|---|---|
| U5 | fixed `f_n/ζ/k/c` 的具体数值 | `M_ref/I_ref` 已冻结 | physics-only transfer-envelope pilot |
| U6 | 传递包络的具体目标区间、评分函数与 tie-break 数值 | U5 | physics pilot 前预注册 |
| U7 | 主计分 `Gamma_star` 的实际数值 | U5/U6 + controller gate | knee-calibration pilot |
| U10 | torsional/rolling friction、condim 与 contact solref/solimp | sliding friction 已冻结 | friction/contact canonical tests |
| U11 | official physics timestep 与 deck `eq_solref/eq_solimp` | excitation `f_max` 已冻结；受 weld/contact time constant 约束 | 三档 timestep + driver conformance 实验 |
| U12 | official gripper force semantics | controller validation | actuator audit |
| U13 | PickPlace 之后的任务 taxonomy | v0 通过 | 后续设计轮次 |

### 10.2 关键依赖链

```text
open-table geometry
→ M_ref / I_ref
→ nominal fn / ζ and frozen k / c
→ physics-only transfer-envelope selection
→ driver/contact/controller gates
→ commanded Gamma knee calibration
→ official seeds and initial states
```

```text
State Oracle-Control contract
→ V0–V3 exact sensor semantics
→ recorder schema
→ dataset schema
→ future method fairness

deferred Vision ablation
→ later define a separate perception contract
→ never mix unlabelled scores with State
```

```text
Can geometry + target region
→ containment semantics
→ settle and velocity thresholds
→ success evaluator
→ scorecard
```

## 11. 剩余实验闭环

不再需要新的产品选择题。实现阶段必须按既定规则产出：

1. physics-only transfer-envelope 与最终 `f_n/ζ/k/c`；
2. timestep/solver/contact profile，以及 deck `eq_solref/eq_solimp` 和六轴 conformance；
3. gripper/OSC applied-force 验证；
4. 100 个 knee-calibration episodes 上的冻结 `Gamma_star`；
5. 400 official states 的 State V0–V3 配对结果。

## 12. Scorecard 与统计

### 12.1 主指标与 Γ 膝点算法 — DECIDED（Q46）

- 主排名只使用一个冻结难度点的成功率：`SR@Gamma_star`；
- 不使用跨 Γ 的 AUC；
- 不把 placement、slip 或 penetration 加权进主分数；
- 不同 Γ 的完整性能曲线只作为后续消融；
- `Gamma_star` 必须在 official evaluation 之前通过独立 calibration scan 冻结；
- 所有 State V0–V3 使用完全相同的 `Gamma_star`，不得逐 tier 选择膝点。
- `Gamma_star` 明确定义在 `Gamma_commanded` 上；`Gamma_deck_actual` 必须通过第 8.2 节 driver conformance，`Gamma_table_actual` 只作为响应量。

已接受的膝点选择：

1. 只用 State-V0 reference oracle controller；
2. 使用与 400 official states 完全不重叠的 100 个 knee-calibration episodes（10 object placements × 10 excitation seed/`t0` pairs）；
3. 先要求 `SR_V0(Gamma=0) >= 0.95`，否则先修任务或控制器；
4. 先在安全 Γ 网格粗扫，再在首次退化区细扫；
5. 对 `SR_V0(Gamma)` 做 non-increasing isotonic fit；
6. 定义 `Gamma_star` 为拟合曲线首次达到绝对 `SR=0.5` 的插值点；
7. 将结果按预注册 `0.05` 精度舍入并冻结；
8. 保存 calibration states、原始 SR、拟合曲线与配置 hash；
9. 若安全范围内没有 crossing，判定当前任务/激励没有形成有效 decision point，不得把边界值冒充膝点。

calibration scan 作为 provenance 公开，但不作为 V0–V3 的正式跨 Γ 比较。

单点主结果增加 saturation/floor 标记：若任一 tier 在 `Gamma_star` 上 `SR>0.90` 或 `<0.10`，受影响的 adjacent contrast 标记为 `ceiling_limited` 或 `floor_limited`。不自动更换主 Γ，也不新增临时主指标；次级 Γ 只能进入预先声明的后续消融。

### 12.2 失败与基础设施错误 — DECIDED（Q47）

- task failure、controller failure 和 action-induced penetration violation 均进入分母，`success=0`；
- 不报告 valid-only SR 作为主结果；
- policy crash 进入分母并记失败；
- simulator/infrastructure crash 只允许重跑完全相同的 state ID；
- 相同 state 重复 infrastructure failure 时整组标为 incomplete，不替换 seed；
- missing、rerun、failure 和 physics-violation 数量全部公开。

### 12.3 V0–V3 单点配对统计 — DECIDED（Q48）

所有 tier 在同一 `Gamma_star` 和同一 400 official state IDs 上运行；主统计只比较 paired binary success。placement、slip、contact loss 等继续记录为诊断，不参与主排名。

每个 tier 报告：

```text
success count / 400
success rate
95% Wilson interval
failure-reason histogram
physics-violation count
```

配对增量固定为 `V1−V0`、`V2−V1`、`V3−V2` 和 `V3−V0`，每项报告 `Delta SR`、固定 seed 的 10,000 次 paired-bootstrap 95% CI、`n_gain` 与 `n_loss`。

### 12.4 样本量与最小可检效应 — DECIDED

`n=40` 不作为正式样本量：在 `SR≈0.5` 时其 Wilson 95% CI 约为 `[0.352, 0.648]`，且配对 `Delta SR=0.10` 在典型 discordant-pair rate 下功效不足。

预注册目标：

```text
minimum effect of interest  = 0.10 Delta SR
two-sided alpha             = 0.05
target power                >= 0.80
official paired state count = 400
```

正态近似下，当总不一致对比例 `q=p_gain+p_loss` 为 `0.2/0.3/0.5` 时，检测 `Delta SR=0.10` 约需 `149/228/385` 个配对 episode；因此 400 是不依赖乐观配对相关性的保守冻结值。最终报告仍必须给效应量、区间、`n_gain/n_loss`，不得用“未显著”代替“效应很小”。
