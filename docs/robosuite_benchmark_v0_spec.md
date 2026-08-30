# robosuite 振动操作 Benchmark v0 规范

> 规范日期：2026-08-28  
> 状态：设计冻结；允许通过预注册 physics pilot 填入第 14 节列出的派生数值。  
> 当前实现范围：State Oracle-Control Track；Vision 仅作为后续消融。

> 2026-08-30 reconciliation：committed state protocol 已与设计树同步为 10 个 dev + 400 个 official states，Can yaw 恒为 0；Phase 01 的 authored spectrum 是新的 candidate profile，不是已证明的旧 ShakeBench exact reuse。物理 timestep、weld/contact `solref/solimp`、isolator 参数和 `Gamma_star` 仍未冻结。

## 1. 目标与非目标

v0 构建一个基于 robosuite 1.5.2 / MuJoCo 的可复现振动操作测试床，研究在任务状态真值已知时，机械振动如何影响控制、接触、抓取、搬运和放置。

当前不声称：

- 已解决视觉感知或腕部相机抗振；
- 已发现现实世界普适滑移规律；
- 已提供训练策略或后训练方法；
- 已覆盖 PickPlace 之外的大多数任务；
- canonical 隔振器是具体硬件的数字孪生。

## 2. 发布与代码边界

- 新实现位于独立 robosuite 仓库。
- 当前 ShakeBench 仓库只作为科学协议、激励算法、视觉语言和验证证据来源。
- 新仓库不得在运行时依赖当前 Isaac/Newton 包。
- `spike/` 作为 conformance oracle，不直接复制成 production `src/`。
- benchmark、未来 policy learning 和未来 Vision ablation 必须分包；benchmark 不依赖作者方法。

建议模块：

```text
shakebench_core/
  excitation.py
  calibration.py
  protocol.py
  scoring.py
  schemas.py

shakebench_robosuite/
  deck.py
  isolator.py
  sensors.py
  contacts.py
  metrics.py

tasks/
  vibration_pick_place_can.py

tools/
  validate_transfer.py
  validate_contact.py
  select_isolator.py
  select_gamma_knee.py
```

## 3. 机械拓扑

```text
world
├── deck_driver                     # prescribed mocap input
├── dynamic_vibrating_deck          # free body, weld to driver
│   ├── Panda robot base            # rigid mount
│   └── canonical 6-DoF isolator
│       └── isolated_worktable
│           ├── industrial visual frame
│           └── shallow target container
├── Can                             # world child + freejoint
└── lab/world assets
```

- Can 只能通过重力、接触和抓取运动，不能运动学挂到工作台。
- target container 与 worktable 是同一个 isolated rigid assembly。
- target container、视觉框架和装饰件不新增隐藏质量或自由度。

### 3.1 Deck driver

- 使用 `mocap driver + equality weld + ordinary dynamic deck`。
- 不允许移动 mocap geom 直接承担 task contact。
- driver 必须在对应 physics step 的位置阶段前更新；不得把现有 `_pre_action + one-step lead` 当成未经测试的永久接口。
- 必须记录 command/actual 六轴 pose、twist、acceleration、weld error 和 constraint wrench。

## 4. Canonical worktable 与隔振器

### 4.1 Worktable inertial contract

```text
tabletop dimensions = [0.65, 0.60, 0.06] m
M_ref               = 32.0 kg
I_ref principal     = [0.9696, 1.1363, 2.0867] kg·m²
```

- `I_ref` 是上述尺寸和质量的等效长方体惯量。
- canonical inertial 显式定义在 isolated worktable root。
- visual frame、横撑、脚板、螺栓和隔振器外观不参与惯量推导。
- 更换 payload 不重新计算 `k/c`；payload 应自然改变实际响应。

### 4.2 Visual contract

- 视觉继承 ShakeBench 深灰 phenolic-resin 工业工作台语言。
- 使用 `assets/textures/phenolic_bench_dark_1k.jpg` 或其经许可复制品。
- 使用 MJCF box/cylinder primitives 重建包边、方管框架、横撑、脚板和螺栓。
- 脚板和 deck 之间必须有可见隔振支座，不能视觉上表现为刚性螺栓直连。
- display-only geoms 必须禁用 collision。

### 4.3 Canonical linear 6-DoF isolator

相对自由度：`tx/ty/tz/rx/ry/rz`。每轴使用独立线性弹簧和粘性阻尼：

```text
k_i = m_eff_i (2π f_n,i)^2
c_i = 2 ζ_i m_eff_i (2π f_n,i)
```

- 首版只冻结一组参数，不把 isolator 作为 official suite 轴。
- `f_n/ζ` 必须由第 14.1 节的 physics-only transfer-envelope 流程选择。
- 选择后一次性派生并冻结 `k/c`；不得根据 task success 或未来方法成绩回调。
- 必须有 travel/angle safety limits 和启动拒绝门。

## 5. Task：VibrationPickPlaceCan

### 5.1 Object

- 使用 robosuite Can 的视觉与碰撞几何。
- 不使用 stock mesh density 产生的约 16 g 质量。
- canonical mass：`0.349 kg`。
- inertia：根据 canonical Can collision envelope 的等效圆柱显式计算。
- mass、inertia 和 COM 必须在编译后自动断言。

### 5.2 Initial open-table region

worktable-local nominal pose：

```text
Can XY = [-0.10, -0.13] m
```

初始区域无墙，允许振动导致抓取前自由体滑移。

### 5.3 Target container

目标是一个随 isolated worktable 运动的浅容器：

```text
center XY        = [-0.10, +0.17] m in worktable frame
outer XY         = [0.18, 0.16] m
wall thickness   = 0.008 m
inner XY         = [0.164, 0.144] m
wall height      = 0.035 m
bottom thickness = 0.012 m
```

- 用一个底板和四个 MJCF box collider 构成。
- source-to-target nominal transport distance 为 `0.30 m`。
- container collision geoms 不改变第 4.1 节的 explicit inertial。

### 5.4 Success

成功只在以下条件连续成立 `0.50 s` 后锁存：

```text
Can collision geometry horizontal projection fully inside target inner footprint
Can supported by target bottom
finger–Can contact absent
Can relative linear speed < 0.02 m/s
Can relative angular speed < 0.20 rad/s
illegal penetration < 0.50 mm
```

- 全部量在 target-container/worktable local frame 中计算。
- containment 使用 collision geometry 的水平支撑点，不用 COM-only、visual mesh 或 world AABB。
- shallow container 不要求整个 Can 低于箱壁。
- v0 不要求特定最终朝向。

## 6. Contact 与摩擦

使用显式、作用域受限的 task contact pairs：

```text
Can ↔ worktable / target bottom / target walls:
  sliding_mu = 0.30

Can ↔ Panda finger pads:
  sliding_mu = 1.00
```

- 两个接口不得由一个 `workpiece_mu` 联动。
- 未来 friction ablation 一次只能改变一个接口。
- `condim`、torsional/rolling friction、`solref/solimp` 和 timestep 通过第 14.2 节 canonical contact tests 冻结。
- 不得根据 PickPlace 成功率反调 contact 参数。

## 7. Excitation

### 7.1 v0 family

Phase 01 remediation 选择 **B：new authored v0 candidate**。旧 ShakeBench 激励实现不在当前仓库中，无法证明 exact reuse；本节数值只表示待 bench-author 批准的 authored candidate：

- 六轴解析随机多频谱；
- 每轴独立 line jitter/phase，episode 级确定性；
- 公共 `seed/t0` 时间窗口；
- 五阶 episode-relative ramp；
- 当前 2.5–8 Hz 中心频带、轴比例和旋转派生规则；
- Γ 标定、25 mm deck displacement gate 和 solver travel gate；
- v0 原有 Γ 候选用于 knee calibration，不直接作为主结果。

### 7.2 Required response records

```text
Gamma_commanded
Gamma_deck_actual
Gamma_table_actual
commanded deck q/qdot/qdd
realized deck pose/twist/acceleration
realized table pose/twist/acceleration
table-relative-to-deck pose/twist/acceleration
```

主计分难度为第 12 节冻结的单点 `Gamma_star`。

## 8. Robot、动作与共享 Oracle controller

```text
robot          Panda
action         7D total = 6D robosuite OSC_POSE + 1D gripper
policy rate    20 Hz
```

- V0–V3 完全共享 TaskExecutive、phase machine、target generator、OSC gains、action schema、policy rate、gripper semantics、initial states 和 horizon。
- 只允许替换 `VibrationProvider`。
- 不复用 spike 的 8D 双指控制、force-switch 或 `move_action_gain`。
- official gripper actuator force 和 OSC 数值必须通过第 14.3 节验证后冻结。

## 9. State Oracle-Control Track

当前唯一实现与正式评测轨道。公共 task state：

```text
robot joint position / velocity
EEF pose in robot-base frame
gripper state
public wrist/fingertip sensors
Can pose in robot-base frame
goal region in robot-base frame
```

goal region 表示为 center、half-extents、z bounds 和 orientation mask，不伪造唯一 target pose。

### 9.1 Vibration information tiers

| Tier | 新增信息 |
|---|---|
| V0 | 无专用振动 channel |
| V1 | delayed/noisy deck IMU window |
| V2 | V1 + current realized support state |
| V3 | V2 + future authored excitation program |

V0 仍可从 Can/goal 相对运动间接推断扰动；其定义只是没有专用振动 channel。

### 9.2 V1 canonical deck IMU

- 安装在 robot-base/deck origin，sensor frame 与 robot base 对齐。
- Policy 每 20 Hz step 只收到最近 10 个 noisy samples。
- clean signal、bias/noise/filter state、clipping 和 acquisition timestamp 仅进 recorder。

```text
ODR                         200 Hz
window                      [10, 6], oldest-to-newest
low-pass                    2nd-order 40 Hz Butterworth
ENBW                        40.7618155662 Hz
additional delivery delay   1 sample = 5 ms
nominal low-frequency total ~9.87 ms

accelerometer               ±16 g, 16 bit, 150 ug/sqrt(Hz)
accel residual bias std     0.02 m/s²
accel bias diffusion        1e-4 m/s²/sqrt(s)

gyroscope                   ±2000 deg/s, 16 bit, 0.005 deg/s/sqrt(Hz)
gyro residual bias std      0.05 deg/s
gyro bias diffusion         1e-4 deg/s/sqrt(s)
```

specific-force 方程必须包含 gravity、`alpha × r` 和 `omega × (omega × r)`；gyro 输出真实角速度，不使用 Euler 导数。

### 9.3 V2 support-state schema

后端无关：

```text
deck_pose_in_nominal_frame          # position + quaternion
deck_twist_in_nominal_frame         # linear + angular
deck_accel_in_nominal_frame         # linear + angular

table_pose_in_deck_frame
table_twist_in_deck_frame
table_accel_in_deck_frame
```

不直接暴露 MuJoCo `qpos/qvel`。

### 9.4 V3 program schema

V3 必须自描述并可在任意 future query time 重建 authored deck command：

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

- `phase_at_episode_zero` 已包含该 episode 的时间窗口偏移，不要求暴露 seed/`t0`。
- 不允许从硬编码 RMS 表反推 line amplitude。
- V3 只能看到 future authored command，不能看到未来 realized table/contact/task state。

## 10. Shared static PolicyTaskContext

所有 V0–V3 公开：

```text
worktable dimensions / M_ref / I_ref
Can dimensions / mass / inertia
isolator fn / zeta / k / c
nominal friction/contact profile IDs
IMU extrinsics / sample rate / profile ID
control rate
target-container geometry / success semantics
support topology ID
```

公开模型参数不等于公开 runtime realized state。

## 11. Initial-state protocol

```text
410 committed task states
├── 10 dev
└── 400 official
```

每个 state：

```text
Can XY perturbation around nominal: independent uniform ±0.02 m
Can yaw: 0 for every State-track episode
target: fixed
object pose
excitation seed / t0 / level_scale
Gamma commanded/deck/table
```

- V0–V3 使用相同 state IDs。
- official evaluation 不在线采样或替换失败 seed。
- 另建 100 个与 official 完全不重叠的 knee-calibration episodes。

## 12. Main score 与 Gamma knee

### 12.1 Gamma selection

只用 State-V0 reference Oracle controller：

1. 在 100 knee-calibration episodes 上要求 `SR(Gamma=0) >= 0.95`；
2. 在物理安全 Γ 范围粗扫；
3. 在首次退化区细扫；
4. 对 `SR_V0(Gamma)` 做 non-increasing isotonic fit；
5. 定义 `Gamma_star` 为拟合曲线首次达到绝对 `SR=0.5` 的插值点；
6. 按 `0.05` 精度舍入并冻结；
7. 保存 calibration states、raw SR、fit 与配置 hash；
8. 若安全范围内没有 crossing，任务/激励没有形成有效 decision point，不得用边界值冒充。

Gamma scan 是 calibration provenance；跨 Γ 性能曲线属于后续消融。

### 12.2 Main score

唯一主指标：

```text
SR@Gamma_star
```

- 不使用 AUC。
- 不把 placement/slip/penetration 加权进主分数。
- 所有 V0–V3 使用相同 `Gamma_star`。

### 12.3 Paired statistics

每个 tier 在相同 400 official state IDs 上报告：

```text
success count / 400
success rate
95% Wilson interval
failure-reason histogram
physics-violation count
```

固定比较：`V1−V0`、`V2−V1`、`V3−V2`、`V3−V0`。每项报告：

```text
Delta SR
95% paired-bootstrap CI, 10,000 resamples, fixed seed
n_gain
n_loss
```

placement、free-table slip、in-hand slip、contact loss、penetration 和 response tracking 只作为诊断。

## 13. Failure 与完整性

- task/controller/action-induced physics failure 进入分母，`success=0`。
- penetration 超过 `0.50 mm` 进入分母并标 `physics_violation`。
- policy crash 进入分母并记失败。
- simulator/infrastructure crash 只重跑相同 state ID。
- 同一 state 重复 infrastructure failure 时整组 incomplete，不替换 seed。
- 不报告 valid-only SR 作为主结果。
- missing、rerun、failure 和 violation 数量全部公开。

## 14. 实现中必须派生并冻结的数值

这些不是新的产品决策；必须按已接受的验证规则由实验产生。

### 14.1 Isolator pilot

- 扫描 `(f_n, zeta)`；
- 比较解析与 MuJoCo transmissibility；
- 计算 `T_accel/R_relative/D_relative/T_peak/travel_margin`；
- 使用预注册 transfer envelope 与 deterministic tie-break；
- 输出最终六轴 `f_n/zeta/k/c`。

### 14.2 Physics/contact pilot

- 至少三档 physics timestep 收敛；
- 六轴单频和联合激励；
- static support、斜面和单轴滑移阈值；
- 冻结 solver、condim、torsional/rolling friction、contact `solref/solimp`；
- 验证 8 mm wall 对 solver travel gate 的约束。

### 14.3 Controller/gripper pilot

- Gamma=0 acquisition、lift、transport、place、release 全链通过；
- 冻结 OSC gains、output scale、gripper actuator force/position semantics；
- 验证 actuator applied force，而不只检查配置值；
- 共享 controller 在 V0–V3 不发生隐式分叉。

### 14.4 Gamma knee pilot

- 仅在第 11 节独立 calibration set 上运行；
- 输出冻结 `Gamma_star` 和 provenance。

## 15. Release gates

1. clean clone 一条命令运行；
2. dependency lock/container；
3. fixed-seed deterministic trajectory；
4. six-axis driver tracking；
5. six-axis isolator analytical agreement；
6. static contact and timestep convergence；
7. friction threshold validation；
8. Gamma=0 task solvability；
9. success evaluator boundary tests；
10. V0 ⊂ V1 ⊂ V2 ⊂ V3 key-set and fail-closed permissions；
11. recorder truth cannot leak to policy observation；
12. 100 knee-calibration episodes and 400 official states committed；
13. all crashes, reruns, failures and violations auditable。

## 16. Deferred work

- Vision Track（agentview + eye-in-hand）只作为后续消融；
- 多物体、多任务和 mechanism taxonomy；
- friction/material suite；
- policy bandwidth/frequency suite；
- author post-training method 与 SOTA 比较；
- real shaker validation / sim-to-real；
- panel operation。

## 17. Supporting records

- `docs/robosuite_benchmark_design_tree.md`
- `docs/research/spike_implementation_formal_review.md`
- `docs/research/robomm_benchmark_positioning.md`
- `docs/research/oracle_design_in_mature_manipulation_benchmarks.md`
- `docs/research/canonical_imu_profile_validation.md`
