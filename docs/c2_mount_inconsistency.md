# C2 安装点与可见场景的矛盾问题说明（当前项目）

更新日期：2026-08-18
状态：**阶段 A 已按 `docs/support_model_implementation_plan.md` v1.2 修复**；本文保留为根因分析记录。修复后 official 1 s ×5 谱探针（`out/stage_a_spectral_probe.json`）不再出现 `robot_link<->platen` 伪接触。
范围：`src/shakebench/{config,mounting,scene,task}.py`、`configs/scenarios.yaml`
结论：当前 C2 写法把**三个不同坐标系**叠加进同一组刚体，导致“振动地板 + Panda + 工作台”不是同一个刚体。该问题已经表现为 official 档可复现的伪接触、穿透指标失效和评分门槛无法通过。

---

## 1. TL;DR

当前实现同时使用三套 XY 坐标：

| 用途 | 数值 |
|---|---|
| C2 振动测点（虚拟测量坐标） | `arm_mount_xy_m = (0.75, -0.45)`，`table_mount_xy_m = (-0.75, 0.45)` |
| 紧凑任务布局（可见物理位置） | `robot_base = (-0.47, 0)`，`worktable_center = (0.18, 0)` |
| 可见振动地板 | `platform_center = (0, 0)`，尺寸 `1.60 × 1.10 m` |

问题不在“用了哪套坐标”，而在**同一块可见地板被当作一个刚体，但机械臂和工作台却用另一套虚拟测点运动直接平移**，两者之间没有任何刚体变换把虚拟测点映射回可见位置。后果：

- Panda 基座参考点相对可见地板表面的动态错位 RMS ≈ **3.0 mm**，峰值约 **±10.0 mm**（实际间隙区间约 `[-2.6, +17.0] mm`，而配置净空只有 7 mm）；
- 工作台腿底相对可见地板表面的动态错位 RMS ≈ **3.0 mm**，实际间隙区间约 `[-2.7, +16.0] mm`；
- 臂/台 Z 向差动峰值 **15.6 mm**（新随机流，seed 17，16 s）；若按可见坐标计算同一差动只有约 **4.0 mm**；
- 新的 official 1 s 谱探针已实测到 `robot_link<->platen` 最大穿透 **0.665 mm**、settle 阶段腕力峰值 **102.4 N**，超过 0.3 mm 评分门槛。

---

## 2. 三套坐标的由来

### 2.1 C2 测点坐标（激励输入）

`src/shakebench/config.py`：

```python
arm_mount_xy_m:  tuple[float, float] = (0.75, -0.45)
table_mount_xy_m: tuple[float, float] = (-0.75, 0.45)
```

这是“车辆测量坐标系”里的两个安装点，相距

```text
sqrt((0.75 - (-0.75))^2 + (-0.45 - 0.45)^2) = 1.749 m
```

它决定了旋转激励在两个安装点产生的差动平动量级，是当前难度/差动输入的来源。

### 2.2 紧凑任务布局坐标（可见场景）

同一文件：

```python
robot_base:      tuple[float, float, float] = (-0.47, 0.0, 0.08)
worktable_center: tuple[float, float, float] = (0.18, 0.0, 0.34)
platform_center:  tuple[float, float, float] = (0.0, 0.0, 0.04)
```

Panda 基座与工作台中心在可见场景里只相距 0.65 m，这是为了让工作半径约 0.855 m 的 Panda 能够到工作台。代码注释明确承认：

> The vehicle measurement coordinates are intentionally separate from the compact task-layout coordinates below.

但当前实现**没有**任何“测量坐标 → 紧凑布局”的映射模型，只是把两套坐标线性叠加。

### 2.3 可见地板中心坐标

`scene.py` 里 `platform` 是 `1.60 × 1.10 × 0.08 m` 的可见/可碰撞 Cuboid，`task._write_supports()` 用**地板中心六轴运动**直接写它的 root pose。也就是说可见地板自己按第三套中心坐标运动。

---

## 3. 当前写法的代码路径

### 3.1 测点映射

`src/shakebench/mounting.py::motion_at_mount()`：

```python
out[..., :3] = motion[..., :3] + rotated_mount - mount
```

即

```math
t_{\mathrm{mount}}(t) = t(t) + R(t)\,r_{\mathrm{mount}} - r_{\mathrm{mount}}
```

其中 `r_arm = (0.75, -0.45, 0)`，`r_table = (-0.75, 0.45, 0)`。速度同理用 `ω × (R r)` 加中心线速度。

### 3.2 支撑状态写入

`src/shakebench/task.py::_support_state()`：

```python
quat = quat_from_euler_xyz(motion[:, 3], -motion[:, 4], motion[:, 5])
anchor = local if rotation_anchor is None else rotation_anchor
rotated_offset = quat_apply(quat, local - anchor)
position = self.scene.env_origins + motion[:, :3] + anchor + rotated_offset
```

`_write_supports()` 的驱动清单是：

```python
(platform,  platform_center,          q,            qd,          None)
(robot,     resolved_robot_base,      arm_q,        arm_qd,      None)
(worktable, resolved_worktable_center, table_q,      table_qd,    None)
(target,    resolved_target_center,    table_q,      table_qd,    worktable_local)
(table_legs, leg_local,                table_q,      table_qd,    worktable_local)
```

于是三个实际写入的位姿为：

| 资产 | 实际写入的位置 |
|---|---|
| 可见地板 | `t + c`，旋转 `R`（`c = (0,0,0.04)`） |
| Panda 根 | `t_arm + l_arm`，旋转 `R`（`l_arm = (-0.47, 0, 0.087)`） |
| 工作台 | `t_table + l_table`，旋转 `R`（`l_table = (0.18, 0, 0.347)`） |
| 桌腿/目标盒 | `t_table + l_table + R(l_i - l_table)`，旋转 `R` |

这里的关键是：`l_arm`、`l_table` 是**可见布局**里的偏移，`t_arm`、`t_table` 却来自**虚拟测点** `r_arm`、`r_table`；两者直接相加，没有把 `l` 旋转到 `r` 所在坐标系。

---

## 4. 为什么这在数学上不构成刚体

如果地板、Panda、工作台真的是同一块刚体，且可见地板中心 `c = (0, 0, 0.04)` 的位姿是 `(t, R)`，那么任意资产在可见坐标 `l` 处的正确位置应是

```math
p^*(l) = t + c + R(l - c)
```

而当前实现给 Panda 根的是

```math
p_{\mathrm{current}}(l_a) = t + R\,r_a - r_a + l_a
```

两者之差：

```math
\Delta_a = p_{\mathrm{current}}(l_a) - p^*(l_a)
         = (I - R)(l_a - r_a - c)
```

`c` 只有 Z 分量 0.04 m，对差动的贡献是 `O(θ² c)` 级（微米以下），因此起决定作用的是 XY 项：

```math
(l_a - r_a)_{xy} = (-1.22,\; 0.45) \quad (\| \cdot \| \approx 1.30\ \mathrm{m})
```

对工作台侧同理：

```math
\Delta_t = (I - R)(l_t - r_t - c),\qquad
(l_t - r_t)_{xy} = (0.93,\; -0.45) \quad (\| \cdot \| \approx 1.03\ \mathrm{m})
```

默认转动激励是毫弧度量级（`rx=4 mrad`、`ry=2 mrad`、`rz=1.2 mrad` RMS），但乘上 1 m 级力臂后，Z 向差动就是毫米级；7 mm 的 `robot_mount_dynamic_clearance_m` / `table_mount_dynamic_clearance_m` 无法覆盖峰值。

因此，“地板 + 臂 + 台”不是刚体，而是一个**可见位置与激励位置不一致的非刚性叠加系统**。

---

## 5. 数值复算结果

复算方式：不依赖 Isaac 环境，用 NumPy 按当前 `vibration.py` 的 `[seed, env_id, axis_index]` 随机流、默认频带、解析谱线，以及 `mounting.py` 的完整 SE(3) 公式，在 1 kHz 外循环下重放 seed 17 的 16 s（跳过 ramp 首 1 s）。

### 5.1 当前随机流（P0 修复后，seed 17）

| 指标 | RMS | 峰值/区间 |
|---|---:|---:|
| Panda 根相对可见地板表面偏差（扣除名义 7 mm 净空） | 3.04 mm | ±10.02 mm |
| Panda 根与可见地板实际间隙 | 7.63 mm | `[-2.61, +17.02] mm` |
| 工作台 FL 腿底相对可见地板表面偏差（扣除名义 7 mm 净空） | 3.03 mm | ±9.71 mm |
| 工作台 FL 腿底与可见地板实际间隙 | 7.63 mm | `[-2.71, +16.03] mm` |
| 臂/台 C2 差动 `dz` | 4.65 mm | 15.61 mm |

若把 `arm_mount_xy_m` 改成可见坐标 `robot_base[:2]`、`table_mount_xy_m` 改成 `worktable_center[:2]`，同样的谱和 seed 下臂/台差动峰值只有 **3.98 mm**——约为当前差动的 1/4。

### 5.2 P0 前旧随机流（seed 17，mechanics 复核值）

旧共享随机流下臂/台差动峰值 **16.66 mm**、RMS 4.50 mm；Panda 根/地板错位 RMS 2.93 mm、峰值 11.88 mm（按名义基座高度计算）。P0 更换随机流后相位变了，但问题结构和量级完全保留。

### 5.3 现有仿真探针证据

`out/p0_5substep_spectral_probe.json`（official 1000 Hz × 5，`--episode-s 1 --vibration spectral --seed 17`）：

| 指标 | 值 |
|---|---|
| `max_penetration_mm` | **0.665** |
| `max_penetration_pair` | `robot_link<->platen` |
| `max_penetration_t` | 1.000 s |
| `max_wrist_force_n` | **102.4 N** |
| `max_wrist_force_phase` | `settle` |
| `contact_pair_at_max_wrist_force` | `robot_link<->platen` |
| `penetration_frames_over_0p5mm` | 0.0030 |
| official 穿透门槛 | 0.3 mm |

该回合控制器还在 `settle` 阶段、尚未开始任何抓取动作，穿透和腕力峰值已经全部来自基座与地板的伪碰撞。这个探针把坐标矛盾从“几何推导”变成了“可复现的评分失败”。

---

## 6. 该矛盾在当前项目中的具体体现

### 6.1 穿透指标被伪接触污染

`penetration_probe()` 会把最深穿透记录为一级指标。现在最深的不是工件↔桌面、指尖↔工件，而是 `panda_link0 ↔ VibrationFloor`。因此 `max_penetration_mm` 不再度量抓取/放置质量，0.3 mm 评分门槛变成了“基座有没有嵌进地板”的门槛。

### 6.2 settle 阶段出现 102 N 腕力峰值

Panda 根是 `fix_root_link=False` 的浮动根，但 root pose 每个 1 ms 外层步被覆写。单子步内基座与地板的深嵌会产生巨大接触冲量，并沿关节链传到腕部。探针中 `max_wrist_force_phase=settle`、接触对 `robot_link<->platen`，说明力峰值与任务动作无关。

### 6.3 桌腿/工作台的视觉与物理错位

桌腿、目标盒被锚定到 `worktable_local` 并按 `table_q` 平移，可见地板却按 `q` 平移。虽然桌腿与地板都是 kinematic 刚体、不产生接触冲量，但视觉上桌腿底会相对地板悬空最多约 16 mm 或嵌入约 2.7 mm。目标盒、工件影子（shadow follower）继承同一差动。

### 6.4 可见画面本身就不自洽

录制视频中，同一块可见地板上：

- Panda 基座时而下陷、时而悬空；
- 桌腿与桌面看似一体，却与地板相对滑动；
- Stewart 12 段腿仍按地板中心姿态渲染。

对不读 metrics 的观众来说，这也是视觉可信度问题。

### 6.5 official 评分当前不可用

因为探针已经超过 0.3 mm 穿透门槛，任何沿用当前 C2 坐标的 official 结果都不能诚实报告为通过。P0 提高了子步数、修好了安全门和 RNG，但子步数不能消除这个几何矛盾——0.665 mm 的伪穿透就是在 ×5 子步下实测到的。

### 6.6 安全门估算被同一个坐标混淆干扰

新的 `validate_impulsive_timestep()` 用 `support_mount_radius_m = hypot(0.75, 0.45) = 0.8746 m` 估算支撑点速度。这个半径对应的是**虚拟测点**，而真实碰撞发生在可见位置 `(-0.47, 0)` / `(0.18, 0)` 与地板的相对运动上。门限能控制“虚拟测点的子步行程”，却控制不了“可见基座与地板的几何错位”。

### 6.7 C2_CLITE 实验模式同样继承该矛盾

`_write_clite_drivers()` 把中心运动写给 platform driver、把 `table_q` 写给 worktable driver；可见坐标与虚拟测点分离的问题没有改变。C2_CLITE 让平台/工作台变成动态刚体后，这个差动还会通过 WELD equality constraint 变成真实内力——项目文档中已记录收紧 solref 时出现 45 kN 指力/4.86 mm 穿透，坐标不一致是重要的潜在来源之一。

### 6.8 失败原因不可解释

`settle` 阶段就可能产生伪接触，后续 `descend_table_contact`、`grasp_contact_lost` 等失败原因也可能被基座/地板碰撞间接触发。评分者无法区分“任务失败”与“几何缺陷”。

---

## 7. 为什么这个缺陷此前看起来“可控”

- 旧随机流 seed 17 的相位恰好没有在 1 s 官方探针内把 `robot_link<->platen` 推成最深接触；第四轮 1 s 探针的最深接触是工件↔桌面 0.222 mm。
- 之前的 1.0 mm contact margin 声明、240 Hz 训练档被拒绝、1000 Hz × 4 档通过，都与“位移门限”相关；而坐标矛盾造成的错位是随相位变化的毫米级现象，容易被解释成接触柔度或离散误差。
- P0 更换为 `[seed, env_id, axis_index]` 独立随机流后，seed 17 的相位序列改变，新的 1 s × 5 探针直接暴露了该接触对。

这说明该问题是**建模错误**，不是数值稳定性问题；继续调子步、margin 或 RNG 只能改变其表现形式，不能消除它。

---

## 8. 修复方向与边界

### 8.1 简单刚性方案：测点改成可见坐标

```python
arm_mount_xy_m   = (-0.47, 0.0)
table_mount_xy_m = (0.18, 0.0)
```

- 优点：改动最小，地板/臂/台立即成为严格刚体。
- 代价：臂/台差动峰值从约 15.6 mm 降到约 4.0 mm，难度显著下降；后续要恢复难度只能重新设计谱型。

### 8.2 保难度的物理可行方案：双支撑板 / 非刚性地板

保留虚拟测点 `±(0.75, 0.45)`，把可见单板拆成两块独立驱动板（或保留可见单板 + 两块不可见碰撞代理板）：

- 臂侧板与 Panda 用同一条 `arm_q` 驱动；
- 台侧板与工作台/桌腿/目标盒用同一条 `table_q` 驱动；
- 中间公共地板可保留为视觉层，或加一块中心驱动的全尺寸不可见碰撞板承接掉落工件。

- 优点：差动输入与难度完全保留；每组“支撑板 + 其上的资产”成为严格局部刚体；伪接触消失；任务可达性不受影响。
- 代价：接触拓扑变化（29 个 MJWarp geometry / 348 candidate pairs 不变式需要重测更新）；若保留可见单板，画面里既有的基座/桌腿相对地板错位仍可见，只是不再产生接触力。

### 8.3 其他被排除的方案

| 方案 | 排除原因 |
|---|---|
| 把 Panda/工作台真摆到 ±(0.75, 0.45) 测点 | 两测点相距 1.75 m，Panda 工作半径约 0.855 m，任务不可达 |
| 只把地板画大 | 不改变“虚拟测点运动 + 可见坐标平移”的叠加错误 |
| 继续提高子步数 | ×5 探针已证明伪穿透仍在，离散误差不是根因 |
| 增大 dynamic clearance | 腿/基座错位峰值约 16–20 mm，且只是掩盖指标，不恢复刚体一致性 |
| 修改上游私有 `contype/conaffinity` 或 `nxn_geom_pair_filtered` | 依赖未公开接口，脆弱且违背项目对上游私有接口的谨慎原则 |

---

## 9. 复现与验证

### 9.1 现有仿真探针

```bash
./run.sh --physics-profile official --episode-s 1 \
  --vibration spectral --seed 17 \
  --metrics-output out/p0_5substep_spectral_probe.json
```

预期看到 `[PHYSICS] ... substeps=5 effective_hz=5000 estimated_peak_substep=0.283mm`，
并在 metrics 中出现 `max_penetration_pair="robot_link<->platen"`、`max_penetration_mm > 0.3`。

### 9.2 数值复算口径

不启动仿真即可复算第 5 节数据：按 `vibration.py` 的 `[seed, env_id, axis_index]` 生成默认频带谱线，按 `mounting.py` 的 `t + R r - r` 计算 `arm_q/table_q`，再按 `task._support_state()` 的位置公式计算：

- Panda 根与可见地板顶面在 `(-0.47, 0)` 处的差；
- FL 桌腿底与可见地板顶面在 `(0.18-0.27, 0-0.245)` 处的差；
- `arm_q[:,2] - table_q[:,2]`。

三个公式直接来自 `_write_supports()` 的资产清单，不需要 Isaac 运行时。

---

## 10. 结论

当前 C2 实现的核心矛盾可以压缩成一句话：

> **激励侧按相距 1.75 m 的两个虚拟测点计算差动，几何侧却把机械臂和工作台摆在相距 0.65 m 的可见位置上，并且用同一块可见刚性地板连接它们。**

任何评分、穿透、力峰值指标，在解决这个坐标一致性问题之前，都混有 `robot_link<->platen` 伪碰撞的成分。修复优先级应高于谱型重定义、重力开关等建模改进，因为它是 official 档能否诚实评分的阻塞项。
