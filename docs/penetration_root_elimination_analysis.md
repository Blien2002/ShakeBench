# 穿模问题根因分析与根绝方案（完整调研文档）

更新日期：2026-08-17  
项目：`ViBench`  
后端：Isaac Lab 3.0 + Newton 1.2.1 / MJWarp（`use_mujoco_contacts=True`）  
文档性质：调研与方案文档。本文只记录分析、实验、结论和待实施方案，**未修改任何项目源码**。

---

## 1. 摘要

当前项目针对“穿模 / 穿透”问题的处理方式，是主动限制振动幅值与频率（`max_substep_displacement_m` 启动门），并通过全局较硬的接触响应 `solref` 把残余穿透压到视觉上不易察觉的程度。该做法没有解决两个真正的物理/数值根因：

1. **支撑体以“传送”方式运动，而非以求解器子步积分**：
   `VibrationBenchmarkTask.step()` 在每个外层物理步开始前，直接把振动地板、Panda 浮动根、工作台、桌腿、目标盒的 root pose/velocity 覆写进仿真。Newton 虽然每个外层步内部执行 4 个 solver substep，但 kinematic 支撑体在这 4 个 substep 内保持不动。因此碰撞检测看到的是“跳完之后的位形”，隧穿尺度等于 **每个外层步的位移**，而不是项目启动门中宣称的“每个有效子步的位移”。

2. **请求的接触 margin 被 MuJoCo 接触路径清零**：
   Newton `SolverMuJoCo` 因上游 issue #2106（NATIVECCD/MULTICCD）在 MJCF 编译期把 `geom_margin` 清零。项目诚实记录了 `nativeccd_margin_honored=false`，但并未恢复 margin。没有 margin 意味着接触只有在表面**已经重叠**后才生成，第一子步就会面对完整的外层阶跃重叠量。

隔离物理模型实验证明了两条可行根治路径：

- **方案 A（接触层根治）**：模型编译完成后，在运行时把 `mjw_model.geom_margin` 写回请求值。官方档真实 3.5σ 外层阶跃（约 1.056 mm）下，几何穿透从 0.827 mm 降到 **0.000 mm**。
- **方案 B（运动注入根治）**：把 `_write_supports()` 的更新频率从外层 1000 Hz 提高到 solver 子步 4000 Hz。这样训练档默认全谱（外层阶跃约 4.4 mm）变为每次支撑更新约 1.15 mm，在恢复 margin 后几何穿透同样降到 **0.000 mm**。
- **方案 C（物理结构根治）**：废除 kinematic 位姿覆写，把支撑体改为动态刚体并用执行器/约束驱动振动轨迹，使支撑运动与接触约束在求解器中联合求解。这是最严格的物理方案，但改动和标定成本最高。

随后在 **真实 ViBench 场景** 中完成 5 组小实验（见第 5A 节），结果推翻了隔离模型的部分结论：

- **简单方案 A 不成立**：只恢复 margin 会产生接触前 speculative 力，真实场景腕力峰值从 36.8 N 暴涨到 21.6 kN，工件被反重力悬空；margin+gap 组合在 box/mesh 混合场景中或失稳、或完全无接触。
- **方案 B 部分成立但有副作用**：在公平的“1000 Hz×4 子步”对照中，支撑逐子步更新把工件↔桌面最大穿透从 0.307 mm 降到 0.161 mm，但暴露了右指↔桌面 1.279 mm 的新碰撞（settle 位姿下连续差动运动会真实扫到桌面，原 1 kHz 阶梯近似恰好跳过该碰撞）。
- **按当前参数粗暴切换方案 D 不成立**：`use_mujoco_contacts=False` 后最大穿透升至 4.302 mm，0.5 mm 以上帧占比 19.2%。

修订后的推荐路线：**C-lite（mocap + WELD 动态支撑）已通过真实子集探针，成为第一候选**；方案 B 作为备选，但必须先修复 settle/手指净空与 C2 差动路径上的几何碰撞；方案 A/D 不能按“运行时填数组”或“换管线”的简单方式落地，需要与上游 Newton 的 margin/gap 语义严格对齐后才可重启。完整结论见 `docs/penetration_route_final.md`。

---

## 2. 当前实现行为回顾

### 2.1 每步调用链

`src/vibench/task.py` 的 `VibrationBenchmarkTask.step()`：

```python
def step(self, arm_target, finger_target):
    self._vibration_q, self._vibration_qd, self._vibration_qdd = self.vibration.sample(self.time_s)
    self._write_supports()          # ① 覆写所有 kinematic 支撑体 root state
    self._update_grasp_assist()
    self.robot.set_joint_position_target_index(...)
    self.scene.write_data_to_sim()
    self.sim.step()                 # ② Newton 内部执行 num_substeps=4 个 solver substep
    self.scene.update(self.cfg.dt)
    self.time_s += self.cfg.dt
    self._update_penetration_metrics()
```

关键事实：`_write_supports()` 每个外层步只调用一次；`sim.step()` 内部的 4 个子步不再更新任何支撑体位姿。

### 2.2 支撑体运动映射

`src/vibench/task.py` 的 `_write_supports()` 对以下对象直接写入 root pose/velocity：

- `platform`（振动地板，kinematic）
- `robot`（Panda 浮动根，`fix_root_link=False`，kinematic 驱动）
- `worktable`（kinematic）
- `table_leg_fl/fr/rl/rr`（kinematic）
- `target`（目标盒，kinematic）

写入方式为 `write_root_pose_to_sim_index()` + `write_root_velocity_to_sim_index()`。这是位置“覆写 / 传送”，不是通过力、执行器或约束让求解器积分到达目标状态。

### 2.3 当前防穿透手段

`src/vibench/vibration.py` 的 `validate_impulsive_timestep()`：

```python
displacement = estimated_peak_velocity_m_s(cfg, mount_radius_m) / (physics_hz * substeps)
if cfg.mode != "off" and displacement > limit_m:
    raise ValueError(...)
```

`src/vibench/config.py`：

```python
# Official 1000 Hz spectral profile is 0.264 mm at the conservative
# 3.5-sigma velocity estimate; 0.3 mm admits it while rejecting the
# 1.10 mm training profile that produced multi-millimetre tunnelling.
max_substep_displacement_m: float = 0.0003
```

`src/vibench/diagnostics.py` 的 `configure_mujoco_contact_solref()` 在运行时统一覆写 `mjw_model.geom_solref`：

- official: `(0.00060 s, 1.0)`
- training: `(0.0025 s, 1.0)`

同时返回：

```json
"nativeccd_margin_honored": false
```

即当前明确承认 margin 未生效。

### 2.4 语义错误：安全门中的“子步”并不是支撑运动的子步

对官方默认谱：

```text
estimated_peak_substep_displacement = 0.264 mm
有效子步频率 = 1000 Hz × 4 = 4000 Hz
```

由 `displacement = peak_velocity / 4000` 反推：

```text
3.5σ 峰值速度 ≈ 1.056 m/s
```

但支撑体每外层步只更新一次，因此支撑体实际每步跳跃：

```text
official: 1.056 m/s × 1/1000 s ≈ 1.056 mm/外层步
training: 1.056 m/s × 1/240 s  ≈ 4.4 mm/外层步
```

结论：当前安全门计算的是“动态物体在每个 solver 子步内的积分增量”，**不是**“kinematic 支撑体每次覆写的位移”。碰撞检测面对的是 1.056 mm（official）或 4.4 mm（training）的位形跳变，且此时 margin 已被清零，隧穿因此无法从接触检测层面被阻止。

---

## 3. 根因一：接触 margin 被上游清零

### 3.1 证据链

1. `src/vibench/scene.py` 中所有任务碰撞体都请求了 `contact_offset=cfg.contact_margin_m`、`rest_offset=cfg.contact_margin_m`（默认 1 mm）。
2. Isaac Lab 的 Newton 绑定注释（`isaaclab/envs/mdp/events.py` 约 831-832 行）明确：
   - `rest_offset -> shape_margin`（Newton margin）
   - `contact_offset -> shape_gap`（Newton gap = contact_offset - margin）
3. Newton `SolverMuJoCo` 上游源码（`.venv/lib/python3.12/site-packages/newton/_src/solvers/mujoco/solver_mujoco.py`）在模型转换时：
   - 检测到 box 几何时设置 `self._zero_margins_for_native_ccd = True`；
   - 编译 MJCF 时把 `geom/pair` 的 `margin` 清零并发出警告；
   - 原因：`mujoco_warp.put_model()` 在 NATIVECCD 开启时拒绝非零 margin（Newton #2106）。
4. 上游回归测试 `newton/tests/test_mujoco_margin_zeroing.py` 明确断言：
   - `use_mujoco_contacts=True` 时 `mjw_model.geom_margin` 必须为零；
   - `notify_model_changed(SHAPE_PROPERTIES)` 后 margin 仍保持为零。
5. 项目历史日志（`out/third_round_smoke.log` 等）中可看到大量：

   ```text
   UserWarning: Geom ...: authored margin=... zeroed for NATIVECCD/MULTICCD
   compatibility (#2106).
   ```

### 3.2 对穿透的直接影响

margin=0 时，MuJoCo 只在两几何表面**实际重叠**后才生成接触。由于支撑体是传送式运动，第一个接触生成时已经存在最大到整段外层位移的初始重叠，随后只能靠 `solref` 弹开，表现为每个接触相位都存在可测穿透。

---

## 4. 根因二：支撑体位姿在子步间不连续

### 4.1 代码证据

- `src/vibench/task.py`：`_write_supports()` 在 `sim.step()` 之前调用一次（见 2.1）。
- Isaac Lab Newton Manager（`isaaclab_newton/physics/newton_manager.py`）的 `step()` 流程为：

  ```text
  写关节目标 -> solver.step × num_substeps -> sensors.update
  ```

  没有在 `_run_solver_substeps()` 的循环内提供任何“重新写入 kinematic root state”的回调入口。

- 因此，即使 `num_substeps=4` 把积分误差减小 4 倍，**支撑体轨迹仍然是以 `cfg.dt` 为周期的阶梯函数**，而工件轨迹是以 `cfg.dt/4` 为周期的积分轨迹。

### 4.2 与外层阶跃的关系

| 档位 | 外层频率 | 支撑更新周期 | 3.5σ 峰值速度 | 支撑体每次阶跃 |
|---|---:|---:|---:|---:|
| official | 1000 Hz | 1.0 ms | ≈1.056 m/s | **≈1.056 mm** |
| training | 240 Hz | 4.167 ms | ≈1.056 m/s | **≈4.4 mm** |

项目当前只允许 official 通过 0.3 mm 启动门，training 默认谱被直接拒绝。这说明“减小振动幅度/提高频率”只是绕开了支撑体阶跃过大的事实。

---

## 5. 隔离实验：验证“运行时恢复 margin”与“子步级支撑更新”

### 5.1 实验原则

- 所有实验都在 `$ISAACLAB_ROOT/.venv` 中运行，使用 `newton.ModelBuilder` 构建最小两刚体场景，**未加载、未修改 ViBench 场景代码**。
- 实验场景刻意复刻项目关键结构：
  - 一个 **kinematic** 地板刚体（box，半尺寸 `0.2×0.2×0.01 m`，质量 1 kg，`is_kinematic=True`）；
  - 一个 **dynamic** 工件刚体（box，半尺寸 `0.02×0.02×0.02 m`，质量 0.1 kg）；
  - 初始表面间隙 0.5 mm，重力 `(0,0,-9.81)`；
  - 地板 qpos/qvel 在“外层步”直接覆写，与项目 `write_root_pose_to_sim_index` 语义一致；
  - 外层步 1000 Hz（official）或 240 Hz（training），每个外层步内 solver 以 4 个 0.25 ms 子步积分；
  - 接触参数使用项目同款 `iterations=80, ls_iterations=24, solver=newton, integrator=implicitfast`。
- 实验中穿透深度统一读取 `solver.mjw_data.contact.dist`，取 `max(0, -dist)`。
- 记录口径已确认：恢复 margin 后 `contact.dist` 仍是**真实几何间隙/穿透**，不是被 margin 偏置的值（见 5.3）。

### 5.2 实验过程中的两次脚本错误修正

为保证记录完整，以下错误在实验过程中被发现并修正：

1. **第一次运行**：对 `contact.dist.numpy()` 取 `[0]` 后对 numpy 标量调用 `len()`，脚本报错。已改为 `np.asarray(...).reshape(-1)`。
2. **第二次运行**：每个外层步把 `state.joint_q` 整体清零后只写地板 7 个坐标，导致动态工件的 7 个坐标也被清零，工件被传送回原点、产生虚假的 29.96 mm 穿透，两个对照分支因此得到相同数值。修正为：**只覆写地板对应索引**，工件坐标由 solver 保留。

修正后的数据见下文。

### 5.3 实验一：编译后恢复 `geom_margin` 是否合法

**目的**：确认上游断言“margin 必须为零”只约束模型编译期；运行时写回 `mjw_model.geom_margin` 是否可被 broadphase 正确使用且不会导致 step 崩溃。

**过程**：
- 构建两个 margin=1 mm 的 box 几何；
- `SolverMuJoCo(use_mujoco_contacts=True)` 构建后，`geom_margin` 初始为 `[[0.0, 0.0]]`；
- 运行时直接 `solver.mjw_model.geom_margin.fill_(0.001)`；
- 将两个 box 表面间隙分别设置为 2.5 mm、1.5 mm、0.5 mm 并执行 solver step；
- 读取 `contact.dist`。

**结果**：

| 真实表面间隙 | 成对 margin 和 | 是否生成接触 | 实测 `contact.dist` |
|---:|---:|---|---:|
| 2.5 mm | 2.0 mm | 否 | 无接触（接触缓冲为零） |
| 1.5 mm | 2.0 mm | 是 | **1.5000 mm**（真实几何间隙） |
| 0.5 mm | 2.0 mm | 是 | **0.5000 mm**（真实几何间隙） |

**结论**：
- `put_model()` 编译成功后，运行时写回 `mjw_model.geom_margin` 是可行的，solver 步骤正常；
- broadphase 使用运行时 `geom_margin` 数组（与 `mujoco_warp/_src/collision_driver.py` 中 `_broadphase_filter(geom_margin, ...)` 的实现一致）；
- `contact.dist` 保持真实几何距离语义，现有 `penetration_probe()` 的 `depth = max(0, -dist)` 口径无需修改。

### 5.4 实验二：official 档 0.264 mm 外层阶跃（项目启动门估算值）

**条件**：
- 外层 1000 Hz × 4 子步；
- 正弦 `amplitude=1.5 mm, f=28 Hz`，峰值外层阶跃 `1.5e-3 × 2π × 28 / 1000 ≈ 0.264 mm`；
- 对照分支：margin=0 vs margin=1 mm（运行时恢复）；
- 两分支均使用 official `geom_solref=(0.00060, 1.0)`；
- 0.5 s 静置后振动 1.0 s，统计所有子步接触。

**结果**：

| 分支 | 最大几何穿透 | 最小接触距离 |
|---|---:|---:|
| margin=0（现状等价） | **0.2728 mm** | -0.2728 mm |
| margin=1 mm + 官方 solref | **0.0000 mm** | 0.0000 mm |

### 5.5 实验三：1000 Hz 外层下 1.106 mm 阶跃

**条件**：
- 外层仍为 1000 Hz × 4 子步；
- 正弦 `amplitude=5.5 mm, f=32 Hz`，峰值外层阶跃 ≈1.106 mm；
- margin=0 vs margin=1 mm，均用 official `solref=(0.00060, 1.0)`。

**结果**：

| 分支 | 最大几何穿透 | 最小接触距离 |
|---|---:|---:|
| margin=0 | **1.1087 mm** | -1.1087 mm |
| margin=1 mm + 官方 solref | **0.0000 mm** | 0.0000 mm |

### 5.6 实验四：official 档真实 3.5σ 外层阶跃 ≈1.056 mm

**条件**：
- 由 `estimated_peak_substep_displacement=0.264 mm × 4000 Hz` 反推峰值速度 ≈1.056 m/s；
- 正弦 `amplitude=4 mm, f=42 Hz`，峰值外层阶跃 `0.004 × 2π × 42 / 1000 ≈ 1.056 mm`；
- 外层 1000 Hz × 4 子步，official `solref=(0.00060, 1.0)`。

**结果**：

| 分支 | 最大几何穿透 | 最小接触距离 |
|---|---:|---:|
| margin=0（现状等价） | **0.8271 mm** | -0.8271 mm |
| margin=1 mm + 官方 solref | **0.0000 mm** | 0.0000 mm |

**结论**：在 official 档真实峰值阶跃下，仅恢复 margin 就能把传送导致的几何穿透压到实验可测的零值。

### 5.7 实验五：training 档真实外层阶跃 ≈4.6 mm，margin 单独不成立

**条件**：
- 外层 240 Hz × 4 子步；
- 正弦 `amplitude=5.5 mm, f=32 Hz`，峰值外层阶跃 `0.0055 × 2π × 32 / 240 ≈ 4.61 mm`；
- 对照 margin=0/1 mm，`solref` 分别取 training `(0.0025, 1.0)` 与 official `(0.00060, 1.0)`。

**结果**：

| margin | solref 时间常数 | 最大几何穿透 |
|---:|---:|---:|
| 0 mm | 0.0025 s（training） | 1.9139 mm |
| 1 mm | 0.0025 s（training） | **45.3865 mm（失稳）** |
| 0 mm | 0.0006 s（official） | 1.4191 mm |
| 1 mm | 0.0006 s（official） | 1.8606 mm |

**结论**：
- 当外层阶跃（4.6 mm）大于成对 margin 包络（2.0 mm）时，margin 不再保证安全；
- 在软接触 `solref=0.0025 s` 下，speculative contact 反而可能向系统注入能量，导致工件被弹起后二次穿透，出现 45 mm 级灾难性失稳；
- 因此**方案 A 单独不足以根治 training 档**，必须与方案 B 组合使用。

### 5.8 实验六：支撑体按 4000 Hz 子步更新 + margin（方案 A+B 组合）

**条件**：
- 用与实验五相同的 32 Hz / 5.5 mm 波形（若按 240 Hz 外层覆写则阶跃为 4.6 mm）；
- 但支撑体每 0.25 ms 更新一次（等价于方案 B：`_write_supports` 与 solver 子步同频）；
- 每个 0.25 ms 直接写 kinematic 地板 qpos/qvel，随后 solver 执行 1 步；
- 对照 margin=0/1 mm，`solref=0.0025 s` 与 `0.0006 s`。

**结果**：

| margin | solref 时间常数 | 最大几何穿透 |
|---:|---:|---:|
| 0 mm | 0.0025 s | 1.3869 mm |
| 0 mm | 0.0006 s | 0.2844 mm |
| **1 mm** | 0.0025 s | **0.0000 mm** |
| **1 mm** | 0.0006 s | **0.0000 mm** |

**结论**：
- 支撑更新频率提升到 4000 Hz 后，training 波形的单次支撑运动从 4.6 mm 降到约 1.15 mm，处于 2 mm margin 包络内；
- “margin 恢复 + 子步级支撑更新”组合可以在官方与训练档波形下把几何穿透压到零。

### 5.9 实验汇总

| 工况 | margin=0（现状等价） | 运行时恢复 1 mm margin + solref | 结论 |
|---|---:|---:|---|
| official 0.264 mm 外层阶跃 | 0.273 mm | **0.000 mm** | A 可根治 |
| 1000 Hz 外层 1.106 mm 阶跃 | 1.109 mm | **0.000 mm** | A 可根治 |
| official 真实 3.5σ ≈1.056 mm 阶跃 | 0.827 mm | **0.000 mm** | A 可根治 |
| training 真实 ≈4.6 mm 阶跃 + training solref | 1.914 mm | 45.39 mm（失稳） | 仅 A 不可用 |
| training 波形 + 支撑按 4 kHz 更新 | 1.387 mm | **0.000 mm** | A+B 可根治 |

---

## 5A. 真实 ViBench 场景验证实验（2026-08-18）

### 5A.1 代码备份

实验前对当前代码做了完整备份：

- 文件：`~/Desktop/ViBench_backups/ViBench_code_backup_20260818_105442.tar.gz`（12 MB）
- SHA-256：`1bca3cf66a357a3822c6bba26ebac27e0c9800ce4df55bbf97a9a0e4cc5d7498`
- 内容：完整仓库（含 `.git`），排除 `out/`、`__pycache__/`、`.pytest_cache/`
- 已解包抽查 `src/` 与 `CLAUDE.md`，与工作区一致

实验脚本均放在 `/tmp/`，**未修改 `src/` 或任何仓库源码**；结果 JSON 已固化到 `out/penetration_experiments_20260818/`（`out/` 被 `.gitignore` 忽略）。

### 5A.2 实验方法

- 真实场景：Panda + 工作台 + sugar_box@0.75 + 目标盒，`386/29/348` 拓扑；
- official 物理档：1000 Hz × 4 子步，`solref=(0.0006 s, 1.0)`；
- 激励：默认 6 轴谱，seed=17，scale=1.0，`grasp_assist=False`；
- 1 秒回合；除特别说明外，控制器处于 settle 阶段（保持复位关节，不发生任务运动），以便单独考察接触物理；
- margin/gap 修改均通过 `NewtonManager._solver.mjw_model.geom_margin/geom_gap` 在模型编译后运行时写入；
- 穿透量统一使用现有 `penetration_probe()` 的口径 `max(0, -contact.dist)`。

### 5A.3 实验一：只恢复 margin（方案 A 的最简形式）

| 分支 | 最大穿透 | 最大穿透对 | 腕力峰值 |
|---|---:|---|---:|
| margin=0（现状等价） | 0.1668 mm | workpiece↔worktable @0.916 s | 36.78 N |
| margin=1 mm 运行时写入 | 0.2798 mm | workpiece↔worktable @0.001 s | **21 638.1 N** |

**结论**：只写 `geom_margin` 会在工件初始 1 mm 悬浮间隙处就产生 large speculative contact force（2 mm 接触包络 > 1 mm 初始间隙），把工件反重力推开/悬空。真实场景腕力放大约 588 倍，**方案 A 不能按“只填 margin”实施**。

### 5A.4 实验二：margin 与 gap 组合

| 分支 | 最大穿透 | 腕力峰值 | 观察 |
|---|---:|---:|---|
| baseline（margin=0, gap=0） | 0.1665 mm | 36.78 N | robot_link↔platen @0.723 s |
| margin=1 mm, gap=1 mm | **26.254 mm** | 40.51 N | workpiece↔worktable @0.639 s，box/mesh 路径失稳 |
| margin=1 mm, gap=0.5 mm | 0.0 mm | 0.0 N | 墙钟 222.6 s，疑似物体被悬空、全程无真实接触 |

**结论**：在真实 box/mesh 混合场景中，编译后运行时改写 `geom_gap` 与上游 NATIVECCD 编译模型不一致，结果或失稳、或无接触。**A 方案以当前形式不可行**。

### 5A.5 实验三：支撑逐子步更新（方案 B，公平对照）

在保持 official `1000 Hz × 4` 外层结构不变的前提下，用 `NewtonManager` 的 substep 接口实现两种注入：

- `current`：每外层步 `_write_supports()` 一次，然后 `sim.step()` 跑 4 个子步（完全等价现状）；
- `schemeB`：每外层步的 **每个子步前** 都重采样振动并 `_write_supports()` + `scene.write_data_to_sim()`，然后单步求解。

| 接触对 | current 最大穿透 | schemeB 最大穿透 |
|---|---:|---:|
| workpiece↔worktable | 0.3070 mm | **0.1606 mm** |
| robot_link↔platen | 0.1665 mm | 0.3315 mm |
| robot_link↔worktable（右指↔桌面） | — | **1.2787 mm**（新增，@0.736 s） |

**结论**：
- 方案 B 确实把最主要的“工件↔桌面”支撑隧穿约减半（0.307→0.161 mm），方向正确；
- 但连续差动运动暴露了 settle 位姿下右指与桌面之间的真实几何碰撞（1.279 mm）。原 1 kHz 阶梯近似恰好跳过该碰撞，说明当前基准的一部分“安全”来自时间量化；
- 因此方案 B 不能单独落地，必须先修 settle/手指净空或 C2 差动路径几何。

> 注：早期一版对比脚本曾把 `sim.step()` 拆成 4 次独立调用，且 B 分支漏掉子步间的 `scene.write_data_to_sim()`，结果不可比；本表来自修正后的公平对照，JSON 见 `04_support_cadence_fair_substep_probe.json`。

### 5A.6 实验四：切换 Newton CollisionPipeline（方案 D 的朴素形态）

配置：`use_mujoco_contacts=False` + `NewtonCollisionPipelineCfg(broad_phase="explicit")` + `NewtonShapeCfg(margin=1 mm, gap=1 mm)`。

| 指标 | 结果 |
|---|---:|
| 最大穿透 | **4.3020 mm** |
| 最大穿透对 | workpiece↔worktable @0.701 s |
| >0.5 mm 帧占比 | **19.18%** |
| 腕力峰值 | 177.18 N |
| 拓扑 | 386 / 29 / 348 不变 |

**结论**：在不完全理解 Newton margin/gap 语义的情况下粗暴切换管线并不可行，反而显著恶化。方案 D 需要按 Newton CollisionPipeline 的 `shape_collision_radius/thickness` 语义重新标定 margin/gap，不能沿用 MuJoCo 直觉。

### 5A.7 实验后的修订结论

1. **最小两刚体实验只证明机制存在，不能外推到真实场景**。真实场景是 box/convex-mesh 混合、多接触对、浮动 Panda 与 C2 差动共同作用，margin/gap 的运行时改写与 NATIVECCD 编译模型存在不一致。
2. **方案 B 仍是当前最值得投入的方向**，但其收益是“工件↔桌面”穿透减半，同时会暴露现有几何近碰撞。实施顺序必须改为：
   - 先修 settle 位姿/手指-桌面净空与 C2 差动路径（例如重新定义 settle 关节姿态或提高初始指-桌净空）；
   - 再实现逐子步支撑注入；
   - 以“全接触对最大穿透 ≤ 0.3 mm”而非“全局最大穿透单值”作为验收标准。
3. **方案 A/D 暂停**，待确认 Newton 上游对 margin/gap 的运行时更新语义（或升级到修复 #2106 的 Newton 版本）后重新评估。
4. 当前“降振幅”门仍必须保留，直到方案 B 及其配套几何修复完成并通过全接触对验收。

---

## 6. 根绝方案

### 6.1 方案 A：接触层根治——编译后恢复 MuJoCo `geom_margin`（真实场景验证未通过，暂停）

> 状态更新（2026-08-18）：按 5A.3/5A.4 的真实场景实验，运行时单独改写 `geom_margin` 或同时改写 `geom_gap` 均不可行。本节仅保留原设计供后续对照，**不作为当前推荐方案**。

**原理**

Newton #2106 只禁止在 **MJCF 编译期**给 NATIVECCD 模型设置非零 margin。`mujoco_warp` 的 broadphase/narrowphase 在**运行时**读取 `mjw_model.geom_margin`，因此模型编译完成后直接写回该数组即可恢复接触包络。5.3 与 5.6 实验证明该写法有效、可执行、且 `contact.dist` 口径不变。

**待实施改动（本阶段只做方案，不实施）**

1. `src/vibench/diagnostics.py`
   - 扩展 `configure_mujoco_contact_solref(solref, friction_mu=None, margin_m=None)`：
     - 现有 `geom_solref` 写入之后，执行 `solver.mjw_model.geom_margin.fill_(margin_m)`；
     - 不要随后调用 `notify_model_changed(SHAPE_PROPERTIES)`，否则 margin 会被上游再次清零；
     - 返回结果增加：
       - `"nativeccd_margin_honored": true`
       - `"applied_unique_geom_margin": [...]`
       - `"margin_restoration": "post_compile_runtime_write"`
       - `"upstream_issue": "#2106"`
   - 可选加固：提供 `assert_margin_intact()`，在 episode 开始与 `reset()` 后校验 `geom_margin` 非零；若被后续模型变更清零，则重写并记录事件。

2. `src/vibench/cli.py`
   - 调用点传入 `margin_m=cfg.contact_margin_m`；
   - 更新 JSON 的 `contact_response` 字段与打印内容。

3. `src/vibench/config.py`
   - 增加“接触包络覆盖校验”：

     ```text
     2 × contact_margin_m ≥ k × estimated_outer_step_displacement
     estimated_outer_step_displacement = estimated_peak_velocity_m_s / physics_hz
     k 建议取 1.25–1.5
     ```

   - 修正注释：现安全门计算的不是“支撑体每个子步的位移”，而是动态物体的积分增量；支撑体真实阶跃应按 `peak_velocity / physics_hz` 计算。
   - 保留原启动门作为第二道保险，但不再把它当作防穿模的唯一机制。

4. `src/vibench/diagnostics.py` 的 `penetration_probe()`
   - 依据 5.3 实验，`contact.dist` 在 margin 恢复后仍为真实几何距离，现有 `max(0, -dist)` 计算保持不变；
   - 在真实场景集成验证时再次确认这一口径。

5. 文档同步
   - `docs/reports/current_implementation.md`
   - `docs/runtime_vibration_issue_resolution.md`
   - `configs/scenarios.yaml` 的 `contact_policy`
   - `docs/fourth_round_validation.md` 相关结论

**验证标准**

- 官方 1 s 谱 probe（seed=17）：期望最大穿透从历史 0.222 mm 降到接近 0；
- 官方 16 s 完整谱回合：`max_penetration_mm < 0.05 mm`，`penetration_frames_over_0p5mm == 0`；
- `max_wrist_force_n` 不显著劣化；
- 接触拓扑保持 `386 / 29 / 348`；
- `nconmax=180` 不溢出：margin 会激活更多“接近但未接触”的 speculative contacts，需监控 `active_contact_count`；
- JSON 记录 `nativeccd_margin_honored=true`。

**风险**

- 依赖 `mjw_model.geom_margin` 的运行时行为，属于对上游 #2106 的 workaround；
- 上游升级后行为可能变化，需要回归测试与版本锁定；
- margin 恢复后接触提前发生，可能轻微改变接触力时序，必须用完整官方回合对比力峰值；
- 训练档默认全谱（4.6 mm 外层阶跃 > 2 mm 包络）**仍不能**仅靠方案 A 放行。

### 6.2 方案 B：运动注入根治——支撑体按 solver 子步更新（方向正确，需先修几何碰撞）

> 状态更新（2026-08-18）：5A.5 的真实场景公平对照显示，B 把 workpiece↔worktable 穿透从 0.307 mm 降至 0.161 mm，但新增右指↔桌面 1.279 mm 碰撞。**必须先修 settle/手指净空，再实施本节内容。**

**原理**

把 `_write_supports()` 的调用周期从外层 `cfg.dt` 缩短到 `cfg.dt / solver_substeps`，支撑轨迹从 1000 Hz 阶梯变为 4000 Hz 阶梯。此时：
- official 真实子步运动 ≈0.264 mm；
- training 真实子步运动 ≈1.15 mm；
- 两者都小于 2 mm margin 包络，实验六已证明该组合可把几何穿透压到零。

**待实施改动（方案阶段）**

1. `src/vibench/config.py`
   - 保留 `dt` 作为控制器/观测节拍（1000 Hz）；
   - 新增 `support_update_decimation=4`（或等价配置 `support_dt=1/4000`）；
   - `SimulationCfg.dt` 改为 `support_dt`，`NewtonCfg.num_substeps=1`；
   - `contact_solref` 的时间常数校验改为基于 `support_dt`：
     - official `0.00060 s ≥ 2 × 0.00025 s` 仍通过；
     - training `0.0025 s` 也通过。

2. `src/vibench/task.py`
   - 拆分 `physics_tick(support_time_s)`：只负责振动采样、`_write_supports()`、`sim.step()`；
   - 保留现有 `step()` 作为 4 个 `physics_tick` 的聚合接口，或改为 `cli.py` 显式 4:1 tick 循环；
   - `time_s`、穿透采样、指标与录制仍按 1000 Hz 控制器节拍；
   - 关节目标每个控制器节拍设置一次，物理 tick 之间保持。

3. `src/vibench/cli.py`
   - 主循环改为 tick 驱动：
     - 每 4 个物理 tick 调用一次 `controller.command(obs)`、`scene.update()`、`observation()`；
     - 每个 tick 都做 `_write_supports()` + `sim.step()`；
   - 录制帧率与现有 30 FPS 对齐，不改变视频输出节奏。

4. `src/vibench/vibration.py`
   - `validate_impulsive_timestep()` 的语义修正：比较对象变为 `peak_velocity × support_dt`；
   - 新增/修改日志字段，明确记录 `support_update_hz`、`outer_peak_displacement_mm` 与 `per_tick_displacement_mm`。

5. 场景策略
   - `configs/scenarios.yaml` 增加显式试验场景（如 `full_spectrum_safe_training`）；
   - 通过真实场景验证后再决定是否恢复训练档默认全谱；**不无声放宽**现有安全门。

**验证标准**

- official 与 training 默认全谱、seed=17：最大穿透接近 0，超过 0.5 mm 帧占比为 0；
- 新增诊断：相邻两个物理 tick 之间支撑体实测位移 ≤ `estimated_peak_velocity × support_dt`；
- 力峰值与抓取行为相对方案 A-only 基线不倒退；
- 壁钟吞吐评估：目标为当前 official 档的 1.5–2 倍以内；若达不到，再评估 NewtonManager 增加 per-substep 回调的上游补丁方案。

**风险**

- 外层 tick 数变为 4 倍，`scene.write_data_to_sim`、传感器更新等固定开销可能放大墙钟时间；
- 需要对 `time_s`、录制、历史缓存、穿透帧计数做一次系统性的节拍梳理，避免指标时间轴错位。

### 6.3 方案 C：物理结构根治——废除 kinematic 位姿覆写，改为动态约束/驱动支撑（高投入，终极方案）

**原理**

只要支撑体是 kinematic + 直接覆写状态，本质上仍是“传送”。最严格的做法是把振动地板/工作台/Panda 根做成**动态刚体**，用执行器或 equality constraint（例如 mocap + weld/connect）跟踪解析振动轨迹。求解器会在每个子步把“支撑驱动”和“工件接触”联合求解，工件可以真实地顶住/推回支撑，支撑也无法瞬间穿入物体。

**实施路线**

1. 先做独立可行性探针，不碰 ViBench 场景：
   - `newton.ModelBuilder` 构造动态自由刚体平台 + 6-DOF 位置/速度跟踪执行器；
   - 在项目级振动速度（约 1.056 m/s 峰值）下测量跟踪误差、穿透与接触力；
   - 对比“动态驱动”与“kinematic 传送”两种注入方式的稳定性。

2. 探针通过后再重构 `src/vibench/scene.py`：
   - `platform` 改为动态刚体并挂 6-DOF 驱动；
   - Panda 根与工作台通过 FIXED/WELD 关节附着到平台，或各自用 mocap+constraint 驱动；
   - 工作台/目标盒/桌腿随同一动态平台运动，保证原 C2 差动测点语义可重建。

3. 作为新 `physics_profile`（例如 `dynamic_support`）并行验证，成熟前不替换 official。

4. 更新架构文档：
   - `CLAUDE.md` 中“机械臂和工作台是同级节点”的架构事实；
   - `docs/reports/current_implementation.md`；
   - 视觉 manifest 父级锚点审计。

**代价与风险**

- 需要标定平台质量/惯量、执行器增益、约束刚度；
- 支撑获得轻微柔度，与“理想刚性振动台”语义有差异，可能影响 benchmark 可比性；
- 接触拓扑可能变化，需要重新执行并记录 `print_contact_snapshot`。

### 6.4 方案 D（备选）：`use_mujoco_contacts=False` + Newton CollisionPipeline

> 状态更新（2026-08-18）：5A.6 的真实场景实验按当前参数切换后最大穿透 4.302 mm，>0.5 mm 帧占比 19.2%。**暂缓，直到按 Newton 自身 margin/gap 语义重新标定后再试。**

Newton #2106 警告中给出的官方替代路径是：关闭 MuJoCo 内部接触生成，改用 Newton 自己的 CollisionPipeline，并通过 `NewtonShapeCfg(margin=...)` 在 Newton 侧保留 margin。

可能的配置形状：

```python
MJWarpSolverCfg(use_mujoco_contacts=False, ...)
NewtonCfg(collision_cfg=NewtonCollisionPipelineCfg(...),
          default_shape_cfg=NewtonShapeCfg(margin=cfg.contact_margin_m, gap=...))
```

**为什么作为备选而非首选**

- 会改变接触生成算法，接触力、摩擦、`contact.dist` 语义都需要重新校核；
- `penetration_probe()`、ContactSensor 读数、`configure_mujoco_contact_solref()` 都需要适配 Newton contacts；
- 验证成本和评分口径风险高于方案 A；
- 仅当方案 A 在上游升级后失效，或需要完全摆脱 #2106 时启用。

---

## 7. 推荐实施顺序

| 阶段 | 内容 | 风险 | 预期收益 |
|---|---|---|---|
| 阶段 0 | 全接触对分级穿透回归（不只记录全局最大值），固化 2026-08-18 实验结果 | 低 | 防止新方案“按下葫芦浮起瓢” |
| 阶段 1 | 修复 settle 位姿/右指-桌面净空与 C2 差动路径几何碰撞 | 中 | 解除方案 B 暴露的真实几何碰撞，且不依赖时间量化 |
| 阶段 2 | 方案 B：4000 Hz 支撑更新 + 4:1 控制解耦 | 中 | workpiece↔worktable 穿透约减半（0.307→0.161 mm 实测） |
| 阶段 3 | 真实场景完整验证：4 种 YCB × 多 seed × 16 s official 无辅助 | 中 | 形成可发布的物理回归证据 |
| 阶段 4 | 方案 C 探针（动态支撑/约束驱动） | 高 | 终极物理保真，视 benchmark 目标决定是否落地 |
| 备选 | 方案 A/D：待上游 Newton margin/gap 语义澄清或 #2106 修复后重启 | 高 | 接触层根治，但当前参数下已验证失败 |

### 7.1 阶段 0 最小验收清单

- [ ] 把 `out/penetration_experiments_20260818/` 纳入回归说明，五组 JSON 可追溯
- [ ] 新增/扩展探针，按 `classify_penetration_pair()` 输出每个接触对的最大穿透、时刻与帧占比
- [ ] 官方 1 s 谱 probe 基线多跑 2–3 次，记录 run-to-run 区间（本日实测 0.167–0.307 mm）

### 7.2 阶段 1 最小验收清单

- [ ] settle 位姿下，连续 C2 差动路径（即子步级支撑注入）不再出现 robot_link↔worktable 或 robot_link↔platen 接触
- [ ] 手指-桌面/手指-platen 净空在完整 6 轴谱峰值下保持 >0
- [ ] 视觉 manifest 与父级锚点审计不受影响

### 7.3 阶段 2 最小验收清单

- [ ] 支撑体每个 solver 子步更新一次，tick 位移在包络内
- [ ] **所有受监控接触对** 的最大穿透 ≤ 0.3 mm，且不出现新接触对
- [ ] 官方与 training 默认全谱的工件↔桌面穿透接近 0
- [ ] 现有 success/failure 判定、录制时间轴、JSON 指标不回归
- [ ] 性能测试：每仿真秒墙钟可接受

---

## 8. 实施时必须遵守的边界

1. **不修改上游 Newton 源码**。方案 A 使用运行时数据数组写回，方案 B 只改 ViBench 自身 tick 结构；如果阶段 2 需要 per-substep 回调，优先提交上游 patch，或在仓库中以显式、带版本断言的适配层实现。
2. **不得用视觉手段掩盖穿透**。穿透探针、视频叠加、JSON 指标保持现状并继续作为第一类指标。
3. **不得恢复 `grasp_assist` 作为官方成绩**，不得放宽 `success`、滑移、接触丢失判据。
4. **训练档仍不计分**，除非后续通过完整验证并显式更新 `configs/scenarios.yaml` 的 scoreable 规则。
5. **所有结论以新生成的 JSON/MP4 为准**；上游 `SIGSEGV`/malloc 崩溃导致的未完成回合不得复用旧产物。

---

## 9. 结论

- 当前“减小振动幅度 + 调硬接触”的路径本质上是在绕开两个根因：**接触 margin 被清零** 与 **kinematic 支撑体传送式运动**。
- **最小两刚体实验证明机制可行**：恢复 margin 或子步级支撑更新能把传送隧穿压到零，但该模型只有 box-box、单接触对，不能外推到真实场景。
- **真实 ViBench 场景实验（2026-08-18）推翻了简单落地方案**：
  - 方案 A（运行时只恢复 margin）造成 21.6 kN 腕力峰值与工件悬空，不可接受；
  - margin+gap 运行时组合在 box/mesh 场景中失稳或无真实接触；
  - 方案 B 的公平对照把 workpiece↔worktable 穿透从 0.307 mm 降到 0.161 mm，但新增右指↔桌面 1.279 mm 碰撞；
  - 朴素切换方案 D 使穿透升至 4.302 mm、超限帧占比 19.2%。
- **修订后的推荐**：C-lite 真实子集探针已通过（穿透 0.9637→0.0128 mm，力 555.5→477.5 N），成为第一候选；下一步扩展到完整场景。方案 B 作为备选并需先修几何；方案 A/D 暂缓，待 Newton 上游 margin/gap 语义澄清或 #2106 修复后重启；方案 C 仍是长期物理保真方向。
- 在阶段 0–2 完成前，**当前降振幅启动门必须保留**；不得以任何实验中间状态替换官方评分路径。
- 本方案解决的是“数值隧穿 / 传送式穿模”。高腕力、无辅助抓取滑移、完整回合成功率属于独立问题，仍需按现有计分完整性规则继续处理。
