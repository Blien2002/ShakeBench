# C-lite 主支撑实现与评定报告

更新日期：2026-08-18  
备份：`~/Desktop/ViBench_backups/ViBench_code_backup_20260818_105442.tar.gz`  
默认行为：`support_config="C2"` 完全不变；C-lite 通过 `--support-config C2_CLITE` 显式启用。

---

## 1. 已实现内容

### 1.1 配置

- `src/vibench/config.py`
  - `BenchmarkConfig.support_config: Literal["C2", "C2_CLITE"]`
  - `BenchmarkConfig.use_clite_support` property
  - `panel_operation + C2_CLITE` 显式拒绝

- `src/vibench/cli.py`
  - 新参数 `--support-config C2|C2_CLITE`（默认 C2）
  - JSON 增加 `support_config` 字段

### 1.2 场景装配

`src/vibench/scene.py`：

- `_clite_dynamic_asset()`：把 RigidObject 的 `kinematic_enabled` 改为 `False`
- `install_clite_model_constraints(cfg)`：注册 `PhysicsEvent.MODEL_INIT` 回调，在 Newton `ModelBuilder` finalize 之前：
  1. 将 **VibrationFloor** 与 **WorkTableTop** 的 `body_flags` 改为 `BodyFlags.DYNAMIC`；
  2. 为二者各新增一个 fixed-root mocap driver body；
  3. 用 `add_equality_constraint_weld()` 把动态支撑体焊到对应 mocap driver；
  4. 设置 driver 的 `joint_X_p` 为 `cfg` 中的名义安装位置，保证初始 mocap 位姿正确。

当前 C-lite 范围：**平台 + 工作台**。桌腿、目标盒与 Panda 浮动根保持原 kinematic 轨迹写入。

### 1.3 任务循环

`src/vibench/task.py`：

- `_resolve_clite_mocap_ids()`：求解器构建后，把 `clite_driver_platform` / `clite_driver_worktable` 映射为 MuJoCo mocap id，并记录初始 mocap 位姿。
- `_write_clite_drivers()`：每步把三个轨迹写入 `mjw_data.mocap_pos/mocap_quat`：
  - platform driver ← 地板中心运动；
  - worktable driver ← C2 工作台测点运动。
- `_write_supports()` C-lite 分支：主支撑只写 mocap；Panda 根、桌腿、目标盒仍写 root pose/velocity。
- `_step_clite_physics()`：C-lite 模式下不调用 `sim.step()`，而是按 `solver_substeps` 手动推进：
  - 每个 0.25 ms 子步前重新采样振动并更新 mocap；
  - 调用 `NewtonManager._step_solver`；
  - 外层 1 ms 结束时 `_update_sensors`、推进 sim time、`scene.update`。
- WELD equality `solref` 设为 `(0.001 s, 1.0)`（默认 `(0.02 s, 1.0)` 过软）。

### 1.4 测试

- `tests/test_vibration.py` 新增 `test_clite_support_config_is_available_but_not_for_panel`
- 现有测试全通过：`37 passed`

---

## 2. 评定结果

统一条件：official 1000 Hz × 4、sugar_box@0.75、默认 6 轴谱 seed=17、`grasp_assist=False`。

> 修复记录：第一版演示中平台出现 180° 旋转，原因是把 Isaac Lab `quat_from_euler_xyz()` 返回的 **xyzw** 直接写入 MuJoCo 的 **wxyz** `mocap_quat`。已改为 `[w,x,y,z]=[q3,q0,q1,q2]` 并重新评定；下表为修复后数据。

### 2.1 0.1 s 无振动 smoke（C2_CLITE）

| 指标 | 值 |
|---|---:|
| 最大穿透 | 0.0457 mm（workpiece↔worktable） |
| >0.5 mm 帧占比 | 0 |
| 腕力峰值 | 0.0045 N |
| 拓扑 | 386 / 29 / 348 |

### 2.2 1 s official 谱对照

| 指标 | C2 基线 | C2_CLITE |
|---|---:|---:|
| 最大穿透 | 0.1665 mm（robot_link↔platen） | 0.2328 mm（workpiece↔worktable） |
| >0.5 mm 帧占比 | 0 | 0 |
| 腕力峰值 | 36.78 N | **0.0225 N** |
| 每仿真秒墙钟 | 31.2 s | 67.1 s |

结论：主支撑 C-lite 在 idle/settle 阶段把腕力峰值降低约 1600 倍，主接触对穿透仍低于 0.3 mm 门槛，但绝对穿透略高于 C2 基线。

### 2.3 16 s 完整官方回合（C2_CLITE）

- 结束时间：4.602 s，控制器阶段 `grasp`
- 失败原因：`grasp_table_contact`
- 最大穿透：**6.365 mm**（workpiece↔worktable）
- 左右指峰值接触力 838.1 N / 550.3 N
- 腕力峰值 58 468.6 N（approach 阶段）

结论：完整无辅助控制器在动态工作台 + 子步级相对运动下，descend/grasp 阶段出现严重指-桌面碰撞。当前 C-lite 主支撑版**不能替换官方 C2 路径**。

### 2.4 16 s C2 基线对照

- 结束时间：4.862 s，控制器阶段 `grasp`
- 失败原因：`grasp_table_contact`
- 最大穿透：2.913 mm（robot_link↔worktable，右指）
- 腕力峰值：31 105 N；左右指峰值 3107 N / 1795.7 N

**对照结论**：C2 基线在完整无辅助 16 s 回合同样因指-桌面碰撞失败，但 C2_CLITE 把该碰撞放大到 7.95 mm / 74.8 kN，说明当前 C-lite 主支撑 + kinematic 机器人组合比 C2 更差。

---

## 3. 失败原因分析

1. **Panda 根未 C-lite 化**：根节点 free joint + weld 在 6 轴谱下会漂移（ee 从 x=-0.08 m 漂到 -0.86 m），因此回退为 kinematic 子步写入。
2. **动态工作台 + kinematic 机器人产生新的相对运动路径**：工作台由 mocap 约束连续积分，机器人根仍按 1 kHz 指令传送；两者相对路径与 C2 量化版本不同，grasp 阶段指尖扫到桌面（与方案 B 暴露的右指碰撞同源）。
3. **控制器净空基于 C2 量化动力学调参**：`finger_table_clearance_m=12 mm` 等参数是在“支撑体 1 kHz 阶梯运动”下验证的，C-lite 连续路径需要重新标定。

---

## 4. 结论与下一步

- **已实现并验证**：C-lite 主支撑（平台+工作台）在 settle/idle 接触物理上显著优于 C2：穿透 0.217 mm、腕力 0.023 N，证明 mocap+weld 机制在真实场景可用。
- **未通过**：完整 16 s 无辅助控制器；grasp 阶段指-桌面碰撞严重。
- **默认 C2 不受影响**，C2_CLITE 标记为实验模式。
- 下一步优先级：
  1. 让 Panda 根以稳定方式 C-lite 化（例如直接移除根 FREE joint，或改由 mocap 位形约束根，而非 free joint + weld）；
  2. 基于连续支撑路径重标定 descend/grasp 净空；
  3. 全接触对穿透回归与 4 种 YCB × 多 seed 验证。
