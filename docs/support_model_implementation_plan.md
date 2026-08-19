# C2 支撑模型重构实施方案（v1.2，实施前冻结决定）

更新日期：2026-08-18
状态：**阶段 A 已实现并验证**（2026-08-18）：38 tests passed；official 1 s 谱探针伪接触消失；完整 16 s 回合为诚实失败（`support_geometry_valid=true`），official 尚未满足 D2 计分资格，详见 `docs/baseline_status.md`。
前置文档：`docs/c2_mount_inconsistency.md`、`docs/prompts/opus.md`
修订说明：v1.2 针对第二轮复核的两条结构性问题作出冻结决定：①同组接触对改为 Newton Builder 的结构性 pair filter，t=0 不再自触发 invalid；②安全门与评分门槛的依赖方向反转，改为几何/物理常数，不再由激励标定。其余意见全部纳入 `v1.2 决定`。凡 v1.1 正文与 `v1.2 决定` 冲突处，以 v1.2 为准。

---

## v1.2 实施前冻结决定

### D1. 同组结构接触：构造性 pair filter，不用装配偏移当逃生舱

1. `build_support_groups()` 除返回组外，还返回 `structural_exclusion_pairs`，硬装档初始表为：

   ```text
   (panda_link0, VibrationFloor)
   (WorkTableLeg*,  VibrationFloor)
   (WorkTableLeg*,  WorkTableTop)
   ```

2. 场景构建时通过 `NewtonManager._builder.add_shape_collision_filter_pair(shape_a, shape_b)` 安装这些过滤对（Builder 公开 API，按 shape label 匹配；与 C2_CLITE 已使用的 MODEL_INIT 回调模式一致）。这不是 contype/conaffinity 掩盖，是“刚体组结构性连接不参与接触”的直接推论。
3. `robot_assembly_offset_m` 改名 `assembly_clearance_m = 0.0005`，语义是**装配公差垫片**：只保证 `dist > 0`，避免 `dist≈0` 的浮点抖动；不是几何修复手段。桌腿底同样使用该公差。
4. 硬装档的 `support_geometry_valid` 运行时扫描降级为两个静态断言：
   - 场景构建后 t=0 同组结构对间隙 `> 0`；
   - pair filter 已实际安装（候选对计数按新期望值断言）。
   运行时逐步扫描只在 `isolated_table` 启用。
5. 禁止接触对按 `support_model` 分表：
   - 硬装：`panda_link0↔VibrationFloor`、`WorkTableLeg↔VibrationFloor`（由 pair filter 排除，扫描只做静态断言）；
   - isolated：`panda_link0↔VibrationFloor` 仍为 invalid；`WorkTableLeg↔VibrationFloor` 变为合法一级事件 `pad_bottomed_out`，不再判 invalid。
6. 阶段 A 验收不再要求 386 / 29 / 348 不变；安装 pair filter 后实测新拓扑并写回 CLAUDE/文档作为新不变量。

### D2. 门限固定为物理/几何常数，不随激励标定

1. **位移门限固定**：

   ```text
   min_task_feature_thickness_m = 0.008   # target bin wall thickness
   alpha_geometry = 0.05                  # 全局求解器安全系数，与谱型无关
   max_substep_displacement_m = alpha_geometry * min_task_feature_thickness_m
                                = 0.00040 m
   ```

   该值只随求解器/几何变化而全局修改；任何 seed 重放峰值超过它，治理动作是**整体提高 solver_substeps**，不允许删除 seed、不允许提高门限迁就激励。

2. **穿透评分门槛固定为工件特征尺寸的函数**：

   ```text
   penetration_threshold_mm = 0.01 * selected_workpiece_min_collider_dimension_mm
   ```

   sugar_box@0.75 实测最小 collider 尺寸 33.602 mm → 门槛 0.336 mm。阈值与物理档位、seed、版本无关。

3. **计分资格由数值地板决定**：
   `official` 只有在“只振动不操作”的数值地板 ≤ `penetration_threshold_mm / 3` 时才有资格计分。达不到就继续改进求解器配置（子步/solref/支持节拍），而不是放宽门槛。

4. 离线重放门必须在 `run.sh` 构建场景前执行；成员表直接消费 `build_support_groups()` 的同一份几何表，二者同源，有断言。

### D3. 阶段 B 的隔振器默认值改为“带外 + 显式推导”

1. 新增强制校验：任意 `fn_*` 不得落入任何活动激励带的 `[0.8·lo, 1.2·hi]`。
2. 阶段 B 暂定默认值（**未验证，实现时必须跑重放校验**）：

   ```text
   fn_z_hz      = 12.0
   fn_rock_hz   = 20.0     # 高于 ry 带上界 12.1×1.2 = 14.52
   shear_stiffness_ratio = 1/3
   table_yaw_rock_inertia_ratio = 2.0
   fn_shear_hz  = fn_z * sqrt(1/3) ≈ 6.93
   fn_torsion_hz ≈ fn_rock * sqrt((1/3)/2) ≈ 8.16  # 高于 rz 带上界 6.72×1.2=8.06
   zeta = 0.10
   ```

3. `fn_torsion` 不手抄常数，由 `shear_stiffness_ratio` 与 `table_yaw_rock_inertia_ratio` 显式导出；假设可见、可复核。
4. 阶段 B 默认 `mount_type="bonded_stud"`、`mount_height_m=0.035`、`max_strain_axial=0.20`、`max_strain_shear=0.30`、`spectral_scale=0.22` 为初值；**全部以离线重放峰值按最不利脚校核**，不通过就在启动时拒绝并打印建议值。

### D4. 应变只吃重放真实峰值

- `validate_isolator_feasibility()` 的输入必须是 `offline_support_travel_report()` 返回的四脚真实峰值（取最不利脚）；
- 轴向与剪切分开：`strain_axial = peak_axial / mount_height`，`strain_shear = peak_shear / mount_height`；
- 3.5σ 和确定性峰值界只用于日志参考，不进入通过/拒绝判定。

### D5. 外层节拍判读

- 重放报告对每个被驱动成员输出三个数：`v·dt`、`0.5·a·dt²`、`substep_travel`；
- 结论规则：只有 `0.5·a·dt²` 是外层写入伪影的有效量；若其 < `max_substep_displacement_m/2.4`，1 kHz 写入节拍足够；
- 另加“速度是否真的参与积分”探针：单轴正弦激励下，取子步中间时刻实际位姿与解析一阶保持位姿比较，逐资产（Panda free joint / kinematic platform / kinematic legs）验证。

### D6. y 偏移不做

`y=0.12` 只换 6.6% 差动却要重设三个布局点和视觉基线，不划算；`y=0.30–0.325` 的关节限位/可达性风险大于收益。硬装档保持 `worktable_center.y=0`，差动只由 `ry·0.65` 产生，并在文档写明这一简化。

### D7. 七条小意见全部执行

1. 重放门在启动时执行，不是事后报告；
2. 重放几何表与 `SupportGroup` 成员表同源，加 `test_replay_uses_same_member_table`；
3. seed 治理：官方 seed 集任何一个超限 → 整体提子步，禁止删除 seed；
4. `ee_tracking_error` 在甲板系下计算（指令与实际都变换到 deck frame）；
5. 重力开启后记录 settle 关节力矩；PD 重调属于阶段 A，不允许“以后再说”；
6. `support_geometry_valid` 分模式语义（见 D1）；
7. 阶段 B 共振带外校验与显式刚度参数（见 D3/D4）。

---

## 0. 决策摘要

1. **official（可评分）= 单坐标系硬装甲板**：删除 `arm_mount_xy_m / table_mount_xy_m`，所有资产由一个 deck 支撑组驱动；差动只由可见几何产生。
2. **`isolated_table`（研究档，不计分）= 六自由度隔振器模型**：默认采用**螺栓贯穿/粘接式减振件（bonded stud）**，垫高 28 mm、最大应变 20%、无预压；可行性用应变和行程校验，不用“预压 + 静态下沉”作主判据。
3. **安全门 = 离线全回合重放**：不再使用 3.5σ / 上边界频率 / 力臂估计式；按每个 seed 的实际解析波形和实际更新节拍重放，取真实 `max|v|·dt`。
4. **几何干净化**：删除两个 7 mm `dynamic_clearance`；机器人根与甲板顶面按实测碰撞体装配，桌腿底与甲板顶面齐平。
5. **评分穿透门槛重新标定**：先用“数值地板”实验测出纯振动、无操作时的最大穿透，再以地板值的 3 倍设评分门槛，不再沿用固定 0.3 mm。

---

## 1. 对 30 mm 垫高的修正结论

v1.0 认为 30 mm 垫高不合理；复核指出该结论只对“无粘接、靠自重坐放的橡胶脚”成立。修正后的可行性判据是**弹性体应变**：

```text
bonded_stud / 螺栓贯穿：  strain = peak_travel_m / mount_height_m
unbonded_pad：            strain = (preload_m + peak_travel_m) / mount_height_m
```

按 `spectral_scale=0.30`、`fn_z=12, fn_rock=11, ζ=0.10` 的角点垂向行程（RMS ~1.0–1.3 mm、3.5σ 峰值 ~4.5 mm）估算：

| 垫高 | 无粘接（预压 4 mm + 行程） | 螺栓贯穿（仅行程） |
|---|---:|---:|
| 15 mm | ~57% | ~30% |
| 25 mm | ~34% | ~18% |
| 30 mm | ~28% | **~15%** |

结论：15 mm + 4 mm 预压的无粘接脚在材料上不可行；**28 mm 螺栓贯穿减振件 + 20% 应变限是正常台架五金规格**。因此：

- official 仍选硬装，不是因为隔振器不可行，而是因为官方基线应最小化自由参数；
- `isolated_table` 默认 `mount_type="bonded_stud", mount_height_m=0.028, preload_m=0.0, max_strain=0.20`；
- 删除 v1.0 的 `mount_height_m <= 0.020` 上限。

---

## 2. 目标架构

```text
VibrationConfig ──> SpectralVibration
                        │
              ┌─────────┴──────────┐
              v                    v
         deck motion q_c      table motion q_t（仅 isolated_table）
              │                    │
              v                    v
      SupportGroup("deck")   SupportGroup("table")
      platform, robot,       worktable, table_legs,
      worktable*(硬装)        target, workpiece_shadow
              │                    │
              v                    v
       write_support_groups()  唯一位姿写入点
```

不变式：**组是唯一刚体来源**。组有且只有一个 `(q, R)` 和一个 `rotation_anchor`；成员只有 `(asset, local, write_strategy)`。

---

## 3. 文件级实施方案

### 3.1 `src/vibench/config.py`

1. **删除**：
   - `arm_mount_xy_m`、`table_mount_xy_m`；
   - `robot_mount_dynamic_clearance_m`、`table_mount_dynamic_clearance_m`；
   - `support_mount_radius_m` property（安全门改为重放，不再需要统计力臂）。

2. **保留并重定义**：
   - `robot_base = (-0.47, 0, platform_top_z)`，即 `z = 0.08`；实际写入时叠加 `robot_assembly_offset_m`（见 §3.6）；
   - `worktable_center = (0.18, 0.12, 0.34)`（y 偏移 0.12 m，让 `rx` 参与硬装差动；见 §3.10）；
   - `workpiece_start = (0.08, -0.01, ...)`、`target_center = (0.08, 0.29, ...)` 随工作台 y 平移；
   - `support_config: Literal["C2", "C2_CLITE", "isolated_table", "legacy_virtual_mount"]`。

3. **新增 `TableIsolatorConfig`**：

```python
@dataclass(frozen=True)
class TableIsolatorConfig:
    enabled: bool = False
    mount_type: Literal["unbonded_pad", "bonded_stud"] = "bonded_stud"
    mount_height_m: float = 0.028
    max_strain: float = 0.20
    preload_m: float = 0.0          # bonded_stud 时必须为 0
    fn_z_hz: float = 12.0
    fn_rock_hz: float = 11.0
    fn_shear_hz: float | None = None   # 默认 fn_z / sqrt(3)
    fn_torsion_hz: float | None = None # 默认 fn_rock / sqrt(3)
    zeta: float = 0.10

    def __post_init__(self):
        # bonded 不允许 preload；unbonded 必须有 preload
        # 所有 mount 参数只在 enabled=True 时校验
```

4. **新增禁止接触对配置**：

```python
forbidden_contact_pairs: tuple[tuple[str, str], ...] = (
    ("panda_link0", "VibrationFloor"),
    ("WorkTableLeg", "VibrationFloor"),
)
```

运行时用子串/glob 匹配 shape label；命中且 `dist < 0` 即 `support_geometry_valid=false`。

### 3.2 新建 `src/vibench/supports.py`

```python
@dataclass(frozen=True)
class SupportMember:
    name: str
    asset: Any
    local: tuple[float, float, float]
    write_strategy: Literal["root", "collection"] = "root"

@dataclass(frozen=True)
class SupportGroup:
    name: str
    motion_source: Literal["deck", "table"]
    rotation_anchor: tuple[float, float, float]   # 唯一，永远取 platform_center
    members: tuple[SupportMember, ...]
```

- `build_support_groups(cfg, scene)` 从 `cfg` 的可见布局生成成员表；桌腿偏移等几何真值只允许在 scene 配置和本函数中出现一次，config 不再复制。
- `write_support_groups` 为**纯关键字参数**：

```python
def write_support_groups(
    *,
    groups: tuple[SupportGroup, ...],
    q_deck: torch.Tensor,
    qd_deck: torch.Tensor,
    q_table: torch.Tensor | None,
    qd_table: torch.Tensor | None,
    env_origins: torch.Tensor,
) -> None:
```

- 成员写入选派：
  - `"root"` → `write_root_pose_to_sim_index`；
  - `"collection"` → 把 Stewart 12 腿与工件阴影整体作为成员，`write_body_pose_to_sim_index`。
- **Stewart 解算不再单独取 platen pose**：它读取 deck 组中 `platform` 成员刚算出的 pose，保证视觉腿与甲板严格同步。
- 组内刚体公式（唯一实现）：

```python
quat = quat_from_euler_xyz(q[3], q[4], q[5])   # 本轮顺手取消 -ry 反手约定，见 §3.11
position = env_origins + q[:3] + anchor + quat_apply(quat, local - anchor)
linear  = qd[:3] + torch.linalg.cross(angular_velocity(q, qd), quat_apply(quat, local - anchor))
```

`angular_velocity` 使用精确的 `E(θ)θ̇`，不再把欧拉角速率当 ω。

### 3.3 `src/vibench/vibration.py`

1. `sample()` 保持为 deck 六轴 `q/qd/qdd`。
2. `reseed()` 额外生成 table 通道的逐 tone 增益/相位表；**六个轴全部走传递率**：
   - `tz` → `(fn_z, ζ)`；
   - `rx/ry` → `(fn_rock, ζ)`；
   - `tx/ty` → `(fn_shear, ζ)`，默认 `fn_z/√3`；
   - `rz` → `(fn_torsion, ζ)`，默认 `fn_rock/√3`。
3. 两个传递函数同时提供：

```text
H_rel(ω) = r² / sqrt((1-r²)² + (2ζr)²)          # Z/Y：相对位移，用于行程预算
H_abs(ω) = sqrt(1+(2ζr)²) / sqrt((1-r²)²+(2ζr)²) # X/Y：绝对运动，用于安全门
```

4. `sample_table(time_s)` 返回绝对 table 六轴运动；速度/加速度逐 tone 解析求导；ramp 对 deck 与 relative 分量分别应用乘积法则后相加。

### 3.4 安全门改为离线全回合重放

删除/停用统计式 `estimated_peak_velocity_m_s + validate_impulsive_timestep` 的 spectral 分支（sine 分支保留解析峰值）。

新增：

```python
def offline_support_travel_report(
    vibration: VibrationConfig,
    isolator: TableIsolatorConfig,
    group_geometry: Sequence[SupportMember],  # 只需 local，不建场景
    physics_hz: int,
    substeps: int,
    episode_s: float,
    update_dt_s: float | None = None,
) -> SupportTravelReport:
    # 逐 seed、逐解析采样点，计算每个成员在“实际支持更新节拍”上的真实位移
    # 返回 max_substep_travel_m / max_outer_update_travel_m / 成员明细
```

规则：

- **评分门（substep）**：`max_substep_travel_m = max|v| · (1/(physics_hz·substeps))`，启动时对当前 seed 全回合重放；`> max_substep_displacement_m` 则拒绝启动。
- **外层阶梯审计**：C2 kinematic 支持实际按外层 `dt` 写，因此同时报告 `max_outer_update_travel_m`；该值与“数值地板”实验一起决定是否需要在后续把支持写入提升到子步节拍（C2_CLITE 已有此能力）。
- `limit_m` 不再沿用固定 0.300 mm：先跑数值地板，按 §7 标定。

该重放是纯 NumPy、不启 Isaac；代价为每 seed 数万次解析采样（预计 < 1 s）。

### 3.5 `src/vibench/task.py`

- `_write_supports()` 改为调用 `write_support_groups()`；
- 硬装 C2：只有一个 deck 组；
- `isolated_table`：deck 组 = `platform, robot`，table 组 = `worktable, legs, target`，工件阴影随 table 组；
- observation：
  - 保留 deck 的 `vibration_q/qd/qdd`；
  - 新增 `vibration_q_table/qd_table/qdd_table`、`table_relative_travel_mm`；
  - 删除 `mount_delta_z`；
- 指标新增：
  - `ee_tracking_error_rms_m`、`max_ee_tracking_error_m`：在甲板系下，控制器指令 EE 位姿与实际 EE 位姿的逐步偏差（settle 后开始累计）；
  - `support_geometry_valid`：运行时逐步扫描 `forbidden_contact_pairs`，任一命中（`dist < 0`）立即置 false 并记录 `support_geometry_invalid_t` 与 pair；
- 硬装下 `robot ↔ worktable` 合法接触（如 `descend_table_contact`）不受禁止清单影响。

### 3.6 `src/vibench/scene.py`

**硬装阶段：**

- 删除 `robot.spawn.rigid_props.disable_gravity = True`（默认 false；是否最终打开见 §3.12 决策门）；
- 机器人根装配：
  - 实测 `panda_link0/collisions` 底相对根原点约 `-0.03 mm`；
  - `robot_base.z = platform_top_z`（0.08 m），即碰撞体底面与甲板顶面齐平；
  - 若 NativeCCD 零 margin 下 t=0 出现伪接触，最小必要 `robot_assembly_offset_m`（如 0.0005 m）写入配置并附测试，不恢复 7 mm；
- 工作台/桌腿：
  - `worktable_center.z = 0.34`，无 +0.007；
  - 桌腿高度 = `table_bottom − floor_top = 0.23 m`，腿底 z=0.08，与甲板顶面齐平；
- `platform` 保持一块、保持碰撞；桌腿/机器人随 deck 组运动后不存在差动间隙。

**`isolated_table` 阶段：**

- 桌腿物理碰撞盒缩短，底端停在工作台底面下方 `mount_height_m` 处；
- 可见层改为“金属腿 + bonded-stud 减振件”，减振件为纯视觉两段式伸缩件，`collision_enabled=False`；
- 桌面、目标盒、工件阴影进 table 组。

### 3.7 `src/vibench/panel_task.py` 与 `C2_CLITE`

- `panel_task.py` 删除重复支撑写入，复用 `supports.py`；
- `isolated_table + panel_operation` 暂不组合（`__post_init__` 拒绝）；
- `C2_CLITE`：
  - driver 命名/焊接关系不变；
  - `_write_clite_drivers()` 按 `SupportGroup.motion_source` 选择 `q_c` 或 `q_t`；
  - `isolated_table + C2_CLITE` 暂不组合。

### 3.8 `src/vibench/cli.py`

- `--support-config` 增加 `isolated_table` 与 `legacy_virtual_mount`；
- `legacy_virtual_mount` **硬性拒绝 `physics_profile=official`**；
- 启动时：
  1. 构建组几何（不建场景）；
  2. 调用 `offline_support_travel_report`；
  3. 检查 `max_substep_travel_m <= cfg.max_substep_displacement_m`，否则拒绝并打印数值；
- JSON 新增：
  - `support_model`
  - `support_geometry_valid`
  - `max_substep_travel_mm`、`max_outer_update_travel_mm`（重放真实值）
  - `ee_tracking_error_rms_m`、`max_ee_tracking_error_m`

### 3.9 `src/vibench/recording.py`

- `C2 Delta-z` → `arm-table surface Δz`（硬装）/ `isolator relative travel`（isolated）；
- isolated 模式叠加 table 相对位移曲线；
- overlay 增加 `support_geometry_valid` 状态。

### 3.10 硬装布局的 y 偏移

当前 `robot_base.y = worktable_center.y = 0`，因此硬装差动只来自 `ry`；`rx`（RMS 为 `ry` 两倍）贡献严格为零。默认改为：

```python
worktable_center = (0.18, 0.12, 0.34)
workpiece_start  = (0.08, -0.01, ...)
target_center    = (0.08, 0.29, ...)
```

`y=0.12` 时硬装臂/台表面差动约 RMS 1.40 mm / 峰值 4.51 mm（y=0 时为 1.32 / 3.98）。实施时重新验证 Panda 可达性；若不可达则回退 y=0 并记录原因。

### 3.11 顺手清掉历史约定

- 取消 `-ry` 反手约定：`quat_from_euler_xyz(q[3], q[4], q[5])`，全链路同步（mounting、task、panel、Stewart 测试、文档）；
- 角速度改为精确 `ω = E(θ) θ̇`，不再把欧拉角速率当 ω；
- 由于 mrad 量级下数值结果几乎不变，本次属于 API 清洁，不改变难度。

### 3.12 重力决策门

硬装叙事要求“6 g 下机械臂自身的跟踪误差是主要难点”，与 `disable_gravity=True` 冲突。实施顺序：

1. 先打开 `disable_gravity=False`，跑 0.2 s 无振动 settle probe；
2. 测关节下垂/EE 误差；若 `FRANKA_PANDA_HIGH_PD_CFG` 能保持目标位姿（误差 < 阈值），保留重力开启；
3. 若不能，二选一并写入文档：
   - 增加显式重力补偿（控制器/执行器层）；
   - 保留零重力但 README **不得**使用“6 g 下机械臂有多难”的叙事，只描述基座运动学激励。

无论结果，`ee_tracking_error` 都作为一级指标。

---

## 4. `isolated_table` 可行性校验（修订版）

```python
def validate_isolator_feasibility(cfg, travel: SupportTravelReport):
    peak = travel.table_relative_peak_m
    if cfg.isolator.mount_type == "bonded_stud":
        assert cfg.isolator.preload_m == 0.0
        strain = peak / cfg.isolator.mount_height_m
    else:
        strain = (cfg.isolator.preload_m + peak) / cfg.isolator.mount_height_m
    assert strain <= cfg.isolator.max_strain
    # 横向剪切行程单独报告：shear_peak <= mount_height * max_shear_strain
```

默认值：

```python
mount_type="bonded_stud", mount_height_m=0.028,
max_strain=0.20, preload_m=0.0
```

`isolated_table_research` scenario 使用 `spectral_scale=0.30`；全谱下该配置需重新校验 strain 是否 ≤ 20%，超限时报告建议 scale。

---

## 5. `support_geometry_valid`（修订版）

不再按“组间接触”判无效（那会把 `robot ↔ worktable` 的合法任务接触误判），改为显式禁止清单：

```yaml
forbidden_contact_pairs:
  - ["panda_link0", "VibrationFloor"]
  - ["WorkTableLeg", "VibrationFloor"]
```

运行时逻辑：

1. 每步从 active contact 的 shape0/shape1 label 匹配；
2. `dist < 0` 且 pair 命中 → `support_geometry_valid=false`，记录首次命中时刻与 pair；
3. episode 结束后 JSON 公开记录；
4. 该判定与 `SupportGroup` 无关，硬装/isolated/legacy 三种模式都有效。

---

## 6. 测试计划（修订版）

### 6.1 删除/替换

- 删除旧 `c2_mount_delta / c2_complete_se3` 测试；
- 删除对 `arm_mount_xy_m/table_mount_xy_m` 和两个 `dynamic_clearance` 的断言。

### 6.2 结构不变量

1. `test_write_root_pose_has_single_call_site`：
   用 AST 扫描包内所有源码，断言 `write_root_pose_to_sim_index` 只出现在 `supports.py::write_support_groups`；
2. `test_support_group_member_distances_preserved`：
   随机 1000 组 `(q, qd)`，断言组内任意两成员 `|p_i-p_j| = |l_i-l_j|` 误差 < 1e-9；
3. `test_support_group_shared_quaternion_and_exact_velocity`：
   组内 quat 全等，且速度满足 `v_i = v + ω×R(l_i-c)`，其中 ω 用精确 `E(θ)θ̇`；
4. `test_stewart_uses_platform_member_pose`：
   断言 Stewart 12 腿的输入 platen pose 就是 deck 组 platform 成员刚生成的 pose，不允许二次求解。

### 6.3 隔振器正确性

1. `test_single_tone_transfer_matches_closed_form`：
   单 tone 配置下，`sample_table` 与 `H_rel/H_abs` 解析增益、相位吻合到 1e-6；
2. `test_table_mount_strain_constraint`：
   默认 research 配置完整回合，`(peak)/height <= max_strain`；
3. `test_all_six_axes_have_finite_transfer`：
   `fn_shear/fn_torsion` 默认导出，且六个轴 table 通道均非恒等于 deck。

### 6.4 安全门

1. `test_offline_replay_matches_direct_bruteforce`：
   任意 seed/短回合，重放报告的 `max_substep_travel_m` 与逐点暴力扫描一致；
2. `test_startup_gate_rejects_over_limit_seed`：
   构造超过 limit 的 seed/scale，启动前被拒绝；
3. `test_official_seed_sweep_travel_below_limit`：
   官方 seed 集在 `max_substep_displacement_m` 标定值以下（标定后写死）。

### 6.5 禁止接触对与几何

1. `test_forbidden_pair_matching_covers_legacy_bug_pairs`；
2. `test_hard_mount_initial_geometry_has_no_initial_contact`：
   t=0 时 link0/桌腿与地板 `dist > 0` 或严格齐平，不得为负；
3. `test_dynamic_clearance_fields_removed`（源码级断言）。

### 6.6 更新现有测试

- `test_default_profile_is_official_and_unassisted`：硬装、无 isolator、`solver_substeps=5`；
- `test_training_spectral_profile_is_rejected...`：改为调用重放门；
- `test_third_round_geometry`：桌腿/阴影随 y=0.12 布局更新；
- `test_visual_manifest`：硬装阶段视觉 manifest 不变；isolated 阶段新增减振垫后单独更新。

---

## 7. 数值地板与评分门槛标定

在阶段 A 的仿真验证中执行：

1. **数值地板实验**：机械臂保持 settle 位姿、不做任何抓取，只开默认谱，跑 official 1 s（seed 17/31/47/73/101），记录 `max_penetration_mm`；
2. 定义 `floor_mm = max(seeds)`；
3. 评分门槛 = `max(0.1, 3 × floor_mm)`，写入 `configs/scenarios.yaml`；
4. `max_substep_displacement_m` 同样按重放真实值标定：取官方 seed 集最大 `max_substep_travel_m × 1.1`，不沿用 0.300 mm；
5. 旧 0.3 mm 数值只保留在历史文档中，不再作为当前门槛。

---

## 8. 基线作废与 legacy 对照

### 8.1 旧产物作废

以下文件产自虚拟测点模型，标记 `legacy/invalid` 并停止引用为通过证据：

- `docs/penetration_baseline.json`
- `docs/penetration_official_final.json`
- `docs/visual_baseline.json`
- `docs/third_round_*` 中的历史视觉/成功 JSON
- `out/benchmark_v2_*`

处理：新增 `docs/baseline_status.md`，逐项标注 `invalidated_by: support_model_refactor`；新基线在阶段 A 验收后生成。

### 8.2 `legacy_virtual_mount`

保留旧模型为显式研究配置（不是默认、不是 official）：

```yaml
support_config: legacy_virtual_mount
arm_mount_xy_m: [0.75, -0.45]
table_mount_xy_m: [-0.75, 0.45]
allowed_profiles: [training]   # official 硬拒绝
```

用途：与硬装档在相同 seed 集上做配对对比，量化 `robot_link<->platen` 伪接触对旧分数的污染；结果写入 `docs/legacy_virtual_mount_comparison.md`，下一个版本删除该模式。

---

## 9. 文档更新

1. `docs/c2_mount_inconsistency.md`：标记方案已选定并链接本文档；
2. `docs/reports/current_implementation.md`：
   - 删除虚拟测点与 7 mm dynamic clearance 描述；
   - 写入硬装甲板、`isolated_table`、重放安全门、禁止接触对、新指标；
3. `CLAUDE.md`：C2 段改为“单坐标系硬装 + 可选隔离台研究档；组写入唯一调用点”；
4. `README.md`：
   - 难度叙事改为“6 g 甲板激励下，机械臂跟踪误差 + 工件滑移/倾覆 + 抓取保持”；
   - 明确 legacy 模式与旧产物状态；
5. `docs/prompts/opus.md` 不改，作为设计讨论记录。

---

## 10. 实施顺序与验收

### 阶段 A：硬装 official

1. 删除虚拟测点参数，保留到 `legacy_virtual_mount` 子配置；
2. 新建 `supports.py`，task/panel 切换到唯一写入点；
3. 归零 dynamic clearance，y=0.12 布局，重力决策门，`-ry`/ω 清理；
4. 实现重放安全门与禁止接触对扫描；
5. 更新测试/场景/文档，旧基线标记 invalid；
6. `./run_tests.sh -q` 全绿；
7. 数值地板实验 → 标定穿透门槛与 `max_substep_displacement_m`；
8. official 1 s ×5 谱探针：

```bash
./run.sh --physics-profile official --episode-s 1 \
  --vibration spectral --seed 17 \
  --metrics-output out/rigid_deck_1s_probe.json
```

验收：
- `support_geometry_valid=true`；
- `max_penetration_pair` 不再是 `robot_link<->platen`；
- `max_penetration_mm <= 标定门槛`；
- 接触拓扑仍为 386 / 29 / 348（阶段 A 不增删碰撞体）。

9. 完整 16 s official seed=17 录制/指标，记录 `success` 或诚实失败原因。

### 阶段 B：`isolated_table` 研究档

1. `TableIsolatorConfig + sample_table()`（六轴传递率）；
2. table 组、桌腿物理缩短、可见 bonded-stud 减振件；
3. strain/行程校验与离线报告；
4. `isolated_table_research` scenario（`spectral_scale=0.30`）；
5. 240 Hz/1000 Hz 短探针，验证相对行程与安全门；
6. 不计分，文档标注。

### 阶段 C：legacy 对照与删除

1. `legacy_virtual_mount` 训练档配对对比（相同 seed）；
2. 写污染量化报告；
3. 下个版本删除 legacy 代码与旧基线引用。

---

## 11. 明确不做

- 不用 `contype/conaffinity` 或 `nxn_geom_pair_filtered` 过滤伪接触；
- 不恢复 7 mm dynamic clearance；
- 不把 `isolated_table` 或 `legacy_virtual_mount` 用于 official；
- 不再使用 3.5σ 统计安全门估算 spectral 档；
- 不沿用固定 0.3 mm 穿透门槛，直到数值地板标定完成。

---

## 12. 与 v1.0 / Opus 方案的关系

- 采纳 Opus：单一坐标系、差动显式化、`SupportGroup` 使错误不可表达、隔振器有物理归属；
- 修正 v1.0：隔振器用应变判据并支持 bonded stud；六轴全传递；安全门用离线重放；7 mm clearance 归零；禁止接触对替代组间判据；
- official 选择硬装是**基线简化决策**，不是隔振器不可行的结论。
