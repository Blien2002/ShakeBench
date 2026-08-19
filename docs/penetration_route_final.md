# 穿模根绝路线决赛：全部候选方案、优劣与实验结论

更新日期：2026-08-18  
项目：`ViBench`（Isaac Lab 3.0 + Newton 1.2.1 / MJWarp）  
文档目的：把穿模问题的**全部候选技术路线**放在同一张表上，给出每个方向的内容、优劣、实验证据、风险和最终排序。  
实验证据目录：`out/penetration_experiments_20260818/`（该目录被 `.gitignore` 忽略）。  
代码备份：`~/Desktop/ViBench_backups/ViBench_code_backup_20260818_105442.tar.gz`。

---

## 1. 问题定义与验收标准

### 1.1 问题

振动地板、Panda 浮动根、工作台、桌腿、目标盒都是 **kinematic 刚体**，每个 1 ms 外层步被直接覆写一次 root pose/velocity。Newton 虽然在内部跑 4 个 0.25 ms 子步，但支撑体在这 4 个子步内不动。因此：

- official 档支撑体实际阶跃 ≈ 1.056 mm/外层步；
- training 档支撑体实际阶跃 ≈ 4.4 mm/外层步；
- 接触 margin 又被 Newton #2106（NATIVECCD/MULTICCD）在编译期清零；
- 碰撞检测只能在支撑“跳完”之后看到重叠，必然产生数值隧穿。

当前项目用“限制振动幅度 + 调硬 `solref`”压住穿透，属于视觉/统计上的绕过，不是根绝。

### 1.2 验收标准（路线决赛用）

1. **全接触对**最大几何穿透 ≤ 0.3 mm，而不仅是全局单值；
2. 不产生新的接触对或新的近碰撞；
3. 腕力峰值不显著劣化（不能靠 speculative 力换取零穿透）；
4. 不依赖减小振动幅度；官方/训练档都能按物理激励运行；
5. 接触拓扑保持 `386 / 29 / 348`（或变化可解释、可记录）；
6. 不恢复 `grasp_assist`、不放松成功判据。

---

## 2. 候选路线总览

| 路线 | 核心思想 | 证据级别 | 状态 |
|---|---|---|---|
| R0 现状 | 降振幅 + 全局 solref | 真实场景（已知行为） | 现行，不根治 |
| A 运行时恢复 margin | 编译后填 `mjw_model.geom_margin` | 真实场景 | **失败** |
| A' 禁用 NATIVECCD + margin | 编译期/运行时让 margin 合法 | 真实场景 | **失败** |
| B 支撑逐子步更新 | kinematic 传送改成 4 kHz 小步 | 真实场景 | **部分成功，有副作用** |
| C-lite mocap+weld 动态支撑 | 动态平台被 mocap 约束拖动 | 最小模型 + 真实子集 | **当前第一候选，真实子集探针已通过** |
| C 全动态支撑 | 平台/机械臂/工作台全部动态执行器驱动 | 概念 | 长期终极方向 |
| D Newton CollisionPipeline | `use_mujoco_contacts=False` | 真实场景（一组参数） | **当前参数失败，待重标定** |

---

## 3. 逐路线详细分析

## 3.1 R0：现状——降振幅 + 全局 solref

**内容**

- `validate_impulsive_timestep()` 拒绝每子步估算位移超过 0.3 mm 的激励；
- 全局覆写 `geom_solref=(0.00060 s, 1.0)`（official）或 `(0.0025 s, 1.0)`（training）；
- JSON 公开记录 `nativeccd_margin_honored=false`。

**优点**

- 已实现、稳定、可复现；
- 穿透在官方短回合可压到 0.167–0.307 mm（多次运行区间）；
- 不影响接触拓扑。

**缺点**

- 直接限制了 benchmark 的激励难度，削弱科学性；
- training 档默认谱无法运行；
- 所谓“子步位移”是误导性表述：支撑体真实阶跃是外层阶跃（official ≈1.056 mm）；
- 不解决根因，只是让数值隧穿不显眼。

**实验证据**

- 2026-08-18 多次 1 s official 谱 idle probe：最大穿透 0.1668 / 0.1822 / 0.3070 mm，接触对在 `workpiece<->worktable` 与 `robot_link<->platen` 之间波动。

**结论**：保留为 fallback；不得作为最终方案。

---

## 3.2 A：运行时恢复 `geom_margin`

**内容**

模型编译完成后直接 `solver.mjw_model.geom_margin.fill_(1e-3)`，绕过 #2106 的编译期清零。

**优点**

- 一行级改动；
- 最小 box-box 模型中，official 1.056 mm 阶跃下穿透可为零。

**缺点（真实场景实验暴露）**

- 成对 margin 和为 2 mm，而工件初始悬浮间隙只有 1 mm：接触在**未接触时**就激活并施力；
- 真实场景腕力峰值 36.78 N → **21 638 N**，工件被反重力推开/悬空；
- 这是 unphysical speculative force，不是根绝。

**实验证据**

| 分支 | 最大穿透 | 腕力峰值 |
|---|---:|---:|
| margin=0 | 0.1668 mm | 36.78 N |
| margin=1 mm | 0.2798 mm | **21 638 N** |

JSON：`01_official_margin_only_probe.json`

**结论**：**否决**。

---

## 3.3 A'：禁用 NATIVECCD 后再使用 margin

**内容**

MuJoCo 3.8 提供 `mjDSBL_NATIVECCD`。实验在模型编译后设置：

```text
opt.disableflags |= mjDSBL_NATIVECCD
geom_margin = 1 mm（或配合 gap）
```

让 margin 绕过 native CCD 的限制。

**优点**

- 不依赖私有数组时序，方向上比 A 更接近上游预期。

**缺点 / 实验结果**

- `margin=0/gap=0` 且禁用 NATIVECCD 后，基线本身变差：穿透 0.423 mm、腕力 124.9 N；
- `margin=1 mm/gap=0` 回合正常应约 40 s，实测 150 s 仍未完成，求解路径被 speculative 接触拖成病态，实验被终止；
- 没有证据表明 A' 能解决 speculative force 或 box/mesh 失稳。

**实验证据**

- `06_nativeccd_off_probe.log`、`06_nativeccd_off_probe_summary.json`

**结论**：**否决当前形态**；若未来 Newton 修复 #2106 或在编译期合法保留 margin，可重测。

---

## 3.4 B：支撑体按 solver 子步更新

**内容**

保持 `1000 Hz × 4` 外层结构，但把 `_write_supports()` 提前到每个 0.25 ms 子步之前；控制器/观测仍按 1 ms 节拍。

**优点**

- 把支撑传送阶跃从 1.056 mm 降到约 0.264 mm；
- 与现行架构距离最近；
- 真实场景公平对照中，主要接触对 `workpiece<->worktable` 穿透从 0.307 mm 降到 **0.161 mm**。

**缺点（真实场景公平对照暴露）**

- 新增 `robot_link<->worktable`（右指↔桌面）1.279 mm 碰撞，发生在 settle 阶段 t≈0.736 s；
- 原因是连续 C2 差动路径真实扫过桌面，原 1 kHz 阶梯近似恰好跳过；这说明当前基线的一部分“安全”来自时间量化；
- 实现依赖 NewtonManager 私有 substep 接口或 4 倍外层 tick，工程侵入性中等偏高；
- 全局最大穿透反而变差，必须在修几何后才能验收。

**实验证据（公平对照，关键表）**

| 接触对 | 现状（支撑 1 ms 更新） | B（支撑 0.25 ms 更新） |
|---|---:|---:|
| workpiece↔worktable | 0.3070 mm | **0.1606 mm** |
| robot_link↔platen | 0.1665 mm | 0.3315 mm |
| robot_link↔worktable | — | **1.2787 mm（新增）** |

JSON：`04_support_cadence_fair_substep_probe.json`

**结论**：**方向正确，但必须与“修复 settle/手指净空”绑定**。单独立项不可通过验收。

---

## 3.5 C-lite：mocap 轨迹 + WELD equality constraint 的动态支撑

**内容**

不直接覆写 kinematic 支撑体位姿，而是：

1. 为每个需要振动的支撑创建一个 **mocap 轨迹体**（fixed-root body 导出为 MuJoCo mocap）；
2. 支撑体改为**动态刚体**；
3. 用 `EqType.WELD` equality constraint 把动态支撑焊在 mocap 轨迹体上；
4. 每步更新 `mocap_pos/mocap_quat`，求解器在每个子步联合求解“轨迹约束 + 接触约束”。

**优点**

- 支撑不再传送：轨迹通过约束进入动力学，接触与约束在同一求解器内协同；
- 工件可以真实地顶住/推回支撑；
- 最小模型实验结果非常干净：
  - official 波形：穿透 0.0149–0.0158 mm，接触力 1.43–1.60 N；
  - training 波形：穿透 0.0675–0.0745 mm，接触力 0.89–0.91 N；
  - 对照 kinematic 传送：穿透 0.827 mm，接触力 84.24 N；
- **真实场景子集探针已通过（2026-08-18）**，子集 = 真实工作台 box（0.65×0.60×0.06 m，45 kg）+ 真实 YCB sugar_box USD@0.75 + 真实 C2 工作台测点运动：
  - kinematic 传送基线：穿透 0.9637 mm，接触力 555.5 N；
  - C-lite mocap 1 kHz：穿透 0.3681 mm，接触力 3012.9 N（更新太慢时约束冲击较大）；
  - **C-lite mocap 4 kHz：穿透 0.0128 mm，接触力 477.5 N**（穿透降低 98.7%，力低于基线）；
- 机制有上游测试佐证（`test_fixed_root_attached_to_world_uses_mocap_and_tracks_pose`）。

**缺点 / 风险**

- 真实子集不含 Panda、目标盒、桌腿与视觉平台，**尚未做完整 ViBench 场景验证**；
- 完整场景集成需要：
  - 平台、Panda 根、工作台、桌腿、目标盒改为动态刚体；
  - 为每个支撑增加 mocap driver + weld 约束；
  - 重新审计接触拓扑、质量/惯量、约束刚度；
- equality constraint 是软约束，支撑会有亚毫米级跟踪误差，需要量化；
- 可能改变 benchmark 的“理想刚性振动台”语义（轻微柔度）；
- mocap 需要按 4 kHz 子步更新才能同时满足低穿透与低力。

**实验证据**

- 最小模型：`07_clite_minimal_probe_summary.json`
- 真实子集：`08_clite_real_subset_probe.json` / `08_clite_real_subset_probe.log`

| 实验 | 配置 | 最大穿透 | 最大接触力 |
|---|---|---:|---:|
| 最小模型 | official，mocap 4 kHz | 0.0149 mm | 1.43 N |
| 最小模型 | training，mocap 4 kHz | 0.0675 mm | 0.89 N |
| 真实子集 | kinematic 传送 | 0.9637 mm | 555.5 N |
| 真实子集 | C-lite mocap 1 kHz | 0.3681 mm | 3012.9 N |
| **真实子集** | **C-lite mocap 4 kHz** | **0.0128 mm** | **477.5 N** |

**结论**：**当前第一候选已升级为“有真实子集证据”**。下一步把该机制扩展到完整 Panda/目标盒/桌腿场景。

---

## 3.6 C：全动态支撑（平台 + 机械臂基座 + 工作台全部动态执行器驱动）

**内容**

彻底废除 kinematic 覆写：振动平台做成动态刚体并由 6-DOF 位置/速度跟踪执行器驱动；Panda 基座与工作台通过 FIXED/WELD 关节或独立执行器附着，让全部振动能量通过执行器进入动力学。

**优点**

- 最物理、最彻底；
- 不依赖 mocap、margin 或私有 substep 接口；
- 接触与支撑动力学完全联合求解。

**缺点**

- 质量/惯量/执行器增益/约束刚度全部需要标定；
- 支撑会产生可测柔度，可能改变 benchmark 语义；
- 需要重构 `scene.py` 的父级结构和大量回归；
- 工程量和风险最大。

**结论**：**长期终极方向**，不作为本轮首选。

---

## 3.7 D / D'：`use_mujoco_contacts=False` + Newton CollisionPipeline

**内容**

关闭 MuJoCo 内部接触生成，改用 Newton 自己的 CollisionPipeline，并配置 `NewtonShapeCfg(margin, gap)` 或依赖 USD authored 的 rest/contact offset。

**优点**

- 是 Newton #2106 警告中明确给出的官方迁移方向；
- 一旦标定正确，可彻底摆脱 NATIVECCD margin 清零问题；
- 拓扑不变（本次实测仍为 386 / 29 / 348）。

**缺点 / 实验结果（一组参数）**

- 朴素配置 `margin=gap=1 mm` 下最大穿透 **4.302 mm**，>0.5 mm 帧占比 **19.18%**，腕力 177 N；
- Newton 的 `margin/gap` 会映射为 `shape_collision_radius/thickness`，与 MuJoCo 直觉不同，需要重新标定；
- 接触力、摩擦、穿透探针口径都需要重新校核。

**实验证据**

JSON：`05_newton_pipeline_probe.json`

**结论**：**当前参数失败，路线未被证伪**。标记为待重标定，不作为本轮首选。

---

## 4. 路线决赛总表

### 4.1 实测数据对照

| 路线 | 证据来源 | 关键穿透指标 | 关键力指标 | 副作用 |
|---|---|---|---|---|
| R0 现状 | 真实场景多次 | workpiece↔table 0.167–0.307 mm | 36.8–124.9 N | 需降振幅 |
| A margin-only | 真实场景 | 0.2798 mm（但 t=0.001） | **21 638 N** | 工件悬空 |
| A margin+gap | 真实场景 | 26.25 mm 或 0 mm（无接触） | 40.5 N 或 0 N | 失稳/悬空 |
| A' NATIVECCD off + margin | 真实场景 | baseline 0.423 mm；m1/g0 超时 | 124.9 N | 求解病态 |
| B 支撑逐子步 | 真实场景公平对照 | workpiece↔table **0.161 mm** | 未测腕力 | 右指↔桌 1.279 mm 新碰撞 |
| C-lite | 最小模型 + 真实子集 | 子集 4 kHz：**0.0128 mm**（基线 0.9637 mm） | **477.5 N**（基线 555.5 N） | 待完整场景扩展 |
| D | 真实场景一组参数 | **4.302 mm** | 177 N | 19.2% 帧超限 |
| C 全动态 | 概念 | 未测 | 未测 | 待标定 |

### 4.2 五维评分

评分：5=最好，1=最差。`证据` 表示在当前实验中的置信度。

| 路线 | 根绝程度 | 物理合理性 | 真实场景证据 | 工程风险 | 基准可比性 | 综合 |
|---|---:|---:|---:|---:|---:|---:|
| R0 现状 | 1 | 3 | 5 | 5 | 5 | 3.0 |
| A | 2 | 1 | 5 | 4 | 2 | 1.5 |
| A' | 2 | 2 | 4 | 3 | 2 | 1.8 |
| B+几何修复 | 4 | 4 | 5 | 2 | 4 | **3.9** |
| C-lite | 4 | 5 | 4（真实子集） | 2 | 4 | **4.4** |
| C 全动态 | 5 | 5 | 1 | 1 | 3 | 3.4 |
| D'（重标定） | 3 | 3 | 2 | 2 | 2 | 2.4 |

### 4.3 最终排序

1. **C-lite（mocap + weld 动态支撑）**：机制证据最强、力/穿透数据最干净，且**真实子集探针已通过**（穿透 0.9637→0.0128 mm）；剩余缺口是完整 Panda/目标盒/桌腿场景扩展。
2. **B + settle/手指净空修复**：真实场景已证明对主接触对有效，是风险最低的次优路线；必须与几何修复绑定。
3. **C 全动态支撑**：长期终极方案，当前性价比低。
4. **D' 重标定 Newton CollisionPipeline**：作为上游 #2106 修复或 C-lite 集成失败后的后备。
5. **A / A' / R0**：维持现状或直接放弃。

---

## 5. 当前状态与建议的下一步

0. **已在主仓库实现 C-lite 主支撑版（2026-08-18 下午）**
   - 新增 `BenchmarkConfig.support_config = "C2_CLITE"` 与 CLI `--support-config C2_CLITE`；
   - 振动平台与工作台改为动态刚体，由 fixed-root mocap driver + WELD equality constraint 驱动；mocap 每个 0.25 ms solver 子步更新；
   - 桌腿、目标盒与 Panda 浮动根暂保留原 kinematic 轨迹写入（Panda 根 free joint+weld 在 6 轴谱下会漂移，已回退）；
   - 官方 1 s 谱 seed=17 评定（四元数方向修复后）：`max_penetration_mm=0.233`（C2 基线 0.167），`max_wrist_force_n=0.0225 N`（C2 基线 36.78 N），>0.5 mm 帧占比 0；
   - 16 s 完整回合（修复方向后）：C2 基线同样 `grasp_table_contact` 失败（2.913 mm / 31.1 kN），C2_CLITE 更差（6.365 mm / 58.5 kN）。详细报告见 `docs/clite_implementation_report.md`。

1. **已完成：C-lite 真实场景最小集成探针**
   - 子集：真实工作台 box + 真实 YCB sugar_box@0.75 + 真实 C2 工作台测点运动；
   - 结果：kinematic 传送 0.9637 mm → C-lite mocap 4 kHz **0.0128 mm**，接触力 555.5 N → 477.5 N；
   - 证据：`out/penetration_experiments_20260818/08_clite_real_subset_probe.json`。

2. **下一步：把 C-lite 扩展到完整 ViBench 场景**
   - 将振动平台、Panda 浮动根、工作台、桌腿、目标盒改为动态刚体，各自用 mocap driver + WELD 约束跟踪原 C2 轨迹；
   - mocap 按 4 kHz 子步更新（1 kHz 更新会引入大约束冲击）；
   - 用 1 s official 谱 idle probe 记录**全接触对**穿透、腕力、拓扑；
   - 通过标准：所有接触对 ≤0.3 mm、力峰值不劣化、拓扑变化可解释。

3. 若完整场景扩展失败，退回 **B + 净空修复**，并同步启动 **D'** 的 Newton 参数标定。

4. 无论选哪条，先把 `penetration_probe()` 扩展为**全接触对**回归，防止“按下葫芦浮起瓢”。

---

## 6. 附录：实验文件与备份

- 备份：`~/Desktop/ViBench_backups/ViBench_code_backup_20260818_105442.tar.gz`
  - SHA-256：`1bca3cf66a357a3822c6bba26ebac27e0c9800ce4df55bbf97a9a0e4cc5d7498`
- 实验结果：`out/penetration_experiments_20260818/`
  - `01_official_margin_only_probe.json`
  - `02_official_margin_gap_probe.json`
  - `03_support_cadence_modified_tick_probe.json`
  - `04_support_cadence_fair_substep_probe.json`
  - `05_newton_pipeline_probe.json`
  - `06_nativeccd_off_probe.log` / `06_nativeccd_off_probe_summary.json`
  - `07_clite_minimal_probe_summary.json`
  - `08_clite_real_subset_probe.json` / `08_clite_real_subset_probe.log`
  - `SHA256SUMS`
- 早期分析文档：`docs/penetration_root_elimination_analysis.md`（含最小模型实验与根因分析）

> 说明：2026-08-18 下午已在 `src/` 实现 C-lite 主支撑版（`support_config="C2_CLITE"`）。实现范围与 1 s/16 s 评定结果见第 5 节；默认 C2 路径未变。
