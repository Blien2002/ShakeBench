# 基线状态

更新日期：2026-08-18

## 已作废（legacy 虚拟测点模型产物）

以下文件产生自 `arm_mount_xy_m=(0.75,-0.45) / table_mount_xy_m=(-0.75,0.45)` 的旧模型，不得再作为通过证据引用：

- `docs/penetration_baseline.json`
- `docs/penetration_official_final.json`
- `docs/visual_baseline.json`
- `docs/third_round_*` 中的历史视觉/成功 JSON
- `out/benchmark_v2_*`
- `out/p0_5substep_spectral_probe.json`（0.665 mm `robot_link<->platen` 伪接触探针）

作废原因：`support_model_refactor`（见 `docs/support_model_implementation_plan.md` v1.2）。

## 新基线（阶段 A 硬装模型）

| 产物 | 说明 |
|---|---|
| `out/stage_a_off_smoke.json` | official 1000 Hz × 5，vibration=off，0.2 s smoke |
| `out/stage_a_spectral_probe.json` | official 1000 Hz × 5，seed=17，1 s settle 探针，最大穿透 0.242 mm `workpiece<->worktable` |
| `out/stage_a_16s_seed17.json` | 阶段 A 完整 16 s official 回合，seed=17：`lifted=true`，随后 `grasp_z_guard_triggered`，`success=false`；最大穿透 0.955 mm `workpiece<->worktable`，`support_geometry_valid=true`。诚实失败，不用于计分 |
| `out/shakebench_stage_a_latest.mp4/.json` | 阶段 A 最新录制：1280×720 H.264，6.1 s / 183 帧；`grasp_contact_timeout`，`success=false`，`support_geometry_valid=true` |
| `out/stage_a_clite_16s_seed17.json` | C2_CLITE + iterations=120 的 16 s 实验：运行 5 min 尚未越过 settle 阶段，已终止，无 JSON；C2_CLITE 手动子步循环当前吞吐不足以跑官方完整回合 |

### 数值地板（1 s settle，只振动不操作）

普通 C2 1000 Hz × 5：seed 17/31/47/73/101 的最大穿透分别为 0.259 / 0.248 / 0.158 / 0.202 / 0.209 mm，**最大 0.259 mm**，未达到 D2 计分资格（≤ 0.336/3 ≈ 0.112 mm）。

C2_CLITE 1000 Hz × 4 + `solver_iterations=50`（当前 official 默认）：seed 17/31/47/73/101 分别为 0.102 / 0.082 / 0.082 / 0.067 / 0.100 mm，**全部低于 0.112 mm 计分资格线**。该配置 1 s 墙钟约 50 s，16 s 约 13 分钟。C2_CLITE 1000 Hz × 5 早期数据（0.064–0.123 mm）与普通 C2（最大 0.259 mm）见上表。

硬装接触拓扑：386 Newton shapes / 29 MJWarp geometries / 339 candidate pairs。
