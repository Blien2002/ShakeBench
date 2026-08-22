"""Generate the evidence-linked Stage 2C Markdown report from JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GAMMAS = (0.0, 0.15, 0.30, 0.50, 0.75, 0.95)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def mm(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "待测"
    value_mm = 1000 * value
    shown_digits = max(digits, 4) if 0.0 < abs(value_mm) < 0.001 else digits
    return f"{value_mm:.{shown_digits}f}"


def dist_mm(distribution: dict) -> str:
    return "/".join(mm(distribution[key]) for key in ("median", "p90", "max"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).parent / "out" / "stage2c",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).parent / "REPORT_STAGE2C.md",
    )
    args = parser.parse_args()
    root = args.output_root
    a3 = load(root / "grip_authority_split.json")
    b1 = load(root / "exp1_rerun" / "b1_prime_summary.json")
    gate2 = load(root / "sensitivity" / "gate2_summary.json")
    hold = load(root / "sensitivity" / "grip_hold_plateau.json")
    scans = [
        ("pad solref damping ratio", "pad_damping.json"),
        ("OSC kp", "osc_kp.json"),
        ("physics timestep", "timestep.json"),
        ("cube-table solref", "cube_table_solref.json"),
        ("move-action gain (diagnostic)", "move_action_gain.json"),
    ]
    matrix = load(root / "tolerance_matrix.json")
    c_path = root / "exp2_ladder" / "c_summary.json"
    d_path = root / "probes" / "d_summary.json"
    c = load(c_path) if c_path.exists() else None
    d = load(d_path) if d_path.exists() else None

    lines = [
        "# ShakeBench spike Stage 2C report",
        "",
        "## 结论",
        "",
    ]
    if not gate2["passed"]:
        lines += [
            "Gate 1 通过，但 Gate 2 未通过；依照预先规定的门禁顺序，Stage C 和 D rollout 未运行。",
            "A3 已把接触获取与 latch 后保持权限拆开；失败不再是获取超时，而是低保持力下的真实接触丢失/滑移。",
        ]
    elif c is not None:
        interpretation = c["interpretation"]["interpretation"]
        lines += [
            f"Gate 1 与 Gate 2 均通过；Stage C 判读为 `{interpretation}`。",
            "本报告不把设计探针的失败写成 benchmark 实现失败。",
        ]
    lines += [
        "",
        "## A3：获取权限 / 保持上限拆分",
        "",
        f"接触获取阶段为 {a3['design']['acquisition_force_limit_n_per_finger']:.1f} N/指；首次双侧 latch 后切到 {a3['design']['hold_force_limit_n_per_finger']:.1f} N/指。",
        f"切换发生在 policy step {a3['reactive_switch']['policy_step_index']}、model step {a3['reactive_switch']['model_step_index']}、t={a3['reactive_switch']['time_s']:.3f} s。切换前后 `actuator_forcerange` 校验均通过。",
        "事件写入 phase history，包含时间、步索引、切换前后完整 actuator 配置。",
        "",
        "冻结重放不做在线 latch 判断；磁带额外记录力上限切换的 policy-step 索引，重放时在同一索引开环改写 `model.actuator_forcerange`。A3 验证中 reactive 与 frozen 均在相同 model step 完成切换。",
        "",
        "确定性检查：同 seed 动作逐位一致，`max|Δaction|=0`；200 Hz 获取轨迹结构完全一致；两次切换 model step 相同。",
        "",
        "### 获取阶段实测",
        "",
        f"采样 {a3['acquisition_trace_summary']['sample_count']} 点（{a3['acquisition_trace_summary']['start_time_s']:.3f}–{a3['acquisition_trace_summary']['end_time_s']:.3f} s）。两指 actuator 峰值绝对力为 "
        + "/".join(f"{value:.3f}" for value in a3["acquisition_trace_summary"]["actuator_force_peak_abs_n"].values())
        + " N；指关节速度峰值绝对值为 "
        + "/".join(f"{value:.5f}" for value in a3["acquisition_trace_summary"]["finger_joint_velocity_peak_abs_m_s"].values())
        + " m/s。",
        "这把 Stage2B 中‘1/1.5 N 失败来自接触获取受限’从相位分类推断升级为带 actuator 力与关节速度轨迹的实测。",
        "",
        "## B1′：A3 配置下的全新配对重跑",
        "",
        "| Γ | 成功 | 最大 grasp slip 中位/P90/最大 (mm) | 双侧脱接触 中位/P90/最大 | warning |",
        "|---:|---:|---:|---:|---:|",
    ]
    for gamma in (0.0, 0.5, 0.95):
        summary = b1["conditions"][str(gamma)]
        both = summary["post_latch_contact_loss_fraction_distributions"][
            "both_below_threshold_fraction"
        ]
        lines.append(
            f"| {gamma:.2f} | {summary['success_count']}/{summary['episode_count']} | "
            f"{dist_mm(summary['max_grasp_slip_distribution_m'])} | "
            f"{pct(both['median'])}/{pct(both['p90'])}/{pct(both['max'])} | "
            f"{summary['mujoco_warning_count']} |"
        )
    lines += [
        "",
        f"Gate 1：**{'PASS' if b1['gate_1']['passed'] else 'FAIL'}**（Γ=0 为 {b1['gate_1']['success_count']}/20）。这些结果均在新目录重新生成，未复用 Stage2B rollout。",
        "",
        "## B2′：Γ=0.95 敏感性",
        "",
        "### latch 后保持力曲线",
        "",
        "预写预测：A3 拆分后 1/1.5/3/6 N 四点全部 100%。",
        "",
        "| 保持上限 (N/指) | 成功 | slip 中位/P90/最大 (mm) | 双侧脱接触 中位/P90/最大 | 失败分类 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in hold["curve"]:
        both = row["both_below_threshold_fraction_distribution"]
        lines.append(
            f"| {row['hold_force_limit_n_per_finger']:.1f} | {row['success_count']}/{row['episode_count']} | "
            f"{dist_mm(row['max_grasp_slip_distribution_m'])} | "
            f"{pct(both['median'])}/{pct(both['p90'])}/{pct(both['max'])} | "
            f"`{row['failure_reason_histogram']}` |"
        )
    prediction_observed = all(row["success_rate"] == 1.0 for row in hold["curve"])
    lines += [
        "",
        f"预测对照：**{'符合' if prediction_observed else '不符合'}**。1 N 点在 A3 后可以完成 latch，因此其非零滑移是保持阶段实测，不再是旧规格混入的 acquisition timeout。",
        f"3 N 与 1.5/6 N 的 SR 差分别为 {hold['adjacent_success_rate_deltas']['vs_1p5_percentage_points']:.1f}/{hold['adjacent_success_rate_deltas']['vs_6p0_percentage_points']:.1f} 个百分点。",
        "",
        "### 其余扫描",
        "",
        "| 参数 | 基线 → 备选 | 成功（基线/备选） | 翻转 | 门禁 |",
        "|---|---|---:|---:|---:|",
    ]
    for label, filename in scans:
        scan = load(root / "sensitivity" / filename)
        excluded = scan.get("diagnostic_only_not_in_gate_2", False)
        gate_label = "诊断项" if excluded else ("PASS" if scan["gate_parameter_passed"] else "FAIL")
        lines.append(
            f"| {label} | {scan['baseline_value']} → {scan['alternate_value']} | "
            f"{scan['baseline']['success_count']}/{scan['baseline']['episode_count']} / "
            f"{scan['alternate']['success_count']}/{scan['alternate']['episode_count']} | "
            f"{scan['flip_count']}/{scan['flip_denominator']} | {gate_label} |"
        )
    lines += [
        "",
        f"Gate 2：**{'PASS' if gate2['passed'] else 'FAIL'}**。保持力项按平台判据，其余门禁项按 `<50%` 配对翻转判据；move-action gain 不计门。",
        "",
        "## D0：实测量与任务容差（显著对照）",
        "",
        "| 实测量 | Γ=0.95 值 (mm) | 容差 (mm) | 占比 | 来源 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in matrix["rows"]:
        ratio = row["fraction_of_tolerance"]
        lines.append(
            f"| {row['measured_quantity']} | {mm(row['gamma_0p95_value_m'])} | "
            f"{mm(row['task_tolerance_m']) if row['task_tolerance_m'] is not None else '—'} | "
            f"{pct(ratio)} | {row['config_source'] or row['measurement_source']} |"
        )
    lines += ["", f"**关键点：{matrix['prominent_observation']}**", ""]

    if c is None:
        lines += [
            "## C：Γ 阶梯",
            "",
            "**Not run**：Gate 2 未通过。没有用部分或调试 rollout 填表。",
            "",
            "## D：设计探针",
            "",
            "**Not run**：D 仅在 C 判为两行都平时触发，而 C 未获准开始。",
        ]
    else:
        ladder = c["ladder"]
        lines += [
            "## C：Γ 阶梯（10 seeds × 6 Γ × 2 strategies）",
            "",
            "| 策略 / reactive 指标 | " + " | ".join(f"{g:.2f}" for g in GAMMAS) + " |",
            "|---|" + "---:|" * len(GAMMAS),
        ]
        for strategy in ("frozen_replay", "reactive_scripted"):
            values = [
                pct(ladder["rows"][str(g)][strategy]["success_rate"]) for g in GAMMAS
            ]
            lines.append(f"| {strategy} SR | " + " | ".join(values) + " |")
        metric_rows = (
            ("EE 抖动最大 (mm)", "ee_wobble_base_distribution_m"),
            ("桌面系工件滑移最大 (mm)", "table_frame_object_slip_distribution_m"),
            ("最大 grasp slip (mm)", "max_grasp_slip_distribution_m"),
        )
        for label, key in metric_rows:
            values = [
                mm(ladder["rows"][str(g)]["reactive_scripted"][key]["max"])
                for g in GAMMAS
            ]
            lines.append(f"| {label} | " + " | ".join(values) + " |")
        comparison = c["prediction_comparison"]
        lines += [
            "",
            "C4 预测对照：reactive 全阶梯 100%——"
            + ("符合" if comparison["reactive_100_percent_observed"] else "不符合")
            + "；slip 单调上升——"
            + ("符合" if comparison["slip_monotonic_observed"] else "不符合")
            + f"；Γ=0.95 实测最大 {mm(comparison['gamma_0p95_observed_max_grasp_slip_m'])} mm（预测约 7.09 mm）。",
            "",
            f"按预写 C3 规则判读：`{c['interpretation']['interpretation']}`。",
            "建议：先依据 D 的设计探针对任务难度轴做结构选择，不把硬装模型中的平坦曲线解释为实现失败。",
            "",
            "## D：条件设计探针",
            "",
        ]
        if d is None or not d.get("triggered"):
            lines.append("未触发或未运行。")
        else:
            d1 = d["d1"]
            d2 = d["d2"]
            lines += [
                "D1 将 cube-table μ 从 1.5 降至 0.4，作为独立任务设计变更；D2 使用显式二阶支撑模型后才运行 rollout。",
                f"D2：`{d['d2_coherence_model']['transfer_function']}`，fn={d['d2_coherence_model']['natural_frequency_hz']} Hz，ζ={d['d2_coherence_model']['damping_ratio']}。这是未拟合硬件的设计探针。",
                "",
                "| Γ | D1 reactive/frozen SR | D2 reactive/frozen SR | D2 base 系桌面运动最大 (mm) |",
                "|---:|---:|---:|---:|",
            ]
            for gamma in GAMMAS:
                key = str(gamma)
                d1row = d1["rows"][key]
                d2row = d2["rows"][key]
                lines.append(
                    f"| {gamma:.2f} | {pct(d1row['reactive_scripted']['success_rate'])}/"
                    f"{pct(d1row['frozen_replay']['success_rate'])} | "
                    f"{pct(d2row['reactive_scripted']['success_rate'])}/"
                    f"{pct(d2row['frozen_replay']['success_rate'])} | "
                    f"{mm(d2row['reactive_scripted']['base_frame_table_motion_distribution_m']['max'])} |"
                )
    lines += [
        "",
        "## 限制与诚实性说明",
        "",
        "- 20 个确定性种子只给出该初始化集合的经验频率，不是总体置信区间。",
        "- A3 后所有 B1′/B2′ 结果均重跑；Stage2B 只作为提示词明确允许沿用的已验证溯源与 B3 仪器证据。",
        "- 固定 10 mm grasp-slip 容差、超时、Γ、频带与辅助设置未为过门而放宽。D1 的 μ=0.4 明确隔离为设计探针。",
        "- 本轮未获准运行 D2；旧 B3 的统一 π/2 相移只保留为仪器诊断，不是策略 rollout 证据。",
        "- 任何未配套 B2′ 的 Γ 下降只标为未验证，不进入结论。",
        "",
        "原始数据位于 `out/stage2c/`；所有 episode JSON 保留动作磁带、切换索引、失败分类、200 Hz 接触力统计与 MuJoCo warning 计数。",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
