"""Final ShakeBench isolated-table force, tracking, and MILD-gap experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from isolator import IsolatorParameters
from run_iso_stage1 import measure_spectral
from run_iso_stage3 import summarize
from run_stage2b import Condition, _distribution, run_condition
from run_stage2c import run_frozen_episode


SPIKE_ROOT = Path(__file__).resolve().parent
ISO_ROOT = SPIKE_ROOT / "out" / "iso"
OUTPUT_ROOT = SPIKE_ROOT / "out" / "iso_final"
SEEDS_10 = list(range(10))
SEEDS_20 = list(range(20))
HOLD_FORCES_N = (3.0, 6.0, 12.0, 20.0)
LADDER_GAMMAS = (0.0, 0.30, 0.50, 0.75, 0.95)
MILD_GAMMAS = (0.15, 0.30, 0.50, 0.75, 0.95)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _strong_parameters() -> IsolatorParameters:
    data = _read(ISO_ROOT / "operating_points.json")["points"]["STRONG"]["isolator"]
    return IsolatorParameters(data["natural_frequency_hz"], data["damping_ratio"])


def _selected_mu() -> float:
    return float(_read(ISO_ROOT / "mu_sweep.json")["selection"]["selected_mu"])


def _condition(gamma: float, hold_force_n: float, **changes) -> Condition:
    values = {
        "gamma": gamma,
        "cube_table_sliding_mu": _selected_mu(),
        "gripper_force_limit_n": hold_force_n,
    }
    values.update(changes)
    return Condition(**values)


def _variant(hold_force_n: float, suffix: str = "") -> str:
    value = f"iso_final_strong_mu_{_slug(_selected_mu())}_hold_{_slug(hold_force_n)}n"
    return f"{value}_{suffix}" if suffix else value


def _force_dir(output_root: Path, force_n: float) -> Path:
    return output_root / "hold_saturation_runs" / f"hold_{_slug(force_n)}n"


def run_hold_saturation(output_root: Path) -> dict:
    parameters = _strong_parameters()
    rows = []
    for force_n in HOLD_FORCES_N:
        print(f"hold saturation STRONG Gamma=.95 force={force_n:g} N", flush=True)
        condition = _condition(0.95, force_n)
        _legacy, episodes = run_condition(
            condition,
            SEEDS_20,
            _force_dir(output_root, force_n),
            table_isolator=parameters,
            environment_variant=_variant(force_n, "saturation"),
        )
        summary = summarize(episodes)
        rows.append(
            {
                "hold_force_n_per_finger": force_n,
                "condition": asdict(condition),
                "summary": summary,
            }
        )

    comparisons = []
    selected = None
    for lower, higher in zip(rows[:-1], rows[1:], strict=True):
        lower_summary = lower["summary"]
        higher_summary = higher["summary"]
        sr_delta_pp = 100.0 * abs(
            lower_summary["success_rate"] - higher_summary["success_rate"]
        )
        lower_p90 = lower_summary["placement_error_distribution_m"]["p90"]
        higher_p90 = higher_summary["placement_error_distribution_m"]["p90"]
        placement_ratio = max(lower_p90, higher_p90) / max(
            min(lower_p90, higher_p90), np.finfo(float).eps
        )
        # The 20-seed grid quantizes SR in exact 5 pp increments.  Keep a tiny
        # numerical guard so an exact 10 pp difference is not admitted by
        # binary floating-point roundoff as 9.999999999999998.
        sr_passed = sr_delta_pp < 10.0 - 1.0e-9
        passed = sr_passed and placement_ratio < 1.3
        comparison = {
            "candidate_hold_force_n_per_finger": lower["hold_force_n_per_finger"],
            "next_higher_force_n_per_finger": higher["hold_force_n_per_finger"],
            "absolute_success_rate_delta_percentage_points": sr_delta_pp,
            "symmetric_placement_p90_ratio": placement_ratio,
            "success_rate_criterion_lt_10pp": sr_passed,
            "placement_p90_ratio_criterion_lt_1p3": placement_ratio < 1.3,
            "passed": passed,
        }
        comparisons.append(comparison)
        if selected is None and passed:
            selected = lower["hold_force_n_per_finger"]

    result = {
        "schema": "shakebench_spike.iso_final.hold_saturation.v1",
        "created_utc": _now(),
        "condition": {
            "operating_point": "STRONG",
            "isolator": parameters.as_dict(),
            "gamma": 0.95,
            "cube_table_sliding_mu": _selected_mu(),
            "acquisition_force_n_per_finger": 20.0,
            "seed_count_per_force": len(SEEDS_20),
            "seeds": SEEDS_20,
        },
        "criterion": {
            "comparison_scope": "candidate versus next higher force only",
            "success_rate": "absolute difference < 10 percentage points",
            "placement_p90": "symmetric larger/smaller ratio < 1.3",
            "both_required": True,
        },
        "rows": rows,
        "adjacent_comparisons": comparisons,
        "saturated_within_panda_limit": selected is not None,
        "selected_hold_force_n_per_finger": selected,
        "stop_after_1a": selected is None,
        "mujoco_warning_count": sum(
            row["summary"]["mujoco_warning_count"] for row in rows
        ),
    }
    _write(output_root / "hold_saturation_strong.json", result)
    return result


def _selected_hold_force(output_root: Path) -> float:
    result = _read(output_root / "hold_saturation_strong.json")
    value = result.get("selected_hold_force_n_per_finger")
    if value is None:
        raise RuntimeError("hold force did not saturate within the 20 N Panda limit")
    return float(value)


def _tape_dir(output_root: Path) -> Path:
    return output_root / "tapes_v2" / "reactive_scripted"


def _old_tape(seed: int) -> dict:
    return _read(
        ISO_ROOT
        / "ladder"
        / "STRONG"
        / "gamma_0p0"
        / "reactive_scripted"
        / f"seed_{seed:03d}.json"
    )


def record_tapes_v2(output_root: Path) -> dict:
    hold_force_n = _selected_hold_force(output_root)
    condition = _condition(0.0, hold_force_n)
    _legacy, episodes = run_condition(
        condition,
        SEEDS_10,
        _tape_dir(output_root),
        table_isolator=_strong_parameters(),
        environment_variant=_variant(hold_force_n, "tape_v2"),
    )
    rows = []
    for new in episodes:
        old = _old_tape(int(new["seed"]))
        old_actions = np.asarray(old["actions"], dtype=np.float64)
        new_actions = np.asarray(new["actions"], dtype=np.float64)
        common = min(len(old_actions), len(new_actions))
        old_switch = old["gripper_force_switch_history"][0]["policy_step_index"]
        new_switch = new["gripper_force_switch_history"][0]["policy_step_index"]
        pre_switch_common = min(common, old_switch, new_switch)
        rows.append(
            {
                "seed": new["seed"],
                "old_action_count": len(old_actions),
                "new_action_count": len(new_actions),
                "action_count_difference": len(new_actions) - len(old_actions),
                "common_prefix_policy_steps": common,
                "max_abs_action_difference_common_prefix": float(
                    np.max(np.abs(new_actions[:common] - old_actions[:common]))
                ),
                "max_abs_action_difference_before_force_switch": float(
                    np.max(
                        np.abs(
                            new_actions[:pre_switch_common]
                            - old_actions[:pre_switch_common]
                        )
                    )
                ),
                "old_force_switch_policy_step_index": old_switch,
                "new_force_switch_policy_step_index": new_switch,
                "force_switch_index_difference": new_switch - old_switch,
            }
        )
    action_differences = [
        row["max_abs_action_difference_common_prefix"] for row in rows
    ]
    pre_switch_differences = [
        row["max_abs_action_difference_before_force_switch"] for row in rows
    ]
    result = {
        "schema": "shakebench_spike.iso_final.tapes_v2.v1",
        "created_utc": _now(),
        "operating_point": "STRONG",
        "gamma": 0.0,
        "selected_hold_force_n_per_finger": hold_force_n,
        "seed_count": len(SEEDS_10),
        "new_tape_summary": summarize(episodes),
        "old_vs_new": {
            "rows": rows,
            "common_prefix_max_abs_action_difference_distribution": _distribution(
                action_differences
            ),
            "pre_force_switch_max_abs_action_difference_distribution": _distribution(
                pre_switch_differences
            ),
            "max_abs_action_count_difference": max(
                abs(row["action_count_difference"]) for row in rows
            ),
            "max_abs_force_switch_index_difference": max(
                abs(row["force_switch_index_difference"]) for row in rows
            ),
            "previous_cross_operating_point_baseline": "median 0.0062-0.0092",
        },
        "mujoco_warning_count": sum(
            int(episode["mujoco_warning_count"]) for episode in episodes
        ),
    }
    _write(output_root / "tapes_v2" / "comparison.json", result)
    return result


def _load_tapes_v2(output_root: Path) -> list[dict]:
    paths = [_tape_dir(output_root) / f"seed_{seed:03d}.json" for seed in SEEDS_10]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("new hold-force Gamma=0 tapes are incomplete")
    return [_read(path) for path in paths]


def _ladder_dir(output_root: Path, gamma: float, strategy: str) -> Path:
    return output_root / "ladder_strong_v2" / f"gamma_{_slug(gamma)}" / strategy


def _load_or_run_frozen(
    path: Path,
    *,
    seed: int,
    gamma: float,
    tape: dict,
    hold_force_n: float,
) -> dict:
    parameters = _strong_parameters()
    expected_isolator = parameters.as_dict()
    variant = _variant(hold_force_n, "ladder_v2")
    if path.is_file():
        cached = _read(path)
        if (
            cached.get("gamma") == gamma
            and cached.get("source_tape_seed") == seed
            and cached.get("environment_variant") == variant
            and cached.get("table_isolator") == expected_isolator
        ):
            print(
                f"resume frozen seed={seed:03d} gamma={gamma:.2f} "
                f"success={cached['success']}",
                flush=True,
            )
            return cached
    episode = run_frozen_episode(
        seed,
        gamma,
        tape,
        table_isolator=parameters,
        environment_variant=variant,
    )
    _write(path, episode)
    print(
        f"frozen seed={seed:03d} gamma={gamma:.2f} success={episode['success']} "
        f"failure={episode['failure_reason']}",
        flush=True,
    )
    return episode


def run_ladder_strong_v2(output_root: Path) -> dict:
    hold_force_n = _selected_hold_force(output_root)
    tapes = _load_tapes_v2(output_root)
    parameters = _strong_parameters()
    rows = {}
    for gamma in LADDER_GAMMAS:
        if gamma == 0.0:
            reactive = tapes
        else:
            _legacy, reactive = run_condition(
                _condition(gamma, hold_force_n),
                SEEDS_10,
                _ladder_dir(output_root, gamma, "reactive_scripted"),
                table_isolator=parameters,
                environment_variant=_variant(hold_force_n, "ladder_v2"),
            )
        frozen = []
        for seed, tape in zip(SEEDS_10, tapes, strict=True):
            frozen.append(
                _load_or_run_frozen(
                    _ladder_dir(output_root, gamma, "frozen_replay")
                    / f"seed_{seed:03d}.json",
                    seed=seed,
                    gamma=gamma,
                    tape=tape,
                    hold_force_n=hold_force_n,
                )
            )
        rows[str(gamma)] = {
            "reactive_scripted": summarize(reactive),
            "frozen_replay": summarize(frozen),
            "paired": [
                {
                    "seed": seed,
                    "reactive_success": reactive_episode["success"],
                    "frozen_success": frozen_episode["success"],
                    "placement_error_delta_frozen_minus_reactive_m": (
                        frozen_episode["placement"]["translation_error_m"]
                        - reactive_episode["placement"]["translation_error_m"]
                    ),
                }
                for seed, reactive_episode, frozen_episode in zip(
                    SEEDS_10, reactive, frozen, strict=True
                )
            ],
        }
    baseline_sr = rows["0.0"]["reactive_scripted"]["success_rate"]
    decline = next(
        (
            gamma
            for gamma in LADDER_GAMMAS[1:]
            if rows[str(gamma)]["reactive_scripted"]["success_rate"] < baseline_sr
        ),
        None,
    )
    result = {
        "schema": "shakebench_spike.iso_final.ladder_strong_v2.v1",
        "created_utc": _now(),
        "operating_point": "STRONG",
        "isolator": parameters.as_dict(),
        "selected_mu": _selected_mu(),
        "selected_hold_force_n_per_finger": hold_force_n,
        "gammas": list(LADDER_GAMMAS),
        "seeds": SEEDS_10,
        "rows": rows,
        "first_reactive_success_rate_decline_gamma": decline,
        "mujoco_warning_count": sum(
            rows[str(gamma)][strategy]["mujoco_warning_count"]
            for gamma in LADDER_GAMMAS
            for strategy in ("reactive_scripted", "frozen_replay")
        ),
    }
    _write(output_root / "ladder_strong_v2" / "summary.json", result)
    return result


def _baseline_for_attribution(
    output_root: Path, gamma: float, hold_force_n: float
) -> list[dict]:
    if gamma == 0.95:
        directory = _force_dir(output_root, hold_force_n)
        episodes = [_read(directory / f"seed_{seed:03d}.json") for seed in SEEDS_20]
        if all(
            episode.get("condition") == asdict(_condition(gamma, hold_force_n))
            for episode in episodes
        ):
            return episodes
    _legacy, episodes = run_condition(
        _condition(gamma, hold_force_n),
        SEEDS_20,
        _ladder_dir(output_root, gamma, "reactive_scripted"),
        table_isolator=_strong_parameters(),
        environment_variant=_variant(hold_force_n, "ladder_v2"),
    )
    return episodes


def run_attribution_v2(output_root: Path) -> dict:
    ladder = _read(output_root / "ladder_strong_v2" / "summary.json")
    gamma = ladder["first_reactive_success_rate_decline_gamma"]
    hold_force_n = _selected_hold_force(output_root)
    if gamma is None:
        result = {
            "schema": "shakebench_spike.iso_final.attribution_v2.v1",
            "created_utc": _now(),
            "status": "not_run_no_reactive_success_rate_decline",
            "reason": "the new STRONG reactive ladder had no success-rate decline",
            "single_parameter_checks": {},
            "any_parameter_bound": False,
            "mujoco_warning_count": 0,
        }
        _write(output_root / "attribution_v2" / "summary.json", result)
        return result

    gamma = float(gamma)
    baseline = _baseline_for_attribution(output_root, gamma, hold_force_n)
    baseline_summary = summarize(baseline)
    baseline_failures = [episode for episode in baseline if not episode["success"]]
    if not baseline_failures:
        raise RuntimeError("attribution decline condition has no failures in 20 seeds")

    doubled_force_n = min(2.0 * hold_force_n, 20.0)
    specs = (
        (
            "hold_force_x2",
            _condition(gamma, doubled_force_n),
            "gripper_force_limit_n_per_finger",
            hold_force_n,
            doubled_force_n,
            doubled_force_n != 2.0 * hold_force_n,
        ),
        (
            "osc_kp_150_to_300",
            _condition(gamma, hold_force_n, osc_kp=300.0),
            "osc_kp",
            150.0,
            300.0,
            False,
        ),
        (
            "move_gain_4_to_3",
            _condition(gamma, hold_force_n, move_action_gain=3.0),
            "move_action_gain",
            4.0,
            3.0,
            False,
        ),
    )
    checks = {}
    parameters = _strong_parameters()
    baseline_by_seed = {int(episode["seed"]): episode for episode in baseline}
    for slug, condition, parameter, base_value, alt_value, capped in specs:
        print(f"attribution v2 {slug}", flush=True)
        _legacy, alternate = run_condition(
            condition,
            SEEDS_20,
            output_root / "attribution_v2" / slug / "reactive_scripted",
            table_isolator=parameters,
            environment_variant=_variant(hold_force_n, f"attribution_{slug}"),
        )
        alternate_summary = summarize(alternate)
        paired = []
        for episode in alternate:
            base = baseline_by_seed[int(episode["seed"])]
            paired.append(
                {
                    "seed": episode["seed"],
                    "baseline_success": base["success"],
                    "alternate_success": episode["success"],
                    "baseline_failure_reason": base["failure_reason"],
                    "alternate_failure_reason": episode["failure_reason"],
                    "baseline_placement_error_m": base["placement"]["translation_error_m"],
                    "alternate_placement_error_m": episode["placement"]["translation_error_m"],
                }
            )
        failed_pairs = [row for row in paired if not row["baseline_success"]]
        rescued = sum(bool(row["alternate_success"]) for row in failed_pairs)
        rescue_fraction = rescued / len(failed_pairs)
        aggregate_improved = (
            alternate_summary["success_rate"] > baseline_summary["success_rate"]
            and alternate_summary["placement_error_distribution_m"]["p90"]
            < baseline_summary["placement_error_distribution_m"]["p90"]
        )
        bound = aggregate_improved and rescue_fraction >= 0.50
        checks[slug] = {
            "parameter": parameter,
            "baseline_value": base_value,
            "alternate_value": alt_value,
            "alternate_was_capped_at_panda_limit": capped,
            "baseline": baseline_summary,
            "alternate": alternate_summary,
            "paired_results": paired,
            "baseline_failure_count": len(failed_pairs),
            "baseline_failures_rescued_count": rescued,
            "baseline_failure_rescue_fraction": rescue_fraction,
            "aggregate_improved": aggregate_improved,
            "effect_parameter_bound": bound,
        }
    result = {
        "schema": "shakebench_spike.iso_final.attribution_v2.v1",
        "created_utc": _now(),
        "status": "completed",
        "effect_condition": {
            "operating_point": "STRONG",
            "gamma": gamma,
            "strategy": "reactive_scripted",
            "selected_hold_force_n_per_finger": hold_force_n,
            "baseline_success_rate": baseline_summary["success_rate"],
            "baseline_seed_count": len(SEEDS_20),
        },
        "binding_rule": (
            "alternate must improve both aggregate success rate and placement P90, "
            "and rescue at least 50% of baseline failures; worsening never binds"
        ),
        "single_parameter_checks": checks,
        "any_parameter_bound": any(
            check["effect_parameter_bound"] for check in checks.values()
        ),
        "mujoco_warning_count": sum(
            check["alternate"]["mujoco_warning_count"] for check in checks.values()
        ),
    }
    _write(output_root / "attribution_v2" / "summary.json", result)
    return result


def _tracking_from_episode(episode: dict) -> dict:
    trace = episode.get("eef_command_tracking_trace_200hz")
    if not trace:
        return {
            "status": "unavailable",
            "reason": "episode lacks time/phase/error EE tracking samples",
        }
    times = np.asarray([row["time_s"] for row in trace], dtype=np.float64)
    phases = [row["phase"] for row in trace]
    errors = np.asarray([row["error_m"] for row in trace], dtype=np.float64)
    transition_times = [
        times[index]
        for index in range(1, len(trace))
        if phases[index] != phases[index - 1]
    ]
    keep = np.ones(len(trace), dtype=bool)
    for transition_time in transition_times:
        keep &= np.abs(times - transition_time) > 0.25
    retained = errors[keep]
    excluded_fraction = 1.0 - float(np.mean(keep))
    return {
        "status": "computed",
        "sample_count": len(trace),
        "retained_sample_count": int(np.sum(keep)),
        "excluded_sample_count": int(np.sum(~keep)),
        "excluded_sample_fraction": excluded_fraction,
        "phase_transition_count": len(transition_times),
        "phase_transition_times_s": [float(value) for value in transition_times],
        "exclusion_half_window_s": 0.25,
        "tracking_error_m": _distribution(retained.tolist()),
        "metric_applicable": excluded_fraction <= 0.30,
    }


def recompute_ee_tracking(output_root: Path) -> dict:
    cells = {}
    unavailable = []
    for gamma in LADDER_GAMMAS:
        gamma_rows = {}
        for strategy in ("reactive_scripted", "frozen_replay"):
            episodes = []
            for seed in SEEDS_10:
                if gamma == 0.0 and strategy == "reactive_scripted":
                    path = _tape_dir(output_root) / f"seed_{seed:03d}.json"
                else:
                    path = _ladder_dir(output_root, gamma, strategy) / f"seed_{seed:03d}.json"
                episode_result = _tracking_from_episode(_read(path))
                episode_result["seed"] = seed
                episodes.append(episode_result)
                if episode_result["status"] != "computed":
                    unavailable.append(str(path))
            computed = [row for row in episodes if row["status"] == "computed"]
            gamma_rows[strategy] = {
                "episodes": episodes,
                "computed_episode_count": len(computed),
                "median_error_episode_distribution_m": _distribution(
                    [row["tracking_error_m"]["median"] for row in computed]
                ),
                "p90_error_episode_distribution_m": _distribution(
                    [row["tracking_error_m"]["p90"] for row in computed]
                ),
                "max_error_episode_distribution_m": _distribution(
                    [row["tracking_error_m"]["max"] for row in computed]
                ),
                "excluded_sample_fraction_distribution": _distribution(
                    [row["excluded_sample_fraction"] for row in computed]
                ),
                "metric_applicable_episode_count": sum(
                    bool(row["metric_applicable"]) for row in computed
                ),
            }
        cells[str(gamma)] = gamma_rows

    old_probe_paths = [
        ISO_ROOT
        / "ladder"
        / "STRONG"
        / "gamma_0p95"
        / strategy
        / "seed_000.json"
        for strategy in ("reactive_scripted", "frozen_replay")
    ]
    old_audit = []
    for path in old_probe_paths:
        episode = _read(path)
        old_audit.append(
            {
                "path": str(path.relative_to(SPIKE_ROOT)),
                "has_full_tracking_trace": bool(
                    episode.get("eef_command_tracking_trace_200hz")
                ),
                "available_related_fields": [
                    key
                    for key in (
                        "eef_command_tracking_error_m",
                        "acquisition_trace_200hz",
                        "target_position_b_by_policy_step",
                    )
                    if key in episode
                ],
            }
        )
    result = {
        "schema": "shakebench_spike.iso_final.ee_tracking_recomputed.v1",
        "created_utc": _now(),
        "method": {
            "source": "authorized iso_final STRONG ladder 200 Hz tracking traces",
            "excluded_window": "within +/-0.25 s of every observed phase change",
            "statistics": ["median", "p90", "max"],
            "downgrade_if_excluded_fraction_gt": 0.30,
        },
        "old_iso_dataset_audit": {
            "status": "not_recomputable_without_new_simulation",
            "reason": (
                "old episodes persisted only aggregate tracking statistics; the "
                "200 Hz acquisition trace has finger force/joints but no EE pose or target"
            ),
            "probes": old_audit,
            "new_rollout_for_old_dataset_was_not_run": True,
        },
        "cells": cells,
        "unavailable_new_episode_paths": unavailable,
        "mujoco_warning_count": 0,
    }
    _write(output_root / "ee_tracking_recomputed.json", result)
    return result


def run_mild_gap_sweep(output_root: Path) -> dict:
    frequencies = (13.0, 12.0, 11.0, 10.0)
    rows = []
    for frequency in frequencies:
        parameters = IsolatorParameters(frequency, 0.10)
        for gamma in MILD_GAMMAS:
            print(f"MILD gap fn={frequency:g} Hz Gamma={gamma:.2f}", flush=True)
            row = measure_spectral(
                gamma=gamma,
                parameters=parameters,
                cube_table_sliding_mu=1.5,
            )
            row["configuration_name"] = f"fn{int(frequency)}_zeta0p10"
            rows.append(row)
    candidates = [row for row in rows if row["gamma"] == 0.5]
    in_range = [
        row
        for row in candidates
        if 0.0008 <= row["base_frame_table_motion_peak_about_mean_m"] <= 0.0020
    ]
    pool = in_range or candidates
    selected = min(
        pool,
        key=lambda row: abs(row["base_frame_table_motion_peak_about_mean_m"] - 0.0015),
    )
    value = selected["base_frame_table_motion_peak_about_mean_m"]
    result = {
        "schema": "shakebench_spike.iso_final.mild_gap_sweep.v1",
        "created_utc": _now(),
        "policy_rollouts_run": False,
        "configuration": {
            "natural_frequencies_hz": list(frequencies),
            "damping_ratio": 0.10,
            "gammas": list(MILD_GAMMAS),
            "cube_table_sliding_mu": 1.5,
            "reason_for_mu": "matches the prior stage-1b instrument sweep",
        },
        "rows": rows,
        "selection": {
            "rule": (
                "Gamma=0.50 base-frame table motion in 0.8-2.0 mm and closest "
                "to 1.5 mm; otherwise closest with deviation reported"
            ),
            "configuration_name": selected["configuration_name"],
            "base_frame_table_motion_m": value,
            "within_predeclared_interval": bool(in_range),
            "deviation_from_interval_m": (
                0.0 if in_range else min(abs(value - 0.0008), abs(value - 0.0020))
            ),
            "isolator": selected["isolator"],
        },
        "mujoco_warning_count": sum(row["mujoco_warning_count"] for row in rows),
    }
    _write(output_root / "mild_gap_sweep.json", result)
    return result


def main() -> int:
    logger = logging.getLogger("robosuite_logs")
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("saturation", "tapes", "ladder", "attribution", "tracking", "mild", "all"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    if args.stage in ("saturation", "all"):
        saturation = run_hold_saturation(args.output)
        print(json.dumps(saturation["adjacent_comparisons"], indent=2), flush=True)
        if saturation["stop_after_1a"]:
            print("20 N did not establish saturation; stopping after task 1a", flush=True)
            return 2
    if args.stage in ("tapes", "all"):
        tapes = record_tapes_v2(args.output)
        print(json.dumps(tapes["old_vs_new"], indent=2), flush=True)
    if args.stage in ("ladder", "all"):
        ladder = run_ladder_strong_v2(args.output)
        print(
            json.dumps(
                {
                    "first_reactive_success_rate_decline_gamma": ladder[
                        "first_reactive_success_rate_decline_gamma"
                    ]
                },
                indent=2,
            ),
            flush=True,
        )
    if args.stage in ("attribution", "all"):
        attribution = run_attribution_v2(args.output)
        print(json.dumps(attribution, indent=2), flush=True)
    if args.stage in ("tracking", "all"):
        tracking = recompute_ee_tracking(args.output)
        print(
            json.dumps(
                {
                    "unavailable_new_episode_count": len(
                        tracking["unavailable_new_episode_paths"]
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    if args.stage in ("mild", "all"):
        mild = run_mild_gap_sweep(args.output)
        print(json.dumps(mild["selection"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
