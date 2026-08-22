"""Stage 3 isolated-worktable policy ladder and attribution runner."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from isolator import IsolatorParameters
from run_stage2b import Condition, _distribution, run_condition
from run_stage2c import run_frozen_episode


OUTPUT_ROOT = Path(__file__).resolve().parent / "out" / "iso"
POINTS = ("RIGID", "MILD", "STRONG")
GAMMAS = (0.0, 0.30, 0.50, 0.75, 0.95)
SEEDS = list(range(10))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configuration(output_root: Path) -> tuple[dict, float]:
    operating = _read(output_root / "operating_points.json")
    mu = float(_read(output_root / "mu_sweep.json")["selection"]["selected_mu"])
    return operating, mu


def _isolator(operating: dict, point: str) -> IsolatorParameters | None:
    data = operating["points"][point]["isolator"]
    if data is None:
        return None
    return IsolatorParameters(data["natural_frequency_hz"], data["damping_ratio"])


def _variant(point: str, mu: float) -> str:
    return f"iso_{point.lower()}_mu_{_slug(mu)}"


def _episode_dir(output_root: Path, point: str, gamma: float, strategy: str) -> Path:
    return output_root / "ladder" / point / f"gamma_{_slug(gamma)}" / strategy


def summarize(episodes: list[dict]) -> dict:
    success_count = sum(bool(episode["success"]) for episode in episodes)
    failures = Counter(
        episode["failure_reason"] for episode in episodes if not episode["success"]
    )

    def episode_max(path: tuple[str, ...]) -> list[float]:
        values = []
        for episode in episodes:
            value = episode
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return values

    return {
        "episode_count": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "placement_error_distribution_m": _distribution(
            episode_max(("placement", "translation_error_m"))
        ),
        "placement_within_70mm_count": sum(
            bool(episode["placement"]["within_tolerance"]) for episode in episodes
        ),
        "eef_command_tracking_episode_max_distribution_m": _distribution(
            episode_max(("eef_command_tracking_error_m", "max"))
        ),
        "eef_command_tracking_episode_p90_distribution_m": _distribution(
            episode_max(("eef_command_tracking_error_m", "p90"))
        ),
        "eef_command_tracking_episode_median_distribution_m": _distribution(
            episode_max(("eef_command_tracking_error_m", "median"))
        ),
        "max_grasp_slip_distribution_m": _distribution(
            [float(episode["max_grasp_slip_m"]) for episode in episodes]
        ),
        "slip_decomposition": {
            "translation_episode_max_distribution_m": _distribution(
                episode_max(("grasp_slip_decomposition", "translation_m", "max"))
            ),
            "rotation_episode_max_distribution_rad": _distribution(
                episode_max(("grasp_slip_decomposition", "rotation_rad", "max"))
            ),
            "any_contact_loss_fraction_distribution": _distribution(
                episode_max(
                    (
                        "grasp_slip_decomposition",
                        "contact_loss",
                        "any_finger_below_threshold_fraction",
                    )
                )
            ),
            "bilateral_contact_loss_fraction_distribution": _distribution(
                episode_max(
                    (
                        "grasp_slip_decomposition",
                        "contact_loss",
                        "both_fingers_below_threshold_fraction",
                    )
                )
            ),
        },
        "table_frame_object_slip_distribution_m": _distribution(
            [float(episode["obj_slip_on_table_m"]) for episode in episodes]
        ),
        "failure_reason_histogram": dict(sorted(failures.items())),
        "mujoco_warning_count": sum(
            int(episode["mujoco_warning_count"]) for episode in episodes
        ),
    }


def record_tapes(output_root: Path) -> dict:
    operating, mu = _configuration(output_root)
    tapes_by_point: dict[str, list[dict]] = {}
    point_summaries = {}
    for point in POINTS:
        parameters = _isolator(operating, point)
        condition = Condition(gamma=0.0, cube_table_sliding_mu=mu)
        print(f"record Gamma=0 tapes for {point}", flush=True)
        _summary, episodes = run_condition(
            condition,
            SEEDS,
            _episode_dir(output_root, point, 0.0, "reactive_scripted"),
            table_isolator=parameters,
            environment_variant=_variant(point, mu),
        )
        tapes_by_point[point] = episodes
        point_summaries[point] = summarize(episodes)

    pairs = {}
    for left, right in (("RIGID", "MILD"), ("RIGID", "STRONG"), ("MILD", "STRONG")):
        rows = []
        for left_tape, right_tape in zip(
            tapes_by_point[left], tapes_by_point[right], strict=True
        ):
            left_actions = np.asarray(left_tape["actions"], dtype=np.float64)
            right_actions = np.asarray(right_tape["actions"], dtype=np.float64)
            common = min(len(left_actions), len(right_actions))
            left_switch = left_tape["gripper_force_switch_history"][0][
                "policy_step_index"
            ]
            right_switch = right_tape["gripper_force_switch_history"][0][
                "policy_step_index"
            ]
            rows.append(
                {
                    "seed": left_tape["seed"],
                    "left_action_count": len(left_actions),
                    "right_action_count": len(right_actions),
                    "action_count_difference": len(right_actions) - len(left_actions),
                    "max_abs_action_difference_common_prefix": float(
                        np.max(np.abs(left_actions[:common] - right_actions[:common]))
                    ),
                    "left_force_switch_policy_step_index": left_switch,
                    "right_force_switch_policy_step_index": right_switch,
                    "force_switch_index_difference": right_switch - left_switch,
                }
            )
        pairs[f"{left}_vs_{right}"] = {
            "rows": rows,
            "max_abs_action_difference": max(
                row["max_abs_action_difference_common_prefix"] for row in rows
            ),
            "max_abs_action_count_difference": max(
                abs(row["action_count_difference"]) for row in rows
            ),
            "max_abs_force_switch_index_difference": max(
                abs(row["force_switch_index_difference"]) for row in rows
            ),
        }
    result = {
        "schema": "shakebench_spike.iso.tape_comparison.v1",
        "created_utc": _now(),
        "recorded_under_new_model": True,
        "source_gamma": 0.0,
        "seed_count_per_operating_point": len(SEEDS),
        "selected_mu": mu,
        "point_summaries": point_summaries,
        "paired_differences": pairs,
    }
    _write(output_root / "tapes" / "comparison.json", result)
    return result


def _load_tapes(output_root: Path, point: str) -> list[dict]:
    paths = [
        _episode_dir(output_root, point, 0.0, "reactive_scripted")
        / f"seed_{seed:03d}.json"
        for seed in SEEDS
    ]
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"Gamma=0 tapes for {point} are incomplete")
    return [_read(path) for path in paths]


def _load_or_run_frozen(
    output_path: Path,
    *,
    seed: int,
    gamma: float,
    tape: dict,
    parameters: IsolatorParameters | None,
    environment_variant: str,
) -> dict:
    expected_isolator = parameters.as_dict() if parameters is not None else None
    if output_path.is_file():
        cached = _read(output_path)
        if (
            cached.get("gamma") == gamma
            and cached.get("source_tape_seed") == seed
            and cached.get("environment_variant") == environment_variant
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
        environment_variant=environment_variant,
    )
    _write(output_path, episode)
    print(
        f"frozen seed={seed:03d} gamma={gamma:.2f} success={episode['success']} "
        f"failure={episode['failure_reason']} "
        f"placement_mm={1000*episode['placement']['translation_error_m']:.2f}",
        flush=True,
    )
    return episode


def _first_decline(rows: dict, point: str, strategy: str) -> float | None:
    baseline = rows[point]["0.0"][strategy]
    for gamma in GAMMAS[1:]:
        current = rows[point][str(gamma)][strategy]
        if (
            current["success_rate"] < baseline["success_rate"]
            or current["placement_error_distribution_m"]["p90"]
            > baseline["placement_error_distribution_m"]["p90"] + 0.005
        ):
            return gamma
    return None


def run_ladder(output_root: Path) -> dict:
    operating, mu = _configuration(output_root)
    if not (output_root / "tapes" / "comparison.json").is_file():
        raise RuntimeError("new-model Gamma=0 tapes must be recorded first")
    rows: dict[str, dict] = {}
    for point in POINTS:
        parameters = _isolator(operating, point)
        variant = _variant(point, mu)
        tapes = _load_tapes(output_root, point)
        point_rows = {}
        for gamma in GAMMAS:
            print(f"ladder reactive {point} Gamma={gamma:.2f}", flush=True)
            _legacy_summary, reactive = run_condition(
                Condition(gamma=gamma, cube_table_sliding_mu=mu),
                SEEDS,
                _episode_dir(output_root, point, gamma, "reactive_scripted"),
                table_isolator=parameters,
                environment_variant=variant,
            )
            frozen = []
            print(f"ladder frozen {point} Gamma={gamma:.2f}", flush=True)
            for seed, tape in zip(SEEDS, tapes, strict=True):
                frozen.append(
                    _load_or_run_frozen(
                        _episode_dir(output_root, point, gamma, "frozen_replay")
                        / f"seed_{seed:03d}.json",
                        seed=seed,
                        gamma=gamma,
                        tape=tape,
                        parameters=parameters,
                        environment_variant=variant,
                    )
                )
            reactive_summary = summarize(reactive)
            frozen_summary = summarize(frozen)
            point_rows[str(gamma)] = {
                "reactive_scripted": reactive_summary,
                "frozen_replay": frozen_summary,
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
                        SEEDS, reactive, frozen, strict=True
                    )
                ],
            }
        rows[point] = point_rows

    rigid_frozen_slip = [
        rows["RIGID"][str(gamma)]["frozen_replay"][
            "max_grasp_slip_distribution_m"
        ]["p90"]
        for gamma in GAMMAS
    ]
    rigid_reactive_slip = [
        rows["RIGID"][str(gamma)]["reactive_scripted"][
            "max_grasp_slip_distribution_m"
        ]["p90"]
        for gamma in GAMMAS
    ]
    strong_frozen_decline = _first_decline(rows, "STRONG", "frozen_replay")
    strong_reactive_decline = _first_decline(rows, "STRONG", "reactive_scripted")
    strong_frozen_failures = Counter()
    for gamma in GAMMAS:
        strong_frozen_failures.update(
            rows["STRONG"][str(gamma)]["frozen_replay"][
                "failure_reason_histogram"
            ]
        )
    dominant = (
        strong_frozen_failures.most_common(1)[0][0] if strong_frozen_failures else None
    )
    predictions = {
        "written_before_stage3_rollouts": True,
        "prediction_1": {
            "text": "RIGID is flat and frozen slip <= reactive slip",
            "observed": (
                all(
                    rows["RIGID"][str(gamma)][strategy]["success_rate"]
                    == rows["RIGID"]["0.0"][strategy]["success_rate"]
                    for gamma in GAMMAS
                    for strategy in ("reactive_scripted", "frozen_replay")
                )
                and all(
                    frozen <= reactive
                    for frozen, reactive in zip(
                        rigid_frozen_slip, rigid_reactive_slip, strict=True
                    )
                )
            ),
        },
        "prediction_2": {
            "text": "STRONG frozen degrades before reactive",
            "frozen_first_decline_gamma": strong_frozen_decline,
            "reactive_first_decline_gamma": strong_reactive_decline,
            "observed": strong_frozen_decline is not None
            and (
                strong_reactive_decline is None
                or strong_frozen_decline < strong_reactive_decline
            ),
        },
        "prediction_3": {
            "text": "STRONG frozen dominant failure is descend_table_contact",
            "failure_histogram": dict(sorted(strong_frozen_failures.items())),
            "dominant_failure": dominant,
            "observed": dominant == "descend_table_contact",
        },
    }
    result = {
        "schema": "shakebench_spike.iso.ladder_summary.v1",
        "created_utc": _now(),
        "seeds": SEEDS,
        "gammas": list(GAMMAS),
        "selected_mu": mu,
        "hold_force_n_per_finger": 3.0,
        "physics_timestep_s": 2.0e-4,
        "policy_frequency_hz": 20,
        "operating_points": operating["points"],
        "rows": rows,
        "prediction_comparison": predictions,
        "mujoco_warning_count": sum(
            rows[point][str(gamma)][strategy]["mujoco_warning_count"]
            for point in POINTS
            for gamma in GAMMAS
            for strategy in ("reactive_scripted", "frozen_replay")
        ),
    }
    _write(output_root / "ladder" / "summary.json", result)
    return result


def run_attribution(output_root: Path) -> dict:
    """Run the three required one-parameter checks at the SR decline point."""

    operating, mu = _configuration(output_root)
    ladder = _read(output_root / "ladder" / "summary.json")
    point = "STRONG"
    gamma = 0.95
    baseline_dir = _episode_dir(
        output_root, point, gamma, "reactive_scripted"
    )
    baseline = [_read(baseline_dir / f"seed_{seed:03d}.json") for seed in SEEDS]
    if summarize(baseline)["success_rate"] >= 1.0:
        raise RuntimeError("attribution requires an observed STRONG/Gamma=0.95 decline")
    parameters = _isolator(operating, point)
    checks = (
        (
            "hold_force_3_to_6_n",
            Condition(
                gamma=gamma,
                cube_table_sliding_mu=mu,
                gripper_force_limit_n=6.0,
            ),
            "gripper_force_limit_n_per_finger",
            3.0,
            6.0,
        ),
        (
            "osc_kp_150_to_300",
            Condition(gamma=gamma, cube_table_sliding_mu=mu, osc_kp=300.0),
            "osc_kp",
            150.0,
            300.0,
        ),
        (
            "move_gain_4_to_3",
            Condition(
                gamma=gamma,
                cube_table_sliding_mu=mu,
                move_action_gain=3.0,
            ),
            "move_action_gain",
            4.0,
            3.0,
        ),
    )
    rows = {}
    baseline_summary = summarize(baseline)
    for slug, condition, parameter, baseline_value, alternate_value in checks:
        print(f"attribution {slug}", flush=True)
        _legacy, alternate = run_condition(
            condition,
            SEEDS,
            output_root / "attribution" / slug / "reactive_scripted",
            table_isolator=parameters,
            environment_variant=f"{_variant(point, mu)}_{slug}",
        )
        paired = []
        for base_episode, alt_episode in zip(baseline, alternate, strict=True):
            paired.append(
                {
                    "seed": base_episode["seed"],
                    "baseline_success": base_episode["success"],
                    "alternate_success": alt_episode["success"],
                    "success_flipped": base_episode["success"]
                    != alt_episode["success"],
                    "baseline_failure_reason": base_episode["failure_reason"],
                    "alternate_failure_reason": alt_episode["failure_reason"],
                    "baseline_placement_error_m": base_episode["placement"][
                        "translation_error_m"
                    ],
                    "alternate_placement_error_m": alt_episode["placement"][
                        "translation_error_m"
                    ],
                }
            )
        baseline_failures = [row for row in paired if not row["baseline_success"]]
        rescued = sum(row["alternate_success"] for row in baseline_failures)
        rescue_fraction = rescued / len(baseline_failures)
        rows[slug] = {
            "parameter": parameter,
            "baseline_value": baseline_value,
            "alternate_value": alternate_value,
            "baseline": baseline_summary,
            "alternate": summarize(alternate),
            "paired_results": paired,
            "baseline_failure_count": len(baseline_failures),
            "baseline_failures_rescued_count": rescued,
            "baseline_failure_rescue_fraction": rescue_fraction,
            "effect_parameter_bound": rescue_fraction >= 0.50,
            "all_seed_success_flip_count": sum(
                row["success_flipped"] for row in paired
            ),
        }
    result = {
        "schema": "shakebench_spike.iso.attribution.v1",
        "created_utc": _now(),
        "effect_condition": {
            "operating_point": point,
            "gamma": gamma,
            "strategy": "reactive_scripted",
            "reason": "first observed reactive success-rate decline",
            "baseline_success_rate": baseline_summary["success_rate"],
            "ladder_prediction_comparison": ladder["prediction_comparison"],
        },
        "seed_count_per_check": len(SEEDS),
        "single_parameter_checks": rows,
        "any_parameter_bound": any(
            row["effect_parameter_bound"] for row in rows.values()
        ),
        "mujoco_warning_count": sum(
            row["alternate"]["mujoco_warning_count"] for row in rows.values()
        ),
    }
    _write(output_root / "attribution" / "summary.json", result)
    return result


def main() -> int:
    logger = logging.getLogger("robosuite_logs")
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", action="store_true")
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if sum((args.tapes, args.ladder, args.attribution, args.all)) != 1:
        parser.error(
            "select exactly one of --tapes, --ladder, --attribution, or --all"
        )
    if args.tapes or args.all:
        tapes = record_tapes(args.output)
        print(json.dumps(tapes["paired_differences"], indent=2), flush=True)
    if args.ladder or args.all:
        ladder = run_ladder(args.output)
        print(json.dumps(ladder["prediction_comparison"], indent=2), flush=True)
    if args.attribution or args.all:
        attribution = run_attribution(args.output)
        print(json.dumps(attribution, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
