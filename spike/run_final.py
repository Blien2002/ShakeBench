"""ShakeBench spike final-round W1/W2/W3 experiment runner."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from run_stage2b import Condition, _distribution, run_condition
from run_stage2c import run_frozen_episode


FINAL_GAMMAS = (0.0, 0.15, 0.30, 0.50, 0.75, 0.95, 1.2, 1.5)
TEN_SEEDS = list(range(10))
TWENTY_SEEDS = list(range(20))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(gamma: float) -> str:
    return str(gamma).replace(".", "p")


def _load_episodes(directory: Path, count: int) -> list[dict]:
    episodes = [_read(directory / f"seed_{seed:03d}.json") for seed in range(count)]
    if len(episodes) != count:
        raise RuntimeError(f"incomplete episode set: {directory}")
    return episodes


def _pooled_airborne(episodes: list[dict]) -> dict:
    recorded = [episode for episode in episodes if "cube_table_airborne_200hz" in episode]
    if not recorded:
        return {
            "available": False,
            "reason": "legacy pre-W2 episode; airborne diagnostic was not recorded",
        }
    phases: dict[str, list[int]] = {}
    for episode in recorded:
        diagnostic = episode["cube_table_airborne_200hz"]
        for phase, row in diagnostic["by_phase"].items():
            counts = phases.setdefault(phase, [0, 0])
            counts[0] += row["sample_count"]
            counts[1] += row["airborne_sample_count"]
    phase_rows = {
        phase: {
            "sample_count": counts[0],
            "airborne_sample_count": counts[1],
            "airborne_fraction": counts[1] / counts[0] if counts[0] else None,
        }
        for phase, counts in sorted(phases.items())
    }
    pre_grasp_names = ("settle", "approach", "descend")
    pre_grasp_rows = []
    for episode in recorded:
        by_phase = episode["cube_table_airborne_200hz"]["by_phase"]
        sample_count = sum(by_phase.get(name, {}).get("sample_count", 0) for name in pre_grasp_names)
        airborne_count = sum(
            by_phase.get(name, {}).get("airborne_sample_count", 0)
            for name in pre_grasp_names
        )
        pre_grasp_rows.append(
            {
                "sample_count": sample_count,
                "airborne_sample_count": airborne_count,
                "airborne_fraction": airborne_count / sample_count if sample_count else None,
            }
        )
    total = sum(row["sample_count"] for row in pre_grasp_rows)
    airborne = sum(row["airborne_sample_count"] for row in pre_grasp_rows)
    return {
        "available": True,
        "diagnostic_frequency_hz": 200,
        "airborne_definition": "cube-table normal force is exactly 0 N",
        "pre_grasp_pooled": {
            "phases": list(pre_grasp_names),
            "sample_count": total,
            "airborne_sample_count": airborne,
            "airborne_fraction": airborne / total if total else None,
        },
        "pre_grasp_episode_fraction_distribution": _distribution(
            [row["airborne_fraction"] for row in pre_grasp_rows]
        ),
        "by_phase_pooled": phase_rows,
    }


def summarize(episodes: list[dict]) -> dict:
    success_count = sum(bool(episode["success"]) for episode in episodes)
    return {
        "episode_count": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "failure_reason_histogram": dict(
            sorted(
                Counter(
                    episode["failure_reason"]
                    for episode in episodes
                    if not episode["success"]
                ).items()
            )
        ),
        "max_grasp_slip_distribution_m": _distribution(
            [episode["max_grasp_slip_m"] for episode in episodes]
        ),
        "table_frame_object_slip_distribution_m": _distribution(
            [episode["obj_slip_on_table_m"] for episode in episodes]
        ),
        "cube_table_airborne_200hz": _pooled_airborne(episodes),
        "mujoco_warning_count": sum(
            episode["mujoco_warning_count"] for episode in episodes
        ),
    }


def write_w0(output_root: Path) -> dict:
    result = {
        "schema": "shakebench_spike.final.gate2_reinterpretation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "old_rule": "3 N must differ from both 1.5 N and 6 N by <5 percentage points",
        "corrected_rule": (
            "one-sided saturation at the highest tested Gamma: increasing the operating "
            "value must not change success rate"
        ),
        "reason": (
            "requiring SR(P/2) approximately equal SR(P) imposes a twofold success-rate "
            "safety factor and incorrectly rejects any physical threshold knee"
        ),
        "gamma": 0.95,
        "operating_force_n_per_finger": 3.0,
        "higher_force_n_per_finger": 6.0,
        "success_rates": {"3.0": 1.0, "6.0": 1.0},
        "difference_percentage_points": 0.0,
        "corrected_gate_2_passed": True,
        "new_rollouts_run_for_w0": False,
    }
    _write(output_root / "gate2_reinterpretation.json", result)
    return result


def run_w1(output_root: Path) -> dict:
    rows = {}
    for gamma in (0.15, 0.30, 0.75):
        condition = Condition(gamma=gamma)
        summary, episodes = run_condition(
            condition,
            TEN_SEEDS,
            output_root / "ladder_reactive" / f"gamma_{_slug(gamma)}",
        )
        rows[str(gamma)] = {
            "condition_summary": summary,
            "final_summary": summarize(episodes),
        }
    result = {
        "schema": "shakebench_spike.final.w1.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_count_per_gamma": 10,
        "rows": rows,
    }
    _write(output_root / "ladder_reactive" / "w1_summary.json", result)
    return result


def run_w2(output_root: Path) -> dict:
    rows = {}
    episodes_by_gamma = {}
    for gamma in (1.2, 1.5):
        condition = Condition(gamma=gamma)
        condition_summary, episodes = run_condition(
            condition,
            TWENTY_SEEDS,
            output_root / "ladder_reactive" / f"gamma_{_slug(gamma)}",
        )
        episodes_by_gamma[gamma] = episodes
        rows[str(gamma)] = {
            "condition_summary": condition_summary,
            "final_summary": summarize(episodes),
        }

    force_rows = {}
    baseline = episodes_by_gamma[1.5][:10]
    force_rows["3.0"] = summarize(baseline)
    for force_n in (6.0, 12.0):
        _condition_summary, episodes = run_condition(
            Condition(gamma=1.5, gripper_force_limit_n=force_n),
            TEN_SEEDS,
            output_root
            / "hold_force_saturation"
            / f"hold_{str(force_n).replace('.', 'p')}n",
        )
        force_rows[str(force_n)] = summarize(episodes)
    saturation = {
        "schema": "shakebench_spike.final.hold_force_saturation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gamma": 1.5,
        "seeds": TEN_SEEDS,
        "acquisition_force_limit_n_per_finger": 20.0,
        "rows": force_rows,
        "three_to_six_difference_percentage_points": 100.0
        * abs(force_rows["3.0"]["success_rate"] - force_rows["6.0"]["success_rate"]),
        "one_sided_saturation_passed": (
            force_rows["3.0"]["success_rate"] == force_rows["6.0"]["success_rate"]
        ),
    }
    _write(output_root / "hold_force_saturation" / "summary.json", saturation)
    result = {
        "schema": "shakebench_spike.final.w2.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_count_per_gamma": 20,
        "authorized_design_change": "Gamma ladder upper bound extended from 0.95 to 1.5",
        "rows": rows,
        "hold_force_saturation": saturation,
    }
    _write(output_root / "ladder_reactive" / "w2_summary.json", result)
    return result


def _reactive_directory(final_root: Path, stage2c_root: Path, gamma: float) -> Path:
    if gamma in (0.0, 0.5, 0.95):
        return stage2c_root / "exp1_rerun" / f"gamma_{_slug(gamma)}"
    return final_root / "ladder_reactive" / f"gamma_{_slug(gamma)}"


def _load_or_run_frozen(path: Path, seed: int, gamma: float, tape: dict) -> dict:
    if path.exists():
        cached = _read(path)
        if cached.get("gamma") == gamma and cached.get("source_tape_seed") == seed:
            print(
                f"resume frozen seed={seed:03d} gamma={gamma:.2f} success={cached['success']}",
                flush=True,
            )
            return cached
    episode = run_frozen_episode(seed, gamma, tape, environment_variant="hard_mounted_mu_1p5")
    _write(path, episode)
    print(
        f"frozen seed={seed:03d} gamma={gamma:.2f} success={episode['success']} "
        f"failure={episode['failure_reason']} slip_mm={1000*episode['max_grasp_slip_m']:.3f}",
        flush=True,
    )
    return episode


def run_w3(output_root: Path, stage2c_root: Path) -> dict:
    source_dir = stage2c_root / "exp1_rerun" / "gamma_0p0"
    tapes = {seed: _read(source_dir / f"seed_{seed:03d}.json") for seed in TEN_SEEDS}
    rows = {}
    for gamma in FINAL_GAMMAS:
        reactive = _load_episodes(
            _reactive_directory(output_root, stage2c_root, gamma),
            10,
        )
        frozen = []
        for seed in TEN_SEEDS:
            frozen.append(
                _load_or_run_frozen(
                    output_root
                    / "ladder_frozen"
                    / f"gamma_{_slug(gamma)}"
                    / f"seed_{seed:03d}.json",
                    seed,
                    gamma,
                    tapes[seed],
                )
            )
        rows[str(gamma)] = {
            "reactive_scripted": summarize(reactive),
            "frozen_replay": summarize(frozen),
        }
    result = {
        "schema": "shakebench_spike.final.w3.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gammas": list(FINAL_GAMMAS),
        "seeds": TEN_SEEDS,
        "action_tape": {
            "source": "Stage2C Gamma=0 reactive policy.command() outputs, seeds 0-9",
            "force_switch": "recorded policy-step index replayed open-loop",
            "policy_frequency_hz": 20,
        },
        "rows": rows,
    }
    _write(output_root / "ladder_frozen" / "summary.json", result)
    return result


def run_decline_sensitivity(output_root: Path) -> dict:
    """Required W3 anti-artifact checks at the only declining point, Gamma=1.5."""

    baseline = _load_episodes(output_root / "ladder_reactive" / "gamma_1p5", 10)
    _condition_summary, alternate = run_condition(
        Condition(gamma=1.5, osc_kp=300.0),
        TEN_SEEDS,
        output_root / "sensitivity" / "gamma_1p5" / "osc_kp_300",
    )
    pairs = []
    for before, after in zip(baseline, alternate, strict=True):
        pairs.append(
            {
                "seed": before["seed"],
                "baseline_success": before["success"],
                "alternate_success": after["success"],
                "flipped": before["success"] != after["success"],
                "baseline_failure_reason": before["failure_reason"],
                "alternate_failure_reason": after["failure_reason"],
            }
        )
    hold = _read(output_root / "hold_force_saturation" / "summary.json")
    result = {
        "schema": "shakebench_spike.final.decline_sensitivity.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gamma": 1.5,
        "trigger": "reactive_scripted success rate declined at Gamma=1.5",
        "baseline": summarize(baseline),
        "hold_force_check": hold,
        "osc_kp_check": {
            "baseline_kp": 150.0,
            "alternate_kp": 300.0,
            "alternate": summarize(alternate),
            "paired_results": pairs,
            "flip_count": sum(row["flipped"] for row in pairs),
            "flip_fraction": sum(row["flipped"] for row in pairs) / len(pairs),
        },
    }
    _write(output_root / "sensitivity" / "gamma_1p5" / "summary.json", result)
    return result


def main() -> int:
    logger = logging.getLogger("robosuite_logs")
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("w0", "w1", "w2", "w3", "sensitivity"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "out" / "final",
    )
    parser.add_argument(
        "--stage2c-output",
        type=Path,
        default=Path(__file__).parent / "out" / "stage2c",
    )
    args = parser.parse_args()
    if args.stage == "w0":
        print(json.dumps(write_w0(args.output), sort_keys=True), flush=True)
        return 0
    if args.stage == "w1":
        result = run_w1(args.output)
        print(json.dumps({g: row["final_summary"]["success_rate"] for g, row in result["rows"].items()}, sort_keys=True), flush=True)
        return 0
    if args.stage == "w2":
        result = run_w2(args.output)
        print(
            json.dumps(
                {
                    "success_rates": {
                        g: row["final_summary"]["success_rate"]
                        for g, row in result["rows"].items()
                    },
                    "hold_saturation_passed": result["hold_force_saturation"][
                        "one_sided_saturation_passed"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.stage == "w3":
        result = run_w3(args.output, args.stage2c_output)
        print(
            json.dumps(
                {
                    gamma: {
                        strategy: row[strategy]["success_rate"]
                        for strategy in ("reactive_scripted", "frozen_replay")
                    }
                    for gamma, row in result["rows"].items()
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.stage == "sensitivity":
        result = run_decline_sensitivity(args.output)
        print(
            json.dumps(
                {
                    "baseline_success_rate": result["baseline"]["success_rate"],
                    "hold_3n_success_rate": result["hold_force_check"]["rows"]["3.0"]["success_rate"],
                    "hold_6n_success_rate": result["hold_force_check"]["rows"]["6.0"]["success_rate"],
                    "kp300_success_rate": result["osc_kp_check"]["alternate"]["success_rate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
