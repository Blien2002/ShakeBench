"""Stage 2B experiment runner with resumable, configuration-stamped outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform

import mujoco
import numpy as np
import robosuite

from env_shakedeck import CONTACT_SOLREF, make_env
from metrics import EpisodeMetrics, contact_snapshot
from policies import ReactiveConfig, ReactiveScriptedPolicy
from vibration import calibrated_vibration


POLICY_HZ = 20
DIAGNOSTIC_HZ = 200
DEFAULT_TIMESTEP_S = 2.0e-4
DEFAULT_EPISODE_S = 16.0
DEFAULT_FORCE_N = 3.0


@dataclass(frozen=True)
class Condition:
    gamma: float
    gripper_force_limit_n: float = DEFAULT_FORCE_N
    pad_solref_dampratio: float = 0.5
    osc_kp: float = 150.0
    physics_timestep_s: float = DEFAULT_TIMESTEP_S
    cube_table_solref_timeconst_s: float = CONTACT_SOLREF[0]
    cube_table_sliding_mu: float = 1.5
    move_action_gain: float = 4.0
    episode_limit_s: float = DEFAULT_EPISODE_S
    policy_frequency_hz: int = POLICY_HZ
    diagnostic_frequency_hz: int = DIAGNOSTIC_HZ
    grip_mode: str = "force_limited_close"


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "robosuite": robosuite.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def _warning_count(env) -> int:
    return int(sum(item.number for item in env.sim.data.warning))


def _vibration(gamma: float, seed: int, physics_hz: int, episode_s: float):
    if gamma == 0.0:
        return None, {
            "gamma_target": 0.0,
            "gamma_realized": 0.0,
            "level_scale": 0.0,
            "motion_sampler": "disabled",
        }
    vibration, calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=episode_s,
    )
    calibration["motion_sampler"] = "shakebench_spectral"
    return vibration, calibration


def run_reactive_episode(seed: int, condition: Condition) -> dict:
    physics_hz = int(round(1.0 / condition.physics_timestep_s))
    if physics_hz % condition.diagnostic_frequency_hz:
        raise ValueError("physics frequency must divide diagnostic frequency exactly")
    vibration, calibration = _vibration(
        condition.gamma,
        seed,
        physics_hz,
        condition.episode_limit_s,
    )
    max_policy_steps = int(round(condition.episode_limit_s * condition.policy_frequency_hz))
    env = make_env(
        seed=seed,
        physics_timestep=condition.physics_timestep_s,
        motion_sampler=None if vibration is None else vibration.sample,
        control_freq=condition.policy_frequency_hz,
        horizon=max_policy_steps + 1,
        direct_gripper=True,
        gripper_force_limit_n=condition.gripper_force_limit_n,
        pad_solref_dampratio=condition.pad_solref_dampratio,
        cube_table_solref=(condition.cube_table_solref_timeconst_s, CONTACT_SOLREF[1]),
        cube_table_sliding_mu=condition.cube_table_sliding_mu,
        osc_kp=condition.osc_kp,
    )
    try:
        policy = ReactiveScriptedPolicy(
            env,
            ReactiveConfig(
                control_freq_hz=condition.policy_frequency_hz,
                grip_mode="force_limited_close",
                move_action_gain=condition.move_action_gain,
            ),
        )
        metrics = EpisodeMetrics.start(env)
        decimation = physics_hz // condition.diagnostic_frequency_hz
        physics_steps = 0

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation == 0:
                metrics.update(stepped_env, policy, contact_snapshot(stepped_env))

        env.physics_step_callback = sample
        actions: list[list[float]] = []
        for _ in range(max_policy_steps):
            action = policy.command(contact_snapshot(env))
            actions.append(action.tolist())
            env.step(action)
            if policy.finished:
                break
        result = metrics.result(policy, _warning_count(env))
        result.update(
            {
                "schema": "shakebench_spike.stage2b.reactive_episode.v1",
                "seed": seed,
                "condition": asdict(condition),
                "elapsed_s": float(env.sim.data.time),
                "policy_steps": len(actions),
                "calibration": calibration,
                "compiled_contact_configuration": env.contact_configuration(),
                "compiled_gripper_actuators": env.gripper_actuator_configuration(),
                "object_mass_kg": float(
                    env.sim.model.body_subtreemass[env.sim.model.body_name2id("cube_main")]
                ),
                "actions": actions,
            }
        )
        return result
    finally:
        env.close()


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p90": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "max": float(np.max(data)),
    }


def summarize_condition(condition: Condition, episodes: list[dict]) -> dict:
    failures = Counter(
        episode["failure_reason"] for episode in episodes if not episode["success"]
    )
    force_keys = (
        "both_zero_fraction",
        "left_below_threshold_fraction",
        "right_below_threshold_fraction",
        "exactly_one_below_threshold_fraction",
        "both_below_threshold_fraction",
    )
    force_distributions = {}
    for key in force_keys:
        force_distributions[key] = _distribution(
            [
                value
                for episode in episodes
                if (value := episode["post_latch_finger_force_n"].get(key)) is not None
            ]
        )
    success_count = sum(bool(episode["success"]) for episode in episodes)
    return {
        "schema": "shakebench_spike.stage2b.condition_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "condition": asdict(condition),
        "seeds": [episode["seed"] for episode in episodes],
        "episode_count": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "failure_reason_histogram": dict(sorted(failures.items())),
        "max_grasp_slip_distribution_m": _distribution(
            [episode["max_grasp_slip_m"] for episode in episodes]
        ),
        "post_latch_force_samples_total": sum(
            episode["post_latch_finger_force_n"]["sample_count"] for episode in episodes
        ),
        "post_latch_contact_loss_fraction_distributions": force_distributions,
        "mujoco_warning_count": sum(episode["mujoco_warning_count"] for episode in episodes),
    }


def run_condition(
    condition: Condition,
    seeds: list[int],
    output_dir: Path,
    *,
    resume: bool = True,
) -> tuple[dict, list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for seed in seeds:
        episode_path = output_dir / f"seed_{seed:03d}.json"
        if resume and episode_path.exists():
            cached = json.loads(episode_path.read_text(encoding="utf-8"))
            if cached.get("condition") == asdict(condition):
                episodes.append(cached)
                print(f"resume seed={seed:03d} success={cached['success']}", flush=True)
                continue
        episode = run_reactive_episode(seed, condition)
        episode_path.write_text(
            json.dumps(episode, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        episodes.append(episode)
        force = episode["post_latch_finger_force_n"]
        print(
            f"seed={seed:03d} gamma={condition.gamma:.2f} success={episode['success']} "
            f"failure={episode['failure_reason']} slip_mm={1000 * episode['max_grasp_slip_m']:.3f} "
            f"one_off={force['exactly_one_below_threshold_fraction']} "
            f"both_off={force['both_below_threshold_fraction']}",
            flush=True,
        )
    summary = summarize_condition(condition, episodes)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary, episodes


def run_b1(output_dir: Path, seeds: list[int]) -> dict:
    control_condition = Condition(gamma=0.0)
    shaken_condition = Condition(gamma=0.5)
    control, control_episodes = run_condition(
        control_condition,
        seeds,
        output_dir / "gamma_0p0",
    )
    shaken, shaken_episodes = run_condition(
        shaken_condition,
        seeds,
        output_dir / "gamma_0p5",
    )
    paired = []
    for zero, half in zip(control_episodes, shaken_episodes, strict=True):
        paired.append(
            {
                "seed": zero["seed"],
                "gamma_0_success": zero["success"],
                "gamma_0p5_success": half["success"],
                "success_changed": zero["success"] != half["success"],
                "gamma_0_max_grasp_slip_m": zero["max_grasp_slip_m"],
                "gamma_0p5_max_grasp_slip_m": half["max_grasp_slip_m"],
            }
        )
    result = {
        "schema": "shakebench_spike.stage2b.b1_paired_rerun.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control": control,
        "gamma_0p5": shaken,
        "paired": paired,
        "gate_1": {
            "criterion": "force-control gamma=0 success count >= 19/20",
            "passed": control["success_count"] >= 19 and control["episode_count"] == 20,
            "success_count": control["success_count"],
            "episode_count": control["episode_count"],
        },
    }
    (output_dir / "b1_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_b1_baseline(output_root: Path, condition: Condition) -> tuple[dict, list[dict]]:
    condition_dir = output_root / "exp1_rerun" / "gamma_0p5"
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(condition_dir.glob("seed_*.json"))
    ]
    if len(episodes) != 20 or any(
        episode.get("condition") != asdict(condition) for episode in episodes
    ):
        raise RuntimeError("B2 requires the complete 20-seed B1 gamma=0.5 baseline")
    return summarize_condition(condition, episodes), episodes


def _paired_flip_result(
    *,
    parameter: str,
    baseline_value,
    alternate_value,
    baseline_summary: dict,
    baseline_episodes: list[dict],
    alternate_summary: dict,
    alternate_episodes: list[dict],
) -> dict:
    pairs = []
    for baseline, alternate in zip(baseline_episodes, alternate_episodes, strict=True):
        if baseline["seed"] != alternate["seed"]:
            raise RuntimeError("sensitivity episode seeds are not paired")
        pairs.append(
            {
                "seed": baseline["seed"],
                "baseline_success": baseline["success"],
                "alternate_success": alternate["success"],
                "flipped": baseline["success"] != alternate["success"],
                "baseline_failure_reason": baseline["failure_reason"],
                "alternate_failure_reason": alternate["failure_reason"],
            }
        )
    flip_count = sum(pair["flipped"] for pair in pairs)
    denominator = len(pairs)
    return {
        "schema": "shakebench_spike.stage2b.sensitivity_flip.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameter": parameter,
        "baseline_value": baseline_value,
        "alternate_value": alternate_value,
        "seed_selection": {
            "rule": "all 20 seeds because the B1 gamma=0.5 failure set is empty",
            "seeds": [pair["seed"] for pair in pairs],
        },
        "baseline": baseline_summary,
        "alternate": alternate_summary,
        "paired_results": pairs,
        "flip_count": flip_count,
        "flip_denominator": denominator,
        "flip_fraction": flip_count / denominator,
        "gate_2_parameter_passed": flip_count < 0.5 * denominator,
    }


def run_b2(output_root: Path, seeds: list[int]) -> dict:
    if seeds != list(range(20)):
        raise ValueError("formal B2 scan requires seeds 0-19")
    output_dir = output_root / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_condition = Condition(gamma=0.5)
    baseline_summary, baseline_episodes = _load_b1_baseline(output_root, baseline_condition)

    # B2-1: pad solref damping ratio, deliberately first.
    pad_condition = Condition(gamma=0.5, pad_solref_dampratio=1.0)
    pad_summary, pad_episodes = run_condition(
        pad_condition,
        seeds,
        output_dir / "pad_damping" / "damping_1p0",
    )
    pad = _paired_flip_result(
        parameter="pad_solref_dampratio",
        baseline_value=0.5,
        alternate_value=1.0,
        baseline_summary=baseline_summary,
        baseline_episodes=baseline_episodes,
        alternate_summary=pad_summary,
        alternate_episodes=pad_episodes,
    )
    (output_dir / "pad_damping.json").write_text(
        json.dumps(pad, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # B2-2: force is a four-point response curve, not a flip-count gate.
    force_summaries = {}
    force_episodes = {}
    for force_n in (1.0, 1.5, 3.0, 6.0):
        key = f"{force_n:.1f}"
        if force_n == DEFAULT_FORCE_N:
            force_summaries[key] = baseline_summary
            force_episodes[key] = baseline_episodes
            continue
        condition = Condition(gamma=0.5, gripper_force_limit_n=force_n)
        summary, episodes = run_condition(
            condition,
            seeds,
            output_dir / "grip_force" / f"force_{str(force_n).replace('.', 'p')}n",
        )
        force_summaries[key] = summary
        force_episodes[key] = episodes
    curve = [
        {
            "force_limit_n_per_finger": force_n,
            "success_count": force_summaries[f"{force_n:.1f}"]["success_count"],
            "episode_count": force_summaries[f"{force_n:.1f}"]["episode_count"],
            "success_rate": force_summaries[f"{force_n:.1f}"]["success_rate"],
            "max_grasp_slip_distribution_m": force_summaries[f"{force_n:.1f}"][
                "max_grasp_slip_distribution_m"
            ],
            "failure_reason_histogram": force_summaries[f"{force_n:.1f}"][
                "failure_reason_histogram"
            ],
        }
        for force_n in (1.0, 1.5, 3.0, 6.0)
    ]
    sr_by_force = {row["force_limit_n_per_finger"]: row["success_rate"] for row in curve}
    adjacent_deltas = {
        "vs_1p5_percentage_points": 100.0 * abs(sr_by_force[3.0] - sr_by_force[1.5]),
        "vs_6p0_percentage_points": 100.0 * abs(sr_by_force[3.0] - sr_by_force[6.0]),
    }
    grip_force = {
        "schema": "shakebench_spike.stage2b.grip_force_plateau.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_selection": "all 20 seeds because the B1 gamma=0.5 failure set is empty",
        "curve": curve,
        "adjacent_success_rate_deltas": adjacent_deltas,
        "plateau_criterion": "absolute SR difference from both adjacent points is <5 percentage points",
        "three_newton_on_platform": all(value < 5.0 for value in adjacent_deltas.values()),
        "force_derivation_scope": (
            "mg/(2*mu)*1.95 covers only vertical gravity-plus-inertial load; "
            "lateral inertia and moment from grasp-point eccentricity are omitted, "
            "so 0.5293 N/finger is a lower bound, not a complete requirement."
        ),
    }
    (output_dir / "grip_force_plateau.json").write_text(
        json.dumps(grip_force, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not grip_force["three_newton_on_platform"]:
        gate = {
            "schema": "shakebench_spike.stage2b.gate2.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "regular_parameter_passes": {
                pad["parameter"]: pad["gate_2_parameter_passed"],
            },
            "grip_force_three_newton_on_platform": False,
            "not_run_due_to_force_platform_failure": [
                "osc_kp",
                "physics_timestep_s",
                "cube_table_solref_timeconst_s",
                "move_action_gain",
            ],
            "passed": False,
            "stop_reason": (
                "3 N is not on a success-rate plateau relative to both adjacent "
                "force points; Stage 2B protocol requires returning to Stage A."
            ),
        }
        (output_dir / "gate2_summary.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return gate

    regular_specs = (
        (
            "osc_kp",
            150.0,
            300.0,
            Condition(gamma=0.5, osc_kp=300.0),
            "osc_kp.json",
        ),
        (
            "physics_timestep_s",
            DEFAULT_TIMESTEP_S,
            1.0e-4,
            Condition(gamma=0.5, physics_timestep_s=1.0e-4),
            "timestep.json",
        ),
        (
            "cube_table_solref_timeconst_s",
            CONTACT_SOLREF[0],
            1.2e-3,
            Condition(gamma=0.5, cube_table_solref_timeconst_s=1.2e-3),
            "cube_table_solref.json",
        ),
        (
            "move_action_gain",
            4.0,
            1.0,
            Condition(gamma=0.5, move_action_gain=1.0),
            "move_action_gain.json",
        ),
    )
    regular_results = {}
    for parameter, baseline_value, alternate_value, condition, filename in regular_specs:
        summary, episodes = run_condition(
            condition,
            seeds,
            output_dir / parameter / "alternate",
        )
        result = _paired_flip_result(
            parameter=parameter,
            baseline_value=baseline_value,
            alternate_value=alternate_value,
            baseline_summary=baseline_summary,
            baseline_episodes=baseline_episodes,
            alternate_summary=summary,
            alternate_episodes=episodes,
        )
        if parameter == "move_action_gain":
            result["diagnostic_only_not_in_gate_2"] = True
        (output_dir / filename).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        regular_results[parameter] = result

    gated_results = [pad] + [
        regular_results[name]
        for name in (
            "osc_kp",
            "physics_timestep_s",
            "cube_table_solref_timeconst_s",
        )
    ]
    gate = {
        "schema": "shakebench_spike.stage2b.gate2.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "regular_parameter_passes": {
            result["parameter"]: result["gate_2_parameter_passed"]
            for result in gated_results
        },
        "grip_force_three_newton_on_platform": grip_force["three_newton_on_platform"],
        "move_action_gain_diagnostic": {
            "flip_fraction": regular_results["move_action_gain"]["flip_fraction"],
            "excluded_from_gate": True,
        },
        "passed": all(result["gate_2_parameter_passed"] for result in gated_results)
        and grip_force["three_newton_on_platform"],
    }
    (output_dir / "gate2_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return gate


def main() -> int:
    robosuite_logger = logging.getLogger("robosuite_logs")
    robosuite_logger.setLevel(logging.ERROR)
    for handler in robosuite_logger.handlers:
        handler.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("b1", "b2"), required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "out" / "stage2b",
    )
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    if args.stage == "b1":
        result = run_b1(args.output / "exp1_rerun", seeds)
        print(json.dumps(result["gate_1"], sort_keys=True), flush=True)
        return 0 if result["gate_1"]["passed"] else 2
    if args.stage == "b2":
        result = run_b2(args.output, seeds)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 2
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
