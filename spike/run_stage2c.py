"""Stage 2C runner: A3 authority split, B1', B2', and gated C experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from env_shakedeck import CONTACT_SOLREF, make_env
from metrics import (
    CONTACT_THRESHOLD_N,
    EpisodeMetrics,
    SLIP_TOLERANCE_M,
    contact_snapshot,
    eef_position_b,
    object_position_b,
)
from run_stage2b import (
    Condition,
    _distribution,
    _vibration,
    _warning_count,
    airborne_summary,
    run_condition,
    run_reactive_episode,
    summarize_condition,
)
from vibration import (
    SecondOrderSupportedVibration,
    ZeroVibration,
    calibrated_vibration,
)


GAMMAS = (0.0, 0.15, 0.30, 0.50, 0.75, 0.95)
FORMAL_SEEDS = list(range(20))
C_SEEDS = list(range(10))


class ReplayState:
    """Metric-facing state for a fixed action and fixed event tape."""

    def __init__(self) -> None:
        self.phase = "settle"
        self.hand_minus_object_b_at_grasp = None
        self.hold_started = False
        self.hold_completed = False
        self.hold_time_s = 0.0
        self.finished = False
        self.failure_reason = None
        self.phase_history = [{"phase": "settle", "time_s": 0.0}]
        self.last_target_b = None


def _force_switch_event(episode: dict) -> dict:
    events = episode["gripper_force_switch_history"]
    if len(events) != 1:
        raise RuntimeError(f"expected one force switch, found {len(events)}")
    return events[0]


def _trace_summary(trace: list[dict]) -> dict:
    if not trace:
        return {"sample_count": 0}
    actuator_names = list(trace[0]["actuator_force_n"])
    joint_names = list(trace[0]["finger_joint_velocity_m_s"])
    return {
        "sample_count": len(trace),
        "start_time_s": trace[0]["time_s"],
        "end_time_s": trace[-1]["time_s"],
        "actuator_force_peak_abs_n": {
            name: max(abs(row["actuator_force_n"][name]) for row in trace)
            for name in actuator_names
        },
        "finger_joint_velocity_peak_abs_m_s": {
            name: max(abs(row["finger_joint_velocity_m_s"][name]) for row in trace)
            for name in joint_names
        },
        "finger_joint_velocity_during_closing_m_s": {
            name: {
                "median": float(
                    np.median(
                        [
                            row["finger_joint_velocity_m_s"][name]
                            for row in trace
                            if row["policy_phase"] in ("approach", "descend", "grasp")
                        ]
                    )
                ),
                "peak_abs": max(
                    abs(row["finger_joint_velocity_m_s"][name])
                    for row in trace
                    if row["policy_phase"] in ("approach", "descend", "grasp")
                ),
            }
            for name in joint_names
        },
    }


def run_frozen_episode(
    seed: int,
    gamma: float,
    tape: dict,
    *,
    table_motion_sampler=None,
    table_isolator=None,
    environment_variant: str = "hard_mounted",
) -> dict:
    source_condition = tape["condition"]
    condition = Condition(
        gamma=gamma,
        gripper_force_limit_n=source_condition["gripper_force_limit_n"],
        pad_solref_dampratio=source_condition["pad_solref_dampratio"],
        osc_kp=source_condition["osc_kp"],
        physics_timestep_s=source_condition["physics_timestep_s"],
        cube_table_solref_timeconst_s=source_condition[
            "cube_table_solref_timeconst_s"
        ],
        cube_table_sliding_mu=source_condition["cube_table_sliding_mu"],
        move_action_gain=source_condition["move_action_gain"],
        episode_limit_s=source_condition["episode_limit_s"],
        policy_frequency_hz=source_condition["policy_frequency_hz"],
        diagnostic_frequency_hz=source_condition["diagnostic_frequency_hz"],
    )
    physics_hz = int(round(1.0 / condition.physics_timestep_s))
    vibration, calibration = _vibration(
        gamma, seed, physics_hz, condition.episode_limit_s
    )
    env = make_env(
        seed=seed,
        physics_timestep=condition.physics_timestep_s,
        motion_sampler=None if vibration is None else vibration.sample,
        control_freq=condition.policy_frequency_hz,
        horizon=len(tape["actions"]) + 1,
        direct_gripper=True,
        gripper_force_limit_n=condition.gripper_force_limit_n,
        pad_solref_dampratio=condition.pad_solref_dampratio,
        cube_table_solref=(condition.cube_table_solref_timeconst_s, CONTACT_SOLREF[1]),
        cube_table_sliding_mu=condition.cube_table_sliding_mu,
        osc_kp=condition.osc_kp,
        table_motion_sampler=table_motion_sampler,
        table_isolator=table_isolator,
    )
    try:
        state = ReplayState()
        metrics = EpisodeMetrics.start(env)
        switch_step = _force_switch_event(tape)["policy_step_index"]
        decimation = physics_hz // condition.diagnostic_frequency_hz
        physics_steps = 0
        airborne_phase_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation == 0:
                contacts = contact_snapshot(stepped_env)
                counts = airborne_phase_counts[state.phase]
                counts[0] += 1
                counts[1] += int(contacts.cube_table_n == 0.0)
                metrics.update(stepped_env, state, contacts)

        env.physics_step_callback = sample
        contact_loss_time_s = 0.0
        actions = np.asarray(tape["actions"], dtype=np.float64)
        phases = tape["phase_by_policy_step"]
        targets = tape.get("target_position_b_by_policy_step")
        if len(actions) != len(phases):
            raise RuntimeError("action and phase tapes have different lengths")
        if targets is not None and len(targets) != len(actions):
            raise RuntimeError("target and action tapes have different lengths")
        executed_steps = 0
        for policy_step_index, action in enumerate(actions):
            state.phase = phases[policy_step_index]
            state.last_target_b = (
                np.asarray(targets[policy_step_index], dtype=np.float64)
                if targets is not None
                else None
            )
            if policy_step_index == switch_step:
                state.hand_minus_object_b_at_grasp = (
                    eef_position_b(env) - object_position_b(env)
                ).copy()
                event = env.activate_hold_force_limit(
                    policy_step_index=policy_step_index,
                    trigger="frozen_tape_index",
                )
                state.phase_history.append(event)
            before = contact_snapshot(env)
            if (
                state.phase in ("descend", "grasp")
                and before.finger_table_n > CONTACT_THRESHOLD_N
                and not before.any_cube
            ):
                state.failure_reason = "descend_table_contact"
                break
            env.step(action)
            executed_steps += 1
            after = contact_snapshot(env)
            if state.hand_minus_object_b_at_grasp is not None and state.phase in (
                "lift",
                "transfer_hold",
                "place",
            ):
                slip = float(
                    np.linalg.norm(
                        (eef_position_b(env) - object_position_b(env))
                        - state.hand_minus_object_b_at_grasp
                    )
                )
                if slip > SLIP_TOLERANCE_M:
                    state.failure_reason = "grasp_slip_exceeded"
                    break
                if after.bilateral:
                    contact_loss_time_s = 0.0
                else:
                    contact_loss_time_s += 1.0 / condition.policy_frequency_hz
                    if contact_loss_time_s >= 0.20:
                        state.failure_reason = "grasp_contact_lost"
                        break
            if state.phase == "transfer_hold":
                state.hold_started = True
                state.hold_time_s += 1.0 / condition.policy_frequency_hz
            if policy_step_index == tape["hold_completed_policy_step_index"]:
                state.hold_completed = True

        state.finished = state.failure_reason is None and executed_steps == len(actions)
        if state.failure_reason is not None:
            state.phase_history.append(
                {
                    "phase": "failed",
                    "reason": state.failure_reason,
                    "time_s": float(env.sim.data.time),
                }
            )
        result = metrics.result(state, _warning_count(env))
        result.update(
            {
                "schema": "shakebench_spike.stage2c.frozen_episode.v1",
                "strategy": "frozen_replay",
                "environment_variant": environment_variant,
                "seed": seed,
                "gamma": gamma,
                "condition": asdict(condition),
                "source_tape_gamma": tape["condition"]["gamma"],
                "source_tape_seed": tape["seed"],
                "source_action_count": len(actions),
                "executed_policy_steps": executed_steps,
                "recorded_force_switch_policy_step_index": switch_step,
                "gripper_force_switch_history": env.gripper_force_switch_history,
                "gripper_force_validation_history": env.gripper_force_validation_history,
                "cube_table_airborne_200hz": airborne_summary(
                    airborne_phase_counts,
                    diagnostic_frequency_hz=condition.diagnostic_frequency_hz,
                ),
                "calibration": calibration,
                "table_isolator": (
                    table_isolator.as_dict() if table_isolator is not None else None
                ),
            }
        )
        return result
    finally:
        env.close()


def run_a3(output_root: Path) -> dict:
    condition = Condition(gamma=0.95)
    first = run_reactive_episode(0, condition)
    second = run_reactive_episode(0, condition)
    first_switch = _force_switch_event(first)
    second_switch = _force_switch_event(second)
    first_actions = np.asarray(first["actions"], dtype=np.float64)
    second_actions = np.asarray(second["actions"], dtype=np.float64)
    action_error = float(np.max(np.abs(first_actions - second_actions)))
    trace_equal = first["acquisition_trace_200hz"] == second["acquisition_trace_200hz"]
    replay = run_frozen_episode(0, 0.95, first)
    replay_switch = _force_switch_event(replay)
    result = {
        "schema": "shakebench_spike.stage2c.grip_authority_split.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "acquisition_force_limit_n_per_finger": 20.0,
            "hold_force_limit_n_per_finger": 3.0,
            "switch_trigger_reactive": "first bilateral latch",
            "switch_trigger_frozen": "recorded policy-step index; no online observation",
            "runtime_mutation": "model.actuator_forcerange for both Panda finger actuators",
        },
        "reactive_switch": first_switch,
        "acquisition_validation": first["gripper_force_validation_history"][0],
        "hold_validation": first_switch["post_switch_validation"],
        "acquisition_trace_summary": _trace_summary(first["acquisition_trace_200hz"]),
        "acquisition_trace_200hz": first["acquisition_trace_200hz"],
        "determinism": {
            "same_seed": 0,
            "gamma": 0.95,
            "max_abs_action_difference": action_error,
            "actions_bitwise_equal": bool(np.array_equal(first_actions, second_actions)),
            "acquisition_trace_structurally_equal": trace_equal,
            "first_switch_policy_step_index": first_switch["policy_step_index"],
            "second_switch_policy_step_index": second_switch["policy_step_index"],
            "first_switch_model_step_index": first_switch["model_step_index"],
            "second_switch_model_step_index": second_switch["model_step_index"],
            "passed": action_error == 0.0
            and trace_equal
            and first_switch["policy_step_index"] == second_switch["policy_step_index"]
            and first_switch["model_step_index"] == second_switch["model_step_index"],
        },
        "frozen_replay_switch_check": {
            "source_policy_step_index": first_switch["policy_step_index"],
            "replay_policy_step_index": replay_switch["policy_step_index"],
            "source_model_step_index": first_switch["model_step_index"],
            "replay_model_step_index": replay_switch["model_step_index"],
            "replay_trigger": replay_switch["trigger"],
            "replay_success": replay["success"],
            "passed": first_switch["policy_step_index"]
            == replay_switch["policy_step_index"]
            and first_switch["model_step_index"] == replay_switch["model_step_index"],
        },
        "engineering_episode": {
            key: value
            for key, value in first.items()
            if key not in ("actions", "acquisition_trace_200hz")
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "grip_authority_split.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_b1_prime(output_root: Path) -> dict:
    summaries = {}
    episodes_by_gamma = {}
    for gamma in (0.0, 0.5, 0.95):
        slug = str(gamma).replace(".", "p")
        summary, episodes = run_condition(
            Condition(gamma=gamma),
            FORMAL_SEEDS,
            output_root / "exp1_rerun" / f"gamma_{slug}",
        )
        summaries[str(gamma)] = summary
        episodes_by_gamma[str(gamma)] = episodes
    result = {
        "schema": "shakebench_spike.stage2c.b1_prime.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": summaries,
        "gate_1": {
            "criterion": "A3 force-control gamma=0 success count >=19/20",
            "success_count": summaries["0.0"]["success_count"],
            "episode_count": 20,
            "passed": summaries["0.0"]["success_count"] >= 19,
        },
    }
    (output_root / "exp1_rerun" / "b1_prime_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_condition(output_dir: Path, condition: Condition) -> tuple[dict, list[dict]]:
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("seed_*.json"))
    ]
    if len(episodes) != 20 or any(
        episode.get("condition") != asdict(condition) for episode in episodes
    ):
        raise RuntimeError(f"incomplete or mismatched baseline: {output_dir}")
    return summarize_condition(condition, episodes), episodes


def _flip_scan(
    parameter: str,
    baseline_value,
    alternate_value,
    baseline_summary: dict,
    baseline_episodes: list[dict],
    alternate_summary: dict,
    alternate_episodes: list[dict],
) -> dict:
    by_seed = {episode["seed"]: episode for episode in baseline_episodes}
    pairs = []
    for alternate in alternate_episodes:
        baseline = by_seed[alternate["seed"]]
        pairs.append(
            {
                "seed": alternate["seed"],
                "baseline_success": baseline["success"],
                "alternate_success": alternate["success"],
                "flipped": baseline["success"] != alternate["success"],
                "baseline_failure_reason": baseline["failure_reason"],
                "alternate_failure_reason": alternate["failure_reason"],
            }
        )
    flip_count = sum(row["flipped"] for row in pairs)
    return {
        "schema": "shakebench_spike.stage2c.sensitivity_flip.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gamma": 0.95,
        "parameter": parameter,
        "baseline_value": baseline_value,
        "alternate_value": alternate_value,
        "baseline": baseline_summary,
        "alternate": alternate_summary,
        "paired_results": pairs,
        "flip_count": flip_count,
        "flip_denominator": len(pairs),
        "flip_fraction": flip_count / len(pairs),
        "gate_parameter_passed": flip_count < 0.5 * len(pairs),
    }


def run_b2_prime(output_root: Path) -> dict:
    sensitivity = output_root / "sensitivity"
    sensitivity.mkdir(parents=True, exist_ok=True)
    baseline_condition = Condition(gamma=0.95)
    baseline_summary, baseline_episodes = _load_condition(
        output_root / "exp1_rerun" / "gamma_0p95", baseline_condition
    )
    failed_seeds = [
        episode["seed"] for episode in baseline_episodes if not episode["success"]
    ]
    scan_seeds = failed_seeds or FORMAL_SEEDS
    seed_rule = (
        "B1' gamma=0.95 failures only"
        if failed_seeds
        else "all 20 seeds because B1' gamma=0.95 failure set is empty"
    )

    pad_condition = Condition(gamma=0.95, pad_solref_dampratio=1.0)
    pad_summary, pad_episodes = run_condition(
        pad_condition,
        scan_seeds,
        sensitivity / "pad_damping_runs" / "damping_1p0",
    )
    pad = _flip_scan(
        "pad_solref_dampratio",
        0.5,
        1.0,
        baseline_summary,
        baseline_episodes,
        pad_summary,
        pad_episodes,
    )
    pad["seed_selection_rule"] = seed_rule
    (sensitivity / "pad_damping.json").write_text(
        json.dumps(pad, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    force_summaries = {}
    for force_n in (1.0, 1.5, 3.0, 6.0):
        key = str(force_n)
        if force_n == 3.0:
            force_summaries[key] = baseline_summary
        else:
            force_summaries[key], _ = run_condition(
                Condition(gamma=0.95, gripper_force_limit_n=force_n),
                scan_seeds,
                sensitivity
                / "grip_hold_runs"
                / f"hold_{str(force_n).replace('.', 'p')}n",
            )
    curve = []
    for force_n in (1.0, 1.5, 3.0, 6.0):
        summary = force_summaries[str(force_n)]
        curve.append(
            {
                "hold_force_limit_n_per_finger": force_n,
                "success_count": summary["success_count"],
                "episode_count": summary["episode_count"],
                "success_rate": summary["success_rate"],
                "max_grasp_slip_distribution_m": summary[
                    "max_grasp_slip_distribution_m"
                ],
                "both_below_threshold_fraction_distribution": summary[
                    "post_latch_contact_loss_fraction_distributions"
                ]["both_below_threshold_fraction"],
                "failure_reason_histogram": summary["failure_reason_histogram"],
            }
        )
    sr = {row["hold_force_limit_n_per_finger"]: row["success_rate"] for row in curve}
    deltas = {
        "vs_1p5_percentage_points": 100.0 * abs(sr[3.0] - sr[1.5]),
        "vs_6p0_percentage_points": 100.0 * abs(sr[3.0] - sr[6.0]),
    }
    hold = {
        "schema": "shakebench_spike.stage2c.grip_hold_plateau.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gamma": 0.95,
        "acquisition_force_limit_n_per_finger": 20.0,
        "seed_selection_rule": seed_rule,
        "prediction_before_run": "all four hold-force points succeed 100%",
        "curve": curve,
        "adjacent_success_rate_deltas": deltas,
        "plateau_criterion": "3 N differs from 1.5 N and 6 N by <5 percentage points",
        "three_newton_on_platform": all(value < 5.0 for value in deltas.values()),
    }
    (sensitivity / "grip_hold_plateau.json").write_text(
        json.dumps(hold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    specs = (
        ("osc_kp", 150.0, 300.0, Condition(gamma=0.95, osc_kp=300.0), "osc_kp.json"),
        (
            "physics_timestep_s",
            2.0e-4,
            1.0e-4,
            Condition(gamma=0.95, physics_timestep_s=1.0e-4),
            "timestep.json",
        ),
        (
            "cube_table_solref_timeconst_s",
            6.0e-4,
            1.2e-3,
            Condition(gamma=0.95, cube_table_solref_timeconst_s=1.2e-3),
            "cube_table_solref.json",
        ),
        (
            "move_action_gain",
            4.0,
            1.0,
            Condition(gamma=0.95, move_action_gain=1.0),
            "move_action_gain.json",
        ),
    )
    regular = {}
    for parameter, baseline_value, alternate_value, condition, filename in specs:
        alt_summary, alt_episodes = run_condition(
            condition,
            scan_seeds,
            sensitivity / f"{parameter}_runs" / "alternate",
        )
        result = _flip_scan(
            parameter,
            baseline_value,
            alternate_value,
            baseline_summary,
            baseline_episodes,
            alt_summary,
            alt_episodes,
        )
        result["seed_selection_rule"] = seed_rule
        if parameter == "move_action_gain":
            result["diagnostic_only_not_in_gate_2"] = True
        (sensitivity / filename).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        regular[parameter] = result

    gated = [pad] + [
        regular[name]
        for name in (
            "osc_kp",
            "physics_timestep_s",
            "cube_table_solref_timeconst_s",
        )
    ]
    gate = {
        "schema": "shakebench_spike.stage2c.gate2.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gamma": 0.95,
        "grip_hold_three_newton_on_platform": hold["three_newton_on_platform"],
        "regular_parameter_passes": {
            result["parameter"]: result["gate_parameter_passed"] for result in gated
        },
        "move_action_gain_diagnostic": {
            "flip_fraction": regular["move_action_gain"]["flip_fraction"],
            "excluded_from_gate": True,
        },
        "passed": hold["three_newton_on_platform"]
        and all(result["gate_parameter_passed"] for result in gated),
    }
    (sensitivity / "gate2_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return gate


def _episode_summary(episodes: list[dict]) -> dict:
    successes = sum(bool(episode["success"]) for episode in episodes)
    return {
        "episode_count": len(episodes),
        "success_count": successes,
        "success_rate": successes / len(episodes),
        "failure_reason_histogram": dict(
            sorted(
                Counter(
                    episode["failure_reason"]
                    for episode in episodes
                    if not episode["success"]
                ).items()
            )
        ),
        "ee_wobble_base_distribution_m": _distribution(
            [episode["ee_wobble_base_m"] for episode in episodes]
        ),
        "table_frame_object_slip_distribution_m": _distribution(
            [episode["obj_slip_on_table_m"] for episode in episodes]
        ),
        "max_grasp_slip_distribution_m": _distribution(
            [episode["max_grasp_slip_m"] for episode in episodes]
        ),
        "base_frame_table_motion_distribution_m": _distribution(
            [episode.get("base_frame_table_motion_m", 0.0) for episode in episodes]
        ),
        "mujoco_warning_count": sum(
            episode["mujoco_warning_count"] for episode in episodes
        ),
    }


def _load_or_run_reactive(
    path: Path,
    seed: int,
    condition: Condition,
    *,
    table_motion_sampler=None,
    environment_variant: str,
) -> dict:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("condition") == asdict(condition)
            and cached.get("environment_variant") == environment_variant
        ):
            print(
                f"resume reactive seed={seed:03d} gamma={condition.gamma:.2f} "
                f"success={cached['success']}",
                flush=True,
            )
            return cached
    episode = run_reactive_episode(
        seed,
        condition,
        table_motion_sampler=table_motion_sampler,
        environment_variant=environment_variant,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"reactive seed={seed:03d} gamma={condition.gamma:.2f} "
        f"success={episode['success']} failure={episode['failure_reason']} "
        f"slip_mm={1000 * episode['max_grasp_slip_m']:.3f}",
        flush=True,
    )
    return episode


def _load_or_run_frozen(
    path: Path,
    seed: int,
    gamma: float,
    tape: dict,
    *,
    table_motion_sampler=None,
    environment_variant: str,
) -> dict:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("gamma") == gamma
            and cached.get("source_tape_seed") == seed
            and cached.get("environment_variant") == environment_variant
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
        table_motion_sampler=table_motion_sampler,
        environment_variant=environment_variant,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"frozen seed={seed:03d} gamma={gamma:.2f} "
        f"success={episode['success']} failure={episode['failure_reason']} "
        f"slip_mm={1000 * episode['max_grasp_slip_m']:.3f}",
        flush=True,
    )
    return episode


def _hard_mounted_table_sampler(_gamma: float, _seed: int, _condition: Condition):
    return None


D2_NATURAL_FREQUENCY_HZ = 8.0
D2_DAMPING_RATIO = 0.20


def _second_order_table_sampler(gamma: float, seed: int, condition: Condition):
    if gamma == 0.0:
        return ZeroVibration().sample
    base, _calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=int(round(1.0 / condition.physics_timestep_s)),
        episode_s=condition.episode_limit_s,
    )
    table = SecondOrderSupportedVibration(
        base.config,
        natural_frequency_hz=D2_NATURAL_FREQUENCY_HZ,
        damping_ratio=D2_DAMPING_RATIO,
    )
    return table.sample


def _run_ladder(
    output_dir: Path,
    *,
    condition_factory,
    table_sampler_factory,
    environment_variant: str,
) -> dict:
    """Run 10 initial states x 6 Gammas x reactive/frozen, resumably."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for seed in C_SEEDS:
        condition = condition_factory(0.0)
        sources[seed] = _load_or_run_reactive(
            output_dir / "reactive_scripted" / "gamma_0p0" / f"seed_{seed:03d}.json",
            seed,
            condition,
            table_motion_sampler=table_sampler_factory(0.0, seed, condition),
            environment_variant=environment_variant,
        )

    rows = {}
    for gamma in GAMMAS:
        slug = str(gamma).replace(".", "p")
        reactive = []
        frozen = []
        for seed in C_SEEDS:
            condition = condition_factory(gamma)
            table_sampler = table_sampler_factory(gamma, seed, condition)
            if gamma == 0.0:
                reactive_episode = sources[seed]
            else:
                reactive_episode = _load_or_run_reactive(
                    output_dir / "reactive_scripted" / f"gamma_{slug}" / f"seed_{seed:03d}.json",
                    seed,
                    condition,
                    table_motion_sampler=table_sampler,
                    environment_variant=environment_variant,
                )
            frozen_episode = _load_or_run_frozen(
                output_dir / "frozen_replay" / f"gamma_{slug}" / f"seed_{seed:03d}.json",
                seed,
                gamma,
                sources[seed],
                table_motion_sampler=table_sampler,
                environment_variant=environment_variant,
            )
            reactive.append(reactive_episode)
            frozen.append(frozen_episode)
        rows[str(gamma)] = {
            "reactive_scripted": _episode_summary(reactive),
            "frozen_replay": _episode_summary(frozen),
        }
    result = {
        "schema": "shakebench_spike.stage2c.ladder.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment_variant": environment_variant,
        "seeds": C_SEEDS,
        "gammas": list(GAMMAS),
        "action_tape": {
            "recorded_at_gamma": 0.0,
            "contents": "policy.command() output and gripper-force switch policy-step index",
            "replay_switch": "open-loop at recorded policy-step index",
        },
        "rows": rows,
    }
    (output_dir / "ladder_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _classify_ladder(ladder: dict) -> dict:
    reactive = [ladder["rows"][str(g)]["reactive_scripted"]["success_rate"] for g in GAMMAS]
    frozen = [ladder["rows"][str(g)]["frozen_replay"]["success_rate"] for g in GAMMAS]
    reactive_flat = len(set(reactive)) == 1
    frozen_flat = len(set(frozen)) == 1
    if reactive_flat and frozen_flat:
        interpretation = "both_flat"
    elif reactive_flat and frozen[-1] < frozen[0]:
        interpretation = "frozen_declines_reactive_flat"
    elif reactive[-1] < reactive[0] and frozen[-1] < frozen[0]:
        interpretation = "both_decline"
    elif max(reactive + frozen) == 0.0:
        interpretation = "both_near_zero"
    else:
        interpretation = "mixed_or_nonmonotonic"
    return {
        "interpretation": interpretation,
        "reactive_success_rates": reactive,
        "frozen_success_rates": frozen,
        "reactive_flat": reactive_flat,
        "frozen_flat": frozen_flat,
        "stage_d_triggered": interpretation == "both_flat",
    }


def run_c(output_root: Path) -> dict:
    gate1 = json.loads(
        (output_root / "exp1_rerun" / "b1_prime_summary.json").read_text(encoding="utf-8")
    )["gate_1"]
    gate2 = json.loads(
        (output_root / "sensitivity" / "gate2_summary.json").read_text(encoding="utf-8")
    )
    if not gate1["passed"] or not gate2["passed"]:
        raise RuntimeError("Stage C requires both Gate 1 and Gate 2")
    predictions = {
        "schema": "shakebench_spike.stage2c.predeclared_predictions.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "written_before_stage_c_rollouts": True,
        "reactive_success_prediction": "100% at every Gamma",
        "grasp_slip_prediction": (
            "reactive maximum increases monotonically, reaches about 7 mm at "
            "Gamma=0.95, and remains below the fixed 10 mm tolerance"
        ),
        "derivation": "3.798 mm at Gamma=0.5 times EE-wobble ratio 2.380/1.275 = 7.09 mm",
    }
    prediction_path = output_root / "exp2_ladder" / "predeclared_predictions.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if not prediction_path.exists():
        prediction_path.write_text(
            json.dumps(predictions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    ladder = _run_ladder(
        output_root / "exp2_ladder",
        condition_factory=lambda gamma: Condition(gamma=gamma),
        table_sampler_factory=_hard_mounted_table_sampler,
        environment_variant="hard_mounted_mu_1p5",
    )
    interpretation = _classify_ladder(ladder)
    reactive_rates = [
        ladder["rows"][str(g)]["reactive_scripted"]["success_rate"] for g in GAMMAS
    ]
    slip_maxima = [
        ladder["rows"][str(g)]["reactive_scripted"]["max_grasp_slip_distribution_m"]["max"]
        for g in GAMMAS
    ]
    gamma95_max = slip_maxima[-1]
    result = {
        "schema": "shakebench_spike.stage2c.c_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate_1": gate1,
        "gate_2": gate2,
        "interpretation": interpretation,
        "prediction_comparison": {
            "reactive_100_percent_predicted": True,
            "reactive_100_percent_observed": all(rate == 1.0 for rate in reactive_rates),
            "slip_monotonic_predicted": True,
            "slip_monotonic_observed": all(
                later >= earlier for earlier, later in zip(slip_maxima, slip_maxima[1:])
            ),
            "gamma_0p95_predicted_max_grasp_slip_m": 0.00709,
            "gamma_0p95_observed_max_grasp_slip_m": gamma95_max,
            "gamma_0p95_below_tolerance": gamma95_max < SLIP_TOLERANCE_M,
        },
        "ladder": ladder,
    }
    (output_root / "exp2_ladder" / "c_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def write_tolerance_matrix(output_root: Path, c_result: dict | None = None) -> dict:
    instrument = json.loads(
        (
            output_root.parent
            / "stage2b"
            / "decoupled_instrument"
            / "decoupled_instrument.json"
        ).read_text(encoding="utf-8")
    )
    gamma95_row = next(row for row in instrument["rows"] if row["gamma"] == 0.95)
    hard_ee = 0.002380
    grasp_slip = None
    grasp_source = "Stage2C C reactive_scripted, n=10"
    if c_result is not None:
        grasp_slip = c_result["ladder"]["rows"]["0.95"]["reactive_scripted"][
            "max_grasp_slip_distribution_m"
        ]["max"]
    else:
        b1 = json.loads(
            (output_root / "exp1_rerun" / "b1_prime_summary.json").read_text(
                encoding="utf-8"
            )
        )
        grasp_slip = b1["conditions"]["0.95"]["max_grasp_slip_distribution_m"][
            "max"
        ]
        grasp_source = "Stage2C B1' reactive_scripted, n=20; C gated off"
    rows = [
        {
            "measured_quantity": "hard-mounted EE wobble",
            "gamma_0p95_value_m": hard_ee,
            "task_tolerance_m": 0.004,
            "config_source": "src/shakebench/config.py:439 descend_clearance_m",
            "fraction_of_tolerance": hard_ee / 0.004,
            "measurement_source": "Stage2B B3 validated instrument",
        },
        {
            "measured_quantity": "hard-mounted maximum grasp slip",
            "gamma_0p95_value_m": grasp_slip,
            "task_tolerance_m": 0.010,
            "config_source": "src/shakebench/config.py:456 grasp_slip_tolerance_m",
            "fraction_of_tolerance": None if grasp_slip is None else grasp_slip / 0.010,
            "measurement_source": grasp_source,
        },
        {
            "measured_quantity": "table-frame object slip",
            "gamma_0p95_value_m": 0.000000700,
            "task_tolerance_m": None,
            "config_source": None,
            "fraction_of_tolerance": None,
            "measurement_source": "Stage2B B3 validated instrument",
        },
        {
            "measured_quantity": "decoupled base-frame table motion (pi/2 diagnostic)",
            "gamma_0p95_value_m": gamma95_row["base_frame_table_motion_m"],
            "task_tolerance_m": 0.004,
            "config_source": "src/shakebench/config.py:439 descend_clearance_m",
            "fraction_of_tolerance": gamma95_row["base_frame_table_motion_m"] / 0.004,
            "measurement_source": "Stage2B B3 validated instrument; diagnostic phase model only",
        },
        {
            "measured_quantity": "hard-mounted EE wobble",
            "gamma_0p95_value_m": hard_ee,
            "task_tolerance_m": 0.012,
            "config_source": "src/shakebench/config.py:440 finger_table_clearance_m",
            "fraction_of_tolerance": hard_ee / 0.012,
            "measurement_source": "Stage2B B3 validated instrument",
        },
    ]
    result = {
        "schema": "shakebench_spike.stage2c.tolerance_matrix.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prominent_observation": (
            "The Stage2B pi/2 decoupled diagnostic table motion is the first measured "
            "quantity above a task tolerance. It is diagnostic-only: if D2 is later "
            "reached, the free phase choice must be replaced by a physically explicit "
            "transfer function before policy rollout."
        ),
        "rows": rows,
    }
    (output_root / "tolerance_matrix.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_d(output_root: Path, c_result: dict) -> dict:
    probe_dir = output_root / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not c_result["interpretation"]["stage_d_triggered"]:
        result = {
            "schema": "shakebench_spike.stage2c.d_summary.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "triggered": False,
            "reason": "Stage C was not classified as both_flat",
        }
        (probe_dir / "d_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    d1 = _run_ladder(
        probe_dir / "friction_mu_0p4",
        condition_factory=lambda gamma: Condition(gamma=gamma, cube_table_sliding_mu=0.4),
        table_sampler_factory=_hard_mounted_table_sampler,
        environment_variant="hard_mounted_mu_0p4_design_probe",
    )
    coherence = {
        "schema": "shakebench_spike.stage2c.coherence_model.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "base-excited linear second-order absolute-displacement transfer",
        "transfer_function": (
            "H(jw)=X_table/Y_base=(wn^2 + 2*j*zeta*wn*w)/"
            "(wn^2 - w^2 + 2*j*zeta*wn*w)"
        ),
        "natural_frequency_hz": D2_NATURAL_FREQUENCY_HZ,
        "damping_ratio": D2_DAMPING_RATIO,
        "application": "same exploratory isotropic response on all six spectral axes",
        "physical_basis": (
            "standard SDOF base-excitation response for a rigid payload on spring-damper "
            "supports; each authored spectral line receives H's amplitude and phase"
        ),
        "scope_and_uncertainty": (
            "Design probe, not a fitted hardware model. The 8 Hz, zeta=0.20 values are "
            "predeclared representative soft-support parameters and must be calibrated "
            "before benchmark claims. They remove the free uniform pi/2 phase knob."
        ),
        "comparison_to_stage2b": (
            "Stage2B imposed pi/2 at every line, making relative amplitude a direct free "
            "choice; this model instead makes phase frequency-dependent and coupled to gain."
        ),
    }
    (probe_dir / "coherence_model.json").write_text(
        json.dumps(coherence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    d2 = _run_ladder(
        probe_dir / "second_order_table",
        condition_factory=lambda gamma: Condition(gamma=gamma),
        table_sampler_factory=_second_order_table_sampler,
        environment_variant="second_order_decoupled_table_fn8Hz_zeta0p20",
    )
    result = {
        "schema": "shakebench_spike.stage2c.d_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "triggered": True,
        "trigger": "Stage C reactive and frozen success-rate rows were both flat",
        "d1_design_change": "cube-table sliding friction mu 1.5 -> 0.4",
        "d1": d1,
        "d2_coherence_model": coherence,
        "d2": d2,
        "claim_scope": "design probes only; no benchmark-failure claim",
    }
    (probe_dir / "d_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    logger = logging.getLogger("robosuite_logs")
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("a3", "b1", "b2", "c", "d0", "d", "remaining"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "out" / "stage2c",
    )
    args = parser.parse_args()
    if args.stage == "a3":
        result = run_a3(args.output)
        passed = result["determinism"]["passed"] and result[
            "frozen_replay_switch_check"
        ]["passed"]
        print(json.dumps({"a3_passed": passed}, sort_keys=True), flush=True)
        return 0 if passed else 2
    if args.stage == "b1":
        result = run_b1_prime(args.output)
        print(json.dumps(result["gate_1"], sort_keys=True), flush=True)
        return 0 if result["gate_1"]["passed"] else 2
    if args.stage == "b2":
        result = run_b2_prime(args.output)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 2
    if args.stage == "c":
        result = run_c(args.output)
        write_tolerance_matrix(args.output, result)
        print(json.dumps(result["interpretation"], sort_keys=True), flush=True)
        return 0
    if args.stage == "d0":
        c_path = args.output / "exp2_ladder" / "c_summary.json"
        c_result = json.loads(c_path.read_text(encoding="utf-8")) if c_path.exists() else None
        result = write_tolerance_matrix(args.output, c_result)
        print(json.dumps({"rows": len(result["rows"])}, sort_keys=True), flush=True)
        return 0
    if args.stage == "d":
        c_result = json.loads(
            (args.output / "exp2_ladder" / "c_summary.json").read_text(encoding="utf-8")
        )
        result = run_d(args.output, c_result)
        print(json.dumps({"triggered": result["triggered"]}, sort_keys=True), flush=True)
        return 0
    if args.stage == "remaining":
        c_result = run_c(args.output)
        write_tolerance_matrix(args.output, c_result)
        d_result = run_d(args.output, c_result)
        print(
            json.dumps(
                {
                    "c_interpretation": c_result["interpretation"]["interpretation"],
                    "d_triggered": d_result["triggered"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
