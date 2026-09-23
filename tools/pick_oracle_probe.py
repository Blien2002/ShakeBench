"""Run one closed-loop oracle episode and report where the state machine stops.

This is the cheap loop for per-object oracle calibration: no evidence archive,
no dataset, just the phase trace, the recovery reason, and the final success
subconditions.  One environment build costs about a minute on its own
(the scene clearance audit dominates), so run one object per process and put
the ones you need side by side.

Usage:
    PYTHONPATH=. python3 -u tools/pick_oracle_probe.py <object_id|legacy> [steps]

Examples:
    PYTHONPATH=. python3 -u tools/pick_oracle_probe.py can 400
    for id in mug apple cereal; do
      PYTHONPATH=. python3 -u tools/pick_oracle_probe.py "$id" 400 >"/tmp/$id.log" 2>&1 &
    done

Read the trace bottom up: the last phase line is the farthest phase reached.
``verify`` means the placement was executed; a latched success additionally
needs the final ``conditions`` line to show every subcondition true and
``reward`` to read 1.0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shakebench.utils.expert import oracle_observation  # noqa: E402
from shakebench.utils.oracle import ShakeBenchOracleController, WorktableTaskContext  # noqa: E402
from shakebench.utils.tasks import OBJECTS, TaskSpec, make_task_env  # noqa: E402


def build(object_id: str, steps: int):
    if object_id == "legacy":
        from shakebench.environments.vibration_pick_place import VibrationPickPlace

        return VibrationPickPlace(hard_reset=False, horizon=steps, seed=5)
    if object_id not in OBJECTS:
        raise SystemExit(f"unknown object_id {object_id!r}; known: legacy, {', '.join(OBJECTS)}")
    return make_task_env(TaskSpec(object_id=object_id), hard_reset=False, horizon=steps, seed=5)


def main() -> int:
    object_id = sys.argv[1] if len(sys.argv) > 1 else "can"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    start = time.time()
    env = build(object_id, steps)
    context = WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
    controller = ShakeBenchOracleController(task_context=context)
    profile = controller.profile
    print(
        f"built in {time.time() - start:.1f} s | grasp_height {profile.grasp_height_m:.4f} "
        f"placement {profile.placement_height_m:.4f} opening_gate {profile.grasp_hold_max_opening_rad:.4f} "
        f"offset_object {profile.grasp_offset_object_m}",
        flush=True,
    )
    observation = oracle_observation(env, env.reset())
    print(
        f"reset at {time.time() - start:.1f} s | object {np.round(observation['object_pos_robot_base'], 4)}", flush=True
    )
    last_phase = None
    reward = 0.0
    for step in range(steps):
        action = np.clip(controller.action(observation, time_s=step / 20.0), -1.0, 1.0)
        observation, reward, done, _ = env.step(action)
        observation = oracle_observation(env, observation)
        phase = controller.executive.phase.value
        if phase != last_phase:
            reason = controller.executive.last_recovery_reason or controller.executive.abort_reason or ""
            print(
                f"  step {step:4d} t={step / 20.0:5.2f}s phase {phase}" f"{' reason ' + reason if reason else ''}",
                flush=True,
            )
            last_phase = phase
        if done or controller.abort_requested:
            break
    metrics = env.get_metrics(update=True)
    success = metrics.get("success", {})
    print(f"final {object_id} reward {reward} conditions {dict(success.get('subconditions', {}))}", flush=True)
    print(
        f"object world {np.round(env.sim.data.body_xpos[env.can_body_id], 4)} "
        f"target {np.round(env.target_frame_world_position(), 4)}",
        flush=True,
    )
    env.close()
    print(f"elapsed {time.time() - start:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
