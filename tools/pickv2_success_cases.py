"""Check the success rule against inside / outside / straddling placements.

For every object this builds the real environment, teleports the object to a
scripted pose, lets it settle, and reports the evaluator's subconditions.  The
three placements are the acceptance cases the task rule has to separate:

* ``inside``     - resting on the crate floor, fully inside the inner
                   footprint: must reach the success latch;
* ``outside``    - resting on the table beside the crate: must fail;
* ``straddling`` - balanced on a crate wall: must fail.

Usage: python tools/pickv2_success_cases.py [--objects mug apple ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shakebench.models.arenas.shakebench_arena import (  # noqa: E402
    TARGET_CONTAINER_BOTTOM_THICKNESS_M,
    TARGET_CONTAINER_CENTER_XY_M,
    TARGET_CONTAINER_INNER_XY_M,
    TARGET_CONTAINER_OUTER_XY_M,
    TARGET_CONTAINER_WALL_HEIGHT_M,
)
from shakebench.utils.tasks import make_task_env, task_variants  # noqa: E402


def place(env, xy_center, *, height_m: float) -> None:
    """Write the object's free joint at ``xy`` with its lowest point ``height_m`` up."""

    lower = env.can_start_pose_envelope[0]
    table_top = float(env.arena.table_top_abs[2])
    position = np.array([xy_center[0], xy_center[1], table_top + height_m - lower], dtype=float)
    env.sim.data.set_joint_qpos(
        env.can.joints[0],
        np.concatenate((position, np.asarray(env.object_start_quat_wxyz, dtype=float))),
    )
    env.sim.data.qvel[:] = 0.0
    env.sim.forward()


def settle(env, seconds: float = 1.0) -> None:
    model, data = env.sim.model._model, env.sim.data._data
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def evaluate(env) -> dict:
    metrics = env.get_metrics(update=True)
    success = metrics.get("success", {})
    return {
        "passed": bool(success.get("passed", False)),
        "subconditions": dict(success.get("subconditions", {})),
    }


def run_cases(spec, cases) -> list[dict]:
    """Run every placement in one environment: building one costs a full audit."""

    env = make_task_env(spec, hard_reset=False, horizon=200, seed=11)
    results = []
    try:
        env.reset()
        center = np.asarray(TARGET_CONTAINER_CENTER_XY_M, dtype=float) + np.asarray(env.arena.table_top_abs[:2])
        for case in cases:
            env.reset()
            if case == "inside":
                place(env, center, height_m=TARGET_CONTAINER_BOTTOM_THICKNESS_M + 0.002)
                settle(env, 1.2)
            elif case == "outside":
                offset = np.array([0.0, -(TARGET_CONTAINER_OUTER_XY_M[1] / 2.0 + 0.06)])
                place(env, center + offset, height_m=0.002)
                settle(env, 0.6)
            elif case == "straddling":
                # Rest on the near wall: half the object overhangs the inner floor.
                wall_y = TARGET_CONTAINER_OUTER_XY_M[1] / 2.0 - 0.004
                place(
                    env,
                    center + np.array([0.0, -wall_y]),
                    height_m=TARGET_CONTAINER_BOTTOM_THICKNESS_M + TARGET_CONTAINER_WALL_HEIGHT_M + 0.004,
                )
                settle(env, 0.8)
            else:
                raise ValueError(case)
            result = evaluate(env)
            result["case"] = case
            result["object_id"] = spec.object_id
            results.append(result)
    finally:
        env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--cases", nargs="*", default=["inside", "outside", "straddling"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    specs = [spec for spec in task_variants() if args.objects is None or spec.object_id in args.objects]
    rows = []
    for spec in specs:
        for result in run_cases(spec, args.cases):
            case = result["case"]
            rows.append(result)
            print(
                f"{spec.object_id:12s} {case:11s} passed={result['passed']!s:5s} {result['subconditions']}", flush=True
            )
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
