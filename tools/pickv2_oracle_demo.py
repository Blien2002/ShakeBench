"""Record one full oracle pick-and-place rollout for a task variant.

Reuses the qualitative demo recorder (`VideoObserver`) and the oracle episode
runner, but takes its state from the frozen task-state artifact instead of the
Phase 07 dev list, so any registered object can be recorded at a requested
Gamma.  One object per process; the environment build dominates the runtime.

Usage:
    MUJOCO_GL=egl PYTHONPATH=. python3 -u tools/pickv2_oracle_demo.py \
        --object mug --grasp-region handle --gamma 0.5 --output out/pickv2_demos/mug_handle.mp4

The episode stops on its own at the oracle abort or at the latched success, so
``--horizon-steps`` is only an upper bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shakebench import models  # noqa: E402
from shakebench.demos.demo_oracle_video import VideoObserver  # noqa: E402
from shakebench.scripts.run_oracle import run_episode  # noqa: E402
from shakebench.utils.geometry import (  # noqa: E402
    DEFAULT_GEOMETRY_PROFILE,
    geometry_scene_path,
)
from shakebench.utils.oracle import OracleControllerProfile  # noqa: E402
from shakebench.utils.scene import load_scene_visual_config  # noqa: E402
from shakebench.utils.tasks import OBJECTS, TaskSpec  # noqa: E402

DEFAULT_STATES = Path(models.assets_root, "shakebench_task_states_official_v3.json")


def select_task_state(path: Path, object_id: str, occurrence: int = 0) -> dict:
    """Return the n-th frozen state of one object variant."""

    rows = json.loads(Path(path).read_text(encoding="utf-8"))["states"]
    matches = [row for row in rows if row["task"]["object_id"] == object_id]
    if not matches:
        raise SystemExit(f"no frozen state for object {object_id!r} in {path}")
    return matches[occurrence % len(matches)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, choices=tuple(OBJECTS))
    parser.add_argument("--grasp-region", choices=("body", "handle"), default="body")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--state-occurrence", type=int, default=0)
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="presentation")
    parser.add_argument("--tier", default="V3")
    parser.add_argument("--geometry-profile", default=DEFAULT_GEOMETRY_PROFILE)
    parser.add_argument("--wrist-inset", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.gamma) or args.gamma < 0.0:
        raise SystemExit("--gamma must be finite and non-negative")

    TaskSpec(object_id=args.object).grasp_plan(args.grasp_region)
    state = select_task_state(args.states, args.object, args.state_occurrence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = OracleControllerProfile()
    observer = VideoObserver(
        args.output,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        tier=args.tier,
        gamma=args.gamma,
        state_id=state["state_id"],
        policy_rate_hz=profile.policy_rate_hz,
        scene_config=load_scene_visual_config(geometry_scene_path(args.geometry_profile)),
        wrist_inset=args.wrist_inset,
    )
    initial_height = None
    max_lift = 0.0
    carry_contacts = set()
    last_phase = None

    def record_step(env, step, observation, controller):
        nonlocal initial_height, max_lift, last_phase
        height = float(env.sim.data.body_xpos[env.can_body_id][2] - env.sim.data.body_xpos[env.worktable_body_id][2])
        if initial_height is None:
            initial_height = height
        max_lift = max(max_lift, height - initial_height)
        phase = controller.executive.phase.value
        if phase != last_phase:
            print(
                f"step {step}: {phase}; reason={controller.executive.last_recovery_reason}; lift={height - initial_height:.4f}",
                flush=True,
            )
            last_phase = phase
        if phase in {"lift", "transport"}:
            for contact in env.sim.data.contact:
                names = [env.sim.model.geom_id2name(index) for index in (contact.geom1, contact.geom2)]
                for finger, obj in (names, names[::-1]):
                    if finger in env.finger_pad_geom_names and obj in env.can.contact_geoms:
                        geom_id = env.sim.model.geom_name2id(obj)
                        mesh_id = int(env.sim.model.geom_dataid[geom_id])
                        mesh = env.sim.model.mesh_id2name(mesh_id) if mesh_id >= 0 else obj
                        carry_contacts.add((finger, mesh))
        observer(env, step, observation, controller)

    try:
        episode = run_episode(
            state,
            gamma_commanded=args.gamma,
            profile=profile,
            horizon_steps=args.horizon_steps,
            hard_reset=False,
            step_observer=record_step,
            grasp_region=args.grasp_region,
            geometry_profile=args.geometry_profile,
        )
    except BaseException:
        observer.close(success=False, hold_seconds=0.0)
        raise
    observer.close(success=bool(episode["success"]))
    metadata = {
        "object_id": args.object,
        "grasp_region": args.grasp_region,
        "state_id": state["state_id"],
        "gamma_commanded": args.gamma,
        "success": bool(episode["success"]),
        "max_object_lift_m": max_lift,
        "carry_pad_contacts": sorted(carry_contacts),
        "termination_category": episode["termination_category"],
        "failure_reason": episode["failure_reason"],
        "steps": len(episode["trace"]),
        "controller_events": episode["controller_events"],
        "video": str(args.output),
        "scoreable": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
