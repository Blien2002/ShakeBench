"""Record a qualitative video of any policy driving the ShakeBench task.

`shakebench_evaluate` writes evidence but no pixels, so this demo reuses the same policy
contract (``module:factory``, ``chunk_size``, ``predict(observation)``) and records the
observation the policy actually saw: the evaluation view on the left, the wrist view on the
right.  Action evidence and denominators stay with the evaluator; this is a qualitative
artifact only.

Two ideas come from the reference recorders: LIBERO's ``libero/libero/utils/video_utils.py``
marks the terminal state with a dimmed frame, and VLA-Adapter's ``save_rollout_video`` puts the
outcome in the filename.  This demo streams frames to the in-repo ffmpeg writer instead of
buffering a whole episode in memory, so a 1200-step horizon stays small.

Example (headless NVIDIA EGL, VLA-Adapter served in its own environment):

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m robosuite.demos.demo_shakebench_policy_video \\
        --policy robosuite.utils.shakebench_vla_adapter:make_policy --policy-arg port=10095 \\
        --state-id shakebench-dev-v0-000 --output out/demo/shakebench_policy.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robosuite.demos.demo_shakebench_oracle_video import FFmpegVideoWriter
from robosuite.scripts.shakebench_evaluate import build_policy, load_policy_factory, parse_policy_args
from robosuite.scripts.shakebench_run_oracle import load_state_asset
from robosuite.utils.shakebench_rollout import ShakeBenchTaskEnv

DEFAULT_STATES = Path("robosuite/models/assets/shakebench_states_dev.json")
DEFAULT_OUTPUT = Path("out/demo/shakebench_policy.mp4")


def compose_frame(observation, caption):
    """Side-by-side policy input with a caption; never resampled before the model saw it."""
    main = np.asarray(observation["observation.images.main"], dtype=np.uint8)
    wrist = np.asarray(observation["observation.images.wrist"], dtype=np.uint8)
    frame = np.concatenate((main, wrist), axis=1)
    cv2.putText(frame, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def record_episode(task, policy, writer, *, action_horizon, hold_frames, preview_marks=()):
    """Step the policy in chunks, writing one frame per executed action.

    Frames are composed from the observation that produced the action, so the video shows the
    policy's real input.  Chunk boundaries match `rollout_policy`: predict, execute
    ``action_horizon`` actions, predict again.
    """
    observation, _ = task.reset()
    state_id = task.state.get("state_id")
    steps, policy_calls, termination_cause = 0, 0, None
    previews = {}
    while termination_cause is None:
        actions = policy.predict(observation)
        policy_calls += 1
        for action in actions[:action_horizon]:
            frame = compose_frame(
                observation,
                f"{state_id}  step {steps:03d}  chunk {policy_calls:02d}  " f"gripper {float(action[6]):+.0f}",
            )
            writer.append_data(frame)
            if steps in preview_marks:
                previews[steps] = frame.copy()
            observation, _, terminated, truncated, info = task.step(action)
            steps += 1
            if terminated or truncated:
                termination_cause = info["termination_cause"]
                break
    # LIBERO marks the end of an episode with a dimmed copy of the final frame; hold it for
    # half a second so the outcome is readable while the video is playing.
    success = termination_cause == "success_latched"
    final = compose_frame(observation, f"{state_id}  step {steps:03d}  END: {termination_cause}  " f"success={success}")
    final = cv2.addWeighted(final, 0.45, np.zeros_like(final), 0.0, 0.0)
    for _ in range(max(1, hold_frames)):
        writer.append_data(final)
    previews[steps] = final.copy()
    return {
        "state_id": state_id,
        "steps": steps,
        "policy_calls": policy_calls,
        "termination_cause": termination_cause,
        "success": success,
        "previews": previews,
    }


def write_preview(path, previews):
    """One strip of the recorded marks, so the demo can be judged without playing the video."""
    frames = [previews[step] for step in sorted(previews)]
    if not frames:
        return None
    image = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {path}")
    return str(path)


def outcome_path(output, *, state_id, success):
    """VLA-Adapter's save_rollout_video convention: the outcome belongs in the file name."""
    return output.with_name(f"{output.stem}--state={state_id}--success={success}{output.suffix}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="module:factory returning a predict()-capable policy")
    parser.add_argument("--policy-arg", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--state-id", default="shakebench-dev-v0-000")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--horizon-steps", type=int, default=120)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--inference-timeout-s", type=float, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, help="Defaults to <output>_preview.png")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.action_horizon < 1 or args.fps < 1:
        raise ValueError("action-horizon and fps must be positive")
    states = {state["state_id"]: state for state in load_state_asset(args.states)["states"]}
    if args.state_id not in states:
        raise ValueError(f"unknown state ID {args.state_id!r}; available: {sorted(states)}")
    preview_path = args.preview or args.output.with_name(f"{args.output.stem}_preview.png")
    policy = build_policy(
        load_policy_factory(args.policy),
        arguments=parse_policy_args(args.policy_arg),
        inference_timeout_s=args.inference_timeout_s,
    )
    task = ShakeBenchTaskEnv(states[args.state_id], gamma=args.gamma, horizon=args.horizon_steps)
    # The declared observation contract gives the frame size without building the scene twice.
    height, width = task.observation_contract()["observation.images.main"]["shape"][:2]
    writer = FFmpegVideoWriter(args.output, width=width * 2, height=height, fps=args.fps)
    try:
        result = record_episode(
            task,
            policy,
            writer,
            action_horizon=args.action_horizon,
            hold_frames=args.fps // 2,
            preview_marks=(0, args.horizon_steps // 2, args.horizon_steps - 1),
        )
    finally:
        writer.close()
        task.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    video = outcome_path(args.output, state_id=result["state_id"], success=result["success"])
    args.output.replace(video)
    summary = {key: value for key, value in result.items() if key != "previews"}
    summary.update(video=str(video), preview=write_preview(preview_path, result["previews"]))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
