"""Record a reproducible qualitative video of the ShakeBench oracle.

The controller consumes only the selected public state-observation tier.
Camera pixels are rendered separately after each policy step and never enter
the action, trace, success metric, or scientific evidence path.

Example (headless NVIDIA EGL):

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m \
        robosuite.demos.demo_shakebench_oracle_video \
        --tier V0 --gamma 0.15 --state-id shakebench-dev-v0-000 \
        --output out/demo/shakebench_oracle_v0_gamma015.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2
import mujoco
import numpy as np

from robosuite.scripts.shakebench_run_oracle import load_dev_states, run_episode
from robosuite.utils.shakebench_geometry import geometry_scene_path, load_geometry_profile
from robosuite.utils.shakebench_oracle import OracleControllerProfile, ShakeBenchOracleController
from robosuite.utils.shakebench_scene import load_scene_visual_config

DEMO_SCHEMA_ID = "shakebench.oracle_demo_video"
DEFAULT_STATE_ID = "shakebench-dev-v0-000"
DEFAULT_OUTPUT = Path("out/demo/shakebench_oracle_v0_gamma015.mp4")
PRESENTATION_CAMERA = "presentation"


class FFmpegVideoWriter:
    """Stream RGB frames to the system FFmpeg without imageio plugins."""

    def __init__(self, output: Path, *, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def append_data(self, frame: np.ndarray) -> None:
        image = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if image.shape != expected:
            raise ValueError(f"video frame must have shape {expected}, got {image.shape}")
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        returncode = self.process.wait()
        detail = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        if returncode != 0:
            raise RuntimeError(f"FFmpeg exited with {returncode}: {detail.strip()}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _git_dirty() -> bool | None:
    completed = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return bool(completed.stdout) if completed.returncode == 0 else None


def _phase_label(controller: ShakeBenchOracleController) -> str:
    trace = controller.last_trace or {}
    phase = trace.get("phase", {})
    return str(phase.get("phase", controller.executive.phase.value))


def annotate_frame(
    frame: np.ndarray,
    *,
    tier: str,
    gamma: float,
    state_id: str,
    step: int,
    policy_rate_hz: float,
    phase: str,
    success: bool,
    final: bool = False,
) -> np.ndarray:
    """Return a presentation frame with deterministic status overlays."""

    image = np.ascontiguousarray(frame).copy()
    height, width = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 91), (10, 14, 22), thickness=-1)
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0.0, image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = (
        "ShakeBench | VibrationPickPlaceCan | Panda oracle",
        f"tier={tier}   Gamma={gamma:.2f}   state={state_id}",
        f"t={step / policy_rate_hz:05.2f}s   step={step:04d}   phase={phase}",
    )
    for row, text in enumerate(lines):
        cv2.putText(image, text, (16, 25 + row * 27), font, 0.57, (245, 248, 252), 1, cv2.LINE_AA)
    if final:
        label = "SUCCESS" if success else "NOT COMPLETED"
        color = (58, 210, 92) if success else (64, 96, 235)
        scale = max(0.8, min(width, height) / 480.0)
        size, _ = cv2.getTextSize(label, font, scale, 2)
        origin = ((width - size[0]) // 2, height - 32)
        cv2.putText(image, label, origin, font, scale, color, 2, cv2.LINE_AA)
    return image


class VideoObserver:
    """Render frames after policy steps without affecting policy inputs."""

    def __init__(
        self,
        output: Path,
        *,
        camera: str,
        width: int,
        height: int,
        fps: int,
        tier: str,
        gamma: float,
        state_id: str,
        policy_rate_hz: float,
        scene_config=None,
    ) -> None:
        self.output = output
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.tier = tier
        self.gamma = gamma
        self.state_id = state_id
        self.policy_rate_hz = policy_rate_hz
        scene_config = scene_config or load_scene_visual_config()
        camera_names = {str(camera["name"]) for camera in scene_config.section("cameras").values()}
        if camera == PRESENTATION_CAMERA:
            camera = str(scene_config.section("cameras")["overview"]["name"])
        if camera not in camera_names:
            raise ValueError(f"camera must be one of {sorted(camera_names)} or {PRESENTATION_CAMERA!r}")
        self.camera = camera
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.writer = FFmpegVideoWriter(output, width=width, height=height, fps=fps)
        self.renderer: mujoco.Renderer | None = None
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[0] = 0
        self.scene_option.geomgroup[1] = 1
        self.last_frame: np.ndarray | None = None
        self.last_step = 0
        self.last_phase = "reset"

    def __call__(
        self,
        env: Any,
        step: int,
        observation: Mapping[str, Any],
        controller: ShakeBenchOracleController,
    ) -> None:
        del observation
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                env.sim.model._model,
                height=self.height,
                width=self.width,
            )
        self.renderer.update_scene(
            env.sim.data._data,
            camera=self.camera,
            scene_option=self.scene_option,
        )
        raw = self.renderer.render().copy()
        metrics = env.get_metrics()
        self.last_step = step
        self.last_phase = _phase_label(controller)
        self.last_frame = annotate_frame(
            raw,
            tier=self.tier,
            gamma=self.gamma,
            state_id=self.state_id,
            step=step,
            policy_rate_hz=self.policy_rate_hz,
            phase=self.last_phase,
            success=bool(metrics["success"]["passed"]),
        )
        self.writer.append_data(self.last_frame)

    def close(self, *, success: bool, hold_seconds: float = 1.5) -> None:
        if self.last_frame is not None:
            final = self.last_frame.copy()
            label = "SUCCESS" if success else "NOT COMPLETED"
            color = (58, 210, 92) if success else (64, 96, 235)
            font = cv2.FONT_HERSHEY_SIMPLEX
            size, _ = cv2.getTextSize(label, font, 1.0, 2)
            cv2.putText(
                final,
                label,
                ((final.shape[1] - size[0]) // 2, final.shape[0] - 32),
                font,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )
            for _ in range(max(1, round(self.fps * hold_seconds))):
                self.writer.append_data(final)
        try:
            self.writer.close()
        finally:
            if self.renderer is not None:
                self.renderer.close()


def _select_state(states: list[dict[str, Any]], state_id: str) -> dict[str, Any]:
    matches = [state for state in states if state["state_id"] == state_id]
    if len(matches) != 1:
        available = ", ".join(state["state_id"] for state in states)
        raise ValueError(f"unknown state-id {state_id!r}; available: {available}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    scene_config = load_scene_visual_config()
    render_config = scene_config.section("render")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("V0", "V1", "V2", "V3"), default="V0")
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--state-id", default=DEFAULT_STATE_ID)
    parser.add_argument("--states", type=Path, default=Path("robosuite/models/assets/shakebench_states_dev.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--camera",
        default=PRESENTATION_CAMERA,
        help="MuJoCo camera name, or 'presentation' for the fixed ShakeBench overview",
    )
    parser.add_argument("--width", type=int, default=int(render_config["default_width"]))
    parser.add_argument("--height", type=int, default=int(render_config["default_height"]))
    parser.add_argument("--fps", type=int, default=int(render_config["default_fps"]))
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--geometry-profile", choices=("canonical", "direct_mount_v1"), default="direct_mount_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gamma < 0.0 or not np.isfinite(args.gamma):
        raise ValueError("--gamma must be finite and non-negative")
    if min(args.width, args.height, args.fps, args.horizon_steps) <= 0:
        raise ValueError("video dimensions, fps, and horizon must be positive")
    state = _select_state(load_dev_states(args.states), args.state_id)
    scene_config = load_scene_visual_config(geometry_scene_path(args.geometry_profile))
    profile = OracleControllerProfile()
    observer = VideoObserver(
        args.output,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        tier=args.tier,
        gamma=args.gamma,
        state_id=args.state_id,
        policy_rate_hz=profile.policy_rate_hz,
        scene_config=scene_config,
    )
    try:
        episode = run_episode(
            state,
            tier=args.tier,
            gamma_commanded=args.gamma,
            profile=profile,
            horizon_steps=args.horizon_steps,
            step_observer=observer,
            geometry_profile=args.geometry_profile,
            # The final direct-mount visual evidence must use the same frozen
            # science authority as the raw scoreable episodes.  A pending or
            # mutated authority fails before rendering rather than yielding a
            # visually plausible but unauthenticated recording.
            allow_unverified_geometry=False,
        )
    except Exception:
        observer.close(success=False, hold_seconds=0.0)
        raise
    observer.close(success=bool(episode["success"]))
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata = {
        "schema_id": DEMO_SCHEMA_ID,
        "schema_version": 1,
        "qualitative_only": False,
        "scoreable_evidence": bool(episode["scoreable"]),
        "geometry_profile": load_geometry_profile(args.geometry_profile),
        "source": {
            "base_commit": _git_head(),
            "worktree_dirty": _git_dirty(),
            "demo_script_sha256": _sha256(Path(__file__)),
            "oracle_runner_sha256": _sha256(Path(run_episode.__code__.co_filename)),
        },
        "video": {
            "path": str(args.output),
            "sha256": _sha256(args.output),
            "bytes": args.output.stat().st_size,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "camera": observer.camera,
            "camera_request": args.camera,
        },
        "episode": {
            "state_id": args.state_id,
            "tier": args.tier,
            "gamma_commanded": args.gamma,
            "success": episode["success"],
            "failure_reason": episode["failure_reason"],
            "termination_category": episode["termination_category"],
            "steps": len(episode["trace"]),
            "trace_sha256": episode["trace_sha256"],
            "controller_profile_sha256": episode["controller_profile"]["profile_sha256"],
            "physics_profile_sha256": episode["physics_profile"]["profile_sha256"],
            "scene_visual": {
                "scene_id": scene_config.scene_id,
                "config_sha256": scene_config.config_sha256,
                "geometry_variant": scene_config.geometry_variant,
                "physics_effect": scene_config.physics_effect,
            },
            "geometry_authority": episode["geometry_authority"],
            "scoreable": episode["scoreable"],
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"video": str(args.output), "metadata": str(metadata_path), **metadata["episode"]}, sort_keys=True)
    )
    return 0 if episode["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEMO_SCHEMA_ID",
    "FFmpegVideoWriter",
    "PRESENTATION_CAMERA",
    "VideoObserver",
    "annotate_frame",
    "build_parser",
    "main",
]
