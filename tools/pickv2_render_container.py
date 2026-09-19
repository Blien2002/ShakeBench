"""Render the current target crate from the compiled arena alone.

No robot, no scene audit: this compiles ``ShakeBenchArena`` plus the target
container and renders a few camera angles, so the container geometry can be
checked visually in seconds instead of minutes.

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. python3 tools/pickv2_render_container.py \
        --output-dir out/pickv2/container
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shakebench.models.arenas import ShakeBenchArena  # noqa: E402
from shakebench.models.arenas.shakebench_arena import (  # noqa: E402
    TARGET_CONTAINER_BOTTOM_THICKNESS_M,
    TARGET_CONTAINER_CENTER_XY_M,
    TARGET_CONTAINER_INNER_XY_M,
    TARGET_CONTAINER_OUTER_XY_M,
    TARGET_CONTAINER_WALL_HEIGHT_M,
    TARGET_CONTAINER_WALL_THICKNESS_M,
)

VIEWS = {
    "oblique": (0.55, 130.0, -22.0),
    "top": (0.42, 90.0, -78.0),
    "front": (0.45, 180.0, -8.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/pickv2/container"))
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--visual", action="store_true", help="also render the display layer")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arena = ShakeBenchArena(worktable_mount="rigid", visual=args.visual)
    arena.add_target_container()
    model = mujoco.MjModel.from_xml_string(arena.get_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lookat = np.array(
        [
            TARGET_CONTAINER_CENTER_XY_M[0],
            TARGET_CONTAINER_CENTER_XY_M[1],
            arena.table_top_abs[2] + 0.02,
        ],
        dtype=float,
    )
    # Group 0 is the collision box that physics and the success rule use;
    # group 1 is the display layer (robosuite's wooden bin panels).
    options = {}
    for label, groups in (("physics", (0,)), ("display", (1,))):
        option = mujoco.MjvOption()
        for group in range(5):
            option.geomgroup[group] = 1 if group in groups else 0
        options[label] = option
    print(
        f"inner {TARGET_CONTAINER_INNER_XY_M[0] * 1e3:.0f} x {TARGET_CONTAINER_INNER_XY_M[1] * 1e3:.0f} mm, "
        f"wall {TARGET_CONTAINER_WALL_HEIGHT_M * 1e3:.0f} mm thick {TARGET_CONTAINER_WALL_THICKNESS_M * 1e3:.0f} mm, "
        f"bottom {TARGET_CONTAINER_BOTTOM_THICKNESS_M * 1e3:.0f} mm, "
        f"outer {TARGET_CONTAINER_OUTER_XY_M[0] * 1e3:.0f} x {TARGET_CONTAINER_OUTER_XY_M[1] * 1e3:.0f} mm",
        flush=True,
    )
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for layer, option in options.items():
            for name, (distance, azimuth, elevation) in VIEWS.items():
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = lookat
                camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
                renderer.update_scene(data, camera=camera, scene_option=option)
                path = args.output_dir / f"container_{name}_{layer}.png"
                Image.fromarray(renderer.render()).save(path)
                print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
