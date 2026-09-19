"""Render the eight pick/place object variants in the current industrial scene.

MUJOCO_GL=egl python -m shakebench.demos.demo_task_variants
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from shakebench.utils.tasks import make_task_env, task_variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/task_variants/pick_place_variants.png"))
    args = parser.parse_args(argv)
    columns = 4
    width, height, label_height = 640, 480, 36
    rows = (len(task_variants()) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, (height + label_height) * rows), (24, 29, 34))
    draw = ImageDraw.Draw(canvas)
    for index, spec in enumerate(task_variants()):
        env = make_task_env(spec, hard_reset=False, seed=17)
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.array(env.arena.table_top_abs) + (0, 0, 0.02)
            camera.distance, camera.azimuth, camera.elevation = 1.08, 140, -48
            option = mujoco.MjvOption()
            option.geomgroup[0] = 0
            option.geomgroup[1] = 1
            with mujoco.Renderer(env.sim.model._model, height=height, width=width) as renderer:
                renderer.update_scene(env.sim.data._data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
            x, y = index % columns * width, index // columns * (height + label_height)
            canvas.paste(frame, (x, y + label_height))
            draw.text(
                (x + 14, y + 10),
                f"METAL / {spec.object_id.upper()}    sliding mu = {spec.table_sliding_mu:.2f}",
                fill=(230, 235, 240),
            )
            print(spec.variant_id, flush=True)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
