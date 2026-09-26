"""View and manually interact with the detailed control panel, without an Oracle.

python -m shakebench.demos.demo_panel
# On this NVIDIA PRIME workstation, prefix the viewer command with:
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
MUJOCO_GL=egl python -m shakebench.demos.demo_panel --output out/panel_scene.png

Viewer keys toggle physical effort: Q/A knob +/-; W/S lever +/-; B press/release
button; X release all; R reset. Mouse perturbation and camera navigation remain
available in the native MuJoCo viewer. Robot control uses the normal 7D env.step
API; these keys only apply manual forces to the panel controls.
"""

import argparse
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from shakebench import PanelOperation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Render one PNG and exit; omit to open the interactive viewer")
    parser.add_argument("--camera", choices=("panel_close", "shakebench_camera_assembly"), default="panel_close")
    parser.add_argument("--gamma", type=float, default=0.0, help="Worktable vibration amplitude multiplier")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--physics-profile", choices=("official", "teleop"), default="teleop")
    args = parser.parse_args(argv)
    env = PanelOperation(
        physics_profile=args.physics_profile,
        vibration={"mode": "multisine_v1", "gamma": args.gamma, "seed": args.seed, "t0_s": 0.0},
        seed=args.seed,
        ignore_done=True,
    )
    try:
        model, data = env.sim.model._model, env.sim.data._data
        # Show authored visuals and hide contact proxies and robot target markers.
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[1] = 1
        if args.output is not None:
            model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
            with mujoco.Renderer(model, height=960, width=1280) as renderer:
                renderer.update_scene(data, camera=args.camera, scene_option=option)
                frame = renderer.render()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(args.output)
            print(args.output.resolve())
            return 0

        from mujoco import viewer as mj_viewer

        lock = threading.Lock()
        effort = np.zeros(3)
        reset_requested = False
        bindings = {
            ord("Q"): (0, 0.04),
            ord("A"): (0, -0.04),
            ord("W"): (1, 0.035),
            ord("S"): (1, -0.035),
            ord("B"): (2, -6.0),
        }

        def on_key(key):
            nonlocal reset_requested
            with lock:
                if key in bindings:
                    i, value = bindings[key]
                    effort[i] = 0.0 if effort[i] == value else value
                elif key in (ord("X"), ord("R")):
                    effort[:] = 0.0
                    reset_requested = key == ord("R")

        print("Q/A: knob +/- | W/S: lever +/- | B: press/release | X: release all | R: reset | Esc: close")
        threads_before = set(threading.enumerate())
        viewer = mj_viewer.launch_passive(model, data, key_callback=on_key)
        viewer_threads = set(threading.enumerate()) - threads_before
        try:
            with viewer:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = model.camera(args.camera).id
                    viewer.opt.geomgroup[:] = option.geomgroup
                while viewer.is_running():
                    start = time.monotonic()
                    with lock:
                        applied = effort.copy()
                        should_reset, reset_requested = reset_requested, False
                    if should_reset:
                        env.reset()
                    data.qfrc_applied[env.panel_qvel_indexes] = applied
                    env.step(np.zeros(env.action_dim))
                    viewer.sync()
                    time.sleep(max(0.0, env.control_timestep - (time.monotonic() - start)))
        finally:
            # MuJoCo closes the daemon render thread asynchronously; let it destroy
            # its GLX context before GLFW's process-exit cleanup runs.
            for thread in viewer_threads:
                thread.join(timeout=5)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
