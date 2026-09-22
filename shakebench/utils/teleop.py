"""SpaceMouse input and a paced camera preview for LeRobot collection."""

import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk


class SpaceMouseTeleop:
    """Reuse robosuite's normalized OSC mapping and display the recorded cameras."""

    def __init__(self, env, reader, *, pos_sensitivity=1.0, rot_sensitivity=1.0):
        from robosuite.devices.spacemouse import SpaceMouse

        self.env = env
        self.window = None
        self.closed = False
        self.device = SpaceMouse(
            env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
        try:
            self.window = tk.Tk()
            self.window.title("ShakeBench: main | wrist - Enter: record, Esc: quit")
            tk.Label(
                self.window,
                text="WASD: move | Q/E: down/up | Space: grasp | R: retry | SpaceMouse: rotate | Esc: quit",
            ).pack()
            self.label = tk.Label(self.window)
            self.label.pack()
            ready = tk.BooleanVar(self.window, False)

            def quit_preview(*_):
                self.closed = True
                ready.set(True)

            self.window.protocol("WM_DELETE_WINDOW", quit_preview)
            self.window.bind("<Escape>", quit_preview)
            self.window.bind("<Return>", lambda _: ready.set(True))
            self._show(reader.read(env._get_observations()))
            print(
                "Press Enter to record. WASD: move; Q/E: down/up; Space: grasp; R: retry; "
                "SpaceMouse: rotate; Esc: quit.",
                flush=True,
            )
            self.window.wait_variable(ready)
            self._poll()
            self.device.start_control()
            self.window.title("ShakeBench: RECORDING main | wrist - R: retry, Esc: quit")
        except BaseException:
            self.close()
            raise

    def _poll(self):
        self.window.update()
        if self.closed:
            raise KeyboardInterrupt("teleoperation preview closed")

    def _show(self, frame):
        pixels = np.concatenate([frame["observation.images.main"], frame["observation.images.wrist"]], axis=1)
        self.photo = ImageTk.PhotoImage(Image.fromarray(pixels), master=self.window)
        self.label.configure(image=self.photo)

    def action(self):
        self._poll()
        self.started = time.monotonic()
        controls = self.device.input2action()
        if controls is None:
            return None
        robot = self.env.robots[0]
        arm = robot.arms[0]
        controller = robot.part_controllers[arm]
        keys = self.device._pressed_keys.copy()
        # Construct movement in world coordinates, then use the OSC input frame.
        forward = np.array(self.env.sim.data.get_body_xpos(self.env.worktable_body_name)) - controller.origin_pos
        forward[2] = 0
        forward /= max(np.linalg.norm(forward), 1e-8)
        right = np.array(self.env.sim.data.get_camera_xmat("robot0_eye_in_hand"))[:, 0].copy()
        right[2] = 0
        right /= max(np.linalg.norm(right), 1e-8)
        translation = (
            forward * (("w" in keys) - ("s" in keys))
            + right * (("d" in keys) - ("a" in keys))
            + np.array([0.0, 0.0, ("e" in keys) - ("q" in keys)])
        )
        if controller.input_ref_frame == "base":
            translation = controller.origin_ori.T @ translation
        controls[f"{arm}_delta"][:3] = np.clip(translation * 0.625 * self.device.pos_sensitivity, -1, 1)
        return robot.create_action_vector({arm: controls[f"{arm}_delta"], f"{arm}_gripper": controls[f"{arm}_gripper"]})

    def sync(self, frame):
        self._show(frame)
        self._poll()
        # Never catch up with bursts of stale human commands after a slow frame.
        time.sleep(max(0.0, 1 / self.env.control_freq - (time.monotonic() - self.started)))

    def close(self):
        try:
            if self.window is not None:
                self.window.destroy()
                self.window = None
        finally:
            self.device.close()
