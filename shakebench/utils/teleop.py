"""Keyboard translation/grasp and SpaceMouse rotation for LeRobot collection."""

import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

_PREVIEW_MARGIN_PX = 32
_PREVIEW_HEADER_PX = 112


class SpaceMouseTeleop:
    """Reuse robosuite's normalized OSC mapping and display the recorded cameras."""

    def __init__(self, env, reader, *, pos_sensitivity=1.0, rot_sensitivity=1.0):
        from robosuite.devices.spacemouse import SpaceMouse

        self.env = env
        self.window = None
        self.closed = False
        self._display_size = None
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
            self.grasp_label = tk.Label(self.window, font=("sans", 18, "bold"), pady=8)
            self.grasp_label.pack(fill="x")
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
        image = Image.fromarray(pixels)
        if self._display_size is None:
            self._display_size = self._fit_preview_size(image.size)
            self.window.geometry(f"{self._display_size[0]}x{self._display_size[1] + _PREVIEW_HEADER_PX}")
        if image.size != self._display_size:
            image = image.resize(self._display_size, Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image, master=self.window)
        self.label.configure(image=self.photo)
        message, color = self._grasp_hint()
        self.grasp_label.configure(text=message, bg=color, fg="white")

    def _grasp_hint(self):
        """Conservative ring-wall grasp guidance from live collision-pad geometry."""
        if not hasattr(self.env, "ring_body_ids"):
            return "Space: grasp | R: retry", "#444444"
        env = self.env
        model, data = env.sim.model, env.sim.data
        pads = env.robots[0].gripper["right"].important_geoms
        ids = [model.geom_name2id(pads[side][0]) for side in ("left_fingerpad", "right_fingerpad")]
        centers = np.array([data.geom_xpos[i] for i in ids])
        context = env.get_policy_task_context()
        name = min(
            env.ring_body_ids, key=lambda n: np.linalg.norm(centers.mean(axis=0) - data.xpos[env.ring_body_ids[n]])
        )
        body = env.ring_body_ids[name]
        rotation = data.xmat[body].reshape(3, 3)
        local = (centers - data.xpos[body]) @ rotation
        # Bounding boxes account for pad thickness and tilted ring/pad frames.
        extents = np.array([np.abs(rotation.T @ data.geom_xmat[i].reshape(3, 3)) @ model.geom_size[i] for i in ids])
        radii = np.linalg.norm(local[:, :2], axis=1)
        inner, outer = np.argsort(radii)
        ring = context["rings"][name]
        clearance = np.linalg.norm(extents[:, :2], axis=1)
        aligned = (
            radii[inner] + clearance[inner] < ring["inner_radius"]
            and radii[outer] - clearance[outer] > ring["outer_radius"]
            and np.dot(local[inner, :2], local[outer, :2]) > 0
        )
        colors = {"large": "蓝环", "medium": "绿环", "small": "黄环"}
        prefix = colors.get(name, name)
        if not aligned:
            return f"{prefix}：调整对齐（两指分别位于孔内、环外）", "#a85b00"
        overlap = context["ring_half_height_m"] + extents[:, 2] - np.abs(local[:, 2])
        if np.min(overlap) >= 0.004:
            return f"{prefix}：可以闭合 · 按空格（几何提示）", "#167a36"
        direction = "继续下降" if local[:, 2].mean() > 0 else "稍微抬高"
        return f"{prefix}：{direction} · 高度差 {local[:, 2].mean() * 1000:+.0f} mm", "#a85b00"

    def _fit_preview_size(self, image_size):
        """Scale the display-only preview to the largest size that fits the screen."""
        image_width, image_height = image_size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        max_width = max(1, screen_width - 2 * _PREVIEW_MARGIN_PX)
        max_height = max(1, screen_height - _PREVIEW_HEADER_PX - 2 * _PREVIEW_MARGIN_PX)
        scale = min(max_width / image_width, max_height / image_height)
        return max(1, round(image_width * scale)), max(1, round(image_height * scale))

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
        success = self.env.get_metrics()["success"]
        if "stage" in success:
            total = len(self.env.get_policy_task_context()["placement_order"])
            self.window.title(f"ShakeBench: RECORDING - Rings {success['stage']}/{total} - R: retry, Esc: quit")
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
