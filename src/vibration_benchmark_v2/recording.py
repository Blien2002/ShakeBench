"""NewtonGL video recorder with task and vibration telemetry."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from isaaclab_newton.video_recording import NewtonGlPerspectiveVideoCfg

from .benchmark_rendering import BenchmarkNewtonGlPerspectiveVideo


class BenchmarkRecorder:
    CAMERA_PRESETS = {
        "main": {
            "eye": (1.95, 1.55, 1.08),
            "lookat": (-0.02, 0.0, -0.10),
            "horiz_fov_deg": 60.0,
        },
        "stewart_side": {
            "eye": (0.18, 2.30, 0.31),
            "lookat": (0.0, 0.0, -0.27),
            "horiz_fov_deg": 58.0,
        },
    }

    def __init__(self, output: Path, fps: int = 30, camera_preset: str = "main", overlays: bool = True):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        if camera_preset not in self.CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset: {camera_preset}")
        camera = self.CAMERA_PRESETS[camera_preset]
        self.capture = BenchmarkNewtonGlPerspectiveVideo(
            NewtonGlPerspectiveVideoCfg(
                window_width=1280,
                window_height=720,
                eye=camera["eye"],
                lookat=camera["lookat"],
                horiz_fov_deg=camera["horiz_fov_deg"],
            )
        )
        self.writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=2)
        self.frames = 0
        self.overlays = overlays

    def add_frame(self, task, controller, obs) -> None:
        frame = np.asarray(self.capture.render_rgb_array(), dtype=np.uint8)
        if not self.overlays:
            self.writer.append_data(frame)
            self.frames += 1
            return
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=14)
        draw.rounded_rectangle((20, 18, 760, 148), radius=10, fill=(4, 9, 17, 205), outline=(72, 205, 255, 220), width=1)
        delta_z_mm = float(obs["mount_delta_z"][0, 0].item()) * 1000.0
        if task.cfg.vibration.mode == "spectral":
            motion_values = []
            for axis in task.cfg.vibration.active_axes:
                index = ("tx", "ty", "tz", "rx", "ry", "rz").index(axis)
                value = float(obs["vibration_q"][0, index].item())
                motion_values.append(
                    f"{axis}={1000.0 * value:+.2f}mm" if axis.startswith("t") else f"{axis}={math.degrees(value):+.2f}deg"
                )
            motion_label = " ".join(motion_values)
        else:
            motion_label = f"Delta-z={delta_z_mm:+.2f}mm"
        left_n = float(obs["left_finger_contact_n"][0, 0].item())
        right_n = float(obs["right_finger_contact_n"][0, 0].item())
        acceleration_rms_g = self._nominal_acceleration_rms_g(task)
        penetration_mm = float(obs["penetration_mm"][0, 0].item())
        penetration_color = (255, 92, 92, 255) if penetration_mm > 0.5 else (165, 235, 205, 255)
        if task.cfg.vibration.mode == "spectral":
            active = task.cfg.vibration.active_axes
            dof_label = f"{len(active)}-DOF {','.join(active)}"
        elif task.cfg.vibration.mode == "sine":
            dof_label = f"1-DOF {task.cfg.vibration.sine_axis}"
        else:
            dof_label = "0-DOF"
        lines = (
            ("Vibration Benchmark v2 | Isaac Lab + Newton", font, (230, 248, 255, 255)),
            (f"{controller.name} | t={task.time_s:5.2f}s | {motion_label} | contact={left_n:.1f}/{right_n:.1f}N | hold={bool(obs['grasped'][0])}", small, (255, 220, 100, 255)),
            (f"{task.cfg.vibration.mode} | seed={task.cfg.vibration.seed} | {dof_label} | a_rms={acceleration_rms_g:.2f} g", small, (190, 255, 195, 255)),
            (
                f"penetration={penetration_mm:.3f} mm | pair={task._current_penetration.pair} | max={task.metrics.max_penetration_mm:.3f} mm",
                small,
                penetration_color,
            ),
        )
        y = 28
        for label, label_font, color in lines:
            draw.text((34, y), label, font=label_font, fill=color)
            y += 29

        self._plot(draw, task)
        wrist_frame = Image.fromarray(task.wrist_camera_rgb(obs)).resize(
            (320, 200), resample=Image.Resampling.BILINEAR
        )
        image.paste(wrist_frame, (936, 490))
        draw.rounded_rectangle(
            (924, 454, 1268, 706),
            radius=14,
            fill=(4, 9, 17, 35),
            outline=(72, 205, 255, 230),
            width=1,
        )
        draw.rectangle((936, 490, 1256, 690), outline=(110, 190, 225, 235), width=1)
        draw.text((942, 466), "PHYSICAL WRIST CAMERA | RGB", font=small, fill=(220, 245, 255, 255))
        if task.metrics.success:
            draw.rounded_rectangle((1080, 20, 1258, 58), radius=10, fill=(0, 92, 42, 220), outline=(105, 235, 145, 235), width=1)
            draw.text((1101, 29), "TASK SUCCESS", font=small, fill=(238, 255, 242, 255))
        self.writer.append_data(np.asarray(image))
        self.frames += 1

    @staticmethod
    def _nominal_acceleration_rms_g(task) -> float:
        if task.cfg.vibration.mode == "off":
            return 0.0
        if task.cfg.vibration.mode == "sine":
            if task.cfg.vibration.sine_axis not in ("tx", "ty", "tz"):
                return 0.0
            omega = 2.0 * math.pi * task.cfg.vibration.sine_frequency_hz
            return omega * omega * task.cfg.vibration.sine_amplitude / math.sqrt(2.0) / 9.80665
        axis_rms = []
        for axis in ("tx", "ty", "tz"):
            if axis not in task.cfg.vibration.active_axes:
                continue
            squared = sum(
                ((2.0 * math.pi * band.center_hz) ** 2 * band.rms) ** 2
                for band in task.cfg.vibration.bands[axis]
            )
            axis_rms.append(task.cfg.vibration.spectral_scale * math.sqrt(squared))
        return math.sqrt(sum(value * value for value in axis_rms)) / 9.80665

    def _plot(self, draw: ImageDraw.ImageDraw, task) -> None:
        times_t, values_t = task.vibration_history()
        times = times_t.detach().cpu().numpy()
        values = values_t.detach().cpu().numpy()
        box = (20, 565, 475, 700)
        draw.rounded_rectangle(box, radius=10, fill=(4, 9, 17, 210), outline=(72, 205, 255, 220), width=1)
        tiny = ImageFont.load_default(size=14)
        draw.text((34, 576), f"rolling 4 s command | a_rms={self._nominal_acceleration_rms_g(task):.2f} g", font=tiny, fill=(220, 242, 255, 255))
        plot = (34, 606, 461, 680)
        x0, y0, x1, y1 = plot
        draw.line((x0, (y0 + y1) // 2, x1, (y0 + y1) // 2), fill=(100, 125, 145, 150), width=1)
        colors = (
            (70, 220, 255, 255),
            (90, 235, 135, 255),
            (255, 205, 70, 255),
            (255, 110, 185, 255),
            (255, 145, 65, 255),
            (175, 125, 255, 255),
        )
        default_limits = (0.002, 0.001, 0.005, 0.012, 0.006, 0.004)
        axes = []
        for axis_index, (axis_name, default_limit, color) in enumerate(
            zip(("tx", "ty", "tz", "rx", "ry", "rz"), default_limits, colors)
        ):
            if task.cfg.vibration.mode == "spectral" and axis_name not in task.cfg.vibration.active_axes:
                continue
            if task.cfg.vibration.mode == "spectral":
                commanded_rms = task.cfg.vibration.spectral_scale * math.sqrt(
                    sum(band.rms * band.rms for band in task.cfg.vibration.bands[axis_name])
                )
                limit = max(3.0 * commanded_rms, 1.0e-9)
            else:
                limit = default_limit
            axes.append((axis_index, axis_name, limit, color))
        if len(times) >= 2:
            span = max(float(times[-1] - times[0]), task.cfg.dt)
            for label_index, (axis, name, limit, color) in enumerate(axes):
                normalized = np.clip(values[:, axis] / limit, -1.0, 1.0)
                points = [
                    (
                        int(x0 + (float(t) - float(times[0])) / span * (x1 - x0)),
                        int((y0 + y1) / 2 - float(v) * (y1 - y0) * 0.45),
                    )
                    for t, v in zip(times, normalized)
                ]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
                draw.text((x0 + 68 * label_index, y0 + 2), name, font=tiny, fill=color)
        draw.rectangle(plot, outline=(110, 165, 195, 180), width=1)

    def close(self) -> None:
        self.writer.close()
