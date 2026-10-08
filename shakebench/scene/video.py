"""MP4 recording of MuJoCo camera views through an ffmpeg subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


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
