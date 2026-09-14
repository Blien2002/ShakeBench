"""Unit check for the local LeRobot preview renderer; synthetic dataset, no renderer needed."""

import io
import json

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from robosuite.scripts.shakebench_visualize_lerobot import main

FRAMES = 3


def png_bytes(value):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (value, value // 2, 255 - value)).save(buffer, format="PNG")
    return buffer.getvalue()


def frame_row(index, with_state):
    row = {
        "observation.images.main": {"bytes": png_bytes(index * 60), "path": None},
        "observation.images.wrist": {"bytes": png_bytes(index * 60 + 20), "path": None},
        "action": [0.1 * index] * 7,
        "observation.table_imu_window": [[0.01 * index + column] * 6 for column in range(10)],
        "observation.table_imu_timestamps_s": [index / 20 + sample * 0.005 - 0.05 for sample in range(10)],
        "observation.table_imu_dt_s": 0.005,
        "next.reward": 0.0,
        "next.done": index == FRAMES - 1,
        "next.success": index == FRAMES - 1,
        "timestamp": index / 20,
        "frame_index": index,
        "episode_index": 0,
        "index": index,
        "task_index": 0,
    }
    if with_state:
        row["observation.state"] = list(np.arange(8, dtype=np.float32))
    return row


def write_dataset(root):
    """Two episodes: one with observation.state, one without, as older collections were written."""
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps({"fps": 20, "total_episodes": 2}))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "synthetic pick"}) + "\n")
    for episode in range(2):
        rows = [frame_row(index, with_state=episode == 0) for index in range(FRAMES)]
        pd.DataFrame(rows).to_parquet(root / f"data/chunk-000/episode_{episode:06d}.parquet")


def test_preview_renders_figure_and_video(tmp_path):
    dataset = tmp_path / "dataset"
    write_dataset(dataset)
    output = tmp_path / "preview"
    assert main(["--dataset", str(dataset), "--output-dir", str(output), "--columns", "3"]) == 0
    for episode in range(2):
        figure = output / f"episode_{episode:06d}.png"
        video = output / f"episode_{episode:06d}.mp4"
        assert figure.stat().st_size > 1000
        with Image.open(figure) as image:
            pixels = np.asarray(image.convert("RGB"))
        # Contact sheet frames and signal plots must actually have drawn ink.
        assert float((pixels.min(axis=2) < 240).mean()) > 0.05
        capture = cv2.VideoCapture(str(video))
        assert capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) == FRAMES
        ok, decoded = capture.read()
        assert ok and decoded.shape[0] == 16 and decoded.shape[1] == 36
        capture.release()


def test_episode_selection_and_range_check(tmp_path):
    dataset = tmp_path / "dataset"
    write_dataset(dataset)
    output = tmp_path / "preview"
    assert main(["--dataset", str(dataset), "--output-dir", str(output), "--episode", "1", "--no-video"]) == 0
    assert sorted(path.name for path in output.iterdir()) == ["episode_000001.png"]
    with pytest.raises(ValueError, match="out of range"):
        main(["--dataset", str(dataset), "--output-dir", str(output), "--episode", "2"])


def test_empty_dataset_is_rejected(tmp_path):
    (tmp_path / "dataset").mkdir()
    with pytest.raises(FileNotFoundError, match="no data/chunk"):
        main(["--dataset", str(tmp_path / "dataset"), "--output-dir", str(tmp_path / "preview")])
