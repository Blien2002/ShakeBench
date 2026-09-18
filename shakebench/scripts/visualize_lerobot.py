"""Render a local LeRobot v2.1 dataset: camera contact sheets, signal plots and an MP4 per episode.

Run: python -m shakebench.scripts.visualize_lerobot --dataset out/lerobot_gamma_zero_task_close
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CAMERAS = (("observation.images.main", "main"), ("observation.images.wrist", "wrist"))
ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
STATE_NAMES = ("eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "finger_l", "finger_r")
IMU_KEY = "observation.table_imu_window"
IMU_TIME_KEY = "observation.table_imu_timestamps_s"
IMU_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")


def stack_nested(value, dtype):
    """Stack Arrow nested lists, which pandas exposes as object arrays of arrays."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.stack([stack_nested(item, dtype) for item in value])
    return np.asarray(value, dtype=dtype)


def read_tasks(dataset):
    path = dataset / "meta/tasks.jsonl"
    if not path.is_file():
        return {}
    return {
        int(record["task_index"]): record["task"] for record in map(json.loads, path.read_text().splitlines()) if record
    }


def load_episode(path, tasks):
    """Read one episode parquet; PNG bytes stay undecoded until a frame is picked."""
    frame = pd.read_parquet(path)
    if len(frame) == 0:
        raise ValueError(f"{path} has no frames")
    time_s = stack_nested(frame["timestamp"].to_numpy(), np.float64)
    task_index = int(frame["task_index"].iloc[0]) if "task_index" in frame else None
    episode = {
        "path": path,
        "images": {key: [row["bytes"] for row in frame[key]] for key, _ in CAMERAS if key in frame},
        "time_s": time_s,
        "done": (
            stack_nested(frame["next.done"].to_numpy(), bool) if "next.done" in frame else np.zeros(len(frame), bool)
        ),
        "success": (
            stack_nested(frame["next.success"].to_numpy(), bool)
            if "next.success" in frame
            else np.zeros(len(frame), bool)
        ),
        "task": tasks.get(task_index, f"task {task_index}"),
        "signals": [],
    }
    if "action" in frame:
        episode["signals"].append(
            {
                "title": "action (normalized OSC_POSE delta, robot base frame)",
                "values": stack_nested(frame["action"].to_numpy(), np.float32),
                "names": ACTION_NAMES,
                "time_s": time_s,
            }
        )
    if "observation.state" in frame:
        episode["signals"].append(
            {
                "title": "observation.state",
                "values": stack_nested(frame["observation.state"].to_numpy(), np.float32),
                "names": STATE_NAMES,
                "time_s": time_s,
            }
        )
    if IMU_KEY in frame:
        imu = stack_nested(frame[IMU_KEY].to_numpy(), np.float32)
        if imu.ndim != 3 or imu.shape[2] != len(IMU_NAMES):
            raise ValueError(f"{IMU_KEY} must be [frames, samples, {len(IMU_NAMES)}], got {imu.shape}")
        samples = imu.reshape(-1, len(IMU_NAMES))
        if IMU_TIME_KEY in frame:
            imu_time_s = stack_nested(frame[IMU_TIME_KEY].to_numpy(), np.float64).reshape(-1)
        else:
            imu_time_s = np.linspace(time_s[0], time_s[-1], len(samples))
        episode["signals"].append(
            {
                "title": f"table IMU acceleration (m/s^2, {len(samples)} samples of the causal window stream)",
                "values": samples[:, :3],
                "names": IMU_NAMES[:3],
                "time_s": imu_time_s,
            }
        )
        episode["signals"].append(
            {
                "title": "table IMU angular velocity (rad/s)",
                "values": samples[:, 3:],
                "names": IMU_NAMES[3:],
                "time_s": imu_time_s,
            }
        )
    return episode


def plot_episode(episode, dataset_name, index, output_path, columns):
    cameras = [(key, label) for key, label in CAMERAS if key in episode["images"]]
    signals = episode["signals"]
    time_s = episode["time_s"]
    fig = plt.figure(figsize=(14.0, 3.6 + 1.9 * (len(cameras) + len(signals))))
    grid = fig.add_gridspec(
        len(cameras) + len(signals),
        columns,
        height_ratios=[1.5] * len(cameras) + [1.8] * len(signals),
        hspace=0.5,
        wspace=0.06,
    )
    picks = np.unique(np.linspace(0, len(time_s) - 1, min(columns, len(time_s))).round().astype(int))
    for row, (key, label) in enumerate(cameras):
        cells = [fig.add_subplot(grid[row, column]) for column in range(columns)]
        for cell, pick in zip(cells, picks):
            with Image.open(io.BytesIO(episode["images"][key][pick])) as image:
                cell.imshow(image)
            cell.set_axis_off()
            cell.set_title(f"{label} t={time_s[pick]:.2f}s", fontsize=6)
        for cell in cells[len(picks) :]:
            cell.set_axis_off()
    done_time = float(time_s[int(np.argmax(episode["done"]))]) if episode["done"].any() else None
    for offset, signal in enumerate(signals):
        axis = fig.add_subplot(grid[len(cameras) + offset, :])
        for column, name in enumerate(signal["names"]):
            axis.plot(signal["time_s"], signal["values"][:, column], linewidth=0.8, label=name)
        axis.set_title(signal["title"], fontsize=9)
        axis.set_xlabel("episode time (s)", fontsize=8)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, ncol=len(signal["names"]))
        axis.tick_params(labelsize=7)
        axis.set_xlim(float(signal["time_s"].min()), float(signal["time_s"].max()))
        if done_time is not None:
            axis.axvline(done_time, color="crimson", linewidth=0.8, linestyle="--")
    fig.suptitle(
        f"{dataset_name} | episode {index} | {episode['task']} | {len(time_s)} frames, "
        f"{time_s[-1]:.2f} s | done={bool(episode['done'].any())} success={bool(episode['success'].any())}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def write_video(episode, episode_index, output_path, fps):
    """Side-by-side MP4 with the episode time burned in; codec is the OpenCV mp4v default."""
    writer = None
    keys = [key for key, _ in CAMERAS if key in episode["images"]]
    if not keys:
        return
    caption = f"episode {episode_index} " + " | ".join(label for key, label in CAMERAS if key in episode["images"])
    try:
        for step, time_s in enumerate(episode["time_s"]):
            frames = [
                cv2.imdecode(np.frombuffer(episode["images"][key][step], np.uint8), cv2.IMREAD_COLOR) for key in keys
            ]
            separator = np.zeros((frames[0].shape[0], 4, 3), np.uint8)
            frame = np.concatenate([part for camera in frames for part in (camera, separator)], axis=1)[:, :-4]
            cv2.putText(
                frame, f"{caption} t={time_s:.2f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open {output_path} for writing; no usable mp4 encoder")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("out/lerobot_gamma_zero_task_close"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset>_preview beside the dataset")
    parser.add_argument("--episode", type=int, action="append", help="Episode index to render; repeatable, default all")
    parser.add_argument("--columns", type=int, default=8, help="Camera frames sampled into the contact sheet")
    parser.add_argument("--no-video", action="store_true", help="Write figures only")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.columns <= 0:
        raise ValueError("columns must be positive")
    paths = sorted(args.dataset.glob("data/chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data/chunk-*/episode_*.parquet under {args.dataset}")
    if args.episode:
        unknown = sorted(set(args.episode) - set(range(len(paths))))
        if unknown:
            raise ValueError(f"episode index out of range 0..{len(paths) - 1}: {unknown}")
        paths = [paths[index] for index in args.episode]
    info_path = args.dataset / "meta/info.json"
    fps = json.loads(info_path.read_text())["fps"] if info_path.is_file() else 20.0
    output_dir = args.output_dir or args.dataset.parent / f"{args.dataset.name}_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(args.dataset)
    for path in paths:
        episode_index = int(path.stem.split("_")[-1])
        episode = load_episode(path, tasks)
        figure_path = output_dir / f"{path.stem}.png"
        plot_episode(episode, args.dataset.name, episode_index, figure_path, args.columns)
        print(f"{path.stem}: {figure_path}", flush=True)
        if not args.no_video:
            video_path = output_dir / f"{path.stem}.mp4"
            write_video(episode, episode_index, video_path, fps)
            print(f"{path.stem}: {video_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
