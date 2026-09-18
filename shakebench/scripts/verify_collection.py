"""Verify a collected ShakeBench LeRobot dataset before it is used for training.

Collectors write what the physics did; this checks that what landed on disk is the dataset the
training registration assumes.  It reads only the artifact, fails closed, and reports every check
individually so a partial failure still tells you which episode or field is wrong.

Checks: collection manifest completeness and state-asset authority, LeRobot v2.1 totals and
indexing, absence of derived training caches, language/task consistency, train-vs-evaluation
state disjointness by execution fingerprint, per-episode schema/actions/outcome/IMU/timing, and
sampled image sanity (blank, wrist==main, writer stalls, world mixing).

    python -m shakebench.scripts.verify_collection --dataset out/lerobot_gpu_can_knee_100

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from shakebench import models
from shakebench.scripts.export_sft_subset import COLLECTION_MANIFEST_FILENAME, is_successful_episode
from shakebench.scripts.run_oracle import load_state_asset
from shakebench.utils.artifacts import write_json_atomic
from shakebench.utils.rollout import STATE_NAMES, task_description
from shakebench.utils.train_states import split_overlap

DEFAULT_ASSETS = (
    Path(models.assets_root, "shakebench_states_dev.json"),
    Path(models.assets_root, "shakebench_states_official.json"),
    Path(models.assets_root, "shakebench_states_knee.json"),
)
# GR00T-style loaders derive these from whatever episodes they were pointed at; shipping them can
# poison normalization statistics for a different selection (audit 2026-09-14, item 2).
DERIVED_CACHE_NAMES = ("stats_gr00t.json", "steps_data_index.pkl")
ACTION_NAMES = ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper")
LANGUAGE_TOLERANCE = 0.0
BLANK_IMAGE_STD = 1.0


class VerificationError(RuntimeError):
    """Raised by a check that cannot complete; recorded as a failure, not a crash."""


def _check(report, name, function, *args, **kwargs):
    """Run one check; a crash inside a check is a failed check, never a lost report."""
    try:
        passed, detail = function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must outlive one broken check
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    report["checks"][name] = {"passed": bool(passed), "detail": detail}
    return passed


def _manifest_contract(dataset):
    manifest = json.loads((dataset / "meta" / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise VerificationError("collection manifest is not complete")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise VerificationError("collection manifest has no episodes")
    return manifest, episodes


def check_manifest(dataset):
    manifest, episodes = _manifest_contract(dataset)
    missing = [
        key
        for key in ("cameras", "action_space", "physics_profile", "physics_backend", "state_authority", "tasks")
        if key not in manifest
    ]
    if missing:
        raise VerificationError(f"manifest is missing {missing}")
    return True, {
        "episodes": len(episodes),
        "physics_backend": manifest["physics_backend"],
        "physics_profile": manifest["physics_profile"],
        "gamma": manifest.get("gamma"),
        "cameras": manifest["cameras"],
        "scoreable": manifest.get("scoreable"),
    }


def check_state_authority(dataset, assets):
    """The states that produced these episodes must still be the committed ones."""
    path, asset = _source_asset(dataset, assets)
    manifest, _ = _manifest_contract(dataset)
    return True, {
        "asset": str(path),
        "sha256": manifest["state_authority"].get("asset_file_sha256"),
        "states": len(manifest.get("requested_states", ())),
        "split": asset.get("split"),
        "scoreable": asset.get("scoreable"),
    }


def _source_asset(dataset, assets):
    """The committed asset these episodes were collected from, by ids and file hash."""
    manifest, _ = _manifest_contract(dataset)
    requested = {str(state_id) for state_id in manifest.get("requested_states", ())}
    declared = manifest["state_authority"].get("asset_file_sha256")
    mismatches = []
    for candidate in assets:
        path = Path(candidate)
        asset = json.loads(path.read_text(encoding="utf-8"))
        if not requested <= {str(state.get("state_id")) for state in asset.get("states", ())}:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() == declared:
            return path, asset
        mismatches.append(str(path))
    raise VerificationError(
        f"no committed asset holds all {len(requested)} requested states with hash {declared}; "
        f"id matches with a different hash: {mismatches}"
    )


def check_source_split(dataset, assets):
    """Training consumes a state pool; if that pool is certified for scoring, it is now burned."""
    path, asset = _source_asset(dataset, assets)
    split, scoreable = asset.get("split"), asset.get("scoreable")
    if scoreable is True:
        raise VerificationError(
            f"training states come from the scoreable {split!r} split of {path.name}; after training that split "
            "can no longer produce a held-out score. Collect training data from a generated train pool "
            "(shakebench.scripts.generate_train_states) or accept that this split is consumed"
        )
    return True, {"source": str(path), "split": split, "scoreable": scoreable}


def check_lerobot_totals(dataset, episodes, tables):
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    lengths = [int(row["length"]) for row in _read_jsonl(dataset / "meta" / "episodes.jsonl")]
    actual_frames = sum(table.num_rows for table in tables.values())
    problems = {
        "total_episodes": (info.get("total_episodes"), len(episodes), len(tables)),
        "total_frames": (info.get("total_frames"), sum(lengths), actual_frames),
    }
    failed = {key: value for key, value in problems.items() if len(set(value)) != 1}
    if failed:
        raise VerificationError(f"totals disagree (info, jsonl, parquet): {failed}")
    if int(info.get("fps", 0)) != 20 or info.get("codebase_version") != "v2.1":
        raise VerificationError(f"unexpected format: v{info.get('codebase_version')} at {info.get('fps')} fps")
    return True, {"episodes": len(episodes), "frames": actual_frames, "fps": info["fps"]}


def check_indexing(dataset, episodes, tables):
    numbers = sorted(int(path.stem.split("_")[-1]) for path in (dataset / "data").rglob("*.parquet"))
    if numbers != list(range(len(numbers))):
        raise VerificationError("episode files are not contiguous from zero")
    manifest_indices = sorted(int(row["episode_index"]) for row in episodes)
    if manifest_indices != numbers:
        raise VerificationError("manifest episode indices do not match the data files")
    running = 0
    for number in numbers:
        table = tables[number]
        frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        episode_column = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        global_index = np.asarray(table["index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(frames, np.arange(len(frames))):
            raise VerificationError(f"episode {number}: frame_index is not 0..T-1")
        if not np.all(episode_column == number):
            raise VerificationError(f"episode {number}: episode_index column disagrees with the file name")
        if not np.array_equal(global_index, np.arange(running, running + len(frames))):
            raise VerificationError(f"episode {number}: global index is not continuous")
        running += len(frames)
    return True, {"episodes": len(numbers), "frames": running}


def check_no_derived_caches(dataset):
    meta = dataset / "meta"
    present = sorted(name for name in DERIVED_CACHE_NAMES if (meta / name).exists())
    stray = sorted(path.name for path in meta.glob("*.pkl")) + sorted(path.name for path in meta.glob("*.pt"))
    if present or stray:
        raise VerificationError(f"derived training caches must not ship with the dataset: {present + stray}")
    return True, {"meta_files": sorted(path.name for path in meta.iterdir())}


def check_language(dataset, episodes, tables):
    """Frame labels, the task table, the manifest and the state's task spec must tell one story."""
    declared = {int(row["task_index"]): row["task"] for row in _read_jsonl(dataset / "meta" / "tasks.jsonl")}
    problems = []
    for episode in episodes:
        number, expected = int(episode["episode_index"]), task_description(episode["state"])["instruction"]
        if episode.get("instruction") != expected:
            problems.append(f"episode {number}: manifest instruction differs from the state's task description")
        indices = {int(value) for value in np.unique(np.asarray(tables[number]["task_index"].to_pylist()))}
        for index in sorted(indices):
            if declared.get(index) != expected:
                problems.append(
                    f"episode {number}: task_index {index} says {declared.get(index)!r}, expected {expected!r}"
                )
    if problems:
        raise VerificationError("; ".join(sorted(set(problems))[:5]))
    return True, {"tasks": [declared[index] for index in sorted(declared)]}


def check_split_disjoint(episodes, assets, source):
    """A training episode must never repeat a state of another committed split."""
    train = [episode["state"] for episode in episodes]
    overlaps = {}
    for path in assets:
        if Path(path) == Path(source):
            continue  # the source split is consumed by construction; check_source_split owns that verdict
        asset = json.loads(Path(path).read_text(encoding="utf-8"))
        shared = split_overlap(train, asset["states"])
        if shared:
            overlaps[str(path)] = shared[:5]
    if overlaps:
        raise VerificationError(f"training states repeat states of another committed split: {overlaps}")
    return True, {
        "compared_train_states": len(train),
        "compared_assets": [str(p) for p in assets if Path(p) != Path(source)],
    }


def check_episode_schema(tables):
    expected = {
        "observation.state": (8,),
        "observation.table_imu_window": (10, 6),
        "observation.table_imu_timestamps_s": (10,),
        "observation.table_imu_dt_s": (1,),
        "action": (7,),
    }
    for number, table in tables.items():
        missing = sorted(set(expected) - set(table.column_names))
        if missing:
            raise VerificationError(f"episode {number}: missing columns {missing}")
        for name, shape in expected.items():
            array = np.asarray(table[name].to_pylist(), dtype=np.float64)
            if array.ndim == 1 and shape == (1,):
                array = array[:, None]  # LeRobot writes shape-[1] features as scalar columns
            if array.shape[1:] != shape:
                raise VerificationError(f"episode {number}: {name} has shape {array.shape[1:]}, expected {shape}")
            if not np.isfinite(array).all():
                raise VerificationError(f"episode {number}: {name} has non-finite values")
        for name in ("observation.images.main", "observation.images.wrist"):
            first = np.asarray(Image.open(_buffer(table[name][0].as_py())))
            if first.shape != (256, 256, 3) or first.dtype != np.uint8:
                raise VerificationError(f"episode {number}: {name} is {first.shape} {first.dtype}")
    return True, {"columns": sorted(tables[min(tables)].column_names)}


def check_actions(tables):
    for number, table in tables.items():
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if np.abs(actions).max() > 1.0 + 1e-6:
            raise VerificationError(f"episode {number}: action outside [-1, 1] (max |a| {np.abs(actions).max():.4f})")
        gripper = np.unique(actions[:, 6])
        if not np.all(np.isin(gripper, (-1.0, 1.0))):
            raise VerificationError(f"episode {number}: gripper commands must be -1 or +1, saw {gripper[:5]}")
        if float(np.abs(actions[:, :6]).max()) == 0.0:
            raise VerificationError(f"episode {number}: every motion command is zero")
    return True, {"episodes": len(tables)}


def check_outcomes(episodes, tables):
    for episode in episodes:
        number = int(episode["episode_index"])
        table = tables[number]
        done = np.asarray(table["next.done"].to_pylist(), dtype=bool).reshape(-1)
        success = np.asarray(table["next.success"].to_pylist(), dtype=bool).reshape(-1)
        if not done[-1] or not success[-1]:
            raise VerificationError(f"episode {number}: episode does not end in a latched success")
        if done[:-1].any():
            raise VerificationError(f"episode {number}: next.done is set before the final frame")
        if not is_successful_episode(episode):
            raise VerificationError(f"episode {number}: manifest outcome is {episode.get('termination_cause')!r}")
        if int(table.num_rows) != int(episode["steps"]):
            raise VerificationError(f"episode {number}: manifest steps {episode['steps']} != frames {table.num_rows}")
    return True, {"episodes": len(episodes)}


def check_time_and_imu(tables):
    for number, table in tables.items():
        stamps = np.asarray(table["observation.table_imu_timestamps_s"].to_pylist(), dtype=np.float64)
        dt = np.asarray(table["observation.table_imu_dt_s"].to_pylist(), dtype=np.float64).reshape(-1)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
        frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.float64)
        if np.any(np.diff(stamps, axis=1) <= 0):
            raise VerificationError(f"episode {number}: IMU timestamps are not strictly increasing")
        if np.any(dt <= 0):
            raise VerificationError(f"episode {number}: IMU dt is not positive")
        if not np.allclose(timestamps, frames / 20.0, atol=1e-6):
            raise VerificationError(f"episode {number}: timestamp is not frame_index / fps")
    return True, {"episodes": len(tables)}


def check_images(dataset, tables, *, stride):
    """Sampled pixels: blank frames, swapped cameras, writer stalls, world mixing."""
    seen = {}
    samples = 0
    for number, table in sorted(tables.items()):
        main_column = table["observation.images.main"]
        wrist_column = table["observation.images.wrist"]
        previous = None
        for index in range(0, table.num_rows, stride):
            main = np.asarray(Image.open(_buffer(main_column[index].as_py())))
            wrist = np.asarray(Image.open(_buffer(wrist_column[index].as_py())))
            samples += 1
            if float(main.std()) < BLANK_IMAGE_STD:
                raise VerificationError(f"episode {number} frame {index}: main view is blank (std {main.std():.3f})")
            if np.array_equal(main, wrist):
                raise VerificationError(f"episode {number} frame {index}: main view equals the wrist view")
            digest = hashlib.sha1(main.tobytes()).hexdigest()
            if digest == previous:
                raise VerificationError(f"episode {number} frame {index}: main view repeats the previous sample")
            previous = digest
            if digest in seen and seen[digest] != number:
                raise VerificationError(f"episode {number} frame {index}: main view duplicates episode {seen[digest]}")
            seen[digest] = number
    return True, {"sampled_frames": samples, "stride": stride, "unique_main_views": len(seen)}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tables(dataset):
    return {
        int(path.stem.split("_")[-1]): pq.read_table(path) for path in sorted((dataset / "data").rglob("*.parquet"))
    }


def _buffer(value):
    import io

    return io.BytesIO(value["bytes"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--eval-assets",
        type=Path,
        nargs="+",
        default=list(DEFAULT_ASSETS),
        help="Committed state assets to check the authority hash and the train/eval split against",
    )
    parser.add_argument("--image-stride", type=int, default=10, help="Sample one frame every N steps")
    parser.add_argument("--report", type=Path, help="Write the full report as JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = {"schema_id": "shakebench.collection_verification", "schema_version": 1, "dataset": str(args.dataset)}
    dataset = args.dataset
    tables = _tables(dataset)
    manifest, episodes = _manifest_contract(dataset)
    report["checks"] = {}
    _check(report, "manifest", check_manifest, dataset)
    _check(report, "state_authority", check_state_authority, dataset, args.eval_assets)
    _check(report, "source_split", check_source_split, dataset, args.eval_assets)
    _check(report, "lerobot_totals", check_lerobot_totals, dataset, episodes, tables)
    _check(report, "indexing", check_indexing, dataset, episodes, tables)
    _check(report, "no_derived_caches", check_no_derived_caches, dataset)
    _check(report, "language", check_language, dataset, episodes, tables)
    source = report["checks"]["state_authority"]["detail"].get("asset") if report["checks"]["state_authority"] else None
    _check(report, "split_disjoint", check_split_disjoint, episodes, args.eval_assets, source)
    _check(report, "episode_schema", check_episode_schema, tables)
    _check(report, "actions", check_actions, tables)
    _check(report, "outcomes", check_outcomes, episodes, tables)
    _check(report, "time_and_imu", check_time_and_imu, tables)
    _check(report, "images", check_images, dataset, tables, stride=max(1, args.image_stride))
    report["passed"] = all(check["passed"] for check in report["checks"].values())
    report["failed_checks"] = sorted(name for name, check in report["checks"].items() if not check["passed"])
    if args.report is not None:
        write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
