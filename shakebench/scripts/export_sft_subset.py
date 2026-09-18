"""Export a success-only SFT dataset from a collected ShakeBench LeRobot dataset.

Failed rollouts stay in the source dataset for audit.  The exported copy holds
only episodes selected by the frozen rule (``success_latched``), renumbered from
zero so the official LeRobot reader can enumerate it, and records the source
manifest hash plus the source index of every exported episode, so a trainer can
prove which demonstrations it consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SUBSET_PROVENANCE_FILENAME = "shakebench_sft_subset.json"
SOURCE_MANIFEST_FILENAME = "shakebench_source_collection.json"
COLLECTION_MANIFEST_FILENAME = "shakebench_collection.json"
SELECTION_RULE = "termination_cause == success_latched"
# Format metadata that is copied verbatim.  GR00T-style loaders derive training caches
# (stats_gr00t.json, steps_data_index.pkl) describe the source episodes and must
# not travel with a subset that holds different ones, so this is an allowlist,
# not a copy of the meta directory.  info.json, episodes.jsonl,
# episodes_stats.jsonl and stats.json are rewritten from the selected episodes.
FORMAT_METADATA_FILES = ("tasks.jsonl", "modality.json")


def is_successful_episode(episode: Mapping[str, Any]) -> bool:
    """The frozen success rule; the only place that decides SFT eligibility."""

    return episode.get("success") is True and episode.get("termination_cause") == "success_latched"


def sft_subset_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the success-only selection for one collection attempt list."""

    selected = [episode for episode in episodes if is_successful_episode(episode)]
    return {
        "selection_rule": SELECTION_RULE,
        "episode_indices": [int(episode["episode_index"]) for episode in selected],
        "success_count": len(selected),
        "attempted_episode_count": len(episodes),
        "failed_episode_count": len(episodes) - len(selected),
        "selected_frame_count": sum(int(episode["steps"]) for episode in selected),
        "exporter": "shakebench.scripts.export_sft_subset",
    }


def select_successful_episodes(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the successful episode records of one complete collection manifest."""

    if not isinstance(manifest, Mapping) or manifest.get("complete") is not True:
        raise ValueError("source collection manifest must be complete")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("source collection manifest has no episodes")
    selected = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping) or episode.get("episode_index") != index:
            raise ValueError("collection episodes must be listed in episode order")
        if episode.get("success") is True and episode.get("termination_cause") != "success_latched":
            raise ValueError(
                f"episode {index} claims success with termination_cause {episode.get('termination_cause')!r}"
            )
        if is_successful_episode(episode):
            selected.append(dict(episode))
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _counter_stats(values: Sequence[float]) -> dict[str, Any]:
    """Per-episode LeRobot stats for one per-frame scalar column."""

    array = np.asarray(values, dtype=np.float64)
    return {
        "min": [array.min().item()],
        "max": [array.max().item()],
        "mean": [array.mean().item()],
        "std": [array.std().item()],
        "count": [int(array.size)],
    }


def aggregate_episode_stats(stats_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge per-episode LeRobot stats with the writer's count-weighted rule."""

    features = set(stats_rows[0])
    for row in stats_rows[1:]:
        features &= set(row)
    aggregated = {}
    for feature in sorted(features):
        entries = [row[feature] for row in stats_rows]
        counts = np.stack([np.asarray(entry["count"], dtype=np.float64) for entry in entries])
        means = np.stack([np.asarray(entry["mean"], dtype=np.float64) for entry in entries])
        variances = np.stack([np.asarray(entry["std"], dtype=np.float64) ** 2 for entry in entries])
        total_count = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        total_mean = (means * counts).sum(axis=0) / total_count
        total_variance = ((variances + (means - total_mean) ** 2) * counts).sum(axis=0) / total_count
        aggregated[feature] = {
            "min": np.min(np.stack([np.asarray(entry["min"], dtype=np.float64) for entry in entries]), axis=0).tolist(),
            "max": np.max(np.stack([np.asarray(entry["max"], dtype=np.float64) for entry in entries]), axis=0).tolist(),
            "mean": np.asarray(total_mean).tolist(),
            "std": np.asarray(np.sqrt(total_variance)).tolist(),
            "count": np.asarray(total_count).tolist(),
        }
    return aggregated


def _write_renumbered_parquet(source: Path, target: Path, *, episode_index: int, first_frame_index: int) -> int:
    """Copy one episode parquet under the exported episode and global frame indices."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    length = table.num_rows
    for name, values in (
        ("episode_index", np.full(length, episode_index, dtype=np.int64)),
        ("index", np.arange(first_frame_index, first_frame_index + length, dtype=np.int64)),
    ):
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
    return length


def export_subset(dataset: Path, output: Path, *, include_failures: bool = False) -> dict[str, Any]:
    """Copy ``dataset`` into ``output`` keeping only the selected episodes.

    Exported episodes are renumbered from zero.  The official reader enumerates
    ``range(total_episodes)`` and refuses a sparse directory, so file names,
    episode metadata, the parquet episode/frame index columns, and the exported
    collection manifest all describe the exported numbering; source indices stay
    in the provenance record.
    """

    meta = dataset / "meta"
    manifest_bytes = (meta / COLLECTION_MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = select_successful_episodes(manifest)
    if include_failures:
        selected = [dict(episode) for episode in manifest["episodes"]]
    elif not selected:
        raise ValueError("no successful episodes; pass --include-failures to train on failures deliberately")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")

    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError("only LeRobot v2.1 datasets can be exported")
    episode_rows = {row["episode_index"]: row for row in _read_jsonl(meta / "episodes.jsonl")}
    stats_rows = {row["episode_index"]: row for row in _read_jsonl(meta / "episodes_stats.jsonl")}
    chunk_size = int(info["chunks_size"])
    source_indices = [int(episode["episode_index"]) for episode in selected]
    for episode in selected:
        index = int(episode["episode_index"])
        if index not in episode_rows or index not in stats_rows:
            raise ValueError(f"episode {index} is missing from the dataset metadata")
        if int(episode_rows[index]["length"]) != int(episode["steps"]):
            raise ValueError(f"episode {index} length disagrees with the collection manifest")

    exported_meta = output / "meta"
    exported_meta.mkdir(parents=True, exist_ok=True)
    for name in FORMAT_METADATA_FILES:
        if (meta / name).is_file():
            shutil.copy2(meta / name, exported_meta / name)
    (exported_meta / SOURCE_MANIFEST_FILENAME).write_bytes(manifest_bytes)

    episodes_jsonl, stats_jsonl, manifest_episodes = [], [], []
    first_frame_index = 0
    for new_index, episode in enumerate(selected):
        source_index = int(episode["episode_index"])
        source_relative = Path(
            info["data_path"].format(episode_chunk=source_index // chunk_size, episode_index=source_index)
        )
        target_relative = Path(info["data_path"].format(episode_chunk=new_index // chunk_size, episode_index=new_index))
        length = _write_renumbered_parquet(
            dataset / source_relative,
            output / target_relative,
            episode_index=new_index,
            first_frame_index=first_frame_index,
        )
        episodes_jsonl.append(dict(episode_rows[source_index], episode_index=new_index))
        stats = dict(stats_rows[source_index].get("stats", {}))
        if "episode_index" in stats:
            stats["episode_index"] = _counter_stats([new_index] * length)
        if "index" in stats:
            stats["index"] = _counter_stats(range(first_frame_index, first_frame_index + length))
        stats_jsonl.append({"episode_index": new_index, "stats": stats})
        manifest_episodes.append(dict(episode, episode_index=new_index, source_episode_index=source_index))
        first_frame_index += length

    _write_jsonl(exported_meta / "episodes.jsonl", episodes_jsonl)
    _write_jsonl(exported_meta / "episodes_stats.jsonl", stats_jsonl)
    if (meta / "stats.json").is_file():
        aggregated = aggregate_episode_stats([row["stats"] for row in stats_jsonl])
        (exported_meta / "stats.json").write_text(json.dumps(aggregated, indent=4) + "\n", encoding="utf-8")
    info.update(
        {
            "total_episodes": len(selected),
            "total_frames": first_frame_index,
            "total_chunks": len({new_index // chunk_size for new_index in range(len(selected))}),
            "splits": {"train": f"0:{len(selected)}"},
        }
    )
    (exported_meta / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    source_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    selection_rule = "all episodes" if include_failures else SELECTION_RULE
    exported_manifest = {
        **{key: value for key, value in manifest.items() if key not in {"episodes", "requested_states", "sft_subset"}},
        "complete": True,
        "source_dataset": dataset.name,
        "source_manifest_sha256": source_manifest_sha256,
        "requested_states": [episode["state"]["state_id"] for episode in manifest_episodes],
        "episodes": manifest_episodes,
        "sft_subset": {
            **sft_subset_summary(manifest_episodes),
            "selection_rule": selection_rule,
            "source_episode_indices": source_indices,
        },
    }
    (exported_meta / COLLECTION_MANIFEST_FILENAME).write_text(
        json.dumps(exported_manifest, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_id": "shakebench.sft_subset",
        "schema_version": 2,
        "source_dataset": dataset.name,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_copy": SOURCE_MANIFEST_FILENAME,
        "selection_rule": selection_rule,
        "source_episode_indices": source_indices,
        "exported_episode_indices": list(range(len(selected))),
        "exported_episode_count": len(selected),
        "exported_frame_count": first_frame_index,
        "attempted_episode_count": len(manifest["episodes"]),
        "excluded_failure_count": len(manifest["episodes"]) - len(selected),
        "failures_retained_in_source_only": not include_failures,
    }
    (exported_meta / SUBSET_PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Collected LeRobot v2.1 dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New subset directory; never overwritten")
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Deliberately keep failed rollouts instead of the success-only rule",
    )
    args = parser.parse_args(argv)
    provenance = export_subset(args.dataset, args.output, include_failures=args.include_failures)

    # Re-open the export with the same reader a trainer uses.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("shakebench/oracle-gamma-zero", root=args.output)
    assert dataset.num_episodes == provenance["exported_episode_count"], dataset.num_episodes
    assert len(dataset) == provenance["exported_frame_count"], len(dataset)
    assert (
        sorted({int(dataset[index]["episode_index"]) for index in range(len(dataset))})
        == provenance["exported_episode_indices"]
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
