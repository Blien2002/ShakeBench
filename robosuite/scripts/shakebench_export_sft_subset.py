"""Export a success-only SFT dataset from a collected ShakeBench LeRobot dataset.

Failed rollouts stay in the source dataset for audit.  The exported copy holds
only episodes selected by the frozen rule (``success_latched``) and records the
source manifest hash, so a trainer can prove which demonstrations it consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SUBSET_PROVENANCE_FILENAME = "shakebench_sft_subset.json"
SELECTION_RULE = "termination_cause == success_latched"


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
        "exporter": "robosuite.scripts.shakebench_export_sft_subset",
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


def export_subset(dataset: Path, output: Path, *, include_failures: bool = False) -> dict[str, Any]:
    """Copy ``dataset`` into ``output`` keeping only the selected episodes."""

    meta = dataset / "meta"
    manifest_bytes = (meta / "shakebench_collection.json").read_bytes()
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
    indices = [int(episode["episode_index"]) for episode in selected]
    frame_count = 0
    for episode in selected:
        index = int(episode["episode_index"])
        if index not in episode_rows or index not in stats_rows:
            raise ValueError(f"episode {index} is missing from the dataset metadata")
        if int(episode_rows[index]["length"]) != int(episode["steps"]):
            raise ValueError(f"episode {index} length disagrees with the collection manifest")
        frame_count += int(episode["steps"])

    shutil.copytree(meta, output / "meta")
    for index in indices:
        relative = Path(info["data_path"].format(episode_chunk=index // chunk_size, episode_index=index))
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / relative, target)
    _write_jsonl(output / "meta" / "episodes.jsonl", [episode_rows[index] for index in indices])
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", [stats_rows[index] for index in indices])
    info.update(
        {
            "total_episodes": len(indices),
            "total_frames": frame_count,
            "total_chunks": len({index // chunk_size for index in indices}),
            "splits": {"train": f"0:{len(indices)}"},
        }
    )
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    provenance = {
        "schema_id": "shakebench.sft_subset",
        "schema_version": 1,
        "source_dataset": dataset.name,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_rule": "all episodes" if include_failures else SELECTION_RULE,
        "selected_episode_indices": indices,
        "selected_frame_count": frame_count,
        "attempted_episode_count": len(manifest["episodes"]),
        "excluded_failure_count": len(manifest["episodes"]) - len(selected),
        "failures_retained_in_source_only": not include_failures,
    }
    (output / "meta" / SUBSET_PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
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
    assert dataset.num_episodes == len(provenance["selected_episode_indices"]), dataset.num_episodes
    assert len(dataset) == provenance["selected_frame_count"], len(dataset)
    assert (
        sorted({int(dataset[index]["episode_index"]) for index in range(len(dataset))})
        == provenance["selected_episode_indices"]
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
