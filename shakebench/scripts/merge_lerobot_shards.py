"""Merge disjoint LeRobot v2.1 collection shards into one episode-indexed dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from shakebench.scripts.export_sft_subset import _counter_stats, _read_jsonl, _write_jsonl, sft_subset_summary


def merge_shards(
    shards, *, output: Path, ordered_state_ids, video_dir: Path, repo_id="shakebench_pickplace_300"
):
    """Merge source episodes in requested state order; videos remain in a shared sidecar directory."""
    if not shards or not ordered_state_ids or output.exists():
        raise ValueError("sources and ordered states are required, and output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(staging)

    try:
        sources, by_state_id = [], {}
        for source in shards:
            source = Path(source)
            meta = source / "meta"
            info = json.loads((meta / "info.json").read_text())
            manifest = json.loads((meta / "shakebench_collection.json").read_text())
            if info.get("codebase_version") != "v2.1" or manifest.get("complete") is not True:
                raise ValueError(f"incomplete or unsupported shard: {source}")
            if info.get("total_episodes") != len(manifest.get("episodes", [])):
                raise ValueError(f"shard manifest/info episode counts disagree: {source}")
            source_record = {
                "path": str(source),
                "info": info,
                "manifest": manifest,
                "episodes": {row["episode_index"]: row for row in _read_jsonl(meta / "episodes.jsonl")},
                "stats": {row["episode_index"]: row for row in _read_jsonl(meta / "episodes_stats.jsonl")},
            }
            if len(source_record["episodes"]) != info["total_episodes"]:
                raise ValueError(f"shard episode metadata count disagrees: {source}")
            for episode in manifest["episodes"]:
                state_id = episode["state"]["state_id"]
                if state_id in by_state_id:
                    raise ValueError(f"duplicate state_id across shards: {state_id}")
                by_state_id[state_id] = (source_record, episode)
            sources.append(source_record)

        if len(ordered_state_ids) != len(set(ordered_state_ids)) or not set(ordered_state_ids) <= set(by_state_id):
            raise ValueError("merge order must contain unique state_ids present in the source shards")
        excluded_state_ids = sorted(set(by_state_id) - set(ordered_state_ids))
        base_info = sources[0]["info"]
        base_features = base_info["features"]
        for source in sources[1:]:
            info = source["info"]
            if info["features"] != base_features or info["fps"] != base_info["fps"]:
                raise ValueError("LeRobot shard feature/fps contracts differ")

        state_rows = [by_state_id[state_id] for state_id in ordered_state_ids]
        instructions = list(dict.fromkeys(episode["instruction"] for _, episode in state_rows))
        task_indices = {instruction: index for index, instruction in enumerate(instructions)}
        modalities = json.loads((Path(sources[0]["path"]) / "meta" / "modality.json").read_text())
        meta_out = staging / "meta"
        meta_out.mkdir(parents=True)
        if (Path(sources[0]["path"]) / ".gitattributes").is_file():
            shutil.copy2(Path(sources[0]["path"]) / ".gitattributes", staging / ".gitattributes")
        (meta_out / "modality.json").write_text(json.dumps(modalities, indent=2) + "\n")
        _write_jsonl(meta_out / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(instructions)])

        from shakebench.scripts.export_sft_subset import _write_renumbered_parquet

        episodes_out, stats_out, manifest_episodes = [], [], []
        first_frame = 0
        chunk_size = int(base_info["chunks_size"])
        for index, (state_id, (source, episode)) in enumerate(zip(ordered_state_ids, state_rows)):
            source_index = int(episode["episode_index"])
            row = source["episodes"].get(source_index)
            stats_row = source["stats"].get(source_index)
            if row is None or stats_row is None or int(row["length"]) != int(episode["steps"]):
                raise ValueError(f"episode metadata mismatch for {state_id}")
            source_info = source["info"]
            source_data = Path(source["path"]) / source_info["data_path"].format(
                episode_chunk=source_index // int(source_info["chunks_size"]), episode_index=source_index
            )
            target_data = staging / base_info["data_path"].format(
                episode_chunk=index // chunk_size, episode_index=index
            )
            task_index = task_indices[episode["instruction"]]
            length = _write_renumbered_parquet(
                source_data,
                target_data,
                episode_index=index,
                first_frame_index=first_frame,
                task_index=task_index,
            )
            episodes_out.append({**row, "episode_index": index, "tasks": [episode["instruction"]]})
            stats = dict(stats_row["stats"])
            if "episode_index" in stats:
                stats["episode_index"] = _counter_stats([index] * length)
            if "index" in stats:
                stats["index"] = _counter_stats(range(first_frame, first_frame + length))
            if "task_index" in stats:
                stats["task_index"] = _counter_stats([task_index] * length)
            stats_out.append({"episode_index": index, "stats": stats})
            videos = list(video_dir.rglob(f"{state_id}.mp4"))
            if len(videos) != 1 or not videos[0].is_file() or videos[0].stat().st_size <= 48:
                raise FileNotFoundError(f"expected one nonempty rollout video for {state_id}, found {videos}")
            video = videos[0]
            manifest_episodes.append(
                {
                    **episode,
                    "episode_index": index,
                    "source_episode_index": source_index,
                    "video_path": str(video),
                }
            )
            first_frame += length

        _write_jsonl(meta_out / "episodes.jsonl", episodes_out)
        _write_jsonl(meta_out / "episodes_stats.jsonl", stats_out)
        info = dict(base_info)
        info.update(
            total_episodes=len(episodes_out),
            total_frames=first_frame,
            total_tasks=len(instructions),
            total_videos=0,
            total_chunks=(len(episodes_out) + chunk_size - 1) // chunk_size,
            splits={"train": f"0:{len(episodes_out)}"},
        )
        (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n")

        first_manifest = sources[0]["manifest"]
        state_dir = Path(sources[0]["path"]).parent.parent / "states"
        pool_manifest_path = state_dir / "state_pools.json"
        pool_manifest = json.loads(pool_manifest_path.read_text()) if pool_manifest_path.is_file() else None
        manifest = {
            **{
                key: value
                for key, value in first_manifest.items()
                if key not in {"episodes", "requested_states", "tasks", "sft_subset", "shards", "device"}
            },
            "complete": True,
            "repo_id": repo_id,
            "device": "multi_gpu",
            "devices": [source["manifest"].get("device") for source in sources],
            "video_output": str(video_dir),
            "legacy_task_instructions": first_manifest.get("legacy_task_instructions", False),
            "tasks": {
                state_id: episode["instruction"] for state_id, (_, episode) in zip(ordered_state_ids, state_rows)
            },
            "requested_states": list(ordered_state_ids),
            "excluded_source_state_ids": excluded_state_ids,
            "state_authority": first_manifest.get("state_authority", {
                "kind": "train",
                "split": "train",
                "scoreable": False,
                "pool_manifest": str(pool_manifest_path) if pool_manifest is not None else None,
                "pools": pool_manifest.get("pools", []) if pool_manifest is not None else [],
            }),
            "shards": [
                {
                    "path": source["path"],
                    "device": source["manifest"].get("device"),
                    "episode_count": source["info"]["total_episodes"],
                    "success_count": source["manifest"].get("sft_subset", {}).get("success_count"),
                }
                for source in sources
            ],
            "episodes": manifest_episodes,
            "sft_subset": sft_subset_summary(manifest_episodes),
        }
        (meta_out / "shakebench_collection.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        return {
            "episodes": len(episodes_out),
            "frames": first_frame,
            "success_count": manifest["sft_subset"]["success_count"],
            "variant_counts": {
                f"{object_id}/{region}": count
                for (object_id, region), count in Counter(
                    (
                        episode["state"].get("task", {}).get("object_id"),
                        episode["state"].get("grasp_region", "body"),
                    )
                    for episode in manifest_episodes
                ).items()
            },
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True, help="ordered LeRobot shard root")
    parser.add_argument("--order-file", type=Path, required=True, help="JSON file with ordered state_ids")
    parser.add_argument("--output", type=Path, required=True, help="New merged LeRobot v2.1 root")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory with per-state MP4 files")
    parser.add_argument("--repo-id", default="shakebench_pickplace_300", help="Dataset identifier in the manifest")
    args = parser.parse_args(argv)
    order = json.loads(args.order_file.read_text())
    result = merge_shards(
        args.source,
        output=args.output,
        ordered_state_ids=order["state_ids"],
        video_dir=args.video_dir,
        repo_id=args.repo_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
