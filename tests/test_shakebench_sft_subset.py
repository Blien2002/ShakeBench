"""Success-only SFT selection and subset export provenance."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robosuite.scripts.shakebench_evaluate import observation_config_from_collection
from robosuite.scripts.shakebench_export_sft_subset import (
    aggregate_episode_stats,
    export_subset,
    is_successful_episode,
    select_successful_episodes,
    sft_subset_summary,
)


def _manifest(*episodes):
    return {"complete": True, "episodes": list(episodes)}


def _episode(index, *, success, cause, steps=10):
    return {
        "episode_index": index,
        "state": {"state_id": f"state-{index:03d}"},
        "steps": steps,
        "success": success,
        "termination_cause": cause,
    }


def test_only_success_latched_episodes_are_sft_eligible():
    attempted = [
        _episode(0, success=True, cause="success_latched", steps=173),
        _episode(1, success=False, cause="horizon_exhausted", steps=1200),
        _episode(2, success=True, cause="success_latched", steps=90),
        _episode(3, success=False, cause="task_rule_violation", steps=40),
    ]

    selected = select_successful_episodes(_manifest(*attempted))

    assert [row["episode_index"] for row in selected] == [0, 2]
    summary = sft_subset_summary(attempted)
    assert summary["episode_indices"] == [0, 2]
    assert summary["selected_frame_count"] == 263
    assert summary["attempted_episode_count"] == 4 and summary["failed_episode_count"] == 2
    assert summary["success_count"] == 2


def test_empty_or_unfinished_collections_cannot_be_exported():
    with pytest.raises(ValueError, match="complete"):
        select_successful_episodes({"complete": False, "episodes": []})
    with pytest.raises(ValueError, match="no episodes"):
        select_successful_episodes(_manifest())
    with pytest.raises(ValueError, match="episode order"):
        select_successful_episodes(_manifest(_episode(1, success=True, cause="success_latched")))
    with pytest.raises(ValueError, match="claims success"):
        select_successful_episodes(_manifest(_episode(0, success=True, cause="horizon_exhausted")))


def test_success_flag_alone_is_not_eligible():
    assert not is_successful_episode({"success": True})
    assert not is_successful_episode({"termination_cause": "success_latched"})
    assert is_successful_episode({"success": True, "termination_cause": "success_latched"})


def _write_source(root, lengths, *, failed):
    """Minimal LeRobot v2.1 collection: two episodes plus StarVLA's derived caches."""

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {
        "action": {"dtype": "float32", "shape": [7], "names": None},
        "observation.images.main": {"dtype": "image", "shape": [4, 4, 3], "names": ["height", "width", "channels"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Panda",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Pick up the can."}) + "\n")
    (root / "meta" / "steps_data_index.pkl").write_bytes(b"starvla derived cache")
    (root / "meta" / "stats_gr00t.json").write_text("{}\n")

    episode_rows, stats_rows, episodes = [], [], []
    offset = 0
    for index, length in enumerate(lengths):
        frames = np.arange(offset, offset + length, dtype=np.int64)
        table = pa.table(
            {
                "action": pa.array([[float(index)] * 7] * length, pa.list_(pa.float32())),
                "timestamp": pa.array(np.arange(length) / 20, pa.float32()),
                "frame_index": pa.array(np.arange(length), pa.int64()),
                "episode_index": pa.array(np.full(length, index), pa.int64()),
                "index": pa.array(frames, pa.int64()),
                "task_index": pa.array(np.zeros(length), pa.int64()),
            }
        )
        pq.write_table(table, root / info["data_path"].format(episode_chunk=0, episode_index=index))
        episode_rows.append({"episode_index": index, "tasks": ["Pick up the can."], "length": length})
        stats_rows.append(
            {
                "episode_index": index,
                "stats": {
                    "action": {
                        "min": [float(index)] * 7,
                        "max": [float(index)] * 7,
                        "mean": [float(index)] * 7,
                        "std": [0.0] * 7,
                        "count": [length],
                    },
                    "episode_index": {
                        "min": [index],
                        "max": [index],
                        "mean": [float(index)],
                        "std": [0.0],
                        "count": [length],
                    },
                    "index": {
                        "min": [offset],
                        "max": [offset + length - 1],
                        "mean": [float(offset + (length - 1) / 2)],
                        "std": [float(np.arange(length).std())],
                        "count": [length],
                    },
                },
            }
        )
        episodes.append(
            {
                "episode_index": index,
                "state": {"state_id": f"state-{index:03d}", "object_xy_m": [0.0, 0.0]},
                "steps": length,
                "success": index not in failed,
                "termination_cause": "horizon_exhausted" if index in failed else "success_latched",
            }
        )
        offset += length
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_rows))
    (root / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stats_rows))
    manifest = {
        "complete": True,
        "scoreable": False,
        "gamma": 0.0,
        "task": "Pick up the can.",
        "cameras": {"observation.images.main": "task_close"},
        "requested_states": [episode["state"]["state_id"] for episode in episodes],
        "episodes": episodes,
    }
    (root / "meta" / "shakebench_collection.json").write_text(json.dumps(manifest))
    return root


def test_sparse_success_selection_is_renumbered_for_the_official_reader(tmp_path):
    """Regression: keeping source indices while shrinking total_episodes broke the reader."""

    source = _write_source(tmp_path / "source", (3, 4), failed={0})
    subset = tmp_path / "subset"

    provenance = export_subset(source, subset)

    assert provenance["source_episode_indices"] == [1]
    assert provenance["exported_episode_indices"] == [0]
    assert provenance["exported_episode_count"] == 1 and provenance["exported_frame_count"] == 4
    assert provenance["excluded_failure_count"] == 1
    assert sorted(path.name for path in (subset / "data/chunk-000").iterdir()) == ["episode_000000.parquet"]

    table = pq.read_table(subset / "data/chunk-000/episode_000000.parquet")
    assert table.column("episode_index").to_pylist() == [0] * 4
    assert table.column("index").to_pylist() == [0, 1, 2, 3]
    assert table.column("action").to_pylist()[0] == [1.0] * 7

    info = json.loads((subset / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["splits"]) == (1, 4, {"train": "0:1"})
    episodes = [json.loads(line) for line in (subset / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert episodes == [{"episode_index": 0, "tasks": ["Pick up the can."], "length": 4}]
    stats = json.loads((subset / "meta" / "episodes_stats.jsonl").read_text().splitlines()[0])
    assert stats["episode_index"] == 0
    assert stats["stats"]["episode_index"] == {
        "min": [0],
        "max": [0],
        "mean": [0.0],
        "std": [0.0],
        "count": [4],
    }
    assert stats["stats"]["index"]["min"] == [0] and stats["stats"]["index"]["max"] == [3]
    assert stats["stats"]["index"]["mean"] == [1.5] and stats["stats"]["action"]["mean"] == [1.0] * 7


def test_exported_metadata_never_carries_the_derived_training_caches(tmp_path):
    """Regression: copied StarVLA caches kept reporting the excluded episodes."""

    source = _write_source(tmp_path / "source", (3, 4), failed={0})
    subset = tmp_path / "subset"

    export_subset(source, subset)

    assert (source / "meta" / "steps_data_index.pkl").is_file()
    assert not (subset / "meta" / "steps_data_index.pkl").exists()
    assert not (subset / "meta" / "stats_gr00t.json").exists()
    assert (subset / "meta" / "tasks.jsonl").read_bytes() == (source / "meta" / "tasks.jsonl").read_bytes()


def test_exported_manifest_describes_only_the_exported_data(tmp_path):
    """Regression: the subset re-used the source manifest, so excluded states stayed 'training'."""

    source = _write_source(tmp_path / "source", (3, 4), failed={0})
    subset = tmp_path / "subset"

    provenance = export_subset(source, subset)

    manifest = json.loads((subset / "meta" / "shakebench_collection.json").read_text())
    assert manifest["complete"] is True
    assert manifest["source_manifest_sha256"] == provenance["source_manifest_sha256"]
    assert [episode["state"]["state_id"] for episode in manifest["episodes"]] == ["state-001"]
    assert manifest["episodes"][0]["episode_index"] == 0 and manifest["episodes"][0]["source_episode_index"] == 1
    assert manifest["requested_states"] == ["state-001"]
    assert manifest["sft_subset"]["selection_rule"] == "termination_cause == success_latched"
    assert (subset / "meta" / "shakebench_source_collection.json").read_bytes() == (
        source / "meta" / "shakebench_collection.json"
    ).read_bytes()

    config = observation_config_from_collection(subset)
    assert [state["state_id"] for state in config["train_states"]] == ["state-001"]
    assert config["subset_provenance"]["source_episode_indices"] == [1]
    # Two different selections of one source must not share a fingerprint.
    assert config["manifest_sha256"] != provenance["source_manifest_sha256"]
    assert len(config["manifest_sha256"]) == 64


def test_include_failures_keeps_every_episode_and_says_so(tmp_path):
    source = _write_source(tmp_path / "source", (3, 4), failed={0})
    subset = tmp_path / "subset"

    provenance = export_subset(source, subset, include_failures=True)

    manifest = json.loads((subset / "meta" / "shakebench_collection.json").read_text())
    assert provenance["selection_rule"] == "all episodes" and provenance["excluded_failure_count"] == 0
    assert manifest["sft_subset"]["selection_rule"] == "all episodes"
    assert [episode["termination_cause"] for episode in manifest["episodes"]] == [
        "horizon_exhausted",
        "success_latched",
    ]
    table = pq.read_table(subset / "data/chunk-000/episode_000001.parquet")
    assert table.column("index").to_pylist() == [3, 4, 5, 6]


def test_aggregated_stats_are_count_weighted_like_the_writer():
    rows = [
        {"f": {"min": [0.0], "max": [1.0], "mean": [0.0], "std": [0.0], "count": [3]}},
        {"f": {"min": [1.0], "max": [2.0], "mean": [2.0], "std": [0.0], "count": [1]}},
    ]

    aggregated = aggregate_episode_stats(rows)

    assert aggregated["f"]["min"] == [0.0] and aggregated["f"]["max"] == [2.0]
    assert aggregated["f"]["mean"] == [0.5] and aggregated["f"]["count"] == [4]
    assert aggregated["f"]["std"] == pytest.approx([np.sqrt(0.75)])
