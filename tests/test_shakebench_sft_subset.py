"""Success-only SFT selection and subset export provenance."""

from __future__ import annotations

import pytest

from robosuite.scripts.shakebench_export_sft_subset import (
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
