"""Deterministic task-state expansion for the four object-pose variants.

Official parents are assigned one variant each, in a balanced round robin.
Knee parents retain all four variants for paired calibration diagnostics.
No outcome-dependent filtering or Gamma selection occurs here.  Version 2
artifacts describe the retired two-surface task set and are rejected: their
records name objects and a surface dimension that no longer exist.
"""

from __future__ import annotations

TASK_STATE_SCHEMA = "shakebench.phase09.task_states"
