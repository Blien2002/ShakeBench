"""Policy protocol used by evaluation and future oracle baselines."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    requires_privileged: tuple[str, ...]

    def reset(self) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray: ...


_PRIVILEGE_KWARGS: dict[str, dict[str, bool]] = {
    "object": {"use_object_obs": True},
    "vibration": {"use_vibration_obs": True},
    "phase": {"use_phase_obs": True},
    "instantaneous_load": {"use_instantaneous_load_obs": True},
}


def policy_env_kwargs(policy: Policy) -> dict[str, Any]:
    """Translate a policy's declared privileges into explicit env switches.

    Evaluators use this function instead of maintaining a second, hand-written
    privilege table. Unknown declarations fail closed.
    """

    result: dict[str, Any] = {
        "use_object_obs": False,
        "use_vibration_obs": False,
        "use_phase_obs": False,
        "use_instantaneous_load_obs": False,
    }
    for privilege in policy.requires_privileged:
        try:
            result.update(_PRIVILEGE_KWARGS[privilege])
        except KeyError as exc:
            raise ValueError(f"unknown privileged observation group {privilege!r}") from exc
    return result


def privilege_label(policy: Policy) -> str:
    """Stable human-readable label used by scorecards and the leaderboard."""

    return ", ".join(policy.requires_privileged) if policy.requires_privileged else "none"
