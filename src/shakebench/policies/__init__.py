"""Public policy registry and privilege-aware construction helpers."""

from __future__ import annotations

from typing import Any

from .base import Policy, policy_env_kwargs, privilege_label
from .classical import ClassicalPolicy
from .oracle import OracleFullPolicy, OraclePhasePolicy, OracleReactivePolicy
from .random import RandomPolicy


POLICY_TYPES = {
    "oracle_full": OracleFullPolicy,
    "oracle_phase": OraclePhasePolicy,
    "oracle_reactive": OracleReactivePolicy,
    "classical": ClassicalPolicy,
    "random": RandomPolicy,
}


def make_policy(name: str, **kwargs: Any):
    try:
        policy_type = POLICY_TYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; available={sorted(POLICY_TYPES)}") from exc
    return policy_type(**kwargs)


__all__ = [
    "ClassicalPolicy",
    "OracleFullPolicy",
    "OraclePhasePolicy",
    "OracleReactivePolicy",
    "POLICY_TYPES",
    "Policy",
    "RandomPolicy",
    "make_policy",
    "policy_env_kwargs",
    "privilege_label",
]
