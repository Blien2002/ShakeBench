"""Phase 06R7/V8 resolved-state compatibility boundary.

The V8 normal-contact family changes only contact candidate-owned fields.
The V6 immutable resolver remains the single materialization boundary for
common runtime, driver, isolator, parity, replay, and frozen child values.
"""

from __future__ import annotations

from robosuite.utils.shakebench_protocol_v6 import (
    ResolvedCommon,
    ResolvedContact,
    ResolvedDriver,
    ResolvedIsolator,
    ResolvedParity,
    ResolvedProbeState,
    ResolvedReplay,
    V6ProtocolStateError,
    canonical_json,
    legal_replay_binding_templates,
    resolve_all_protocol_states_v6,
    resolve_protocol_state_v6,
    resolve_replay_binding_templates,
    sha256_json,
)


V8ProtocolStateError = V6ProtocolStateError
resolve_protocol_state_v8 = resolve_protocol_state_v6
resolve_all_protocol_states_v8 = resolve_all_protocol_states_v6


__all__ = [
    "ResolvedCommon",
    "ResolvedContact",
    "ResolvedDriver",
    "ResolvedIsolator",
    "ResolvedParity",
    "ResolvedProbeState",
    "ResolvedReplay",
    "V8ProtocolStateError",
    "canonical_json",
    "legal_replay_binding_templates",
    "resolve_all_protocol_states_v8",
    "resolve_protocol_state_v8",
    "resolve_replay_binding_templates",
    "sha256_json",
]
