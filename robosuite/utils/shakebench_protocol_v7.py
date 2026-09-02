"""Phase 06R6/V7 resolved-state compatibility boundary.

V7 deliberately keeps the V6 resolver as the single protocol-to-adapter
boundary.  The contact table is expanded, but the common runtime, driver,
isolator, parity, replay, and deep-freeze semantics are unchanged.  Aliasing
the resolver here makes that inheritance explicit to callers and prevents a
second, subtly different state materializer from appearing in the V7 runner.
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


V7ProtocolStateError = V6ProtocolStateError
resolve_protocol_state_v7 = resolve_protocol_state_v6
resolve_all_protocol_states_v7 = resolve_all_protocol_states_v6


__all__ = [
    "ResolvedCommon",
    "ResolvedContact",
    "ResolvedDriver",
    "ResolvedIsolator",
    "ResolvedParity",
    "ResolvedProbeState",
    "ResolvedReplay",
    "V7ProtocolStateError",
    "canonical_json",
    "legal_replay_binding_templates",
    "resolve_all_protocol_states_v7",
    "resolve_protocol_state_v7",
    "resolve_replay_binding_templates",
    "sha256_json",
]
