"""Fail-closed Phase 06F official-physics finalization.

This module owns the ordering and consistency rules for the final contact
freeze.  Physics and filesystem operations are intentionally expressed by the
small ``EvidenceStore`` protocol so the same state machine can be exercised by
synthetic tests and the production MuJoCo runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from pathlib import Path

from robosuite.utils.shakebench_physics import physics_profile_hash


SCHEMA_ID = "shakebench.phase06f.finalization"
BINDING_RULE = "contact_candidate_id = selection.contact_winner"


class FinalizationError(ValueError):
    """Raised when a Phase 06F protocol or evidence contract is invalid."""


def canonical_json(value: Any) -> str:
    """Return the canonical representation used by Phase 06F artifacts."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artifact_hash(value: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding only its self-referential hash."""

    payload = copy.deepcopy(dict(value))
    payload.pop("payload_sha256", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def seal_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["payload_sha256"] = artifact_hash(payload)
    return payload


@runtime_checkable
class EvidenceStore(Protocol):
    """Internal I/O seam used by :func:`finalize_official_physics`."""

    @property
    def protocol_sha256(self) -> str: ...

    def verify_inherited(self, protocol: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def execute_contact(self, candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def verify_contact(
        self, candidate: Mapping[str, Any], evidence: Mapping[str, Any], protocol: Mapping[str, Any]
    ) -> tuple[bool, Sequence[str]]: ...

    def freeze_selection(self, selection: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def freeze_dependent_manifest(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def execute_dependent(self, state: Mapping[str, Any], protocol: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def verify_dependent(
        self, state: Mapping[str, Any], evidence: Mapping[str, Any], protocol: Mapping[str, Any]
    ) -> tuple[bool, Sequence[str]]: ...


@dataclass(frozen=True)
class BlockedResult:
    status: str
    blocking_stage: str
    reason: str
    selection_artifact: Mapping[str, Any] | None
    dependent_manifest: None
    official_profile: None
    final_status: Mapping[str, Any]
    handoff: Mapping[str, Any]


@dataclass(frozen=True)
class PublicationPlan:
    status: str
    contact_winner: str
    selection_artifact: Mapping[str, Any]
    dependent_manifest: Mapping[str, Any]
    dependent_evidence: tuple[Mapping[str, Any], ...]
    official_profile: Mapping[str, Any]
    final_status: Mapping[str, Any]
    handoff: Mapping[str, Any]


def _rows(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, Mapping) for row in value):
        raise FinalizationError(f"{name} must be a non-empty list of mappings")
    return list(value)


def validate_finalization_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate winner-independent registration before any evidence call."""

    if not isinstance(protocol, Mapping):
        raise FinalizationError("protocol must be a mapping")
    if protocol.get("schema_id") != "shakebench.phase06f.physics_selection_protocol":
        raise FinalizationError("unexpected Phase 06F protocol schema")
    if int(protocol.get("schema_version", 0)) != 1 or protocol.get("phase") != "06F":
        raise FinalizationError("unsupported Phase 06F protocol version")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise FinalizationError("Phase 06F protocol must be immutable and pre-registered")
    if protocol.get("selection_authority") != "physics_probes_only":
        raise FinalizationError("selection authority must be physics_probes_only")

    candidates = _rows(protocol.get("contact_candidates"), "contact_candidates")
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if any(not candidate_id for candidate_id in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise FinalizationError("contact candidate IDs must be non-empty and unique")
    if any(not str(row.get("output", "")).endswith(".json") for row in candidates):
        raise FinalizationError("each contact candidate must pre-register an output filename")
    priority = list(protocol.get("selection", {}).get("minimal_intervention_priority", ()))
    if priority != candidate_ids:
        raise FinalizationError("candidate list must exactly follow minimal-intervention priority")
    if protocol.get("selection", {}).get("binding_rule") != BINDING_RULE:
        raise FinalizationError("winner-dependent binding rule is missing")

    templates = _rows(protocol.get("dependent_manifest_template"), "dependent_manifest_template")
    required_groups = {
        "gamma_zero_parity",
        "driver",
        "isolator",
        "contact",
        "gamma_zero_parity_replay",
        "selected_diagnostics",
    }
    groups = {str(row.get("group", "")) for row in templates}
    if groups != required_groups:
        raise FinalizationError("dependent manifest template has incomplete groups")
    for row in templates:
        if row.get("contact_binding") != "selection.contact_winner":
            raise FinalizationError("winner-dependent row is not bound to selection.contact_winner")
        rendered = canonical_json(row)
        concrete = sorted(candidate_id for candidate_id in candidate_ids if candidate_id in rendered)
        if concrete:
            raise FinalizationError("winner-dependent protocol row contains concrete contact ID: " + ", ".join(concrete))
        if "{contact_candidate_id}" not in str(row.get("state_id_template", "")):
            raise FinalizationError("dependent state ID must include the winner placeholder")
        if "{selection_artifact_sha256}" not in str(row.get("state_id_template", "")):
            raise FinalizationError("dependent state ID must include the selection-artifact hash placeholder")

    profile = protocol.get("official_profile")
    if not isinstance(profile, Mapping) or profile.get("profile_id") != "shakebench.official.physics.v2":
        raise FinalizationError("official profile schema/id is not pre-registered")
    if not isinstance(profile.get("physics_template"), Mapping):
        raise FinalizationError("official profile physics template is missing")
    handoff = protocol.get("phase07_handoff")
    if not isinstance(handoff, Mapping) or handoff.get("schema_id") != "shakebench.phase06.to_phase07_handoff":
        raise FinalizationError("Phase 07 handoff schema is not pre-registered")
    return {"candidate_ids": candidate_ids, "dependent_template_count": len(templates), "groups": sorted(groups)}


def materialize_dependent_manifest(
    protocol: Mapping[str, Any], *, winner: str, selection_artifact_sha256: str
) -> dict[str, Any]:
    """Bind dependent rows to the actual winner and frozen selection hash."""

    validation = validate_finalization_protocol(protocol)
    if winner not in validation["candidate_ids"]:
        raise FinalizationError("winner is not a registered contact candidate")
    rows = []
    for template in protocol["dependent_manifest_template"]:
        state = copy.deepcopy(dict(template))
        state["state_id"] = str(state.pop("state_id_template")).format(
            contact_candidate_id=winner,
            selection_artifact_sha256=selection_artifact_sha256,
        )
        output_template = state.pop("output_template", None)
        if output_template is not None:
            state["output"] = str(output_template).format(
                contact_candidate_id=winner,
                selection_artifact_sha256=selection_artifact_sha256,
            )
        state["contact_candidate_id"] = winner
        state["selection_artifact_sha256"] = selection_artifact_sha256
        rows.append(state)
    return seal_artifact(
        {
            "schema_id": SCHEMA_ID + ".dependent_manifest",
            "schema_version": 1,
            "protocol_sha256": None,
            "contact_winner": winner,
            "selection_artifact_sha256": selection_artifact_sha256,
            "states": rows,
        }
    )


def _blocked(
    *, protocol_hash: str, stage: str, reason: str, selection: Mapping[str, Any] | None = None
) -> BlockedResult:
    status = seal_artifact(
        {
            "schema_id": "shakebench.phase06.final_status",
            "schema_version": 1,
            "status": "BLOCKED",
            "blocking_stage": stage,
            "reason": reason,
            "protocol_sha256": protocol_hash,
            "official_profile": None,
            "phase07_authorized": False,
        }
    )
    handoff = seal_artifact(
        {
            "schema_id": "shakebench.phase06.to_phase07_handoff",
            "schema_version": 1,
            "status": "BLOCKED",
            "blocking_stage": stage,
            "reason": reason,
            "protocol_sha256": protocol_hash,
            "selection_artifact_sha256": None if selection is None else selection.get("payload_sha256"),
            "dependent_manifest_sha256": None,
            "official_profile_sha256": None,
            "phase07_authorized": False,
        }
    )
    return BlockedResult("BLOCKED", stage, reason, selection, None, None, status, handoff)


def _profile(protocol: Mapping[str, Any], *, winner: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    spec = protocol["official_profile"]
    physics = copy.deepcopy(dict(spec["physics_template"]))
    physics["contact"] = {
        key: copy.deepcopy(winner[key])
        for key in (
            "candidate_id",
            "condim",
            "sliding_mu",
            "torsional_mu",
            "rolling_mu",
            "margin_m",
            "gap_m",
            "solref",
            "solimp",
            "pair_scope",
        )
    }
    physics["contact"]["friction_encoding"] = "isotropic_pair_5d"
    if "interfaces" in winner:
        physics["contact"]["interfaces"] = copy.deepcopy(winner["interfaces"])
    payload = {
        "schema_id": "shakebench.official.physics",
        "schema_version": 1,
        "profile_id": spec["profile_id"],
        "status": "official_immutable",
        "scoreable": True,
        "selection_basis": "phase06f_task_native_physics_only",
        "protocol_file": protocol.get("protocol_filename"),
        "protocol_sha256": provenance["protocol_sha256"],
        "selection_artifact_sha256": provenance["selection_artifact_sha256"],
        "dependent_manifest_sha256": provenance["dependent_manifest_sha256"],
        "evidence_manifest_sha256": provenance["evidence_manifest_sha256"],
        "registration_commit": protocol.get("registration_commit"),
        "physics": physics,
    }
    payload["profile_sha256"] = physics_profile_hash(payload)
    return payload


def finalize_official_physics(protocol: Mapping[str, Any], evidence_store: EvidenceStore) -> BlockedResult | PublicationPlan:
    """Execute the single fail-closed Phase 06F finalization chain."""

    try:
        validation = validate_finalization_protocol(protocol)
    except (FinalizationError, TypeError, ValueError) as exc:
        return _blocked(protocol_hash=getattr(evidence_store, "protocol_sha256", ""), stage="protocol", reason=str(exc))
    protocol_hash = str(evidence_store.protocol_sha256)
    try:
        inherited = dict(evidence_store.verify_inherited(protocol))
    except Exception as exc:
        return _blocked(protocol_hash=protocol_hash, stage="inherited_evidence", reason=str(exc))

    contact_rows = []
    winner_row: Mapping[str, Any] | None = None
    for candidate in protocol["contact_candidates"]:
        candidate_id = str(candidate["candidate_id"])
        try:
            evidence = dict(evidence_store.execute_contact(candidate, protocol))
            eligible, errors = evidence_store.verify_contact(candidate, evidence, protocol)
        except Exception as exc:
            evidence, eligible, errors = {}, False, (f"{type(exc).__name__}: {exc}",)
        evidence_sha256 = artifact_hash(evidence) if evidence else None
        row = {
            "candidate_id": candidate_id,
            "eligible": bool(eligible),
            "errors": [str(error) for error in errors],
            "evidence_sha256": evidence_sha256,
        }
        contact_rows.append(row)
        if winner_row is None and eligible:
            winner_row = candidate

    selection = seal_artifact(
        {
            "schema_id": SCHEMA_ID + ".contact_selection",
            "schema_version": 1,
            "protocol_sha256": protocol_hash,
            "inherited": inherited,
            "priority": validation["candidate_ids"],
            "candidates": contact_rows,
            "contact_winner": None if winner_row is None else winner_row["candidate_id"],
            "selection_rule": "eligibility_then_minimal_intervention_priority",
            "forbidden_inputs_used": [],
        }
    )
    try:
        frozen_selection = dict(evidence_store.freeze_selection(selection))
    except Exception as exc:
        return _blocked(protocol_hash=protocol_hash, stage="selection_freeze", reason=str(exc))
    if frozen_selection.get("payload_sha256") != selection["payload_sha256"]:
        return _blocked(protocol_hash=protocol_hash, stage="selection_freeze", reason="selection artifact changed while freezing")
    if winner_row is None:
        return _blocked(
            protocol_hash=protocol_hash,
            stage="contact",
            reason="physics_gate_failure: zero eligible contact candidates",
            selection=frozen_selection,
        )

    winner = str(winner_row["candidate_id"])
    manifest = materialize_dependent_manifest(
        protocol, winner=winner, selection_artifact_sha256=str(frozen_selection["payload_sha256"])
    )
    manifest["protocol_sha256"] = protocol_hash
    manifest["payload_sha256"] = artifact_hash(manifest)
    try:
        frozen_manifest = dict(evidence_store.freeze_dependent_manifest(manifest))
    except Exception as exc:
        return _blocked(protocol_hash=protocol_hash, stage="dependent_manifest", reason=str(exc), selection=frozen_selection)
    if frozen_manifest.get("payload_sha256") != manifest["payload_sha256"]:
        return _blocked(
            protocol_hash=protocol_hash,
            stage="dependent_manifest",
            reason="dependent manifest changed while freezing",
            selection=frozen_selection,
        )

    dependent = []
    for state in frozen_manifest["states"]:
        try:
            evidence = dict(evidence_store.execute_dependent(state, protocol))
            passed, errors = evidence_store.verify_dependent(state, evidence, protocol)
        except Exception as exc:
            evidence, passed, errors = {}, False, (f"{type(exc).__name__}: {exc}",)
        dependent.append(evidence)
        if not passed:
            return _blocked(
                protocol_hash=protocol_hash,
                stage="dependent_evidence",
                reason="; ".join(f"{state['state_id']}: {error}" for error in errors),
                selection=frozen_selection,
            )

    replay_spec = protocol.get("replay", {})
    if isinstance(replay_spec, Mapping):
        process_count = int(replay_spec.get("process_count", 0))
        replay_groups = set(replay_spec.get("groups", ()))
        for group in replay_groups:
            rows = [row.get("evidence", {}) for row in dependent if row.get("group") == group]
            valid = bool(
                process_count >= 3
                and len(rows) == process_count
                and {int(row.get("process_index", 0)) for row in rows} == set(range(1, process_count + 1))
                and len({row.get("trace_digest") for row in rows}) == 1
                and len({row.get("metric_digest") for row in rows}) == 1
                and all(row.get("complete_trace") is True and row.get("same_process_reset_used") is False for row in rows)
            )
            if not valid:
                return _blocked(
                    protocol_hash=protocol_hash,
                    stage="dependent_evidence",
                    reason=f"three-process replay equality failed: {group}",
                    selection=frozen_selection,
                )

    evidence_manifest_hash = hashlib.sha256(
        canonical_json(
            {
                "inherited": inherited,
                "contacts": contact_rows,
                "dependent": [artifact_hash(row) for row in dependent],
            }
        ).encode("utf-8")
    ).hexdigest()
    profile = _profile(
        protocol,
        winner=winner_row,
        provenance={
            "protocol_sha256": protocol_hash,
            "selection_artifact_sha256": frozen_selection["payload_sha256"],
            "dependent_manifest_sha256": frozen_manifest["payload_sha256"],
            "evidence_manifest_sha256": evidence_manifest_hash,
        },
    )
    status = seal_artifact(
        {
            "schema_id": "shakebench.phase06.final_status",
            "schema_version": 1,
            "status": "PASS",
            "blocking_stage": None,
            "reason": None,
            "protocol_sha256": protocol_hash,
            "contact_winner": winner,
            "selection_artifact_sha256": frozen_selection["payload_sha256"],
            "dependent_manifest_sha256": frozen_manifest["payload_sha256"],
            "official_profile_sha256": profile["profile_sha256"],
            "phase07_authorized": True,
        }
    )
    handoff = seal_artifact(
        {
            "schema_id": "shakebench.phase06.to_phase07_handoff",
            "schema_version": 1,
            "status": "PASS",
            "phase07_authorized": True,
            "protocol": {"path": protocol.get("protocol_filename"), "sha256": protocol_hash},
            "registration_commit": protocol.get("registration_commit"),
            "selected_profile": {"profile_id": profile["profile_id"], "sha256": profile["profile_sha256"]},
            "selection_artifact_sha256": frozen_selection["payload_sha256"],
            "dependent_manifest_sha256": frozen_manifest["payload_sha256"],
            "evidence_manifest_sha256": evidence_manifest_hash,
            "contact_evidence": [
                {
                    "candidate_id": row["candidate_id"],
                    "path": next(
                        candidate["output"]
                        for candidate in protocol["contact_candidates"]
                        if candidate["candidate_id"] == row["candidate_id"]
                    ),
                    "sha256": row["evidence_sha256"],
                }
                for row in contact_rows
            ],
            "dependent_evidence": [
                {
                    "state_id": state["state_id"],
                    "path": state.get("output"),
                    "sha256": artifact_hash(evidence),
                }
                for state, evidence in zip(frozen_manifest["states"], dependent)
            ],
            "package_hashes": {
                **copy.deepcopy(dict(protocol["phase07_handoff"].get("package_hashes", {}))),
                "protocol_asset_sha256": protocol_hash,
                "official_profile_payload_sha256": profile["profile_sha256"],
            },
        }
    )
    return PublicationPlan("PASS", winner, frozen_selection, frozen_manifest, tuple(dependent), profile, status, handoff)


def verify_official_publication_bundle(asset_root: str | Path) -> dict[str, Any]:
    """Verify the packaged status/handoff/profile and every bound evidence hash.

    This deliberately performs no MuJoCo work, making it safe for the official
    loader and for a clean installed wheel. The execution verifier adds an
    independent numeric-gate recomputation on top of this package proof.
    """

    root = Path(asset_root)
    errors: list[str] = []

    def read_json(filename: str) -> dict[str, Any]:
        try:
            value = json.loads((root / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FinalizationError(f"unreadable publication asset {filename}: {exc}") from exc
        if not isinstance(value, dict):
            raise FinalizationError(f"publication asset {filename} is not a mapping")
        return value

    try:
        status = read_json("shakebench_phase_06_final_status.json")
        handoff = read_json("shakebench_phase_06_to_07_handoff.json")
        selection = read_json("shakebench_phase_06f_contact_selection.json")
        manifest = read_json("shakebench_phase_06f_dependent_manifest.json")
        protocol_path = root / "shakebench_phase_06f_protocol.yaml"
        protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        try:
            import yaml

            profile = yaml.safe_load((root / "shakebench_official_physics.yaml").read_text(encoding="utf-8"))
        except ImportError:
            profile = json.loads((root / "shakebench_official_physics.yaml").read_text(encoding="utf-8"))
        if not isinstance(profile, Mapping):
            raise FinalizationError("official profile is not a mapping")
    except (FinalizationError, OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)]}

    for name, value in (("status", status), ("handoff", handoff), ("selection", selection), ("manifest", manifest)):
        if value.get("payload_sha256") != artifact_hash(value):
            errors.append(f"{name} payload hash mismatch")
    if status.get("status") != "PASS" or status.get("phase07_authorized") is not True:
        errors.append("final status does not authorize Phase 07")
    if handoff.get("status") != "PASS" or handoff.get("phase07_authorized") is not True:
        errors.append("handoff does not authorize Phase 07")
    if status.get("protocol_sha256") != protocol_sha256 or handoff.get("protocol", {}).get("sha256") != protocol_sha256:
        errors.append("protocol asset hash mismatch")
    if handoff.get("package_hashes", {}).get("protocol_asset_sha256") != protocol_sha256:
        errors.append("handoff package protocol hash mismatch")

    winner = selection.get("contact_winner")
    selection_hash = selection.get("payload_sha256")
    manifest_hash = manifest.get("payload_sha256")
    if not isinstance(winner, str) or not winner:
        errors.append("contact selection has no winner")
    if manifest.get("contact_winner") != winner or manifest.get("selection_artifact_sha256") != selection_hash:
        errors.append("dependent manifest is not bound to the selected winner/hash")
    if status.get("contact_winner") != winner or status.get("selection_artifact_sha256") != selection_hash:
        errors.append("final status selection binding mismatch")
    if status.get("dependent_manifest_sha256") != manifest_hash or handoff.get("dependent_manifest_sha256") != manifest_hash:
        errors.append("dependent manifest hash binding mismatch")

    contact_records = []
    for reference in handoff.get("contact_evidence", ()):
        try:
            record = read_json(str(reference["path"]))
        except (FinalizationError, KeyError) as exc:
            errors.append(str(exc))
            continue
        digest = artifact_hash(record)
        if digest != reference.get("sha256") or record.get("payload_sha256") != digest:
            errors.append("contact evidence hash mismatch: " + str(reference.get("path")))
        if record.get("candidate_id") != reference.get("candidate_id"):
            errors.append("contact evidence candidate binding mismatch")
        contact_records.append(record)

    dependent_records = []
    states = {str(row.get("state_id")): row for row in manifest.get("states", ()) if isinstance(row, Mapping)}
    for reference in handoff.get("dependent_evidence", ()):
        try:
            record = read_json(str(reference["path"]))
        except (FinalizationError, KeyError) as exc:
            errors.append(str(exc))
            continue
        digest = artifact_hash(record)
        state = states.get(str(reference.get("state_id")))
        if digest != reference.get("sha256") or record.get("payload_sha256") != digest:
            errors.append("dependent evidence hash mismatch: " + str(reference.get("path")))
        if state is None or any(record.get(key) != state.get(key) for key in ("state_id", "contact_candidate_id", "selection_artifact_sha256", "group")):
            errors.append("dependent evidence manifest binding mismatch")
        if record.get("contact_candidate_id") != winner:
            errors.append("dependent evidence does not use the actual winner")
        dependent_records.append(record)
    if len(contact_records) != len(selection.get("candidates", ())) or len(dependent_records) != len(states):
        errors.append("publication evidence coverage is incomplete")

    replay_groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in dependent_records:
        group = str(record.get("group"))
        if group in {"driver", "isolator", "contact", "gamma_zero_parity_replay"}:
            replay_groups.setdefault(group, []).append(record.get("evidence", {}))
    for group in ("driver", "isolator", "contact", "gamma_zero_parity_replay"):
        rows = replay_groups.get(group, [])
        if not (
            len(rows) == 3
            and {int(row.get("process_index", 0)) for row in rows} == {1, 2, 3}
            and len({row.get("trace_digest") for row in rows}) == 1
            and len({row.get("metric_digest") for row in rows}) == 1
            and all(row.get("contact_candidate_id") == winner and row.get("complete_trace") is True for row in rows)
        ):
            errors.append(f"three-process replay bundle mismatch: {group}")

    evidence_manifest_hash = hashlib.sha256(
        canonical_json(
            {
                "inherited": selection.get("inherited"),
                "contacts": selection.get("candidates"),
                "dependent": [artifact_hash(row) for row in dependent_records],
            }
        ).encode("utf-8")
    ).hexdigest()
    if handoff.get("evidence_manifest_sha256") != evidence_manifest_hash:
        errors.append("handoff evidence-manifest hash mismatch")
    if profile.get("evidence_manifest_sha256") != evidence_manifest_hash:
        errors.append("official profile evidence-manifest hash mismatch")
    computed_profile_hash = physics_profile_hash(profile)
    if profile.get("profile_sha256") != computed_profile_hash:
        errors.append("official profile embedded hash mismatch")
    if profile.get("profile_id") != "shakebench.official.physics.v2" or profile.get("status") != "official_immutable" or profile.get("scoreable") is not True:
        errors.append("official profile identity/status mismatch")
    if profile.get("physics", {}).get("contact", {}).get("candidate_id") != winner:
        errors.append("official profile contact differs from winner")
    if handoff.get("selected_profile", {}).get("sha256") != computed_profile_hash or status.get("official_profile_sha256") != computed_profile_hash:
        errors.append("profile/status/handoff hash mismatch")
    if handoff.get("package_hashes", {}).get("official_profile_payload_sha256") != computed_profile_hash:
        errors.append("handoff package profile hash mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "contact_winner": winner,
        "profile_sha256": computed_profile_hash,
        "protocol_sha256": protocol_sha256,
        "evidence_manifest_sha256": evidence_manifest_hash,
    }


__all__ = [
    "BINDING_RULE",
    "BlockedResult",
    "EvidenceStore",
    "FinalizationError",
    "PublicationPlan",
    "artifact_hash",
    "finalize_official_physics",
    "materialize_dependent_manifest",
    "seal_artifact",
    "validate_finalization_protocol",
    "verify_official_publication_bundle",
]
