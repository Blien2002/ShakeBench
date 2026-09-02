"""Phase 06R5/V6 resolved-state and adapter-contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robosuite.models import assets_root
from robosuite.scripts.shakebench_select_physics_v6 import (
    PROTOCOL_FILENAME,
    RAW_PREFIX,
    _select_contact_or_none,
    adapter_contract,
    dry_run_manifest,
    load_v6_protocol,
    validate_v6_protocol,
    verify_selection_artifact,
)
from robosuite.utils.shakebench_phase06_adapters import (
    NoPhysicsBackend,
    prepare_contact_probe,
    prepare_driver_probe,
    prepare_isolator_probe,
    prepare_parity_probe,
    prepare_replay_probe,
)
from robosuite.utils.shakebench_protocol_v6 import (
    V6ProtocolStateError,
    resolve_all_protocol_states_v6,
)


ASSETS = Path(assets_root)
PROTOCOL = ASSETS / PROTOCOL_FILENAME


def _protocol() -> dict:
    value, _, _ = load_v6_protocol(PROTOCOL)
    return copy.deepcopy(dict(value))


def _write(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def test_v6_protocol_resolves_all_stages_and_derived_parameters():
    protocol = _protocol()
    structure = validate_v6_protocol(protocol)
    states = resolve_all_protocol_states_v6(protocol)
    assert structure["state_count"] == 37
    assert structure["stage_counts"] == {"driver": 18, "isolator": 3, "contact": 3, "parity": 1, "replay": 12}
    assert len(states) == 37
    assert {state.stage for state in states} == {"driver", "isolator", "contact", "parity", "replay"}
    isolator = next(state for state in states if state.stage == "isolator" and state.candidate_id == "low_frequency_damped")
    assert isolator.isolator is not None
    assert isolator.isolator.fn_hz == (4.0, 4.0, 4.0, 3.0, 3.0, 2.0)
    assert isolator.isolator.k[2] == pytest.approx(20212.949813431005, abs=1e-12)
    assert isolator.isolator.springref[2] == pytest.approx(0.015530637680177088, abs=1e-15)
    contact = next(state for state in states if state.stage == "contact")
    assert contact.isolator is not None
    assert contact.contact is not None
    assert contact.contact.condim == 3


def test_v6_resolved_arrays_and_mappings_are_immutable():
    state = next(state for state in resolve_all_protocol_states_v6(_protocol()) if state.stage == "driver")
    assert isinstance(state.driver.deck["inertia_kg_m2"], tuple)
    with pytest.raises(TypeError):
        state.driver.program["seed"] = 99
    with pytest.raises(TypeError):
        state.common.protocol_identity["schema_id"] = "mutated"


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (("driver", "convergence_candidates", 0), "sample_rate_hz"),
        (("driver", "convergence_candidates", 0), "refresh_stride"),
        (("components", "isolator_candidates", 0), "fn_hz"),
        (("components", "isolator_candidates", 0), "zeta"),
        (("components", "contact_candidates", 0), "condim"),
        (("components", "contact_candidates", 0), "solref"),
        (("components", "contact_candidates", 0), "solimp"),
        (("parity",), "geometry_tolerance_m"),
        (("execution_manifest", 25), "binding"),
    ],
)
def test_v6_missing_contract_fields_fail_before_any_adapter_or_physics_call(path, key):
    protocol = _protocol()
    if path[0] == "execution_manifest":
        target = protocol["execution_manifest"][path[1]]
    elif len(path) == 1:
        target = protocol[path[0]]
    else:
        target = protocol
        for part in path[:-1]:
            target = target[part]
        if isinstance(target, list):
            target = target[path[-1]]
    target.pop(key, None)
    with pytest.raises((V6ProtocolStateError, ValueError), match="missing|required|binding|candidate|geometry"):
        validate_v6_protocol(protocol)


def test_v6_adapter_contract_uses_no_physics_backend_and_subprocess(tmp_path):
    protocol = _protocol()
    result = adapter_contract(protocol)
    assert result["mujoco_model_created"] is False
    assert result["backend"]["physics_calls"] == 0
    assert result["legal_replay_binding_count"] >= 3
    path = tmp_path / "protocol.yaml"
    _write(path, protocol)
    completed = subprocess.run(
        [sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v6", "--protocol", str(path), "--adapter-contract"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["backend"]["physics_calls"] == 0


def test_each_adapter_accepts_resolved_state_without_protocol_mapping():
    states = resolve_all_protocol_states_v6(_protocol())
    backend = NoPhysicsBackend()
    for state in states:
        if state.stage == "driver":
            prepare_driver_probe(state, backend)
        elif state.stage == "isolator":
            prepare_isolator_probe(state, backend)
        elif state.stage == "contact":
            prepare_contact_probe(state, backend)
        elif state.stage == "parity":
            prepare_parity_probe(state, backend)
        else:
            prepare_replay_probe(state, backend)
    assert len(backend.preparation_events) == len(states)


def test_dry_run_digest_is_stable_and_v6_does_not_consume_v5_raw_files():
    protocol = _protocol()
    first = dry_run_manifest(protocol)
    second = dry_run_manifest(protocol)
    assert first["resolved_state_digest"] == second["resolved_state_digest"]
    assert all(path.name.startswith(RAW_PREFIX) for path in [Path(item["output"]) for item in first["states"] for item in [item["common"]]])
    v5 = sorted(ASSETS.glob("shakebench_phase_06r4_v5_raw_driver_*.json"))
    assert len(v5) == 18


def test_v6_verifier_rejects_missing_matrix_external_profile_and_raw_mutation(tmp_path):
    protocol = _protocol()
    selected = {
        "schema_id": "shakebench.phase06r5.v6.physics_selection",
        "schema_version": 6,
        "status": "PASS",
        "protocol": {"bytes_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "normalized_sha256": "wrong"},
        "resolved_state_digest": "wrong",
        "raw_files": [],
        "official_profile": "/tmp/external.yaml",
        "official_profile_alignment": True,
    }
    selected["payload_sha256"] = "not-the-hash"
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(selected), encoding="utf-8")
    result = verify_selection_artifact(path, protocol_path=PROTOCOL)
    assert result["passed"] is False
    assert any("driver" in error or "raw" in error or "external" in error for error in result["errors"])


def test_v6_zero_eligible_contact_never_becomes_a_selected_candidate():
    historical = json.loads((ASSETS / "shakebench_phase_06r5_v6_selected_candidates.json").read_text(encoding="utf-8"))
    assert historical["status"] == "BLOCKED"
    assert historical["eligible_counts"]["contact"] == 0
    assert historical["selection"]["selected_candidate_ids"]["contact"] == "c3_nominal"
    assert _select_contact_or_none([], []) is None
    assert _select_contact_or_none([], ["c3_nominal"]) is None
    assert _select_contact_or_none(["c3_nominal"], ["c3_nominal"]) == "c3_nominal"
