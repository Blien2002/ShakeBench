"""V5 resolver contract tests; all execution uses temporary protocol files."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robosuite.scripts.shakebench_select_physics_v5 import (
    RAW_PREFIX,
    V5ProtocolError,
    dry_run_manifest,
    validate_v5_protocol,
)
from robosuite.utils.shakebench_protocol import ProtocolStateError, resolve_all_protocol_states, resolve_protocol_state


def _v5_protocol() -> dict:
    drivers = [
        {"candidate_id": "dt_fine", "physics_timestep_s": 0.0001, "refresh_stride": 50, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_mass_kg": 400.0, "deck_inertia_kg_m2": [12.12, 14.083333333333334, 26.033333333333332], "deck_eq_solref": [0.0002, 0.5], "deck_eq_solimp": [0.9, 0.95, 0.001, 0.5, 2.0]},
        {"candidate_id": "dt_medium", "physics_timestep_s": 0.000125, "refresh_stride": 40, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_mass_kg": 400.0, "deck_inertia_kg_m2": [12.12, 14.083333333333334, 26.033333333333332], "deck_eq_solref": [0.00025, 0.5], "deck_eq_solimp": [0.9, 0.95, 0.001, 0.5, 2.0]},
        {"candidate_id": "dt_nominal", "physics_timestep_s": 0.0002, "refresh_stride": 25, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_mass_kg": 400.0, "deck_inertia_kg_m2": [12.12, 14.083333333333334, 26.033333333333332], "deck_eq_solref": [0.0004, 0.5], "deck_eq_solimp": [0.9, 0.95, 0.001, 0.5, 2.0]},
    ]
    manifest = []
    for driver in drivers:
        for gamma in (0.15, 0.30, 0.50):
            for load in ("empty", "panda_plus_worktable_reference_proxy"):
                state_id = f"driver.{driver['candidate_id']}.{str(gamma).replace('.', '_')}.{load}"
                manifest.append({"state_id": state_id, "stage": "driver", "candidate_id": driver["candidate_id"], "gamma": gamma, "load_case": load, "duration_s": 34.71991810092712, "output": RAW_PREFIX + state_id.replace('.', '_') + ".json", "retry_policy": "same_state_config_once_then_block"})
    for stage, candidate in (("isolator", "balanced_nominal"), ("contact", "c3_nominal"), ("parity", "gamma_zero")):
        state_id = f"{stage}.{candidate}"
        manifest.append({"state_id": state_id, "stage": stage, "candidate_id": candidate, "duration_s": 1.0, "output": RAW_PREFIX + state_id.replace('.', '_') + ".json", "retry_policy": "same_state_config_once_then_block"})
    for group in ("driver", "isolator", "contact", "gamma_zero_parity"):
        for process in (1, 2, 3):
            state_id = f"replay.{group}.{process}"
            manifest.append({"state_id": state_id, "stage": "replay", "candidate_id": group, "duration_s": 1.0, "output": RAW_PREFIX + state_id.replace('.', '_') + ".json", "retry_policy": "same_state_config_once_then_block"})
    return {
        "schema_id": "shakebench.phase06r4.v5.physics_selection_protocol",
        "schema_version": 5,
        "status": "pre_registered",
        "immutable_after_registration": True,
        "manifest_state_count": len(manifest),
        "frozen_facts": {"control_frequency_hz": 20.0, "f_max_hz": 8.87, "safe_gamma_candidates": [0.15, 0.30, 0.50], "load_cases": ["empty", "panda_plus_worktable_reference_proxy"]},
        "driver": {"convergence_candidates": drivers},
        "components": {"isolator_candidates": [{"candidate_id": "balanced_nominal", "fn_hz": [5, 5, 5, 4, 4, 2.5], "zeta": [0.1] * 6}], "contact_candidates": [{"candidate_id": "c3_nominal"}]},
        "runtime_defaults": {"physics_timestep_s": 0.0002, "refresh_stride": 25, "sample_dt_s": 0.005, "sample_rate_hz": 200.0, "integrator": "Euler", "solver": "Newton", "iterations": 100, "tolerance": 1e-12, "deck_mass_kg": 400.0, "deck_inertia_kg_m2": [12.12, 14.083333333333334, 26.033333333333332]},
        "measurement": {"trace_schema_id": "shakebench.deck_driver.trace", "trace_schema_version": 3},
        "execution_manifest": manifest,
    }


def _write_protocol(path: Path, protocol: dict) -> None:
    path.write_text(yaml.safe_dump(protocol, sort_keys=True), encoding="utf-8")


def test_resolver_materializes_all_driver_runtime_fields_and_is_deterministic():
    protocol = _v5_protocol()
    states = resolve_all_protocol_states(protocol)
    assert len(states) == 33
    drivers = [state for state in states if state.stage == "driver"]
    assert {state.candidate_id for state in drivers} == {"dt_fine", "dt_medium", "dt_nominal"}
    assert {(state.physics_timestep_s, state.control_steps, state.refresh_stride, state.measurement_rate_hz) for state in drivers} == {(0.0001, 500, 50, 200.0), (0.000125, 400, 40, 200.0), (0.0002, 250, 25, 200.0)}
    assert all(state.output.startswith(RAW_PREFIX) for state in states)
    assert resolve_protocol_state(protocol, drivers[0].state_id).to_dict() == drivers[0].to_dict()


def test_resolver_rejects_manifest_owned_measurement_override_and_v4_shape():
    protocol = _v5_protocol()
    protocol["execution_manifest"][0]["sample_rate_hz"] = 200.0
    with pytest.raises(ProtocolStateError, match="candidate-owned"):
        resolve_all_protocol_states(protocol)
    protocol = _v5_protocol()
    protocol["execution_manifest"][0]["refresh_stride"] = 50
    with pytest.raises(ProtocolStateError, match="candidate-owned"):
        resolve_all_protocol_states(protocol)


def test_v5_validator_and_cli_dry_run_use_same_resolved_state_count(tmp_path):
    protocol = _v5_protocol()
    path = tmp_path / "protocol.yaml"
    _write_protocol(path, protocol)
    result = validate_v5_protocol(protocol)
    assert result["state_count"] == 33
    dry = dry_run_manifest(protocol)
    assert dry["state_count"] == result["state_count"]
    completed = subprocess.run([sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v5", "--protocol", str(path), "--dry-run-manifest"], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["state_count"] == 33


def test_v5_cli_rejects_v4_shaped_manifest_before_probe(tmp_path):
    protocol = _v5_protocol()
    protocol["execution_manifest"][0]["sample_rate_hz"] = 200.0
    path = tmp_path / "v4-shaped.yaml"
    _write_protocol(path, protocol)
    completed = subprocess.run([sys.executable, "-m", "robosuite.scripts.shakebench_select_physics_v5", "--protocol", str(path), "--validate-protocol"], capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "candidate-owned" in completed.stdout or "candidate-owned" in completed.stderr


def test_v5_resolver_has_no_official_profile_dependency():
    source = Path("robosuite/utils/shakebench_protocol.py").read_text(encoding="utf-8")
    assert "load_official_physics_profile" not in source
    assert "resolve_all_protocol_states" in source
