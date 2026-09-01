"""State-isolated Phase 06 archive and fail-closed tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robosuite.models import assets_root
import robosuite.utils.shakebench_physics as physics
from robosuite.utils.shakebench_driver_measurement import DriverMeasurementPlan
from robosuite.utils.shakebench_physics import (
    OFFICIAL_PHYSICS_PROFILE_FILENAME,
    PhysicsProfileIntegrityError,
    load_official_physics_profile,
    make_probe_physics_profile,
    physics_profile_hash,
)


ASSETS = Path(assets_root)


def _temporary_official_fixture(tmp_path: Path) -> Path:
    """Create an isolated PASS package fixture; never mutate checkout state."""

    protocol = tmp_path / "shakebench_selection_protocol_v5.yaml"
    protocol.write_text("schema_id: test.v4\nstatus: pre_registered\n", encoding="utf-8")
    payload = make_probe_physics_profile().to_dict()
    payload.update(
        {
            "profile_id": "shakebench.official.physics.v1",
            "status": "official_immutable",
            "scoreable": True,
            "protocol_file": protocol.name,
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "freeze_commit": "test-fixture",
        }
    )
    payload["profile_sha256"] = physics_profile_hash(payload)
    (tmp_path / OFFICIAL_PHYSICS_PROFILE_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )
    (tmp_path / "shakebench_phase_06r4_v5_status.json").write_text(
        json.dumps(
            {
                "schema_id": "test.status",
                "schema_version": 1,
                "status": "PASS",
                "protocol": {"path": protocol.name},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_packaged_baseline_is_blocked_before_v5_publication():
    with pytest.raises(PhysicsProfileIntegrityError, match="Phase 06R4 V5 selection"):
        load_official_physics_profile()


def test_official_loader_accepts_only_an_isolated_packaged_pass_fixture(tmp_path, monkeypatch):
    fixture = _temporary_official_fixture(tmp_path)
    monkeypatch.setattr(physics.models, "assets_root", str(fixture))
    profile = load_official_physics_profile()
    assert profile.scoreable is True
    assert profile.status == "official_immutable"
    assert profile.profile_sha256 == physics_profile_hash(profile.to_dict())


def test_external_scoreable_profile_path_is_rejected(tmp_path):
    fixture = _temporary_official_fixture(tmp_path)
    with pytest.raises(PhysicsProfileIntegrityError, match="packaged canonical asset"):
        load_official_physics_profile(fixture / OFFICIAL_PHYSICS_PROFILE_FILENAME)


def test_probe_profile_remains_explicitly_non_scoreable():
    profile = make_probe_physics_profile()
    assert profile.scoreable is False
    assert profile.status == "probe_non_scoreable"


def test_v3_protocol_and_raw_evidence_are_unchanged_and_blocked():
    protocol = ASSETS / "shakebench_selection_protocol_v3.yaml"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        "8175160da40e66d39b85f2f3dccf75338d713b309b7612a968f053fbe808fbee"
    )
    status = json.loads((ASSETS / "shakebench_phase_06r2_status.json").read_text(encoding="utf-8"))
    raw = json.loads((ASSETS / "shakebench_phase_06r2_raw_driver_dt_medium.json").read_text(encoding="utf-8"))
    assert status["status"] == "BLOCKED"
    assert status["failure_taxonomy"] == "invalid_protocol_configuration"
    assert raw["record_count"] == 1
    assert raw["records"][0]["retry_ledger"][0]["exception"].startswith("SimulationError:")


def test_v2_protocol_bytes_and_v1_v2_v3_statuses_are_preserved():
    v2 = ASSETS / "shakebench_selection_protocol_v2.yaml"
    assert hashlib.sha256(v2.read_bytes()).hexdigest() == (
        "4d15e7a20ce242edad6a41ed4bf74e2a00f4e04276accbc6362dce73ef5b875c"
    )
    for name in (
        "shakebench_phase_06_v1_invalidated.json",
        "shakebench_phase_06r_status.json",
        "shakebench_phase_06r2_status.json",
    ):
        assert json.loads((ASSETS / name).read_text(encoding="utf-8"))["status"] in {
            "invalidated_phase06r",
            "BLOCKED",
        }


def test_bounded_measurement_contract_and_capacity_evidence():
    plan = DriverMeasurementPlan(physics_timestep_s=0.0001, cadence_hz=200.0)
    assert plan.refresh_stride == 50
    assert plan.sample_dt_s == pytest.approx(0.005, abs=1e-15)
    artifact = json.loads(
        (ASSETS / "shakebench_phase_06r2_capacity_preflight.json").read_text(encoding="utf-8")
    )
    assert artifact["passed"] is True
    assert artifact["selection_input"] is False
    assert artifact["mujoco_step_count"] == 347200
    assert artifact["peak_retained_sample_count"] == 6950


def test_v3_dt_medium_scheduler_rejection_is_not_infrastructure_failure():
    status = json.loads((ASSETS / "shakebench_phase_06r2_status.json").read_text(encoding="utf-8"))
    assert status["failure_taxonomy"] == "invalid_protocol_configuration"
    assert status["driver_gate"]["dt_medium_disposition"] == "preflight_rejected_before_MuJoCo_integration"


def test_v4_status_is_blocked_and_first_v4_raw_artifact_is_preserved():
    status = json.loads((ASSETS / "shakebench_phase_06r3_v4_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "BLOCKED"
    assert status["failure_taxonomy"] == "invalid_protocol_configuration"
    assert (ASSETS / "shakebench_selection_protocol_v4.yaml").exists()
    raw = ASSETS / "shakebench_phase_06r3_v4_raw_driver_dt_fine_gamma_0_15_empty.json"
    assert raw.is_file()
    assert json.loads(raw.read_text(encoding="utf-8"))["state_id"] == "driver.dt_fine.gamma_0_15.empty"


def test_v5_driver_matrix_is_preserved_but_active_status_is_blocked():
    status = json.loads((ASSETS / "shakebench_phase_06r4_v5_status.json").read_text(encoding="utf-8"))
    files = sorted(ASSETS.glob("shakebench_phase_06r4_v5_raw_driver_*.json"))
    assert status["status"] == "BLOCKED"
    assert status["failure_taxonomy"] == "invalid_protocol_configuration"
    assert len(files) == 18
    assert all(json.loads(path.read_text(encoding="utf-8"))["evidence"]["passed"] is True for path in files)
    assert not list(ASSETS.glob("shakebench_phase_06r4_v5_raw_isolator_*.json"))


def test_profile_assets_remain_flat_and_packaged():
    assert (ASSETS / OFFICIAL_PHYSICS_PROFILE_FILENAME).is_file()
    assert not (ASSETS / "shakebench").exists()
    manifest = Path(__file__).parents[1] / "MANIFEST.in"
    assert "recursive-include robosuite/models/assets/ *" in manifest.read_text(encoding="utf-8")
