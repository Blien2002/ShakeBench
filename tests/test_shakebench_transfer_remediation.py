"""Regression tests for the Phase 03R transfer-remediation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robosuite.utils.shakebench_isolator import (
    PHASE03_TRANSFER_SCHEMA_ID,
    PHASE03_TRANSFER_SCHEMA_VERSION,
    compare_harmonic_fit,
    fit_harmonic_transfer,
    verify_phase03_transfer_artifact,
)


ARTIFACT = Path(__file__).with_name("shakebench_phase_03_transfer.json")


def test_complex_harmonic_fit_preserves_amplitude_and_phase_convention():
    time = np.linspace(0.0, 8.0, 8001)
    frequency_hz = 2.5
    input_signal = 0.4 * np.sin(2.0 * np.pi * frequency_hz * time + 0.35) + 0.07
    output_signal = 0.9 * np.sin(2.0 * np.pi * frequency_hz * time - 0.20) - 0.11
    fit = fit_harmonic_transfer(time, input_signal, output_signal, frequency_hz)
    assert fit.amplitude_ratio == pytest.approx(0.9 / 0.4, rel=0.0, abs=1e-12)
    assert fit.phase_difference_rad == pytest.approx(-0.55, rel=0.0, abs=1e-12)
    assert fit.normalized_fit_residual < 1e-10
    comparison = compare_harmonic_fit(fit, fit.transfer_complex)
    assert comparison["amplitude_gate_passed"]
    assert comparison["phase_gate_passed"]
    assert comparison["residual_gate_passed"]
    assert comparison["passed"]
    assert fit.to_dict()["input_complex_coefficient"] != fit.to_dict()["input_amplitude"]


def test_phase03_transfer_artifact_is_read_only_verified_and_handed_off():
    before = ARTIFACT.read_bytes()
    summary = verify_phase03_transfer_artifact(ARTIFACT)
    assert summary["passed"]
    assert summary["phase04_handoff"] == "PASS"
    assert all(summary["checks"].values())
    assert ARTIFACT.read_bytes() == before

    payload = json.loads(before.decode("utf-8"))
    assert payload["schema_id"] == PHASE03_TRANSFER_SCHEMA_ID
    assert payload["schema_version"] == PHASE03_TRANSFER_SCHEMA_VERSION
    assert payload["harmonic_grid"]["passed"]
    assert len(payload["harmonic_grid"]["records"]) == 18
    assert payload["joint_spectrum"]["gate_summary"]["passed"]
    assert payload["joint_spectrum"]["line_fit_count"] == 64
    assert payload["payload_sensitivity"]["passed"]
    assert len(payload["payload_sensitivity"]["records"]) == 4
    assert payload["provenance"]["right_limit_time_contract"].startswith("write q(t)")
    assert payload["joint_spectrum"]["line_fits"][0]["relative_fit"]["transfer_complex"]
    assert payload["joint_spectrum"]["line_fits"][0]["relative_comparison"]["phase_gate_passed"]


def test_phase03_transfer_artifact_mutation_fails_verification(tmp_path):
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["thresholds"]["phase_absolute_error_deg_max"] = 999.0
    mutated = tmp_path / "mutated_phase03_transfer.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    summary = verify_phase03_transfer_artifact(mutated)
    assert not summary["passed"]
    assert not summary["checks"]["integrity"]
