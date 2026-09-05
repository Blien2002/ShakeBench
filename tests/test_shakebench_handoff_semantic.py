"""Phase 06F-R2 semantic verifier and adversarial trace tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from robosuite.models import assets_root
from robosuite.scripts.shakebench_handoff_remediation import load_protocol as load_r1_protocol
from robosuite.utils.shakebench_handoff_semantic import (
    recompute_convergence,
    verify_gamma_zero_parity,
    verify_phase06fr2_handoff,
)


ASSETS = Path(assets_root)


def _protocol():
    import yaml

    return yaml.safe_load((ASSETS / "shakebench_phase_06fr2_protocol.yaml").read_text(encoding="utf-8"))


def test_r2_handoff_semantic_bundle_passes_after_anchor_generation():
    result = verify_phase06fr2_handoff(ASSETS)
    assert result.passed, result.errors


def test_gamma_zero_parity_is_recomputed_from_raw_trace_values():
    protocol = _protocol()
    parity = json.loads((ASSETS / "shakebench_phase_06fr_raw_parity_gamma_zero.json").read_text(encoding="utf-8"))
    assert verify_gamma_zero_parity(parity["trace"], parity["parity"], protocol)["passed"]
    mutated = copy.deepcopy(parity["trace"])
    mutated[100]["decoded_action"][0] = 1.0
    assert verify_gamma_zero_parity(mutated, parity["parity"], protocol)["passed"] is False
    shifted = copy.deepcopy(parity["parity"])
    shifted["worktable_pose"][2] += 0.1
    assert verify_gamma_zero_parity(parity["trace"], shifted, protocol)["passed"] is False
    thresholds = copy.deepcopy(parity["parity"])
    thresholds["success_thresholds"]["max_relative_linear_speed_m_s"] = 99.0
    assert verify_gamma_zero_parity(parity["trace"], thresholds, protocol)["passed"] is False


def test_convergence_recomputes_cumulative_impulse_not_one_step_force():
    protocol = _protocol()
    records = {}
    for support_token, support in (("open", "open_worktable"), ("target", "target_bottom")):
        for token, dt in (("dt_fine", 0.0001), ("dt_medium", 0.000125), ("dt_nominal", 0.0002)):
            records[(support, dt)] = json.loads((ASSETS / f"shakebench_phase_06fr_raw_real_{support_token}_{token}.json").read_text())
    result = recompute_convergence(protocol, records)
    assert result["passed"]
    mutated = copy.deepcopy(records)
    mutated[("open_worktable", 0.0001)]["trace"][2499]["normal_impulse_Ns"] += 1.0
    assert recompute_convergence(protocol, mutated)["passed"] is False
