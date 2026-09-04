"""Execute and verify the fail-closed Phase 06F physics publication."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from robosuite import models
from robosuite.scripts.shakebench_diagnose_normal_impact_v8 import (
    run_incline_calibration,
    run_level_control,
    run_normal_impact_trace,
)
from robosuite.utils.shakebench_artifacts import file_sha256, write_json_atomic
from robosuite.utils.shakebench_contact_force import (
    compiled_pair_audit,
    force_threshold_gates,
    run_horizontal_force_threshold,
    run_single_axis_slip,
    run_task_native_settling,
    run_timestep_convergence,
)
from robosuite.utils.shakebench_physics import PhysicsProfile, physics_profile_hash
from robosuite.utils.shakebench_physics_finalizer import (
    BlockedResult,
    PublicationPlan,
    artifact_hash,
    finalize_official_physics,
    seal_artifact,
    validate_finalization_protocol,
    verify_official_publication_bundle,
)


PROTOCOL_FILENAME = "shakebench_phase_06f_protocol.yaml"
FEASIBILITY_FILENAME = "shakebench_phase_06f_feasibility.json"
SELECTION_FILENAME = "shakebench_phase_06f_contact_selection.json"
DEPENDENT_MANIFEST_FILENAME = "shakebench_phase_06f_dependent_manifest.json"
FINAL_STATUS_FILENAME = "shakebench_phase_06_final_status.json"
HANDOFF_FILENAME = "shakebench_phase_06_to_07_handoff.json"
OFFICIAL_PROFILE_FILENAME = "shakebench_official_physics.yaml"
REPORT_FILENAME = "phase_06f_final_physics_report.md"
REPLAY_GROUPS = {"driver", "isolator", "contact", "gamma_zero_parity_replay"}


class Phase06FError(RuntimeError):
    """Raised when final evidence or publication is inconsistent."""


def _asset(filename: str) -> Path:
    return Path(models.assets_root) / filename


def load_protocol(path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    source = _asset(PROTOCOL_FILENAME) if path is None else Path(path)
    raw = source.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise Phase06FError("Phase 06F protocol root must be a mapping")
    return value, source, hashlib.sha256(raw).hexdigest()


def validate_registered_sources(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate inherited evidence and exact V6/V8 candidate tuples."""

    errors = []
    authenticated = []
    inherited = protocol.get("inherited_evidence", {})
    for label in ("protocol", "selected", "status", "feasibility"):
        reference = inherited.get(label, {})
        path = _asset(str(reference.get("path", "")))
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            errors.append(f"inherited {label} hash mismatch")
        else:
            authenticated.append({"kind": "inherited_" + label, "path": path.name, "sha256": file_sha256(path)})
    v6 = yaml.safe_load(_asset("shakebench_selection_protocol_v6.yaml").read_text(encoding="utf-8"))
    v8 = yaml.safe_load(_asset("shakebench_selection_protocol_v8.yaml").read_text(encoding="utf-8"))
    sources = {}
    for row in v6["components"]["contact_candidates"]:
        if row["candidate_id"] == "c3_nominal":
            sources[row["candidate_id"]] = row
    for row in v8["components"]["contact_candidates"]:
        if row["candidate_id"] in {"n6_critical_negative_control", "n6_overdamped_2", "n6_overdamped_4"}:
            sources[row["candidate_id"]] = row
    tuple_fields = (
        "condim",
        "sliding_mu",
        "torsional_mu",
        "rolling_mu",
        "margin_m",
        "gap_m",
        "solref",
        "solimp",
        "iterations",
        "interfaces",
    )
    for candidate in protocol["contact_candidates"]:
        candidate_id = str(candidate["candidate_id"])
        source = sources.get(candidate_id)
        if source is None or any(candidate.get(field) != source.get(field) for field in tuple_fields):
            errors.append(f"candidate tuple differs from authenticated source: {candidate_id}")
        raw_ref = candidate.get("source", {})
        raw_path = _asset(str(raw_ref.get("path", "")))
        if not raw_path.is_file() or file_sha256(raw_path) != raw_ref.get("sha256"):
            errors.append(f"candidate source raw hash mismatch: {candidate_id}")
        else:
            authenticated.append({"kind": "candidate_source", "candidate_id": candidate_id, "path": raw_path.name, "sha256": file_sha256(raw_path)})
    if errors:
        raise Phase06FError("; ".join(errors))
    return {"authenticated": authenticated, "candidate_tuple_count": len(sources)}


def _candidate_profile(candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> PhysicsProfile:
    physics = copy.deepcopy(dict(protocol["official_profile"]["physics_template"]))
    physics["contact"] = {
        key: copy.deepcopy(candidate[key])
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
    payload = {
        "schema_id": "shakebench.official.physics",
        "schema_version": 1,
        "profile_id": "shakebench.phase06f.probe." + str(candidate["candidate_id"]),
        "status": "probe_non_scoreable",
        "scoreable": False,
        "physics": physics,
    }
    payload["profile_sha256"] = physics_profile_hash(payload)
    return PhysicsProfile(payload=payload, source="Phase 06F candidate", profile_sha256=payload["profile_sha256"]).assert_valid()


def _finger_and_compiled_evidence(candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan
    from robosuite.scripts.shakebench_diagnose_contact_recovery_v7 import (
        _declared_finger_load_trace,
        _force_envelope_audit,
    )

    profile = _candidate_profile(candidate, protocol)
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=float(profile.model_timestep_s),
        target_container_friction=(
            0.30,
            float(candidate["torsional_mu"]),
            float(candidate["rolling_mu"]),
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=20,
        seed=17,
    )
    try:
        compiled = env.audit_compiled_model()
        envelope = _force_envelope_audit(
            env.sim.model._model,
            finger_actuator_names=(
                "gripper0_right_gripper_finger_joint1",
                "gripper0_right_gripper_finger_joint2",
            ),
        )
        env.reset()
        finger = _declared_finger_load_trace(
            env,
            duration_s=float(protocol["contact"]["finger_load_duration_s"]),
            envelope=envelope,
        )
        warnings = int(np.sum(env.sim.data.warning.number))
    finally:
        env.close()
    return {"compiled_task_model": compiled, "finger_force_envelope": envelope, "finger_load": finger, "warning_count": warnings}


def execute_contact_candidate(candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    contact = protocol["contact"]
    force = contact["force_threshold"]
    evidence = {
        "schema_id": "shakebench.phase06f.contact_evidence",
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "compiled_explicit_pairs": {
            "open_worktable": compiled_pair_audit(candidate, support="open_worktable"),
            "target_bottom": compiled_pair_audit(candidate, support="target_bottom"),
        },
        "task_native_settling": {
            "open_worktable": run_task_native_settling(candidate, support="open_worktable"),
            "target_bottom": run_task_native_settling(candidate, support="target_bottom"),
        },
        "level_force_threshold": run_horizontal_force_threshold(
            candidate,
            force_factors=force["force_factors"],
            reference_mu=float(force["reference_mu"]),
            mass_kg=float(force["mass_kg"]),
            gravity_m_s2=float(force["gravity_m_s2"]),
            plateau_duration_s=float(force["plateau_duration_s"]),
            static_displacement_max_m=float(force["static_displacement_max_m"]),
            monotonic_tolerance_m=float(force["monotonic_tolerance_m"]),
            bracket_relative_tolerance=float(force["bracket_relative_tolerance"]),
        ),
        "single_axis_slip": run_single_axis_slip(
            candidate,
            initial_speed_m_s=float(contact["slip_initial_speed_m_s"]),
            duration_s=float(contact["slip_duration_s"]),
        ),
        "timestep_convergence": run_timestep_convergence(candidate, timesteps_s=contact["timestep_candidates_s"]),
    }
    evidence.update(_finger_and_compiled_evidence(candidate, protocol))
    return seal_artifact(evidence)


def verify_contact_candidate(
    candidate: Mapping[str, Any], evidence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    errors = []
    gates = protocol["contact"]["hard_gates"]
    expected_friction = [
        float(candidate["sliding_mu"]["table_object"]),
        float(candidate["sliding_mu"]["table_object"]),
        float(candidate["torsional_mu"]),
        float(candidate["rolling_mu"]),
        float(candidate["rolling_mu"]),
    ]
    if evidence.get("candidate_id") != candidate["candidate_id"] or evidence.get("payload_sha256") != artifact_hash(evidence):
        errors.append("contact evidence identity/hash mismatch")
    for support, audit in evidence.get("compiled_explicit_pairs", {}).items():
        if (
            int(audit.get("matching_pair_count", 0)) != 1
            or int(audit.get("condim", 0)) != int(candidate["condim"])
            or not np.allclose(audit.get("friction", ()), expected_friction, rtol=0.0, atol=1.0e-14)
            or not np.allclose(audit.get("solref", ()), candidate["solref"], rtol=0.0, atol=1.0e-14)
            or not np.allclose(audit.get("solimp", ()), candidate["solimp"], rtol=0.0, atol=1.0e-14)
            or float(audit.get("can_mass_kg", math.nan)) != 0.349
        ):
            errors.append(f"{support} compiled tuple mismatch")
    if set(evidence.get("compiled_explicit_pairs", {})) != {"open_worktable", "target_bottom"}:
        errors.append("compiled explicit-pair coverage incomplete")
    for support, result in evidence.get("task_native_settling", {}).items():
        if not (
            result.get("final_window_named_support_continuous") is True
            and float(result.get("final_window_max_linear_speed_m_s", math.inf)) <= float(gates["linear_speed_max_m_s"])
            and float(result.get("final_window_max_angular_speed_rad_s", math.inf)) <= float(gates["angular_speed_max_rad_s"])
            and float(result.get("maximum_penetration_m", math.inf)) <= float(gates["maximum_illegal_penetration_m"])
            and result.get("finite_contact_forces") is True
            and result.get("finite_state") is True
            and int(result.get("warning_count", 1)) == 0
        ):
            errors.append(f"{support} task-native settling failed")
    if set(evidence.get("task_native_settling", {})) != {"open_worktable", "target_bottom"}:
        errors.append("task-native settling coverage incomplete")
    force_checks = force_threshold_gates(
        evidence.get("level_force_threshold", {}),
        penetration_max_m=float(gates["maximum_illegal_penetration_m"]),
        transverse_drift_max_m=float(gates["transverse_drift_max_m"]),
    )
    errors.extend("force gate failed: " + name for name, passed in force_checks.items() if not passed)
    slip = evidence.get("single_axis_slip", {})
    if not (
        float(slip.get("final_speed_m_s", math.inf)) <= float(gates["slip_final_speed_max_m_s"])
        and float(slip.get("maximum_penetration_m", math.inf)) <= float(gates["maximum_illegal_penetration_m"])
        and slip.get("finite_state") is True
        and slip.get("contact_identity_valid") is True
        and int(slip.get("warning_count", 1)) == 0
    ):
        errors.append("single-axis slip sanity failed")
    convergence = evidence.get("timestep_convergence", {})
    if (
        list(convergence.get("timesteps_s", ())) != list(protocol["contact"]["timestep_candidates_s"])
        or float(convergence.get("maximum_relative_force_error", math.inf)) > float(gates["timestep_trace_relative_error_max"])
        or any(int(row.get("warning_count", 1)) != 0 or row.get("final_window_named_support_continuous") is not True for row in convergence.get("records", ()))
    ):
        errors.append("three-timestep convergence failed")
    finger = evidence.get("finger_load", {})
    named = {tuple(sorted(row)) for row in finger.get("named_pair_audit", ())}
    if not (
        float(finger.get("maximum_normal_force_N", 0.0)) >= float(gates["finger_force_min_N"])
        and float(finger.get("maximum_normal_force_N", math.inf)) <= float(gates["finger_force_max_N"])
        and float(finger.get("maximum_penetration_m", math.inf)) <= float(gates["finger_penetration_max_m"])
        and len(named) == 2
        and float(evidence.get("finger_force_envelope", {}).get("finite_force_envelope_N", math.nan)) == float(gates["finger_force_max_N"])
        and int(evidence.get("warning_count", 1)) == 0
    ):
        errors.append("finger force/penetration envelope failed")
    return not errors, errors


def _parity_probe(candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Run Gamma=0 physics without calling task reward or success."""

    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    profile = _candidate_profile(candidate, protocol)
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=profile.model_timestep_s,
        target_container_friction=(0.30, float(candidate["torsional_mu"]), float(candidate["rolling_mu"])),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=30,
        seed=17,
    )
    try:
        audit = env.audit_compiled_model()
        env.reset()
        zero = np.zeros(env.action_dim)
        for _ in range(20):
            for internal_index in range(env._control_steps):
                time_s = float(env.sim.data.time)
                env._pre_physics_step(time_s, internal_index == 0)
                for hook in env._pre_physics_step_hooks:
                    hook(time_s, internal_index == 0)
                env.sim.forward()
                env._pre_action(zero, internal_index == 0)
                env.sim.step()
                env._physics_step_index += 1
            env.cur_time += env.control_timestep
        result = {
            "gamma": 0.0,
            "zero_action": True,
            "action_dimension": int(env.action_dim),
            "control_frequency_hz": float(env.control_freq),
            "worktable_mass_kg": float(audit["worktable"]["mass_kg"]),
            "can_mass_kg": float(audit["can"]["mass_kg"]),
            "target_inner_xy_m": audit["target_container"]["inner_xy_m"],
            "target_outer_xy_m": audit["target_container"]["outer_xy_m"],
            "warning_count": int(np.sum(env.sim.data.warning.number)),
            "physics_only": True,
            "task_reward_or_success_called": False,
        }
    finally:
        env.close()
    expected = protocol["parity"]
    result["passed"] = bool(
        result["action_dimension"] == int(expected["action_dimension"])
        and result["control_frequency_hz"] == float(expected["control_frequency_hz"])
        and result["worktable_mass_kg"] == float(expected["worktable_mass_kg"])
        and result["can_mass_kg"] == float(expected["can_mass_kg"])
        and np.allclose(result["target_inner_xy_m"], expected["target_inner_xy_m"], rtol=0.0, atol=1.0e-12)
        and np.allclose(result["target_outer_xy_m"], expected["target_outer_xy_m"], rtol=0.0, atol=1.0e-12)
        and result["warning_count"] == 0
    )
    return result


def _selected_diagnostics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    level = run_level_control(candidate)
    return {
        "selection_authority": False,
        "publication_gate": False,
        "drop_15cm": run_normal_impact_trace(candidate, duration_s=0.50),
        "incline": run_incline_calibration(candidate, level) if level.get("passed") else {"blocked_by_level_control": True},
        "labels": {"drop_15cm": "public_non_blocking_diagnostic", "incline": "public_non_blocking_diagnostic"},
        "passed": True,
    }


def _v6_component_probe(group: str) -> Mapping[str, Any]:
    from robosuite.scripts.shakebench_select_physics_v6 import (
        _resolve,
        _v6_driver_probe,
        _v6_isolator_probe,
        load_v6_protocol,
    )

    path = _asset("shakebench_selection_protocol_v6.yaml")
    protocol, _, protocol_hash = load_v6_protocol(path)
    states = _resolve(protocol, protocol_bytes_hash=protocol_hash)
    if group == "driver":
        state = next(
            row
            for row in states
            if row.stage == "driver"
            and row.candidate_id == "dt_nominal"
            and row.driver is not None
            and row.driver.gamma == 0.30
            and row.driver.load_case == "empty"
        )
        return _v6_driver_probe(state)
    state = next(row for row in states if row.stage == "isolator" and row.candidate_id == "low_frequency_damped")
    return _v6_isolator_probe(state)


def _replay_payload(group: str, candidate: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if group in {"driver", "isolator"}:
        evidence = _v6_component_probe(group)
        metrics = evidence.get("metrics", {}) if group == "driver" else {"summary": evidence.get("summary"), "hard_gates": evidence.get("hard_gates")}
    elif group == "contact":
        force = protocol["contact"]["force_threshold"]
        evidence = {
            "open": run_task_native_settling(candidate, support="open_worktable"),
            "target": run_task_native_settling(candidate, support="target_bottom"),
            "force": run_horizontal_force_threshold(
                candidate,
                force_factors=force["force_factors"],
                reference_mu=force["reference_mu"],
                mass_kg=force["mass_kg"],
                gravity_m_s2=force["gravity_m_s2"],
                plateau_duration_s=force["plateau_duration_s"],
                static_displacement_max_m=force["static_displacement_max_m"],
                monotonic_tolerance_m=force["monotonic_tolerance_m"],
                bracket_relative_tolerance=force["bracket_relative_tolerance"],
            ),
        }
        metrics = {
            "open_speed": evidence["open"]["final_window_max_linear_speed_m_s"],
            "target_speed": evidence["target"]["final_window_max_linear_speed_m_s"],
            "transition": evidence["force"]["transition"],
        }
    else:
        evidence = _parity_probe(candidate, protocol)
        metrics = {"passed": evidence["passed"], "warning_count": evidence["warning_count"]}
    return evidence, metrics


def replay_worker(protocol_path: Path, candidate_id: str, group: str, process_index: int) -> dict[str, Any]:
    protocol, _, protocol_hash = load_protocol(protocol_path)
    candidate = next(row for row in protocol["contact_candidates"] if row["candidate_id"] == candidate_id)
    evidence, metrics = _replay_payload(group, candidate, protocol)
    trace_digest = (
        str(evidence.get("trace", {}).get("digest"))
        if group == "driver" and isinstance(evidence.get("trace"), Mapping)
        else artifact_hash(dict(evidence))
    )
    return {
        "schema_id": "shakebench.phase06f.replay",
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "contact_candidate_id": candidate_id,
        "group": group,
        "process_index": int(process_index),
        "process_count": 3,
        "trace_digest": trace_digest,
        "metric_digest": artifact_hash(dict(metrics)),
        "digest_scope": "declared_complete_trace_schema",
        "complete_trace": True,
        "same_process_reset_used": False,
        "passed": bool(evidence.get("passed", True)),
    }


class ProductionEvidenceStore:
    """Filesystem/MuJoCo adapter for the deep finalizer."""

    def __init__(self, protocol: Mapping[str, Any], protocol_path: Path, protocol_sha256: str, output_dir: Path):
        self.protocol = protocol
        self.protocol_path = protocol_path
        self._protocol_sha256 = protocol_sha256
        self.output_dir = output_dir
        self.replay_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    @property
    def protocol_sha256(self) -> str:
        return self._protocol_sha256

    def verify_inherited(self, protocol):
        from robosuite.scripts.shakebench_select_physics_v8 import _verify_v7_noncontact_evidence

        evidence = dict(_verify_v7_noncontact_evidence())
        if evidence.get("selected_driver") != "dt_nominal" or evidence.get("selected_isolator") != "low_frequency_damped":
            raise Phase06FError("inherited driver/isolator selection changed")
        return evidence

    def execute_contact(self, candidate, protocol):
        evidence = execute_contact_candidate(candidate, protocol)
        filename = str(candidate["output"])
        write_json_atomic(self.output_dir / filename, evidence)
        return evidence

    def verify_contact(self, candidate, evidence, protocol):
        return verify_contact_candidate(candidate, evidence, protocol)

    def freeze_selection(self, selection):
        path = self.output_dir / SELECTION_FILENAME
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != selection:
                raise Phase06FError("immutable selection artifact already exists with different bytes")
        else:
            write_json_atomic(path, selection)
        return json.loads(path.read_text(encoding="utf-8"))

    def freeze_dependent_manifest(self, manifest):
        path = self.output_dir / DEPENDENT_MANIFEST_FILENAME
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise Phase06FError("immutable dependent manifest already exists with different bytes")
        else:
            write_json_atomic(path, manifest)
        return json.loads(path.read_text(encoding="utf-8"))

    def execute_dependent(self, state, protocol):
        output_path = self.output_dir / str(state["output"])
        if output_path.is_file():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("payload_sha256") == artifact_hash(existing)
                and all(existing.get(key) == state.get(key) for key in ("state_id", "contact_candidate_id", "selection_artifact_sha256", "group"))
                and (
                    state.get("group") not in REPLAY_GROUPS
                    or existing.get("evidence", {}).get("digest_scope") == "declared_complete_trace_schema"
                )
            ):
                return existing
        candidate = next(row for row in protocol["contact_candidates"] if row["candidate_id"] == state["contact_candidate_id"])
        group = str(state["group"])
        if group == "selected_diagnostics":
            evidence = _selected_diagnostics(candidate)
        elif group == "gamma_zero_parity":
            evidence = _parity_probe(candidate, protocol)
        else:
            command = [
                sys.executable,
                "-m",
                "robosuite.scripts.shakebench_finalize_physics",
                "--worker",
                "--protocol",
                str(self.protocol_path),
                "--candidate-id",
                str(candidate["candidate_id"]),
                "--group",
                group,
                "--process-index",
                str(state["process_index"]),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise Phase06FError(f"fresh-process {group} replay failed: {completed.stderr or completed.stdout}")
            evidence = json.loads(completed.stdout.strip().splitlines()[-1])
        record = seal_artifact(
            {
                "schema_id": "shakebench.phase06f.dependent_evidence",
                "schema_version": 1,
                "state_id": state["state_id"],
                "contact_candidate_id": state["contact_candidate_id"],
                "selection_artifact_sha256": state["selection_artifact_sha256"],
                "group": group,
                "evidence": evidence,
                "passed": evidence.get("passed") is True,
            }
        )
        write_json_atomic(output_path, record)
        return record

    def verify_dependent(self, state, evidence, protocol):
        errors = []
        for key in ("state_id", "contact_candidate_id", "selection_artifact_sha256", "group"):
            if evidence.get(key) != state.get(key):
                errors.append(f"dependent {key} binding mismatch")
        if evidence.get("payload_sha256") != artifact_hash(evidence) or evidence.get("passed") is not True:
            errors.append("dependent evidence hash/gate failed")
        if state["group"] in REPLAY_GROUPS:
            replay = evidence.get("evidence", {})
            if replay.get("contact_candidate_id") != state["contact_candidate_id"] or replay.get("group") != state["group"]:
                errors.append("fresh-process replay winner/group mismatch")
            self.replay_records[str(state["group"])].append(replay)
            if int(state.get("process_index", 0)) == 3:
                rows = self.replay_records[str(state["group"])]
                if (
                    {row.get("process_index") for row in rows} != {1, 2, 3}
                    or len({row.get("trace_digest") for row in rows}) != 1
                    or len({row.get("metric_digest") for row in rows}) != 1
                    or not all(row.get("complete_trace") is True and row.get("same_process_reset_used") is False for row in rows)
                ):
                    errors.append("three-process replay equality failed")
        return not errors, errors


def dry_run_all_winners(protocol: Mapping[str, Any], protocol_sha256: str) -> dict[str, Any]:
    validation = validate_finalization_protocol(protocol)
    sources = validate_registered_sources(protocol)
    branches = []
    for winner in validation["candidate_ids"]:
        selection_hash = hashlib.sha256((protocol_sha256 + winner).encode("utf-8")).hexdigest()
        from robosuite.utils.shakebench_physics_finalizer import materialize_dependent_manifest

        manifest = materialize_dependent_manifest(protocol, winner=winner, selection_artifact_sha256=selection_hash)
        branches.append(
            {
                "winner": winner,
                "selection_artifact_sha256": selection_hash,
                "manifest_sha256": manifest["payload_sha256"],
                "state_count": len(manifest["states"]),
                "state_digest": artifact_hash({"states": manifest["states"]}),
                "all_rows_bound": all(row["contact_candidate_id"] == winner for row in manifest["states"]),
            }
        )
    return {
        "protocol_sha256": protocol_sha256,
        "candidate_count": len(validation["candidate_ids"]),
        "branches": branches,
        "mujoco_calls": 0,
        "source_authentication": sources,
    }


def adapter_contract(protocol: Mapping[str, Any], protocol_sha256: str) -> dict[str, Any]:
    validation = validate_finalization_protocol(protocol)
    validate_registered_sources(protocol)
    plans = []
    for candidate in protocol["contact_candidates"]:
        plans.append({"candidate_id": candidate["candidate_id"], "stages": ["compiled_pairs", "open_settle", "target_settle", "level_force", "single_axis_slip", "finger_envelope", "three_dt"]})
    return {
        "protocol_sha256": protocol_sha256,
        "candidate_ids": validation["candidate_ids"],
        "plans": plans,
        "adapter_contract_digest": artifact_hash({"plans": plans}),
        "backend": "NoPhysicsBackend",
        "mujoco_calls": 0,
    }


def write_feasibility(protocol_path: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    protocol, source, protocol_sha256 = load_protocol(protocol_path)
    validation = validate_finalization_protocol(protocol)
    sources = validate_registered_sources(protocol)
    dry = dry_run_all_winners(protocol, protocol_sha256)
    contract = adapter_contract(protocol, protocol_sha256)
    payload = seal_artifact(
        {
            "schema_id": "shakebench.phase06f.feasibility",
            "schema_version": 1,
            "status": "PASS",
            "protocol": {"path": source.name, "sha256": protocol_sha256},
            "validation": validation,
            "source_authentication": sources,
            "dry_run": dry,
            "adapter_contract": contract,
            "commands": {
                "validate": {"exit_code": 0},
                "dry_run_all_winners": {"exit_code": 0},
                "adapter_contract": {"exit_code": 0},
            },
            "mujoco_calls": 0,
        }
    )
    destination = _asset(FEASIBILITY_FILENAME) if output_path is None else Path(output_path)
    write_json_atomic(destination, payload)
    return payload


def publish_result(result: BlockedResult | PublicationPlan, output_dir: Path) -> None:
    if isinstance(result, PublicationPlan):
        profile_path = output_dir / OFFICIAL_PROFILE_FILENAME
        profile_path.write_text(yaml.safe_dump(dict(result.official_profile), sort_keys=False), encoding="utf-8")
    write_json_atomic(output_dir / FINAL_STATUS_FILENAME, result.final_status)
    write_json_atomic(output_dir / HANDOFF_FILENAME, result.handoff)


def verify_final_publication(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Recompute hard gates, priority, bindings, hashes, and profile alignment."""

    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    errors = []
    bundle = verify_official_publication_bundle(output)
    errors.extend(bundle.get("errors", ()))
    try:
        protocol, source, protocol_sha256 = load_protocol(output / PROTOCOL_FILENAME)
        validate_finalization_protocol(protocol)
        validate_registered_sources(protocol)
        feasibility = json.loads((output / FEASIBILITY_FILENAME).read_text(encoding="utf-8"))
        if (
            feasibility.get("payload_sha256") != artifact_hash(feasibility)
            or feasibility.get("status") != "PASS"
            or feasibility.get("protocol", {}).get("sha256") != protocol_sha256
            or feasibility.get("mujoco_calls") != 0
        ):
            errors.append("Phase 06F feasibility artifact failed authentication")
        selection = json.loads((output / SELECTION_FILENAME).read_text(encoding="utf-8"))
        handoff = json.loads((output / HANDOFF_FILENAME).read_text(encoding="utf-8"))
        records = {}
        eligible = []
        for candidate in protocol["contact_candidates"]:
            path = output / str(candidate["output"])
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if file_sha256(path) == candidate.get("source", {}).get("sha256"):
                errors.append("Phase 06F contact evidence reused historical candidate bytes")
            passed, candidate_errors = verify_contact_candidate(candidate, evidence, protocol)
            records[candidate["candidate_id"]] = {"passed": passed, "errors": candidate_errors}
            if passed:
                eligible.append(candidate["candidate_id"])
        winner = eligible[0] if eligible else None
        if winner != selection.get("contact_winner") or winner != bundle.get("contact_winner"):
            errors.append("minimal-intervention priority recomputation differs from publication")
        selected_rows = [row for row in selection.get("candidates", ()) if row.get("eligible")]
        if [row.get("candidate_id") for row in selected_rows] != eligible:
            errors.append("selection eligibility table differs from recomputed hard gates")
        profile = yaml.safe_load((output / OFFICIAL_PROFILE_FILENAME).read_text(encoding="utf-8"))
        PhysicsProfile(payload=profile, source=str(output / OFFICIAL_PROFILE_FILENAME), profile_sha256=str(profile.get("profile_sha256"))).assert_valid()
        candidate = next(row for row in protocol["contact_candidates"] if row["candidate_id"] == winner)
        expected_contact = {
            key: copy.deepcopy(candidate[key])
            for key in ("candidate_id", "condim", "sliding_mu", "torsional_mu", "rolling_mu", "margin_m", "gap_m", "solref", "solimp", "pair_scope", "interfaces")
        }
        expected_contact["friction_encoding"] = "isotropic_pair_5d"
        if profile.get("physics", {}).get("contact") != expected_contact:
            errors.append("official profile does not exactly match the selected contact tuple")
        if profile.get("physics", {}).get("timestep", {}).get("candidate_id") != "dt_nominal" or profile.get("physics", {}).get("isolator", {}).get("candidate_id") != "low_frequency_damped":
            errors.append("official profile does not bind the inherited driver/isolator")
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(source)],
            cwd=source.resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        registration_commit = completed.stdout.strip()
        if completed.returncode == 0 and registration_commit and handoff.get("registration_commit") != registration_commit:
            errors.append("handoff registration commit mismatch")
    except Exception as exc:
        errors.append(f"final verifier exception: {type(exc).__name__}: {exc}")
        records = {}
        eligible = []
        winner = None
    return {
        "passed": not errors,
        "errors": errors,
        "contact_winner": winner,
        "eligible_contacts": eligible,
        "contact_gates": records,
        "bundle": bundle,
        "physics_only": True,
        "task_success_used": False,
        "controller_outcome_used": False,
    }


def run_finalization(protocol_path: str | Path | None = None, output_dir: str | Path | None = None) -> BlockedResult | PublicationPlan:
    protocol, source, protocol_sha256 = load_protocol(protocol_path)
    validate_registered_sources(protocol)
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(source)],
        cwd=source.resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or len(completed.stdout.strip()) != 40:
        raise Phase06FError("cannot resolve the protocol registration commit")
    protocol = copy.deepcopy(protocol)
    protocol["registration_commit"] = completed.stdout.strip()
    output = Path(models.assets_root) if output_dir is None else Path(output_dir)
    store = ProductionEvidenceStore(protocol, source, protocol_sha256, output)
    result = finalize_official_physics(protocol, store)
    publish_result(result, output)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--dry-run-all-winners", action="store_true")
    parser.add_argument("--adapter-contract", action="store_true")
    parser.add_argument("--write-feasibility", action="store_true")
    parser.add_argument("--verify-final", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--process-index", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if None in (args.protocol, args.candidate_id, args.group, args.process_index):
                raise Phase06FError("worker requires protocol, candidate-id, group, and process-index")
            result = replay_worker(Path(args.protocol), args.candidate_id, args.group, args.process_index)
        else:
            protocol, source, protocol_sha256 = load_protocol(args.protocol)
            if args.validate_protocol:
                result = {"passed": True, "protocol_sha256": protocol_sha256, "validation": validate_finalization_protocol(protocol), "source_authentication": validate_registered_sources(protocol)}
            elif args.dry_run_all_winners:
                result = dry_run_all_winners(protocol, protocol_sha256)
            elif args.adapter_contract:
                result = adapter_contract(protocol, protocol_sha256)
            elif args.write_feasibility:
                result = write_feasibility(source)
            elif args.verify_final:
                result = verify_final_publication(args.output_dir)
            else:
                final = run_finalization(source, args.output_dir)
                result = {"status": final.status, "reason": getattr(final, "reason", None), "contact_winner": getattr(final, "contact_winner", None)}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
        return 0 if result.get("status", "PASS") == "PASS" or result.get("passed") is True else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
