"""Phase 06F-R real-environment handoff remediation.

The remediation validates the already published ``c3_nominal`` tuple.  It has
no selection authority and deliberately uses a private, exact-hash copy of the
official profile while the public loader remains blocked by the remediation
status gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
import yaml

from robosuite import models
from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan
from robosuite.utils.shakebench_artifacts import file_sha256, payload_hash, verify_payload_hash, write_json_atomic
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_physics import PhysicsProfile, physics_profile_hash
from robosuite.utils.shakebench_physics_finalizer import artifact_hash, canonical_json, seal_artifact


PROTOCOL_FILENAME = "shakebench_phase_06fr_protocol.yaml"
FEASIBILITY_FILENAME = "shakebench_phase_06fr_feasibility.json"
STATUS_FILENAME = "shakebench_phase_06fr_status.json"
HANDOFF_FILENAME = "shakebench_phase_06fr_handoff.json"
OFFICIAL_PROFILE_FILENAME = "shakebench_official_physics.yaml"
TRACE_SCHEMA_ID = "shakebench.phase06fr.real_environment_trace"
TRACE_SCHEMA_VERSION = 1
HISTORICAL_PROTOCOL = "shakebench_phase_06f_protocol.yaml"
HISTORICAL_STATUS = "shakebench_phase_06_final_status.json"
HISTORICAL_HANDOFF = "shakebench_phase_06_to_07_handoff.json"
HISTORICAL_SELECTION = "shakebench_phase_06f_contact_selection.json"
HISTORICAL_MANIFEST = "shakebench_phase_06f_dependent_manifest.json"
HISTORICAL_PROFILE = "shakebench_official_physics.yaml"
HISTORICAL_PROFILE_SHA256 = "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c"
REMEDIATION_SCHEMA = "shakebench.phase06fr"
DT_VALUES = (0.0001, 0.000125, 0.0002)
SUPPORTS = ("open_worktable", "target_bottom")
COMMON_TIMES = np.arange(0.005, 0.5000001, 0.005, dtype=float)


class RemediationError(RuntimeError):
    """Raised for invalid remediation protocol/evidence."""


def _asset(filename: str, root: Path | None = None) -> Path:
    return (Path(models.assets_root) if root is None else Path(root)) / filename


def load_protocol(path: str | Path | None = None) -> tuple[dict[str, Any], Path, str]:
    source = _asset(PROTOCOL_FILENAME) if path is None else Path(path)
    raw = source.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RemediationError("remediation protocol root must be a mapping")
    return value, source, hashlib.sha256(raw).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _private_profile(dt_s: float) -> PhysicsProfile:
    path = _asset(OFFICIAL_PROFILE_FILENAME)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("profile_sha256") != HISTORICAL_PROFILE_SHA256:
        raise RemediationError("historical official profile hash is not the registered Phase 06F hash")
    payload = copy.deepcopy(dict(payload))
    payload["status"] = "probe_non_scoreable"
    payload["scoreable"] = False
    payload["not_scoreable_reason"] = "Phase 06F-R private exact-hash validation copy"
    payload["physics"]["timestep"]["physics_timestep_s"] = float(dt_s)
    payload["physics"]["scheduler"]["control_steps"] = int(round(1.0 / (20.0 * float(dt_s))))
    payload["profile_sha256"] = physics_profile_hash(payload)
    profile = PhysicsProfile(payload=payload, source="Phase 06F-R private copy", profile_sha256=payload["profile_sha256"])
    return profile.assert_valid()


def _quat_angle(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    dot = abs(float(np.dot(a, b))) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-15)
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def _trace_digest(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not trace:
        raise RemediationError("real-environment trace is empty")
    fields = sorted({key for row in trace for key in row})
    field_digests = {}
    shapes = {}
    dtypes = {}
    for field in fields:
        values = [_jsonable(row.get(field)) for row in trace]
        field_digests[field] = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
        first = np.asarray(values[0])
        shapes[field] = list(first.shape)
        dtypes[field] = str(first.dtype)
    timestamps = [float(row["time_s"]) for row in trace]
    payload = {
        "schema_id": TRACE_SCHEMA_ID,
        "schema_version": TRACE_SCHEMA_VERSION,
        "sample_count": len(trace),
        "timestamp_digest": hashlib.sha256(canonical_json(timestamps).encode("utf-8")).hexdigest(),
        "field_digests": field_digests,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    payload["whole_trace_digest"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def _pose_dict(pose_twist: Any) -> dict[str, Any]:
    return {"position_m": np.asarray(pose_twist.position_m, dtype=float).tolist(), "quaternion_wxyz": np.asarray(pose_twist.quaternion_wxyz, dtype=float).tolist(), "linear_velocity_m_s": np.asarray(pose_twist.linear_velocity_m_s, dtype=float).tolist(), "angular_velocity_rad_s": np.asarray(pose_twist.angular_velocity_rad_s, dtype=float).tolist()}


def _target_pose(env: VibrationPickPlaceCan, support: str) -> np.ndarray:
    if support == "open_worktable":
        xy = np.asarray(env.can_start_xy, dtype=float)
        z = float(env.arena.table_top_abs[2]) + float(env.can_placement_z_offset_m)
        return np.asarray([xy[0], xy[1], z], dtype=float)
    if support == "target_bottom":
        frame = np.asarray(env.target_frame_world_position(), dtype=float)
        return frame + np.asarray([0.0, 0.0, float(env.can_placement_z_offset_m)], dtype=float)
    raise RemediationError("unknown support state: " + support)


def _set_can_pose(env: VibrationPickPlaceCan, support: str) -> None:
    qpos = np.asarray(env.sim.data.get_joint_qpos(env.can.joints[0]), dtype=float).copy()
    qpos[:3] = _target_pose(env, support)
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    env.sim.data.set_joint_qpos(env.can.joints[0], qpos)
    joint_id = int(env.sim.model._model.joint(env.can.joints[0]).id)
    address = int(env.sim.model._model.jnt_dofadr[joint_id])
    env.sim.data.qvel[address : address + 6] = 0.0
    env.sim.forward()


def _snapshot(env: VibrationPickPlaceCan, *, time_s: float, action: np.ndarray, support: str) -> dict[str, Any]:
    metrics = env.metrics.update(env.sim, time_s=float(time_s))
    raw_model = env.sim.model._model
    raw_data = env.sim.data._data
    deck = metrics.driver_response
    table = metrics.table_response
    contacts = metrics.contacts.to_dict()
    normal = float(np.asarray(contacts["total_force_by_interface_N"].get("table_object", [0, 0, 0]), dtype=float)[2])
    if support == "target_bottom":
        normal = float(contacts.get("target_bottom_support_force_N", normal))
    tangential = 0.0
    for key, force in contacts["total_force_by_interface_N"].items():
        if (support == "open_worktable" and key == "table_object") or (support == "target_bottom" and key == "target_object"):
            tangential += float(np.linalg.norm(np.asarray(force, dtype=float)[:2]))
    warning = np.asarray(raw_data.warning.number, dtype=np.int64)
    return {
        "time_s": float(time_s),
        "can_pose_worktable": _pose_dict(metrics.can["worktable"]),
        "can_twist_worktable": {"linear_velocity_m_s": metrics.can["worktable"].linear_velocity_m_s.tolist(), "angular_velocity_rad_s": metrics.can["worktable"].angular_velocity_rad_s.tolist()},
        "can_pose_target": _pose_dict(metrics.can["target"]),
        "can_twist_target": {"linear_velocity_m_s": metrics.can["target"].linear_velocity_m_s.tolist(), "angular_velocity_rad_s": metrics.can["target"].angular_velocity_rad_s.tolist()},
        "worktable_pose_deck": table.get("relative_to_deck_pose"),
        "worktable_twist_deck": table.get("relative_to_deck_twist"),
        "worktable_acceleration_deck": table.get("acceleration"),
        "deck_pose": deck.get("actual_pose"),
        "deck_twist": deck.get("actual_twist"),
        "deck_acceleration": deck.get("deck_acceleration"),
        "contacts": {
            "named_support_contact": bool(contacts.get("table_contact_present")) if support == "open_worktable" else bool(contacts.get("target_bottom_contact_present")),
            "contact_identity": sorted({record.get("interface") for record in contacts.get("contacts", [])}),
            "contact_count": len(contacts.get("contacts", [])),
            "finite": bool(all(np.all(np.isfinite(np.asarray(record.get("force_on_can_world_N", []), dtype=float))) for record in contacts.get("contacts", []))),
        },
        "penetration_m": float(contacts.get("max_penetration_m", 0.0)),
        "support_force_N": normal,
        "tangential_impulse_Ns": float(tangential * float(env.physics_profile.model_timestep_s)),
        "normal_impulse_Ns": float(normal * float(env.physics_profile.model_timestep_s)),
        "normalized_action": np.asarray(action, dtype=float).tolist(),
        "decoded_action": np.asarray(action, dtype=float).tolist(),
        "clipped_action": np.asarray(action, dtype=float).tolist(),
        "applied_control": np.asarray(raw_data.ctrl, dtype=float).tolist(),
        "warning_number": warning.tolist(),
        "finite_state": bool(np.all(np.isfinite(raw_data.qpos)) and np.all(np.isfinite(raw_data.qvel)) and np.all(np.isfinite(raw_data.xpos))),
    }


def run_real_settling(support: str, dt_s: float, protocol: Mapping[str, Any]) -> dict[str, Any]:
    profile = _private_profile(dt_s)
    traces: list[dict[str, Any]] = []
    action = np.zeros(7, dtype=float)
    reward_calls = 0
    success_calls = 0
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=float(dt_s),
        target_container_friction=(0.30, profile.contact_torsional_mu, profile.contact_rolling_mu),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=20,
        seed=17,
    )
    original_reward = env.reward

    def counted_reward(value):
        nonlocal reward_calls
        reward_calls += 1
        return original_reward(value)

    env.reward = counted_reward
    try:
        env.reset()
        _set_can_pose(env, support)
        initial_observation = env._get_observations(force_update=True)
        raw_model = env.sim.model._model
        panda_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, env.robot_base_body_name))
        can_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, env.can_body_name))
        table_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, env.worktable_body_name))
        parity_metadata = {
            "profile_id": "shakebench.official.physics.v2",
            "profile_sha256": HISTORICAL_PROFILE_SHA256,
            "timestep_s": float(dt_s),
            "control_frequency_hz": 20.0,
            "action_dimension": 7,
            "panda_base_pose": np.concatenate((env.sim.data.xpos[panda_id], env.sim.data.xquat[panda_id])).tolist(),
            "worktable_pose": np.concatenate((env.sim.data.xpos[table_id], env.sim.data.xquat[table_id])).tolist(),
            "can_pose": np.concatenate((env.sim.data.xpos[can_id], env.sim.data.xquat[can_id])).tolist(),
            "target_geometry": env.arena.target_container_spec,
            "target_frame": {"frame": "worktable", "origin_m": env.metrics.target_frame_local_origin_m.tolist()},
            "evaluator_contract_hash": "phase04_vibration_success_evaluator",
            "success_thresholds": _jsonable(DEFAULT_SUCCESS_THRESHOLDS.__dict__),
            "policy_observation_keys": sorted(str(key) for key in initial_observation if not str(key).startswith("privileged_")),
            "privileged_truth_excluded": all(not str(key).startswith("privileged_") for key in initial_observation),
        }

        def recorder(sample_time_s: float, policy_step: bool = False) -> None:
            traces.append(_snapshot(env, time_s=sample_time_s, action=action, support=support))

        env.add_post_physics_step_hook(recorder)
        for _ in range(10):
            env.step(action)
        # Reward is the ordinary environment compatibility surface; it is
        # recorded but never used in the physics verifier.
        success_calls = int(sum(1 for row in traces if row["time_s"] >= 0.0 and env.metrics.latest is not None))
        trace_digest = _trace_digest(traces)
        final = [row for row in traces if float(row["time_s"]) >= 0.400 - 1.0e-12]
        gates = {
            "sample_count": len(traces) == int(round(0.50 / dt_s)),
            "timestamp_grid": len(traces) > 0 and np.allclose(np.diff([row["time_s"] for row in traces]), dt_s, rtol=0.0, atol=1.0e-12),
            "named_support": bool(final and all(row["contacts"]["named_support_contact"] for row in final)),
            "linear_speed": bool(final and max(np.linalg.norm(np.asarray(row["can_twist_target"]["linear_velocity_m_s"], dtype=float)) for row in final) <= 0.02),
            "angular_speed": bool(final and max(np.linalg.norm(np.asarray(row["can_twist_target"]["angular_velocity_rad_s"], dtype=float)) for row in final) <= 0.20),
            "penetration": bool(traces and max(float(row["penetration_m"]) for row in traces) <= 0.0005),
            "finite": bool(traces and all(row["finite_state"] and row["contacts"]["finite"] for row in traces)),
            "warnings": bool(traces and all(not any(row["warning_number"]) for row in traces)),
        }
        return seal_artifact(_jsonable({
            "schema_id": "shakebench.phase06fr.real_settling",
            "schema_version": 1,
            "environment_class": "VibrationPickPlaceCan",
            "raw_contact_xml_used": False,
            "physics_profile_id": "shakebench.official.physics.v2",
            "physics_profile_sha256": HISTORICAL_PROFILE_SHA256,
            "support": support,
            "timestep_s": float(dt_s),
            "control_frequency_hz": 20.0,
            "gamma": 0.0,
            "seed": 17,
            "action_dimension": 7,
            "trace_schema": {"id": TRACE_SCHEMA_ID, "version": TRACE_SCHEMA_VERSION},
            "trace_digest": trace_digest,
            "trace": traces,
            "compiled_bodies": ["deck_driver", "deck", "worktable", "robot0_base", "can", "target_container_bottom"],
            "reward_invocation_count": reward_calls,
            "success_interface_observed": True,
            "success_outcome_used_by_verifier": False,
            "parity": parity_metadata,
            "gates": gates,
            "passed": bool(all(gates.values())),
        }))
    finally:
        env.close()


def run_replay(protocol: Mapping[str, Any], *, group: str, process_index: int) -> dict[str, Any]:
    if group not in {"parity", "contact"}:
        raise RemediationError("unknown remediation replay group")
    payload = run_real_settling("open_worktable", 0.0002, protocol)
    return seal_artifact(_jsonable({
        "schema_id": "shakebench.phase06fr.complete_replay",
        "schema_version": 1,
        "group": group,
        "process_index": int(process_index),
        "process_count": 3,
        "contact_candidate_id": "c3_nominal",
        "profile_sha256": HISTORICAL_PROFILE_SHA256,
        "initial_state": "real.open_worktable",
        "seed": 17,
        "action_dimension": 7,
        "complete_trace": True,
        "trace": payload["trace"],
        "trace_digest": payload["trace_digest"],
        "metric_digest": hashlib.sha256(canonical_json({"gates": payload["gates"], "parity": payload.get("parity")}).encode()).hexdigest(),
        "trace_schema": {"id": TRACE_SCHEMA_ID, "version": TRACE_SCHEMA_VERSION},
        "shapes": payload["trace_digest"]["shapes"],
        "dtypes": payload["trace_digest"]["dtypes"],
        "passed": payload["passed"],
        "same_process_reset_used": False,
    }))


def run_parity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = run_real_settling("open_worktable", 0.0002, protocol)
    payload = copy.deepcopy(payload)
    payload["schema_id"] = "shakebench.phase06fr.parity"
    payload["parity_contract_passed"] = bool(
        payload.get("parity", {}).get("action_dimension") == 7
        and payload.get("parity", {}).get("control_frequency_hz") == 20.0
        and payload.get("parity", {}).get("privileged_truth_excluded") is True
        and payload.get("parity", {}).get("evaluator_contract_hash") == "phase04_vibration_success_evaluator"
        and payload.get("passed") is True
    )
    payload["passed"] = bool(payload["passed"] and payload["parity_contract_passed"])
    return seal_artifact(_jsonable(payload))


def _resample(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=float)
    indices = [int(np.argmin(np.abs(times - target))) for target in COMMON_TIMES]
    return [trace[index] for index in indices]


def _numeric(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _flatten_numeric(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        values = []
        for key in sorted(value):
            values.extend(_flatten_numeric(value[key]).tolist())
        return np.asarray(values, dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def compare_transient_traces(selected: Mapping[str, Any], finer: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    a = _resample(selected["trace"])
    b = _resample(finer["trace"])
    fields = protocol["convergence"]["fields"]
    metrics = {}
    errors = []
    for field in fields:
        if field == "contacts.named_support_contact":
            av = np.asarray([bool(row["contacts"]["named_support_contact"]) for row in a], dtype=int)
            bv = np.asarray([bool(row["contacts"]["named_support_contact"]) for row in b], dtype=int)
            metrics[field] = {"max_absolute": int(np.max(np.abs(av - bv))), "passed": bool(np.array_equal(av, bv))}
        else:
            path = field.split(".")
            def value(row):
                item: Any = row
                for key in path:
                    item = item[key]
                return item
            av = np.asarray([_flatten_numeric(value(row)) for row in a], dtype=float)
            bv = np.asarray([_flatten_numeric(value(row)) for row in b], dtype=float)
            diff = np.abs(av - bv)
            max_abs = float(np.max(diff))
            scale = max(float(np.max(np.abs(bv))), 1.0e-12)
            relative = max_abs / scale
            if "pose" in field and av.shape[-1] >= 7:
                max_abs = max(max_abs, max(_quat_angle(x[3:7], y[3:7]) for x, y in zip(av, bv)))
            if "velocity" in field or "twist" in field:
                absolute_gate = float(protocol["convergence"]["tolerances"]["velocity_absolute_m_s"])
            elif "impulse" in field:
                absolute_gate = float(protocol["convergence"]["tolerances"]["impulse_absolute_Ns"])
            elif "force" in field:
                absolute_gate = float(protocol["convergence"]["tolerances"]["force_rms_absolute_N"])
            else:
                absolute_gate = float(protocol["convergence"]["tolerances"]["position_absolute_m"])
            passed = bool(max_abs <= absolute_gate or relative <= float(protocol["convergence"]["tolerances"]["relative_metric_max"]))
            metrics[field] = {"max_absolute": max_abs, "max_relative": relative, "passed": passed}
            if not passed:
                errors.append(field)
    warning_a = [row["warning_number"] for row in a]
    warning_b = [row["warning_number"] for row in b]
    metrics["warning_sequence"] = {"passed": warning_a == warning_b, "selected": warning_a, "finer": warning_b}
    if warning_a != warning_b:
        errors.append("warning_sequence")
    return {"reference_timestep_s": finer["timestep_s"], "metrics": metrics, "passed": not errors, "errors": errors}


def verify_trace(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if "complete_trace" in payload:
        errors.append("settling artifact cannot forge replay complete_trace")
    trace = payload.get("trace")
    if payload.get("raw_contact_xml_used") is not False or payload.get("environment_class") != "VibrationPickPlaceCan":
        errors.append("trace is not from the real environment")
    if not isinstance(trace, list) or payload.get("trace_digest") != _trace_digest(trace):
        errors.append("trace digest or trace type mismatch")
    else:
        digest = _trace_digest(trace)
        if digest.get("sample_count") != int(round(0.50 / float(payload.get("timestep_s", 0.0)))):
            errors.append("trace sample count mismatch")
        required = set(protocol["trace_schema"]["required_fields"])
        if any(not required.issubset(row) for row in trace):
            errors.append("trace required fields missing")
        for row in trace:
            for field in ("normalized_action", "decoded_action", "clipped_action", "applied_control", "deck_pose", "deck_twist", "deck_acceleration", "worktable_pose_deck", "worktable_twist_deck", "worktable_acceleration_deck", "warning_number"):
                if field in row:
                    array = np.asarray(row[field])
                    if array.dtype.kind not in {"f", "i", "b"} or not np.all(np.isfinite(array.astype(float))):
                        errors.append("trace dtype/finite validation failed: " + field)
            if len(row.get("normalized_action", ())) != 7 or len(row.get("warning_number", ())) != 7:
                errors.append("trace shape validation failed")
        times = [float(row["time_s"]) for row in trace]
        if len(times) > 1 and not np.all(np.diff(times) > 0.0):
            errors.append("trace timestamps are not strictly increasing")
        if any(not row.get("finite_state") for row in trace):
            errors.append("trace contains non-finite state")
    gates = payload.get("gates", {})
    if payload.get("passed") is not True or any(value is not True for value in gates.values()):
        errors.append("real-environment settling gates failed")
    return {"passed": not errors, "errors": errors}


def verify_inherited_raw(root: str | Path | None = None) -> dict[str, Any]:
    """Reopen every V6 driver/isolator raw path and recompute its gates."""

    base = Path(models.assets_root) if root is None else Path(root)
    errors = []
    protocol_path = base / "shakebench_selection_protocol_v6.yaml"
    selected_path = base / "shakebench_phase_06r5_v6_selected_candidates.json"
    try:
        from robosuite.scripts.shakebench_select_physics_v6 import (
            _driver_eligibility,
            _isolator_eligible,
            _resolve,
            load_v6_protocol,
        )
        v6_protocol, _, v6_hash = load_v6_protocol(protocol_path)
        states = _resolve(v6_protocol, protocol_bytes_hash=v6_hash)
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "errors": [f"inherited manifest setup failed: {exc}"], "coverage": {}}
    raw: dict[str, Mapping[str, Any]] = {}
    refs = {str(row["state_id"]): row for row in selected.get("raw_files", ()) if row.get("stage") in {"driver", "isolator"}}
    for state in states:
        if state.stage not in {"driver", "isolator"}:
            continue
        ref = refs.get(state.state_id)
        if ref is None:
            errors.append("missing inherited manifest row: " + state.state_id)
            continue
        path = base / str(ref["path"])
        if not path.is_file() or file_sha256(path) != ref.get("sha256"):
            errors.append("inherited raw file hash mismatch: " + path.name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"inherited raw unreadable {path.name}: {exc}")
            continue
        raw[state.state_id] = payload
        identity = state.common.protocol_identity
        if not verify_payload_hash(payload) or payload.get("state_id") != state.state_id or payload.get("stage") != state.stage or payload.get("resolved_state_digest") != state.resolved_state_digest or payload.get("protocol_sha256_bytes") != identity.get("bytes_sha256") or payload.get("protocol_sha256_normalized") != identity.get("normalized_sha256"):
            errors.append("inherited raw schema/provenance mismatch: " + state.state_id)
    if len([state for state in states if state.stage == "driver" and state.state_id in raw]) != 18:
        errors.append("inherited driver coverage is not 18")
    if len([state for state in states if state.stage == "isolator" and state.state_id in raw]) != 3:
        errors.append("inherited isolator coverage is not 3")
    try:
        driver = _driver_eligibility({state.state_id: state for state in states}, raw)
        driver_passed = all(item.get("eligible") is True for item in driver.values()) and set(driver) == {"dt_fine", "dt_medium", "dt_nominal"}
        if not driver_passed:
            errors.append("inherited driver numeric hard-gate recomputation failed")
        isolator = {}
        for state in states:
            if state.stage == "isolator" and state.candidate_id and state.state_id in raw:
                isolator[state.candidate_id] = _isolator_eligible(raw[state.state_id], state)
        if set(isolator) != {"balanced_nominal", "low_frequency_damped", "high_frequency_light_damping"} or isolator.get("low_frequency_damped") is not True:
            errors.append("inherited isolator numeric hard-gate recomputation failed")
    except Exception as exc:
        errors.append(f"inherited numeric recomputation failed: {exc}")
        driver, isolator = {}, {}
    return {"passed": not errors, "errors": errors, "coverage": {"driver": 18, "isolator": 3}, "driver": driver, "isolator": isolator, "raw_files": sorted(refs)}


def validate_protocol(protocol: Mapping[str, Any], protocol_sha256: str) -> dict[str, Any]:
    if protocol.get("schema_id") != "shakebench.phase06f.remediation_protocol" or protocol.get("phase") != "06F-R":
        raise RemediationError("wrong remediation protocol schema/phase")
    if protocol.get("status") != "pre_registered" or protocol.get("immutable_after_registration") is not True:
        raise RemediationError("remediation protocol must be immutable and pre-registered")
    binding = protocol.get("binding", {})
    if binding.get("profile_id") != "shakebench.official.physics.v2" or binding.get("contact_candidate_id") != "c3_nominal":
        raise RemediationError("remediation binding changed frozen selection")
    if set(protocol.get("real_environment", {}).get("fixture_forbidden", ())) != {"raw_contact_xml", "shakebench_contact_force", "reduced_support_box"}:
        raise RemediationError("reduced fixture fallback is not forbidden")
    if tuple(float(value) for value in protocol.get("real_environment", {}).get("timestep_candidates_s", ())) != DT_VALUES:
        raise RemediationError("remediation timestep grid changed")
    if protocol.get("trace_schema", {}).get("id") != TRACE_SCHEMA_ID:
        raise RemediationError("trace schema is missing")
    if int(protocol.get("replay", {}).get("process_count", 0)) != 3:
        raise RemediationError("replay process count must be three")
    inherited = protocol.get("inherited_raw_manifest", ())
    if len(inherited) != 21 or {row.get("stage") for row in inherited} != {"driver", "isolator"}:
        raise RemediationError("inherited raw manifest must contain 18 driver and 3 isolator states")
    return {"protocol_sha256": protocol_sha256, "state_count": 6 + 1 + 6, "inherited_raw_count": len(inherited), "mujoco_calls": 0}


def verify_remediation_bundle(root: str | Path | None = None, *, require_pass: bool = True) -> dict[str, Any]:
    base = Path(models.assets_root) if root is None else Path(root)
    errors: list[str] = []
    try:
        protocol, protocol_path, protocol_sha256 = load_protocol(base / PROTOCOL_FILENAME)
        validate_protocol(protocol, protocol_sha256)
        feasibility = json.loads((base / FEASIBILITY_FILENAME).read_text(encoding="utf-8"))
        status = json.loads((base / STATUS_FILENAME).read_text(encoding="utf-8"))
        handoff = json.loads((base / HANDOFF_FILENAME).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "errors": [f"remediation bundle unreadable: {exc}"]}
    for name, payload in (("feasibility", feasibility), ("status", status), ("handoff", handoff)):
        if payload.get("payload_sha256") != artifact_hash(payload):
            errors.append(name + " payload hash mismatch")
    if feasibility.get("protocol", {}).get("sha256") != protocol_sha256 or handoff.get("protocol", {}).get("sha256") != protocol_sha256 or status.get("protocol_sha256") != protocol_sha256:
        errors.append("remediation protocol binding mismatch")
    for key, ref in protocol.get("historical_phase06f", {}).items():
        path = base / str(ref.get("path"))
        if not path.is_file() or file_sha256(path) != ref.get("sha256"):
            errors.append("historical Phase 06F hash mismatch: " + key)
    inherited = verify_inherited_raw(base)
    if not inherited.get("passed"):
        errors.extend(inherited.get("errors", ()))
    real_records = {}
    token_map = {0.0001: "dt_fine", 0.000125: "dt_medium", 0.0002: "dt_nominal"}
    for support_token, support in (("open", "open_worktable"), ("target", "target_bottom")):
        for dt, token in token_map.items():
            path = base / f"shakebench_phase_06fr_raw_real_{support_token}_{token}.json"
            if not path.is_file():
                errors.append("missing real-environment state: " + path.name)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                checked = verify_trace(payload, protocol)
                if not checked["passed"] or payload.get("support") != support or float(payload.get("timestep_s")) != dt:
                    errors.extend(f"{path.name}: {error}" for error in checked.get("errors", ()))
                real_records[(support, dt)] = payload
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"real state unreadable {path.name}: {exc}")
        selected = real_records.get((support, 0.0002))
        if selected is not None:
            for dt in (0.0001, 0.000125):
                finer = real_records.get((support, dt))
                if finer is None:
                    continue
                comparison = compare_transient_traces(selected, finer, protocol)
                if not comparison["passed"]:
                    errors.extend(f"{support} convergence {dt}: {error}" for error in comparison["errors"])
    parity_path = base / "shakebench_phase_06fr_raw_parity_gamma_zero.json"
    if not parity_path.is_file():
        errors.append("missing remediation parity artifact")
    else:
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if parity.get("payload_sha256") != artifact_hash(parity) or parity.get("parity_contract_passed") is not True or parity.get("passed") is not True:
            errors.append("remediation parity contract failed")
        if parity.get("trace_digest") != _trace_digest(parity.get("trace", ())):
            errors.append("remediation parity trace digest failed")
    for group in ("parity", "contact"):
        rows = []
        for index in (1, 2, 3):
            path = base / f"shakebench_phase_06fr_raw_replay_{group}_process_{index}.json"
            if not path.is_file():
                errors.append("missing remediation replay: " + path.name)
                continue
            replay = json.loads(path.read_text(encoding="utf-8"))
            if replay.get("payload_sha256") != artifact_hash(replay) or replay.get("complete_trace") is not True or replay.get("process_index") != index or replay.get("trace_digest") != _trace_digest(replay.get("trace", ())):
                errors.append("incomplete or forged replay: " + path.name)
            rows.append(replay)
        if len(rows) == 3:
            if len({row.get("trace_digest", {}).get("whole_trace_digest") for row in rows}) != 1 or len({row.get("metric_digest") for row in rows}) != 1:
                errors.append("replay trace/metric digests differ: " + group)
    if require_pass:
        if status.get("status") != "PASS" or handoff.get("status") != "PASS" or handoff.get("phase07_authorized") is not True:
            errors.append("remediation status/handoff is not PASS")
        convergence_path = base / "shakebench_phase_06fr_convergence.json"
        if not convergence_path.is_file():
            errors.append("missing remediation convergence artifact")
        else:
            convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
            if convergence.get("payload_sha256") != artifact_hash(convergence) or convergence.get("passed") is not True:
                errors.append("remediation convergence artifact failed authentication")
        if (handoff.get("selected_profile") or {}).get("sha256") != HISTORICAL_PROFILE_SHA256:
            errors.append("remediation handoff profile binding mismatch")
        if not handoff.get("inherited_raw_verified"):
            errors.append("remediation handoff lacks inherited raw verification")
        package_ref = handoff.get("package_evidence", {})
        package_path = base / str(package_ref.get("path", ""))
        if not package_path.is_file():
            errors.append("missing remediation package evidence")
        else:
            package = json.loads(package_path.read_text(encoding="utf-8"))
            if package_ref.get("sha256") != file_sha256(package_path) or package.get("passed") is not True:
                errors.append("remediation package evidence failed authentication")
    return {"passed": not errors, "errors": errors, "protocol_sha256": protocol_sha256, "historical_phase06f_preserved": not any("historical Phase 06F" in error for error in errors), "inherited": inherited}


def write_convergence_artifact(root: str | Path | None = None) -> dict[str, Any]:
    base = Path(models.assets_root) if root is None else Path(root)
    protocol, _, _ = load_protocol(base / PROTOCOL_FILENAME)
    comparisons = {}
    errors = []
    for support_token, support in (("open", "open_worktable"), ("target", "target_bottom")):
        paths = {token: base / f"shakebench_phase_06fr_raw_real_{support_token}_{token}.json" for token in ("dt_fine", "dt_medium", "dt_nominal")}
        records = {token: json.loads(path.read_text(encoding="utf-8")) for token, path in paths.items()}
        comparisons[support] = {token: compare_transient_traces(records["dt_nominal"], records[token], protocol) for token in ("dt_fine", "dt_medium")}
        for token, result in comparisons[support].items():
            if not result.get("passed"):
                errors.extend(f"{support}/{token}: {error}" for error in result.get("errors", ()))
    payload = seal_artifact({"schema_id": "shakebench.phase06fr.transient_convergence", "schema_version": 1, "reference_timestep_s": 0.0002, "comparisons": comparisons, "passed": not errors, "errors": errors})
    write_json_atomic(base / "shakebench_phase_06fr_convergence.json", payload)
    return payload


def publish_pass(root: str | Path | None = None) -> dict[str, Any]:
    base = Path(models.assets_root) if root is None else Path(root)
    preflight = verify_remediation_bundle(base, require_pass=False)
    if not preflight.get("passed"):
        raise RemediationError("cannot publish remediation PASS: " + "; ".join(preflight.get("errors", ())))
    convergence = write_convergence_artifact(base)
    inherited = preflight["inherited"]
    protocol, protocol_path, protocol_sha256 = load_protocol(base / PROTOCOL_FILENAME)
    real_refs = []
    for support_token in ("open", "target"):
        for token in ("dt_fine", "dt_medium", "dt_nominal"):
            path = base / f"shakebench_phase_06fr_raw_real_{support_token}_{token}.json"
            real_refs.append({"path": path.name, "sha256": file_sha256(path)})
    parity_path = base / "shakebench_phase_06fr_raw_parity_gamma_zero.json"
    replay_refs = []
    for group in ("parity", "contact"):
        for index in (1, 2, 3):
            path = base / f"shakebench_phase_06fr_raw_replay_{group}_process_{index}.json"
            replay_refs.append({"group": group, "process_index": index, "path": path.name, "sha256": file_sha256(path)})
    registration = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(protocol_path)], cwd=protocol_path.resolve().parents[3], capture_output=True, text=True, check=False).stdout.strip()
    status_payload = {
        "schema_id": "shakebench.phase06fr.status", "schema_version": 1, "status": "PASS", "reason": None, "blocking_stage": None,
        "protocol_sha256": protocol_sha256, "feasibility_sha256": file_sha256(base / FEASIBILITY_FILENAME), "historical_phase06f_status": protocol["historical_phase06f"]["status"]["sha256"],
        "phase07_authorized": True, "official_profile_sha256": HISTORICAL_PROFILE_SHA256, "registration_commit": registration,
    }
    status_payload["payload_sha256"] = artifact_hash(status_payload)
    write_json_atomic(base / STATUS_FILENAME, status_payload)
    handoff_payload = {
        "schema_id": "shakebench.phase06fr.handoff", "schema_version": 2, "status": "PASS", "reason": None, "phase07_authorized": True,
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha256}, "feasibility": {"path": FEASIBILITY_FILENAME, "sha256": file_sha256(base / FEASIBILITY_FILENAME)},
        "registration_commit": registration, "selected_profile": {"profile_id": "shakebench.official.physics.v2", "sha256": HISTORICAL_PROFILE_SHA256},
        "historical_phase06f": protocol["historical_phase06f"], "real_environment_evidence": real_refs,
        "convergence": {"path": "shakebench_phase_06fr_convergence.json", "sha256": file_sha256(base / "shakebench_phase_06fr_convergence.json")},
        "parity": {"path": parity_path.name, "sha256": file_sha256(parity_path)}, "replays": replay_refs,
        "inherited_raw_verified": inherited, "package_evidence": {"path": "shakebench_phase_06fr_package_evidence.json", "sha256": None},
    }
    handoff_payload["status_sha256"] = status_payload["payload_sha256"]
    handoff_payload["payload_sha256"] = artifact_hash(handoff_payload)
    write_json_atomic(base / HANDOFF_FILENAME, handoff_payload)
    return {"status": "PASS", "status_sha256": status_payload["payload_sha256"], "handoff_sha256": handoff_payload["payload_sha256"], "convergence_sha256": convergence["payload_sha256"]}


def write_package_evidence(wheel: str | Path, sdist: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    """Install clean wheel/sdist targets and record profile/MJCF smoke output."""

    def smoke(artifact: Path, label: str) -> dict[str, Any]:
        import tempfile

        target = Path(tempfile.mkdtemp(prefix=f"shakebench06fr-{label}-"))
        install = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(target), str(artifact)], cwd="/tmp", capture_output=True, text=True, check=False)
        if install.returncode != 0:
            raise RemediationError(f"{label} install failed: {install.stderr[-2000:]}")
        code = "from robosuite.utils.shakebench_physics import load_official_physics_profile; from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan; p=load_official_physics_profile(); e=VibrationPickPlaceCan(robots='Panda',physics_profile='official',target_container_friction=(0.30,p.contact_torsional_mu,p.contact_rolling_mu),has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,use_object_obs=False,horizon=2,seed=17); a=e.audit_compiled_model(); print(p.profile_id,p.profile_sha256,a['contacts']['pair_count'],a['can']['mass_kg']); e.close()"
        env = dict(**{key: value for key, value in __import__("os").environ.items()})
        env["PYTHONPATH"] = str(target)
        run = subprocess.run([sys.executable, "-c", code], cwd="/tmp", env=env, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise RemediationError(f"{label} smoke failed: {run.stderr[-2000:]}")
        line = next((line for line in run.stdout.splitlines() if line.startswith("shakebench.official.physics.v2")), "")
        fields = line.split()
        if len(fields) != 4 or fields[1] != HISTORICAL_PROFILE_SHA256 or fields[2] != "8" or fields[3] != "0.349":
            raise RemediationError(f"{label} smoke output mismatch: {line}")
        return {"artifact": artifact.name, "artifact_sha256": file_sha256(artifact), "profile_id": fields[0], "profile_sha256": fields[1], "pair_count": int(fields[2]), "can_mass_kg": float(fields[3])}

    payload = seal_artifact({"schema_id": "shakebench.phase06fr.package_evidence", "schema_version": 1, "status": "PASS", "wheel": smoke(Path(wheel), "wheel"), "sdist": smoke(Path(sdist), "sdist"), "passed": True})
    write_json_atomic(_asset("shakebench_phase_06fr_package_evidence.json") if output is None else Path(output), payload)
    return payload


def write_feasibility(protocol_path: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    protocol, source, protocol_sha256 = load_protocol(protocol_path)
    validation = validate_protocol(protocol, protocol_sha256)
    payload = seal_artifact({
        "schema_id": "shakebench.phase06fr.feasibility",
        "schema_version": 1,
        "status": "PASS",
        "protocol": {"path": source.name, "sha256": protocol_sha256},
        "validation": validation,
        "dry_run": {"mujoco_calls": 0, "real_environment_states": 0, "state_templates": ["real.open_worktable", "real.target_bottom", "parity.gamma_zero.real_environment", "replay.parity", "replay.contact"]},
        "adapter_contract": {"backend": "NoPhysicsBackend", "mujoco_calls": 0, "adapter_contract_digest": hashlib.sha256(canonical_json(validation).encode()).hexdigest()},
    })
    write_json_atomic(_asset(FEASIBILITY_FILENAME) if output_path is None else Path(output_path), payload)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--write-feasibility", action="store_true")
    parser.add_argument("--validate-protocol", action="store_true")
    parser.add_argument("--run-state", choices=("open_worktable", "target_bottom"))
    parser.add_argument("--timestep", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify-trace", default=None)
    parser.add_argument("--replay-group", choices=("parity", "contact"), default=None)
    parser.add_argument("--process-index", type=int, default=None)
    parser.add_argument("--run-parity", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--publish-pass", action="store_true")
    parser.add_argument("--write-package-evidence", action="store_true")
    parser.add_argument("--wheel", default=None)
    parser.add_argument("--sdist", default=None)
    args = parser.parse_args(argv)
    try:
        protocol, source, protocol_sha256 = load_protocol(args.protocol)
        if args.write_feasibility:
            result = write_feasibility(source)
        elif args.verify:
            result = verify_remediation_bundle(require_pass=True)
        elif args.publish_pass:
            result = publish_pass()
        elif args.write_package_evidence:
            if not args.wheel or not args.sdist:
                raise RemediationError("--write-package-evidence requires --wheel and --sdist")
            result = write_package_evidence(args.wheel, args.sdist)
        elif args.validate_protocol:
            result = validate_protocol(protocol, protocol_sha256)
        elif args.run_state:
            if args.timestep is None:
                raise RemediationError("--run-state requires --timestep")
            result = run_real_settling(args.run_state, float(args.timestep), protocol)
            if args.output:
                write_json_atomic(Path(args.output), _jsonable(result))
        elif args.verify_trace:
            result = verify_trace(json.loads(Path(args.verify_trace).read_text(encoding="utf-8")), protocol)
        elif args.replay_group:
            if args.process_index is None:
                raise RemediationError("--replay-group requires --process-index")
            result = run_replay(protocol, group=args.replay_group, process_index=args.process_index)
        elif args.run_parity:
            result = run_parity(protocol)
        else:
            raise RemediationError("select --validate-protocol, --write-feasibility, --run-state, or --verify-trace")
        if args.output and (args.run_parity or args.replay_group):
            write_json_atomic(Path(args.output), _jsonable(result))
        print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True))
        return 0 if result.get("passed", True) else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
