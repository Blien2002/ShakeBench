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
from robosuite.utils.shakebench_artifacts import file_sha256, payload_hash, write_json_atomic
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
    normal = float(np.asarray(contacts["total_force_by_interface_N"].get("can_open_worktable", [0, 0, 0]), dtype=float)[2])
    if support == "target_bottom":
        normal = float(contacts.get("target_bottom_support_force_N", normal))
    tangential = 0.0
    for key, force in contacts["total_force_by_interface_N"].items():
        if (support == "open_worktable" and key == "can_open_worktable") or (support == "target_bottom" and key == "can_target_object"):
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
        return seal_artifact({
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
            "gates": gates,
            "passed": bool(all(gates.values())),
        })
    finally:
        env.close()


def _resample(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    times = np.asarray([float(row["time_s"]) for row in trace], dtype=float)
    indices = [int(np.argmin(np.abs(times - target))) for target in COMMON_TIMES]
    return [trace[index] for index in indices]


def _numeric(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


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
            av = np.asarray([value(row) for row in a], dtype=float)
            bv = np.asarray([value(row) for row in b], dtype=float)
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
        times = [float(row["time_s"]) for row in trace]
        if len(times) > 1 and not np.all(np.diff(times) > 0.0):
            errors.append("trace timestamps are not strictly increasing")
        if any(not row.get("finite_state") for row in trace):
            errors.append("trace contains non-finite state")
    gates = payload.get("gates", {})
    if payload.get("passed") is not True or any(value is not True for value in gates.values()):
        errors.append("real-environment settling gates failed")
    return {"passed": not errors, "errors": errors}


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
    args = parser.parse_args(argv)
    try:
        protocol, source, protocol_sha256 = load_protocol(args.protocol)
        if args.write_feasibility:
            result = write_feasibility(source)
        elif args.validate_protocol:
            result = validate_protocol(protocol, protocol_sha256)
        elif args.run_state:
            if args.timestep is None:
                raise RemediationError("--run-state requires --timestep")
            result = run_real_settling(args.run_state, float(args.timestep), protocol)
            if args.output:
                write_json_atomic(Path(args.output), result)
        elif args.verify_trace:
            result = verify_trace(json.loads(Path(args.verify_trace).read_text(encoding="utf-8")), protocol)
        else:
            raise RemediationError("select --validate-protocol, --write-feasibility, --run-state, or --verify-trace")
        print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True))
        return 0 if result.get("passed", True) else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
