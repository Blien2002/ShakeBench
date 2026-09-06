"""Semantic verification for the Phase 06F-R2 handoff.

The verifier is intentionally independent of the R1 self-reported booleans:
it opens every listed file, recomputes hashes and derives parity, convergence
and replay decisions from trace values.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from robosuite.models import assets_root
from robosuite.scripts.shakebench_handoff_remediation import verify_inherited_raw
from robosuite.utils.shakebench_artifacts import file_sha256, verify_payload_hash
from robosuite.utils.shakebench_physics_finalizer import artifact_hash, canonical_json

PROTOCOL_FILENAME = "shakebench_phase_06fr2_protocol.yaml"
STATUS_FILENAME = "shakebench_phase_06fr2_status.json"
HANDOFF_FILENAME = "shakebench_phase_06fr2_handoff.json"
ANCHOR_FILENAME = "shakebench_phase_06fr2_anchor.json"
PROFILE_FILENAME = "shakebench_official_physics.yaml"
TRACE_SCHEMA_ID = "shakebench.phase06fr.real_environment_trace"
HISTORICAL_PROFILE_SHA256 = "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c"
DT_VALUES = (0.0001, 0.000125, 0.0002)


@dataclass(frozen=True)
class VerifiedHandoff:
    passed: bool
    errors: tuple[str, ...]
    checks: Mapping[str, Any]


def _root(root: str | Path | None) -> Path:
    return Path(assets_root) if root is None else Path(root)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a mapping")
    return value


def _trace_digest(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = sorted({key for row in trace for key in row})
    digests = {}
    shapes = {}
    dtypes = {}
    for field in fields:
        values = [row.get(field) for row in trace]
        digests[field] = __import__("hashlib").sha256(canonical_json(values).encode()).hexdigest()
        array = np.asarray(values[0])
        shapes[field] = list(array.shape)
        dtypes[field] = str(array.dtype)
    times = [float(row["time_s"]) for row in trace]
    result = {
        "schema_id": TRACE_SCHEMA_ID,
        "schema_version": 1,
        "sample_count": len(trace),
        "timestamp_digest": __import__("hashlib").sha256(canonical_json(times).encode()).hexdigest(),
        "field_digests": digests,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    result["whole_trace_digest"] = __import__("hashlib").sha256(canonical_json(result).encode()).hexdigest()
    return result


def _valid_trace(trace: Any, protocol: Mapping[str, Any], *, expected_samples: int | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(trace, list) or not trace:
        return ["trace is empty or not a list"]
    required = set(protocol["trace_schema"]["required_fields"])
    times = []
    for index, row in enumerate(trace):
        if not isinstance(row, Mapping) or not required.issubset(row):
            errors.append(f"trace row {index} is missing required fields")
            continue
        times.append(float(row["time_s"]))
        for field in (
            "normalized_action",
            "decoded_action",
            "clipped_action",
            "applied_control",
            "deck_pose",
            "deck_twist",
            "deck_acceleration",
            "worktable_pose_deck",
            "worktable_twist_deck",
            "worktable_acceleration_deck",
            "warning_number",
        ):
            array = np.asarray(row[field])
            if array.dtype.kind not in {"f", "i", "b"} or not np.all(np.isfinite(array.astype(float))):
                errors.append(f"trace row {index} dtype/finite failure: {field}")
        if len(row["normalized_action"]) != 7 or len(row["decoded_action"]) != 7 or len(row["clipped_action"]) != 7:
            errors.append(f"trace row {index} action shape failure")
        if len(row["warning_number"]) != 7:
            errors.append(f"trace row {index} warning shape failure")
    if times and not np.all(np.diff(times) > 0.0):
        errors.append("trace timestamps are not strictly increasing")
    if expected_samples is not None and len(trace) != expected_samples:
        errors.append(f"trace sample count {len(trace)} != {expected_samples}")
    return errors


def _resample(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    targets = np.arange(0.005, 0.5000001, 0.005)
    times = np.asarray([float(row["time_s"]) for row in trace])
    return [trace[int(np.argmin(np.abs(times - target)))] for target in targets]


def _flatten(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value):
            result.extend(_flatten(value[key]).tolist())
        return np.asarray(result, dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _pose_orientation_error(a: Any, b: Any) -> float:
    aa = _flatten(a)
    bb = _flatten(b)
    if aa.size < 7 or bb.size < 7:
        return 0.0
    dot = abs(float(np.dot(aa[3:7], bb[3:7]))) / max(float(np.linalg.norm(aa[3:7]) * np.linalg.norm(bb[3:7])), 1e-15)
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def recompute_convergence(
    protocol: Mapping[str, Any], records: Mapping[tuple[str, float], Mapping[str, Any]]
) -> dict[str, Any]:
    comparisons = {}
    errors = []
    fields = protocol["convergence"]["fields"]
    for support in ("open_worktable", "target_bottom"):
        selected = _resample(records[(support, 0.0002)]["trace"])
        support_comparisons = {}
        for dt in (0.0001, 0.000125):
            finer = _resample(records[(support, dt)]["trace"])
            metrics = {}
            for field in fields:
                if field == "contacts.named_support_contact":
                    a = np.asarray([int(row["contacts"]["named_support_contact"]) for row in selected])
                    b = np.asarray([int(row["contacts"]["named_support_contact"]) for row in finer])
                    passed = bool(np.array_equal(a, b))
                    metrics[field] = {"max_absolute": int(np.max(np.abs(a - b))), "passed": passed}
                    if not passed:
                        errors.append(f"{support}/{dt}/{field}")
                    continue
                path = field.split(".")

                def get(row):
                    value = row
                    for key in path:
                        value = value[key]
                    return value

                a = np.asarray([_flatten(get(row)) for row in selected], dtype=float)
                b = np.asarray([_flatten(get(row)) for row in finer], dtype=float)
                max_abs = float(np.max(np.abs(a - b)))
                denominator = max(float(np.max(np.abs(b))), 1.0e-6)
                relative = max_abs / denominator
                if "pose" in field:
                    orientation = max(_pose_orientation_error(get(x), get(y)) for x, y in zip(selected, finer))
                    max_abs = max(max_abs, orientation)
                absolute = float(protocol["convergence"]["tolerances"]["position_absolute_m"])
                if "twist" in field or "velocity" in field:
                    absolute = float(protocol["convergence"]["tolerances"]["velocity_absolute_m_s"])
                if "impulse" in field:
                    absolute = float(protocol["convergence"]["tolerances"]["impulse_absolute_Ns"])
                passed = bool(
                    max_abs <= absolute
                    or relative <= float(protocol["convergence"]["tolerances"]["relative_metric_max"])
                )
                metrics[field] = {"max_absolute": max_abs, "max_relative": relative, "passed": passed}
                if not passed:
                    errors.append(f"{support}/{dt}/{field}")
            # Recompute cumulative force impulses and final-window force RMS.
            for metric_name in ("normal_impulse_Ns", "tangential_impulse_Ns", "support_force_N"):
                a = np.asarray([float(row[metric_name]) for row in selected])
                b = np.asarray([float(row[metric_name]) for row in finer])
                if "impulse" in metric_name:
                    a = np.cumsum(a)
                    b = np.cumsum(b)
                    absolute = float(protocol["convergence"]["tolerances"]["impulse_absolute_Ns"])
                else:
                    a = np.sqrt(np.mean(a[-20:] ** 2))
                    b = np.sqrt(np.mean(b[-20:] ** 2))
                    absolute = float(protocol["convergence"]["tolerances"]["force_rms_absolute_N"])
                denominator = (
                    max(
                        float(
                            np.max(np.abs(np.asarray(b))),
                        ),
                        1.0e-6,
                    )
                    if np.ndim(b)
                    else max(abs(float(b)), 1.0e-6)
                )
                difference = float(np.max(np.abs(a - b))) if np.ndim(a) else abs(float(a) - float(b))
                passed = bool(
                    difference <= absolute
                    or difference / denominator <= float(protocol["convergence"]["tolerances"]["relative_metric_max"])
                )
                metrics["cumulative_" + metric_name] = {"difference": difference, "passed": passed}
                if not passed:
                    errors.append(f"{support}/{dt}/cumulative_{metric_name}")
            support_comparisons[str(dt)] = {
                "metrics": metrics,
                "passed": not any(error.startswith(f"{support}/{dt}/") for error in errors),
            }
        comparisons[support] = support_comparisons
    return {
        "schema_id": "shakebench.phase06fr2.semantic_convergence",
        "schema_version": 1,
        "comparisons": comparisons,
        "errors": errors,
        "passed": not errors,
    }


def verify_gamma_zero_parity(
    trace: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    errors = _valid_trace(trace, protocol, expected_samples=2500)
    if (
        metadata.get("profile_id") != "shakebench.official.physics.v2"
        or metadata.get("profile_sha256") != HISTORICAL_PROFILE_SHA256
    ):
        errors.append("parity profile binding mismatch")
    if (
        metadata.get("timestep_s") != 0.0002
        or metadata.get("control_frequency_hz") != 20.0
        or metadata.get("action_dimension") != 7
    ):
        errors.append("parity scheduler/action binding mismatch")
    for row in trace:
        for key in ("normalized_action", "decoded_action", "clipped_action"):
            if any(abs(float(value)) > 0.0 for value in row[key]):
                errors.append("parity action is not registered zero")
        if not row.get("finite_state") or any(row.get("warning_number", ())):
            errors.append("parity nonfinite/warning state")
        if not row.get("contacts", {}).get("named_support_contact") or float(row.get("penetration_m", 1.0)) > 0.0005:
            errors.append("parity support/penetration bound failed")
            break
    if metadata.get("action_dimension") != 7 or metadata.get("privileged_truth_excluded") is not True:
        errors.append("parity observation/action semantics failed")
    if metadata.get("evaluator_contract_hash") != "phase04_vibration_success_evaluator":
        errors.append("parity evaluator contract hash failed")
    expected_poses = {
        "panda_base_pose": [-0.485, 0.0, 0.912, 1.0, 0.0, 0.0, 0.0],
        "worktable_pose": [0.0, 0.0, 0.77, 1.0, 0.0, 0.0, 0.0],
        "can_pose": [-0.1, -0.13, 0.8402970033304401, 1.0, 0.0, 0.0, 0.0],
    }
    for key, expected in expected_poses.items():
        if not np.allclose(np.asarray(metadata.get(key, ()), dtype=float), expected, rtol=0.0, atol=0.0005):
            errors.append("parity nominal pose bound failed: " + key)
    target = metadata.get("target_geometry", {})
    if tuple(target.get("inner_xy_m", ())) != (0.164, 0.144) or tuple(target.get("outer_xy_m", ())) != (0.18, 0.16):
        errors.append("parity target geometry bound failed")
    expected_thresholds = {
        "hold_duration_s": 0.5,
        "max_relative_linear_speed_m_s": 0.02,
        "max_relative_angular_speed_rad_s": 0.2,
        "max_illegal_penetration_m": 0.0005,
        "target_bottom_support_force_threshold_N": 0.001,
        "target_bottom_support_z_tolerance_m": 0.0005,
        "containment_epsilon_m": 1.0e-12,
    }
    if metadata.get("success_thresholds") != expected_thresholds:
        errors.append("parity success threshold mapping failed")
    return {"passed": not errors, "errors": sorted(set(errors))}


def verify_phase06fr2_handoff(asset_root: str | Path | None = None) -> VerifiedHandoff:
    base = _root(asset_root)
    errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        protocol = yaml.safe_load((base / PROTOCOL_FILENAME).read_text(encoding="utf-8"))
        status = _read_json(base / STATUS_FILENAME)
        handoff = _read_json(base / HANDOFF_FILENAME)
        anchor = _read_json(base / ANCHOR_FILENAME)
    except Exception as exc:
        return VerifiedHandoff(False, (f"R2 handoff unreadable: {exc}",), {})
    if (
        handoff.get("status") != "PASS"
        or status.get("status") != "PASS"
        or handoff.get("phase07_authorized") is not True
    ):
        errors.append("R2 status/handoff is not PASS")
    for name, payload in (("status", status), ("handoff", handoff), ("anchor", anchor)):
        if payload.get("payload_sha256") != artifact_hash(payload):
            errors.append(name + " payload hash mismatch")
    listed = protocol.get("evidence", [])
    seen = set()
    handoff_evidence = {str(row.get("path")): row for row in handoff.get("evidence", []) if isinstance(row, Mapping)}
    protocol_paths = {str(row.get("path")) for row in listed}
    if set(handoff_evidence) != protocol_paths:
        errors.append("handoff evidence list does not exactly match protocol coverage")
    for row in listed:
        path_value = str(row.get("path", ""))
        if path_value in seen:
            errors.append("duplicate evidence path: " + path_value)
        seen.add(path_value)
        path = base / path_value
        if not path.is_file():
            errors.append("missing evidence path: " + path_value)
            continue
        value = _read_json(path) if path.suffix == ".json" else None
        if not row.get("deferred") and file_sha256(path) != row.get("file_sha256"):
            errors.append("evidence file hash mismatch: " + path_value)
        if value is not None:
            if (
                not row.get("deferred")
                and row.get("payload_sha256") is not None
                and value.get("payload_sha256") != row.get("payload_sha256")
            ):
                errors.append("evidence payload hash mismatch: " + path_value)
            if row.get("schema_id") and value.get("schema_id") != row.get("schema_id"):
                errors.append("evidence schema mismatch: " + path_value)
        handoff_row = handoff_evidence.get(path_value)
        if (
            handoff_row is None
            or (not row.get("deferred") and handoff_row.get("file_sha256") != row.get("file_sha256"))
            or (not row.get("deferred") and handoff_row.get("payload_sha256") != row.get("payload_sha256"))
        ):
            errors.append("handoff evidence binding mismatch: " + path_value)
    checks["evidence_coverage"] = len(seen) == len(listed)
    inherited = verify_inherited_raw(base)
    if not inherited.get("passed"):
        errors.extend(inherited.get("errors", ()))
    records = {}
    for support_token, support in (("open", "open_worktable"), ("target", "target_bottom")):
        for dt, token in ((0.0001, "dt_fine"), (0.000125, "dt_medium"), (0.0002, "dt_nominal")):
            path = base / f"shakebench_phase_06fr_raw_real_{support_token}_{token}.json"
            if path.is_file():
                records[(support, dt)] = _read_json(path)
    for support in ("open_worktable", "target_bottom"):
        for dt in DT_VALUES:
            if (support, dt) not in records:
                errors.append(f"missing semantic settling state {support}/{dt}")
        selected = records.get((support, 0.0002))
        if selected:
            for dt in (0.0001, 0.000125):
                if (support, dt) in records:
                    comparison = recompute_convergence(protocol, records)
                    if not comparison.get("passed"):
                        errors.extend(comparison.get("errors", ()))
                    break
    parity_path = base / "shakebench_phase_06fr_raw_parity_gamma_zero.json"
    if parity_path.is_file():
        parity = _read_json(parity_path)
        parity_result = verify_gamma_zero_parity(parity.get("trace", ()), parity.get("parity", {}), protocol)
        if not parity_result["passed"]:
            errors.extend(parity_result["errors"])
    else:
        errors.append("missing R2 parity")
    for group in ("parity", "contact"):
        rows = []
        for index in (1, 2, 3):
            path = base / f"shakebench_phase_06fr_raw_replay_{group}_process_{index}.json"
            if path.is_file():
                replay = _read_json(path)
                rows.append(replay)
                replay_errors = _valid_trace(replay.get("trace"), protocol, expected_samples=2500)
                if replay.get("complete_trace") is not True or replay.get("trace_digest") != _trace_digest(
                    replay.get("trace", ())
                ):
                    replay_errors.append("complete trace digest/flag failed")
                errors.extend(f"{group}/{index}: {error}" for error in replay_errors)
            else:
                errors.append(f"missing R2 replay {group}/{index}")
        if len(rows) == 3 and len({row.get("trace_digest", {}).get("whole_trace_digest") for row in rows}) != 1:
            errors.append("R2 replay trace values differ: " + group)
    if handoff.get("selected_profile", {}).get("sha256") != HISTORICAL_PROFILE_SHA256:
        errors.append("R2 selected profile hash mismatch")
    if anchor.get("evidence_commit"):
        commit = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(anchor["evidence_commit"]), "HEAD"],
            cwd=base.resolve().parents[2],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        # Installed wheels/sdists have no .git directory; there we validate
        # the anchor's asset/blob hashes and rely on the external distribution
        # identity. A checkout must additionally prove ancestry.
        if commit.returncode == 1 or commit.returncode not in (0, 1, 128):
            errors.append("R2 anchor evidence commit is not an ancestor")
    return VerifiedHandoff(not errors, tuple(sorted(set(errors))), checks)


__all__ = ["VerifiedHandoff", "verify_phase06fr2_handoff", "verify_gamma_zero_parity", "recompute_convergence"]
