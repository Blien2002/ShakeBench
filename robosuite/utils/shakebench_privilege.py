"""Explicit privileged recorder and policy-observation isolation helpers.

The recorder is intentionally not an ``Observable`` and is not installed as a
wrapper attribute that participates in observation construction.  It accepts
only names in the ``privileged_`` namespace, stores defensive copies, and can
forward each snapshot to an explicit user callback.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Optional

import numpy as np


PRIVILEGED_NAMESPACE = "privileged_"


class ShakeBenchPrivilegeError(ValueError):
    """Raised when privileged data crosses the explicit recorder boundary."""


PrivilegeError = ShakeBenchPrivilegeError


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return np.array(value, copy=True)
        copied = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            copied[index] = _copy_value(value[index])
        return copied
    if isinstance(value, np.generic):
        return value.copy() if hasattr(value, "copy") else value.item()
    if isinstance(value, Mapping):
        return {_copy_value(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [_copy_value(item) for item in value]
        return type(value)(copied)
    if isinstance(value, set):
        return {_copy_value(item) for item in value}
    if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
        return value
    try:
        copied = deepcopy(value)
    except Exception as exc:
        raise ShakeBenchPrivilegeError(
            f"privileged value of type {type(value).__name__} is not defensively copyable"
        ) from exc
    if copied is value:
        raise ShakeBenchPrivilegeError(f"privileged value of type {type(value).__name__} was not defensively copied")
    return copied


def _serializable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _serializable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_value(item) for item in value]
    return value


def _validate_privileged_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ShakeBenchPrivilegeError("privileged snapshot must be a mapping")
    result = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.startswith(PRIVILEGED_NAMESPACE):
            raise ShakeBenchPrivilegeError(f"privileged snapshot key {key!r} must start with {PRIVILEGED_NAMESPACE!r}")
        result[key] = _copy_value(value)
    return result


@dataclass(frozen=True)
class PrivilegeAudit:
    """Immutable result of a policy-boundary key audit."""

    passed: bool
    leaked_keys: tuple[str, ...]
    policy_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "leaked_keys": list(self.leaked_keys),
            "policy_keys": list(self.policy_keys),
        }


def audit_policy_observation(observation: Mapping[str, Any]) -> PrivilegeAudit:
    """Check that an ordinary observation mapping has no privileged namespace."""

    if not isinstance(observation, Mapping):
        raise ShakeBenchPrivilegeError("policy observation must be a mapping")
    policy_keys = tuple(str(key) for key in observation)
    leaked = tuple(sorted(key for key in policy_keys if key.startswith(PRIVILEGED_NAMESPACE)))
    return PrivilegeAudit(passed=not leaked, leaked_keys=leaked, policy_keys=policy_keys)


def assert_policy_observation_is_clean(observation: Mapping[str, Any]) -> None:
    audit = audit_policy_observation(observation)
    if not audit.passed:
        raise ShakeBenchPrivilegeError(
            "privileged fields leaked into policy observation: " + ", ".join(audit.leaked_keys)
        )


assert_no_privileged_leak = assert_policy_observation_is_clean
audit_privilege_boundary = audit_policy_observation


class PrivilegedRecorder:
    """In-memory recorder for full simulator truth.

    ``callback`` is optional and must be supplied explicitly by the caller. It
    receives a defensive copy of the already namespace-validated snapshot.
    The callback's return value is ignored, which prevents callback results
    from being inserted into policy observations accidentally.
    """

    def __init__(self, callback: Optional[Callable[[Mapping[str, Any]], Any]] = None) -> None:
        if callback is not None and not callable(callback):
            raise ShakeBenchPrivilegeError("privileged recorder callback must be callable")
        self.callback = callback
        self._records: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._records = []

    def record(self, snapshot: Optional[Mapping[str, Any]] = None, **fields: Any) -> dict[str, Any]:
        if snapshot is None:
            snapshot = {}
        if fields:
            if snapshot:
                combined = dict(snapshot)
                combined.update(fields)
            else:
                combined = fields
        else:
            combined = snapshot
        validated = _validate_privileged_mapping(combined)
        self._records.append(validated)
        callback_payload = _copy_value(validated)
        if self.callback is not None:
            self.callback(callback_payload)
        return _copy_value(validated)

    record_step = record
    snapshot = record

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(_copy_value(record) for record in self._records)

    @property
    def latest(self) -> Optional[dict[str, Any]]:
        return None if not self._records else _copy_value(self._records[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": PRIVILEGED_NAMESPACE,
            "record_count": len(self._records),
            "records": [_serializable_value(record) for record in self._records],
        }

    def __len__(self) -> int:
        return len(self._records)


def make_privileged_recorder(value: Any) -> Optional[PrivilegedRecorder]:
    """Normalize the explicit environment recorder/callback argument."""

    if value is None:
        return None
    if isinstance(value, PrivilegedRecorder):
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ShakeBenchPrivilegeError("privileged_recorder must be an explicit recorder or callback, not a flag")
    if callable(value):
        return PrivilegedRecorder(callback=value)
    raise ShakeBenchPrivilegeError("privileged_recorder must be a PrivilegedRecorder or callable callback")


def namespace_snapshot(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix a recorder-owned field mapping and validate it."""

    if not isinstance(fields, Mapping):
        raise ShakeBenchPrivilegeError("fields must be a mapping")
    prefixed = {}
    for key, value in fields.items():
        if not isinstance(key, str):
            raise ShakeBenchPrivilegeError("privileged field names must be strings")
        prefixed[key if key.startswith(PRIVILEGED_NAMESPACE) else PRIVILEGED_NAMESPACE + key] = value
    return _validate_privileged_mapping(prefixed)


PHASE05_OBSERVATION_ARTIFACT_SCHEMA_ID = "shakebench.phase05.observation"
PHASE05_OBSERVATION_ARTIFACT_SCHEMA_VERSION = 1


def phase05_observation_artifact_hash(payload: Mapping[str, Any]) -> str:
    """Hash an observation artifact while excluding only its self-hash."""

    if not isinstance(payload, Mapping):
        raise ShakeBenchPrivilegeError("Phase 05 observation artifact must be a mapping")
    normalized = json.loads(json.dumps(dict(payload), sort_keys=True, ensure_ascii=True, allow_nan=False))
    lock = normalized.get("artifact_lock")
    if isinstance(lock, dict):
        lock.pop("payload_sha256", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_phase05_observation_artifact(path: Any) -> dict[str, Any]:
    """Read-only verification of the Phase 05R controlled evidence artifact."""

    checks = {
        "schema": False,
        "integrity": False,
        "exact_policy_keys": False,
        "imu_mount": False,
        "imu_profile": False,
        "timeline": False,
        "quantizer": False,
        "noise_filter": False,
        "v2_v3": False,
        "privilege_isolation": False,
    }
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        checks["schema"] = (
            payload.get("schema_id") == PHASE05_OBSERVATION_ARTIFACT_SCHEMA_ID
            and payload.get("schema_version") == PHASE05_OBSERVATION_ARTIFACT_SCHEMA_VERSION
            and payload.get("phase") == "05R"
            and payload.get("status") == "PASS"
        )
        lock = payload.get("artifact_lock", {})
        expected_hash = lock.get("payload_sha256") if isinstance(lock, dict) else None
        checks["integrity"] = (
            isinstance(lock, dict)
            and isinstance(lock.get("update_reason"), str)
            and bool(lock["update_reason"].strip())
            and isinstance(lock.get("previous_payload_sha256"), str)
            and len(lock["previous_payload_sha256"]) == 64
            and isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and expected_hash == phase05_observation_artifact_hash(payload)
        )
        from robosuite.utils.shakebench_providers import COMMON_STATE_KEYS, TIER_POLICY_KEYS

        artifact_keys = payload.get("exact_policy_key_sets", {})
        checks["exact_policy_keys"] = all(
            isinstance(artifact_keys.get(tier), list)
            and set(artifact_keys[tier]) == set(COMMON_STATE_KEYS) | set(TIER_POLICY_KEYS[tier])
            and len(artifact_keys[tier]) == len(set(artifact_keys[tier]))
            for tier in ("V0", "V1", "V2", "V3")
        )
        mount = payload.get("imu_mount", {})
        checks["imu_mount"] = (
            mount.get("sensor_body_name") == "robot0_base"
            and mount.get("sensor_frame_parent") == "robot_base"
            and mount.get("deck_body_name") == "deck"
            and mount.get("parent_body_name") == "deck"
            and np.allclose(mount.get("sensor_position_m_in_robot_base"), [0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
            and np.allclose(
                mount.get("sensor_quaternion_wxyz_in_robot_base"),
                [1.0, 0.0, 0.0, 0.0],
                rtol=0.0,
                atol=1e-12,
            )
            and np.linalg.norm(np.asarray(mount.get("robot_base_pose_in_deck"), dtype=float)[:3]) > 0.0
        )
        from robosuite.utils.shakebench_sensors import CANONICAL_IMU_PROFILE, canonical_imu_profile_hash

        profile = payload.get("imu_profile", {})
        checks["imu_profile"] = (
            profile.get("profile_id") == CANONICAL_IMU_PROFILE.profile_id
            and profile.get("profile_hash") == canonical_imu_profile_hash()
            and profile.get("sample_rate_hz") == 200.0
            and profile.get("window_shape") == [10, 6]
            and profile.get("delivery_delay_samples") == 1
        )
        timeline = payload.get("timeline", {})
        checks["timeline"] = (
            timeline.get("reset_window_acquisition_s")
            == [-0.05, -0.045, -0.04, -0.035, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005]
            and timeline.get("reset_pending_delivery_s") == [0.0]
            and timeline.get("first_step_window_delivered_s")
            == [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
            and timeline.get("first_step_acquisition_s")
            == [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
            and timeline.get("first_step_pending_delivery_s") == [0.05]
        )
        quantizer = payload.get("quantizer", {})
        checks["quantizer"] = (
            quantizer.get("code_min") == -32768
            and quantizer.get("code_max") == 32767
            and quantizer.get("accel_plus_endpoint_code") == 32767
            and quantizer.get("accel_minus_endpoint_code") == -32768
            and quantizer.get("gyro_plus_endpoint_code") == 32767
            and quantizer.get("gyro_minus_endpoint_code") == -32768
            and quantizer.get("round_trip_int16") is True
        )
        checks["noise_filter"] = payload.get("noise_filter", {}).get("passed") is True
        checks["v2_v3"] = (
            payload.get("v2_v3", {}).get("current_only") is True
            and payload.get("v2_v3", {}).get("independent_reconstruction") is True
            and payload.get("v2_v3", {}).get("no_future_realized_state") is True
        )
        checks["privilege_isolation"] = payload.get("privilege_isolation", {}).get("passed") is True
    except (OSError, TypeError, ValueError, KeyError, IndexError, AttributeError, json.JSONDecodeError):
        pass
    return {
        "passed": bool(all(checks.values())),
        "integrity_valid": bool(checks["integrity"]),
        "checks": checks,
    }


__all__ = [
    "PRIVILEGED_NAMESPACE",
    "PrivilegeAudit",
    "PrivilegedRecorder",
    "PrivilegeError",
    "ShakeBenchPrivilegeError",
    "assert_policy_observation_is_clean",
    "assert_no_privileged_leak",
    "audit_policy_observation",
    "audit_privilege_boundary",
    "make_privileged_recorder",
    "namespace_snapshot",
    "PHASE05_OBSERVATION_ARTIFACT_SCHEMA_ID",
    "PHASE05_OBSERVATION_ARTIFACT_SCHEMA_VERSION",
    "phase05_observation_artifact_hash",
    "verify_phase05_observation_artifact",
]
