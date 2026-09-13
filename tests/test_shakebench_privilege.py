"""Privilege-boundary tests for the State observation API."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.utils.shakebench_privilege import (
    PrivilegedRecorder,
    ShakeBenchPrivilegeError,
    assert_policy_observation_is_clean,
    audit_policy_observation,
    make_privileged_recorder,
)


def test_recorder_is_namespace_scoped_and_defensively_copied():
    received = []
    recorder = PrivilegedRecorder(callback=lambda payload: received.append(payload))
    mutable = np.array([1.0, 2.0])
    record = recorder.record({"privileged_signal": mutable})
    mutable[0] = 99.0
    record["privileged_signal"][1] = 99.0
    assert recorder.latest["privileged_signal"].tolist() == [1.0, 2.0]
    assert received[0]["privileged_signal"].tolist() == [1.0, 2.0]
    assert recorder.to_dict()["namespace"] == "privileged_"
    assert len(recorder) == 1

    with pytest.raises(ShakeBenchPrivilegeError, match="privileged_"):
        recorder.record({"clean_signal": np.zeros(1)})
    with pytest.raises(ShakeBenchPrivilegeError, match="not a flag"):
        make_privileged_recorder(True)


def test_recorder_rejects_uncopyable_values_and_separates_nested_copies():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("deliberately uncopyable")

    recorder = PrivilegedRecorder()
    with pytest.raises(ShakeBenchPrivilegeError, match="defensively copy"):
        recorder.record({"privileged_bad": Uncopyable()})

    source = {"privileged_nested": {"array": np.array([1.0, 2.0]), "items": [{"x": 3.0}]}}
    returned = recorder.record(source)
    latest = recorder.latest
    history = recorder.records[0]
    callback = []
    separate = PrivilegedRecorder(callback=lambda value: callback.append(value))
    separate_returned = separate.record(source)
    source["privileged_nested"]["array"][0] = 99.0
    returned["privileged_nested"]["array"][1] = 98.0
    latest["privileged_nested"]["items"][0]["x"] = 97.0
    history["privileged_nested"]["array"][0] = 96.0
    separate_returned["privileged_nested"]["array"][0] = 95.0
    callback[0]["privileged_nested"]["array"][0] = 94.0
    assert recorder.latest["privileged_nested"]["array"].tolist() == [1.0, 2.0]
    assert recorder.latest["privileged_nested"]["items"][0]["x"] == 3.0
    assert separate.latest["privileged_nested"]["array"].tolist() == [1.0, 2.0]


def test_policy_audit_rejects_only_privileged_namespace_keys():
    clean = audit_policy_observation({"table_imu_window": np.zeros((10, 6))})
    assert clean.passed
    assert_policy_observation_is_clean({"ordinary": np.zeros(1)})
    leaked = audit_policy_observation({"ordinary": 1, "privileged_bias": np.zeros(6)})
    assert not leaked.passed
    assert leaked.leaked_keys == ("privileged_bias",)
    with pytest.raises(ShakeBenchPrivilegeError, match="privileged_bias"):
        assert_policy_observation_is_clean({"privileged_bias": np.zeros(6)})
