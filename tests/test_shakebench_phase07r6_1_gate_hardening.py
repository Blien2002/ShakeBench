"""Fail-closed regression tests for the Phase 07R6.1 authority gates."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from robosuite.scripts import shakebench_audit_evidence as audit
from robosuite.scripts import shakebench_build_evidence_index as index_builder
from robosuite.scripts.shakebench_verify_evidence_archive import verify_root
from robosuite.scripts.shakebench_verify_phase07_handoff import validate_package_evidence


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _package_payload(*, wheel_passed=True, sdist_passed=True, source_leak=False):
    records = []
    for kind, passed in (("wheel", wheel_passed), ("sdist", sdist_passed)):
        filename = f"robosuite-test-{kind}"
        records.append(
            {
                "kind": kind,
                "artifact_filename": filename,
                "artifact_bytes": 10,
                "artifact_sha256": "a" * 64 if kind == "wheel" else "b" * 64,
                "content_check": {"passed": passed},
                "check": {
                    "passed": passed,
                    "install_root": f"/tmp/install-{kind}",
                    "module_paths": {"robosuite": f"/tmp/install-{kind}/robosuite/__init__.py"},
                    "source_checkout_in_import_path": source_leak,
                },
                "passed": passed,
            }
        )
    return {
        "schema_id": "shakebench.phase07.package_evidence",
        "schema_version": 2,
        "records": records,
        "passed": all(record["passed"] for record in records),
    }


def _package_manifest(payload):
    return {
        "sha256": _digest(_canonical(payload)),
        "wheel_filename": "robosuite-test-wheel",
        "wheel_bytes": 10,
        "wheel_sha256": "a" * 64,
        "sdist_filename": "robosuite-test-sdist",
        "sdist_bytes": 10,
        "sdist_sha256": "b" * 64,
    }


def test_package_authority_requires_complete_true_records():
    payload = _package_payload()
    assert validate_package_evidence(payload, _package_manifest(payload)) == []

    payload = _package_payload(wheel_passed=False)
    assert validate_package_evidence(payload, _package_manifest(payload))

    payload = _package_payload(source_leak=True)
    assert validate_package_evidence(payload, _package_manifest(payload))


@pytest.mark.parametrize("field", ["payload_sha256", "trace_sha256", "trace_whole_digest"])
def test_index_binds_declared_digest_fields(tmp_path, field):
    path = tmp_path / "run.json"
    payload = {"schema_id": "shakebench.phase07.oracle_run", field: "c" * 64}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert index_builder._json_bindings(path)[field] == "c" * 64


def test_index_rejects_malformed_declared_digest(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"schema_id": "shakebench.phase07.oracle_run", "payload_sha256": "bad"}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        index_builder._json_bindings(path)


def test_index_does_not_invent_bindings_for_ordinary_json(tmp_path):
    path = tmp_path / "ordinary.json"
    path.write_text(json.dumps({"schema_id": "ordinary", "name": "payload"}), encoding="utf-8")
    assert not {"payload_sha256", "trace_sha256", "trace_whole_digest"} & index_builder._json_bindings(path).keys()


def test_root_verifier_enumerates_real_members(tmp_path):
    root = tmp_path / "root"
    (root / "index").mkdir(parents=True)
    payload = root / "payload.txt"
    payload.write_bytes(b"payload")
    index = {
        "schema_id": "shakebench.evidence.index",
        "schema_version": 1,
        "archive_sha256": None,
        "files": [
            {
                "archive_member": "payload.txt",
                "sha256": _digest(b"payload"),
                "size": 7,
            }
        ],
    }
    index["index_sha256"] = _digest(_canonical(index))
    (root / "index" / "evidence_index_v1.json").write_text(json.dumps(index), encoding="utf-8")
    assert verify_root(root)["passed"] is True
    (root / "extra.txt").write_bytes(b"extra")
    assert verify_root(root)["passed"] is False
    index["files"].append({"archive_member": "extra.txt", "sha256": _digest(b"extra"), "size": 5})
    index.pop("index_sha256")
    index["index_sha256"] = _digest(_canonical(index))
    (root / "index" / "evidence_index_v1.json").write_text(json.dumps(index), encoding="utf-8")
    assert verify_root(root)["passed"] is True


def test_scientific_only_root_audit_never_claims_release_authority(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(audit, "_verify_index", lambda *args, **kwargs: {"index_sha256": "x"})
    monkeypatch.setattr(audit, "_run_scientific_audit", lambda _: {"passed": True, "errors": []})
    result = audit.audit_root(root)
    assert result["passed"] is True
    assert result["release_authority_verified"] is False


def test_package_evidence_can_be_read_from_a_safe_archive(tmp_path):
    payload = _package_payload()
    package_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    archive = tmp_path / "gate.tar.gz"
    source = tmp_path / "package_evidence_final.json"
    source.write_bytes(package_bytes)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="provenance/package_evidence_final.json")
    assert archive.is_file()
