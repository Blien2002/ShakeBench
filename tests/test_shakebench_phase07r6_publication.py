"""Phase 07R6 publication-integrity regression tests."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from robosuite.scripts import shakebench_verify_package as package_verifier
from robosuite.scripts.shakebench_verify_evidence_archive import verify_archive
from robosuite.utils.shakebench_runtime_verifier import verify_runtime_publication_bundle


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _archive(tmp_path: Path, *, extra: tuple[str, bytes] = (), symlink: bool = False) -> tuple[Path, str]:
    stage = tmp_path / "stage"
    (stage / "provenance").mkdir(parents=True)
    (stage / "provenance" / "shakebench_history_rewrite_map_v2.json").write_text("{}", encoding="utf-8")
    (stage / "provenance" / "shakebench_evidence_removed_paths_v1.txt").write_text("", encoding="utf-8")
    (stage / "payload.txt").write_bytes(b"payload")
    files = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "archive_member": path.relative_to(stage).as_posix(),
                    "original_path": path.relative_to(stage).as_posix(),
                    "phase_status": "diagnostic",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                }
            )
    index = {
        "archive_format": "tar.gz",
        "archive_release_tag": "test",
        "archive_sha256": None,
        "expected_asset_url": "https://example.invalid/test",
        "file_count": len(files),
        "files": files,
        "history_rewrite_map_sha256": hashlib.sha256(
            (stage / "provenance" / "shakebench_history_rewrite_map_v2.json").read_bytes()
        ).hexdigest(),
        "path_list_sha256": hashlib.sha256(
            (stage / "provenance" / "shakebench_evidence_removed_paths_v1.txt").read_bytes()
        ).hexdigest(),
        "schema_id": "shakebench.evidence.index",
        "schema_version": 1,
    }
    index["index_sha256"] = hashlib.sha256(_canonical(index)).hexdigest()
    (stage / "index").mkdir()
    (stage / "index" / "evidence_index_v1.json").write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for name, value in extra:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    archive = tmp_path / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(stage.rglob("*")):
            stream.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_runtime_verifier_authenticates_runtime_physics_assets():
    result = verify_runtime_publication_bundle(Path("robosuite/models/assets"))
    assert result["passed"], result["errors"]


def test_runtime_verifier_ignores_release_history_map(tmp_path):
    source = Path("robosuite/models/assets")
    for name in (
        "shakebench_runtime_contract.json",
        "shakebench_history_rewrite_map_v2.json",
        "shakebench_official_physics.yaml",
        "shakebench_phase_06f_protocol.yaml",
        "shakebench_states_dev.json",
    ):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    contract = json.loads((tmp_path / "shakebench_runtime_contract.json").read_text())
    contract["history_rewrite"]["map_sha256"] = "0" * 64
    contract.pop("payload_sha256")
    contract["payload_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    (tmp_path / "shakebench_runtime_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    assert verify_runtime_publication_bundle(tmp_path)["passed"] is True


@pytest.mark.parametrize(
    "stdout,returncode",
    [
        ('{"passed": false}\n', 0),
        ('{"passed": true}\n', 1),
        ("not-json\n", 0),
    ],
)
def test_package_outer_check_requires_successful_true_json(monkeypatch, tmp_path, stdout, returncode):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(package_verifier.subprocess, "run", fake_run)
    result = package_verifier._run_installed_check(tmp_path, tmp_path / "fixture.json", tmp_path / "repo")
    assert result["passed"] is False


def test_archive_requires_detached_authority(tmp_path):
    archive, _ = _archive(tmp_path)
    assert verify_archive(archive)["passed"] is False
    assert "detached" in verify_archive(archive)["errors"][0]


def test_archive_outer_hash_mutation_fails_even_with_null_index_hash(tmp_path):
    archive, expected = _archive(tmp_path)
    assert verify_archive(archive, expected_sha256=expected)["passed"] is True
    archive.write_bytes(archive.read_bytes() + b"mutation")
    assert verify_archive(archive, expected_sha256=expected)["passed"] is False


def test_archive_extra_member_fails_closed(tmp_path):
    archive, expected = _archive(tmp_path, extra=(("extra.txt", b"extra"),))
    assert verify_archive(archive, expected_sha256=expected)["passed"] is False
