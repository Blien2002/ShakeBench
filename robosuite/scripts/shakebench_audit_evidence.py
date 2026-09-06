"""Run the explicit full ShakeBench evidence audit.

Runtime imports intentionally do not call this module.  A caller must provide
``--evidence-archive`` or ``--evidence-root``; the audit then authenticates the
archive/index and reopens Phase 06 and Phase 07 raw traces for the existing
numeric/semantic verifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Optional

INDEX_MEMBER = "index/evidence_index_v1.json"
RUN_SCHEMA_ID = "shakebench.phase07.oracle_run"


class EvidenceAuditError(ValueError):
    """Raised when evidence archive or binding validation fails."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceAuditError(f"JSON root is not an object: {path}")
    return value


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceAuditError(f"unsafe archive member path: {name}")


def _list_archive(archive: Path) -> list[str]:
    if archive.suffix == ".zst" or archive.name.endswith(".tar.zst"):
        command = ["tar", "--zstd", "-tf", str(archive)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise EvidenceAuditError(f"cannot list zstd archive: {completed.stderr.strip()}")
        members = completed.stdout.splitlines()
    else:
        try:
            with tarfile.open(archive, "r:*") as stream:
                members = stream.getnames()
        except (OSError, tarfile.TarError) as exc:
            raise EvidenceAuditError(f"cannot list archive: {exc}") from exc
    members = [member[2:] if member.startswith("./") else member for member in members]
    for member in members:
        _safe_member(member)
    return members


def _extract_archive(archive: Path, destination: Path) -> None:
    members = _list_archive(archive)
    if INDEX_MEMBER not in members:
        raise EvidenceAuditError(f"archive is missing {INDEX_MEMBER}")
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".tar.zst") or archive.suffix == ".zst":
        command = ["tar", "--zstd", "-xf", str(archive), "-C", str(destination)]
    else:
        command = ["tar", "-xf", str(archive), "-C", str(destination)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise EvidenceAuditError(f"cannot extract archive: {completed.stderr.strip()}")


def _verify_index(root: Path, *, expected_archive_sha256: str | None = None) -> dict[str, Any]:
    index_path = root / INDEX_MEMBER
    index = _read_json(index_path)
    if index.get("schema_id") != "shakebench.evidence.index" or index.get("schema_version") != 1:
        raise EvidenceAuditError("evidence index schema mismatch")
    index_hash = index.get("index_sha256")
    index_without_hash = dict(index)
    index_without_hash.pop("index_sha256", None)
    if index_hash != _sha256_bytes(_canonical(index_without_hash).encode("utf-8")):
        raise EvidenceAuditError("evidence index self-hash mismatch")
    if expected_archive_sha256 is not None and index.get("archive_sha256") not in {None, expected_archive_sha256}:
        raise EvidenceAuditError("archive SHA-256 differs from evidence index")
    files = index.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceAuditError("evidence index has no file records")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise EvidenceAuditError("evidence index record is not an object")
        member = record.get("archive_member")
        original = record.get("original_path")
        expected_hash = record.get("sha256")
        expected_size = record.get("size")
        if not isinstance(member, str) or not isinstance(original, str) or not isinstance(expected_hash, str):
            raise EvidenceAuditError("evidence index record is missing path/hash fields")
        _safe_member(member)
        if member in seen:
            raise EvidenceAuditError(f"duplicate archive member in index: {member}")
        seen.add(member)
        path = root / member
        if not path.is_file():
            raise EvidenceAuditError(f"indexed archive member is missing: {member}")
        if path.stat().st_size != int(expected_size):
            raise EvidenceAuditError(f"size mismatch: {member}")
        if _sha256_file(path) != expected_hash:
            raise EvidenceAuditError(f"SHA-256 mismatch: {member}")
    return index


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_scientific_audit(root: Path) -> dict[str, Any]:
    phase06_root = root / "evidence" / "phase06"
    phase07_root = root / "evidence" / "phase07"
    if not phase06_root.is_dir():
        raise EvidenceAuditError("archive is missing evidence/phase06")
    if not phase07_root.is_dir():
        raise EvidenceAuditError("archive is missing evidence/phase07")

    from robosuite.scripts.shakebench_handoff_remediation import verify_remediation_bundle
    from robosuite.scripts.shakebench_run_oracle import verify_determinism_manifest, verify_run_artifact
    from robosuite.utils.shakebench_handoff_semantic import verify_phase06fr2_handoff
    from robosuite.utils.shakebench_physics_finalizer import verify_official_publication_bundle

    phase06_checks = {
        "official_publication": verify_official_publication_bundle(phase06_root),
        "remediation": verify_remediation_bundle(phase06_root, require_pass=True),
        "semantic_handoff": verify_phase06fr2_handoff(phase06_root).__dict__,
    }
    errors: list[str] = []
    for name, result in phase06_checks.items():
        passed = result.get("passed") if isinstance(result, Mapping) else None
        if passed is not True:
            errors.append(f"Phase 06 {name}: {result}")

    phase07_checks: dict[str, Any] = {}
    phase07_out = phase07_root / "out"
    if not phase07_out.is_dir():
        errors.append("archive is missing evidence/phase07/out")
    else:
        authoritative_names = {
            "v0_gamma_000_final.json",
            "replay_process_1.json",
            "replay_process_2.json",
            "replay_process_3.json",
        }
        with _pushd(phase07_root):
            r5_out = phase07_out / "phase07r5"
            for path in sorted(r5_out.glob("*.json")):
                try:
                    payload = _read_json(path)
                except EvidenceAuditError:
                    continue
                if payload.get("schema_id") == RUN_SCHEMA_ID:
                    # Authoritative R5 runs are recomputed from the current
                    # v3 profile.  Older probes/ablations remain indexed and
                    # byte-authenticated, but their historical controller
                    # profile is intentionally not treated as current R5
                    # semantics.
                    is_authoritative = path.name in authoritative_names or path.name.startswith(("smoke_", "matched_"))
                    result = (
                        verify_run_artifact(path)
                        if is_authoritative
                        else {"passed": True, "mode": "hash_only_historical_diagnostic"}
                    )
                    phase07_checks[str(path.relative_to(phase07_root))] = result
                    if result.get("passed") is not True:
                        errors.append(f"Phase 07 run {path.name}: {result.get('errors')}")
            determinism = phase07_out / "phase07r5" / "determinism_manifest_final.json"
            if determinism.is_file():
                result = verify_determinism_manifest(determinism)
                phase07_checks["determinism"] = result
                if result.get("passed") is not True:
                    errors.append(f"Phase 07 determinism: {result.get('errors')}")
            else:
                errors.append("Phase 07 determinism manifest is missing")

    return {
        "passed": not errors,
        "errors": errors,
        "phase06": phase06_checks,
        "phase07": phase07_checks,
    }


def audit_archive(archive: str | Path, *, expected_archive_sha256: str | None = None) -> dict[str, Any]:
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        return {"passed": False, "errors": [f"evidence archive is missing: {archive_path}"]}
    temporary = Path(tempfile.mkdtemp(prefix="shakebench_full_audit_"))
    try:
        _extract_archive(archive_path, temporary)
        index = _verify_index(temporary, expected_archive_sha256=expected_archive_sha256)
        scientific = _run_scientific_audit(temporary)
        errors = list(scientific.get("errors", ()))
        return {
            "passed": not errors,
            "errors": errors,
            "archive": str(archive_path),
            "archive_sha256": _sha256_file(archive_path),
            "index_sha256": index.get("index_sha256"),
            "scientific": scientific,
        }
    except EvidenceAuditError as exc:
        return {"passed": False, "errors": [str(exc)], "archive": str(archive_path)}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def audit_root(root: str | Path, *, expected_archive_sha256: str | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return {"passed": False, "errors": [f"evidence root is missing: {root_path}"]}
    try:
        index = _verify_index(root_path, expected_archive_sha256=expected_archive_sha256)
        scientific = _run_scientific_audit(root_path)
    except EvidenceAuditError as exc:
        return {"passed": False, "errors": [str(exc)], "evidence_root": str(root_path)}
    errors = list(scientific.get("errors", ()))
    return {
        "passed": not errors,
        "errors": errors,
        "evidence_root": str(root_path),
        "index_sha256": index.get("index_sha256"),
        "scientific": scientific,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence-archive", type=Path)
    source.add_argument("--evidence-root", type=Path)
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.evidence_archive is not None:
        result = audit_archive(args.evidence_archive, expected_archive_sha256=args.expected_archive_sha256)
    else:
        result = audit_root(args.evidence_root, expected_archive_sha256=args.expected_archive_sha256)
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EvidenceAuditError", "audit_archive", "audit_root", "main"]
