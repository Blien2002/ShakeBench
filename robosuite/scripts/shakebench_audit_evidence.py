"""Run the explicit full ShakeBench evidence audit.

Runtime imports intentionally do not call this module.  A caller must provide
an evidence archive (or an already extracted evidence root) and detached
authority.  Archive byte authentication is performed before the existing
Phase 06/07 scientific recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from robosuite.scripts import shakebench_verify_evidence_archive as archive_verifier

INDEX_MEMBER = archive_verifier.INDEX_MEMBER
RUN_SCHEMA_ID = "shakebench.phase07.oracle_run"


class EvidenceAuditError(ValueError):
    """Raised when evidence archive or binding validation fails."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return archive_verifier.sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceAuditError(f"JSON root is not an object: {path}")
    return value


def _safe_member(name: str) -> None:
    archive_verifier.normalize_member(name)


def _list_archive(archive: Path) -> list[str]:
    try:
        names, _, _ = archive_verifier._archive_members(archive)
    except (archive_verifier.ArchiveVerificationError, OSError, ValueError) as exc:
        raise EvidenceAuditError(str(exc)) from exc
    return names


def _extract_archive(archive: Path, destination: Path) -> list[str]:
    try:
        regular, _ = archive_verifier._extract(archive, destination)
    except (archive_verifier.ArchiveVerificationError, OSError, ValueError) as exc:
        raise EvidenceAuditError(str(exc)) from exc
    return regular


def _verify_index(
    root: Path,
    *,
    expected_archive_sha256: str | None = None,
    regular_members: list[str] | None = None,
    repo_index: Path | None = None,
    repo_map: Path | None = None,
    repo_removed_path_list: Path | None = None,
) -> dict[str, Any]:
    if regular_members is None:
        try:
            regular_members = archive_verifier._root_regular_members(root)
        except archive_verifier.ArchiveVerificationError as exc:
            raise EvidenceAuditError(str(exc)) from exc
    try:
        index, _ = archive_verifier._verify_index(
            root,
            regular_members=regular_members,
            repo_index=repo_index,
            repo_map=repo_map,
            repo_removed_path_list=repo_removed_path_list,
        )
    except archive_verifier.ArchiveVerificationError as exc:
        raise EvidenceAuditError(str(exc)) from exc
    if expected_archive_sha256 is not None and index.get("archive_sha256") not in (None, expected_archive_sha256):
        raise EvidenceAuditError("archive SHA-256 differs from evidence index")
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

    return {"passed": not errors, "errors": errors, "phase06": phase06_checks, "phase07": phase07_checks}


def _repo_metadata(repo_root: Path | None) -> tuple[Path | None, Path | None, Path | None]:
    if repo_root is None:
        return None, None, None
    root = repo_root.resolve()
    index = root / "docs" / "shakebench_evidence_index_v1.json"
    mapping = root / "docs" / "shakebench_history_rewrite_map_v2.json"
    removed = root / "docs" / "shakebench_evidence_removed_paths_v1.txt"
    return (
        index if index.is_file() else None,
        mapping if mapping.is_file() else None,
        removed if removed.is_file() else None,
    )


def audit_archive(
    archive: str | Path,
    *,
    expected_archive_sha256: str | None = None,
    release_manifest: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the outer archive and then run the full scientific audit."""

    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        return {"passed": False, "errors": [f"evidence archive is missing: {archive_path}"]}
    if expected_archive_sha256 is None and release_manifest is None:
        return {"passed": False, "errors": ["detached release authority is required"]}
    actual_archive_sha256 = _sha256_file(archive_path)
    errors: list[str] = []
    release_authority_verified = expected_archive_sha256 is not None
    if expected_archive_sha256 is not None and actual_archive_sha256 != expected_archive_sha256:
        errors.append("actual archive SHA-256 differs from expected SHA-256")
    repo_index, repo_map, repo_removed = _repo_metadata(Path(repo_root) if repo_root else None)
    temporary = Path(tempfile.mkdtemp(prefix="shakebench_full_audit_"))
    index: dict[str, Any] = {}
    disk: dict[str, Any] | None = None
    try:
        try:
            regular_members = _extract_archive(archive_path, temporary)
            index = _verify_index(
                temporary,
                expected_archive_sha256=expected_archive_sha256,
                regular_members=regular_members,
                repo_index=repo_index,
                repo_map=repo_map,
                repo_removed_path_list=repo_removed,
            )
            if release_manifest is not None:
                try:
                    release = archive_verifier._verify_release_manifest(
                        Path(release_manifest).resolve(),
                        archive=archive_path,
                        actual_archive_sha256=actual_archive_sha256,
                        index=index,
                        index_bytes_sha256=_sha256_file(temporary / INDEX_MEMBER),
                        repo_map=repo_map,
                        repo_removed_path_list=repo_removed,
                    )
                    package_member = str(release.get("package_evidence", {}).get("path", ""))
                    if package_member and not (temporary / package_member).is_file():
                        errors.append("release manifest package evidence is missing from archive")
                    else:
                        release_authority_verified = True
                except (archive_verifier.ArchiveVerificationError, OSError, TypeError, ValueError) as exc:
                    errors.append(str(exc))
            scientific = _run_scientific_audit(temporary)
            errors.extend(scientific.get("errors", ()))
        except EvidenceAuditError as exc:
            errors.append(str(exc))
            scientific = {"passed": False, "errors": [str(exc)]}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    result: dict[str, Any] = {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "archive": str(archive_path),
        "archive_sha256": actual_archive_sha256,
        "index_sha256": index.get("index_sha256"),
        "release_authority_verified": release_authority_verified and not errors,
        "scientific": scientific,
    }
    return result


def audit_root(
    root: str | Path,
    *,
    expected_archive_sha256: str | None = None,
    release_manifest: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit an extracted root; an outer archive is required for release authority."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return {"passed": False, "errors": [f"evidence root is missing: {root_path}"]}
    if expected_archive_sha256 is not None or release_manifest is not None:
        return {"passed": False, "errors": ["full audit release authority requires the original archive bytes"]}
    repo_index, repo_map, repo_removed = _repo_metadata(Path(repo_root) if repo_root else None)
    try:
        index = _verify_index(root_path, repo_index=repo_index, repo_map=repo_map, repo_removed_path_list=repo_removed)
        scientific = _run_scientific_audit(root_path)
    except EvidenceAuditError as exc:
        return {"passed": False, "errors": [str(exc)], "evidence_root": str(root_path)}
    errors = list(scientific.get("errors", ()))
    return {
        "passed": not errors,
        "errors": errors,
        "evidence_root": str(root_path),
        "index_sha256": index.get("index_sha256"),
        "release_authority_verified": False,
        "scientific_only": True,
        "scientific": scientific,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence-archive", type=Path)
    source.add_argument("--evidence-root", type=Path)
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument("--release-manifest", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scientific-only", action="store_true")
    args = parser.parse_args(argv)
    if args.evidence_root is not None and not args.scientific_only:
        result = {"passed": False, "errors": ["--evidence-root requires --scientific-only"]}
    elif args.evidence_root is not None and (
        args.expected_archive_sha256 is not None or args.release_manifest is not None
    ):
        result = {
            "passed": False,
            "errors": ["scientific-only root audit cannot claim release authority"],
            "release_authority_verified": False,
        }
    elif args.evidence_archive is not None and args.expected_archive_sha256 is None and args.release_manifest is None:
        result = {
            "passed": False,
            "errors": [
                "detached release authority is required: provide --expected-archive-sha256 or --release-manifest"
            ],
        }
    elif args.evidence_archive is not None:
        result = audit_archive(
            args.evidence_archive,
            expected_archive_sha256=args.expected_archive_sha256,
            release_manifest=args.release_manifest,
            repo_root=args.repo_root,
        )
    else:
        result = audit_root(
            args.evidence_root,
            repo_root=args.repo_root,
        )
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EvidenceAuditError", "audit_archive", "audit_root", "main"]
