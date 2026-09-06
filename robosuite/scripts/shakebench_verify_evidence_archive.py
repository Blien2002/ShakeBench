"""Verify a ShakeBench evidence archive with Python's standard library only.

The archive's content index authenticates members and their scientific
metadata.  The detached release manifest authenticates the *outer archive*
bytes; an index is deliberately not allowed to authenticate itself as the
outer archive.  The script uses the system ``tar`` command for zstd support,
but has no dependency on robosuite, NumPy, PyYAML, or any other package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

INDEX_MEMBER = "index/evidence_index_v1.json"
MAP_MEMBER = "provenance/shakebench_history_rewrite_map_v2.json"
REMOVED_PATH_MEMBER = "provenance/shakebench_evidence_removed_paths_v1.txt"
RELEASE_SCHEMA_ID = "shakebench.evidence.release_manifest"
RELEASE_SCHEMA_VERSION = 1


class ArchiveVerificationError(ValueError):
    """Raised when an archive or its detached authority is invalid."""


class DiskSpaceError(ArchiveVerificationError):
    """Raised before extraction when the destination cannot hold the data."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_member(name: str) -> str:
    """Return a safe canonical POSIX member name or raise."""

    if not isinstance(name, str) or "\x00" in name:
        raise ArchiveVerificationError(f"invalid archive member path: {name!r}")
    if name in {".", "./"}:
        return "."
    if not name:
        raise ArchiveVerificationError(f"invalid archive member path: {name!r}")
    if name.endswith("/"):
        name = name.rstrip("/")
        if not name:
            return "."
    if name.startswith("./"):
        name = name[2:]
        if not name:
            return "."
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveVerificationError(f"unsafe archive member path: {name}")
    if name == ".":
        return name
    normalized = path.as_posix()
    if normalized != name or not normalized:
        raise ArchiveVerificationError(f"non-canonical archive member path: {name}")
    return normalized


def _archive_kind(path: Path) -> str:
    return "zstd" if path.name.endswith(".tar.zst") or path.suffix == ".zst" else "tar"


def _tar_command(archive: Path, *args: str) -> list[str]:
    command = ["tar"]
    if _archive_kind(archive) == "zstd":
        command.append("--zstd")
    command.extend(args)
    command.append(str(archive))
    return command


def _run(command: list[str], *, error: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown tar error"
        raise ArchiveVerificationError(f"{error}: {detail}")
    return completed


def _archive_members(archive: Path) -> tuple[list[str], list[str], int]:
    """Return (all names, regular file names, declared regular bytes)."""

    if _archive_kind(archive) == "zstd":
        names = _run(_tar_command(archive, "-tf"), error="cannot list zstd archive").stdout.splitlines()
        verbose = _run(_tar_command(archive, "-tvf"), error="cannot inspect zstd archive").stdout.splitlines()
        regular: list[str] = []
        declared_bytes = 0
        for line in verbose:
            fields = line.split(maxsplit=5)
            if not fields:
                continue
            mode = fields[0]
            if mode[0] not in {"-", "d"}:
                raise ArchiveVerificationError(f"archive contains non-regular member: {line}")
            if mode[0] == "-":
                if len(fields) < 4:
                    raise ArchiveVerificationError(f"cannot parse zstd member metadata: {line}")
                try:
                    declared_bytes += int(fields[2])
                except ValueError as exc:
                    raise ArchiveVerificationError(f"cannot parse zstd member size: {line}") from exc
                regular.append(fields[-1])
    else:
        try:
            with tarfile.open(archive, "r:*") as stream:
                members = stream.getmembers()
        except (OSError, tarfile.TarError) as exc:
            raise ArchiveVerificationError(f"cannot read archive: {exc}") from exc
        names = [member.name for member in members]
        regular = []
        declared_bytes = 0
        for member in members:
            if (
                member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
                or member.ischr()
                or member.isblk()
            ):
                raise ArchiveVerificationError(f"archive contains non-regular member: {member.name}")
            if member.isfile():
                regular.append(member.name)
                declared_bytes += int(member.size)

    normalized_names = [normalize_member(name) for name in names]
    if len(normalized_names) != len(set(normalized_names)):
        raise ArchiveVerificationError("archive contains duplicate members")
    normalized_regular = [normalize_member(name) for name in regular]
    return normalized_names, normalized_regular, declared_bytes


def _read_index_from_archive(archive: Path) -> dict[str, Any]:
    names, _, _ = _archive_members(archive)
    if INDEX_MEMBER not in names:
        raise ArchiveVerificationError(f"archive is missing {INDEX_MEMBER}")
    if _archive_kind(archive) == "zstd":
        raw = None
        for candidate in (INDEX_MEMBER, "./" + INDEX_MEMBER):
            completed = subprocess.run(
                ["tar", "--zstd", "-xOf", str(archive), candidate],
                check=False,
                capture_output=True,
            )
            if completed.returncode == 0:
                raw = completed.stdout
                break
        if raw is None:
            raise ArchiveVerificationError("cannot read embedded content index from zstd archive")
    else:
        try:
            with tarfile.open(archive, "r:*") as stream:
                member = stream.getmember(INDEX_MEMBER)
                extracted = stream.extractfile(member)
                if extracted is None:
                    raise ArchiveVerificationError("embedded content index is not a regular file")
                raw = extracted.read()
        except (OSError, KeyError, tarfile.TarError) as exc:
            raise ArchiveVerificationError(f"cannot read embedded content index: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(f"embedded content index is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveVerificationError("embedded content index is not an object")
    return value


def _disk_preflight(index: Mapping[str, Any], declared_bytes: int, destination: Path) -> dict[str, int]:
    records = index.get("files")
    if not isinstance(records, list):
        raise ArchiveVerificationError("content index files must be a list")
    indexed_bytes = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ArchiveVerificationError("content index record is not an object")
        try:
            size = int(record["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveVerificationError("content index record has invalid size") from exc
        if size < 0:
            raise ArchiveVerificationError("content index record has negative size")
        indexed_bytes += size
    # Include the embedded index, directory entries, extraction metadata, and a
    # small safety margin.  This is intentionally conservative: failing before
    # extraction is preferable to leaving a multi-gigabyte partial tree.
    required = max(indexed_bytes, declared_bytes) + max(1024 * 1024, len(records) * 4096)
    usage = shutil.disk_usage(destination)
    result = {"required_bytes": required, "available_bytes": int(usage.free), "indexed_bytes": indexed_bytes}
    if result["available_bytes"] < required:
        raise DiskSpaceError("insufficient disk space before archive extraction: " + json.dumps(result, sort_keys=True))
    return result


def _extract(archive: Path, destination: Path) -> tuple[list[str], dict[str, int]]:
    names, regular, declared_bytes = _archive_members(archive)
    index = _read_index_from_archive(archive)
    disk = _disk_preflight(index, declared_bytes, destination)
    destination.mkdir(parents=True, exist_ok=True)
    if _archive_kind(archive) == "zstd":
        command = [
            "tar",
            "--zstd",
            "--no-same-owner",
            "--no-same-permissions",
            "-xf",
            str(archive),
            "-C",
            str(destination),
        ]
    else:
        command = [
            "tar",
            "--no-same-owner",
            "--no-same-permissions",
            "-xf",
            str(archive),
            "-C",
            str(destination),
        ]
    _run(command, error="cannot extract archive")
    return regular, disk


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(f"{description} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveVerificationError(f"{description} is not an object")
    return value


def _verify_index(
    root: Path,
    *,
    regular_members: Iterable[str] | None = None,
    repo_index: Path | None = None,
    repo_map: Path | None = None,
    repo_removed_path_list: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    index_path = root / INDEX_MEMBER
    index = _read_json(index_path, "embedded content index")
    if index.get("schema_id") != "shakebench.evidence.index" or index.get("schema_version") != 1:
        raise ArchiveVerificationError("evidence index schema mismatch")
    if index.get("archive_sha256") not in (None, ""):
        # The v1 field is retained for compatibility, but R6's source of truth
        # for outer bytes is detached.  A non-null value must never bypass it.
        if not isinstance(index.get("archive_sha256"), str) or len(index["archive_sha256"]) != 64:
            raise ArchiveVerificationError("evidence index archive_sha256 is malformed")
    without_hash = dict(index)
    expected_index_hash = without_hash.pop("index_sha256", None)
    if expected_index_hash != sha256_bytes(canonical(without_hash)):
        raise ArchiveVerificationError("evidence index self-hash mismatch")
    files = index.get("files")
    if not isinstance(files, list) or not files:
        raise ArchiveVerificationError("evidence index has no file records")
    indexed: set[str] = set()
    errors: list[str] = []
    for record in files:
        if not isinstance(record, Mapping):
            errors.append("evidence index record is not an object")
            continue
        try:
            member = normalize_member(record["archive_member"])
        except (KeyError, ArchiveVerificationError) as exc:
            errors.append(str(exc))
            continue
        if member in indexed:
            errors.append("duplicate archive member in index: " + member)
            continue
        indexed.add(member)
        expected_hash = record.get("sha256")
        expected_size = record.get("size")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append("malformed SHA-256 in index: " + member)
            continue
        try:
            expected_size = int(expected_size)
        except (TypeError, ValueError):
            errors.append("malformed size in index: " + member)
            continue
        path = root / member
        if not path.is_file() or path.is_symlink():
            errors.append("indexed archive member is missing: " + member)
            continue
        if path.stat().st_size != expected_size:
            errors.append("size mismatch: " + member)
        if sha256_file(path) != expected_hash:
            errors.append("SHA-256 mismatch: " + member)
    if INDEX_MEMBER in indexed:
        errors.append("content index must not index itself")
    actual = set(regular_members or [])
    actual.discard(INDEX_MEMBER)
    missing = sorted(indexed - actual)
    extra = sorted(actual - indexed)
    errors.extend("missing archive member: " + member for member in missing)
    errors.extend("unindexed archive member: " + member for member in extra)
    if repo_index is not None:
        try:
            if index_path.read_bytes() != repo_index.read_bytes():
                errors.append("embedded content index differs from repository index")
        except OSError as exc:
            errors.append(f"repository content index unreadable: {exc}")
    for path, member, field in (
        (repo_map, MAP_MEMBER, "history_rewrite_map_sha256"),
        (repo_removed_path_list, REMOVED_PATH_MEMBER, "path_list_sha256"),
    ):
        if path is None:
            continue
        expected = index.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"content index {field} missing")
            continue
        if sha256_file(path) != expected:
            errors.append(f"repository {field} differs from content index")
        member_path = root / member
        if not member_path.is_file() or member_path.read_bytes() != path.read_bytes():
            errors.append(f"archive {member} differs from repository metadata")
    if errors:
        raise ArchiveVerificationError("; ".join(sorted(set(errors))))
    return index, sorted(indexed)


def _verify_release_manifest(
    release_manifest: Path,
    *,
    archive: Path,
    actual_archive_sha256: str,
    index: Mapping[str, Any] | None = None,
    index_bytes_sha256: str | None = None,
    repo_map: Path | None = None,
    repo_removed_path_list: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_json(release_manifest, "detached release manifest")
    if manifest.get("schema_id") != RELEASE_SCHEMA_ID or manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ArchiveVerificationError("detached release manifest schema mismatch")
    archive_spec = manifest.get("archive")
    if not isinstance(archive_spec, Mapping):
        raise ArchiveVerificationError("detached release manifest archive binding is missing")
    if archive_spec.get("filename") != archive.name:
        raise ArchiveVerificationError("detached release manifest archive filename mismatch")
    if int(archive_spec.get("byte_size", -1)) != archive.stat().st_size:
        raise ArchiveVerificationError("detached release manifest archive byte size mismatch")
    if archive_spec.get("sha256") != actual_archive_sha256:
        raise ArchiveVerificationError("detached release manifest archive SHA-256 mismatch")
    if index is not None:
        if index_bytes_sha256 is None or manifest.get("embedded_content_index_sha256") != index_bytes_sha256:
            raise ArchiveVerificationError("release manifest/content index binding mismatch")
        for manifest_field, index_field, label in (
            ("history_rewrite_map_sha256", "history_rewrite_map_sha256", "rewrite map"),
            ("removed_path_list_sha256", "path_list_sha256", "removed path list"),
        ):
            if manifest.get(manifest_field) != index.get(index_field):
                raise ArchiveVerificationError(f"release manifest/{label} binding mismatch")
    if repo_map is not None and manifest.get("history_rewrite_map_sha256") != sha256_file(repo_map):
        raise ArchiveVerificationError("release manifest/repository rewrite map mismatch")
    if repo_removed_path_list is not None and manifest.get("removed_path_list_sha256") != sha256_file(
        repo_removed_path_list
    ):
        raise ArchiveVerificationError("release manifest/repository removed path list mismatch")
    package = manifest.get("package_evidence")
    if not isinstance(package, Mapping) or not isinstance(package.get("sha256"), str):
        raise ArchiveVerificationError("release manifest package evidence binding is missing")
    return manifest


def verify_root(
    root: Path,
    *,
    regular_members: Iterable[str] | None = None,
    repo_index: Path | None = None,
    repo_map: Path | None = None,
    repo_removed_path_list: Path | None = None,
) -> dict[str, Any]:
    index, indexed = _verify_index(
        root,
        regular_members=regular_members,
        repo_index=repo_index,
        repo_map=repo_map,
        repo_removed_path_list=repo_removed_path_list,
    )
    return {"passed": True, "errors": [], "index_sha256": index.get("index_sha256"), "members": len(indexed)}


def verify_archive(
    archive: str | Path,
    *,
    release_manifest: str | Path | None = None,
    expected_sha256: str | None = None,
    repo_index: str | Path | None = None,
    repo_map: str | Path | None = None,
    repo_removed_path_list: str | Path | None = None,
) -> dict[str, Any]:
    """Verify outer bytes, detached authority, member contract, and hashes."""

    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        return {"passed": False, "errors": [f"evidence archive is missing: {archive_path}"]}
    if release_manifest is None and expected_sha256 is None:
        return {
            "passed": False,
            "errors": ["detached release authority is required: provide --release-manifest or --expected-sha256"],
        }
    errors: list[str] = []
    actual_sha = sha256_file(archive_path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        errors.append("actual archive SHA-256 differs from expected detached SHA-256")
    temporary = Path(tempfile.mkdtemp(prefix="shakebench_archive_verify_"))
    disk: dict[str, int] | None = None
    try:
        if release_manifest is not None:
            try:
                # Validate the detached outer binding before spending disk on
                # extraction.  The index binding is checked after extraction.
                _verify_release_manifest(
                    Path(release_manifest).resolve(),
                    archive=archive_path,
                    actual_archive_sha256=actual_sha,
                    repo_map=Path(repo_map).resolve() if repo_map else None,
                    repo_removed_path_list=Path(repo_removed_path_list).resolve() if repo_removed_path_list else None,
                )
            except (ArchiveVerificationError, OSError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        try:
            regular, disk = _extract(archive_path, temporary)
            index, _ = _verify_index(
                temporary,
                regular_members=regular,
                repo_index=Path(repo_index).resolve() if repo_index else None,
                repo_map=Path(repo_map).resolve() if repo_map else None,
                repo_removed_path_list=Path(repo_removed_path_list).resolve() if repo_removed_path_list else None,
            )
            if release_manifest is not None:
                try:
                    _verify_release_manifest(
                        Path(release_manifest).resolve(),
                        archive=archive_path,
                        actual_archive_sha256=actual_sha,
                        index=index,
                        index_bytes_sha256=sha256_file(temporary / INDEX_MEMBER),
                        repo_map=Path(repo_map).resolve() if repo_map else None,
                        repo_removed_path_list=(
                            Path(repo_removed_path_list).resolve() if repo_removed_path_list else None
                        ),
                    )
                except (ArchiveVerificationError, OSError, TypeError, ValueError) as exc:
                    errors.append(str(exc))
        except (ArchiveVerificationError, OSError, tarfile.TarError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    result: dict[str, Any] = {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "archive": str(archive_path),
        "archive_sha256": actual_sha,
    }
    if disk is not None:
        result["disk_preflight"] = disk
    return result


def verify(root: Path, **kwargs: Any) -> dict[str, Any]:
    """Compatibility helper for an already extracted tree."""

    try:
        return verify_root(root, **kwargs)
    except (ArchiveVerificationError, OSError, TypeError, ValueError) as exc:
        return {"passed": False, "errors": [str(exc)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--release-manifest", type=Path, default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--repo-index", type=Path, default=None)
    parser.add_argument("--rewrite-map", type=Path, default=None)
    parser.add_argument("--removed-path-list", type=Path, default=None)
    args = parser.parse_args(argv)
    result = verify_archive(
        args.archive,
        release_manifest=args.release_manifest,
        expected_sha256=args.expected_sha256,
        repo_index=args.repo_index,
        repo_map=args.rewrite_map,
        repo_removed_path_list=args.removed_path_list,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArchiveVerificationError",
    "DiskSpaceError",
    "INDEX_MEMBER",
    "MAP_MEMBER",
    "REMOVED_PATH_MEMBER",
    "canonical",
    "sha256_file",
    "verify",
    "verify_archive",
    "verify_root",
]
