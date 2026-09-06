"""Build a deterministic, preflighted ShakeBench evidence archive.

The builder accepts a deliberately staged tree.  It refuses package/install
trees, caches, archives, links, and unindexed files before creating the tar
file.  The embedded index is written before the outer archive and never
contains the outer archive hash; that binding belongs to the detached release
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from robosuite.scripts.shakebench_build_evidence_index import build_index

FORBIDDEN_COMPONENTS = {
    ".git",
    "__pycache__",
    "build",
    "cache",
    "caches",
    "dist",
    "install",
    "package_install",
    "site-packages",
    "sdist",
    "virtualenv",
    "venv",
    "wheel",
    "wheels",
}
FORBIDDEN_SUFFIXES = (".whl", ".tar.gz", ".tar.zst", ".bundle")
LARGE_DUPLICATE_THRESHOLD = 1024 * 1024


class ArchiveBuildError(ValueError):
    """Raised when the evidence stage is not publishable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification(relative: str) -> str:
    lower = relative.lower()
    if lower.startswith("evidence/phase06/") or lower.startswith("evidence/phase07/"):
        if "invalidated" in lower:
            return "invalidated"
        return "authoritative" if "phase07r5" in lower or "selected" in lower else "diagnostic"
    if lower.startswith("provenance/"):
        return "provenance"
    if lower.startswith("audit/"):
        return "verifier"
    return "metadata"


def preflight_stage(stage_root: str | Path, *, allowed_duplicate_paths: set[str] | None = None) -> dict[str, Any]:
    root = Path(stage_root).resolve()
    if not root.is_dir():
        raise ArchiveBuildError(f"stage root is missing: {root}")
    allowed = allowed_duplicate_paths or set()
    files: list[Path] = []
    inode_seen: dict[tuple[int, int], str] = {}
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if path.is_symlink():
            errors.append(f"symlink member is forbidden: {relative}")
            continue
        if path.is_dir():
            if parts & FORBIDDEN_COMPONENTS:
                errors.append(f"forbidden directory component: {relative}")
            continue
        if not path.is_file():
            errors.append(f"non-regular stage member is forbidden: {relative}")
            continue
        if parts & FORBIDDEN_COMPONENTS or path.name.lower().endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden archive content: {relative}")
        inode = (path.stat().st_dev, path.stat().st_ino)
        prior = inode_seen.get(inode)
        if prior is not None:
            errors.append(f"hardlink-like duplicate stage member: {prior} and {relative}")
        inode_seen[inode] = relative
        files.append(path)
    digests: dict[str, list[str]] = {}
    total_bytes = 0
    categories: Counter[str] = Counter()
    category_bytes: Counter[str] = Counter()
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _sha256(path)
        digests.setdefault(digest, []).append(relative)
        total_bytes += size
        category = _classification(relative)
        categories[category] += 1
        category_bytes[category] += size
    duplicate_large = [
        paths
        for paths in digests.values()
        if len(paths) > 1 and any((root / item).stat().st_size >= LARGE_DUPLICATE_THRESHOLD for item in paths)
    ]
    for paths in duplicate_large:
        if not set(paths).issubset(allowed):
            errors.append("duplicate large-file SHA-256 requires an explicit reason: " + ", ".join(paths))
    if errors:
        raise ArchiveBuildError("; ".join(errors))
    return {
        "stage_root": str(root),
        "member_count": len(files),
        "uncompressed_bytes": total_bytes,
        "categories": dict(sorted(categories.items())),
        "category_bytes": dict(sorted(category_bytes.items())),
        "duplicate_large_sha256_groups": [sorted(paths) for paths in duplicate_large],
        "forbidden_content_checks": {
            "package_install": True,
            "build": True,
            "dist": True,
            "cache": True,
            "wheel_sdist": True,
            "links": True,
        },
    }


def _tar_create(stage_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name.endswith(".tar.zst"):
        command = [
            "tar",
            "--zstd",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-cf",
            str(output),
            "-C",
            str(stage_root),
            ".",
        ]
    elif output.name.endswith(".tar.gz"):
        command = [
            "tar",
            "-czf",
            str(output),
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-C",
            str(stage_root),
            ".",
        ]
    else:
        raise ArchiveBuildError("archive output must end with .tar.zst or .tar.gz")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ArchiveBuildError(completed.stderr.strip() or "tar failed")


def build_archive(
    stage_root: str | Path,
    output: str | Path,
    *,
    index_output: str | Path,
    archive_release_tag: str,
    expected_asset_url: str,
    path_list_sha256: str,
    history_map_sha256: str,
    preflight_output: str | Path | None = None,
) -> dict[str, Any]:
    stage = Path(stage_root).resolve()
    archive = Path(output).resolve()
    index_destination = Path(index_output).resolve()
    if archive.parent == stage or archive.is_relative_to(stage):
        raise ArchiveBuildError("outer archive must not be placed inside the staging tree")
    report = preflight_stage(stage)
    disk = shutil.disk_usage(archive.parent if archive.parent.exists() else Path("/tmp"))
    required = int(report["uncompressed_bytes"]) + max(1024 * 1024, int(report["member_count"]) * 4096)
    if disk.free < required:
        raise ArchiveBuildError(
            "insufficient disk space before archive creation: "
            + json.dumps({"required_bytes": required, "available_bytes": int(disk.free)}, sort_keys=True)
        )

    index_path = stage / "index" / "evidence_index_v1.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_index(
        stage,
        output=index_path,
        archive_release_tag=archive_release_tag,
        expected_asset_url=expected_asset_url,
        path_list_sha256=path_list_sha256,
        history_map_sha256=history_map_sha256,
        archive_sha256=None,
        archive_format="tar.zst" if archive.name.endswith(".tar.zst") else "tar.gz",
    )
    index_bytes = index_path.read_bytes()
    index_destination.parent.mkdir(parents=True, exist_ok=True)
    index_destination.write_bytes(index_bytes)
    # The index was added after the first preflight; validate the final stage
    # again so every ordinary member is covered by the generated index.
    report = preflight_stage(stage)
    _tar_create(stage, archive)
    result = {
        "archive": str(archive),
        "archive_filename": archive.name,
        "archive_byte_size": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "embedded_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "member_count": len(index.get("files", [])),
        "uncompressed_bytes": report["uncompressed_bytes"],
        "preflight": report,
    }
    if preflight_output is not None:
        target = Path(preflight_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--archive-release-tag", required=True)
    parser.add_argument("--expected-asset-url", required=True)
    parser.add_argument("--path-list-sha256", required=True)
    parser.add_argument("--history-map-sha256", required=True)
    parser.add_argument("--preflight-output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build_archive(
        args.stage_root,
        args.output,
        index_output=args.index_output,
        archive_release_tag=args.archive_release_tag,
        expected_asset_url=args.expected_asset_url,
        path_list_sha256=args.path_list_sha256,
        history_map_sha256=args.history_map_sha256,
        preflight_output=args.preflight_output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ArchiveBuildError", "build_archive", "main", "preflight_stage"]
