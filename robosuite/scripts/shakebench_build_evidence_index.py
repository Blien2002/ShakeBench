"""Build the deterministic compact index for a staged evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bindings(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(value, Mapping):
        return {}
    bindings: dict[str, Any] = {}
    for key in (
        "profile_id",
        "profile_sha256",
        "protocol_sha256",
        "state_id",
        "tier",
        "gamma_commanded",
        "phase",
        "status",
        "schema_id",
        "schema_version",
    ):
        if key in value and isinstance(value[key], (str, int, float, bool)):
            bindings[key] = value[key]
    for key in ("physics_authority", "controller_profile", "physics_profile", "selected_profile"):
        item = value.get(key)
        if isinstance(item, Mapping):
            selected = {
                name: item[name]
                for name in ("profile_id", "profile_sha256", "sha256")
                if name in item and isinstance(item[name], (str, int, float, bool))
            }
            if selected:
                bindings[key] = selected
    return bindings


def _original_path(member: str) -> str:
    if member.startswith("evidence/phase06/"):
        return "robosuite/models/assets/" + member.removeprefix("evidence/phase06/")
    if member.startswith("evidence/phase07/"):
        return member.removeprefix("evidence/phase07/")
    if member.startswith("provenance/"):
        return member.removeprefix("provenance/")
    return member


def _status(original: str) -> str:
    if original.startswith("out/phase07_invalidated/"):
        return "invalidated"
    if original.startswith("out/phase07"):
        return "authoritative" if "phase07r5" in original else "diagnostic"
    if "diagnostic" in original or "raw" in original or "replay" in original:
        return "diagnostic"
    if original.startswith("history_rewrite/") or original.startswith("audit/"):
        return "build evidence"
    return "authoritative"


def build_index(
    stage_root: str | Path,
    *,
    output: str | Path,
    archive_release_tag: str,
    expected_asset_url: str,
    path_list_sha256: str,
    history_map_sha256: str | None = None,
    archive_sha256: str | None = None,
    archive_format: str = "tar.zst",
) -> dict[str, Any]:
    root = Path(stage_root)
    index_member = Path("index/evidence_index_v1.json")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root) == index_member:
            continue
        member = path.relative_to(root).as_posix()
        original = _original_path(member)
        record: dict[str, Any] = {
            "archive_member": member,
            "archive_group": "phase06/raw" if original.startswith("robosuite/models/assets/") else "release",
            "original_path": original,
            "phase_status": _status(original),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        payload = _json_bindings(path)
        for field in ("payload_sha256", "trace_sha256", "trace_whole_digest"):
            if field in payload:
                record[field] = payload[field]
        if payload:
            record["bindings"] = payload
        files.append(record)

    index: dict[str, Any] = {
        # The outer archive is authenticated only by the detached release
        # manifest.  Keeping this compatibility field null avoids an archive
        # -> index -> archive hash cycle.
        "archive_format": archive_format,
        "archive_release_tag": archive_release_tag,
        "archive_sha256": archive_sha256,
        "expected_asset_url": expected_asset_url,
        "files": files,
        "history_rewrite_map_sha256": history_map_sha256,
        "path_list_sha256": path_list_sha256,
        "schema_id": "shakebench.evidence.index",
        "schema_version": 1,
    }
    index["file_count"] = len(files)
    index["index_sha256"] = hashlib.sha256(_canonical(index).encode("utf-8")).hexdigest()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-release-tag", required=True)
    parser.add_argument("--expected-asset-url", required=True)
    parser.add_argument("--path-list-sha256", required=True)
    parser.add_argument("--history-map-sha256", default=None)
    parser.add_argument("--archive-sha256", default=None)
    args = parser.parse_args(argv)
    build_index(
        args.stage_root,
        output=args.output,
        archive_release_tag=args.archive_release_tag,
        expected_asset_url=args.expected_asset_url,
        path_list_sha256=args.path_list_sha256,
        history_map_sha256=args.history_map_sha256,
        archive_sha256=args.archive_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_index", "main"]
