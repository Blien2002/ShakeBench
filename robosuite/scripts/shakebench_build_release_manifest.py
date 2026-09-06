"""Create the detached release authority for a ShakeBench evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_ID = "shakebench.evidence.release_manifest"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _package_record(package: dict[str, Any], kind: str) -> dict[str, Any]:
    records = package.get("records")
    if not isinstance(records, list):
        raise ValueError("package evidence records are missing")
    matches = [record for record in records if isinstance(record, dict) and record.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"package evidence must contain exactly one {kind} record")
    record = matches[0]
    artifact_sha = record.get("artifact_sha256")
    artifact_path = record.get("artifact_path") or record.get("artifact")
    if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
        raise ValueError(f"package {kind} SHA-256 is missing")
    return {"filename": Path(str(artifact_path)).name, "sha256": artifact_sha, "bytes": record.get("artifact_bytes")}


def build_release_manifest(
    archive: str | Path,
    *,
    output: str | Path,
    release_tag: str,
    release_url: str,
    content_index: str | Path,
    rewrite_map: str | Path,
    removed_path_list: str | Path,
    package_evidence: str | Path,
    r6_manifest: str | Path,
    implementation_commit: str,
    handoff_rule: str = "final_commit must be an ancestor of the commit containing this manifest",
    sums_output: str | Path | None = None,
) -> dict[str, Any]:
    archive_path = Path(archive).resolve()
    index_path = Path(content_index).resolve()
    map_path = Path(rewrite_map).resolve()
    removed_path = Path(removed_path_list).resolve()
    package_path = Path(package_evidence).resolve()
    r6_path = Path(r6_manifest).resolve()
    index = _read_json(index_path)
    package = _read_json(package_path)
    release = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "release_tag": release_tag,
        "release_url": release_url,
        "archive": {
            "filename": archive_path.name,
            "byte_size": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        },
        "embedded_content_index_sha256": _sha256(index_path),
        "history_rewrite_map_sha256": _sha256(map_path),
        "removed_path_list_sha256": _sha256(removed_path),
        "r6_manifest_sha256": _sha256(r6_path),
        "r6_implementation_commit": implementation_commit,
        "expected_handoff": {
            "implementation_commit": implementation_commit,
            "rule": handoff_rule,
        },
        "package_evidence": {
            "path": "provenance/package_evidence_final.json",
            "sha256": _sha256(package_path),
            "wheel": _package_record(package, "wheel"),
            "sdist": _package_record(package, "sdist"),
        },
    }
    index_payload = dict(index)
    index_payload.pop("index_sha256", None)
    if index.get("index_sha256") != hashlib.sha256(_canonical(index_payload)).hexdigest():
        raise ValueError("content index self-hash does not match its bytes")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(release, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if sums_output is not None:
        sums = Path(sums_output)
        sums.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (release["archive"]["sha256"], release["archive"]["filename"]),
            (_sha256(target), target.name),
            (release["history_rewrite_map_sha256"], map_path.name),
            (release["embedded_content_index_sha256"], index_path.name),
        ]
        sums.write_text("".join(f"{digest}  {name}\n" for digest, name in rows), encoding="utf-8")
    return release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--content-index", type=Path, required=True)
    parser.add_argument("--rewrite-map", type=Path, required=True)
    parser.add_argument("--removed-path-list", type=Path, required=True)
    parser.add_argument("--package-evidence", type=Path, required=True)
    parser.add_argument("--r6-manifest", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--sums-output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build_release_manifest(
        args.archive,
        output=args.output,
        release_tag=args.release_tag,
        release_url=args.release_url,
        content_index=args.content_index,
        rewrite_map=args.rewrite_map,
        removed_path_list=args.removed_path_list,
        package_evidence=args.package_evidence,
        r6_manifest=args.r6_manifest,
        implementation_commit=args.implementation_commit,
        sums_output=args.sums_output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_release_manifest", "main"]
