"""Standalone archive/index checker with only Python stdlib dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

INDEX = "index/evidence_index_v1.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    index_path = root / INDEX
    value = json.loads(index_path.read_text(encoding="utf-8"))
    expected = value.get("index_sha256")
    copied = dict(value)
    copied.pop("index_sha256", None)
    errors: list[str] = []
    if expected != hashlib.sha256(_canonical(copied)).hexdigest():
        errors.append("index self-hash mismatch")
    for record in value.get("files", []):
        member = str(record["archive_member"])
        safe = PurePosixPath(member)
        if safe.is_absolute() or ".." in safe.parts:
            errors.append("unsafe member: " + member)
            continue
        path = root / member
        if not path.is_file():
            errors.append("missing member: " + member)
            continue
        if path.stat().st_size != int(record["size"]):
            errors.append("size mismatch: " + member)
        if _sha256(path) != record["sha256"]:
            errors.append("hash mismatch: " + member)
    return {"passed": not errors, "errors": errors, "index_sha256": expected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    temporary = Path(tempfile.mkdtemp(prefix="shakebench_archive_check_"))
    try:
        command = ["tar", "--zstd", "-xf", str(args.archive), "-C", str(temporary)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            print(json.dumps({"passed": False, "errors": [completed.stderr.strip()]}, indent=2))
            return 1
        result = verify(temporary)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["passed"] else 1
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
