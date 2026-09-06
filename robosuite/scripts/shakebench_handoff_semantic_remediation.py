"""CLI for the Phase 06F-R2 semantic handoff verifier."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from robosuite.scripts.shakebench_audit_evidence import audit_archive, audit_root


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--evidence-root", default=None)
    parser.add_argument("--evidence-archive", default=None)
    parser.add_argument("--release-manifest", default=None)
    parser.add_argument("--expected-archive-sha256", default=None)
    args = parser.parse_args(argv)
    if not args.verify:
        parser.error("--verify is required")
    if args.evidence_archive and args.evidence_root:
        parser.error("--evidence-archive and --evidence-root are mutually exclusive")
    evidence_root = args.evidence_root or args.asset_root
    if not args.evidence_archive and not evidence_root:
        result = {
            "passed": False,
            "errors": ["explicit external evidence required: provide --evidence-root or --evidence-archive"],
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2
    if args.evidence_archive:
        result = audit_archive(
            args.evidence_archive,
            expected_archive_sha256=args.expected_archive_sha256,
            release_manifest=args.release_manifest,
        )
    else:
        result = audit_root(evidence_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
