"""CLI for the Phase 06F-R2 semantic handoff verifier."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from robosuite.utils.shakebench_handoff_semantic import verify_phase06fr2_handoff


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--asset-root", default=None)
    args = parser.parse_args(argv)
    if not args.verify:
        parser.error("--verify is required")
    result = verify_phase06fr2_handoff(args.asset_root)
    print(json.dumps({"passed": result.passed, "errors": list(result.errors), "checks": result.checks}, ensure_ascii=False, sort_keys=True))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
