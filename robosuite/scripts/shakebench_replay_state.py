"""Verify a frozen Phase 08 state artifact without replaying an environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robosuite.utils.shakebench_committed_states import verify_committed_state_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--split", choices=("official", "knee"), required=True)
    args = parser.parse_args(argv)
    verdict = verify_committed_state_artifact(args.states, expected_split=args.split)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
