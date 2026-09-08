"""Verify a Phase 7.5A direct-mount handoff manifest fail-closed."""

from __future__ import annotations

import argparse
import json

from robosuite.utils.shakebench_handoff import verify_phase07_5a_handoff


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="docs/phase_07_5a_requalification_manifest.json")
    args = parser.parse_args(argv)
    verdict = verify_phase07_5a_handoff(args.manifest)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
