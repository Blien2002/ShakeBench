"""Write a deterministic train-split state artifact for demonstration collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from shakebench.utils.artifacts import write_json_atomic
from shakebench.utils.dev_states import CAN_XY_HALF_RANGE_M
from shakebench.utils.train_states import build_train_state_artifact, verify_train_state_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New train-state JSON path; never overwritten")
    parser.add_argument("--count", type=int, required=True, help="Number of train states to generate")
    parser.add_argument("--seed", type=int, required=True, help="Generation seed; same seed reproduces the pool")
    parser.add_argument("--half-range-m", type=float, default=CAN_XY_HALF_RANGE_M)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = build_train_state_artifact(args.count, seed=args.seed, half_range_m=args.half_range_m)
    verdict = verify_train_state_artifact(payload)
    if not verdict["passed"]:
        raise RuntimeError(f"generated train artifact failed verification: {verdict['checks']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, payload)
    print(f"{len(payload['states'])} train states -> {args.output}")
    print(f"payload_sha256={payload['artifact_lock']['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
