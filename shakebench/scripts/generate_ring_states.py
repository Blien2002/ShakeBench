"""Generate reproducible random tabletop starts for ring-stack collection."""

import argparse
import json
from pathlib import Path

import numpy as np

from shakebench.environments.ring_on_peg import STATE_SCHEMA, sample_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.count <= 0 or not 0 <= args.seed < 2**32:
        parser.error("--count must be positive and --seed must be in [0, 2**32)")
    rng = np.random.default_rng(args.seed)
    payload = {
        "schema_id": STATE_SCHEMA,
        "schema_version": 5,
        "generator": {"seed": args.seed},
        "states": [sample_state(rng, state_id=f"ring-stack-s{args.seed}-{i:04d}") for i in range(args.count)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
