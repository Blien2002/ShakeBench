"""Generate reproducible starts for all three upright object variants."""

import argparse
import json
from pathlib import Path

import numpy as np

from shakebench.environments.upright import OBJECT_IDS, SCHEMA_VERSION, STATE_SCHEMA, sample_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    args = parser.parse_args(argv)
    if args.count <= 0 or not 0 <= args.seed < 2**32:
        parser.error("--count must be positive and --seed must be in [0, 2**32)")
    rng = np.random.default_rng(args.seed)
    payload = {
        "schema_id": STATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": {"seed": args.seed, "split": args.split},
        "states": [
            sample_state(
                rng,
                object_id=OBJECT_IDS[i % len(OBJECT_IDS)],
                split=args.split,
                state_id=f"upright-{OBJECT_IDS[i % len(OBJECT_IDS)]}-s{args.seed}-{i:04d}",
            )
            for i in range(args.count)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
