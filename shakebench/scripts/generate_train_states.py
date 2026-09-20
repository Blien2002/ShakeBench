"""Write a deterministic train-split state artifact for demonstration collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from shakebench.utils.artifacts import write_json
from shakebench.utils.train_states import TRAIN_XY_HALF_RANGE_M, max_xy_half_range_m
from shakebench.utils.train_states import build_train_state_artifact, verify_train_state_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New train-state JSON path; never overwritten")
    parser.add_argument("--count", type=int, required=True, help="Number of train states to generate")
    parser.add_argument("--seed", type=int, required=True, help="Generation seed; same seed reproduces the pool")
    parser.add_argument(
        "--half-range-m",
        type=float,
        default=TRAIN_XY_HALF_RANGE_M,
        help=f"Square half range in metres; checked ceiling is {max_xy_half_range_m():.4f}",
    )
    parser.add_argument(
        "--objects",
        default=None,
        help=(
            "Comma-separated registry object ids to sample; every object contributes its registered body "
            "pose, and the mug contributes both its upright and its lying double-wall pose"
        ),
    )
    parser.add_argument(
        "--per-variant",
        type=int,
        default=None,
        help=(
            "Episodes to place on each object-pose-grip entry of the pool, filled in order instead of "
            "sampled at random; --count is normally --per-variant times that pool size"
        ),
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    object_ids = None if args.objects is None else [value.strip() for value in args.objects.split(",") if value.strip()]
    payload = build_train_state_artifact(
        args.count,
        seed=args.seed,
        half_range_m=args.half_range_m,
        object_ids=object_ids,
        per_variant=args.per_variant,
    )
    verdict = verify_train_state_artifact(payload)
    if not verdict["passed"]:
        raise RuntimeError(f"generated train artifact failed verification: {verdict['checks']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(f"{len(payload['states'])} train states -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
