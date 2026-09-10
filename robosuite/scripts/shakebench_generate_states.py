"""Generate and freeze the Phase 08 official and knee state authorities only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robosuite.utils.shakebench_committed_states import freeze_committed_state_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--task-variants",
        action="store_true",
        help="prepare Phase 09 v2 states: every placement crossed with six task variants",
    )
    args = parser.parse_args(argv)
    if args.task_variants:
        from robosuite.utils.shakebench_task_states import freeze_task_state_assets

        paths = freeze_task_state_assets(args.output_dir)
    else:
        paths = freeze_committed_state_assets(args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
