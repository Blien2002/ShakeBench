"""Run the optional four-scenario suite through the existing evaluation interface.

Example::

    python -m shakebench.scripts.evaluate_scenarios --output-dir out/scenarios \
        --policy my_policy:make_policy --state-ids shakebench-dev-v0-000

The suite is an experimental workload. Gamma is an amplitude multiplier here;
these runs do not replace the existing benchmark peak-Gamma grid.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from shakebench import models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=1.0, help="Scenario amplitude multiplier")
    parser.add_argument("--level", type=float, default=1.0, help="Additional scenario amplitude multiplier")
    parser.add_argument("--backend", choices=("cpu",), default="cpu")
    args, evaluation_args = parser.parse_known_args(argv)
    reserved = {"--mode", "--mode-params", "--sway-v1", "--output"}
    if any(item.split("=", 1)[0] in reserved for item in evaluation_args):
        parser.error("mode, mode-params and output are owned by the suite")
    suite = json.loads(Path(models.assets_root, "shakebench_vibration_suite_v2.json").read_text())
    output_paths = [args.output_dir / (item["scenario"] + ".json") for item in suite["scenarios"]]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite an existing scenario result")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = "shakebench.scripts.evaluate" if args.backend == "cpu" else "shakebench.scripts.evaluate_gpu"
    evaluate = importlib.import_module(module).main
    for item, output in zip(suite["scenarios"], output_paths):
        params = {**item, "level": args.level}
        result = evaluate(
            [
                *evaluation_args,
                "--mode",
                suite["mode"],
                "--mode-params",
                json.dumps(params),
                "--gamma",
                str(args.gamma),
                "--output",
                str(output),
            ]
        )
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
