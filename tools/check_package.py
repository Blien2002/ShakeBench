"""Public package smoke checks; full simulation regressions are maintained separately."""

import argparse
import hashlib
import importlib
import json
import os
import pkgutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_MODULES = (
    "shakebench.scripts.evaluate",
    "shakebench.scripts.evaluate_async",
    "shakebench.scripts.eval_websocket_policy",
    "shakebench.demos.demo_policy_video",
)


def check_cli():
    # Help must work even when simulator/graphics imports are impossible.
    code = """import importlib.abc, runpy, sys
class NoSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'robosuite', 'mujoco', 'OpenGL'}:
            raise RuntimeError('help attempted simulator/graphics import: ' + fullname)
sys.meta_path.insert(0, NoSimulation())
module = sys.argv[1]
sys.argv = [module, '--help']
runpy.run_module(module, run_name='__main__', alter_sys=True)
"""
    for module in CLI_MODULES:
        result = subprocess.run([sys.executable, "-c", code, module], capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or "usage:" not in result.stdout:
            raise RuntimeError(f"{module} help failed: {result.stderr[-3000:]}")
    print(json.dumps({"cli_modules": list(CLI_MODULES), "simulator_imports_blocked": True}))


def check_imports():
    # Import MuJoCo's disabled GL path before upstream robosuite chooses its renderer defaults.
    os.environ["MUJOCO_GL"] = "disable"
    import mujoco  # noqa: F401

    import shakebench

    optional = {"torch", "safetensors", "draccus", "deployment"}
    imported, unavailable = [], []
    for item in pkgutil.walk_packages(shakebench.__path__, prefix="shakebench."):
        try:
            importlib.import_module(item.name)
        except ModuleNotFoundError as exc:
            if not exc.name or exc.name.split(".")[0] not in optional:
                raise
            unavailable.append({"module": item.name, "dependency": exc.name})
        else:
            imported.append(item.name)
    # Request the lazy public exports too: default environments must remain registered.
    from robosuite.environments.base import REGISTERED_ENVS

    for name in shakebench.__all__:
        getattr(shakebench, name)
    if not {"PanelOperation", "VibrationPickPlace"} <= REGISTERED_ENVS.keys():
        raise RuntimeError("default public exports did not register environments")
    print(json.dumps({"modules_imported": len(imported), "optional_unavailable": unavailable, "public_exports": True}))


def check_wheel(path):
    expected = {
        str(p.relative_to(ROOT)).replace(os.sep, "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "shakebench").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    with zipfile.ZipFile(path) as wheel:
        names = [n for n in wheel.namelist() if not n.endswith("/")]
        actual = {n: hashlib.sha256(wheel.read(n)).hexdigest() for n in names if n.startswith("shakebench/")}
        forbidden = [n for n in names if n.startswith(("tests/", "docs/")) or "__pycache__" in n or n.endswith(".pyc")]
    if actual != expected or forbidden:
        raise RuntimeError(
            json.dumps(
                {
                    "missing": sorted(expected.keys() - actual.keys()),
                    "extra": sorted(actual.keys() - expected.keys()),
                    "changed": [k for k in expected.keys() & actual.keys() if expected[k] != actual[k]],
                    "forbidden": forbidden,
                }
            )
        )
    print(
        json.dumps(
            {
                "package_files": len(actual),
                "asset_files": sum(n.startswith("shakebench/models/assets/") for n in actual),
                "hashes_match": True,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cli", "imports", "wheel"])
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.mode == "cli":
        check_cli()
    elif args.mode == "imports":
        check_imports()
    else:
        if args.wheel is None:
            parser.error("wheel mode requires --wheel")
        check_wheel(args.wheel)


if __name__ == "__main__":
    main()
