"""Package structure: every module imports, layering holds, robosuite stays external."""

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "shakebench"
OPTIONAL = {"torch", "safetensors", "draccus", "deployment", "gymnasium", "gym", "h5py"}
CORE = {"environments", "tasks", "physics", "sensors", "scene", "models", "utils"}
UPPER = {"rollout", "policies", "evaluation", "wrappers"}
ENTRY = {"scripts", "demos"}
MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__") for path in PACKAGE.rglob("*.py")
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in OPTIONAL:
            pytest.skip(f"optional dependency {exc.name} is not installed")
        raise


def _imported_areas(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == "shakebench" and len(parts) > 1:
                yield parts[1], name


def test_layering():
    problems = []
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE).parts
        area = parts[0] if len(parts) > 1 else None
        for target, name in _imported_areas(path):
            if (target in ENTRY and area not in ENTRY) or (area in CORE and target in UPPER):
                problems.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not problems, "\n".join(problems)


def test_tasks_use_the_shakebench_seam():
    from robosuite.environments.base import REGISTERED_ENVS

    from shakebench.environments.base import ShakeBenchEnv

    tasks = {name: cls for name, cls in REGISTERED_ENVS.items() if cls.__module__.startswith("shakebench.")}
    expected = {"VibrationPickPlace", "PanelOperation", "PushT", "RingOnPeg", "RingOnRail", "StackBlocks"}
    assert expected | {"Upright", "WineRack"} <= set(tasks)
    assert all(issubclass(cls, ShakeBenchEnv) for cls in tasks.values())
    assert not {"ShakeBenchEnv", "ShakeBenchTask"} & set(REGISTERED_ENVS)


def test_robosuite_is_an_external_dependency():
    import robosuite

    assert not (ROOT / "robosuite").exists()
    assert ROOT not in Path(robosuite.__file__).resolve().parents
