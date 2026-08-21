"""ShakeBench: an embodied-intelligence benchmark for manipulation under vibration.

ShakeBench is a standalone Isaac Lab + Newton/MJWarp benchmark in which a Franka
Panda performs a pick-and-place task while its base and worktable receive
independent six-axis vibration through a shared C2-mounted Stewart platform.

Heavy Isaac Lab modules are loaded lazily through :func:`__getattr__` so that
``shakebench.config``, ``shakebench.vibration`` and the offline Gamma
calibration test suite import without Isaac Lab installed.
"""

from pathlib import Path
import json
from typing import Any

from .config import AssetConfig, BenchmarkConfig, PanelConfig, SpectralBand, VibrationConfig

__version__ = "0.2.0"

__all__ = [
    "AssetConfig",
    "BenchmarkConfig",
    "BenchmarkSceneCfg",
    "PanelConfig",
    "SpectralBand",
    "SpectralVibration",
    "ShakerGeometryCfg",
    "VibrationBenchmarkTask",
    "VibrationConfig",
    "__version__",
    "make_scene_cfg",
    "make_sim_cfg",
    "make",
    "load_controller_config",
    "benchmark",
    "solve_leg_transforms",
]

_LAZY_EXPORTS = {
    "BenchmarkSceneCfg": ("models.scene", "BenchmarkSceneCfg"),
    "make_scene_cfg": ("models.scene", "make_scene_cfg"),
    "make_sim_cfg": ("models.scene", "make_sim_cfg"),
    "ShakerGeometryCfg": ("models.supports.shaker", "ShakerGeometryCfg"),
    "solve_leg_transforms": ("models.supports.shaker", "solve_leg_transforms"),
    "SpectralVibration": ("vibration", "SpectralVibration"),
    "VibrationBenchmarkTask": ("envs.manipulation.pick_place", "VibrationBenchmarkTask"),
}


def load_controller_config(name: str) -> dict[str, Any]:
    """Load one of the versioned controller JSON configurations."""

    normalized = name.upper()
    path = Path(__file__).resolve().parent / "controllers" / "config" / f"{normalized}.json"
    if not path.is_file():
        available = sorted(item.stem for item in path.parent.glob("*.json"))
        raise ValueError(f"unknown controller config {name!r}; available={available}")
    return json.loads(path.read_text(encoding="utf-8"))


def make(env_name: str, **kwargs: Any):
    """Construct a policy-facing ShakeBench environment, like robosuite.make."""

    from .envs import make_env

    return make_env(env_name, **kwargs)


def __getattr__(name: str):
    if name == "benchmark":
        import importlib

        value = importlib.import_module(".benchmark", __name__)
        globals()[name] = value
        return value
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
