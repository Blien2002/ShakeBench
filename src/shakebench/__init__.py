"""ShakeBench: an embodied-intelligence benchmark for manipulation under vibration.

ShakeBench is a standalone Isaac Lab + Newton/MJWarp benchmark in which a Franka
Panda performs a pick-and-place task while its base and worktable receive
independent six-axis vibration through a shared C2-mounted Stewart platform.

Heavy Isaac Lab modules are loaded lazily through :func:`__getattr__` so that
``shakebench.config``, ``shakebench.vibration`` and the offline Gamma
calibration test suite import without Isaac Lab installed.
"""

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
    "solve_leg_transforms",
]

_LAZY_EXPORTS = {
    "BenchmarkSceneCfg": ("scene", "BenchmarkSceneCfg"),
    "make_scene_cfg": ("scene", "make_scene_cfg"),
    "make_sim_cfg": ("scene", "make_sim_cfg"),
    "ShakerGeometryCfg": ("shaker", "ShakerGeometryCfg"),
    "solve_leg_transforms": ("shaker", "solve_leg_transforms"),
    "SpectralVibration": ("vibration", "SpectralVibration"),
    "VibrationBenchmarkTask": ("task", "VibrationBenchmarkTask"),
}


def __getattr__(name: str):
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
