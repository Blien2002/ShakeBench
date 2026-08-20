"""ShakeBench: an embodied-intelligence benchmark for manipulation under vibration.

ShakeBench is a standalone Isaac Lab + Newton/MJWarp benchmark in which a Franka
Panda performs a pick-and-place task while its base and worktable receive
independent six-axis vibration through a shared C2-mounted Stewart platform.
"""

from .config import AssetConfig, BenchmarkConfig, PanelConfig, SpectralBand, VibrationConfig
from .scene import BenchmarkSceneCfg, make_scene_cfg, make_sim_cfg
from .shaker import ShakerGeometryCfg, solve_leg_transforms
from .task import VibrationBenchmarkTask
from .vibration import SpectralVibration

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
