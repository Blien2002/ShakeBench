"""ShakeBench: the shaken-worktable manipulation benchmark built on robosuite.

The public environment and make exports register the default ShakeBench tasks
when first requested. Package import alone leaves simulator and renderer setup
until runtime, so command-line help also works without a graphics driver.

Optional policy dependencies load inside the adapter that needs them.
"""

__all__ = ["PanelOperation", "ShakeBenchArena", "VibrationPickPlace", "make"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from robosuite.environments.base import make

    from shakebench.environments.panel_operation import PanelOperation
    from shakebench.environments.vibration_pick_place import VibrationPickPlace
    from shakebench.models.arenas import ShakeBenchArena

    exports = {
        "PanelOperation": PanelOperation,
        "ShakeBenchArena": ShakeBenchArena,
        "VibrationPickPlace": VibrationPickPlace,
        "make": make,
    }
    globals().update(exports)
    return exports[name]
