"""ShakeBench MJCF models and the package asset root.

Package-owned paths are available without initializing robosuite or graphics.
Upstream materials and meshes are resolved when their asset root is requested.
"""

import os

assets_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def __getattr__(name):
    if name != "robosuite_assets_root":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from robosuite import models

    root = models.assets_root
    globals()[name] = root
    return root


def xml_path_completion(xml_path, root=None):
    """Return the absolute path of a ShakeBench package asset."""
    from robosuite.utils.mjcf_utils import xml_path_completion as upstream_completion

    return upstream_completion(xml_path, root=assets_root if root is None else root)
