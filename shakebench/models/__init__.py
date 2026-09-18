"""ShakeBench MJCF models and the package asset root.

ShakeBench owns its arena, task objects, textures and JSON contracts.  A few
resources stay upstream (robosuite materials and meshes that the tasks reuse);
those are always addressed through robosuite_assets_root by an explicit call,
never by searching several directories in turn.
"""

import os

from robosuite import models as _robosuite_models
from robosuite.utils.mjcf_utils import xml_path_completion as _xml_path_completion

assets_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
robosuite_assets_root = _robosuite_models.assets_root


def xml_path_completion(xml_path, root=None):
    """Return the absolute path of a ShakeBench package asset."""

    return _xml_path_completion(xml_path, root=assets_root if root is None else root)
