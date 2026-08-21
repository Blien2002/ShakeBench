"""Vision-language-action adapter extension point."""

from .dp import DiffusionPolicyAdapter


class VLAAdapter(DiffusionPolicyAdapter):
    pass
