#!/usr/bin/env python3
"""Fail-fast source/runtime probe for the NewtonGL material and light path."""

from __future__ import annotations

import inspect
from pathlib import Path


def main() -> int:
    from newton._src.usd import utils as usd_utils
    from newton._src.viewer import viewer
    from newton._src.viewer.gl import opengl, shaders

    usd_source = inspect.getsource(usd_utils)
    viewer_source = inspect.getsource(viewer)
    shader_source = inspect.getsource(shaders)
    renderer_source = inspect.getsource(opengl.RendererGL)
    checks = {
        "diffuseColor": "diffuseColor" in usd_source and "ObjectColor" in shader_source,
        "roughness": "roughness" in usd_source and "float roughness" in shader_source,
        "metallic": "metallic" in usd_source and "float metallic" in shader_source,
        "albedo_texture": "texture" in viewer_source and "albedo_map" in shader_source,
        "normal_map": "normal_map" in shader_source or "normal texture" in shader_source.lower(),
        "opacity": "FragColor = vec4(color, 1.0)" not in shader_source,
        "emissiveColor": "emissiveColor" in usd_source and "emissive" in shader_source,
        "directional_sun": "sun_direction" in shader_source,
        "camera_spotlight": "spotlight_enabled" in renderer_source,
        "shadow_map": "draw_shadows" in renderer_source and "ShadowCalculation" in shader_source,
        "point_lights": "point_light" in renderer_source or "pointLights" in shader_source,
    }
    print(f"Newton USD parser: {Path(inspect.getsourcefile(usd_utils) or '').resolve()}")
    print(f"NewtonGL shader: {Path(inspect.getsourcefile(shaders) or '').resolve()}")
    for name, supported in checks.items():
        print(f"{name:20s}: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    required = ("diffuseColor", "roughness", "metallic", "directional_sun", "shadow_map")
    return 0 if all(checks[name] for name in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
