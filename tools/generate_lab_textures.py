#!/usr/bin/env python3
"""Generate deterministic cool-gray laboratory material textures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
TEXTURES = ROOT / "assets" / "textures"


def _save(array: np.ndarray, name: str) -> None:
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    image.save(TEXTURES / name, quality=95, subsampling=0, optimize=False, progressive=False)


def epoxy(rng: np.random.Generator) -> None:
    size = 1024
    y, x = np.mgrid[0:size, 0:size]
    broad = 5.0 * np.sin(2.0 * np.pi * (0.73 * x + 0.24 * y) / size)
    broad += 3.0 * np.cos(2.0 * np.pi * (0.17 * x - 0.91 * y) / size)
    grain = rng.normal(0.0, 1.8, (size, size))
    # NewtonGL's calibrated benchmark exposure lifts this albedo by roughly
    # 30 gray levels. An 86-level source therefore lands near the requested
    # rendered luminance of 118 instead of competing with the platen.
    value = 86.0 + broad + grain
    image = np.stack((value + 3.0, value + 1.0, value - 3.0), axis=-1)
    # Low-frequency cured-resin clouding and a few restrained wear arcs.
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.6))
    draw = ImageDraw.Draw(pil, "RGBA")
    for box, alpha in (((110, 140, 760, 590), 18), ((420, 410, 1100, 930), 12)):
        draw.arc(box, 195, 345, fill=(74, 78, 80, alpha), width=5)
    pil.save(TEXTURES / "epoxy_floor_cool_gray_1k.jpg", quality=95, subsampling=0, optimize=False, progressive=False)


def wall(rng: np.random.Generator) -> None:
    size = 1024
    grain = rng.normal(0.0, 1.25, (size, size))
    base = np.stack((199.0 + grain, 197.0 + grain, 191.0 + grain), axis=-1)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for x in (256, 512, 768):
        draw.line((x, 0, x, size), fill=(181, 181, 178), width=2)
        draw.line((x + 2, 0, x + 2, size), fill=(210, 208, 202), width=1)
    image.save(TEXTURES / "industrial_wall_light_gray_1k.jpg", quality=95, subsampling=0, optimize=False, progressive=False)


def phenolic(rng: np.random.Generator) -> None:
    size = 1024
    y, x = np.mgrid[0:size, 0:size]
    grain = rng.normal(0.0, 1.4, (size, size))
    brushed = 1.2 * np.sin(2.0 * np.pi * y / 137.0) + 0.6 * np.sin(2.0 * np.pi * x / 431.0)
    value = 86.0 + grain + brushed
    _save(np.stack((value - 2.0, value, value + 1.0), axis=-1), "phenolic_bench_dark_1k.jpg")


def main() -> None:
    TEXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1703)
    epoxy(rng)
    wall(rng)
    phenolic(rng)
    for name in (
        "epoxy_floor_cool_gray_1k.jpg",
        "industrial_wall_light_gray_1k.jpg",
        "phenolic_bench_dark_1k.jpg",
    ):
        print(TEXTURES / name)


if __name__ == "__main__":
    main()
