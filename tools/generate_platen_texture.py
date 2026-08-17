#!/usr/bin/env python3
"""Generate the deterministic threaded-hole platen albedo."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "textures" / "platen_threaded_holes_1k.jpg"


def main() -> None:
    scale = 3
    width, height = 1024 * scale, 704 * scale
    image = Image.new("RGB", (width, height), (166, 170, 173))
    draw = ImageDraw.Draw(image)
    margin_x, margin_y = 102 * scale, 82 * scale
    rows, columns = 7, 11
    radius = 8 * scale
    for row in range(rows):
        y = margin_y + row * (height - 2 * margin_y) / (rows - 1)
        for column in range(columns):
            x = margin_x + column * (width - 2 * margin_x) / (columns - 1)
            # Bright bevel ring, softly shaded wall and dark lower bore create
            # depth without adding dozens of Newton render shapes.
            draw.ellipse((x - radius - 2 * scale, y - radius - 2 * scale, x + radius + 2 * scale, y + radius + 2 * scale), fill=(194, 198, 200))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(76, 80, 83))
            draw.ellipse((x - 0.58 * radius, y - 0.58 * radius, x + 0.58 * radius, y + 0.58 * radius), fill=(48, 52, 55))
            draw.arc((x - 0.84 * radius, y - 0.84 * radius, x + 0.84 * radius, y + 0.84 * radius), 195, 345, fill=(226, 229, 230), width=2 * scale)
            draw.arc((x - 0.70 * radius, y - 0.70 * radius, x + 0.70 * radius, y + 0.70 * radius), 15, 165, fill=(38, 42, 45), width=2 * scale)
    image = image.resize((1024, 704), Image.Resampling.LANCZOS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, quality=95, subsampling=0, optimize=False, progressive=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
