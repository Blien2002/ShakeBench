"""Rebuild the panel's printed face and label atlas using Pillow.

python -m shakebench.scripts.build_panel_markings
The shipped PNG is a runtime asset; fonts are only needed to rebuild it.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shakebench.models import xml_path_completion


def build_markings(output):
    """Draw millimetre-aligned legends, graduations and three nameplates."""
    width, height, face_height = 1536, 1280, 972
    rng = np.random.default_rng(11)
    grain = rng.normal(0, 0.55, (height, width, 1))
    pixels = np.clip(np.array([36, 46, 55]) + grain, 0, 255).astype(np.uint8)
    atlas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(atlas)
    scale = width / 302
    white, muted, accent = (224, 230, 230), (150, 168, 177), (81, 166, 183)

    def point(x, y):
        # Face coordinates in mm: right is -panel Y, up follows the slope.
        return ((x + 151) * scale, (95.5 - y) / 191 * face_height)

    def font(size, bold=False):
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", round(size * scale))

    def label(x, y, text, size=3.0, color=white, bold=False):
        draw.text(point(x, y), text, font=font(size, bold), fill=color, anchor="mm")

    def line(start, end, color=muted, thickness=0.25):
        draw.line([point(*start), point(*end)], fill=color, width=max(1, round(thickness * scale)))

    # Restrained silkscreen on the existing face; the lamps remain unlit.
    label(0, 87.5, "SHAKEBENCH", 4.8, bold=True)
    label(0, 41, "CONTROL PANEL", 3.4, muted)
    line((-30, 35), (30, 35), accent, 0.35)
    for i in range(24):
        angle = i * np.pi / 12
        inner, outer = (36.8, 39.5) if i % 2 == 0 else (37.7, 39.5)
        start = (85 + inner * np.cos(angle), 55 + inner * np.sin(angle))
        end = (85 + outer * np.cos(angle), 55 + outer * np.sin(angle))
        line(start, end, white, 0.32 if i % 2 == 0 else 0.20)
    # A fine outline makes the pushbutton mounting washer legible in shadow.
    center_x, center_y = point(-85, 55)
    radius = 34.0 * scale
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=muted, width=2)
    for y, text in ((-29, "+"), (-55, "0"), (-81, "−")):
        line((-34, y), (-38, y), white, 0.35)
        label(-43, y, text, 3.2)
    label(-92, -63, "SB / 01", 4.2, bold=True)
    label(-92, -70, "OPERATOR CONSOLE", 2.6, muted)
    label(91, -66, "PUSH · TURN · TOGGLE", 2.5, muted)
    line((-120, -80), (-64, -80), accent, 0.4)

    # Atlas UVs in panel.xml address these 512 x 152 pixel nameplates.
    for i, text in enumerate(("ROTARY", "TOGGLE", "PUSH")):
        x = i * 512
        draw.rectangle((x, 1000, x + 511, 1151), fill=(20, 27, 33))
        draw.rounded_rectangle((x + 5, 1005, x + 506, 1146), radius=8, outline=(145, 160, 167), width=3)
        draw.text((x + 256, 1076), text, font=ImageFont.truetype("DejaVuSans-Bold.ttf", 49), fill=white, anchor="mm")
        draw.line((x + 150, 1127, x + 362, 1127), fill=accent, width=3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output, optimize=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(xml_path_completion("objects/panel/panel_markings.png")))
    args = parser.parse_args(argv)
    print(build_markings(args.output).resolve())


if __name__ == "__main__":
    main()
