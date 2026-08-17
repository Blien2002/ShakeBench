#!/usr/bin/env python3
"""Quantitative visual regression audit for benchmark videos and stills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROI = (200, 520, 300, 900)
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be y0,y1,x0,x1")
    y0, y1, x0, x1 = values
    if min(values) < 0 or y0 >= y1 or x0 >= x1:
        raise argparse.ArgumentTypeError("ROI must satisfy 0 <= y0 < y1 and 0 <= x0 < x1")
    return values


def read_frame(path: Path, time_s: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an RGB frame from a still image or time-indexed video."""

    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return np.asarray(Image.open(path).convert("RGB")), {"source_kind": "image", "time_s": None}
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 30.0))
        frame_index = max(0, int(round(time_s * fps)))
        frame = np.asarray(reader.get_data(frame_index))[..., :3]
    finally:
        reader.close()
    return frame, {"source_kind": "video", "time_s": time_s, "frame_index": frame_index, "fps": fps}


def audit_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[dict[str, Any], np.ndarray]:
    """Return luminance/saturation metrics and the audited RGB ROI."""

    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("expected an HxWx3 RGB frame")
    y0, y1, x0, x1 = roi
    if y1 > frame.shape[0] or x1 > frame.shape[1]:
        raise ValueError(f"ROI {roi} exceeds frame size {frame.shape[1]}x{frame.shape[0]}")
    crop = np.asarray(frame[y0:y1, x0:x1, :3], dtype=np.float32)
    luminance = np.rint(0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2])
    high = crop.max(axis=2)
    low = crop.min(axis=2)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0.0)
    quantiles = np.percentile(luminance, PERCENTILES)
    metrics: dict[str, Any] = {
        "min": float(luminance.min()),
        "max": float(luminance.max()),
        "mean": float(luminance.mean()),
        "std": float(luminance.std()),
        "percentiles": {f"p{p}": float(value) for p, value in zip(PERCENTILES, quantiles)},
        "iqr": float(quantiles[4] - quantiles[2]),
        "narrow_band_120_215_ratio": float(((luminance >= 120.0) & (luminance <= 215.0)).mean()),
        "saturation_mean": float(saturation.mean()),
        "low_saturation_lt_0_08_ratio": float((saturation < 0.08).mean()),
    }
    return metrics, crop.astype(np.uint8)


def audit_regions(frame: np.ndarray, config_path: Path) -> dict[str, Any]:
    """Measure named material regions and scene-wide warm/cool consistency."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    regions: dict[str, Any] = {}
    warmth_values: list[float] = []
    for entry in payload["regions"]:
        y0, y1, x0, x1 = (int(value) for value in entry["roi"])
        if y0 < 0 or x0 < 0 or y1 > frame.shape[0] or x1 > frame.shape[1] or y0 >= y1 or x0 >= x1:
            raise ValueError(f"invalid region ROI for {entry['name']}: {entry['roi']}")
        crop = np.asarray(frame[y0:y1, x0:x1, :3], dtype=np.float32)
        rgb = crop.mean(axis=(0, 1))
        luminance = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        red_minus_blue = float(rgb[0] - rgb[2])
        regions[entry["name"]] = {
            "roi_y0_y1_x0_x1": [y0, y1, x0, x1],
            "mean_rgb": [float(value) for value in rgb],
            "mean_luminance": luminance,
            "mean_red_minus_blue": red_minus_blue,
            "excluded_from_warmth_span": bool(entry.get("exclude_from_warmth_span", False)),
        }
        if not entry.get("exclude_from_warmth_span", False):
            warmth_values.append(red_minus_blue)
    adjacent: dict[str, Any] = {}
    for lhs, rhs in payload.get("adjacent_large_surfaces", []):
        delta = abs(regions[lhs]["mean_luminance"] - regions[rhs]["mean_luminance"])
        adjacent[f"{lhs}__{rhs}"] = {"luminance_delta": float(delta), "gte_25": bool(delta >= 25.0)}
    span = max(warmth_values) - min(warmth_values) if warmth_values else 0.0
    return {
        "regions": regions,
        "warmth_span_red_minus_blue": float(span),
        "adjacent_large_surfaces": adjacent,
        "gates": {
            "warmth_span_lte_50": bool(span <= 50.0),
            "adjacent_luminance_delta_gte_25": bool(all(item["gte_25"] for item in adjacent.values())),
        },
    }


def save_histogram(metrics: dict[str, Any], roi: np.ndarray, path: Path) -> None:
    """Save a dependency-light luminance histogram as a PNG."""

    luminance = np.rint(0.2126 * roi[..., 0] + 0.7152 * roi[..., 1] + 0.0722 * roi[..., 2]).astype(np.uint8)
    histogram = np.bincount(luminance.ravel(), minlength=256).astype(np.float64)
    histogram /= max(histogram.max(), 1.0)
    image = Image.new("RGB", (900, 420), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    plot = (64, 48, 868, 360)
    draw.rectangle(plot, fill=(255, 255, 255), outline=(80, 90, 100), width=1)
    for value in (120, 215):
        x = plot[0] + value / 255.0 * (plot[2] - plot[0])
        draw.line((x, plot[1], x, plot[3]), fill=(230, 155, 40), width=2)
    points = [
        (
            plot[0] + value / 255.0 * (plot[2] - plot[0]),
            plot[3] - probability * (plot[3] - plot[1]),
        )
        for value, probability in enumerate(histogram)
    ]
    draw.line(points, fill=(30, 100, 170), width=2)
    draw.text((64, 14), f"Luminance histogram | mean={metrics['mean']:.2f} std={metrics['std']:.2f} IQR={metrics['iqr']:.2f}", fill=(20, 30, 40), font=font)
    draw.text((64, 376), "0", fill=(40, 40, 40), font=font)
    draw.text((842, 376), "255", fill=(40, 40, 40), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def comparison_table(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    """Return a compact Markdown comparison table."""

    keys = (
        ("mean", "Mean"),
        ("std", "Std"),
        ("iqr", "IQR"),
        ("narrow_band_120_215_ratio", "Narrow-band ratio"),
        ("low_saturation_lt_0_08_ratio", "Low-saturation ratio"),
    )
    lines = ["| Metric | Baseline | Current | Delta |", "|---|---:|---:|---:|"]
    for key, label in keys:
        old = float(baseline["metrics"][key])
        new = float(current["metrics"][key])
        lines.append(f"| {label} | {old:.4f} | {new:.4f} | {new - old:+.4f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--time-s", type=float, default=5.5)
    parser.add_argument("--roi", type=_parse_roi, default=DEFAULT_ROI, metavar="Y0,Y1,X0,X1")
    parser.add_argument("--histogram", type=Path)
    parser.add_argument("--frame-output", type=Path)
    parser.add_argument("--baseline", type=Path, help="Save this result as the baseline JSON")
    parser.add_argument("--compare", type=Path, help="Compare against a saved baseline JSON")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--regions-config", type=Path, help="YAML table of named material-region ROIs")
    args = parser.parse_args()

    frame, source = read_frame(args.input, args.time_s)
    metrics, roi_image = audit_frame(frame, args.roi)
    result = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        **source,
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "roi_y0_y1_x0_x1": list(args.roi),
        "metrics": metrics,
        "gates": {
            "iqr_gte_60": metrics["iqr"] >= 60.0,
            "std_gte_55": metrics["std"] >= 55.0,
            "narrow_band_lt_0_50": metrics["narrow_band_120_215_ratio"] < 0.50,
        },
    }
    if args.regions_config:
        result["region_audit"] = audit_regions(frame, args.regions_config)
    if args.histogram:
        save_histogram(metrics, roi_image, args.histogram)
        result["histogram"] = str(args.histogram.resolve())
    if args.frame_output:
        args.frame_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args.frame_output)
        result["frame_output"] = str(args.frame_output.resolve())
    payload = json.dumps(result, indent=2) + "\n"
    output = args.baseline or args.json_output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        print(comparison_table(baseline, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
