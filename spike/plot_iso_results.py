#!/usr/bin/env python3
"""Render the measured/analytic isolator transmissibility comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "out" / "iso" / "transmissibility_check.json"
OUTPUT = ROOT / "out" / "iso" / "transmissibility_curve.png"


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/shakebench-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/shakebench-cache")
    import matplotlib.pyplot as plt

    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = payload["checks"]["transmissibility_vs_analytic"]["rows"]

    grouped: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        damping_ratio = float(row["parameters"]["damping_ratio"])
        grouped.setdefault(damping_ratio, []).append(row)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = {0.05: "#d95f02", 0.10: "#1b78b4", 0.20: "#2b9348"}
    for damping_ratio, group in sorted(grouped.items()):
        group.sort(key=lambda item: float(item["parameters"]["natural_frequency_hz"]))
        frequencies = [float(item["parameters"]["natural_frequency_hz"]) for item in group]
        analytic = [float(item["transmissibility_analytic"]) for item in group]
        measured = [float(item["transmissibility_measured"]) for item in group]
        color = colors[damping_ratio]
        ax.plot(
            frequencies,
            analytic,
            color=color,
            linewidth=1.8,
            label=f"analytic, zeta={damping_ratio:.2f}",
        )
        ax.scatter(
            frequencies,
            measured,
            color=color,
            marker="o",
            s=42,
            edgecolor="white",
            linewidth=0.7,
            label=f"measured, zeta={damping_ratio:.2f}",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Natural frequency (Hz)")
    ax.set_ylabel("Table/base transfer ratio")
    excitation_hz = float(rows[0]["excitation_frequency_hz"])
    ax.set_title(f"ShakeBench isolator transmissibility at {excitation_hz:g} Hz excitation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
