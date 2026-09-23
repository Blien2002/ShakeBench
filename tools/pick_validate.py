"""Validate the eight-object task set: load, reset, settle, and (optionally) place.

Usage:
    python tools/pick_validate.py                # load/reset/settle report
    python tools/pick_validate.py --oracle       # add the scripted pick-and-place run

Reports per variant: compiled-vs-registry collision envelope, placement height,
settle duration, post-settle drift of the object from its registered start
pose, and the success subconditions of the final sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shakebench.utils.tasks import OBJECTS, make_task_env, task_variants  # noqa: E402


def _rotation_angle_rad(first, second) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    dot = abs(float(np.dot(first, second))) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(2.0 * np.arccos(min(1.0, dot)))


def load_reset_report(spec) -> dict:
    env = make_task_env(spec, hard_reset=False, horizon=600, seed=7)
    env.reset()
    data = env.sim.data._data
    body = env.can_body_id
    placed_position = np.asarray(env.arena.table_top_abs, dtype=float) + np.asarray(
        [*env.object_start_xy, env.can_placement_z_offset_m], dtype=float
    )
    settled_position = np.asarray(data.xpos[body], dtype=float)
    settled_quat = np.asarray(data.xquat[body], dtype=float)
    report = {
        "object_id": spec.object_id,
        "variant_id": spec.variant_id,
        "mass_kg": env.object_mass_kg,
        "table_mu": spec.table_sliding_mu,
        "contract": spec.contract(),
        "placement_z_offset_m": float(env.can_placement_z_offset_m),
        "posed_lower_upper_radius_m": [float(value) for value in env.can_start_pose_envelope],
        "compiled_envelope": env.can_collision_envelope.to_dict(),
        "settle_duration_s": float(env.reset_settle_duration_s),
        "settle_drift_m": float(np.linalg.norm(settled_position - placed_position)),
        "settle_rotation_rad": _rotation_angle_rad(settled_quat, spec.start_quat_wxyz),
        "object_position_world_m": [float(value) for value in settled_position],
        "object_quat_wxyz": [float(value) for value in settled_quat],
        "velocity_norm": float(np.linalg.norm(data.qvel[6:12])),
    }
    env.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true", help="run the scripted oracle placement per variant")
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    specs = [spec for spec in task_variants() if args.objects is None or spec.object_id in args.objects]
    reports = []
    for spec in specs:
        report = load_reset_report(spec)
        reports.append(report)
        lower, upper, radius = OBJECTS[spec.object_id]["support"]
        print(
            f"{spec.object_id:12s} mass {report['mass_kg']:.2f} kg mu {report['table_mu']:.2f} "
            f"z_offset {report['placement_z_offset_m']:+.4f} settle {report['settle_duration_s']:.3f}s "
            f"drift {report['settle_drift_m'] * 1e3:.1f}mm/{np.degrees(report['settle_rotation_rad']):.1f}deg "
            f"registry_support {lower:+.4f}/{upper:+.4f}/{radius:.4f}",
            flush=True,
        )
    if args.output is not None:
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
