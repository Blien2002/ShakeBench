"""Collect colocated native MuJoCo and ShakeBench IMUs in one live rollout.

Run: python -m shakebench.scripts.compare_table_imu --output-dir out/imu_comparison
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from shakebench.environments.vibration_pick_place import VibrationPickPlace


def add_native_imu(xml):
    root = ET.fromstring(xml)
    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    for kind in ("accelerometer", "gyro"):
        ET.SubElement(sensors, kind, name="comparison_" + kind, site="table_imu_site")
    return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    vibration = {"mode": "multisine_v1", "gamma": args.gamma, "seed": 42, "t0_s": 0.0}
    env = VibrationPickPlace(
        physics_profile="probe",
        vibration=vibration,
        seed=42,
        imu_seed=42,
        imu_mode="canonical_noisy_v1",
        load_model_on_init=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        hard_reset=False,
        horizon=40,
    )
    env.set_xml_processor(add_native_imu)
    rows = []
    try:
        env.reset()
        model = env.sim.model._model
        scratch = mujoco.MjData(model)
        indices = []
        for kind in ("accelerometer", "gyro"):
            sensor = model.sensor("comparison_" + kind)
            address = int(sensor.adr[0])
            indices.extend(range(address, address + 3))

        def collect(time_s, policy_step=False):
            sample = env.table_imu_provider.imu.last_sample
            if sample is None or (rows and sample.acquisition_time_s <= rows[-1][0]):
                return
            # Evaluate native sensors on the exact derived state used by our
            # provider, without forwarding/integrating or changing the rollout.
            mujoco.mj_copyData(scratch, model, env.sim.data._data)
            mujoco.mj_sensorVel(model, scratch)
            mujoco.mj_sensorAcc(model, scratch)
            native = scratch.sensordata[indices].copy()
            rows.append(
                np.r_[
                    sample.acquisition_time_s,
                    sample.delivered_acquisition_time_s,
                    native,
                    sample.clean_measurement,
                    sample.delivered_measurement,
                ]
            )

        env.add_post_physics_step_hook(collect)
        for step in range(40):
            env.step(np.zeros(env.action_dim))
            if (step + 1) % 10 == 0:
                print(f"Collected {(step + 1) / 20:.1f} s", flush=True)
        values = np.asarray(rows)
        assert values.shape == (400, 20) and np.isfinite(values).all()
        np.testing.assert_allclose(np.diff(values[:, 0]), 0.005, atol=1e-12)
        channels = ["ax_m_s2", "ay_m_s2", "az_m_s2", "gx_rad_s", "gy_rad_s", "gz_rad_s"]
        header = ["time_s", "delivered_source_time_s"] + [
            prefix + channel for prefix in ("native_", "clean_", "delivered_") for channel in channels
        ]
        with (args.output_dir / "paired.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(values)
        for name, start in (("mujoco_native", 2), ("shakebench_delivered", 14)):
            with (args.output_dir / (name + ".csv")).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s"] + channels)
                writer.writerows(np.c_[values[:, 0], values[:, start : start + 6]])
        difference = values[:, 8:14] - values[:, 2:8]
        delivered_difference = values[:, 14:20] - values[:, 2:8]
        summary = {
            "mujoco_version": mujoco.__version__,
            "vibration": vibration,
            "physics_profile": "probe",
            "duration_s": 2.0,
            "samples": len(values),
            "sample_rate_hz": 200,
            "site": "table_imu_site",
            "body": "worktable",
            "reset_settle_duration_s": env.reset_settle_duration_s,
            "native_settings": "default accelerometer + gyro; no added noise/filter/delay",
            "same_derived_state": True,
            "channels": channels,
            "clean_vs_native_max_abs": np.max(np.abs(difference), axis=0).tolist(),
            "delivered_vs_native_rmse": np.sqrt(np.mean(delivered_difference**2, axis=0)).tolist(),
            "imu_profile": env.table_imu_provider.imu.profile.to_dict(),
            "example_at_1s": dict(zip(header, values[199].tolist())),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for index in range(6):
            ax = axes[index % 3, index // 3]
            ax.plot(values[:, 0], values[:, 2 + index], label="MuJoCo native", lw=1.5)
            ax.plot(values[:, 0], values[:, 8 + index], label="ShakeBench clean", lw=1, ls="--")
            ax.plot(values[:, 0], values[:, 14 + index], label="ShakeBench delivered", lw=1, alpha=0.85)
            ax.set_ylabel(channels[index])
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("Simulation time (s)")
        fig.suptitle(f"Same worktable IMU site | multisine, Gamma={args.gamma:g}, seed=42 | 200 Hz")
        fig.tight_layout()
        fig.savefig(args.output_dir / "comparison.png", dpi=160)
        plt.close(fig)
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in ("mujoco_version", "samples", "clean_vs_native_max_abs", "delivered_vs_native_rmse")
                }
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
