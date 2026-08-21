# Code structure

ShakeBench uses a single `src/` package layout.

| Path | Responsibility |
|---|---|
| `benchmark/` | LIBERO-style task descriptors, suite registry, and HDF5 loading |
| `envs/` | policy-rate scheduling, Gymnasium API, manipulation tasks, wrappers |
| `controllers/` | joint-position, OSC pose, and variable-impedance action transforms |
| `policies/` | policy protocol, scripted compatibility, random sentinel, adapters |
| `models/` | arena/object/robot/support extension points |
| `vibration/` | analytic spectral synthesis, Γ calibration, displacement gates, IMU |
| `sensors/` | physical wrist-camera model |
| `init_files/` | 50 phase-complete initial states per suite task |
| `config.py` | validated benchmark, asset, panel, and vibration dataclasses |
| `scene.py` | Isaac/Newton scene and solver construction |

The legacy module imports (`shakebench.task`, `shakebench.panel_task`, `shakebench.vibration`, and `shakebench.wrist_camera`) remain compatibility surfaces. The physical support pose has one source of truth in `supports.py`: every hard-mounted member uses `p=q+c+R(l-c)`.
