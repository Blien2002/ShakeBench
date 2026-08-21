# Code structure

ShakeBench uses a single `src/` package layout.

| Path | Responsibility |
|---|---|
| `benchmark/` | LIBERO-style task descriptors, suite registry, and HDF5 loading |
| `envs/` | policy-rate scheduling, Gymnasium API, manipulation tasks, wrappers |
| `controllers/` | joint-position, OSC pose, and variable-impedance action transforms |
| `policies/` | policy protocol, scripted compatibility, random sentinel, adapters |
| `models/` | scene construction, arenas, panel objects, visual assets, and support mechanics |
| `vibration/` | analytic spectral synthesis, Γ calibration, displacement gates, IMU |
| `sensors/` | physical wrist-camera model |
| `init_files/` | 50 phase-complete initial states per suite task |
| `config.py` | validated benchmark, asset, panel, and vibration dataclasses |
| package root | only public exports, module entry point, CLI, and shared validated configuration |

The physical support pose has one source of truth in `models/supports/base.py`: every hard-mounted member uses `p=q+c+R(l-c)`. Scripted policies live in `policies/`, render and diagnostic helpers in `utils/`, and no implementation-only compatibility modules remain at package root.
