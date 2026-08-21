# Code structure

ShakeBench uses a single `src/` package layout.

| Path | Responsibility |
|---|---|
| `benchmark/` | LIBERO-style task descriptors, suite registry, HDF5 loading, bootstrap scorecards |
| `envs/` | policy-rate scheduling, Gymnasium API, manipulation tasks, wrappers |
| `controllers/` | joint-position, OSC pose, and variable-impedance action transforms |
| `policies/` | policy protocol, three Oracle tiers, FxLMS/classical and random baselines, adapters |
| `models/` | scene construction, arenas, panel objects, visual assets, and support mechanics |
| `vibration/` | analytic spectral synthesis, Γ calibration, displacement gates, IMU |
| `sensors/` | physical wrist-camera model |
| `init_files/` | 50 phase-complete initial states per suite task |
| `config.py` | validated benchmark, asset, panel, and vibration dataclasses |
| package root | only public exports, module entry point, CLI, and shared validated configuration |

The physical support pose has one source of truth in `models/supports/base.py`: every hard-mounted member uses `p=q+c+R(l-c)`. Scripted policies live in `policies/`, render and diagnostic helpers in `utils/`, and no implementation-only compatibility modules remain at package root.

Paper-facing entry points live in `scripts/`: `run_scorecard.py`,
`aggregate_scorecard.py`, `collect_demonstrations.py`, and
`get_dataset_info.py`. The scorecard and collection CLIs distinguish the
non-scoreable state-contract backend from physical Isaac/Newton evidence.
