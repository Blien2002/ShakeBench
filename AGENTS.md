# Repository Guidelines

ShakeBench evaluates robot manipulation policies on a vibrating worktable. This repository is evaluation-only: task
environments, state assets, the rollout boundary, policy adapters and evaluation entry points. Data collection, scripted
experts (Oracles), teleoperation and training do not belong here.

## Layout

`shakebench/` is the only package.

- `environments/`: one module per task, named after its `task_type`; `base.py` holds `ShakeBenchEnv` (the robosuite
  seam) and `ShakeBenchTask` (the shared worktable scaffold every task builds on).
- `tasks/`: registry, pick-place catalog, success metrics, observation privilege, shared CPU runtime, and state assets
  under `tasks/states/`.
- `physics/`, `sensors/`, `scene/`, `models/`: excitation and deck dynamics, the table IMU, scene configuration and
  cameras, MJCF models and packaged assets.
- `rollout/`, `policies/`, `evaluation/`, `wrappers/`: the policy-facing boundary, adapters, evaluation runner, Gym.
- `scripts/` and `demos/`: command-line entry points only. Nothing imports them; reused code belongs in the library.

The simulation core (`environments`, `tasks`, `physics`, `sensors`, `scene`, `models`, `utils`) never imports `rollout`,
`policies`, `evaluation` or `wrappers`. Local regression checks enforce these rules; public package smoke checks do not replace them.

robosuite is an unmodified, pinned dependency (see `pyproject.toml`). Never vendor or patch it; extend `ShakeBenchEnv`.

## Commands

- `pip install -e ".[dev,gym]"`: editable install with test tooling.
- `python tools/check_package.py cli`: help without simulator or graphics imports.
- `MUJOCO_GL=disable python tools/check_package.py imports`: module and public-export smoke.
- `python tools/check_package.py wheel --wheel PATH`: package/asset hash verification.
- `pre-commit run --all-files`: Black and isort.

## Style

Python 3.10+, four-space indentation, 120-character lines, Black and isort (black profile), Google-style docstrings,
`snake_case` functions and modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.

## Tests and documentation

Full regression tests and detailed validation documents are retained in the local validation tree. The public
repository excludes `tests/`, `docs/`, and generated audit reports. Keep local validation evidence when changing
physics, sensing, task rules, observations or results, and summarize material changes in README.md. Asset licenses,
source notices, runtime JSON configuration and state banks remain in the package.

## Behavior-preserving changes

Refactors must not change rollouts. Run `python tools/compare_rollouts.py record --out before.json` on the old checkout
and `--out after.json` on the new one (each first on `PYTHONPATH`), then `python tools/compare_rollouts.py diff
before.json after.json`; every task must be identical. Moving the robosuite pin requires the same check.

## Commits and pull requests

Short, imperative, sentence-case subjects; one change per commit. Pull requests list what changed, the test commands
and results, and the rollout diff when relevant. Never commit `out/`, videos, datasets or local configuration.
