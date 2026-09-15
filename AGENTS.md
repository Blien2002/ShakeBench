# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `robosuite/`. ShakeBench environments are under `robosuite/environments/manipulation/`, reusable simulation logic under `robosuite/utils/`, command-line entry points under `robosuite/scripts/`, and MuJoCo XML, JSON, and textures under `robosuite/models/assets/`. Tests mirror these areas in `tests/`, including focused `tests/test_shakebench_*.py` modules. Documentation lives in `docs/`; optional StarVLA integration code lives in `integrations/starvla/`. Treat `out/`, `build/`, caches, videos, and generated datasets as local artifacts unless a change explicitly requires a fixture. The `docs/` and `tests/` trees are local working copies: the published repository serves the upstream robosuite versions of those files, so keep local edits there uncommitted.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt` installs the package in editable mode and core dependencies.
- `python -m pip install -r requirements-gpu.txt` adds pinned MJWarp dependencies for GPU collection.
- `python -m pytest` runs the full suite; use `python -m pytest tests/test_shakebench_scene.py` for a focused check.
- `python -m pytest -m "not renderer"` skips tests requiring EGL, Mesa, or an on-screen renderer.
- `pre-commit run --all-files` applies Black and isort checks.
- `python -m robosuite.utils.shakebench_runtime_verifier` validates committed physics and scene assets. Use `--write` only when intentionally updating that contract.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax, four-space indentation, and a 120-character line limit. Black formats code; isort uses Black-compatible import ordering. Follow Google-style docstrings. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Add a module docstring to scripts and tests when their purpose or invocation is not obvious.

## Testing Guidelines

Use pytest. Name files `test_*.py` and tests `test_<behavior>`. Add the smallest regression test beside the affected subsystem. Mark graphics-dependent tests with `@pytest.mark.renderer`. No fixed coverage threshold is configured; changed behavior must have direct assertions and pass relevant focused tests before the full suite.

## Commit & Pull Request Guidelines

Recent commits use short, sentence-case imperative subjects such as `Add mjwarp functionality and tests`. Keep each commit scoped to one change. Pull requests should include an itemized summary, test commands and results, linked issues when applicable, and example scripts or screenshots for new APIs, environments, or visual scene changes. Do not mix generated outputs, `docs/` or `tests/` edits, or unrelated local files into the patch.

## Local-Only `docs/` and `tests/`

After code edits, commit and push only changes outside `docs/` and `tests/`. Leave local edits in those folders unstaged, even when the task touches them, because the remote keeps the upstream robosuite content for both folders. `git status` will keep reporting those local changes; that is expected. To confirm the published trees still match upstream, run `git diff a85139df HEAD -- docs tests`, which should print nothing.
