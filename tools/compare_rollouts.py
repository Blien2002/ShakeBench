#!/usr/bin/env python3
"""Record deterministic task rollouts and compare two recordings.

A refactor of ShakeBench must not change physics, sensing, task logic or
rendering.  ``record`` drives every registered task through the shared runtime
with vibration enabled and a fixed pseudo-random action sequence, hashing the
full MuJoCo state, every observation (including the table IMU) and the reward
after each control step.  ``--render`` additionally hashes the main and wrist
camera images of two tasks.  Run ``record`` once per checkout, with that
checkout first on ``PYTHONPATH``, then ``diff`` the two files; any divergence
reports the first differing control step.

    PYTHONPATH=/path/to/old python tools/compare_rollouts.py record --out old.json
    PYTHONPATH=/path/to/new python tools/compare_rollouts.py record --out new.json
    python tools/compare_rollouts.py diff old.json new.json

Bitwise equality is expected on one machine and one MuJoCo build; recordings
from different platforms are not comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TASK_MODULES = {
    "pick_place": "shakebench.environments.vibration_pick_place",
    "panel_operation": "shakebench.environments.panel_operation",
    "push_t": "shakebench.environments.push_t",
    "ring_on_peg": "shakebench.environments.ring_on_peg",
    "ring_on_rail": "shakebench.environments.ring_on_rail",
    "stack_blocks": "shakebench.environments.stack_blocks",
    "upright": "shakebench.environments.upright",
    "wine_rack": "shakebench.environments.wine_rack",
}
# Tasks whose default_state() lives outside the environment module.
DEFAULT_STATE_MODULES = {"panel_operation": "shakebench.environments.panel_state"}
RENDER_TASKS = ("pick_place", "push_t")


def _runtime():
    """Return (make_environment, load_state_asset, ShakeBenchTaskEnv) for this checkout."""
    from shakebench.rollout.task_env import ShakeBenchTaskEnv
    from shakebench.tasks.runtime import make_environment
    from shakebench.tasks.states.assets import load_state_asset

    return make_environment, load_state_asset, ShakeBenchTaskEnv


def _initial_state(task, load_state_asset):
    importlib.import_module(TASK_MODULES[task])
    if task == "pick_place":
        from shakebench import models

        return load_state_asset(Path(models.assets_root) / "shakebench_states_dev.json")["states"][0]
    module = importlib.import_module(DEFAULT_STATE_MODULES.get(task, TASK_MODULES[task]))
    return module.default_state()


def _update(digest, value):
    array = np.asarray(value)
    if array.dtype.kind in "biufc":
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        digest.update(repr(value).encode())


def _digest_observation(digest, observation):
    for key in sorted(observation):
        digest.update(key.encode())
        _update(digest, observation[key])


def _actions(dimension, steps, seed=1234):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-0.5, 0.5, dimension) for _ in range(steps)]


def record_task(task, *, gamma, steps):
    make_environment, load_state_asset, _ = _runtime()
    state = _initial_state(task, load_state_asset)
    started = time.perf_counter()
    env, _ = make_environment(state, gamma=gamma, horizon=steps + 10)
    try:
        digest = hashlib.sha256()
        trace = []
        data = env.sim.data
        for action in _actions(env.action_dim, steps):
            observation, reward, _, _ = env.step(action)
            for array in (data.time, data.qpos, data.qvel, data.act, data.ctrl, data.sensordata):
                _update(digest, array)
            _digest_observation(digest, observation)
            _update(digest, reward)
            trace.append(digest.hexdigest()[:16])
        metrics = json.dumps(env.get_metrics(), sort_keys=True, default=repr)
        site_rgba = np.ascontiguousarray(env.sim.model.site_rgba)
        return {
            "trace": trace,
            "metrics_sha256": hashlib.sha256(metrics.encode()).hexdigest(),
            "model_timestep": float(env.sim.model.opt.timestep),
            # Site colors decide what cameras see; hidden sites must use alpha 0, never a negative alpha.
            "site_rgba_sha256": hashlib.sha256(site_rgba.tobytes()).hexdigest(),
            "negative_site_alpha": int((site_rgba[:, 3] < 0).sum()),
            "seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        env.close()


def record_render(task, *, gamma, steps):
    _, load_state_asset, ShakeBenchTaskEnv = _runtime()
    state = _initial_state(task, load_state_asset)
    env = ShakeBenchTaskEnv(state, gamma=gamma, observation_source="cameras")
    try:
        observation, _ = env.reset()
        digest = hashlib.sha256()
        _digest_observation(digest, observation)
        trace = [digest.hexdigest()[:16]]
        for action in _actions(7, steps, seed=99):
            observation, *_ = env.step(action)
            _digest_observation(digest, observation)
            trace.append(digest.hexdigest()[:16])
        return {"trace": trace, "keys": sorted(observation)}
    finally:
        env.close()


def record(args):
    result = {"python": sys.version.split()[0], "gamma": args.gamma, "steps": args.steps, "tasks": {}, "render": {}}
    tasks = args.tasks or list(TASK_MODULES)
    for task in tasks:
        try:
            result["tasks"][task] = record_task(task, gamma=args.gamma, steps=args.steps)
        except Exception as exc:  # noqa: BLE001 - recorded so the diff names the failing task
            result["tasks"][task] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        print(task, result["tasks"][task].get("error") or result["tasks"][task]["seconds"], flush=True)
    if args.render:
        for task in RENDER_TASKS:
            try:
                result["render"][task] = record_render(task, gamma=args.gamma, steps=5)
            except Exception as exc:  # noqa: BLE001
                result["render"][task] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            print("render", task, result["render"][task].get("error", "ok"), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


def diff(args):
    first, second = (json.loads(Path(path).read_text()) for path in (args.first, args.second))
    failures = 0
    for section in ("tasks", "render"):
        for task in sorted(set(first[section]) | set(second[section])):
            a, b = first[section].get(task), second[section].get(task)
            if a is None or b is None:
                print(f"[{section}] {task}: missing in one recording")
                failures += 1
            elif "error" in a or "error" in b:
                print(f"[{section}] {task}: error {a.get('error')!r} vs {b.get('error')!r}")
                failures += 1
            else:
                mismatch = next((i for i, (x, y) in enumerate(zip(a["trace"], b["trace"])) if x != y), None)
                extra = {k for k in a if k not in ("trace", "seconds") and a.get(k) != b.get(k)}
                if mismatch is not None or len(a["trace"]) != len(b["trace"]) or extra:
                    print(f"[{section}] {task}: DIFFERS first_step={mismatch} fields={sorted(extra)}")
                    failures += 1
                else:
                    print(f"[{section}] {task}: identical over {len(a['trace'])} steps")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--out", required=True)
    rec.add_argument("--gamma", type=float, default=0.6)
    rec.add_argument("--steps", type=int, default=25)
    rec.add_argument("--render", action="store_true", help="also hash camera images (needs EGL or OSMesa)")
    rec.add_argument("--tasks", nargs="*", choices=list(TASK_MODULES))
    cmp = sub.add_parser("diff")
    cmp.add_argument("first")
    cmp.add_argument("second")
    args = parser.parse_args(argv)
    return record(args) or 0 if args.command == "record" else diff(args)


if __name__ == "__main__":
    raise SystemExit(main())
