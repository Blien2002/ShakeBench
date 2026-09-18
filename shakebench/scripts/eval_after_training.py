"""Wait for a successful training run, then record fresh gamma-zero checkpoint evaluations.

The best static checkpoint is then re-evaluated under shaken Gamma levels; both phases
reuse the same 20 held-out states so they are directly comparable.

Run with the ShakeBench Python environment; --prepare-only validates assets without a GPU.
The generated state artifact is private held-out evidence, not an official benchmark split.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from shakebench.scripts.evaluate import observation_config_from_collection
from shakebench.scripts.gpu_batch import make_environment
from shakebench.utils.artifacts import write_json_atomic
from shakebench.utils.train_states import assert_split_disjoint, build_train_state_artifact


def asset_signature(context):
    """Compare actual training geometry, object, visual scene and physics, not state IDs."""
    return {
        key: context[key]
        for key in (
            "geometry_profile",
            "scene_visual",
            "object",
            "physics_profile",
            "friction",
            "target_container",
            "worktable",
        )
    }


def checkpoints(run, minimum, maximum, interval):
    found = {}
    for path in (run / "checkpoints").glob("steps_*_pytorch_model.pt"):
        match = re.fullmatch(r"steps_(\d+)_pytorch_model.pt", path.name)
        if match and minimum <= int(match[1]) <= maximum:
            found[int(match[1])] = path
    expected = {step for step in range(interval, maximum + 1, interval) if step >= minimum}
    if not expected or expected - found.keys():
        raise RuntimeError(f"missing checkpoints: {sorted(expected - found.keys())}")
    if any(path.stat().st_size == 0 for path in found.values()):
        raise RuntimeError("empty checkpoint")
    return sorted(found.items())


def prepare(args):
    camera = observation_config_from_collection(args.dataset)
    manifest = json.loads((args.dataset / "meta/shakebench_collection.json").read_text())
    expected = asset_signature(manifest["episodes"][0]["task_context"])
    if any(asset_signature(episode["task_context"]) != expected for episode in manifest["episodes"]):
        raise ValueError("training dataset contains inconsistent assets")
    payload = build_train_state_artifact(20, seed=args.seed)
    states = payload["states"]
    existing = list(camera["train_states"])
    for path in (Path(__file__).resolve().parents[1] / "models/assets").glob("shakebench*states*.json"):
        existing.extend(json.loads(path.read_text()).get("states", []))
    assert_split_disjoint(existing, states)
    old_xy = {tuple(state["object_xy_m"]) for state in existing if "object_xy_m" in state}
    if any(tuple(state["object_xy_m"]) in old_xy or "task" in state for state in states):
        raise ValueError("new states must have novel poses and use the training legacy can asset")
    env, _ = make_environment(states[0], gamma=0, horizon=600, physics_profile="official")
    try:
        if asset_signature(env.get_policy_task_context()) != expected:
            raise ValueError("current environment assets differ from the training dataset")
    finally:
        env.close()
    state_path = args.output / "fresh_states.json"
    if state_path.exists():
        if json.loads(state_path.read_text()) != payload:
            raise ValueError("existing fresh state artifact differs; use a new output directory")
    else:
        write_json_atomic(state_path, payload)
    contract = {
        "gamma": 0,
        "episodes_per_checkpoint": 20,
        "seed": args.seed,
        "minimum_step_inclusive": args.min_step,
        "assets": expected,
        "dataset_manifest_sha256": camera["manifest_sha256"],
        "states": str(state_path),
        "best_checkpoint_rule": (
            "most successes, then fewest execution " "errors, then highest step over the static sweep"
        ),
        "num_worlds": args.num_worlds,
        "gpu_pairs": [pair.strip() for pair in args.gpu_pairs.split(",") if pair.strip()],
        "shaken": {"gammas": [float(g) for g in args.shaken_gammas], "rollouts_per_gamma": 20},
    }
    contract_path = args.output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("evaluation contract changed since preparation")
    write_json_atomic(contract_path, contract)
    return state_path, expected


def evaluate(args, checkpoint, destination, states, *, gamma, num_worlds, model_gpu, eval_gpu):
    destination.mkdir(parents=True)
    # One owned server per checkpoint; never attach to or kill an unrelated service.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{args.policy_server_root}:{Path.cwd()}"}
    with (destination / "server.log").open("w") as log:
        server = subprocess.Popen(
            [
                str(args.policy_server_python),
                "deployment/model_server/server_policy.py",
                "--ckpt_path",
                str(checkpoint),
                "--port",
                str(port),
                "--use_bf16",
                "--seed",
                "42",
                "--idle_timeout",
                "-1",
            ],
            cwd=args.policy_server_root,
            env={**environment, "CUDA_VISIBLE_DEVICES": model_gpu},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"model server exited; see {destination / 'server.log'}")
                ready = "server running" in (destination / "server.log").read_text()
                if ready:
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("model server did not become ready within 600 seconds")
                time.sleep(2)
            with (destination / "evaluation.log").open("w") as eval_log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "shakebench.scripts.evaluate_gpu",
                        "--policy",
                        "shakebench.utils.websocket_policy:make_policy",
                        "--policy-arg",
                        "host=127.0.0.1",
                        "--policy-arg",
                        f"port={port}",
                        "--inference-timeout-s",
                        "60",
                        "--policy-id",
                        checkpoint.name,
                        "--states",
                        str(states),
                        "--dataset",
                        str(args.dataset),
                        "--gamma",
                        str(gamma),
                        "--num-worlds",
                        str(num_worlds),
                        "--physics-profile",
                        "official",
                        "--device",
                        "cuda:0",
                        "--video-dir",
                        str(destination / "videos"),
                        "--output",
                        str(destination / "result.json"),
                    ],
                    env={**environment, "CUDA_VISIBLE_DEVICES": eval_gpu},
                    check=True,
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                )
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def summarize_rollouts(destination, states, assets, **extra):
    result = json.loads((destination / "result.json").read_text())
    episodes = result["episodes"]
    expected_ids = [state["state_id"] for state in json.loads(states.read_text())["states"]]
    if [episode["state_id"] for episode in episodes] != expected_ids:
        raise ValueError("rollout set must contain all 20 fresh states exactly once")
    for episode in episodes:
        if asset_signature(episode["task_context"]) != assets:
            raise ValueError("rollout assets differ from training")
        if not Path(episode["video"]).is_file() or episode["video_frames"] < 1:
            raise ValueError("rollout video missing")
    errors = sum(episode["termination_cause"] in {"policy_error", "invalid_execution"} for episode in episodes)
    successes = sum(bool(episode["success"]) for episode in episodes)
    return {
        **extra,
        "rollouts": len(episodes),
        "successes": successes,
        "mean_success_rate": successes / len(episodes),
        "execution_errors": errors,
        "result": str(destination / "result.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dir", type=Path, required=True, help="Directory containing train.yaml and exit_status"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--policy-server-root",
        type=Path,
        required=True,
        help="Checkout providing deployment/model_server/server_policy.py",
    )
    parser.add_argument(
        "--policy-server-python", type=Path, required=True, help="Interpreter that can run that policy server"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2026091701)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--shaken-gammas", type=float, nargs="+", default=[0.35, 0.5, 0.75, 0.95])
    parser.add_argument("--num-worlds", type=int, default=8, help="Fresh states simulated together in one MJWarp batch")
    parser.add_argument(
        "--gpu-pairs",
        default="0:1,2:3,4:5,6:7",
        help="Comma-separated model:eval CUDA device pairs, one worker per pair",
    )
    args = parser.parse_args()
    for key in ("training_dir", "dataset", "policy_server_root", "policy_server_python", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states, assets = prepare(args)
        print("Preflight passed: 20 novel states; training assets match; no GPU allocated.", flush=True)
        if args.prepare_only:
            return
        status = args.training_dir / "exit_status"
        print(f"Waiting for successful training completion: {status}", flush=True)
        while not status.exists():
            time.sleep(30)
        if status.read_text().strip() != "0":
            raise RuntimeError(f"training failed ({status.read_text().strip()}); evaluation not started")
        config = yaml.safe_load((args.training_dir / "train.yaml").read_text())
        run = Path(config["run_root_dir"]) / config["run_id"]
        selected = checkpoints(
            run, args.min_step, config["trainer"]["max_train_steps"], config["trainer"]["save_interval"]
        )
        prepare(args)  # Fail closed if assets changed while waiting.
        pairs = [pair.split(":") for pair in args.gpu_pairs.split(",")]
        if not pairs or any(len(pair) != 2 or not all(part.strip() for part in pair) for pair in pairs):
            raise ValueError("--gpu-pairs must be model:eval device pairs, e.g. 0:1,2:3")
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            pending = {}
            for index, (step, checkpoint) in enumerate(selected):
                model_gpu, eval_gpu = (part.strip() for part in pairs[index % len(pairs)])
                destination = args.output / f"step_{step}"
                print(
                    f"Evaluating {checkpoint.name} on GPU {model_gpu}/{eval_gpu} "
                    f"({args.num_worlds} worlds per batch)",
                    flush=True,
                )
                future = pool.submit(
                    evaluate,
                    args,
                    checkpoint,
                    destination,
                    states,
                    gamma=0.0,
                    num_worlds=args.num_worlds,
                    model_gpu=model_gpu,
                    eval_gpu=eval_gpu,
                )
                pending[future] = (step, destination)
            for future in concurrent.futures.as_completed(pending):
                step, destination = pending[future]
                future.result()
                row = summarize_rollouts(destination, states, assets, step=step)
                rows.append(row)
                rows.sort(key=lambda item: item["step"])
                write_json_atomic(args.output / "summary.json", {"checkpoints": rows})
                with (args.output / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(row, flush=True)
        best = max(rows, key=lambda row: (row["successes"], -row["execution_errors"], row["step"]))
        best_checkpoint = dict(selected)[best["step"]]
        print(
            "Best static checkpoint: step {} ({}/{})".format(best["step"], best["successes"], best["rollouts"]),
            flush=True,
        )
        shaken = []
        workers = max(1, min(len(pairs), len(args.shaken_gammas)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            for index, gamma in enumerate(args.shaken_gammas):
                model_gpu, eval_gpu = (part.strip() for part in pairs[index % len(pairs)])
                destination = args.output / "shaken" / f"gamma_{gamma:g}"
                print(
                    f"Evaluating {best_checkpoint.name} under Gamma={gamma:g} on GPU "
                    f"{model_gpu}/{eval_gpu} ({args.num_worlds} worlds per batch)",
                    flush=True,
                )
                future = pool.submit(
                    evaluate,
                    args,
                    best_checkpoint,
                    destination,
                    states,
                    gamma=float(gamma),
                    num_worlds=args.num_worlds,
                    model_gpu=model_gpu,
                    eval_gpu=eval_gpu,
                )
                pending[future] = (float(gamma), destination)
            for future in concurrent.futures.as_completed(pending):
                gamma, destination = pending[future]
                future.result()
                row = summarize_rollouts(destination, states, assets, gamma=gamma)
                shaken.append(row)
                shaken.sort(key=lambda item: item["gamma"])
                write_json_atomic(args.output / "shaken_summary.json", {"best_step": best["step"], "gammas": shaken})
                with (args.output / "shaken_summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(shaken[0]))
                    writer.writeheader()
                    writer.writerows(shaken)
                print(row, flush=True)
        write_json_atomic(
            args.output / "completed.json",
            {
                "checkpoints": len(rows),
                "rollouts": len(rows) * 20,
                "best_step": best["step"],
                "best_successes": best["successes"],
                "shaken_gammas": [float(g) for g in args.shaken_gammas],
                "shaken_rollouts": len(shaken) * 20,
            },
        )


if __name__ == "__main__":
    main()
