"""Wait for a successful StarVLA run, then record fresh gamma-zero checkpoint evaluations.

Run with the ShakeBench Python environment; --prepare-only validates assets without a GPU.
The generated state artifact is private held-out evidence, not an official benchmark split.
"""

from __future__ import annotations

import argparse
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

from robosuite.scripts.shakebench_evaluate import observation_config_from_collection
from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.utils.shakebench_train_states import assert_split_disjoint, build_train_state_artifact


def asset_signature(context):
    """Compare actual training geometry, object, visual scene and physics, not state IDs."""
    return {key: context[key] for key in (
        "geometry_profile", "scene_visual", "object", "physics_profile", "friction", "target_container", "worktable"
    )}


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
    contract = {"gamma": 0, "episodes_per_checkpoint": 20, "seed": args.seed,
                "minimum_step_inclusive": args.min_step, "assets": expected,
                "dataset_manifest_sha256": camera["manifest_sha256"], "states": str(state_path)}
    contract_path = args.output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("evaluation contract changed since preparation")
    write_json_atomic(contract_path, contract)
    return state_path, expected


def evaluate(args, checkpoint, destination, states):
    destination.mkdir()
    # One owned server per checkpoint; never attach to or kill an unrelated service.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1",
                   "PYTHONPATH": f"{args.starvla_root}:{Path.cwd()}"}
    with (destination / "server.log").open("w") as log:
        server = subprocess.Popen(
            [str(args.starvla_python), "deployment/model_server/server_policy.py", "--ckpt_path", str(checkpoint),
             "--port", str(port), "--use_bf16", "--seed", "42", "--idle_timeout", "-1"],
            cwd=args.starvla_root, env={**environment, "CUDA_VISIBLE_DEVICES": args.model_gpu},
            stdout=log, stderr=subprocess.STDOUT,
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
                    [sys.executable, "-m", "robosuite.scripts.shakebench_evaluate_gpu",
                     "--policy", "robosuite.utils.shakebench_starvla:make_policy",
                     "--policy-arg", "host=127.0.0.1", "--policy-arg", f"port={port}",
                     "--inference-timeout-s", "60",
                     "--policy-id", checkpoint.name, "--states", str(states), "--dataset", str(args.dataset),
                     "--gamma", "0", "--num-worlds", "1", "--action-horizon", "8", "--horizon-steps", "600",
                     "--physics-profile", "official", "--device", "cuda:0",
                     "--video-dir", str(destination / "videos"), "--output", str(destination / "result.json")],
                    env={**environment, "CUDA_VISIBLE_DEVICES": args.eval_gpu}, check=True,
                    stdout=eval_log, stderr=subprocess.STDOUT,
                )
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def summarize_checkpoint(step, destination, states, assets):
    result = json.loads((destination / "result.json").read_text())
    episodes = result["episodes"]
    expected_ids = [state["state_id"] for state in json.loads(states.read_text())["states"]]
    if [episode["state_id"] for episode in episodes] != expected_ids:
        raise ValueError("checkpoint must contain all 20 fresh rollouts exactly once")
    for episode in episodes:
        if asset_signature(episode["task_context"]) != assets:
            raise ValueError("rollout assets differ from training")
        if not Path(episode["video"]).is_file() or episode["video_frames"] < 1:
            raise ValueError("rollout video missing")
    errors = sum(episode["termination_cause"] in {"policy_error", "invalid_execution"} for episode in episodes)
    successes = sum(bool(episode["success"]) for episode in episodes)
    return {
        "step": step,
        "rollouts": 20,
        "successes": successes,
        "mean_success_rate": successes / 20,
        "execution_errors": errors,
        "result": str(destination / "result.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True, help="Directory containing train.yaml and exit_status")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--starvla-root", type=Path, required=True)
    parser.add_argument("--starvla-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2026091701)
    parser.add_argument("--model-gpu", default="0")
    parser.add_argument("--eval-gpu", default="1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for key in ("training_dir", "dataset", "starvla_root", "starvla_python", "output"):
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
        rows = []
        for step, checkpoint in selected:
            destination = args.output / f"step_{step}"
            print(f"Evaluating {checkpoint}", flush=True)
            evaluate(args, checkpoint, destination, states)
            rows.append(summarize_checkpoint(step, destination, states, assets))
            write_json_atomic(args.output / "summary.json", {"checkpoints": rows})
            with (args.output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(rows[-1], flush=True)
        write_json_atomic(args.output / "completed.json", {"checkpoints": len(rows), "rollouts": len(rows) * 20})


if __name__ == "__main__":
    main()
