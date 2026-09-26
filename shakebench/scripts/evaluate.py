"""Evaluate any policy factory on ShakeBench states with one frozen result contract.

The runner is model-agnostic: a policy is any importable factory returning an
object with "chunk_size" and "predict(observation)".  Results record the state,
Gamma, horizon, observation contract, policy identity, executed-action evidence,
termination cause, and typed policy errors, so failures keep their denominators.

Official tier scorecards still come from shakebench.scripts.run_oracle;
these records are evaluation evidence, not certified scorecard inputs.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

from shakebench import models
from shakebench.scripts.export_sft_subset import SUBSET_PROVENANCE_FILENAME
from shakebench.scripts.run_oracle import load_state_asset
from shakebench.utils.artifacts import write_json
from shakebench.utils.excitation import SWAY_V1, VIBRATION_MODES
from shakebench.utils.rollout import OBSERVATION_SOURCES, ShakeBenchTaskEnv, invalid_episode_result, rollout_policy
from shakebench.utils.task_registry import assert_split_disjoint, split_overlap

EVALUATION_SCHEMA_ID = "shakebench.policy_evaluation"
EVALUATION_SCHEMA_VERSION = 1
COLLECTION_MANIFEST_NAME = "shakebench_collection.json"


def load_policy_factory(spec):
    """Import "module:attribute" and return the policy factory callable."""
    module_name, separator, attribute = str(spec).partition(":")
    if not module_name or not separator or not attribute:
        raise ValueError("--policy must be given as module:factory")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError(f"policy factory {spec!r} is not callable")
    return factory


def parse_policy_args(items):
    """Parse repeated NAME=VALUE policy arguments without a shell-specific format."""
    arguments = {}
    for item in items:
        name, separator, value = str(item).partition("=")
        if not name or not separator:
            raise ValueError(f"--policy-arg must be NAME=VALUE, got {item!r}")
        try:
            arguments[name] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            arguments[name] = value
    return arguments


def build_policy(factory, *, arguments, inference_timeout_s):
    """Call the factory with the arguments it declares, nothing else."""
    accepted = set(inspect.signature(factory).parameters)
    kwargs = {name: value for name, value in arguments.items() if name in accepted}
    if "inference_timeout_s" in accepted:
        kwargs["inference_timeout_s"] = inference_timeout_s
    unknown = sorted(set(arguments) - accepted)
    if unknown:
        raise ValueError(f"policy factory does not accept: {unknown}")
    policy = factory(**kwargs)
    for attribute in ("chunk_size", "predict"):
        if not hasattr(policy, attribute):
            raise ValueError(f"policy object is missing {attribute!r}")
    if int(policy.chunk_size) < 1:
        raise ValueError("policy chunk_size must be positive")
    return policy


def observation_config_from_collection(path):
    """Read the observation contract of the dataset a policy was trained on.

    Training states come from the manifest of the data actually present, so an
    exported SFT subset reports only the states it kept; the subset provenance,
    when present, links that selection back to its source manifest.
    """

    source = Path(path)
    manifest_path = source / "meta" / COLLECTION_MANIFEST_NAME if source.is_dir() else source
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("training dataset manifest must be complete before evaluation")
    info = json.loads((manifest_path.parent / "info.json").read_text(encoding="utf-8"))
    shape = info["features"]["observation.images.main"]["shape"]
    provenance_path = manifest_path.parent / SUBSET_PROVENANCE_FILENAME
    return {
        "main_camera": manifest.get("cameras", {}).get("observation.images.main", "task_close"),
        "height": int(shape[0]),
        "width": int(shape[1]),
        "train_states": [episode["state"] for episode in manifest["episodes"]],
        "dataset": manifest_path.parent.parent.name,
        "selection_rule": manifest.get("sft_subset", {}).get("selection_rule"),
        "subset_provenance": (
            json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else None
        ),
    }


def requested_state_ids(states, selectors):
    """Return the ordered, unique state selection; unknown IDs fail closed."""
    if not selectors:
        return list(states)
    requested = []
    for selector in selectors:
        requested.extend(part.strip() for part in str(selector).split(",") if part.strip())
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate --state-ids entries")
    by_id = {state["state_id"]: state for state in states}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown state IDs: {unknown}")
    return [by_id[state_id] for state_id in requested]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-module", action="append", default=[], help="Import a trusted task registration module (repeatable)"
    )
    parser.add_argument("--policy", required=True, help="module:factory returning a predict()-capable policy")
    parser.add_argument("--policy-arg", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--policy-id", default=None, help="Label recorded with the results; defaults to --policy")
    parser.add_argument("--states", type=Path, default=Path(models.assets_root, "shakebench_states_dev.json"))
    parser.add_argument("--state-ids", action="append", help="Comma-separated or repeated exact state IDs")
    parser.add_argument("--dataset", type=Path, help="Collected dataset or manifest used for training")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--mode", choices=VIBRATION_MODES, default="multisine_v1")
    parser.add_argument("--mode-params", type=json.loads, default={}, help="JSON excitation parameters")
    parser.add_argument("--sway-v1", action="store_true", help="Use the low-frequency SWAY_V1 preset")
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--inference-timeout-s", type=float, default=None)
    parser.add_argument("--observation-source", choices=OBSERVATION_SOURCES, default="cameras")
    parser.add_argument("--height", type=int, default=None, help="Defaults to the training dataset, else 256")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--main-camera", default=None, help="Defaults to the training dataset, else task_close")
    parser.add_argument(
        "--allow-train-states", action="store_true", help="Permit evaluating states that trained the policy"
    )
    parser.add_argument("--output", type=Path, help="New JSON result path; omit to print the record")
    return parser


def summarize(episodes):
    """Report success with the failure denominator kept explicit."""
    causes = Counter(episode["termination_cause"] for episode in episodes)
    error_types = Counter(error["error_type"] for episode in episodes for error in episode["policy_errors"])
    valid = sum(episode["episode_validity"] == "valid" for episode in episodes)
    successes = sum(episode["score_outcome"] == "success" for episode in episodes)
    return {
        "attempted_episodes": len(episodes),
        "valid_episodes": valid,
        "invalid_execution_episodes": len(episodes) - valid,
        "success": successes,
        "unsuccessful": valid - successes,
        "success_rate_over_attempts": (successes / len(episodes)) if episodes else None,
        "success_rate_over_valid": (successes / valid) if valid else None,
        "termination_causes": dict(sorted(causes.items())),
        "policy_error_types": dict(sorted(error_types.items())),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.sway_v1:
        if args.mode_params or args.mode != "multisine_v1":
            raise ValueError("--sway-v1 cannot be combined with --mode or --mode-params")
        args.mode, args.mode_params = SWAY_V1["mode"], SWAY_V1["mode_params"]
    for module_name in args.task_module:
        importlib.import_module(module_name)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if min(args.horizon_steps, args.action_horizon) < 1:
        raise ValueError("horizon-steps and action-horizon must be positive")

    states = requested_state_ids(load_state_asset(args.states)["states"], args.state_ids)
    if not states:
        raise ValueError("no states selected")
    camera_config = {
        "main_camera": args.main_camera or "task_close",
        "height": args.height or 256,
        "width": args.width or 256,
    }
    train_split = {"status": "unavailable", "reason": "no --dataset was given"}
    train_states = []
    if args.dataset is not None:
        trained_on = observation_config_from_collection(args.dataset)
        train_states = trained_on.pop("train_states")
        camera_config = {
            "main_camera": args.main_camera or trained_on["main_camera"],
            "height": args.height or trained_on["height"],
            "width": args.width or trained_on["width"],
        }
        train_split = {"status": "checked", **trained_on, "train_state_count": len(train_states)}
    if train_states and not args.allow_train_states:
        assert_split_disjoint(train_states, states)
        train_split["overlap"] = 0
    elif train_states:
        train_split["overlap"] = len(split_overlap(train_states, states))
        train_split["status"] = "train_states_allowed"

    policy = build_policy(
        load_policy_factory(args.policy),
        arguments=parse_policy_args(args.policy_arg),
        inference_timeout_s=args.inference_timeout_s,
    )
    episodes = []

    def result_payload():
        return {
            "schema_id": EVALUATION_SCHEMA_ID,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "scoreable": False,
            "scoreable_reason": "policy evaluation evidence; official scorecards come from shakebench.scripts.run_oracle",
            "policy": {
                "spec": args.policy,
                "policy_id": args.policy_id or args.policy,
                "identity": policy.identity if isinstance(getattr(policy, "identity", None), dict) else None,
                "chunk_size": int(policy.chunk_size),
            },
            "run_contract": {
                "gamma": args.gamma,
                "mode": args.mode,
                "mode_params": args.mode_params,
                "horizon_steps": args.horizon_steps,
                "action_horizon": args.action_horizon,
                "inference_timeout_s": args.inference_timeout_s,
                "observation_source": args.observation_source,
                "observation": {
                    "source": args.observation_source,
                    **({} if args.observation_source == "contract" else camera_config),
                    "identity": episodes[0].get("observation_identity") if episodes else None,
                },
                "action_space": (episodes[0].get("task_context") or {}).get("action_space") if episodes else None,
                "timing": episodes[0].get("timing") if episodes else None,
                "state_asset": str(args.states),
            },
            "train_split_check": train_split,
            "episodes": episodes,
            "summary": summarize(episodes),
        }

    def save_results():
        """Persist what finished; a mid-batch failure must not lose earlier episodes."""

        payload = result_payload()
        if args.output is not None:
            write_json(args.output, payload)
        return payload

    try:
        for state in states:
            task = None
            try:
                task = ShakeBenchTaskEnv(
                    state,
                    gamma=args.gamma,
                    mode=args.mode,
                    mode_params=args.mode_params,
                    horizon=args.horizon_steps,
                    observation_source=args.observation_source,
                    **camera_config,
                )
                episodes.append(
                    rollout_policy(
                        task,
                        policy,
                        action_horizon=args.action_horizon,
                        inference_timeout_s=args.inference_timeout_s,
                        policy_id=args.policy_id or args.policy,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one unusable state must not end the batch
                episodes.append(
                    invalid_episode_result(
                        state_id=state.get("state_id"),
                        reason=f"{type(exc).__name__}: {exc}",
                        task=task,
                        policy_id=args.policy_id or args.policy,
                        gamma=args.gamma,
                        horizon_steps=args.horizon_steps,
                        action_horizon=args.action_horizon,
                        inference_timeout_s=args.inference_timeout_s,
                    )
                )
                print(f"{state.get('state_id')}: invalid_execution: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                if task is not None:
                    task.close()
            save_results()
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    payload = save_results()
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
