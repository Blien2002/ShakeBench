"""Fine-tune a local LeRobot-style checkpoint on native ShakeBench demonstrations.

Run in the external model's Python environment with its source on PYTHONPATH.
Images, language, 8D state and normalized 7D OSC actions retain their original meaning.
"""

import argparse
import json
from pathlib import Path

from shakebench.utils.pretrained_policy import (
    CONTRACT_FILE,
    FEATURE_KEYS,
    load_config,
    policy_class,
    validate_offsets,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-class", required=True, help="External module:class")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Success-only LeRobot v2.1 export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-offsets", type=int, nargs="+", default=[-2, 0])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    offsets = validate_offsets(args.observation_offsets)
    if min(args.steps, args.batch_size) < 1 or not 0 < args.lr < float("inf"):
        raise ValueError("steps, batch size and learning rate must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from safetensors.torch import load_file

    from shakebench.scripts.export_sft_subset import select_successful_episodes

    manifest = json.loads((args.dataset / "meta/shakebench_collection.json").read_text())
    selected = select_successful_episodes(manifest)
    if len(selected) != len(manifest["episodes"]) or not selected:
        raise ValueError("export a nonempty success-only dataset with shakebench_export_sft_subset first")
    if manifest["action_space"]["controller"] != "OSC_POSE":
        raise ValueError("expected ShakeBench OSC_POSE demonstrations")
    torch.manual_seed(args.seed)
    cls = policy_class(args.policy_class)
    config = load_config(cls, args.checkpoint, args.device)
    meta = LeRobotDatasetMetadata("shakebench/local", root=args.dataset)
    if meta.info["codebase_version"] != "v2.1":
        raise ValueError("LeRobot v2.1 required")
    if meta.fps != 20:
        raise ValueError("dataset must match the 20 Hz ShakeBench control rate")
    features = dataset_to_policy_features(meta.features)
    config.input_features = {key: features[key] for key in FEATURE_KEYS[:-1]}
    config.output_features = {"action": features["action"]}
    if tuple(features["observation.state"].shape) != (8,) or tuple(features["action"].shape) != (7,):
        raise ValueError("expected 8D state and 7D actions")
    if config.n_obs_steps != len(offsets):
        raise ValueError("history length must match pretrained temporal architecture")
    for key in ("use_delta_action", "enable_streaming", "adapt_to_pi_aloha"):
        if hasattr(config, key):
            setattr(config, key, False)
    timestamps = {key: [offset / meta.fps for offset in offsets] for key in config.input_features}
    timestamps["action"] = [i / meta.fps for i in range(config.chunk_size)]
    dataset = LeRobotDataset("shakebench/local", root=args.dataset, delta_timestamps=timestamps)
    stats = {
        key: {name: torch.as_tensor(value).clone() for name, value in values.items()}
        for key, values in meta.stats.items()
        if key in FEATURE_KEYS
    }
    for values in stats.values():
        values["std"] = values["std"].clamp_min(1e-6)
    model = cls(config, dataset_stats=stats)
    # Transfer learned weights; dataset normalizers have different names and state dimensions.
    normalizers = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")
    weights = load_file(str(args.checkpoint / "model.safetensors"))
    missing, unexpected = model.load_state_dict(
        {key: value for key, value in weights.items() if not key.startswith(normalizers)}, strict=False
    )
    if unexpected or any(not key.startswith(normalizers) for key in missing):
        raise ValueError(f"incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    model.to(args.device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    iterator = iter(loader)
    # ponytail: single-device fine-tuning; use a distributed trainer when one GPU is insufficient.
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
        }
        # Some external policies consume this spelling for the standard LeRobot padding mask.
        batch["actions_id_pad"] = batch["action_is_pad"]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}", flush=True)
    model.save_pretrained(args.output)
    contract = {
        "policy_class": args.policy_class,
        "action_contract": "shakebench.normalized_osc.v1",
        "observation_offsets": offsets,
        "dataset": str(args.dataset.resolve()),
        "fps": meta.fps,
        "steps": args.steps,
    }
    (args.output / CONTRACT_FILE).write_text(json.dumps(contract, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
