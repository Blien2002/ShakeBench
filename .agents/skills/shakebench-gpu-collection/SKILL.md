---
name: shakebench-gpu-collection
description: Create ShakeBench GPU Oracle videos or pick-place, ring, and Push-T datasets with MJWarp CUDA physics and MuJoCo EGL rendering. Use for collection, resume, success-only LeRobot export, and train/evaluation split checks.
---

# ShakeBench GPU collection

Use this skill from the ShakeBench repository root. Read the current `AGENTS.md` and the entry points named below before running a collection; script arguments and contracts take precedence over these examples.

## Contract

- `shakebench.scripts.collect_lerobot_gpu` supports pick-place, ring-on-peg, and Push-T states. MJWarp runs Oracle physics at `gamma=0` on `cuda:N`; host MuJoCo renders synchronized states through EGL.
- Load ring states with `--task-module shakebench.environments.ring_on_peg`. Load Push-T with `--task-module shakebench.environments.push_t --states shakebench/models/assets/shakebench_push_t_states.json --horizon-steps 1500`. Push-T states with different `goal_id` values form separate batches.
- `--video-only` writes one MP4 per selected state without LeRobot data or a collection manifest. Dataset mode can also take `--video-output` to save synchronized rollout MP4s beside LeRobot data. Use a new output directory for each run, and verify frame counts with `ffprobe`. Ring and Push-T success use each task's outcome rule on synchronized device state; Push-T also requires the Oracle's post-retreat verification.
- Without `--video-only`, output is a local, non-scoreable LeRobot **v2.1** dataset: 20 Hz, main and wrist RGB images (default 256×256), 8D proprioception, normalized 7D `OSC_POSE` actions, task text, and next-step outcome fields. IL data contains no IMU. `--repo-id` is local metadata; the collector does not upload.
- Generate a separate train state pool. Preserve official and knee state assets for held-out evaluation. Choose a new output path for every independent run; never delete or overwrite a dataset to make a retry work.

## Run

1. **Prepare the environment.** Use a Python 3.10+ environment with `requirements.txt`, `requirements-collection.txt` (`lerobot==0.3.3`), and `requirements-gpu.txt` (pinned MuJoCo/MJWarp/Warp). Set `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` **before Python starts**. Check the real CUDA device and EGL availability. GPU/driver commands in this repository require the sandbox escalation described in `AGENTS.md`; do not infer missing hardware from a sandbox failure or switch to CPU silently.
2. **Choose the state pool and budget.** Use `generate_train_states` for pick-place; `--objects` accepts registry IDs such as `can`, `mug`, and `apple`. The `can` pool includes standing and side-lying poses. Keep the generated JSON because the verifier matches the collection to its source asset. The GPU collector's default `--states` points to dev states, so always pass the train asset explicitly for training data.
3. **Collect.** Start with a small, separate output if runtime or scene health is unknown. Run the intended full collection with an unused output directory. The collector writes one episode per selected state; `--limit` is the per-shard episode budget, and `--num-worlds` is the batch size (default 4). Reduce `--num-worlds` if device or host memory is insufficient. Keep `--physics-profile official` for the calibrated device path; `probe` is exploratory.
4. **Resume only an incomplete run.** Reuse the original `--output`, `--states`, state selection, shard settings, `--repo-id`, image dimensions, camera, horizon, device, and physics profile, adding `--resume`. The script checks some but not all of these parameters. It refuses a completed dataset and prevents duplicate state IDs in the manifest. Inspect `meta/shakebench_collection.json` and `meta/info.json` if a save was interrupted; stop if their episode counts disagree.
5. **Export and verify.** Wait for `"complete": true` in the source manifest. Inspect `sft_subset.success_count`; if zero, investigate collection failures before exporting. `export_sft_subset` keeps only `success_latched` episodes, renumbers them, and records source indices. Run `verify_collection` on this exported dataset, not the raw attempt set: its outcome check requires every episode to succeed. Pass the train source asset **and** every held-out evaluation asset to `--eval-assets`, because that option replaces the verifier's default list.

Example for one unsharded `can` collection; choose count, seed, output names, and batch size for the actual request:

```bash
python -m pip install -r requirements.txt -r requirements-collection.txt -r requirements-gpu.txt

python -m shakebench.scripts.generate_train_states \
  --output out/can_train_states.json --count 100 --seed 42 --objects can

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m shakebench.scripts.collect_lerobot_gpu \
  --states out/can_train_states.json --output out/can_gpu_raw \
  --device cuda:0 --physics-profile official --num-worlds 4

python -m shakebench.scripts.export_sft_subset \
  --dataset out/can_gpu_raw --output out/can_gpu_sft

python -m shakebench.scripts.verify_collection \
  --dataset out/can_gpu_sft \
  --eval-assets out/can_train_states.json \
    shakebench/models/assets/shakebench_states_dev.json \
    shakebench/models/assets/shakebench_states_official.json \
    shakebench/models/assets/shakebench_states_knee.json
```

For a resume, repeat the collection command exactly with `--resume`. For multiple shards, use a distinct output directory per shard; the repository has no automatic shard merge in this workflow. Do not claim that separate shard outputs form one LeRobot dataset.

Report the source and exported paths, requested/attempted/successful episode counts from the manifest, the verifier's `passed` result and failed checks, and the actual MJWarp device/EGL status. The exported dataset is the training input; `shakebench.scripts.train_policy` additionally needs an external policy class and checkpoint.

For `--video-only`, report the MP4 directory, state IDs, frame counts, episode outcomes printed by the collector, and actual MJWarp device/EGL status. Do not claim a LeRobot dataset or a success-only export was made.
