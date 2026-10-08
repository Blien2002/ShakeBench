# ShakeBench

**Robot manipulation under persistent worktable vibration.**

ShakeBench tests whether a manipulation policy can complete tasks while its
support surface moves. A Panda arm is fixed to the foundation beside a worktable
in a six-degree-of-freedom driven-platform scene. Reproducible excitation
drives the table; an underside IMU measures its motion. Policies act at 20 Hz
using the same seven-dimensional operational-space action interface across
eight task families.

The benchmark provides task environments, packaged initial states and scene
assets, camera and sensor observations, policy adapters, and synchronous and
asynchronous evaluation runners. Physics uses [MuJoCo](https://mujoco.org/) and
an unmodified, pinned [robosuite](https://github.com/ARISE-Initiative/robosuite).

[Tasks](docs/tasks.md) · [Evaluation guide](docs/evaluation.md) ·
[Architecture](docs/architecture.md) ·
[Training demonstrations](https://huggingface.co/datasets/Blien2002/ShakeBench)

## Tasks and state assets

Task modules live under `shakebench.environments`. Importing a module registers
its task and state loader. `pick_place` is registered by default; pass
`--task-module` for the other tasks.

| Task | Module | Available variations | Packaged states |
| --- | --- | --- | --- |
| Pick and place | `vibration_pick_place` | Object/start-pose variants; task-state assets cover mug, apple, standing can and lying can | `shakebench_states_dev.json`, `shakebench_states_official.json`, `shakebench_states_knee.json`, `shakebench_task_states_official.json`, `shakebench_task_states_knee.json` |
| Control panel | `panel_operation` | Press, toggle up/down, turn on/off; ordered operation chains | `shakebench_panel_operation_states.json`, `shakebench_panel_operation_chain_states.json` |
| Push-T | `push_t` | Two target poses; pushing without lifting or grasping | `shakebench_push_t_states.json` |
| Ring on peg | `ring_on_peg` | Place large and small rings in order | `shakebench_ring_on_peg_states.json` |
| Ring on rail | `ring_on_rail` | Single blue ring placement | `shakebench_ring_on_rail_states.json` |
| Stack blocks | `stack_blocks` | Plain and tapered-tenon joints; blue → green → yellow | `shakebench_stack_blocks_plain_states.json`, `shakebench_stack_blocks_tenon_states.json` |
| Upright | `upright` | Mug, wine bottle, boxed drink and power drill | `shakebench_upright_states.json` |
| Wine rack | `wine_rack` | Left/middle/right slots; wine and water bottles | `shakebench_wine_rack_states.json` |

Paths above are relative to `shakebench/models/assets/`. State counts, split
labels and task versions are encoded in the files. Some bundled states are
development or training states: a filename alone does not establish that a state
is held out. See [task details](docs/tasks.md) before selecting an evaluation set.

## Physics, vibration and success

- **Control:** 20 Hz Panda OSC pose increments; 250 physics substeps per action.
- **Physics:** 0.2 ms timestep, with the packaged `official` physics profile.
- **Assembly:** the runner uses `world_fixed_rigid_table_v1`: a fixed robot
  base and a table rigidly mounted to the driven platen. The package also
  includes a compliant isolation assembly for direct environment construction;
  the runner does not expose a geometry-profile switch.
- **Excitation:** `multisine_v1`, `single_sine_v1`, `custom_multisine_v1`, or the opt-in `scenario_lines_v1` workload.
  `--sway-v1` selects the built-in low-frequency sway preset.
- **Intensity:** nonnegative `--gamma`; `0` removes excitation. For legacy modes, the default
  `normal_peak_v1` calibration scales the authored program so its sampled peak
  workpiece-point acceleration along the support normal, divided by gravity,
  matches the requested gamma. This calibrates the command, not each object's
  measured acceleration.
- **Randomness:** each state fixes excitation/IMU seeds, layout and start time.
  Use the same states and horizon when comparing gamma values.

Each environment owns its success and task-rule checks. Ring placement requires
the specified order and released, stable placement. Wine rack success requires
the correct slot, alignment, support and release. Success is latched once
achieved; intermediate metrics are kept out of policy observations.

The runner reports success over **all attempted episodes**. Policy errors and
timeouts stay in that denominator; invalid execution is reported separately.
The JSON records are experimental evidence, not certified leaderboard scores.


## Optional vibration scenarios

The opt-in `scenario_lines_v1` mode provides S1 rocking, S2 rough-road motion, S3 orbital motion (10 mm primary radius / 20 mm diameter at the deck origin), and S4 alternating press shocks. Here `gamma` is an amplitude multiplier (`scenario_level_v1`), so scenario gamma=1 is not equivalent to the existing peak-calibrated benchmark gamma=1. Defaults, gamma=0, scoring and IMU conventions are preserved.

Run `python -m shakebench.scripts.evaluate_scenarios --policy my_policy:make_policy --state-ids shakebench-dev-v0-000 --output-dir out/scenarios` for the optional four-scenario workload. See [scenario definitions, units, parameters and replay](docs/vibration_scenarios.md).

## Installation

Use Python **3.10 or later** and Git. Linux with an NVIDIA driver/EGL is the
tested headless image-rendering configuration. State-contract checks do not
require a GPU. Other platforms depend on MuJoCo/robosuite support and have not
been validated here.

```bash
git clone https://github.com/Blien2002/ShakeBench.git
cd ShakeBench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The install pins robosuite to upstream commit
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c` and requires
`mujoco>=3.3.0,<3.10`. The PyPI robosuite 1.5.2 release alone is not an equivalent
replacement for that commit. MuJoCo 3.10 changes a mass-matrix API used by the
controllers. Source installation needs access to GitHub and does not install a
vendored or patched robosuite.

```bash
python -m pip install -e ".[gym]"       # Gymnasium and h5py
python -m pip install -e ".[policies]"  # Local checkpoint adapter dependencies
python -m pip install -e ".[dev,gym]"   # Tests and formatting tools
```

Local checkpoint adapters also need the external model implementation that
created the checkpoint. Extras do not install every model framework. The
WebSocket adapter needs a compatible external `deployment` client package.
Video recording additionally requires `ffmpeg` on `PATH`.

## Quick start

Run a short state-contract rollout without rendering:

```bash
MUJOCO_GL=disable python - <<'PY'
import numpy as np
from shakebench.environments.ring_on_peg import default_state
from shakebench.rollout.task_env import ShakeBenchTaskEnv

task = ShakeBenchTaskEnv(
    default_state(), gamma=0.5, horizon=3, observation_source="contract"
)
try:
    observation, info = task.reset()
    for _ in range(3):
        observation, reward, terminated, truncated, info = task.step(np.zeros(7))
        if terminated or truncated:
            break
    print(info)
finally:
    task.close()
PY
```

For images, set `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` **before Python starts**
and use `observation_source="cameras"`. This renders main and wrist RGB views
on the GPU; physics runs on the CPU. The supplied runners do not expose an
MJWarp GPU-physics backend.

Run a one-state, three-step end-to-end evaluation:

```bash
MUJOCO_GL=disable python -m shakebench.scripts.evaluate \
  --task-module shakebench.environments.ring_on_peg \
  --states shakebench/models/assets/shakebench_ring_on_peg_states.json \
  --state-ids ring-stack-s0-0000 \
  --policy shakebench.policies.zero:make_policy \
  --observation-source contract --gamma 0.5 --horizon-steps 3 \
  --output out/smoke.json
```

The zero-action policy checks installation, simulation and result writing. It
is not a manipulation baseline and is expected to time out. Output paths must
be new: the evaluator refuses to overwrite an existing result.

## Policy interface

Pass `--policy module:factory`. The factory returns an object with a positive
integer `chunk_size` and a `predict(observation)` method. A minimal external
policy scaffold is:

```python
import numpy as np

class MyPolicy:
    chunk_size = 4

    def predict(self, observation):
        # Replace with inference and your checkpoint's action decoding.
        return np.zeros((self.chunk_size, 7), dtype=np.float32)

def make_policy():
    return MyPolicy()
```

Return `[7]` or `[N, 7]` finite actions in `[-1, 1]`. The action order is
`delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper`: normalized
robot-base OSC pose increments and a gripper command. These are not absolute
poses, joint angles or position deltas in metres. `--action-horizon` sets how
many actions from a prediction are executed before the next call.

Camera observations contain:

| Key | Contract |
| --- | --- |
| `observation.images.main` | RGB `uint8` H×W×3; default 256×256, `task_close` camera |
| `observation.images.wrist` | RGB `uint8` H×W×3; `robot0_eye_in_hand` camera |
| `observation.state` | 8 values: EEF position, EEF rotation vector, signed left/right finger positions |
| `observation.table_imu_window` | 10×6 table IMU measurement window |
| `observation.table_imu_timestamps_s` | Ten sample timestamps |
| `observation.table_imu_dt_s` | IMU sample interval |
| `task` | Current language instruction |
| `timestamp` | Elapsed policy time in seconds |

EEF position is in metres and orientation is a rotation vector in radians,
both in the robot-base frame. Finger positions are signed slide-joint
coordinates, not a binary open/close value. `contract` mode returns the
environment's declared policy fields without images; its schema differs from
camera mode.

The boundary rejects privileged observation namespaces and invalid actions.
Repeated `--policy-arg NAME=VALUE` options are passed only to parameters named
in the factory signature; values are parsed as Python literals when possible.
A synchronous timeout requires the adapter to declare and enforce a positive
`deadline_s`; arbitrary blocking Python inference cannot be interrupted by
the runner.

## Evaluate a policy

For a camera policy whose factory is importable:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m shakebench.scripts.evaluate \
  --task-module shakebench.environments.ring_on_peg \
  --states shakebench/models/assets/shakebench_ring_on_peg_states.json \
  --policy my_policy:make_policy --policy-id my-checkpoint \
  --gamma 0.5 --horizon-steps 1200 --action-horizon 4 \
  --main-camera task_close --height 256 --width 256 \
  --output out/ring_gamma0.5.json
```

For pick-place, omit `--task-module` and select the intended pick-place asset.
`--state-ids` restricts a run; unknown or duplicate IDs are rejected.
`--mode-params` takes JSON for custom excitation. Run each gamma into a distinct
file and record state IDs, code commit, simulator versions and checkpoint.

If your training dataset includes `meta/shakebench_collection.json`, pass
`--dataset path/to/dataset`. The runner reads training states and camera/image
settings and rejects overlapping train/evaluation state identities.
`--allow-train-states` explicitly permits overlap and records it. A standard
LeRobot export without that completed manifest is not accepted by this option;
its dataset name or split label alone does not prove independence.

### Asynchronous inference

`python -m shakebench.scripts.evaluate_async --help` exposes the continuous
chunk runner. For a compatible external LeRobot-style model/checkpoint:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m shakebench.scripts.evaluate_async \
  --task-module shakebench.environments.ring_on_peg \
  --states shakebench/models/assets/shakebench_ring_on_peg_states.json \
  --policy shakebench.policies.async_policy:make_chunk_policy \
  --policy-arg checkpoint=path/to/checkpoint --policy-arg device=cuda \
  --gamma 0.5 --time-scale 2.0 --output out/async_ring.json
```

Inference runs in a spawned process while simulation advances. `--time-scale`
sets wall seconds per simulation second and scales measured inference latency.
Overruns and timing validity are reported; increase the scale if simulation
cannot maintain the pace. This example requires an external model
implementation; no pretrained weights are bundled.

### WebSocket policy services

The supplied adapter uses a compatible StarVLA-style `deployment` client:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m shakebench.scripts.evaluate \
  --policy shakebench.policies.websocket:make_policy \
  --policy-arg host=127.0.0.1 --policy-arg port=10093 \
  --task-module shakebench.environments.ring_on_peg \
  --states shakebench/models/assets/shakebench_ring_on_peg_states.json \
  --gamma 0.5 --output out/websocket_ring.json
```

Start the model server separately and make its client package importable.
`shakebench.scripts.eval_websocket_policy` is also available as a one-episode
pick-place convenience wrapper; its `--help` lists the narrower options.
The adapter requires `available_unnorm_keys` to include `new_embodiment`,
`action_keys=["action.osc", "action.gripper"]`, empty `state_keys`, and a positive
`action_chunk_size`. It sends main then wrist images resized to 224×224 plus
language, and expects `ok=true` with actions shaped `[1, N, 7]`. The server must
restore normalized ShakeBench OSC units.

Other models/transports, including DynamicVLA, can use an external
`module:factory` implementing the policy interface. A model-specific DynamicVLA
service adapter is not included. Match training camera, view order,
preprocessing, state conventions, normalization and action units before
interpreting success rates.

## Results and reproducibility

The synchronous result uses `schema_id="shakebench.policy_evaluation"` and
`schema_version=1`. It contains evaluation settings, `train_split_check`, `episodes`
and `summary`. Each episode records state ID, gamma, horizon, policy identity,
observation identity, executed-action count, timing, validity, outcome and typed
policy errors.

| Termination cause | Meaning |
| --- | --- |
| `success_latched` | Task success achieved |
| `horizon_exhausted` | Time limit reached without success |
| `task_rule_violation` | Task-specific rule broken |
| `policy_error` | Inference exception, timeout or invalid output |
| `policy_abort` | Policy aborted the episode |
| `invalid_execution` | Environment construction or execution failed |

`summary.success_rate_over_attempts` uses all attempted episodes;
`success_rate_over_valid` excludes invalid execution. Report the denominator
and failure causes. Results contain an executed-action count, not a complete
action-trajectory archive.

Fixed seeds aid reproducibility, but changing simulator versions, contact
geometry or camera assets can change rollouts/images. State loaders validate
schemas and task versions; they do not authenticate asset origin or certify a
score. Check a behavior-preserving change with:

```bash
MUJOCO_GL=disable python tools/compare_rollouts.py record --out before.json
# Run on the changed checkout with that package first on PYTHONPATH.
MUJOCO_GL=disable python tools/compare_rollouts.py record --out after.json
python tools/compare_rollouts.py diff before.json after.json
```

## Training demonstrations

Demonstrations are hosted separately in the
[ShakeBench Hugging Face dataset](https://huggingface.co/datasets/Blien2002/ShakeBench).
The seven published directories are:

- `shakebench_pickplace_300`
- `shakebench_panel_operation_400`
- `shakebench_push_t_400`
- `shakebench_ring_on_peg_200`
- `shakebench_stack_blocks_400`
- `shakebench_upright_400`
- `shakebench_wine_rack_300`

Eight evaluation environments and seven training directories are distinct
inventories: no ring-on-rail training directory is currently listed. Follow the
dataset card for versions, format and provenance; pin a dataset revision rather
than relying on mutable `main`. Train in your model framework: the repository
provides the evaluation boundary, not a training/collection runner.

## Development

```bash
python -m pip install -e ".[dev,gym]"
MUJOCO_GL=disable pytest -q
black --check shakebench tools tests
isort --check-only shakebench tools tests
```

`pytest -m "not slow"` skips tests that initialize every task, though shared
runtime tests still build a simulation. See [AGENTS.md](AGENTS.md) and
[architecture](docs/architecture.md) for dependency boundaries and new task
registration.

## Frequently asked questions

**Does EGL mean GPU physics?** EGL provides offscreen rendering. The task
runtime steps CPU MuJoCo; there is no MJWarp runner.

**Can I change the robosuite pin or use MuJoCo 3.10?** The extension mirrors
the pinned simulation loop. Upgrades require reviewing it and comparing task
rollouts; MuJoCo 3.10 is outside the supported range.

**Are optional robosuite warnings failures?** Private macros, extra robot
models and whole-body IK are optional upstream features and are not required
for Panda tasks. Renderer exceptions or failed rollouts still need investigation.

**Why does a camera policy fail in `contract` mode?** This mode has no camera
images. Use `cameras` for image policies.

**Why does `--dataset` reject a LeRobot export?** The split checker needs the
completed ShakeBench collection manifest and `meta/info.json`. An export
without that manifest needs provenance-aware split checking outside this option.

## Citation and licenses

When using ShakeBench, cite this repository URL and the exact commit used.
No ShakeBench paper citation is supplied here. Acknowledge the underlying
[robosuite framework](https://robosuite.ai/) and [MuJoCo](https://mujoco.org/)
as appropriate.

The code license is in [LICENSE](LICENSE); existing copyright and notices
are retained. Assets have separate terms and attribution:

| Component | Attribution and terms |
| --- | --- |
| RoboCasa models | [Source notices](shakebench/models/assets/objects/robocasa/LICENSE.md), CC BY 4.0 |
| YCB-Sim drill | [Source notices](shakebench/models/assets/objects/ycb_sim/power_drill/LICENSE.md), Apache-2.0 model and CC BY 4.0 YCB data |
| ambientCG wood textures | CC0-1.0; source records beside the PNGs |
| Panel scene and controls | [Source records](shakebench/models/assets/objects/panel/SOURCES.md) for the current asset revision |

The current Panel controls use independently constructed analytic meshes, revision
`original_controls`, including fitted compound knob contacts and
mesh contacts for the lever, collar and shoulder. Moving masses are preserved;
explicit control inertia follows the new analytic surfaces. These contacts differ
from older scene revisions. The linked historical Panel demonstrations have not been
updated to this appearance; see [asset provenance](docs/assets.md).

The code license does not override third-party asset terms or the separate
training dataset's terms.
