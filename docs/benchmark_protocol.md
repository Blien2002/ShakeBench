# Benchmark protocol

Select a suite through `shakebench.benchmark.get_benchmark_dict()`, then select a task and one of its 50 committed initial states:

```python
from shakebench import benchmark
suite = benchmark.get_benchmark_dict()["shakebench_ladder"]()
task = suite.get_task(0)
state = suite.get_task_init_states(0)[0]
env = shakebench.make(**task.config)
env.set_init_state(state)
```

Each HDF5 record contains `object_placement (N,7)`, `workpiece`, scalar `t0`, calibrated `level_scale`, `gamma_realized`, and `seed`. Replayers validate Γ and level to `1e-6`. Evaluation must keep `episode_s=16`, derive horizon from policy rate, use the official physics profile, start from the supplied state, and record controller type, intra-step mode, observation privileges, failure reason, seed, `t0`, Γ, and physics profile.

## Round-7 official scoring

One official configuration contains the first 10 committed init states for
each of the four workpieces (40 episodes), at scale 0.75 and 16 seconds on the
official physics profile. An episode is valid only when all of the following
hold:

```text
support_geometry_valid == true
grasp_assist_used == false
max_penetration_mm < 0.01 * thinnest_runtime_collider_dimension_mm
abs(gamma_realized - init_state.gamma_realized) <= 1e-6
```

Voided episodes are counted separately. A configuration with more than 5%
voided episodes is not paper-eligible. `scripts/run_scorecard.py` writes the
fixed policy/config fields, `n_valid`, `n_voided`, per-episode evidence, and
means with deterministic 95% bootstrap intervals. Critical Γ, frequency, and
control-rate values use linear interpolation at 50% success.

`aggregate_scorecard.py --decision-output ...` performs a paired bootstrap on
matching task/init-state episodes for the phase-vs-reactive decision point. It
issues a continue/stop decision only for the prescribed Γ=0.50,
`frequency_scale=1`, 5 Hz protocol; other inputs remain `pending`.

The package's default Gym environment is an API/state-contract backend and
reports `scoreable=false`; it can smoke-test the pipeline but cannot produce
official numbers. Scorecard and dataset CLIs refuse it by default. This guard
prevents a successful schema test from being mistaken for Isaac/Newton
physics evidence.

## Demonstration schema

`DataCollectionWrapper` writes robomimic-compatible `data/demo_N` groups with
`states`, `actions`, `rewards`, `dones`, observations, JSON `init_state`, and
root train/valid masks. The five supervision labels are
`privileged_object_pose`, `privileged_vibration_qdd`, `privileged_phase`,
`privileged_object_mu`, and `privileged_object_mass`. Policy-time privileged
keys are removed before persistence so training code can exclude all truth
labels with one prefix filter. In `mode="off"`, HDF5 `env_args.level_scale`
is JSON `null`.

Paper-facing collection additionally requires machine-readable
`docs/reports/fidelity_throughput_correlation.json` with `spearman_rho > 0.9`
and `training_eligible=true`. `--allow-contract-backend` bypasses this only for
an explicitly non-scoreable schema smoke test.

The ladder, frequency, policy-bandwidth, and predictability suites contain 20, 20, 24, and 12 tasks respectively. The frequency sweep holds Γ at 0.15 because Γ=0.50 at the 0.25× frequency point exceeds the 25 mm displacement gate. Every task includes a natural-language instruction for VLA policies. Report aggregate success and failure categories over all official states; do not replace failed or crashed states. A process crash with no metrics artifact is missing data, not success.
