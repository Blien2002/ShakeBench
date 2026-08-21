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

The ladder, frequency, policy-bandwidth, and predictability suites contain 20, 20, 24, and 12 tasks respectively. The frequency sweep holds Γ at 0.15 because Γ=0.50 at the 0.25× frequency point exceeds the 25 mm displacement gate. Every task includes a natural-language instruction for VLA policies. Report aggregate success and failure categories over all official states; do not replace failed or crashed states. A process crash with no metrics artifact is missing data, not success.
