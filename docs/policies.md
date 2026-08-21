# Policies

A policy implements `reset()`, `act(observation)`, and a `requires_privileged` tuple.

```python
class Policy:
    requires_privileged = ()
    def reset(self):
        self.hidden = None
    def act(self, obs):
        return network(obs)
```

Choose `OSC_POSE` for common VLA outputs, `JOINT_POSITION` for the bundled scripted controller, or `VARIABLE_IMPEDANCE` for vibration-aware compliance. Keep outputs normalized to `[-1,1]`; clipping is disclosed in `info`.

Ordinary policies leave `requires_privileged=()`. Round 7 defines four
explicit groups: `object`, `vibration`, `phase`, and `instantaneous_load`.
`policy_env_kwargs(policy)` enables exactly the declared groups and fails
closed on unknown names. Camera, robot proprioception, and synthetic IMU stay
non-privileged.

The disclosed baselines are:

| Policy | `requires_privileged` | Rate | Inner loop |
|---|---|---:|---|
| `oracle_full` | `object, vibration` | 1000 Hz | phase feedforward |
| `oracle_phase` | `phase` | target control rate | phase feedforward |
| `oracle_reactive` | `object, instantaneous_load` | target control rate | ZOH |
| `classical` | none | 200 Hz | FxLMS + fixed impedance |

The reactive tier consumes only the current acceleration sample and keeps no
history, estimator, or prediction. The phase tier reconstructs acceleration
from the disclosed spectral line frequencies and phases. Demonstration
recording uses a separate truth-label channel: the policy never receives
object pose or vibration truth merely because the HDF5 recorder stores them.
