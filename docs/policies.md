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

Ordinary policies should leave `requires_privileged=()`. Future oracle policies may declare `("object",)` or `("vibration",)`. The evaluator must enable exactly those groups and record them, so privileged baselines cannot be compared silently with camera/IMU policies. Camera, robot proprioception, and synthetic IMU are non-privileged; object pose and true vibration state are privileged.
