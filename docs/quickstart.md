# Quick start

```python
import shakebench
env = shakebench.make("PickPlace", control_freq=5, use_camera_obs=False)
obs, info = env.reset(seed=17)
while True:
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    if done or truncated: break
env.close()
```

Run it with `./run_python.sh quickstart.py`. For the full Isaac demonstration, use:

```bash
./run.sh --physics-profile official --episode-s 16 --gamma 0.50 --seed 17
```
