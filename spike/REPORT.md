# ShakeBench robosuite spike: corrected contact-scope experiment

## Decision

The previous report's localization was too broad because it lacked a
zero-vibration control. That control is now part of the experiment.

After correcting the XML contact scope, the reactive policy succeeds in 20/20
episodes at Gamma=0 and 12/20 at Gamma=0.5. The fix raises Gamma=0.5 success
from the old implementation's 2/20 to 12/20, but it still misses the required
16/20 gate. **Stop after experiment 1; do not run experiment 2 yet.**

The corrected conclusion is: the no-vibration grasp is stable, the original
all-geom XML rewrite was a major defect, and the remaining failures are
vibration-dependent contact-loss events. The present evidence does not support
either the old claim that grasping is unconditionally broken or the stronger
claim that the XML scope bug was the only cause.

## Implementation correction

The original processor applied the cube-table parameters to every collidable
geom, including both Panda finger pads. The compiled pads therefore had
`margin=gap=0.001`, `solref=[0.0006, 1]`, and sliding friction 1.5 instead of
their robosuite defaults.

The processor now creates one explicit `cube_g0`--`table_collision` contact
pair. That pair alone receives `margin=gap=0.001`,
`solref=[0.0006, 1]`, and mu=1.5. The compiled finger pads are checked at every
environment construction and must remain exactly:

```text
margin=0  gap=0  solref=[0.01, 0.5]  friction=[2, 0.05, 0.0001]
```

An explicit pair is preferable to merely excluding names ending in
`_pad_collision`: leaving the 1 mm gap on the cube geom could still influence
pad-cube contacts through MuJoCo's contact-parameter combination rules.

`run_exp1.py` now runs Gamma=0 with `motion_sampler=None` before every positive
Gamma condition unless `--no-zero-control` is explicitly supplied for debug.
Each episode also records compiled contact parameters, object mass, post-latch
left/right normal-force statistics, and the fraction of samples where both
finger forces are exactly zero.

## Reproduction

Environment: Python 3.11.14, `robosuite==1.5.2`, `mujoco==3.3.7`, NumPy
1.26.4. Scored physics uses a 0.2 ms timestep and the policy runs at 20 Hz.

```bash
NUMBA_CACHE_DIR=/tmp/shakebench_numba spike/.venv/bin/python \
  spike/run_exp1.py --self-check --seed 17 --timestep 0.0002 --window-s 6 \
  --output spike/out/fixed/exp1_selfcheck.json

NUMBA_CACHE_DIR=/tmp/shakebench_numba spike/.venv/bin/python \
  spike/run_exp1.py --mocap-lag-check --seed 17 --timestep 0.0002 \
  --window-s 6 --output spike/out/fixed/mocap_step_lag_ab.json

NUMBA_CACHE_DIR=/tmp/shakebench_numba spike/.venv/bin/python \
  spike/run_exp1.py --episodes 20 --seed 0 --timestep 0.0002 \
  --episode-s 16 --gamma 0.5 --output spike/out/fixed/formal
```

The final command intentionally exits with status 2 because the Gamma=0.5
gate fails. `fixed/formal/gamma_0p0` and `gamma_0p5` contain the 40 raw episode
JSON files and their per-condition summaries. `exp1_comparison.json` is the
paired aggregate.

## Experiment 1a: corrected assembly self-check

All mandatory checks still pass after the contact-scope and split-step timing
corrections.

| Check | Corrected result | Reference / gate | Result |
|---|---:|---:|---:|
| Zero-motion EE drift | 0.000653 mm | < 0.05 mm | pass |
| Zero-motion joint drift | 0.00000156 rad | < 0.001 rad | pass |
| Gamma=0.5 command amplitude, 6 s | 1.8140 mm | 1.81 mm | pass |
| Peak weld tracking error | 77.54 um (4.274%) | 88.8 um (about 4.9%; <20% gate) | pass |
| Static cube-table penetration, median | 0.0984 um | about 0.2 um; not 1 mm suspension | pass |
| MuJoCo runtime warning count | 0 / 0 / 0 | 0 | pass |
| Same-seed maximum qpos difference | 0.0 | exactly 0 | pass |

The Gamma calibration remains `level_scale=1.355673`,
`gamma_realized=0.5`, with full-episode peak deck displacement 2.15973 mm and
peak speed 0.0958344 m/s. Common-deck cube motion remains 0.269 um; this is the
clean value after removing Lift's unrelated 10 mm initial free fall.

### `lite_physics` split-step timing

robosuite calls `mj_step1`, then `_pre_action`, then `mj_step2`. Writing the
current-time mocap target in `_pre_action` therefore leaves `mj_step1` using
the preceding target. A controlled A/B with all other inputs fixed measured:

| Mocap command lead | Peak weld error | Error / command | P90 error | Warnings |
|---:|---:|---:|---:|---:|
| 0 physics steps | 92.49 um | 5.099% | 61.20 um | 0 |
| 1 physics step | 77.54 um | 4.274% | 51.37 um | 0 |

The implementation now writes the next physical step's target. This is a
small fidelity correction; it did not rescue the failing Gamma=0.5 seed 1 in
the four-seed regression test.

## Experiment 1b: paired Gamma=0 / Gamma=0.5 result

Both conditions use seeds 0--19, identical initial-state generation, a 16 s
limit, the same reactive controller, no grasp assist, and the unchanged 10 mm
slip threshold. The Gamma=0 condition uses no motion sampler at all.

| Metric | Gamma=0 control | Gamma=0.5 |
|---|---:|---:|
| Success | 20/20 (100%) | 12/20 (60%) |
| `grasp_slip_exceeded` | 0 | 8 |
| Slip median | 1.760 mm | 5.333 mm |
| Slip P90 | 1.899 mm | 20.711 mm |
| Slip maximum | 2.161 mm | 22.537 mm |
| Both-force-zero fraction, median | 0.000% | 1.235% |
| Both-force-zero fraction, P90 | 0.000% | 45.000% |
| Both-force-zero fraction, maximum | 0.621% | 55.556% |
| MuJoCo runtime warnings | 0 in all episodes | 0 in all episodes |

Gamma=0.5 successful seeds are 0, 2, 3, 4, 6, 7, 8, 9, 11, 16, 17, and
19. Failing seeds are 1, 5, 10, 12, 13, 14, 15, and 18; every failure is
`grasp_slip_exceeded`.

For successful Gamma=0.5 episodes, the median post-latch left/right mean
normal forces are 0.905 N / 0.899 N and the median both-zero fraction is
0.619%. For failed episodes, the corresponding values are 0.435 N / 0.631 N
and 38.889%. This strongly associates the failures with contact loss rather
than insufficient static friction. The 20 randomized cubes weigh 70.11--83.00
g, while nominal two-pad friction capacity at roughly 0.9 N per pad is much
larger than their gravity and Gamma=0.5 inertial load.

### Corrected localization

- The 20/20 Gamma=0 result demonstrates that the baseline grasp and scripted
  trajectory work without vibration.
- Restoring pad parameters and scoping the cube-table settings improves
  Gamma=0.5 from 10% to 60%. The previous global geom rewrite was therefore a
  material implementation bug.
- The remaining eight failures are still vibration-dependent and exhibit much
  more simultaneous force loss than successful episodes. They need a narrower
  investigation of pad-cube contact continuity, grasp centering, and the
  controller/contact interaction under relative deck motion.
- Because the corrected result remains below 80%, Gamma discrimination cannot
  yet be measured cleanly with experiment 2.

## Experiment 2: not run by gate

| Strategy | Gamma=0 | 0.15 | 0.30 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| `frozen_replay` | not run | not run | not run | not run | not run | not run |
| `reactive_scripted` | not run | not run | not run | not run | not run | not run |

Recommendation: **keep the experiment stopped and isolate the remaining
vibration-induced contact loss; do not retune score thresholds or start the
Gamma ladder until the fixed Gamma=0.5 implementation reaches 80%.**

## Caveats

- The force diagnostics are sampled at the 20 Hz policy rate, not every 5 kHz
  physics step. Failed episodes contribute only 9--10 post-latch samples, so
  the exact zero-contact fractions are coarse, though their separation from
  successful episodes is large.
- The disposable object is robosuite Lift's near-cube, not ShakeBench's 0.514
  kg elongated workpiece. These results diagnose the prototype implementation;
  they do not validate the final asset's grasp robustness.
- The pre-fix JSON files at the root of `spike/out` remain historical evidence.
  Only `spike/out/fixed/formal` supports the corrected result.
- The one-step mocap lead reduces measured tracking error, but no analytic
  continuous-time ground truth was used to prove that it removes every phase
  convention error.
