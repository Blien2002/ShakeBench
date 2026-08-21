# ShakeBench robosuite spike

## Decision

**Stop after experiment 1. Fix the grasp/contact model before continuing the
port.** The mandatory assembly gate (1a) passed, but the reactive policy passed
only 2 of 20 episodes at Gamma=0.5 (10%, gate: at least 80%). Experiment 2 was
therefore not run.

This is a disposable robosuite prototype. It imports ShakeBench's NumPy
spectral vibration and scoring semantics, but does not import Isaac Lab or move
USD assets, cameras, recording, room geometry, or the Stewart platform.

## Reproduction

The isolated environment uses Python 3.11.14, `robosuite==1.5.2`,
`mujoco==3.3.7`, and NumPy 1.26.4. The commands used for the scored artifacts
were:

```bash
NUMBA_CACHE_DIR=/tmp/shakebench_numba spike/.venv/bin/python \
  spike/run_exp1.py --self-check --seed 17 --timestep 0.0002 --window-s 6

NUMBA_CACHE_DIR=/tmp/shakebench_numba spike/.venv/bin/python \
  spike/run_exp1.py --episodes 20 --seed 0 --timestep 0.0002 \
  --episode-s 16 --gamma 0.5 --output spike/out
```

The second command intentionally exits with status 2 when the experiment-1
gate fails. `spike/out/exp1_selfcheck.json` and
`spike/out/exp1_summary.json` are the aggregate evidence; the 20
`spike/out/exp1_seed_*.json` files retain every 20 Hz post-latch action and the
per-episode metrics.

## Experiment 1a: assembly self-check

Measured with seed 17, a 6 s window, 20 Hz policy commands, and the scored
0.2 ms timestep (5 kHz physics). All mandatory checks passed.

| Check | Measured | Prompt reference / gate | Result |
|---|---:|---:|---:|
| Zero-motion EE drift | 0.000653 mm | < 0.05 mm | pass |
| Zero-motion maximum joint drift | 0.00000156 rad | < 0.001 rad | pass |
| Gamma 0.5 command amplitude, 6 s | 1.8140 mm | 1.81 mm | pass |
| Peak weld tracking error | 92.49 um (5.10%) | 88.8 um (about 5%; <20% gate) | pass |
| Static cube-table penetration, median | 0.0984 um | about 0.2 um; not 1 mm suspension | pass |
| MuJoCo runtime warning count | 0 / 0 / 0 | 0 | pass |
| Same-seed maximum qpos difference | 0.0 | exactly 0 | pass |

The full 16 s calibration also reproduced the supplied reference:
`level_scale=1.355673`, `gamma_realized=0.5`, peak deck displacement
2.15973 mm, and peak deck speed 0.0958344 m/s.

The shaken EE wobble in the robot-base frame was 1.2767 mm, close to the
1.20 mm reference. Cube motion relative to the common deck was 0.269 um,
below the supplied 26 um reference. This is a non-gating single-seed
discrepancy. The cube starts in static table contact, so robosuite Lift's
unrelated default 10 mm free fall is not counted as table slip.

The robosuite log messages about unused controller components are not MuJoCo
dynamics warnings. Every scored rollout had zero entries in `mjData.warning`.

## Experiment 1b: reactive scripted policy

Configuration: Gamma=0.5, seeds 0 through 19, 16 s episode limit, 20 Hz policy,
0.2 ms physics timestep, mu=1.5, no grasp assist, and the unchanged 10 mm
ShakeBench slip tolerance. The controller reads base-frame hand/object state,
gripper opening, and finger contact only; it does not read deck displacement,
vibration phase, or `t0`. A bilateral contact above 0.05 N freezes the two
physical finger targets with a 0.9 mm preload. Successful episodes completed a
4.05 s transfer hold.

| Metric | Result |
|---|---:|
| Success | 2 / 20 (10%) |
| Required gate | >= 16 / 20 (80%) |
| Successful seeds | 8, 9 |
| `grasp_slip_exceeded` failures | 18 / 20 |
| Other failure categories | 0 |
| Maximum grasp slip, median | 17.456 mm |
| Maximum grasp slip, P90 | 31.210 mm |
| Maximum grasp slip, maximum | 68.065 mm |
| Slip tolerance | 10.000 mm |
| Object lift, median / maximum | 2.020 / 129.182 mm |
| Object table slip before grasp, median / maximum | 1.478 / 2.817 mm |
| MuJoCo runtime warnings | 0 in every episode |

Failure-reason histogram:

```text
grasp_slip_exceeded  18  (90%)
success               2  (10%)
```

The two successful episodes had maximum slip of 1.704 mm and 2.096 mm. Most
failures crossed 10 mm during initial lift: the median maximum lift over all
episodes was only 2.020 mm. Seeds 7 and 19 briefly lifted much farther but
still accumulated 68.065 mm and 67.295 mm of hand-object vector drift.

### Localization conclusion

**Yes, `grasp_slip_exceeded` reproduced in MuJoCo.** It reproduced in 18 of 20
scored episodes despite a passing deck assembly, mu=1.5, zero runtime dynamics
warnings, and no grasp assistance. Under the prompt's decision rule, this
localizes the blocker to grasp/contact modeling rather than specifically to
Isaac/Newton.

The next investigation should measure pad normal force after latching, retune
finger contact stiffness/damping without changing the 10 mm score threshold,
and inspect grasp-point offset from the object centre of mass. The disposable
Lift cube is a near-cube with 20--22 mm half extents, not the benchmark's
0.514 kg elongated workpiece, so it does not exercise the requested aspect
ratio or mass exactly. Since the simpler object already slips, the result is
evidence of a contact/preload/controller interaction, but it does not by itself
identify which of those three is causal.

## Experiment 2: not run by gate

The required 2 x 6 table is deliberately marked as not run rather than filled
with old, fast-timestep, or fabricated evidence:

| Strategy | Gamma=0 | 0.15 | 0.30 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| `frozen_replay` | not run | not run | not run | not run | not run | not run |
| `reactive_scripted` | not run | not run | not run | not run | not run | not run |

The pre-declared rule for this situation is "both rows near zero: return to
experiment 1 and fix grasping." Experiment 1 itself is already at 10%, so the
Gamma-axis comparison would be confounded by the baseline grasp failure.
Recommendation: **fix grasping first**.

## Caveats and insufficient evidence

- Only experiment 1 has 20-seed scored evidence. No claim is made about Gamma
  discrimination, frozen replay, low-friction sensitivity, or an independently
  driven table.
- A five-seed 1 ms debug run reached 60% whereas the scored 0.2 ms run reached
  10%. That timestep sensitivity is suspicious and reinforces the contact-model
  diagnosis; the debug run is not used as scored evidence.
- `max_penetration_m` is the maximum over all active contacts, not only the
  cube-table pair. Its scored median was 2.61 um and its maximum was 0.978 mm;
  the latter needs contact-pair attribution before being interpreted.
- Hold-phase EE wobble is undefined for episodes that fail before hold and is
  recorded as zero there. Its aggregate distribution should not be read as a
  vibration rejection score.
- The self-check's 0.269 um common-deck cube motion differs from the 26 um
  reference by roughly two orders of magnitude. It passed the mandatory gates,
  but the difference should be explained before using table-slip magnitude as
  a cross-engine validation target.
