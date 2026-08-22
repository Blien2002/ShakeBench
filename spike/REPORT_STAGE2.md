# ShakeBench spike Stage 2 — Stage A checkpoint

## Scope and pause point

This checkpoint implements **Stage A only**. It measures the Gamma-to-EE-wobble
instrument and replaces the position-latched elastic budget with a
force-limited full-close command. Stages B, C, and D have not run: the Stage 2
protocol explicitly requires confirmation of this design before the paired
rerun and sensitivity gate.

No task parameter changed. Gamma, spectral band, 20 Hz policy rate, cube-table
mu=1.5, the 10 mm slip tolerance, timeouts, geometry, and contact thresholds
remain fixed.

## A0. EE wobble as the artifact predictor

Measurement configuration: seed 17, 0.2 ms physics timestep, 20 Hz controller,
6 s window, and 200 Hz diagnostic sampling from post-physics-step state. Each
condition contains 1,200 samples and has zero MuJoCo runtime warnings.

| Gamma | Measured base-frame EE wobble | Prompt reference | Measured / reference | Object motion in base frame |
|---:|---:|---:|---:|---:|
| 0.15 | 0.3574 mm | 0.3300 mm | 1.083 | 0.247 um |
| 0.30 | 0.7692 mm | 0.7000 mm | 1.099 | 0.247 um |
| 0.50 | 1.2766 mm | 1.2000 mm | 1.064 | 0.273 um |
| 0.75 | 1.9057 mm | 1.8000 mm | 1.059 | 0.319 um |
| 0.95 | 2.3845 mm | 2.2700 mm | 1.050 | 0.363 um |

The ladder independently reproduces the supplied scale to within 5.0--9.9%.
The measured 2.3845 mm maximum is 2.65 times the previous spike preload of
0.9 mm. It also exceeds the main repository's 2.0 mm validation ceiling. A
fixed opening therefore remains capable of turning `wobble / preload` into an
unintended threshold even at the largest currently permitted preload.

Raw evidence: `out/stage2/ee_wobble_ladder.json`.

## A2. Selected grip design: force-limited full close

After bilateral pad contact, both finger position commands are set to fully
closed (0 m). The two stock `kp=1000 N/m` position actuators are independently
limited to +/-3 N. Thus the full-close target keeps requesting inward force as
relative motion unloads one pad; a frozen opening no longer defines the
available elastic displacement.

This is accurately described as a **force-limited saturated position
actuator**, not an ideal torque controller. The compiled MuJoCo model is
validated to contain two `forcerange=[-3, 3] N` actuators. The stock limit is
20 N per finger.

### Force selection

The observed Lift cube mass range is 70.1107--83.0013 g. The limiting contact
coefficient is min(workpiece-side 1.5, pad-side 2.0) = 1.5.

```text
static per-finger requirement = m*g / (2*mu)
                              = 0.2293--0.2714 N

Gamma=0.95 requirement       = static requirement * 1.95
                              = 0.4471--0.5293 N

selected limit / worst case  = 3.0 / 0.5293 = 5.67
```

Three newtons was selected because it is above the entire predicted load range,
matches the approximately 3 N stable-pad result from the earlier 3 mm preload
diagnostic, and remains far below the Panda actuator's stock 20 N cap.

### One-seed engineering check (not a gate result)

Seed 0 at Gamma=0.95 completed the task with 4.198 mm maximum grasp slip and a
4.05 s transfer hold. At 200 Hz, 1,610 post-latch samples were recorded:

| Metric | Left | Right |
|---|---:|---:|
| Mean pad normal force | 2.948 N | 2.805 N |
| Minimum | 0.000 N | 0.800 N |
| Maximum instantaneous contact force | 28.649 N | 22.353 N |

Both pads were simultaneously at exactly zero in 0% of samples, and MuJoCo
reported zero runtime warnings. Instantaneous contact-force peaks can exceed
the 3 N actuator cap because impact and constraint impulses also contribute to
`mj_contactForce`; the steady mean is the relevant check of the selected
actuation limit. One seed is only a wiring/scale check and is not evidence that
Gate 1 or Gate 2 passes.

Raw design and action evidence: `out/stage2/grip_design.json`.

## Required main-repository finding

At commit `ececfa0`, `src/shakebench/config.py` sets
`gripper_contact_preload_m = 0.0003` (0.3 mm), and validation rejects values
above 0.002 m (2.0 mm). The current Stage A measurement is:

| Quantity | Value |
|---|---:|
| Committed main-repo preload | 0.300 mm |
| Validation upper bound | 2.000 mm |
| Measured EE wobble at Gamma=0.50 | 1.277 mm |
| Measured EE wobble at Gamma=0.95 | 2.384 mm |

The committed preload is about one quarter of the Gamma=0.5 wobble, and even
the validation maximum is below the Gamma=0.95 wobble. The formal benchmark's
position-latch design therefore appears exposed to the same elastic-budget
failure mode. This is a plausible explanation for Round-5 ending in
`grasp_slip_exceeded` despite privileged information, but it remains an
inference until reproduced in the main simulator. Stage A intentionally does
not edit `src/shakebench`.

The working tree contains an unrelated, pre-existing uncommitted change from
0.3 mm to 0.9 mm in that file. The committed 0.3 mm value is used here when
describing the repository design; neither value is modified by this spike.

## Non-task parameter change log

| Change | Before | Stage A | Reason |
|---|---:|---:|---|
| Grip hold scheme | frozen opening | full close + force cap | remove displacement budget from difficulty equation |
| Per-finger actuation cap | stock 20 N | 3 N | 5.67x worst predicted need without excessive squeeze |
| Contact-force diagnostic rate | 20 Hz | 200 Hz | avoid 9--10-sample contact-loss estimates |

Finger-pad contact parameters, cube-table pair parameters, OSC kp, physics
timestep, and every task parameter remain unchanged.

## Stage B gates — pending confirmation

| Gate | Required result | Current Stage A status |
|---|---|---|
| Gate 1: baseline usable | Gamma=0 >=19/20 after A2 | not run |
| Gate 2: no single-point artifact | no physical non-task perturbation flips >=50% of failures | not run |

The prior 20/20 Gamma=0 result used the position latch and cannot substitute
for the required A2 rerun.

## B2 sensitivity matrix — predeclared, not run

| Non-task parameter | Baseline / alternate | Failure flips | Gate status |
|---|---|---:|---|
| Grip force | 3 N; 1.5 N / 6 N | pending | pending |
| Pad solref damping ratio | 0.5 / 1.0 | pending | pending |
| OSC kp | 150 / 300 | pending | pending |
| Physics timestep | 2e-4 / 1e-4 s | pending | pending |
| Cube-table solref | 6e-4 / 1.2e-3 s | pending | pending |
| `_move_action` empirical gain | 4.0 / removed or kp-compensated | pending | diagnostic |

No B2 row has been executed or inferred from old artifacts.

## Stage C table — not run

| Strategy | Gamma=0 | 0.15 | 0.30 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| `frozen_replay` | pending | pending | pending | pending | pending | pending |
| `reactive_scripted` | pending | pending | pending | pending | pending | pending |
| EE wobble | pending C run | pending C run | pending C run | pending C run | pending C run | pending C run |
| Object/table slip | pending | pending | pending | pending | pending | pending |

The interpretation rule is fixed before seeing C data:

- both policy rows flat: Gamma does not carry useful difficulty; proceed to D;
- frozen declines while reactive stays flat: feedback is sufficient and the
  meaningful axis is control bandwidth;
- both decline and frozen declines faster: Gamma is potentially valid, but the
  strongest decline must receive a B2 sensitivity scan before any claim;
- both near zero: return to A/B.

## Caveats and next action

- The wobble ladder uses one seed and one six-second window per Gamma. It is an
  artifact-scale instrument, not a confidence interval over task outcomes.
- The object-motion column measures motion in the common robot-base frame; it
  is not post-grasp slip and should not be interpreted as task success.
- The 3 N engineering check is one seed. Its success only verifies that the A2
  command, force cap, and 200 Hz diagnostics are connected correctly.
- Saturated position control and an ideal force controller can differ during
  fast impacts. The B2 half/double-force scan is still mandatory.

**Recommendation at this checkpoint:** accept or revise the 3 N A2 design,
then run Stage B1 and B2. Do not start the Gamma discrimination ladder before
both gates pass.
