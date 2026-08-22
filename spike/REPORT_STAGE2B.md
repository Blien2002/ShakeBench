# ShakeBench spike Stage 2B — gate-2 stop report

## Outcome

Stage 2B stopped at B2-2 as required. Gate 1 passed, but Gate 2 failed because
the selected 3 N per-finger force limit is not on a success-rate plateau:

| Force limit per finger | Success | Dominant failure |
|---:|---:|---|
| 1.0 N | 0/20 | `descend_contact_timeout` (20) |
| 1.5 N | 0/20 | `descend_contact_timeout` (20) |
| 3.0 N | 20/20 | none |
| 6.0 N | 20/20 | none |

The 3 N point differs from its 1.5 N neighbor by 100 percentage points and
from its 6 N neighbor by 0 points. The required difference is less than five
points for both neighbors. Therefore Stage C was not started, and Stage D was
not triggered.

This is not evidence that 3 N cannot hold the cube. It shows that the current
implementation applies the force cap throughout approach and contact
acquisition, so the scan mixes finger-closing dynamics and the fixed descend
timeout with post-contact holding force. All 1.0/1.5 N failures occurred before
latching. The precise joint-load mechanism was not separately instrumented,
so that causal refinement remains an inference from the phase and failure
classification.

## B0. Configuration provenance

The uncommitted `src/shakebench/config.py` change from 0.3 mm to 0.9 mm could
not be attributed to a commit or formal output. Its inline comment described
an intended approximately 1.8 N squeeze and a roughly two-times static-load
margin, but no matching formal artifact was found. It was restored to the
committed 0.3 mm value; no Stage 2B main-repository numeric design change was
made.

Twenty-two local `out/round5*.json` files were inspected through their embedded
`controller_motion_limits.gripper_contact_preload_m` field:

| Embedded preload | Artifact count | Interpretation |
|---:|---:|---|
| 0.3 mm | 21 | all official, baseline, settle, smoke, and Pareto outputs |
| 1.5 mm | 1 | explicitly named `round5_official_grasp_probe_preload15_seed17.json` |
| 0.9 mm | 0 | no Round-5 artifact |

Thus the Round-5 finding is attributable to 0.3 mm, except for the explicitly
separated 1.5 mm diagnostic probe. Raw provenance is in
`out/stage2b/config_provenance.json`.

## B1. Paired force-control rerun and Gate 1

Configuration: seeds 0–19, 20 Hz policy, 0.2 ms physics timestep, 200 Hz
post-model-step diagnostics, force-limited full-close grip, 3 N per finger.

| Condition | Success | Slip median / P90 / max | Warning count | Force samples |
|---|---:|---:|---:|---:|
| Gamma=0 | 20/20 | 1.592 / 1.732 / 1.798 mm | 0 | 32,300 |
| Gamma=0.5 | 20/20 | 2.402 / 3.262 / 3.798 mm | 0 | 32,370 |

Gate 1 requires at least 19/20 at Gamma=0. It **passes at 20/20**.

The median of each episode's post-latch mean normal force was 2.988 N left and
2.988 N right at Gamma=0, and 2.979 N left and 2.977 N right at Gamma=0.5.

### Single-side versus double-side contact loss

Fractions use the 0.05 N contact threshold and are calculated per episode from
200 Hz samples. Values below are the distribution across 20 episodes.

| Gamma | Metric | Median | P90 | Maximum |
|---:|---|---:|---:|---:|
| 0 | exactly one side below threshold | 0% | 0% | 0.0621% |
| 0 | both sides below threshold | 0% | 0% | 0% |
| 0.5 | exactly one side below threshold | 0.1852% | 0.3848% | 0.6832% |
| 0.5 | both sides below threshold | 0% | 0.3168% | 0.4908% |
| 0.5 | left below threshold | 0.1858% | 0.4465% | 0.7453% |
| 0.5 | right below threshold | 0.1543% | 0.6139% | 0.6832% |

The force controller therefore prevented the sustained all-contact loss seen
in the original position-latch failure while allowing short one-side unloading
events. Exact double-zero fractions equal the reported both-below-threshold
distribution at Gamma=0.5 for this dataset; neither caused a task failure.

Raw paired results are under `out/stage2b/exp1_rerun/`.

## B2. Sensitivity scan and Gate 2

The B1 Gamma=0.5 failure set was empty, so all 20 seeds were used for robustness
checks as required.

| Order | Parameter | Baseline / alternate | Result | Gate interpretation |
|---:|---|---|---|---|
| 1 | pad solref damping ratio | 0.5 / 1.0 | 20/20 vs 20/20; 0 flips | pass for this row |
| 2 | grip force | 1.0 / 1.5 / 3.0 / 6.0 N | 0% / 0% / 100% / 100% | **3 N not on platform** |
| 3 | OSC `kp` | 150 / 300 | not run | stopped after B2-2 |
| 4 | physics timestep | 2e-4 / 1e-4 s | not run | stopped after B2-2 |
| 5 | cube-table solref | 6e-4 / 1.2e-3 s | not run | stopped after B2-2 |
| 6 | `_move_action` gain | 4.0 / 1.0 | not run | diagnostic; stopped after B2-2 |

Gate 2 **fails** on its special grip-force plateau criterion. Placeholder JSON
files for rows 3–6 explicitly say `not_run`; they contain no inferred results.

### Scope correction for the Stage A force derivation

The Stage A expression `mg/(2*mu) * 1.95` accounts only for vertical
gravity-plus-inertial load. It omits lateral inertia and moment caused by an
eccentric grasp point. The reported 0.5293 N per-finger worst-case value is
therefore a lower bound, not a complete force requirement. The 5.67-times
ratio for 3 N must be read as margin over that lower bound only.

There is a second coupling: the current compiled `forcerange` is active before
bilateral contact, although A2's purpose is post-contact holding. The observed
0-to-100% transition between 1.5 N and 3 N is dominated by contact acquisition,
not measured hold failure. Stage A should separate acquisition authority from
the post-contact cap before selecting a holding-force plateau.

## B3. Decoupled-table instrument (separate structural probe)

This instrument ran no task policy. The Panda base remained on the original
deck; the table was attached to a second mocap+weld support. Both supports used
the same calibrated spectral amplitudes and seed, with a uniform pi/2 phase
offset applied to every table spectral line. Each row has 1,200 samples at
200 Hz over six seconds and zero MuJoCo runtime warnings.

| Gamma | Base-frame EE wobble | Table-frame object slip | Base-frame table motion |
|---:|---:|---:|---:|
| 0.15 | 0.3576 mm | 0.000247 mm | 0.8066 mm |
| 0.30 | 0.7688 mm | 0.000247 mm | 1.6130 mm |
| 0.50 | 1.2753 mm | 0.000275 mm | 2.6883 mm |
| 0.75 | 1.9019 mm | 0.000345 mm | 4.0324 mm |
| 0.95 | 2.3801 mm | 0.000700 mm | 5.1078 mm |

Decoupling creates millimeter-scale robot-table relative motion, but static
table-frame object slip remains sub-micrometer at mu=1.5. This makes D2 useful
for testing task-relative geometry, not evidence that the resting cube itself
slides under the current friction setting. B3 is excluded from B1/B2 gates and
from the hard-mounted Stage C definition.

Raw results are in `out/stage2b/decoupled_instrument/decoupled_instrument.json`.

## Stage C Gamma ladder — not run

Both gates must pass before C. Gate 2 did not, so all cells remain unverified.

| Strategy / diagnostic | Gamma=0 | 0.15 | 0.30 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| `frozen_replay` | not run | not run | not run | not run | not run | not run |
| `reactive_scripted` | not run | not run | not run | not run | not run | not run |
| EE wobble during C | not run | not run | not run | not run | not run | not run |
| table-frame object slip during C | not run | not run | not run | not run | not run | not run |

The predeclared C4 prediction was that both policy rows would stay flat within
plus or minus ten percentage points because A0 showed sub-micrometer object
motion and at most 2.384 mm EE wobble. There is **no observed C curve to compare
with that prediction**; treating B1's two Gamma points as C would violate the
predeclared 10-state frozen/reactive design.

Consequently none of the C3 interpretation branches can be selected, and no
claim about Gamma discrimination is made.

## Stage D — not triggered

D1 and D2 are authorized only after C identifies non-discrimination. Since C
was forbidden by Gate 2, that trigger was never evaluated. No D rollout was
run. B3's instrument-only structural result is not a substitute for D2.

## Caveats and recommendation

- B1 and each completed B2 condition use 20 deterministic seeds; these are
  empirical rates over the authored initialization set, not population
  confidence bounds.
- Force-threshold fractions are sensitive to the exact 0.05 N definition, so
  both exact-zero and threshold-based episode data remain in the raw JSON.
- The 1.0/1.5 N acquisition failures lack a finger-joint effort/velocity trace;
  the acquisition-coupling explanation is strongly indicated by phase labels
  but not fully instrumented.
- B3 uses a uniform pi/2 phase shift on each spectral line. Other cross-support
  coherence models can change relative motion and require separate design
  justification.
- B2-3 through B2-6, C, and D are honestly unverified, not implicitly passed.

**Recommendation:** return to Stage A and redesign the force limit so contact
acquisition authority is separated from the post-contact holding cap; then
reselect a force on a demonstrated plateau and rerun B1/B2 before any Gamma
ladder claim.
