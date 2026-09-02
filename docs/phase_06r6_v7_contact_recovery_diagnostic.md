# Phase 06R6 / V7 contact-recovery diagnostic

Status: **DESIGN_EVIDENCE_ONLY**. This artifact is not a V7 candidate result,
has no selection authority, is not scoreable, and is not a hidden score.
It never calls environment reward, success, metrics, controller-outcome, or
task APIs and it never publishes a physics profile.

The machine-readable output is
`robosuite/models/assets/shakebench_phase_06r6_v7_contact_recovery_diagnostic.json`.
It is generated from the V6 resolved contact states and direct MuJoCo
`mj_forward`/`mj_step` calls. The V6 raw contact files remain unchanged and
are not relabeled as V7 evidence.

## Frozen convention

Every candidate is released from the same Can pose, with zero Can linear and
angular velocity, and is traced at the compiled `0.0002 s` model timestep from
`release_time_s = 0` through exactly `0.50 s`. The first impact is the first
sample with a named Can contact whose raw distance is non-positive. The report
retains Can pose and quaternion, six-component velocity, contact count,
normal/tangential force, cumulative normal impulse, contact distance,
penetration, interface names, warning counters, and mechanical energy at every
sample. Terminal values and a fixed final `0.05 s` maximum/RMS window are
recorded for both linear and angular speed.

The limits are unchanged from V6: terminal and trailing-window linear speed
`<= 0.02 m/s`, angular speed `<= 0.20 rad/s`, and maximum penetration
`<= 0.0005 m`. No horizon extension or threshold relaxation is used.

## Observed mechanism

The direct release trace reaches first impact at `0.1498 s`. Its terminal
linear speed remains about `0.1325 m/s` for `c3_nominal`, `c4_torsional`, and
`c3_softer`, so the failure is not a timestamp omission. The trailing window
also remains above the linear gate. Angular speed in this isolated release
trace is small (`< 0.001 rad/s`), so this particular fixture does not support
calling the V6 blocker persistent rolling spin; the registered V6 raw
contact probe separately reports approximately `11.10`, `15.94`, and
`11.11 rad/s` for the three candidates. The combined evidence is classified
as rebound/solver-oscillation with persistent post-impact translation, not a
measurement-cadence error. `c3_softer` additionally reaches about
`0.0004797 m` in the direct trace and `0.0005415 m` in the historical V6
probe, over the fixed V6 penetration gate in the latter.

The exact historical V6 raw values are retained in the Phase 06 report:

| candidate | V6 raw recovery speed | V6 raw angular speed | V6 raw max penetration |
|---|---:|---:|---:|
| `c3_nominal` | `0.462543 m/s` | `11.102875 rad/s` | `0.000216543 m` |
| `c3_softer` | `0.525657 m/s` | `15.937918 rad/s` | `0.000541545 m` |
| `c4_torsional` | `0.462556 m/s` | `11.111834 rad/s` | `0.000216543 m` |

## Finger-force envelope

The compiled Panda model contains the two named position actuators
`gripper0_right_gripper_finger_joint1` and
`gripper0_right_gripper_finger_joint2`. Their compiled `forcerange` is
`[-20, 20] N`, and the first transmission gear is unit gain for each slide
joint. Therefore the finite declared actuator envelope is derived as

```text
2 actuators × max(abs([-20, 20] N)) × abs(1.0 gear) = 40 N total
```

This is an actuator/transmission envelope, not permission for the contact
solver to inject arbitrary force through deep interpenetration. The historical
V6 midpoint audit records approximately `46.5–290.7 kN` across the three raw
files. The design diagnostic reproduces the same failure mode and records
approximately `9.2–57.4 kN` depending on the candidate and direct fixture
state, far above the derived `40 N` envelope. Both named Can–finger pairs are
enumerated in the trace. V7 therefore makes maximum declared finger force
and penetration hard gates in addition to the existing positive minimum-force
gate; the V7 trajectory uses separate shallow, named face holds rather than
the historical deep midpoint placement.

The force audit is a design constraint for V7 candidate registration. It does
not select a contact candidate, rank V6 candidates, or authorize Phase 07.
