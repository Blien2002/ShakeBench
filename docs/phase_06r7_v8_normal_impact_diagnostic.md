# Phase 06R7 / V8 normal-impact diagnostic

Status: **DESIGN_EVIDENCE_ONLY**. The machine-readable output is
`robosuite/models/assets/shakebench_phase_06r7_v8_normal_impact_diagnostic.json`.
It is not a V8 candidate result, has no selection authority, is not scoreable,
and is not a hidden score. The diagnostic compiles raw MuJoCo XML only; it
does not instantiate a robosuite task environment or call reward, success,
metrics, controller, or policy-observation APIs.

## V7 mechanism retained

The V7 raw contact traces are preserved unchanged. They show negligible
terminal angular speed but no named table contact at `0.50 s`, with the Can
moving downward at about `0.1325 m/s`. This is unresolved normal rebound/
flight, not persistent rolling. The V7 incline fixture is also retained as
invalid design evidence: its first angle was `0.10 rad`, so it had no zero-
angle stable lower control.

## Raw normal-impact fixture

The V8 diagnostic uses the committed Can collision mesh, the exact
`0.325 x 0.30 x 0.03 m` table collision box, Can mass `0.349 kg`, Can
inertia `[0.00024106572568945823, 0.00024106572568945823,
0.00010986473347417683] kg*m²`, table mass/inertia `[32 kg,
0.9696, 1.1363, 2.0867]`, V7 low-frequency isolator parameters, the V7
dynamic-deck weld, and drop height `0.15 m`. The release pose is
`[0, 0, 0.95] m` above the table top at `0.80 m`; all initial Can velocities
are zero. Each of the seven V7 contact tuples is traced at every `0.0002 s`
step through exactly `0.50 s`.

Each trace records Can pose/quaternion, six-component velocity, vertical
velocity, named table-contact count, normal/tangential force, cumulative
normal impulse, contact distance, penetration, warnings, and kinetic/
potential/mechanical energy. Contact-loss and contact-reentry times, peak
downward/rebound vertical speed, terminal direction, terminal support, and
terminal/final-`0.05 s` velocity maxima/RMS values are derived from those raw
rows.

With the exact raw fixture and V7 `c6_rolling_high` default, first impact is
about `0.1462 s`, the Can has several contact-loss/re-entry intervals, and by
the terminal sample it has named table support with about `1.56 N` normal
force and about `0.00452 m/s` linear speed. The largest recorded penetration
is about `0.0002721 m`. The different terminal state from the V7 task-side
trace is explicitly evidence that the V7 no-contact result includes a fixture/
model-seam component; V8 therefore makes terminal named support an explicit
hard gate and tests the normal response in a raw fixture.

## Level-first incline calibration

Before any angle is measured, the V8 fixture settles the V7 default from the
same non-penetrating drop pose for `0.50 s`, then holds the resulting state
with zero velocity for `0.10 s`. The level control passes: terminal support
has four named contacts and about `1.57 N` normal force; maximum hold
horizontal speed is about `7.49e-6 m/s` and displacement relative to the
post-settle pose is about `5.99e-7 m`.

The inclined fixture reuses that settled Can pose, zeroes velocity, starts at
`0 rad`, includes sub-critical angles, extends through and above
`atan(0.30)`, and refines the measured stable/sliding transition by five
binary steps. The measured raw bracket is approximately
`[0.003125, 0.00625] rad` with width `0.003125 rad`; the Coulomb angle
`0.2914567944778671 rad` is retained only as an analytic comparison, never as
the measurement or a substitute for raw MuJoCo evidence.

The V8 protocol retains the level-control prerequisite and the binary
refinement convention. No friction, recovery horizon, speed limit,
penetration limit, or force envelope was relaxed to obtain these observations.
