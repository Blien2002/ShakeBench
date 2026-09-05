# Phase 06F-R Prompt: Close the Real-Environment Handoff Gaps

Work in `/home/miracle04/Desktop/ShakeBench`. Phase 06F has a structurally
consistent PASS bundle, but it does **not yet authorize Phase 07**: its
task-native settling and timestep convergence were measured in a reduced
support-box fixture, Gamma=0 parity omitted required state/action semantics,
parity replay labeled terminal summaries as complete traces, and the final
verifier did not reopen every inherited driver/isolator raw artifact.

This remediation validates the already selected official tuple. It must not
search new physics, change the `c3_nominal` contact tuple, change driver or
isolator parameters, modify task-success thresholds, or implement the Phase
07 controller. Any failed physical check leaves the handoff BLOCKED and must
be reported honestly.

Read before editing:

- `docs/shakebench_prompts/phase_06f_final_physics_closure.md`
- `docs/phase_06f_final_physics_report.md`
- `docs/robosuite_benchmark_design_tree.md`
- `docs/robosuite_benchmark_v0_spec.md`
- `docs/shakebench_prompts/phase_07_oracle_controller.md`
- `robosuite/models/assets/shakebench_phase_06f_protocol.yaml`
- `robosuite/models/assets/shakebench_phase_06_to_07_handoff.json`
- `robosuite/models/assets/shakebench_official_physics.yaml`
- `robosuite/scripts/shakebench_finalize_physics.py`
- `robosuite/utils/shakebench_contact_force.py`
- `robosuite/utils/shakebench_physics_finalizer.py`
- `robosuite/utils/shakebench_physics.py`

## 1. Fail closed and preserve Phase 06F

Preserve all published Phase 06F files byte-for-byte as historical evidence.
Create a flat remediation status/addendum and make the official loader require
the remediated handoff version before Phase 07 starts. During remediation the
active handoff must report BLOCKED even though the historical Phase 06F
bundle remains internally hash-consistent.

Create a new, pre-registered supplemental validation protocol rather than
editing the immutable Phase 06F protocol. It binds exactly:

```text
profile  = shakebench.official.physics.v2
driver   = dt_nominal
isolator = low_frequency_damped
contact  = c3_nominal
```

The protocol contains no candidate list and no ranking. It specifies only the
missing validation states, trace schemas, tolerances, inherited raw paths and
output names. Commit the protocol and no-physics dry-run/adapter-contract
artifact before running its MuJoCo states.

Completion criterion: historical Phase 06F hashes still verify, while the
official Phase 7 gate is fail-closed until remediation PASS.

## 2. Validate settling in the real benchmark environment

The reduced box fixture remains a unit/positive-control test, not handoff
evidence. Add a recorder adapter at the existing post-physics seam and run
the real `VibrationPickPlaceCan` model with:

- packaged Phase 06F physics tuple, loaded through a private, exact-hash
  non-scoreable remediation path while the public loader is blocked;
- one Panda, dynamic deck, six-axis isolator, 32 kg worktable, full target
  geometry and explicit Can pairs;
- explicit zero excitation (`Gamma=0`), seed 17, zero 7D policy action,
  ordinary environment step lifecycle and all deck right-limit hooks;
- camera/rendering disabled.

Run two separately reset states:

1. Can at the registered open-worktable initial pose;
2. Can at the target-bottom support pose, centered clear of the walls.

After pose assignment, zero the Can free-joint velocity, call the required
forward/reset synchronization, and record every physics sample for `0.50 s`.
The final `0.10 s` must satisfy the existing hard gates: continuous named
support, linear speed `<=0.02 m/s`, angular speed `<=0.20 rad/s`, penetration
`<=0.50 mm`, finite forces/state, no warning/NaN. Also record deck/table pose,
twist, acceleration, isolator displacement and Can pose/twist in the table
frame.

Use normal `env.step(zero_action)` for the action/control path. Reward or
success outputs may be recorded only to prove interface compatibility; they
must not affect physics validity or change the frozen tuple. Instrument calls
so the artifact states whether reward/success was invoked and confirms no
outcome was consumed by the verifier.

Tests must prove the real-environment recorder observes deck, table, target,
Panda and Can bodies, and must fail if the test silently falls back to
`raw_contact_xml()` or another reduced fixture.

Completion criterion: both real-environment states pass and their artifacts
contain the declared full trace schema.

## 3. Replace fake contact convergence with transient convergence

Run the same two real-environment settling states at physics timesteps:

```text
0.0001 s
0.000125 s
0.0002 s  # selected official timestep
```

Keep the 20 Hz control period, scheduler divisibility, contact tuple, deck,
isolator and initial state identical. Resample or select the native traces on
a pre-registered common `200 Hz` right-limit grid. Never compare arrays by
index when their physical timestamps differ.

For each support state, compare the selected `0.0002 s` trace against both
finer references using at least:

- Can position and orientation in the worktable frame;
- Can linear and angular velocity in that frame;
- worktable pose/twist and isolator relative displacement;
- named support-contact indicator;
- contact penetration;
- integrated normal/tangential impulse and final-window force RMS;
- warning and finite-state sequences.

Pre-register numerical absolute/relative tolerances in the supplemental
protocol before running. Retain the existing `10%` upper bound for relative
contact-trace/impulse metrics unless a dimensioned absolute tolerance is more
appropriate; position tolerance may not exceed the existing `0.50 mm`
penetration scale. Require both finer-reference comparisons to pass.

The old mean-`mg` comparison stays in the archive but cannot satisfy this
gate. Add a negative test where mean support force matches while an injected
velocity/penetration transient differs; the verifier must reject it.

Completion criterion: convergence is derived from timestamped transient
traces, not one steady-state scalar.

## 4. Implement bounded Gamma=0 parity completely

Run parity through the real environment and record one full, timestamped
physics/control trace. Verify all pre-registered semantics:

- official profile/hash, timestep, control frequency and 7D action shape;
- normalized, decoded, clipped and applied zero action;
- Panda base, worktable, Can and target nominal poses in their declared
  frames;
- worktable equilibrium / dynamic-deck static response at `Gamma=0`;
- Can support, relative pose/twist, contact identity and penetration;
- target containment/evaluator contract hash and unchanged success thresholds;
- no warning/NaN and no privileged truth in policy observations.

Bounded parity does not require trajectory equality with stock robosuite and
does not require task success. It requires that the declared nominal geometry,
action semantics, evaluator definition and passive Gamma=0 state are all
present and within their registered bounds.

Do not manually reproduce `MujocoEnv.step()` with a private loop. Use the
ordinary environment step lifecycle so post-integration refresh, post-physics
hooks, metrics and timestamps execute exactly as they will in Phase 07.

Completion criterion: the parity artifact contains raw values for every
declared field, and its verifier independently recomputes every bound.

## 5. Make replay evidence genuinely complete

Generate three fresh-process parity replays from the remediation protocol.
Each process must store the full declared timestamped trace, or a canonical
lossless trace payload plus schema, shapes and dtypes from which its digest is
recomputed. A terminal summary alone is insufficient.

Set `complete_trace=true` only after validating:

- exact sample count and timestamp sequence;
- all required fields, shapes, dtypes and finite values;
- recomputed per-field and whole-trace digests;
- process indices exactly `{1,2,3}`;
- identical profile, initial state, action and seed;
- equal trace and metric digests across independent processes.

Add negative tests for missing middle samples, changed dtype/shape, equal
summary with different intermediate state, and a forged `complete_trace=true`.

Also rerun the selected-contact replay in three fresh processes using the real
environment settling recorder. Existing driver and isolator replay may be
inherited only after the raw-evidence audit in the next section passes.

Completion criterion: parity/contact replay equality is demonstrated from
trace values, not asserted by a Boolean.

## 6. Reopen every inherited raw artifact

Extend the final verifier so it follows the inherited manifest to every V6
driver and isolator raw path. For each file it must:

- recompute file SHA-256 and embedded payload hash;
- validate schema, protocol hash, state ID, candidate ID and resolved-state
  digest;
- recompute driver/isolator hard gates from numeric raw fields;
- confirm complete expected coverage: 18 driver states and three isolator
  candidates;
- recompute the selected values `dt_nominal` and
  `low_frequency_damped` using the registered ordering.

Hashing the embedded `inherited` description again is not verification. The
verifier must fail if any referenced raw file is missing or mutated even when
the selection/handoff JSON is rehashed consistently.

Add mutation tests against one driver raw metric, one isolator raw metric, one
missing file and one reordered/incomplete matrix.

Completion criterion: inherited evidence is independently reopened,
recomputed and cross-bound to the official profile.

## 7. Repair release hygiene

Before the final handoff:

- add the Phase 06R6, Phase 06R7, Phase 06F and this remediation prompt to
  version control together with the README index;
- reconcile the advertised Python version with actual imports: either use a
  compatible `typing_extensions` dependency for Python 3.7, or raise the
  minimum Python version and update setup metadata, formatter targets, docs
  and CI consistently; add an import smoke for the declared minimum version;
- retain the renderer/EGL limitation as an explicit infrastructure lane;
  define a reproducible headless skip/marker rule and require renderer-capable
  CI before release, without calling the local full suite PASS;
- record the current large raw-evidence/package size as Phase 10 debt; do not
  rewrite Git history or remove evidence during this remediation.

Completion criterion: `git status --short` is clean, the originating specs are
tracked, and package metadata no longer advertises an interpreter that cannot
import the official loader.

## 8. Regenerate the only valid Phase 07 handoff

Create versioned remediation evidence and regenerate final status/handoff
without changing official physics values. The new handoff must bind:

- original Phase 06F protocol, selection and official profile hashes;
- supplemental remediation protocol/registration commit;
- two real-environment settling traces at all three timesteps;
- transient convergence results;
- complete Gamma=0 parity trace and three-process replay;
- selected-contact real-environment replay;
- independently reverified inherited driver/isolator raw manifest;
- package/install evidence.

The public official loader and Phase 07 preflight must require this new
handoff version. Historical PASS/status files alone cannot authorize startup.

Run and report exact results for:

- remediation focused and adversarial tests;
- all Phase 06F/profile tests;
- affected Phase 02–05 environment/deck/isolator/provider/sensor tests;
- three fresh-process `--verify-final` calls;
- clean sdist and wheel install, official profile load and MJCF compile;
- full non-renderer suite and the separately reported renderer lane.

Phase 07 may start only when all are true:

```text
remediation final status = PASS
official profile tuple is byte/value-identical to Phase 06F
real-environment open-table and target-bottom settling pass
three-timestep transient convergence passes
bounded parity includes all required state/action/evaluator fields
parity/contact replay is backed by complete traces in three processes
every inherited driver/isolator raw file is reopened and verified
official loader and clean installed packages verify the new handoff
declared Python compatibility is truthful
worktree is clean and all originating prompts are tracked
```

Any failed condition leaves Phase 07 BLOCKED. Stop without implementing the
controller, scanning Gamma, or modifying frozen physics/contact/task success.
