# Phase 06F Prompt: Final Official Physics Closure

Work in `/home/miracle04/Desktop/ShakeBench`. This is the final Phase 06
closure, not another open-ended contact sweep. Preserve V1–V8 protocols,
artifacts, reports and commits byte-for-byte. V8 is correctly BLOCKED because
its physics winner `n6_overdamped_4` differs from the contact candidate that
its parity/replay rows pre-bound. No official Phase 06 profile is active.

The goal is to reconcile the contact gate with task-native benchmark practice,
freeze one minimally modified contact profile, generate dependent evidence
from the actual winner, publish one authenticated official physics profile,
and emit a machine-readable Phase 07 handoff. A PASS is allowed only when all
specified gates really pass; otherwise remain fail-closed and report the exact
blocker. Do not implement the Phase 07 controller in this phase.

Read before editing:

- `docs/contact_physics_validation_in_embodied_benchmarks.md`
- `docs/robosuite_benchmark_design_tree.md`
- `docs/robosuite_benchmark_v0_spec.md` sections 14.2 and 15
- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/shakebench_prompts/phase_06r7_v8_normal_impact.md`
- `docs/shakebench_prompts/phase_07_oracle_controller.md`
- `docs/phase_06_report.md`
- V6–V8 protocols, statuses, selected/excluded tables and raw evidence
- `robosuite/scripts/shakebench_select_physics_v8.py`
- `robosuite/utils/shakebench_physics.py`
- `tests/test_shakebench_physics_profile.py`

## 1. Reconcile the specification before selection

Update the design tree and v0 spec in one dedicated documentation commit.
Record the following distinction without changing the already frozen task
success evaluator:

### Official contact hard gates

- complete compiled contact tuple and explicit Can pair scope;
- task-native Can initialization on the open worktable and target bottom;
- `Gamma=0`, zero action, `0.50 s` passive settling;
- named support contact through a final `0.10 s` window;
- final-window linear speed `<=0.02 m/s`, angular speed `<=0.20 rad/s`;
- illegal penetration `<=0.50 mm`, finite contact forces, no warning/NaN;
- level-surface horizontal force/slip sanity for frozen `mu=0.30`;
- finger contact force/penetration envelope;
- three-timestep contact convergence and deterministic replay.

### Public diagnostics, not publication gates

- `0.15 m` free-drop/recovery stress;
- inclined-plane threshold/bracket;
- placement, free-table slip, in-hand slip and contact-loss metrics.

The target-region success rule remains unchanged: after policy release, the
Can must satisfy the existing continuous `0.50 s` containment, relative
velocity, no-finger-contact, support and penetration conditions. Demoting the
standalone drop fixture does not weaken task success.

Document why: mature embodied benchmarks normally freeze simulator/material
configuration and validate task-native initialization/settling; the high-drop
test is outside the reset/release distribution of no-bin PickPlaceCan and must
not drive contact overfitting.

Completion criterion: design tree, v0 spec, research report and Phase 06F
protocol language agree on exactly which metrics are hard gates.

## 2. Implement one deep finalization module

Create or refactor one deep module with this conceptual interface:

```text
finalize_official_physics(protocol, evidence_store)
    -> BlockedResult | PublicationPlan
```

The interface owns the full ordering:

```text
verify inherited driver/isolator evidence
→ execute and verify contact candidate states
→ select exactly one eligible contact
→ materialize winner-dependent parity/replay manifest
→ execute and verify dependent evidence
→ build official profile and Phase-07 handoff
```

Callers and tests exercise this interface. Candidate selection, dependent
manifest creation, publication and status consistency must not be spread
across scripts. Keep MuJoCo and filesystem access behind existing production
and no-physics/test adapters at internal seams.

The protocol pre-registers the **binding rule**, not a guessed winner:

```text
contact_candidate_id = selection.contact_winner
```

After candidate evidence is complete, write an intermediate immutable
selection artifact. The module then deterministically materializes parity and
replay state IDs from the winner plus that artifact hash. Dry-run must test
every possible winner branch. No protocol row may contain a concrete contact
ID for winner-dependent evidence.

Add regression tests reproducing V8 exactly: when `n6_overdamped_4` wins, all
dependent states must bind to `n6_overdamped_4`; pre-binding
`n6_overdamped_2` must fail before physics. A zero-eligible set produces no
dependent manifest, no profile and a BLOCKED result.

Completion criterion: the full PASS and BLOCKED flows work on temporary
synthetic evidence before Phase 06F registration.

## 3. Replace the broken incline hard gate with a level force test

Implement a raw-MuJoCo, task-geometry force-threshold fixture. Settle the Can
on a static level worktable from the exact task-native support pose, zero its
velocity, then apply horizontal body-COM force plateaus using the same compiled
contact pair as the candidate.

For `m=0.349 kg`, `mu=0.30`, `g=9.81 m/s²`, record the analytic reference
`F_mu = mu*m*g` but recompute it from protocol values. Pre-register force
levels bracketing this reference, including at least `0`, `0.5 F_mu`,
`0.8 F_mu`, `1.0 F_mu`, `1.2 F_mu` and `1.5 F_mu`. Reset to the same settled
state for every plateau. Record along-force displacement/velocity, transverse
drift, support force, penetration, contact identity and warnings.

Hard requirements:

- level zero-force control remains supported and stationary;
- sub-threshold plateaus do not exceed the registered static displacement;
- at least one above-threshold plateau produces monotonic along-force motion;
- the measured transition brackets `F_mu` within a pre-registered tolerance;
- no transverse instability, illegal penetration or solver warning.

The old incline fixture remains runnable and its large disagreement is
published as a diagnostic; it is never substituted into the force-test gate.

Completion criterion: positive and intentionally broken-friction controls
prove the force fixture is sensitive to compiled `mu`, rather than merely
returning PASS for every profile.

## 4. Pre-register a finite minimal-intervention contact choice

Create one final flat protocol asset,
`robosuite/models/assets/shakebench_phase_06f_protocol.yaml`, only after the
specification, finalizer, fixtures and negative tests above are complete.
Commit the protocol and its no-physics feasibility artifact before any Phase
06F candidate run.

Do not invent new contact parameters. The finite candidate list is:

1. `c3_nominal` from V6/V7 — benchmark canonical baseline;
2. `n6_critical_negative_control` from V8;
3. `n6_overdamped_2` from V8;
4. `n6_overdamped_4` from V8.

Copy every candidate tuple exactly from its authenticated source artifact:
`condim`, sliding/torsional/rolling friction, margin/gap, `solref`, `solimp`,
iterations and pair scope. Sliding friction remains table/Can `0.30` and
finger/Can `1.00`. Task geometry, Can mass/inertia, driver and isolator are
unchanged.

Selection is eligibility first, then this pre-registered minimal-intervention
priority:

```text
c3_nominal
→ n6_critical_negative_control
→ n6_overdamped_2
→ n6_overdamped_4
```

This ordering deliberately prefers the closest canonical MuJoCo/robosuite
contact behavior. Drop-stress score is not a selection input. If the baseline
passes task-native settling, level force/slip, force envelope, convergence and
numerical gates, it wins; more heavily damped/expanded contacts are considered
only when a less modified profile is ineligible.

Pre-register:

- exact candidate tuples and priority;
- open-worktable and target-bottom settling states;
- the force-plateau fixture and thresholds;
- failure taxonomy and retry policy;
- inherited evidence hashes;
- winner-dependent manifest template;
- official profile schema/id and final handoff schema;
- every raw/output filename.

Run protocol validation, complete dry-run and every-winner adapter-contract
before registration. The feasibility artifact records their exit codes,
resolved-state/template digests and confirms zero MuJoCo calls.

## 5. Execute the final contact freeze

Independently re-verify and inherit only the already valid non-contact results:

```text
driver   = dt_nominal
isolator = low_frequency_damped
```

Recompute source raw hashes, coverage, hard gates and ordering. Do not inherit
any V6–V8 contact selection or the stale V1 official profile.

For all four Phase 06F contact candidates, newly execute:

- compiled explicit-pair audit;
- task-native open-table settling;
- task-native target-bottom settling;
- level horizontal force-threshold test;
- level single-axis slip sanity;
- finger force/penetration envelope;
- three-dt contact convergence;
- warning/NaN/penetration gates.

Select by the registered priority among eligible candidates. Freeze the
intermediate selection artifact, materialize the dependent manifest from the
actual winner, then newly run:

- Gamma=0 bounded parity;
- three-process driver replay;
- three-process isolator replay;
- three-process selected-contact replay;
- three-process Gamma=0 parity replay.

Record the 15 cm drop and incline results for the selected profile as
non-blocking diagnostics with explicit labels. They cannot change candidate
eligibility or ordering.

## 6. Publish an authenticated official profile

The final verifier must recompute all hard gates, candidate eligibility,
minimal-intervention ordering, intermediate selection hash, dependent-manifest
binding, raw hashes, replay equality and final profile alignment. It must prove
that every dependent artifact uses the actual winner.

On PASS only:

1. archive the stale V1 official-profile bytes under a flat invalidated asset
   if they are not already preserved outside Git history;
2. atomically write `shakebench_official_physics.yaml` with a new, distinct
   profile ID and embedded canonical hash;
3. write `shakebench_phase_06_final_status.json` with `status: PASS`;
4. write `shakebench_phase_06_to_07_handoff.json`, binding protocol,
   registration commit, selected profile, all evidence and package hashes;
5. make the official loader accept only this packaged profile when the final
   status and handoff both verify.

External paths/mappings remain explicitly non-scoreable. Historical V1–V8
status files cannot activate the loader.

On any failure, emit consistent BLOCKED final status/handoff, publish no
profile and stop. Never manufacture PASS to satisfy this prompt.

## 7. Phase 07 handoff gate

Update `phase_07_oracle_controller.md` so its first executable step calls a
single verifier for `shakebench_phase_06_to_07_handoff.json`. Phase 07 must
refuse to start if the handoff, official profile, package asset, protocol or
evidence hashes disagree.

Run and record:

- Phase 06F focused and negative tests;
- affected Phase 02–05 driver/isolator/environment/provider/sensor tests;
- clean source sdist and wheel installation, profile load and MJCF compile;
- three fresh-process final verifier calls;
- full `python -m pytest`, with renderer/EGL infrastructure limitations
  reported separately rather than hidden.

Track all code, tests, documents and authoritative evidence. `git diff
--check` must pass and a clean checkout must reproduce the final verifier.

Phase 06F is complete, and Phase 07 may start, only when all of these are true:

```text
final status = PASS
exactly one official profile loads
exactly one eligible contact is selected by registered priority
task-native settling and force/slip gates pass
dependent manifest and every replay bind the actual winner
driver/isolator/contact/parity evidence hashes verify
clean installed package reproduces the same profile and handoff hashes
no task success, controller outcome or tier ranking influenced physics
```
