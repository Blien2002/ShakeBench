# Phase 07R5 current conclusion: Oracle Motion Semantics closure

Status: **BLOCKED_BY_PHASE_07R5_FINAL_COMMIT_HANDOFF**

R4 entry is independently verified by `docs/phase_07_r4_manifest.json` and the
new R4 entry replay at `out/phase07r5/r4_entry_replay.json`.  The compact R4
manifest pointer was corrected to the actual determinism-file byte hash; the
R4 raw payloads were not changed.  R4 state-002 is an environment success,
V0/Gamma=0 is 10/10, and the preserved semantic/actuator/determinism/package
evidence passes its entry verifier.

R5 implementation is complete for the motion-semantics and evidence core:

- `TaskPhase` has one explicit `MotionCapability` mapping.  Contact limits are
  applied after vibration compensation and before normalized action output;
  `VERTICAL_DESCEND`/`GRASP_CLOSE`, capability, pre-limit and post-limit
  values are recorded in every R5 trace.
- `PublicRelativeKinematicsTracker` computes `T_worktable_can` and
  `T_target_can` from public poses, differences relative translation before
  finite differencing, uses shortest rotation vectors, is q/-q invariant, and
  resets on repeated/backward timestamps and episode boundaries.
- `RelativeSupportMotionEstimate` v1 gives V0--V3 one semantic quantity:
  worktable/target relative to robot base at `target_origin`, with explicit
  frame, units, pose/twist/acceleration, timestamps, prediction grid,
  validity/confidence and provenance.  The shared law never branches on the
  tier string.  V2 converts public table state through the compiled deck-to-
  robot-base transform and target-origin lever arm.
- V3 preview uses only public authored lines, reconstructs quintic ramp first
  and second derivative terms, applies the frozen six-axis transfer and frame
  rotation, and initializes its short trajectory from current public state.
  Preview failure falls back to V2-equivalent current compensation.
- Oracle run schema is v4; the verifier replays tracker, estimate, law,
  capability, action and semantic trace fields, and authenticates state/post-
  state and actuator payload digests.  Explicit diagnostic profiles are
  hashed and separated from the main reference profile.

The final R5 profile is `shakebench.reference_oracle.v3`, profile SHA-256
`60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c`, with
task-context SHA-256
`04b9a8d354fad6d27fc6d54024f9faf332c406f0399ce49fe270b55446594155`.
Frozen physics remains `shakebench.official.physics.v2` with SHA-256
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.

## R5 final evidence

The V0/Gamma=0 frozen dev matrix is **10/10 environment successes** with
2131 trace rows.  State-002 is a success with 207 policy samples.  Matched
V0/V1/V2/V3 Gamma=0 dev-000 runs are all environment successes and all four
independent semantic verifications pass.

Positive-Gamma diagnostics are complete and are not ranking gates:

| commanded Gamma | V0 | V1 | V2 | V3 |
|---:|---:|---:|---:|---:|
| 0.15 | 3/3 | 3/3 | 3/3 | 3/3 |
| 0.30 | 3/3 | 1/3 | 3/3 | 3/3 |

The two V1/Gamma=0.30 failures are retained as
`public_object_unrecoverable`; no failure was removed from a denominator.
Fixed-subset diagnostic ablations for compensation-off, common public-history,
V3 preview-off and V3 preview-on are complete, each with three attempts and
independent semantic verification.

Final authority verifiers are all green: determinism is 3/3 fresh processes
with common trace SHA-256
`cb42c58a9cfadcb1822464f5173d54866292982e428cb19cc09bc5108fea531d`,
actuator controls are 14/14, and clean wheel/sdist installs both pass the
R5 artifact and environment smoke check.  The compact machine-readable index
is [phase_07_r5_manifest.json](/home/miracle04/Desktop/ShakeBench/docs/phase_07_r5_manifest.json:1).

## R5 gate and handoff boundary

The remaining blocker is repository handoff, not controller execution:
`final_commit` in the compact manifest is intentionally null because the
worktree contains pre-existing uncommitted Phase 05--R4 changes alongside
this R5 work.  Until those changes are explicitly committed as the final
phase handoff, Phase 07 remains blocked and Phase 08 is not authorized.
No knee or official state was read, generated, scanned or evaluated.

The authoritative contract is:

```text
public goal/worktable pose → T_worktable_can / T_target_can tracker
                           → typed relative-support estimate
                           → shared tier-invariant law
                           → capability limit
                           → normalized 6D OSC_POSE + gripper action
                           → independent environment evaluator
```

The earlier R2--R4 sections below are retained as disclosed provenance and
historical evidence; their artifacts are not final R5 authority.

# Phase 07: task semantics and shared Oracle controller

Status: **BLOCKED_BY_PHASE_07R4_EVIDENCE_REGENERATION**

## Phase 07R4 entry and active remediation

R4 retains the R3 candidate hashes and frozen anchors below.  The Phase 06
semantic handoff was rechecked before this work (`passed=true`), and the
package-owned dev asset remains byte-identical to the `dd6fe2ed` anchor.
The official compiled task context reports worktable size `(0.65, 0.60) m`,
half-extents `(0.325, 0.30) m`, target-frame origin
`(-0.10, 0.17, 0.042) m`, table surface `0.03 m`, and Can collision radius
`0.02509177806572465 m`.

R4 adds a typed immutable `WorktableTaskContext`, explicit target↔worktable
transforms, a frozen worktable-local grasp anchor, the
`CLEARANCE_LIFT -> LATERAL_ALIGN_ABOVE_CAN -> ALIGN_SETTLE ->
VERTICAL_DESCEND -> GRASP_CLOSE` acquisition trajectory, swept fingertip
clearance certificates, actual asymmetric table-envelope edge margins, and
reverse-path recovery that holds the gripper closed until public support and
stability are confirmed.

The final shared profile is `shakebench.reference_oracle.v2`, profile
SHA-256 `bc6a049ecde1e9d4b5eb50f0e0b1df20d26027f7c852663ca7fc3827e24074c8`.
Its 10 frozen V0/Gamma=0 episodes all passed environment success and the
independent semantic verifier (2946 trace rows in
`out/phase07r4/v0_gamma_000_final.json`; payload SHA-256
`802b56b89c8c3cc81f57193b5094b70398bbbfa36e73c986be1895c9f7c0049c`).
State-002's final trace has 301 policy samples and includes the bounded
close recovery followed by PRELIFT_VERIFY, LIFT, PLACE, RELEASE, and VERIFY.

Matched V0/V1/V2/V3 Gamma=0 dev-000 runs, actuator controls (14/14), fresh
wheel/sdist package installs, and three-process determinism all independently
pass under this profile.  Gamma=0.15 diagnostics finished with explicit
success/failure reasons for all V0--V3 episodes.  Gamma=0.30 diagnostics were
stopped after 40 minutes without complete raw artifacts; this evidence gap,
along with the pre-existing dirty worktree/final-commit handoff requirement,
keeps the report blocked and no Phase 08 work is authorized.

The state-002 before/after timeline is:

| checkpoint | outcome | trace |
|---|---|---:|
| R3 blocker | `controller_failed/public_object_out_of_workspace` | 117 |
| R4 final | `environment_success` (one close recovery, second grasp through PRELIFT_VERIFY) | 301 |

The R4 final 10-state summary is `10/10` environment successes with trace
lengths `193, 227, 301, 200, 309, 193, 397, 402, 343, 381`.  The compact
binding record is [phase_07_r4_manifest.json](/home/miracle04/Desktop/ShakeBench/docs/phase_07_r4_manifest.json:1).

The frozen controller choices are: 10 mm prelift; 18 mm translation-slip
gate with an immediate 0.75-Can-radius severe-slip gate; Can-diameter-plus-
10-mm aperture upper bound (60.2 mm); 0.35 transport position limit; three
recovery attempts; and a 1.20 s public settle budget.  These are shared
profile parameters, selected only after nominal dev solvability and the
public geometry/noise controls.

Final evidence hashes are indexed in `docs/phase_07_r4_manifest.json`; the
independent verdicts are semantic `passed=true` (2946 rows), actuator 14/14,
determinism 3/3, and package wheel+sdist 2/2.  Gamma=0.15 smoke semantic
verdicts also pass independently (V0/V1/V2/V3 trace counts 979/876/748/931),
while their task outcomes remain diagnostics rather than ranking claims.
The complete R4 red suite passed 33 tests, and the specified affected
non-renderer regression passed 66 tests with 1 renderer test deselected.

Gamma=0.15 diagnostic outcomes were V0 `3/3`, V1 `2/3`, V2 `2/3`, and V3
`3/3`; the two unsuccessful episodes retain
`failure_reason=public_object_unrecoverable`.  Gamma=0.30 was attempted with
the same commands but stopped after the long-running workers produced no
complete artifacts; it is deliberately not promoted to a ranking or nominal
solvability claim.

## Phase 07R3 entry and active remediation

The R2 blocker is retained below as provenance.  R3 began from the same
frozen dev-state anchor `dd6fe2edb6384ccdb5116be44f07592b4864e377`, official
physics profile `shakebench.official.physics.v2`, and physics SHA-256
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.
The Phase 06 semantic handoff was independently rechecked before controller
changes and returned `passed=true`.

The deterministic R3 minimal reproduction was saved as
`out/phase07r3/blocker_before.json`: V0, Gamma=0, frozen dev state 002,
horizon 120, `success=false`, `termination_category=controller_failed`,
`failure_reason=public_object_out_of_workspace`, and a 117-sample trace.  It
is an execution/controller defect rather than an asset, physics, or task
corruption.

R3 replaces the inverted ambiguous hold field with the fixed Panda contract
`gripper_open_action=-1.0` and `gripper_close_action=+1.0`.  It adds public
finger-pad corridor / expected-transform grasp checks, candidate
`PRELIFT_VERIFY`, sub-Can-radius translation-slip confirmation, and explicit
`RECOVERY_HOLD -> RECOVERY_LOWER_IF_NEEDED -> RECOVERY_OPEN -> RETREAT`
transitions.  Wrench is trace-only confidence and cannot establish a hold.

This work is still blocked pending an environment-backed 10/10 V0/Gamma=0
rerun and the consequent matched-tier, actuator, determinism, package, and
semantic-evidence regeneration.  No Phase 08, knee, or official-state work
has been performed.

Phase 07R2 has revoked the completion-remediation PASS authority before any R2
production-code changes. The existing remediation artifacts are diagnostic
provenance only and cannot satisfy an R2 hard gate, Phase 08 handoff, or paper
result. The frozen entry anchors remain `dd6fe2edb6384ccdb5116be44f07592b4864e377`,
`shakebench.official.physics.v2`, and physics hash
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.

R2 entry verification, run before code changes:

```text
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
exit=0; passed=true; evidence_coverage=true
```

The remediation raw payload hashes recorded before R2 work are:

| artifact group | payload SHA-256 |
|---|---|
| `v0_gamma_000_remediation_final.json` | `2a31dd92edf206384f83c2e20ca9cc5d4c18c7bc4b3850d4b72e03c776059a35` |
| `matched_v1/v2/v3_gamma_000_final.json` | `ea0bbf122f413f81d64d431cd546286a879ee54afa28aa47180d928391ebef93`; `07c206c3c77e9c3bd9db0f453e1cc3e689ac1be6df22bb86fcb16a818c8a2a2a`; `e71458063bf8952fc0323f196b9bc53fa01a206751d5e6398575d09ab48885f0` |
| `smoke_v0_gamma_015/030_final.json` | `cf249b3f13d1197a29b583fe717574b01228f15b9130fe613927246c447297b9`; `cd8082af80f5471466dca615c9f9269cfbeaff4a468d6d7c1c48fcee4fb3c254` |
| `smoke_v1_gamma_015/030_final.json` | `7e13612013a5efa12f22faf9fde12ac6ff8f325e17f438288576a01c8653debc`; `a441a29486ffba722df87083f7fe826dc2e9a659a947b3e2aed29a2f770fd614` |
| `smoke_v2_gamma_015/030_final.json` | `58d50c80855645b2bd49c0c9867d038a2d925234b1a251b48a26ed934e7b6820`; `9164a7231f04818542323599bfbb46ed5bbcff9be3e8f29a7bc12324b4dfb9a4` |
| `smoke_v3_gamma_015/030_final.json` | `e7a76ca0b9a846e2f80f392b5fa646ab87a45fac504814e2b1ece2f06d313b46`; `8ec62a105dacdbf3afea3bc0a8c1e9f63a36e35dff345b5ab4b37897fe629b00` |
| `determinism_manifest_final.json` records | `037f4bdde0a4843fe5e1da18084aac225c262d30f435cdcde1ee3537e76fa8b2` (repeated PID/path evidence; invalid) |

The earlier invalidated payload hashes remain listed below as historical
provenance. R2 must produce a new semantic, actuator, replay, and package
evidence set.

The prior Phase 07 PASS has no authority. Its controller semantics and raw
evidence are retained only as diagnostic provenance and must not be used for a
Phase 08 handoff, paper result, or any Phase 07 hard gate.

Invalidated rollout payload hashes:

- `8299882b61f8dcd9eb296898c63ed6d172f297a2173d1176ce9b8d0fafe50539`
  (old V0 Gamma=0 dev run);
- `2d76da79e931e75839aade6078a2e015e8d745e139a95076c282686b041bf8b3`,
  `06149ce35f2a68044e4bcf0616bedf825bb5cdcf82ed7d4c3f61ba01618d3990`,
  `c2e7187aae02c5b965d294163261c62d4634e23f5418febab140eeaf999cd9b7`
  (old matched V1/V2/V3 Gamma=0 runs);
- all prior `shakebench_phase_07_smoke_v*_gamma_*.json` payloads; and
- `e9224be7b99fa36a6e424994d5816d3e157ab8b29722b9e90c17388ba27ed97a`
  (old three-process replay payload).

Review found six root causes: the target quaternion was converted twice; V2/V3
gravity semantics were wrong; the V3 future program was not used; the action
law/recovery were not fully six-axis or public-signal complete; unsuccessful
episodes could have missing failure integrity; and actuator/determinism
evidence was not independently verifiable. The official physics profile remains
`shakebench.official.physics.v2` with hash
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.

This report records the completed semantic/controller implementation and the
dependency repair. No knee or official state was generated, scanned,
or evaluated; frozen physics, contacts, geometry, and success thresholds were
not changed.

## Entry verification

Start commit: `f4644308e9f9acfa5107ac78cabeab57bf32f38d`.

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
```

Exited `0` with `{"checks":{"evidence_coverage":true},"passed":true}`.
The accepted profile is `shakebench.official.physics.v2`, hash
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.

## Completed hard-gate work

- The public State schema now has exactly one target-frame representation:
  position/quaternion in robot base, target-local half extents/z bounds, and
  orientation mask. It is identical in V0--V3; V0 still has no dedicated
  vibration channel, while V2's increment remains realized support state.
- `ShakeBenchMetrics.success_snapshot()` and the environment's post-physics
  hook evaluate every internal 0.2 ms step. Reports remain 20 Hz, but any
  intermediate violation resets the 0.50 s candidate window.
- `shakebench_oracle.py` implements public-payload `VibrationEstimate`, one
  `SharedVibrationControlLaw`, `TaskExecutive`, a profile hash, and normalized
  seven-channel OSC_POSE + Panda gripper actions. It has no `env.sim` input.
- `shakebench_run_oracle.py` records task state, provider payload, typed
  estimate, law output, normalized/decoded/clipped actions, actual actuator
  controls and realized `actuator_force`. It refuses to create or replace dev
  states.

The instantaneous-substep regression, target-frame transformation/public-vs-
evaluator convention, provider isolation, controller profile/hash and action
validity are in `tests/test_shakebench_oracle.py` and
`tests/test_shakebench_providers.py`.

## Targeted revalidation

Both official-profile Gamma=0 task-native settling probes passed after the
sampling change, with all finite/contact/velocity/penetration/warning gates
true:

```bash
python -m robosuite.scripts.shakebench_handoff_remediation --run-state open_worktable --timestep 0.0002
python -m robosuite.scripts.shakebench_handoff_remediation --run-state target_bottom --timestep 0.0002
```

## Dev-state dependency repair

Git-history and current-tree inspection proved that the referenced dev asset
had never existed: Phase 08 owned all committed-state generation while also
requiring Phase 07 dev success as a prerequisite. The repair moves only the
ten non-scoreable dev states forward. `shakebench_states_dev.json` is now
frozen before any rollout using a SHA-256-to-uniform mapping, root seed
`20260905`, exact IDs, bounds, regeneration verifier, and payload hash. No
rollout outcome participated in state selection. Phase 08 still owns the 400
official and 100 knee states and must reuse, not regenerate, this dev subset.

The runner defaults to the package asset:

```bash
python -m robosuite.scripts.shakebench_run_oracle --tier V0 --gamma 0.00 --output RAW.json
```

The remediation reruns use the single frozen controller profile
`shakebench.reference_oracle.v1`, hash
`308b8570980e335c321517e01e6634f3c2867f319ef6177474adbc84cbb7af98`.

The dependency regression first failed with `FileNotFoundError`, then passed
after the repair. The controlled asset verifies schema, dev-only scope, count,
hash, and exact regeneration; clean wheel and sdist checks load the same
package-owned controller/profile and dev-state verifier.

## Execution evidence

Hard gates:

- Gamma=0 State-V0 dev is **10/10**. Raw evidence:
  `docs/phase_07_raw/v0_gamma_000_remediation_final.json`, payload hash
  `2a31dd92edf206384f83c2e20ca9cc5d4c18c7bc4b3850d4b72e03c776059a35`.
- The matched Gamma=0 pipeline on `shakebench-dev-v0-000` succeeds for V0--V3.
  V0 is the matching row in the dev artifact; V1/V2/V3 are recorded in the
  three `matched_v*_gamma_000_final.json` raw assets. Every record
  contains public task state, provider payload, typed estimate, control-law
  output, normalized/decoded/clipped action, applied actuator control, and
  realized actuator force.
- Three independent V0/Gamma=0 processes produced the same complete trace
  SHA-256 `9461413b6ddfada0b1cdc8f0e3a241274243337d53c73ed2b860fcbfdcf6452f`.
  `docs/phase_07_raw/determinism_manifest_final.json` records each process
  identity, file/payload/trace hash, complete flag, and independent verdict.
- Privilege isolation and applied-actuator checks pass. All recorded action and
  force values across the dev/matched/smoke raw artifacts are finite.
- A clean wheel and sdist installation contain `shakebench_oracle.py`,
  `shakebench_dev_states.py`, the dev-state asset, and the read-only run
  verifier. Complete raw traces remain outside runtime package assets under
  `docs/phase_07_raw/`; `MANIFEST.in` preserves the source package contract.

Diagnostics (not ranking gates):

| Gamma commanded | V0 | V1 | V2 | V3 |
|---:|---:|---:|---:|---:|
| 0.15 | 2/3 | 3/3 | 3/3 | 3/3 |
| 0.30 | 2/3 | 2/3 | 2/3 | 0/3 |

These smoke results are retained in the eight
`docs/phase_07_raw/smoke_v*_gamma_*_final.json` assets. They do not drive a tier
ordering claim or any physics/controller change.

The hard gates protect nominal solvability, public-only information flow,
continuous success semantics, reproducibility, and actual actuator evidence.
Their thresholds are the pre-registered task contract (0.50 s, speed,
penetration and support bounds) and the fixed Phase 07 dev matrix. A failure
would block this phase's State Oracle controller evidence, not alter official
physics or official scoring.

## Local verification

```bash
pytest -q tests/test_shakebench_oracle.py tests/test_shakebench_metrics.py
pytest -q tests/test_shakebench_providers.py
pytest -q tests/test_shakebench_privilege.py
```

The remediation red/green suite has 12 tests covering target quaternion,
gravity semantics, future-program causality, failure integrity, six-axis,
recovery, cache-equivalence and artifact mutation. The current Oracle/metrics/
provider suite passes 30 tests. The final affected non-renderer invocation
passed 50 tests with 1 source-distribution test deselected. The handoff verifier
passed again after the Phase 07 changes. Renderer remains outside this
non-renderer work and no renderer claim is made here.

The final combined affected non-renderer invocation passed `50` tests with
`1` source-distribution test deselected:

```bash
pytest -q tests/test_shakebench_oracle.py tests/test_shakebench_metrics.py \
  tests/test_shakebench_providers.py tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
```

The old invalidated traces are isolated in
`out/phase07_invalidated/`; no old summary is used with the final runs.
No knee scan, 100 knee states, 400 official states, or Phase 08 evaluation was
run.

## Phase 07R2 closure audit — BLOCKED

The R2 implementation work is present, but the phase cannot be promoted to
PASS. The sole blocking hard gate is nominal solvability: the current public
controller does not achieve V0 Gamma=0 10/10 on the frozen ten-state dev set.

### correctness fixes

- V1 now owns persistent attitude, gravity-filter state, window time, reset
  generation, and canonical measurement/policy timestamps. Its public latency
  is the frozen 4.87 ms filter delay plus 5 ms delivery delay (`~9.87 ms`),
  not the 5 ms sample interval.
- Can-in-EEF grasp references use the full public SE(3) transform. Rotation
  slip is reported with the shortest rotation vector, while v0 Can yaw remains
  an explicitly diagnostic/unmasked degree of freedom.
- VERIFY uses conservative public Can envelope containment, public finite-
  difference speed/stability, and open-gripper state; COMPLETE is not used as
  an irreversible policy terminal state.
- V3 future feed-forward uses the public isolator relative transfer law and
  explicit frame/unit-separated gains. The shared law remains tier invariant.
- The runner and read-only semantic verifier are schema v3; the actuator
  positive-control runner covers 14 OSC/gripper directions, and the real
  three-child determinism command/verifier exists.

### hard gates

- Phase 06 semantic handoff: PASS; official physics hash remains
  `c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.
- Dev asset remains byte-identical to `dd6fe2ed…`; file SHA-256 is
  `07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6`,
  payload SHA-256 is `2b0924a361182f34f5660a739a027021dfdd335c0b7ba38fc861e2bba589cb9c`.
- Current minimal blocker artifact:
  `out/phase07r2/blocker_v0_gamma0_dev002_current.json`, file SHA-256
  `c64f590212b45744bf07804ad6698c809ac480a64e1c81cfe786e71870d981fb`,
  payload SHA-256 `aaa70685ede36756519237329e0409a5cddcab42f47eb6bdec51468c1c9b36b8`.
  It is independently accepted by `verify_run_artifact()` but records
  `success=false`, `termination_category=controller_failed`,
  `failure_reason=public_object_out_of_workspace` for frozen state
  `shakebench-dev-v0-002` at Gamma=0.
- The observed final public Can position is approximately
  `[0.012, -0.209, -1.297]` m in robot-base. The trace shows bounded public
  recovery attempts before the object leaves the public workspace; no
  evaluator/contact/privileged field is used to trigger this decision.
- Therefore the required V0 Gamma=0 10/10 gate is **FAIL/BLOCKED**. This is a
  controller/task nominal-solvability blocker, not permission to alter frozen
  physics, contact tuples, task geometry, success thresholds, or dev states.

### positive-Gamma diagnostics

The earlier smoke raw files under `out/phase07r2/` are retained as diagnostic
provenance, but profile changes made while closing the nominal blocker mean
they are not final R2 authority. Their observed outcomes are not used for
tier ranking or physics changes.

### adversarial verifier results

The semantic verifier rejects resealed schema, policy-input, action, actuator,
state/profile/physics-binding, trace-order, termination, and determinism
mutations in the focused red suite. The current blocker artifact itself passes
semantic recomputation, demonstrating that the blocker is a real episode
verdict rather than a digest/schema failure.

### package/determinism evidence

An earlier package evidence run passed isolated wheel/sdist imports, dev-state
verification, compact artifact verification, and no-renderer environment smoke.
That evidence is retained under `out/phase07r2/`, but is not a final PASS
because the controller profile changed during the nominal blocker audit. A
fresh package/determinism closure must follow any controller remediation.

### infrastructure limitations

Renderer/EGL was not used. No renderer claim is made; this does not explain the
State-lane nominal failure.

### known limitations and next action

Complete R2 is intentionally stopped here. The next responsible action is to
repair the public recovery/grasp/action seam for the minimal state
`shakebench-dev-v0-002`, then rerun all ten Gamma=0 states before regenerating
matched, smoke, determinism, and package evidence. Phase 08 implementation,
knee states, and official states remain unrun.
