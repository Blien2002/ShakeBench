# Phase 06F-R2 Prompt: Semantic Evidence Handoff Remediation

Work in `/home/miracle04/Desktop/ShakeBench`. Phase 06F-R execution produced
real-environment traces and a PASS status, but its verifier still trusts
self-reported parity/replay booleans and does not bind every evidence path in
the handoff to the file bytes it claims. Therefore Phase 07 remains BLOCKED.

This is an evidence-verification remediation only. Do not change the official
profile, contact tuple, driver, isolator, excitation, task geometry, task
success thresholds, controller, or any Phase 06F/06F-R historical artifact.
The binding remains exactly:

```text
profile   shakebench.official.physics.v2
driver    dt_nominal
isolator  low_frequency_damped
contact   c3_nominal
```

Read before editing:

- `docs/shakebench_prompts/phase_06f_handoff_remediation.md`
- `docs/phase_06fr_handoff_report.md`
- `docs/robosuite_benchmark_design_tree.md`
- `docs/robosuite_benchmark_v0_spec.md`
- `docs/shakebench_prompts/phase_07_oracle_controller.md`
- `robosuite/models/assets/shakebench_phase_06fr_protocol.yaml`
- `robosuite/models/assets/shakebench_phase_06fr_handoff.json`
- `robosuite/scripts/shakebench_handoff_remediation.py`
- `robosuite/utils/shakebench_physics.py`
- `tests/test_shakebench_handoff_remediation.py`

## 1. Preserve prior evidence and fail closed

Keep every Phase 06F and 06F-R asset byte-for-byte. Create a flat,
pre-registered `shakebench_phase_06fr2_protocol.yaml` that binds their hashes
and specifies only semantic re-verification/trace outputs. Before this
protocol passes, the public official loader and Phase 07 prompt must require
the new 06F-R2 handoff and reject all earlier PASS statuses.

Implement one deep module with this interface:

```text
verify_phase06fr2_handoff(asset_root) -> VerifiedHandoff | VerificationErrors
```

It owns path/hash binding, raw trace decoding, parity semantics, convergence,
replay comparison, inheritance verification and final Phase 07 authorization.
The CLI only parses arguments and calls this module; the official profile
loader calls the same module. Tests cross the same interface with a temporary
asset root.

## 2. Bind every evidence path to immutable bytes

The 06F-R2 handoff must list every evidence file with:

```text
path
file_sha256
payload_sha256
schema_id/schema_version
state or group identity
profile_sha256
protocol_sha256
trace whole digest (when applicable)
```

This includes all six real settling traces, convergence artifact, parity trace,
six replay traces, package evidence, every inherited V6 driver/isolator raw
file, and all referenced Phase 06F files.

`verify_phase06fr2_handoff` must open every listed path, recompute file and
payload hashes, validate identity fields, then verify complete coverage against
the 06F-R2 protocol. Do not merely hash the handoff's embedded description.

Add adversarial tests that mutate a raw trace and recompute only its own
payload hash; the handoff must still fail because its stored file SHA differs.
Also test missing file, swapped state IDs, duplicate path, stale profile hash,
and an evidence list with one omitted settling/replay entry.

## 3. Recompute parity from trace values

Replace `parity_contract_passed` as an authority with a pure
`verify_gamma_zero_parity(trace, metadata, protocol)` computation. It must
verify from raw values, not a summary Boolean:

- profile ID/hash and `dt_nominal`, 20 Hz and seven action channels;
- every normalized, decoded and clipped action equals registered zero action;
- applied control is present with declared shape/dtype and finite values;
- Panda base, worktable, Can and target nominal poses in their registered
  frames are within pre-registered absolute tolerances;
- Gamma=0 deck/table equilibrium/static response is within registered bounds;
- Can named support, relative pose/twist, contact identity and penetration
  satisfy the existing bounds throughout the final settling window;
- evaluator contract is an authenticated source/content hash and its threshold
  mapping is a canonical hash; both match the frozen implementation;
- policy observation keys contain no privileged truth key;
- warnings are zero and state is finite.

Reward/success calls may appear in compatibility metadata, but their outcomes
must not influence any parity decision. Add adversarial tests for a changed
applied action, shifted nominal pose, changed evaluator threshold, removed
support contact, nonzero penetration and forged parity PASS.

## 4. Recompute transient convergence meaningfully

Use the already recorded six real-environment traces on the common 200 Hz
right-limit grid. For each open-table and target-bottom state, compare
`dt_nominal=0.0002` against `dt_fine=0.0001` and `dt_medium=0.000125`.

For every comparison, independently compute:

- timestamp alignment and full field/schema/shape/dtype/finite checks;
- Can pose/twist in worktable frame, table pose/twist/acceleration and
  support-contact sequence;
- penetration trace and final-window force RMS;
- **cumulative normal and tangential impulse** by integrating the complete
  raw contact-force time series with the recorded physical timestep;
- relative metrics with a nonzero physical denominator floor, plus the
  protocol's absolute tolerance.

Do not compare per-step `force * dt` as if it were the full impulse. Require
both finer-reference comparisons to satisfy the registered transient metric
set. Preserve existing trace artifacts; write a new 06F-R2 convergence result
derived from them. Test that equal mean support force with altered middle
velocity/penetration, altered integrated impulse, or altered final-window RMS
fails.

## 5. Re-verify replay as complete trace evidence

For parity and contact replay, decode every one of the three stored trace
payloads and apply the same trace validator used for settling. Verify:

- exact sample count, timestamps, required fields, dtypes, shapes and finite
  values;
- profile, protocol, initial state, seed, zero action and group binding;
- whole-trace and per-field digests recomputed from values;
- identical full trace/metric digests across process indices `{1,2,3}`;
- `complete_trace=true` only as a derived conclusion after those checks.

The replay artifact must not be accepted because three summary digests match.
Add adversarial tests for missing middle sample, changed dtype/shape, a changed
intermediate value with the same terminal summary, forged `complete_trace`,
and three equally mutated traces.

## 6. Anchor the remediation handoff for Phase 7 development

Use this commit sequence:

1. Commit verifier implementation and tests, without 06F-R2 protocol/evidence.
2. Commit the immutable 06F-R2 protocol and no-physics feasibility artifact.
3. Generate semantic results and the 06F-R2 status/handoff, then commit them.
4. Commit one flat `shakebench_phase_06fr2_anchor.json` that records the exact
   evidence commit SHA plus file hashes of protocol, status, handoff and
   official profile.

In a Git checkout, the Phase 7 verifier requires the anchor's evidence commit
to be an ancestor and verifies the listed Git blob hashes. In an installed
wheel/sdist, it verifies the same asset hashes and relies on the externally
published package SHA; record that distribution SHA in package evidence for
Phase 10. Do not claim a self-hash alone protects against coordinated edits.

## 7. Final authorization

Add `shakebench_phase_06fr2_status.json` and
`shakebench_phase_06fr2_handoff.json`. The public loader and the first Phase
07 command must require:

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
```

Update the Phase 07 prompt accordingly. Historical Phase 06F/06F-R PASS files
cannot authorize startup. Keep the physics tuple byte/value-identical and
assert it against the original official profile.

Run and record:

- all 06F-R2 focused and adversarial tests;
- existing 06F/06F-R/profile compatibility tests;
- affected Phase 02–05 tests;
- three fresh-process semantic verifier calls;
- clean wheel and sdist install, profile load, MJCF compile and semantic
  handoff verification;
- full non-renderer suite and separately marked renderer lane.

Track all prompts and docs, keep the worktree clean, and make Python support
metadata truthful. Phase 07 may start only when the semantic verifier returns
PASS from a clean checkout and a clean installed package, every listed raw
file is byte-verified, and all parity/convergence/replay decisions are
recomputed from complete trace values.
