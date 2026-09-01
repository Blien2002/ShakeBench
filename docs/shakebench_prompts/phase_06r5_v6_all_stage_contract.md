# Phase 06R5 Prompt: V6 All-Stage Resolved-State Contract

Work in `/home/miracle04/Desktop/ShakeBench`. Preserve the record: V1 is
invalidated, V2/V3/V4 are BLOCKED, and V5 is BLOCKED after a fully passing
18-state driver matrix because the resolved state did not materialize the
isolator and contact parameters required by their adapters. V5 driver raw
evidence is valuable archive/provenance but is not V6 selection evidence.

V6 changes only the protocol-to-runner contract. Keep V5's physical
candidates, 64-line excitation, Gamma/load matrix, hard gates, scoring and
selection ordering unchanged. Do not use task success, reward, controller
outcome, State tiers, policy observations or `Gamma_star`. Phase 07 remains
forbidden until a V6 verifier authorizes the official profile.

Read first:

- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/shakebench_prompts/phase_06r4_v5_protocol_execution_contract.md`
- `docs/robosuite_benchmark_design_tree.md` (physics/failure integrity)
- `docs/phase_06_report.md`
- `robosuite/models/assets/shakebench_selection_protocol_v5.yaml`
- `robosuite/models/assets/shakebench_phase_06r4_v5_status.json`
- `robosuite/scripts/shakebench_select_physics_v5.py`
- `robosuite/utils/shakebench_physics.py`
- `tests/test_shakebench_physics_profile.py`

## 1. Archive V5 and make the blocked baseline green

V5's stop is `invalid_protocol_configuration`, not a physics failure. Preserve
its 18 passed driver artifacts and its BLOCKED status without editing their
bytes. Update `docs/phase_06_report.md` to say explicitly that no V5
isolator/contact/parity/replay evidence exists.

Before a V6 protocol exists, ensure a fresh checkout has all V1–V5 source,
tests, reports, status and raw artifacts required to reproduce the historical
verdicts. The focused Phase 06 suite must pass while the packaged benchmark is
fail-closed; no test may still assume a historical V1 profile is active.

## 2. One resolved contract for every stage

Replace the driver-shaped resolved state with an immutable tagged union or
immutable nested model that can materialize every stage's real input:

```text
ResolvedProbeState
  common: state_id, stage, output, retry policy, protocol identity,
          timestep/control/measurement/trace fields
  driver: full driver/deck/weld/Gamma/load/program fields
  isolator: candidate ID, fn_hz, zeta, derived k/c/springref,
            transfer regions, combined spectrum, payload/equilibrium fields
  contact: candidate ID, condim, sliding/torsional/rolling friction,
           margin/gap, solref/solimp, iterations, interfaces and all probes
  parity: static/dynamic duration, geometry/action/support tolerances
  replay: selected-component binding, process index and complete trace schema
```

Only the resolver may read the YAML protocol. It deep-copies and freezes all
resolved numeric arrays/mappings. Every real stage adapter receives its
`ResolvedProbeState` (or stage-specific immutable child) and has no protocol
mapping/path parameter. This is an API boundary: reading YAML in an adapter is
a test failure, not an acceptable convenience.

Derive isolator `k`, `c` and preload from the frozen physical definitions in
the resolver, then compare them to declared/expected values before returning.
Resolve selected-component replay states only after the upstream selections
exist; dry-run must enumerate their legal binding templates for every possible
selected candidate.

## 3. Adapter-contract preflight before V6 registration

Implement `robosuite/scripts/shakebench_select_physics_v6.py`, with a pure
resolver, read-only verifier and these commands:

```text
--validate-protocol
--dry-run-manifest
--adapter-contract
--stage {driver,isolator,contact,parity,replay,all}
--verify PATH
```

`--adapter-contract` is stricter than a schema check. For every planned state
and every legal replay selection binding, it must invoke the **actual** driver,
isolator, contact, parity and replay adapter preparation path using a
`NoPhysicsBackend` that raises if MuJoCo model creation, stepping or an
environment/task API is attempted. The command succeeds only when each adapter
accepts its resolved state, requests no undeclared field and produces a
complete probe plan.

The test suite must exercise this same command in a subprocess. Include
negative tests where a protocol omits each of: driver measurement fields,
isolator `fn_hz`, isolator `zeta`, contact `condim`, contact `solref`, contact
`solimp`, parity tolerance and replay binding. Each must fail before any
physics call. Include a sentinel/mocked protocol object proving adapters never
consult YAML after resolution.

Complete all stage implementations and raw-evidence verifier against
temporary/synthetic inputs before V6 registration. The V6 verifier recomputes
gates, eligibility, score/tie-break, raw/protocol hashes, parity and replay
completeness; it never trusts a result Boolean or report prose.

Commit this code and its tests before creating V6 protocol/assets. It contains
no V6 protocol, feasibility file or V6 physical raw evidence.

## 4. Register V6 without changing physics

Create `robosuite/models/assets/shakebench_selection_protocol_v6.yaml` and
copy V5's physical candidate values exactly, including:

| candidate | dt | control steps | stride | sample rate | deck solref |
|---|---:|---:|---:|---:|---|
| `dt_fine` | `0.0001` | 500 | 50 | 200 Hz | `[0.0002, 0.5]` |
| `dt_medium` | `0.000125` | 400 | 40 | 200 Hz | `[0.00025, 0.5]` |
| `dt_nominal` | `0.0002` | 250 | 25 | 200 Hz | `[0.0004, 0.5]` |

Keep `dt_coarse_negative_control=0.0004` non-scoring; keep the V5 three
isolator candidates, three contact candidates, `[0.15, 0.30, 0.50]` safe
Gamma set, two load cases and all original numerical gates. Candidate-owned
fields appear only in candidate tables. State rows reference IDs; no state row
may shadow candidate-owned values.

Run all three pure commands against V6 protocol before registration:

```text
python -m robosuite.scripts.shakebench_select_physics_v6 --protocol PATH --validate-protocol
python -m robosuite.scripts.shakebench_select_physics_v6 --protocol PATH --dry-run-manifest
python -m robosuite.scripts.shakebench_select_physics_v6 --protocol PATH --adapter-contract
```

Write `shakebench_phase_06r5_v6_feasibility.json` from these exact runs. It
must contain protocol SHA, full resolved-state digest, adapter-contract digest,
state/binding counts, per-state arithmetic proofs, V3 capacity-artifact hash
and all exit codes. Commit only V6 protocol plus feasibility artifact as the
registration commit. Byte-compare both with `git show` before any V6 MuJoCo
call.

## 5. Execute all physics gates and publish only on verification

Re-run the complete V6 driver matrix; do not relabel V5 driver raw files with
the V6 protocol hash. Then run every isolator candidate, every contact
candidate, real Gamma=0 parity and all three-process replay groups. Persist
flat V6 artifacts with prefix `shakebench_phase_06r5_v6_`.

Classify failures exactly:

- invalid protocol configuration: impossible after V6 adapter-contract; BLOCK
  without retry;
- completed physics gate failure: record/exclude only as V6 protocol allows;
- infrastructure failure: retry the identical resolved state once, then BLOCK;
- evidence-integrity failure: BLOCK without substitution.

Add negative verifier tests for missing matrix cells, missing component probes,
fewer than three driver passes, reversed isolator margin ordering, missing
parity/replay, raw mutation, status/profile hash mismatch, external scoreable
profile and every manifest/resolved-state mismatch.

Only if the independent V6 verifier passes may publication atomically write
the official profile and V6 PASS status, binding them to V6 protocol,
registration commit, feasibility and adapter-contract digests, raw manifest,
selection, parity and determinism evidence. Otherwise preserve a
machine-readable BLOCKED status and stop.

Update report/index, run focused and affected Phase 02–05 tests, clean source
sdist/wheel/MJCF smoke and full `python -m pytest`; record exact results.
Track all resulting code, docs, tests and selected evidence. Phase 07 starts
only after V6 PASS.
