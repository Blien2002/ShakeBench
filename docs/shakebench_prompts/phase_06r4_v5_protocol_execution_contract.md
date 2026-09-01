# Phase 06R4 Prompt: V5 Protocol–Runner Contract Repair

Work in `/home/miracle04/Desktop/ShakeBench`. V1 is invalidated; V2 and V3
are BLOCKED; V4 is BLOCKED before MuJoCo integration because its immutable
manifest and its runner disagreed about where `sample_rate_hz` and
`refresh_stride` live. Preserve all their commits, raw files and statuses.

V5 repairs that software contract and then performs the complete Phase 06
physics-only freeze. It does **not** change the physics question: retain V4's
physical candidate values, Gamma/load matrix, 64-line excitation, thresholds,
selection orders and failure rules. Do not read task success, reward,
controller outcome, State tiers, policy observations or `Gamma_star`. Do not
begin Phase 07 unless V5 finishes with verified PASS.

Read before editing:

- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/shakebench_prompts/phase_06r3_v4_physics_freeze.md`
- `docs/robosuite_benchmark_design_tree.md` (physics and failure integrity)
- `docs/phase_06_report.md`
- `robosuite/models/assets/shakebench_selection_protocol_v4.yaml`
- `robosuite/models/assets/shakebench_phase_06r3_v4_status.json`
- `robosuite/scripts/shakebench_select_physics_v4.py`
- `robosuite/utils/shakebench_driver_measurement.py`
- `robosuite/utils/shakebench_physics.py`
- `tests/test_shakebench_physics_profile.py`

## 1. Keep V4 blocked and create one configuration authority

Mark V4's failure as `invalid_protocol_configuration`, preserve its raw
two-attempt `KeyError` ledger, and do not reinterpret it as a physics failure.

Implement one immutable resolved-state type, for example
`ResolvedPhysicsProbeState`. It is the only object a V5 probe may consume. A
single pure function, for example
`resolve_protocol_state(protocol, state_id, selection_context)`, must:

- resolve a manifest state reference to its driver/isolator/contact candidate;
- attach every runtime field used by the runner, including physics timestep,
  control steps, deck solref/solimp, measurement rate, refresh stride, sample
  interval, duration, output path, retry policy and trace schema;
- validate types, identities, arithmetic invariants and forbidden inputs;
- return an immutable resolved state or a named validation error.

Neither dry-run nor real execution may index raw YAML mappings for these
fields. Both must consume this same resolved-state object. The manifest is a
reference plan; candidate tables are the sole authority for candidate-owned
parameters. Do not copy `sample_rate_hz`, `refresh_stride`, timestep or solver
values into each manifest row.

## 2. Establish a green V4-BLOCKED baseline

Before creating a V5 protocol:

1. Ensure V1–V4 reports, raw artifacts, statuses, code and tests are tracked
   and reproduce their published result from a fresh checkout.
2. Repair Phase 06 tests so the packaged V4-BLOCKED state is a passing,
   fail-closed baseline rather than five stale V1-PASS failures.
3. Add regression tests that construct temporary protocols and prove:
   - missing candidate-owned measurement fields fail validation before any
     probe call;
   - a manifest row cannot override candidate-owned fields;
   - a valid `dt_fine`, `dt_medium=0.000125`, `dt_nominal=0.0002` matrix
     resolves all states and all values are identical for dry-run/execution;
   - V4's precise missing-field shape is rejected by V5 validation;
   - no resolver/probe path loads an official profile.
4. Run the focused Phase 06 suite while active status remains BLOCKED. It must
   be green before V5 registration.

## 3. Implement and test V5 runner before registration

Create `robosuite/scripts/shakebench_select_physics_v5.py` and a read-only
V5 verifier. They may reuse tested V4 mechanics, but all configuration access
must go through `resolve_protocol_state`.

Support:

```text
--protocol PATH
--validate-protocol
--dry-run-manifest
--stage {driver,isolator,contact,parity,replay,all}
--verify PATH
```

`--dry-run-manifest` must traverse **every** planned V5 state through the
same resolver used by execution, without creating a MuJoCo model. It writes a
canonical list of resolved states and a digest covering every runtime field.
It exits nonzero for any missing/unknown/duplicate/inconsistent field.

Create tests that invoke the CLI in a subprocess on a valid temporary protocol
and a deliberately V4-shaped invalid temporary protocol. The valid run must
emit the exact expected resolved-state count; the invalid run must fail before
any probe function can be called. This is the missing gate that would have
prevented V4.

Finish all selector stages and verifier logic now—driver, isolator, contact,
Gamma=0 parity, three-process replay, profile publication and raw-evidence
recomputation—using temporary/synthetic fixtures. Do not wait to implement a
stage until a prior long physics run succeeds.

Commit this reusable V5 resolver/runner/test code before writing the V5
protocol. That commit must contain no V5 protocol, feasibility artifact or V5
raw physics evidence.

## 4. Register V5 exactly once

Create `robosuite/models/assets/shakebench_selection_protocol_v5.yaml` and
retain the V4 physical values unchanged:

| driver candidate | dt | control steps | refresh stride | sample rate | deck solref |
|---|---:|---:|---:|---:|---|
| `dt_fine` | `0.0001` | 500 | 50 | 200 Hz | `[0.0002, 0.5]` |
| `dt_medium` | `0.000125` | 400 | 40 | 200 Hz | `[0.00025, 0.5]` |
| `dt_nominal` | `0.0002` | 250 | 25 | 200 Hz | `[0.0004, 0.5]` |

Keep `dt_coarse_negative_control=0.0004` explicitly non-scoring. Keep V4's
isolator candidates, contact candidates, safe Gamma set `[0.15, 0.30, 0.50]`,
both load cases, full 64-line spectrum, all gates and ranking rules unchanged.

The V5 protocol validator must require, for every resolved state:

```text
0.05 / timestep is an integer
refresh_stride is a positive integer
sample_dt == refresh_stride * timestep
sample_rate == 1 / sample_dt >= 177.4 Hz
timestep <= 1 / (20 * 8.87 Hz)
positive solref >= 2 * timestep
all IDs and artifact paths are unique
duration / timestep and retained sample count match the resolved plan
```

Use the existing V3 capacity artifact only as non-decisional runtime evidence,
authenticated by file hash. It is valid because V5's longest state is the
same `dt_fine` workload; its measured physics remains outside V5 selection.

Run both V5 CLI commands before registration:

```text
python -m robosuite.scripts.shakebench_select_physics_v5 \
  --protocol robosuite/models/assets/shakebench_selection_protocol_v5.yaml \
  --validate-protocol

python -m robosuite.scripts.shakebench_select_physics_v5 \
  --protocol robosuite/models/assets/shakebench_selection_protocol_v5.yaml \
  --dry-run-manifest
```

Write `shakebench_phase_06r4_v5_feasibility.json` from those exact commands.
It includes protocol bytes hash, the full resolved-state digest, state count,
per-state integer proofs, capacity evidence hash and command exit codes.
Commit **only** the V5 protocol and feasibility artifact as the V5
registration commit. Re-read both using `git show` and byte-compare them to
the worktree before any V5 MuJoCo call.

## 5. Execute, verify and publish

Run every V5 driver state (three dt × three Gamma × two loads × complete
64-line spectrum) using the resolved state. Then run every pre-registered
isolator and contact candidate, parity, and all three-process replay groups.
Keep the original physics gates and ranking. Raw artifacts are flat under
`robosuite/models/assets/` with prefix `shakebench_phase_06r4_v5_`.

Failure taxonomy:

- `invalid_protocol_configuration`: resolver/validation failure; must be
  impossible after registration. It BLOCKS without retry.
- `physics_gate_failure`: a completed measurement fails a registered metric;
  record and exclude only as the protocol permits.
- `infrastructure_failure`: external host/service failure only; retry the
  identical resolved state once, then BLOCK.
- `evidence_integrity_failure`: missing/tampered raw evidence; BLOCK.

The V5 verifier recomputes all gates, eligibility, scores, tie-breaks, raw
hashes, protocol hash, parity and replay completeness from raw artifacts. Add
negative tests for raw mutation, missing matrix element, missing component
probe, fewer than three dt passes, reversed isolator margin ordering, missing
parity/replay group, profile/status mismatch, external scoreable profile, and
any manifest/resolved-state disagreement.

Only if this verifier passes may publication atomically replace the official
profile and set a V5 active status to PASS. Bind that profile to V5 protocol,
registration commit, feasibility digest, selected/excluded artifacts, parity
and determinism hashes. Otherwise write a machine-readable BLOCKED status,
preserve evidence and stop.

Update `docs/phase_06_report.md` with a separate V4-blocked/V5 section, update
the prompt index, run focused plus affected Phase 02–05 tests, clean source
package/MJCF smoke and full `python -m pytest`. Record exact commands and
results. Do not start Phase 07 unless V5 PASS is fully verified.
