# Phase 06R3 Prompt: V4 Official Physics Freeze

Work in `/home/miracle04/Desktop/ShakeBench`. Preserve the scientific record:

- V1 is invalidated.
- V2 is BLOCKED by repeated external termination.
- V3 is BLOCKED because its immutable `dt_medium=0.00015 s` cannot divide
  the frozen 20 Hz control period.

Create a new **V4** protocol and selection. Do not edit or delete V1/V2/V3
protocols, statuses, reports, or raw artifacts. Phase 07 remains forbidden
until V4 has independently produced a verified official physics profile.

Read first:

- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/shakebench_prompts/phase_06_review_remediation.md`
- `docs/shakebench_prompts/phase_06r2_feasible_physics_freeze.md`
- `docs/robosuite_benchmark_design_tree.md` (physics and failure-integrity gates)
- `docs/phase_06_report.md`
- `robosuite/models/assets/shakebench_selection_protocol_v3.yaml`
- `robosuite/models/assets/shakebench_phase_06r2_status.json`
- `robosuite/scripts/shakebench_select_physics_v3.py`
- `robosuite/utils/shakebench_driver_measurement.py`
- `robosuite/utils/shakebench_physics.py`
- `tests/test_shakebench_physics_profile.py`

## 1. Archive V3 and restore a green BLOCKED baseline

Treat V3's scheduler rejection as
`invalid_pre_registered_configuration`, not an infrastructure crash. Preserve
its two-attempt ledger verbatim, but update report language and any derived
summary without rewriting raw evidence.

Before registering V4:

1. Commit the current V1/V2/V3 code, reports, statuses and raw evidence needed
   to reproduce each historical conclusion. Do not commit temporary files or
   partial artifacts that lack declared provenance.
2. Replace stale V1-PASS assumptions in
   `tests/test_shakebench_physics_profile.py` with state-isolated tests:
   archived/BLOCKED tests must use temporary status/profile fixtures and prove
   fail-closed behavior; official-profile tests must use a self-contained
   temporary PASS fixture until V4 is published.
3. Add focused tests for the V3 protocol hash, bounded 200 Hz measurement,
   capacity evidence, scheduler rejection classification and unchanged
   V1/V2/V3 artifacts.
4. Run the Phase 06 focused tests. This checkpoint is complete only when they
   pass while the packaged benchmark is still BLOCKED.

## 2. Complete the V4 machinery before registering candidates

Implement a new flat runner
`robosuite/scripts/shakebench_select_physics_v4.py` and a read-only V4
verifier. Complete and test all stages before executing any V4 physics probe:

- driver/timestep/weld;
- isolator analytic + MuJoCo transfer and combined spectrum;
- contact raw-physics probes;
- Gamma=0 bounded parity;
- three-independent-process replay;
- official-profile publication.

The runner must support `--validate-protocol`, `--dry-run-manifest`, staged
resume and final `--verify`. A dry run lists every state ID, dependency,
expected duration, timestep count, retained-sample count, output filename and
retry policy without running MuJoCo.

Use one shared canonical JSON/hash/atomic-write utility for V2/V3/V4/replay
where it can be introduced without changing archived hashes. Archived
verifiers must retain their original canonicalization rules explicitly.

The verifier must recompute gates, eligibility, normalized scores, secondary
ordering, tie-breaks, raw hashes, protocol hash and official-profile alignment
from raw values. A `passed` Boolean, self-hash or report sentence is never
sufficient evidence.

Checkpoint: all runner stages are callable against synthetic/temporary
fixtures, all negative verifier tests pass, and no V4 physics artifact exists.

## 3. Structural feasibility gate

Create `robosuite/models/assets/shakebench_selection_protocol_v4.yaml` from a
finite, explicit execution manifest. Keep component candidate tables
independent: driver selection must not be coupled to one isolator or contact
candidate. Every planned run is nevertheless enumerated in the manifest; do
not generate a hidden Cartesian grid in Python.

The protocol validator must fail before registration unless every entry
satisfies all applicable structural constraints:

```text
control_period_s = 1 / 20 = 0.05
control_period_s / physics_timestep_s is an integer within 1e-12
refresh_stride is a positive integer
sample_dt_s = refresh_stride * physics_timestep_s
sample_rate_hz >= 20 * 8.87 Hz
physics_timestep_s <= 1 / (20 * 8.87 Hz)
every positive solref time constant >= 2 * physics_timestep_s
all state IDs, candidate IDs and artifact paths are unique
all referenced component candidates and fixtures exist
all planned durations, step counts and sample counts are finite and positive
```

Use exactly these driver convergence candidates:

| id | dt (s) | control steps | sample stride | sample rate | deck solref |
|---|---:|---:|---:|---:|---|
| `dt_fine` | `0.0001` | 500 | 50 | 200 Hz | `[0.0002, 0.5]` |
| `dt_medium` | `0.000125` | 400 | 40 | 200 Hz | `[0.00025, 0.5]` |
| `dt_nominal` | `0.0002` | 250 | 25 | 200 Hz | `[0.0004, 0.5]` |

Keep `dt_coarse_negative_control=0.0004` labelled as a negative control; it
does not count toward the required three convergence passes. Freeze Euler,
Newton, 100 iterations and `1e-12` tolerance for the three convergence
candidates unless the original Phase 06 protocol already preregisters a
separate solver-sensitivity state.

The V3 capacity artifact may justify V4 runtime because V4's slowest candidate
is the identical `dt_fine` state. Authenticate its file hash and
`selection_input=false`; do not reuse its physics metrics as V4 selection
evidence.

Run the pure structural validator and write
`shakebench_phase_06r3_v4_feasibility.json`, containing the normalized
protocol hash, exact run manifest, integer-ratio proofs and capacity-evidence
hash. Commit the V4 protocol and feasibility artifact together in one
registration commit. After that commit, neither may change.

Checkpoint: re-reading both files from `git show <registration-commit>:<path>`
must byte-match the worktree, and no V4 MuJoCo raw artifact predates the
registration commit.

## 4. Execute the complete physics selection

Use only V4 protocol values, Phase 01–05 frozen physical facts and raw MuJoCo
state. Selection and probes must not load an official profile or any V1/V2/V3
selected result as candidate input. Task success, reward, controller outcome,
State-tier ranking, policy observation and `Gamma_star` are forbidden inputs.

### Driver

For all three convergence candidates, run every safe Gamma
`[0.15, 0.30, 0.50]`, both load cases and the complete frozen six-axis
64-line spectrum. Advance every MuJoCo physics step while retaining the
pre-registered 200 Hz right-limit trace. Record per-line amplitude/phase,
commanded and realized deck Gamma, weld diagnostics, warnings, solver data,
wall time, step/sample counts, schema/dtype/shape and trace digest.

Require all three candidates to pass amplitude `<=1%`, phase `<=1 degree`,
Gamma error `<=1%`, residual, warning, solref and scheduler gates. Select the
largest passing dt, then minimum maximum phase error, then lexicographic ID.

### Isolator

Run every registered isolator candidate through analytic and MuJoCo six-axis
tracking/resonance/isolation, the combined 64-line spectrum, empty/payload
equilibrium, sag/preload/travel, payload offsets and timestep/solver stability.
Apply the protocol's numerical targets and weights. The secondary
`minimum_normalized_margin` ordering is descending.

### Contact

Run every registered contact candidate through actual MuJoCo static support,
incline threshold, slip/deceleration, impact plus at least 0.50 s recovery,
finger load, three-dt convergence, and compiled explicit-pair audit. A
candidate may be excluded only by a recorded protocol gate, not by comparison
with a preferred canonical value.

### Parity and determinism

Produce real Gamma=0 bounded-parity evidence for geometry, action semantics,
passive support, nominal poses, Panda reachability prerequisites and dynamic
residual. Replay driver, selected isolator, selected contact and parity in
three independent processes. Digests cover the full declared trace schema,
including names, shapes, dtypes and values.

## 5. Failure taxonomy

Classify every unsuccessful state before aggregation:

- `invalid_protocol_configuration`: a deterministic structural/configuration
  error; V4 registration should make this unreachable and its occurrence
  blocks the whole protocol without retry.
- `physics_gate_failure`: the simulation completed but measured physics,
  warnings, NaN/divergence or safety metrics failed; exclude that candidate
  according to the protocol and continue only where the protocol permits.
- `infrastructure_failure`: external termination, host OOM, renderer/service
  outage or unavailable execution resource; retry the identical state once.
  A repeated failure makes the group incomplete and BLOCKED.
- `evidence_integrity_failure`: missing/tampered/mismatched raw evidence;
  BLOCKED without substituting a seed, state or candidate.

Do not catch all Python exceptions as infrastructure failures. Persist type,
message, state ID, candidate hash, attempt count and disposition.

## 6. Publication and tests

Only after the independent V4 verifier passes may publication atomically:

1. write the single `shakebench_official_physics.yaml`;
2. write V4 selected/excluded/status artifacts;
3. bind the official profile to the V4 protocol, registration commit, raw
   manifest, parity and determinism hashes;
4. enable `load_official_physics_profile()` for the packaged profile only.

External paths and mappings remain explicitly non-scoreable. Every
score-affecting timestep, scheduler, deck, weld, isolator, geometry/contact
pair and solver field must match the official profile.

Add negative tests for all structural gates and for: fewer than three driver
passes, missing Gamma/load spectrum, unprobed component candidate, inverted
isolator margin order, missing parity/replay group, raw mutation, status/profile
hash mismatch, external scoreable path, and selection reading archived
official data. Tests must not mutate the checkout.

Run and record exact results for:

- Phase 06 focused tests;
- affected Phase 02–05 driver/isolator/environment/provider/sensor tests;
- clean source sdist/wheel asset and MJCF compile smoke;
- full `python -m pytest`, with renderer/EGL limitations reported separately.

Update `docs/phase_06_report.md` with distinct V1/V2/V3/V4 sections and
machine-readable artifact hashes. Track all required code, tests, protocols,
reports and selected raw evidence so a fresh clone reproduces the verdict.

V4 may report `PASS` and authorize Phase 07 only when every original Phase 06
completion condition and every V4 gate above passes. Any missing or failed
condition produces `BLOCKED`; preserve evidence and stop without implementing
Phase 07.
