# Phase 06R2 Prompt: feasible, immutable Official Physics Freeze

Work in `/home/miracle04/Desktop/ShakeBench`. Phase 06R is correctly
**BLOCKED**, not failed evidence to overwrite: its immutable V2 protocol
required `dt_ultrafine=50 us` and its complete 64-line empty-load driver run
was externally terminated twice. Preserve all V1/V2 files and their hashes.

Your objective is to perform one new, self-contained **Phase 06R2 / V3**
physics-only selection. It may un-block Phase 06 only if every V3 hard gate
passes. It must not use task success, rewards, `_check_success`, controller
outcomes, State-tier ranking, policy observations, or `Gamma_star`; do not
start Phase 07.

Read before editing:

- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/shakebench_prompts/phase_06_review_remediation.md`
- `docs/robosuite_benchmark_design_tree.md` (physics gates and failure integrity)
- `docs/phase_06_report.md`
- `robosuite/models/assets/shakebench_selection_protocol_v2.yaml`
- `robosuite/models/assets/shakebench_phase_06r_status.json`
- `robosuite/scripts/shakebench_select_physics_v2.py`
- `robosuite/utils/shakebench_physics.py`

## Outcome and invariants

V2 remains permanently archived and BLOCKED. Create V3 rather than modifying
V2, the archived V1 profile, or their raw evidence. The environment stays
fail-closed for `scoreable=True` until a V3 `PASS` status exists. A V3 failure
must leave it blocked, preserving its raw evidence and stopping here.

The V3 driver evidence must use the same frozen Phase-01 authored six-axis,
64-line excitation, safe-Gamma set, empty and Panda/base-inclusive
representative-load fixtures, physical definitions, and amplitude/phase/Gamma
gates. Do not shorten the fit window, reduce the spectrum, substitute an
analytic trace, lower tolerances, or skip a load/Gamma/axis to make runtime
easier.

## 1. Make measurement efficient without changing the experiment

First identify why V2 was externally terminated. Add a narrow driver probe
measurement seam if needed, with these properties:

- MuJoCo still advances every physics timestep; no physics step is skipped.
- A pre-registered measurement cadence (at least 20x the 8.87 Hz maximum
  spectral line) records the full named measurement schema. It is a sampled
  measurement trace, not an ambiguous "complete simulation trace".
- Use streaming / online accumulation or bounded chunking for fit data and
  trace digesting; do not retain a Python object for every integration step.
- The line fitter must use timestamps of retained samples, and a synthetic
  regression must demonstrate <0.1% amplitude and <0.1 degree phase error at
  every authored line after the chosen cadence/decimation.
- Record wall time, peak retained sample count, cadence, sampling phase rule,
  MuJoCo step count, and trace schema in each raw artifact. These are audit
  metadata, not selection metrics.

Before V3 registration, run only a non-decisional engineering capacity check
on the final probe code: simulate the exact V3 longest duration at `dt=0.0001`
with the fixed program and measurement cadence. It may establish that the
probe completes under a declared local runtime ceiling; it must not supply
selection metrics or choose a candidate. Record its command, elapsed time,
machine-independent step/sample counts, and result in a separate
`shakebench_phase_06r2_capacity_preflight.json`. If it cannot complete, keep
Phase 06 blocked and report the concrete bottleneck rather than changing the
science contract.

## 2. V3 pre-registration

Create flat asset `robosuite/models/assets/shakebench_selection_protocol_v3.yaml`.
It is the single source of truth and must be committed *alone* before any V3
physics probe. Record that commit SHA and the normalized file hash inside the
protocol and later artifacts. Do not edit it after running a V3 probe.

V3 must contain a finite composed candidate table (not a Cartesian grid hidden
in Python) and all hard gates, scoring weights, targets, tie-breaks, windows,
cadence, trace schema, retry/crash ledger and artifact filenames. Its driver
convergence table must contain exactly these three complete candidates:

| id | dt (s) | integrator | solver | iterations | tolerance | deck solref |
|---|---:|---|---|---:|---:|---|
| `dt_fine` | `0.0001` | Euler | Newton | 100 | `1e-12` | `[0.0002, 0.5]` |
| `dt_medium` | `0.00015` | Euler | Newton | 100 | `1e-12` | `[0.0003, 0.5]` |
| `dt_nominal` | `0.0002` | Euler | Newton | 100 | `1e-12` | `[0.0004, 0.5]` |

Keep `dt_coarse_negative_control=0.0004` as a labelled negative control only;
it never counts toward the required three passes. Each candidate must meet
`dt <= 1/(20 f_max)` and positive solref >= 2dt. Select by the protocol's
pre-registered ordering: largest dt among complete hard-gate passes, then
smallest maximum absolute line phase error, then lexicographic id.

All V3 candidate profiles must be constructed from V3 protocol + upstream
frozen facts only. Neither selection nor probe execution may load the package
official profile, V1/V2 selection output, or official status as input.

## 3. Complete V3 selection

Implement `robosuite/scripts/shakebench_select_physics_v3.py` as a new,
flat script. Reuse well-tested helpers only where doing so does not import an
official profile. It must:

1. Run each of the three dt candidates for every safe Gamma, both load cases,
   and the entire authored spectrum; record per-line amplitude/phase,
   commanded/deck Gamma, raw weld diagnostics, warnings, solver diagnostics,
   sampled trace digest, and raw file hash.
2. Run every pre-registered isolator candidate through analytic and MuJoCo
   six-axis tracking/resonance/isolation, combined 64-line spectrum, empty and
   payload equilibrium, travel/sag/preload/payload sensitivity and stability.
   Recompute V3 protocol score and descending normalized-margin tie-break from
   raw numbers.
3. Run every non-preflight-excluded contact candidate in raw MuJoCo static
   support, actual incline threshold, slip/deceleration, >=0.50 s impact
   recovery, finger-load, 3-dt convergence and compiled pair-scope audits.
4. Produce an actual Gamma=0 bounded-parity artifact.
5. Run three independent-process replays for driver, selected isolator,
   selected contact, and parity. Each digest covers the full declared sampled
   trace schema, including field names, dtypes, shapes and values.

The pre-registered repeated-crash policy remains: one identical retry;
repeated infrastructure termination makes that group incomplete and V3
BLOCKED. Never replace a failed state, seed or candidate. Persist the retry
ledger in raw JSON.

## 4. Integrity and official profile publication

Create a V3 verifier that recomputes hard gates, eligibility, scores,
tie-breaks, raw hashes and protocol hash directly from raw artifacts. It must
not trust a `passed` Boolean, self-hash, or report prose. Negative tests must
prove rejection of: changed raw artifact, missing load-spectrum matrix,
unprobed contact candidate, fewer than three dt passes, inverted margin
tie-break, missing parity, missing replay group, external `scoreable` profile,
and V2 profile/status used as selection input.

Only after all V3 gates pass may the selector atomically replace
`shakebench_official_physics.yaml` with the one V3 profile, change the active
Phase-06 status to `PASS`, and make `load_official_physics_profile()` accept
it. The profile must authenticate V3 protocol and evidence hashes. External
paths/mappings always remain explicit non-scoreable probes. If any gate fails,
do not alter the active official YAML or status.

## 5. Evidence, tests and handoff

Keep assets flat under `robosuite/models/assets/`, prefixed
`shakebench_phase_06r2_`; do not create a directory. Update:

- `docs/phase_06_report.md`, clearly separating V1 invalidated, V2 blocked,
  and V3 result;
- `tests/test_shakebench_physics_profile.py` plus focused driver/isolator/env
  tests;
- `docs/shakebench_prompts/README.md` Phase 06 index.

Run the Phase 06 focused suite, affected Phase 02–05 tests, package
sdist/MJCF smoke, and report exact commands/results. Treat EGL separately.
Report V3 as PASS only when all ten Phase-06 completion conditions from the
original prompt hold. Otherwise state BLOCKED with a machine-readable reason
and stop; do not implement Phase 07.
