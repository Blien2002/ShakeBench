# Phase 06：Official Physics Profile

> Phase 06R status: **BLOCKED**. This document's former V1 PASS claim is
> invalidated by `shakebench_phase_06_v1_invalidated.json`. The V1 evidence
> below is retained only as archive/provenance; it is not a scoreable freeze
> and must not authorize Phase 07.

Phase 06R protocol `shakebench_selection_protocol_v2.yaml` was committed as
`13c7bd5e` before V2 probes. Its complete first driver state requires a
33.969818100927114 s 64-line fit window. The same `dt_ultrafine` empty-load
state was terminated twice before the trace completed. Per the pre-registered
failure-integrity rule, the driver group is incomplete and selection is
BLOCKED; no replacement candidate, seed, official profile, parity result, or
determinism PASS was produced. The raw crash ledger is
`shakebench_phase_06r_raw_driver_dt_ultrafine.json`.

The package loader now fails closed for the archived V1 official profile
until `shakebench_phase_06r_status.json` is PASS. Phase 06-only IMU semantic
changes were reverted.

## Phase 06R2 / V3 capacity result (PASS; before registration)

R2 preserves V2 unchanged and does not register V3 until a non-decisional
capacity preflight succeeds. The final bounded-measurement seam advances every
MuJoCo `dt=0.0001 s` step but retains the named right-limit trace at `200 Hz`
(`50` physics steps/sample, above the required `177.4 Hz`). Its exact 64-line
run still requires `34.71991810092712 s` / `347200` MuJoCo steps.

The preflight completed in `71.8817107869545 s`, retaining `6950` samples and
advancing `347200` MuJoCo steps, below its declared `120 s` local runtime
ceiling. The machine-readable record is
`shakebench_phase_06r2_capacity_preflight.json`; it is explicitly
non-decisional (`selection_input=false`). The same `200 Hz` line estimator
passes the synthetic 64-line regression by large margin (amplitude error
`2.11e-14`, phase error `3.86e-12°`). V3 may therefore be registered, but
Phase 06 remains **BLOCKED** until every V3 physics gate passes; Phase 07
remains prohibited.

## Phase 06R2 / V3 driver result (BLOCKED)

V3 was pre-registered and committed alone in `2f6cac5e`. All six complete
`dt_fine` Gamma×load×64-line states completed and passed their driver hard
gates using the V3 bounded measurement trace. The next mandatory convergence
candidate, `dt_medium=0.00015 s`, is deterministically invalid for the frozen
`20 Hz` control scheduler: `0.05 / 0.00015` is not an integer. The no-task
fixture rejected the same `Gamma=0.15`, empty-load state twice with:

```text
SimulationError: control timestep 0.05 must be an integer multiple of model timestep 0.00015
```

This is an `invalid_protocol_configuration`, not an infrastructure crash. The
two-attempt ledger in the raw artifact is retained verbatim as historical
evidence; the derived status classification does not rewrite it. Three
complete convergence passes are required, and the immutable V3 protocol
cannot replace or alter this candidate. V3 is therefore **BLOCKED** before
isolator/contact/parity/determinism stages. See
`shakebench_phase_06r2_raw_driver_dt_medium.json` and
`shakebench_phase_06r2_status.json`. No official profile was published; V2
remains unchanged and the active loader remains fail-closed.

## Phase 06R4 / V5 result (BLOCKED)

V5 protocol and resolved-state feasibility were registered as `d53bcccc`.
The complete driver matrix—18 states (`3 dt × 3 Gamma × 2 load`)—finished
with all amplitude, phase, Gamma, weld-residual and warning gates passing.
Raw evidence uses the V5 canonical protocol hash
`59e544893520851935a6496ca9f47b7718f1cd251bf6e5681bd2814c66627078` and is
preserved under `shakebench_phase_06r4_v5_raw_driver_*.json`.

Before the isolator stage, the resolved-state contract exposed a missing
authority: `ResolvedPhysicsProbeState` materialized `candidate_id` but not the
isolator `fn_hz/zeta` or contact tuple needed by later probes. Reading those
values directly from YAML in an adapter would violate the V5 contract. This
deterministic implementation/configuration mismatch is classified as
`invalid_protocol_configuration`; V5 stopped before isolator/contact/parity/
replay and official-profile publication. See
`shakebench_phase_06r4_v5_status.json`. No official profile was published and
Phase 07 remains forbidden.

No V5 isolator, contact, bounded-parity, or replay/determinism evidence
exists. The 18 V5 driver files are retained as archive/provenance only and
are not relabeled or reused as V6 selection evidence.

## Phase 06R5 / V6 result (BLOCKED)

The immutable V6 protocol and feasibility artifact were registered in
`379c412f`. The protocol bytes hash is
`835cabd4b47650ebd7362f127f35dcd0729457963e399bc748947620c972322a`; its
resolved-state digest is
`44f331d6a16880d33a5edda273fb02502f64ef9310ab817fc4c1bc7f994ec7f9`, and
the feasibility artifact hash is
`41c9930474443d56798cfbc4b77675faf49a3e0df503286c69d011600a8ac38a`.
The validate, dry-run, and adapter-contract commands all exited `0`. The
adapter contract covered 37 manifest states plus 66 legal replay bindings
(169 preparation plans) with `NoPhysicsBackend` reporting zero physics calls.

The complete V6 run produced 37 flat raw artifacts: all 18 driver states
passed all driver gates; all three isolator candidates completed and only
`low_frequency_damped` was eligible; parity and all four three-process replay
groups passed independent verification. All three contact candidates
completed, but none passed the registered recovery gates: `c3_nominal` and
`c4_torsional` measured approximately `0.463 m/s` and `11.1 rad/s` recovery
velocity/angle, above the `0.02 m/s` and `0.20 rad/s` limits, while
`c3_softer` also exceeded the penetration limit. This is a completed
`physics_gate_failure`, not an infrastructure failure.

The independent V6 verifier recomputed hashes, state coverage, driver and
component eligibility, selection ordering, bounded parity and replay
completeness; it rejected publication because no contact candidate was
eligible. `shakebench_phase_06r5_v6_status.json` therefore remains
`BLOCKED`, no official profile was published, and Phase 07 remains forbidden.

### V6 validation and regression record

- `--validate-protocol`: exit `0`; `--dry-run-manifest`: exit `0`; and
  `--adapter-contract`: exit `0`.
- Focused Phase 06/V1–V5 archive suite: `36 passed`.
- Affected Phase 02–05 suite: `167 passed`; test-only helpers explicitly use
  the non-scoreable `probe` profile while the production official loader stays
  fail-closed.
- Clean source sdist and wheel builds both included the V6 protocol,
  feasibility, and 37 V6 raw artifacts; direct `shakebench_arena.xml` MJCF
  compilation passed.
- Full non-EGL regression (`python -m pytest -q` excluding the two
  renderer-dependent environment collections): `460 passed, 58 skipped, 2
  failed`. The two failures are `test_camera_transforms` and
  `test_environment_determinism`, both stopped by the container's missing EGL
  device/swrast driver. The exact unfiltered full command was also attempted;
  it is blocked by the same EGL failure, not by a V6 assertion.

## Phase 06R3 / V4 result (BLOCKED)

V4 structural registration was committed as `80712165`; the protocol and
feasibility artifact were byte-checked before any V4 raw output. The first
manifest state, `driver.dt_fine.gamma_0_15.empty`, never reached MuJoCo. The
new runner expected per-state `sample_rate_hz` and `refresh_stride` fields that
the immutable V4 manifest did not declare. The identical state failed twice
with `KeyError: 'sample_rate_hz'`, and the raw ledger is preserved at
`shakebench_phase_06r3_v4_raw_driver_dt_fine_gamma_0_15_empty.json`.

This is classified as `invalid_protocol_configuration`, not a physics result
and not a candidate exclusion. V4 cannot be amended or resumed after this
registration-time defect. The derived status is
`shakebench_phase_06r3_v4_status.json`; no V4 isolator, contact, parity,
determinism, or official-profile publication stage was executed. Phase 07
remains forbidden.

## Archived V1 report (invalidated)

状态：**INVALIDATED（not a PASS）**

本阶段只冻结 environment-owned physics：timestep / integrator / solver /
deck weld / six-axis isolator / explicit Can contact pairs。没有读取
V0–V3 ranking、task success rate 或 controller outcome，也没有修改
controller。完整 selection 的输入规则先写入并提交于 commit
`8d34171a`，之后没有改动 protocol。

## 1. 唯一 official profile

唯一 scoreable physics profile 是
`robosuite/models/assets/shakebench_official_physics.yaml`。

```text
profile_id       = shakebench.official.physics.v1
profile_sha256   = 72dc38077cb706aa1f8d4be9d382321b2183cada114182d2e39f5fcbf6fac414
selection_status = official_immutable
selection_source = physics probes only
```

冻结值：

| component | frozen value |
|---|---|
| timestep | `0.0002 s` |
| integrator / solver | `implicitfast / Newton` |
| solver iterations / tolerance | `100 / 1e-10` |
| deck mass / diagonal inertia | `400 kg / [12.12, 14.083333333333334, 26.033333333333332] kg·m²` |
| deck `eq_solref` | `[0.0004, 0.5]` |
| deck `eq_solimp` | `[0.9, 0.95, 0.001, 0.5, 2.0]` |
| isolator `f_n` | `[4, 4, 4, 3, 3, 2] Hz` |
| isolator `ζ` | `[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]` |
| isolator `k` | `[20212.949813431005, 20212.949813431005, 20212.949813431005, 344.5044633826647, 403.7339333144822, 329.5184560600506]` |
| isolator `c` | `[321.6990877275948, 321.6990877275948, 321.6990877275948, 7.310611768609593, 8.567500157457797, 10.488898224393315]` |
| nominal-table `springref` | `[0, 0, 0.015530637680177088, 0, 0, 0]` |
| contact scope | explicit Can pairs only |
| contact | `condim=3`, torsional `0.005`, rolling `0.0001`, margin/gap `0`, `solref=[0.0004,1]`, canonical `solimp` |
| sliding friction | table/object `0.30`; finger/object `1.00` |

`k/c/springref` are checked against the analytic derivation at profile load
time. The table preload compensates only the 32 kg table; the 0.349 kg Can is
not retuned.

## 2. Selection evidence

The machine-readable selection is
`robosuite/models/assets/shakebench_phase_06_selected_candidates.json` and
its excluded-candidate table is
`robosuite/models/assets/shakebench_phase_06_excluded_candidates.json`.
The selected artifact payload hash is
`e699ed8461a90d10bfef024acdc0953cee613d1b7b14d736ea64ed5974b07d0a`.

### Timestep / solver / weld

All three registered dt candidates were measured with the same six-axis
no-task driver fixture and explicit deck inertial/weld values:

| candidate | dt | max line phase error | max line amplitude error | result |
|---|---:|---:|---:|---|
| `dt_fine` | `0.0001` | `0.334400°` | `2.53e-6` | pass; lower-priority |
| `dt_nominal` | `0.0002` | `0.668804°` | `1.01e-5` | **selected** |
| `dt_conservative_coarse` | `0.0004` | `1.337629°` | `4.05e-5` | excluded: phase gate |

The selected dt satisfies `dt <= 1/(20×8.87 Hz)` and both equality and
contact positive solref time constants satisfy `>=2×dt`. All registered
Gamma values `0, 0.15, 0.30, 0.50, 0.75, 0.95` were checked for both empty and
Panda-plus-reference-load no-task fixtures. Maximum target-Gamma error was
`0.2958%`, maximum realized deck-Gamma error was `0.0621%`; warning count was
zero.

The selected option passed the independent solver sensitivity probe for
official settings, 150 iterations, and the `implicit` integrator. The driver
convergence summary and complete trace digests are included in the selected
artifact.

### Isolator transfer envelope

The selected `low_frequency_damped` candidate passed the registered analytic
envelope:

```text
T_accel mean / median       = 0.915023 / 0.632997
R_relative median / max     = 1.157878 / 2.5
T_peak                      = 2.692582
D_relative                  = 0.0025 m (probe base amplitude 0.001 m)
minimum travel margin       = 0.0225 m
uncompensated static sag    = 0.015530637680177088 m
payload offset max          = 0.002167572243086722 m
payload sensitivity         = 0.0867029
```

The selected candidate also passed six direct MuJoCo axis transfer probes
(amplitude error `<4.2e-5`, phase error `<2.0e-4°`) and the 64-line combined
authored spectrum. Combined diagonal leakage was `0.0444382`, below the
pre-registered `0.05` tolerance; diagonal complex-transfer and safety gates
passed. Empty and 0.349 kg payload equilibrium are recorded analytically and
the existing Phase 03 payload path remains unchanged.

### Contact

The selected `canonical_explicit_pair` candidate compiled all 8 required
Can/table-target/finger pairs with exact pair-local parameters. Physics-only
probes passed:

```text
static support        = 4 contacts, 3.3573 N normal force
incline threshold     = atan(0.30) = 0.2914568 rad; 0.1 rad test passed
single-axis slip      = 0.05 m/s → 0.0386621 m/s in the probe window
impact/recovery       = 0.0001067 m maximum penetration; zero warnings
finger load           = 2 contacts; finite normal force
contact dt convergence = 0.346% maximum support-force deviation across 3 dt
```

The sliding friction values remain exactly `0.30` and `1.00`; torsional,
rolling, `condim`, margin/gap and contact solref/solimp are now explicit and
audited rather than inherited from MuJoCo defaults.

## 3. Bounded parity, integrity and determinism

Gamma=0 parity is bounded parity: the dynamic table is not required to match
stock static trajectories sample-for-sample. The same Panda/task geometry,
action semantics, passive support topology and solvability prerequisites are
retained; the dynamic residual is reported separately. The Phase 04
environment artifact and current compiled audit cover this geometry/contact
boundary.

`robosuite/models/assets/shakebench_phase_06_replay_determinism.json` contains
three independent-process replays of the same state/config. Every field in the
complete 21-field `DeckDriverTrace` was included in the digest; all three
digests, shapes, dtypes, options and program hash matched. The replay payload
hash is
`c8cc96349a0fd0d8b8e7496430e2295594b2c5d1377847bfa9f03dc4d4bb9a9f`.
No same-process reset was used as the determinism gate.

The official and protocol assets are flat files under
`robosuite/models/assets/` and are covered by the existing recursive
`MANIFEST.in` rule. `load_official_physics_profile()` verifies the embedded
SHA-256 before environment use; `make_probe_physics_profile()` is explicitly
`probe_non_scoreable`.

## 4. Artifacts

Raw candidate artifacts are all flat under `robosuite/models/assets/` with the
prefix `shakebench_phase_06_raw_`; they include candidate status, exclusions,
compiled options, trace digests and physics metrics. The protocol hash is
`9f992b9a7277772604cbdc761e80bb61156dfadb806773042f8ddab36c9b2578`.

The selection verifier and replay verifier are read-only and both return
`passed=true` in this checkout. The Phase 06 implementation does not make the
full benchmark scoreable yet: controller/state/protocol freezes remain later
phase responsibilities.
