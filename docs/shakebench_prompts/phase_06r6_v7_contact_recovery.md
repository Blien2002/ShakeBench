# Phase 06R6 Prompt: V7 Contact-Recovery Selection

Work in `/home/miracle04/Desktop/ShakeBench`. Phase 06 is still BLOCKED for a
real physics reason, not a software-contract issue: V6 completed its driver,
isolator, parity and replay evidence, but all three V6 contact candidates
failed the fixed impact-recovery gate. Preserve every V1–V6 artifact and
status. Do not start Phase 07 and do not use task success, reward, controller
outcome, State tiers, policy observations or `Gamma_star`.

Read first:

- `docs/shakebench_prompts/phase_06_physics_freeze.md`
- `docs/shakebench_prompts/phase_06r5_v6_all_stage_contract.md`
- `docs/robosuite_benchmark_design_tree.md` (contact/failure-integrity gates)
- `docs/phase_06_report.md`
- `robosuite/models/assets/shakebench_selection_protocol_v6.yaml`
- `robosuite/models/assets/shakebench_phase_06r5_v6_status.json`
- all three `shakebench_phase_06r5_v6_raw_contact_*.json` files
- `robosuite/models/assets/shakebench_phase_06r5_v6_selected_candidates.json`
- `robosuite/models/assets/shakebench_phase_06r5_v6_excluded_candidates.json`
- `robosuite/scripts/shakebench_select_physics_v6.py`

## 1. Record the V6 finding accurately

V6 proved the software/physics pipeline can run: 18/18 driver states passed,
only `low_frequency_damped` isolator was eligible, parity and four
three-process replay groups passed. The blocker is contact recovery:

```text
c3_nominal:  v≈0.463 m/s, |omega|≈11.10 rad/s after 0.50 s
c4_torsional:v≈0.463 m/s, |omega|≈11.11 rad/s after 0.50 s
c3_softer:   v≈0.526 m/s, |omega|≈15.94 rad/s and penetration > 0.5 mm
```

Do not relabel these raw files as V7 evidence or modify their bytes. Add a V6
integrity addendum: its excluded table names `c3_nominal` as selected even
though eligibility count is zero; V6 status is authoritative BLOCKED. Fix the
selection-table code and add a regression so a zero-eligible component yields
`selected_contact=null`, a non-PASS table and no profile payload. Preserve the
historical V6 table as evidence rather than rewriting it.

## 2. Diagnose the recovery mechanism before V7 registration

Create a non-scoreable diagnostic script and report using V6 physical facts.
It must not call environment reward/success/task logic and must not publish a
profile. For each V6 candidate, record a time-aligned impact trace from
release through the fixed `0.50 s` recovery horizon:

- release and first-impact times;
- Can position/orientation, linear and angular velocity;
- contact count, normal/tangential force or impulse, contact distance;
- penetration, active contact interfaces, and mechanical energy where
  well-defined;
- the terminal value and a declared trailing-window maximum/RMS for velocity
  and angular velocity.

Use this diagnostic to distinguish persistent rolling/spin, rebound, solver
oscillation and measurement-timestamp error. The original recovery horizon
and limits remain fixed: `0.50 s`, `0.02 m/s`, `0.20 rad/s`, and `0.5 mm`.
Do not lengthen the horizon or relax a threshold to turn a failure into PASS.

Also audit the V6 finger-load evidence: recorded forces of roughly 46–291 kN
for a 0.349 kg Can are not an adequate realism certificate merely because
they exceed a minimum. Derive a finite, documented upper force envelope from
the compiled Panda gripper actuator/transmission/contact configuration, and
record the derivation. This becomes a V7 hard gate alongside the existing
minimum; do not invent a numerical limit without the derivation.

The diagnostic report is design evidence only. It must explicitly state that
it is not a V7 candidate result, has no selection authority and is not used as
a hidden score.

## 3. Prepare a genuine V7 contact selection

Implement and test the V7 contact runner/verifier before writing its protocol.
The runner must keep V6's all-stage resolved-state/adapter contract and add
the complete recovery trace plus force-envelope audit. It must compute gates
from raw numeric traces, not nested `passed` flags or an analytic substitute.

Use a raw MuJoCo bracketing experiment for incline threshold (pre-registered
angle grid/resolution); analytic Coulomb angle is a comparison, not the
measurement. Finger-load testing must use a declared force/penetration
trajectory and audit all named Can–finger pairs.

Add tests for:

- recovery timestamps and the frozen 0.50-s terminal/tail-window convention;
- recovery trace mutation, missing trace or wrong contact interface;
- candidate profile compiled equality (`condim`, all friction dimensions,
  margin/gap, solref/solimp, iterations);
- force envelope derivation and over-force rejection;
- zero eligible contacts producing no selected contact/profile;
- verifier rejection of a V6-style inconsistent selected/excluded table.

## 4. V7 protocol: expand the physical contact family deliberately

Create `shakebench_selection_protocol_v7.yaml` after the diagnostic and
runner tests are complete, then commit it alone with its feasibility artifact
before any V7 candidate probe. It must retain every V6 non-contact parameter
and frozen task geometry exactly.

V7 must retain `c3_nominal` and `c4_torsional` as labelled negative controls,
and add an explicit finite family that activates rolling resistance:

- a `condim=6` baseline with the existing torsional/rolling values;
- at least two `condim=6` rolling-resistance levels that bracket the baseline
  logarithmically;
- one `condim=6` candidate that varies contact damping (`solref` damping
  ratio) based on the diagnostic mechanism, while preserving legal MuJoCo
  values and `solref >= 2dt`.

The protocol must name every candidate's complete tuple, including all three
friction components, `condim`, margin/gap, solref/solimp and solver settings.
It must pre-register the finite candidate list, diagnostic-derived rationale,
fixed recovery trace convention, force envelope, hard gates, numeric scoring,
tie-breaks, retry policy and output paths. No code-generated sweep is an
authority.

Keep sliding friction fixed at table/Can `0.30` and finger/Can `1.00`; do not
change Can mass/geometry, drop height, recovery horizon, task geometry,
driver, isolator or any controller action. The contact choice must result
solely from V7 physics metrics.

## 5. Inherit only verified V6 non-contact evidence

V7 may inherit the V6 driver and isolator results only after a V7 verifier
recomputes their raw hashes, coverage, gates and selection ordering from the
listed V6 artifacts. It must establish exactly:

```text
driver   = dt_nominal
isolator = low_frequency_damped
```

The V6 global BLOCKED state and its contact selection are never inherited.
If any inherited raw hash/gate/profile field differs, V7 is BLOCKED and must
not silently rerun or substitute it. V7 must newly run every contact candidate,
new Gamma=0 bounded parity, and all four three-process replay groups using the
final V7 contact candidate. It must revalidate that inherited driver/isolator
values exactly match the final profile.

## 6. Publication gate

The V7 verifier must independently recompute inherited evidence, V7 contact
eligibility/scores/tie-break, recovery trace gates, force envelope, parity,
replay and every raw/protocol hash. It may emit exactly one selected contact
only if that contact is eligible. Otherwise it writes a machine-readable
BLOCKED status with `selected_contact=null` and no official profile payload.

Only when all V7 checks pass may it atomically write the official physics
profile with V6 driver/isolator and V7 contact parameters, bind it to all
evidence hashes, and enable the official loader. Update `docs/phase_06_report.md`,
the prompt index, focused/affected tests and clean sdist/wheel/MJCF smoke.
Run full `python -m pytest`, recording EGL limitations separately. Phase 07
may start only after verified V7 PASS.
