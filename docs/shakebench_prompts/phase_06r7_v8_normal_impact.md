# Phase 06R7 Prompt: V8 Normal-Impact and Incline-Fixture Selection

Work in `/home/miracle04/Desktop/ShakeBench`. V7 is a completed contact
physics failure. Preserve all V1–V7 artifacts, commits and statuses. Do not
enter Phase 07 and do not use task success, reward, controller outcome, State
tiers, policy observations or `Gamma_star`.

Read first:

- `docs/shakebench_prompts/phase_06r6_v7_contact_recovery.md`
- `docs/phase_06r6_v7_contact_recovery_diagnostic.md`
- `docs/robosuite_benchmark_design_tree.md` (contact/failure integrity)
- `docs/phase_06_report.md`
- `robosuite/models/assets/shakebench_selection_protocol_v7.yaml`
- `robosuite/models/assets/shakebench_phase_06r6_v7_status.json`
- V7 contact raw files, diagnostic JSON and selected/excluded artifacts
- `robosuite/scripts/shakebench_select_physics_v7.py`

## 1. Record the actual V7 mechanism

V7 eliminated the earlier ambiguity: all candidates have negligible terminal
angular speed, but at `t=0.50 s` the Can has no named table contact and is
moving down at about `0.1325 m/s`. This is unresolved normal rebound / flight,
not persistent rolling. The frozen terminal and final-50-ms linear-speed gate
`<=0.02 m/s`, angular gate `<=0.20 rad/s`, penetration `<=0.5 mm`, and
recovery horizon `0.50 s` remain unchanged.

V7 incline evidence is also not a valid friction bracket: its first tested
angle is `0.10 rad`, so it lacks a zero-angle stable control and reports no
stable lower bound. Preserve V7 raw evidence; document this fixture limitation
without changing its bytes.

## 2. Diagnose normal contact and fixture semantics before V8 registration

Create a non-scoreable, raw-MuJoCo diagnostic with the exact task Can/table
geometry, mass, drop height and V7 physical defaults. It must not create an
environment task, reward, success result or profile.

### Normal-impact trace

For each V7 normal-contact tuple, record every physics step from release to
0.50 s: Can vertical position/velocity, full velocity, named contact count,
normal force/impulse, contact distance, penetration, first impact, every
contact-loss/re-contact time and kinetic/potential energy. Calculate:

- peak downward and rebound vertical speed;
- contact-loss intervals and terminal contact/support state;
- normal energy retained after each impact where defined;
- terminal and final-50-ms maximum velocity with direction.

The diagnostic must establish whether insufficient normal damping, compliance,
initial placement, or an integration/fixture error causes the flight. It is
design evidence only and not a candidate result or hidden score.

### Incline fixture calibration

Before any incline sweep, run a level (`0 rad`) support control: settle from a
non-penetrating pose, require named support contact, near-zero horizontal
speed and displacement relative to the **post-settle** pose. The inclined
fixture must use that same settled state, zero velocity and a declared
observation horizon. Its angle grid starts at `0`, includes sub-critical
angles and extends above `atan(0.30)`; use a pre-registered binary refinement
to form a measured stable/sliding bracket. The analytic Coulomb angle remains
a comparison only.

If level support cannot pass, V8 is BLOCKED as a fixture error; do not tune
friction or contact parameters to hide it.

## 3. Build a normal-contact selection runner before the protocol

Create a V8 runner/verifier that retains the V7 resolved-state and
adapter-contract guarantees. Add a distinct normal-impact probe plan and
compiled contact audit. The recovery gate must additionally require terminal
named table support with positive normal force; a low speed while airborne is
not recovered support.

Tests must prove:

- zero-angle fixture control passes before an incline bracket is accepted;
- a trace with terminal no-contact fails recovery even if speed is below the
  threshold;
- vertical rebound, terminal downward flight and contact re-entry are
  separately identified from rolling;
- recovery timestamps use exactly the fixed 0.50-s horizon and final 0.05-s
  window;
- candidate normal `solref`/`solimp`, condim and all pair settings compile
  exactly; and
- missing normal trace, altered raw trace or a fake stable lower bracket makes
  the verifier fail.

## 4. V8 protocol: select normal damping, not task performance

After diagnostic and runner tests, pre-register
`shakebench_selection_protocol_v8.yaml` plus a feasibility artifact in a
dedicated commit before V8 probes. Keep V7 driver, isolator, Can/task geometry,
drop height, recovery limits, sliding friction (`0.30` table/Can and `1.00`
finger/Can), force envelope and action/controller semantics unchanged.

Use a finite normal-response candidate family. Retain one V7 normal tuple as
a negative control. Hold the V7 tangential/rolling tuple fixed while varying
only legal normal contact parameters, with at least:

- a critical-damping control;
- two over-damped `solref` damping-ratio candidates;
- a slower compliant over-damped candidate; and
- one `solimp` normal-impedance variation justified by the diagnostic.

Every candidate must explicitly state `condim`, all friction dimensions,
margin/gap, both `solref` terms, full `solimp`, iterations and pair scope.
All positive contact time constants must be `>=2dt`. Do not change the 0.50-s
horizon, speed/penetration gates or observed force envelope to make a result
pass.

Pre-register the normal-impact trace schema, terminal-support condition,
level-control requirements, incline grid/refinement, candidate scoring and
tie-break. Choose contacts only from raw physics metrics: recovery, support,
penetration, normal-energy/rebound behavior, actual incline bracket, slip,
finger force/penetration, convergence and warnings.

## 5. Evidence inheritance and final gate

V8 may inherit V6/V7 driver and isolator evidence only after recomputing their
raw hashes, gates and selected values (`dt_nominal`,
`low_frequency_damped`). It inherits no V7 contact selection. V8 newly runs
every contact candidate, Gamma=0 bounded parity and all four three-process
replay groups with the selected V8 contact.

The verifier must output `selected_contact=null` whenever no candidate passes;
it must never publish a profile in that case. Only a fully verified V8 PASS
may atomically write the official physics profile and enable the official
loader. Update the Phase 06 report/index, focused and affected tests, package
smoke and full non-renderer test record. Phase 07 begins only after V8 PASS.
