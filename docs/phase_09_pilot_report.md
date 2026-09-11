# Phase 09 requalification pilot

Status: **INVALIDATED**. On 2026-09-10 the user requested replacement of the
steel and wood blocks with RoboCasa Objaverse canned-food and cookie-box
models. The results below describe the previous geometry only. The new
task contract and state payloads require a fresh selection and requalification.
Historical gate status: **BLOCKED**. This is non-scoreable evidence; the v2
state assets remain `prepared_for_requalification` and `scoreable=false`.

## Frozen selection and setup

The pilot selected knee-v2 parent indices `0, 24, 49, 74, 99`, retaining all
six task variants for each parent (30 states; five per configuration). The
selection, its parent/state payload hashes, and input/code authority are in
`out/phase09_pilot_invalidated_20mm/selection.json`.

The nominal gate used the common V0 State Oracle, `Gamma=0`, official
1200-step horizon, official physics, and `direct_mount_v1`. Every episode
executed the complete runner rather than a contact probe or smoke test.

## Static nominal gate

All 30 episodes were valid. The gate reached 20/30 environment-latched
successes, so it fails its required 30/30 criterion.

| Configuration | Success | Valid controller abort |
| --- | ---: | ---: |
| metal / steel_block | 0/5 | 5/5 |
| mat / steel_block | 0/5 | 5/5 |
| metal / wood_cube | 5/5 | 0/5 |
| mat / wood_cube | 5/5 | 0/5 |
| metal / bread | 5/5 | 0/5 |
| mat / bread | 5/5 | 0/5 |

Each steel-block failure was `valid / unsuccessful / policy_abort` in
`grasp_close`: the controller detected `public_anchor_drift`, used all three
recovery attempts, then aborted. This pattern occurred for every selected
parent and both surfaces, so it is not a Gamma effect or a state-selection
effect. The evidence supports classifying it as a common-controller grasp
qualification failure pending root-cause remediation; it does not establish
that the physical task is unreachable.

The full raw run is `out/phase09_pilot_invalidated_20mm/static_gamma000.json`.
Each of its 30 episodes was independently re-verified in
`out/phase09_pilot_invalidated_20mm/verdicts/`.
`static_summary.json` contains the aggregate and failed-state list.

## Not run

The static gate blocks the rest of the phase. Therefore Gamma 0.60 and 0.95
paired pilot rollouts, videos, and the three-process six-state replay were
not run. No knee scan, `Gamma_star` fit, scorecard, or benchmark ranking was
generated.

## Next action

Repair and test the shared controller's steel-block grasp behavior without
changing state identities, friction, task success thresholds, or the frozen
selection. That changes controller authority, so regenerate the affected v2
binding and rerun the complete static gate before any positive-Gamma work.
