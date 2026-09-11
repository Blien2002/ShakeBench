# Phase 08R: failure boundary remediation

## Outcome authority

New raw artifacts use `shakebench.outcome_contract` v1 and run/episode schema
v6. The contract hash is embedded in every run, episode, scorecard projection
and determinism trace projection. Its three independent terminal fields are:

| Field | Values |
| --- | --- |
| `episode_validity` | `valid`, `invalid` |
| `score_outcome` | `success`, `unsuccessful`, `null` |
| `termination_cause` | `success_latched`, `horizon_exhausted`, `task_rule_violation`, `policy_abort`, `policy_error`, `invalid_execution` |

Only `valid/success/success_latched`, `valid/unsuccessful/{horizon_exhausted, task_rule_violation, policy_abort, policy_error}`, and `invalid/null/invalid_execution` are accepted. Controller events are diagnostics and cannot encode or overwrite an outcome.

## Migration table

| Previous field/reason | v6 meaning |
| --- | --- |
| `episode_deadline` | Removed; runner horizon is the sole episode budget. |
| `phase_deadline` | `phase_deadline` controller event; recover/replan. |
| `public_grasp_*` | `grasp_not_established`, `grasp_loss`, or `grasp_slip` event; only a later controller decision may produce `policy_abort`. |
| `public_anchor_drift`, `public_tool_clearance` | `anchor_drift` / `tool_clearance_blocked` controller event. |
| `public_placement_loss`, `public_placement_rebound` | Controller event and same-rollout recovery. |
| `public_object_unrecoverable` | Prohibited in v6. |
| `public_object_edge_unrecoverable` | `edge_risk` event and, when no safe controller action remains, `policy_abort`. |
| `public_object_out_of_workspace` | `workspace_risk` event and optional `policy_abort`. |
| `evaluator_not_latched_after_public_verify` | Controller event; runner continues to success, abort, or horizon. |
| `physics_violation` | Registered finite penetration gate is `task_rule_violation`; backend failure is `invalid_execution`. |
| `nonfinite_action_or_state` | Policy action is `policy_error`; environment/backend state is `invalid_execution`. |
| `horizon_exhausted` | `valid/unsuccessful/horizon_exhausted`. |

## Measurement and retry

Success rate is `valid_success / (valid_success + valid_failure)`. Policy abort, policy error, penetration-rule violation and horizon exhaustion remain in that denominator. An invalid execution is excluded from the rate and marks a complete tier matrix ineligible until its exact job identity is retried. Old schema-v5 artifacts remain historical artifacts and are not admissible inputs to a v6 Phase 09 group.

## Evidence status

The fast synthetic outcome and controller tests live in `tests/test_shakebench_phase08r_outcomes.py`. The Phase 08 official and knee state assets were regenerated without changing their IDs, positions, seeds, `t0`, split sizes, or generator root seed. Their controller binding changed from `60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c` to `ce4182b352ddf04b2655f5930ce93d486c34944906ba227b86d3f5830711f7da`, and both now bind outcome contract `808ba17086d47b95e21435b6f0f5e90cfc794e3c9ace12d87836aab819ed0709`.

The regenerated official asset was exercised through a one-step V0/Gamma=0 v6 runner smoke and independently passed `verify_run_artifact`. The focused outcome, protocol, scorecard, CPU batch, and controller suites passed 60 tests after remediation. The verifier now derives success, task-rule violation, policy abort, and horizon exhaustion from terminal trace evidence, rather than accepting a rehashed terminal-cause string. Scorecards bind and recompute the outcome-contract hash; batch aggregates now authenticate every record, retry ledger, output hash, and incomplete-group mapping.

The V0/Gamma=0 ten-state nominal gate completed as `10/10` `success_latched` episodes at `out/phase08r_fix_validation/dev_v0_gamma000.json`. Its v6 artifact passed semantic verification with run ID `132f6a81be9942255c33e14ac49ca39339c4b7b583a80b2bd89e2c46f26e46e9` and 1,982 retained policy trace rows. A V0/Gamma=0.30 positive-excitation dev diagnostic on state 000 also passed semantic verification and reached `success_latched`; it did not trigger a recovery event. The fresh three-process V0/Gamma=0 replay manifest at `out/phase08r_fix_validation/determinism_manifest_gamma000.json` passed with one identical trace digest across all children. Fresh wheel and sdist clean installs both passed content inspection, installed-root import checks, environment smoke, and the v6 artifact verifier; evidence is `out/phase08r_fix_validation/package_v6_clean.json`.

The clean package evidence was regenerated after clearing the build cache and passed for both wheel and sdist at `out/phase08r_fix_validation/package_v6_final.json`. The new handoff verifier is `robosuite/scripts/shakebench_verify_phase08r_handoff.py`; its current verdict is recorded at `out/phase08r_fix_validation/handoff_verdict.json` and has one blocker: the positive-Gamma artifact is valid but contains no recovery event.

No knee scan or official measurement was run. A positive-Gamma diagnostic that exercises a recovery event remains required before Phase 09 is reopened.
