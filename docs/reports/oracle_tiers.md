# Oracle tiers and Round-7 decision point

Status: **blocked at the prerequisite gate; no decision claimed**.

The code now provides `oracle_full`, `oracle_phase`, and `oracle_reactive`
with distinct, automatically enabled privilege declarations. `oracle_full`
is fixed to 1 kHz; the other tiers emit slow `VARIABLE_IMPEDANCE` variables at
the configured policy rate. None uses `grasp_assist`.

The Γ=0.50 decision experiment has not been run because the committed Round-5
full official episode still ends in `grasp_slip_exceeded` rather than
`success=true` (see `docs/baseline_status.md`). Per `round7.md` §0/§1, the
project must repair and re-qualify the physical grasp before running or
interpreting this comparison.

| Metric | oracle_phase | oracle_reactive | Difference |
|---|---:|---:|---:|
| Success rate | not run | not run | — |
| `ee_tracking_error_rms_m` | not run | not run | — |
| `grip_margin_min` | not run | not run | — |
| `max_grasp_slip_m` | not run | not run | — |

The continue/stop decision is therefore **pending**, not negative. Contract
backend smoke tests are explicitly non-scoreable and are not evidence for
this table.
