# Training/official fidelity correlation

Status: **not run; training profile remains ineligible for data collection**.

The required comparison is `scripted + oracle_reactive + oracle_phase` across
five Γ levels and five episodes on both physics profiles. Acceptance requires
Spearman ρ > 0.9, plus reported mean/max success-rate bias and penetration
distributions.

No correlation value is claimed. Until this experiment passes, the
`training` profile must not be used for paper datasets or policy training.
The collection CLI also refuses the current non-physical state-contract
backend unless an explicit schema-smoke-test flag is supplied.

After the experiment, its machine-readable companion must contain at least:

```json
{"spearman_rho": 0.0, "training_eligible": false}
```

The collection CLI requires `spearman_rho > 0.9` and
`training_eligible=true`; this example is deliberately failing, not a result.
