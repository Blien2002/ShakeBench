# Phase 08 remediation record

This remediation regenerated only the deterministic committed official and
knee state authorities after adding `t0_s` and the canonical task-contract
hash binding. The frozen ten-state development asset was not modified.

| Asset | Previous SHA-256 | Regenerated SHA-256 |
| --- | --- | --- |
| `shakebench_states_official.json` | `f458dc46e99f76c312314698a936d46d22f67d793aca4d1258da32eb268fcd12` | `4453477d9178d1803e20151f1eabe980ee9da2cde1b87f9073348eae1ae2bedc` |
| `shakebench_states_knee.json` | `59ffdd79df0287a05e461a76909b5005fa02a201cda0af35887b49c8ff8e6eed` | `3890204cd0682984e164a82bd40b868de0367e9f797c075355b4fc71a714d362` |
| `shakebench_states_dev.json` | `07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6` | unchanged: `07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6` |

Validation was limited to synthetic, static, and contract tests. No knee
scan, official rollout, or physics backend execution was performed.
