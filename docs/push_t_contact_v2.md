# Push-T contact physics v2

The user selected Push-T table sliding friction 0.50 on 2026-10-08. The effective explicit T-block/table pair uses 0.50 on both sliding axes in the official isotropic 5-D contact encoding. Pusher friction remains 0.50. Torsional/rolling friction, solver settings, shape, mass, inertia, state assets, controllers, action semantics and success conditions retain their existing values. Global official table/object friction remains 0.30: only the Push-T pair builder overrides its table interface. The same task-specific coefficient is supplied to all supported physics profiles; only the official compiled contact encoding is covered by the accompanying regression.

New task contexts record `contact_physics_revision: push_t_contact_v2`, `table_sliding_mu: 0.5` and the unchanged `pusher_sliding_mu: 0.5`. Task/state schema version 5 remains unchanged; the contact revision is separate from the scoring protocol. Keep the source commit and these fields with each result. Historical results and frozen collection sources are not rewritten. Water physics v2, Plain Stack v2 and Pick collection inputs are unchanged by this revision.

Changing friction changes the task dynamics. Existing data and checkpoints trained under mu030 remain archived under their original physical version and are not regenerated or relabelled; a mu050 evaluation measures performance under a different physical condition. Preserve the mu030 evaluation freeze (source-manifest SHA256 `95cf29c6fb745158a1ba57401f8d84a5a960e24e701c20c6f4783d869a40a3ee`) and its original results. Do not pool mu030/mu050 success rates or attribute their difference to the policy alone. An attribution study would require the same checkpoint, states, seeds and protocol on both explicit physical versions. No such paired policy evaluation, recollection or retraining is performed by this release.

## Existing passive evidence

The repaired friction-screen attempt02 tested the original goal-1 state with excitation seed 3780952855. The mu050 development source freeze was `77d16b5e6290f855f3dc8d851bb207fc840c63d78dfdbea54a2fb88d9a7c9572`; common initial states were recorded under state-manifest SHA256 `b795292eca86e3d0df0ede4a13554dbf61e90c4a26ba5a81a45b508b56320406`. The former S2w/S4r names map to canonical S2/S4. These are historical development-preview examples, not a new eval-only policy score or a distribution-level result.

| Scenario | mu050 max displacement | mu030 max displacement | mu050 planar path | mu030 planar path |
| --- | ---: | ---: | ---: | ---: |
| S2 (former S2w) | 42.874 mm | 67.854 mm | 84.465 mm | 411.432 mm |
| S4 (former S4r) | 30.682 mm | 28.709 mm | 343.923 mm | 524.515 mm |

Both mu050 examples satisfied the passive safety contract without table departure or robot-scene contact. The older screen required at least 5% lower maximum displacement and path than both the original Push and every Stack block in both scenarios. Mu050 failed that combined screen because S4 maximum displacement exceeded its 27.273 mm threshold; the screen had no winner. The present parameter adoption follows the user's explicit selection and does not convert that failed screen into a pass.

Recorded passive previews used a common pre-clock reset at mu030 and then the selected friction for recorded physics. That experimental reset hook is excluded from this release: the native Push-T reset uses the selected mu050 throughout. Old passive numbers therefore do not establish exact rollout equivalence with this release.

Active policy/oracle Push-T performance at mu050, with either gamma0 or gamma1, remains unverified. Mu060 passive/active evidence cannot establish mu050 success. Parameter search is stopped; this release adds no collector, oracle, experimental reset hook, training data or policy-score claim.
