# Phase 06F-R Real-Environment Handoff Remediation

## Final result

The remediation is **PASS** and the public loader now requires this versioned
handoff before Phase 07. Historical Phase 06F remains byte/hash verified and
is preserved as evidence; its tuple and selection were not changed.

Frozen binding remains:

```text
profile = shakebench.official.physics.v2
driver  = dt_nominal
isolator = low_frequency_damped
contact = c3_nominal
```

Registration commit: `096e4c7ea7a48b2936c101b755e34a9b23af2fb3`.

Remediation protocol SHA-256:
`80b62f4af50d4c0f0b0f0037d7d663fc52a8037fef44de19131f89006a336006`.

Official profile SHA-256 remains:
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.

## Real-environment settling

All six states used `VibrationPickPlaceCan`, Panda, dynamic deck, full target
geometry, explicit Can pairs, ordinary `env.step(zero_action)`, Gamma 0,
seed 17, rendering disabled, and the private exact-hash profile copy. No
`raw_contact_xml` or reduced support-box fixture was used.

| support | dt | native samples | final support | max penetration | gate |
|---|---:|---:|---:|---:|---:|
| open worktable | 0.0001 | 5000 | named/finite | `<0.50 mm` | PASS |
| open worktable | 0.000125 | 4000 | named/finite | `<0.50 mm` | PASS |
| open worktable | 0.0002 | 2500 | named/finite | `<0.50 mm` | PASS |
| target bottom | 0.0001 | 5000 | named/finite | `<0.50 mm` | PASS |
| target bottom | 0.000125 | 4000 | named/finite | `<0.50 mm` | PASS |
| target bottom | 0.0002 | 2500 | named/finite | `<0.50 mm` | PASS |

Every final `0.10 s` window passed named support, `0.02 m/s` linear speed,
`0.20 rad/s` angular speed, finite state/forces, zero warnings and `0.50 mm`
penetration. The artifacts contain deck/table/Can state, contact identity,
actions, applied controls, warnings and full timestamped physics traces.

## Transient convergence

The selected `0.0002 s` traces were compared on a common 200 Hz right-limit
grid against both finer traces for both support states. Position/orientation,
relative velocity, worktable/deck response, support identity, penetration,
force/impulse and warning sequences passed the registered tolerances. The
convergence artifact is
`b26e7a670352aaa9d9c1f3818adde9b31d7235dc163d889cea16be4dd9e8fc84`.

The verifier includes an adversarial transient mutation: equal mean support
force with a changed middle penetration/velocity sample is rejected.

## Gamma=0 parity and replay

The parity artifact contains a complete 2500-sample real-environment trace and
raw values for profile/hash, timestep, 7D action semantics, Panda/worktable/
Can/target poses, target frame, evaluator contract, unchanged success
thresholds, policy observation keys and privileged-truth exclusion. Reward and
success interface invocations are recorded for compatibility but are not used
by the physics verifier.

Parity and selected-contact replay each ran in three fresh processes. Both
groups have identical whole-trace digest
`a16233d17d19a512a6257474d140481fe7cd22dc97b7a444813646061719d7c9`, identical
metric digest, process indices `{1,2,3}`, exact timestamp/schema/shape/dtype
validation and `complete_trace=true` backed by the lossless trace payload.

## Inherited evidence reopening

The final verifier reopens all 18 V6 driver files and three isolator files,
recomputes file and payload hashes, checks protocol/state/resolved-state
identity, recomputes numeric gates, and confirms `dt_nominal` and
`low_frequency_damped`. Mutation, missing-file and incomplete-matrix tests
fail as expected.

## Release verification

- remediation focused/adversarial tests: `38 passed`;
- Phase 06F/profile/V8 preservation tests remain passing;
- affected Phase 02–05 environment/deck/isolator/provider/sensor lane: `204 passed`;
- full non-renderer lane: `495 passed, 58 skipped, 7 deselected`;
- public loader: loads v2 profile only after the remediated PASS handoff;
- clean wheel and sdist installs: profile load, remediation verify and MJCF
  compile pass (`8` explicit pairs, Can mass `0.349 kg`);
- Python compatibility is now honestly `>=3.10`, with import smoke coverage;
- renderer lane remains separately marked and fails locally only because EGL/
  MESA lacks `swrast_dri.so` / `EGL_EXT_platform_device`;
- `git status --short` is clean after the final commit.

No Phase 07 controller, Gamma scan, physics search, contact change or task
success threshold change was made.
