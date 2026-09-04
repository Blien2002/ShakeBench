# Phase 06F Final Official Physics Closure

## Result

Phase 06F is **PASS**. The only published profile is
`shakebench.official.physics.v2`; Phase 07 is authorized exclusively through
`shakebench_phase_06_to_07_handoff.json` and the single final verifier.

The pre-registered eligibility-first, minimal-intervention priority selected
`c3_nominal`. All four finite candidates were eligible, so the canonical
baseline won without using the 15 cm drop, incline fixture, task success,
reward, controller outcome, tier ranking, policy observations, or
`Gamma_star`.

## Registration and bindings

- specification reconciliation commit: `68ca4286`;
- finalizer/fixture preparation commit: `d0418b2a`;
- immutable protocol and no-physics feasibility registration commit:
  `de2e60df56bce83dd2867d36e6c2d78b63bb52fd`;
- protocol SHA-256:
  `54e756ca004ab43380e7bf571c5bb92cef249aab0761268bfca225ad471f75e0`;
- frozen contact-selection SHA-256:
  `c2d1363c3750bfd32154080b0f9d25903358fbbe3b1b11bf83f45e534ed295b8`;
- dependent-manifest SHA-256:
  `ed2342fa70ab36f0015fe032b5cca5ce64e39f509f9378a4d598644153adf94d`;
- evidence-manifest SHA-256:
  `935b01a32f717a868877b51b62715a5613b7e2732d1e0c69271a388e035b9ba6`;
- official profile SHA-256:
  `c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.

The protocol contains only the binding rule
`contact_candidate_id = selection.contact_winner`. After selection, every
parity/replay state ID and output name was materialized from `c3_nominal` plus
the immutable selection hash. Dry-run covered all four possible winner
branches with zero MuJoCo calls. The V8 regression test proves that an
`n6_overdamped_4` winner binds every dependency to itself and that a concrete
`n6_overdamped_2` pre-binding fails before physics.

## Contact hard gates

| candidate | eligible | open-table final-window linear / angular | target-bottom final-window linear / angular | max penetration | level-force transition | finger force / penetration | 3-dt max relative force error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c3_nominal` | yes | `5.99e-7 m/s` / `1.29e-5 rad/s` | `5.99e-7 m/s` / `1.29e-5 rad/s` | `3.91e-7 m` | `[1.027107, 1.2325284] N` | `14.0775 N` / `9.50e-6 m` | `3.50e-13` |
| `n6_critical_negative_control` | yes | `2.39e-7` / `5.17e-6` | `2.39e-7` / `5.17e-6` | `3.91e-7 m` | `[1.027107, 1.2325284] N` | `14.1069 N` / `9.50e-6 m` | `6.22e-13` |
| `n6_overdamped_2` | yes | `2.39e-7` / `5.17e-6` | `2.39e-7` / `5.17e-6` | `3.91e-7 m` | `[1.027107, 1.2325284] N` | `3.8395 N` / `9.50e-6 m` | `6.58e-13` |
| `n6_overdamped_4` | yes | `2.39e-7` / `5.17e-6` | `2.39e-7` / `5.17e-6` | `3.91e-7 m` | `[1.027107, 1.2325284] N` | `1.3587 N` / `9.50e-6 m` | `3.63e-14` |

All task-native settling windows maintained named support continuously for the
final `0.10 s`, with finite forces, finite state, zero warnings, linear speed
below `0.02 m/s`, angular speed below `0.20 rad/s`, and penetration below
`0.50 mm`.

The force reference was recomputed from registered values:
`F_mu = 0.30 * 0.349 * 9.81 = 1.027107 N`. The `0`, `0.5`, and `0.8 F_mu`
plateaus remained within the `0.50 mm` static displacement gate; `1.2` and
`1.5 F_mu` produced monotonic along-force motion. The transition bracket
contains `F_mu`. An intentionally compiled zero-friction control moves below
the reference threshold and fails both the static and bracket gates.

Phase 06F also fixes the explicit MuJoCo pair encoding: the registered
isotropic sliding coefficient is compiled into both tangent dimensions,
followed by torsional and two rolling dimensions. Historical V1--V8 profiles
retain their legacy encoding path and bytes.

## Dependent evidence and public diagnostics

The Gamma=0 bounded parity row and all 12 replay rows bind `c3_nominal` and the
selection artifact SHA-256. Driver, isolator, selected-contact and Gamma=0
parity replay each use three fresh processes; every group has equal complete
trace and metric digests, process IDs `{1,2,3}`, and no same-process reset.

The selected profile's `0.15 m` drop finishes at `0.00452 m/s` with a maximum
final-window speed of `0.00499 m/s`. This is public, non-blocking diagnostic
evidence. The old incline fixture reports a transition bracket of
`[0.003125, 0.00625] rad`, far from `atan(0.30) ~= 0.29146 rad`; the large
disagreement is explicitly retained as a non-publication diagnostic and was
not substituted for the level force gate.

The stale V1 official YAML bytes are preserved flat as
`shakebench_phase_06_v1_official_profile_invalidated.yaml`; its byte SHA-256
matches the pre-06F Git blob (`69e08202d934dc31c82db69e583dbedfd48070b2c641f76db58ec4f9ee8a9c66`).

## Verification record

- focused Phase 06F/profile/V8-preservation tests: `33 passed`;
- affected Phase 02--05 driver/isolator/environment/provider/sensor tests:
  `167 passed`;
- full non-renderer suite: `489 passed, 58 skipped`;
- three fresh-process final verifier calls: all exit `0`, with identical
  winner, protocol, evidence-manifest and profile hashes;
- wheel installation outside the checkout: official profile load and MJCF
  compile passed (`8` explicit pairs, Can mass `0.349 kg`);
- sdist installation outside the checkout: the same profile hash loaded and
  the same MJCF audit passed;
- `git diff --check`: PASS.

The literal full `python -m pytest -q` run encountered renderer-only
infrastructure failures. EGL initialization reports missing
`/usr/lib/dri/swrast_dri.so` and no usable `EGL_EXT_platform_device`; the
offscreen context destructor then reports an uninitialized `con`. These are
reported separately and are not hidden as physics or benchmark passes. All
tests that do not require the unavailable renderer infrastructure completed
successfully.

## Phase 07 gate

Phase 07's first executable command is:

```bash
python -m robosuite.scripts.shakebench_finalize_physics --verify-final
```

The official loader also invokes the package-only handoff verifier. It rejects
external scoreable mappings/paths and cannot be activated by any historical
V1--V8 status file. No Phase 07 controller was implemented in this phase.
