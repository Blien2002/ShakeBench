# Phase 06F-R2 Semantic Handoff Remediation

Phase 06F-R2 is **PASS**. The frozen profile remains
`shakebench.official.physics.v2` with hash
`c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c`.
No physics, contact tuple, driver, isolator, task geometry, success threshold
or controller value changed.

The R2 protocol is `shakebench_phase_06fr2_protocol.yaml`; its protocol hash is
`80cddbc8cf2ca2984336fb54ec197a427aedf597e80735300a11e4f73318a2fc`.
The evidence commit is `76aa2615`; the Git anchor is committed separately and
requires that evidence commit to be an ancestor. The semantic CLI and public
loader call the same `verify_phase06fr2_handoff(asset_root)` implementation.

The handoff lists and verifies every Phase 06F/06F-R reference, six real
settling traces, semantic convergence result, parity trace, six complete replay
traces, package evidence, and all 21 inherited V6 driver/isolator raw files.
Each listed JSON path is reopened and checked for file SHA-256, payload hash,
schema, identity and trace digest. Missing, duplicate, swapped, omitted or
mutated evidence fails closed.

Parity is recomputed from raw values: zero normalized/decoded/clipped actions,
finite applied controls, seven channels, nominal Panda/worktable/Can/target
poses, target geometry, evaluator contract and threshold mapping, support,
penetration, warning and policy-observation privilege boundaries. The stored
parity Boolean is not an authority.

Transient convergence is recomputed on the common 200 Hz right-limit grid for
both open-worktable and target-bottom states against both finer timesteps. It
uses timestamp alignment, pose/twist/deck/contact sequences, cumulative
normal/tangential impulses obtained by integrating raw per-step forces, and
final-window force RMS. The old mean-support scalar is not used.

Parity and selected-contact replay each decode all three full traces and
recompute per-field/whole-trace digests, sample count, timestamps, dtypes,
shapes, finite values and process identity. Matching terminal summaries alone
cannot pass.

Verification record:

- R2 semantic/adversarial tests: `14 passed` in the final focused run;
- prior Phase 06F-R focused/profile tests remain passing;
- full non-renderer lane: `495 passed, 58 skipped, 7 deselected`;
- three fresh semantic verifier calls: all exit `0`;
- clean wheel and sdist installs: official profile, semantic handoff and MJCF
  compile pass (`8` explicit pairs, Can mass `0.349 kg`);
- renderer lane remains separately marked and locally blocked only by missing
  EGL/MESA `swrast_dri.so` / `EGL_EXT_platform_device` infrastructure;
- final worktree is clean and all Phase 06F/06F-R/R2 prompts are tracked.

Phase 07 may start only through:

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
```
