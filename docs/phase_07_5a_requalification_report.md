# Phase 7.5C direct-mount local closure

Status: **PASS**. `phase08_authorized=true`.

## Authority closure

The authorization chain is acyclic and fail-closed:

```text
frozen science authority -> evidence-core manifest -> final authority
                                             final package -> outer handoff manifest
```

Scoreable episodes bind only the stable science authority. The evidence core binds
the final source inventory and all non-package scientific evidence. The final
authority binds that core. The wheel/sdist contain the final authority, and the
outer handoff manifest binds their package evidence without feeding package hashes
back into the authority. Neither finalization nor package rebuilding changes raw
episode identity.

## Scientific behavior

- Fresh schema-v5 V0/Gamma=0 evidence passed all ten untouched dev states: 10/10.
- State000 V0/Gamma=0 passed three independent spawn-process determinism.
- State000 V0--V3/Gamma=0.15 matched smoke passed all four semantic verifiers.
- The final scoreable state000 probe and direct-mount preflight passed.
- Geometry, scene, controller, official physics and dev-state bindings remain
  authenticated by the science authority.

## Visual and package evidence

The new 1280x720, 20 fps native MuJoCo EGL/FFmpeg video completed in 183 steps.
Approach, grasp/prelift, transport, place and success-hold keyframes were reviewed;
no visible penetration, jump or redundant down-up-down motion was observed.

Final wheel and sdist are stored under `out/phase07_5a_requalification/package/`.
A clean wheel install outside the checkout verified both authorities, direct scene,
three required textures, reset, scoreability and preflight. The final handoff
verifier passed from both the working tree and the installed wheel using an
sdist-derived evidence bundle with `PYTHONSAFEPATH=1`.

## CPU and tests

The accepted runner policy is 2 workers on CPUs `[0,2]`, one numerical-library
thread per worker. The formal spawn/atomic/resume batch passed at 0.013276 jobs/s
in a controlled no-swap timing window; `/swapfile` was restored afterward. The
4/8-worker measurements remain preserved as `resource_limited` negative evidence.
MJWarp/MJX/CUDA physics was not used.

Final verification: 615 tests passed and 82 renderer/environment tests skipped;
Black, isort and `git diff --check` passed for all modified Python files. No Phase
8/9 knee scan, 100 knee rollouts, 400x4 official rollouts, commit, push, PR or
release was performed, and the ten dev states were not modified or replaced.
