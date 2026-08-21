# Physics fidelity

## Profiles and support model

The official profile is 1000 Hz × 4 substeps with C2_CLITE dynamic platform/worktable support, 50 solver iterations, and `solref=(0.00060 s,1.0)`. The training profile is 240 Hz × 4 with `solref=(0.0025 s,1.0)` and is not scoreable. The solref time constant must span at least two solver substeps.

The deck is modeled as one rigid assembly. Robot, worktable, legs, and target use the same transform `p=q+c+R(l-c)` from visible local coordinates; this replaces the earlier inconsistent virtual-mount coordinates. Same-assembly structural collision pairs are filtered. C2_CLITE uses mocap plus weld constraints for dynamic main supports while kinematic followers retain the same geometric trajectory.

## Threshold provenance

Two thresholds answer different questions:

- The startup travel gate is `0.05 × 8 mm = 0.40 mm`, where 8 mm is the thinnest task collision feature and 0.05 is the solver/geometry safety factor.
- The scoring penetration limit is 1% of the selected workpiece's thinnest runtime collider dimension. For sugar box at 0.75 scale, this is about 0.336 mm.

Runtime collider AABBs, not nominal YCB dimensions, determine placement and grasp geometry. This rule was introduced after a nominal/runtime size mismatch allowed roughly 0.10 m of free fall and produced a 21.6 mm false penetration baseline.

## Contact-path limitation

With MuJoCo contacts, NativeCCD/MULTICCD clears the requested 1 mm margin. Metrics therefore state `nativeccd_margin_honored=false`. Restoring the margin array after compilation is not a valid fix: experiments produced speculative forces up to 21.6 kN. The benchmark relies on explicit profile qualification and measured signed contact distance instead of claiming that margin is active.

## Known limits

- C2 kinematic support at 1000 Hz × 5 had a five-seed numerical floor up to 0.259 mm and did not meet the stricter one-third qualification threshold. C2_CLITE at 1000 Hz × 4/50 iterations measured 0.067–0.102 mm in the historical five-seed qualification.
- Official C2_CLITE execution is slow (historically about 13 minutes for 16 simulated seconds); most time is in the Newton solver.
- The training profile can materially increase penetration and cannot support official claims.
- Upstream Newton construction can terminate without a Python traceback.
- Grasp assist is off for scoring. When explicitly enabled for demos, it requires bilateral contact and penetration below 0.5 mm, and releases above 1.0 mm.

## Round-7 Pareto gate

The completed Γ=0.50 sweep is recorded in
`docs/reports/fidelity_throughput_pareto.md`. At 1000 Hz × 4, lowering the
Newton main iterations from 50 to 10 retained the penetration gate (0.048 mm
for the measured seed) but did not improve wall time. The 500 Hz and 240 Hz
profiles were 2–4× faster but measured 0.246–0.297 mm penetration and failed
the 0.112 mm qualification line. Because no qualifying profile achieved the
required ≥4× speedup, the official profile remains 1000 Hz × 4 / 50 iterations;
prior penetration baselines are not invalidated.
