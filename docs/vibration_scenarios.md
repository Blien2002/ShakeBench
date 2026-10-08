# Optional vibration scenarios

`scenario_lines_v1` adds the supplied explicit six-axis line tables. It is opt-in:
the default remains `multisine_v1`, gamma defaults to zero, and existing state assets,
physics, scoring, sensor conventions and evaluation schemas are unchanged.

| Scenario | Status | Nominal excitation at gamma=level=1 |
| --- | --- | --- |
| S1 | Recommended | 1.25–1.5 Hz rocking about a seeded horizontal axis; primary rotation 3.5°, plus smaller secondary/yaw lines |
| S2 | Recommended | Independent 12-line translations in 2–6 Hz; line-energy RMS 1.98 m/s² on x/y and 2.34 m/s² on z |
| S3 | Recommended | 2.0–2.9 Hz horizontal circular primary orbit: radius 10 mm, diameter 20 mm; small 2nd/3rd harmonics and vertical motion |
| S4 | Recommended | 0.75–1 Hz press strokes; nominal vertical carrier peak 0.7 g; horizontal peak 1.05 g along a seeded azimuth, reversing every stroke |

“Recommended” describes the attachment's optional scenario workload. It does not
change the default benchmark protocol or certify equal task difficulty. Attachment
object-response results are historical supplied evidence, not validation on this build.

## Coordinates, units and physical point

The line table uses deck axes `tx ty tz rx ry rz`. Translation coordinates, velocity
and acceleration are in m, m/s and m/s²; angular coordinates, velocity and acceleration
are in rad, rad/s and rad/s². Frequencies are in Hz and line phases in radians.
`rx ry rz` encode a rotation vector, as in the existing driver; its derivatives are
coordinate derivatives and must not be confused with the measured IMU angular velocity.

Rotations act about the deck origin. The attachment's nominal tabletop-centre offset
from that origin is `[0.11, 0, 0.299]` m. For small rotations its additional displacement
is `rotation_vector × offset`; the simulation uses the existing exact rotation path.
S3's 20 mm is the **primary deck-origin orbit diameter**, not a half-amplitude or a
guarantee for the total harmonic trajectory or an object's measured motion. S1's
tilt and yaw can add table-centre translation even though its translation lines are zero.

For each carrier line, `qdd=A*sin(2*pi*f*t+phase)` and
`q=-A/(2*pi*f)^2*sin(2*pi*f*t+phase)`. The unchanged quintic startup ramp and its analytic
derivatives apply at episode zero. `t0_s` shifts carrier phase, not the ramp's clock;
time-shift equivalence holds once both samples are past the startup ramp.

The attachment's S3 direction label has been corrected: `ccw` means positive oriented
area in deck x/y viewed from +z toward the origin. With the existing negative-sine
displacement convention, `ty` phase = `tx` phase + pi/2 is clockwise. Authored line
amplitudes, frequencies and phases are preserved; only that latent label is corrected.

## Intensity and replay

Actual line amplitude = authored amplitude × `mode_params.level` × `gamma`.
`gamma=0` or `level=0` gives zero q/qdot/qdd on every axis. For this mode,
`gamma_definition="scenario_level_v1"`: gamma is an **amplitude multiplier**, with
no peak-Gamma calibration. Explicitly requesting `normal_peak_v1` or `magnitude_peak_v1`
for this mode fails. Conversely, old modes still use their original peak calibration.
Scenario gamma=1 must not be grouped as the same intensity as old benchmark gamma=1.

`build_vibration_program` validates the recipe before environment construction.
`vibration_record(program)` records `gamma_definition`, the absence of calibration,
seed, t0, mode parameters, latents and the full program hash. CPU, async and development
MJWarp evaluator `run_contract` records add `vibration_semantics` only for the new mode,
including units and `comparable_to_peak_gamma_grid=false`; old mode records stay unchanged.
The existing `scoreable=false` policy-evaluation boundary remains in force for all modes. This flag was already unconditional before integration; it is not caused by the short smoke-test horizon and does not permanently prohibit the new mode. Valid episodes still report the existing task success/failure and success rates. This optional suite has not been added to the current certified scorecard protocol. Publishing a scenario benchmark later requires a versioned suite/intensity/state/horizon contract, grouping by scenario and explicit scorecard handling of `scenario_level_v1`, followed by the backend and rendering validations. Episode
seeds/t0 continue to come from the selected state. Replay latents are regenerated from
that seed and recipe, rather than used as a second source of randomness.

Parameters must be finite numbers, with nonnegative amplitudes, strictly positive
damping, integer line counts (S2: 1–12), and
`0 < f_lo < f_hi < 8.87` Hz. Unknown keys fail. Every axis has at most 12 lines.
The registry records each scenario's accepted keyword parameters and status.

## Run one scenario or the optional suite

Use an existing policy factory and state selection:

```bash
python -m shakebench.scripts.evaluate \
  --policy my_policy:make_policy --state-ids shakebench-dev-v0-000 \
  --mode scenario_lines_v1 --mode-params '{"scenario":"S2","level":1,"params":{}}' \
  --gamma 1 --output out/scenario_S2.json

python -m shakebench.scripts.evaluate_scenarios \
  --policy my_policy:make_policy --state-ids shakebench-dev-v0-000 \
  --gamma 1 --level 1 --output-dir out/scenario_suite
```

`shakebench/models/assets/shakebench_vibration_suite_v2.json` packages the optional
S1/S2/S3/S4 recipes and their intensity semantics. The suite runner creates one
ordinary evaluation result per scenario and refuses to overwrite an existing result.
Group results by `run_contract.mode_params.scenario`. It does not collect data or
automatically run with the old Gamma grid.

API module: `shakebench.physics.vibration_scenarios`; entry points `build_scenario_program` and
`build_vibration_program` both accept deterministic seed/t0. Set `params` to override
an authored scalar, for example `{"scenario":"S3","params":{"radius_m":0.008}}`.

The official package runs CPU MuJoCo with the existing EGL camera path.

## Canonical naming contract v2

Only S1/S2/S3/S4 are accepted. They retain the previously recommended physical recipes. No retired alias is accepted. Historical results retain their original labels and must be replayed with their original frozen source; this catalog does not reinterpret their provenance. The current S3 remains the 10 mm radius orbit.

## Result provenance

Canonical S1/S2/S3/S4 denote the former S1r/S2w/S3/S4r waveforms. Evaluation run semantics and per-episode vibration records identify this catalog as `scenario_catalog_version: canonical_v2`. The additional record field does not change the waveform program hash or the replay recipe. Historical S1/S4 records must retain their original source version and must not be relabelled as canonical v2.

