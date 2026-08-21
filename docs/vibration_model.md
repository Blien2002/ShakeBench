# Vibration model

ShakeBench authors vibration as relative six-axis acceleration spectra and solves one absolute level per `(seed, t0)`. Translation channels are in m/s² and rotation channels in rad/s². Each narrow band contains equal-acceleration spectral lines with deterministic frequency jitter and random phase. For a line with angular frequency `ω` and acceleration amplitude `A`, displacement amplitude is `A/ω²`; velocity and acceleration are analytic derivatives, never finite differences.

## Difficulty Γ

At workpiece offset `r_wp`, the calibrated vertical acceleration is

```text
a_wp,z(t) = q̈z(t) + [α(t) × r_wp]z
Γ = max over the 16 s window |a_wp,z(t)| / g
```

Because every channel is linear in the global `level_scale`, calibration needs one unit-level replay:

```text
level_scale = Γ_target g / max |a_wp,z,unit|
```

Multi-environment calibration uses the largest unit-level peak, equivalently the minimum safe scale. This makes realized Γ exact for every seed and removes seed-to-seed difficulty variance caused by random peak factors. Γ≥1 is ballistic and requires an explicit opt-in.

## Spectral family and rotational rule

Default translation centers are `tx=5 Hz`, `ty=6.5 Hz`, and `tz=8 Hz`, with relative RMS ratios 0.50:0.35:1.00. Rotation is derived rather than hand-tuned. For a rotation linked to a translation band,

```text
σ_alpha = kappa_rot σ_a / reference_lever_m
```

with `kappa_rot=0.30` and a 0.65 m reference lever. `rx` and `ry` link to vertical translation; `rz` links to longitudinal translation. `frequency_scale` moves all band centers together and `bandwidth_ratio` controls predictability.

## Feasibility wedge

At low frequency, displacement grows as `a/ω²`. After Γ calibration, startup replays the entire episode and rejects a deck translation peak above 25 mm. The resulting displacement-limited ceiling grows approximately with frequency squared until it reaches the acceleration ceiling `Γ=4`. This wedge prevents a low-frequency scan from silently changing the contact-injection regime.

The solver gate is separate: maximum substep support travel is bounded by `alpha_geometry × thinnest_feature = 0.05 × 8 mm = 0.40 mm`. It uses the real support-group geometry and seed waveform, not a nominal single-axis estimate.

## Initial phase

An initial state contains one scalar time offset `t0`. Spectral angles are evaluated at `ω(t+t0)+φ`; all axes therefore select a common later window of the same infinite realization. Independent per-axis phase offsets are forbidden because they alter cross-axis physics and, for multiple tones, do not represent a time translation.

The correct generation order is: sample `t0` in `[0,100)`, replay `(seed,t0)`, calibrate `level_scale`, apply the displacement gate, then store the state. The fifth-order startup ramp remains episode-relative rather than shifting with `t0`.
