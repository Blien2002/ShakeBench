# Scene maturation plan and validation log

This plan records the implementation sequence from
`vibration_benchmark_v2_场景改造提示词.md`. Every phase keeps the original seven
tests and the spectral pick-and-place reproduction command as release gates.

| Phase | Scope | Expected render/FPS effect | MJWarp contact effect | Tests |
|---|---|---|---|---|
| P0 | NewtonGL capability and contact baseline | Probe only | Establish baseline | Original 7 |
| P7.4 | Replace C2 linear displacement with exact SE(3) | None | None | Add 4/50 mrad boundary test |
| P1 | Collision-disabled visual Stewart platform | Small render-only cost from 29 Newton shapes | Geometry/pair delta must be 0 | Add analytic IK and 1000-step stroke tests |
| P2 | Recessed pit and larger laboratory | Moderate static render cost; no per-environment room clone | New visual items must be collision-disabled | Preserve deterministic room test |
| P4 | Baked contact-shadow appearance and lighting tune | Small fill-rate cost | Delta 0 | Texture SHA-256 gate |

## Completed checks

### P0 baseline

- Original test suite: `7 passed`.
- Newton shapes: 210.
- MJWarp collision geometries: 156.
- MJWarp filtered candidate contact pairs: 4030.
- Initial active contacts: 0 before task reset.
- Capability conclusions: see `newtongl_capabilities.md`.

### P7.4 exact C2 mapping

- C2 position uses `t + R @ r - r`, retaining the benchmark's established
  pitch-axis sign convention.
- The velocity path uses the exact rigid-body relation `v + omega x (R @ r)`;
  no finite differences were introduced.
- At 4 mrad, the exact and old vertical components differ by less than 1 um.
- At 50 mrad, the old full-position approximation differs by more than 900 um.

### P1 Stewart platform

- Six paired base/platen joint locations, twelve fixed-length telescoping visual
  bodies, twelve universal-joint spheres, a hexagonal base plate, inertia block
  and six air springs are configured through `ShakerGeometryCfg`.
- Default nominal leg length: approximately 0.83815 m.
- Post-change Newton shapes: 239 (+29 render shapes).
- Post-change MJWarp collision geometries: 156 (delta 0).
- Post-change filtered candidate contact pairs: 4030 (delta 0).
- Test suite after P7.4/P1: `10 passed`.

The command-line runner prints `[CONTACT:episode_start]` and
`[CONTACT:episode_end]` and stores both snapshots in its JSON output. This makes
future pure-visual changes fail visibly if they enter MJWarp contacts.

### P2 recessed laboratory

- The room is now `6.00 x 5.00 x 3.00 m` and is authored once at
  `/World/RoomArena`, so it does not scale with `num_envs`.
- A `2.05 x 1.55 x 0.78 m` recessed pit, four-piece deck, safety border and a
  three-sided guard rail replace the former continuous floor.
- All room decoration, UV skins, pit structure and rail components explicitly
  set `collisionEnabled=false`; the simulation ground plane is below the pit.
- Post-change Newton shapes: 257.
- Post-change MJWarp collision geometries: 29.
- Post-change filtered candidate contact pairs: 348.
- The lower totals are intentional: legacy room/texture visual meshes were
  removed from MJWarp instead of merely adding the new room on top of them.
- A 10 s pure-roll (`rx=0.02 rad`) visual acceptance run produced a simultaneous
  Stewart leg-length spread of 11.68 mm without stroke-limit violations.

### P4 lighting subset

- Ambient and key-light balance was tuned for the larger pit scene.
- Fake transparent shadow decals were not added because the installed ViewerGL
  probe reports opacity as unsupported. This avoids opaque artifacts and keeps
  the collision boundary unchanged.
- No new texture files were introduced, so `configs/assets.yaml` requires no new
  SHA-256 entry; the existing texture hash gate remains active.

### Final release gates

- Test suite: `11 passed` (the original seven plus four P7.4/P1/R3 tests).
- Spectral reproduction: 430-frame H.264 MP4 and JSON with `success=true`.
- Episode-start contact snapshot: 426 Newton shapes, 29 MJWarp geometries,
  348 candidate pairs, 0 active contacts.

## Second-round implementation and validation

The second-round work follows
`vibration_benchmark_v2_第二轮改造提示词.md` and starts from the completed
P0/P7.4/P1/P2 implementation above.

### R0 objective visual baseline

- Added `tools/visual_audit.py` for MP4/PNG extraction, ROI metrics,
  baseline/compare JSON and dependency-light histogram PNG output.
- The archived 5.5 s baseline is `docs/visual_baseline.json`: mean 148.58,
  std 34.79, IQR 21, narrow-band ratio 73.78%, low-saturation ratio 68.46%.
- The final formal video is `docs/visual_final.json`: mean 110.32, std 60.81,
  IQR 122, narrow-band ratio 49.27%, low-saturation ratio 1.06%.
- All three R1 gates pass: IQR >= 60, std >= 55, narrow-band ratio < 50%.

### R1 material and lighting hierarchy

- The physical platen is now a dark side skirt with a brighter collision-free
  top skin, producing an explicit top/side luminance separation.
- Pit, Stewart cylinder, rod, joint, base and table-frame colors are separated
  into ordered dark/cool material groups.
- ViewerGL uses sky/ground 0.36/0.14, specular scale 0.30 and exposure 0.88.
- Because the installed ViewerGL ignores opacity, transparent radial decals
  were not used. Static contact cues use three concentric collision-disabled
  discs, avoiding opaque square artifacts. A dynamic height-fading workpiece
  decal remains unimplemented and is recorded as a renderer limitation.

### R2 worktable

- Visible 60 mm tabletop edges, dark perimeter band, tube-frame appearance,
  lower stretchers, foot plates and 16 bolt heads were added without new
  colliders; only the original top and four leg box colliders remain.
- The floor texture proves the UV path is functional. Source-image measurement
  isolated the invisible marble issue to the former light-gray texture
  (luminance std 6.35), so the active texture is the already-vendored,
  higher-contrast `marble_01_diff_1k.jpg`.

### R3 Stewart topology

- Six paired actuators remain analytic, continuous and in stroke. The joint
  radius is 36 mm, giving joint diameter / rod radius = 2.4.
- Added six joint flanges and three collision-free base cross braces.
- The clean side-view preset and 5 s pure-roll acceptance video are
  `out/r3_stewart_side_frame.png` and `out/r3_stewart_side_roll.mp4`.
- The 20 mrad roll run reached 12.39 mm maximum instantaneous leg spread while
  MJWarp remained 29 geometries / 348 pairs.
- ViewerGL exposes no verified wireframe mode, so the requested wireframe
  evidence is replaced by the clean side-view geometry frame.

### R4-R7 scene detail and composition

- The platen top uses one deterministic 7 x 11 hole-array UV skin rather than
  77 cylinder shapes. Its SHA-256 is registered in `configs/assets.yaml`.
- Added mount-zone lines, four accelerometer models, a nameplate region and
  alternating warning stripes.
- Guard rails are round tubes with mid rails, base plates, 20 anchor bolts and
  connection collars; both side endpoints stop in the rear half, preserving
  the camera-facing maintenance opening.
- Added one global control cabinet, chair, cable bridge, tool cart, emergency
  stop, ceiling, two fixtures and three routed power cables. These assets are
  authored once under `/World/RoomArena` and explicitly disable collision.
- The main camera is lower and closer. Telemetry panels are compact and report
  the configured translational acceleration RMS (6.22 g).
- The workpiece default remains 0.55 to preserve the validated manipulation
  trajectory; R7 improves its readability through the closer camera instead.

### Second-round hard-gate results

- Formal video: H.264, 1280 x 720, 30 FPS, 430 frames, full FFmpeg decode.
- Formal JSON: `lifted=true`, `placed=true`, `success=true`.
- Final horizontal error: 0.06975 m.
- Contact snapshot: 426 Newton render shapes, 29 MJWarp geometries,
  348 candidate pairs; visual contact-pair delta is zero.
- Tests: `11 passed`.
- Active textures and all SHA-256 values pass the manifest test.
- The project directory is not a Git checkout, so this round records a
  file-level implementation log rather than a Git diff.

## Third-round implementation and validation

The third-round work follows `vibration_benchmark_v2_第三轮改造提示词.md`.
Measurements below come from the repository state and the successful formal
run, not from the target values in the prompt. The preceding second-round
section is retained as historical evidence and is superseded by this section
where the two differ.

### S0 fail-first geometry and regional visual audit

- Added `tests/test_third_round_geometry.py` before changing geometry. All five
  new checks initially failed as expected: platen-joint X/Y clearance, joint
  depth below the skirt, actuator/platen AABB clearance, static-equipment
  grounding, and shadow completeness/alignment. The unmodified failure output
  is archived in `docs/third_round_s0_failure_baseline.txt`.
- Extended `tools/visual_audit.py` with fixed regions from
  `configs/visual_regions.yaml`. It now records per-region RGB, luminance and
  red-minus-blue values, excludes safety yellow from the warmth gate, and
  checks adjacent large-surface luminance separation.

### S2 Stewart geometry correction

- Base joints now lie on an ellipse with semi-axes `0.85 / 0.60 m`; platen
  joints use `0.62 / 0.40 m`. The cylinder radius is `0.05 m`.
- Platen-joint clearances are `79.42 mm` in X and `64.12 mm` in Y. The joint
  geometry is fully below the 80 mm side skirt with `10 mm` vertical clearance.
- The complete deterministic 1000-step spectral episode has no actuator
  segment/platen AABB intersection and all legs remain inside the configured
  `0.67-0.95 m` stroke.

### S1/S3 industrial room hierarchy

- Active surfaces are a generated cool-gray epoxy floor, light-gray industrial
  wall panels, dark baseboard and dark phenolic worktop. The former active wood,
  wallpaper and marble surfaces are no longer used.
- Pit interior, border, Stewart cylinders, rods, joints, base and table frame
  use ordered neutral/cool luminance levels. The pit is dark but not clipped to
  black, and added grates preserve depth cues.
- Only the rear safety-yellow guard rail remains; camera-facing sides are open.
  Cabinet, chair, tool cart, cable bridge, emergency stop and routed cables are
  collision-disabled room-level visuals. Cabinet and cart feet are grounded.

### S4/S5 benchmark hardware and composition

- The platen has a true 80 mm side skirt, 5 mm cap/bevel cues, a seam-free
  generated 7 x 11 chamfered-hole texture, a continuous segmented warning band,
  four accelerometer assemblies and routed cables.
- Worktable details include square feet, anchor bolts, frame shadows and a
  thick Panda flange with eight bolts. Shadow coverage includes the workpiece,
  moving platen, table feet, robot, target, chair, cart, cabinet and room cues.
- The dynamic workpiece shadow is the 13th collision-disabled follower in the
  existing Stewart visual collection; this avoids introducing a separate
  physics collection and leaves MJWarp contact topology unchanged.
- Default YCB scale is now `0.75`. The success indication is a compact
  upper-right badge rather than a large central overlay.

### Third-round hard-gate results

- Tests: `16 passed`.
- Formal video: H.264, 1280 x 720, 30 FPS, 430 frames, 14.334 s, full FFmpeg
  decode. JSON reports `lifted=true`, `placed=true`, `success=true`, bilateral
  contact confirmed and grasp assist used; final XY error is `0.05092 m`.
- Contact snapshots remain `29 MJWarp geometries / 348 candidate pairs` at both
  episode start and end, so the third-round visual contact-pair delta is zero.
  Newton render shapes are 385 after removal of the former plank visuals.
- At 6.0 s the central ROI measures IQR `127`, standard deviation `59.81`, and
  narrow-band ratio `43.90%`. Excluding safety yellow, regional warmth span is
  `45.20` (gate `<= 50`). Platen/floor and platen/table luminance deltas are
  `27.65` and `63.94` (gate `>= 25`). Evidence is in
  `docs/third_round_final_visual.json` and `docs/third_round_final_frame.png`.
- All four active generated texture SHA-256 values match `configs/assets.yaml`.

### Recorded deviations and renderer/runtime limits

- The prompt's preferred dark phenolic worktop was selected. Exact source RGB
  targets do not map one-to-one through NewtonGL lighting: for example the wall
  renders near luminance 147 rather than the nominal 195 target, while the
  objective warmth, adjacency and global hierarchy gates all pass.
- NewtonGL in this environment has no verified opacity path. Shadows therefore
  use opaque layered geometry cues; the warning belt uses continuous narrow
  segments instead of alpha decals or texture overlays.
- A requested follow-up low-angle run twice encountered an intermittent native
  Newton initialization `SIGSEGV` before model construction. This did not affect
  the clean formal run above, but is recorded as an upstream runtime risk rather
  than hidden or treated as a benchmark result.
- The project directory is not a Git checkout. Changes are recorded at file
  level and by test/render artifacts rather than as a Git diff.
