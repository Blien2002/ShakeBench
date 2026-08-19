# NewtonGL capability baseline

Verified on 2026-08-13 against the Newton package installed in the Isaac Lab
3.0 virtual environment. Re-run with:

```bash
./run_python.sh scripts/probe_newtongl_capabilities.py
```

## Material inputs

| UsdPreviewSurface input | Result | Evidence / consequence |
|---|---|---|
| `diffuseColor` | Effective | The USD importer resolves it into `shape_display_color`; the fragment shader consumes it as albedo. |
| Albedo texture connected to `diffuseColor` | Effective | UVs and one RGB albedo texture are retained and sampled by `albedo_map`. |
| `roughness` | Effective | Imported into the mesh source and used by the ViewerGL GGX shader. |
| `metallic` | Effective | Imported into the mesh source and used for F0, diffuse suppression and environment/specular response. |
| `normal` | Not effective | No normal-texture slot reaches ViewerGL; only authored/geometric vertex normals are used. Bake small relief into albedo or use actual low-cost geometry. |
| `opacity` | Not effective | The shape fragment shader always writes alpha `1.0`; use opaque baked decals, geometry, or a recorder-side composite. |
| `emissiveColor` | Not effective | It is absent from the importer-to-ViewerGL material record and the shape shader. Bake glow into albedo; it will not illuminate nearby objects. |

This means P1 can use genuine roughness/metallic differences, while P3/P4 must
still bake hole relief, ambient occlusion and soft-shadow appearance into RGB
textures.

## Lighting and shadows

- One directional sun is supported.
- One camera-relative spotlight/fill approximation is supported.
- Hemispherical sky/ground ambient light and an environment map are supported.
- Shadow mapping is supported for the directional light and can be toggled with
  `draw_shadows`; the benchmark deliberately disables it because the current
  4096 px hard/PCF shadow path is visually unstable in the moving close-up.
- Arbitrary USD point lights and a general multi-light list are not imported by
  ViewerGL. Additional point/area lights authored in USD therefore do not create
  NewtonGL illumination.

## Collision/rendering boundary

ViewerGL renders Newton model shapes, but a shape does not need to participate in
MJWarp contacts. Applying `UsdPhysics.CollisionAPI` with
`collisionEnabled=false` preserves a visible Newton shape while excluding it
from the MuJoCo-Warp geometry/pair tables. All Stewart follower hardware uses
that explicit setting.

## P0 runtime baseline

Before the Stewart platform change (`num_envs=1`, sugar box):

| Counter | Baseline |
|---|---:|
| Newton shapes | 210 |
| MJWarp collision geometries | 156 |
| Filtered candidate contact pairs | 4030 |
| Active contacts immediately after initialization | 0 |

The P1 acceptance invariant was met: MJWarp collision geometries and filtered
candidate pairs remained **156 / 4030**, while the Stewart hardware increased
the Newton render-shape count to 239.

After P2, converting the legacy room and UV skins to explicit visual-only
geometry reduced the scene to **29 MJWarp geometries / 348 candidate pairs**;
the larger room, recessed pit and guard rail increased the Newton render-shape
count to 257. Thus P2 added no visual-only objects to the contact tables and
also removed the pre-existing room-side contact pollution.
