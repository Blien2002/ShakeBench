# Asset attribution and scene revisions

Code and model assets have separate provenance. [LICENSE](../LICENSE) retains
the code's existing MIT copyright notice and the included MuJoCo notice. It
does not relicense third-party meshes, textures or demonstration datasets.

| Packaged component | Source and terms | Records |
| --- | --- | --- |
| RoboCasa objects | Official `robocasa/robocasa-assets` release; CC BY 4.0 | `shakebench/models/assets/objects/robocasa/LICENSE.md` and `SOURCES.json` |
| YCB-Sim drill | Vikash Kumar's YCB_sim; Apache-2.0 model, CC BY 4.0 underlying YCB data | `objects/ycb_sim/power_drill/LICENSE.md`, full Apache license and `SOURCES.json` |
| Wood049 and Wood095 | ambientCG, CC0-1.0 | `textures/ambientcg_wood*_color_*.json` |
| Stack wood atlases | Repository-generated procedural textures; repository license | `textures/stack_wood_*_1k.json` |
| Laboratory textures | Repository-generated reference-scene textures | `textures/shakebench_lab_materials.md` |
| Light phenolic surface | Generated image; generation provenance is retained | `textures/shakebench_phenolic_table_light_1k.json` |
| Console scene/button/indicators | MIT-licensed demo geometry with retained notice | `objects/panel/DEMO_CODE_LICENSE.txt`, `objects/panel/SOURCES.md` |
| Panel rotary grip, dial, collar and lever | Original native primitives; repository license | `tools/generate_panel_controls.py`, `objects/panel/SOURCES.md` |
| Wine rack and task primitives | Repository-authored geometry; ambientCG wood texture | `wine_rack_visual_sources.json`, `models/objects/` |

Credit RoboCasa, Vikash Kumar and the YCB Object and Model Set when using their
assets. The drill directory contains the full Apache-2.0 text. AmbientCG
source records link its license. Upstream robosuite meshes/materials are loaded
from the separately installed dependency and remain subject to its notices.

## Panel controls

The independently constructed controls use the stable asset identifier `original_controls`.
The knob has 12 grip convex pieces, 24 dark-root pieces, a dial and an inset: 38 moving
contacts. The lever, fixed collar and shoulder follow their analytic visual meshes.
`tools/fit_panel_collisions.py` derives contacts solely from local visual vertices.
Convex piece unions approximate curved surfaces; they are not exact concave collisions.

The gray waist-shaped flat grip is 27.511588 mm long, at most 8.705727 mm wide,
and 6.800 mm wide at its waist. Both long sides and both end faces expose 8.925 mm
of straight contact height above the unchanged 14.950 mm shoulder datum. The
straight faces reach normal coordinate 23.875 mm; a 0.5525 mm edge radius joins
them to the planar crown at 24.4275 mm, 9.4775 mm above the shoulder.
The black direction strip is 2.200 mm wide and occupies only the positive tangent
half: its projected span is 12.931169 mm. It has a rounded inner cap, follows the
flat crown and wraps down that end face. It points toward OFF at 0 degrees and ON
at 120 degrees; 60 degrees is an inspection pose.

Use the single regeneration entry point `python tools/generate_panel_controls.py`.
It updates the approved grip, paint, 12 upper convex hulls and knob COM/inertia;
the other controls and lower rotary contacts remain unchanged. Public generation
contains no scripted grasp strategy. Physical handle-only and wrist-path validation
belongs to the separate private collection workflow. Record its exact source and
model hashes; appearance or older demonstrations cannot establish grasp success.

Knob mass is 45 g and lever mass is 25 g. Explicit centers of mass and inertia use
normalized volume weighting over slightly overlapping analytic components; these
are geometric estimates, not measured density or hardware properties. Collision
geoms add no mass. Every moving contact receives the official finger-pad pair
settings. Release checks include all control pieces against robot geoms at the
existing 1 mm contact threshold. Official contact friction, solver, timestep,
controller, joint axes/ranges/damping, camera settings and scoring thresholds are
retained. Historical demonstrations do not automatically acquire these contacts
or this appearance; record the source hash or commit with each experiment.

Validate the current controls with the official CPU runtime before creating a
new demonstration release. Scripted oracles, collection and training remain
external to this evaluation package.
