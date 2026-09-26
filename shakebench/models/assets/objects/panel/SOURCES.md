# Control panel visuals

Source: [wyl03291211/ShakeBench](https://github.com/wyl03291211/ShakeBench),
commit `22f526a7e73cea8e0b9f8c09a597602484a814a0` (the detailed inclined console,
before the flat-panel replacement). The console layout, bezels, fasteners,
indicator lamps and controls follow that version's `visual_assets.py` and
`panel_controls.py`. Demo code is MIT licensed; see `DEMO_CODE_LICENSE.txt`.

## Adaptation

The source-to-panel axis mapping has negative determinant. Its reflected
triangle winding must be reversed for MuJoCo's single-sided rendering; otherwise
all four imported meshes expose their interiors through culled outer faces.
All shipped meshes now have outward winding.

`apollo_knob_visual.stl` uses the original full-resolution Apollo source STL
(50,944 triangles), replacing the earlier 1,624-triangle runtime LOD. Source
millimetres are mapped as `-X * tangent + (Y - minY) * normal - Z * lateral`,
scaled by 0.001, then offset 19 mm along the normal. Triangle order is reversed
and facet normals recomputed. The XML applies the same 0.7 scale to this mesh
and its independent physical cylinder.

The switch base (472 triangles) and handle (574 triangles) remain inline MJCF
runtime meshes. The source switch pivot is 31.0949356 mm above its mounting
surface before the compact adaptation below. The copied switch source notice
describes an earlier LOD; the counts here describe the shipped meshes.

The compact adaptation scales the controls and their contact proxies to 70% of
those source dimensions, then rotates their mounting surface from 37.15 degrees
to 12 degrees above horizontal. The shell is rebuilt as one rounded wedge,
224 mm wide and 152 mm deep, with 9 mm corner radii and 2.5 mm rolled edges.
A thin rounded faceplate replaces the rectangular side rails and rear cap;
the shell and faceplate collision meshes follow the new silhouette. The shell
base mounts 1 mm above the tabletop. The main assembly camera is unchanged.

Display-only annular washers, rounded button cap, screw slots, spindle sleeve
and lamp rims refine the console. Materials distinguish powder coating, satin
metal, rubber and glossy plastic. All visual geoms have zero mass and disabled
collision. Moving-control masses and joint resistance are retained; inertia is
recomputed from the smaller contact proxies. Lever stop impedance is tightened
to keep limit deflection below 0.001 rad at the demo effort with the reduced
inertia. The lever still travels +/-30 degrees and the button retains its
4 mm spring-return travel.

`panel_markings.png` is original procedural artwork: face lettering, rotary
graduations and three nameplates. Rebuild it with
`python -m shakebench.scripts.build_panel_markings` using Pillow and DejaVu Sans
fonts. The PNG is packaged; fonts are not required at runtime. The inset label
surfaces sit 0.105 mm above the metal plaques to avoid coplanar flicker. The
colored indicator lenses have no task-state or success behavior.

## Attribution and terms

Mesh terms are independent of the repository's code license:

- Apollo knob: James / Jamesteam, Smithsonian source data, **CC BY-NC 4.0**.
  See `apollo_command_module_control_panel_knob.LICENSE.txt`, copied verbatim
  from the demo. The listing also carries a **NoAI** restriction.
- Switch base and handle: demo meshes from the user-provided
  `source/Interruptor.fbx`; source author and license are unspecified. See
  `switch_control.SOURCE.txt`, copied verbatim. Reuse here was requested by the
  user; this notice does not grant downstream redistribution rights.

This remains an interactive visualization and physics scene without Oracle,
collection/export integration, task sequence, success rules or scoring.
