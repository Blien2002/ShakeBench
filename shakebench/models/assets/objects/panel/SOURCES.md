# Control panel visuals

Source: [wyl03291211/ShakeBench](https://github.com/wyl03291211/ShakeBench),
commit `22f526a7e73cea8e0b9f8c09a597602484a814a0` (the detailed inclined console,
before the flat-panel replacement). The console, bezels, fasteners, plaques,
indicator lamps and red button follow that version's `visual_assets.py` and
`panel_controls.py`. Demo code is MIT licensed; see `DEMO_CODE_LICENSE.txt`.

`panel.xml` contains the transformed runtime STL triangle meshes as inline MJCF
vertices/faces. Millimetres were converted to metres and the original panel
tangent/lateral/normal mapping retained. Mesh triangle counts: housing 16,
Apollo knob 1,624, switch base 472, switch handle 574. The verbatim switch
notice describes an earlier LOD; these are the counts in this MJCF. MuJoCo hinge/slide joints, contact
proxies, inertias, resistance and a close camera were added here. The switch
handle pivot is retained at 31.0949356 mm above its mounting surface. Moving
collision proxies are separate from visual meshes; the fixed switch base uses
the convex hull of its visual mesh. All visual geoms have zero mass.

Mesh attribution and terms are independent of the repository's code license:

- Apollo knob: James / Jamesteam, Smithsonian source data, **CC BY-NC 4.0**.
  See `apollo_command_module_control_panel_knob.LICENSE.txt`, copied verbatim
  from the demo. The listing also carries a **NoAI** restriction.
- Switch base and handle: demo runtime meshes from the user-provided
  `source/Interruptor.fbx`; source author and license are unspecified. See
  `switch_control.SOURCE.txt`, copied verbatim. Reuse here was requested by the
  user; this notice does not grant downstream redistribution rights.

This is an interactive visualization and physics scene, without Oracle,
collection/export integration, a task sequence, success rules or scoring.
