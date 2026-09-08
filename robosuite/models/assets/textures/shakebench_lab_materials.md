# ShakeBench laboratory texture provenance

These three textures are original deterministic assets from the user-owned
ShakeBench reference project. Its `assets/textures/README.md` identifies them
as repository-generated, from `tools/generate_lab_textures.py` and
`tools/generate_platen_texture.py`, rather than the legacy photographic assets.
Reference file hashes and purposes are recorded in `shakebench_scene_visual_v1.json`.

The source JPGs were decoded and saved as PNG for native MuJoCo compatibility.
Decoded RGB pixels are identical; no color grading or AI texture generation was
applied. All runtime paths resolve inside the package. The platen's apparent
bores are an albedo pattern, with no additional collision surfaces.

| Packaged PNG | Original JPG | Packaged SHA-256 |
|---|---|---|
| `shakebench_epoxy_floor_cool_gray_1k.png` | `epoxy_floor_cool_gray_1k.jpg` | `ca5b0de5fff11c3497fa9edb9bca6d2654ddca6f1949e253df9b09145f9fdaef` |
| `shakebench_industrial_wall_light_gray_1k.png` | `industrial_wall_light_gray_1k.jpg` | `f9a8a169c72307c5efa5164d8aaeabfc7ef19beee5ad4a6a50425a95a43612b7` |
| `shakebench_platen_threaded_holes_1k.png` | `platen_threaded_holes_1k.jpg` | `76c897d7892270b9abb7943e92885225ddac54ce8ba33889b84ef47581c41663` |
