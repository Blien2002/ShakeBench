# Texture provenance

All active 1K JPG diffuse maps are deterministic repository-generated assets,
so benchmark runs remain reproducible and do not require network access.

| Use | Source | Local file | SHA-256 |
|---|---|---|---|
| Floor | `tools/generate_lab_textures.py` | `epoxy_floor_cool_gray_1k.jpg` | `7da2141fb9b7a38966114d5876d9616899f749222c9299360f341fa4b4c2a474` |
| Wall | `tools/generate_lab_textures.py` | `industrial_wall_light_gray_1k.jpg` | `9910939ec5a98cfb3863ba33b752266126756103422b1a54c384a39d68f564f4` |
| Tabletop | `tools/generate_lab_textures.py` | `phenolic_bench_dark_1k.jpg` | `6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb` |
| Platen | `tools/generate_platen_texture.py` | `platen_threaded_holes_1k.jpg` | `f686cdc8166275e3ea54a763b70a6aa65adee0ea1a22c4caa7a8cbf9a763f3a0` |

The laboratory generator creates a cool-gray epoxy floor without plank seams,
light-gray industrial wall panels, and a dark neutral phenolic-resin worktop.
The platen generator creates an aligned 7 x 11 chamfered-hole albedo without
the former seam-like zone lines. Legacy Poly Haven CC0 files remain beside the
active assets solely for provenance and historical reproduction; their license
is <https://polyhaven.com/license>.
