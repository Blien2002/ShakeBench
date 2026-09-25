# RoboCasa task-object provenance

The instance XML, visual meshes, textures and collision meshes in this
directory come from the official RoboCasa asset release, `objaverse.zip` of
the `robocasa/robocasa-assets` Hugging Face dataset:

| ShakeBench object | RoboCasa instance |
|---|---|
| `cereal` | `objaverse/cereal/cereal_0` |
| `mug` | `objaverse/mug/mug_1` |
| `apple` | `objaverse/apple/apple_0` |
| `spatula` | `objaverse/spatula/spatula_0` |
| `bar_soap` | `objaverse/bar_soap/bar_soap_0` |
| `rolling_pin` | `objaverse/rolling_pin/rolling_pin_0` |
| `potato` | `objaverse/potato/potato_1` |

Source: <https://huggingface.co/datasets/robocasa/robocasa-assets>

The single documented transform is the task's category scale, baked into the
mesh/geom/body attributes plus the robosuite object sites, and recorded with
the upstream and generated SHA-256 digests in `SOURCES.json`.  RoboCasa
licenses its assets and datasets under CC BY 4.0; this derivative keeps this
notice and must credit RoboCasa.

The `pot/pot_061` model is the RoboCasa Lightwheel `Pot061` from NVIDIA's
[Manipulation Objects Kitchen MJCF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF),
[objects_lightwheel/pot.zip](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/blob/main/objects_lightwheel/pot.zip).
It is CC BY 4.0. ShakeBench scales its MJCF geometry to 0.62 at runtime;
`SOURCES.json` records the source and this modification. Credit NVIDIA and RoboCasa.
