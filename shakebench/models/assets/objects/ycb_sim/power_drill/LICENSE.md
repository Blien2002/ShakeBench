# YCB-Sim power drill

The mesh and texture are copied from [vikashplus/YCB_sim](https://github.com/vikashplus/YCB_sim), revision `57546b87f4724c947eadd4241a7892473febb88d`. Its MuJoCo model is Apache License 2.0; the unmodified license text is in `LICENSE.apache-2.0`. The underlying [YCB Object and Model Set](https://ycb-benchmarks.s3.amazonaws.com/index.html) data are CC BY 4.0. Credit both Vikash Kumar and the YCB Object and Model Set.

ShakeBench rotates the visual mesh to define the battery-down upright pose and replaces YCB_sim's single convex collision mesh with four primitive contacts, leaving the handle accessible to the gripper. `SOURCES.json` records upstream file digests.
