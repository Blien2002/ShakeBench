# 视觉附件父变换审计

更新日期：2026-08-13。坐标为默认单环境名义世界坐标；`{ENV_NS}` 在运行时展开为 `/World/envs/env_0`。验收门槛为世界位置到期望表面锚点的偏差小于 5 mm。

| 附件 | 父 prim | 局部偏移 m | 世界位置 m | 期望锚点 m | 偏差 mm |
|---|---|---:|---:|---:|---:|
| `accelerometer_0` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (-0.7100,-0.4600,0.0626) | (-0.7100,-0.4600,0.1026) | (-0.7100,-0.4600,0.1026) | 0.000 |
| `accelerometer_cable_0` | `{ENV_NS}/VibrationFloor/LayeredAppearance/Accelerometer0` | (0,-0.0750,-0.0060) | (-0.7100,-0.5350,0.0966) | (-0.7100,-0.5350,0.0966) | 0.000 |
| `accelerometer_1` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (-0.7100,0.4600,0.0626) | (-0.7100,0.4600,0.1026) | (-0.7100,0.4600,0.1026) | 0.000 |
| `accelerometer_cable_1` | `{ENV_NS}/VibrationFloor/LayeredAppearance/Accelerometer1` | (0,0.0750,-0.0060) | (-0.7100,0.5350,0.0966) | (-0.7100,0.5350,0.0966) | 0.000 |
| `accelerometer_2` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0.7100,-0.4600,0.0626) | (0.7100,-0.4600,0.1026) | (0.7100,-0.4600,0.1026) | 0.000 |
| `accelerometer_cable_2` | `{ENV_NS}/VibrationFloor/LayeredAppearance/Accelerometer2` | (0,-0.0750,-0.0060) | (0.7100,-0.5350,0.0966) | (0.7100,-0.5350,0.0966) | 0.000 |
| `accelerometer_3` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0.7100,0.4600,0.0626) | (0.7100,0.4600,0.1026) | (0.7100,0.4600,0.1026) | 0.000 |
| `accelerometer_cable_3` | `{ENV_NS}/VibrationFloor/LayeredAppearance/Accelerometer3` | (0,0.0750,-0.0060) | (0.7100,0.5350,0.0966) | (0.7100,0.5350,0.0966) | 0.000 |
| `robot_mount_flange` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (-0.4700,0,0.0546) | (-0.4700,0,0.0946) | (-0.4700,0,0.0946) | 0.000 |
| `robot_mount_bolt_0` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (0.0920,0,0.0100) | (-0.3780,0,0.1046) | (-0.3780,0,0.1046) | 0.000 |
| `robot_mount_bolt_1` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (0.0651,0.0651,0.0100) | (-0.4049,0.0651,0.1046) | (-0.4049,0.0651,0.1046) | 0.000 |
| `robot_mount_bolt_2` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (0,0.0920,0.0100) | (-0.4700,0.0920,0.1046) | (-0.4700,0.0920,0.1046) | 0.000 |
| `robot_mount_bolt_3` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (-0.0651,0.0651,0.0100) | (-0.5351,0.0651,0.1046) | (-0.5351,0.0651,0.1046) | 0.000 |
| `robot_mount_bolt_4` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (-0.0920,0,0.0100) | (-0.5620,0,0.1046) | (-0.5620,0,0.1046) | 0.000 |
| `robot_mount_bolt_5` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (-0.0651,-0.0651,0.0100) | (-0.5351,-0.0651,0.1046) | (-0.5351,-0.0651,0.1046) | 0.000 |
| `robot_mount_bolt_6` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (0,-0.0920,0.0100) | (-0.4700,-0.0920,0.1046) | (-0.4700,-0.0920,0.1046) | 0.000 |
| `robot_mount_bolt_7` | `{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange` | (0.0651,-0.0651,0.0100) | (-0.4049,-0.0651,0.1046) | (-0.4049,-0.0651,0.1046) | 0.000 |
| `nameplate` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0,0.5508,0.0140) | (0,0.5508,0.0540) | (0,0.5508,0.0540) | 0.000 |
| `warning_band` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0,0.5508,-0.0170) | (0,0.5508,0.0230) | (0,0.5508,0.0230) | 0.000 |
| `shadow_table_foot_0` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (-0.0900,-0.2450,0.0466) | (-0.0900,-0.2450,0.0866) | (-0.0900,-0.2450,0.0866) | 0.000 |
| `shadow_table_foot_1` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (-0.0900,0.2450,0.0466) | (-0.0900,0.2450,0.0866) | (-0.0900,0.2450,0.0866) | 0.000 |
| `shadow_table_foot_2` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0.4500,-0.2450,0.0466) | (0.4500,-0.2450,0.0866) | (0.4500,-0.2450,0.0866) | 0.000 |
| `shadow_table_foot_3` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0.4500,0.2450,0.0466) | (0.4500,0.2450,0.0866) | (0.4500,0.2450,0.0866) | 0.000 |
| `shadow_robot_base` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (-0.4700,0,0.0466) | (-0.4700,0,0.0866) | (-0.4700,0,0.0866) | 0.000 |
| `shadow_target_bin` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0.0800,0.1700,0.0466) | (0.0800,0.1700,0.0866) | (0.0800,0.1700,0.0866) | 0.000 |
| `shadow_workpiece` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0.0800,-0.1300,0.0466) | (0.0800,-0.1300,0.0866) | (0.0800,-0.1300,0.0866) | 0.000 |
| `shadow_platen` | `{ENV_NS}/VibrationFloor/LayeredAppearance` | (0,0,0.0466) | (0,0,0.0866) | (0,0,0.0866) | 0.000 |
| `guardrail_base_WestSouth` | `/World/RoomArena/RailPostWestSouth` | (0,0,-0.0160) | (-1.1850,-0.9350,-0.0160) | (-1.1850,-0.9350,-0.0160) | 0.000 |
| `guardrail_base_EastSouth` | `/World/RoomArena/RailPostEastSouth` | (0,0,-0.0160) | (1.1850,-0.9350,-0.0160) | (1.1850,-0.9350,-0.0160) | 0.000 |
| `guardrail_base_SouthMid` | `/World/RoomArena/RailPostSouthMid` | (0,0,-0.0160) | (0,-0.9350,-0.0160) | (0,-0.9350,-0.0160) | 0.000 |

结论：30 项全部低于 5 mm。法兰的 8 个螺栓现在以 `RobotMountFlange` 为父 prim，坐标为半径 92 mm 的局部环；加速度计壳体、连接器和走线也都使用传感器根的局部坐标。`tests/test_visual_manifest.py` 同时断言误差门槛和这些父子关系的源代码模式。
