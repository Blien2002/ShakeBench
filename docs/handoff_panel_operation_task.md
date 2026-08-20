# 交接文档：ShakeBench 新增控制面板任务（panel_operation）

状态：**物理建模已替换，基线策略仍未全部收敛**。本仓库最后一次全量单测通过；knob 单控件已由真实碰撞和真实转动关节完成，lever/button 关节与碰撞模型已接入，但参考控制器尚未把两者操作成功，因此三控件完整序列仍未通过。

> 2026-08-17 最新实现说明：下文保留了旧实现的调试历史；其中“控件 visual-only / proximity 推进状态”的描述已经失效。当前权威实现是 `panel_controls.py` + `panel_task.py` 中的三套独立 articulation，任务进度只读取仿真关节位置，`request_panel_progress()` 会直接拒绝脚本化推进。

### 最新物理验证快照

- knob：`success=true`，真实关节进度 1.0，最大穿透 0.317 mm，0 帧超过 0.5 mm；右指接触峰值约 1813 N（读数偏高，仍需力学标定）。指标：`/tmp/shakebench_panel_knob_physical_v3.json`。
- lever：已产生左右指接触并推动真实关节，最好一次约 6.2% 目标角；随后 contact-lost，未成功。已补齐球形握把 collider，最大穿透在最近安全轨迹中约 0.685 mm。指标：`/tmp/shakebench_panel_lever_physical_v6.json`。
- button：正确按钮接触可触发 operate 阶段，但 prismatic 关节仍保持 0，最终 operation-timeout；同时出现约 7.86 mm 手部穿透，未成功。指标：`/tmp/shakebench_panel_button_physical_v2.json`。
- 三控件序列、随机种子组、official 1000 Hz：因 lever/button 未通过，未做成功声明。
- panel 任务暂时禁用 wrist joint-wrench sensor：Newton 多 articulation 场景会把全局 body index 当成 robot-local index，启动时报 out-of-range；六个按链接过滤的指-控件接触传感器仍启用。

---

## 1. 需求（已与用户澄清）

在现有 ShakeBench 中新增一个 `panel_operation` 任务：

- **不要新建工作台**：直接在当前 C2 工作台上放一块固定的控制面板。
- 面板上只有三个控件：**旋钮 knob、拨杆 lever、按钮 button**。
- 三个控件在面板上**呈三角形排列**（当前实现：knob 左上、lever 下中、button 右上）。
- 每次 rollout 的**操作子集和顺序随机指定**（不是固定的“顺序操作”）。未显式给定指令时，按 `panel_seed` 确定性采样一个 1~3 个控件的随机排列；也支持显式指定顺序。
- 建模尽量复用现有素材：房间、工作台、振动地板、Panda、Stewart 均复用；控件用参数化 primitive（圆柱/方块）建模。

---

## 2. 当前实现了什么

### 2.1 配置与随机指令

- `config.py`：
  - 新增 `CONTROL_KINDS = ("knob", "lever", "button")`。
  - 新增 `sample_panel_sequence(seed)`：确定性采样随机子集 + 随机顺序。
  - 新增 `PanelConfig`：面板几何、三角布局 UV、控件尺寸/行程/目标角度、操作速度、超时、接触阈值。
  - `BenchmarkConfig` 新增 `task: Literal["pick_place","panel_operation"]` 与 `panel: PanelConfig`。
- `panel.py`：
  - `PanelLayout` / `ControlLayout`：从配置解析面板与三个控件的 task-local 坐标。
  - 三角布局：knob `uv=(-0.085, +0.055)`，lever `uv=(0, -0.055)`，button `uv=(+0.085, +0.055)`。
  - 辅助函数：`padded_sequence_ids`、`control_speed_1_s` 等。

### 2.2 场景建模

- `visual_assets.py`：新增 `ControlPanelAppearanceCfg` / `spawn_control_panel_appearance`，在面板正面加显示用的边框、标签块、状态灯（全部 collision-disabled）。
- `scene.py`：
  - `BenchmarkSceneCfg` 增加可选字段：`panel`、`knob`、`lever`、`button`、`panel_appearance` 以及 6 个 per-control 接触传感器；`pick_place` 时这些字段保持 `None`，**不影响原拓扑**。
  - `_configure_panel_task()`：
    - panel 是 kinematic cuboid，固定在当前工作台上方。
    - knob/button 为 axis-X 圆柱；lever 为绕面板局部 X 轴侧摆的竖杆（以底部为 pivot，root pose 写为 COM 绕 pivot 的偏移）。
    - **面板任务里 wrist camera 碰撞被关闭**（保留渲染），否则相机支架会撞到 knob。
    - **knob/lever/button 三个控件碰撞已关闭**（保留 Newton 渲染形状），面板本体仍可碰撞；原因见“问题 4”。
    - `pick_place` 时仍走原路径，workpiece/target/sensors 不变；panel 时把这些 pick 专用资产置 None。
  - `make_scene_cfg()` 按 `cfg.task` 分支；panel 任务的 shaker shadow 用面板中心点。
- `visual_manifest.yaml` / `visual_manifest.py`：为面板外观新增 trim/label/status light 三条 manifest feature 及 facts（单测已通过）。

### 2.3 任务与控制器

- `panel_task.py`：`PanelBenchmarkTask`
  - 完全复用 C2 支撑运动逻辑（platform/robot/worktable/table legs + panel/controls 都走 `_support_state` 的完整 SE(3) 映射）。
  - 维护 `panel_sequence`、`_control_state`（3 维归一化进度）、每个控件的 root pose/velocity 写出。
  - knob：绕局部 X 旋转；lever：绕局部 X 侧摆（不会摆进面板）；button：沿局部 +X 按压。
  - observation 含 `panel_sequence_ids`、`panel_state`、knob/lever-tip/lever-pivot/button-face 的 world/base pose、各控件左右指接触力等。
  - `PanelEpisodeMetrics`：成功 = 指定序列全部完成且无 wrong_order / wrong_control_contact。
- `panel_controller.py`：`ScriptedPanelController`
  - 阶段：`settle -> pre -> approach -> move -> operate -> retreat`，每个控件重复。
  - 关键防碰撞措施（都在这个文件里）：
    - 保持 settle 手部姿态，不旋转到 panel-frame 姿态（旋转会让 DLS 换肘分支并扫过桌面）。
    - 使用 settle 时捕获的 hand→finger-center 基座偏移，不动态重算。
    - IK 输出 clamp 到 joint limits 后，再做 joint-space 速率限制（1.2 rad/s）。
    - 阶段切换用**实测 hand pose**误差，而不是 commanded pose 误差。
    - 操作时手指停在控件前方 `REACH_STANDOFF_M=0.025 m`，以 finger-center 距离 `OPERATION_PROXIMITY_M=0.03 m` 作为操作门控；**不是真实接触门控**。
  - 当前模式是“脚本化 reach-and-hold + 控件视觉状态推进”，保证可复现、不依赖脆弱的指间夹取。

### 2.4 CLI / 场景配置

- `cli.py`：
  - 新增 `--task pick_place|panel_operation`、`--panel-sequence knob,lever,button`、`--panel-seed`。
  - 场景 `task` / `panel_sequence` / `panel_seed` 可从 `scenarios.yaml` 读取。
  - **`configure_mujoco_contact_solref()` 已移到 `task.reset()` 之后**（原顺序在 panel 任务会触发问题 2）。
  - 日志与 metrics JSON 已按任务分支；panel JSON 包含 `panel_instruction`、`panel_seed`、`panel_collision_shapes`。
- `recording.py`：overlay 支持 panel 任务（显示 sequence、三控件状态；不再读取 pick 专用字段）。
- `configs/scenarios.yaml`：新增
  - `panel_operation`（vibration off，seeds 0/17/31/47/73）
  - `panel_operation_spectral_demo`（safe scale 0.15）
  - `panel_sequence_knob_button`（显式指令）
- `diagnostics.py`：`classify_penetration_pair` 增加 finger/panel 与 finger/control 语义对。

---

## 3. 已完成的验证

- `./run_tests.sh`：**36 passed**（改动后跑通过一次；新增 manifest features 已包含在测试内）。
- pick_place 原路径可正常构建、机械臂可正常运动（用 training + vibration off 短跑验证过）。
- panel 场景可正常构建，Newton/MJWarp 拓扑可打印。
- `knob` 单控件：training、vibration off、seed/sequence 显式 `knob`，**success=true**（手部轨迹稳定、无桌面穿透，knob 状态推进到 1.0）。
- panel 三控件：早前版本 knob 完成后，lever 阶段发生 `move_timeout`；随后做了多项调整，最近一次三控件运行被用户 KeyboardInterrupt，**尚未确认是否成功**。

---

## 4. 遇到的问题 / 现象 / 初步结论

### 问题 1：DLS IK 在面板前方姿态下输出关节角远超限位

- 现象：joint1 目标约 `-5.13 rad`，而 Panda joint1 限位约 `[-2.897, 2.897]`。
- 处理：IK 输出显式 clamp 到 `robot.data.joint_pos_limits`；并加 joint-space 速率限制。
- 结论：panel-front 的 pose 目标会让 DLS 产生 unwrapped 关节角，必须 clamp + rate-limit。

### 问题 2：`configure_mujoco_contact_solref()` 与 panel 控制器组合时，机械臂完全不响应关节目标

- 现象：任务收到正确 arm target、target buffer 也是正确值，但 `joint_pos` 不变化；手工恒定/ramp target 则能运动；不带 controller 也能运动；pick_place 不受影响。
- 复现实验：
  - 无 configure + controller：臂运动。
  - configure + controller：臂冻结。
  - configure + 手工目标：臂运动。
  - 先写 145 个 settle 目标再跳变到 IK 目标：冻结。
  - 直接跳变到同一 IK 目标：运动。
- 关联现象：起初 panel 与 knob/button 存在 2~6 mm 的 kinematic-kinematic 初始穿透，episode_start 有 2 个 active contact（`ControlKnob<->ControlPanel`、`ControlButton<->ControlPanel`）。
- 初步结论：configure 后，长时间保持目标 + 面板/控件持续接触约束 + 大目标跳变，会让 Newton/MJWarp 忽略 articulation target。**消除控件与面板几何重叠后，active_contacts=0，臂恢复运动**。同时把 configure 调用移到 `task.reset()` 后。
- 遗留：没有找到 Newton 侧根因；如果未来恢复真实接触门控，需要重新验证该组合。

### 问题 3：panel-front 目标用 pose-mode DLS 容易扫过桌面

- 现象：直接 move 到 knob 目标时，link5/link6/hand 穿透桌面，wrist force 数十万 N、穿透几十 mm。
- 处理：
  - 增加 `pre`（后上方）和 `approach`（正前方高位）两段路点。
  - 保持 settle 手部姿态。
  - 用实测手部位姿作为阶段切换条件。
- 结果：knob 单控件已无桌面穿透（max_penetration=0）。

### 问题 4：手/腕部相机与 knob 的物理碰撞

- 现象：
  - 原先 wrist camera 的 collision-enabled 支架会撞 knob（`MountBracket<->ControlKnob`，穿透约 29 mm）。
  - 即使相机碰撞关闭，`panda_hand` 本体在 front-reach 时仍会与 knob 碰撞（约 3~5 mm）。
- 处理：
  - panel 任务中关闭 wrist camera collision。
  - 最终把 knob/lever/button 三个控件设为 **visual-only（collision_enabled=False）**，panel 本体保留碰撞。
  - 控制器改为 proximity-gated 的 reach-and-hold，不再依赖真实指-控件接触。
- 结论：如果后续要恢复真实接触，需要重新设计手部位姿或控件伸出量，避免 hand 本体与控件干涉。

### 问题 5：lever 阶段 move_timeout

- 现象：lever 目标低于 knob，手臂在接近目标时实测误差无法稳定进入 `MOVE_TOLERANCE_M=0.006`，约 4 s 超时；误差大约 x≈0.010 m、z≈0.018 m。
- 处理：把 `MOVE_TOLERANCE_M` 放宽到 `0.012`；之后未完成完整验证即被中断。
- 初步结论：lever 目标在可达到边缘，DLS 末端收敛慢；如果仍超时，可进一步把 lever 交互点沿 panel 局部 +Z 抬高 10~20 mm，或对该阶段单独放大容差。

### 问题 6：最新一次三控件运行被中断

- 最后一次命令：`./run.sh --task panel_operation --panel-sequence knob,lever,button --vibration off --physics-profile training --episode-s 24 ...`
- 日志 `/tmp/panel_all2.log` 末尾为 warp `pack_arg` 中的 `KeyboardInterrupt`（用户主动打断），没有 metrics 输出，无法判断 lever 后续是否成功。

---

## 5. 文件清单（谁对应什么）

| 文件 | 改动/作用 |
|---|---|
| `src/shakebench/config.py` | `CONTROL_KINDS`、`sample_panel_sequence`、`PanelConfig`、`BenchmarkConfig.task/panel` |
| `src/shakebench/panel.py` | 面板布局解析（三角排列、pivot/尺寸/目标） |
| `src/shakebench/scene.py` | panel/knob/lever/button 场景资产、面板外观、6 个接触传感器、`_configure_panel_task` |
| `src/shakebench/visual_assets.py` | 面板正面显示细节 spawner |
| `src/shakebench/panel_task.py` | `PanelBenchmarkTask`、C2 支撑写出、控件状态写出、obs、metrics |
| `src/shakebench/panel_controller.py` | `ScriptedPanelController`、分段路点、IK clamp/rate-limit、proximity 操作门控 |
| `src/shakebench/cli.py` | `--task`/`--panel-sequence`/`--panel-seed`、任务分支、metrics JSON |
| `src/shakebench/recording.py` | 录像 overlay 的 panel 分支 |
| `src/shakebench/diagnostics.py` | 新增 finger<->control / finger<->panel 穿透语义 |
| `src/shakebench/wrist_camera.py` | `WristCameraAssemblyCfg.collision_enabled`（panel 任务关闭） |
| `src/shakebench/visual_manifest.py` / `configs/visual_manifest.yaml` | 面板视觉 feature facts/清单 |
| `configs/scenarios.yaml` | 三个 panel 场景 |
| `docs/handoff_panel_operation_task.md` | 本交接文档 |

---

## 6. 建议的下一步（给接手者）

1. 先跑单测：`./run_tests.sh`，确认无回归。
2. 重跑 knob 单控件确认仍 `success=true`：
   ```bash
   ./run.sh --task panel_operation --panel-sequence knob \
     --vibration off --physics-profile training --episode-s 18 \
     --metrics-output out/panel_knob.json
   ```
3. 继续验证三控件完整序列，重点看 lever 是否仍 `move_timeout`：
   ```bash
   ./run.sh --task panel_operation --panel-sequence knob,lever,button \
     --vibration off --physics-profile training --episode-s 24 \
     --metrics-output out/panel_all.json
   ```
   若超时：抬高 lever 交互点、单独放宽 lever move 容差、或缩短 lever 长度/目标角度。
4. 随机指令验证：`--panel-sequence` 留空，用 `--panel-seed 0/17/31` 各跑一次。
5. 官方 1000 Hz 档验证：`--physics-profile official`（注意耗时会明显增加；先 vibration off）。
6. 若要更真实的物理操作，需要重新解决控件/手碰撞与 configure 冻结问题，而不是保留当前 proximity 门控。
