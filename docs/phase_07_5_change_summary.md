# Phase 7 / 7.5 修改汇总与 Phase 8 准入结论

审计日期：2026-09-07。

## 结论

原 Phase 7 已完成：远端 `master` 与 Phase 7R6.1 gate tag 均指向
`4238e37cc3b57d626a37f7950df974fc10b0d1e8`，R6 evidence tag 解引用到
`1824060986949c3dd618075499fb8a709e08fb36`。R6/R6.1 manifests 对 canonical
配置记录 `PASS` 和 `phase08_authorized=true`；本次复核的 R6/R6.1 gate tests 为
`17 passed`。

该授权最初不能直接覆盖后续 `direct_mount_v1`。它删除了 RethinkMount 的质量、惯量和碰撞体，
改变机器人/桌面变换，因此候选 manifest 曾正确标记为 `scoreable=false`、
`phase08_ready=false`。受影响科学准入现已通过；最终 `direct_mount_v1` profile 为
`scoreable=true`，详见 `docs/phase_07_5a_requalification_manifest.json`。

CPU 部分只完成了短程可行性探针，没有完成或验收端到端加速。Phase 8 应复用 CPU 设计约束，
但必须重新实现/验证 batch runner 与最终配置吞吐。

## 工作副本与历史

- `/home/miracle04/Desktop/ShakeBench` 是 history rewrite 前的 recovery workspace，本地
  `master`/`origin/master` 仍为 `dd6fe2ed...`，包含大量 Phase 7 dirty 文件、原始证据和
  `out/phase07_5b_cpu` 短探针。
- `/home/miracle04/Desktop/ShakeBench/.phase07_5a_clean` 基于远端最新 Phase 7R6.1
  `4238e37c...`，分支为 `codex/phase-07-5a-scene-restoration`，承载当前场景实现。
- Phase 7 已完成 history slimming、独立 evidence release、runtime/full audit 分离、package
  gate 和 R6.1 fail-closed gate hardening。旧 recovery workspace 不应重新作为发布 authority。

## 场景与背景物

- 新增平铺、可哈希的 scene authority；记录参考文件 SHA-256、材质来源、frame ownership、
  room/pit/platen/Stewart/table-support/equipment/camera/render 参数。
- 场景补齐 6.0 x 5.0 x 3.0 m 实验室外壳、可见地坑、shaker foundation、六根 Stewart
  装饰杆、振动台、黄黑护栏、急停和安全边界。
- 新增精细控制机柜、功放/控制面板、仪器桌、示波器式仪器、电脑、键盘鼠标、工具车、
  储物柜、传感器托盘、信号线、校准砝码和记录本。
- 新增浅灰工业墙面、环氧地坪、振动台孔阵列贴图、踢脚线、顶面/灯具；使用 native
  MuJoCo shadow map。电脑屏幕改为空白黑屏，控制机柜转向振动台。
- 黄黑护栏改为连续圆管与弯头/套接/底板/锚栓结构；工具车移到后墙电脑桌左侧，储物柜
  放在控制机柜旁。
- 提供 overview、assembly、side、cabinet、instruments、cart、storage 等固定相机。
- 装饰几何均为零密度、`contype=conaffinity=0`、无 joint/constraint，不进入 policy 或
  evaluator。物理 floor 保留，只以非接触 slab 表现地坑。

## 场景/几何接口与运行元数据

- `shakebench_scene.py` 实现配置加载、canonical hash、MJCF 注入、材质/灯光、compiled audit、
  physics signature、AABB/geom distance 和 safe-envelope clearance。
- `shakebench_geometry.py` 实现版本化 geometry profile 与 scene 绑定。
- arena XML 保留 canonical tabletop/isolator 物理体，新增显式 `shakebench_platen_visual` role；
  原分散视觉桌架由 scene authority 生成。
- `VibrationPickPlaceCan` 新增 `scene_config`、`scene_visual`、`geometry_profile`；修复
  `base_types` 被硬编码为 `default` 的问题；增加 mount/scene/clearance compiled audit。
- deck processor 新增 `deck_visual` role，保证 platen visual 是 ordinary dynamic deck 后代。
- Oracle run artifact、run ID 和 verifier 增加 scene metadata；geometry profile run 强制
  `scoreable=false`，直到独立准入；当前 direct-mount requalification 已完成并提升为 true。
- demo 新增 policy-step observer，只用于 renderer；native MuJoCo EGL + FFmpeg 生成视频、
  sidecar、关键帧与视频 hash，像素不反馈给控制器。
- 新增无 renderer 的 scene preflight，区分 visual-visual、visual-collision proxy、active contact
  和白名单装配界面。

## direct_mount_v1 的物理修改

```text
platen visible size:     1.60 x 1.10 -> 2.00 x 1.40 m
tabletop size:           0.65 x 0.60 x 0.06 m（不变）
tabletop position:       (+0.18, 0, 0.299) m
Panda base position:     (-0.47, 0, 0.009) m
mount:                   RethinkMount -> NullMount / direct link0 mount
removed mount mass:      274.5941 kg
physics signature:       a1849e266a9ac705dba9abec5be8ece9e1443ad173c063cb24198a62141f0591
current scene hash:       8a23e7c39b1e182b4e70171eb6631a866000e9163f4ff2db445fa457034a7dde
current geometry hash:    a70750e0a6f8cd3615f222525b823d9fcb3418caf0948a15caf61f1bc0336395
```

工作台显式 32 kg、局部惯量、隔振 k/c、preload、接触参数和 deck driver 参数保留。
349 g preload 和新 arena 六轴隔振测试通过；21 个 safe-envelope clearance 样本通过。
但完整任务准入未通过：V0/Gamma=0.15/state000 的 500-step direct-mount demo 中 Can 倾倒，
以 `horizon_exhausted` 结束。后续纯外观修改保持上述 physics signature 和 20-step fixed-action
trace 不变，不能替代 10-state Gamma=0 solvability/controller 适配。

## 测试、打包与文档修改

新增/修改的实现与资产：

```text
robosuite/models/assets/shakebench_scene_visual_v1.json
robosuite/models/assets/shakebench_scene_direct_mount_v1.json
robosuite/models/assets/shakebench_geometry_direct_mount_v1.json
robosuite/models/assets/textures/shakebench_*.png
robosuite/models/assets/textures/shakebench_lab_materials.md
robosuite/utils/shakebench_scene.py
robosuite/utils/shakebench_geometry.py
robosuite/models/arenas/shakebench_arena.py
robosuite/models/assets/arenas/shakebench_arena.xml
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/scripts/shakebench_scene_preflight.py
robosuite/scripts/shakebench_run_oracle.py
robosuite/demos/demo_shakebench_oracle_video.py
setup.py
```

新增/修改的验证与说明：

```text
tests/test_shakebench_scene.py
tests/test_shakebench_scene_finishes.py
tests/test_shakebench_direct_mount.py
tests/test_shakebench_oracle_demo_video.py
docs/phase_07_5a_*_report.md
docs/phase_07_5a_*_manifest.json
docs/lighting_benchmark_references.md
docs/demos.md
docs/integration_map.md
docs/robosuite_benchmark_design_tree.md
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_09_knee_official_eval.md
```

报告记录 scene/arena/demo/environment 测试、Black、isort、`git diff --check`、独立 wheel/sdist
加载、headless environment 和 NVIDIA EGL 渲染通过。最终场景改动仍未提交，相关 report/manifest
中的 `worktree_dirty=true` 不能当成最终发布 handoff。

## CPU 运行准备的实际状态

Phase 7.5B 提示词记录了 i9-14900HX 的 8 P-core + 16 E-core 拓扑、P-core 首 sibling
`0,2,4,6,8,10,12,14`、BLAS/OMP 单线程、spawn worker、atomic result、resume/retry 和
scientific parity 设计。旧 workspace 运行了 6 个 40-step JSON 探针：默认/绑核各一项，
V0/V3 串行/并行各两项。

这些 JSON 没有 wall-clock/resource 字段，绑定旧 `dd6fe2ed...` 和 canonical 模型；仓库中没有
`shakebench_batch.py`、正式 batch CLI、benchmark plan、throughput report/manifest 或经 A/B
等价性门保留的热路径优化。因此当前只能确认“短任务可并行启动”，不能确认单回合加速、
完整回合吞吐、1/2/4/8 worker scaling 或 Phase 7.5B 完成。

## Phase 8 提示词调整

新版 `docs/shakebench_prompts/phase_08_protocol_scorecard.md` 保留该 fail-closed 准入记录；
direct-mount 已完成第一段，现可从 committed-state 协议开始执行：

1. 已对 `direct_mount_v1` 完成 static/preload/transfer/contact/task/controller、10-state
   Gamma=0、matched smoke、determinism、actuator、package/runtime 门，并生成
   `phase08_authorized=true` handoff；
2. 现在再冻结 400 official + 100 knee states，建立 CPU batch、EpisodeResult/RunManifest、
   scorecard 和 MDE；仍不运行 knee/official evaluation。

这样既保留历史 Phase 7 的有效性，也不会把纯背景变化、真实物理变化和尚未完成的 CPU
优化混成同一个授权结论。
