# Phase 7.5A：直接安装、矮桌架与扩大振动台（候选阶段记录）

本文件记录初始候选装配及其失败诊断；其结论已被
`docs/phase_07_5a_requalification_report.md` 取代。不要使用本文件中旧的
`scoreable=false`、旧 geometry/scene hash 或失败 demo 作为当前 Phase 7 状态。

## 本轮装配

根据用户要求，在现有精细实验室场景上建立 `direct_mount_v1`：

| 项目 | 上一版 | 新装配 |
|---|---|---|
| 振动台可见尺寸 | 1.60 × 1.10 m | **2.00 × 1.40 m** |
| 桌面尺寸 | 0.65 × 0.60 × 0.06 m | 保持 |
| 桌面距振动台上表面 | 0.791 m | **0.290 m** |
| 桌面中心 XY | (0, 0) | **(+0.18, 0) m** |
| 机器人安装 | RethinkMount 带轮底座 | **Panda link0 直接固定于 deck** |
| 地坑开口 | 2.05 × 1.55 m | 2.45 × 1.90 m |

参考原仓库 `config.py`：platen top=0.08 m、robot base=(-0.47,0,0.08) m、worktable COM=(0.18,0,0.34) m。
保留原版 0.29 m 相对台高；按当前 platen top=0.009 m 换算机器人 base=(-0.47,0,0.009) m、tabletop=(0.18,0,0.299) m。
采用四角支撑布局和低位横撑。工作台上部几何属于 worktable，底板属于 deck；两侧 XY 使用各自 frame 正确换算。
护栏降低为 0.46 m，并将急停移至扩大后的护栏外。实验室材质和器材精细模型继续复用。

[全景](../out/phase07_5a/direct_mount_revision/overview.png) · [装配近景](../out/phase07_5a/direct_mount_revision/assembly.png) · [侧面](../out/phase07_5a/direct_mount_revision/side.png)

## 实际物理变化

从模型中移除 RethinkMount 的 controller box、pedestal、feet 及其 collision proxies 和 inertia，总质量减少 **274.5941 kg**。
NullMount 仅留下零质量的空挂接节点。Panda 自身 link0 的 4 kg 惯性与 collision mesh 保留，robot root 仍刚性属于普通 dynamic deck。
新位置影响 Can 的绝对高度、可达域和跌落高度；因此本轮不能标为纯视觉等价。
工作台显式 32 kg、既有 COM 局部惯量、k/c、预紧、接触系数和 deck driver 参数保留。缩短桌架是这些集中参数对应的可视装配，不按 visual geoms 自动重算质量。

历史 canonical 模型默认仍供科学 runner 使用；新装配明确 `scoreable=false`，绑定独立 geometry hash 和 scene hash。
演示 CLI 默认使用新装配，环境需显式传入 `geometry_profile="direct_mount_v1"`。

## 验证结果

- 11 项新装配测试通过：移除旧底座及质量、安装点、四脚对齐、visual on/off 物理与一步状态一致、10 mm 埋入机器人负例、配置冲突拒绝、空载预紧、六轴新 arena 隔振传递。
- 37 项原 scene/arena/demo/environment 回归通过。
- 21 个预注册 safe envelope 采样通过，无非预期穿透。
- Panda link0 mesh 在作者安装基准下方约 0.033 mm 的坐合界面仅对白名单中的 link0/platen 配对接受，上限 0.1 mm；更深的负间隙继续拒绝。
- 349 g 负载：理论压缩 -0.000108403851 m，实测 -0.000108403851 m，误差 2.47e-13 m。
- wheel 独立解包后加载新 geometry/scene authority、创建环境及 native MuJoCo EGL 渲染通过。
- Black、isort 和 `git diff --check` 通过。

演示 evaluator success=False，结束类别 horizon_exhausted，failure_reason=horizon_exhausted。

关键帧显示 Can 已倾倒，200 帧处为 `clearance_retreat`，499 帧仍为 `approach`；未完成放置。
该录像是失败诊断预览，不是正式成功 demo。控制器适配、Gamma=0 新布局可解性和完整任务级科学门控留待后续工作；本轮保留原控制器参数，未按成功率反调装配尺寸。
[失败演示 sidecar](../out/demo/phase07_5a_direct_mount_v5_v0_gamma015.json) · [关键帧](../out/phase07_5a/direct_mount_revision/keyframes.json)

新物理签名：`a1849e266a9ac705dba9abec5be8ece9e1443ad173c063cb24198a62141f0591`。
geometry payload hash：`69e933679eb7ed8a9595bc6e6390fc2eb411ce0bd059d302f2ddb5f5fac1209c`。
scene payload hash：`e1920369d0713497796536d6a9eb6579bfcea896cc222430045be49b7b2d1d5b`。

本次通过的六轴测试针对新 arena 的隔振子系统；完整任务的科学准入仍需针对新 geometry 单独重开。
旧 Phase 6/7 的科学证据保留其历史有效性，不能用于证明新装配。未运行 knee/official states 或进入 Phase 8。

## 复现

```python
VibrationPickPlaceCan(robots="Panda", geometry_profile="direct_mount_v1", ...)
```

```sh
MUJOCO_GL=egl python -m robosuite.demos.demo_shakebench_oracle_video \
  --geometry-profile direct_mount_v1 \
  --output out/demo/phase07_5a_direct_mount_v5_v0_gamma015.mp4

python -m robosuite.scripts.shakebench_scene_preflight \
  --geometry-profile direct_mount_v1 \
  --output out/phase07_5a/direct_mount_revision/scene_preflight.json
```

用 `--geometry-profile canonical` 可复现上一版布局。证据目录：`out/phase07_5a/direct_mount_revision`。
