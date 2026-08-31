# Phase 04：VibrationPickPlaceCan、任务几何与接触语义

状态：**Phase 04R remediation PASS；Phase 05 handoff = PASS**

`docs/phase_04_task_contract_remediation_report.md` supersedes the initial
Phase 04 contract details below where they conflict. In particular, the old
site-derived `0.100 m` Can height is not a physical authority anymore.

本阶段从工作区已有的 Phase 03 dirty state 继续，保留了其他阶段的修改；最近已提交基线为 `02aa46f9`。实现没有调整 controller 来追求 task success，也没有创建新的 ShakeBench 子目录。

## 1. 实现落点

| 路径 | 内容 |
| --- | --- |
| `robosuite/environments/manipulation/vibration_pick_place_can.py` | 单 Panda、标准 PandaGripper、开放桌面 Can、浅目标箱、Phase 02 deck driver 装配、任务生命周期和 success seam |
| `robosuite/utils/shakebench_metrics.py` | moving-frame pose/twist、碰撞支撑点、接触接口 metrics、force/wrench/impulse/penetration、slip/contact-loss、driver/table response、连续 success latch |
| `robosuite/__init__.py` | 显式导入环境，依靠 `EnvMeta` 注册 `VibrationPickPlaceCan` |
| `tests/test_environments/test_vibration_pick_place_can.py` | 注册、headless lifecycle、compiled topology/inertia/geometry/contact、passive settle 和 target contact probes |
| `tests/test_shakebench_metrics.py` | success subcondition boundaries、0.50 s latch、moving-frame invariance 和 artifact schema |
| `tests/shakebench_phase_04_environment.json` | 本阶段机器可读 evidence 摘要 |

为使 Phase 02 driver 在第一次编译前安装，环境使用既有 `ManipulationEnv` 的 `load_model_on_init` seam；默认行为仍保持 eager reset。Phase 00 registry bootstrap 和通用 environment smoke 已同步支持新环境的 single-Panda fail-closed 约束。

## 2. Compiled topology

编译后的 task topology 为：

```text
world
├── deck_driver                 # mocap, no contact geom
├── deck                       # free body + equality weld
│   ├── robot0_base            # Panda rigid subtree
│   └── worktable              # 6-DoF isolator + target geoms
└── can_main                   # world child + can_joint0 freejoint
```

`worktable` 和 `robot0_base` 通过显式 role handle 交给 `ShakeBenchDeckXMLProcessor`；Can 没有被 reparent 到桌面。目标底和四面墙是 `worktable` 的 direct geom children，没有 freejoint 或新增 inertial。compiled audit 对 parent graph、freejoint、target body 和 deck contract 均通过。

## 3. Can inertial and placement

Can 复用仓内 `CanObject` 的 mesh visual/collision geometry。任务层移除 mesh density 的隐式质量，添加：

```text
mass       = 0.349 kg
COM        = [0, 0, 0] m
inertia    = [0.00024106572568945823,
              0.00024106572568945823,
              0.00010986473347417683] kg·m²
```

唯一 collision-envelope authority 是从 canonical compiled Can collision geom
`can_g0` 的 support vertices 提取的 envelope：

```text
radius       = 0.02509177806572465 m
height       = 0.08000000550552341 m
lower z      = -0.040297003330440104 m
upper z      =  0.03970300217508332 m
source hash  = c5332fb76e8b2f8c36fe10ac99e51f5e79c189e248d626dacd54cc983629e669
algorithm    = shakebench.can_collision_envelope.compiled_support.v1
```

等效圆柱惯量只使用上述 compiled envelope 的 radius/height；编译后自动断言 body mass、COM、principal inertia 和单个 freejoint。v0 初始 yaw 固定为零。stock placement sites 仍保留用于 robosuite API 兼容，但不再进入惯量或 placement correction。

placement sampler 使用同一 envelope 的 `lower z`，correction 为
`-lower_support_z = 0.040297003330440104 m`；不存在第二个 site-based
placement correction authority。构建和 audit 都会检查 source geom list、source
hash、radius、height、lower/upper support、inertia 和 correction 的漂移。

默认 sampler 是既有 `UniformRandomSampler`，相对于 worktable top 使用：

```text
Can start XY = [-0.10, -0.13] m
rotation     = 0
source       = open table; no source wall/bin
```

## 4. Target container

目标箱由一个 bottom 和四个 named collision wall geoms 构成，并随 isolated worktable 刚性运动：

```text
center XY        = [-0.10, +0.17] m
outer XY         = [0.18, 0.16] m
inner XY         = [0.164, 0.144] m
wall thickness    = 0.008 m
wall height       = 0.035 m
bottom thickness  = 0.012 m
```

compiled audit 同时检查五个 geom 的 parent、local position 和 half-size；target bottom 为 `size=[0.09,0.08,0.006]`，x-walls 为 `size=[0.004,0.08,0.0175]`，y-walls 为 `size=[0.082,0.004,0.0175]`。display-only visual geoms 的 contact bits 为零。

## 5. Contact roles

任务只添加显式 Can pairs：

```text
Can ↔ table_collision                 1 pair, sliding μ = 0.30
Can ↔ target_container_bottom/walls   5 pairs, sliding μ = 0.30
Can ↔ Panda finger pads               2 pairs, sliding μ = 1.00
```

Can 使用专用 contact bit，listed partner geoms 使用对应 affinity；未列出的 world/robot geoms 不会通过默认 bit test 生成 Can contact。compiled pair friction audit 通过，且 geometry friction 未被 pair 参数改写：Can `0.95`、table `1.0`、Panda pads `2.0` 的原始 sliding friction 均保持可审计。

`condim`、torsional/rolling friction 的最终选择、margin/gap 和 pair `solref/solimp` 没有在本阶段冻结；它们保留 compiled provisional defaults，由 Phase 06 physics freeze 负责。

Contact report 对每个 active contact 保存 interface、geom names、point、distance/penetration、contact-frame force、Can-frame world force、world wrench、impulse 和 wrench impulse，并提供 table/object、target/object、finger/object 聚合。当前 sample 字段统一为 `table_contact_present`、`target_bottom_contact_present`、`target_wall_contact_present` 和 `finger_can_contact_present`；stateful 丢失字段只有 `finger_contact_loss_after_grasp`，不再输出歧义的 `contact_loss`。

## 6. Metrics and success

`shakebench_metrics.py` 的 moving-frame seam 返回 robot-base、worktable 和 target-local 的 Can pose/twist。相对线速度包含 rotating-frame transport term，因此 deck/table 共同刚体运动时 pose/twist 不产生虚假 slip。

状态 metrics 包含：

- table-relative slip distance/speed 和 first-slip timestamp；
- in-hand translation/rotation slip；
- instantaneous `finger_can_contact_present` 与 stateful `finger_contact_loss_after_grasp`；
- per-interface force/wrench/impulse/penetration；
- deck/driver trace 和 table pose/twist/acceleration/relative-to-deck response。

success evaluator 只有一套定义，并由 reward 直接调用同一 latched evaluator。每个 policy-boundary sample 需要连续满足 `0.50 s`：

```text
collision support points fully inside target inner footprint
target bottom contact + target-local upward support force > 0.001 N
Can lower support z >= -0.00050 m relative to target bottom plane
no finger–Can contact
relative linear speed < 0.02 m/s
relative angular speed < 0.20 rad/s
illegal penetration < 0.50 mm
```

containment 使用 compiled collision mesh support points 的 target-local XY 投影；不使用中心点、visual mesh 或 world AABB。浅箱不添加“Can 必须低于墙”的条件。speed、penetration 使用严格小于边界；containment 的几何边界允许落在内边界上。成功窗口在不连续样本时清零，完整窗口后锁存。

`use_object_obs=True` 时的 Can observables 只提供显式
`can_pos_robot_base` / `can_quat_robot_base`（以及 target-local
`can_to_target_pos`），不再提供 world-frame `can_pos/can_quat`，为 Phase 05
保留明确的 robot-base common-state seam。

本阶段没有实现 tier observables、IMU、V0–V3 providers、Oracle controller、committed states、scorecard 或 task SR 优化。

## 7. Verification

Phase 04 + registry bootstrap：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_metrics.py \
  tests/test_environments/test_vibration_pick_place_can.py \
  tests/test_shakebench_config.py
```

结果：Phase 04R 后 `38 passed`（含 artifact lock、support-force、contact-loss 和 policy-frame regressions）。

Phase 03 handoff regression：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_arena.py \
  tests/test_shakebench_isolator.py \
  tests/test_shakebench_transfer_remediation.py
```

结果：`22 passed`。

不依赖 EGL 的既有上游集合：

```bash
MUJOCO_GL=disable PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_controllers/test_composite_controllers.py \
  tests/test_environments/test_action_playback.py \
  tests/test_grippers tests/test_robots --tb=short
```

结果：`255 passed, 58 skipped`。

静态检查：`py_compile`、`pyflakes`、`flake8 --ignore=E203,W503 --max-line-length=120`、`isort --check-only` 和 `git diff --check` 通过。

发布资产：`robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png`
已纳入 Git index，SHA-256 为
`87a5478e7325b7d79fc8073afefd3ac7c44b44d6c804ecb586c1641d8157e162`；source
distribution clean-compile smoke 通过，且实际被 `shakebench_arena.xml` 引用。

当前主机的 `tests/test_environments/test_all_environments.py`、`test_camera_transforms.py` 和 `test_env_determinism.py` 仍依赖 off-screen EGL；运行容器缺少可用 EGL device display / `swrast_dri.so`，因此这些图形相关 smoke 不能作为本机 PASS。该限制与 Phase 00/03 报告一致；新环境自身的 no-renderer headless lifecycle 已通过。

## 8. Handoff and deferred gates

Phase 04 完成并交接 Phase 05。后续仍需按阶段完成：

1. canonical IMU 和 V0–V3 policy key-set / privilege isolation；
2. physics-only contact、timestep、solver 和 isolator freeze；
3. controller/gripper pilot、Oracle controller 和 Gamma=0 task solvability；
4. committed states、Gamma knee、official scorecard 与 release audit。

`tests/shakebench_phase_04_environment.json` 保存本阶段 evidence 摘要；其中 `model_timestep=0.0002 s`、deck `eq_solref=[0.0004,0.5]` 和其它 solver/contact 字段均标为 provisional，不代表 Phase 06 official freeze。
