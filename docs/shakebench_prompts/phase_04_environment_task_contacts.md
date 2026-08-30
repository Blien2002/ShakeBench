# Phase 04 Prompt：注册 VibrationPickPlaceCan、任务几何与接触语义

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接在现有 manipulation 目录内新增标准环境文件并修改 `robosuite/__init__.py` 注册；实现 Can、开放起点、浅目标箱、contact metrics 和 success evaluator；不调 controller 追求成功，不新建子文件夹。

## 前置与必读

- Phase 03 arena/isolator probes 通过。
- 读取 `docs/integration_map.md` 及 `docs/robosuite_benchmark_design_tree.md` 第 4、7、8 节。
- 审计 `Lift`、`PickPlace`、`ManipulationTask`、`CanObject`、placement sampler 和 environment registration 实际源码。

## Environment 集成

新增（均在现有目录内）：

```text
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/utils/shakebench_metrics.py
tests/test_environments/test_vibration_pick_place_can.py
```

- 直接继承 `ManipulationEnv` 或 Phase 00 确认的最小合适基类；不要继承 BinsArena 的双 bin 语义或硬编码 cube 逻辑。
- 使用 `ShakeBenchArena`、Panda、standard gripper、CanObject 和 Phase 02 deck processor。
- 在 `robosuite/__init__.py` 的 manipulation imports 中导入新环境，使 metaclass 注册并可由 `robosuite.make("VibrationPickPlaceCan", ...)` 创建。
- 新环境参数默认不影响任何现有 env；初始阶段只支持单 Panda 并 fail closed。

## Task topology与几何

```text
deck
├── Panda base
└── isolated worktable
    └── shallow target container

world
└── Can free body
```

```text
Can nominal XY      = [-0.10,-0.13] m in worktable frame
target center XY    = [-0.10,+0.17] m
target outer XY     = [0.18,0.16] m
wall thickness      = 0.008 m
inner XY            = [0.164,0.144] m
wall height         = 0.035 m
bottom thickness    = 0.012 m
```

Can：复用 robosuite 视觉/碰撞几何，显式 mass=0.349 kg，按 collision envelope 等效圆柱显式 inertia，COM/mass/inertia 编译后断言，v0 yaw=0。

目标箱：一个 bottom+四个 named wall geoms，rigid child of worktable，不新增 freejoint 或 hidden inertia。

## 显式contact pair

```text
Can ↔ table/target bottom/walls: sliding_mu=0.30
Can ↔ Panda finger pads:         sliding_mu=1.00
```

`condim`、torsional/rolling friction、margin/gap、`solref/solimp` 保持 provisional 到 Phase 06。编译后证明 pair 参数没有污染 finger pads/其他 geoms。

## Metrics与success

实现 robot-base/worktable/target frame 的 Can pose/twist，以及 table slip、first-slip、in-hand translation/rotation slip、contact-loss、per-interface force/wrench/impulse/penetration 和 driver/table response。

success 连续 0.50 s 满足：

```text
Can collision horizontal support fully inside target inner footprint
supported by target bottom
no finger–Can contact
relative linear speed <0.02 m/s
relative angular speed <0.20 rad/s
illegal penetration <0.50 mm
```

使用 collision support points 和 target-local frame；允许 Can 高于浅箱壁。reward 可调用同一 success latch，但不得产生第二套成功定义。

## 本阶段验证

- environment registration / `robosuite.make`；
- compiled topology、Can inertial、target dimensions 和 contact roles；
- Gamma=0 passive settle；
- open-table free-slip probes；
- target bottom/wall contacts；
- state injection 覆盖 containment、wall-straddling、support、speed、release、penetration 和 0.50 s latch 边界；
- frame invariance under deck/table motion；
- existing Lift/PickPlace/all-environments tests 无回归。

## 完成条件

1. 新环境可 headless reset/step/close；
2. topology 和每个 contact role 可由 compiled model 审计；
3. evaluator 每个 subcondition 有正负边界测试；
4. passive scene 稳定且无非法 penetration；
5. 输出 `docs/phase_04_report.md`；
6. 未实现 tier observables 或调整 controller 获得 task SR；未新建目录。

完成后停止。
