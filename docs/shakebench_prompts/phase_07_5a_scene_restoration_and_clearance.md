# Phase 7.5A：原振动场景恢复与装配间隙修复

本阶段位于 Phase 7R6.1 与 Phase 8 之间，只修复场景和装配几何。完成后必须得到：

1. 可辨识的 ShakeBench 振动实验室与振动台；
2. 工作台、隔振支座、platen 和 Panda mount 没有非预期穿透或悬空；
3. 场景frame归属与物理拓扑一致；
4. 一段经过抽帧检查的正式成功demo；
5. 明确结论说明旧Phase 6/7科学证据能否沿用。

本阶段不实现CPU/GPU批量执行，不运行knee或official states，不调控制器，不根据成功率
选择场景尺寸。性能工作属于后续Phase 7.5B/8。

## 1. 基线、工作副本与必读资料

已审远端基线：

```text
4238e37cc3b57d626a37f7950df974fc10b0d1e8
```

开始时获取远端最新master并记录SHA。若远端已有后续提交，以远端最新提交为基线，
先审查差异；不得reset、覆盖或强推他人工作。当前
`/home/miracle04/Desktop/ShakeBench` 可能仍是旧dirty recovery workspace，禁止在其中
直接大范围同步或提交。创建独立、可长期保留的clean clone/branch，记录路径和commit。
把用户已批准但尚未提交的demo、tracker和本提示词按文件移植，逐项核对，不使用`git add -A`。

必读：

- `docs/robosuite_benchmark_design_tree.md` 第2、4、8、9节；
- `docs/robosuite_benchmark_v0_spec.md`；
- Phase 02/03/04/06/07 reports与当前R6/R6.1 manifests；
- `robosuite/models/arenas/shakebench_arena.py`；
- `robosuite/models/assets/arenas/shakebench_arena.xml`；
- `robosuite/utils/shakebench_deck.py`、`shakebench_isolator.py`；
- `robosuite/environments/manipulation/vibration_pick_place_can.py`；
- `robosuite/demos/demo_shakebench_oracle_video.py`及其sidecar/test（如已存在）。

用户已明确要求参考原ShakeBench场景，允许只读检查：

```text
/home/miracle04/Desktop/ShakeBench_backup/src/shakebench/models/scene.py
/home/miracle04/Desktop/ShakeBench_backup/src/shakebench/models/arenas/room.py
/home/miracle04/Desktop/ShakeBench_backup/src/shakebench/models/supports/shaker.py
/home/miracle04/Desktop/ShakeBench_backup/src/shakebench/models/visual_assets.py
/home/miracle04/Desktop/ShakeBench_backup/configs/room.yaml
/home/miracle04/Desktop/ShakeBench_backup/out/benchmark_v2_vibrating_resolved.mp4
```

记录实际读取文件的SHA-256和用途。正式package/runtime不得引用backup绝对路径。
不得复制整个backup、Isaac Lab依赖、USD运行时或历史out目录。原场景只作为布局、造型、
材质和相机参考；用robosuite现有MJCF/primitive接口重新实现。

## 2. 当前已复现的问题

在修改前把以下检查固化为可重复preflight/test：

### 2.1 场景缺失

当前编译模型有`deck_driver`和`deck`物理body，但缺少原场景的主要视觉角色：

```text
1.60 × 1.10 m visible vibration platen
shaker foundation
six Stewart / actuator legs
pit and safety border
guardrails
control cabinet
emergency stop
```

当前phenolic tabletop、方管桌架、简单墙面属于已迁入部分；报告不得再写成“完全没有
迁移”，应明确partial integration及本轮新增内容。

### 2.2 桌架与机器人底座重叠

当前Panda默认`RethinkMount`包含大型pedestal、controller box和collision proxy。
已观察到工作台左侧桌腿/支座与pedestal区域重叠；一个primitive/convex距离探针得到：

```text
minimum signed candidate distance ≈ -0.0875 m
```

负值是可靠红灯，但不能自动解释为真实非凸visual mesh穿透深度。preflight必须分别输出：

- visual ↔ visual候选；
- visual ↔ collision proxy候选；
- active physical contacts；
- geom/body名称、frame、最近点和距离方法。

工作台visual geoms目前大多`contype=conaffinity=0`，所以contact gate不会发现视觉穿透。
新增独立scene-clearance gate，不把“没有contact”当作“没有视觉重叠”。

### 2.3 尺寸判断

原场景和当前canonical tabletop均为：

```text
[0.65, 0.60, 0.06] m
```

原场景更宽阔主要因为约`1.60 × 1.10 m`的platen、worktable布局和不同robot base造型。
环境当前会根据`table_full_size[0]`调用Panda table base offset；单纯增大桌面会同时移动
robot base和桌腿，未必增加间隙。不得把“增大桌面”预设成修复答案。

## 3. 冻结场景的frame与物理含义

实现前写出machine-readable scene inventory，并按以下所有权构建：

```text
world
  laboratory shell / visual floor slabs
  pit / safety border / guardrails / cabinet / emergency stop
  shaker foundation and lower Stewart segments

dynamic deck
  visible vibration platen
  robot mounting assembly
  deck-side isolation base plates
  upper Stewart segments

isolated worktable
  canonical tabletop collision and visual
  target container
  table upper frame / legs / worktable-side isolation sleeves

world free body
  Can
```

需要保持的物理拓扑：

```text
mocap driver -> equality weld -> ordinary dynamic deck
dynamic deck -> rigid robot base
dynamic deck -> six compliant worktable joints -> isolated worktable
Can motion -> gravity/contact/grasp only
```

视觉杆件和装饰不能增加并联约束、隐藏mass或新的task contact。新增visual默认：

```text
group=1
contype=0
conaffinity=0
无joint
无inertial
```

若MJCF编译器会为某个body自动推导惯性，重构为附着在已有物理body上的geom或显式
静态/mocap视觉路径；不得接受自动小质量。视觉开关前后必须比较命名body、joint、weld、
actuator、contact pair、solver/timestep及质量惯量。

## 4. 具体场景配置

新增平铺配置：

```text
robosuite/models/assets/shakebench_scene_visual_v1.json
```

至少记录：schema/version、原参考文件及hash、frame ownership、room/platen/pit/Stewart/
table-support/camera参数、material/texture来源、`physics_effect`、配置自身payload hash。
XML、Python和demo从一个配置authority取值，禁止复制三套常量。

默认作者值：

```text
room envelope              = [6.00, 5.00, 3.00] m
pit opening XY             = [2.05, 1.55] m
pit depth                  = 0.78 m
pit border width           = 0.16 m
guardrail height/thickness = 0.72 / 0.035 m
platen size                = [1.60, 1.10, 0.08] m
Stewart base ellipse       = [0.85, 0.60] m semi-axes
Stewart platen ellipse     = [0.62, 0.40] m semi-axes
joint pair spread          = 20 deg
leg nominal/min/max        = 0.82 / 0.67 / 0.95 m
outer/rod radius           = 0.050 / 0.030 m
```

这些值来自原场景，是MuJoCo实现的初始authoring值。不要直接复制原Isaac世界Z坐标。
从当前编译模型读取pedestal和worktable foot的最低支撑点，派生platen nominal top；
报告派生值和装配误差。脚板允许与platen形成预期接触/贴合，不能埋入板体。

当前无限物理floor必须保留原接触语义。为了显示地坑，可以将其放入不显示的geom group
并用四块non-contact visual floor slabs围出开口，在下方增加dark pit visuals；不得删除
physical floor或让Can失败后出现新的坑底接触。

至少提供稳定名称：

```text
shakebench_platen_visual
shakebench_shaker_foundation_*
shakebench_stewart_outer_0..5
shakebench_stewart_rod_0..5
shakebench_pit_*
shakebench_guardrail_*
shakebench_control_cabinet_*
shakebench_emergency_stop_*
shakebench_table_upper_*
shakebench_table_lower_mount_*
shakebench_camera_overview
shakebench_camera_assembly
shakebench_camera_side
```

控制柜和急停是首版必需场景线索。椅子和工具车可省略，但须写入known omissions。
只复用许可证明确的纹理；Isaac/Omniverse远程资产不得直接复制。优先使用MJCF primitives、
仓库已有phenolic/steel/floor textures和确定性颜色。

## 5. Dynamic deck和隔振外观接入

arena source XML中声明显式`shakebench_platen_visual` role body；通过现有role-based deck
processor将它挂到generated dynamic deck。扩展body handles：

```text
isolated_worktable -> worktable          # required, unchanged
robot_base        -> robot0_base         # required, unchanged
deck_visual       -> shakebench_platen_visual
```

processor/audit必须确认deck_visual是dynamic deck后代。不得在最终XML字符串中按文本查找
`<body name="deck">`并手工拼接，也不得变成mocap parent。

工作台支撑外观拆为两侧：deck侧foot/base plates与worktable侧legs/sleeves。两段式隔振
mount可以在名义位置重叠形成套筒，但不得有单个刚性geom跨越两个frame。

Stewart杆件优先使用两段式visual：lower outer属于world/foundation，upper rod属于deck。
按原ellipse端点计算名义方向、长度和重叠，并在safe运动包络中验证不断开、不穿platen。
不要增加5kHz Python visual hook，不要在运行时逐步修改共享`mjModel.geom_size`。
若必须动态更新杆件，只允许renderer/demo cadence，且headless scoreable路径不注册该hook。

## 6. 穿透修复和桌面最小变更

先在同一preflight中评估：

```text
A. tabletop/body不动，仅重排non-contact桌架和deck侧支座
B. tabletop尺寸不动，worktable恢复原参考布局约+0.18m X偏移
C. 使用显式ShakeBench mount替换默认RethinkMount
```

默认选择A。A必须保持结构合理，不能把四腿挤到一侧或使支撑看起来不稳定。
若A无法在预注册safe motion envelope内取得合理间隙，选择B并运行受影响task/physics回归。
C会改变mount质量、惯量、碰撞proxy和arm高度，只有A/B均不可接受时才采用。

修复`base_types`被硬编码为`"default"`的问题，使构造参数要么真正生效，要么从公共接口
删除并由冻结scene/geometry profile唯一选择。最终mount类型、质量、碰撞体和robot-base
变换进入compiled audit。禁止隐藏Rethink visual后保留未说明的大型proxy来伪造clearance。

clearance报告覆盖nominal与实际pre-registered safe excitation envelope；机械joint极限另列
stress检查。每项记录采样方法和覆盖，不能把有限角点说成连续数学证明。对于primitive使用
MuJoCo geom distance；mesh先以convex/AABB筛选，疑似负间隙再用visual mesh或更精确代理确认。
白名单仅包括预期装配界面，例如foot/base plate贴合platen、套筒内部重叠。

如果用户仍要求扩大可操作桌面，先提供不超过两个候选尺寸与以下报告：

- Panda可达域、初始Can和target margin；
- base offset是否随table length隐式移动；
- 新增接触面是否改变Can跌落/支撑结果；
- 保持显式M/I时的等效刚体解释；
- 重算M/I/k/c时需要重开的physics gates。

用户未给出具体尺寸时不得自行冻结新尺寸。最小物理变更方案B-size定义为：tabletop
visual与collision同步扩大，保持中心、上表面高度、worktable root、Can/target局部坐标和
显式M/I；新建geometry profile，不覆盖旧canonical配置。由于新增接触面会改变失败边界，
必须补测contact/edge/reach，不能称为纯视觉修改。

如果质量、COM、惯量、worktable位置或mount物理发生变化，建立新physics/geometry profile，
按影响重测static equilibrium、preload、six-axis transfer、deck conformance、task contacts、
Gamma=0 solvability和controller reachability。不要自动重跑全部Phase 00–07，也不得把旧R6.1
PASS解释为覆盖新profile。

## 7. 代码落点

优先使用以下现有目录；允许合并实现以保持接口小，但不得把逻辑散进demo：

```text
robosuite/models/assets/shakebench_scene_visual_v1.json
robosuite/models/assets/arenas/shakebench_arena.xml
robosuite/models/arenas/shakebench_arena.py
robosuite/utils/shakebench_scene.py
robosuite/utils/shakebench_deck.py                 # 仅role/audit所需最小扩展
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/demos/demo_shakebench_oracle_video.py
tests/test_shakebench_scene.py
tests/test_shakebench_arena.py
tests/test_vibration_pick_place_can.py 或现有对应environment test
docs/phase_07_5a_report.md
docs/phase_07_5a_manifest.json
```

`shakebench_scene.py`形成一个深模块，外部接口保持在以下量级：

```text
load_scene_visual_config(path) -> SceneVisualConfig
augment_scene_mjcf(arena, config) -> SceneInventory
audit_compiled_scene(sim, config) -> SceneAudit
scene_clearance_report(sim, config, envelope) -> ClearanceReport
```

调用者不传几十个单独几何参数。测试通过同一接口验证配置、编译结构和clearance。
若只用XML即可表达某些静态visual，仍由config/audit统一认证。

## 8. 回归与证据边界

先保存before基线：编译XML/physics signature、命名body的M/I、joint/weld/actuator/contact配置、
robot-base/worktable/target transforms、10个dev state文件hash，以及固定action序列的命名状态
trace。新增geom可能导致整数ID重排；对比稳定名称与数值，不对比raw ID或整个XML字节。

若采用A且所有新增/移动内容均为visual：

1. visual on/off编译physics signature一致；
2. 同初态/同action的命名qpos/qvel/body pose/contact/metrics/observations/actions在注册容差内一致；
3. 10个dev state原文件字节和payload hash不变；
4. V0/Gamma=0 10/10重新运行；
5. state000、Gamma=0.15运行V0–V3 matched smoke；
6. 三个独立进程的科学trace一致；
7. 只把时间、UUID、路径及新增scene metadata造成的外层文件hash变化排除，不忽略科学字段。

若B/C或桌面物理变更，先列出哪些冻结hash失效，再执行第6节规定的扩大验证。
检测到physics、task或controller输出变化时不得把差异归因于“只是视觉”，必须定位来源。

新增测试至少覆盖：

- 配置schema/hash、缺失/非法尺寸/重复角色fail closed；
- platen属于deck，upper table visuals属于worktable，lower mounts属于deck/world；
- visual geoms不增加contact/inertial/joints/actuators；
- visual开关的physics signature不变；
- nominal和safe envelope非预期penetration为空；
- 一个故意移动桌腿进入pedestal的negative fixture能让clearance gate变红；
- floor/pit visual开关不改变物理floor contact；
- 相机能看到platen、worktable、Can、target和robot；
- package wheel/sdist包含scene config、textures和demo script。

## 9. 正式demo与视觉验收

使用native MuJoCo Renderer、NVIDIA EGL和FFmpeg。摄像头帧只用于展示，不进入State policy、
evaluator或scoreable evidence。默认录制：

```text
environment = VibrationPickPlaceCan
robot       = Panda
tier        = V0
Gamma       = 0.15
state       = shakebench-dev-v0-000
resolution  = 1280 × 720（资源不足可640 × 360，同时保留原尺寸记录）
fps         = 20
```

scene config提供overview/assembly/side三相机。主视频使用overview，画面必须同时看到：

- platen与至少部分shaker机构；
- Panda及其真实安装位置；
- isolated worktable与可见支座；
- Can和target；
- pit/safety border或其他明确场景线索。

叠加sim time、tier、Gamma、phase和result。不得视觉放大实际振动；如另做慢放或位移放大解释版，
在画面和sidecar标注倍率。保存MP4、JSON sidecar、起始/抓取/搬运/放置/成功关键帧和视频hash。
人工检查关键帧：无穿透、无悬空、杆件无明显断开、相机不被墙/机器人遮挡、SUCCESS与真实
evaluator一致。旧简化demo保留或标为superseded，不用新文件静默覆盖其provenance。

## 10. 文档、manifest与完成条件

更新设计树：完整实验室视觉从新增用户需求开始，过去只冻结工业worktable视觉；明确物理
拓扑复用与场景外观复用的区别。更新v0 spec、integration map、prompt README及Phase 8/9前置：

- Phase 8 committed states继续复用原10个dev rows/IDs/seeds/poses；
- visual-only scene hash进入run/scene metadata，不改旧state payload；
- geometry/physics profile变化时，Phase 8/9必须绑定新profile并拒绝混合结果；
- Phase 9只接受本阶段选定scene/geometry profile。

`docs/phase_07_5a_manifest.json`分别记录：

```text
scene_ready
clearance_passed
visual_physics_invariant
geometry_variant = A | B | C | B-size
science_compatibility = unchanged | revalidated | invalidated
demo_verified
phase08_ready
```

不得用一个总PASS掩盖子项。只有以下全部满足才可`phase08_ready=true`：

1. 原振动台核心视觉角色已集成且不依赖backup；
2. 编译scene inventory和frame归属正确；
3. nominal/safe envelope没有非预期穿透或悬空；
4. 预期装配接触白名单明确；
5. 所有物理/任务变化均已识别并完成对应回归；
6. 旧dev states和新scene/geometry authority关系明确；
7. demo和关键帧人工验收通过；
8. focused tests、相关ShakeBench non-renderer regression、Black、isort、`git diff --check`通过；
9. package安装后可加载scene config、创建无renderer环境并渲染demo环境；
10. 报告没有运行knee/official states，也没有按任务成功率调场景。

任一必需项失败时只标记对应子项与`phase08_ready=false`，给出直接原因和最小下一步。
旧R6/R6.1 tag/release保持历史可验证，不修改旧verifier让其覆盖新代码。本阶段不做历史重写、
强推或多轮gate release。完成后停止，不开始Phase 8、7.5B或正式大规模实验。
