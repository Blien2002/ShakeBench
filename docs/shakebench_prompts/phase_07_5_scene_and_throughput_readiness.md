# Phase 7.5：场景修复、桌面最小变更与实验吞吐准备

本阶段由用户在查看 demo 后新增，位于 Phase 7R6.1 与 Phase 8 之间。
目标：恢复原 ShakeBench 振动实验室的可辨识场景，消除工作台与机器人底座的穿透，
并为本机及 8×A100 服务器建立可复现的批量执行方案。交付场景、视频、可运行的
CPU 批量工具、性能报告和 GPU 可行性结论；不执行 knee 扫描或正式评测。

先完成最小且可验证的修改。桌面尺寸由几何与任务需要决定，不依据成功率调优。
既有 R6/R6.1 发布证据作为历史基线保留；新场景是否能沿用证据由实际影响决定。

## 1. 输入、工作副本和实施顺序

必读：

- `docs/robosuite_benchmark_design_tree.md`、`docs/robosuite_benchmark_v0_spec.md`
- Phase 03/04/06/07 reports 及 R6/R6.1 manifests
- `docs/shakebench_prompts/phase_08_protocol_scorecard.md`
- `docs/shakebench_prompts/phase_09_knee_official_eval.md`
- `docs/agents/issue-tracker.md`（若存在）以及 GitHub issues #1、#2
- `robosuite/models/arenas/shakebench_arena.py`、对应 arena XML
- `vibration_pick_place_can.py`、`shakebench_run_oracle.py`、`shakebench_metrics.py`
- 本地尚未提交的 `demo_shakebench_oracle_video.py`、相关测试与 demo sidecar

已审基线为 `4238e37cc3b57d626a37f7950df974fc10b0d1e8`。
检查远端最新状态与该基线的差异，保留他人后续工作；不要强制要求远端永远等于旧 SHA。
当前 `/home/miracle04/Desktop/ShakeBench` 可能仍是旧 dirty recovery workspace。
创建可长期保留的独立 branch/checkout，记录绝对路径与基线；按文件移植未提交的 demo、
tracker 和本提示词，核对差异，不覆盖原目录。最终交付可应用补丁或明确工作分支。

原场景参考位于 `/home/miracle04/Desktop/ShakeBench_backup/`：

- `src/shakebench/models/scene.py`
- `src/shakebench/models/arenas/room.py`
- `src/shakebench/models/supports/shaker.py`
- `src/shakebench/models/visual_assets.py`
- `configs/room.yaml`、相关 textures 与 license/attribution
- `out/benchmark_v2_vibrating_resolved.mp4`

用户已要求恢复原场景，允许只读检查该备份。读取源码、配置和参考帧，记录 source hash。
正式运行不得依赖 backup 路径，也不得直接引入 Isaac Lab/USD/Torch 场景依赖；将选中的
资产和必要几何定义移入现有 robosuite 目录并记录来源。不得复制整个备份或 out/。

实施顺序：基线捕获 → 场景/布局修复 → 受影响验证与视频 → CPU 吞吐优化与批量工具
→ 服务器部署方案和可选 GPU 探针 → Phase 8 交接。场景与性能改动分开提交、分别验证。

## 2. 已知事实和不得沿用的推断

此前检查发现：

- 当前与原场景 tabletop 均为 `0.65 × 0.60 × 0.06 m`；场景尺度差异还来自布局、
  桌高、mount 和缺失的约 `1.60 × 1.10 m` platen。
- 当前大型 `RethinkMount` 的 pedestal feet 与左桌腿/支座的几何距离约为负数，
  一个探针最小值约 `-0.0875 m`。视觉件为 non-contact，所以常规 contact gate未发现。
- 该数值含 primitive collision proxy 与 mesh 凸包距离，不能一律解释成真实视觉
  mesh 的穿透深度；必须区分 proxy overlap、visual overlap、active physical contact。
- `base_types` 参数曾被父类调用中的 `base_types="default"` 覆盖，修改时核对真实路径。
- 单次 V0、Gamma=0.15、205步完整 episode曾约150秒；每policy step含250个0.2ms子步。
- 5步 profile 中 success/contact 提取约占47%，说明 Python 热路径值得测量。

这些只是诊断线索。零动作环境与成功 oracle 不是同一条轨迹，不能从145.8秒与150.1秒
相减推断“记录仅占3%”；4个5步进程同时结束也不能证明完整回合并行近线性。
此前8-worker耗时估算并非实测承诺。重新测量，不将这些估算写入正式吞吐结论。

## 3. 默认路线：保留任务桌面，修复可见布局

默认保留：tabletop collision 尺寸、工作台root/COM/弹性中心、机器人base变换、
Can与target的工作台局部坐标、质量惯量、激励/隔振器/接触/控制参数。
先调整桌腿、脚板、横撑的支撑布局，使它们避开现有mount的实际占用体积。
不得隐藏重叠的碰撞体，或用较小外观包住仍然突出的巨大碰撞proxy来制造通过截图。

建立场景角色与变换关系：

```text
world：实验室、地坑、护栏、基础与固定执行器锚点
dynamic deck：可见振动platen、机器人安装结构
isolated worktable：任务桌面、target、桌架上部
decorative shaker：根据actual deck pose更新杆件几何
free Can：保持现有自由体与接触驱动
```

迁入可见platen、shaker基础与六杆外观、坑位/安全边界、护栏及必要实验室设备；
设备详情按参考图选取，家具数量不设硬门。原platen尺寸作为候选，需容纳当前robot
mount和桌架，不能在平台边界外留下悬空脚板。场景配置集中管理布局、材质、相机。

platen跟随actual dynamic deck，工作台跟随actual isolated body；decorative杆件
从固定端与运动端求端点/长度，明确不是新的驱动力学机构。不得改变deck质量惯量，
不得引入并联的六杆约束或将dynamic deck替换成kinematic parent。
接入renderer-only geoms或经验证不影响物理的MJCF visuals；避免零质量动态body、
自动惯性推导和隐藏mass。地坑不能通过误删物理floor改变物体跌落行为。
隔振器外观须表现真实连接两端，刚性桌腿不得视觉上贯穿隔振支座并形成旁路。

输出 scene semantic inventory：每个新增/移动件属于world/deck/worktable哪个frame，
visual与collision各负责什么，是否影响inertial、contact mask、sensor或evaluator。

### 3.1 默认场景配置与命名

新增平铺资产 `robosuite/models/assets/shakebench_scene_visual_v1.json`，禁止把布局常量
散落在XML、environment和demo三处。至少记录schema/version、source paths及SHA-256、
room/platen/pit/Stewart/table-support/camera配置和`physics_effect=false`声明。
实现可按编译模型调整Z基准，但以下是默认起点，任何偏离都在报告中解释：

```text
room envelope              = [6.00, 5.00, 3.00] m
pit opening XY             = [2.05, 1.55] m
pit depth                  = 0.78 m
pit border width           = 0.16 m
guardrail height/thickness = 0.72 / 0.035 m
platen size                = [1.60, 1.10, 0.08] m
platen nominal top         = current table/base support plane
Stewart base ellipse       = [0.85, 0.60] m semi-axes
Stewart platen ellipse     = [0.62, 0.40] m semi-axes
joint-pair spread          = 20 deg
leg nominal/min/max        = 0.82 / 0.67 / 0.95 m
outer/rod radius           = 0.050 / 0.030 m
```

不要盲目把原Isaac世界Z值复制到当前MuJoCo。先从编译模型读取robot pedestal和
worktable foot的最低支撑点，派生platen nominal top；断言二者在预期装配面附近，
而不是穿入板体。保持现有物理floor geom及其接触属性；如需地坑开口，将物理plane
设为不显示但保持原contact，使用四块non-contact visual floor slab围出开口，避免
无意增加“掉进坑里”的新失败模式。

编译模型至少具有稳定角色名或前缀：

```text
shakebench_platen_visual          # dynamic deck child
shakebench_shaker_foundation_*    # world visual
shakebench_stewart_outer_0..5     # world/lower visual
shakebench_stewart_rod_0..5       # deck/upper visual
shakebench_pit_* / guardrail_*    # world visual
shakebench_control_cabinet_*      # world visual
shakebench_emergency_stop_*       # world visual
shakebench_table_upper_*          # isolated worktable visual
shakebench_table_lower_mount_*    # deck visual
shakebench_camera_overview
shakebench_camera_assembly
shakebench_camera_side
```

platen visual body先在arena中声明为显式role，并通过现有role-based deck processor
挂到generated dynamic deck；不得按XML字符串猜测parent。扩展body handle时保持
`isolated_worktable`和`robot_base`必需角色不变，并在compiled audit中断言platen
是deck直接/间接后代。工作台下部视觉分成deck侧与worktable侧，两侧可有重叠套筒，
但不得存在一根刚性visual从deck跨到isolated table形成旁路。

Stewart杆件优先采用两段式、无运行时Python hook的视觉实现：固定外筒属于world或
foundation，内杆属于deck；用名义端点计算方向和重叠，并检查safe envelope内不脱节。
若两段方案在实际姿态不可接受，再引入低频renderer-only更新；不得在5kHz物理hook中
为12个装饰body更新pose。不得每帧修改共享`mjModel.geom_size`造成多环境串扰。

控制柜和急停为原场景辨识所需；椅子与工具车可在首次版本省略并明确记录。
视觉资产全部检查license、来源和hash；Isaac/Omniverse远程资产不得未经许可复制。
首版优先使用MJCF primitives和已入库纹理。

### 3.2 穿透修复的具体决策

先输出以下三个compiled AABB/geom-distance候选，而不是直接改桌面：

```text
A: 原tabletop不动，只重排non-contact桌架与deck侧支座
B: tabletop不动，worktable整体恢复原参考布局约+0.18m X偏移
C: 显式选择更合适的ShakeBench robot mount并保持任务相对变换
```

`table_full_size`增大时，Panda的`base_xpos_offset["table"]`会随table length移动base，
桌腿若也随边缘移动，二者间隙可能几乎不变，因此“加大桌面”不是默认修复。
环境当前接收`base_types`却曾向父类硬编码`"default"`；修复接口，使最终选择被编译
审计和geometry profile记录。不得仅为通过截图隐藏`RethinkMount`，同时保留突出到
桌架中的未说明collision proxy。

在不改变物理合同的前提下优先A。若A导致明显不合理的偏置支撑或safe envelope间隙
不足，选择B并触发本提示词规定的task/physics受影响回归。C会改变mount质量、惯量、
碰撞和arm高度，除非A/B均不可用不得选择。报告nominal及safe envelope最小间隙、
对应geom名称和closest points；白名单只允许脚板/支座与platen预期装配接触。

## 4. 如果要改桌面尺寸：最小变更路径

先输出尺寸/布局建议并以默认路线实施。用户尚未指定新的物理桌面尺寸，不任意冻结
`0.8×0.7` 或更大值；提供至多两个有clearance与可达性依据的候选。
若默认路线已满足无穿透和视觉需求，结论可为“不需扩大接触桌面”。

按影响由小到大区分：

| 方案 | 改动 | 能否沿用旧证据 |
|---|---|---|
| A，默认 | 调整non-contact桌架/支座，恢复platen与房间；任务表面不变 | 编译物理合同与运行对照通过后沿用 |
| B，接触表面扩大 | tabletop visual与collision一起扩大，root/局部任务坐标/显式M/I保持 | 新geometry版本，补测接触边缘/可达性/失败语义；不能宣称物理完全不变 |
| C，真实机械尺寸修订 | 桌面尺寸及质量/COM/惯量、mount或相对位置变化 | 重开受影响的动力学与控制验证，冻结新profile |

方案B是用户明确要求扩大“可操作桌面”时的最小科学变更：参数化一个尺寸来源，
保留tabletop中心、厚度、上表面高度和target/Can坐标，不随尺寸自动把robot base后移。
检查现有 `base_xpos_offset["table"](table_length)` 的隐式联动并使相对base变换显式。
新增scene/geometry profile允许新尺寸；原canonical配置仍可回放，禁止全局替换常量。

保留旧M/I意味着“给定惯性的等效刚体加大接触面”，不是声称真实均匀板变大却惯量
不变。该建模解释写入设计文档；新表面可能接住原本掉落的Can，故即使实际接触未
发生在新增区域，也不能把它视为纯视觉变更。禁止扩大visual而留下未标识的悬空接触面。

若采用方案C，明确总质量是否变化。均匀等效长方体参考惯量为：
`Ixx=M*(Ly²+h²)/12, Iyy=M*(Lx²+h²)/12, Izz=M*(Lx²+Ly²)/12`。
若COM/弹性中心不重合，额外处理平行轴与耦合；不能简单套公式到任意真实桌架。
保持目标fn/zeta时按新M/I派生k/c；这是一项新physics profile，不覆盖旧hash。

B/C未选定时，提交候选和影响表即可，不让尺寸选择阻塞默认视觉修复和性能工作。

## 5. 几何与科学回归：只验证受影响部分

检查名义姿态、可达工作空间及已注册safe激励下的相对位移/转角范围；报告采样覆盖
与保守包络，不把有限角点采样叫作完整连续证明。clearance按实体、运动包络与装配
余量决定；20mm可作为初始设计目标，不作为对所有装饰件一刀切的门槛。
白名单列出预期连接面，检查非预期穿透、悬空和非接触部件之间的机械旁路。
convex mesh距离给候选对，复杂非凸结构用更精确代理或可解释的visual检查确认。

默认A须验证：

- 开关visual不改变命名body的M/I、关节、weld、actuator、contact参数及调度；
- 对相同action序列作前后回放，比较命名状态/动力学量，允许新增geom导致ID重排；
- 对冻结10个dev做V0/Gamma=0回归；state000在Gamma=0.15做V0–V3 matched smoke；
- 不以UUID、墙钟、文件路径、新增metadata导致的文件hash变化认定科学轨迹变化；
- 振动场景全景、装配近景、实际成功/失败视频可见，无幅值暗中放大。

B追加：新增/旧桌边的支撑、接触/跌落、目标containment、接近/放置clearance回归；
对发生变化的task geometry重冻结并重跑相关dev证据。C再追加受影响的静力/preload、
六轴transfer、deck实际跟踪、任务接触及控制可达性验证。
不自动重做全部Phase 00–07，也不沿用与新geometry/profile不兼容的scoreable结论。

视频使用已验证的native MuJoCo Renderer/NVIDIA EGL和FFmpeg；GPU探针在允许访问
宿主驱动的环境运行，沙箱内nvidia-smi失败不能证明驱动缺失。帧不进入State policy。
旁注显示仿真时间、实际Γ/运动量和结果；如果另做慢放/放大解释版，明确标注。

## 6. CPU性能：先用同轨迹测量，再优化

本机已知为i9-14900HX、32GB RAM、RTX4070 Laptop 8GB，重新采集可用资源。
服务器只有“8×A100”是已知事实；CPU/NUMA、内存、40/80GB显存、PCIe/SXM、
MIG/共享配额、调度器与存储均需清点，不根据GPU数量假定CPU吞吐。

用预先选定的dev workloads测量startup/reset/compile、excitation、OSC、
physics integration/forward、success/contact、provider、trace、I/O和渲染。
热点profile用于定位；正式吞吐用未插桩运行测量。区分：

- raw mj_step微基准（诊断下界，不代表可评分环境）；
- 完整无renderer oracle回合（实际吞吐依据）；
- 同任务含视频（仅定性展示成本）。

优先优化重复name lookup、重复模型验证、临时对象、支持点/接触数据构造，
预计算冻结解析激励在原时间网格上的值，缓存编译后常量；必要时把纯数值热路径
移入编译函数。每项单独做profile与语义差分。
禁止通过增大dt、降低OSC频率、跳过瞬时success条件、减少solver精度或延迟IMU
采样来实现“无语义改变加速”。优化success fast path须有短暂失稳重置窗口的反例。
model reset复用仅在验证sensor历史、控制器积分态、mocap/warmstart和RNG均重置后启用。

CPU runner默认spawn独立进程，各进程独立env、RNG和输出。控制BLAS/OMP线程避免
进程内线程过量；这类配置也做串并行差分。GPU渲染不在无图像评测worker中初始化。
本机试workers=1/2/4/8，更多worker只在内存/温度/吞吐结果支持时增加；每组固定
同一job清单，含完整成功与失败回合、预热与多次重复。报告jobs/s、P50/P95耗时、
RSS峰值、CPU温度/降频、swap、启动/I/O成本，不预先承诺8倍加速。

## 7. 批量执行与8×A100服务器使用

实现与Phase8共用的episode job runner，不复制第二套state协议/scorecard。
输入为显式job list，包含state_id、tier、Gamma、seeds和配置hash。
job身份不依赖worker/rank、GPU编号或完成顺序。按matched state block分片，
块内tier顺序由预注册规则给出，避免某tier固定在一台机器或某GPU。

要求：

- dispatch前落盘job manifest，单job临时目录，结果原子完成标记；
- resume按job identity和science/config hash判断，拒绝混合版本；
- 单写入者/锁保证重复调度不形成重复计分；聚合排序稳定；
- task/controller失败保留，infrastructure故障按同job重试并记录ledger；
- worker崩溃、超时、取消、重复提交和跨机器归并均有集成测试；
- 用冻结dev与synthetic jobs测调度，不生成或筛选official/knee states。

服务器第一路线是服务器CPU运行classic MuJoCo无渲染任务；只要CPU/RAM适合，
无需GPU迁移便可获益。提供普通CLI和可选Slurm array模板，明确cpus-per-task、
memory、每分片job数、本地scratch与持久归档；CPU任务不默认占用8张GPU。
服务器连接/调度配置可从已有用户配置发现，缺失时只请求必要信息；其余工作继续。
没有服务器执行结果时标`server_validation=pending`，不得编造A100测速。

第二路线是MJWarp批量探针，先1张A100，再2/4/8张，以独立进程分别持有GPU，
每GPU批量多个world。先试batch=1/16/64，资源足够再试256；记录OOM和约束容量。
根据真实模型支持矩阵核验weld/mocap、六轴弹簧/preload、Can mesh/contact pairs、
friction/solver、OSC所需Jacobian/mass matrix、IMU和持续成功判据。

必须将解析激励、子步OSC/控制、sensor时序与success/contact计算一并纳入设备侧
循环或明确测量host往返成本。每个0.2ms子步copy到CPU并调用旧Python逻辑，不能
称作完成GPU加速。外层20Hz policy即便保留CPU也需测批量同步与数据搬运成本。
分别报告cold compile、warm full rollout、host transfer、证据导出和总作业时间。
纯mjw.step吞吐不等于benchmark吞吐。

8张GPU采用独立job分片；通常无需把单个环境跨8卡拆分，也无需为独立rollout设置DDP。
NVLink不自动加速独立任务。先用同总job数测strong scaling，再测每卡同batch的
weak scaling；报告跨卡负载不均、失败早停/长回合、编译重复与容量开销。
吞吐评测使用无renderer模型；本机4070承担开发与少量视频。

MJWarp在本阶段是非计分候选，classic MuJoCo仍是默认official reference。
跨后端的浮点、solver/contact顺序及确定性可能不同；先冻结对照指标/容差再测试，
同后端跨进程重现与CPU/GPU数值一致性分开评价，不要求不合理的跨后端逐位相同。
若GPU能跑但需要修改数值合同，记录兼容性缺口，不能悄悄混入1600回合主结果。
GPU试验失败不阻塞视觉修复或Phase8；若决定将GPU升为official，另列影响范围、
校准与确定性要求，必须在Phase9测量前冻结，且knee与official使用同一后端。

MJWarp探针先限制为：模型加载、解析驱动的短开环物理对照、代表性接触与端到端
闭环可行性评估。基础兼容性不通过就停止扩卡；不要为本阶段重建整个Isaac Lab应用。
必要时比较MJX，但不同时建设两个production GPU backend。
`mujoco.rollout`可用于给定控制序列的CPU线程池诊断；无法直接替代当前闭环OSC、
每子步driver和evaluator，移植前必须保持这些操作的时间语义。

## 8. 耗时模型、交接和有限完成条件

对实际部署配置给出Phase9耗时估计，并列出测量与假设：

```text
T_total = startup/compile + T_knee + T_official + I/O/audit + retry allowance
N_knee = 100 × 实際预注册Gamma候选数量
N_official = 400 × 4
T_stage ≈ N_stage / 实测对应worker与backend的完整episode吞吐
```

不同Gamma、tier、早停/失败长回合分别估计；knee各Gamma阶段按扫描协议顺序执行。
如用户未指定目标，用“24小时内完成一次预计knee+official”作为可调整规划目标，
不是科学资格门。未达目标时交付实测最佳配置、预计耗时和进一步优化成本。

建议代码落点（允许结合现有代码精简，不要求创建所有文件）：

```text
robosuite/models/arenas/shakebench_arena.py
robosuite/models/assets/arenas/shakebench_arena.xml
robosuite/utils/shakebench_scene.py
robosuite/utils/shakebench_batch.py
robosuite/scripts/shakebench_scene_preflight.py
robosuite/scripts/shakebench_benchmark_throughput.py
robosuite/scripts/shakebench_run_batch.py
robosuite/demos/demo_shakebench_oracle_video.py
tests/test_shakebench_scene.py
tests/test_shakebench_batch.py
docs/phase_07_5_report.md
docs/phase_07_5_manifest.json
```

产物：修复代码、配置及来源记录、原/新场景对照图和MP4、clearance报告、
compiled-physics差分与受影响回归、串并行一致性、吞吐JSON/表格、服务器启动模板、
GPU兼容性与实測或pending状态、Phase8/9受影响文档更新。
视频/raw/profiling/install trees放gitignored out或外部存储，Git只保留compact记录。
复用已有evidence验证工具；多episode index问题应按字段路径/episode ID表示digest，
不要递归把不同层级的合法hash压成一个值。现有冻结archive保持可追溯。

完成条件仅四组：

1. **场景可用**：振动台可辨识，连接关系可解释，非预期穿透已修复且关键帧人工可审。
2. **实验不混淆**：A的物理/任务语义对照通过；B/C若选用，影响表与新版本回归完成。
3. **批量可用**：CPU runner完成真实dev回合，串并行一致且故障/resume/重复处理正确。
4. **交接明确**：本机实测与服务器计划可复现，GPU状态独立记录，选定Phase8场景与
   Phase9候选执行配置；服务器不可用或GPU试验失败不得伪造PASS或拖住其余交付。

分别记录 `scene_ready`、`science_compatibility`、`cpu_batch_ready`、
`server_validation`、`gpu_candidate_status` 和 `phase08_ready`，不要让可选GPU项
决定整阶段BLOCKED。Phase8实施可以在服务器验证pending时继续；正式测量前必须
在实际目标硬件重验执行与数值兼容性。不要运行official/knee样本提前看成绩。

旧R6.1 gate验证的是历史handoff；它拒绝后续代码变化是预期行为。保留旧tag/release，
为7.5新增清晰的场景/执行版本与baseline关系，不能改旧verifier让旧PASS覆盖新代码。
本任务无需为了文档再做历史重写、强推或新建多轮gate release。

## 9. 官方资料与版本约束

以下资料用于设计依据；实施时固定实际版本并核对支持矩阵，不能把不同版本的
classic MuJoCo、MJWarp、Newton及Isaac Lab能力合并为一个保证。

- MuJoCo Warp：批量NVIDIA GPU仿真、容量/graph capture、数值与确定性差异：
  https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
- MuJoCo Python rollout：给定controls的CPU原生线程池执行：
  https://mujoco.readthedocs.io/en/stable/python.html#rollout
- MJWarp源码与当前兼容性信息： https://github.com/google-deepmind/mujoco_warp

执行本提示词时，完成上述范围的实施、验证和交接；Phase 8–10 的正式任务另行执行。
