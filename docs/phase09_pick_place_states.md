# Pick&place 摩擦组合与 Phase 09/09.5 state

当前实现将三种物体和两种桌面交叉为 **一个 `pick_place` 任务的六种 state 配置**。每个 episode 仍只操作一个物体。Phase 09 先执行非计分重认证 pilot；尚未启动 Phase 09.5 knee scan 或 official measurement。

## 资产与接触配置

全部采用仓库已有的 robosuite 物体、几何生成器和贴图，无需下载第三方模型。

| object_id | 原生资产 | 实验摩擦档位 | 金属桌面 μ | 桌垫 μ |
|---|---|---|---:|---:|
| `food_can` | RoboCasa Objaverse `canned_food_18`，直径 50 mm、高 65 mm | 低 | 0.15 | 0.60 |
| `cookie_box` | RoboCasa Objaverse `boxed_food_0`，45 × 120 × 72 mm 饼干盒 | 中 | 0.30 | 0.90 |
| `bread` | `BreadObject`，原面包网格及贴图 | 高 | 0.50 | 1.20 |

这些 μ 是本实验指定的接触参数，**不是对真实材料摩擦系数的测量值**。原 can、milk、bread XML 的 geom friction 实际相同，不能直接据此划分摩擦档位。2026-09-10 按用户要求，低、中摩擦物体分别换为 RoboCasa Objaverse `canned_food_18` 食品罐头和 `boxed_food_0` 饼干盒。罐头视觉网格横向缩放为 50 mm 直径，并配同尺寸圆柱碰撞体；饼干盒视觉网格配 45 × 120 × 72 mm 盒体碰撞，并以 90° 初始 yaw 使 45 mm 窄边对准夹爪。两者质量固定为 349 g，接触 μ 保持低／中档；许可证与来源记录随资产保存。新物体的完整抓放可解性与振动响应仍需重新认证。位移统计同时记录直立余弦，`upright cosine < 0.95` 后的位移不归类为滑移。

下文已有滑移演示和 Phase 09 的 20/30 静态结果均来自旧钢块，仅作历史记录，不适用于新物体；旧 selection 不得续跑为新物体证据。

摩擦定义在“物体—接触面”上。MuJoCo 显式接触对提供自己的接触参数，所以修改 geom 默认摩擦不足以修改当前任务。[MuJoCo 接触参数说明](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。六种配置均通过 pair 设置两个切向方向的摩擦，并审计编译后的 `pair_friction`。扭转、滚动摩擦和求解器沿用既有 physics profile。

桌垫是浅灰绿色、低反光的 3 mm 刚性薄层，表面复用原生 `gray-felt.png` 毛毡贴图（3 × 3 重复）并增加细包边，附着在隔振桌上，顶面与原工作面齐平；使用独立的 `table_mat_collision` 接触几何。物体只与选中的工作面配对，避免金属与桌垫重复接触。保持桌面总质量 32 kg、惯量、顶面高度和隔振器不变。该模型表达表面材料差异，不模拟软垫压缩变形。

目标区域的视觉现为浅色涂层金属置物篮，具有圆润边框、细栅条与两处握持套；底板和四壁仍使用原来的实心碰撞近似。篮体细节为无质量、无接触的显示几何，实际放置空间与成功判定尺寸不变。目标接触 μ 使用表中金属列；夹爪接触 μ 固定为 1.0。三个物体质量统一为 0.349 kg，以控制质量变量；惯量、质心和惯性坐标系来自按质量归一化的编译几何。

## 统一任务接口

```python
from robosuite.utils.shakebench_tasks import TaskSpec, make_task_env

env = make_task_env(
    TaskSpec(task_type="pick_place", object_id="cookie_box", surface_id="mat"),
    object_start_xy=(-0.10, -0.13),
    observation_tier="V0",
    hard_reset=False,
    seed=17,
)
try:
    observation = env.reset()
    context = env.get_task_context()
    observation, reward, done, info = env.step([0.0] * env.action_dim)
finally:
    env.close()
```

也可通过 `robosuite.make("VibrationPickPlace", task=...)` 构造。新任务默认采用当前 `direct_mount_v1` 场景。`TaskSpec` 验证任务类型、物体和桌面；`make_task_env` 集中选择实现，未来任务从此处扩展。`ShakeBenchTask` 定义 `reset/step/close`、动作范围、观测契约、静态任务上下文和 evaluator metrics 的共同接口。

State tiers 的物体位姿统一命名为 `object_pos_robot_base`、`object_quat_robot_base`，四元数为 xyzw；目标使用 `goal_*`。V0–V3 的振动信息权限保持原契约。`get_task_context()` 返回静态 `task` 和 `object` 信息；`get_metrics()` 是 evaluator 数据，不提供给策略。

物体初始高度根据编译接触几何计算。未知组合、非有限位置、与当前 state 契约不符的 yaw/速度/高度会被拒绝。TaskSpec 独占桌面接触配置，不接受另行覆盖 `table_friction` 或 `target_container_friction`。

旧 `VibrationPickPlaceCan` 和其默认 μ=0.30 行为保留，用于历史实验。既有 oracle 的内部 wire format 与 metrics 仍使用 `can_*` 名称，通过 `legacy_oracle_observation` 适配新任务；runner 根据 state 中的 `task` 自动选用新环境，并记录独立 task contract。这是兼容层，新增策略应使用 object 命名接口。

### 实验对象：轻量木块

`light_wood_block` 是一个只用于单次实验的低重心木块：尺寸 60 × 50 × 20 mm、质量 100 g、金属／桌垫接触 μ 为 0.15／0.60。它不属于 `task_variants()`，不写入 official 或 knee state，也不参与 state 聚合。它用于测试“低摩擦且低质量”这一联合条件；由于质量和材质同时改变，结果不能解释为单独的摩擦效应。

在 official profile、seed=17、Gamma=0.95、5 s 的金属桌面单次 rollout 中，其峰值工作台相对位移为 32.637 mm、末态为 32.070 mm，最小直立余弦为 0.999996，未倾倒。视频、100 点时序和汇总位于 `out/experimental_light_wood_gamma095_metal_5s/`。

## Phase 09/09.5 输入与合并规则

最初的 2,400 来自将既有 400 个位置／种子与六种组合做全交叉，并非必须数量。按用户选择，official 现调整为固定总量 400、组合尽量均衡；生成器不依赖试跑结果进行筛选。

新资产：

- `robosuite/models/assets/shakebench_task_states_official_v2.json`：保留 **400 个原始位置/种子，各分配一种组合，共 400 states**。
- `robosuite/models/assets/shakebench_task_states_knee_v2.json`：100 个原始位置/种子 × 6 = **600 states**。

例如 `shakebench-official-v2-000.pick_place.mat.cookie_box`。official 按原始 parent 顺序轮流分配六种组合，各组合为 67、67、67、67、66、66 个；每个原始位置与种子仅产生一个 state。knee 仍以同一 parent 的六种配置共享位置、excitation seed、IMU seed 和 t0，供配对诊断使用。每个配置绑定自己的物体、表面、初始位姿、资产哈希、task contract 和 Phase 8R outcome/controller 哈希。每个 state 有唯一 ID 和 canonical payload hash，验证器进行完整确定性再生成，拒绝缺失组合、篡改参数及错配资产。

所有配置统一放在一个 pick&place state 池中，各 state 等权。V0–V3 按完整 state_id 配对；按物体/表面分层的结果作为诊断补充。official 的六种组合不再全部具有相同 parent，物体／表面之间按分层比较；每个 tier 仍使用相同的 400 个 state 做严格配对。knee 的六个共享 parent 配置并非六个独立随机种子，不确定性估计应以 parent 分组。

```bash
# 仅生成输入，不运行 rollout
python -m robosuite.scripts.shakebench_generate_states --task-variants

# 单个 state 的短程 smoke；不是正式 benchmark 结果
python -m robosuite.scripts.shakebench_run_oracle \
  --states robosuite/models/assets/shakebench_task_states_knee_v2.json \
  --state-id shakebench-knee-v2-000.pick_place.mat.cookie_box \
  --tier V0 --gamma 0 --horizon-steps 2 \
  --geometry-profile direct_mount_v1 --output out/task_variants/oracle_cube_smoke.json

# 渲染六种状态
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m robosuite.demos.demo_shakebench_task_variants
```

CPU batch 默认读取新的 v2 official state 池，也支持显式指定 knee/v2 或历史 committed 资产；job identity 包含完整 state digest，组合之间不会复用输出或 resume 记录。生成 CLI 不带 `--task-variants` 时保留历史 Phase 08 生成行为。

v2 资产当前标记为 `prepared_for_requalification`、`scoreable=false`。任务接触和物体发生变化后，历史 can 认证不能自动授权新组合；Phase 09 通过完整 nominal gate、振动诊断和跨进程 replay 生成资格清单，Phase 09.5 验证该清单后才发布计分 authority。现有短程 smoke 和接触探针不会被当作该认证或策略成功率。

## 验证

`tests/test_shakebench_task_variants.py` 覆盖六种真实环境的编译、reset/step、接触对、质量、几何和观测契约，state 平衡与哈希防篡改，以及每个原生物体的冲量滑移和水平振动接触探针。

独立接触探针使用 349 g 物体、10 mm / 4 Hz 水平振动、0.3 s 平滑启动，总时长 1 s；三种物体的冲量和水平振动检查均通过。此探针用于验证摩擦差异进入动力学；不是完整隔振桌的响应，也不是 Phase 09 Gamma 标定。

验证结果（2026-09-08）：新增测试 18 项、历史任务与 protocol/oracle/outcome/batch 回归 62 项全部通过。木块＋桌垫的 2-step oracle smoke 通过结果语义验证，重新计算外层哈希后的物体身份替换仍被拒绝。该短程结果为 `horizon_exhausted`，不代表完整 pick&place 成功。

本地诊断输出位于 `out/task_variants/`：`contact_probes.json`、`oracle_cube_smoke.json`、`oracle_verification.json` 和 `pick_place_variants_final.png`。复现命令与测试都在仓库内，输出目录不是正式 Phase 09 结果权威。

## Official Gamma=0.6、5 s 滑移演示

正式版本使用 `physics_profile="official"`、`geometry_profile="direct_mount_v1"`，并经正式 runner 相同的 seed=17、2 s Gamma 校准路径得到 `level_scale=2.79021683593475`。命令的 Gamma 精确为 0.600；在完整 5 s 观测窗内，该固定程序的实际最大法向 Gamma 为 0.6496。每段视频为 5 秒实时播放、100 个 20 Hz 控制步。

| 组合 | 峰值工作台相对位移 | 5 s 末态位移 | 备注 |
|---|---:|---:|---|
| metal / steel_block | 19.039 mm | 17.597 mm | 直立滑移 |
| metal / wood_cube | 8.851 mm | 8.741 mm | 直立滑移 |
| metal / bread | 2.087 mm | 1.710 mm | 直立滑移 |
| mat / steel_block | 1.108 mm | 1.107 mm | 直立滑移 |
| mat / wood_cube | 0.00259 mm | 0.00254 mm | 基本静止 |
| mat / bread | 0.00349 mm | 0.00301 mm | 基本静止 |

六种组合的 `tip_time_s` 均为空，最小直立余弦均大于 0.99999，因此表内位移未混入翻倒后的滚动。原始视频、100 点时序、姿态门限、以及含 Gamma、物理 profile 和 level scale 的汇总位于 `out/task_slip_gamma060_5s_v2/`。旧 `out/task_slip_gamma060_5s/` 是已废弃的 can 对照，不属于当前任务状态。

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m \
  robosuite.demos.demo_shakebench_task_slip \
  --output-dir out/task_slip_gamma060_5s_v2 \
  --gamma 0.6 --duration-s 5 --width 480 --height 360
```

### Official Gamma=0.95、5 s rollout

在相同的 official profile、seed=17 和 2 s 校准路径下，`Gamma=0.95` 对应 `level_scale=4.417843323563354`；完整 5 s 窗口中的实际最大法向 Gamma 为 1.0285。所有组合均完成 100 个 20 Hz 控制步，且 `tip_time_s` 为空、直立余弦高于 0.9985，因此数据只包含未翻倒状态的相对位移。

| 组合 | 峰值工作台相对位移 | 5 s 末态位移 |
|---|---:|---:|
| metal / steel_block | 28.081 mm | 27.097 mm |
| metal / wood_cube | 38.539 mm | 37.062 mm |
| metal / bread | 21.902 mm | 21.873 mm |
| mat / steel_block | 11.681 mm | 11.681 mm |
| mat / wood_cube | 3.540 mm | 3.540 mm |
| mat / bread | 1.282 mm | 0.033 mm |

桌垫在这个强度下仍降低钢块与木块的累积滑移，但不再让它们近似静止。bread 的峰值会在试验结束前回落，表现为可恢复的相对位移。原始视频、100 点时序和汇总位于 `out/task_slip_gamma095_5s/`。

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m \
  robosuite.demos.demo_shakebench_task_slip \
  --output-dir out/task_slip_gamma095_5s \
  --gamma 0.95 --duration-s 5 --width 480 --height 360
```

本次视觉与数量修订：原框体／置物篮编译后的物理签名一致；更新后真实环境 reset/step 和渲染验证通过。新预览位于 `out/task_visual_refresh/mat_basket.png` 及 `basket_closeup.png`。此前 `out/task_variants/` 内的旧预览和旧 smoke 描述修订前版本，不作为当前输入集证据。
