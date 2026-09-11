# 当前仓库的 robosuite ShakeBench 实现清单

本文按**可执行代码与其被加载的包内资产**盘点当前工作树中的 ShakeBench 实现；不把 `out/` 的运行产物、`__pycache__/` 或纯历史报告当作另一套实现。结论来自本仓库一手源码（文件名后的 `L` 为当前行号）。其中 `VibrationPickPlace` 是现在的统一任务接口，`VibrationPickPlaceCan` 是保留给历史实验的兼容实现。

## 1. 任务环境、场景和任务变体

| 实现 | 功能 | 关键源码 |
| --- | --- | --- |
| `VibrationPickPlaceCan` | 旧版 Phase 04：单 Panda、开放工作台、Can 抓取/放置环境；编译场景、注册观测、在动作前驱动振动台、提供 metrics / policy context / success 判断。 | `robosuite/environments/manipulation/vibration_pick_place_can.py` L106、L563、L688、L911、L1029、L1190、L1411 |
| `VibrationPickPlace` | 新版对象无关接口，继承旧环境但以 `TaskSpec` 选择物体/表面，强制 `direct_mount_v1`，把 `can_*` 公共观测改成 `object_*`；仍通过显式 adapter 供旧 oracle 使用。 | `robosuite/environments/manipulation/vibration_pick_place.py` L13-L32、L34-L50 |
| robosuite 导出/工厂入口 | 两个环境均由 `robosuite.__init__` 导入，因此可用 `robosuite.make("VibrationPickPlace", task=...)` 或旧名称构造。 | `robosuite/__init__.py` L1、L15-L16；`robosuite/environments/base.py` 中 `make` |
| `ShakeBenchArena` | 继承 robosuite `Arena` 的工业场景：创建隔振的规范工作台，并为 XML 处理器保留 `isolated_worktable` 角色和目标容器装配缝。 | `robosuite/models/arenas/shakebench_arena.py` L1、L28-L49、L74-L109 |
| `shakebench_arena.xml` 及世界/直装 XML | MuJoCo 资产层：工作台和环境世界定义；由 Arena 和场景配置加载。 | `robosuite/models/assets/arenas/shakebench_arena.xml` L1；`robosuite/models/assets/arenas/shakebench_world_visuals.xml` L1；`robosuite/models/assets/arenas/shakebench_direct_mount_world_additions.xml` L1 |
| `TaskSpec` / `ShakeBenchTask` / `make_task_env` | 定义 `pick_place`、物体和表面选择，生成物体、统一的 `reset/step/close` 协议，并集中选择新环境。对象表含 `food_can`、`light_wood_block`、`cookie_box`、`bread`；表面为 `metal`、`mat`。冻结 state 使用前三种物体与两种表面的六个组合，`light_wood_block` 保持 opt-in。 | `robosuite/utils/shakebench_tasks.py` L19-L59、L98-L110、L181-L208、L211-L251、L281 |

## 2. 运行时 benchmark 子系统

这些模块是上表环境在 robosuite/MuJoCo 执行路径中直接使用的 ShakeBench 实现，或是生成、验证其可运行 benchmark 输入的代码。

| 模块 | 功能（主入口） | 关键源码 |
| --- | --- | --- |
| `shakebench_deck` | 动态六自由度 deck driver、可审计 XML 变换和 trace；`DeckDriver` 将激励命令接到仿真。 | `robosuite/utils/shakebench_deck.py` L1-L3、L315、L622、L1159、L1330 |
| `shakebench_excitation` | 不依赖仿真器的六轴频带、确定性谱线、quintic ramp 和运动导数生成。 | `robosuite/utils/shakebench_excitation.py` L1-L17、L30-L57、L626 |
| `shakebench_isolator` | 规范线性六自由度隔振器：参数、预载/静态偏移和传递函数。 | `robosuite/utils/shakebench_isolator.py` L1、L201、L348、L473、L643 |
| `shakebench_sensors` | 规范 deck IMU：刚体运动到 specific force/gyro、滤波和有噪声的 `CanonicalIMU` trace。 | `robosuite/utils/shakebench_sensors.py` L1、L106-L201、L393、L723-L1040 |
| `shakebench_providers` | V0–V3 policy-safe vibration information provider；从 MuJoCo 读取支持体状态或重建已公开的激励，不泄露 evaluator 真值。 | `robosuite/utils/shakebench_providers.py` L1-L6、L131-L143、L463-L509、L681-L865 |
| `shakebench_privilege` | 将 privileged recorder 与 policy observation 隔离，并审计 policy payload。 | `robosuite/utils/shakebench_privilege.py` L1、L85-L122、L180-L230 |
| `shakebench_metrics` | Can/物体接触、相对位姿/速度和连续稳定成功窗口；`VibrationSuccessEvaluator` 与 `ShakeBenchMetrics` 是环境计量层。 | `robosuite/utils/shakebench_metrics.py` L1、L51、L248、L1111-L1120、L1445-L1495 |
| `shakebench_oracle` | 只读取 public State/context/provider 的可检查 FSM reference controller；不持有 MuJoCo object/contact truth。 | `robosuite/utils/shakebench_oracle.py` L1-L7、L35-L69、L1030、L1353、L1594 |
| `shakebench_outcomes` | 独立版本化 outcome contract：校验 episode validity、score outcome、termination cause 和 controller events。 | `robosuite/utils/shakebench_outcomes.py` L1、L49、L57-L129、L170 |
| `shakebench_tasks` / `shakebench_task_states` | 前者提供任务身份/适配；后者把 v1 parent states 扩展为 v2：official 每 parent 一个轮换变体，knee 为全部变体，并校验/冻结资产。 | `robosuite/utils/shakebench_task_states.py` L1-L6、L25-L96、L99-L118 |
| `shakebench_committed_states` / `shakebench_dev_states` | Phase 08 official/knee state authority 与 Phase 07 dev-state authority；构建、哈希、验证和冻结状态资产。 | `robosuite/utils/shakebench_committed_states.py` L1、L46、L184、L264、L353；`robosuite/utils/shakebench_dev_states.py` L1、L56、L116 |
| `shakebench_physics` / `shakebench_geometry` / `shakebench_scene` | 分别负责不可变 official/probe physics profile、几何 profile、以及场景视觉/库存/净空审计；三者将物理、装配和视觉配置分离。 | `robosuite/utils/shakebench_physics.py` L1-L14、L177、L574-L752；`robosuite/utils/shakebench_geometry.py` L1-L16；`robosuite/utils/shakebench_scene.py` L1、L496、L545 |
| `shakebench_calibration` / `shakebench_safety` | 将 authored acceleration 校准为 Gamma，并为六轴激励做 fail-closed 安全界检查。 | `robosuite/utils/shakebench_calibration.py` L1、L177、L256-L399；`robosuite/utils/shakebench_safety.py` L1、L80、L134、L226-L392 |
| `shakebench_scoring` | 不运行物理的纯 Phase 08 scorecard：episode/run schema、Wilson 区间、paired MDE/bootstrap 和验证。 | `robosuite/utils/shakebench_scoring.py` L1、L58、L194、L224-L250、L291-L472 |
| `shakebench_contact_force` / `shakebench_driver_measurement` | 物理选择使用的任务原生接触/水平力夹具，以及长 deck-driver probe 的有界测量计划。 | `robosuite/utils/shakebench_contact_force.py` L1、L125、L194-L426；`robosuite/utils/shakebench_driver_measurement.py` L1、L30、L90 |
| `shakebench_protocol_v6`、`shakebench_phase06_adapters`、`shakebench_physics_finalizer` | Phase 06 的 immutable resolved state、无物理 adapter 和 fail-closed official physics 发布/校验。V7/V8 直接复用 V6 resolver。 | `robosuite/utils/shakebench_protocol_v6.py` L1、L157-L469；`robosuite/utils/shakebench_phase06_adapters.py` L1、L25-L48、L150-L381；`robosuite/utils/shakebench_physics_finalizer.py` L1、L275、L467 |
| `shakebench_artifacts`、`shakebench_authority`、`shakebench_runtime_verifier`、`shakebench_handoff*` | canonical JSON/hash/原子写入、Phase 7.5A/08 scoreability authority、紧凑发布包验证与 handoff 语义验证。 | `robosuite/utils/shakebench_artifacts.py` L1、L19-L67；`robosuite/utils/shakebench_authority.py` L1、L107-L201；`robosuite/utils/shakebench_runtime_verifier.py` L1、L57；`robosuite/utils/shakebench_handoff.py` L1、L130；`robosuite/utils/shakebench_handoff_semantic.py` L1、L140-L286 |
| `shakebench_config`、`shakebench_rotations` | Phase 00 closed configuration/provenance envelope，以及 MuJoCo `wxyz` 精度四元数工具。 | `robosuite/utils/shakebench_config.py` L1、L83-L111；`robosuite/utils/shakebench_rotations.py` L1、L13-L90 |

## 3. 可直接调用的命令和演示

所有下列文件都有 `main()` 与 `if __name__ == "__main__"`，所以通用调用形式是 `python -m robosuite.scripts.<文件名>` 或 `python -m robosuite.demos.<文件名>`；括号中是 `main` 的行号。

| 入口 | 功能 |
| --- | --- |
| `shakebench_run_oracle`（`robosuite/scripts/shakebench_run_oracle.py` L499、L751） | 在显式 dev/committed states 上运行共享 State Oracle，生成 rollout/run artifact。`run_episode` 从 state 构造激励，按是否含 `task` 选择新 `VibrationPickPlace` 或兼容 `VibrationPickPlaceCan`，再以 public observation 执行 controller。 |
| `shakebench_generate_states`（L12）和 `shakebench_replay_state`（L12） | 生成/冻结、或离线验证 Phase 08 state asset；前者支持 `--task-variants`。 |
| `shakebench_scorecard`（L25） | 从已有 raw episode JSON 构建 scorecard，不运行 physics。 |
| `shakebench_cpu_batch`（L799） | 原子、可恢复的两 worker CPU rollout 批执行层，不改变 science 输入。 |
| `shakebench_scene_preflight`（L153）、`shakebench_verify_actuators`（L174） | 分别运行 scene/frame/clearance gate，及 Panda OSC_POSE + gripper 七通道动作路径验证。 |
| `shakebench_probe_arm_effect`（L497） | 执行 E1/E2/E3 振动对 Panda 臂保持、相对轨迹和稳态抓取影响的非计分诊断。 |
| `shakebench_probe_deck_driver`（L1525）、`shakebench_replay_physics`（L267） | 前者以小 MJCF 实物探测 deck driver / XML processor，后者跨独立进程复放物理 trace 以验证确定性。 |
| `shakebench_select_physics`（L513）、`_v6`（L1819）、`_v7`（L2313）、`_v8`（L1060）、`shakebench_finalize_physics`（L779） | Phase 06 的物理候选选择、协议修订选择、接触恢复/法向冲击选择及最终 fail-closed 发布。 |
| `shakebench_diagnose_contact_recovery_v7`（L416）、`shakebench_diagnose_normal_impact_v8`（L375） | 非计分的真实 MuJoCo 接触恢复与法向冲击/斜坡夹具诊断。 |
| `shakebench_generate_excitation_golden`（L452） | 校验 authored excitation golden fixture；只有带显式更新开关才替换 fixture。 |
| `shakebench_cli`（L88） | Phase 00 config envelope 的 `version`、`print-schema`、`validate-config` CLI。 |
| `shakebench_handoff_remediation`（L730）、`shakebench_handoff_semantic_remediation`（L12） | 对既发布 tuple 运行 Phase 06F-R 实环境 remediation，以及 Phase 06F-R2 semantic handoff verifier。 |
| `shakebench_verify_phase07_handoff`（L909）、`shakebench_verify_phase08r_handoff`（L82） | 各阶段 handoff 的 fail-closed 验证，不运行 rollout。 |
| `shakebench_build_evidence_index`（L166）、`shakebench_build_evidence_archive`（L237）、`shakebench_build_release_manifest`（L118）、`shakebench_verify_evidence_archive`（L525）、`shakebench_audit_evidence`（L296） | 构建、封装、签出、验证并完整审计 evidence archive。 |
| `shakebench_verify_package`（L372） | 在干净 wheel/sdist install 中检查无 source-checkout 泄漏。 |
| `demo_shakebench_task_variants`（`robosuite/demos/demo_shakebench_task_variants.py` L16） | 渲染六个 pick/place 变体的工业场景预览。 |
| `demo_shakebench_task_slip`（L176） | 在夹爪保持张开且静止时录制六个变体的同条件 vibration/slip 视频；展示物体相对表面的运动而非策略能力。 |
| `demo_shakebench_oracle_video`（L336） | 录制可复现 oracle 定性视频；相机像素不进入 policy、metrics 或科学证据链。 |

## 4. 被运行时加载的冻结配置与资产

| 资产组 | 功能 |
| --- | --- |
| `shakebench_official_physics.yaml` | scoreable、immutable 的 official physics profile，绑定 Phase 06F protocol/evidence hash。`robosuite/models/assets/shakebench_official_physics.yaml` L1-L16。 |
| `shakebench_geometry_direct_mount_v1.json` 和 `shakebench_scene_direct_mount_v1.json` | 当前 `direct_mount_v1` 的机器人初始关节/几何，及不改变 physics topology 的场景视觉配置。`robosuite/models/assets/shakebench_geometry_direct_mount_v1.json` L1-L16；`robosuite/models/assets/shakebench_scene_direct_mount_v1.json` L1-L16。 |
| `shakebench_task_states_official_v2.json` / `shakebench_task_states_knee_v2.json` | v2 任务扩展状态：official 从 400 parent 分配一个变体，knee 从 100 parent 做完整变体组合；均锁定 controller、physics、scene、task contract hashes。`robosuite/models/assets/shakebench_task_states_official_v2.json` L1-L16；`robosuite/models/assets/shakebench_task_states_knee_v2.json` L1-L16。 |
| `shakebench_runtime_contract.json` | 紧凑运行时发布包的受认证资产清单。`robosuite/models/assets/shakebench_runtime_contract.json` L1-L16。 |
| `shakebench_selection_protocol*.yaml`、`shakebench_phase_06*.json/yaml`、`shakebench_phase07_5a*.json` | 各 Phase 06/07 选择、状态、handoff 和 authority 的冻结证据；由上述 physics/authority/verifier 模块按 hash 读取，而不是由环境随意采样。示例索引常量见 `robosuite/utils/shakebench_physics.py` L34-L61。 |

## 5. 最小使用示例

```bash
# 新版任务环境（程序内也可调用 robosuite.make）
python -m robosuite.demos.demo_shakebench_task_variants

# 一个显式 task-state 上运行 reference controller
python -m robosuite.scripts.shakebench_run_oracle \
  --states robosuite/models/assets/shakebench_task_states_knee_v2.json \
  --state-id shakebench-knee-v2-000.pick_place.mat.cookie_box

# 录制 wrist-camera oracle 视频（需要可用的 MuJoCo 渲染后端）
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m robosuite.demos.demo_shakebench_oracle_video --wrist-inset \
  --output out/demo/shakebench_direct_mount_wrist.mp4
```

上述第二条命令的 state/参数调用方式也由仓库文档给出：`docs/phase09_pick_place_states.md` L74-L85；它与源码 `make_task_env`（`robosuite/utils/shakebench_tasks.py` L240）和 runner（`robosuite/scripts/shakebench_run_oracle.py` L751）一致。
