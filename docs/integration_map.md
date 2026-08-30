# ShakeBench integration map

状态：Phase 00 骨架与 Phase 01/01R simulator-independent candidate excitation 完成；后续 physics/task 尚未实现。
审计路径：`/home/miracle04/Desktop/ShakeBench`。  
审计基线 commit：`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。

本文件把 ShakeBench 直接集成到现有 robosuite 目录的落点和真实源码 seam 固定下来。内部 Python package 仍然是 `robosuite`；下表中的 ShakeBench 文件都是现有目录内的 additive module，不创建 `shakebench/` 子目录。

## 1. Phase 00 实际新增文件

| 路径 | 职责 | 默认行为 |
| --- | --- | --- |
| `robosuite/utils/shakebench_config.py` | typed configuration envelope、JSON schema、canonical JSON、SHA-256 config hash、深度不可变 payload、official-field/provenance validators、`UNFROZEN` gate | 只在显式导入/调用时运行；不导入环境、不触碰 `robosuite.macros` |
| `robosuite/scripts/shakebench_cli.py` | `version`、`validate-config`、`print-schema` 命令行入口 | 不创建环境，不修改 registry 或 simulation globals |
| `tests/test_shakebench_config.py` | import/registry、schema round-trip、稳定 hash、unknown field、scoreability gate 的 bootstrap tests | 只验证既有环境注册表，不加入环境 |
| `docs/integration_map.md` | 直接集成落点、seam 和后续模块地图 | 文档 |
| `docs/phase_00_report.md` | 改名、哈希、版本、基线、测试和完成条件证据 | 文档 |

## 1.1 Phase 01 实际新增文件

| 路径 | 职责 | 默认行为 |
| --- | --- | --- |
| `robosuite/utils/shakebench_excitation.py` | 六轴 authored candidate band table、确定性 line jitter/phase、quintic ramp 和解析 `q/qdot/qdd` | 仅显式调用时生成 NumPy program；不创建 simulator |
| `robosuite/utils/shakebench_calibration.py` | workpiece-point `Gamma_commanded`、self-contained unit replay、peak factor 和 per-axis RMS/peak | 仅对显式 candidate program 做 authored-command 计算 |
| `robosuite/utils/shakebench_safety.py` | 六自由度 displacement、non-ballistic、frequency、feature solver-travel 和可选 angle gate | candidate fail-closed report；不裁剪输入 |
| `robosuite/scripts/shakebench_generate_excitation_golden.py` | 生成平铺 golden fixture 和机器可读误差摘要 | 不创建环境；输出到 `tests/` |
| `tests/test_shakebench_excitation.py`、`tests/test_shakebench_calibration.py`、`tests/test_shakebench_config.py` | golden 独立 evaluator、解析导数、Gamma、t0、ramp、六自由度 safety 和 configuration closure | 纯 NumPy/stdlib 测试 |

本阶段没有修改上游核心源码、`robosuite/__init__.py` 或任何既有环境/模型文件；为使新增代码的 Python 最低版本声明与语法一致，更新了 `setup.py`/`pyproject.toml`，并为已存在的 candidate texture 添加了最小 `.gitignore` 例外。

## 2. 已审计的真实上游入口

| 领域 | 实际路径与 symbol | 集成结论 |
| --- | --- | --- |
| 环境注册 | `robosuite/environments/base.py:register_env`、`EnvMeta.__new__`、`make` | 具体环境 class 在 class creation 时写入 `REGISTERED_ENVS`；`robosuite.make` 实际由 `robosuite/environments/base.py:make` 提供 |
| 根导入路径 | `robosuite/__init__.py` 中的 `from robosuite.environments.base import make`，以及随后逐项导入 manipulation env | 后续 `VibrationPickPlaceCan` 应在现有根导入链中显式 import，依靠 `EnvMeta` 注册；Phase 00 不添加它 |
| 环境基类 | `robosuite/environments/base.py:MujocoEnv` | 公共 constructor 保存 `_xml_processors`、`control_freq`、`lite_physics`、`model_timestep`、`control_timestep`；后续可选 seam 必须保持旧默认路径 |
| XML processor | `MujocoEnv.__init__` 的 `_xml_processors = [self.edit_model_xml]`、`set_xml_processor`、`_initialize_sim` 的 processor loop、`edit_model_xml` | deck processor 应是显式环境配置添加的 callable；不在 import 时注册 callback。`edit_model_xml` 负责把 XML 内 robosuite asset 路径解析到当前 package |
| reset | `MujocoEnv.reset`、`_reset_internal`、`reset_from_xml_string` | 新环境 reset 状态和 sensor state 应挂在已有 reset 生命周期；不要改变现有环境的 hard/deterministic reset 语义 |
| lite-physics step | `MujocoEnv.step`：每个 control step 内执行 `sim.step1()`/`sim.forward()` → `_pre_action` → `sim.step2()`/`sim.step()` → `_update_observables`，之后 `_post_action` | deck command 的写入时序必须在后续阶段用 probe 锁定，不能把旧的 `_pre_action` 一步超前行为直接当成合同 |
| robot step hook | `RobotEnv._pre_action` | 现有实现按 robot action 分发至 `robot.control`；若要增加 deck 写入，应以最小、可选接口组合，不改变旧 action 语义 |
| timestep | `MujocoEnv.initialize_time` 读取 `robosuite.macros.SIMULATION_TIMESTEP`；`MujocoWorldBase.__init__` 将同值写入 `<option timestep>` | Phase 00 不修改 global macro。后续 environment-owned timestep 必须先在新环境显式传入并锁定兼容 fallback；不能在模块 import 时改 `SIMULATION_TIMESTEP` |
| manipulation 基类 | `robosuite/environments/manipulation/manipulation_env.py:ManipulationEnv` | 提供 `_check_grasp`、EEF/object 相对 pose sensors、`_get_arm_prefixes` 和 gripper visualization；新任务可复用这些 helper |
| Lift 装配模板 | `robosuite/environments/manipulation/lift.py:Lift._load_model`、`_setup_references`、`_setup_observables`、`_reset_internal`、`_check_success` | 这是 `TableArena` + robot + object + `ManipulationTask` 的实际装配样板；ShakeBench 任务必须去掉 Lift 的 cube/height 硬编码 |
| task merge | `robosuite/models/tasks/task.py:Task.__init__`、`merge_arena`、`merge_robot`、`merge_objects` | 实际 world assembly 在 `Task`，`ManipulationTask` 目前只是 placeholder subclass；所有 XML reparent 必须适配此 merge 顺序 |
| XML merge | `robosuite/models/base.py:MujocoXML.merge`、`merge_assets`、`get_xml`；`robosuite/models/world.py:MujocoWorldBase` | body、asset、actuator、sensor、tendon、equality、contact 都在 merge 时搬入 task；`MujocoXML` 的 asset dependency 在源 XML 读取时转成绝对路径 |
| table arena | `robosuite/models/arenas/table_arena.py:TableArena.__init__`、`configure_location`、`table_top_abs`；模板 `robosuite/models/assets/arenas/table_arena.xml` | 现有 table 是一个 arena body，collision/visual/legs 是不同 geom；ShakeBench arena 应在现有 `models/arenas/` 增加 class，在 `models/assets/arenas/` 增加 XML |
| Can object | `robosuite/models/objects/xml_objects.py:CanObject`；`MujocoXMLObject`/`MujocoObject` properties 在 `robosuite/models/objects/objects.py`；资产 `robosuite/models/assets/objects/can.xml` | stock Can XML 使用 free joint、mesh density 和默认摩擦；后续任务必须显式覆盖 canonical mass/inertia/接触接口，而不是默默复用 mesh-derived 质量 |
| Panda base | `robosuite/models/robots/manipulators/panda_robot.py:Panda.default_base`、`base_xpos_offset`；`robosuite/models/robots/robot_model.py:RobotModel.add_base`/`add_mount` | Panda 通过 `Robot.load_model` → `RobotModel.add_base` 合并 mount；robot base 隶属 deck 的 reparent 必须发生在正式 task XML assembly 层 |
| Panda gripper | `robosuite/robots/robot.py:Robot.load_model`；`ManipulatorModel.add_gripper`；`robosuite/models/grippers/panda_gripper.py:PandaGripper` | 现有 Panda 是 7D arm+single gripper action 的来源；Phase 00 不改 force、gripper semantics 或 controller |
| observables | `robosuite/utils/observables.py:sensor`、`Observable.update`；`MujocoEnv._setup_observables`、`Robot.setup_observables`、`Lift._setup_observables` | 后续 V0–V3 应在任务 `_setup_observables` / additive provider 中建立明确 key set；不能通过 privileged recorder 反向污染 policy observation |
| wrappers | `robosuite/wrappers/wrapper.py:Wrapper.step/reset`；`GymWrapper` 的 `reset`、`step`、`_flatten_obs` | wrapper 是外围适配层，不是 task physics seam；后续 score/recording wrapper 需保持四元组原生 API与 Gym 五元组转换边界清楚 |

## 3. 后续模块的最终落点

下表是依据当前设计树、v0 spec 和后续 phase prompts 的落点表。每个路径都位于仓库已有目录；不得把任一路径改成新的 `shakebench/` package 或 test/docs 子目录。

| 阶段 | 最终路径 | 职责与接入 seam |
| --- | --- | --- |
| 01 | `robosuite/utils/shakebench_excitation.py` | **已实现 candidate**：simulator-independent 六轴谱、seed/`t0`、quintic ramp；exact reuse 未证明 |
| 01 | `robosuite/utils/shakebench_calibration.py` | **已实现**：Γ/频谱校准与可审计 authored-command 解析；不读取 task SR |
| 01 | `robosuite/utils/shakebench_safety.py` | **已实现 candidate gate**：六自由度位移、non-ballistic、频率、feature solver travel 和 safety rejection |
| 01 | `robosuite/scripts/shakebench_generate_excitation_golden.py` | **已实现**：生成 tests 根目录的平铺 golden fixture 和误差摘要 |
| 01 | `tests/test_shakebench_excitation.py`、`tests/test_shakebench_calibration.py`、`tests/test_shakebench_config.py` | **已实现**：excitation/calibration/config/safety 纯工具测试；golden 用独立 evaluator 验收 |
| 02 | `robosuite/utils/shakebench_deck.py` | dynamic deck model、mocap driver、weld XML processor；挂接 `MujocoEnv.set_xml_processor` 和经过测试的 step timing |
| 02 | `robosuite/scripts/shakebench_probe_deck_driver.py`、`tests/test_shakebench_deck_driver.py` | zero/six-axis/spectrum/Gamma、empty/load、多 dt 和 weld conformance |
| 02 | `robosuite/environments/base.py`（仅在测试锁定后） | 可选 environment-owned timestep / pre-step hook 的最小向后兼容 seam；旧 env 未传参数时完全走现有 `macros.SIMULATION_TIMESTEP` |
| 03 | `robosuite/utils/shakebench_isolator.py` | canonical linear 6-DoF support、解析 transfer、preload/k/c 派生 |
| 03 | `robosuite/models/arenas/shakebench_arena.py` | 继承/组合现有 `TableArena` 结构，提供 explicit tabletop inertial root、industrial visual primitives、target assembly 接口 |
| 03 | `robosuite/models/assets/arenas/shakebench_arena.xml` | arena XML；只放现有 assets/arenas 目录 |
| 03 | `tests/test_shakebench_arena.py`、`tests/test_shakebench_isolator.py` | XML、惯量、preload、解析 transfer 和 physics-only probes |
| 04 | `robosuite/environments/manipulation/vibration_pick_place_can.py` | `ManipulationEnv` 子类；复用 `Lift` 的生命周期，但显式处理 Can、开放起点、浅目标箱和 success handles |
| 04 | `robosuite/utils/shakebench_metrics.py` | 接触接口、containment、settle/velocity/penetration 和 recorder-only metrics |
| 04 | `robosuite/__init__.py`（仅注册时） | 显式 import `VibrationPickPlaceCan`，让 `EnvMeta` 通过已有 `REGISTERED_ENVS` 注册；不得改动已有 registry entries |
| 04 | `tests/test_environments/test_vibration_pick_place_can.py`、`tests/test_shakebench_metrics.py` | task assembly、static contact、boundary evaluator 和 metric tests |
| 05 | `robosuite/utils/shakebench_sensors.py` | canonical 200 Hz deck IMU（specific force、gravity、lever arm、filter/delay/noise） |
| 05 | `robosuite/utils/shakebench_providers.py` | V0–V3 vibration information providers；只生成 policy-allowed keys |
| 05 | `robosuite/utils/shakebench_privilege.py` | `privileged_` recorder/evaluator truth 与 fail-closed key audit |
| 05 | `VibrationPickPlaceCan._setup_observables`（已有 task 文件） | 将公共 task state 与 V0–V3 provider 接到既有 `Observable` 生命周期，不修改 stock env defaults |
| 05 | `tests/test_shakebench_sensors.py`、`tests/test_shakebench_providers.py`、`tests/test_shakebench_privilege.py` | IMU/key-set/privilege isolation tests |
| 07 | `robosuite/utils/shakebench_oracle.py` | shared 7D State Oracle controller；调用既有 OSC/action schema，不创建新 controller default |
| 07 | `robosuite/scripts/shakebench_run_oracle.py`、`tests/test_shakebench_oracle.py` | Gamma=0 acquisition/transport/place/release 与 V0–V3 shared-controller tests |
| 08 | `robosuite/utils/shakebench_protocol.py` | committed state IDs、replay、manifest 和 hash provenance |
| 08 | `robosuite/utils/shakebench_scoring.py` | Wilson、paired bootstrap、MDE、single `Gamma_star` scorecard |
| 08 | `robosuite/scripts/shakebench_generate_states.py`、`shakebench_replay_state.py`、`shakebench_scorecard.py` | 运行协议脚本 |
| 08 | `robosuite/models/assets/shakebench_states_dev.json`、`shakebench_states_official.json`、`shakebench_states_knee.json` | 直接平铺在现有 `robosuite/models/assets/` 根；由既有 `MANIFEST.in` 打包 |
| 08 | `tests/test_shakebench_protocol.py`、`tests/test_shakebench_scoring.py` | state integrity、replay、denominator、paired statistics tests |
| 06/09 | `robosuite/scripts/shakebench_select_physics.py`、`shakebench_replay_physics.py`、`shakebench_select_isolator.py`、`shakebench_select_gamma_knee.py` | physics-only selection 和独立 knee calibration；输出 raw evidence，不读取 task SR 选择 physics |
| 10 | `robosuite/scripts/shakebench_release_audit.py`（如需） | clean-room audit orchestration；只审计，不在 release phase 修功能 |

## 4. XML、reparent 与 timestep 的 seam 约束

### 4.1 Dynamic deck 与 robot/table 拓扑

目标拓扑是 `prescribed driver → ordinary dynamic deck → {Panda base, 6-DoF isolator → isolated worktable}`。`Can` 必须继续是 world child + freejoint，不能放进 worktable body。实现时需要同时满足：

- deck driver 只驱动 equality weld 对应的动态 deck，不让 mocap geom 承担 task contact；
- Panda 的 base attachment 在 `RobotModel.add_base` / `MujocoXML.merge` 后可被识别为 deck-owned rigid subtree；
- worktable、浅目标箱、装饰 frame 属于 isolated assembly，但视觉 primitives 不添加隐藏 inertial；
- `Task.merge_objects` 仍把 Can 放到 task worldbody，不能把 Can 作为 arena/table 的 child；
- 任何 reparent 都要在 XML 编译前完成，并保留 `MujocoEnv.edit_model_xml` 的绝对 asset-path 修复。

### 4.2 Physics timing

当前 `MujocoEnv.step` 的真实顺序是：

```text
control action
  → for each model step: step1/forward
  → _pre_action(action, policy_step)
  → step2/step
  → _update_observables
  → _post_action
```

当前 model timestep 由 `MujocoEnv.initialize_time` 和 `MujocoWorldBase.__init__` 的 macro 路径共同确定。Phase 00 不写入任何 global timestep/callback。Phase 02 若需要 pre-step 或 environment-owned timestep，必须：

1. 只在新环境/显式参数启用；
2. 对旧 env 保留 constructor 与旧执行路径；
3. 先添加 regression tests，再修改 `base.py`；
4. 记录 command/actual deck pose、twist、acceleration、weld error 和 constraint wrench；
5. 按 `dt`、weld time constant、频率 sampling 和 solver travel 的约束 fail closed。

## 5. Asset、texture 和 package data

- XML 相对 asset 路径由 `robosuite/utils/mjcf_utils.py:xml_path_completion` 从 `robosuite/models/assets/` 解析；`MujocoXML.resolve_asset_dependency` 在加载时转绝对路径。
- `MujocoEnv.edit_model_xml` 会对包含 `robosuite` 路径段的 mesh/texture 做当前 package 路径修复。
- `MANIFEST.in` 已有 `recursive-include robosuite/models/assets/ *`，`setup.py` 已有 `include_package_data=True`；因此未来 profile/state/arena XML 和 texture 应直接平铺在已有 assets 目录或其已有子目录，不需要新增打包配置。
- Phase 00 已验证 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg` 存在，大小 `163720` bytes，SHA-256 为 `6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb`。
- Phase 03 可直接使用该 texture；若材质代码需要注册纹理名，应在 additive arena/XML 层处理，不修改 stock `TEXTURE_FILES` 以避免改变已有材质行为。

## 6. Registry、API 和兼容性护栏

- Phase 00 的 `shakebench_config` 不 import `robosuite`，不会触发环境 class creation，也不会触碰 `REGISTERED_ENVS`。
- `ShakeBenchConfig` 默认 `scoreable=False`，其 official map 以 `UNFROZEN` 表示尚未由 physics/controller/state phases 冻结的值。
- 只有 `validate_config` / `ShakeBenchConfig.validate()` 在 scoreable mode 下执行 official gate；scoreable config 必须提供完整 official-field map，且任何 sentinel（包括嵌套值）都会失败。
- `canonical_json` 使用排序 key、无空白 separators、UTF-8、拒绝 non-finite number；`config_hash` 是该字节串的 lower-case SHA-256。
- CLI 的 `version` 只报告 extension identity 和当前 package version；`print-schema` 输出上述 config envelope；`validate-config` 不会实例化环境。
- 不修改 `robosuite.macros.SIMULATION_TIMESTEP`，不注册全局 callbacks，不改现有 action/observation/model timestep/XML 默认行为。

## 7. 测试落点与命令

纯配置/协议测试直接放在 `tests/` 根，环境测试放在现有 `tests/test_environments/`。Phase 00 使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_shakebench_config.py
python -m robosuite.scripts.shakebench_cli version
python -m robosuite.scripts.shakebench_cli print-schema
python -m robosuite.scripts.shakebench_cli validate-config <config.json>
```

仓库当前安装环境有 ROS pytest entrypoint 与两个模块级 argparse 测试；完整上游基线的兼容 invocation 和阻断原因记录在 `docs/phase_00_report.md`。后续新增测试不得依赖这些环境外部条件；测试应显式关闭插件自动发现，或在 CI 中提供与上游测试匹配的 runner。

## 8. 明确不在 Phase 00 的工作

本阶段没有实现 excitation、deck、isolator、arena、Can task、target box、contact/friction、IMU、V0–V3 provider、oracle controller、committed states 或 scorecard；没有复制任何外部 package；没有新建目录；没有改 package name/version。

Phase 01 已实现 authored excitation，但仍未实现 deck、isolator、arena、task、contact、IMU 或任何 MuJoCo model/environment；这些继续由后续 phases 负责。

Phase 01R 将该 excitation 明确标记为 `new authored v0 candidate`；没有可审计的旧 ShakeBench algorithm/reference，因此不宣称 exact reuse。`scoreable=True` 仍需后续 freeze authority。
