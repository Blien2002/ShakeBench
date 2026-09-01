# ShakeBench-in-robosuite：多阶段实现提示词

本目录中的 prompts 用于把**原 robosuite 1.5.2 仓库整体改名为 ShakeBench**，并在改名后的仓库内实现 ShakeBench v0 benchmark。ShakeBench 是仓库本身：实现**直接在 robosuite 原有目录与文件上修改**，必要时只在现有目录内新增 `shakebench_` 前缀的模块文件，**不新建任何 shakebench 子文件夹**。

## 仓库与路径约定

```text
implementation repo: /home/miracle04/Desktop/ShakeBench
    = 原 /home/miracle04/Desktop/robosuite（ARISE-Initiative/robosuite fork）整目录改名
    - 内部 Python package 保持 robosuite（import robosuite、注册表、资产路径不变）
    - 仓库目录名与项目名称为 ShakeBench
```

仓库改名由 Phase 00 步骤 0 完成。权威设计文档已随仓（`docs/` 根，见 `docs/IMPORT_PROVENANCE.md`）；实现与提示词**不引用任何外部目录**。

## 架构原则

- **不新建任何 shakebench 子文件夹**：`robosuite/utils/shakebench/`、`tests/test_shakebench/`、`docs/shakebench/`、`robosuite/models/assets/shakebench/` 全部禁止。
- **直接在原有目录上改**：允许在现有目录内新增模块文件（文件名 `shakebench_` 前缀），公共核心（如 `MujocoEnv`）直接做最小、向后兼容的修改。
- 阶段报告直接写进现有 `docs/`：`docs/phase_XX_report.md`、`docs/integration_map.md`。
- 权威设计文档位于 `docs/` 根：`docs/robosuite_benchmark_design_tree.md` 等；提示词目录 `docs/shakebench_prompts/` 保留。
- 测试直接加入现有 `tests/` 结构：环境测试进 `tests/test_environments/`，纯工具测试在 `tests/` 根新增 `test_shakebench_*.py` 文件。
- 配置/profile/state 数据文件直接平铺在 `robosuite/models/assets/` 根（该目录已被 MANIFEST.in 递归打包），文件名 `shakebench_*`。

## 权威资料

每个阶段从 implementation repo 根目录执行，并完整读取：

```text
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/shakebench_prompts/README.md
```

按需读取：

```text
docs/spike_implementation_formal_review.md
docs/canonical_imu_profile_validation.md
docs/repro_weld_sag_20260828.md
```

冲突优先级：最新设计树 > v0 spec > research reports。

## 直接集成原则

- 修改当前 `robosuite/` package、`tests/`、`docs/` 和必要的 package assets；仓库名已是 ShakeBench，但 Python package 名保持 robosuite。
- 保持现有 robosuite 公共 API 和全部上游测试兼容；新行为默认关闭，只有新环境或显式 ShakeBench 配置启用。
- 优先 additive modules；修改 `MujocoEnv`、robot mounting、step loop 等公共核心时，提供最小、向后兼容的可选 seam 和现有环境回归测试。
- 新环境命名 `VibrationPickPlaceCan`，通过 robosuite 现有 metaclass/import 机制注册，可由 `robosuite.make(...)` 创建。
- 实现只依赖随仓权威文档与 robosuite 自身，不引用任何外部目录；不复制 Isaac/Newton 代码。
- 当前只实现 State Oracle-Control Track；Vision、训练方法、多任务和真机验证保持 deferred。

建议落点（Phase 00 可按实际上游结构细化文件名，但不得新建子文件夹）：

```text
robosuite/utils/shakebench_config.py        # schema/config/hash/UNFROZEN
robosuite/utils/shakebench_excitation.py    # 六轴激励
robosuite/utils/shakebench_deck.py          # deck driver / XML processor
robosuite/utils/shakebench_isolator.py      # 6-DoF isolator
robosuite/utils/shakebench_sensors.py       # canonical IMU
robosuite/utils/shakebench_providers.py     # V0–V3 providers
robosuite/utils/shakebench_metrics.py       # 任务/接触指标
robosuite/utils/shakebench_privilege.py     # recorder/evaluator 隔离
robosuite/utils/shakebench_oracle.py        # State Oracle controller
robosuite/utils/shakebench_protocol.py      # committed states/回放
robosuite/utils/shakebench_scoring.py       # scorecard/MDE
robosuite/models/arenas/shakebench_arena.py # 工业工作台
robosuite/models/assets/arenas/shakebench_arena.xml
robosuite/models/assets/textures/           # 现有目录内新增 texture 文件
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/scripts/shakebench_*.py           # probes/标定/正式运行脚本
tests/test_environments/test_vibration_pick_place_can.py
tests/test_shakebench_*.py                  # 纯工具测试文件
docs/phase_XX_report.md
docs/robosuite_benchmark_design_tree.md
```

## 全局科学合同

- dynamic deck 通过 mocap driver + equality weld 获得 prescribed motion；robot base 刚接 deck，32 kg worktable 通过 linear 6-DoF isolator 接 deck。
- Can 是 world child + freejoint；开放初始区，有墙浅目标箱随 worktable 运动。
- `Gamma_commanded` 是实验自变量；deck/table actual Gamma 分别是 conformance/response。
- V0–V3 共享 TaskExecutive、controller、action schema、rate、gripper、states 和 horizon，只替换 vibration-information provider。
- task/controller/physics failure 进入分母；infrastructure crash 只重跑相同 state ID。
- policy observation 与 privileged recorder/evaluator truth fail-closed 隔离。
- 物理参数按 physics metrics 选择，不根据 task SR 反调。

## 执行规则

1. 按编号顺序运行，一个 agent turn 只执行一个阶段。
2. 开始阶段前检查 `git status`，保留其他人的修改；记录起始 commit。
3. 阶段完成必须同时满足代码、测试、机器可读 artifacts 和 `docs/phase_XX_report.md`。
4. 阶段失败时保留 raw evidence，标记 blocked，停在当前阶段。
5. Phase 09 之前不运行 official evaluation；Phase 10 之前不宣称 release-ready。

## 阶段索引

| 阶段 | Prompt | 直接修改重点 | 硬门 |
|---:|---|---|---|
| 00 | `phase_00_upstream_integration.md` | robosuite→ShakeBench 改名、上游结构审计、直接集成骨架 | 现有 tests 不回归 |
| 01 | `phase_01_excitation_core.md` | `robosuite/utils` 内纯 NumPy 激励模块 | golden 解析不变量 |
| 01R | `phase_01_review_remediation.md` | 封闭 scoreable 配置、独立 golden、六自由度 safety 与文档 provenance | 审查问题全部可证伪地关闭 |
| 02 | `phase_02_dynamic_deck_driver.md` | env-owned timestep、XML processor、dynamic deck | six-axis driver conformance |
| 02R | `phase_02_review_remediation.md` | 修复 driver 生命周期、Γ/坐标系/weld 测量语义与 conformance matrix | Phase 02 P1 测量门关闭 |
| 02R2 | `phase_02_final_conformance_remediation.md` | 闭合 post-step state、谱线分辨率、Γ×load 与 artifact 锁定 | Phase 03 handoff PASS |
| 02R3 | `phase_02_time_contract_remediation.md` | 闭合 mocap 左右极限与 acceleration/Gamma 时间契约 | Phase 03 handoff PASS |
| 02R4 | `phase_02_driver_selection_and_load_remediation.md` | 数据驱动选择 driver 参数、Panda/base-inclusive Γ×load conformance | Phase 03 handoff PASS |
| 03 | `phase_03_arena_and_isolator.md` | `robosuite/models/arenas` 工业工作台、6-DoF isolator | analytic transfer/preload |
| 03R | `phase_03_transfer_remediation.md` | 闭合六轴复数传递、联合谱、偏载平衡与 XML default | Phase 04 handoff PASS |
| 04 | `phase_04_environment_task_contacts.md` | VibrationPickPlaceCan、Can/浅箱/contact/evaluator | passive physics/evaluator gates |
| 04R | `phase_04_task_contract_remediation.md` | 闭合 Can collision inertia、support/contact 语义与运行时 asset | Phase 05 handoff PASS |
| 05 | `phase_05_state_tiers_imu.md` | observables、V0–V3、IMU、privilege isolation | sensor/key-set gates |
| 05R | `phase_05_observation_imu_remediation.md` | 闭合 robot-base IMU、exact key-set、连续时间链、int16 与 privilege immutability | Phase 06 handoff PASS |
| 05R-W | `phase_05_gym_wrapper_cleanup.md` | 移除 production fake-Gym、用 test-only harness 验证 wrapper 隔离 | Phase 06 handoff PASS |
| 06 | `phase_06_physics_freeze.md` | dt/solver/weld/contact/isolator freeze | all physics gates |
| 06R | `phase_06_review_remediation.md` | 重新预注册 V2，闭合 candidate coverage、profile authority 与 raw-evidence verifier | all physics gates |
| 06R2 | `phase_06r2_feasible_physics_freeze.md` | 保留 V2 BLOCKED，预注册可完成的 V3 测量与完整物理选择 | all physics gates（当前 V3 因 dt_medium 与 20 Hz scheduler 非整倍数而 BLOCKED） |
| 06R3 | `phase_06r3_v4_physics_freeze.md` | 保留 V3 BLOCKED，先验证 scheduler/采样/solref 可行性，再执行完整 V4 freeze | all physics gates |
| 06R4 | `phase_06r4_v5_protocol_execution_contract.md` | 保留 V4 BLOCKED，用同一 resolver 驱动 dry-run 与执行后完成 V5 freeze | all physics gates（当前 driver 通过，组件参数 materialize 缺失导致 BLOCKED） |
| 07 | `phase_07_oracle_controller.md` | shared 7D State Oracle controller | Gamma=0 10/10 dev |
| 08 | `phase_08_protocol_scorecard.md` | committed states、MDE、scorecard | replay/stat/integrity gates |
| 09 | `phase_09_knee_official_eval.md` | Gamma* + 400×4 official evaluation | complete paired matrix |
| 10 | `phase_10_release_audit.md` | clean-room ShakeBench-fork release audit | no release blocker |

## 禁止跨阶段

- Phase 00 先完成 robosuite→ShakeBench 改名（权威设计文档已随仓），再开始代码实现；不实现 benchmark physics；不新建 shakebench 子文件夹。
- Phase 02 只验证 deck driver，不加入 task。
- Phase 03 不运行 task SR。
- Phase 04 不调 controller 追求成功。
- Phase 06 只看 physics metrics，不看 tier ranking。
- Phase 07 不改已经冻结的 physics/contact/task success。
- Phase 09 是 measurement run；发现 bug 后整轮作废并回到责任阶段。
