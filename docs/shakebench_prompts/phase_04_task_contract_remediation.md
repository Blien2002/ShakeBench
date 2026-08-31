# Phase 04R 修复提示词：闭合 Can collision inertia、support/contact 语义与发布资产

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 04 环境可运行，但 task physical contract 仍有三个 blocker：Can 惯量取自 stock placement sites 而非 compiled collision geometry、`contact_loss` 有两种冲突含义、MJCF 引用的 runtime PNG 未纳入 Git。本阶段只修复任务物理/metrics/reproducibility contract；保留工作树，不重置，不新建目录，不实现 Phase 05 的 IMU、V0–V3 providers、privileged recorder 或 Oracle。

leading word 是 **权威**：每个可报告量只能有一个可编译、可审计的来源。collision envelope、placement correction、惯量、support、contact loss 和 package asset 不能各自使用近似或隐含的第二套定义。

## 必读与范围

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_04_environment_task_contacts.md
docs/robosuite_benchmark_design_tree.md       # task, Can, contact, success sections
docs/robosuite_benchmark_v0_spec.md
docs/phase_04_report.md
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/utils/shakebench_metrics.py
robosuite/models/assets/arenas/shakebench_arena.xml
tests/test_environments/test_vibration_pick_place_can.py
tests/test_shakebench_metrics.py
```

允许修改 Phase 04 environment/metrics/tests/docs、`.gitignore`、`MANIFEST.in` 和已有 texture asset。不得创建 Phase 05 modules 或把 privileged truth 放进普通 observation。

## 1. 冻结唯一 Can collision-envelope authority

当前错误是：惯量从 `CanObject.top_offset - bottom_offset`（placement sites）派生，而 Phase 04 已测得 compiled collision mesh 的实际支持高度不同。解决方案：

1. 在一个明确的 canonical compiled pose 中，从**已编译 Can collision geoms**的 vertices/primitive support points 定义 `CanCollisionEnvelope`：source geom names、radius、height、lower/upper support z、source model hash、提取算法版本；
2. 用这个 envelope 的 radius/height 计算 canonical equivalent-cylinder inertia，并将 resulting numeric mass/COM/inertia 作为 compiled assertion；
3. placement height 也从同一 envelope 的 lower support z 单独校正。placement site 可保留给 stock visual/API，但不能再进入惯量公式；
4. 删除或显式弃用将 `top_offset/bottom_offset` 命名为 collision envelope 的常量/API；不得保留两个可能漂移的 authority；
5. 增加 regression：若 compiled mesh height/radius、source geom list、inertia 或 placement correction 有任一漂移，则环境 compile/audit fail closed；
6. 更新 artifact/report，说明旧 0.100 m site-based 数字无效，记录新 collision-derived values与 hashes。

## 2. 修复 support 和 contact-loss 的语义

### 2.1 Target bottom support

`target_bottom_contact=True` 不能单独等于 “supported by target bottom”。success 需要物理 support：

- 从 target-local 或 worktable-local contact force / normal 求 Can 受到的向上支撑分量，并使用一个明确、预注册的严格 positive support-force threshold；
- 同时验证 Can 碰撞 lower support 不在 target bottom 的下方（带明确 tolerance），防止 Can 从底板下方/侧面接触被判成功；
- 保持 containment、无 finger contact、relative speed、penetration 和 0.50 s latch 的原有独立条件；
- 加状态注入正负测试：箱内底板上静止通过；箱底下方接触/侧面擦碰失败；wall-straddling失败；支持力边界失败；Can 高于墙仍可通过 containment。

### 2.2 Contact loss

一个 metrics report 内只能有一个含义：

```text
finger_can_contact_present              # current sample instantaneous contact
finger_contact_loss_after_grasp         # stateful: a prior grasp contact existed, current sample lost it
```

- 删除或弃用歧义 `contact_loss` 字段；不能让 `ContactReport` 与 `MetricsSnapshot` 对同名字段给出冲突值；
- 保留 first-slip、in-hand translation/rotation 与 interface contact reports，但逐字段记录是否 instantaneous / stateful；
- reset、首次接触、持续接触、释放、从未抓住的无接触都写 regression 和 JSON schema tests。

## 3. 处理 Phase 05 的 world-frame 防线，但不实现 tiers

当前 generic object observables 可能提供 world-frame Can pose。Phase 04R 不实现 V0–V3，但必须做其中一种明确边界：

1. 将任何 policy-facing Can task state 改为 robot-base frame；或
2. 在 `use_object_obs=True` 时 fail closed / 标记为 non-benchmark legacy output，并让 Phase 05 只能从明确的 robot-base common-state seam 构建 policy observation。

增加 test，确保 Phase 05 不会无意复用 world-frame `can_pos/can_quat` 作为 State V0 公共字段。不要在本阶段添加 IMU、program 或 privileged fields。

## 4. 发布资产与 artifact lock

- 将 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png` 纳入 Git：添加最小 `.gitignore` 例外，确认 `git ls-files` 可见；
- 测试应验证 **MJCF 实际引用的 PNG**，而不只检查 source JPG；
- 增加 source-distribution/clean-compile smoke：构建 source artifact 或等价临时安装输入，确认 PNG 存在并可被 `shakebench_arena.xml` 编译；
- 更新 Phase 04 evidence artifact：记录 collision-envelope authority、support/contact schema version、runtime texture hash、clean-compile result；默认验证只读，更新需显式 reason 和 old/new hash。

## 5. 报告与停止

更新 `docs/phase_04_report.md`，新增 `docs/phase_04_task_contract_remediation_report.md`。报告必须列出 collision envelope numeric authority、support-force definition、contact-loss state machine、policy-frame boundary、asset tracking/compile evidence，以及唯一 `Phase 05 handoff = PASS|BLOCKED`。

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_metrics.py \
  tests/test_environments/test_vibration_pick_place_can.py \
  tests/test_shakebench_config.py --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_arena.py \
  tests/test_shakebench_isolator.py \
  tests/test_shakebench_transfer_remediation.py --tb=short
git ls-files --error-unmatch \
  robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png
```

完成定义：Can inertia/placement 均从单一 compiled collision envelope 派生；success 证明 bottom support 而非任意接触；metrics contact loss 语义不矛盾；runtime PNG 可从 clean source 输入编译；Phase 05 无 world-frame common-state leakage。任一项不满足，报告 BLOCKED 并停止在 Phase 04R。
