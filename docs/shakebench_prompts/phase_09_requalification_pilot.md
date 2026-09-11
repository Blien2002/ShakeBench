# Phase 09 Prompt：六种任务配置重认证与配对 Pilot

当前低、中摩擦物体分别为 RoboCasa Objaverse `canned_food_18` 食品罐头和 `boxed_food_0` 饼干盒；饼干盒以 90° 初始 yaw 放置。旧钢块和木方块的 selection、raw runs 和资格结论仅作为历史证据，新认证应绑定当前资产重新冻结 selection。

在 `/home/miracle04/Desktop/ShakeBench` 工作。本阶段回答两个问题：六种新配置在静态条件下是否可由共同 State Oracle 完成；当前受控波形在完整 pick-and-place 中是否产生可记录、可解释的变化。本阶段只生成非计分资格证据，不运行 600-state knee scan 或 4×400 official measurement。

## 先读与边界

完整读取：

- `README.md` 顶部 ShakeBench 状态；
- `docs/phase09_pick_place_states.md`；
- `docs/phase_08r_failure_boundary_report.md`；
- `docs/manipulation_benchmark_failure_definitions.md`；
- `robosuite/utils/shakebench_task_states.py`；
- `robosuite/scripts/shakebench_run_oracle.py`。

保留 `shakebench_task_states_{knee,official}_v2.json` 的 `prepared_for_requalification`、`scoreable=false` 语义。Phase 09 的输出不能进入正式 scorecard。复用现有任务、runner、outcome contract、语义 verifier 和 artifact hash；只在发现阻断本阶段结论的实现缺陷时修复根因并重跑受影响证据。

## 1. 冻结 Pilot 选择

从 knee-v2 的 100 个共享 parent 中按 parent 索引固定选择 `0, 24, 49, 74, 99`，每个 parent 包含全部六种物体—表面配置，共 30 个 state。开始 rollout 前写出 `out/phase09_pilot/selection.json`，保存选择规则、30 个完整 state ID、parent/state payload hash、代码状态和所有输入 authority hash。

选择规则与 state 列表写出后保持不变。不得按运行结果替换位置、物体、表面、seed 或初始状态。

完成标准：从原 knee-v2 资产可独立重建完全相同的 30 个 state，且六种配置各 5 个。

## 2. 静态完整任务门

使用 V0、`Gamma=0` 和正式 horizon 运行 30 个 state。每个 episode 必须经历完整的接近、抓取、搬运、释放和释放后稳定判定，短程 smoke、接触探针和物体自由滑移不计入本门。

静态门要求：

- 30 个 episode 全部通过 raw artifact 和 outcome 语义验证；
- 每种配置 5/5 成功；
- 成功来自环境判据，controller event 不直接写入 outcome；
- 无效执行、数值异常、越权观测或不一致任务上下文为门失败。

若失败，先用现有 phase/event trace 分类为任务不可达、共同 controller 缺陷、接触/资产缺陷、评分缺陷或基础设施错误。修复实现缺陷后从同一 `selection.json` 全量重跑静态门。不能删除困难 state，也不能修改物理、摩擦或成功阈值来追求通过。

完成标准：保存 30 个完整 raw episode、独立 verifier verdict、按配置与阶段聚合的静态报告。

## 3. 正振动配对 Pilot

静态门通过后，对相同 30 个 state、相同 V0 controller 和相同 horizon 分别运行 `Gamma=0.60` 与 `Gamma=0.95`。三组条件严格复用 object pose、excitation seed、IMU seed 与 `t0`。记录 commanded Gamma 及实际 deck/table 六轴响应。

正振动条件只要求全部 episode 有效且可验证；任务成功率可以下降、持平或上升。报告：

- 每种配置的成功率及相对静态的配对变化；
- 首个异常阶段与最终失败原因；
- 抓取前物体滑移、抓持丢失、搬运误差、目标外释放和释放后失稳；
- 恢复事件、是否恢复及恢复时间；
- 实际激励与 commanded Gamma 的一致性和适用范围。

完成标准：90 个主 pilot episode 全部可追溯；每个配对单元都能从 raw artifact 重算，不以必须观察到退化作为通过条件。

## 4. 跨进程重放

从五个 parent 中固定使用 parent 索引 49 的六种配置，在三个全新进程中分别重放 V0/`Gamma=0`。使用仓库已有 determinism 与语义检查；如现有入口不能表达六 state 的重放，只增加最小调用层。

完成标准：18 次重放的初始状态、任务上下文、终止结果和既有 determinism 投影一致，且每次都通过独立 artifact verifier。

## 5. 资格清单与结论

生成：

- `out/phase09_pilot/`：selection、raw runs、verdicts、聚合表和少量失败视频；
- `docs/phase_09_pilot_report.md`：设置、结果、失败分类和限制；
- `docs/phase_09_qualification_manifest.json`：绑定代码状态、v2 knee/official payload、30-state selection、90 个主 episode、18 次重放及所有 verifier hash。

资格清单状态只能是：

- `PASS`：静态 30/30、90 个主 episode 有效、跨进程重放通过；授权进入 Phase 09.5 的协议冻结与计分 authority 发布；
- `BLOCKED`：列出阻断的具体配置、state 和受影响结论；
- `INVALIDATED`：测量期间影响任务、物理、controller、outcome 或 state identity 的代码发生变化，需重跑相关证据。

本阶段结束时 v2 资产仍为 `scoreable=false`。不要执行 knee 扫描、拟合 `Gamma_star`、生成正式排名或发布 benchmark 分数。
