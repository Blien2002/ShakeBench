# Phase 09.5 Prompt：协议冻结、Gamma Knee 与 Official Measurement

在 `/home/miracle04/Desktop/ShakeBench` 工作。本阶段只在 Phase 09 资格通过后发布计分 authority，冻结单一受控振动协议，选择 `Gamma_star`，并执行 400 states × V0–V3 的正式测量。

## 前置硬门

完整读取：

- `docs/shakebench_prompts/phase_09_requalification_pilot.md`；
- `docs/phase_09_pilot_report.md`；
- `docs/phase_09_qualification_manifest.json`；
- `docs/phase09_pick_place_states.md`；
- Phase 08R outcome、scorecard、CPU batch 与 direct-mount authority 的当前实现。

开始正式 rollout 前必须满足：

- Phase 09 qualification manifest 为 `PASS`，且所有绑定 hash 可独立验证；
- 六种配置的静态门、正振动 pilot 和跨进程重放仍有效；
- task、physics、controller、outcome、state records 与 geometry authority 已冻结；
- 工作树对上述计分语义无未提交修改；
- scorecard synthetic tests 和两 worker batch smoke 通过；
- 输出目录是新的 run ID，未复用历史 episode。

任一项失败时停止正式测量，记录失败项并回到 Phase 09 或对应责任模块修复。

## 1. 发布计分 State Authority

增加最小的资格发布与验证路径：由非计分 v2 state records 和 Phase 09 qualification manifest 派生冻结的 knee/official 计分资产。计分资产必须绑定完整资格清单、原 v2 payload、task contracts、controller/outcome/physics/geometry authority，并证明 state records 未被增删、替换或按 pilot 结果筛选。

runner、CPU batch 和 scorecard 只在资格验证通过时将该 authority 识别为 scoreable。直接把 v2 的布尔值改成 true、仅依赖文件名或跳过资格验证均不构成发布。

完成标准：重新生成的计分资产逐字段一致；篡改资格清单、state、task contract 或 authority hash 时 loader 和 batch fail closed；相关测试通过。

## 2. 冻结振动协议

Phase 09.5 只使用当前受控六轴波形族。冻结 JSON 协议，至少包含波形/频谱定义、六轴方向与相位关系、Gamma 缩放规则、safe range、coarse/fine grid、ramp、时长、采样率、坐标系、seed、deck conformance 和 table response 字段。

`Gamma_commanded` 是该固定波形族内的强度参数，不解释为跨车辆、船体或飞行器的通用等效强度。真实载具 profile 留给后续泛化实验，不与本阶段的 `Gamma_star` 混合。

完成标准：运行前将完整 grid、插值、舍入、停止和无 crossing 规则写入机器可读协议并绑定 hash；看过任务结果后不得修改。

## 3. Knee Calibration

使用计分 knee authority、V0 和全部 600 states；统计时以 100 个共享 parent 为独立配对/聚类单位，六个 variant 不是 600 个独立 seed。

1. 先运行 `Gamma=0`。总体成功率必须不低于 0.95，每个 variant 不低于 0.90；否则停止并回 Phase 09 扩大静态诊断。
2. 按冻结 coarse grid 扫描 safe range。每个 Gamma 复用相同 600 states。
3. 只在预注册 bracket 内运行 fine grid，不临时增加有利点。
4. 对总体 `SR_V0(Gamma_commanded)` 做 non-increasing isotonic fit；同时报告六个 variant 曲线和 parent-cluster bootstrap 区间。
5. `Gamma_star` 定义为拟合曲线首次达到 absolute SR=0.5 的插值点，并按 0.05 舍入。
6. safe range 内没有 crossing 时输出 `no_valid_decision_point`，停止 official measurement，不外推 `Gamma_star`。
7. freeze artifact 保存 raw grid、fit、区间、舍入、失败分类、actual deck/table response 和全部 authority hash。

完成标准：独立程序能从 raw knee episodes 重算相同 `Gamma_star` 或相同的 `no_valid_decision_point`。

## 4. Official Evaluation

仅在得到有效 `Gamma_star` 后，使用计分 official authority，在同一 `Gamma_star` 对 V0、V1、V2、V3 各运行 400 states，共 1600 episodes。四个 tier 使用相同完整 state ID；按预注册 state block 交错 tier 顺序，避免 tier 与热状态或运行时间混淆。

每 episode 前写 manifest。任务、controller 或物理失败保留在分母；invalid execution 排除分数并使完整 tier matrix 保持 incomplete，直到同一 job identity 成功重试。measurement 中发现影响计分语义的 bug 时，本轮作废，修复、重新冻结并只按明确依赖范围重跑。

## 5. 聚合与交付

唯一主指标为 `SR@Gamma_star`。生成：

- 每 tier 成功率、Wilson 区间、失败类型和 violation；
- V0→V1、V1→V2、V2→V3、V0→V3 的逐 state 配对比较；
- 按物体—表面配置的诊断结果，清楚标注 official 的配置间并非共享 parent；
- ceiling/floor、恢复事件、actual response、rerun 与 incomplete ledger；
- 1600 个 raw results、manifests、scorecard JSON/CSV/Markdown、图和 `docs/phase_09_5_report.md`。

完成条件：

1. knee 与 `Gamma_star` 可从 raw 独立重算；
2. 4×400 matrix 完整，否则报告 incomplete 且不排名；
3. 每个 paired count 可追溯到完整 state ID；
4. 所有正式结果绑定同一冻结协议和计分 authority；
5. measurement 期间计分代码与 authority 不变，仅新的 gitignored `out/` 运行产物增长。

完成后停止。真实载具振动、Vision 策略、训练方法和 release audit 属于后续独立阶段。
