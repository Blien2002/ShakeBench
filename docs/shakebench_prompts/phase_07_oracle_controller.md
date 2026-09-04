# Phase 07 Prompt：在新环境上实现共享 State Oracle Controller

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接为 `VibrationPickPlaceCan` 增加 reference State Oracle policy 和运行脚本；physics/contact/task success 已冻结，本阶段不得修改它们，不新建子文件夹。

## 前置与必读

- Phase 06 official physics profile 全门通过。
- 读取 `docs/robosuite_benchmark_design_tree.md` shared controller、State Track、failure integrity 和 Gamma=0 gates。
- 参考随仓 `docs/spike_implementation_formal_review.md` 记录的失败机制，不复制其 8D gripper、force-switch 或 `move_action_gain`。

## 首个可执行步骤：Phase 06 handoff gate

在 import controller、创建环境或读取任何 Phase 07 state 之前，只运行：

```bash
python -m robosuite.scripts.shakebench_finalize_physics --verify-final
```

这个单一 verifier 必须验证
`shakebench_phase_06_to_07_handoff.json`、final status、official profile、
Phase 06F protocol、selection/dependent manifest、全部 evidence 与 package hash。
任一 hash、winner binding、三进程 replay 或 profile tuple 不一致时 Phase 07
必须拒绝启动；V1–V8 historical status 不能替代该 handoff。

## 修改落点（不新建目录）

```text
robosuite/utils/shakebench_oracle.py
robosuite/utils/shakebench_providers.py   # 与 Phase 05 集成
robosuite/scripts/shakebench_run_oracle.py
tests/test_shakebench_oracle.py
```

policy 使用环境公开 observation/action API，不通过 `env.sim` 旁路读取 privileged truth。

## 唯一执行链

```text
State task observation
→ VibrationProvider(V0/V1/V2/V3)
→ typed VibrationEstimate
→ one SharedVibrationControlLaw
→ one TaskExecutive
→ robosuite 7D total action = 6D OSC_POSE + 1D gripper
```

V0 zero dedicated estimate；V1 从 noisy IMU 估计；V2 current realized state；V3 在固定 latency horizon 查询 authored future。V2 不保存 support history/辨识 program。

## TaskExecutive

实现明确 phase/deadline/failure：

```text
settle→approach→descend→grasp→lift→transport→place→release→verify
```

- target pose 每步由 moving target-container frame 计算；
- bilateral contact、finger-table contact、grasp loss、slip、timeout fail closed；
- success 只由 environment privileged evaluator 返回，policy 不自报。

## Action/controller验证

- 使用 robosuite 标准 Panda OSC_POSE+default gripper action shape；
- policy rate=20 Hz；
- 记录 normalized/raw/decoded/clipped/applied action；
- 验证 OSC translation/rotation scale、kp/damping、gripper actuator force/position；
- applied force/torque 是证据，不只检查配置。

## 调参边界

- 只使用 10 dev states 和 Gamma=0；
- 先完成完整 task chain；
- controller profile 冻结后，positive-Gamma smoke 只验证稳定和 provider 路径，不回调参数；
- V0–V3 的 TaskExecutive、gains、rate、grip、horizon 和 action injection hash 相同。

## Gamma=0 gate

10 个 dev states 上 State-V0 必须 10/10 成功。失败时只修 controller/task implementation；physics/contact/success threshold 保持冻结。

## 必须测试

- action channel positive controls；
- phase/deadline/failure transitions；
- moving-target frame commands；
- bilateral/contact-loss/slip gates；
- identical controller hash across tiers；
- V2 current-only / V3 fixed-latency query；
- Gamma=0 matched action path；
- applied actuator validation；
- 3-process deterministic controller trace；
- no `env.sim`/privileged dependency；
- existing controllers/env tests 无回归。

## 完成条件

1. Gamma=0 dev 10/10；
2. frozen controller profile 可由 package 安装读取；
3. V0–V3 仅 provider input 不同；
4. positive-Gamma smoke 未触发参数修改；
5. 输出 raw dev episodes、action/force evidence 和 `docs/phase_07_report.md`；
6. 未新建目录。

完成后停止，不生成 official states 或扫描 Gamma knee。
