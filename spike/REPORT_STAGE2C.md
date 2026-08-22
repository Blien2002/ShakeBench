# ShakeBench spike Stage 2C report

## 结论

Gate 1 通过，但 Gate 2 未通过；依照预先规定的门禁顺序，Stage C 和 D rollout 未运行。
A3 已把接触获取与 latch 后保持权限拆开；失败不再是获取超时，而是低保持力下的真实接触丢失/滑移。

## A3：获取权限 / 保持上限拆分

接触获取阶段为 20.0 N/指；首次双侧 latch 后切到 3.0 N/指。
切换发生在 policy step 81、model step 20250、t=4.050 s。切换前后 `actuator_forcerange` 校验均通过。
事件写入 phase history，包含时间、步索引、切换前后完整 actuator 配置。

冻结重放不做在线 latch 判断；磁带额外记录力上限切换的 policy-step 索引，重放时在同一索引开环改写 `model.actuator_forcerange`。A3 验证中 reactive 与 frozen 均在相同 model step 完成切换。

确定性检查：同 seed 动作逐位一致，`max|Δaction|=0`；200 Hz 获取轨迹结构完全一致；两次切换 model step 相同。

### 获取阶段实测

采样 810 点（0.005–4.050 s）。两指 actuator 峰值绝对力为 5.779/5.830 N；指关节速度峰值绝对值为 0.04295/0.04286 m/s。
这把 Stage2B 中‘1/1.5 N 失败来自接触获取受限’从相位分类推断升级为带 actuator 力与关节速度轨迹的实测。

## B1′：A3 配置下的全新配对重跑

| Γ | 成功 | 最大 grasp slip 中位/P90/最大 (mm) | 双侧脱接触 中位/P90/最大 | warning |
|---:|---:|---:|---:|---:|
| 0.00 | 20/20 | 1.592/1.733/1.789 | 0.0%/0.0%/0.0% | 0 |
| 0.50 | 20/20 | 2.388/3.263/3.798 | 0.0%/0.3%/0.4% | 0 |
| 0.95 | 20/20 | 4.601/6.672/8.809 | 0.2%/0.6%/0.7% | 0 |

Gate 1：**PASS**（Γ=0 为 20/20）。这些结果均在新目录重新生成，未复用 Stage2B rollout。

## B2′：Γ=0.95 敏感性

### latch 后保持力曲线

预写预测：A3 拆分后 1/1.5/3/6 N 四点全部 100%。

| 保持上限 (N/指) | 成功 | slip 中位/P90/最大 (mm) | 双侧脱接触 中位/P90/最大 | 失败分类 |
|---:|---:|---:|---:|---:|
| 1.0 | 3/20 | 15.736/21.100/22.722 | 38.4%/62.5%/71.1% | `{'grasp_slip_exceeded': 17}` |
| 1.5 | 14/20 | 5.851/13.274/18.025 | 1.5%/51.1%/52.0% | `{'grasp_slip_exceeded': 6}` |
| 3.0 | 20/20 | 4.601/6.672/8.809 | 0.2%/0.6%/0.7% | `{}` |
| 6.0 | 20/20 | 4.305/6.230/6.790 | 0.0%/0.2%/0.5% | `{}` |

预测对照：**不符合**。1 N 点在 A3 后可以完成 latch，因此其非零滑移是保持阶段实测，不再是旧规格混入的 acquisition timeout。
3 N 与 1.5/6 N 的 SR 差分别为 30.0/0.0 个百分点。

### 其余扫描

| 参数 | 基线 → 备选 | 成功（基线/备选） | 翻转 | 门禁 |
|---|---|---:|---:|---:|
| pad solref damping ratio | 0.5 → 1.0 | 20/20 / 20/20 | 0/20 | PASS |
| OSC kp | 150.0 → 300.0 | 20/20 / 12/20 | 8/20 | PASS |
| physics timestep | 0.0002 → 0.0001 | 20/20 / 20/20 | 0/20 | PASS |
| cube-table solref | 0.0006 → 0.0012 | 20/20 / 20/20 | 0/20 | PASS |
| move-action gain (diagnostic) | 4.0 → 1.0 | 20/20 / 0/20 | 20/20 | 诊断项 |

Gate 2：**FAIL**。保持力项按平台判据，其余门禁项按 `<50%` 配对翻转判据；move-action gain 不计门。

## D0：实测量与任务容差（显著对照）

| 实测量 | Γ=0.95 值 (mm) | 容差 (mm) | 占比 | 来源 |
|---|---:|---:|---:|---|
| hard-mounted EE wobble | 2.380 | 4.000 | 59.5% | src/shakebench/config.py:439 descend_clearance_m |
| hard-mounted maximum grasp slip | 8.809 | 10.000 | 88.1% | src/shakebench/config.py:456 grasp_slip_tolerance_m |
| table-frame object slip | 0.0007 | — | — | Stage2B B3 validated instrument |
| decoupled base-frame table motion (pi/2 diagnostic) | 5.108 | 4.000 | 127.7% | src/shakebench/config.py:439 descend_clearance_m |
| hard-mounted EE wobble | 2.380 | 12.000 | 19.8% | src/shakebench/config.py:440 finger_table_clearance_m |

**关键点：The Stage2B pi/2 decoupled diagnostic table motion is the first measured quantity above a task tolerance. It is diagnostic-only: if D2 is later reached, the free phase choice must be replaced by a physically explicit transfer function before policy rollout.**

## C：Γ 阶梯

**Not run**：Gate 2 未通过。没有用部分或调试 rollout 填表。

## D：设计探针

**Not run**：D 仅在 C 判为两行都平时触发，而 C 未获准开始。

## 限制与诚实性说明

- 20 个确定性种子只给出该初始化集合的经验频率，不是总体置信区间。
- A3 后所有 B1′/B2′ 结果均重跑；Stage2B 只作为提示词明确允许沿用的已验证溯源与 B3 仪器证据。
- 固定 10 mm grasp-slip 容差、超时、Γ、频带与辅助设置未为过门而放宽。D1 的 μ=0.4 明确隔离为设计探针。
- 本轮未获准运行 D2；旧 B3 的统一 π/2 相移只保留为仪器诊断，不是策略 rollout 证据。
- 任何未配套 B2′ 的 Γ 下降只标为未验证，不进入结论。

原始数据位于 `out/stage2c/`；所有 episode JSON 保留动作磁带、切换索引、失败分类、200 Hz 接触力统计与 MuJoCo warning 计数。
