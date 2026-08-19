# 第四轮物理与资产回归验证

更新日期：2026-08-14

> **P0 更新（2026-08-18）**：official 档已从 1000 Hz × 4 改为 **1000 Hz × 5** 子步，启动门改为六轴矢量合成 + 频带上边界 + 真实支撑半径。本文件以下内容记录的是第四轮 **× 4 子步** 的历史验证，不能代表 × 5 档的当前结果。× 5 档 1 s seed=17 谱探针最大穿透 0.665 mm（`robot_link<->platen`，settle 阶段），未通过 0.3 mm 门槛，见 `out/p0_5substep_spectral_probe.json`；安装点坐标一致性修复后需重新验证。

## 配置边界

- 第四轮验证时官方评测与录制：1000 Hz × 4 子步，`--physics-profile official`。
- 训练吞吐档：240 Hz × 4 子步，`--physics-profile training`；不用于正式评分。
- 谱激励的估算峰值子步位移分别为 0.264 mm 与 1.100 mm。
- 接触响应按档位显式设置：官方 `solref=(0.00060 s, 1.0)`，训练 `solref=(0.0025 s, 1.0)`；两者时间常数均不低于各自两个子步。
- 请求 margin 为 1.0 mm，但 `use_mujoco_contacts=True` 的 NativeCCD/MULTICCD 会在转换时将其清零（Newton #2106）。JSON 中记录 `nativeccd_margin_honored=false`，不宣称 margin 已生效。
- `grasp_assist` 默认关闭；只有显式 `--grasp-assist` 才启用，而且进入保持要求穿透小于 0.5 mm，保持中超过 1.0 mm 会解除。

固定加速度 RMS 下 `v=a/ω`。因此不能通过降低频率并增大位移来提高难度，那会增大速度和单步穿透。

## V0 基线与诊断

240 Hz × 4、seed=17、1 s 谱激励的改造前实测见 `docs/penetration_baseline.json`：最大穿透 21.633 mm，发生在工件与桌面之间。这一数值远大于提示词估算的 1 mm，排查确认旧初始化让工件从桌面上方约 0.10 m 自由落下，且名义 YCB 高度与 Newton 实际 collider 不一致。

修复后，任务在运行时读取 Newton collider 包围盒来初始化高度并计算抓取几何。默认 sugar_box@0.75 的物理网格实测为 69.508 × 132.188 × 33.602 mm。调 `solref` 前的 20 ms 无振动探针为 1.197 mm，证明自由落体/尺寸错配已被分离出来，剩余误差来自默认接触柔度。

## 当前验证状态

- 单元与几何回归：`29 passed`。
- 接触拓扑：386 Newton shapes / 29 MJWarp geometries / 348 candidate pairs；相对第三轮 385 / 29 / 348，纯视觉改造的 MJWarp geometry 与 candidate pair 增量均为 0。
- 父变换审计：30 个锚点，最大名义偏差 0 mm，小于 5 mm 门槛。
- 官方 1 s 谱回合：最大穿透 0.222 mm（工件↔桌面，t=0.924 s），超过 0.5 mm 的帧占比 0，腕力峰值 36.78 N；通过 0.3 mm 门槛。机器可读结果见 `docs/penetration_official_final.json`。
- 当前训练档同场景：最大穿透 1.725 mm，明确不用于评分。训练/官方壁钟时间分别为 16.23 s / 62.47 s，官方档约慢 3.85 倍；两者 outer-step 吞吐约 14.85 / 16.01 steps/s，差异来自每模拟秒的 outer-step 数为 240 / 1000。
- 原生 smoke：本机仍可复现既有的 Isaac/Newton 模型构建阶段 `malloc(): unaligned tcache chunk detected` 或无 Python traceback 退出；重试后可完整生成上述 JSON。

## 控制器连续命令与指尖净空追加验证

在官方 4 s smoke 暴露 1682.63 N 腕力峰值后，进一步诊断确认旧控制器存在两个命令不连续：approach→descend 时末端目标一次下跳约 0.196 m，进入 grasp 时每个指关节目标一次从 0.040 m 跳到 0.012 m。现已加入分阶段笛卡尔速度限制、夹爪速度限制和 0.1 mm 接触预载。

精确 shape 记录随后确认残余冲击来自 `panda_leftfinger/collisions` 与桌面。Newton 实测该 collider 相对指节原点向下伸出 53.85 mm，而 sugar_box@0.75 高度只有 33.60 mm；把指节原点放在工件顶面上方 4 mm 必然先撞桌。控制器现在用实时指尖 collider、实时工件投影高度和 1 mm 桌面净空计算安全预抓高度。碰桌不会进入 grasp，而是记录 `descend_table_contact` 并失败。

训练档、无振动、seed=17 的逐步结果：

| 控制器版本 | 时长 | 最大穿透 | 腕力峰值 | 双指接触 | 主要结论 |
|---|---:|---:|---:|:---:|---|
| 旧阶段跳变 | 5 s | 1.498 mm | 1202.79 N | 是 | 命令跳变与指尖碰桌 |
| 仅加入轨迹限速 | 5 s | 0.819 mm | 161.52 N | 否 | 精确定位为左指 collider 碰桌 |
| collider 自适应净空 | 6 s | 0.280 mm | 148.55 N | 是 | 指尖—桌面碰撞消失，但抬升时工件滑落 |
| 慢速闭合与丢失保护 | 6.5 s | 0.348 mm | 183.95 N | 曾确认 | 接触丢失被诚实记录，未继续假搬运 |

最后一项之后又修正了接触保持目标：首次接触时使用“实测关节位置与已下发目标中更闭合者”作为预载基准，防止因执行器滞后而在确认接触后重新张开。该修改的后续运行两次都在 Newton 模型构建阶段原生退出，未得到新的 JSON，不能宣称已通过动态验证。

新增的力诊断会记录峰值时刻、控制阶段、同时发生的精确 contact shapes，以及左右指尖峰值。当前结论仍是：穿透和假成功路径得到进一步控制，但参考抓取的瞬时力与保持能力尚未达到可发布基线；不应恢复 assist 或放宽阈值掩盖失败。

最终 0.10 s 训练档录制 smoke 也在同一模型构建阶段原生退出，`out/fourth_round_recording_smoke.mp4` 与对应 JSON 均未生成。因此录制链路在本次继续修改后仍标记为“未重新验证”，不能沿用第三轮历史 MP4 作为通过证据。

## 复现命令

```bash
./run.sh --physics-profile training --episode-s 1 --vibration spectral \
  --seed 17 --metrics-output out/penetration_training.json

./run.sh --physics-profile official --episode-s 1 --vibration spectral \
  --seed 17 --metrics-output out/penetration_official.json

./run.sh --physics-profile official --record --episode-s 16 \
  --vibration spectral --seed 17 --workpiece sugar_box \
  --workpiece-scale 0.75 --output out/benchmark_v2_fourth_round.mp4 \
  --metrics-output out/benchmark_v2_fourth_round.json
```

新的控制器允许困难回合诚实失败；不能通过恢复默认 assist、放宽成功阈值或放宽穿透门槛来制造 `success=true`。
