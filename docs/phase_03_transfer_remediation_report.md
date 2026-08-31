# ShakeBench Phase 03R transfer-remediation report

状态：**PASS；Phase 04 handoff = PASS**

本阶段修复并验证了 Phase 03 的三个证据缺口：复数 harmonic transfer、联合 authored
spectrum 的频线分离与 leakage、以及 world-child payload 的偏载平衡。所有验证仍是
physics-only；没有实现或运行 Can task、success evaluator、controller、IMU、official
scoring，也没有冻结最终 `f_n/zeta`。

## 1. 物理定义与 default equality

唯一 canonical worktable 定义：

```text
dimensions = [0.65, 0.60, 0.06] m
M_ref      = 32.0 kg
I_ref      = [0.9696, 1.1363, 2.0867] kg m^2
COM = elastic center = body origin = principal frame
```

`robosuite/models/assets/arenas/shakebench_arena.xml` 的 raw asset 直接交给 MuJoCo
编译，不调用 `ShakeBenchArena.configure_isolator()`，再逐项与
`DEFAULT_ISOLATOR_CONFIG` 派生值比较：mass、COM、inertia、六个 joint type/axis、
stiffness、damping、springref、range 和 limit 共 **33/33 checks passed**。

默认 probe candidate 为六轴 `f_n=5 Hz`、`zeta=0.1`，仅用于 sweep/evidence，不是
official operating point。对应：

```text
k = [31582.7340835, 31582.7340835, 31582.7340835,
       956.9568427,  1121.4831481,  2059.4903504]
c = [201.0619298, 201.0619298, 201.0619298,
        6.0921765,   7.1395835,   13.1111228]
springref = [0, 0, 0.0099396081153, 0, 0, 0]
```

## 2. 复数 harmonic transfer

每个 fit 使用同一 reference point/frame：

```text
input point  = dynamic deck body origin / deck site
output point = worktable body origin = COM = elastic center
input frame  = nominal deck frame
relative output = worktable COM relative to deck origin, realized deck frame
absolute output = worktable COM in nominal worktable frame
```

时间合同为：在 `t` 写入 `q(t)`，积分到 `t+dt`，写入 right-limit `q(t+dt)`，执行
`forward`，在 `t+dt` 采样。fit 使用：

```text
x(t) = a_s sin(wt) + a_c cos(wt) + b
complex coefficient = a_c - i*a_s
H = output coefficient / input coefficient
phase = arg(H), wrapped to [-pi, pi)
```

harmonic grid 为每个 axis 的：

```text
tracking  = 0.2 f_n = 1 Hz
resonance = 1.0 f_n = 5 Hz
isolation = 2.0 f_n = 10 Hz
```

共 18 格，使用 10-cycle transient discard、20-cycle fit、`dt=0.0002 s`、sample
stride 16（312.5 Hz）。每格同时验证 `H_rel` 和 `H_abs` 的复数系数，独立检查
amplitude、phase 和 normalized residual：

```text
18/18 records passed
max |relative amplitude error| = 8.633e-4
max relative phase error       = 0.1763 deg
max relative fit residual      = 0.004345
max cross-axis leakage         = 0
```

预注册阈值为 amplitude error `<=5%`、phase error `<=3 deg`、normalized residual
`<=2%`。这些是 measurement gates，不是最终 physics-freeze 选择规则。

## 3. 联合 six-axis authored spectrum

联合 probe 使用 Phase 01 candidate：

```text
profile_id             = shakebench.authored_v0_candidate
authored version       = candidate-2026-08-30
seed                   = 17
t0                     = 0.137 s
level_scale            = 0.3
active lines           = 64
minimum global spacing = 0.0087876394667 Hz
fit window             = 227.592404942 s = 2 / minimum spacing
```

与短窗口试跑不同，正式 fit 对全部 64 条 active lines 使用一个共享多频设计矩阵，
同时提取六个 deck input、六个 relative table output 和六个 absolute table output。
这样近邻 authored lines 不会被错误地计为跨轴响应或 residual。

```text
diagonal complex transfer gate = PASS
cross-axis leakage gate        = PASS
safety/travel gate             = PASS
line fit count                 = 64
max relative amplitude error   = 0.1710%
max relative phase error       = 0.1788 deg
max relative residual          = 0.1303%
max cross-axis leakage         = 5.194e-4
```

联合 metrics 使用同一 reference contract，并记录：

```text
T_peak             = 5.1227671
D_relative        = 0.00232031 m / 0.00144578 rad
travel margins     = [0.0227053, 0.0244667, 0.0242792] m
angle margins      = [0.0864793, 0.0859131, 0.0870043] rad
```

harmonic linearization 明确记录为 zero gravity、`isolator_tz springref=0`。这是为了
测量线性 base-excitation transfer；正常重力与 `M_ref*g/k_z` preload 没有被删除，
而是在下一节的 static payload probe 中独立验证。

## 4. world-child payload sensitivity

payload 保持 world child + freejoint，通过真实 tabletop contact 传递重力；为避免
sphere 在偏载姿态下 rolling 导致 COM 漂移，probe 使用有平底的显式质量 box。它仍
不是 task object，也没有运动学挂到 worktable。

```text
mass = 0.349 kg
COM xy offsets = [(0,0), (0.10,0), (0,0.20), (0.10,0.20)] m
settle duration = 8 s
equilibrium window = 1 s
```

四格均有最终 `payload_geom ↔ table_collision` contact、zero warnings，且 compiled
worktable support parameters 与 payload-free signature 相同。结果：

| COM offset (m) | measured `[tz, rx, ry]` | analytic `[tz, rx, ry]` | COM offset error (m) |
| --- | --- | --- | ---: |
| `(0, 0)` | `[-0.108403851 mm, ~0, ~0]` | `[-0.108403851 mm, 0, 0]` | `2.8e-16` |
| `(0.10, 0)` | `[-0.108403851 mm, ~0, 0.305355588 mrad]` | `[-0.108403851 mm, 0, 0.305282340 mrad]` | `1.26e-5` |
| `(0, 0.20)` | `[-0.108403851 mm, -0.715738258 mrad, ~0]` | `[-0.108403851 mm, -0.715536970 mrad, 0]` | `2.96e-5` |
| `(0.10, 0.20)` | `[-0.108403851 mm, -0.715738306 mrad, 0.305355504 mrad]` | `[-0.108403851 mm, -0.715536970 mrad, 0.305282340 mrad]` | `3.22e-5` |

四格 equilibrium/sign/magnitude/support/safety gates均通过。`k/c` 和 springref 没有
按 payload 或 COM offset 重算。

## 5. Visual invariance and artifact

visible/hidden arena 的编译 physics hashes 完全一致，RGBA hash 不同；同一 `tz, 5 Hz`
transfer trace 的 `H_rel/H_abs` 逐项一致。visual layer 不改变 compiled physics 或
transfer trace。

机器可读证据：

[tests/shakebench_phase_03_transfer.json](../tests/shakebench_phase_03_transfer.json)

```text
schema_id/version = shakebench.phase03r.transfer / 1
file SHA-256      = 95dc0e40fadf0d82f5e022b3f5b02097c734f7841d8fbe8612b67b3e04413aef
payload SHA-256   = 86d3c7bb9e177ca957c670140f958b1210881bfb1121cdde2668534d7ddc5fb6
integrity         = PASS
phase04_handoff   = PASS
```

artifact 更新使用 explicit reason，并保存 previous/new hash 字段；默认 verifier 只读，
不会改写 artifact。

## 6. Verification

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_transfer_remediation.py --tb=short
3 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_isolator.py tests/test_shakebench_arena.py --tb=short
19 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_deck_driver.py \
  tests/test_shakebench_isolator.py tests/test_shakebench_arena.py --tb=short
78 passed
```

Phase 03R 的 handoff 只证明 isolator transfer/default/payload/visual physics gates；
`f_n/zeta/k/c` official freeze 仍由 Phase 06 负责。
