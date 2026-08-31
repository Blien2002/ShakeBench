# ShakeBench Phase 03 report

状态：**基础实现报告；transfer evidence 已由 Phase 03R remediation supersede**

Phase 03 的原始报告只覆盖单一 4 Hz 幅值 probe。由于它没有覆盖复数 phase、三频段
grid、联合谱 leakage 和完整 payload COM sensitivity，不能单独作为 Phase 04 handoff。
请以 [Phase 03R transfer remediation report](phase_03_transfer_remediation_report.md)
和 [Phase 03R artifact](../tests/shakebench_phase_03_transfer.json) 为当前证据。

执行范围是 `ShakeBenchArena`、canonical linear 6-DoF isolator、Phase 02
role-based deck handoff 和无 task-success 的 MuJoCo probes。没有注册新环境、没有运行
task SR，也没有根据任务结果选择 `f_n` / `zeta`。

## 1. 实现落点

| 路径 | 结果 |
| --- | --- |
| `robosuite/models/arenas/shakebench_arena.py` | 新增工业工作台、显式 inertial、6 个 isolator joint、视觉开关、target assembly seam 和 compiled audit |
| `robosuite/models/assets/arenas/shakebench_arena.xml` | 新增 arena MJCF；table collision 与工业装饰层分离 |
| `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png` | 由仓内 phenolic JPEG 确定性 PNG 转码，供当前 MuJoCo loader 使用 |
| `robosuite/utils/shakebench_isolator.py` | typed config、`k/c` 与 preload 派生、解析 transfer、payload equilibrium 和 fail-closed safety report |
| `tests/test_shakebench_arena.py` | arena、deck handoff、visual invariance、target seam、空载 / payload probes |
| `tests/test_shakebench_isolator.py` | 解析式、sweep、safety 和六轴 MuJoCo harmonic probes |

`robosuite/models/arenas/__init__.py` 已导出 `ShakeBenchArena`；`TableArena` 未修改。

## 2. Canonical worktable contract

```text
dimensions = [0.65, 0.60, 0.06] m
mass       = 32.0 kg
inertia    = [0.9696, 1.1363, 2.0867] kg m^2
COM / elastic center / body origin / principal frame = same point and frame
```

MuJoCo compiled audit：

```text
nbody = 6   (after Phase 02 deck processing)
njnt  = 7   (deck freejoint + six worktable joints)
ngeom = 38
worktable mass    = 32.0 kg
worktable COM     = [0, 0, 0] m in body frame
worktable inertia = [0.9696, 1.1363, 2.0867] kg m^2
```

工作台只有一个 direct explicit inertial。tabletop collision 是唯一默认接触
geom；frame、横撑、脚板、螺栓、mount 和 tabletop visual 都是 `group=1` 且
`contype=conaffinity=0`，不会贡献 contact。可选 target container 由五个 collision
boxes 加对应 display-only boxes 组成，仍作为 worktable 的刚性 assembly，不增加 joint
或隐藏质量；Phase 03 默认不添加它。

仓内 JPEG 源保持不变，SHA-256 为
`6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb`。
当前 probe 环境的 MuJoCo texture loader 对该 JPEG 报 `Non-PNG texture`，所以 MJCF
使用同一源图的确定性 RGB PNG 转码，SHA-256 为
`87a5478e7325b7d79fc8073afefd3ac7c44b44d6c804ecb586c1641d8157e162`。这只影响视觉
asset，不影响物理 signature。

## 3. Deck handoff and topology

`ShakeBenchArena.deck_body_handles` 显式返回：

```text
isolated_worktable -> worktable
```

`make_deck_processor()` 将该 handle 与可选的显式 Panda/base handles 一起交给 Phase 02
processor。处理后 parent graph 为：

```text
world
├── deck_driver  (mocap, no geom)
└── deck         (ordinary free body + explicit inertial)
    └── worktable (six compliant relative joints)
```

compiled audit 确认 driver 没有 geom，dynamic deck 不是 mocap，worktable 已挂到 deck，
generated weld 的 body/order 与 Phase 02 config 一致。没有通过猜测 `table` / `cube`
名称重挂 body。

## 4. Isolator derivation

对六轴分别使用：

```text
m_eff = [M, M, M, Ixx, Iyy, Izz]
omega_n = 2*pi*f_n
k = m_eff * omega_n^2
c = 2*zeta*m_eff*omega_n
springref = [0, 0, M_ref*g/k_z, 0, 0, 0]
```

默认 probe profile 是六轴 `f_n=5 Hz`、`zeta=0.1`，仅用于可重复测试和 sweep 起点，
不是最终 operating point。对应 derived values 为：

```text
k = [31582.7341, 31582.7341, 31582.7341,
      956.9568,  1121.4831,  2059.4904]
c = [201.0619, 201.0619, 201.0619,
       6.0922,   7.1396,   13.1111]
```

`springref_z = 0.0099396081 m`，与未补偿的
`static_sag_uncompensated = g/(2*pi*f_n,z)^2` 相同。该 preload 只补偿 32 kg
worktable 自重；payload 不触发 `k/c` 重算。

## 5. Transfer and equilibrium probes

解析传递函数使用同一 COM/reference frame：

```text
H_abs = (1 + i*2*zeta*r) / (1 - r^2 + i*2*zeta*r)
H_rel = r^2 / (1 - r^2 + i*2*zeta*r)
```

默认 profile 在 `f=[1, 5, 10] Hz` 的 envelope 中：

```text
T_peak             = 5.0990195
R_relative @ 5 Hz = 5.0
D_relative        = 0.005 m  (base displacement amplitude = 0.001 m)
```

六个独立 4 Hz 小振幅 MuJoCo probes 使用 `dt=0.0002 s`、Phase 02 nominal weld
`eq_solref=[0.0004, 0.5]`。相对振幅与解析 `|H_rel|=1.6245539` 的结果如下：

| axis | MuJoCo measured | analytic | relative error |
| --- | ---: | ---: | ---: |
| tx | 1.6441388 | 1.6245539 | 1.206% |
| ty | 1.6441204 | 1.6245539 | 1.204% |
| tz | 1.6385420 | 1.6245539 | 0.861% |
| rx | 1.6452834 | 1.6245539 | 1.276% |
| ry | 1.6455833 | 1.6245539 | 1.294% |
| rz | 1.6415976 | 1.6245539 | 1.049% |

所有单轴 probe 的 MuJoCo warning counters 为零。测试将 analytic formula 作为
exact contract，并将当前 weld/discrete implementation 的 measured envelope 设为
5% provisional bound；Phase 06 仍需用冻结前的预注册 transfer envelope 重新选择和
确认 candidate。

空载 fixed-deck probe 运行 5 s 后：

```text
isolator_tz q = 2.99e-13 m
warnings     = zero
```

加入一个 world-child、显式质量 `0.349 kg` 的 payload 后，工作台产生额外：

```text
tz = -0.000108404 m
```

这与解析 `-m_payload*g/k_z` 一致；payload COM 偏移 `(0.10, 0.20, 0) m` 时还得到
非零 `rx/ry` gravity torque offset。该 probe 不创建 task，也不计算 task success。

## 6. Safety and visual invariance

- translation limits 默认是 `[0.025, 0.025, 0.025] m`，rotation limits 默认是
  `[5, 5, 5] deg`；每个轴独立检查。
- `NaN`、错误 shape、边界值和超限值都 fail closed；严格边界值不会被视为通过。
- Phase 02 的 environment-owned timestep / weld gate 保持有效：probe 使用
  `eq_solref[0] = 2*dt`，小于 `2*dt` 的配置拒绝由已有 Phase 02 contract 负责。
- visible / hidden arena 编译后的 mass、COM、inertia、joint type/axis、stiffness、
  damping、springref、limits 和 contact flags 逐项相同；只有 RGBA 不同。
- 同一 deck driver 在两种 visual setting 下的 actual pose/twist/acceleration、raw
  weld diagnostics 和 timestamps 逐值相同。

## 7. Verification

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
tests/test_shakebench_isolator.py tests/test_shakebench_arena.py --tb=short
18 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_config.py tests/test_shakebench_excitation.py \
  tests/test_shakebench_calibration.py --tb=short
34 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_deck_driver.py --tb=short
59 passed
```

Phase 03 完成后保留的边界：最终 `f_n/zeta/k/c`、driver/contact/timestep 的 official
freeze、task contacts、Can evaluator 和 task SR 均属于后续 phase；本报告不把
provisional probe profile 宣称为 official physics profile。

现有 environment smoke 的 `test_action_playback.py` 通过（1 passed）。仓库环境套件中
需要 off-screen EGL 的 `test_all_environments.py`、`test_camera_transforms.py` 和
`test_env_determinism.py` 在本运行容器于创建环境前失败，原因是已知的
`EGL_BAD_DISPLAY` / 缺少 `swrast_dri.so`，不是 arena 或 isolator assertion；Phase 03
新增测试和 Phase 02 artifact verifier 均不依赖 EGL。
