# Phase 06 Prompt：冻结 Official Physics Profile

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。只依据 physics probes 冻结 environment-owned timestep/solver、deck weld、isolator 和 contact profile；不得使用 V0–V3 task ranking 或修改 controller，不新建子文件夹。

## 前置与必读

- Phase 05 通过；Phase 02–05 probes 可重放。
- 读取 `docs/robosuite_benchmark_design_tree.md` physics gates、transfer-envelope 和 failure integrity。

## 预注册先行

先新增并提交 `robosuite/models/assets/shakebench_selection_protocol.yaml`（平铺进现有 assets 目录，该目录已被 MANIFEST.in 递归打包），包含 candidate grids、target transfer envelope、hard exclusions、scoring、deterministic tie-break、tolerances 和 crash handling。运行 selection 后不得改规则；无候选通过即 blocked。

## 1. timestep/solver/weld

至少三档 dt，满足 `dt<=1/(20*f_max)` 与 positive solref time constant `>=2dt`。扫描/验证 integrator、solver、iterations、deck inertial、`eq_solref/eq_solimp` 和 scheduler。

所有 safe Gamma candidates、六轴、empty/representative load 下硬门：line amplitude error<=1%、phase<=1°、Gamma deck error<=1%，无 warning/unstable residual。

## 2. isolator selection

对候选 six-axis `f_n/zeta` 计算 analytic/MuJoCo transfer、combined spectrum、`T_accel/R_relative/D_relative/T_peak/travel margin/static sag/preload offset/payload sensitivity`。按预注册规则选唯一 candidate 并冻结 `k/c`。不读取 task SR。

## 3. contact selection

保持 sliding μ=0.30/1.00，只通过 static support、incline、single-axis slip、impact/recovery、finger load 和 timestep convergence 冻结 `condim`、torsional/rolling、margin/gap、`solref/solimp` 和 iterations。使用显式 pair 作用域。

## 4. bounded parity与确定性

Gamma=0 不要求 dynamic table 与 stock static table 逐轨迹相等；比较 task geometry、action semantics、passive support 和 solvability prerequisites，单独报告 dynamic residual。

每个关键 probe 相同 state/config 至少 3 个独立进程，比较完整 trace。

## 直接修改与产出

- 唯一 immutable official physics profile：`robosuite/models/assets/shakebench_official_physics.yaml` 及 hash（平铺现有 assets 目录）；
- environment 构造只从该 profile 读取 scoreable physics；
- 保留 training/probe profile 但明确 non-scoreable；
- 新增 `robosuite/scripts/shakebench_select_physics.py`、`robosuite/scripts/shakebench_replay_physics.py` 和 `tests/test_shakebench_physics_profile.py`；
- 输出 raw candidate artifacts、selected/excluded table、`docs/phase_06_report.md`。

## 完成条件

1. 一个且仅一个 official profile 通过全部 hard gates；
2. wheel 安装后 profile 可发现且 hash 一致；
3. driver/isolator/contact/timestep 均有解析或收敛证据；
4. selection 未读取 task SR；
5. 3-process determinism 通过；
6. 所有现有 robosuite physics/env tests 无回归；
7. 任何门失败则 blocked，不进入 controller；
8. 未新建目录；profile/protocol 文件全部平铺在现有 `robosuite/models/assets/` 根。

完成后停止。
