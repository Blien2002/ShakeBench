# Phase 03 Prompt：ShakeBenchArena、工业工作台与 6-DoF Isolator

你在 `/home/miracle04/Desktop/ShakeBench`（原 robosuite 仓库改名而来）中工作。直接在现有 robosuite arena/model 目录内新增文件并接入 Phase 02 deck processor；只做 physics probes，不运行 task success，不新建子文件夹。

## 前置与必读

- Phase 02 driver gate 通过。
- 读取 `docs/integration_map.md`、`docs/robosuite_benchmark_design_tree.md` 第 2 节和视觉合同。
- 参考随仓证据：`docs/robosuite_benchmark_design_tree.md` 第 2 节与视觉合同、`docs/spike_implementation_formal_review.md`；不得引用任何外部目录。

## 修改落点（不新建目录）

```text
robosuite/models/arenas/shakebench_arena.py             # 新文件，导出于 arenas/__init__.py
robosuite/models/assets/arenas/shakebench_arena.xml     # 新 MJCF（现有 assets 目录内）
robosuite/models/assets/textures/                       # 现有目录内新增 texture 文件
robosuite/utils/shakebench_isolator.py
tests/test_shakebench_arena.py
tests/test_shakebench_isolator.py
```

更新 `robosuite/models/arenas/__init__.py`，不改变现有 `TableArena`。

## Canonical worktable

```text
dimensions = [0.65,0.60,0.06] m
mass       = 32 kg
inertia    = [0.9696,1.1363,2.0867] kg m^2
elastic center = body origin = COM = principal frame
```

- 显式 inertial，不从 visual/collision density 推导；
- industrial frame、横撑、脚板、螺栓、isolation mounts 仅视觉，non-contact 且不贡献 mass；
- phenolic texture：若 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg` 已随仓则直接使用（SHA-256 见 `docs/IMPORT_PROVENANCE.md`）；否则先用确定性深灰中性材质占位并在报告中标注视觉合同待补。纹理仅视觉，不参与物理。

## 6-DoF isolator

- 3 slide + 3 hinge，axes 穿过 COM/elastic center；
- 每轴独立 `f_n/zeta` config；
- `k/c` 由 32 kg 和显式惯量派生；
- z `springref=M_ref*g/k_z` 只补偿 table 自重；
- Can/payload 不触发重算 `k/c`；
- travel/angle limits fail closed。

Phase 03 参数化 sweep，不冻结最终 operating point。

## Physics probes

- six single-axis harmonic transfer；
- tracking/resonance/isolation regions；
- empty-table nominal equilibrium；
- 0.349 kg payload offset；
- payload mass/COM sensitivity；
- six-axis spectrum；
- timestep and safety rejection；
- `T_accel/R_relative/D_relative/T_peak/travel_margin/static sag`。

解析量与 MuJoCo 使用同一 reference point/frame。

## 完成条件

1. compiled mass/COM/inertia/joints/preload 逐项断言；
2. six-axis transfer 与解析解在预注册容差内；
3. empty table 回到 nominal pose，payload 产生非零额外 offset；
4. visual layer 开关不改变 compiled physics 或 trace；
5. 现有 arenas/environments tests 无回归；
6. 输出 `docs/phase_03_report.md`，不选择最终 `f_n/zeta`，不运行 task SR，未新建目录。

完成后停止。
