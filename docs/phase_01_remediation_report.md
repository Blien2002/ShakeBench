# Phase 01R remediation report：封口官方配置、激励权威性与安全语义

状态：Phase 01 review remediation 完成候选实现；当前阶段不宣称 official-ready，也没有推进 Phase 02+ runtime physics。

## 1. 起止与证据边界

- implementation repo：`/home/miracle04/Desktop/ShakeBench`。
- 起始 commit：`5e5829ee80c544125c47eba5bc760ce90ebf70ad`。
- 目标 remote：`https://github.com/Blien2002/ShakeBench.git`。
- implementation checkpoint commit：`b4ee4808` (`fix(shakebench): close phase 01 review findings`)；本报告随该 commit 提交。
- 没有读取、运行或引用仓库外的旧 ShakeBench 备份目录；所有 runtime 和测试输入来自当前仓库。

## 2. Blocked evidence：exact reuse

**Blocked / unavailable：旧 ShakeBench exact reuse 无法在当前仓库内证明。** 当前 `robosuite` fork 中没有旧 ShakeBench 激励算法、可核对的 source commit 或独立 reference dataset；设计文档和已有 Phase 01 实现本身不能作为证明。因此本阶段明确选择 **B：new authored v0 candidate**：profile id 为 `shakebench.authored_v0_candidate`，spectrum version 为 `candidate-2026-08-30`，decision id 为 `phase-01-remediation-B-20260830`。

不再使用“原样复用当前 ShakeBench”或 exact-reuse 表述；candidate 数值仍等待 bench-author 批准，不能解除 scoreable gate。六轴中心频率、relative RMS、bandwidth、tones、`kappa_rot=0.30`、`reference_lever_m=0.65 m`、jitter、ramp、gravity、Gamma point 和 RNG/phase 语义均只有当前 candidate 实现证据；旧实现的逐项匹配为 blocked，不以 production generator 输出反证等价。

## 3. 配置封口

`ShakeBenchConfig` 已升级为 schema version `2`：

- `official_fields`、`provenance` 和 `options` 在构造后递归冻结为只读 mapping/tuple；`to_dict()` 返回独立可修改副本，hash 不受外部 mutation 影响；
- official 字段逐项检查标量、向量长度、有限性、正值范围、MuJoCo `solref` 两值格式、`solimp=[dmin,dmax,width,midpoint,power]` 五值格式、`condim` 域和 10 dev + 400 official state-ID 结构；
- provenance 明确定义 profile identity/hash、spectrum version、gravity、ramp、frequency scale、Gamma point、safety profile identity/hash、完整 safety limits、state-manifest hash 和 freeze status；
- scoreable config 禁止非空 `options`；Phase 01 `SCOREABLE_FREEZE_AUTHORITY=None`，因此即使结构上填入一组候选数值，`scoreable=True` 仍 fail closed；
- 默认 draft 保留所有未冻结 official/provenance 值为 `UNFROZEN`。

负向测试覆盖深层 mapping/sequence mutation、零值/空向量、错误 `solimp`、未知 provenance、伪造全零 official map、隐藏 options 和无 freeze authority。

## 4. Golden 与独立验收

fixture 维护工具的默认命令只验证，不覆盖：`python -m robosuite.scripts.shakebench_generate_excitation_golden`。只有显式 `--update` 或 `--accept-reference-change` 并提供 `--reason` 才能替换 fixture；更新记录保留 reason、previous fixture hash 和 new fixture hash。

当前 candidate fixture provenance：

- source profile：`shakebench.authored_v0_candidate`；decision：`phase-01-remediation-B-20260830`；
- config-envelope hash：`91c0f38575eccb429d18948c0d8f47050407e837ccd2b65c803cf7560eef7661`；
- canonical fixture payload hash（排除 self-hash 和 update history）：`3111107a8db4e91ba38b684a354c2826b3dc197b901df9c35718c183add753d5`；
- safety profile hash：`e04395ab50798aef2c0ad7760a739152bf623caf4dc3ce96b6cc5b2417c5b5a4`。

`tests/test_shakebench_excitation.py` 的 golden 验收使用独立 NumPy evaluator：从 fixture 固定 line table 解析 `q/qdot/qdd`、quintic ramp、Gamma point normal acceleration 和 time-shift；production generator 只作为当前实现与已提交 line table 的 drift 对照，不生成 expected motion。rejection cases 也保存完整 program，测试独立重算 displacement、frequency、Gamma 和 solver-step bounds。

## 5. 六自由度 safety

新增 candidate `SafetyGeometry`，包含 table corners、target-wall top edge 和 workpiece support point。displacement gate 同时检查 `||translation||₂` 与 `||translation + rotation × feature_offset||₂` 的保守界。

solver-travel gate 明确使用 geometry clearance（默认 8 mm 是 candidate clearance，不是自动允许的运动量）和 `solver_step_fraction=0.25`；对 translation 与每个 feature 分别计算速度、加速度和 `required_step = speed_bound × dt + 0.5 × acceleration_bound × dt²`，再与 `allowed_step = clearance × solver_step_fraction` 比较。

non-ballistic gate 是 command-level preflight，按 workpiece-support normal 检查 translation、`alpha × r` 和 centripetal term；realized deck/table runtime gate 留给 Phase 02+。red tests 覆盖合成 translation 范数超限、rotation-only feature 超限，以及 translation step 通过但 table-corner feature step 失败。

## 6. 时间语义与发布资产

- `ExcitationProgram.to_dict()` 只输出静态程序参数和 `line_phase_at_episode_zero`，不输出 `episode_time_s`；`to_runtime_payload(episode_time_s)` 要求有限、非负的运行时当前时刻；
- `shakebench_phenolic_bench_dark_1k.jpg` 已用 `.gitignore` 最小例外纳入 Git，SHA-256 为 `6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb`，`MANIFEST.in` 递归规则覆盖它；
- `docs/robosuite_benchmark_v0_spec.md` 已同步为 10 dev + 400 official、Can yaw=0，并标明 candidate/exact-reuse 状态；
- `docs/reports/m05_driven_deck_spike_20260830.md` 已标记为依赖 `/tmp` 的 exploratory evidence，不作为 Phase 02 审计依据。

## 7. 修改文件清单

本轮 remediation 修改或新增：`.gitignore`、`docs/IMPORT_PROVENANCE.md`、`docs/integration_map.md`、`docs/phase_00_report.md`、`docs/phase_01_report.md`、`docs/phase_01_remediation_report.md`、`docs/reports/m05_driven_deck_spike_20260830.md`、`docs/robosuite_benchmark_design_tree.md`、`docs/robosuite_benchmark_v0_spec.md`、`docs/shakebench_prompts/README.md`（保留用户新增 01R 索引）、`pyproject.toml`、`robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg`、`robosuite/scripts/shakebench_generate_excitation_golden.py`、`robosuite/utils/shakebench_calibration.py`、`robosuite/utils/shakebench_config.py`、`robosuite/utils/shakebench_excitation.py`、`robosuite/utils/shakebench_safety.py`、`setup.py`、两个 golden JSON 和三个 Phase 01 测试文件。

`docs/shakebench_prompts/phase_01_review_remediation.md` 是本阶段输入 prompt，保留在工作树中；未创建任何禁止的 `shakebench/` 子目录。

## 8. 验证记录

必须验收命令：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_shakebench_config.py tests/test_shakebench_excitation.py tests/test_shakebench_calibration.py --tb=short
34 passed（修复后；最终验收将再次运行）

python -m compileall -q robosuite/utils/shakebench_*.py robosuite/scripts/shakebench_*.py

git ls-files --error-unmatch robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg
```

已有上游 non-EGL regression：`255 passed, 58 skipped`；EGL/headless renderer 阻断仍记录在 `docs/phase_00_report.md`，不是本阶段新增代码引起。

source distribution smoke：`python setup.py sdist --dist-dir /tmp/shakebench_sdist_phase01r` 生成的 `robosuite-1.5.2.tar.gz` 中包含 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg`；`robosuite.egg-info/SOURCES.txt` 同样列出该文件。

仍未冻结的 official fields：`gamma_star`、`physics_timestep_s`、deck `eq_solref/eq_solimp`、deck mass/inertia、isolator `fn/zeta/k/c`、contact `condim/solref/solimp`、gripper force、OSC profile、official state-manifest hash，以及 candidate excitation/safety profile 的 freeze evidence。

Phase 01R 完成后停止；不实现 dynamic deck、isolator、task、controller 或 official evaluation。
