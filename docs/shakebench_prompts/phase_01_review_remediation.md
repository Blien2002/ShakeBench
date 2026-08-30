# Phase 01 修复提示词：封闭官方配置、激励权威性与安全语义

你在 `/home/miracle04/Desktop/ShakeBench` 工作。当前仓库已有 Phase 00–01 的实现；本阶段的任务是**修复审查发现的可复现性与物理语义问题**，不是继续推进 dynamic deck、isolator、任务或 controller。

直接修改现有 `robosuite/`、`tests/`、`docs/` 和既有 assets 目录中的文件；不得建立任何 `shakebench` 子目录，不得在运行时读取仓库外文件，也不得删除或回退其他人的修改。

## 必读与权威顺序

完整阅读：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_00_upstream_integration.md
docs/shakebench_prompts/phase_01_excitation_core.md
docs/robosuite_benchmark_design_tree.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_00_report.md
docs/phase_01_report.md
robosuite/utils/shakebench_config.py
robosuite/utils/shakebench_excitation.py
robosuite/utils/shakebench_calibration.py
robosuite/utils/shakebench_safety.py
tests/test_shakebench_config.py
tests/test_shakebench_excitation.py
tests/test_shakebench_calibration.py
```

设计冲突按以下顺序处理：最新设计树 > v0 spec > phase report > 旧 prompt。设计树当前的有效事实包括：**400 official + 10 dev states，Can yaw 恒为 0；物理 timestep、weld/contact `solref/solimp`、isolator 参数和 `Gamma_star` 仍未冻结。**

本阶段的 leading word 是 **封口（closure）**：任何进入 `scoreable=True`、被称为 “v0 reuse”、或被提交为 golden reference 的东西，都必须能由独立代码、不可变数据或严格 schema 验证。不能证明的内容保持 draft / `UNFROZEN`，不能以“测试通过”替代证据。

## 范围与非范围

本阶段允许修改：

```text
robosuite/utils/shakebench_config.py
robosuite/utils/shakebench_excitation.py
robosuite/utils/shakebench_calibration.py
robosuite/utils/shakebench_safety.py
robosuite/scripts/shakebench_cli.py
robosuite/scripts/shakebench_generate_excitation_golden.py
tests/test_shakebench_config.py
tests/test_shakebench_excitation.py
tests/test_shakebench_calibration.py
tests/golden_shakebench_excitation_*.json
.gitignore
docs/phase_00_report.md
docs/phase_01_report.md
docs/robosuite_benchmark_v0_spec.md
docs/phase_01_remediation_report.md
```

如需新增测试或纯工具模块，只能平铺到现有 `tests/` 或 `robosuite/utils/`。可以新增一个小型、纯 NumPy 的独立 reference evaluator，但它必须是与 production generator **不同的实现路径**，且只在测试中使用。

本阶段不得：创建 MuJoCo model/environment、实现 driver/isolator/task、修改公共 simulation loop、选择任何未冻结物理参数、运行官方评测、或借 task success 调参数。

## 1. 修复 official configuration：scoreable 必须真正不可钻空子

将 `ShakeBenchConfig` 从“JSON 容器”收紧为可靠的配置边界。

1. `official_fields` 和 `options` 在构造后必须深度不可变。推荐把嵌套 mapping 转为只读 mapping、sequence 转为 tuple；`to_dict()` 仍返回可安全修改的深拷贝。以下行为必须失败，且不会改变原 config hash：

   ```python
   config.official_fields["gamma_star"] = 1.0
   config.options["anything"] = 1
   config.official_fields["isolator_k"][0] = 1.0
   ```

2. `scoreable=True` 必须校验**每个** official 字段的结构、长度、单位域和数值范围，而不只是“不等于 `UNFROZEN`”。为每个字段建立显式 validator；不接受“所有字段填 0”这样的伪冻结配置。

3. 尚未由后续 physics/controller/protocol phase 冻结的字段必须使 scoreable config fail closed。Phase 01 不得伪造这些值来宣称可 official score。一个正确的 Phase 01 outcome 是：draft config 可以完整序列化和 hash；`scoreable=True` 因冻结证据尚不存在而被拒绝，并给出具体字段和原因。

4. 任何影响 official replay 的参数都不能藏在自由的 `options`。至少将 excitation profile identity/hash、authored-spectrum version、gravity、ramp duration、frequency scale、Gamma point、safety-profile identity/hash 和 official state-manifest hash 放入明确定义的、可验证的官方 provenance 字段；或在 `scoreable=True` 时禁止非空 `options`。选择一种方案并在 schema 与文档中明确。

5. `solref` / `solimp` 必须按 MuJoCo 所需的完整向量结构表示并验证长度/有限性/合法范围；不能以“任意 JSON array”冒充配置。对仍未冻结的具体数值保留 `UNFROZEN`，而不是降低 schema 约束。

6. 明确 Python 支持策略：要么把新代码改写为仓库声明支持的最低 Python 版本可运行的 typing 语法，要么提高 `setup.py`/格式工具/文档中的最低版本并给出理由。不能一边声明 Python `>=3`，一边使用仅 Python 3.10+ 可解析的 `X | Y`、`TypeAlias`、内建泛型 API。

为以上每条先写 red test；修复后测试必须证明嵌套 mutation、零值/空向量、错误 `solimp`、未知 provenance、以及 scoreable 的隐藏 options 都 fail closed。

## 2. 修复激励的权威来源与 golden 证据

设计树要求 v0 “原样复用当前 ShakeBench 数值”。现有 Phase 01 报告若使用了不同 ramp、episode duration、gravity、jitter/RNG、符号或默认时间窗，就不能继续称为 reuse。

先完成一次可审计的**权威选择**，只能二选一：

- **A. exact reuse：** 将由随仓 provenance 可获得的旧 ShakeBench 激励算法和全部冻结默认值逐项恢复；在当前仓库内提交一个独立 reference fixture（或只读 reference dataset）以及其来源、commit/hash、字段对照。生产 generator 必须逐项匹配该 reference，包括 per-axis RNG stream、frequency jitter、phase、符号、gravity、ramp、episode-relative time semantics 和 Gamma calibration grid。
- **B. new authored profile：** 若无法在当前仓库内证明 exact reuse，则把该 profile 明确改名为新的 authored v0 candidate；从设计树和 Phase 01 report 删除“原样复用”的陈述，并把其作为尚待 bench-author 批准的 draft。不得用新实现生成的结果反向证明它等于旧实现。

不要隐式选择。将选择、逐字段差异表和 provenance 写入 `docs/phase_01_remediation_report.md`。只有 A 才可保留“reuse”字样。

golden 的修复要求：

1. `tests/golden_shakebench_excitation_*.json` 是**只读验收输入**，不是 production generator 的自我快照。测试必须以独立 evaluator 解析 fixture 后重建并比对 line table、采样 `q/qdot/qdd`、Gamma、time-shift 和 safety cases。
2. fixture generator 可以保留作维护工具，但生成 fixture 的命令不得是日常测试证据。更新 fixture 时要求显式 `--update` / `--accept-reference-change`、人工记录原因、旧/新 SHA-256 和完整 invariant suite；默认 invocation 只能验证而不能覆盖 fixture。
3. fixture provenance 使用确定性信息（source profile identity、source commit/hash 或 authored-decision id、config hash、fixture SHA）；不要写随机器时间为主要可复现证据。
4. 最小覆盖：多 seed、nonzero `t0`、six-axis/single-axis、ramp 前中后、Gamma point 含 `alpha × r`、以及每一种 safety rejection。对于 exact reuse，这些值必须来自独立 reference；对于 new authored profile，它们必须来自预先写入 fixture 的固定数值，测试不允许调用 production generator 来产生 expected values。

## 3. 把 safety gate 变成六自由度物理 gate

保留 fail-closed 的原则，但修复“逐轴 max 代替整体运动”的错误。

1. 位移 gate 必须检查相对位移的三维向量范数，以及 rotation 造成的任务相关点（桌角、target wall/top edge、workpiece support point）的位移；不能只比较单一 translation axis 的最大值。
2. solver-travel gate 必须以明确的几何 clearance / contact feature 为基础，同时检查 translation、rotation-induced feature travel、速度/加速度和所需的 per-step displacement。给出每个限值的单位、来源与 conservative formula；8 mm target wall 不是自动等于 8 mm 可允许运动。
3. non-ballistic 判断必须针对 workpiece-support normal 的实际有效法向加速度，包含 authored translation 和 `alpha × r`；明确何时只做 command-level preflight、何时需要后续 actual-state runtime check。本阶段不假装拥有 dynamic deck 的 actual state。
4. 当物理参数尚未冻结时，safety profile 只能是 candidate。其 profile hash 与所有限值必须进入配置 provenance；未冻结 candidate 不得解除 scoreable gate。
5. 添加针对“每轴分别过关但合成向量超限”、rotation-only 超限、以及 solver travel 因 joint translation 合格却因桌角运动失败的 red tests。

## 4. 修复 V3 时间语义的早期 API 缺陷

当前 excitation serialization 不得把 `t0` 命名或输出为 `episode_time_s`。`t0` 是程序的共同时间偏移/phase 语义；`episode_time_s` 是运行时当前时刻，二者不同。

- 静态 `ExcitationProgram.to_dict()` 只输出静态程序参数（含每条 phase-at-episode-zero），不伪造当前 episode time；
- 如需 V3 payload，新增显式函数/对象，调用者必须传入当前 `episode_time_s`，并断言其有限且非负；
- 在没有 V3 provider 的本阶段，为该 serializer 写纯单元测试，不暴露 policy observation。

## 5. 修复可发布的 assets、报告和单一事实源

1. 将 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg` 纳入 Git 和 source distribution：为该特定文件在 `.gitignore` 写最小例外（或用等价受控方式），检查 `git ls-files` 能列出它，并加入 manifest/package-data smoke test。不要 force-add 一个仍被 `.gitignore` 隐藏、未来容易再次消失的文件。
2. 同步或降级 `docs/robosuite_benchmark_v0_spec.md`：不能再把其中的“40 official/random yaw”与设计树的“400 official/yaw=0”并列称为权威。优先将旧段落替换为设计树所指的有效值；若其余 spec 还具有历史价值，在文首醒目标注它是 superseded summary，并逐项链接设计树。
3. 更新 `docs/phase_00_report.md` 与 `docs/phase_01_report.md`，只陈述当前可验证事实：正确 remote/commit、正确测试命令和结果、是否 exact reuse、fixture 的独立性、哪些值仍 `UNFROZEN`。不把 Phase 01 称为 official-ready。
4. 若 `docs/reports/m05_driven_deck_spike_*.md` 依赖 `/tmp` 中未提交脚本/数据，把报告明确标为 exploratory evidence，或将最小可复现脚本、输入和结果以平铺方式提交到现有允许目录。不能将临时目录当作 Phase 02 的可审计依据。

## 验收与停止条件

按 red → green → refactor 执行，保留失败测试到它们被正确修复。至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_config.py \
  tests/test_shakebench_excitation.py \
  tests/test_shakebench_calibration.py --tb=short
python -m compileall -q robosuite/utils/shakebench_*.py \
  robosuite/scripts/shakebench_*.py
git ls-files --error-unmatch \
  robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg
```

并在报告中列出每项命令、输出摘要、起止 commit、完整修改文件清单、fixture SHA、official-config negative tests 和仍未冻结字段。

完成的定义是：

1. Phase 01 的草稿能力保持可用，但不存在可以伪装为 official/scoreable 的空配置、零配置、隐藏配置或可变配置；
2. “reuse”声明有独立证据，或被诚实替换为 new authored candidate；
3. golden 能捕捉 production generator 的语义漂移，而不仅是证明它今天能重算自己的输出；
4. safety 不再把六自由度合成问题缩减为单轴 max；
5. 资产、文档和测试证据在 clean clone/source distribution 中可获得；
6. 没有触及 Phase 02+ 的 runtime physics 实现。

完成后停止，并在最终回复中先报告任何 blocked evidence（尤其是 exact reuse 的来源缺失），再报告已通过的 gates。
