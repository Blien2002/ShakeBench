# Phase 07R5 Final Commit Handoff

你在 `/home/miracle04/Desktop/ShakeBench` 中完成 Phase 07R5 的 Git 冻结和 Phase 8 交接。
当前 controller、motion semantics、V0/Gamma=0 10/10、matched、Gamma=0.15/0.30、ablations、
semantic verifier、三进程 determinism、actuator 14/14 和 clean wheel/sdist evidence 已全部
通过。当前唯一 blocker 是 `docs/phase_07_r5_manifest.json` 中：

```text
status             = BLOCKED_BY_PHASE_07R5_FINAL_COMMIT_HANDOFF
final_commit       = null
phase08_authorized = false
```

本任务只做 repository handoff。不得修改 controller、provider、environment、physics、tests 的
语义或任何 profile 参数；不得重跑 knee/official states；不得创建 Phase 07R6。

## 1. 重新确认当前证据

读取：

```text
docs/phase_07_report.md 顶部 R5 current section
docs/phase_07_r5_manifest.json
docs/phase_07_r4_manifest.json
docs/shakebench_prompts/README.md 的门控治理
```

运行并记录：

```bash
python -m robosuite.scripts.shakebench_handoff_semantic_remediation --verify
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_oracle.py \
  tests/test_shakebench_phase07r5_motion_semantics.py \
  tests/test_shakebench_phase07_completion_red.py \
  tests/test_shakebench_actuators.py \
  tests/test_shakebench_metrics.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py \
  tests/test_environments/test_vibration_pick_place_can.py -k 'not runtime_png'
python -m isort --check-only <全部 Phase 07 修改的 Python 文件>
python -m black --check -W 1 <全部 Phase 07 修改的 Python 文件>
git diff --check
```

使用 manifest 中已有 reproduction commands 重新打开并验证最终 semantic、determinism、
actuator 和 package artifacts。逐个重算 manifest 所列 `path` 的 file SHA-256；不得仅信任 stored
PASS Boolean。

若任何 verifier/hash/test 失败，保持原 BLOCKED 状态并报告该具体 blocker。若 production
代码或 profile 必须修改，停止本任务：相关 rollout/package/determinism evidence 必须回到 R5
重新生成，不能在本 Git-only handoff 中修代码。

完成标准：所有已有 R5 evidence 仍通过，且从 evidence生成后到当前没有 production byte
变化。

## 2. 整理提交边界

检查 `git status --short` 和完整 diff，把文件分为：

```text
提交：Phase 05 contract synchronization、Phase 07 controller/provider/environment、
      tests、scripts、design/spec、R2–R5 prompts、R4/R5 compact manifests、Phase 07 report、
      .gitignore 中的 raw-output rules

排除：out/ raw traces、build/、*.egg-info、__pycache__、pytest caches、MUJOCO_LOG.TXT、
      本地 .codex/config.toml
```

`.codex/config.toml` 是本地 Codex sandbox/network 配置，不属于 benchmark。将 `.codex/` 加入
`.gitignore`，除非仓库已有明确决定要共享该配置。保留文件本身，不删除。

不要使用未经检查的 `git add -A`。先输出准备提交的精确路径列表，确认每个文件都属于
Phase 05–07 合同同步或 R5 实现；再显式 stage。不要提交 ignored `out/` 中约数百 MB 的 raw
traces，Git 只保留 compact manifests 和外部 raw path/size/hash。

完成标准：staged diff 包含完整可安装实现、tests、docs 和 compact manifests；不包含本地配置、
build/cache 或 full raw evidence。

## 3. 创建 implementation/evidence commit

提交第一个 commit，例如：

```bash
git commit -m "Complete Phase 07 oracle motion semantics"
```

提交后记录 `git rev-parse HEAD`，将该 SHA 记为 `IMPLEMENTATION_COMMIT`。这个 commit 冻结
所有影响 action、observation、profile、artifact verification 和 package behavior 的 bytes。

不要尝试在 `IMPLEMENTATION_COMMIT` 自身的 manifest 中预先写入它自己的 SHA；Git commit
hash 依赖文件内容，这会形成不可能的自引用。

完成标准：`IMPLEMENTATION_COMMIT` 存在，且 Phase 07 production/tests/manifests 的实现内容均为
其 tree 内容。

## 4. 更新 handoff metadata

第一个 commit 完成后，只修改：

```text
docs/phase_07_r5_manifest.json
docs/phase_07_report.md
必要时 docs/shakebench_prompts/README.md 的当前状态文字
```

在 manifest 中设置：

```text
status             = PASS
final_commit       = IMPLEMENTATION_COMMIT
phase08_authorized = true
final_commit_role  = implementation_and_evidence_commit
handoff_rule       = final_commit must be an ancestor of the commit containing this manifest
```

`final_commit` 指向第一个实现/证据 commit，不指向包含更新后 manifest 的第二个 commit。报告顶部
改为 Phase 07R5 PASS，记录 implementation commit、profile/task-context/physics/dev-state hashes、
R5 manifest path 和 ancestor rule。报告把“包含该 manifest/report 的 Git commit”定义为
handoff commit，但不在文件中预写该 commit 的 literal SHA，避免第二次自引用。历史 R2–R4 只
保留 provenance，不再混排为当前状态。

运行 `git diff --name-only IMPLEMENTATION_COMMIT`；输出只能是 handoff metadata 文件。出现
production/test/profile 变化时停止并解释 evidence 为什么可能失效。

完成标准：manifest/report 对 Phase 07 当前状态只有一个 PASS 结论，并绑定
`IMPLEMENTATION_COMMIT`。

## 5. 创建 metadata handoff commit

显式 stage 第 4 节的 metadata 文件并提交，例如：

```bash
git commit -m "Authorize Phase 07 to Phase 08 handoff"
```

把当前 HEAD 记为 `HANDOFF_COMMIT`，运行：

```bash
git merge-base --is-ancestor IMPLEMENTATION_COMMIT HANDOFF_COMMIT
git show IMPLEMENTATION_COMMIT:robosuite/utils/shakebench_oracle.py >/dev/null
git show HANDOFF_COMMIT:docs/phase_07_r5_manifest.json >/dev/null
git status --short
```

再从 `HANDOFF_COMMIT` tree 读取 manifest，验证：

```text
status=PASS
final_commit=IMPLEMENTATION_COMMIT
phase08_authorized=true
IMPLEMENTATION_COMMIT is ancestor of HANDOFF_COMMIT
controller profile hash=60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c
physics hash=c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c
dev-state anchor=dd6fe2edb6384ccdb5116be44f07592b4864e377
```

`git status --short` 必须为空；ignored `out/` 可保留在磁盘。不要删除 raw evidence。

## 6. 完成条件

只有以下全部成立才允许进入 Phase 8：

1. R5 controller/evidence gate 仍全部通过；
2. implementation commit 与 metadata handoff commit 均存在；
3. manifest 不自引用，采用可验证 ancestor binding；
4. manifest status 为 PASS、`phase08_authorized=true`；
5. 第二个 commit 相对第一个 commit 只改变 handoff metadata；
6. 工作树干净，full raw evidence 未进入 Git 且未被删除；
7. Phase 07 report 顶部只呈现当前 PASS、implementation commit 和 handoff ancestor rule；
8. 未修改 physics/contact/task success/dev states，未访问 knee/official states。

完成后停止并报告两个 commit SHA、manifest SHA-256、profile hash、验证命令和 `git status`。
不要开始 Phase 8 implementation。
