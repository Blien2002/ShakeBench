# Phase 07R6 Publication Integrity Closure

你需要在 ShakeBench 仓库中执行 Phase 07R6：Publication Integrity Closure。

目标不是开始 Phase 8，而是修复 Phase 07R5 的 archive、Git anchor、package verifier 和最终 handoff 闭环。只有全部 R6 门控通过后，才能重新设置 `phase08_authorized=true`。

重要：不要在旧的 dirty recovery workspace 上实施。必须从远端 master 的
`5fe46051d1c1cbe5a748c9330b5e338fd563df15` 创建新的干净 clone。旧 workspace、repo 外 backup bundle 和原始 raw evidence 均不得删除。

开始前读取：

- `docs/shakebench_prompts/phase_07r5_repository_slimming_and_handoff.md`
- `docs/phase_07_report.md` 顶部 R5 section
- `docs/phase_07_r5_manifest.json`
- `docs/shakebench_history_rewrite_map_v1.json`
- `docs/shakebench_evidence_index_v1.json`
- `robosuite/scripts/shakebench_audit_evidence.py`
- `robosuite/scripts/shakebench_verify_evidence_archive.py`
- `robosuite/scripts/shakebench_verify_package.py`
- `robosuite/utils/shakebench_runtime_verifier.py`
- `robosuite/scripts/shakebench_run_oracle.py`

新建并维护：

- `docs/shakebench_prompts/phase_07r6_publication_integrity_closure.md`
- `docs/phase_07_r6_manifest.json`
- `docs/phase_07_report.md` 顶部 R6 current section

## 1. 权威边界与禁止事项

禁止：

- 不修改 controller 数值、状态机、观测 tier、物理参数、成功条件或 dev-state rows；
- 不重跑 knee/official states；
- 不重新选择 physics/controller profile；
- 不修改已有 raw trace、payload、action、state 或 metric 数值；
- 不覆盖或删除旧 release；
- 不再次重写 Git 历史，除非证明 removed blobs 仍可从 master 到达；
- 不在验证完成前设置 `phase08_authorized=true`；
- 不使用 `git add -A`；
- 不推送 `refs/codex/*`；
- 不删除 repo 外备份或旧 workspace。

冻结的科学身份必须保持：

```text
controller profile:
  shakebench.reference_oracle.v3
  60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c

official physics:
  shakebench.official.physics.v2
  c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c

dev-state asset:
  07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6

pre-rewrite anchor:
  dd6fe2edb6384ccdb5116be44f07592b4864e377

rewritten anchor:
  08626ea5a5e107df503e266be9065b929d47f882

R5 rewritten implementation:
  64039dbc77ac5becc67f8b9f33860d7bf12372ac

R5 handoff:
  5fe46051d1c1cbe5a748c9330b5e338fd563df15
```

## 2. 恢复真实状态

在任何修改前执行并记录：

```bash
git status --short
git rev-parse HEAD
git remote -v
git ls-remote origin refs/heads/master
git count-objects -vH
git fsck --full
```

确认：

1. 工作树干净；
2. HEAD 和远端 master 均为 `5fe46051…`；
3. `64039db…` 是 `5fe4605…` 的祖先；
4. removed-path list 中的路径在 master 全历史不可达；
5. rewritten anchor `08626ea5…` 存在；
6. old anchor `dd6fe2ed…` 在 rewritten history 中不可达，这是预期状态，不得继续把它当成当前 ancestry anchor。

如果远端 master 已变化，停止并报告，不得 force push。

## 3. 修复 Git anchor 迁移模型

当前错误：

- R5 manifest 仍把 `dd6fe2ed…` 当作当前 dev-state anchor；
- `shakebench_run_oracle.py` 也只保存旧 SHA；
- runtime verifier 只检查 rewrite-map hash 是否为 64 字符；
- 没有 verifier 检查 rewritten anchor、tree hash 和 handoff ancestry。

必须改为显式双 anchor 模型：

```yaml
dev_state_anchor:
  pre_history_rewrite_commit: dd6fe2edb6384ccdb5116be44f07592b4864e377
  rewritten_commit: 08626ea5a5e107df503e266be9065b929d47f882
  asset_path: robosuite/models/assets/shakebench_states_dev.json
  asset_sha256: 07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6
```

要求：

1. 历史 raw artifacts 中保存的旧 SHA 不得被静默改写；
2. 验证历史 artifact 时，通过 rewrite map 把旧 SHA 解析到 rewritten SHA；
3. 新生成的 provenance 使用 rewritten SHA，同时保留 original SHA；
4. verifier 必须运行 `git cat-file` / `git show`，确认 rewritten commit 存在；
5. 验证 rewritten commit 中 dev-state asset 的 SHA 为 `07de20b4…`；
6. 验证当前 checkout 中同一 asset 内容相同；
7. old→new mapping 不一致、commit 不存在、path 不存在、内容 hash 改变时 fail closed。

增加独立 repo handoff verifier，例如：

```bash
python -m robosuite.scripts.shakebench_verify_phase07_handoff \
  --manifest docs/phase_07_r6_manifest.json \
  --rewrite-map docs/shakebench_history_rewrite_map_v2.json
```

它必须检查：

- manifest schema/status；
- `phase08_authorized`；
- `final_commit` 存在；
- `final_commit` 是 handoff commit 的祖先；
- rewritten dev-state anchor 存在；
- old→new mapping 完整；
- relevant tree/payload hashes 不变；
- implementation commit 与 metadata handoff commit 的差异仅限允许的 metadata 文件；
- release manifest、archive index、runtime contract 和 rewrite map 相互绑定。

不要仅检查字符串格式。

## 4. 解决 archive hash 的循环依赖

不得再试图把 archive 自身 SHA 写进 archive 内部 index 后重新打包，这会产生循环依赖。采用三层结构。

### A. Archive 内嵌 content index

- 记录每个 archive member 的 path、size、SHA-256、phase/status/binding；
- 有 `index_sha256` 自校验；
- 不声称包含最终 outer archive SHA；
- 仓库中的 content index 必须与 archive 内嵌 index字节完全一致。

### B. Compact history rewrite map v2

新增 `docs/shakebench_history_rewrite_map_v2.json`，包含：

- original/rewritten commits；
- rewritten anchor；
- implementation commits；
- path-list hash；
- filter-repo version/command；
- science hashes；
- mapping payload hash。

它不包含 outer archive SHA，以避免循环。将完全相同的 map v2 放入：

- repository docs；
- runtime package compact assets；
- evidence archive `provenance/`。

Runtime contract 必须打开并验证实际 map 文件内容，不能只检查 map hash 长度。

### C. Detached release manifest

新增 `docs/shakebench_evidence_release_manifest_v1.json`，在 archive 构建完成后生成，记录：

- schema/version；
- release tag 和 URL；
- archive filename、byte size、SHA-256；
- embedded content-index SHA-256；
- rewrite-map v2 SHA-256；
- removed-path-list SHA-256；
- R6 implementation commit；
- expected handoff relationship；
- package evidence SHA-256。

该 detached manifest 不放入 archive；将它作为独立 release asset 和仓库 metadata 发布。同时生成 `SHA256SUMS`，至少包含 archive、release manifest、rewrite map 和 compact index。

## 5. 重建精简 evidence archive

不要复用当前 1.61 GB archive 作为最终 R6 archive。旧 release 保留为历史，不覆盖、不删除。新 tag 使用新的不可变名称，例如：

```text
shakebench-evidence-v0-2026-09-r6
```

Archive 只包含：

- 从 Git 外置的 authoritative/diagnostic raw scientific evidence；
- 必要的 R2–R5 provenance JSON；
- compact R4/R5/R6 manifests；
- embedded content index；
- removed-path list；
- history rewrite map v2；
- standalone stdlib verifier；
- README/recovery instructions。

明确排除：

- `package_install*/`、`build/`、`dist/`；
- wheel/sdist 文件本体、`*.whl` 和 package `*.tar.gz`；
- `__pycache__/`、`*.pyc`；
- virtualenv/site-packages；
- Git bundle；
- 重复 source checkout；
- 可重新生成的安装目录；
- 相同 raw evidence 的 wheel-install/sdist-install 副本。

只保留 package evidence JSON 和 wheel/sdist hashes。

新增 archive preflight：

- 拒绝绝对路径、`..`、symlink、hardlink、device member；
- 拒绝未被 index 记录的普通文件；
- 拒绝重复 archive member；
- 对相同 SHA 的重复大文件给出清单，未经显式理由不得重复；
- 统计 member count、压缩大小、解压大小和分类大小；
- 检测 package install、cache、wheel/sdist，发现即失败；
- 解压前检查磁盘空间并给出明确 required/available bytes。

目标不是追求任意固定大小，而是确认 package/install duplication 已完全消失，并报告 before/after。

## 6. 修复 full audit 和 standalone verifier

修改 `shakebench_audit_evidence.py`：

1. `--expected-archive-sha256` 或 `--release-manifest` 二者至少提供一个；
2. 对 archive 文件实际字节计算 SHA-256；
3. 必须直接比较 actual archive SHA 和 expected SHA；
4. 不得因为内部 index 的 `archive_sha256=null` 而绕过；
5. release manifest、actual archive、embedded index、rewrite map 必须形成完整绑定；
6. embedded index 与仓库 compact index必须字节一致；
7. archive 额外成员、缺失成员、member mutation、size/hash mismatch 均失败；
8. full scientific audit 必须重新执行原有 Phase 6/7 semantic/numeric verifier；
9. 无论成功失败均清理临时目录；
10. 磁盘不足时返回结构化错误，不留下数 GB 临时文件。

修改 standalone verifier，使其支持：

```bash
python shakebench_verify_evidence_archive.py \
  ARCHIVE \
  --release-manifest RELEASE_MANIFEST \
  --expected-sha256 EXPECTED_SHA
```

它必须认证 outer archive，而不只是内部 index。

新增负向测试：

- archive 外层字节变化；
- 合法重新打包但 outer hash 不同；
- release manifest archive hash 被篡改；
- embedded index 被替换并重新计算 self-hash；
- rewrite map 被替换；
- member 缺失/增加/换绑；
- `archive_sha256=null` 时仍必须依赖 detached authority；
- path traversal、symlink、hardlink；
- 磁盘空间不足；
- 缺少 detached release authority。

所有负向测试必须 fail closed。

## 7. 修复 package verifier

修复 `robosuite/scripts/shakebench_verify_package.py`。当前错误：

- 外层 `passed` 没有检查 `parsed["passed"]`；
- cwd 硬编码为 `out/phase07r2`；
- clean install 可能从 source checkout 泄漏 import。

要求：

1. `_run_installed_check` 的 passed 必须同时满足：
   - subprocess return code 为 0；
   - parsed 是 dict；
   - `parsed["passed"] is True`；
   - `robosuite.__file__` 位于 install root 内；
   - 被验证模块均来自 install root；
   - source checkout 不在有效 import path 中。
2. 在新建的临时空目录中运行 installed check，不依赖历史 `out/`；
3. fixture 使用显式绝对路径；
4. wheel 和 sdist 分别安装到隔离目录；
5. 验证 official profile load；
6. 创建 `VibrationPickPlaceCan` 无 renderer 环境并 reset；
7. 验证 runtime loader 不访问 raw archive 或网络；
8. 验证 wheel/sdist 不包含 removed-path list 中的路径；
9. 验证 wheel/sdist 不包含 package install、cache、raw evidence；
10. 无论成功失败都清理 install root，除非显式 `--keep-temp`。

必须新增测试：

- 子进程输出 `{"passed": false}` 时外层失败；
- 子进程非零退出时失败；
- JSON malformed 时失败；
- import 来自 source checkout 时失败；
- 历史 `out/` 不存在时仍能运行；
- 缺失 compact runtime asset 时失败；
- runtime map/release binding mutation 时失败；
- clean wheel 和 clean sdist 正向测试通过。

只保留一份最终 package evidence：

```text
out/phase07r6/package_evidence_final.json
```

Manifest、report、release manifest 和 index 必须全部指向同一份 package evidence 及同一组 wheel/sdist hashes。

## 8. 处理遗留 verifier

`shakebench_handoff_semantic_remediation --verify` 当前在 slim fresh clone 中会因缺失 raw evidence和旧 commit失败。必须二选一：

A. 将其明确改成 full-audit-only CLI，强制要求 `--evidence-root` 或 `--evidence-archive`；或

B. 标记为 historical/deprecated，并将所有 runtime/official入口迁移到新 runtime verifier。

无论选择哪种方案：

- official profile loader 不得调用 raw-evidence verifier；
- CLI 不得在缺少 archive 时输出大量误导性 hash mismatch；
- 应返回清晰的 `explicit external evidence required`；
- 增加调用链测试，证明 runtime load 不访问旧 raw paths。

## 9. 重新建立唯一 authority

在修复阶段先将 R6 manifest 设置为：

```text
status: BLOCKED_BY_PHASE_07R6_PUBLICATION_INTEGRITY
phase08_authorized: false
final_commit: null
```

完成代码、tests、map v2、archive builder 和 verifier 后创建 `R6_IMPLEMENTATION_COMMIT`，然后：

1. 构建一次最终 wheel/sdist；
2. 生成唯一 package evidence；
3. 构建精简 archive；
4. 生成 embedded content index；
5. 计算 archive actual SHA；
6. 生成 detached release manifest 和 `SHA256SUMS`；
7. 本地 standalone verification PASS；
8. 本地 full scientific audit PASS；
9. 创建 metadata handoff commit；
10. manifest 设置：

```text
status: PASS
phase08_authorized: true
final_commit: R6_IMPLEMENTATION_COMMIT
final_commit_role: publication_integrity_implementation_commit
handoff_rule: final_commit must be an ancestor of the commit containing this manifest
```

不得在 handoff commit 自身文件中写 handoff commit SHA。

## 10. 发布和远端验证

不要覆盖旧 release。创建新 tag：

```text
shakebench-evidence-v0-2026-09-r6
```

Release 必须至少上传：

- 精简 evidence archive；
- `shakebench_evidence_release_manifest_v1.json`；
- `shakebench_history_rewrite_map_v2.json`；
- embedded content index 的独立副本；
- `SHA256SUMS`。

发布顺序：

1. 记录远端 master 精确 lease；
2. 推送新 tag，不移动旧 tag；
3. 创建 release并上传全部 assets；
4. 从公开 release URL 下载所有 assets到新临时目录；
5. 校验 `SHA256SUMS`；
6. 用 detached manifest验证 actual archive bytes；
7. standalone archive verification PASS；
8. full scientific audit PASS；
9. 确认 release tag不指向旧 pre-rewrite历史；
10. 远端 master 未变化后，用精确 `--force-with-lease` 或普通 fast-forward push更新 master；
11. 不使用 `--mirror`；
12. 不删除无关 branch/tag。

若上传、下载或 audit 任一失败：

- 保持 `phase08_authorized=false`；
- 不更新 master 授权状态；
- 保留本地 artifacts 和错误报告；
- 不覆盖旧 release。

## 11. Fresh-clone 最终门控

从远端重新创建完全独立的 fresh clone，不复用旧 object cache。必须验证：

- `git status` 干净；
- default HEAD 正确；
- `git fsck --full` PASS；
- removed paths 在 master 全历史不可达；
- rewritten dev-state anchor 存在且内容 hash 正确；
- runtime publication verifier PASS；
- runtime verifier mutation tests fail closed；
- handoff verifier PASS；
- implementation commit 是 handoff commit 祖先；
- archive 从 release 可下载；
- actual archive SHA 与 detached manifest相同；
- embedded index 与仓库 index 字节一致；
- history map v2 在 repo/package/archive 中内容一致；
- full scientific audit PASS；
- archive 中不存在 install/build/cache/wheel/sdist 副本；
- clean wheel install PASS；
- clean sdist install PASS；
- official profile load PASS；
- 无 renderer 环境创建/reset PASS；
- focused Phase 07 tests PASS；
- full available non-renderer regression PASS；
- Black、isort、`git diff --check` PASS。

至少运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_oracle.py \
  tests/test_shakebench_phase07r5_motion_semantics.py \
  tests/test_shakebench_phase07_completion_red.py \
  tests/test_shakebench_actuators.py \
  tests/test_shakebench_physics_profile.py \
  <新增的 R6 archive/runtime/package/handoff tests>
```

并运行所有可用的 ShakeBench 非 renderer 测试。

## 12. 最终报告

更新 `docs/phase_07_report.md` 顶部 R6 section，明确记录：

- R6 status 和 `phase08_authorized`；
- implementation commit 和 handoff关系；
- old/new dev-state anchors；
- archive URL、实际字节数和 SHA；
- embedded index、release manifest、rewrite map v2 hash；
- package evidence及 wheel/sdist hashes；
- archive before/after大小；
- install/build/cache重复成员 before/after数量；
- repo `.git` 大小；
- removed paths reachability；
- 所有测试和 audit 命令及结果；
- fresh clone路径和 HEAD；
- remote lease/push结果；
- backup和恢复命令。

报告只能有一个当前结论。旧 R5 结论放入 Historical section，不得让多个 PASS/BLOCKED 标题并列造成歧义。

## 13. 最终授权规则

只有以下全部成立才允许 `phase08_authorized=true`：

1. actual archive bytes 由 detached release manifest认证；
2. release 可重新下载且 full audit PASS；
3. embedded index、repo index、rewrite map一致；
4. archive 无 package/install/build/cache重复内容；
5. rewritten anchor 可验证；
6. handoff ancestry 由可执行 verifier验证；
7. package verifier 不存在假阳性；
8. manifest/report/index/package evidence hashes完全一致；
9. runtime package 不依赖 raw archive或网络；
10. fresh clone 全部门控通过。

任一条件不满足时保持 BLOCKED，并停止；不要开始 Phase 8 implementation。
