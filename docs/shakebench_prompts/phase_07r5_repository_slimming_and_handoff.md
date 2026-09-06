# Phase 07R5 Repository Slimming + Final Handoff

你在 `/home/miracle04/Desktop/ShakeBench` 中完成两件连续工作：

1. 把大型 raw evidence 从代码仓库和 runtime package 外置到可校验的 release archive，并在
   隔离 clone 中重写只包含 ShakeBench 历史的相关 Git refs；
2. 在重写后的历史上完成 Phase 07R5 implementation/evidence commit 与 metadata handoff，
   最终授权 Phase 8。

当前 R5 controller 和 evidence core 已全部通过，唯一原 blocker 是 final Git handoff。仓库
当前 `.git` 约 `1.1 GB`，pack objects 约 `874 MiB`；当前 HEAD 中至少 21 个 ≥10 MB 文件共约
`587 MiB`，主要是 Phase 06 raw JSON。`.gitignore` 只能阻止新增，不能删除历史 blob。

本任务涉及历史重写和 force-with-lease。严格执行“备份 → archive upload → isolated rewrite →
anchor migration → clean verification → remote update”的顺序。任一不可恢复前置条件失败时，
停止在本地并保留原仓库；不要部分更新远程。

## 1. 权威输入与禁止项

读取：

```text
docs/phase_07_report.md 顶部 R5 current section
docs/phase_07_r5_manifest.json
docs/phase_07_r4_manifest.json
docs/shakebench_prompts/phase_07r5_oracle_motion_semantics_closure.md
docs/shakebench_prompts/phase_07r5_final_commit_handoff.md
docs/robosuite_benchmark_design_tree.md 的 provenance/release sections
```

保持以下科学内容不变：

```text
controller profile = shakebench.reference_oracle.v3
profile SHA-256     = 60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c
official physics   = shakebench.official.physics.v2
physics SHA-256    = c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c
dev state rows, IDs, seeds and poses
physics/contact/task/success/controller numeric semantics
all raw payload and trace hashes
```

不要修改 controller 行为、重新选择 profile、重跑 knee/official states，或用仓库瘦身改变实验
结果。不要把 robot meshes、运行时纹理、official profile、compact protocol/status/handoff、
schemas 或小型回归 fixtures 当作 raw evidence 删除。

## 2. 建立大小与引用基线

保存以下输出到 compact migration report：

```bash
git status --short
git remote -v
git branch --show-current
git count-objects -vH
du -sh .git
git for-each-ref --format='%(refname) %(objectname)'
git rev-list --objects --all
```

生成按 blob size 排序的历史对象清单，以及当前 HEAD 中 ≥5 MB 文件清单。分类为：

```text
runtime-required asset
compact provenance/control artifact
raw scientific evidence
invalidated/historical raw evidence
regenerable build/package output
upstream documentation/video
```

对所有候选移除路径建立显式
`docs/shakebench_evidence_removed_paths_v1.txt`。每一行记录 path、当前/历史状态、移除理由和
archive group。路径选择只覆盖 ShakeBench 生成的 raw/diagnostic/replay traces 和误入历史的
`out/` 文件；上游 branch、robot meshes 和未评估资产保持不动。

列出哪些 local/remote refs 能到达这些大对象。`refs/codex/*` 是本地 checkpoint，不得推送；
远程无关的 upstream branches 不删除、不重写。记录远程 default branch 当前 SHA，作为稍后
`--force-with-lease` 的精确 lease。

完成标准：移除路径和受影响 refs 是有限、显式、可审查集合；有 before-size 基线。

## 3. 备份 dirty worktree、Git 历史和 raw evidence

在任何删除、history rewrite 或 remote mutation 前，在仓库外创建带日期的备份目录，例如：

```text
/tmp/shakebench_repo_slimming_backup_<timestamp>/
```

保存并校验：

1. `git diff --binary` 的 tracked-worktree patch；
2. 所有准备提交的 untracked Phase 05–R5 source/tests/docs/manifests 的 tar archive；
3. `git bundle create ... --all` 和 `git bundle verify`；
4. 当前 `git status`、refs、remote URL、HEAD、object-size report；
5. R5 manifest 和全部 ignored `out/phase07*` raw evidence；
6. 每个备份文件的 SHA-256。

排除 build/cache/package-install 的重复副本，但保留 package-evidence JSON 和 wheel/sdist hashes。
原始 raw traces不得删除。不要在同一远程创建指向旧历史的 backup branch/tag，因为它会继续
保留全部大对象；backup bundle 只保存在仓库外，并在最终报告给出路径/hash/recovery 命令。

完成标准：bundle 验证通过，dirty worktree 与 raw evidence 都能从仓库外备份恢复。

## 4. 先冻结当前 R5 实现

在原仓库中重新运行 R5 manifest 所列 semantic、determinism、actuator、package 和测试命令，
逐项重算 artifact file hashes。确认唯一 blocker 仍是：

```text
final_commit=null
phase08_authorized=false
```

明确排除 `.codex/config.toml`、ignored `out/`、build/cache 后，显式 stage Phase 05–R5
implementation、tests、docs、prompts、R4/R5 compact manifests 和 `.gitignore`，创建一个
pre-rewrite implementation commit。不要使用未经审查的 `git add -A`。

该 commit 仍保持 R5 manifest BLOCKED；它只是让 dirty worktree 成为可迁移的 Git tree。记录
commit SHA 为 `PRE_REWRITE_IMPLEMENTATION_COMMIT`。如果 commit 后仍有非 ignored 的项目文件，
分类并处理；本地 `.codex/` 应加入 `.gitignore` 而不删除。

完成标准：所有需迁移内容已进入一个 commit，原仓库工作树除 ignored raw 外干净，备份 bundle
仍可恢复 rewrite 前历史。

## 5. 构建可发布 evidence archive

根据第 2 节 path list 创建版本化 evidence index，至少记录：

```text
schema/version
archive release tag and expected asset URL
original repository/path
phase/status: authoritative, diagnostic, invalidated, or build evidence
uncompressed size
file SHA-256
payload/trace SHA-256 where applicable
profile/state/protocol bindings
archive member path
```

创建压缩 archive（优先 tar.zst；环境不支持时使用 tar.gz），包含：

- 从 Git 移出的 Phase 06 raw/diagnostic/replay evidence；
- R5 final raw evidence与必要的 R2–R4 provenance；
- R4/R5 compact manifests 和一份独立校验脚本；
- old→new commit mapping 的预留路径；
- README，说明 full audit 和恢复方式。

不重复打包可重新构建的 wheel/sdist/install trees；保留其 hashes 和 package-evidence JSON。
archive 大于发布平台单文件限制时按确定性分卷，逐卷记录 hash。先在本地解包到新临时目录，
验证每个 member 的 path/size/hash 和所有 semantic raw payload。

在代码仓库中只保留 compact `docs/shakebench_evidence_index_v1.json`、path list 和 audit CLI。
预先确定一个不会指向旧历史的 release tag，例如 `shakebench-evidence-v0-2026-09`。

完成标准：archive 自包含、可解包、逐文件 hash 全通过；代码仓库内没有 archive 本体。

## 6. 拆分 runtime verification 与 full audit

当前 official loader/Phase 06 handoff 会打开 package-owned raw evidence。把验证接口拆成：

```text
runtime verification
  - package-owned official profile
  - compact protocols/status/manifests
  - schema、profile、selected tuple、payload hashes 和 current package contract
  - import/load/environment creation 不访问网络或完整 raw archive

full audit
  - 用户显式提供 --evidence-archive 或 --evidence-root
  - 校验 archive/index/hash
  - 重新打开全部 raw evidence并执行现有 semantic/numeric recomputation
```

默认 official profile loader 使用 runtime verification；release audit CLI 使用 full audit。缺少
archive 时 full audit 明确报告缺失，runtime import 仍可用。不得把 stored PASS Boolean 变成
runtime authority：compact manifest 必须绑定 profile/protocol/payload hashes和 history-rewrite
mapping。

更新 `MANIFEST.in` / package-data，排除外置 raw/diagnostic traces，保留 runtime meshes、textures、
profiles、compact manifests 和小 fixtures。新增测试：

- clean wheel/sdist 不包含 removed-path list 中任何成员；
- clean install 可 load official profile 和创建无 renderer 环境；
- runtime verifier 对 compact manifest mutation fail closed；
- full audit 对 archive 缺失、member mutation、hash mismatch、换绑 fail closed；
- 完整 archive full audit PASS。

这些修改只改变证据存放和验证入口，不改变 physics/controller output。若任何 action/state/metric
trace 因代码变化而不同，停止并回到 R5 重建相关 evidence。

完成标准：runtime package 显著缩小且无需 raw archive；full audit 保留原科学可复算能力。

## 7. 创建 slimming implementation commit

在原仓库或专用 slimming branch 中：

1. 加入 compact evidence index、removed-path list、runtime/full-audit代码、tests和文档；
2. 从当前 tree 移除 path list 中的 raw evidence；
3. 保证 raw bytes 已在外部 archive/backup 中；
4. 运行 focused tests、full available non-renderer regression、Black、isort、`git diff --check`；
5. 构建新的 wheel/sdist，记录 before/after size 和内容列表；
6. full audit 使用外部 archive通过。

提交为 `SLIMMING_SOURCE_COMMIT`。此时还没有重写旧历史，远程也未改变。

完成标准：当前 tree 和 package 已精简，所有运行/完整审计测试通过，archive 可恢复所有移出文件。

## 8. 在隔离 clone 中重写历史

从包含 `SLIMMING_SOURCE_COMMIT` 的本地仓库创建新的隔离 clone 到 `/tmp`，不要直接对唯一工作
副本运行 filter-repo。记录 clone source 和 HEAD。确保 `git filter-repo` 可用；不可用时安装或
停止，不用 `filter-branch` 临时替代。

使用 `docs/shakebench_evidence_removed_paths_v1.txt` 对 `master` 和明确识别出的 ShakeBench-owned
refs 执行 `git filter-repo --invert-paths`。不要 rewrite/delete 无关 upstream remote branches，
不要包含 `refs/codex/*`。保存 filter-repo `commit-map` 和 `ref-map`，加入 evidence archive，
同时把 compact commit mapping 加入 rewritten tree。

历史重写会改变 Phase 6/7 registration、dev-state anchor 和 R5 commit SHA。根据 commit-map 迁移
所有 Git-anchor verifier：

- 保留 original commit SHA 作为 `pre_history_rewrite_commit`；
- 记录对应 `rewritten_commit`；
- ancestry gate 使用 rewritten commit；
- 验证对应 commit 的 relevant tree/payload hashes 与迁移前一致；
- dev state rows/IDs/seeds/poses 不变；
- official physics/controller/profile/payload hashes 不变。

不要静默用新 SHA 覆盖历史 SHA。增加
`docs/shakebench_history_rewrite_map_v1.json`，记录 old→new、path-list hash、archive hash、工具版本
和 rewrite command。所有受影响的 Phase 6/7 handoff、installed-package fallback 和 provenance
tests 必须更新并通过。

在隔离 clone 中运行：

```text
git fsck --full
git count-objects -vH
current-tree large-file audit
removed path absent from all rewritten target refs
Phase 6 runtime handoff
Phase 6 full audit against archive
Phase 7 semantic/determinism/actuator/package manifests
focused + available non-renderer regression
clean wheel/sdist installation
```

完成标准：rewritten refs 不再包含 removed blobs；commit mapping 可验证；科学 hashes 不变；
Git pack 和 package size 相比基线显著下降。

## 9. 在重写历史上完成 R5 handoff

在隔离 clone 的 rewritten branch 上，把当前通过全部 tests/audits 的 source commit记为
`REWRITTEN_IMPLEMENTATION_COMMIT`。只修改 R5 handoff metadata：

```text
docs/phase_07_r5_manifest.json
docs/phase_07_report.md
docs/shakebench_prompts/README.md 当前状态
compact evidence/history-rewrite indexes（仅最终 URL/hash）
```

设置：

```text
status             = PASS
final_commit       = REWRITTEN_IMPLEMENTATION_COMMIT
phase08_authorized = true
final_commit_role  = rewritten_implementation_and_evidence_commit
handoff_rule       = final_commit must be an ancestor of the commit containing this manifest
```

创建 metadata `HANDOFF_COMMIT`。不要在它自己的文件中预写 `HANDOFF_COMMIT` literal SHA；由当前
HEAD 解析。验证两个 commit 的差异只有 metadata/index，且
`REWRITTEN_IMPLEMENTATION_COMMIT` 是 `HANDOFF_COMMIT` 的祖先。

完成标准：两提交 handoff 无自引用，manifest/report 只有一个当前 PASS 结论。

## 10. 先发布 evidence archive，再更新 master

从 remote URL解析 owner/repository。若为 GitHub，优先使用已登录 `gh`：

1. 在 `HANDOFF_COMMIT` 创建新的 evidence release tag；
2. 只推送该新 tag，不推送旧 backup refs；
3. 创建 release 并上传 archive/index/checksums/commit-map；
4. 从 release URL 下载到新临时目录；
5. 重算 archive hash并运行 full audit；
6. 确认 release tag 指向 rewritten history，不可到达旧大对象。

若不是 GitHub或没有已授权的外部存储，停止在 force-push 之前，报告 archive 路径/hash和所需
外部操作；远程 master 保持不变。不要在 archive 尚未可下载校验时删除远程历史。

完成标准：远程 evidence archive 可独立下载，hash/full audit 通过，release tag 不保留旧历史。

## 11. 安全更新远程历史

重新获取远程 default branch SHA，必须等于第 2 节保存的 lease；否则远程在本轮期间被他人更新，
停止并重新协调，不能覆盖新提交。

使用精确：

```text
--force-with-lease=refs/heads/<default>:<captured-old-sha>
```

只推送 rewritten default branch 和明确迁移的 ShakeBench-owned refs。不要使用 `git push --mirror`，
不要推送 `refs/codex/*`，不要删除无关 upstream branches。对仍引用旧 ShakeBench raw history 的
tags/branches逐一列出并迁移或删除；未授权/不明确的 ref 先停止报告。

push 后从远程执行全新 clone，不复用 object cache，验证：

- default HEAD=`HANDOFF_COMMIT`；
- history-rewrite map 和 R5 manifest 可读取；
- removed raw paths 在目标 refs 全历史不可达；
- Git clone/.git/current checkout/package size 与 before 对比；
- runtime profile/environment smoke PASS；
- 下载 release archive 后 full audit PASS；
- Phase 07R5 status PASS、`phase08_authorized=true`、ancestor binding PASS。

Git hosting 可能延迟回收 unreachable objects；报告 push 后可达对象大小和 fresh-clone size。若
平台显示的 repository size 暂未下降，等待服务端 GC或按平台流程申请回收，不再次重写历史。

## 12. 最终完成条件

只有以下全部满足才允许进入 Phase 8：

1. repo 外 backup bundle、dirty-worktree backup 和 evidence archive均可恢复；
2. raw evidence release 可下载并通过 full audit；
3. runtime loader/package 不依赖完整 raw archive；
4. removed paths 不存在于 rewritten ShakeBench refs 的任何 commit；
5. old→new commit map 完整，Phase 6/7 Git anchors 可验证；
6. official physics/controller/dev-state rows和所有 raw payload/trace hashes未改变；
7. rewritten fresh clone 与 package显著小于 before baseline；
8. R5 manifest status=PASS、`phase08_authorized=true`；
9. implementation/evidence commit 是 metadata handoff commit 的祖先；
10. remote update 使用精确 force-with-lease，没有覆盖并发提交或无关 branches；
11. fresh remote clone 的 runtime audit、full audit、tests 和 package smoke通过；
12. 原 workspace 和 ignored raw evidence未删除，最终报告包含恢复命令。

完成后更新 `docs/phase_07_report.md` 顶部，报告 before/after Git/package size、release URL/hash、
removed path count、commit-map hash、rewritten implementation/handoff commit、remote lease/push结果和
fresh-clone验证。停止，不开始 Phase 8 implementation。
