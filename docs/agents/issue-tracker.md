# Issue tracker: GitHub

ShakeBench 的需求、缺陷、门控遗漏和后续技术债以 GitHub Issues 为权威记录：

```text
https://github.com/Blien2002/ShakeBench/issues
```

仓库内的 phase prompt、manifest 和 report 是设计与证据文档，不替代 issue 状态。若二者冲突，以代码和可重放证据判断事实，并在对应 issue 中记录差异。

## Repository

```text
owner: Blien2002
repository: ShakeBench
default branch: master
```

使用 `gh` CLI 操作，并显式指定仓库，避免当前旧 recovery workspace 的本地 refs 影响目标判断：

```bash
gh issue list --repo Blien2002/ShakeBench --state open
gh issue view <number> --repo Blien2002/ShakeBench --comments
```

## Pull requests as a request surface

**PRs as a request surface: no.**

Pull request 可以引用或关闭已有 issue，但未关联 issue 的外部 PR 不自动进入需求队列。GitHub Issues 和 PR 共用编号；遇到裸 `#<number>` 时，先用 `gh issue view`，找不到再用 `gh pr view` 判断。

## Issue conventions

每个 issue 只描述一个可验证的问题。标题使用：

```text
[Phase NN] imperative summary
[Tooling] imperative summary
[Docs] imperative summary
```

正文至少包含：

- Context：问题出现在哪个 phase/commit/artifact；
- Evidence：可复现命令、错误输出和文件行号；
- Impact：是否阻塞 phase entry、只阻塞 evidence publication，或属于非阻塞技术债；
- Acceptance criteria：可执行且能明确判断 PASS/FAIL；
- Protected scope：不得改变的 physics/controller/state/raw-evidence authority；
- Verification：修复后必须运行的测试或审计命令。

严重度：

- `P0`：可能错误授权 official/scoreable 结果，或破坏科学证据真实性；
- `P1`：确定的功能/工具错误，但不改变当前已冻结科学结论；
- `P2`：测试覆盖、可维护性或文档问题；
- `P3`：低优先级重构和体验改进。

状态使用 GitHub 原生状态：

- OPEN：尚未满足 acceptance criteria；
- CLOSED：修复已合入，并在关闭评论中记录 commit 和验证结果；
- 若暂时无法推进，在正文顶部记录 `Blocked by: #...`，不要关闭 issue；
- 重新出现或原验证不足时重新打开，不创建内容相同的新 issue。

## Agent workflow

### 查找与读取

在开始实现或审查前：

```bash
gh issue list \
  --repo Blien2002/ShakeBench \
  --state open \
  --json number,title,body,labels,comments,url

gh issue view <number> \
  --repo Blien2002/ShakeBench \
  --comments \
  --json number,title,state,body,labels,comments,url
```

先搜索现有 issue，避免重复：

```bash
gh issue list --repo Blien2002/ShakeBench --state all --search '<keywords>'
```

### 创建

```bash
gh issue create \
  --repo Blien2002/ShakeBench \
  --title '[Phase NN] concise imperative title' \
  --body '<context, evidence, impact, acceptance criteria, verification>'
```

不要仅以“某个 stored PASS 为 false/true”创建 issue；必须提供行为、代码或可重放证据。

### Commit 和文档引用

实现 commit 使用：

```text
Refs #<number>
```

只有 acceptance criteria 已全部满足时才使用：

```text
Fixes #<number>
```

Phase prompt、manifest、report 中引用 issue 时写成 GitHub 链接或 `#<number>`，并明确它是 entry blocker、publication blocker 还是 follow-up。

### 更新与关闭

```bash
gh issue comment <number> --repo Blien2002/ShakeBench --body '<evidence update>'
gh issue close <number> --repo Blien2002/ShakeBench --comment '<commit and verification summary>'
```

关闭评论必须包含：

- 修复 commit；
- 实际执行的验证命令；
- PASS/skip 数量；
- 尚未执行的环境依赖测试；
- 对 phase authorization 的影响。

## Phase-gate policy

Issue 的存在本身不自动阻塞下一阶段。只有满足以下任一条件才是 entry blocker：

1. 会导致错误的 official/scoreable 判定；
2. 会改变已冻结的 physics/controller/state semantics；
3. 当前阶段的权威 artifact 无法独立认证或重放；
4. 下一阶段的首个必需操作必然触发该缺陷，且没有安全替代路径。

只影响未来 archive 构建、非权威工具入口、额外测试覆盖或内部结构的 P1/P2，可以作为下一阶段首个任务或 publication blocker 跟踪，不应无限追加 phase-entry remediation。

## Current authority baseline

建立本文件时，已审核的 Phase 07R6.1 基线为：

```text
remote master / R6.1 gate tag:
4238e37cc3b57d626a37f7950df974fc10b0d1e8

R6 evidence tag:
1824060986949c3dd618075499fb8a709e08fb36

R6 archive SHA-256:
dda863dd9941a498f0702b88105a7b50bceb523d59b7fc29cd48a2cdc9596553
```

Phase 8 可以从该远端基线开始。已知非 entry-blocking 问题必须在 GitHub Issues 中跟踪，并在首次生成 Phase 8 evidence index/archive 前解决相关 publication blocker。

## Initial tracked issues

- [#1 — Support multi-episode digests in the evidence index](https://github.com/Blien2002/ShakeBench/issues/1): P1 publication blocker；不阻塞 Phase 8 implementation，但必须在首次生成或发布 Phase 8 evidence index/archive 前关闭。
- [#2 — Exercise archive-backed package evidence end to end](https://github.com/Blien2002/ShakeBench/issues/2): P2 test coverage；不阻塞 Phase 8 entry。

## Wayfinding operations

若后续使用 wayfinding：

- map issue 使用标签 `wayfinder:map`；
- child issue 使用 `wayfinder:research`、`wayfinder:prototype`、`wayfinder:grilling` 或 `wayfinder:task`；
- 优先使用 GitHub 原生 sub-issues 和 issue dependencies；
- 若仓库未启用 dependencies，则在正文顶部使用 `Blocked by: #<number>`；
- frontier 是 map 中第一个 OPEN、无 blocker、无 assignee 的 child issue；
- claim 时先指派当前执行者，完成时追加证据后再关闭。
