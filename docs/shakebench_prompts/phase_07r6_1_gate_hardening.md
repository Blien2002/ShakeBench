# Phase 07R6.1 Gate Hardening and Final Phase 8 Authorization

本阶段只修复授权门控，不重做控制器、任务状态机、观测、物理实验、raw
evidence、dev-state rows、profile、seed、pose 或 Git 历史。R6 科学证据和
release 保持不可变；R6.1 必须在独立 `--no-local` clone 中完成。

执行顺序：

1. 先以 `BLOCKED_BY_PHASE_07R6_1_GATE_HARDENING`、
   `phase08_authorized=false`、`final_commit=null` 冻结
   `docs/phase_07_r6_1_manifest.json`。
2. 使 package evidence 成为 PASS/authorization 的强制 authority。必须验证
   schema v2、wheel/sdist 各一条、记录和 content/install checks、artifact
   filename/size/SHA-256、install-root module paths 以及 source-checkout
   import 泄漏；显式 JSON 和安全读取 archive member 两种入口都必须绑定同一
   manifest hash。
3. 验证 detached R6 manifest 的 `r6_manifest_sha256`、implementation/handoff
   rule、release tag/URL/archive size/hash、package metadata、index/map/removed
   path hashes。重新封装 self-hash 不能绕过这些绑定。
4. 对 R6 tag 和 R6.1 gate tag 执行 Git object、peeled commit、祖先关系和
   tag-tree bytes 校验；提交区间必须逐提交覆盖 R6 handoff 到 R6.1
   implementation、implementation 到 metadata handoff，以及 handoff 后的
   HEAD，不能只寻找最后一次 manifest 修改。
5. `--evidence-root --scientific-only` 只能执行 extracted-root scientific
   audit，并明确返回 `release_authority_verified=false`；只有带原始 archive
   bytes 和 detached authority 的 archive 模式可以作为 release authority。
   root audit 必须枚举实际 regular members，发现缺失、增加或篡改即失败。
6. 修复 evidence-index digest bindings：对实际声明的
   `payload_sha256`、`trace_sha256`、`trace_whole_digest` 正确绑定，错误的
   64-hex 值必须失败，普通 JSON 不得伪造不存在的 binding。
7. 从 R6.1 implementation tree 构建一次 clean wheel/sdist，生成唯一的
   `out/phase07r6_1/package_evidence_final.json`；不把安装树、wheel、sdist
   或 raw evidence 提交进 Git 或 R6 scientific archive。新建轻量 gate release
   `shakebench-gate-v0-2026-09-r6-1`，只发布 R6.1 manifest、package evidence、
   gate attestation/checksums 和 `SHA256SUMS`，引用既有 R6 archive。

R6.1 implementation commit 完成后，manifest 才能在独立 metadata handoff
commit 中改为 `PASS` 和 `phase08_authorized=true`。handoff 自身不得预写
自己的 SHA；最终 fresh clone 必须从远端 HEAD 验证全部 gate、tag、package、
archive、focused tests、non-renderer regression、Black、isort 和
`git diff --check`。任一 gate 失败都保持 BLOCKED，不开始 Phase 8。
