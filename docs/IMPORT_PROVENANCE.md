# 权威设计文档导入记录（IMPORT PROVENANCE）

导入日期：2026-08-29
目标仓库：`Blien2002/ShakeBench`（Python package 仍为 `robosuite`）
原则：本仓库 `docs/` 根是这些文档的唯一权威副本；实现与提示词不得引用任何外部备份目录。

| 文件 | 来源（导入前） | SHA-256 |
|---|---|---|
| robosuite_benchmark_design_tree.md | 当前仓库权威副本；2026-08-30 与 Phase 01R candidate/reconciliation 同步 | 47ad5a9200dbca708ca1577756435a866f8b7d340c7ac747fcfc17bf7c49af10 |
| robosuite_benchmark_v0_spec.md | 当前仓库权威副本；2026-08-30 与设计树状态协议同步 | a8eb3e568851199386c1e0c89120d3811bbb08029c57a1209dbffc675904f346 |
| spike_implementation_formal_review.md | 原设计仓库 docs/research/ | fe59f27b0a35d3fc96276292a586114fa352de426544962d6185e67ad2e8ec9f |
| canonical_imu_profile_validation.md | 原设计仓库 docs/research/ | a81a0bd5e27bdea4f4363c8475edd0faa4be3256c89d600cad24ac68e2d3f833 |
| repro_weld_sag_20260828.md | 原设计仓库 docs/reports/ | 7af3fe4963470c98079d5e1ba6f2fbfc7b01d4701bca63404207767c8684b0de |

phenolic 纹理 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg` 已纳入当前仓库并由 `.gitignore` 最小例外保护（SHA-256 `6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb`，163720 bytes）；`MANIFEST.in` 的递归 assets 规则覆盖它。

2026-08-30 reconciliation：设计树和 v0 spec 的原始导入 hash 已由上述当前权威副本 hash 替代；原始导入证据保留在 Git 历史 commit `d1db196e` 中。Phase 01R 的 authored profile 选择 B，旧 ShakeBench exact-reuse 来源未在当前仓库内证明。
