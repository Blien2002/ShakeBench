# 权威设计文档导入记录（IMPORT PROVENANCE）

导入日期：2026-08-29
目标仓库：robosuite fork（仓库将改名为 ShakeBench）
原则：本仓库 `docs/` 根是这些文档的唯一权威副本；实现与提示词不得引用任何外部备份目录。

| 文件 | 来源（导入前） | SHA-256 |
|---|---|---|
| robosuite_benchmark_design_tree.md | 原设计仓库 docs/（经 docs/shakebench_prompts 副本核对为最新版） | ad2c5f947b16fd26575aadc4d9aa72801859c3087a7c3e5faead0cabfbea1094 |
| robosuite_benchmark_v0_spec.md | 原设计仓库 docs/（经 docs/shakebench_prompts 副本核对为最新版） | 6adcf97a3b98a83fb62f1419f1246b0ad0a4a17485adae1cb23549dac27798a8 |
| spike_implementation_formal_review.md | 原设计仓库 docs/research/ | fe59f27b0a35d3fc96276292a586114fa352de426544962d6185e67ad2e8ec9f |
| canonical_imu_profile_validation.md | 原设计仓库 docs/research/ | a81a0bd5e27bdea4f4363c8475edd0faa4be3256c89d600cad24ac68e2d3f833 |
| repro_weld_sag_20260828.md | 原设计仓库 docs/reports/ | 7af3fe4963470c98079d5e1ba6f2fbfc7b01d4701bca63404207767c8684b0de |

phenolic 纹理 `assets/textures/phenolic_bench_dark_1k.jpg`（SHA-256 6fb5d97aa0169d7e1f6897d61687148d00566cf7bcd9ec8fe0315c23ad9d3bdb，163720 bytes）为二进制文件，需在仓库改名步骤中一并复制到 `robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.jpg`；Phase 03 在该文件就位前使用确定性深灰中性材质占位。
