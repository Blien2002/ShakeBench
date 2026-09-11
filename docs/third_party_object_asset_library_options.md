# ShakeBench 第三方物品素材库候选

调研日期：2026-09-10。目标是选择一个有明确生活语义、可用于低摩擦与中等重心配置、并能在视觉和物理复杂度上接近 robosuite 原生物体的素材来源。

## 仓库风格基线

当前 robosuite 的 `can / milk / bread / cereal` 视觉网格约为 104–956 个三角面，通常使用一张贴图；模型由 Blender 导出，通过 MJCF 指定网格、材质、质量和接触参数。第三方高精度扫描不应原样进入物理碰撞：建议将视觉网格降至约 500–1,500 个三角面，纹理缩至 512–1024 px，并使用圆柱、盒体或少量凸包作为碰撞体。

## 候选库

| 优先级 | 素材库 | 适合的具体对象 | 风格与接入 | 许可证 |
| --- | --- | --- | --- | --- |
| 1 | RoboCasa Object Library | `canned_food`（23 个）、`can`（21 个） | 与 robosuite 同一 MuJoCo / MJCF 生态，已有视觉、碰撞和对象区域标注；从 Objaverse 来源的人工模型中选一件即可，接入成本最低 | 官方仓库声明 assets / datasets 为 CC BY 4.0 |
| 2 | YCB Object and Model Set | `010_potted_meat_can`、`005_tomato_soup_can`、`007_tuna_fish_can` | 实物扫描并带物理尺寸、质量与高分辨率模型，机器人研究中使用广泛；需要转换为 MJCF并简化碰撞 | 模型通常按 CC BY 4.0 分发；正式纳入前应保存下载包中的许可证和署名信息 |
| 3 | Google Scanned Objects + MuJoCo port | 矮罐、塑料杯、陶瓷小盅、食品容器 | 1,030 件真实家居物体；已有社区 MuJoCo 版本，包含 OBJ、纹理、MJCF 和 V-HACD 碰撞凸包。视觉更写实，需要降面和降纹理 | 3D 资产 CC BY 4.0，社区 MJCF 为 MIT |
| 4 | NVIDIA HOPE | `canned_mushrooms`、`butter`、`yogurt_cup` 等玩具食品 | 28 件专门为机器人抓取选择的食品/日用品，提供高、低分辨率纹理网格；语义很清楚，比例也适合桌面操作 | CC BY-NC-SA 4.0，只适合非商业用途，派生资产须同许可共享 |
| 5 | Poly Haven | 杯、罐、厨房容器等零散模型 | 资产质量稳定、尺寸规范、全部由维护方审核；多数模型和 PBR 纹理明显比 robosuite 更精细，需要较多简化，且食品小物品覆盖不如前三者 | CC0，可修改和再分发 |

## 最匹配的具体选择

如果优先考虑代码与视觉风格的一致性，首选 **RoboCasa 的人工建模 `canned_food`**。RoboCasa 建立在 robosuite 上，官方对象页列出 3,200 多件物体、150 多个类别，其中 `canned_food` 有 16 件 Objaverse 来源和 7 件 AI 生成来源；本项目应只从 16 件 Objaverse 来源中选择外形规则、无透明件、无开口内容物的一件。

如果优先考虑可解释的真实物理参数，首选 **YCB `010_potted_meat_can`**。YCB 论文给出的实物参数为 370 g、约 50 × 97 × 82 mm，与 ShakeBench 的 349 g 目标质量接近，而且是明确的食品罐头。`005_tomato_soup_can` 恰好为 349 g，但高度约 101 mm，重心更高；`007_tuna_fish_can` 只有约 33 mm 高，重心偏低。

如果希望直接获得现成 MuJoCo 文件，首选 **Google Scanned Objects 的 MuJoCo port**。其每件物体已有 `model.obj`、`texture.png`、`model.xml` 和 32 个 V-HACD 碰撞子网格；接入前仍建议将碰撞体进一步缩减为一个圆柱或少数凸包，以维持 ShakeBench 接触可解释性。

HOPE 很适合论文中的“牛奶、面包、罐头”生活情境，但其非商业与相同方式共享条款会限制 benchmark 的后续发布。ABO 暂不列为候选：官方落地页写 CC BY 4.0，而 CVPR 论文和 AWS Registry 写 CC BY-NC 4.0，且完整 3D 包约 154 GB，许可证表述与接入成本都不够干净。BigBIRD 官方页面未清楚给出模型再分发许可证，也暂不采用。

## 第一手来源

- [RoboCasa 官方仓库](https://github.com/robocasa/robocasa)：基于 robosuite、3,200+ 对象、assets / datasets 为 CC BY 4.0。
- [RoboCasa 对象目录](https://robocasa.ai/docs/build/html/assets/objects.html)：各类别数量与来源拆分。
- [YCB Object and Model Set 论文](https://www.ri.cmu.edu/pub_files/2015/7/ICAR-FINAL.pdf)：实物选择原则、扫描模型和物理参数。
- [Google Scanned Objects 官方介绍](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)：1,030 件扫描物体及 CC BY 4.0。
- [Google Scanned Objects 的 MuJoCo 适配仓库](https://github.com/kevinzakka/mujoco_scanned_objects)：MJCF、纹理和 V-HACD 碰撞结构。
- [NVIDIA HOPE 官方仓库](https://github.com/swtyree/hope-dataset)：28 件玩具食品、网格下载与 CC BY-NC-SA 4.0。
- [Poly Haven 许可证](https://polyhaven.com/license)与[模型标准](https://docs.polyhaven.com/en/technical-standards/models)：CC0、尺寸和网格质量要求。
- [ABO 官方数据页](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)与[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Collins_ABO_Dataset_and_Benchmarks_for_Real-World_3D_Object_Understanding_CVPR_2022_paper.pdf)：规模、PBR 模型及许可证表述差异。
