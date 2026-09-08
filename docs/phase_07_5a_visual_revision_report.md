# Phase 7.5A 实验室外观与器材精细建模整改

2026-09-07。在 `.phase07_5a_clean` 的既有工作上完成本轮整改，保留前序未提交工作。
本轮对应用户提出的光照、墙地面材质、器材布局，以及后续提出的器材模型精度要求。

## 最终效果

- 恢复中性白色主光、顶光和补光，使用浅灰工业墙面、环氧地坪和原版振动台孔阵列贴图。
- 墙面增加踢脚线，房间补齐顶面与灯具。工作台、机器人和振动台仍为画面主体。
- 控制柜恢复为约 0.60 × 0.65 × 1.80 m 的机柜，补上倒角、机架导轨、功放模块、旋钮滚花、拉手、螺钉、铰链、散热百叶、接口和状态灯。
- 后墙仪器台补齐示波器式仪器的屏幕边框、波形网格、旋钮、按键、BNC 外观接口、散热槽和提手；显示器配支架、逐键键盘、鼠标，台下抽屉有缝隙及紧固件。
- 工具车具备托盘翻边、内衬、把手、轮叉、轮毂、转向座，以及工具箱、螺丝刀、开口扳手和套筒。
- 急停位于护栏外，具备底板、锚固件、盒体倒角、垫圈、面板螺钉和蘑菇按钮。
- 机柜、仪器台、工具车有独立近景相机。全部细节由同一 scene JSON 描述，倒角使用确定性生成的凸网格。

[最终全景](../out/phase07_5a/lab_detail_revision/overview.png) · [机柜近景](../out/phase07_5a/lab_detail_revision/cabinet_detail.png) · [仪器近景](../out/phase07_5a/lab_detail_revision/instruments_detail.png) · [工具车近景](../out/phase07_5a/lab_detail_revision/cart_detail.png)

[最终 1280×720 / 20 fps 视频](../out/demo/phase07_5a_lab_detail_v4_v0_gamma015.mp4) · [视频 sidecar](../out/demo/phase07_5a_lab_detail_v4_v0_gamma015.json) · [关键帧](../out/phase07_5a/lab_detail_revision/keyframes.json)

## 验证

37 个测试通过，覆盖 scene、arena、demo 和 VibrationPickPlaceCan 环境；包括从 sdist 解包后编译 arena。
Black、isort 和 `git diff --check` 通过。wheel 独立解包后可加载全部贴图、创建无 renderer 环境，并以 native MuJoCo EGL 渲染。

nominal 和预注册 safe envelope 的 21 个采样均通过间隙检查，无非预期穿透对。
工具车轮与大尺寸薄地板盒相切时，MuJoCo convex distance 曾报告毫米量级假负距离；在圆柱完整位于水平地板顶面范围内时，改用精确圆柱支撑函数复核，记录独立方法名。测试确认轮子埋入地面 10 mm 仍返回 -10 mm，未增加穿透白名单。

修改前后物理签名一致：`f5f5ad4e91b0366e6d7870849858e5d21e17af15d698a6ffd9ac2f99fc01d07f`。
同初态、10 步固定动作的 11 个命名状态样本（关节 qpos/qvel、关键 body pose、接触、观测和 metrics）完全一致，只排除 scene metadata。
新增细节均为零密度、无接触的 visual，质量、惯量、关节、约束、机器人安装、canonical tabletop 和控制器保持原有物理定义。

最终演示：V0，Gamma=0.15，state000，205 steps，evaluator success=true。
科学 trace hash 与上一次光照修正版 v3 完全相同：`2b44213b4ceae45f7684724eff2ef801703f26f76accfaa4af212f804baa5192`。
检查了起始、夹持/预抬升、搬运、放置、成功保持关键帧，以及六个静态视角。

## 材质来源与证据范围

三张参考工程原创程序化 JPG 被解码为 PNG，逐像素验证 RGB 一致。
参考原文件、生成脚本及用途的 SHA-256 记录在 scene JSON；包内说明见
[texture provenance](../robosuite/models/assets/textures/shakebench_lab_materials.md)。运行时不访问 backup。

最终 scene payload SHA-256：`e301721fd3e1cc3eccfbd04cc8b19f5ac8f8e7179a7edd831769103699d72214`。
孔阵列为贴图效果，仪器屏幕为固定装饰图形。相机图像用于展示；本轮物理兼容性结论来自上述签名和轨迹对比。
本轮未运行 knee、official states 或 Phase 8；完整 Phase 7.5A readiness 由该阶段清单单独审定。
前序 v1/v2 场景视频及本轮 v3 光照视频均保留，v4 是当前精细器材版本。

原始证据位于 `out/phase07_5a/lab_detail_revision`，整改前快照位于 `out/phase07_5a/lab_style_revision/before`。
