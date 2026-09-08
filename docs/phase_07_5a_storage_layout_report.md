# 储物柜与工具车布局更新

2026-09-07。在 `.phase07_5a_clean` 的 direct_mount_v1 实验室场景中完成两项用户要求：

- 机柜旁新增 0.82 × 0.54 m、台面高 0.88 m 的矮储物柜，位置 (-2.20,0.72,0) m，柜门朝向振动台外的通道。模型包含底座、倒角外壳、双抽屉、双开门、拉手、铰链及台面挡边。
- 储物柜上摆放带衬垫的六只传感器托盘、一卷带接头的信号线、两只校准参考砝码及一本闭合记录本。均为展示模型，无任务质量或接触。
- 工具车从原位置 (-1.93,1.55,0) m 移至 (-0.78,2.12,0) m，即电脑/仪器桌左侧、靠后墙。工具车后缘与墙面保留约 0.1 m 间距，右侧与桌面边缘保留约 0.18 m 间距；工具和脚轮一起移动。

[全景](../out/phase07_5a/storage_layout_revision/overview.png) · [储物柜及台面物品](../out/phase07_5a/storage_layout_revision/storage_detail.png) · [工具车新位置](../out/phase07_5a/storage_layout_revision/cart_detail.png)

19 项现有 scene/scene finishes 测试通过；名义及 21 个 safe envelope 采样通过间隙检查。
前后物理签名以及 20 步固定动作、21 个状态样本完全一致。零密度装饰归属 world，未新增接触或约束。
物理签名：`a1849e266a9ac705dba9abec5be8ece9e1443ad173c063cb24198a62141f0591`。
scene hash：`8a23e7c39b1e182b4e70171eb6631a866000e9163f4ff2db445fa457034a7dde`。

新增储物柜、摆件、工具车位置及近景相机仍由同一个 scene JSON 描述。既有灯光、黄黑栏杆、直接安装方式和空白电脑屏幕继续保留。
此前控制器在 direct_mount_v1 下的抓取适配问题仍未解决；本轮未重跑任务成功率。
证据目录：`out/phase07_5a/storage_layout_revision`。

最终 wheel 独立解包后可加载并用 native MuJoCo EGL 渲染新布局，scene hash 与工作目录一致。
