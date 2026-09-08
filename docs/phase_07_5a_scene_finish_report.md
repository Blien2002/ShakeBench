# Phase 7.5A：灯光、机柜、黄黑栏杆与空白显示屏

2026-09-07。在 `.phase07_5a_clean` 的 `direct_mount_v1` 装配上完成用户提出的四项外观修改。

## 光照与阴影

参考 [ManiSkill 的照明源码与教程](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/docs/source/user_guide/tutorials/domain_randomization.md)、robosuite 标准场景及 MuJoCo 3.9 原生渲染约束。
完整来源、固定 commit 与默认/可选设置区别记录在 [研究笔记](lighting_benchmark_references.md)。
ManiSkill 与 robosuite 的这些默认示例关闭阴影；本场景根据用户要求使用可控投影，而非宣称直接复制默认设置。

减少原先 headlight 与三盏较强方向光的叠加；现在只有位于顶面和灯具下方的 spot 主光投影，另有不投影的弱补光。
主光 diffuse=0.32、辅光 diffuse=0.28，headlight ambient/diffuse/specular=0.31/0.18/0.025。
较低主光占比减少硬影对比；阴影图为 8192，`shadowscale=1.0` 覆盖完整照明范围，避免锥形截断。
这是根据本实验室画面调试后的 native MuJoCo 参数，不与 SAPIEN 光强作数值等价。
阴影来自 native shadow mapping，保留真实遮挡关系；没有使用后期涂黑阴影，也不把它描述成面积光路径追踪柔影。

固定全景像素平均亮度由 109.32 降至 82.63，约下降 24.4%。这是整幅画面的描述统计，包含新增黑色栏杆与机柜朝向变化，不是独立的曝光测量。

## 机柜与屏幕

机柜保持位置 (-2.05,-0.25,0) m，绕 Z 轴旋转至约 96.95°，屏幕面法向指向振动台中心。
编译后的方向点积检查 >0.99999。
移除电脑显示屏上的全部四条图形，使用无自发光的黑色空屏材质。仪器的波形显示和独立电源指示灯继续保留。
机柜近景机位已跟随正面朝向调整。

## 黄黑栏杆

横杆与立柱改为连续圆管加黄黑相间色带。转角为有正确朝外法线的弯管网格，端点与直管中心线精确相接；装配误差测试 <1 µm。
增加端部连接套、T 形套接、立柱承口、接地底板、锚固螺栓和螺栓槽。
装饰全部归 world/guardrails，零密度、零接触；没有新增物理约束。

## 验证与证据

- 30 项 scene finishes、scene 和 direct mount 测试通过。
- 名义与 safe envelope 的 21 个采样通过，无非预期穿透。
- 修改前后物理签名一致：`a1849e266a9ac705dba9abec5be8ece9e1443ad173c063cb24198a62141f0591`。
- 同初态和 20 步固定动作，21 个样本的 qpos/qvel、关键 body pose、接触、观测及 metrics 完全一致；仅排除 scene/geometry metadata。
- Black、isort、`git diff --check` 通过；wheel 独立解包后可加载并用 NVIDIA EGL 渲染。
- 检查了全景、装配、侧面、机柜、仪器、工具车和栏杆七个视角。

[全景](../out/phase07_5a/lighting_rail_revision/overview.png) · [栏杆接头](../out/phase07_5a/lighting_rail_revision/guardrail_detail.png) · [机柜正面](../out/phase07_5a/lighting_rail_revision/cabinet_detail.png) · [空白电脑屏幕](../out/phase07_5a/lighting_rail_revision/instruments_detail.png)

scene SHA-256：`4c8e22dbaf173e583731b0654870fd2bd606c41624b39be4f5d5e16391cd190b`。
geometry metadata SHA-256：`55ae9d8a8bf707751552bd069a278fc31d67c1db2b4387468a3f47279b44b6fe`（随 scene authority hash 更新，数值装配与物理签名保持不变）。

本轮只完成外观整改。此前 direct_mount_v1 的抓取控制器适配问题仍未解决；未重跑或宣称通过完整抓取演示，Phase 08 readiness 继续为 false。
前后快照、测试、独立包和渲染证据：`out/phase07_5a/lighting_rail_revision`。
