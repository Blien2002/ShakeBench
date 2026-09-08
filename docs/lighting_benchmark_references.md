# 实验室光照与阴影参考

核对日期：2026-09-07。目的：降低 ShakeBench 实验室的过曝感，保留器材形体和接触阴影。以下区分官方配置事实与本场景的设计建议；SAPIEN 与 native MuJoCo 的光强数值不能直接等价搬用。

## 官方配置事实

| 项目 | 已核对的光照与阴影设置 | 对本场景的启示 |
| --- | --- | --- |
| ManiSkill 默认环境 | `enable_shadow=False`；ambient 为 `[0.3, 0.3, 0.3]`。第一盏 directional 方向 `[1, 1, -1]`、颜色 `[1, 1, 1]`；第二盏方向 `[0, 0, -1]`、颜色 `[1, 1, 1]`。第一盏阴影开关继承 `enable_shadow`，启用时 `shadow_scale=5`、`shadow_map_size=2048`。第二盏调用未主动开启阴影。 | 学习有限数量的主光与补光，以及可控阴影开关；不要把默认场景描述成所有光源都投影。 |
| ManiSkill 随机化教程 | 示例把 ambient 每通道随机设为 `0.2–0.6`，第一盏显式 `shadow=True`、`shadow_scale=5`、`shadow_map_size=4096`，另有一盏向下补光。 | 4096 是教程中的可选投影配置，区别于当前默认加载函数的 2048。随机化范围并非室内布光推荐值。 |
| robosuite table arena | 一盏 directional：`pos="1 1 1.5"`、`dir="-0.2 -0.2 -1"`、`specular="0.3 0.3 0.3"`，明确 `castshadow="false"`。该灯没有显式覆盖 diffuse / ambient。base XML 只设置 `map znear="0.001"`，未设置 headlight 或 shadowsize。 | 该标准桌面环境偏重均匀可辨识照明，不应把它作为强烈硬阴影的依据。继承参数须结合 MuJoCo 版本判断。 |
| MuJoCo Menagerie Panda scene | headlight：diffuse `0.6`、ambient `0.3`、specular `0`；另有 `pos="0 0 1.5"`、`dir="0 0 -1"` 的 directional。没有显式 `castshadow`，在 MuJoCo 3.9 中继承 `true`；没有单独设置阴影贴图大小。 | 可借鉴低镜面 headlight 加一盏主要投影灯；其开放地面展示场景没有封闭顶面，不宜照搬到实验室。Menagerie 是模型库示例，不能称作统一的 benchmark 光照标准。 |

对应的固定版本来源：

- [ManiSkill BaseEnv：默认开关与 `_load_lighting`](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/sapien_env.py#L848-L857)。`enable_shadow: bool = False` 在同文件第 199 行。
- [ManiSkill 域随机化教程](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/docs/source/user_guide/tutorials/domain_randomization.md)。
- [robosuite table_arena.xml](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/models/assets/arenas/table_arena.xml#L40)；[base.xml](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/models/assets/base.xml)。
- [Menagerie Panda scene.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/scene.xml)。

## native MuJoCo 3.9 的实际约束

以下已核对 [MuJoCo 3.9.0 XMLreference.rst](https://github.com/google-deepmind/mujoco/blob/3.9.0/doc/XMLreference.rst) 的 `body/light`、`visual/headlight`、`visual/quality` 和 `visual/map` 定义：

- Native renderer 使用 Phong 光照与 shadow mapping。Headlight 不投影且与显式灯光叠加；添加场景灯时需要相应降低 headlight。3.9 默认 headlight ambient / diffuse / specular 分别为 `0.1 / 0.4 / 0.5`。
- 显式灯的默认 diffuse / ambient / specular 分别为 `0.7 / 0 / 0.3`，`castshadow` 默认开启。投影是灯级属性；每盏投影灯增加一次场景渲染。
- `quality.shadowsize` 默认 `4096`，是阴影贴图分辨率；`offsamples` 默认 `4`，改善离屏图像抗锯齿。提高分辨率能减少阴影采样粗糙，但不等于真实面积光源的柔软半影。
- 定向灯 `pos` 不影响照明位置。其阴影横向范围由 `stat.extent * map.shadowclip` 控制，默认 `shadowclip=1`。Spot 的阴影锥角由 `cutoff * shadowscale` 控制，默认 `shadowscale=0.6`。太小的范围会使阴影突兀截断。
- `intensity` 和 image texture 不用于默认 Phong renderer。`bulbradius` 的作用依赖 renderer，不能承诺在 native renderer 调大它就会获得面积光柔影。核对 3.9 [classic/render_gl3.c](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/render/classic/render_gl3.c) 与 [render_context.c](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/render/classic/render_context.c) 后，没有发现原生阴影路径读取 `bulbradius`；其实现为投影矩阵、深度贴图及 shadow pass。

## 本实验室的实施建议（设计推论，不是 benchmark 原值）

1. 使用一盏位于顶面下方、朝向振动台中心的较宽 spot 作为唯一投影主光，辅以弱而不投影的定向补光。这样投影有统一方向，器材背光面仍可辨识；避免多盏强投影灯叠加多重硬影。
2. 先以 headlight ambient `0.12–0.18`、diffuse `0.12–0.20`、specular `0.03–0.06`，主光 diffuse `0.40–0.55`，辅光 diffuse `0.15–0.25` 作画面调试起点。所有灯 ambient 除 headlight 外先设零，灯 specular 保持低值。最终依据固定机位对照图而不是数字决定亮度。
3. 保持 `shadowsize=4096` 和 `offsamples=4` 起步；spot `cutoff` 可取 `65–75` 度，`shadowscale` 可先取 `1.0`，验证主工作区和栏杆影子没有突然消失再缩小。不要只增大贴图而保留过大的无关投影范围。
4. 室内完整顶面会遮挡来自无限远的投影定向光；降低 directional 的 `pos` 无法解决这一问题。优先把有限位置 spot 放在顶面下方、灯具实体下方并避开灯罩几何体。另保留不投影的弱补光。必须检查天花板/灯罩是否把主工作区整体遮黑，不能以增加全局 ambient 掩盖错误的遮挡关系。
5. 校验至少包括同曝光的全景、振动台装配近景、机柜近景：浅色墙面仍有层次；地面无大块纯白；轮脚/立柱/支架接地感清楚；金属高光不过曝；阴影方向一致且无方框或锥形截断。原生 shadow map 的硬阴影可通过降低主光与补光的对比变得较自然，但不能声称达到路径追踪面积光源的柔影效果。

本次来源研究不涉及动力学、接触、布局或控制器修改；具体已落地参数以场景配置和最终渲染记录为准。
