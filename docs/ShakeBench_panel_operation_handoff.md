# ShakeBench `panel_operation` 控制面板任务交接文档

更新时间：2026-08-19  
对应项目：`/home/miracle04/Desktop/ShakeBench`  
远程源码快照：`Blien2002/ShakeBench`，`main@7dd5abdfba707475c39a00b8cc545828739c7718`

## 1. 当前结论

控制面板的视觉方案、场景接入、三类物理控件、任务状态与指标框架均已实现。旋钮单控件已经在 GPU 上通过真实碰撞操作；拨杆和按钮的物理模型已经进入 Newton/MJWarp 求解，但随仓库提供的参考控制器尚未成功完成，因此三控件完整序列、随机指令组和 official 频率验收不能标记为完成。

| 项目 | 实现状态 | 运行验证状态 |
|---|---|---|
| 斜面控制台及外观 | 已完成 | NewtonGL 场景可构建、可渲染 |
| 旋钮物理关节与碰撞 | 已完成 | 单控件 `success=true` |
| 拨杆物理关节与碰撞 | 已完成 | 能接触并推动，但未达到目标角 |
| 按钮物理关节与碰撞 | 已完成 | 能接触，但直线关节未产生有效行程 |
| 顺序、随机子集、错误操作判定 | 已完成 | 未完成多控件端到端验收 |
| 静态测试 | 已通过 | 36 passed，53 warnings |
| 三控件完整序列 | 代码路径已存在 | 未通过 |
| 随机种子矩阵 | 代码路径已存在 | 未执行完整验收 |
| official 1000 Hz | 配置路径已存在 | 未执行面板成功验收 |
| 录像与最终演示 | overlay 已支持 | 未生成三控件成功录像 |

## 2. 需求与最终设计

- 复用原有 C2 工作台、Panda、Stewart 振动平台和房间，不增加第二张工作台。
- 控制台安装在 C2 工作台上，外形为带斜操作面的五棱柱式工业控制台。
- 控件按三角形布局：旋钮左上、拨杆下中、按钮右上。
- 支持显式顺序，例如 `knob,lever,button`；未指定时，根据 `panel_seed` 确定性抽取 1～3 个互不重复控件并排列。
- 成功必须来自真实关节位移，而不是视觉动画、距离触发或直接写入进度。
- 外观参考用户提供的 Sketchfab Apollo 控制旋钮和工业控制台，但实现为项目内程序化几何，没有复制或打包第三方网格。

默认关键参数：

| 参数 | 当前值 |
|---|---:|
| 控制台深度 × 宽度 × 高度 | 0.190 × 0.320 × 0.180 m |
| 前缘高度 | 0.055 m |
| 旋钮目标角 | 72° |
| 拨杆目标角 | 30° |
| 按钮目标行程 | 4 mm |
| 完成阈值 | 目标进度的 95% |
| 移动超时 | 7 s |
| 操作超时 | 4 s |
| 接触丢失超时 | 0.30 s |

## 3. 已完成内容及具体实现

### 3.1 控制台与视觉建模

`src/shakebench/visual_assets.py` 和 `src/shakebench/panel.py` 实现了程序化控制台：

- 楔形/五棱柱式机箱，控件安装在斜面而不是垂直面。
- 独立操作面、边缘压条、后部平台、紧固件、铭牌、状态灯和三个固定座圈。
- 固定外观只包含机箱和座圈；旋钮指针、拨杆杆体、按钮帽属于各自运动链接，避免“视觉动、碰撞体不动”。
- 面板局部坐标显式计算斜面切向、法向和横向，布局及控制器共用同一坐标定义。

### 3.2 三类物理控件

`src/shakebench/panel_controls.py` 为每个控件创建两刚体、单自由度 articulation：根刚体随工作台运动，运动链接承担碰撞与可视几何。

旋钮：

- Revolute joint，局部 Z 轴沿面板法向。
- 任务目标 72°，硬上限为目标的 110%，避免成功瞬间直接加载无限硬限位。
- 碰撞代理为隐藏圆柱；外观为 Apollo 风格楔形指针及浅色索引条。
- 质量 0.06 kg，阻尼 0.08。

拨杆：

- Revolute joint，局部 Y 轴，使拨杆在斜面切向—法向平面内转动。
- 任务目标 30°，硬上限同样保留 10% 余量。
- 碰撞体由 22 mm 直径杆体圆柱和 22 mm 直径球形握把组成。
- 球形碰撞体是后续补充的；早期只有视觉握把，夹爪实际只能碰到圆柱平端。
- 质量 0.04 kg，阻尼 0.05。

按钮：

- Prismatic joint，当前限位 `[-4.6 mm, 0]`，任务行程 4 mm。
- 线性 drive：最大力 30 N、刚度 180、阻尼 2.5。
- 碰撞代理为隐藏圆柱，运动链接上有红色圆柱帽和圆顶。
- 当前模型已构建，但按钮受压时关节仍保持 0，是首要未解决问题。

### 3.3 场景接入

`src/shakebench/scene.py` 在 `panel_operation` 分支中完成：

- 面板固定刚体、三个 articulation 和外观资产的创建。
- 每个控件分别建立左指、右指接触传感器，共六个按运动链接过滤的传感器。
- `pick_place` 专用 workpiece、target 和相关传感器在面板任务中置空，避免改变原任务语义。
- 面板任务关闭 wrist camera 的物理碰撞，但保留渲染；否则相机支架会撞旋钮。
- 面板任务暂时不创建 wrist joint-wrench sensor，原因见第 6.1 节。

### 3.4 任务状态、观测和判定

`src/shakebench/panel_task.py` 的权威状态来自仿真关节：

```text
knob_progress  = knob_q / 72°
lever_progress = lever_q / 30°
button_progress = -button_q / 4 mm
```

- 三项进度裁剪到 `[0, 1]`，同时保存 episode 内峰值。
- `request_panel_progress()` 会抛出异常，禁止脚本直接推进状态。
- `mark_control_complete()` 只在真实峰值进度不小于 0.95 时锁存完成。
- 检查操作顺序、非当前控件的持续误接触、移动/操作超时和接触丢失。
- observation 包含面板三个基向量、控件姿态、拨杆 pivot/tip、按钮 face、序列 ID、关节状态及六路接触力。
- metrics 记录完成序列、三控件峰值、接触峰值、穿透深度、穿透对象、发生时间和超过 0.5 mm 的帧比例。
- 面板、三个控件根、工作台及桌腿都复用 C2 支撑运动的完整 SE(3) 写出，振动时不会只平移不旋转。

### 3.5 参考控制器

`src/shakebench/panel_controller.py` 实现确定性 DLS IK 基线：

```text
settle -> pre -> approach -> move -> operate -> retreat
```

- 使用实测手部位置判断阶段是否到达，不能用已限速的 command 自己判断。
- 保存初始 hand-to-finger-center 偏移，避免动态偏移随姿态漂移。
- IK 输出先裁剪到 Panda 关节限位，再以 1.2 rad/s 做关节空间限速。
- `move/operate` 笛卡尔速度限制到 0.05 m/s。
- 始终保持 settle 时的手部姿态，避免 DLS 在面板前换肘分支。
- 旋钮由一侧张开手指横向扫动指针。
- 拨杆闭合到每指 10 mm，并按绕 pivot 的 30° 圆弧运动；圆弧持续 3 s。
- 按钮正确接触后可提前由 `move` 进入 `operate`，随后沿面板负法向按压。
- 无论控制器处于什么阶段，完成判据仍是物理关节进度。

### 3.6 配置、CLI 与记录

- `src/shakebench/config.py`：`PanelConfig`、随机指令采样、几何/速度/阈值校验。
- `src/shakebench/cli.py`：`--task panel_operation`、`--panel-sequence`、`--panel-seed` 及面板 metrics 输出。
- `configs/scenarios.yaml`：面板关闭振动、频谱演示和显式序列场景。
- `src/shakebench/recording.py`：录像 overlay 可显示指令和三控件状态。
- `src/shakebench/diagnostics.py`：支持 `finger<->knob/lever/button/panel` 穿透分类。

## 4. 已完成验证

### 4.1 静态检查

最后一次执行：

```bash
PYTHONDONTWRITEBYTECODE=1 ./run_tests.sh -p no:cacheprovider
```

结果：`36 passed, 53 warnings`。相关源码还通过了 `git diff --check`。

### 4.2 GPU 场景构建

面板场景和三个 articulation 可在 Newton/MJWarp 下完成构建。一次短构建中观测到约 412～413 个 Newton shapes、22～23 个 MJWarp geometries；添加拨杆球形握把后 geometry 数增加 1，符合预期。

### 4.3 旋钮单控件

命令：

```bash
./run.sh --task panel_operation --panel-sequence knob \
  --vibration off --episode-s 16 --physics-profile training \
  --metrics-output /tmp/shakebench_panel_knob_physical_v3.json
```

结果：

- `success=true`
- 真实旋钮关节进度：1.0
- `move_timeout=false`
- 最大穿透：0.3168 mm
- 超过 0.5 mm 的帧：0
- 最大接触：右指/旋钮约 1813 N

旋钮的任务逻辑和运动链路已经跑通，但峰值力明显不可信或过硬，不能视为力学标定完成。

## 5. 未完成部分

### 5.1 按钮：首要阻塞项

最新隔离运行中，右指成功接触按钮，状态机也由 `move_button` 进入 `operate_button`，但按钮关节进度始终为 0，最终 `operation_timeout=true`。

最新指标：

- `button_progress=0.0`
- 右指/按钮峰值约 2426.6 N
- 最大穿透约 7.855 mm
- 最大穿透对象最终包含 `panda_hand` 与按钮碰撞体

这表明控制器已发出向内轨迹，但 prismatic joint 没有沿预期方向释放位移。下一位开发者应首先检查 joint frame、轴向符号、上下限和 drive 是否在 Newton 转换后与预期一致，不能继续单纯增加按压距离。

建议最小诊断：

1. 每个物理步打印原始 `button_q/button_qd`，不要只看裁剪后的 progress。
2. 暂时去掉按钮 drive，仅对运动链接施加小的正/负轴向力，分别验证哪一方向可动。
3. 在 USD/Newton builder 输出中检查 prismatic axis、local rotation 和最终限位。
4. 加入穿透提前终止；当前 7.855 mm 已说明继续加压无意义。
5. 轴向确认后再恢复弹簧，并从低刚度、低最大力逐步增加。

### 5.2 拨杆：物理模型可动，基线抓取未稳定

拨杆已产生左右指接触并推动真实关节。多次迭代中最好约达到目标角的 6.2%，随后接触丢失；最新降速版本约 2.8% 后丢失。

当前问题不是“关节不存在”，而是夹爪—球形握把—手掌之间的几何和轨迹不匹配：

- 夹爪圆弧目标领先于被动拨杆，握把滑出。
- 手部圆弧半径过小时，手掌会撞入握把/面板；过大时只有指尖边缘短暂接触。
- `contact_loss_timeout_s=0.30` 较严格，但仅放宽超时不会自动产生正确扭矩。

建议下一步：

1. 用阻抗/速度控制跟随真实拨杆角，而不是按时间直接走完理想角。
2. 目标角采用“当前拨杆角 + 小领先量”的闭环轨迹，限制最大领先角。
3. 单独优化腕部姿态，让两指长边与拨杆旋转平面匹配。
4. 持续监测 `panda_hand` 与握把的穿透，不能只看 finger contact sensor。
5. 在稳定完成后再调接触丢失超时。

### 5.3 尚未进行成功验收的项目

- `knob,lever,button` 三控件完整序列。
- 其他排列和随机 1～3 控件子集。
- seeds `0/17/31/47/73` 的确定性回归。
- vibration spectral 条件下的成功率。
- official 1000 Hz 外层频率验收。
- 成功三控件录像和最终演示说明。
- 参考策略之外的学习策略/VLA 策略评估。

## 6. 实现中遇到的问题与踩坑

### 6.1 Newton 多 articulation 与 wrist wrench sensor 不兼容

三个控件变成 articulation 后，wrist joint-wrench sensor 在初始化时出现类似错误：

```text
RuntimeError: joint_child contains out-of-range body indices for '/World/envs/env_.*/Robot'
```

现象表明传感器把场景全局 body index 当成 robot-local index。当前处理是在 `panel_operation` 中将 `scene.wrist_wrench=None`，面板评分改用六路 link-filtered 接触传感器。`pick_place` 的 wrist sensor 路径未改。

后续若修复该传感器，必须先确认 IsaacLab/Newton 版本是否已解决多 articulation 索引问题，不要直接恢复配置。

### 6.2 早期 visual-only / 脚本进度方案不能作为最终任务

早期为了先跑通场景，三个控件曾关闭碰撞，通过 proximity 门控推进视觉状态。这只能验证 UI 和路径，不能证明机器人操作了物体。

最终已替换为真实关节状态；`request_panel_progress()` 现在故意报错。旧交接记录里出现的“visual-only”“reach-and-hold”属于历史方案，不是当前实现。

### 6.3 DLS IK 会输出未展开的大角度并换肘分支

面板前方姿态曾让 Panda joint1 目标达到约 `-5.13 rad`，超过约 `[-2.897, 2.897]` 的机械限位；某些姿态切换还会使 link5/link6/hand 扫入桌面。

有效处理：

- pre 和 approach 两段路点；
- 始终保持 settle 手姿态；
- IK 结果裁剪到关节限位；
- 关节空间和笛卡尔空间双重限速；
- 用实测 pose 而不是 command pose 切换阶段。

### 6.4 相机支架碰撞会污染操作

腕部相机安装支架曾与旋钮发生约 29 mm 穿透。面板任务关闭了相机组件碰撞，只保留视觉模型；否则手指是否正确接触无法与相机支架碰撞区分。

### 6.5 控件与面板初始重叠会造成冻结或伪接触

早期 knob/button 与面板存在 2～6 mm 初始重叠，episode start 带持续 contact。与 `configure_mujoco_contact_solref()`、长时间保持目标和突然的大 IK 目标组合时，机械臂出现目标已写入但关节不响应的现象。

处理包括消除初始几何重叠，并把 contact solref 配置放到 task reset 之后。以后调整控件 standoff 时要先做 episode-start 穿透检查。

### 6.6 任务目标不能等于硬限位

旋钮最初把 72° 同时作为成功目标和硬上限。手指到达成功帧时会继续把关节加载到硬限位，产生很高的瞬时接触力。

当前旋钮和拨杆的硬限位均为目标的 110%，按钮行程也保留 15% 余量；控制器在 95% 完成后撤离。这个改动解决了旋钮超时，但峰值力仍高，需要继续做阻抗/材料标定。

### 6.7 降低速度后必须同步调整超时

旋钮操作速度降到 0.05 m/s 后，原 4 s move timeout 在刚建立接触时就触发，真实进度只有约 37%。将 move timeout 调到 7 s 后，旋钮可以达到目标并回撤。

### 6.8 拨杆抓取的几次失败模式

- 每指 13 mm 时，夹爪内间距约 26 mm，而杆体直径约 22 mm，留下约 4 mm 空气间隙，全程零接触。
- 改到每指 10 mm 后建立了真实接触，但直线扫动只推动约 2%，随后滑脱。
- 把抓取点降到杆身中段后，手掌直接撞入拨杆，最大穿透约 22.6 mm；该方案已否决。
- 恢复手掌净空并走圆弧后，穿透回到亚毫米量级，但视觉握把没有物理 collider，指尖仍从圆柱端部滑脱。
- 增加球形握把 collider 后，能稳定地产生握把接触和小幅关节运动，但时间驱动圆弧仍会领先于拨杆。

这说明拨杆后续重点应是闭环接触控制，而不是继续反复修改静态目标点。

### 6.9 按钮状态机曾被不可达目标卡住

按钮 `move` 阶段最初要求机械臂到达位于按钮碰撞体之后的目标点；机器人已接触按钮却不能满足位置误差，因此从未进入按压阶段，最终 `move_timeout`。

现已改成按钮一旦检测到正确指接触即可进入 `operate`。该修复暴露出更底层的问题：即使进入 operate，prismatic joint 仍不移动。

### 6.10 NATIVECCD/MULTICCD 会忽略 authored contact margin

运行日志明确提示，MuJoCo contact pipeline 在 `NATIVECCD/MULTICCD` 下会把 authored 1 mm margin 清零。不要把配置文件里的 margin 当作运行时已生效；解释接触和穿透结果时应以实际 solver 日志与 penetration probe 为准。

### 6.11 警告与证据管理

- 三个控件根刚体没有 collider，Newton 会提示 inertia tensor 无效并使用小球近似惯量；虽然能运行，但应补显式 inertia 或禁用碰撞的小代理几何。
- GPU training 回合耗时数分钟，且终端长时间不输出不等于 configure 卡死；应区分“进程仍在物理步进”和“构建阶段冻结”。
- `/tmp` 中的 metrics 只是临时证据，重跑前应复制到项目 `out/` 或独立工件目录。
- 静态测试通过、场景能构建、控件发生接触、任务 `success=true` 是四个不同层次，不能互相替代。

## 7. 推荐接手顺序

1. 保留当前旋钮成功基线，不先改公共 IK 和场景布局。
2. 隔离按钮，验证原始 prismatic `q/qd`、轴向和限位；修复后要求穿透小于 0.5 mm 再算完成。
3. 隔离拨杆，改为基于真实杆角的小领先量闭环圆弧控制。
4. 三个单控件都通过后，按 `knob,lever,button` 跑无振动 training 序列。
5. 再覆盖其他显式排列和随机 seeds。
6. 最后运行 official 频率、spectral vibration、录像与评估。
7. 全部验证结果保存到持久目录，并在文档中记录命令、提交号和 metrics 文件。

建议命令：

```bash
cd /home/miracle04/Desktop/ShakeBench

PYTHONDONTWRITEBYTECODE=1 ./run_tests.sh -p no:cacheprovider

./run.sh --task panel_operation --panel-sequence button \
  --vibration off --physics-profile training --episode-s 16 \
  --metrics-output out/panel_button.json

./run.sh --task panel_operation --panel-sequence lever \
  --vibration off --physics-profile training --episode-s 18 \
  --metrics-output out/panel_lever.json

./run.sh --task panel_operation --panel-sequence knob,lever,button \
  --vibration off --physics-profile training --episode-s 45 \
  --metrics-output out/panel_all.json
```

## 8. 关键文件索引

| 文件 | 职责 |
|---|---|
| `src/shakebench/config.py` | 面板参数、随机顺序、配置校验 |
| `src/shakebench/panel.py` | 斜面布局和控件 pivot 解析 |
| `src/shakebench/panel_controls.py` | 三套物理 articulation、关节、碰撞和运动视觉 |
| `src/shakebench/visual_assets.py` | 控制台固定外观 |
| `src/shakebench/scene.py` | 场景资产、actuator、六路接触传感器 |
| `src/shakebench/panel_task.py` | 真实关节状态、观测、顺序判定和 metrics |
| `src/shakebench/panel_controller.py` | DLS IK 参考控制器和操作轨迹 |
| `src/shakebench/cli.py` | CLI、运行循环和 metrics JSON |
| `src/shakebench/recording.py` | 面板录像 overlay |
| `src/shakebench/diagnostics.py` | 穿透检测与语义分类 |
| `configs/scenarios.yaml` | 面板场景预设 |
| `configs/visual_manifest.yaml` | 外观特征清单 |

## 9. 完成标准建议

只有同时满足以下条件，才建议把控制面板任务标为完成：

- 三个单控件均由真实关节状态达到目标并成功回撤。
- 三控件完整序列至少一种显式排列成功。
- 随机子集/顺序在约定 seeds 上通过且可复现。
- 无错误顺序和非目标控件误接触。
- 每种操作最大穿透达到项目接受阈值，且不存在手掌/腕部相机参与的伪操作。
- 接触力经过合理标定，不再出现数千牛级峰值而无人解释。
- training 与 official 的行为差异得到记录。
- spectral vibration 条件完成至少一次成功运行。
- 保存 metrics、日志和录像；静态测试继续全绿。

## 10. 仓库状态说明

2026-08-17 已将不含 `docs/`、`tests/`、`tools/` 的源码快照强制发布到远程 `Blien2002/ShakeBench` 的 `main`，提交为 `7dd5abd`。因此本交接文档当前作为独立工件交付，不在该远程源码快照内。本地 `/home/miracle04/Desktop/ShakeBench` 仍保留文档、测试和未提交工作区，接手时不要用远程 checkout 覆盖本地未提交内容。
