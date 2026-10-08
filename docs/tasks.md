# 任务说明

## 共同设定

**场景。** 所有任务共用同一个场景：
- Panda 机械臂不带移动底座，固定在岸边的实体地基上；
- 工作台面装在六自由度隔振平台上，台面下方装有 IMU；
- 振动激励由每条状态记录的种子决定，可以复现。

**动作与时间。** 动作空间统一为 7 维归一化 OSC 增量：末端位置 3 维、旋转向量 3 维、夹爪 1 维。控制频率 20 Hz，物理步长 0.2 ms。

**任务模块。** 每个任务模块在被导入时注册自己，并提供：
- `default_state()`：一条固定的示例状态；
- 状态校验函数和状态资产加载函数。

**成功判定。** 成功判定在环境内部完成，一经达成即锁存。阶段性进度只写入 metrics，不作为观测提供给策略。

## pick_place

环境 `shakebench.environments.vibration_pick_place.VibrationPickPlace`。

从台面上抓取物体，放入目标箱。物体集合由 `shakebench.tasks.catalog` 定义，包括 can、mug、apple 等，物体取自 robosuite 与 RoboCasa 的资产。

状态资产：
- `shakebench_states_dev.json`：开发状态，用于调试，不作为正式结果；
- `shakebench_states_official.json`、`shakebench_states_knee.json`：保留评测状态集；
- `shakebench_task_states_official.json`、`shakebench_task_states_knee.json`：按四种物体位姿变体展开的版本。

## ring_on_peg

环境 `shakebench.environments.ring_on_peg.RingOnPeg`。

**任务。** 机械臂从台面随机位置依次抓取蓝色大环和黄色小环，按从大到小的顺序套到中央的圆头木杆上。

**成功条件。**
- 每个阶段要求：已放置的环位置正确、已松手并稳定 0.5 秒，且后续的环还没有放上。
- 两个环都完成后奖励为 1。
- 顺序放错时，可以把提前放上的环取下来纠正。

**几何尺寸。**
- 环的内/外半径（从大到小）分别为 30/55 mm 和 24/45 mm，厚 16 mm。
- 木杆为陡圆台，底/顶半径 18/9 mm，高 120 mm，顶端球头半径 12.5 mm，小于最小环的有效孔径。
- 底盘为实心圆盘，半径 62 mm，厚 12 mm。

**状态。**
- `shakebench_ring_on_peg_states.json` 含 20 个随机布局，记录双环与木杆的位置、各环的初始朝向，以及激励和 IMU 种子。
- 不传状态时，每次 reset 随机放置两个环；传入 `seed` 可以复现。

## wine_rack

环境 `shakebench.environments.wine_rack.WineRack`。

**任务。** 参考 RLBench/PerAct 的 place_wine_at_rack_location，保留 left、middle、right 三个目标槽位。瓶子初始竖立在台面上，需要横放进指定槽位。瓶型为 RoboCasa 的葡萄酒瓶 `wine_3` 或白色半透明塑料水瓶 `bottled_water_16`。

**成功条件。** 以下条件同时满足，并连续保持 0.5 秒：
- 瓶体的完整碰撞几何进入指定槽位；
- 瓶颈朝向酒架 +x 方向，瓶轴偏差不超过 15°；
- 前后两根托梁同时承重；
- 瓶子与机械臂脱离接触。

水瓶因瓶肩更宽，竖向上界放宽 15 mm，其余条件相同。放错槽位、悬空、竖放、反向或越界都不算成功。

**物理设定。** 酒架是自由刚体：质量 0.9 kg，与台面的滑动摩擦系数 0.30。所有位置判断都在随台面运动的酒架坐标系中进行。

**状态。**
- `shakebench_wine_rack_states.json` 含 30 条状态，两种瓶型各 15 条，记录酒架的位置和朝向。
- 不传状态时，每次 reset 在 x ∈ [0.035, 0.135] m、y ∈ [−0.08, 0.08] m、朝向 ±10° 的范围内随机放置酒架。

## stack_blocks

环境 `shakebench.environments.stack_blocks.StackBlocks`。

按顺序叠放三个木块。有两种接头变体，各有独立的状态资产：
- plain：普通接头，`shakebench_stack_blocks_plain_states.json`；
- tenon：带锥度的榫卯接头，`shakebench_stack_blocks_tenon_states.json`。

## push_t

环境 `shakebench.environments.push_t.PushT`。

三维版本的非抓取 Push-T：只能推动木质 T 形块，不能抓取或抬起，要把它推到台面上的目标贴花处。目标位姿有两个固定选项，由状态中的 `goal_id` 选择。

状态资产：`shakebench_push_t_states.json`。

## ring_on_rail

环境 `shakebench.environments.ring_on_rail.RingOnRail`。

单环放置任务，只放置一个蓝色环。

状态资产：`shakebench_ring_on_rail_states.json`。

## upright

环境 `shakebench.environments.upright.Upright`。

把倒在台面上的物体扶正。物体为杯子、葡萄酒瓶、盒装饮料或电钻之一。

状态资产：`shakebench_upright_states.json`，覆盖全部四种物体。

## panel_operation

环境 `shakebench.environments.panel_operation.PanelOperation`。

操作控制面板上的按钮、旋钮和拨杆。状态资产：
- `shakebench_panel_operation_states.json`：每条状态对应一次单控件操作；
- `shakebench_panel_operation_chain_states.json`：按顺序完成多个控件操作的评测链，仅用于评测。

当前控件资产版本为 `original_controls`：旋钮、拨杆和底座采用独立构造的解析网格。旋钮使用
38 个复合凸接触体；拨杆、固定螺母和肩部接触体与解析外观对应。旋钮/拨杆质量保留，显式质心与惯量按
外观体积归一化估算。关节、官方接触参数、控制参数与成功阈值保留。释放检查覆盖目标控件的全部接触体，
不能假定与旧版本接触等价。训练图像版本应与评测版本核对；详见 [assets.md](assets.md)。


Plain stacking success rule `ordered_released_supported_stable_v2` accepts staggered centres and yawed cubes when the blue/green/yellow order is physically supported, released, and stable. Projected cube footprints must overlap with positive area; edge-only contact does not qualify. Positive contact force is projected onto worktable-up so side contact alone does not count as support. Existing height, tilt, pose-drift windows, stage order, release clearance and hold durations remain in force. Tenon, tenon_tight and tenon_loose retain their existing condition rules. Plain policy context reports `align_tolerance_m: null` and the new success-rule identity. No historical result is rescored without the required contact and pose history.
