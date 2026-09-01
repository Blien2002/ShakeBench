# Phase 05R 修复提示词：闭合 State key-set、robot-base IMU、连续时间链与 privilege immutability

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 05 已实现 V0–V3、IMU 与 recorder 骨架，但当前 policy key-set、IMU physical mounting、prefill timeline、16-bit encoding 和 privileged defensive-copy contract 仍不符合规范。本阶段只修复 Phase 05；保留工作树，不重置，不新建目录，不进入 Phase 06 physics freeze，也不实现 Oracle controller。

leading word 是 **边界**：policy/privileged、deck/base、acquisition/delivery、signed code range 和 current/future 每一侧都必须有唯一、可测试的定义；subset、alias 或内部自洽不能替代边界证明。

## 必读与范围

完整读取：

```text
docs/shakebench_prompts/README.md
docs/shakebench_prompts/phase_05_state_tiers_imu.md
docs/robosuite_benchmark_design_tree.md       # Observation / V0–V3 / IMU / privilege
docs/robosuite_benchmark_v0_spec.md
docs/research/canonical_imu_profile_validation.md
docs/phase_05_report.md
robosuite/environments/manipulation/vibration_pick_place_can.py
robosuite/utils/shakebench_sensors.py
robosuite/utils/shakebench_providers.py
robosuite/utils/shakebench_privilege.py
tests/test_shakebench_sensors.py
tests/test_shakebench_providers.py
tests/test_shakebench_privilege.py
```

允许修改 Phase 05 environment integration、三份 utils、Phase 05 tests/docs 和按需的 test dependency metadata。不得修改已通过的 Phase 02 driver time contract、Phase 03 isolator physics、Phase 04 success semantics，亦不得实现 Phase 06/07。

## 1. 把 IMU 安装在真实 robot-base origin

canonical metadata 已冻结为 `parent=robot_base, position=[0,0,0], quaternion=identity`，实现必须匹配该物理位置。

- V1 直接从 compiled `robot0_base` body 的 world pose/spatial twist/spatial acceleration生成 clean IMU；不得继续从 deck origin 以 zero lever 采样；
- 或者显式、自动从 compiled model 派生 deck→robot-base extrinsics，再用 deck state与该 lever arm计算，但 metadata、provider和 recorder必须保存并审计 exact extrinsics。优先直接采 robot base，减少重复转换；
- robot base 刚接 deck 不等于两者原点相同。编译后断言 body names、relative pose/extrinsics和 sensor frame；
- 增加真实 MuJoCo rotational-deck regression：非零 `alpha` / `omega` 下，robot-base IMU 的 specific force必须包含由实际 offset产生的 `alpha×r` 和 `omega×(omega×r)`，并与独立刚体解析式一致；同时证明 deck-origin zero-lever结果不同；
- static +g、free fall、gyro frame和右极限 sample-time contract继续成立。

## 2. State tiers 必须返回 exact policy keys

显式 `observation_tier` 下，最终 `env.reset()` / `env.step()` 返回值必须严格满足：

```text
set(observation) == set(COMMON_STATE_KEYS + TIER_POLICY_KEYS[tier])
```

不能只断言声明 keys 是 returned observation 的子集。必须移除或在最终 policy boundary fail-close过滤 stock extras，包括但不限于：

```text
robot0_joint_acc
joint sin/cos
stock gripper qpos/qvel aliases
robot0_proprio-state / modality aggregates
world-frame EEF/object fields
```

保留的 common fields仍是 joint pos/vel、robot-base EEF、明确 gripper state、wrist/fingertip、robot-base Can和 robot-base goal region。V0 不得因 stock observables获得 joint acceleration或其它专用 vibration cue。

推荐在显式 State tier 的 `_get_observations()` 最终边界按 contract构建新 dict、缺 key/多 key 都抛错，并返回 defensive copies。legacy `observation_tier=None` 保持上游行为。

tests 必须分别对 reset、step、renderer/debug flags、wrapper path断言 exact equality和无 aggregate extras。

## 3. 冻结连续的 200 Hz acquisition / delivery timeline

定义 reset `t=0` 的 static-history语义：

```text
policy window at reset: delivered acquisitions [-0.050, -0.045, ..., -0.005] s
delivery queue at reset: static acquisition at 0.000 s
after first 20 Hz step: delivered window [0.000, 0.005, ..., 0.045] s
acquisition at 0.050 s remains queued for delivery at 0.055 s
```

每个相邻 acquisition timestamp严格差 `0.005 s`。不得出现 `[-0.005, 0.005]` 的 10 ms gap。

- 修正 prefill start和 delivery queue，不通过伪造 output timestamp隐藏 gap；
- live environment每个 20 Hz policy step必须正好产生 10 个 acquisition；model timestep必须整除5 ms；
- recorder保存 acquisition、delivery和 delivered-acquisition timestamps；policy只见 noisy delayed `[10,6]` window与 dt；
- hard/non-hard reset、episode timestamp回退、连续两个 policy steps都测试；
- oldest-to-newest、无重复/缺失 sample与一帧 delivery delay逐项断言。

## 4. 合法 signed 16-bit quantization

冻结编码为 signed two's-complement code range：

```text
code_min = -32768
code_max = +32767
LSB = 2*range / 65536
```

`+range` 不可产生 `+32768`；应量化/饱和到 `+32767`，`-range` 为 `-32768`。明确区分 physical clipping flag 与 quantizer endpoint saturation。所有 codes必须可无损转换为 `np.int16` 并 round-trip。

更新 profile metadata、step properties、clip/round logic和错误测试；删除当前期望 `+32768` 的断言。覆盖 zero、±0.5 LSB、两个 endpoint、超量程、单调性和 deterministic replay。

## 5. 冻结 sensor/privileged truth 的不可变性

- 将 process-global gravity/filter constants改为 immutable tuples或只读 storage；所有 constructor内部先 defensive copy，外部 mutation不得改变后来创建的 sensor；
- `IMUSample(frozen=True)` 的所有 arrays必须在**最终 copy 后**设置 `writeable=False`；`_DeliveryItem`、records、bias/filter trace同样不能暴露可写内部 storage；
- `RigidBodyState` / `SupportState` / provider payload继续返回 defensive arrays；
- `PrivilegedRecorder._copy_value` 在 `deepcopy` 失败时必须抛 `ShakeBenchPrivilegeError`，禁止返回原对象。可选择显式支持 JSON/NumPy/domain snapshots；unsupported mutable object fail closed；
- recorder records/latest/callback payload分别独立，任一调用方 mutation不得改变另外两份或内部历史；
- 为全局常量 mutation、IMUSample array mutation、uncopyable object、nested arrays/maps写 adversarial regressions。

## 6. 补全定量和隔离证据

### Noise/filter

- 用足够长的 seeded zero-input sequence，去除 filter transient后，验证 filtered accel/gyro RMS分别等于 `noise_density * sqrt(ENBW)`，使用运行前固定的统计 tolerance；
- bias diffusion std、40 Hz −3 dB、ENBW、clip和quantization分别独立验证；不能用只测 bias代替 noise RMS。

### V2/V3

- V2 live test证明 current-only：对象不保存 support history，不输出 seed/t0/frequency/program/future；同刚体 motion下 relative state为零；
- V3 test必须只读取公开 payload arrays和 `episode_time_s/ramp metadata`，由独立 reconstruction函数重建未来 `q/qdot/qdd`；禁止调用 provider持有的原 `program.evaluate()`作为“重建”；
- 对多个 future query times与独立 reference比较；确认 V0–V2完全没有 program fields，V3没有未来 realized/contact/outcome。

### Wrapper/debug

- GymWrapper测试不能作为 handoff 的 skip。可将 gymnasium加入 test extra/当前环境，或提供能导入并执行真实 GymWrapper代码路径的 dependency fixture；
- 另加不依赖可选 Gym的 adversarial wrapper，尝试请求 `privileged_*`、modality aggregate、debug/renderer字段，必须 fail closed；
- report只能在真实 wrapper lane通过后宣称 GymWrapper无泄漏。

## 7. 报告、artifact 与停止条件

新增受控 artifact `tests/shakebench_phase_05_observation.json`，记录：exact key sets、IMU extrinsics/profile/hash、timeline examples、quantizer endpoints、noise/filter statistics、V2/V3 reconstruction、privilege adversarial结果与 handoff。默认只读验证；更新需 reason和 old/new hash。

更新 `docs/phase_05_report.md`，新增 `docs/phase_05_observation_imu_remediation_report.md`，并同步清理 `docs/integration_map.md` 中“未实现/已实现”的冲突表述。

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_shakebench_sensors.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_privilege.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_environments/test_vibration_pick_place_can.py \
  tests/test_shakebench_metrics.py \
  tests/test_shakebench_deck_driver.py
```

Phase 06 handoff只有在以下全部满足时为 PASS：robot-base IMU物理位置正确；V0–V3 returned key-set exact；200 Hz timeline连续；int16 encoding合法；sensor/recorder truth不可变且copy fail closed；noise RMS、independent V3 reconstruction和真实 wrapper leakage tests通过。否则报告 `BLOCKED` 并停止在 Phase 05R。
