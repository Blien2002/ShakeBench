# Phase 05R-W 修复提示词：移除 production fake-Gym，保留真实 wrapper 隔离验证

你在 `/home/miracle04/Desktop/ShakeBench` 工作。Phase 05R 的科学合同已通过；当前只修复一个上游兼容性回归：`robosuite/wrappers/gym_wrapper.py` 在 Gym/Gymnasium 缺失或 Gym 版本过低时静默安装 `_FallbackGym/_FallbackSpaces`。该 fallback 只能属于测试，不得成为生产 API。

本阶段不修改 IMU 数值、V0–V3 key-set、providers、privileged schema、Phase 02–04 physics，也不实现 Phase 06。

## 目标生产行为

恢复并测试原有依赖语义：

1. 优先导入 `gymnasium`；
2. gymnasium 不存在时允许导入 `gym>=0.26`；
3. 两者均不存在时，导入/使用 `GymWrapper` 明确抛出依赖错误；
4. 仅安装旧 `gym<0.26` 时继续抛出版本错误，不能被外层 `except ImportError` 吞掉后切换到 fallback；
5. 删除 production `_FallbackEnv/_FallbackBox/_FallbackDict/_FallbackText/_FallbackSpaces/_FallbackGym`；发布 wheel/sdist 中不得包含这些符号；
6. 保留 Phase 05R 对 explicit State tier 的 exact policy allowlist、string/bool spaces、privileged/unavailable key fail-closed 行为。

不要通过扩大 `except Exception`、静默返回普通 dict，或伪造 `gym.Env` inheritance 使测试变绿。

## Test-only dependency harness

当前 workspace 没有 Gymnasium/Gym，测试仍须不 skip。把 shim 完全放在 tests 内，并在隔离 subprocess 中运行，避免污染主进程的 `sys.modules` 或 production package。

推荐流程：

1. test 在 `tmp_path` 创建最小 `gymnasium` test package，只实现 GymWrapper 实际调用的 `Env`、`spaces.Box/Dict/MultiBinary/Text`；
2. subprocess 的 `PYTHONPATH` 将该 test package 放在 repo 前，并导入仓库真实 `robosuite.wrappers.GymWrapper`；
3. 用真实 `VibrationPickPlaceCan` V0 和 V3 分别 reset/step；
4. 验证 returned keys严格等于 `env.policy_observation_keys`，`observation_space.contains(observation)` 为真，action space包含合法 action；
5. 显式请求 `privileged_*`、`robot0_proprio-state`、debug/renderer/aggregate keys 均 fail closed；
6. callback recorder收到 truth，但 wrapper observation不含任何 privileged key；
7. 另开无 shim subprocess，证明缺依赖时 production import/constructor明确失败；
8. 再开仅有 `gym.__version__ < 0.26` 的 shim subprocess，证明旧版本错误不会被 fallback吞掉。

test shim 文件只能位于临时目录或现有 tests 文件内，不得新增 production module或被 MANIFEST/package-data 收录。

## 回归与报告

- 复跑所有现有 robosuite wrapper tests，确认有真实 Gymnasium 时的行为不变；
- 复跑 Phase 05R privilege/provider/sensor suites，无 skip；
- 更新 `docs/phase_05_observation_imu_remediation_report.md`：声明 wrapper验证使用 test-only isolated harness，而非 production fallback；
- `docs/phase_05_report.md` 与 artifact provenance不得声称 bundled fallback；
- `git diff` 只能包含 wrapper 的 State-tier allowlist/string support等必要变化，不能保留 fake Gym class。

至少运行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_shakebench_privilege.py \
  tests/test_shakebench_providers.py \
  tests/test_shakebench_sensors.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --tb=short \
  tests/test_shakebench_privilege.py \
  tests/test_environments/test_action_playback.py
```

完成定义：production GymWrapper 对缺失/过旧依赖恢复明确失败；测试 shim不进入 production/package；真实 wrapper代码路径在 test-only Gym API下通过 exact key/space/privilege tests；Phase 05R 28 项及相关回归无 skip。完成后可正式进入 Phase 06。
