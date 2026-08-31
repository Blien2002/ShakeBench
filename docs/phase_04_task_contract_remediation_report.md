# Phase 04R：Task contract remediation

状态：**PASS；唯一 Phase 05 handoff = PASS**

本报告修复并关闭 Phase 04 的三个 task-contract blocker：Can collision/inertia authority、support/contact-loss semantics 和 runtime PNG reproducibility。工作树中 Phase 03/04 的既有修改均保留；没有 reset、没有新建目录、没有实现 IMU/V0–V3、privileged recorder 或 Oracle controller。

## 1. Collision-envelope authority

旧实现把 `CanObject.bottom_offset/top_offset` placement sites 当作 collision envelope，得到错误的 `h=0.100 m` 惯量。04R 将 authority 改为：在 canonical compiled Can model 中，对明确的 collision geom `can_g0` 提取 mesh vertices / primitive support points。提取算法固定为：

```text
algorithm version = shakebench.can_collision_envelope.compiled_support.v1
source geom list = [can_g0]
```

当前 canonical compiled values：

```text
support radius   = 0.02509177806572465 m
height           = 0.08000000550552341 m
lower support z  = -0.040297003330440104 m
upper support z  =  0.03970300217508332 m
source model SHA = c5332fb76e8b2f8c36fe10ac99e51f5e79c189e248d626dacd54cc983629e669
algorithm        = shakebench.can_collision_envelope.compiled_support.v1
```

canonical Can inertial is now derived only from that envelope's radius and height：

```text
mass             = 0.349 kg
COM              = [0, 0, 0] m
Ixx = Iyy        = 0.00024106572568945823 kg·m²
Izz              = 0.00010986473347417683 kg·m²
```

placement uses the same envelope's lower support z with one correction：

```text
placement z offset = -lower_support_z = 0.040297003330440104 m
```

Stock placement sites remain available for upstream object API compatibility, but no longer participate in inertia or placement correction. Construction and compiled audit both re-extract and compare source geom list, source hash, radius, height, lower/upper support, inertia and correction; any drift fails closed. The old site-derived `0.100 m` authority is explicitly invalidated.

## 2. Support semantics

`target_bottom_contact_present` is now only an instantaneous contact fact. It is not success support. Success derives `supported_by_target_bottom` from three independent values：

```text
target_bottom_contact_present = true
target-local/worktable-local upward support force > 0.001 N (strict)
lower_support_z_m >= -0.00050 m relative to target bottom plane
```

The force is computed by summing the positive local-z component of the Can force for contacts with the named target-bottom geom; world wrench/force records remain available separately. The lower support z comes from the same compiled collision support vertices expressed in the target-local frame.

State-injection regressions cover：

- static Can inside the target：bottom contact, positive support force and valid lower support pass；
- force exactly at `0.001 N`：fails；
- Can below the bottom plane：contact may remain present, but the height gate fails；
- side/wall scrape without valid bottom support：support fails；
- wall-straddling：target wall contact can be present while containment fails；
- Can above the shallow wall：height is not a success condition, so geometric containment remains independently valid。

The existing containment, release, relative-speed, angular-speed, penetration and continuous `0.50 s` latch gates remain independent。

## 3. Contact-loss schema

There is no ambiguous `contact_loss` field in a metrics report. The schema is：

```text
instantaneous: finger_can_contact_present
stateful:     finger_contact_loss_after_grasp
```

The state machine is reset at episode reset；before any grasp, no finger contact means `finger_contact_loss_after_grasp=false`；during a prior grasp it remains false while contact is present；the first sample after loss becomes true. Reports still contain first-slip, in-hand translation/rotation slip and interface-resolved contact force/wrench/impulse/penetration。

`tests/test_shakebench_metrics.py` and the environment integration test cover reset, never-grasped, first-contact, sustained-contact, release and post-release semantics. The artifact records `support_contact_schema_version=2` and the two exact field names above。

## 4. Policy-frame boundary

Phase 04R does not implement tiers. It closes the Phase 05 common-state seam by making `use_object_obs=True` expose only：

```text
can_pos_robot_base
can_quat_robot_base
can_to_target_pos       # target-local region-relative position
```

World-frame `can_pos` / `can_quat` are no longer emitted. The report and artifact explicitly mark the common Can state frame as `robot_base`；no IMU, future program, realized support-state privilege or `privileged_` field was added。

## 5. Runtime asset and artifact lock

The actual MJCF reference is：

```text
robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png
SHA-256 = 87a5478e7325b7d79fc8073afefd3ac7c44b44d6c804ecb586c1641d8157e162
```

`.gitignore` now contains the minimal PNG exception, and the PNG is staged in the Git index；the required command succeeds：

```bash
git ls-files --error-unmatch \
  robosuite/models/assets/textures/shakebench_phenolic_bench_dark_1k.png
```

`MANIFEST.in` already recursively includes `robosuite/models/assets/`, so no broad packaging rule was added. The source-distribution smoke builds a temporary sdist, extracts it, verifies the PNG member, verifies that `shakebench_arena.xml` references that PNG, and compiles the extracted XML with MuJoCo。

`tests/shakebench_phase_04_environment.json` is schema version 2 and records the envelope authority, support/contact schema, policy frame, runtime hash and clean-compile result. Its lock is read-only by default：

```text
previous payload SHA-256 = 5a964a49d4918d0d7ada074c5c5a105160e671b547ca1387e4c1662dbabbf0cf
new payload SHA-256      = 0e16fb2bc4b69bcecfc09613bea1853c60d5e8fce29d1d16c72b1009feadddb0
update reason            = explicit Phase 04R contract remediation
```

The verifier checks schema, lock integrity, envelope/inertia/placement authority, support/contact names, policy frame, runtime file hash and absence of the ambiguous `contact_loss` key；it never writes the artifact. Mutating the payload without recomputing the lock fails verification。

## 6. Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_metrics.py \
  tests/test_environments/test_vibration_pick_place_can.py \
  tests/test_shakebench_config.py --tb=short
```

Result after 04R：`38 passed`。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_shakebench_arena.py \
  tests/test_shakebench_isolator.py \
  tests/test_shakebench_transfer_remediation.py --tb=short
```

Result：`22 passed`。

Additional checks passed：source-distribution clean compile、`git ls-files` runtime asset check、`py_compile`、`pyflakes`、`flake8`、`isort`、JSON parsing、read-only artifact verification and artifact mutation rejection。The existing no-EGL upstream regression remains `255 passed, 58 skipped`；renderer-dependent all-environment/camera smoke remains host-blocked by unavailable EGL/swrast and is not reported as PASS。

## 7. Completion decision

All four 04R gates are closed：one compiled collision-envelope authority drives inertia and placement, target-bottom support requires force and height evidence, contact-loss has one unambiguous schema, and the actual runtime PNG is Git/package/clean-compile verifiable. **Phase 05 handoff = PASS.**
