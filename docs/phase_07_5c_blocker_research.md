# Phase 7.5C local blocker research

Date: 2026-09-07

Scope: two local-execution blockers only: the `mjviewer` / GLFW test lane and the
fully occupied Linux swap device. This note distinguishes infrastructure evidence
from ShakeBench science evidence; neither workaround authorizes Phase 8 by itself.

## Executive finding

1. The renderer failure is not evidence of a MuJoCo physics or ShakeBench scene
   failure. `mjviewer` is an **interactive GLFW window** while the already working
   video path is an **offscreen EGL context**. Those are different context and
   lifecycle paths. On this host, the on-screen precondition is falsely accepted
   merely because `DISPLAY` / `WAYLAND_DISPLAY` exists, then GLFW cannot create a
   window because Mesa's driver dependency resolves against an incompatible
   Anaconda `libstdc++.so.6`. After that is corrected, the test itself passes but
   the process exits with signal 11 because the environment does not explicitly
   close the passive viewer. Both defects have locally reproducible remedies.
2. `SwapUsed == SwapTotal` together with available RAM does not prove active
   thrashing. Linux defines `MemAvailable` as the RAM estimate available to start
   applications without swapping, and swapped-out cold pages can remain in swap
   after the pressure that placed them there. The local spot check showed zero
   `si` / `so` and zero recent memory PSI, so the machine was not thrashing at that
   instant. It is nevertheless an unsuitable starting state for an official
   resource-sensitive 1/2/4/8-worker comparison: there is no swap headroom and
   the retained residency state is neither normalized nor guaranteed equal across
   runs. Clear and re-enable the specific swap device before the official matrix,
   or keep the affected result `resource_limited`.

## 1. Interactive viewer / GLFW blocker

### 1.1 Why EGL success and `mjviewer` failure can coexist

MuJoCo documents `mujoco.viewer.launch_passive` as an interactive GUI and requires
the user to call `sync`; the returned handle has an explicit, thread-safe `close`
method. In contrast, `mujoco.GLContext` is provided for offscreen rendering and
must be made current on the rendering thread. These are separate public APIs, not
two names for the same backend ([MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html#interactive-viewer),
[offscreen context documentation](https://mujoco.readthedocs.io/en/stable/python.html#rendering)).

The source makes that separation concrete. MuJoCo's interactive viewer calls
`glfw.init()` and fails if GLFW cannot initialize, whereas `MUJOCO_GL=egl` selects
the EGL implementation of the offscreen `GLContext`; it does not replace the
interactive viewer's GLFW window ([MuJoCo viewer source](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/viewer.py),
[MuJoCo GL-context selector](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/gl_context.py)).
Robosuite's `mjviewer` adapter directly calls `viewer.launch_passive(...)` and
later `sync()` ([robosuite viewer adapter](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/renderers/viewer/mjviewer_renderer.py)).
If native Simulate cannot create its GLFW window, its adapter calls `mju_error`
([MuJoCo GLFW adapter source](https://github.com/google-deepmind/mujoco/blob/main/simulate/glfw_adapter.cc#L49-L61));
MuJoCo specifies that `mju_error` immediately terminates the process
([MuJoCo error semantics](https://mujoco.readthedocs.io/en/stable/python.html#error-handling)).
That explains why this failure can remove the entire pytest process instead of
arriving as an ordinary Python assertion.
Therefore the successful native EGL video is valid evidence for offscreen video,
but cannot satisfy an interactive-window test.

GLFW also distinguishes the selected *window system* (Wayland, X11, or the
explicit-only Null platform) from the context creation API. Its default platform
selection excludes Null, so an interactive test still needs a usable Wayland/X11
connection and a working OpenGL driver even when EGL is present
([GLFW initialization guide](https://www.glfw.org/docs/latest/intro_guide.html#platform),
[GLFW context guide](https://www.glfw.org/docs/latest/context_guide.html)).

### 1.2 Local evidence and actual failure chain

Read-only probes on this host established the following:

- Python loads `glfw 2.10.2`, whose initially selected bundled runtime reports
  `3.4.0 Wayland Null EGL OSMesa shared`. Both `DISPLAY=:0` and
  `WAYLAND_DISPLAY=wayland-0` are set.
- `glfw.init()` succeeds, but even a 64x64 invisible `glfw.create_window` returns
  null. Mesa reports that it cannot load `iris_dri.so` or `swrast_dri.so`.
- Loading `iris_dri.so` directly identifies the dependency error: the Anaconda
  `libstdc++.so.6` lacks `GLIBCXX_3.4.30`, required by the system LLVM/Mesa stack.
  GCC's ABI table confirms that `GLIBCXX_3.4.30` was introduced by GCC 12.1
  ([GCC libstdc++ ABI policy](https://gcc.gnu.org/onlinedocs/libstdc%2B%2B/manual/abi.html)).
- With the compatible system C++ runtime preloaded and pyGLFW's X11 variant
  selected, GLFW window creation succeeds. pyGLFW officially supports choosing
  `x11` or `wayland` using `PYGLFW_LIBRARY_VARIANT`
  ([pyGLFW README](https://github.com/FlorianRhiem/pyGLFW#linux)).
- Under that corrected window runtime, the exact `test_mjviewer_renderer` body
  reports PASS and the enclosing Python process then exits 139. The equivalent
  body wrapped in `try/finally: env.close()` exits 0. This matches robosuite's
  adapter lifecycle: `close()` closes the passive-viewer handle, whereas the
  current upstream test does not close its environment
  ([upstream renderer test](https://github.com/ARISE-Initiative/robosuite/blob/master/tests/test_renderers/test_all_renderers.py),
  [robosuite viewer adapter](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/renderers/viewer/mjviewer_renderer.py)).

The existing test's `is_display_available()` only checks whether either variable
name exists. It does not prove the socket is reachable, that the loaded GLFW build
supports that window system, or that a context-bearing window can be created.
This is why its collection-time skip condition does not protect this host.

### 1.3 Correct remediation and CI classification

Recommended order for this repository:

1. **Fix the runtime coherently.** Prefer a clean Python environment whose Mesa,
   LLVM and `libstdc++` dependencies come from a compatible stack; for the current
   Conda environment that means upgrading `libstdcxx-ng` / `libgcc-ng` to a
   mutually compatible locked set exporting `GLIBCXX_3.4.30` or newer. Do not
   symlink a system C++ library into Conda. Treat
   `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` plus
   `PYGLFW_LIBRARY_VARIANT=x11` as a narrowly scoped diagnostic or recorded test
   launcher, not an implicit global fix. If it is used for final evidence, record
   the exact Python, MuJoCo, pyGLFW, loaded GLFW and loaded `libstdc++` paths and
   versions. X11 also requires an actually reachable X server (the desktop's Xwayland
   or a deliberately provisioned Xvfb).
2. **Make teardown part of the test.** Enclose environment use in `try/finally`
   and always call `env.close()`. The pass criterion is both the assertions passing
   **and the child process exiting zero**. A post-PASS signal 11 is a test failure,
   not a cosmetic warning. Keep pytest's fault handler enabled; pytest documents
   that it emits Python tracebacks on segfaults and timeouts
   ([pytest failure diagnostics](https://docs.pytest.org/en/stable/how-to/failures.html#fault-handler)).
3. **Run the interactive lane in a subprocess.** Native GUI threads and drivers
   can terminate or corrupt the interpreter during shutdown; subprocess isolation
   lets the suite retain an unambiguous exit status and preserves the remainder of
   the test report. Use a bounded timeout and retain stdout/stderr and return code.
4. **Use a capability probe, not environment-variable presence.** Before the
   interactive lane, run a short child that calls `glfw.init()`, records the
   selected platform and version string, creates an invisible OpenGL window,
   destroys it, calls `glfw.terminate()`, and exits nonzero on any failure.
   GLFW specifies that initialization and termination belong on the main thread
   ([GLFW API reference](https://www.glfw.org/docs/latest/group__init.html#ga317aac130a235ab08c6db0834907d85e)).
   In a GUI-enabled lane, a failed probe is a renderer **failure**, not a reason
   to skip; otherwise this host's real ABI/driver regression would be hidden.
5. **Keep three outcomes separate:** non-renderer regression, offscreen EGL, and
   interactive `mjviewer`. On a genuinely headless runner, skipping the interactive
   test with a precise reason is legitimate—pytest defines skip for a test whose
   external prerequisite is unavailable—but the Phase 7.5C report must then say
   `BLOCKED_BY_MJVIEWER_RUNTIME`, not “interactive renderer PASS.” Do not use
   `xfail` for a crash or missing display: pytest defines xfail for an expected
   product failure, not an unavailable external resource
   ([pytest skip/xfail semantics](https://docs.pytest.org/en/stable/how-to/skipping.html)).

This approach neither changes ShakeBench traces nor converts the already valid EGL
video into a different kind of evidence.

## 2. Full swap with available RAM

### 2.1 What the counters mean

At the time of this investigation the host reported approximately 31 GiB RAM,
12.7--12.9 GiB `MemAvailable`, and a 2 GiB `/swapfile` with only tens of KiB free.
A short `vmstat 1` sample showed `si=0`, `so=0`; `/proc/pressure/memory` showed
`avg10=avg60=avg300=0.00` for both `some` and `full`.

Linux defines `MemAvailable` as an estimate of memory available for new
applications *without swapping*. It separately defines `SwapCached` as pages that
have been read back into RAM but still have a copy in swap
([proc_meminfo(5)](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html)).
Consequently, a large static swap allocation can be residue from earlier pressure;
it is active `si` / `so` and stall pressure that demonstrate current interference.
`vmstat` defines `si` and `so` as swap-in and swap-out per second, and notes that
its first row is an average since boot, so only subsequent interval rows are useful
for a timed run ([vmstat(8)](https://man7.org/linux/man-pages/man8/vmstat.8.html)).

PSI supplies the missing latency signal: `memory some` is time with at least one
task stalled and `memory full` is time when all non-idle tasks are stalled; sustained
`full` indicates thrashing and wasted CPU cycles. Its cumulative `total` field makes
before/after deltas suitable for a benchmark interval
([Linux PSI documentation](https://docs.kernel.org/accounting/psi.html)).

Thus the current spot check says **no observed active thrash**, not **clean official
benchmark state**. With `SwapFree` effectively zero, a later allocation burst has
no swap safety margin, and worker-count comparisons can inherit different page-in,
reclaim or OOM behavior.

### 2.2 Safe normalization procedure

Do this only between runs, with all ShakeBench workers stopped:

1. Capture `/proc/meminfo`, `swapon --show --bytes`, `/proc/vmstat`,
   `/proc/pressure/{memory,io,cpu}`, and, when under cgroup v2,
   `memory.current`, `memory.swap.current`, `memory.events` and
   `memory.swap.events`.
2. Confirm that `MemAvailable` comfortably exceeds current swap use **plus** the
   expected benchmark working set and a safety reserve. This is a conservative
   operational check, not a kernel guarantee.
3. With administrative privileges, cycle the explicit known target rather than a
   broad wildcard:

   ```bash
   sudo swapoff /swapfile
   sudo swapon /swapfile
   ```

   `swapoff` brings swapped pages back to memory and can fail when memory is
   insufficient; util-linux documents a distinct insufficient-memory exit status.
   Check the exit status before `swapon` and do not continue the benchmark after a
   failed transition ([swapoff(8)](https://man7.org/linux/man-pages/man8/swapoff.8.html),
   [swapoff(2)](https://man7.org/linux/man-pages/man2/swapoff.2.html)). A reboot is
   the clean fallback when the host cannot be safely quiesced or the cycle fails.
4. Verify that the swap device is re-enabled, swap use is near zero, no OOM event
   occurred, and the intended worker cpuset is still valid. Allow the machine to
   settle, then perform the same untimed warm-up for every worker count.
5. Do **not** use `drop_caches` as a substitute for clearing swap. The kernel says
   caches are reclaimed automatically and warns that forced dropping can add
   substantial I/O and CPU cost, which would itself contaminate throughput
   measurements ([Linux VM sysctl documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html#drop-caches)).

### 2.3 Acceptance rule for the CPU matrix

For every 1/2/4/8-worker point, save pre/post values and interval samples. Baseline
and candidate must start from the same normalized policy and receive the same
warm-up. In addition to the Phase 7.5B latency/RSS/frequency/temperature fields,
record deltas for:

- `pswpin`, `pswpout`, `pgmajfault`, all available `allocstall*`, and `oom_kill`
  from `/proc/vmstat`;
- `some.total` and `full.total` from memory and I/O PSI;
- `SwapFree`, `MemAvailable`, and per-worker or cgroup peak memory;
- cgroup v2 `memory.events` (`high`, `max`, `oom`, `oom_kill`) and
  `memory.swap.events` where available. The kernel documents these as reclaim,
  limit, OOM and swap-allocation-failure counters
  ([cgroup v2 memory interface](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)).

Apply the following fail-closed interpretation:

- Any swap-in/out during the timed region, sustained positive memory/I/O PSI,
  allocation stalls attributable to the run, OOM event, swap allocation failure,
  or thermal throttling makes that worker point `resource_limited`.
- A full swap device with zero interval activity may be used for a **diagnostic**
  run, but not to select or authorize the official worker count. If swap cannot be
  normalized, retain raw outputs and mark the matrix
  `BLOCKED_BY_UNNORMALIZED_SWAP_STATE` / `resource_limited`.
- A normalized run is still not a performance PASS unless the requested full
  baseline/candidate semantic parity also passes. Memory cleanup cannot compensate
  for missing reset, observation, action, actuator, trace, metrics, success-window,
  termination, or determinism parity.

## 3. Minimal next execution sequence

1. Establish a compatible on-screen GLFW runtime in a dedicated launcher or clean
   environment; save the active window probe.
2. Add explicit `env.close()` teardown and run only the `mjviewer` test in a child;
   require return code 0. Then run the complete suite and separately report all
   renderer skips/failures.
3. Quiesce the host, cycle `/swapfile` safely, verify counters, and save a preflight
   snapshot.
4. Run CPU parity and the atomic/resumable 1/2/4/8 matrix while sampling VM/PSI and
   thermal counters. Reject only the affected worker point as `resource_limited`,
   but keep Phase 8 unauthorized until every Phase 7.5C hard gate passes.
