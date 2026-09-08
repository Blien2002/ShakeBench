# Phase 7.5B CPU throughput closure

Status: **PASS_WITH_2_WORKER_LIMIT**. No Phase 8 authorization is implied by this CPU result alone.

The host swap device was normalized to zero before testing. The retained candidate
`hard_reset=False` passed a full state000 / V0 / Gamma=0 / 1200-step episode
parity check against `hard_reset=True`: reset-derived state, observations, action
trace, actuator controls and forces, metrics, success window and termination were
identical. This is a diagnostic parity result only; it has not yet passed the
required fresh/reuse order matrix or three-process reuse determinism checks.

Each timing job used classic MuJoCo only, a frozen direct-mount science authority,
V0/Gamma=0, a full 1200-step horizon, spawn-independent process execution, and
one numerical-library thread. All raw outputs under
`out/phase07_5a_requalification/cpu/jobs/` passed the v5 semantic verifier.

| Workers / affinity | Jobs | Makespan s | Jobs/s | Sim-s/wall-s | P50/P95 s | Max RSS MiB | Efficiency | Resource result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 / [0] | 1 | 136.63 | 0.00732 | 0.0655 | 136.63 / 136.63 | 878.5 | 1.000 | diagnostic only |
| 2 / [0,2] | 2 | 141.48 | 0.01414 | 0.1265 | 139.27 / 141.48 | 900.0 | 0.966 | diagnostic only |
| 4 / [0,2,4,6] | 4 | 147.15 | 0.02718 | 0.2433 | 140.07 / 147.15 | 888.7 | 0.929 | **resource_limited**: swap grew 0 → 11 MiB |
| 8 / [0,2,4,6,8,10,12,14] | 8 | 149.44 | 0.05353 | 0.4791 | 144.94 / 149.44 | 890.6 | 0.914 | **resource_limited**: swap grew 11 MiB → 1.61 GiB |

No OOM was observed. A final formal run used
`robosuite.scripts.shakebench_cpu_batch` with spawn, immutable dispatch identity,
atomic per-job publication, resume/retry ledger, stable aggregation and before/after
memory, VM, frequency and temperature snapshots. In a controlled swap-disabled
window, 2 workers on `[0,2]` completed both full-horizon jobs with semantic PASS,
`jobs/s=0.0132762`, and zero swap growth. `/swapfile` was restored after the run.

The accepted operational policy is therefore exactly 2 workers. The 4/8 results
remain `resource_limited` negative evidence and cannot be selected automatically.
`hard_reset=False` remains disabled until the full fresh/reuse order matrix and
three-process reuse determinism are completed; no unverified optimization is retained.
