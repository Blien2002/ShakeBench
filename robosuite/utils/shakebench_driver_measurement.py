"""Bounded measurement seam for long ShakeBench deck-driver probes.

The seam changes only observability: MuJoCo continues to advance every model
timestep, while the declared measurement trace is retained at a fixed cadence
that exceeds twenty times the maximum authored spectral frequency.  It is
usable by a pre-registration capacity check and by a later immutable protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosuite.scripts.shakebench_select_physics import _run_driver_trace, _trace_digest, _trace_shapes
from robosuite.utils.shakebench_deck import TRACE_FIELD_CONTRACT, TRACE_SCHEMA_ID, TRACE_SCHEMA_VERSION


MEASUREMENT_SCHEMA_ID = "shakebench.driver.bounded_measurement"
MEASUREMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DriverMeasurementPlan:
    """Pre-registered sample cadence and right-limit trace contract."""

    physics_timestep_s: float
    cadence_hz: float = 200.0
    f_max_hz: float = 8.87

    def __post_init__(self) -> None:
        dt = float(self.physics_timestep_s)
        cadence = float(self.cadence_hz)
        f_max = float(self.f_max_hz)
        if not all(np.isfinite(value) and value > 0.0 for value in (dt, cadence, f_max)):
            raise ValueError("measurement plan values must be finite and positive")
        if cadence < 20.0 * f_max:
            raise ValueError("measurement cadence must be at least 20 * f_max")
        stride = 1.0 / (cadence * dt)
        if not np.isclose(stride, round(stride), rtol=0.0, atol=1.0e-12):
            raise ValueError("measurement cadence must be an integral multiple of physics timestep")

    @property
    def refresh_stride(self) -> int:
        return int(round(1.0 / (self.cadence_hz * self.physics_timestep_s)))

    @property
    def sample_dt_s(self) -> float:
        return self.refresh_stride * self.physics_timestep_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": MEASUREMENT_SCHEMA_ID,
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "physics_timestep_s": self.physics_timestep_s,
            "cadence_hz": self.cadence_hz,
            "minimum_required_cadence_hz": 20.0 * self.f_max_hz,
            "f_max_hz": self.f_max_hz,
            "refresh_stride": self.refresh_stride,
            "sample_dt_s": self.sample_dt_s,
            "sampling_phase_rule": "post-integration right-limit refresh; sample at data.time after every refresh_stride MuJoCo steps",
            "retention": "bounded named measurement trace; no record is retained for unmeasured integration steps",
            "trace_schema_id": TRACE_SCHEMA_ID,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_fields": list(TRACE_FIELD_CONTRACT),
        }


def _candidate_for_plan(plan: DriverMeasurementPlan) -> dict[str, Any]:
    return {
        "candidate_id": "capacity_preflight_dt_fine",
        "physics_timestep_s": float(plan.physics_timestep_s),
        "integrator": "Euler",
        "solver": "Newton",
        "solver_iterations": 100,
        "solver_tolerance": 1.0e-12,
        "deck_mass_kg": 400.0,
        "deck_inertia_kg_m2": [12.12, 14.083333333333334, 26.033333333333332],
        "deck_eq_solref": [max(2.0 * float(plan.physics_timestep_s), 0.0002), 0.5],
        "deck_eq_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    }


def run_capacity_preflight(*, trajectory: Any, duration_s: float, plan: DriverMeasurementPlan, runtime_ceiling_s: float) -> dict[str, Any]:
    """Run a non-decisional exact-duration capacity check.

    The resulting trace is deliberately not fitted or scored; it only records
    whether the final measurement mechanism can complete under its declared
    local runtime ceiling.
    """

    if not np.isfinite(duration_s) or float(duration_s) <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(runtime_ceiling_s) or float(runtime_ceiling_s) <= 0.0:
        raise ValueError("runtime_ceiling_s must be finite and positive")
    started = time.perf_counter()
    trace, audit, options = _run_driver_trace(
        _candidate_for_plan(plan),
        duration_s=float(duration_s),
        trajectory=trajectory,
        load_case="empty",
        refresh_stride=plan.refresh_stride,
    )
    elapsed = time.perf_counter() - started
    expected_steps = int(math.ceil(float(duration_s) / plan.physics_timestep_s))
    metadata = {
        "kind": "engineering_capacity_preflight_non_decisional",
        "measurement_plan": plan.to_dict(),
        "duration_s": float(duration_s),
        "runtime_ceiling_s": float(runtime_ceiling_s),
        "elapsed_wall_time_s": float(elapsed),
        "mujoco_step_count": expected_steps,
        "peak_retained_sample_count": int(trace.sample_time_s.size),
        "sampled_trace_digest": _trace_digest(trace),
        "sampled_trace_shapes": _trace_shapes(trace),
        "options": options,
        "audit": audit,
        "passed": bool(elapsed <= float(runtime_ceiling_s)),
        "selection_input": False,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    metadata["metadata_sha256"] = hashlib.sha256(encoded).hexdigest()
    return metadata
