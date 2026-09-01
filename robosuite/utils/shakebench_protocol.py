"""Pure protocol-state resolution for Phase 06R4/V5.

Manifest rows are references only.  Candidate-owned physics values live in
the component tables and are copied into one immutable resolved state before
either dry-run or execution.  This module deliberately imports no official
profile and creates no MuJoCo model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ProtocolStateError(ValueError):
    """A manifest reference cannot be resolved into a safe runtime state."""


_MANIFEST_OWNED_KEYS = {
    "physics_timestep_s",
    "control_steps",
    "refresh_stride",
    "sample_rate_hz",
    "sample_dt_s",
    "solver",
    "integrator",
    "iterations",
    "tolerance",
    "deck_eq_solref",
    "deck_eq_solimp",
    "deck_mass_kg",
    "deck_inertia_kg_m2",
}


@dataclass(frozen=True)
class ResolvedPhysicsProbeState:
    """Complete immutable runtime state consumed by a V5 probe."""

    state_id: str
    stage: str
    candidate_id: str
    gamma: float | None
    load_case: str | None
    physics_timestep_s: float
    control_period_s: float
    control_steps: int
    integrator: str
    solver: str
    solver_iterations: int
    solver_tolerance: float
    deck_mass_kg: float
    deck_inertia_kg_m2: tuple[float, ...]
    deck_eq_solref: tuple[float, ...]
    deck_eq_solimp: tuple[float, ...]
    measurement_rate_hz: float
    refresh_stride: int
    sample_dt_s: float
    duration_s: float
    mujoco_step_count: int
    retained_sample_count: int
    output: str
    retry_policy: str
    trace_schema_id: str
    trace_schema_version: int
    dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "stage": self.stage,
            "candidate_id": self.candidate_id,
            "gamma": self.gamma,
            "load_case": self.load_case,
            "physics_timestep_s": self.physics_timestep_s,
            "control_period_s": self.control_period_s,
            "control_steps": self.control_steps,
            "integrator": self.integrator,
            "solver": self.solver,
            "solver_iterations": self.solver_iterations,
            "solver_tolerance": self.solver_tolerance,
            "deck_mass_kg": self.deck_mass_kg,
            "deck_inertia_kg_m2": list(self.deck_inertia_kg_m2),
            "deck_eq_solref": list(self.deck_eq_solref),
            "deck_eq_solimp": list(self.deck_eq_solimp),
            "measurement_rate_hz": self.measurement_rate_hz,
            "refresh_stride": self.refresh_stride,
            "sample_dt_s": self.sample_dt_s,
            "duration_s": self.duration_s,
            "mujoco_step_count": self.mujoco_step_count,
            "retained_sample_count": self.retained_sample_count,
            "output": self.output,
            "retry_policy": self.retry_policy,
            "trace_schema_id": self.trace_schema_id,
            "trace_schema_version": self.trace_schema_version,
            "dependencies": list(self.dependencies),
        }


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ProtocolStateError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolStateError(f"{label} must be a finite number") from exc
    if not np.isfinite(number) or (positive and number <= 0.0):
        raise ProtocolStateError(f"{label} must be finite and positive")
    return number


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ProtocolStateError(f"{label} must contain {length} values")
    try:
        values = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise ProtocolStateError(f"{label} must contain {length} values") from exc
    if len(values) != length:
        raise ProtocolStateError(f"{label} must contain {length} values")
    return values


def _candidate(protocol: Mapping[str, Any], section: str, candidate_id: str) -> Mapping[str, Any]:
    rows = protocol.get("components", {}).get(section, ())
    for row in rows:
        if row.get("candidate_id") == candidate_id:
            return row
    raise ProtocolStateError(f"unknown {section} candidate: {candidate_id}")


def _driver(protocol: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    rows = protocol.get("driver", {}).get("convergence_candidates", ())
    for row in rows:
        if row.get("candidate_id") == candidate_id:
            return row
    raise ProtocolStateError(f"unknown driver candidate: {candidate_id}")


def _manifest_row(protocol: Mapping[str, Any], state_id: str) -> Mapping[str, Any]:
    rows = [row for row in protocol.get("execution_manifest", ()) if row.get("state_id") == state_id]
    if len(rows) != 1:
        raise ProtocolStateError(f"state_id must resolve to exactly one manifest row: {state_id}")
    row = rows[0]
    if _MANIFEST_OWNED_KEYS.intersection(row):
        names = sorted(_MANIFEST_OWNED_KEYS.intersection(row))
        raise ProtocolStateError(f"manifest cannot override candidate-owned fields: {', '.join(names)}")
    return row


def _driver_runtime(protocol: Mapping[str, Any], row: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    dt = _number(candidate.get("physics_timestep_s"), "physics_timestep_s", positive=True)
    control_period = 1.0 / _number(protocol.get("frozen_facts", {}).get("control_frequency_hz"), "control_frequency_hz", positive=True)
    control_ratio = control_period / dt
    control_steps = int(round(control_ratio))
    if not np.isclose(control_ratio, control_steps, rtol=0.0, atol=1e-12):
        raise ProtocolStateError("control period is not an integer multiple of candidate timestep")
    stride = int(candidate.get("refresh_stride"))
    if stride <= 0 or candidate.get("refresh_stride") != stride:
        raise ProtocolStateError("candidate refresh_stride must be a positive integer")
    sample_dt = _number(candidate.get("sample_dt_s"), "sample_dt_s", positive=True)
    rate = _number(candidate.get("sample_rate_hz"), "sample_rate_hz", positive=True)
    if not np.isclose(sample_dt, stride * dt, rtol=0.0, atol=1e-12):
        raise ProtocolStateError("candidate sample_dt_s does not equal refresh_stride * timestep")
    if not np.isclose(rate, 1.0 / sample_dt, rtol=0.0, atol=1e-9):
        raise ProtocolStateError("candidate sample_rate_hz does not equal inverse sample_dt_s")
    if rate < 20.0 * _number(protocol["frozen_facts"]["f_max_hz"], "f_max_hz", positive=True):
        raise ProtocolStateError("candidate measurement rate is below 20 * f_max")
    if dt > 1.0 / (20.0 * float(protocol["frozen_facts"]["f_max_hz"])):
        raise ProtocolStateError("candidate timestep violates f_max gate")
    solref = _vector(candidate.get("deck_eq_solref"), 2, "deck_eq_solref")
    if solref[0] < 2.0 * dt:
        raise ProtocolStateError("candidate deck solref time constant is below 2 * timestep")
    duration = _number(row.get("duration_s"), "duration_s", positive=True)
    controls = int(math.ceil(duration / control_period))
    steps = controls * control_steps
    samples = int(math.ceil(steps / stride))
    if steps <= 0 or samples <= 0:
        raise ProtocolStateError("resolved driver step/sample counts must be positive")
    return {"physics_timestep_s": dt, "control_period_s": control_period, "control_steps": control_steps, "integrator": str(candidate["integrator"]), "solver": str(candidate["solver"]), "solver_iterations": int(candidate["iterations"]), "solver_tolerance": _number(candidate["tolerance"], "tolerance"), "deck_mass_kg": _number(candidate["deck_mass_kg"], "deck_mass_kg", positive=True), "deck_inertia_kg_m2": _vector(candidate["deck_inertia_kg_m2"], 3, "deck_inertia_kg_m2"), "deck_eq_solref": solref, "deck_eq_solimp": _vector(candidate["deck_eq_solimp"], 5, "deck_eq_solimp"), "measurement_rate_hz": rate, "refresh_stride": stride, "sample_dt_s": sample_dt, "duration_s": duration, "mujoco_step_count": steps, "retained_sample_count": samples}


def _generic_runtime(protocol: Mapping[str, Any], row: Mapping[str, Any], candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        candidate = protocol.get("runtime_defaults", {})
    dt = _number(candidate.get("physics_timestep_s"), "physics_timestep_s", positive=True)
    stride = int(candidate.get("refresh_stride", 1))
    if stride <= 0:
        raise ProtocolStateError("refresh_stride must be positive")
    sample_dt = _number(candidate.get("sample_dt_s", dt * stride), "sample_dt_s", positive=True)
    rate = _number(candidate.get("sample_rate_hz", 1.0 / sample_dt), "sample_rate_hz", positive=True)
    if not np.isclose(sample_dt, stride * dt, rtol=0.0, atol=1e-12) or not np.isclose(rate, 1.0 / sample_dt, rtol=0.0, atol=1e-9):
        raise ProtocolStateError("generic measurement cadence is inconsistent")
    duration = _number(row.get("duration_s"), "duration_s", positive=True)
    steps = int(math.ceil(duration / dt))
    samples = int(math.ceil(steps / stride))
    return {"physics_timestep_s": dt, "control_period_s": 0.05, "control_steps": int(round(0.05 / dt)) if np.isclose(0.05 / dt, round(0.05 / dt), atol=1e-12) else 0, "integrator": str(candidate.get("integrator", "Euler")), "solver": str(candidate.get("solver", "Newton")), "solver_iterations": int(candidate.get("iterations", 100)), "solver_tolerance": _number(candidate.get("tolerance", 1e-12), "tolerance"), "deck_mass_kg": _number(candidate.get("deck_mass_kg", 400.0), "deck_mass_kg", positive=True), "deck_inertia_kg_m2": _vector(candidate.get("deck_inertia_kg_m2", [12.12, 14.083333333333334, 26.033333333333332]), 3, "deck_inertia_kg_m2"), "deck_eq_solref": _vector(candidate.get("deck_eq_solref", [max(2.0 * dt, 0.0002), 0.5]), 2, "deck_eq_solref"), "deck_eq_solimp": _vector(candidate.get("deck_eq_solimp", [0.9, 0.95, 0.001, 0.5, 2.0]), 5, "deck_eq_solimp"), "measurement_rate_hz": rate, "refresh_stride": stride, "sample_dt_s": sample_dt, "duration_s": duration, "mujoco_step_count": steps, "retained_sample_count": samples}


def resolve_protocol_state(protocol: Mapping[str, Any], state_id: str, selection_context: Mapping[str, Any] | None = None) -> ResolvedPhysicsProbeState:
    """Resolve one manifest reference without accepting manifest overrides."""

    if not isinstance(state_id, str) or not state_id:
        raise ProtocolStateError("state_id must be a nonempty string")
    row = _manifest_row(protocol, state_id)
    stage = row.get("stage")
    if stage not in {"driver", "isolator", "contact", "parity", "replay"}:
        raise ProtocolStateError(f"unsupported stage: {stage}")
    candidate_id = str(row.get("candidate_id", ""))
    runtime: dict[str, Any]
    dependencies = tuple(str(item) for item in row.get("dependencies", ()))
    if stage == "driver":
        candidate = _driver(protocol, candidate_id)
        runtime = _driver_runtime(protocol, row, candidate)
    elif stage == "isolator":
        _candidate(protocol, "isolator_candidates", candidate_id)
        runtime = _generic_runtime(protocol, row, protocol.get("runtime_defaults"))
    elif stage == "contact":
        _candidate(protocol, "contact_candidates", candidate_id)
        runtime = _generic_runtime(protocol, row, protocol.get("runtime_defaults"))
    else:
        if stage == "replay" and candidate_id not in {"driver", "isolator", "contact", "gamma_zero_parity"}:
            raise ProtocolStateError(f"unknown replay group: {candidate_id}")
        runtime = _generic_runtime(protocol, row, protocol.get("runtime_defaults"))
    gamma = row.get("gamma")
    if gamma is not None:
        gamma = _number(gamma, "gamma")
    load_case = None if row.get("load_case") is None else str(row["load_case"])
    output = row.get("output")
    if not isinstance(output, str) or not output:
        raise ProtocolStateError("manifest output must be a nonempty path")
    retry_policy = row.get("retry_policy")
    if retry_policy != "same_state_config_once_then_block":
        raise ProtocolStateError("unexpected retry policy")
    return ResolvedPhysicsProbeState(
        state_id=state_id,
        stage=str(stage),
        candidate_id=candidate_id,
        gamma=gamma,
        load_case=load_case,
        integrator=runtime["integrator"],
        solver=runtime["solver"],
        solver_iterations=runtime["solver_iterations"],
        solver_tolerance=runtime["solver_tolerance"],
        deck_mass_kg=runtime["deck_mass_kg"],
        deck_inertia_kg_m2=runtime["deck_inertia_kg_m2"],
        deck_eq_solref=runtime["deck_eq_solref"],
        deck_eq_solimp=runtime["deck_eq_solimp"],
        dependencies=dependencies,
        **{key: runtime[key] for key in ("physics_timestep_s", "control_period_s", "control_steps", "measurement_rate_hz", "refresh_stride", "sample_dt_s", "duration_s", "mujoco_step_count", "retained_sample_count")},
        output=output,
        retry_policy=retry_policy,
        trace_schema_id=str(protocol.get("measurement", {}).get("trace_schema_id", "shakebench.deck_driver.trace")),
        trace_schema_version=int(protocol.get("measurement", {}).get("trace_schema_version", 3)),
    )


def resolve_all_protocol_states(protocol: Mapping[str, Any]) -> tuple[ResolvedPhysicsProbeState, ...]:
    rows = protocol.get("execution_manifest")
    if not isinstance(rows, list):
        raise ProtocolStateError("execution_manifest must be a list")
    states = tuple(resolve_protocol_state(protocol, str(row.get("state_id"))) for row in rows)
    if len({state.state_id for state in states}) != len(states) or len({state.output for state in states}) != len(states):
        raise ProtocolStateError("resolved state IDs and outputs must be unique")
    return states
