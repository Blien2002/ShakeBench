"""Immutable Phase 06R5/V6 protocol resolution.

The V5 resolver intentionally remains available for archive/provenance tests.
V6 uses the models in this module as the configuration boundary between the
registered YAML and every stage adapter.  The resolver is the only place that
knows the shape of the protocol mapping; adapters consume only the returned
``ResolvedProbeState`` (or one of its frozen children).

This module imports no official physics profile and never creates a MuJoCo
model.  It is consequently safe to use for protocol validation and dry-run
preflight in a clean checkout.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from robosuite.utils.shakebench_isolator import (
    DEFAULT_ANGLE_LIMITS_RAD,
    DEFAULT_TRAVEL_LIMITS_M,
    IsolatorConfig,
    derive_isolator_parameters,
)


class V6ProtocolStateError(ValueError):
    """A V6 manifest reference cannot be materialized safely."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively freeze mappings and sequences, including NumPy values."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return tuple(_freeze(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any, label: str, *, required: bool = True) -> Mapping[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise V6ProtocolStateError(f"{label} must be a mapping")
    return value


def _number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise V6ProtocolStateError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise V6ProtocolStateError(f"{label} must be a finite number") from exc
    if not np.isfinite(number):
        raise V6ProtocolStateError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise V6ProtocolStateError(f"{label} must be positive")
    if nonnegative and number < 0.0:
        raise V6ProtocolStateError(f"{label} must be nonnegative")
    return number


def _integer(value: Any, label: str, *, positive: bool = False, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise V6ProtocolStateError(f"{label} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise V6ProtocolStateError(f"{label} must be an integer") from exc
    if not np.isfinite(number) or number != round(number):
        raise V6ProtocolStateError(f"{label} must be an integer")
    result = int(number)
    if positive and result <= 0:
        raise V6ProtocolStateError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise V6ProtocolStateError(f"{label} must be >= {minimum}")
    return result


def _vector(value: Any, length: int, label: str, *, positive: bool = False, nonnegative: bool = False) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V6ProtocolStateError(f"{label} must contain {length} values")
    if len(value) != length:
        raise V6ProtocolStateError(f"{label} must contain {length} values")
    return tuple(
        _number(item, f"{label}[{index}]", positive=positive, nonnegative=nonnegative)
        for index, item in enumerate(value)
    )


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V6ProtocolStateError(f"{label} must be a nonempty string")
    return value


def _same_vector(actual: Sequence[float], expected: Sequence[float], label: str, *, atol: float = 1.0e-12) -> None:
    if len(actual) != len(expected) or not np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0.0, atol=atol
    ):
        raise V6ProtocolStateError(f"{label} does not match resolver-derived value")


@dataclass(frozen=True)
class ResolvedCommon:
    """Fields shared by every physical-selection stage."""

    state_id: str
    stage: str
    output: str
    retry_policy: str
    protocol_identity: Mapping[str, Any]
    physics_timestep_s: float
    control_period_s: float
    control_steps: int
    measurement_rate_hz: float
    refresh_stride: int
    sample_dt_s: float
    duration_s: float
    mujoco_step_count: int
    retained_sample_count: int
    trace_schema_id: str
    trace_schema_version: int
    integrator: str
    solver: str
    solver_iterations: int
    solver_tolerance: float
    deck_mass_kg: float
    deck_inertia_kg_m2: tuple[float, ...]
    deck_eq_solref: tuple[float, ...]
    deck_eq_solimp: tuple[float, ...]
    dependencies: tuple[str, ...] = ()
    runtime_source_candidate_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_identity", _freeze(self.protocol_identity))
        object.__setattr__(self, "dependencies", tuple(str(item) for item in self.dependencies))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "stage": self.stage,
            "output": self.output,
            "retry_policy": self.retry_policy,
            "protocol_identity": _thaw(self.protocol_identity),
            "physics_timestep_s": self.physics_timestep_s,
            "control_period_s": self.control_period_s,
            "control_steps": self.control_steps,
            "measurement_rate_hz": self.measurement_rate_hz,
            "refresh_stride": self.refresh_stride,
            "sample_dt_s": self.sample_dt_s,
            "duration_s": self.duration_s,
            "mujoco_step_count": self.mujoco_step_count,
            "retained_sample_count": self.retained_sample_count,
            "trace_schema_id": self.trace_schema_id,
            "trace_schema_version": self.trace_schema_version,
            "integrator": self.integrator,
            "solver": self.solver,
            "solver_iterations": self.solver_iterations,
            "solver_tolerance": self.solver_tolerance,
            "deck_mass_kg": self.deck_mass_kg,
            "deck_inertia_kg_m2": list(self.deck_inertia_kg_m2),
            "deck_eq_solref": list(self.deck_eq_solref),
            "deck_eq_solimp": list(self.deck_eq_solimp),
            "dependencies": list(self.dependencies),
            "runtime_source_candidate_id": self.runtime_source_candidate_id,
        }


@dataclass(frozen=True)
class ResolvedDriver:
    """Frozen driver/deck/weld/program input."""

    candidate_id: str
    gamma: float
    load_case: str
    integrator: str
    solver: str
    solver_iterations: int
    solver_tolerance: float
    deck: Mapping[str, Any]
    weld: Mapping[str, Any]
    program: Mapping[str, Any]
    hard_gates: Mapping[str, Any]
    candidate_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("deck", "weld", "program", "hard_gates", "candidate_fields"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @property
    def physics_timestep_s(self) -> float:
        return float(self.deck["physics_timestep_s"])

    @property
    def deck_mass_kg(self) -> float:
        return float(self.deck["mass_kg"])

    @property
    def deck_inertia_kg_m2(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.deck["inertia_kg_m2"])

    @property
    def deck_eq_solref(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.weld["solref"])

    @property
    def deck_eq_solimp(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.weld["solimp"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "gamma": self.gamma,
            "load_case": self.load_case,
            "integrator": self.integrator,
            "solver": self.solver,
            "solver_iterations": self.solver_iterations,
            "solver_tolerance": self.solver_tolerance,
            "deck": _thaw(self.deck),
            "weld": _thaw(self.weld),
            "program": _thaw(self.program),
            "hard_gates": _thaw(self.hard_gates),
            "candidate_fields": _thaw(self.candidate_fields),
        }


@dataclass(frozen=True)
class ResolvedIsolator:
    """Frozen isolator candidate, derived parameters, and probe contract."""

    candidate_id: str
    driver_candidate_id: str
    fn_hz: tuple[float, ...]
    zeta: tuple[float, ...]
    effective_mass: tuple[float, ...]
    omega_n_rad_s: tuple[float, ...]
    k: tuple[float, ...]
    c: tuple[float, ...]
    springref: tuple[float, ...]
    mass_kg: float
    inertia_kg_m2: tuple[float, ...]
    gravity_m_s2: float
    travel_limits_m: tuple[float, ...]
    angle_limits_rad: tuple[float, ...]
    transfer_regions: Mapping[str, Any]
    combined_spectrum: Mapping[str, Any]
    payload: Mapping[str, Any]
    equilibrium: Mapping[str, Any]
    hard_gates: Mapping[str, Any]
    scoring: Mapping[str, Any]
    candidate_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("transfer_regions", "combined_spectrum", "payload", "equilibrium", "hard_gates", "scoring", "candidate_fields"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        for name in ("fn_hz", "zeta", "effective_mass", "omega_n_rad_s", "k", "c", "springref", "inertia_kg_m2", "travel_limits_m", "angle_limits_rad"):
            object.__setattr__(self, name, tuple(float(item) for item in getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "driver_candidate_id": self.driver_candidate_id,
            "fn_hz": list(self.fn_hz),
            "zeta": list(self.zeta),
            "effective_mass": list(self.effective_mass),
            "omega_n_rad_s": list(self.omega_n_rad_s),
            "k": list(self.k),
            "c": list(self.c),
            "springref": list(self.springref),
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "gravity_m_s2": self.gravity_m_s2,
            "travel_limits_m": list(self.travel_limits_m),
            "angle_limits_rad": list(self.angle_limits_rad),
            "transfer_regions": _thaw(self.transfer_regions),
            "combined_spectrum": _thaw(self.combined_spectrum),
            "payload": _thaw(self.payload),
            "equilibrium": _thaw(self.equilibrium),
            "hard_gates": _thaw(self.hard_gates),
            "scoring": _thaw(self.scoring),
            "candidate_fields": _thaw(self.candidate_fields),
        }


@dataclass(frozen=True)
class ResolvedContact:
    """Frozen contact candidate and every contact probe input."""

    candidate_id: str
    driver_candidate_id: str
    isolator_candidate_id: str
    condim: int
    sliding_mu: Mapping[str, Any]
    torsional_mu: float
    rolling_mu: float
    margin_m: float
    gap_m: float
    solref: tuple[float, ...]
    solimp: tuple[float, ...]
    iterations: int
    interfaces: tuple[str, ...]
    probes: Mapping[str, Any]
    hard_gates: Mapping[str, Any]
    scoring: Mapping[str, Any]
    candidate_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("sliding_mu", "probes", "hard_gates", "scoring", "candidate_fields"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "interfaces", tuple(str(item) for item in self.interfaces))
        object.__setattr__(self, "solref", tuple(float(item) for item in self.solref))
        object.__setattr__(self, "solimp", tuple(float(item) for item in self.solimp))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "driver_candidate_id": self.driver_candidate_id,
            "isolator_candidate_id": self.isolator_candidate_id,
            "condim": self.condim,
            "sliding_mu": _thaw(self.sliding_mu),
            "torsional_mu": self.torsional_mu,
            "rolling_mu": self.rolling_mu,
            "margin_m": self.margin_m,
            "gap_m": self.gap_m,
            "solref": list(self.solref),
            "solimp": list(self.solimp),
            "iterations": self.iterations,
            "interfaces": list(self.interfaces),
            "probes": _thaw(self.probes),
            "hard_gates": _thaw(self.hard_gates),
            "scoring": _thaw(self.scoring),
            "candidate_fields": _thaw(self.candidate_fields),
        }


@dataclass(frozen=True)
class ResolvedParity:
    """Bounded Gamma=0 parity contract."""

    candidate_id: str
    driver_candidate_id: str
    isolator_candidate_id: str
    contact_candidate_id: str
    static_duration_s: float
    dynamic_duration_s: float
    geometry_tolerance_m: float
    action_dimension_exact: bool
    action_dimension: int
    support_force_min_N: float
    support_force_tolerance_N: float
    expected_geometry: Mapping[str, Any]
    expected_action: Mapping[str, Any]
    expected_support: Mapping[str, Any]
    candidate_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("expected_geometry", "expected_action", "expected_support", "candidate_fields"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "driver_candidate_id": self.driver_candidate_id,
            "isolator_candidate_id": self.isolator_candidate_id,
            "contact_candidate_id": self.contact_candidate_id,
            "static_duration_s": self.static_duration_s,
            "dynamic_duration_s": self.dynamic_duration_s,
            "geometry_tolerance_m": self.geometry_tolerance_m,
            "action_dimension_exact": self.action_dimension_exact,
            "action_dimension": self.action_dimension,
            "support_force_min_N": self.support_force_min_N,
            "support_force_tolerance_N": self.support_force_tolerance_N,
            "expected_geometry": _thaw(self.expected_geometry),
            "expected_action": _thaw(self.expected_action),
            "expected_support": _thaw(self.expected_support),
            "candidate_fields": _thaw(self.candidate_fields),
        }


@dataclass(frozen=True)
class ResolvedReplay:
    """Independent-process replay binding and complete trace schema."""

    selected_component: str
    selected_candidate_id: str
    binding: Mapping[str, Any]
    process_index: int
    process_count: int
    trace_schema_id: str
    trace_schema_version: int
    trace_fields: tuple[str, ...]
    determinism_tolerances: Mapping[str, Any]
    candidate_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("binding", "determinism_tolerances", "candidate_fields"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "trace_fields", tuple(str(item) for item in self.trace_fields))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_component": self.selected_component,
            "selected_candidate_id": self.selected_candidate_id,
            "binding": _thaw(self.binding),
            "process_index": self.process_index,
            "process_count": self.process_count,
            "trace_schema_id": self.trace_schema_id,
            "trace_schema_version": self.trace_schema_version,
            "trace_fields": list(self.trace_fields),
            "determinism_tolerances": _thaw(self.determinism_tolerances),
            "candidate_fields": _thaw(self.candidate_fields),
        }


@dataclass(frozen=True)
class ResolvedProbeState:
    """Tagged immutable union consumed by all V6 stage adapters."""

    common: ResolvedCommon
    driver: ResolvedDriver | None = None
    isolator: ResolvedIsolator | None = None
    contact: ResolvedContact | None = None
    parity: ResolvedParity | None = None
    replay: ResolvedReplay | None = None

    @property
    def state_id(self) -> str:
        return self.common.state_id

    @property
    def stage(self) -> str:
        return self.common.stage

    @property
    def output(self) -> str:
        return self.common.output

    @property
    def candidate_id(self) -> str:
        ordered = {
            "driver": (self.driver,),
            "isolator": (self.isolator,),
            "contact": (self.contact,),
            "parity": (self.parity,),
        }.get(self.stage, (self.driver, self.isolator, self.contact, self.parity))
        for child in ordered:
            if child is not None:
                return str(child.candidate_id)
        if self.replay is not None:
            return self.replay.selected_candidate_id
        raise V6ProtocolStateError(f"resolved state has no candidate: {self.state_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "common": self.common.to_dict(),
            "driver": None if self.driver is None else self.driver.to_dict(),
            "isolator": None if self.isolator is None else self.isolator.to_dict(),
            "contact": None if self.contact is None else self.contact.to_dict(),
            "parity": None if self.parity is None else self.parity.to_dict(),
            "replay": None if self.replay is None else self.replay.to_dict(),
        }

    @property
    def resolved_state_digest(self) -> str:
        return sha256_json(self.to_dict())


_MANIFEST_CANDIDATE_OWNED_KEYS = {
    "physics_timestep_s",
    "control_steps",
    "refresh_stride",
    "sample_rate_hz",
    "measurement_rate_hz",
    "sample_dt_s",
    "solver",
    "integrator",
    "iterations",
    "solver_iterations",
    "tolerance",
    "solver_tolerance",
    "deck_eq_solref",
    "deck_eq_solimp",
    "deck_mass_kg",
    "deck_inertia_kg_m2",
    "fn_hz",
    "f_n_hz",
    "zeta",
    "damping_ratio",
    "k",
    "c",
    "springref",
    "derived_k",
    "derived_c",
    "derived_springref",
    "condim",
    "sliding_mu",
    "torsional_mu",
    "rolling_mu",
    "margin_m",
    "gap_m",
    "solref",
    "solimp",
    "iterations",
}


def _find_candidate(protocol: Mapping[str, Any], section: str, candidate_id: str) -> Mapping[str, Any]:
    components = _mapping(protocol.get("components"), "components")
    rows = components.get(section)
    if not isinstance(rows, list):
        raise V6ProtocolStateError(f"components.{section} must be a list")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise V6ProtocolStateError(f"{section} candidate must resolve exactly once: {candidate_id}")
    return matches[0]


def _find_driver(protocol: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    driver = _mapping(protocol.get("driver"), "driver")
    rows = driver.get("convergence_candidates")
    if not isinstance(rows, list):
        raise V6ProtocolStateError("driver.convergence_candidates must be a list")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise V6ProtocolStateError(f"driver candidate must resolve exactly once: {candidate_id}")
    return matches[0]


def _manifest_row(protocol: Mapping[str, Any], state_id: str) -> Mapping[str, Any]:
    rows = protocol.get("execution_manifest")
    if not isinstance(rows, list):
        raise V6ProtocolStateError("execution_manifest must be a list")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("state_id") == state_id]
    if len(matches) != 1:
        raise V6ProtocolStateError(f"state_id must resolve to exactly one manifest row: {state_id}")
    row = matches[0]
    shadowed = sorted(_MANIFEST_CANDIDATE_OWNED_KEYS.intersection(row))
    if shadowed:
        raise V6ProtocolStateError("manifest cannot override candidate-owned fields: " + ", ".join(shadowed))
    return row


def _facts(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(protocol.get("frozen_facts"), "frozen_facts")


def _candidate_id_from(row: Mapping[str, Any], context: Mapping[str, Any] | None, key: str, default: Any = None) -> str:
    context = context or {}
    row_value = row.get(key)
    context_value = context.get(key)
    if row_value is not None and context_value is not None and str(row_value) != str(context_value):
        raise V6ProtocolStateError(f"resolved binding disagrees with manifest reference: {key}")
    value = context_value if context_value is not None else (row_value if row_value is not None else default)
    if value is None:
        raise V6ProtocolStateError(f"missing resolved binding: {key}")
    return _required_string(str(value), key)


def _runtime_candidate(protocol: Mapping[str, Any], row: Mapping[str, Any], stage: str, context: Mapping[str, Any] | None) -> tuple[str | None, Mapping[str, Any]]:
    if stage == "driver":
        candidate_id = _required_string(str(row.get("candidate_id", "")), "driver candidate_id")
        return candidate_id, _find_driver(protocol, candidate_id)
    selection = _mapping(protocol.get("selection"), "selection", required=False)
    defaults = _mapping(selection.get("default"), "selection.default", required=False)
    driver_id = context.get("driver_candidate_id") if context else None
    binding = row.get("binding", row.get("selection_binding"))
    if stage == "replay" and isinstance(binding, Mapping):
        binding_driver_id = binding.get("driver_candidate_id")
        if driver_id is not None and binding_driver_id is not None and str(driver_id) != str(binding_driver_id):
            raise V6ProtocolStateError("replay binding disagrees with selected driver")
        driver_id = binding_driver_id if binding_driver_id is not None else driver_id
    row_driver_id = row.get("driver_candidate_id")
    if row_driver_id is not None and driver_id is not None and str(row_driver_id) != str(driver_id):
        raise V6ProtocolStateError("manifest driver binding disagrees with selected driver")
    if row_driver_id is not None:
        driver_id = row_driver_id
    if driver_id is None:
        driver_id = defaults.get("driver_candidate_id", defaults.get("driver"))
    if driver_id is not None:
        driver_id = _required_string(str(driver_id), "driver_candidate_id")
        return driver_id, _find_driver(protocol, driver_id)
    runtime_defaults = protocol.get("runtime_defaults")
    if not isinstance(runtime_defaults, Mapping):
        raise V6ProtocolStateError(f"{stage} state has no driver binding or runtime_defaults")
    return None, runtime_defaults


def _runtime(protocol: Mapping[str, Any], row: Mapping[str, Any], stage: str, context: Mapping[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    source_id, candidate = _runtime_candidate(protocol, row, stage, context)
    facts = _facts(protocol)
    frequency = _number(facts.get("control_frequency_hz"), "frozen_facts.control_frequency_hz", positive=True)
    dt = _number(candidate.get("physics_timestep_s"), f"{stage}.physics_timestep_s", positive=True)
    control_period = 1.0 / frequency
    ratio = control_period / dt
    control_steps = _integer(round(ratio), f"{stage}.control_steps", positive=True)
    if not np.isclose(ratio, control_steps, rtol=0.0, atol=1.0e-12):
        raise V6ProtocolStateError("control period is not an integer multiple of candidate timestep")
    if "control_steps" in candidate and _integer(candidate.get("control_steps"), f"{stage}.candidate.control_steps", positive=True) != control_steps:
        raise V6ProtocolStateError("declared control_steps does not match timestep/control frequency")
    stride = _integer(candidate.get("refresh_stride"), f"{stage}.refresh_stride", positive=True)
    sample_dt = _number(candidate.get("sample_dt_s"), f"{stage}.sample_dt_s", positive=True)
    sample_rate = _number(candidate.get("sample_rate_hz"), f"{stage}.sample_rate_hz", positive=True)
    if not np.isclose(sample_dt, stride * dt, rtol=0.0, atol=1.0e-12):
        raise V6ProtocolStateError("sample_dt_s does not equal refresh_stride * timestep")
    if not np.isclose(sample_rate, 1.0 / sample_dt, rtol=0.0, atol=1.0e-9):
        raise V6ProtocolStateError("sample_rate_hz does not equal inverse sample_dt_s")
    minimum_cadence = _number(
        protocol.get("measurement", {}).get("minimum_cadence_hz", 20.0 * _number(facts.get("f_max_hz"), "frozen_facts.f_max_hz", positive=True)),
        "measurement.minimum_cadence_hz",
        positive=True,
    )
    if sample_rate < minimum_cadence:
        raise V6ProtocolStateError("sample_rate_hz is below the registered minimum cadence")
    f_max = _number(facts.get("f_max_hz"), "frozen_facts.f_max_hz", positive=True)
    if dt > 1.0 / (20.0 * f_max):
        raise V6ProtocolStateError("physics timestep violates f_max gate")
    deck_solref = _vector(candidate.get("deck_eq_solref"), 2, f"{stage}.deck_eq_solref", positive=True)
    if deck_solref[0] < 2.0 * dt:
        raise V6ProtocolStateError("deck solref time constant is below 2 * timestep")
    duration = _number(row.get("duration_s"), f"{row.get('state_id', stage)}.duration_s", positive=True)
    control_count = int(math.ceil(duration / control_period))
    mujoco_steps = control_count * control_steps
    retained_samples = int(math.ceil(mujoco_steps / stride))
    if mujoco_steps <= 0 or retained_samples <= 0:
        raise V6ProtocolStateError("resolved step and sample counts must be positive")
    runtime = {
        "physics_timestep_s": dt,
        "control_period_s": control_period,
        "control_steps": control_steps,
        "measurement_rate_hz": sample_rate,
        "refresh_stride": stride,
        "sample_dt_s": sample_dt,
        "duration_s": duration,
        "mujoco_step_count": mujoco_steps,
        "retained_sample_count": retained_samples,
        "integrator": _required_string(str(candidate.get("integrator")), f"{stage}.integrator"),
        "solver": _required_string(str(candidate.get("solver")), f"{stage}.solver"),
        "solver_iterations": _integer(candidate.get("iterations"), f"{stage}.iterations", positive=True),
        "solver_tolerance": _number(candidate.get("tolerance"), f"{stage}.tolerance", positive=True),
        "deck_mass_kg": _number(candidate.get("deck_mass_kg"), f"{stage}.deck_mass_kg", positive=True),
        "deck_inertia_kg_m2": _vector(candidate.get("deck_inertia_kg_m2"), 3, f"{stage}.deck_inertia_kg_m2", positive=True),
        "deck_eq_solref": deck_solref,
        "deck_eq_solimp": _vector(candidate.get("deck_eq_solimp"), 5, f"{stage}.deck_eq_solimp"),
    }
    solimp = runtime["deck_eq_solimp"]
    if not 0.0 <= solimp[0] <= solimp[1] <= 1.0 or solimp[2] <= 0.0 or not 0.0 <= solimp[3] <= 1.0 or solimp[4] <= 0.0:
        raise V6ProtocolStateError("deck_eq_solimp is invalid")
    return runtime, source_id


def _protocol_identity(protocol: Mapping[str, Any], context: Mapping[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    return {
        "schema_id": str(protocol.get("schema_id", "")),
        "schema_version": int(protocol.get("schema_version", 0)),
        "bytes_sha256": context.get("protocol_sha256_bytes", protocol.get("protocol_sha256_bytes")),
        "normalized_sha256": context.get("protocol_sha256_normalized", sha256_json(protocol)),
    }


def _common(protocol: Mapping[str, Any], row: Mapping[str, Any], stage: str, context: Mapping[str, Any] | None) -> tuple[ResolvedCommon, Mapping[str, Any], str | None]:
    runtime, source_id = _runtime(protocol, row, stage, context)
    output = _required_string(row.get("output"), "manifest.output")
    if output.startswith("/") or "/" in output or "\\" in output or output in {".", ".."}:
        raise V6ProtocolStateError("manifest.output must be a flat artifact filename")
    retry_policy = row.get("retry_policy")
    if retry_policy != "same_state_config_once_then_block":
        raise V6ProtocolStateError("unexpected retry policy")
    measurement = _mapping(protocol.get("measurement"), "measurement")
    trace_schema_id = _required_string(measurement.get("trace_schema_id"), "measurement.trace_schema_id")
    trace_schema_version = _integer(measurement.get("trace_schema_version"), "measurement.trace_schema_version", minimum=1)
    state_id = _required_string(str(row.get("state_id", "")), "manifest.state_id")
    dependencies = row.get("dependencies", ())
    if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
        raise V6ProtocolStateError(f"{state_id}.dependencies must be a list")
    common = ResolvedCommon(
        state_id=state_id,
        stage=stage,
        output=output,
        retry_policy=retry_policy,
        protocol_identity=_protocol_identity(protocol, context),
        physics_timestep_s=runtime["physics_timestep_s"],
        control_period_s=runtime["control_period_s"],
        control_steps=runtime["control_steps"],
        measurement_rate_hz=runtime["measurement_rate_hz"],
        refresh_stride=runtime["refresh_stride"],
        sample_dt_s=runtime["sample_dt_s"],
        duration_s=runtime["duration_s"],
        mujoco_step_count=runtime["mujoco_step_count"],
        retained_sample_count=runtime["retained_sample_count"],
        trace_schema_id=trace_schema_id,
        trace_schema_version=trace_schema_version,
        integrator=runtime["integrator"],
        solver=runtime["solver"],
        solver_iterations=runtime["solver_iterations"],
        solver_tolerance=runtime["solver_tolerance"],
        deck_mass_kg=runtime["deck_mass_kg"],
        deck_inertia_kg_m2=runtime["deck_inertia_kg_m2"],
        deck_eq_solref=runtime["deck_eq_solref"],
        deck_eq_solimp=runtime["deck_eq_solimp"],
        dependencies=tuple(str(item) for item in dependencies),
        runtime_source_candidate_id=source_id,
    )
    return common, runtime, source_id


def _driver_child(protocol: Mapping[str, Any], row: Mapping[str, Any], runtime: Mapping[str, Any], candidate_id: str) -> ResolvedDriver:
    candidate = _find_driver(protocol, candidate_id)
    gamma = _number(row.get("gamma"), f"{row.get('state_id')}.gamma")
    load_case = _required_string(row.get("load_case"), f"{row.get('state_id')}.load_case")
    facts = _facts(protocol)
    allowed_gammas = tuple(float(item) for item in facts.get("safe_gamma_candidates", ()))
    allowed_loads = tuple(str(item) for item in facts.get("load_cases", ()))
    if allowed_gammas and not any(np.isclose(gamma, item, rtol=0.0, atol=1.0e-12) for item in allowed_gammas):
        raise V6ProtocolStateError("driver gamma is not in the registered safe set")
    if allowed_loads and load_case not in allowed_loads:
        raise V6ProtocolStateError("driver load_case is not in the registered load set")
    driver_spec = _mapping(protocol.get("driver"), "driver")
    program_spec = dict(_mapping(driver_spec.get("program"), "driver.program", required=False))
    authored = _mapping(facts.get("authored_spectrum"), "frozen_facts.authored_spectrum")
    for key in ("seed", "t0_s", "line_count", "transient_discard_s"):
        program_spec.setdefault(key, authored.get(key))
    if program_spec.get("seed") is None or program_spec.get("t0_s") is None or program_spec.get("line_count") is None or program_spec.get("transient_discard_s") is None:
        raise V6ProtocolStateError("driver measurement/program fields are incomplete")
    program_spec.setdefault("level_scale", 0.30)
    program_spec.setdefault("ramp_type", "quintic_smoothstep")
    program_spec.setdefault("program_frame", "nominal_deck_frame")
    program_spec["seed"] = _integer(program_spec["seed"], "driver.program.seed", minimum=0)
    program_spec["line_count"] = _integer(program_spec["line_count"], "driver.program.line_count", positive=True)
    program_spec["t0_s"] = _number(program_spec["t0_s"], "driver.program.t0_s")
    program_spec["transient_discard_s"] = _number(program_spec["transient_discard_s"], "driver.program.transient_discard_s", nonnegative=True)
    program_spec["level_scale"] = _number(program_spec["level_scale"], "driver.program.level_scale", nonnegative=True)
    driver_names = _mapping(driver_spec.get("deck"), "driver.deck", required=False)
    deck = {
        "physics_timestep_s": runtime["physics_timestep_s"],
        "mass_kg": runtime["deck_mass_kg"],
        "inertia_kg_m2": list(runtime["deck_inertia_kg_m2"]),
        "body_name": str(driver_names.get("body_name", "deck")),
        "driver_body_name": str(driver_names.get("driver_body_name", "deck_driver")),
        "freejoint_name": str(driver_names.get("freejoint_name", "deck_freejoint")),
        "driver_site_name": str(driver_names.get("driver_site_name", "deck_driver_site")),
        "site_name": str(driver_names.get("site_name", "deck_site")),
        "site_size_m": _number(driver_names.get("site_size_m", 0.01), "driver.deck.site_size_m", positive=True),
        "pos_m": list(_vector(driver_names.get("pos_m", (0.0, 0.0, 0.0)), 3, "driver.deck.pos_m")),
        "quat_wxyz": list(_vector(driver_names.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)), 4, "driver.deck.quat_wxyz")),
    }
    weld = {
        "name": str(driver_names.get("weld_name", "deck_weld")),
        "solref": list(runtime["deck_eq_solref"]),
        "solimp": list(runtime["deck_eq_solimp"]),
        "positive_solref_min_s": 2.0 * runtime["physics_timestep_s"],
    }
    hard_gates = _mapping(driver_spec.get("hard_gates"), "driver.hard_gates")
    return ResolvedDriver(
        candidate_id=candidate_id,
        gamma=gamma,
        load_case=load_case,
        integrator=runtime["integrator"],
        solver=runtime["solver"],
        solver_iterations=runtime["solver_iterations"],
        solver_tolerance=runtime["solver_tolerance"],
        deck=deck,
        weld=weld,
        program=program_spec,
        hard_gates=hard_gates,
        candidate_fields=candidate,
    )


def _derived_expected(candidate: Mapping[str, Any], key: str) -> Any:
    derived = candidate.get("derived")
    if isinstance(derived, Mapping) and key in derived:
        return derived[key]
    for name in (key, "derived_" + key, "expected_" + key):
        if name in candidate:
            return candidate[name]
    return None


def _isolator_child(protocol: Mapping[str, Any], row: Mapping[str, Any], candidate_id: str, driver_candidate_id: str) -> ResolvedIsolator:
    candidate = _find_candidate(protocol, "isolator_candidates", candidate_id)
    fn_hz = _vector(candidate.get("fn_hz"), 6, f"isolator candidate {candidate_id}.fn_hz", positive=True)
    zeta = _vector(candidate.get("zeta"), 6, f"isolator candidate {candidate_id}.zeta", nonnegative=True)
    facts = _facts(protocol)
    mass = _number(candidate.get("mass_kg", facts.get("table_mass_kg")), "isolator.mass_kg", positive=True)
    inertia = _vector(candidate.get("inertia_kg_m2", facts.get("table_inertia_kg_m2")), 3, "isolator.inertia_kg_m2", positive=True)
    gravity = _number(candidate.get("gravity_m_s2", facts.get("gravity_m_s2")), "isolator.gravity_m_s2", positive=True)
    spec = _mapping(protocol.get("isolator"), "isolator")
    travel = _vector(candidate.get("travel_limits_m", spec.get("travel_limits_m", DEFAULT_TRAVEL_LIMITS_M)), 3, "isolator.travel_limits_m", positive=True)
    angles = _vector(candidate.get("angle_limits_rad", spec.get("angle_limits_rad", DEFAULT_ANGLE_LIMITS_RAD)), 3, "isolator.angle_limits_rad", positive=True)
    parameters = derive_isolator_parameters(
        IsolatorConfig(
            fn_hz=fn_hz,
            zeta=zeta,
            mass_kg=mass,
            inertia_kg_m2=inertia,
            gravity_m_s2=gravity,
            travel_limits_m=travel,
            angle_limits_rad=angles,
        )
    )
    for key, value in (("k", parameters.stiffness), ("c", parameters.damping), ("springref", parameters.springref)):
        expected = _derived_expected(candidate, key)
        if expected is not None:
            _same_vector(value, _vector(expected, 6, f"isolator candidate {candidate_id}.{key}"), f"isolator {candidate_id}.{key}")
    regions = _mapping(spec.get("regions"), "isolator.regions")
    combined = _mapping(spec.get("combined_spectrum"), "isolator.combined_spectrum")
    payload = _mapping(spec.get("payload"), "isolator.payload")
    equilibrium = _mapping(spec.get("equilibrium", payload), "isolator.equilibrium")
    hard_gates = _mapping(spec.get("hard_gates"), "isolator.hard_gates")
    scoring = _mapping(spec.get("scoring"), "isolator.scoring")
    return ResolvedIsolator(
        candidate_id=candidate_id,
        driver_candidate_id=driver_candidate_id,
        fn_hz=fn_hz,
        zeta=zeta,
        effective_mass=tuple(parameters.effective_mass),
        omega_n_rad_s=tuple(parameters.omega_n_rad_s),
        k=tuple(parameters.stiffness),
        c=tuple(parameters.damping),
        springref=tuple(parameters.springref),
        mass_kg=mass,
        inertia_kg_m2=inertia,
        gravity_m_s2=gravity,
        travel_limits_m=travel,
        angle_limits_rad=angles,
        transfer_regions=regions,
        combined_spectrum=combined,
        payload=payload,
        equilibrium=equilibrium,
        hard_gates=hard_gates,
        scoring=scoring,
        candidate_fields=candidate,
    )


def _contact_child(protocol: Mapping[str, Any], row: Mapping[str, Any], candidate_id: str, driver_candidate_id: str, isolator_candidate_id: str) -> ResolvedContact:
    candidate = _find_candidate(protocol, "contact_candidates", candidate_id)
    contact = _mapping(protocol.get("contact"), "contact")
    facts = _facts(protocol)
    # These are candidate-owned in V6.  The frozen-facts fallback only keeps
    # synthetic pre-registration fixtures readable; the V6 validator requires
    # the fields in every contact candidate row.
    condim = _integer(candidate.get("condim"), f"contact candidate {candidate_id}.condim", positive=True)
    sliding = candidate.get("sliding_mu", facts.get("sliding_mu"))
    sliding = _mapping(sliding, f"contact candidate {candidate_id}.sliding_mu")
    torsional = _number(candidate.get("torsional_mu"), f"contact candidate {candidate_id}.torsional_mu", nonnegative=True)
    rolling = _number(candidate.get("rolling_mu"), f"contact candidate {candidate_id}.rolling_mu", nonnegative=True)
    margin = _number(candidate.get("margin_m"), f"contact candidate {candidate_id}.margin_m", nonnegative=True)
    gap = _number(candidate.get("gap_m"), f"contact candidate {candidate_id}.gap_m", nonnegative=True)
    solref = _vector(candidate.get("solref"), 2, f"contact candidate {candidate_id}.solref", positive=True)
    solimp = _vector(candidate.get("solimp"), 5, f"contact candidate {candidate_id}.solimp")
    iterations = _integer(candidate.get("iterations"), f"contact candidate {candidate_id}.iterations", positive=True)
    interfaces = candidate.get("interfaces", contact.get("interfaces"))
    if isinstance(interfaces, (str, bytes)) or not isinstance(interfaces, Sequence) or not interfaces:
        raise V6ProtocolStateError(f"contact candidate {candidate_id}.interfaces must be a nonempty list")
    probes = _mapping(contact.get("probes"), "contact.probes")
    hard_gates = _mapping(contact.get("hard_gates"), "contact.hard_gates")
    scoring = _mapping(contact.get("scoring"), "contact.scoring")
    return ResolvedContact(
        candidate_id=candidate_id,
        driver_candidate_id=driver_candidate_id,
        isolator_candidate_id=isolator_candidate_id,
        condim=condim,
        sliding_mu=sliding,
        torsional_mu=torsional,
        rolling_mu=rolling,
        margin_m=margin,
        gap_m=gap,
        solref=solref,
        solimp=solimp,
        iterations=iterations,
        interfaces=tuple(str(item) for item in interfaces),
        probes=probes,
        hard_gates=hard_gates,
        scoring=scoring,
        candidate_fields=candidate,
    )


def _parity_child(protocol: Mapping[str, Any], row: Mapping[str, Any], driver_id: str, isolator_id: str, contact_id: str) -> ResolvedParity:
    spec = _mapping(protocol.get("parity"), "parity")
    static_duration = _number(spec.get("static_duration_s", spec.get("duration_s")), "parity.static_duration_s", positive=True)
    dynamic_duration = _number(spec.get("dynamic_duration_s", spec.get("duration_s")), "parity.dynamic_duration_s", positive=True)
    tolerance = spec.get("geometry_tolerance_m", spec.get("geometry_absolute_m"))
    tolerance = _number(tolerance, "parity.geometry_tolerance_m", nonnegative=True)
    action_exact = spec.get("action_dimension_exact")
    if not isinstance(action_exact, bool):
        raise V6ProtocolStateError("parity.action_dimension_exact must be boolean")
    action_dim = _integer(spec.get("action_dimension", 7), "parity.action_dimension", positive=True)
    support_min = _number(spec.get("support_force_min_N", spec.get("passive_support_force_min_N")), "parity.support_force_min_N", positive=True)
    support_tolerance = _number(spec.get("support_force_tolerance_N", support_min), "parity.support_force_tolerance_N", nonnegative=True)
    return ResolvedParity(
        candidate_id=str(row.get("candidate_id", "gamma_zero_parity")),
        driver_candidate_id=driver_id,
        isolator_candidate_id=isolator_id,
        contact_candidate_id=contact_id,
        static_duration_s=static_duration,
        dynamic_duration_s=dynamic_duration,
        geometry_tolerance_m=tolerance,
        action_dimension_exact=action_exact,
        action_dimension=action_dim,
        support_force_min_N=support_min,
        support_force_tolerance_N=support_tolerance,
        expected_geometry=_mapping(spec.get("expected_geometry"), "parity.expected_geometry"),
        expected_action=_mapping(spec.get("expected_action"), "parity.expected_action"),
        expected_support=_mapping(spec.get("expected_support"), "parity.expected_support", required=False),
        candidate_fields=spec,
    )


def _replay_child(protocol: Mapping[str, Any], row: Mapping[str, Any], common: ResolvedCommon, context: Mapping[str, Any] | None) -> ResolvedReplay:
    spec = _mapping(protocol.get("replay"), "replay")
    selected_component = row.get("selected_component", row.get("component"))
    if selected_component not in {"driver", "isolator", "contact", "parity", "gamma_zero_parity"}:
        raise V6ProtocolStateError("replay.selected_component is missing or unsupported")
    binding = row.get("binding", row.get("selection_binding"))
    if not isinstance(binding, Mapping) or not binding:
        raise V6ProtocolStateError("replay binding is required")
    binding = dict(binding)
    context = context or {}
    for key, value in context.items():
        if key.endswith("_candidate_id") and key in binding and str(binding[key]) != str(value):
            raise V6ProtocolStateError(f"replay binding disagrees with selected upstream state: {key}")
    selected_key = {
        "driver": "driver_candidate_id",
        "isolator": "isolator_candidate_id",
        "contact": "contact_candidate_id",
        "parity": "contact_candidate_id",
        "gamma_zero_parity": "contact_candidate_id",
    }[selected_component]
    selected_id = binding.get(selected_key)
    if selected_id is None:
        selected_id = binding.get("selected_candidate_id", row.get("selected_candidate_id"))
    selected_id = _required_string(str(selected_id) if selected_id is not None else "", "replay.selected_candidate_id")
    process_count = _integer(row.get("process_count", spec.get("process_count", 3)), "replay.process_count", minimum=3)
    process_index = _integer(row.get("process_index"), "replay.process_index", minimum=1)
    if process_index > process_count:
        raise V6ProtocolStateError("replay.process_index exceeds replay.process_count")
    trace_fields = row.get("trace_fields", spec.get("trace_fields", ()))
    if isinstance(trace_fields, (str, bytes)) or not isinstance(trace_fields, Sequence) or not trace_fields:
        raise V6ProtocolStateError("replay complete trace schema is missing")
    trace_fields = tuple(_required_string(str(item), "replay.trace_fields[]") for item in trace_fields)
    tolerances = _mapping(spec.get("determinism_tolerances", spec.get("tolerances", {})), "replay.determinism_tolerances", required=False)
    return ResolvedReplay(
        selected_component=str(selected_component),
        selected_candidate_id=selected_id,
        binding=binding,
        process_index=process_index,
        process_count=process_count,
        trace_schema_id=common.trace_schema_id,
        trace_schema_version=common.trace_schema_version,
        trace_fields=trace_fields,
        determinism_tolerances=tolerances,
        candidate_fields=row,
    )


def _resolve_row(protocol: Mapping[str, Any], row: Mapping[str, Any], context: Mapping[str, Any] | None = None, *, state_id_override: str | None = None, output_override: str | None = None) -> ResolvedProbeState:
    row = dict(row)
    if state_id_override is not None:
        row["state_id"] = state_id_override
    if output_override is not None:
        row["output"] = output_override
    stage = row.get("stage")
    if stage not in {"driver", "isolator", "contact", "parity", "replay"}:
        raise V6ProtocolStateError(f"unsupported stage: {stage}")
    common, runtime, runtime_driver_id = _common(protocol, row, str(stage), context)
    driver = isolator = contact = parity = replay = None
    if stage == "driver":
        driver = _driver_child(protocol, row, runtime, str(runtime_driver_id))
    elif stage == "isolator":
        driver_id = _candidate_id_from(row, context, "driver_candidate_id", runtime_driver_id)
        isolator_id = _required_string(str(row.get("candidate_id", "")), "isolator candidate_id")
        isolator = _isolator_child(protocol, row, isolator_id, driver_id)
    elif stage == "contact":
        driver_id = _candidate_id_from(row, context, "driver_candidate_id", runtime_driver_id)
        selection = _mapping(protocol.get("selection"), "selection", required=False)
        defaults = _mapping(selection.get("default"), "selection.default", required=False)
        isolator_id = _candidate_id_from(row, context, "isolator_candidate_id", defaults.get("isolator_candidate_id", defaults.get("isolator")))
        contact_id = _required_string(str(row.get("candidate_id", "")), "contact candidate_id")
        isolator = _isolator_child(protocol, row, isolator_id, driver_id)
        contact = _contact_child(protocol, row, contact_id, driver_id, isolator_id)
    elif stage == "parity":
        selection = _mapping(protocol.get("selection"), "selection", required=False)
        defaults = _mapping(selection.get("default"), "selection.default", required=False)
        driver_id = _candidate_id_from(row, context, "driver_candidate_id", runtime_driver_id)
        isolator_id = _candidate_id_from(row, context, "isolator_candidate_id", defaults.get("isolator_candidate_id", defaults.get("isolator")))
        contact_id = _candidate_id_from(row, context, "contact_candidate_id", defaults.get("contact_candidate_id", defaults.get("contact")))
        isolator = _isolator_child(protocol, row, isolator_id, driver_id)
        contact = _contact_child(protocol, row, contact_id, driver_id, isolator_id)
        parity = _parity_child(protocol, row, driver_id, isolator_id, contact_id)
    else:
        replay = _replay_child(protocol, row, common, context)
        binding = dict(replay.binding)
        driver_id = binding.get("driver_candidate_id")
        isolator_id = binding.get("isolator_candidate_id")
        contact_id = binding.get("contact_candidate_id")
        if replay.selected_component == "driver" and driver_id is not None:
            driver_row = {
                "state_id": common.state_id,
                "stage": "driver",
                "candidate_id": driver_id,
                "gamma": row.get("gamma", 0.15),
                "load_case": row.get("load_case", "empty"),
                "duration_s": common.duration_s,
                "output": common.output,
                "retry_policy": common.retry_policy,
            }
            driver = _driver_child(protocol, driver_row, runtime, str(driver_id))
        if replay.selected_component == "isolator" and isolator_id is not None:
            isolator = _isolator_child(protocol, row, str(isolator_id), str(driver_id or runtime_driver_id))
        if replay.selected_component == "contact" and contact_id is not None:
            isolator = _isolator_child(protocol, row, str(isolator_id), str(driver_id or runtime_driver_id))
            contact = _contact_child(protocol, row, str(contact_id), str(driver_id or runtime_driver_id), str(isolator_id))
        if replay.selected_component in {"parity", "gamma_zero_parity"}:
            isolator = _isolator_child(protocol, row, str(isolator_id), str(driver_id or runtime_driver_id))
            contact = _contact_child(protocol, row, str(contact_id), str(driver_id or runtime_driver_id), str(isolator_id))
            parity = _parity_child(protocol, row, str(driver_id or runtime_driver_id), str(isolator_id), str(contact_id))
    return ResolvedProbeState(common=common, driver=driver, isolator=isolator, contact=contact, parity=parity, replay=replay)


def resolve_protocol_state_v6(protocol: Mapping[str, Any], state_id: str, selection_context: Mapping[str, Any] | None = None) -> ResolvedProbeState:
    """Resolve one V6 state and deep-freeze every runtime input."""

    if not isinstance(protocol, Mapping):
        raise V6ProtocolStateError("protocol must be a mapping")
    if not isinstance(state_id, str) or not state_id:
        raise V6ProtocolStateError("state_id must be a nonempty string")
    return _resolve_row(protocol, _manifest_row(protocol, state_id), selection_context)


def resolve_all_protocol_states_v6(protocol: Mapping[str, Any], selection_context: Mapping[str, Any] | None = None) -> tuple[ResolvedProbeState, ...]:
    rows = protocol.get("execution_manifest")
    if not isinstance(rows, list):
        raise V6ProtocolStateError("execution_manifest must be a list")
    states = tuple(resolve_protocol_state_v6(protocol, str(row.get("state_id")), selection_context) for row in rows)
    if len({state.state_id for state in states}) != len(states):
        raise V6ProtocolStateError("resolved state IDs must be unique")
    if len({state.output for state in states}) != len(states):
        raise V6ProtocolStateError("resolved artifact outputs must be unique")
    return states


def legal_replay_binding_templates(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Enumerate every candidate binding a replay adapter may legally consume."""

    replay = _mapping(protocol.get("replay"), "replay")
    explicit = replay.get("legal_bindings")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise V6ProtocolStateError("replay.legal_bindings must be a list")
        templates = []
        for item in explicit:
            if not isinstance(item, Mapping):
                raise V6ProtocolStateError("each replay legal binding must be a mapping")
            templates.append(dict(item))
        if not templates:
            raise V6ProtocolStateError("replay.legal_bindings must not be empty")
        return tuple(templates)
    drivers = [_required_string(str(row.get("candidate_id", "")), "driver candidate_id") for row in _mapping(protocol.get("driver"), "driver")["convergence_candidates"]]
    isolators = [_required_string(str(row.get("candidate_id", "")), "isolator candidate_id") for row in _mapping(protocol.get("components"), "components")["isolator_candidates"]]
    contacts = [_required_string(str(row.get("candidate_id", "")), "contact candidate_id") for row in _mapping(protocol.get("components"), "components")["contact_candidates"]]
    templates: list[dict[str, Any]] = []
    for driver_id in drivers:
        templates.append({"selected_component": "driver", "binding": {"driver_candidate_id": driver_id}})
    for driver_id, isolator_id in itertools.product(drivers, isolators):
        templates.append({"selected_component": "isolator", "binding": {"driver_candidate_id": driver_id, "isolator_candidate_id": isolator_id}})
    for driver_id, isolator_id, contact_id in itertools.product(drivers, isolators, contacts):
        binding = {"driver_candidate_id": driver_id, "isolator_candidate_id": isolator_id, "contact_candidate_id": contact_id}
        templates.append({"selected_component": "contact", "binding": binding})
        templates.append({"selected_component": "gamma_zero_parity", "binding": binding})
    return tuple(templates)


def resolve_replay_binding_templates(protocol: Mapping[str, Any], *, selection_context: Mapping[str, Any] | None = None) -> tuple[ResolvedProbeState, ...]:
    rows = protocol.get("execution_manifest")
    if not isinstance(rows, list):
        raise V6ProtocolStateError("execution_manifest must be a list")
    replay_rows = [row for row in rows if isinstance(row, Mapping) and row.get("stage") == "replay"]
    if not replay_rows:
        raise V6ProtocolStateError("execution_manifest has no replay template row")
    base = dict(replay_rows[0])
    states = []
    for index, template in enumerate(legal_replay_binding_templates(protocol)):
        row = dict(base)
        row.update(template)
        row["process_index"] = 1
        row["process_count"] = int(_mapping(protocol.get("replay"), "replay").get("process_count", 3))
        row["state_id"] = f"{base.get('state_id', 'replay.template')}.binding_{index + 1}"
        output = f"{base.get('output', 'shakebench_phase_06r5_v6_raw_replay_template.json').rsplit('.', 1)[0]}_binding_{index + 1}.json"
        states.append(_resolve_row(protocol, row, selection_context, state_id_override=row["state_id"], output_override=output))
    return tuple(states)


__all__ = [
    "ResolvedCommon",
    "ResolvedDriver",
    "ResolvedIsolator",
    "ResolvedContact",
    "ResolvedParity",
    "ResolvedReplay",
    "ResolvedProbeState",
    "V6ProtocolStateError",
    "canonical_json",
    "sha256_json",
    "legal_replay_binding_templates",
    "resolve_protocol_state_v6",
    "resolve_all_protocol_states_v6",
    "resolve_replay_binding_templates",
]
