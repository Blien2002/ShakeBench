"""Phase 06R5 adapter boundary and no-physics preparation plans.

The preparation functions below are deliberately small, but they are the
same constructors used by the V6 runner before a physics call.  Their input
is a resolved immutable state only.  ``NoPhysicsBackend`` makes accidental
model creation, stepping, or environment/task access fail during the
registration-time adapter contract check.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from robosuite.utils.shakebench_protocol_v6 import ResolvedProbeState, sha256_json


class AdapterContractError(ValueError):
    """A resolved state cannot produce a complete stage probe plan."""


class NoPhysicsBackend:
    """Backend used by adapter preflight; every physics operation is illegal."""

    def __init__(self) -> None:
        self.preparation_events: list[dict[str, Any]] = []

    def record_preparation(self, stage: str, state_id: str) -> None:
        self.preparation_events.append({"stage": str(stage), "state_id": str(state_id)})

    def create_model(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("NoPhysicsBackend forbids MuJoCo model creation")

    def step(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("NoPhysicsBackend forbids physics stepping")

    def create_environment(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("NoPhysicsBackend forbids environment creation")

    def call_task_api(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("NoPhysicsBackend forbids task/environment APIs")


@dataclass(frozen=True)
class PreparedProbePlan:
    """Canonical output of one adapter's preparation path."""

    stage: str
    state_id: str
    requested_fields: tuple[str, ...]
    complete: bool
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "state_id": self.state_id,
            "requested_fields": list(self.requested_fields),
            "complete": self.complete,
            "details": _json_ready(self.details),
        }


def adapter_contract_payload(
    states: Iterable[ResolvedProbeState],
    replay_states: Iterable[ResolvedProbeState],
    *,
    schema_id: str,
    schema_version: int,
    plan_payload: Callable[[ResolvedProbeState, PreparedProbePlan, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Exercise all adapter plans without a physics backend.

    V6--V8 differ only in their resolved states and, for V8 contact plans,
    their serialized plan details.  This keeps the common preflight loop in
    one place while callers retain their protocol-specific payload shape.
    """

    states = tuple(states)
    replay_states = tuple(replay_states)
    backend = NoPhysicsBackend()
    encode = plan_payload or (lambda _state, plan, _kind: plan.to_dict())
    plans: list[dict[str, Any]] = []

    def prepare(state: ResolvedProbeState) -> PreparedProbePlan:
        handlers = {
            "driver": prepare_driver_probe,
            "isolator": prepare_isolator_probe,
            "contact": prepare_contact_probe,
            "parity": prepare_parity_probe,
            "replay": prepare_replay_probe,
        }
        try:
            plan = handlers[state.stage](state, backend)
        except KeyError as exc:
            raise AdapterContractError(f"unsupported adapter stage: {state.stage}") from exc
        if plan.complete is not True or not plan.requested_fields:
            raise AdapterContractError(f"incomplete {state.stage} adapter plan")
        return plan

    for state in states:
        plans.append({"kind": "manifest_state", "state_digest": state.resolved_state_digest, "plan": encode(state, prepare(state), "manifest_state")})
    for state in replay_states:
        replay_plan = prepare_replay_probe(state, backend)
        if state.replay is None:
            raise AdapterContractError("replay binding resolver produced no replay child")
        plans.append({"kind": "legal_replay_binding", "state_digest": state.resolved_state_digest, "plan": encode(state, replay_plan, "legal_replay_binding")})
        component = state.replay.selected_component
        component_state = state
        handlers = {
            "driver": prepare_driver_probe,
            "isolator": prepare_isolator_probe,
            "contact": prepare_contact_probe,
            "parity": prepare_parity_probe,
        }
        plan = handlers.get(component, prepare_parity_probe)(component_state, backend)
        if plan.complete is not True:
            raise AdapterContractError(f"incomplete replay component plan: {component}")
        plans.append({"kind": "legal_replay_component", "state_digest": state.resolved_state_digest, "plan": encode(state, plan, "legal_replay_component")})
    return {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "state_count": len(states),
        "legal_replay_binding_count": len(replay_states),
        "prepared_plan_count": len(plans),
        "adapter_contract_digest": sha256_json(plans),
        "plans": plans,
        "backend": {"type": "NoPhysicsBackend", "preparation_event_count": len(backend.preparation_events), "physics_calls": 0},
        "mujoco_model_created": False,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _finish(state: ResolvedProbeState, fields: tuple[str, ...], details: Mapping[str, Any], backend: NoPhysicsBackend | None) -> PreparedProbePlan:
    if backend is not None:
        backend.record_preparation(state.stage, state.state_id)
    return PreparedProbePlan(state.stage, state.state_id, fields, True, dict(details))


def prepare_driver_probe(state: ResolvedProbeState, backend: NoPhysicsBackend | None = None) -> PreparedProbePlan:
    """Prepare the real deck driver/XML path without compiling a model."""

    if state.stage not in {"driver", "replay"} or state.driver is None:
        raise AdapterContractError("driver adapter requires a driver resolved state")
    driver = state.driver
    common = state.common
    from robosuite.utils.shakebench_deck import DeckDriver, DeckDriverConfig

    config = DeckDriverConfig(
        driver_body_name=str(driver.deck["driver_body_name"]),
        deck_body_name=str(driver.deck["body_name"]),
        deck_freejoint_name=str(driver.deck["freejoint_name"]),
        weld_name=str(driver.weld["name"]),
        driver_site_name=str(driver.deck["driver_site_name"]),
        deck_site_name=str(driver.deck["site_name"]),
        deck_mass_kg=driver.deck_mass_kg,
        deck_inertia_kg_m2=driver.deck_inertia_kg_m2,
        deck_pos_m=tuple(driver.deck["pos_m"]),
        deck_quat_wxyz=tuple(driver.deck["quat_wxyz"]),
        eq_solref=tuple(driver.weld["solref"]),
        eq_solimp=tuple(driver.weld["solimp"]),
        physics_timestep_s=common.physics_timestep_s,
        site_size_m=float(driver.deck["site_size_m"]),
    )
    deck_driver = DeckDriver(config=config, refresh_stride=common.refresh_stride)
    import xml.etree.ElementTree as ET

    source_xml = ET.tostring(ET.Element("mujoco", {"model": "shakebench_v6_preflight"}), encoding="unicode")
    root = ET.fromstring(source_xml)
    ET.SubElement(root, "worldbody")
    # The real XML processor is invoked here; compilation is intentionally not.
    processed_xml = deck_driver.processor(ET.tostring(root, encoding="unicode"))
    if "shakebench_deck_processor" not in processed_xml:
        raise AdapterContractError("driver XML processor did not emit its audit marker")
    fields = (
        "common.physics_timestep_s",
        "common.control_steps",
        "common.measurement_rate_hz",
        "common.refresh_stride",
        "driver.deck",
        "driver.weld",
        "driver.gamma",
        "driver.load_case",
        "driver.program",
        "driver.hard_gates",
    )
    return _finish(
        state,
        fields,
        {
            "config": config.to_dict(),
            "refresh_stride": common.refresh_stride,
            "program": driver.program,
            "gamma": driver.gamma,
            "load_case": driver.load_case,
            "trace_schema": {"id": common.trace_schema_id, "version": common.trace_schema_version},
            "processed_xml_sha256": hashlib.sha256(processed_xml.encode("utf-8")).hexdigest(),
        },
        backend,
    )


def prepare_isolator_probe(state: ResolvedProbeState, backend: NoPhysicsBackend | None = None) -> PreparedProbePlan:
    """Prepare the canonical analytic/MuJoCo isolator probe inputs."""

    if state.stage not in {"isolator", "replay"} or state.isolator is None:
        raise AdapterContractError("isolator adapter requires an isolator resolved state")
    isolator = state.isolator
    from robosuite.utils.shakebench_isolator import IsolatorConfig, derive_isolator_parameters, six_axis_transfer

    config = IsolatorConfig(
        fn_hz=isolator.fn_hz,
        zeta=isolator.zeta,
        mass_kg=isolator.mass_kg,
        inertia_kg_m2=isolator.inertia_kg_m2,
        gravity_m_s2=isolator.gravity_m_s2,
        travel_limits_m=isolator.travel_limits_m,
        angle_limits_rad=isolator.angle_limits_rad,
    )
    parameters = derive_isolator_parameters(config)
    transfer = six_axis_transfer((0.2,), parameters)
    fields = (
        "common.physics_timestep_s",
        "isolator.candidate_id",
        "isolator.fn_hz",
        "isolator.zeta",
        "isolator.k",
        "isolator.c",
        "isolator.springref",
        "isolator.transfer_regions",
        "isolator.combined_spectrum",
        "isolator.payload",
        "isolator.equilibrium",
        "isolator.hard_gates",
        "isolator.scoring",
    )
    return _finish(
        state,
        fields,
        {
            "parameters": parameters.to_dict(),
            "transfer_probe_shape": list(transfer.shape),
            "transfer_regions": isolator.transfer_regions,
            "combined_spectrum": isolator.combined_spectrum,
            "payload": isolator.payload,
            "equilibrium": isolator.equilibrium,
        },
        backend,
    )


def probe_profile_from_state(state: ResolvedProbeState):
    """Build the non-scoreable candidate profile used by contact/parity probes."""

    if state.contact is None or state.isolator is None:
        raise AdapterContractError("contact profile requires isolator and contact resolved children")
    from robosuite.utils.shakebench_physics import PhysicsProfile, make_probe_physics_profile, physics_profile_hash

    payload = make_probe_physics_profile().to_dict()
    physics = payload["physics"]
    physics["timestep"].update(
        {
            "physics_timestep_s": state.common.physics_timestep_s,
            "integrator": state.common.integrator,
            "solver": state.common.solver,
            "iterations": state.common.solver_iterations,
            "tolerance": state.common.solver_tolerance,
        }
    )
    physics["scheduler"]["control_steps"] = state.common.control_steps
    physics["deck"].update(
        {
            "mass_kg": state.common.deck_mass_kg,
            "inertia_kg_m2": list(state.common.deck_inertia_kg_m2),
            "eq_solref": list(state.common.deck_eq_solref),
            "eq_solimp": list(state.common.deck_eq_solimp),
        }
    )
    physics["isolator"].update(
        {
            "candidate_id": state.isolator.candidate_id,
            "fn_hz": list(state.isolator.fn_hz),
            "zeta": list(state.isolator.zeta),
            "k": list(state.isolator.k),
            "c": list(state.isolator.c),
            "springref": list(state.isolator.springref),
            "mass_kg": state.isolator.mass_kg,
            "inertia_kg_m2": list(state.isolator.inertia_kg_m2),
            "gravity_m_s2": state.isolator.gravity_m_s2,
            "travel_limits_m": list(state.isolator.travel_limits_m),
            "angle_limits_rad": list(state.isolator.angle_limits_rad),
        }
    )
    physics["contact"].update(
        {
            "condim": state.contact.condim,
            "sliding_mu": dict(state.contact.sliding_mu),
            "torsional_mu": state.contact.torsional_mu,
            "rolling_mu": state.contact.rolling_mu,
            "margin_m": state.contact.margin_m,
            "gap_m": state.contact.gap_m,
            "solref": list(state.contact.solref),
            "solimp": list(state.contact.solimp),
            "interfaces": list(state.contact.interfaces),
        }
    )
    payload["profile_id"] = f"shakebench.phase06r5.v6.probe.{state.contact.candidate_id}"
    payload["profile_sha256"] = physics_profile_hash(payload)
    return PhysicsProfile(payload=payload, source="V6 resolved contact state", profile_sha256=payload["profile_sha256"]).assert_valid()


def prepare_contact_probe(state: ResolvedProbeState, backend: NoPhysicsBackend | None = None) -> PreparedProbePlan:
    """Prepare the real candidate contact profile and pair scope."""

    if state.stage not in {"contact", "replay"} or state.contact is None:
        raise AdapterContractError("contact adapter requires a contact resolved state")
    profile = probe_profile_from_state(state)
    fields = (
        "common.physics_timestep_s",
        "contact.candidate_id",
        "contact.condim",
        "contact.sliding_mu",
        "contact.torsional_mu",
        "contact.rolling_mu",
        "contact.margin_m",
        "contact.gap_m",
        "contact.solref",
        "contact.solimp",
        "contact.iterations",
        "contact.interfaces",
        "contact.probes",
        "contact.hard_gates",
        "contact.scoring",
    )
    return _finish(state, fields, {"profile": profile.to_dict(), "interfaces": list(state.contact.interfaces)}, backend)


def prepare_parity_probe(state: ResolvedProbeState, backend: NoPhysicsBackend | None = None) -> PreparedProbePlan:
    """Prepare bounded parity geometry/action/support checks."""

    if state.stage not in {"parity", "replay"} or state.parity is None:
        raise AdapterContractError("parity adapter requires a parity resolved state")
    parity = state.parity
    if parity.action_dimension_exact and parity.action_dimension <= 0:
        raise AdapterContractError("parity action dimension is invalid")
    fields = (
        "parity.static_duration_s",
        "parity.dynamic_duration_s",
        "parity.geometry_tolerance_m",
        "parity.action_dimension_exact",
        "parity.expected_geometry",
        "parity.expected_action",
        "parity.expected_support",
    )
    return _finish(
        state,
        fields,
        {
            "static_duration_s": parity.static_duration_s,
            "dynamic_duration_s": parity.dynamic_duration_s,
            "geometry_tolerance_m": parity.geometry_tolerance_m,
            "action_dimension": parity.action_dimension,
            "expected_geometry": parity.expected_geometry,
            "expected_action": parity.expected_action,
            "expected_support": parity.expected_support,
        },
        backend,
    )


def prepare_replay_probe(state: ResolvedProbeState, backend: NoPhysicsBackend | None = None) -> PreparedProbePlan:
    """Prepare a replay binding without resolving a new protocol mapping."""

    if state.stage != "replay" or state.replay is None:
        raise AdapterContractError("replay adapter requires a replay resolved state")
    replay = state.replay
    if replay.process_count < 3 or not 1 <= replay.process_index <= replay.process_count:
        raise AdapterContractError("replay process binding is incomplete")
    if not replay.binding or not replay.trace_fields:
        raise AdapterContractError("replay binding/trace schema is incomplete")
    fields = (
        "replay.selected_component",
        "replay.selected_candidate_id",
        "replay.binding",
        "replay.process_index",
        "replay.process_count",
        "replay.trace_schema_id",
        "replay.trace_schema_version",
        "replay.trace_fields",
        "replay.determinism_tolerances",
    )
    return _finish(
        state,
        fields,
        {
            "selected_component": replay.selected_component,
            "selected_candidate_id": replay.selected_candidate_id,
            "binding": replay.binding,
            "process_index": replay.process_index,
            "process_count": replay.process_count,
            "trace_schema": {"id": replay.trace_schema_id, "version": replay.trace_schema_version},
            "trace_fields": list(replay.trace_fields),
            "binding_digest": sha256_json(replay.binding),
        },
        backend,
    )


__all__ = [
    "AdapterContractError",
    "NoPhysicsBackend",
    "PreparedProbePlan",
    "probe_profile_from_state",
    "prepare_driver_probe",
    "prepare_isolator_probe",
    "prepare_contact_probe",
    "prepare_parity_probe",
    "prepare_replay_probe",
]
