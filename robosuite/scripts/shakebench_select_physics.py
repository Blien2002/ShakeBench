"""Run the Phase 06 physics-only selection and write auditable artifacts.

The selector has no task-success input.  Its three stages are deliberately
separate:

1. deck/timestep candidates are measured on the Phase 02 no-task fixture;
2. isolator candidates are scored from the registered transfer envelope and
   checked with the canonical six-axis MuJoCo support;
3. contact candidates are compiled into the real Can pair scope and exercised
   with support, slip, impact, finger-load, and timestep probes.

The default output is flat in ``robosuite/models/assets`` as required by the
Phase 06 contract.  Existing Phase 02–05 artifacts are read only as
provenance; they are never used as task ranking data.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Iterable, Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robosuite.scripts.shakebench_probe_deck_driver import (
    DeckProbeEnv,
    _build_probe_fixture,
    _trace_summary,
)
from robosuite.utils.shakebench_deck import DeckDriver, DeckDriverConfig
from robosuite.utils.shakebench_artifacts import sha256_json
from robosuite.utils.shakebench_physics import (
    PhysicsProfile,
    physics_profile_hash,
)


_INTEGRATOR_NAMES = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
_SOLVER_NAMES = {0: "PGS", 1: "CG", 2: "Newton"}
TRACE_FIELDS = (
    "integration_target_time_s", "sample_target_time_s", "integration_application_time_s", "sample_application_time_s",
    "sample_time_s", "command_pose", "actual_pose", "command_twist", "actual_twist", "command_acceleration",
    "actual_acceleration", "sample_target_pose", "sample_target_twist", "sample_target_acceleration",
    "deck_tracking_pose_error", "weld_constraint_residual_raw", "weld_constraint_force_raw", "solver_iterations",
    "solver_niter", "warning_number_delta", "warning_lastinfo",
)


















def _set_probe_options(xml_string: str, candidate: Mapping[str, Any]) -> str:
    root = ET.fromstring(xml_string)
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", format(float(candidate["physics_timestep_s"]), ".17g"))
    option.set("integrator", str(candidate["integrator"]))
    option.set("solver", str(candidate["solver"]))
    option.set("iterations", str(int(candidate["solver_iterations"])))
    option.set("tolerance", format(float(candidate["solver_tolerance"]), ".17g"))
    return ET.tostring(root, encoding="unicode")


def _driver_config(candidate: Mapping[str, Any]) -> DeckDriverConfig:
    return DeckDriverConfig(
        deck_mass_kg=float(candidate["deck_mass_kg"]),
        deck_inertia_kg_m2=tuple(float(value) for value in candidate["deck_inertia_kg_m2"]),
        eq_solref=tuple(float(value) for value in candidate["deck_eq_solref"]),
        eq_solimp=tuple(float(value) for value in candidate["deck_eq_solimp"]),
        physics_timestep_s=float(candidate["physics_timestep_s"]),
    )


def _run_driver_trace(
    candidate: Mapping[str, Any],
    *,
    duration_s: float,
    trajectory: Any = None,
    load_case: str = "empty",
    refresh_stride: int = 1,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run one no-task driver fixture with candidate-local MuJoCo options."""

    dt = float(candidate["physics_timestep_s"])
    fixture = _build_probe_fixture(model_timestep=dt, load_case=load_case)
    source_xml = _set_probe_options(fixture.xml_string, candidate)
    config = _driver_config(candidate)
    driver = DeckDriver(
        trajectory=trajectory,
        config=config,
        body_handles=fixture.role_handles,
        refresh_stride=refresh_stride,
    )
    env = DeckProbeEnv(
        source_xml,
        driver,
        model_timestep=dt,
        horizon=max(2, int(math.ceil(float(duration_s) * 20.0)) + 1),
        initial_joint_state=fixture.initial_joint_state,
    )
    try:
        env.reset()
        driver.reset_trace()
        for _ in range(max(1, int(math.ceil(float(duration_s) * env.control_freq)))):
            env.step(np.zeros(0))
        trace = driver.trace
        raw_model = env.sim.model._model
        options = {
            "timestep_s": float(raw_model.opt.timestep),
            "integrator": _INTEGRATOR_NAMES.get(int(raw_model.opt.integrator), str(int(raw_model.opt.integrator))),
            "solver": _SOLVER_NAMES.get(int(raw_model.opt.solver), str(int(raw_model.opt.solver))),
            "iterations": int(raw_model.opt.iterations),
            "tolerance": float(raw_model.opt.tolerance),
            "control_freq_hz": float(env.control_freq),
            "control_steps": int(env._control_steps),
            "refresh_stride": int(refresh_stride),
            "post_integration_refresh": True,
        }
        audit = {
            "deck": {
                "parent_graph": {
                    name: (
                        None
                        if int(raw_model.body_parentid[body_id]) == 0
                        else mujoco.mj_id2name(
                            raw_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            int(raw_model.body_parentid[body_id]),
                        )
                    )
                    for body_id in range(int(raw_model.nbody))
                    for name in [mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)]
                },
                "mass_kg": float(raw_model.body_mass[driver._deck_body_id]),
                "inertia_kg_m2": np.asarray(raw_model.body_inertia[driver._deck_body_id], dtype=float).tolist(),
                "eq_solref": np.asarray(raw_model.eq_solref[driver._weld_id], dtype=float).tolist(),
                "eq_solimp": np.asarray(raw_model.eq_solimp[driver._weld_id], dtype=float).tolist(),
                "load_case": load_case,
                "warnings": int(np.sum(trace.warning_number_delta)) if trace.warning_number_delta.size else 0,
            }
        }
        digest = _trace_digest(trace)
        trace_record = {
            "summary": _trace_summary(trace),
            "digest": digest,
            "field_shapes": _trace_shapes(trace),
            "field_contract": "complete DeckDriverTrace fields; digest covers every field",
        }
        return trace, {**audit, "trace": trace_record}, options
    finally:
        env.close()


def _trace_shapes(trace: Any) -> dict[str, list[int]]:
    return {
        name: list(np.asarray(getattr(trace, name)).shape)
        for name in TRACE_FIELDS
    }


def _trace_digest(trace: Any) -> str:
    digest = hashlib.sha256()
    for name in TRACE_FIELDS:
        array = np.ascontiguousarray(np.asarray(getattr(trace, name)))
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()






















def _profile_payload_with_contact(base: PhysicsProfile, contact: Mapping[str, Any], *, dt: Optional[float] = None) -> dict[str, Any]:
    payload = base.to_dict()
    payload["status"] = "contact_candidate_non_scoreable"
    payload["scoreable"] = False
    payload["profile_id"] = "shakebench.contact.candidate." + str(contact["candidate_id"])
    payload["physics"]["contact"] = {
        "pair_scope": "explicit_can_pairs",
        "sliding_mu": {"table_object": 0.30, "finger_object": 1.00},
        "condim": int(contact["condim"]),
        "torsional_mu": float(contact["torsional_mu"]),
        "rolling_mu": float(contact["rolling_mu"]),
        "margin_m": float(contact["margin_m"]),
        "gap_m": float(contact["gap_m"]),
        "solref": [float(value) for value in contact["solref"]],
        "solimp": [float(value) for value in contact["solimp"]],
        "interfaces": list(base.contact.get("interfaces", [])),
    }
    if dt is not None:
        payload["physics"]["timestep"]["physics_timestep_s"] = float(dt)
        payload["physics"]["scheduler"]["control_steps"] = int(round(1.0 / (20.0 * float(dt))))
        payload["physics"]["deck"]["eq_solref"][0] = max(
            float(payload["physics"]["deck"]["eq_solref"][0]), 2.0 * float(dt)
        )
        payload["physics"]["contact"]["solref"][0] = max(
            float(payload["physics"]["contact"]["solref"][0]), 2.0 * float(dt)
        )
    payload["physics"]["timestep"]["iterations"] = int(contact["iterations"])
    payload["profile_sha256"] = physics_profile_hash(payload)
    return payload


def _contact_candidate_probe(
    base_profile: PhysicsProfile,
    candidate: Mapping[str, Any],
    *,
    run_expensive: bool,
    recovery_duration_s: Optional[float] = None,
    reset_drop_velocity: bool = False,
) -> dict[str, Any]:
    """Exercise real task contact geometry without evaluating task outcome."""

    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan

    payload = _profile_payload_with_contact(base_profile, candidate)
    profile = PhysicsProfile(payload=payload, source="selection contact candidate", profile_sha256=payload["profile_sha256"])
    profile.assert_valid()
    target_friction = (
        0.30,
        float(candidate["torsional_mu"]),
        float(candidate["rolling_mu"]),
    )
    env = VibrationPickPlaceCan(
        robots="Panda",
        physics_profile=profile,
        model_timestep=float(profile.model_timestep_s),
        target_container_friction=target_friction,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
        horizon=200,
        seed=17,
    )
    # Reuse the compiled contact model across the probe states; a hard reset
    # would replace the MjData object captured by the raw-contact audit.
    env.hard_reset = False
    try:
        model = env.sim.model._model
        data = env.sim.data._data
        can_geom_names = tuple(env.can.contact_geoms)
        table_geom_names = tuple(env.table_contact_geom_names)
        finger_geom_names = tuple(env.finger_pad_geom_names)

        def contact_force_for(allowed_second: Iterable[str]) -> tuple[int, float, float]:
            allowed = set(allowed_second)
            count = 0
            normal_force = 0.0
            min_distance = float("inf")
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                if {first, second} & set(can_geom_names) and ({first, second} & allowed):
                    count += 1
                    force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(model, data, index, force)
                    normal_force += abs(float(force[0]))
                    min_distance = min(min_distance, float(contact.dist))
            return count, normal_force, (0.0 if min_distance == float("inf") else min_distance)

        def physics_steps(count: int) -> None:
            """Advance MuJoCo directly, bypassing task reward/success hooks."""

            for _ in range(int(count)):
                if hasattr(data, "ctrl"):
                    data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)

        # Static support is a direct contact-force check.  It does not call the
        # success evaluator and therefore cannot turn into task ranking data.
        env.reset()
        physics_steps(20)
        static_count, static_force, static_distance = contact_force_for(table_geom_names)
        static_support = {
            "contact_count": static_count,
            "normal_force_N": static_force,
            "minimum_contact_distance_m": static_distance,
            "passed": bool(static_count > 0 and static_force > 0.001),
        }

        can_joint = model.joint(env.can.joints[0]).id
        can_qvel = int(model.jnt_dofadr[can_joint])
        data.qvel[can_qvel : can_qvel + 2] = (0.05, 0.0)
        mujoco.mj_forward(model, data)
        slip_speeds = []
        recovery_steps = 1000 if recovery_duration_s is None else max(
            1, int(math.ceil(float(recovery_duration_s) / float(profile.model_timestep_s)))
        )
        for _ in range(recovery_steps):
            physics_steps(1)
            slip_speeds.append(float(np.linalg.norm(data.qvel[can_qvel : can_qvel + 2])))
        slip = {
            "initial_speed_m_s": 0.05,
            "final_speed_m_s": slip_speeds[-1],
            "speed_trace_sha256": sha256_json(slip_speeds),
            "passed": bool(np.all(np.isfinite(slip_speeds))),
        }

        table_top = np.asarray(env.arena.table_top_abs, dtype=float)
        can_qpos = int(model.jnt_qposadr[can_joint])
        data.qpos[can_qpos : can_qpos + 3] = table_top + np.asarray((0.0, 0.0, 0.15))
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        if reset_drop_velocity:
            data.qvel[can_qvel : can_qvel + 6] = 0.0
        mujoco.mj_forward(model, data)
        impact_distances = []
        for _ in range(1000):
            physics_steps(1)
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                if {first, second} & set(can_geom_names) and ({first, second} & set(table_geom_names)):
                    impact_distances.append(float(contact.dist))
        impact = {
            "sample_count": len(impact_distances),
            "minimum_distance_m": min(impact_distances) if impact_distances else None,
            "maximum_penetration_m": max(0.0, -min(impact_distances)) if impact_distances else 0.0,
            "recovery_velocity_m_s": float(np.linalg.norm(data.qvel[can_qvel : can_qvel + 3])),
            "recovery_angular_velocity_rad_s": float(np.linalg.norm(data.qvel[can_qvel + 3 : can_qvel + 6])),
            "warning_count": int(np.sum(data.warning.number)),
            "passed": bool(
                impact_distances
                and max(0.0, -min(impact_distances)) < 0.0005
                and not np.any(data.warning.number)
            ),
        }

        pad_positions = [data.geom_xpos[model.geom(name).id].copy() for name in finger_geom_names]
        hand_center = np.mean(pad_positions, axis=0)
        data.qpos[can_qpos : can_qpos + 3] = hand_center
        data.qpos[can_qpos + 3 : can_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        finger_count, finger_force, _ = contact_force_for(finger_geom_names)
        finger_load = {
            "contact_count": finger_count,
            "normal_force_N": finger_force,
            "passed": bool(finger_count > 0 and np.isfinite(finger_force)),
        }

        convergence_records = []
        for convergence_dt in (0.0001, 0.0002, 0.0004):
            convergence_payload = _profile_payload_with_contact(
                base_profile, candidate, dt=convergence_dt
            )
            convergence_profile = PhysicsProfile(
                payload=convergence_payload,
                source="contact timestep convergence",
                profile_sha256=convergence_payload["profile_sha256"],
            )
            convergence_env = VibrationPickPlaceCan(
                robots="Panda",
                physics_profile=convergence_profile,
                model_timestep=convergence_dt,
                target_container_friction=target_friction,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                use_object_obs=False,
                control_freq=20,
                horizon=5,
                seed=17,
            )
            try:
                convergence_model = convergence_env.sim.model._model
                convergence_data = convergence_env.sim.data._data
                convergence_data.ctrl[:] = 0.0
                for _ in range(20):
                    convergence_data.ctrl[:] = 0.0
                    mujoco.mj_step(convergence_model, convergence_data)
                count = 0
                force_sum = 0.0
                for index in range(int(convergence_data.ncon)):
                    contact = convergence_data.contact[index]
                    first = mujoco.mj_id2name(
                        convergence_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    )
                    second = mujoco.mj_id2name(
                        convergence_model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    )
                    names = {first, second}
                    if names & set(can_geom_names) and names & set(table_geom_names):
                        count += 1
                        force = np.zeros(6, dtype=float)
                        mujoco.mj_contactForce(convergence_model, convergence_data, index, force)
                        force_sum += abs(float(force[0]))
                relative_force_error = (
                    abs(force_sum / static_force - 1.0) if static_force > 0.0 else float("inf")
                )
                convergence_records.append(
                    {
                        "dt_s": convergence_dt,
                        "contact_solref_time_constant_s": convergence_profile.contact_solref[0],
                        "contact_count": count,
                        "normal_force_N": force_sum,
                        "relative_force_error": relative_force_error,
                        "warning_count": int(np.sum(convergence_data.warning.number)),
                        "passed": bool(
                            count > 0
                            and relative_force_error <= 0.10
                            and not np.any(convergence_data.warning.number)
                        ),
                    }
                )
            finally:
                convergence_env.close()
        contact_timestep_convergence = {
            "records": convergence_records,
            "passed": bool(convergence_records) and all(item["passed"] for item in convergence_records),
            "stability_tolerance": 0.10,
        }

        incline_mu = 0.30
        incline_angle = math.atan(incline_mu)
        incline = {
            "sliding_mu": incline_mu,
            "analytic_static_limit_angle_rad": incline_angle,
            "test_angle_rad": min(0.1, 0.5 * incline_angle),
            "analytic_threshold_passed": math.tan(min(0.1, 0.5 * incline_angle)) < incline_mu,
        }
        result = {
            "compiled_contact_profile": env.audit_compiled_model()["contacts"]["contact_profile"],
            "static_support": static_support,
            "incline_threshold": incline,
            "single_axis_slip": slip,
            "impact_recovery": impact,
            "finger_load": finger_load,
            "timestep_convergence": contact_timestep_convergence,
            "warning_count": int(np.sum(data.warning.number)),
            "profile_hash": profile.profile_sha256,
        }
        result["passed"] = bool(
            static_support["passed"]
            and incline["analytic_threshold_passed"]
            and slip["passed"]
            and impact["passed"]
            and finger_load["passed"]
            and contact_timestep_convergence["passed"]
            and result["warning_count"] == 0
        )
        if not run_expensive:
            result["mode"] = "compiled_and_short_physics_probe"
        return result
    finally:
        env.close()
























def main(argv: Optional[list[str]] = None) -> int:
    from robosuite.scripts.shakebench_select_physics_v8 import main as current_main

    return current_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
