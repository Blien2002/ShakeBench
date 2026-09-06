"""Verify the seven-channel OSC_POSE + Panda gripper action path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.shakebench_run_oracle import _actuator_metadata_from_model, _digest, _json_ready
from robosuite.utils.shakebench_oracle import OracleControllerProfile

CHANNEL_NAMES = (
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "gripper",
)


def _rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-10:
        return np.zeros(3)
    axis = np.asarray(
        (rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]),
        dtype=float,
    )
    return angle * axis / np.linalg.norm(axis)


def _bounds_ok(values: np.ndarray, metadata: list[dict[str, Any]], field: str) -> bool:
    for value, actuator in zip(values, metadata):
        if not np.isfinite(value):
            return False
        limited = actuator["ctrllimited"] if field == "ctrl" else actuator["forcelimited"]
        if limited and not (actuator[f"{field}range"][0] - 1.0e-9 <= value <= actuator[f"{field}range"][1] + 1.0e-9):
            return False
    return True


def run_actuator_positive_controls(output: str | Path | None = None) -> dict[str, Any]:
    profile = OracleControllerProfile()
    env = robosuite.make(
        "VibrationPickPlaceCan",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        observation_tier="V0",
        imu_mode="ideal_smoke",
        horizon=2,
        ignore_done=True,
        seed=0,
    )
    results: list[dict[str, Any]] = []
    try:
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        actuator_metadata = _actuator_metadata_from_model(raw_model)
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            for sign in (-1, 1):
                observation = env.reset()
                arm = env.robots[0].part_controllers["right"]
                action = np.zeros(7, dtype=np.float32)
                action[channel_index] = sign
                decoded = action.astype(float).copy()
                decoded[:3] *= profile.position_action_range_m
                decoded[3:6] *= profile.orientation_action_range_rad
                scaled_arm_action = np.asarray(arm.scale_action(action[:6]), dtype=float)
                expected_pos = np.asarray(arm.compute_goal_pos(scaled_arm_action[:3]), dtype=float)
                expected_ori = np.asarray(arm.compute_goal_ori(scaled_arm_action[3:6]), dtype=float)
                for _ in range(20 if channel_index == 6 else 1):
                    _, _, _, _ = env.step(action)
                after_pos = np.asarray(arm.goal_pos, dtype=float).copy()
                after_ori = np.asarray(arm.goal_ori, dtype=float).copy()
                ctrl = np.asarray(env.sim.data.ctrl, dtype=float).copy()
                force = np.asarray(env.sim.data.actuator_force, dtype=float).copy()
                position_delta = after_pos - expected_pos
                rotation_delta = after_ori - expected_ori
                if channel_index < 3:
                    target_response = float(scaled_arm_action[channel_index] * sign)
                    non_target_response = float(np.linalg.norm(np.delete(scaled_arm_action[:3], channel_index)))
                    finite_response = (
                        np.isfinite(target_response)
                        and target_response > 0.0
                        and np.allclose(after_pos, expected_pos, rtol=0.0, atol=1.0e-10)
                    )
                    no_mapping_swap = (
                        non_target_response <= 1.0e-10 and np.linalg.norm(scaled_arm_action[3:]) <= 1.0e-10
                    )
                elif channel_index < 6:
                    axis = channel_index - 3
                    target_response = float(scaled_arm_action[3 + axis] * sign)
                    non_target_response = float(np.linalg.norm(np.delete(scaled_arm_action[3:], axis)))
                    finite_response = (
                        np.isfinite(target_response)
                        and target_response > 0.0
                        and np.allclose(after_ori, expected_ori, rtol=0.0, atol=1.0e-10)
                    )
                    no_mapping_swap = (
                        non_target_response <= 1.0e-10 and np.linalg.norm(scaled_arm_action[:3]) <= 1.0e-10
                    )
                else:
                    gripper_ctrl = ctrl[-2:]
                    # The compiled Panda pair uses opposite signs for the two
                    # symmetric finger actuators.  Positive normalized action
                    # is close; negative normalized action is open.
                    finite_response = bool(np.all(np.isfinite(gripper_ctrl))) and (
                        (gripper_ctrl[0] > 0.0 and gripper_ctrl[1] < 0.0)
                        if sign < 0
                        else np.all(np.abs(gripper_ctrl) <= 1.0e-9)
                    )
                    target_response = float(abs(gripper_ctrl[0]))
                    non_target_response = 0.0
                    no_mapping_swap = True
                record = {
                    "channel": channel_name,
                    "channel_index": channel_index,
                    "sign": sign,
                    "decoded_action": decoded,
                    "target_response": target_response,
                    "non_target_response_norm": non_target_response,
                    "finite_explainable_response": bool(finite_response),
                    "no_mapping_swap": bool(no_mapping_swap),
                    "ctrl": ctrl,
                    "force": force,
                    "ctrl_bounds_ok": _bounds_ok(ctrl, actuator_metadata, "ctrl"),
                    "force_bounds_ok": _bounds_ok(force, actuator_metadata, "force"),
                }
                record["passed"] = all(
                    record[key]
                    for key in (
                        "finite_explainable_response",
                        "no_mapping_swap",
                        "ctrl_bounds_ok",
                        "force_bounds_ok",
                    )
                )
                results.append(record)
    finally:
        env.close()
    payload = {
        "schema_id": "shakebench.phase07.actuator_controls",
        "schema_version": 1,
        "controller": "OSC_POSE + PandaGripper",
        "physics_profile": {
            "profile_id": "shakebench.official.physics.v2",
            "profile_sha256": "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c",
        },
        "profile": profile.to_dict(),
        "actuators": actuator_metadata,
        "results": results,
        "passed": len(results) == 14 and all(row["passed"] for row in results),
    }
    payload["payload_sha256"] = _digest(payload)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_json_ready(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_actuator_positive_controls(args.output)
    print(
        json.dumps({"output": args.output, "passed": result["passed"], "count": len(result["results"])}, sort_keys=True)
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
