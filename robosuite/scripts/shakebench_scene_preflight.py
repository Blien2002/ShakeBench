"""Audit the world-fixed arm scene without granting experiment certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import robosuite
from robosuite.utils.shakebench_artifacts import json_ready, payload_hash
from robosuite.utils.shakebench_geometry import geometry_scene_path, load_geometry_profile
from robosuite.utils.shakebench_scene import compiled_physics_signature, load_scene_visual_config


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def run_preflight(
    output: str | Path, geometry_profile: str = "world_fixed_arm_v1", preview: str | Path | None = None
) -> dict[str, Any]:
    """Compile the native scene and write the machine-readable gate result."""

    scene_config = load_scene_visual_config(geometry_scene_path(geometry_profile))
    env = robosuite.make(
        "VibrationPickPlace",
        robots="Panda",
        controller_configs=None,
        physics_profile="official",
        geometry_profile=geometry_profile,
        observation_tier=None,
        use_camera_obs=False,
        use_object_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=1,
        ignore_done=True,
    )
    try:
        visible_signature = compiled_physics_signature(env.sim)
        visible_rgba_sha256 = hashlib.sha256(np.asarray(env.sim.model._model.geom_rgba).tobytes()).hexdigest()
        forbidden_initial_contacts = []
        model = env.sim.model._model
        for contact in env.sim.data.contact[: env.sim.data.ncon]:
            names = {model.geom(contact.geom1).name or "", model.geom(contact.geom2).name or ""}
            if "table_collision" in names and any(
                token in " ".join(names) for token in ("link5", "link6", "link7", "hand")
            ):
                forbidden_initial_contacts.append(sorted(names))
        if preview is not None:
            import mujoco
            from PIL import Image

            options = mujoco.MjvOption()
            options.geomgroup[0] = 0
            with mujoco.Renderer(model, height=720, width=1280) as renderer:
                renderer.update_scene(env.sim.data._data, camera="shakebench_camera_assembly", scene_option=options)
                target = Path(preview)
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(target)
        report = {
            "schema_id": "shakebench.world_fixed_arm.scene_preflight",
            "schema_version": 1,
            "base_commit": _git_head(),
            "geometry_profile": load_geometry_profile(geometry_profile),
            "scene_visual": {
                "scene_id": scene_config.scene_id,
                "config_sha256": scene_config.config_sha256,
                "geometry_variant": scene_config.geometry_variant,
                "physics_effect": scene_config.physics_effect,
            },
            "inventory": env.arena.scene_inventory.to_dict(),
            "compiled_audit": env._scene_audit.to_dict(),
            "clearance": env._scene_clearance.to_dict(),
            "robot_mount": env._compiled_contract["robot_mount"],
            "experiment_certification": "pending",
            "physics_signature": env._scene_audit.physics_signature,
            "gates": {
                "scene_ready": bool(env._scene_audit.passed),
                "clearance_passed": bool(env._scene_clearance.passed),
                "visual_physics_invariant": None,
                "initial_contact_gate": not forbidden_initial_contacts,
            },
            "initial_contacts": forbidden_initial_contacts,
        }
    finally:
        env.close()

    hidden_env = robosuite.make(
        "VibrationPickPlace",
        robots="Panda",
        controller_configs=None,
        physics_profile="official",
        observation_tier=None,
        use_camera_obs=False,
        use_object_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        scene_visual=False,
        geometry_profile=geometry_profile,
        seed=0,
        horizon=1,
        ignore_done=True,
    )
    try:
        hidden_signature = compiled_physics_signature(hidden_env.sim)
        hidden_rgba_sha256 = hashlib.sha256(np.asarray(hidden_env.sim.model._model.geom_rgba).tobytes()).hexdigest()
        report["visual_invariance"] = {
            "passed": visible_signature["sha256"] == hidden_signature["sha256"],
            "visible_physics_signature_sha256": visible_signature["sha256"],
            "hidden_physics_signature_sha256": hidden_signature["sha256"],
            "visible_rgba_sha256": visible_rgba_sha256,
            "hidden_rgba_sha256": hidden_rgba_sha256,
            "rgba_differs": visible_rgba_sha256 != hidden_rgba_sha256,
        }
        report["gates"]["visual_physics_invariant"] = report["visual_invariance"]["passed"]
        report["payload_sha256"] = payload_hash(report)
    finally:
        hidden_env.close()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_ready(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/world_fixed_arm/scene_preflight.json"))
    parser.add_argument("--geometry-profile", choices=("world_fixed_arm_v1",), default="world_fixed_arm_v1")
    parser.add_argument("--preview", type=Path, help="Optional PNG; requires an EGL/MESA renderer")
    args = parser.parse_args(argv)
    report = run_preflight(args.output, geometry_profile=args.geometry_profile, preview=args.preview)
    print(json.dumps({"output": str(args.output), **report["gates"]}, sort_keys=True))
    return (
        0
        if all(
            report["gates"][key]
            for key in (
                "scene_ready",
                "clearance_passed",
                "visual_physics_invariant",
                "initial_contact_gate",
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_preflight"]
