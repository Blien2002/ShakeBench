"""Camera identities for policy observations, datasets and videos."""

from __future__ import annotations

import mujoco
import numpy as np

from shakebench.scene.geometry import load_geometry_profile

CAMERAS = {"observation.images.main": "task_close", "observation.images.wrist": "robot0_eye_in_hand"}


# LIBERO pins the camera behind every benchmark observation in
# libero/libero/envs/bddl_base_domain.py::BenchmarkEnv._setup_camera: pos
# [0.5886, 0, 1.4904], quat wxyz [0.6380, 0.3049, 0.3049, 0.6380] for a tabletop
# centred at (0, 0, 0.8). That "agentview" is camera_names[0] of
# libero/libero/envs/env_wrapper.py; its sibling "canonical_agentview" is the same
# pose 0.05 m further back.
LIBERO_AGENTVIEW_POS_M = (0.5886131746834771, 0.0, 1.4903500240372423)


LIBERO_AGENTVIEW_QUAT_WXYZ = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)


LIBERO_AGENTVIEW_TABLE_TOP_M = (0.0, 0.0, 0.8)


LIBERO_AGENTVIEW_TABLE_M = 0.8


def task_close_camera() -> mujoco.MjvCamera:
    """Return the main task view: LIBERO's agentview pose transplanted onto the ShakeBench deck.

    The camera-to-tabletop offset is scaled by the deck-to-table size ratio, which
    makes the deck span LIBERO's share of the 45 degree frame (its near corners
    leave the frame there too); objects are 1/scale closer than in LIBERO.
    LIBERO's orientation carries no roll, which is what MuJoCo free cameras
    provide (right axis horizontal to 1e-7).
    """
    position, quat, deck_top = task_close_camera_pose()
    rotated = np.zeros(9)
    mujoco.mju_quat2Mat(rotated, quat)
    forward = -rotated.reshape(3, 3)[:, 2]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Aim the optical axis at the deck plane, as LIBERO's axis lands on its own tabletop.
    camera.distance = float((position[2] - deck_top[2]) / -forward[2])
    camera.lookat[:] = position + camera.distance * forward
    # MuJoCo free-camera convention: forward = (cos el cos az, cos el sin az, sin el).
    camera.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
    camera.elevation = float(np.degrees(np.arcsin(forward[2])))
    return camera


def task_close_camera_pose(profile=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (position_m, quat_wxyz, deck_top_m) of the main task view in world coordinates.

    Renderers that only accept model cameras (mujoco_warp's ray tracer, for example)
    use this pose directly, so their images frame the task exactly like the CPU
    free-camera preset does.
    """
    profile = profile or load_geometry_profile()
    deck_top = np.asarray(profile["table_top_pos_m"], dtype=float)
    scale = float(profile["retained_table_dimensions_m"][0]) / LIBERO_AGENTVIEW_TABLE_M
    position = deck_top + scale * (np.array(LIBERO_AGENTVIEW_POS_M) - np.array(LIBERO_AGENTVIEW_TABLE_TOP_M))
    return position, np.array(LIBERO_AGENTVIEW_QUAT_WXYZ, dtype=float), deck_top


WRIST_CAMERA = CAMERAS["observation.images.wrist"]
