"""Task, contact, and success metrics for the Phase 04 ShakeBench task.

The module deliberately keeps the measurement seam separate from the task
environment.  It can therefore be used with a live ``MjSim`` as well as with
compiled MuJoCo model/data objects in physics-only probes.  All relative pose
and velocity calculations are made in a moving frame; no world AABB or
center-only containment shortcut is used for success.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Optional

import mujoco
import numpy as np

import robosuite.utils.transform_utils as T


class ShakeBenchMetricsError(ValueError):
    """Raised when a metric input cannot satisfy its explicit contract."""


CONTACT_INTERFACE_TABLE_OBJECT = "table_object"
CONTACT_INTERFACE_TARGET_OBJECT = "target_object"
CONTACT_INTERFACE_FINGER_OBJECT = "finger_object"
CONTACT_INTERFACE_OTHER = "other"

# Short aliases make the role vocabulary convenient for callers without
# creating a second set of semantics.
TABLE_OBJECT_INTERFACE = CONTACT_INTERFACE_TABLE_OBJECT
TARGET_OBJECT_INTERFACE = CONTACT_INTERFACE_TARGET_OBJECT
FINGER_OBJECT_INTERFACE = CONTACT_INTERFACE_FINGER_OBJECT

CANONICAL_CAN_MASS_KG = 0.349
CANONICAL_CAN_COM_M = (0.0, 0.0, 0.0)
CAN_COLLISION_ENVELOPE_ALGORITHM_VERSION = "shakebench.can_collision_envelope.compiled_support.v1"
CANONICAL_CAN_COLLISION_ENVELOPE_SOURCE_GEOM_NAMES = ("can_g0",)
CANONICAL_CAN_COLLISION_ENVELOPE_SOURCE_MODEL_HASH = "c5332fb76e8b2f8c36fe10ac99e51f5e79c189e248d626dacd54cc983629e669"
TARGET_BOTTOM_SUPPORT_FORCE_THRESHOLD_N = 0.001
TARGET_BOTTOM_SUPPORT_Z_TOLERANCE_M = 0.00050


@dataclass(frozen=True)
class CanCollisionEnvelope:
    """Compiled collision support envelope; the sole Can geometry authority."""

    source_geom_names: tuple[str, ...]
    support_radius_m: float
    height_m: float
    lower_support_z_m: float
    upper_support_z_m: float
    source_model_hash: str
    extraction_algorithm_version: str = CAN_COLLISION_ENVELOPE_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        names = tuple(self.source_geom_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ShakeBenchMetricsError("source_geom_names must contain non-empty strings")
        object.__setattr__(self, "source_geom_names", names)
        for name in ("support_radius_m", "height_m", "lower_support_z_m", "upper_support_z_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ShakeBenchMetricsError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.support_radius_m <= 0.0 or self.height_m <= 0.0:
            raise ShakeBenchMetricsError("Can collision support radius and height must be positive")
        if not self.lower_support_z_m < self.upper_support_z_m:
            raise ShakeBenchMetricsError("Can collision lower support must be below upper support")
        if not np.isclose(
            self.height_m,
            self.upper_support_z_m - self.lower_support_z_m,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ShakeBenchMetricsError("Can collision envelope height does not match lower/upper support")
        if not isinstance(self.source_model_hash, str) or len(self.source_model_hash) != 64:
            raise ShakeBenchMetricsError("source_model_hash must be a SHA-256 hex digest")
        try:
            int(self.source_model_hash, 16)
        except ValueError as exc:
            raise ShakeBenchMetricsError("source_model_hash must be a SHA-256 hex digest") from exc
        if self.extraction_algorithm_version != CAN_COLLISION_ENVELOPE_ALGORITHM_VERSION:
            raise ShakeBenchMetricsError("unsupported Can collision envelope extraction algorithm")

    def assert_matches(self, other: "CanCollisionEnvelope", *, tolerance: float = 1e-12) -> None:
        if not isinstance(other, CanCollisionEnvelope):
            raise ShakeBenchMetricsError("Can collision envelope comparison requires CanCollisionEnvelope")
        if self.source_geom_names != other.source_geom_names:
            raise ShakeBenchMetricsError("Can collision source geom list drifted")
        if self.source_model_hash != other.source_model_hash:
            raise ShakeBenchMetricsError("Can collision source model hash drifted")
        if self.extraction_algorithm_version != other.extraction_algorithm_version:
            raise ShakeBenchMetricsError("Can collision envelope algorithm version drifted")
        for name in ("support_radius_m", "height_m", "lower_support_z_m", "upper_support_z_m"):
            if not np.isclose(getattr(self, name), getattr(other, name), rtol=0.0, atol=tolerance):
                raise ShakeBenchMetricsError(f"Can collision envelope field {name!r} drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_geom_names": list(self.source_geom_names),
            "support_radius_m": self.support_radius_m,
            "height_m": self.height_m,
            "lower_support_z_m": self.lower_support_z_m,
            "upper_support_z_m": self.upper_support_z_m,
            "source_model_hash": self.source_model_hash,
            "extraction_algorithm_version": self.extraction_algorithm_version,
        }


CANONICAL_CAN_COLLISION_ENVELOPE = CanCollisionEnvelope(
    source_geom_names=CANONICAL_CAN_COLLISION_ENVELOPE_SOURCE_GEOM_NAMES,
    support_radius_m=0.02509177806572465,
    height_m=0.08000000550552341,
    lower_support_z_m=-0.040297003330440104,
    upper_support_z_m=0.03970300217508332,
    source_model_hash=CANONICAL_CAN_COLLISION_ENVELOPE_SOURCE_MODEL_HASH,
)

# These aliases are retained only as deprecated read-only names for callers
# from the first Phase 04 implementation.  They point to the compiled
# envelope authority above; they are not placement-site constants.
CAN_COLLISION_ENVELOPE_RADIUS_M = CANONICAL_CAN_COLLISION_ENVELOPE.support_radius_m
CAN_COLLISION_ENVELOPE_HEIGHT_M = CANONICAL_CAN_COLLISION_ENVELOPE.height_m


def equivalent_cylinder_inertia(
    mass_kg: float = CANONICAL_CAN_MASS_KG,
    radius_m: Optional[float] = None,
    height_m: Optional[float] = None,
) -> tuple[float, float, float]:
    """Return principal inertia for a solid cylinder collision envelope."""

    if radius_m is None:
        radius_m = CANONICAL_CAN_COLLISION_ENVELOPE.support_radius_m
    if height_m is None:
        height_m = CANONICAL_CAN_COLLISION_ENVELOPE.height_m
    values = np.asarray((mass_kg, radius_m, height_m), dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mass_kg, radius_m, and height_m must be finite positive values")
    transverse = mass_kg * (3.0 * radius_m**2 + height_m**2) / 12.0
    axial = mass_kg * radius_m**2 / 2.0
    return float(transverse), float(transverse), float(axial)


CANONICAL_CAN_INERTIA_KG_M2 = equivalent_cylinder_inertia()


def _finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ShakeBenchMetricsError(f"{name} must contain {length} finite values") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ShakeBenchMetricsError(f"{name} must contain {length} finite values")
    return np.array(array, dtype=float, copy=True)


def _normalise_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        names = tuple(value)
    except TypeError as exc:
        raise ShakeBenchMetricsError("geom names must be a string or iterable of strings") from exc
    if any(not isinstance(name, str) or not name for name in names):
        raise ShakeBenchMetricsError("geom names must be non-empty strings")
    return names


def _normalise_quat_wxyz(value: Any, name: str = "quaternion_wxyz") -> np.ndarray:
    quat = _finite_vector(name, value, 4)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ShakeBenchMetricsError(f"{name} must have non-zero norm")
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _quat_wxyz_to_mat(quaternion_wxyz: Any) -> np.ndarray:
    quat = _normalise_quat_wxyz(quaternion_wxyz)
    return np.asarray(T.quat2mat(quat[[1, 2, 3, 0]]), dtype=float)


def _mat_to_quat_wxyz(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ShakeBenchMetricsError("rotation must be a finite 3x3 matrix")
    quat_xyzw = np.asarray(T.mat2quat(matrix), dtype=float)
    return _normalise_quat_wxyz(quat_xyzw[[3, 0, 1, 2]])


def _rotation_vector_from_matrix(rotation: Any) -> np.ndarray:
    quat_wxyz = _mat_to_quat_wxyz(rotation)
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return np.asarray(T.quat2axisangle(np.array(quat_xyzw, dtype=float, copy=True)), dtype=float)


def _raw_model_data(sim_or_model: Any, data: Any = None) -> tuple[Any, Any]:
    """Return raw MuJoCo model/data for either robosuite or native bindings."""

    if data is None and hasattr(sim_or_model, "data"):
        data = sim_or_model.data
    model = getattr(sim_or_model, "model", sim_or_model)
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    if raw_model is None or raw_data is None:
        raise ShakeBenchMetricsError("a live model and data pair is required")
    return raw_model, raw_data


def _raw_model_only(sim_or_model: Any) -> Any:
    model = getattr(sim_or_model, "model", sim_or_model)
    raw_model = getattr(model, "_model", model)
    if raw_model is None:
        raise ShakeBenchMetricsError("a compiled MuJoCo model is required")
    return raw_model


def _mujoco_id(model: Any, object_type: Any, name: str) -> int:
    try:
        identifier = int(mujoco.mj_name2id(model, object_type, name))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ShakeBenchMetricsError(f"could not resolve {name!r}") from exc
    if identifier < 0:
        raise ShakeBenchMetricsError(f"compiled model is missing {name!r}")
    return identifier


def _mujoco_name(model: Any, object_type: Any, identifier: int) -> str:
    name = mujoco.mj_id2name(model, object_type, int(identifier))
    if name is None:
        return f"<unnamed:{int(identifier)}>"
    return str(name)


@dataclass(frozen=True)
class PoseTwist:
    """Pose and coordinate twist of a body expressed in a moving frame.

    ``pose`` is ``[x, y, z, qw, qx, qy, qz]`` and ``twist`` is
    ``[linear_xyz, angular_xyz]``.  The linear velocity is the derivative of
    the position coordinates in the moving frame, so a rigidly co-moving
    object has a zero twist even when the frame itself is translating or
    rotating in world coordinates.
    """

    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _finite_vector("position_m", self.position_m, 3))
        object.__setattr__(self, "quaternion_wxyz", _normalise_quat_wxyz(self.quaternion_wxyz))
        object.__setattr__(
            self,
            "linear_velocity_m_s",
            _finite_vector("linear_velocity_m_s", self.linear_velocity_m_s, 3),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_vector("angular_velocity_rad_s", self.angular_velocity_rad_s, 3),
        )

    @property
    def pose(self) -> np.ndarray:
        return np.concatenate((self.position_m, self.quaternion_wxyz))

    @property
    def twist(self) -> np.ndarray:
        return np.concatenate((self.linear_velocity_m_s, self.angular_velocity_rad_s))

    @property
    def quaternion(self) -> np.ndarray:
        return self.quaternion_wxyz

    @property
    def linear_speed_m_s(self) -> float:
        return float(np.linalg.norm(self.linear_velocity_m_s))

    @property
    def angular_speed_rad_s(self) -> float:
        return float(np.linalg.norm(self.angular_velocity_rad_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose": self.pose.copy(),
            "twist": self.twist.copy(),
            "position_m": self.position_m.copy(),
            "quaternion_wxyz": self.quaternion_wxyz.copy(),
            "linear_velocity_m_s": self.linear_velocity_m_s.copy(),
            "angular_velocity_rad_s": self.angular_velocity_rad_s.copy(),
        }


def relative_pose_twist(
    child_position_m: Any,
    child_rotation: Any,
    child_twist: Any,
    frame_position_m: Any,
    frame_rotation: Any,
    frame_twist: Any,
) -> PoseTwist:
    """Compute a child's pose and moving-frame coordinate twist.

    Inputs use world-frame positions, rotation matrices, and six-vectors in
    linear-then-angular order.  The frame velocity is evaluated at the frame
    origin.  The formula includes the rotating-frame transport term, which is
    what makes the result invariant when deck/table motion is applied to both
    bodies.
    """

    child_position = _finite_vector("child_position_m", child_position_m, 3)
    child_rotation = np.asarray(child_rotation, dtype=float)
    if child_rotation.shape != (3, 3) or not np.all(np.isfinite(child_rotation)):
        raise ShakeBenchMetricsError("child_rotation must be a finite 3x3 matrix")
    child_twist = _finite_vector("child_twist", child_twist, 6)
    frame_position = _finite_vector("frame_position_m", frame_position_m, 3)
    frame_rotation = np.asarray(frame_rotation, dtype=float)
    if frame_rotation.shape != (3, 3) or not np.all(np.isfinite(frame_rotation)):
        raise ShakeBenchMetricsError("frame_rotation must be a finite 3x3 matrix")
    frame_twist = _finite_vector("frame_twist", frame_twist, 6)

    frame_linear = frame_twist[:3]
    frame_angular = frame_twist[3:]
    relative_position = frame_rotation.T.dot(child_position - frame_position)
    relative_rotation = frame_rotation.T.dot(child_rotation)
    relative_linear_velocity = frame_rotation.T.dot(child_twist[:3] - frame_linear)
    relative_linear_velocity -= np.cross(frame_rotation.T.dot(frame_angular), relative_position)
    relative_angular_velocity = frame_rotation.T.dot(child_twist[3:] - frame_angular)
    return PoseTwist(
        position_m=relative_position,
        quaternion_wxyz=_mat_to_quat_wxyz(relative_rotation),
        linear_velocity_m_s=relative_linear_velocity,
        angular_velocity_rad_s=relative_angular_velocity,
    )


def _body_spatial_twist(model: Any, data: Any, body_id: int) -> np.ndarray:
    velocity_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity_rotlin, 0)
    # MuJoCo's object API returns angular then linear components.
    return np.concatenate((velocity_rotlin[3:], velocity_rotlin[:3]))


def _body_spatial_acceleration(model: Any, data: Any, body_id: int) -> np.ndarray:
    """Return world-frame body acceleration in linear-then-angular order."""

    # The RNE post-constraint buffers are the backend-independent source for
    # object acceleration; qacc alone is a body-joint tangent coordinate.
    mujoco.mj_rnePostConstraint(model, data)
    acceleration_rotlin = np.zeros(6, dtype=float)
    mujoco.mj_objectAcceleration(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        int(body_id),
        acceleration_rotlin,
        0,
    )
    return np.concatenate((acceleration_rotlin[3:], acceleration_rotlin[:3]))


def _body_world_pose_twist(
    model: Any, data: Any, body_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(data.xpos[int(body_id)], dtype=float).copy()
    rotation = np.asarray(data.xmat[int(body_id)], dtype=float).reshape(3, 3).copy()
    twist = _body_spatial_twist(model, data, body_id)
    return position, rotation, twist[:3], twist[3:]


def _frame_world_pose_twist(
    model: Any,
    data: Any,
    frame_body_name: Optional[str],
    frame_local_origin_m: Any = (0.0, 0.0, 0.0),
    frame_local_quat_wxyz: Any = (1.0, 0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_origin = _finite_vector("frame_local_origin_m", frame_local_origin_m, 3)
    local_rotation = _quat_wxyz_to_mat(frame_local_quat_wxyz)
    if frame_body_name is None:
        return np.zeros(3), local_rotation, np.zeros(3), np.zeros(3)
    body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body_name)
    body_position, body_rotation, body_linear, body_angular = _body_world_pose_twist(model, data, body_id)
    world_offset = body_rotation.dot(local_origin)
    origin_position = body_position + world_offset
    origin_linear = body_linear + np.cross(body_angular, world_offset)
    return origin_position, body_rotation.dot(local_rotation), origin_linear, body_angular


def pose_twist_in_frame(
    sim_or_model: Any,
    child_body_name: str,
    frame_body_name: Optional[str] = None,
    *,
    data: Any = None,
    frame_local_origin_m: Any = (0.0, 0.0, 0.0),
    frame_local_quat_wxyz: Any = (1.0, 0.0, 0.0, 0.0),
) -> PoseTwist:
    """Return a compiled body's pose/twist in a world or body frame."""

    model, raw_data = _raw_model_data(sim_or_model, data)
    child_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, child_body_name)
    child_position, child_rotation, child_linear, child_angular = _body_world_pose_twist(model, raw_data, child_id)
    frame_position, frame_rotation, frame_linear, frame_angular = _frame_world_pose_twist(
        model,
        raw_data,
        frame_body_name,
        frame_local_origin_m,
        frame_local_quat_wxyz,
    )
    return relative_pose_twist(
        child_position,
        child_rotation,
        np.concatenate((child_linear, child_angular)),
        frame_position,
        frame_rotation,
        np.concatenate((frame_linear, frame_angular)),
    )


def can_pose_twist_in_frame(
    sim_or_model: Any,
    can_body_name: str,
    frame_body_name: Optional[str] = None,
    *,
    data: Any = None,
    frame_local_origin_m: Any = (0.0, 0.0, 0.0),
    frame_local_quat_wxyz: Any = (1.0, 0.0, 0.0, 0.0),
) -> PoseTwist:
    """Task-named wrapper around :func:`pose_twist_in_frame`."""

    return pose_twist_in_frame(
        sim_or_model,
        can_body_name,
        frame_body_name,
        data=data,
        frame_local_origin_m=frame_local_origin_m,
        frame_local_quat_wxyz=frame_local_quat_wxyz,
    )


def can_pose_in_frame(*args: Any, **kwargs: Any) -> np.ndarray:
    """Return Can pose ``[xyz, qwxyz]`` in a selected frame."""

    return can_pose_twist_in_frame(*args, **kwargs).pose


def can_twist_in_frame(*args: Any, **kwargs: Any) -> np.ndarray:
    """Return Can moving-frame twist ``[linear, angular]``."""

    return can_pose_twist_in_frame(*args, **kwargs).twist


def frame_world_position(
    sim_or_model: Any,
    frame_body_name: Optional[str],
    *,
    data: Any = None,
    frame_local_origin_m: Any = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the world position of a body-frame point."""

    model, raw_data = _raw_model_data(sim_or_model, data)
    position, _, _, _ = _frame_world_pose_twist(model, raw_data, frame_body_name, frame_local_origin_m)
    return position


def _mesh_vertices_in_body_frame(model: Any, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[int(geom_id)])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[int(geom_id)])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float).reshape(-1, 3)
        geom_quat = _normalise_quat_wxyz(model.geom_quat[int(geom_id)], "compiled geom quaternion")
        geom_rotation = _quat_wxyz_to_mat(geom_quat)
        geom_position = np.asarray(model.geom_pos[int(geom_id)], dtype=float)
        return geom_position + vertices.dot(geom_rotation.T)

    # This fallback is useful for probe fixtures.  The production Can uses a
    # mesh, but a primitive collision geom still gets exact corner support
    # points rather than a center/AABB approximation.
    size = np.asarray(model.geom_size[int(geom_id)], dtype=float).reshape(-1)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        local = np.asarray(
            [[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
        local = np.asarray(
            [[sx * radius, sy * radius, sz * radius] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
    else:
        radius = float(size[0])
        half_height = float(size[1]) if size.size > 1 else radius
        local = np.asarray(
            [
                [radius * np.cos(angle), radius * np.sin(angle), z]
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                for z in (-half_height, half_height)
            ],
            dtype=float,
        )
    geom_quat = _normalise_quat_wxyz(model.geom_quat[int(geom_id)], "compiled geom quaternion")
    geom_rotation = _quat_wxyz_to_mat(geom_quat)
    geom_position = np.asarray(model.geom_pos[int(geom_id)], dtype=float)
    return geom_position + local.dot(geom_rotation.T)


def _collision_source_hash(model: Any, geom_names: tuple[str, ...]) -> str:
    """Hash compiled collision geometry inputs used by envelope extraction."""

    if len(geom_names) == 1:
        geom_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_names[0])
        payload = {
            "algorithm": CAN_COLLISION_ENVELOPE_ALGORITHM_VERSION,
            "source_geom_names": list(geom_names),
            "geom_type": int(model.geom_type[geom_id]),
            "geom_pos": np.asarray(model.geom_pos[geom_id], dtype=float).tolist(),
            "geom_quat": np.asarray(model.geom_quat[geom_id], dtype=float).tolist(),
        }
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            payload["mesh_vertices"] = np.asarray(model.mesh_vert[start : start + count], dtype=float).tolist()
        else:
            payload["geom_size"] = np.asarray(model.geom_size[geom_id], dtype=float).tolist()
    else:
        geometries = []
        for geom_name in geom_names:
            geom_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            geometry = {
                "name": geom_name,
                "geom_type": int(model.geom_type[geom_id]),
                "geom_pos": np.asarray(model.geom_pos[geom_id], dtype=float).tolist(),
                "geom_quat": np.asarray(model.geom_quat[geom_id], dtype=float).tolist(),
                "geom_size": np.asarray(model.geom_size[geom_id], dtype=float).tolist(),
            }
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                geometry["mesh_vertices"] = np.asarray(model.mesh_vert[start : start + count], dtype=float).tolist()
            geometries.append(geometry)
        payload = {
            "algorithm": CAN_COLLISION_ENVELOPE_ALGORITHM_VERSION,
            "source_geom_names": list(geom_names),
            "geometries": geometries,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_can_collision_envelope(
    sim_or_model: Any,
    can_body_name: str,
    collision_geom_names: Iterable[str],
) -> CanCollisionEnvelope:
    """Extract one Can envelope from compiled collision geoms in body frame."""

    model = _raw_model_only(sim_or_model)
    body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, can_body_name)
    geom_names = _normalise_names(collision_geom_names)
    points = []
    for geom_name in geom_names:
        geom_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if int(model.geom_bodyid[geom_id]) != body_id:
            raise ShakeBenchMetricsError(f"collision geom {geom_name!r} is not attached to Can body")
        points.append(_mesh_vertices_in_body_frame(model, geom_id))
    if not points:
        raise ShakeBenchMetricsError("at least one compiled Can collision geom is required")
    support_points = np.concatenate(points, axis=0)
    lower = float(np.min(support_points[:, 2]))
    upper = float(np.max(support_points[:, 2]))
    radius = float(np.max(np.linalg.norm(support_points[:, :2], axis=1)))
    return CanCollisionEnvelope(
        source_geom_names=geom_names,
        support_radius_m=radius,
        height_m=upper - lower,
        lower_support_z_m=lower,
        upper_support_z_m=upper,
        source_model_hash=_collision_source_hash(model, geom_names),
    )


def collision_support_points_in_frame(
    sim_or_model: Any,
    body_name: str,
    collision_geom_names: Iterable[str],
    frame_body_name: Optional[str] = None,
    *,
    data: Any = None,
    frame_local_origin_m: Any = (0.0, 0.0, 0.0),
    frame_local_quat_wxyz: Any = (1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return collision-geometry support vertices expressed in a frame.

    The points are the compiled collision mesh vertices (or exact primitive
    corners for probe fixtures), transformed into the requested frame.  The
    caller can project them to XY for rectangular target containment.
    """

    model, raw_data = _raw_model_data(sim_or_model, data)
    body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    body_rotation = np.asarray(raw_data.xmat[body_id], dtype=float).reshape(3, 3)
    body_position = np.asarray(raw_data.xpos[body_id], dtype=float)
    points = []
    for geom_name in _normalise_names(collision_geom_names):
        geom_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if int(model.geom_bodyid[geom_id]) != body_id:
            raise ShakeBenchMetricsError(f"collision geom {geom_name!r} is not attached to body {body_name!r}")
        body_points = _mesh_vertices_in_body_frame(model, geom_id)
        points.append(body_position + body_points.dot(body_rotation.T))
    if not points:
        raise ShakeBenchMetricsError("at least one collision geom is required")
    world_points = np.concatenate(points, axis=0)
    frame_position, frame_rotation, _, _ = _frame_world_pose_twist(
        model,
        raw_data,
        frame_body_name,
        frame_local_origin_m,
        frame_local_quat_wxyz,
    )
    return (world_points - frame_position).dot(frame_rotation)


@dataclass(frozen=True)
class ContactRecord:
    """One Can contact measured at the current MuJoCo state."""

    interface: str
    geom1: str
    geom2: str
    point_world_m: np.ndarray
    distance_m: float
    penetration_m: float
    force_contact_frame_N: np.ndarray
    force_on_can_world_N: np.ndarray
    wrench_on_can_world_N_Nm: np.ndarray
    impulse_on_can_world_Ns: np.ndarray
    wrench_impulse_on_can_world_Ns_Nms: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_world_m", _finite_vector("point_world_m", self.point_world_m, 3))
        object.__setattr__(self, "distance_m", float(self.distance_m))
        object.__setattr__(self, "penetration_m", float(self.penetration_m))
        object.__setattr__(
            self,
            "force_contact_frame_N",
            _finite_vector("force_contact_frame_N", self.force_contact_frame_N, 6),
        )
        for field_name in (
            "force_on_can_world_N",
            "wrench_on_can_world_N_Nm",
            "impulse_on_can_world_Ns",
            "wrench_impulse_on_can_world_Ns_Nms",
        ):
            expected_length = 6 if "wrench" in field_name else 3
            object.__setattr__(self, field_name, _finite_vector(field_name, getattr(self, field_name), expected_length))

    @property
    def normal_force_N(self) -> float:
        return float(abs(self.force_contact_frame_N[0]))

    @property
    def force_world_N(self) -> np.ndarray:
        return self.force_on_can_world_N

    @property
    def wrench_world_N_Nm(self) -> np.ndarray:
        return self.wrench_on_can_world_N_Nm

    @property
    def impulse_world_Ns(self) -> np.ndarray:
        return self.impulse_on_can_world_Ns

    @property
    def penetration_depth_m(self) -> float:
        return self.penetration_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "geom1": self.geom1,
            "geom2": self.geom2,
            "point_world_m": self.point_world_m.copy(),
            "distance_m": self.distance_m,
            "penetration_m": self.penetration_m,
            "normal_force_N": self.normal_force_N,
            "force_contact_frame_N": self.force_contact_frame_N.copy(),
            "force_on_can_world_N": self.force_on_can_world_N.copy(),
            "wrench_on_can_world_N_Nm": self.wrench_on_can_world_N_Nm.copy(),
            "impulse_on_can_world_Ns": self.impulse_on_can_world_Ns.copy(),
            "wrench_impulse_on_can_world_Ns_Nms": self.wrench_impulse_on_can_world_Ns_Nms.copy(),
        }


@dataclass(frozen=True)
class ContactReport:
    """Interface-resolved contact report for one simulation sample."""

    contacts: tuple[ContactRecord, ...]
    by_interface: Mapping[str, tuple[ContactRecord, ...]]
    max_penetration_m: float
    table_contact_present: bool
    target_bottom_contact_present: bool
    target_wall_contact_present: bool
    finger_can_contact_present: bool
    target_bottom_support_force_N: float
    total_force_by_interface_N: Mapping[str, np.ndarray]
    total_wrench_by_interface_N_Nm: Mapping[str, np.ndarray]
    total_impulse_by_interface_Ns: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contacts", tuple(self.contacts))
        object.__setattr__(self, "by_interface", MappingProxyType(dict(self.by_interface)))
        object.__setattr__(self, "max_penetration_m", float(self.max_penetration_m))
        object.__setattr__(self, "target_bottom_support_force_N", float(self.target_bottom_support_force_N))
        for field_name in (
            "total_force_by_interface_N",
            "total_wrench_by_interface_N_Nm",
            "total_impulse_by_interface_Ns",
        ):
            value = {key: np.array(item, dtype=float, copy=True) for key, item in getattr(self, field_name).items()}
            object.__setattr__(self, field_name, MappingProxyType(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contacts": [contact.to_dict() for contact in self.contacts],
            "by_interface": {key: [contact.to_dict() for contact in value] for key, value in self.by_interface.items()},
            "max_penetration_m": self.max_penetration_m,
            "table_contact_present": self.table_contact_present,
            "target_bottom_contact_present": self.target_bottom_contact_present,
            "target_wall_contact_present": self.target_wall_contact_present,
            "finger_can_contact_present": self.finger_can_contact_present,
            "target_bottom_support_force_N": self.target_bottom_support_force_N,
            "total_force_by_interface_N": {key: value.copy() for key, value in self.total_force_by_interface_N.items()},
            "total_wrench_by_interface_N_Nm": {
                key: value.copy() for key, value in self.total_wrench_by_interface_N_Nm.items()
            },
            "total_impulse_by_interface_Ns": {
                key: value.copy() for key, value in self.total_impulse_by_interface_Ns.items()
            },
        }


def _contact_interface(
    other_geom: str,
    table_geom_names: set[str],
    target_bottom_geom_names: set[str],
    target_wall_geom_names: set[str],
    finger_pad_geom_names: set[str],
) -> str:
    if other_geom in table_geom_names:
        return CONTACT_INTERFACE_TABLE_OBJECT
    if other_geom in target_bottom_geom_names or other_geom in target_wall_geom_names:
        return CONTACT_INTERFACE_TARGET_OBJECT
    if other_geom in finger_pad_geom_names:
        return CONTACT_INTERFACE_FINGER_OBJECT
    return CONTACT_INTERFACE_OTHER


def collect_contact_metrics(
    sim_or_model: Any,
    *,
    can_geom_names: Iterable[str] = (),
    can_body_name: Optional[str] = None,
    table_geom_names: Iterable[str] = (),
    target_bottom_geom_names: Iterable[str] = (),
    target_wall_geom_names: Iterable[str] = (),
    finger_pad_geom_names: Iterable[str] = (),
    data: Any = None,
    dt_s: Optional[float] = None,
    support_frame_rotation: Any = None,
) -> ContactReport:
    """Collect Can contacts and per-interface force/wrench/impulse metrics."""

    model, raw_data = _raw_model_data(sim_or_model, data)
    can_names = set(_normalise_names(can_geom_names))
    if not can_names and can_body_name is not None:
        can_body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, can_body_name)
        can_names = {
            _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) == can_body_id and int(model.geom_group[geom_id]) == 0
        }
    table_names = set(_normalise_names(table_geom_names))
    bottom_names = set(_normalise_names(target_bottom_geom_names))
    wall_names = set(_normalise_names(target_wall_geom_names))
    finger_names = set(_normalise_names(finger_pad_geom_names))
    if not can_names:
        raise ShakeBenchMetricsError("can_geom_names must not be empty")
    if dt_s is None:
        dt_s = float(model.opt.timestep)
    if isinstance(dt_s, (bool, np.bool_)) or not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ShakeBenchMetricsError("dt_s must be finite and positive")
    dt_s = float(dt_s)
    if support_frame_rotation is None:
        support_rotation = np.eye(3, dtype=float)
    else:
        support_rotation = np.asarray(support_frame_rotation, dtype=float)
        if support_rotation.shape != (3, 3) or not np.all(np.isfinite(support_rotation)):
            raise ShakeBenchMetricsError("support_frame_rotation must be a finite 3x3 matrix")

    records = []
    by_interface = defaultdict(list)
    total_force = defaultdict(lambda: np.zeros(3, dtype=float))
    total_wrench = defaultdict(lambda: np.zeros(6, dtype=float))
    total_impulse = defaultdict(lambda: np.zeros(3, dtype=float))

    for contact_index in range(int(raw_data.ncon)):
        contact = raw_data.contact[contact_index]
        geom1 = _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        can_is_geom1 = geom1 in can_names
        can_is_geom2 = geom2 in can_names
        if not (can_is_geom1 or can_is_geom2) or (can_is_geom1 and can_is_geom2):
            continue
        other_geom = geom2 if can_is_geom1 else geom1
        interface = _contact_interface(other_geom, table_names, bottom_names, wall_names, finger_names)
        contact_force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, raw_data, contact_index, contact_force)
        contact_frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_world = contact_frame.T.dot(contact_force[:3])
        torque_world_at_contact = contact_frame.T.dot(contact_force[3:])
        # MuJoCo reports the contact-frame force applied to geom2.  Orient it
        # onto the Can so a static tabletop support has an upward positive Z
        # force regardless of the runtime geom ordering.
        if can_is_geom1:
            force_world *= -1.0
            torque_world_at_contact *= -1.0
        point_world = np.asarray(contact.pos, dtype=float).copy()
        wrench_world = np.concatenate((force_world, np.cross(point_world, force_world) + torque_world_at_contact))
        impulse_world = force_world * dt_s
        wrench_impulse = wrench_world * dt_s
        distance = float(contact.dist)
        record = ContactRecord(
            interface=interface,
            geom1=geom1,
            geom2=geom2,
            point_world_m=point_world,
            distance_m=distance,
            penetration_m=max(0.0, -distance),
            force_contact_frame_N=contact_force,
            force_on_can_world_N=force_world,
            wrench_on_can_world_N_Nm=wrench_world,
            impulse_on_can_world_Ns=impulse_world,
            wrench_impulse_on_can_world_Ns_Nms=wrench_impulse,
        )
        # Force and impulse fields are linear three-vectors; wrench fields are
        # six-vectors.  Aggregate fields below use the same distinction.
        records.append(record)
        by_interface[interface].append(record)
        total_force[interface] += force_world
        total_wrench[interface] += wrench_world
        total_impulse[interface] += impulse_world

    # The record stores six-vectors for a uniform spatial API.  The public
    # aggregate names explicitly distinguish linear force/impulse from wrench.
    for interface in (
        CONTACT_INTERFACE_TABLE_OBJECT,
        CONTACT_INTERFACE_TARGET_OBJECT,
        CONTACT_INTERFACE_FINGER_OBJECT,
        CONTACT_INTERFACE_OTHER,
    ):
        by_interface.setdefault(interface, [])
        total_force.setdefault(interface, np.zeros(3, dtype=float))
        total_wrench.setdefault(interface, np.zeros(6, dtype=float))
        total_impulse.setdefault(interface, np.zeros(3, dtype=float))
    target_bottom_contacts = [
        record
        for record in by_interface[CONTACT_INTERFACE_TARGET_OBJECT]
        if record.geom1 in bottom_names or record.geom2 in bottom_names
    ]
    target_bottom_support_force = float(
        sum(
            max(0.0, float(support_rotation.T.dot(record.force_on_can_world_N)[2])) for record in target_bottom_contacts
        )
    )
    return ContactReport(
        contacts=tuple(records),
        by_interface={key: tuple(value) for key, value in by_interface.items()},
        max_penetration_m=max((record.penetration_m for record in records), default=0.0),
        table_contact_present=bool(by_interface.get(CONTACT_INTERFACE_TABLE_OBJECT)),
        target_bottom_contact_present=any(
            record.geom1 in bottom_names or record.geom2 in bottom_names
            for record in by_interface.get(CONTACT_INTERFACE_TARGET_OBJECT, ())
        ),
        target_wall_contact_present=any(
            record.geom1 in wall_names or record.geom2 in wall_names
            for record in by_interface.get(CONTACT_INTERFACE_TARGET_OBJECT, ())
        ),
        finger_can_contact_present=bool(by_interface.get(CONTACT_INTERFACE_FINGER_OBJECT)),
        target_bottom_support_force_N=target_bottom_support_force,
        total_force_by_interface_N={key: value for key, value in total_force.items()},
        total_wrench_by_interface_N_Nm={key: value for key, value in total_wrench.items()},
        total_impulse_by_interface_Ns={key: value for key, value in total_impulse.items()},
    )


@dataclass(frozen=True)
class SuccessThresholds:
    """Frozen v0 success thresholds."""

    hold_duration_s: float = 0.50
    max_relative_linear_speed_m_s: float = 0.02
    max_relative_angular_speed_rad_s: float = 0.20
    max_illegal_penetration_m: float = 0.00050
    target_bottom_support_force_threshold_N: float = TARGET_BOTTOM_SUPPORT_FORCE_THRESHOLD_N
    target_bottom_support_z_tolerance_m: float = TARGET_BOTTOM_SUPPORT_Z_TOLERANCE_M
    containment_epsilon_m: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "hold_duration_s",
            "max_relative_linear_speed_m_s",
            "max_relative_angular_speed_rad_s",
            "max_illegal_penetration_m",
            "target_bottom_support_force_threshold_N",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ShakeBenchMetricsError(f"{name} must be finite and positive")
        if (
            isinstance(self.target_bottom_support_z_tolerance_m, (bool, np.bool_))
            or not np.isfinite(float(self.target_bottom_support_z_tolerance_m))
            or float(self.target_bottom_support_z_tolerance_m) < 0.0
        ):
            raise ShakeBenchMetricsError("target_bottom_support_z_tolerance_m must be finite and non-negative")
        if (
            isinstance(self.containment_epsilon_m, (bool, np.bool_))
            or not np.isfinite(float(self.containment_epsilon_m))
            or float(self.containment_epsilon_m) < 0.0
        ):
            raise ShakeBenchMetricsError("containment_epsilon_m must be finite and non-negative")


DEFAULT_SUCCESS_THRESHOLDS = SuccessThresholds()


@dataclass(frozen=True)
class SuccessSnapshot:
    """Inputs to the success evaluator at one time sample."""

    support_points_target_xy: np.ndarray
    target_inner_half_extents_m: tuple[float, float]
    target_bottom_contact_present: bool
    target_bottom_support_force_N: float
    lower_support_z_m: float
    finger_can_contact_present: bool
    relative_linear_speed_m_s: float
    relative_angular_speed_rad_s: float
    illegal_penetration_m: float

    def __post_init__(self) -> None:
        points = np.asarray(self.support_points_target_xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise ShakeBenchMetricsError("support_points_target_xy must be a finite (N, 2) array")
        extents = _finite_vector("target_inner_half_extents_m", self.target_inner_half_extents_m, 2)
        if np.any(extents <= 0.0):
            raise ShakeBenchMetricsError("target_inner_half_extents_m must be positive")
        object.__setattr__(self, "support_points_target_xy", np.array(points, dtype=float, copy=True))
        object.__setattr__(self, "target_inner_half_extents_m", (float(extents[0]), float(extents[1])))
        for name in (
            "target_bottom_support_force_N",
            "relative_linear_speed_m_s",
            "relative_angular_speed_rad_s",
            "illegal_penetration_m",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)) or float(value) < 0.0:
                raise ShakeBenchMetricsError(f"{name} must be finite and non-negative")
        if not np.isfinite(float(self.lower_support_z_m)):
            raise ShakeBenchMetricsError("lower_support_z_m must be finite")

    def is_supported_by_target_bottom(
        self,
        *,
        force_threshold_N: Optional[float] = None,
        z_tolerance_m: Optional[float] = None,
    ) -> bool:
        """Require bottom contact, strict positive support force, and valid height."""

        if force_threshold_N is None:
            force_threshold_N = TARGET_BOTTOM_SUPPORT_FORCE_THRESHOLD_N
        if z_tolerance_m is None:
            z_tolerance_m = TARGET_BOTTOM_SUPPORT_Z_TOLERANCE_M
        if (
            isinstance(force_threshold_N, (bool, np.bool_))
            or not np.isfinite(float(force_threshold_N))
            or float(force_threshold_N) <= 0.0
        ):
            raise ShakeBenchMetricsError("force_threshold_N must be finite and strictly positive")
        if (
            isinstance(z_tolerance_m, (bool, np.bool_))
            or not np.isfinite(float(z_tolerance_m))
            or float(z_tolerance_m) < 0.0
        ):
            raise ShakeBenchMetricsError("z_tolerance_m must be finite and non-negative")
        return bool(
            self.target_bottom_contact_present
            and self.target_bottom_support_force_N > float(force_threshold_N)
            and self.lower_support_z_m >= -float(z_tolerance_m)
        )

    @property
    def supported_by_target_bottom(self) -> bool:
        """Default-threshold support result (derived, never an independent input)."""

        return self.is_supported_by_target_bottom()

    @property
    def finger_can_contact(self) -> bool:
        """Deprecated input alias; report schemas use ``finger_can_contact_present``."""

        return self.finger_can_contact_present

    @property
    def containment(self) -> bool:
        return self.containment_with_epsilon(DEFAULT_SUCCESS_THRESHOLDS.containment_epsilon_m)

    def containment_with_epsilon(self, epsilon_m: float) -> bool:
        if isinstance(epsilon_m, (bool, np.bool_)) or not np.isfinite(float(epsilon_m)) or float(epsilon_m) < 0.0:
            raise ShakeBenchMetricsError("epsilon_m must be finite and non-negative")
        half_x, half_y = self.target_inner_half_extents_m
        return bool(
            np.all(np.abs(self.support_points_target_xy[:, 0]) <= half_x + float(epsilon_m))
            and np.all(np.abs(self.support_points_target_xy[:, 1]) <= half_y + float(epsilon_m))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "containment": self.containment,
            "support_points_target_xy": self.support_points_target_xy.copy(),
            "target_inner_half_extents_m": list(self.target_inner_half_extents_m),
            "target_bottom_contact_present": bool(self.target_bottom_contact_present),
            "target_bottom_support_force_N": self.target_bottom_support_force_N,
            "lower_support_z_m": self.lower_support_z_m,
            "supported_by_target_bottom": bool(self.supported_by_target_bottom),
            "finger_can_contact_present": bool(self.finger_can_contact_present),
            "relative_linear_speed_m_s": self.relative_linear_speed_m_s,
            "relative_angular_speed_rad_s": self.relative_angular_speed_rad_s,
            "illegal_penetration_m": self.illegal_penetration_m,
        }


@dataclass(frozen=True)
class SuccessEvaluation:
    """One evaluator result, including every subcondition."""

    passed: bool
    latched: bool
    candidate_since_s: Optional[float]
    subconditions: Mapping[str, bool]
    snapshot: SuccessSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "latched", bool(self.latched))
        object.__setattr__(self, "subconditions", MappingProxyType(dict(self.subconditions)))

    @property
    def success(self) -> bool:
        return self.passed

    @property
    def conditions(self) -> Mapping[str, bool]:
        return self.subconditions

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "latched": self.latched,
            "candidate_since_s": self.candidate_since_s,
            "subconditions": dict(self.subconditions),
            "snapshot": self.snapshot.to_dict(),
        }


def _coerce_success_snapshot(snapshot: Any) -> SuccessSnapshot:
    if isinstance(snapshot, SuccessSnapshot):
        return snapshot
    if not isinstance(snapshot, Mapping):
        raise ShakeBenchMetricsError("success input must be a SuccessSnapshot or mapping")
    values = dict(snapshot)
    # Mapping support is intentionally explicit so recorder/probe code can
    # pass JSON-shaped state without bypassing validation.
    return SuccessSnapshot(**values)


class VibrationSuccessEvaluator:
    """Continuous, fail-closed v0 success latch."""

    def __init__(self, thresholds: Optional[SuccessThresholds] = None):
        self.thresholds = thresholds if thresholds is not None else DEFAULT_SUCCESS_THRESHOLDS
        if not isinstance(self.thresholds, SuccessThresholds):
            raise ShakeBenchMetricsError("thresholds must be SuccessThresholds")
        self.reset()

    def reset(self) -> None:
        self._candidate_since_s: Optional[float] = None
        self._last_time_s: Optional[float] = None
        self._latched = False
        self._last_evaluation: Optional[SuccessEvaluation] = None

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def candidate_since_s(self) -> Optional[float]:
        return self._candidate_since_s

    def _conditions(self, snapshot: SuccessSnapshot) -> dict[str, bool]:
        return {
            "containment": snapshot.containment_with_epsilon(self.thresholds.containment_epsilon_m),
            "supported_by_target_bottom": bool(snapshot.supported_by_target_bottom),
            "finger_can_contact_absent": not bool(snapshot.finger_can_contact_present),
            "relative_linear_speed": snapshot.relative_linear_speed_m_s < self.thresholds.max_relative_linear_speed_m_s,
            "relative_angular_speed": snapshot.relative_angular_speed_rad_s
            < self.thresholds.max_relative_angular_speed_rad_s,
            "illegal_penetration": snapshot.illegal_penetration_m < self.thresholds.max_illegal_penetration_m,
        }

    def evaluate(self, snapshot: Any, time_s: float) -> SuccessEvaluation:
        """Evaluate and update the latch at an episode-relative timestamp."""

        snapshot = _coerce_success_snapshot(snapshot)
        if isinstance(time_s, (bool, np.bool_)) or not np.isfinite(float(time_s)) or float(time_s) < 0.0:
            raise ShakeBenchMetricsError("time_s must be finite and non-negative")
        time_s = float(time_s)
        if self._last_time_s is not None and time_s < self._last_time_s - 1e-12:
            # A backwards timestamp is a reset boundary, not a successful
            # continuation from a different episode.
            self.reset()
        self._last_time_s = time_s
        conditions = self._conditions(snapshot)
        all_conditions = all(conditions.values())
        if not self._latched:
            if all_conditions:
                if self._candidate_since_s is None:
                    self._candidate_since_s = time_s
                if time_s - self._candidate_since_s >= self.thresholds.hold_duration_s - 1e-12:
                    self._latched = True
            else:
                self._candidate_since_s = None
        result = SuccessEvaluation(
            passed=self._latched,
            latched=self._latched,
            candidate_since_s=self._candidate_since_s,
            subconditions=conditions,
            snapshot=snapshot,
        )
        self._last_evaluation = result
        return result

    def check(self, snapshot: Any, time_s: float) -> bool:
        """Boolean convenience wrapper around :meth:`evaluate`."""

        return self.evaluate(snapshot, time_s).passed

    def update(self, snapshot: Any, time_s: float) -> SuccessEvaluation:
        """Stateful-update alias used by recorder/evaluator integrations."""

        return self.evaluate(snapshot, time_s)

    __call__ = check


def evaluate_success(
    snapshot: Any,
    time_s: float,
    evaluator: Optional[VibrationSuccessEvaluator] = None,
) -> SuccessEvaluation:
    """Evaluate a snapshot with an optional caller-owned continuous latch."""

    active_evaluator = evaluator if evaluator is not None else VibrationSuccessEvaluator()
    return active_evaluator.evaluate(snapshot, time_s)


def audit_can_compiled_model(
    sim_or_model: Any,
    can_body_name: str,
    can_geom_names: Iterable[str],
    *,
    expected_mass_kg: float = CANONICAL_CAN_MASS_KG,
    expected_com_m: Iterable[float] = CANONICAL_CAN_COM_M,
    expected_inertia_kg_m2: Iterable[float] = CANONICAL_CAN_INERTIA_KG_M2,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Assert compiled Can mass, COM, inertia, freejoint, and geoms."""

    model, _ = _raw_model_data(sim_or_model)
    body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, can_body_name)
    expected_com = _finite_vector("expected_com_m", expected_com_m, 3)
    expected_inertia = _finite_vector("expected_inertia_kg_m2", expected_inertia_kg_m2, 3)
    if not np.isclose(float(model.body_mass[body_id]), float(expected_mass_kg), rtol=0.0, atol=tolerance):
        raise ShakeBenchMetricsError("compiled Can mass differs from canonical mass")
    if not np.allclose(model.body_ipos[body_id], expected_com, rtol=0.0, atol=tolerance):
        raise ShakeBenchMetricsError("compiled Can COM differs from canonical COM")
    if not np.allclose(model.body_inertia[body_id], expected_inertia, rtol=0.0, atol=tolerance):
        raise ShakeBenchMetricsError("compiled Can inertia differs from canonical equivalent cylinder")
    freejoints = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_bodyid[joint_id]) == body_id
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(freejoints) != 1:
        raise ShakeBenchMetricsError("Can must have exactly one freejoint")
    geom_names = _normalise_names(can_geom_names)
    geom_ids = [_mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names]
    if any(int(model.geom_bodyid[geom_id]) != body_id for geom_id in geom_ids):
        raise ShakeBenchMetricsError("a Can collision geom is attached to another body")
    return {
        "body_name": can_body_name,
        "body_id": body_id,
        "mass_kg": float(model.body_mass[body_id]),
        "com_m": np.asarray(model.body_ipos[body_id], dtype=float).tolist(),
        "inertia_kg_m2": np.asarray(model.body_inertia[body_id], dtype=float).tolist(),
        "freejoint_names": [_mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in freejoints],
        "collision_geom_names": list(geom_names),
    }


def audit_contact_pairs(
    sim_or_model: Any,
    *,
    can_geom_names: Iterable[str],
    table_geom_names: Iterable[str],
    target_bottom_geom_names: Iterable[str],
    target_wall_geom_names: Iterable[str],
    finger_pad_geom_names: Iterable[str],
    table_sliding_mu: float = 0.30,
    target_sliding_mu: Optional[float] = None,
    finger_sliding_mu: float = 1.00,
    contact_profile: Optional[Mapping[str, Any]] = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Audit exact compiled contact roles and their pair-local friction.

    The function rejects extra Can pairs.  When ``contact_profile`` is
    supplied, every score-affecting pair field is checked against that
    profile; without it the historical Phase 04 structural-only audit is
    preserved.
    """

    model, _ = _raw_model_data(sim_or_model)
    can_names = _normalise_names(can_geom_names)
    table_names = _normalise_names(table_geom_names)
    bottom_names = _normalise_names(target_bottom_geom_names)
    wall_names = _normalise_names(target_wall_geom_names)
    finger_names = _normalise_names(finger_pad_geom_names)
    expected_condim = None
    expected_torsional_mu = None
    expected_rolling_mu = None
    expected_margin_m = None
    expected_gap_m = None
    expected_solref = None
    expected_solimp = None
    isotropic_pair_5d = False
    if contact_profile is not None:
        if not isinstance(contact_profile, Mapping):
            raise ShakeBenchMetricsError("contact_profile must be a mapping")
        expected_condim = int(contact_profile["condim"])
        expected_torsional_mu = float(contact_profile["torsional_mu"])
        expected_rolling_mu = float(contact_profile["rolling_mu"])
        expected_margin_m = float(contact_profile["margin_m"])
        expected_gap_m = float(contact_profile["gap_m"])
        expected_solref = np.asarray(contact_profile["solref"], dtype=float)
        expected_solimp = np.asarray(contact_profile["solimp"], dtype=float)
        isotropic_pair_5d = contact_profile.get("friction_encoding") == "isotropic_pair_5d"
        if expected_solref.shape != (2,) or expected_solimp.shape != (5,):
            raise ShakeBenchMetricsError("contact_profile solref/solimp has the wrong shape")
    expected = {}
    for can_name in can_names:
        for other_name in table_names:
            expected[frozenset((can_name, other_name))] = (CONTACT_INTERFACE_TABLE_OBJECT, table_sliding_mu)
        for other_name in bottom_names + wall_names:
            expected[frozenset((can_name, other_name))] = (
                CONTACT_INTERFACE_TARGET_OBJECT,
                table_sliding_mu if target_sliding_mu is None else target_sliding_mu,
            )
        for other_name in finger_names:
            expected[frozenset((can_name, other_name))] = (CONTACT_INTERFACE_FINGER_OBJECT, finger_sliding_mu)
    roles = defaultdict(list)
    seen = set()
    pair_records = []
    for pair_id in range(int(model.npair)):
        geom1 = _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair_id]))
        geom2 = _mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair_id]))
        key = frozenset((geom1, geom2))
        if key not in expected:
            if geom1 in can_names or geom2 in can_names:
                raise ShakeBenchMetricsError(f"unexpected explicit Can contact pair {geom1!r}, {geom2!r}")
            continue
        if key in seen:
            raise ShakeBenchMetricsError(f"duplicate explicit Can contact pair {geom1!r}, {geom2!r}")
        seen.add(key)
        interface, expected_mu = expected[key]
        friction = np.asarray(model.pair_friction[pair_id], dtype=float)
        if friction.size < 1 or not np.isclose(friction[0], expected_mu, rtol=0.0, atol=tolerance):
            raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong sliding friction")
        if contact_profile is not None:
            if friction.size < 3:
                raise ShakeBenchMetricsError("compiled contact pair does not expose torsional/rolling friction")
            if isotropic_pair_5d:
                if friction.size != 5 or not np.allclose(friction[:2], expected_mu, rtol=0.0, atol=tolerance):
                    raise ShakeBenchMetricsError(
                        f"contact pair {geom1!r}, {geom2!r} is not isotropic in both sliding directions"
                    )
                if not np.isclose(friction[2], expected_torsional_mu, rtol=0.0, atol=tolerance):
                    raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong torsional friction")
                if not np.allclose(friction[3:], expected_rolling_mu, rtol=0.0, atol=tolerance):
                    raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong rolling friction")
            else:
                if not np.isclose(friction[1], expected_torsional_mu, rtol=0.0, atol=tolerance):
                    raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong torsional friction")
                if not np.allclose(friction[2:], expected_rolling_mu, rtol=0.0, atol=tolerance):
                    raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong rolling friction")
            if not hasattr(model, "pair_dim") or int(model.pair_dim[pair_id]) != expected_condim:
                raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong condim")
            if not np.isclose(model.pair_margin[pair_id], expected_margin_m, rtol=0.0, atol=tolerance):
                raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong margin")
            if not np.isclose(model.pair_gap[pair_id], expected_gap_m, rtol=0.0, atol=tolerance):
                raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong gap")
            if not np.allclose(model.pair_solref[pair_id], expected_solref, rtol=0.0, atol=tolerance):
                raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong solref")
            if not np.allclose(model.pair_solimp[pair_id], expected_solimp, rtol=0.0, atol=tolerance):
                raise ShakeBenchMetricsError(f"contact pair {geom1!r}, {geom2!r} has the wrong solimp")
        record = {
            "pair_id": pair_id,
            "geom1": geom1,
            "geom2": geom2,
            "interface": interface,
            "friction": friction.tolist(),
            "sliding_mu": float(friction[0]),
            "condim": int(model.pair_dim[pair_id]) if hasattr(model, "pair_dim") else None,
            "margin_m": float(model.pair_margin[pair_id]) if hasattr(model, "pair_margin") else None,
            "gap_m": float(model.pair_gap[pair_id]) if hasattr(model, "pair_gap") else None,
            "solref": np.asarray(model.pair_solref[pair_id], dtype=float).tolist(),
            "solimp": np.asarray(model.pair_solimp[pair_id], dtype=float).tolist(),
        }
        pair_records.append(record)
        roles[interface].append(record)
    missing = set(expected) - seen
    if missing:
        raise ShakeBenchMetricsError(f"compiled model is missing {len(missing)} explicit Can contact pairs")

    all_geom_names = [_mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(int(model.ngeom))]
    geometry_friction = {
        name: np.asarray(model.geom_friction[geom_id], dtype=float).tolist()
        for geom_id, name in enumerate(all_geom_names)
    }
    geometry_contact_bits = {
        name: [int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])]
        for geom_id, name in enumerate(all_geom_names)
    }
    return {
        "pair_count": len(pair_records),
        "pairs": pair_records,
        "roles": {key: list(value) for key, value in roles.items()},
        "geometry_friction": geometry_friction,
        "geometry_contact_bits": geometry_contact_bits,
        "contact_profile": (
            {
                "scope": "explicit_pair",
                "condim": expected_condim,
                "torsional_mu": expected_torsional_mu,
                "rolling_mu": expected_rolling_mu,
                "margin_m": expected_margin_m,
                "gap_m": expected_gap_m,
                "solref": None if expected_solref is None else expected_solref.tolist(),
                "solimp": None if expected_solimp is None else expected_solimp.tolist(),
                "frozen": contact_profile is not None,
            }
            if contact_profile is not None
            else None
        ),
        "provisional_fields": {
            "condim": "validated only when an explicit Phase 06 contact profile is supplied",
            "margin_gap": "validated only when an explicit Phase 06 contact profile is supplied",
            "solref_solimp": "validated only when an explicit Phase 06 contact profile is supplied",
        },
    }


@dataclass(frozen=True)
class MetricsSnapshot:
    """Stateful task metrics at one sample."""

    time_s: float
    can: Mapping[str, PoseTwist]
    contacts: ContactReport
    table_slip_distance_m: float
    table_slip_speed_m_s: float
    first_slip_time_s: Optional[float]
    in_hand_translation_slip_m: float
    in_hand_rotation_slip_rad: float
    finger_contact_loss_after_grasp: bool
    driver_response: Mapping[str, Any]
    table_response: Mapping[str, Any]
    success_snapshot: SuccessSnapshot
    success: Optional[SuccessEvaluation] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "can": {key: value.to_dict() for key, value in self.can.items()},
            "contacts": self.contacts.to_dict(),
            "max_illegal_penetration_m": self.contacts.max_penetration_m,
            "table_slip_distance_m": self.table_slip_distance_m,
            "table_slip_speed_m_s": self.table_slip_speed_m_s,
            "first_slip_time_s": self.first_slip_time_s,
            "in_hand_translation_slip_m": self.in_hand_translation_slip_m,
            "in_hand_rotation_slip_rad": self.in_hand_rotation_slip_rad,
            "finger_contact_loss_after_grasp": self.finger_contact_loss_after_grasp,
            "driver_response": dict(self.driver_response),
            "table_response": dict(self.table_response),
            "success_snapshot": self.success_snapshot.to_dict(),
            "success": None if self.success is None else self.success.to_dict(),
        }


class ShakeBenchMetrics:
    """Stateful metrics collector used by ``VibrationPickPlaceCan``."""

    def __init__(
        self,
        *,
        can_body_name: str,
        can_geom_names: Iterable[str],
        robot_base_body_name: str,
        worktable_body_name: str,
        target_frame_local_origin_m: Iterable[float],
        target_inner_xy_m: Iterable[float],
        table_geom_names: Iterable[str],
        target_bottom_geom_names: Iterable[str],
        target_wall_geom_names: Iterable[str],
        finger_pad_geom_names: Iterable[str],
        gripper_body_name: str,
        deck_body_name: str = "deck",
        deck_driver: Any = None,
        dt_s: Optional[float] = None,
        slip_speed_threshold_m_s: float = 1e-3,
    ):
        self.can_body_name = can_body_name
        self.can_geom_names = tuple(_normalise_names(can_geom_names))
        self.robot_base_body_name = robot_base_body_name
        self.worktable_body_name = worktable_body_name
        self.target_frame_local_origin_m = _finite_vector("target_frame_local_origin_m", target_frame_local_origin_m, 3)
        target_xy = _finite_vector("target_inner_xy_m", target_inner_xy_m, 2)
        if np.any(target_xy <= 0.0):
            raise ShakeBenchMetricsError("target_inner_xy_m must be positive")
        self.target_inner_half_extents_m = target_xy / 2.0
        self.table_geom_names = tuple(_normalise_names(table_geom_names))
        self.target_bottom_geom_names = tuple(_normalise_names(target_bottom_geom_names))
        self.target_wall_geom_names = tuple(_normalise_names(target_wall_geom_names))
        self.finger_pad_geom_names = tuple(_normalise_names(finger_pad_geom_names))
        self.gripper_body_name = gripper_body_name
        self.deck_body_name = deck_body_name
        self.deck_driver = deck_driver
        self.dt_s = dt_s
        if (
            isinstance(slip_speed_threshold_m_s, (bool, np.bool_))
            or not np.isfinite(float(slip_speed_threshold_m_s))
            or float(slip_speed_threshold_m_s) < 0.0
        ):
            raise ShakeBenchMetricsError("slip_speed_threshold_m_s must be finite and non-negative")
        self.slip_speed_threshold_m_s = float(slip_speed_threshold_m_s)
        self._compiled_can_body_points: Optional[np.ndarray] = None
        self._compiled_can_model_identity: Optional[int] = None
        self.reset()

    def reset(self) -> None:
        self._initial_can_worktable_xy: Optional[np.ndarray] = None
        self._initial_can_gripper_pose: Optional[PoseTwist] = None
        self._first_slip_time_s: Optional[float] = None
        self._had_finger_contact = False
        self._last_finger_contact_loss_after_grasp = False
        self.latest: Optional[MetricsSnapshot] = None

    def _response(self, model: Any, data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        deck = pose_twist_in_frame(self._as_sim_or_model(model, data), self.deck_body_name)
        table = pose_twist_in_frame(self._as_sim_or_model(model, data), self.worktable_body_name)
        table_in_deck = pose_twist_in_frame(
            self._as_sim_or_model(model, data), self.worktable_body_name, self.deck_body_name
        )
        deck_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, self.deck_body_name)
        table_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, self.worktable_body_name)
        driver_response = {
            "deck_pose": deck.pose.copy(),
            "deck_twist": deck.twist.copy(),
            "deck_acceleration": _body_spatial_acceleration(model, data, deck_id),
            "actual_pose": deck.pose.copy(),
            "actual_twist": deck.twist.copy(),
        }
        if self.deck_driver is not None:
            try:
                driver_body_name = self.deck_driver.config.driver_body_name
                driver = pose_twist_in_frame(self._as_sim_or_model(model, data), driver_body_name)
                driver_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, driver_body_name)
                driver_response.update(
                    {
                        "driver_pose": driver.pose.copy(),
                        "driver_twist": driver.twist.copy(),
                        "driver_acceleration": _body_spatial_acceleration(model, data, driver_id),
                    }
                )
            except (AttributeError, ShakeBenchMetricsError):
                driver_response["driver_pose_available"] = False
            try:
                trace = self.deck_driver.trace
                if len(trace.sample_time_s):
                    driver_response.update(
                        {
                            "sample_time_s": float(trace.sample_time_s[-1]),
                            "command_pose": trace.sample_target_pose[-1].copy(),
                            "command_twist": trace.sample_target_twist[-1].copy(),
                            "command_acceleration": trace.sample_target_acceleration[-1].copy(),
                            "deck_tracking_pose_error": trace.deck_tracking_pose_error[-1].copy(),
                        }
                    )
            except (AttributeError, IndexError, TypeError):
                # Metrics remain useful in a physics-only fixture without a
                # runtime DeckDriver; absence is represented explicitly.
                driver_response["trace_available"] = False
        table_response = {
            "pose": table.pose.copy(),
            "twist": table.twist.copy(),
            "acceleration": _body_spatial_acceleration(model, data, table_id),
            "relative_to_deck_pose": table_in_deck.pose.copy(),
            "relative_to_deck_twist": table_in_deck.twist.copy(),
        }
        return driver_response, table_response

    def _success_primitive(self, model: Any, raw_data: Any) -> tuple[PoseTwist, np.ndarray, ContactReport]:
        """Shared compiled-geometry/contact extraction for success and reports."""

        sim_view = self._as_sim_or_model(model, raw_data)
        worktable_body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, self.worktable_body_name)
        worktable_rotation = np.asarray(raw_data.xmat[worktable_body_id], dtype=float).reshape(3, 3)
        can_target = can_pose_twist_in_frame(
            sim_view,
            self.can_body_name,
            self.worktable_body_name,
            frame_local_origin_m=self.target_frame_local_origin_m,
        )
        model_identity = id(model)
        if self._compiled_can_body_points is None or self._compiled_can_model_identity != model_identity:
            body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, self.can_body_name)
            points = []
            for geom_name in self.can_geom_names:
                geom_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if int(model.geom_bodyid[geom_id]) != body_id:
                    raise ShakeBenchMetricsError(
                        f"collision geom {geom_name!r} is not attached to body {self.can_body_name!r}"
                    )
                points.append(_mesh_vertices_in_body_frame(model, geom_id))
            if not points:
                raise ShakeBenchMetricsError("at least one compiled Can collision geom is required")
            self._compiled_can_body_points = np.concatenate(points, axis=0)
            self._compiled_can_model_identity = model_identity
        body_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, self.can_body_name)
        body_rotation = np.asarray(raw_data.xmat[body_id], dtype=float).reshape(3, 3)
        body_position = np.asarray(raw_data.xpos[body_id], dtype=float)
        world_points = body_position + self._compiled_can_body_points.dot(body_rotation.T)
        frame_position, frame_rotation, _, _ = _frame_world_pose_twist(
            model, raw_data, self.worktable_body_name, self.target_frame_local_origin_m
        )
        support_points_target = (world_points - frame_position).dot(frame_rotation)
        contacts = collect_contact_metrics(
            sim_view,
            can_geom_names=self.can_geom_names,
            table_geom_names=self.table_geom_names,
            target_bottom_geom_names=self.target_bottom_geom_names,
            target_wall_geom_names=self.target_wall_geom_names,
            finger_pad_geom_names=self.finger_pad_geom_names,
            dt_s=self.dt_s,
            support_frame_rotation=worktable_rotation,
        )
        return can_target, support_points_target, contacts

    def _success_snapshot_from_primitive(
        self, can_target: PoseTwist, support_points_target: np.ndarray, contacts: ContactReport
    ) -> SuccessSnapshot:
        return SuccessSnapshot(
            support_points_target_xy=support_points_target[:, :2],
            target_inner_half_extents_m=tuple(self.target_inner_half_extents_m),
            target_bottom_contact_present=contacts.target_bottom_contact_present,
            target_bottom_support_force_N=contacts.target_bottom_support_force_N,
            lower_support_z_m=float(np.min(support_points_target[:, 2])),
            finger_can_contact_present=contacts.finger_can_contact_present,
            relative_linear_speed_m_s=float(np.linalg.norm(can_target.linear_velocity_m_s)),
            relative_angular_speed_rad_s=float(np.linalg.norm(can_target.angular_velocity_rad_s)),
            illegal_penetration_m=contacts.max_penetration_m,
        )

    def success_snapshot(self, sim_or_model: Any, data: Any = None) -> SuccessSnapshot:
        """Return only the evaluator inputs for one internal physics step.

        This deliberately avoids the stateful diagnostic/report assembly in
        :meth:`update`.  The environment calls it at the physics rate so that
        a one-substep loss of containment, support, finger release, velocity,
        or penetration resets the continuous-success candidate window.
        """

        model, raw_data = _raw_model_data(sim_or_model, data)
        return self._success_snapshot_from_primitive(*self._success_primitive(model, raw_data))

    @staticmethod
    def _as_sim_or_model(model: Any, data: Any) -> Any:
        class _SimView:
            pass

        view = _SimView()
        view.model = model
        view.data = data
        return view

    def update(self, sim_or_model: Any, data: Any = None, *, time_s: Optional[float] = None) -> MetricsSnapshot:
        model, raw_data = _raw_model_data(sim_or_model, data)
        if time_s is None:
            time_s = float(raw_data.time)
        if isinstance(time_s, (bool, np.bool_)) or not np.isfinite(float(time_s)) or float(time_s) < 0.0:
            raise ShakeBenchMetricsError("time_s must be finite and non-negative")
        time_s = float(time_s)
        if self.latest is not None and time_s < self.latest.time_s - 1e-12:
            self.reset()

        sim_view = self._as_sim_or_model(model, raw_data)
        can = {
            "robot_base": can_pose_twist_in_frame(sim_view, self.can_body_name, self.robot_base_body_name),
            "worktable": can_pose_twist_in_frame(sim_view, self.can_body_name, self.worktable_body_name),
        }
        can_target_primitive, support_points_target, contacts = self._success_primitive(model, raw_data)
        can["target"] = can_target_primitive
        if self._initial_can_worktable_xy is None:
            self._initial_can_worktable_xy = can["worktable"].position_m[:2].copy()
        table_slip = float(np.linalg.norm(can["worktable"].position_m[:2] - self._initial_can_worktable_xy))
        table_slip_speed = float(np.linalg.norm(can["worktable"].linear_velocity_m_s[:2]))
        if (
            self._first_slip_time_s is None
            and contacts.table_contact_present
            and table_slip_speed > self.slip_speed_threshold_m_s
        ):
            self._first_slip_time_s = time_s

        hand_pose = can_pose_twist_in_frame(sim_view, self.can_body_name, self.gripper_body_name)
        if contacts.finger_can_contact_present:
            if self._initial_can_gripper_pose is None:
                self._initial_can_gripper_pose = hand_pose
            self._had_finger_contact = True
        if self._initial_can_gripper_pose is None:
            in_hand_translation = 0.0
            in_hand_rotation = 0.0
        else:
            in_hand_translation = float(
                np.linalg.norm(hand_pose.position_m - self._initial_can_gripper_pose.position_m)
            )
            initial_rotation = _quat_wxyz_to_mat(self._initial_can_gripper_pose.quaternion_wxyz)
            current_rotation = _quat_wxyz_to_mat(hand_pose.quaternion_wxyz)
            in_hand_rotation = float(
                np.linalg.norm(_rotation_vector_from_matrix(initial_rotation.T.dot(current_rotation)))
            )
        finger_contact_loss_after_grasp = bool(self._had_finger_contact and not contacts.finger_can_contact_present)
        self._last_finger_contact_loss_after_grasp = finger_contact_loss_after_grasp
        success_snapshot = self._success_snapshot_from_primitive(can_target_primitive, support_points_target, contacts)
        driver_response, table_response = self._response(model, raw_data)
        self.latest = MetricsSnapshot(
            time_s=time_s,
            can=can,
            contacts=contacts,
            table_slip_distance_m=table_slip,
            table_slip_speed_m_s=table_slip_speed,
            first_slip_time_s=self._first_slip_time_s,
            in_hand_translation_slip_m=in_hand_translation,
            in_hand_rotation_slip_rad=in_hand_rotation,
            finger_contact_loss_after_grasp=finger_contact_loss_after_grasp,
            driver_response=driver_response,
            table_response=table_response,
            success_snapshot=success_snapshot,
        )
        return self.latest

    def attach_success(self, evaluation: SuccessEvaluation) -> MetricsSnapshot:
        if self.latest is None:
            raise ShakeBenchMetricsError("cannot attach success before the first metrics update")
        self.latest = replace(self.latest, success=evaluation)
        return self.latest

    def to_dict(self) -> dict[str, Any]:
        if self.latest is None:
            return {}
        return self.latest.to_dict()


PHASE04_ENVIRONMENT_ARTIFACT_SCHEMA_ID = "shakebench.phase04.environment"
PHASE04_ENVIRONMENT_ARTIFACT_SCHEMA_VERSION = 2








# Compatibility-friendly names for callers that prefer evaluator/collector
# language over the concrete class names.
CanPoseTwist = PoseTwist
compute_can_pose_twist = can_pose_twist_in_frame
compute_support_points = collision_support_points_in_frame
CanSuccessEvaluator = VibrationSuccessEvaluator
SuccessEvaluator = VibrationSuccessEvaluator
SuccessLatch = VibrationSuccessEvaluator
MetricsCollector = ShakeBenchMetrics
ContactMetrics = ShakeBenchMetrics
ShakeBenchContactMetrics = ContactReport
audit_contact_roles = audit_contact_pairs

MU_TABLE_OBJECT = 0.30
MU_FINGER_OBJECT = 1.00
SUCCESS_HOLD_TIME_S = DEFAULT_SUCCESS_THRESHOLDS.hold_duration_s
