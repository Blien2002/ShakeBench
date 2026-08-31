"""Dynamic deck driver and auditable XML integration for ShakeBench.

The production topology used in this phase is deliberately explicit:

``deck_driver (mocap, no geoms) -> equality weld -> deck (free body)``

Bodies supplied through ``body_handles`` are moved under the ordinary dynamic
``deck`` body before MuJoCo compilation.  The processor never searches for
task-specific names such as ``table`` or ``cube``.  An omitted handle list is
valid for an empty physics probe; a requested but missing handle is an error.

``DeckDriver`` is an opt-in runtime companion.  When installed on a
``MujocoEnv`` it writes the mocap target in the new pre-physics hook, before
the corresponding ``step1`` / ``forward`` call, and records command,
application, and post-step sample timestamps together with six-axis state and
constraint diagnostics.  No one-step lead is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robosuite.utils import transform_utils as T


AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
TRACE_SCHEMA_ID = "shakebench.deck_driver.trace"
TRACE_SCHEMA_VERSION = 3
TRACE_FIELD_CONTRACT = MappingProxyType({
    "integration_target_time_s": "left-limit target q(t) evaluated for integration, episode-relative seconds",
    "sample_target_time_s": "right-limit target q(t+dt) evaluated after integration, episode-relative seconds",
    "integration_application_time_s": "mocap write time for q(t), episode-relative seconds",
    "sample_application_time_s": "mocap write time for q(t+dt), episode-relative seconds",
    "sample_time_s": "post-integration state sample data.time, episode-relative seconds",
    "command_pose": "world deck-origin left-limit integration target pose at integration_target_time_s [x,y,z,qw,qx,qy,qz], m and unit quaternion",
    "actual_pose": "world deck-origin pose [x,y,z,qw,qx,qy,qz], m and unit quaternion",
    "command_twist": "world spatial left-limit integration target twist at integration_target_time_s [linear_xyz,angular_xyz], m/s and rad/s",
    "actual_twist": "world spatial twist at deck origin [linear_xyz,angular_xyz], m/s and rad/s",
    "command_acceleration": "world spatial left-limit integration target acceleration at integration_target_time_s [linear_xyz,angular_xyz], m/s^2 and rad/s^2",
    "actual_acceleration": "world spatial acceleration at deck origin [linear_xyz,angular_xyz], m/s^2 and rad/s^2",
    "sample_target_pose": "world deck-origin right-limit target pose at sample_target_time_s [x,y,z,qw,qx,qy,qz], m and unit quaternion",
    "sample_target_twist": "world spatial right-limit target twist at sample_target_time_s [linear_xyz,angular_xyz], m/s and rad/s",
    "sample_target_acceleration": "world spatial right-limit target acceleration at sample_target_time_s [linear_xyz,angular_xyz], m/s^2 and rad/s^2",
    "deck_tracking_pose_error": "world pose error at sample_time_s: actual minus right-limit command, [translation,rotation-vector], m and rad",
    "weld_constraint_residual_raw": "MuJoCo efc_pos rows for this weld at sample_time_s, raw constraint-coordinate order",
    "weld_constraint_force_raw": "MuJoCo efc_force rows for this weld at sample_time_s, raw constraint-coordinate order",
    "solver_iterations": "maximum post-step solver_niter across islands",
    "solver_niter": "post-step MuJoCo solver_niter per island",
    "warning_number_delta": "post-step delta of MuJoCo warning.number since previous sample",
    "warning_lastinfo": "post-step MuJoCo warning.lastinfo per warning type",
})


class DeckDriverError(ValueError):
    """Base error for invalid dynamic-deck configuration or runtime state."""


class DeckXMLProcessingError(DeckDriverError):
    """Raised when a deck XML transformation cannot be applied safely."""


class DeckAuditError(DeckDriverError):
    """Raised when a compiled or uncompiled deck does not meet its contract."""


def _finite_float(name: str, value: Any, *, minimum: Optional[float] = None, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DeckDriverError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DeckDriverError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise DeckDriverError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        comparator = ">" if strict else ">="
        raise DeckDriverError(f"{name} must be {comparator} {minimum}")
    return result


def _vector(name: str, value: Iterable[Any], length: int) -> tuple[float, ...]:
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise DeckDriverError(f"{name} must contain {length} finite values") from exc
    if len(values) != length:
        raise DeckDriverError(f"{name} must contain {length} finite values")
    return tuple(_finite_float(f"{name}[{index}]", item) for index, item in enumerate(values))


def _format_vector(values: Iterable[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _parse_vector(name: str, value: Optional[str], length: int) -> tuple[float, ...]:
    if value is None:
        raise DeckAuditError(f"{name} is missing")
    try:
        values = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise DeckAuditError(f"{name} must contain {length} numbers") from exc
    if len(values) != length or not np.all(np.isfinite(values)):
        raise DeckAuditError(f"{name} must contain {length} finite numbers")
    return values


def _normalise_quat_wxyz(quaternion: Iterable[Any], name: str = "quaternion") -> np.ndarray:
    result = np.asarray(_vector(name, quaternion, 4), dtype=float)
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise DeckDriverError(f"{name} must not be the zero quaternion")
    return result / norm


def _quat_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=float,
    )


def _quat_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.array((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]), dtype=float) / float(
        np.dot(quaternion, quaternion)
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3-by-3 cross-product matrix for a finite vector."""

    x, y, z = np.asarray(vector, dtype=float)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)


def _so3_left_jacobian_coefficients(angle: float) -> tuple[float, float, float, float]:
    """Return ``A, B, dA/dtheta, dB/dtheta`` for the SO(3) left Jacobian."""

    if angle < 1e-4:
        angle_squared = angle * angle
        angle_fourth = angle_squared * angle_squared
        # Series are used only for the scalar coefficients; the mapping and
        # its time derivative remain analytic and do not finite-difference the
        # authored trajectory.
        a = 0.5 - angle_squared / 24.0 + angle_fourth / 720.0
        b = 1.0 / 6.0 - angle_squared / 120.0 + angle_fourth / 5040.0
        da = angle * (-1.0 / 12.0 + angle_squared / 180.0 - angle_fourth / 6720.0)
        db = angle * (-1.0 / 60.0 + angle_squared / 1260.0 - angle_fourth / 60480.0)
        return a, b, da, db
    sine = math.sin(angle)
    cosine = math.cos(angle)
    a = (1.0 - cosine) / angle**2
    b = (angle - sine) / angle**3
    da = (angle * sine - 2.0 * (1.0 - cosine)) / angle**3
    db = (angle * (1.0 - cosine) - 3.0 * (angle - sine)) / angle**4
    return a, b, da, db


def rotation_vector_to_spatial_angular_velocity(
    rotation_vector: Iterable[Any], rotation_vector_rate: Iterable[Any]
) -> np.ndarray:
    """Map exponential-coordinate rotation rate to spatial angular velocity.

    Args:
        rotation_vector: Relative SO(3) exponential-coordinate vector in the
            nominal/deck frame, in radians.
        rotation_vector_rate: Time derivative of ``rotation_vector`` in
            radians per second.

    Returns:
        numpy.ndarray: Spatial angular velocity expressed in the same nominal
            frame, in radians per second.
    """

    vector = np.asarray(_vector("rotation_vector", rotation_vector, 3), dtype=float)
    rate = np.asarray(_vector("rotation_vector_rate", rotation_vector_rate, 3), dtype=float)
    angle = float(np.linalg.norm(vector))
    a, b, _, _ = _so3_left_jacobian_coefficients(angle)
    cross_matrix = _skew(vector)
    return (np.eye(3) + a * cross_matrix + b * cross_matrix.dot(cross_matrix)).dot(rate)


def rotation_vector_to_spatial_angular_acceleration(
    rotation_vector: Iterable[Any],
    rotation_vector_rate: Iterable[Any],
    rotation_vector_acceleration: Iterable[Any],
) -> np.ndarray:
    """Map exponential-coordinate derivatives to spatial angular acceleration.

    Args:
        rotation_vector: Relative SO(3) exponential-coordinate vector in the
            nominal/deck frame, in radians.
        rotation_vector_rate: First derivative in radians per second.
        rotation_vector_acceleration: Second derivative in radians per second
            squared.

    Returns:
        numpy.ndarray: Spatial angular acceleration in the nominal frame, in
            radians per second squared.

    Notes:
        The derivative of the SO(3) left Jacobian is evaluated analytically.
        This preserves the distinction between a rotation-vector derivative
        and a spatial angular quantity for finite rotations.
    """

    vector = np.asarray(_vector("rotation_vector", rotation_vector, 3), dtype=float)
    rate = np.asarray(_vector("rotation_vector_rate", rotation_vector_rate, 3), dtype=float)
    acceleration = np.asarray(
        _vector("rotation_vector_acceleration", rotation_vector_acceleration, 3), dtype=float
    )
    angle = float(np.linalg.norm(vector))
    a, b, da, db = _so3_left_jacobian_coefficients(angle)
    cross_matrix = _skew(vector)
    rate_cross_matrix = _skew(rate)
    cross_square = cross_matrix.dot(cross_matrix)
    jacobian = np.eye(3) + a * cross_matrix + b * cross_square
    if angle <= 1e-14:
        angle_rate = 0.0
    else:
        angle_rate = float(np.dot(vector, rate) / angle)
    jacobian_rate = (
        da * angle_rate * cross_matrix
        + a * rate_cross_matrix
        + db * angle_rate * cross_square
        + b * (rate_cross_matrix.dot(cross_matrix) + cross_matrix.dot(rate_cross_matrix))
    )
    return jacobian.dot(acceleration) + jacobian_rate.dot(rate)


def rotation_vector_to_quat(rotation_vector: Iterable[Any]) -> np.ndarray:
    """Convert an exponential-coordinate rotation vector to wxyz quaternion."""

    vector = np.asarray(_vector("rotation_vector", rotation_vector, 3), dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-14:
        return np.array((1.0, 0.0, 0.0, 0.0), dtype=float)
    axis = vector / angle
    half = 0.5 * angle
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def quat_to_rotation_vector(quaternion: Iterable[Any]) -> np.ndarray:
    """Convert a wxyz quaternion to the shortest exponential-coordinate vector."""

    quat = _normalise_quat_wxyz(quaternion)
    if quat[0] < 0.0:
        quat = -quat
    sine_half = float(np.linalg.norm(quat[1:]))
    if sine_half <= 1e-14:
        return 2.0 * quat[1:]
    angle = 2.0 * math.atan2(sine_half, float(np.clip(quat[0], -1.0, 1.0)))
    return quat[1:] * (angle / sine_half)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        result = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * math.sqrt(max(1e-16, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            result = np.array(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(max(1e-16, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            result = np.array(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * math.sqrt(max(1e-16, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            result = np.array(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    return _normalise_quat_wxyz(result, name="rotation matrix quaternion")


def _local_body_pose(body: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(_parse_vector("body pos", body.get("pos", "0 0 0"), 3), dtype=float)
    orientation_attributes = [
        name for name in ("quat", "euler", "axisangle", "xyaxes", "zaxis") if body.get(name) is not None
    ]
    if not orientation_attributes:
        return position, np.eye(3)
    if len(orientation_attributes) != 1:
        raise DeckXMLProcessingError(
            "body pose uses multiple orientation attributes: " + ", ".join(orientation_attributes)
        )
    attribute = orientation_attributes[0]
    if attribute == "quat":
        return position, T.quat2mat(np.asarray(_parse_vector("body quat", body.get(attribute), 4))[[1, 2, 3, 0]])
    if attribute == "euler":
        return position, T.euler2mat(np.asarray(_parse_vector("body euler", body.get(attribute), 3)))
    if attribute == "axisangle":
        values = np.asarray(_parse_vector("body axisangle", body.get(attribute), 4))
        axis = values[:3]
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise DeckXMLProcessingError("body axisangle axis must not be zero")
        return position, T.quat2mat(rotation_vector_to_quat(axis / norm * values[3])[[1, 2, 3, 0]])
    if attribute == "xyaxes":
        values = np.asarray(_parse_vector("body xyaxes", body.get(attribute), 6)).reshape(2, 3)
        x_axis = values[0] / np.linalg.norm(values[0])
        y_axis = values[1] - x_axis * np.dot(x_axis, values[1])
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm <= 0.0:
            raise DeckXMLProcessingError("body xyaxes must contain independent axes")
        y_axis /= y_norm
        z_axis = np.cross(x_axis, y_axis)
        return position, np.column_stack((x_axis, y_axis, z_axis))
    raise DeckXMLProcessingError("body zaxis orientation is not supported by the reparent processor")


def _pose_compose(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]):
    first_position, first_rotation = first
    second_position, second_rotation = second
    return first_position + first_rotation.dot(second_position), first_rotation.dot(second_rotation)


def _pose_inverse(pose: tuple[np.ndarray, np.ndarray]):
    position, rotation = pose
    inverse_rotation = rotation.T
    return -inverse_rotation.dot(position), inverse_rotation


def _is_identity_pose(pose: tuple[np.ndarray, np.ndarray]) -> bool:
    return np.allclose(pose[0], 0.0, rtol=0.0, atol=1e-14) and np.allclose(
        pose[1], np.eye(3), rtol=0.0, atol=1e-14
    )


@dataclass(frozen=True, init=False)
class DeckDriverConfig:
    """Configuration for the generated deck driver and equality weld.

    Attributes:
        eq_solref: Canonical MuJoCo equality ``solref`` pair.
        eq_solimp: Canonical MuJoCo equality ``solimp`` quintuple.
        physics_timestep_s: Optional compiled model timestep used for the
            ``eq_solref >= 2 * dt`` fail-closed check.

    ``deck_mass_kg`` and ``deck_inertia_kg_m2`` are explicit provisional
    numerical implementation parameters.  They are intentionally not an
    official physics profile; Phase 06 owns the freeze decision. The
    constructor accepts ``weld_solref`` / ``weld_solimp`` only as boundary
    compatibility names; serialized configuration uses ``eq_solref`` /
    ``eq_solimp``.
    """

    driver_body_name: str = "deck_driver"
    deck_body_name: str = "deck"
    deck_freejoint_name: str = "deck_freejoint"
    weld_name: str = "deck_weld"
    driver_site_name: str = "deck_driver_site"
    deck_site_name: str = "deck_site"
    deck_mass_kg: float = 400.0
    deck_inertia_kg_m2: tuple[float, float, float] = (
        12.12,
        14.083333333333334,
        26.033333333333332,
    )
    deck_pos_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    deck_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    eq_solref: tuple[float, float] = (0.0002, 1.0)
    eq_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    physics_timestep_s: Optional[float] = None
    site_size_m: float = 0.01

    def __init__(
        self,
        driver_body_name: str = "deck_driver",
        deck_body_name: str = "deck",
        deck_freejoint_name: str = "deck_freejoint",
        weld_name: str = "deck_weld",
        driver_site_name: str = "deck_driver_site",
        deck_site_name: str = "deck_site",
        deck_mass_kg: float = 400.0,
        deck_inertia_kg_m2: Iterable[float] = (12.12, 14.083333333333334, 26.033333333333332),
        deck_pos_m: Iterable[float] = (0.0, 0.0, 0.0),
        deck_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        eq_solref: Optional[Iterable[float]] = None,
        eq_solimp: Optional[Iterable[float]] = None,
        physics_timestep_s: Optional[float] = None,
        site_size_m: float = 0.01,
        *,
        weld_solref: Optional[Iterable[float]] = None,
        weld_solimp: Optional[Iterable[float]] = None,
    ) -> None:
        if weld_solref is not None and eq_solref is not None and tuple(weld_solref) != tuple(eq_solref):
            raise DeckDriverError("eq_solref and legacy weld_solref specify different values")
        if weld_solimp is not None and eq_solimp is not None and tuple(weld_solimp) != tuple(eq_solimp):
            raise DeckDriverError("eq_solimp and legacy weld_solimp specify different values")
        if eq_solref is None:
            eq_solref = weld_solref
        if eq_solimp is None:
            eq_solimp = weld_solimp
        object.__setattr__(self, "driver_body_name", driver_body_name)
        object.__setattr__(self, "deck_body_name", deck_body_name)
        object.__setattr__(self, "deck_freejoint_name", deck_freejoint_name)
        object.__setattr__(self, "weld_name", weld_name)
        object.__setattr__(self, "driver_site_name", driver_site_name)
        object.__setattr__(self, "deck_site_name", deck_site_name)
        object.__setattr__(self, "deck_mass_kg", deck_mass_kg)
        object.__setattr__(self, "deck_inertia_kg_m2", deck_inertia_kg_m2)
        object.__setattr__(self, "deck_pos_m", deck_pos_m)
        object.__setattr__(self, "deck_quat_wxyz", deck_quat_wxyz)
        object.__setattr__(self, "eq_solref", (0.0002, 1.0) if eq_solref is None else eq_solref)
        object.__setattr__(
            self,
            "eq_solimp",
            (0.9, 0.95, 0.001, 0.5, 2.0) if eq_solimp is None else eq_solimp,
        )
        object.__setattr__(self, "physics_timestep_s", physics_timestep_s)
        object.__setattr__(self, "site_size_m", site_size_m)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "driver_body_name",
            "deck_body_name",
            "deck_freejoint_name",
            "weld_name",
            "driver_site_name",
            "deck_site_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DeckDriverError(f"{name} must be a non-empty string")
        if self.driver_body_name == self.deck_body_name:
            raise DeckDriverError("driver_body_name and deck_body_name must differ")
        if self.driver_site_name == self.deck_site_name:
            raise DeckDriverError("driver_site_name and deck_site_name must differ")
        mass = _finite_float("deck_mass_kg", self.deck_mass_kg, minimum=0.0, strict=True)
        inertia = _vector("deck_inertia_kg_m2", self.deck_inertia_kg_m2, 3)
        if any(value <= 0.0 for value in inertia):
            raise DeckDriverError("deck_inertia_kg_m2 must be strictly positive")
        position = _vector("deck_pos_m", self.deck_pos_m, 3)
        quaternion = _normalise_quat_wxyz(self.deck_quat_wxyz, "deck_quat_wxyz")
        solref = _vector("eq_solref", self.eq_solref, 2)
        if solref[0] <= 0.0 or solref[1] <= 0.0:
            raise DeckDriverError("eq_solref must contain positive time constant and damping ratio")
        solimp = _vector("eq_solimp", self.eq_solimp, 5)
        if not 0.0 <= solimp[0] <= solimp[1] <= 1.0:
            raise DeckDriverError("eq_solimp requires 0 <= dmin <= dmax <= 1")
        if solimp[2] <= 0.0 or not 0.0 <= solimp[3] <= 1.0 or solimp[4] <= 0.0:
            raise DeckDriverError("eq_solimp width, midpoint, and power are invalid")
        if self.physics_timestep_s is not None:
            dt = _finite_float("physics_timestep_s", self.physics_timestep_s, minimum=0.0, strict=True)
            if solref[0] < 2.0 * dt:
                raise DeckDriverError("eq_solref time constant must be at least 2 * physics_timestep_s")
        size = _finite_float("site_size_m", self.site_size_m, minimum=0.0, strict=True)
        object.__setattr__(self, "deck_mass_kg", mass)
        object.__setattr__(self, "deck_inertia_kg_m2", inertia)
        object.__setattr__(self, "deck_pos_m", position)
        object.__setattr__(self, "deck_quat_wxyz", tuple(quaternion))
        object.__setattr__(self, "eq_solref", solref)
        object.__setattr__(self, "eq_solimp", solimp)
        object.__setattr__(self, "site_size_m", size)
        if self.physics_timestep_s is not None:
            object.__setattr__(self, "physics_timestep_s", dt)

    @property
    def weld_solref(self) -> tuple[float, float]:
        """Compatibility alias for the canonical :attr:`eq_solref` field."""

        return self.eq_solref

    @property
    def weld_solimp(self) -> tuple[float, float, float, float, float]:
        """Compatibility alias for the canonical :attr:`eq_solimp` field."""

        return self.eq_solimp

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_body_name": self.driver_body_name,
            "deck_body_name": self.deck_body_name,
            "deck_freejoint_name": self.deck_freejoint_name,
            "weld_name": self.weld_name,
            "driver_site_name": self.driver_site_name,
            "deck_site_name": self.deck_site_name,
            "deck_mass_kg": self.deck_mass_kg,
            "deck_inertia_kg_m2": list(self.deck_inertia_kg_m2),
            "deck_pos_m": list(self.deck_pos_m),
            "deck_quat_wxyz": list(self.deck_quat_wxyz),
            "eq_solref": list(self.eq_solref),
            "eq_solimp": list(self.eq_solimp),
            "physics_timestep_s": self.physics_timestep_s,
            "site_size_m": self.site_size_m,
        }


@dataclass(frozen=True)
class DeckBodyHandle:
    """An explicit semantic role-to-body handle supplied by the environment."""

    role: str
    body_name: str


@dataclass(frozen=True)
class DeckBodyHandles:
    """Immutable-ish serializable container for role-based body handles."""

    roles: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalise_body_handles(self.roles)
        object.__setattr__(self, "roles", MappingProxyType(dict(normalized)))

    def to_dict(self) -> dict[str, str]:
        return dict(self.roles)


def _normalise_body_handles(body_handles: Any) -> dict[str, str]:
    if body_handles is None:
        return {}
    if isinstance(body_handles, DeckBodyHandles):
        body_handles = body_handles.roles
    if isinstance(body_handles, DeckBodyHandle):
        body_handles = {body_handles.role: body_handles.body_name}
    if not isinstance(body_handles, Mapping):
        raise DeckXMLProcessingError("body_handles must be a role-to-body mapping")
    normalized = {}
    for role, value in body_handles.items():
        if not isinstance(role, str) or not role.strip():
            raise DeckXMLProcessingError("body handle roles must be non-empty strings")
        if isinstance(value, DeckBodyHandle):
            value = value.body_name
        elif isinstance(value, Mapping):
            value = value.get("body_name", value.get("name"))
        if not isinstance(value, str) or not value.strip():
            raise DeckXMLProcessingError(f"body handle for role {role!r} must name a body")
        if role in normalized:
            raise DeckXMLProcessingError(f"duplicate body handle role {role!r}")
        if value in normalized.values():
            raise DeckXMLProcessingError(f"body {value!r} is assigned to multiple roles")
        normalized[role] = value
    return normalized


@dataclass(frozen=True)
class DeckXMLAudit:
    """Uncompiled deck audit returned by :func:`audit_deck_xml`."""

    parent_graph: Mapping[str, Optional[str]]
    role_body_names: Mapping[str, str]
    driver_body_name: str
    deck_body_name: str
    freejoint_name: str
    site_names: tuple[str, str]
    weld_name: str
    weld_body1: str
    weld_body2: str
    eq_solref: tuple[float, float]
    eq_solimp: tuple[float, float, float, float, float]
    deck_mass_kg: float
    deck_inertia_kg_m2: tuple[float, float, float]
    driver_geom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_graph", MappingProxyType(dict(self.parent_graph)))
        object.__setattr__(self, "role_body_names", MappingProxyType(dict(self.role_body_names)))

    @property
    def body_parent(self) -> Mapping[str, Optional[str]]:
        return self.parent_graph

    @property
    def role_handles(self) -> Mapping[str, str]:
        return self.role_body_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_graph": dict(self.parent_graph),
            "role_body_names": dict(self.role_body_names),
            "driver_body_name": self.driver_body_name,
            "deck_body_name": self.deck_body_name,
            "freejoint_name": self.freejoint_name,
            "site_names": list(self.site_names),
            "weld_name": self.weld_name,
            "weld_body1": self.weld_body1,
            "weld_body2": self.weld_body2,
            "eq_solref": list(self.eq_solref),
            "eq_solimp": list(self.eq_solimp),
            "deck_mass_kg": self.deck_mass_kg,
            "deck_inertia_kg_m2": list(self.deck_inertia_kg_m2),
            "driver_geom_names": list(self.driver_geom_names),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    @property
    def weld_solref(self) -> tuple[float, float]:
        """Compatibility alias for the canonical equality parameter."""

        return self.eq_solref

    @property
    def weld_solimp(self) -> tuple[float, float, float, float, float]:
        """Compatibility alias for the canonical equality parameter."""

        return self.eq_solimp


def _body_entries(worldbody: ET.Element):
    entries = []

    def visit(parent: ET.Element, ancestors: tuple[ET.Element, ...]) -> None:
        for child in list(parent):
            if child.tag != "body":
                continue
            entries.append((child, parent, ancestors))
            visit(child, ancestors + (child,))

    visit(worldbody, ())
    return entries


def _named_elements(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    result = {}
    for element in root.iter(tag):
        name = element.get("name")
        if name is None:
            continue
        if name in result:
            raise DeckXMLProcessingError(f"duplicate {tag} name {name!r} in XML")
        result[name] = element
    return result


def _find_named_child(parent: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    matches = [child for child in list(parent) if child.tag == tag and child.get("name") == name]
    if len(matches) > 1:
        raise DeckAuditError(f"duplicate {tag} {name!r}")
    return matches[0] if matches else None


class ShakeBenchDeckXMLProcessor:
    """Create and validate a dynamic deck topology in a merged MJCF string."""

    PROCESSOR_MARKER_NAME = "shakebench_deck_processor"

    def __init__(
        self,
        config: Optional[DeckDriverConfig] = None,
        body_handles: Any = None,
        required_roles: Iterable[str] = (),
        role_handles: Any = None,
    ) -> None:
        if body_handles is not None and role_handles is not None:
            raise DeckXMLProcessingError("provide only one of body_handles and role_handles")
        self.config = config if config is not None else DeckDriverConfig()
        self.body_handles = _normalise_body_handles(body_handles if body_handles is not None else role_handles)
        self.required_roles = (required_roles,) if isinstance(required_roles, str) else tuple(required_roles)
        for role in self.required_roles:
            if not isinstance(role, str) or not role.strip():
                raise DeckXMLProcessingError("required_roles must contain non-empty strings")
            if role not in self.body_handles:
                raise DeckXMLProcessingError(f"required body role {role!r} has no explicit handle")

    def __call__(self, xml_string: str) -> str:
        return self.process(xml_string)

    @property
    def role_handles(self) -> dict[str, str]:
        return dict(self.body_handles)

    def _validate_handles(
        self,
        body_index: Mapping[str, tuple[ET.Element, ET.Element, tuple[ET.Element, ...]]],
    ) -> None:
        for role, body_name in self.body_handles.items():
            if body_name not in body_index:
                raise DeckXMLProcessingError(
                    f"body role {role!r} refers to missing body {body_name!r}; refusing implicit name lookup"
                )
            body = body_index[body_name][0]
            if any(
                element.tag == "freejoint" or (element.tag == "joint" and element.get("type") == "free")
                for element in body.iter()
            ):
                raise DeckXMLProcessingError(
                    f"body role {role!r} contains a freejoint and cannot be nested under dynamic deck"
                )
        selected = [body_index[name][0] for name in self.body_handles.values()]
        for index, body in enumerate(selected):
            for other in selected[index + 1 :]:
                if body in body_index[other.get("name")][2] or other in body_index[body.get("name")][2]:
                    raise DeckXMLProcessingError("ancestor and descendant body handles are ambiguous")

    def _reparent_body(
        self,
        body: ET.Element,
        old_parent: ET.Element,
        old_world_pose: tuple[np.ndarray, np.ndarray],
        deck_world_pose: tuple[np.ndarray, np.ndarray],
        deck: ET.Element,
    ) -> None:
        old_parent.remove(body)
        if not _is_identity_pose(deck_world_pose):
            local_pose = _pose_compose(_pose_inverse(deck_world_pose), old_world_pose)
            body.set("pos", _format_vector(local_pose[0]))
            body.set("quat", _format_vector(_matrix_to_quat_wxyz(local_pose[1])))
            for attribute in ("euler", "axisangle", "xyaxes", "zaxis"):
                body.attrib.pop(attribute, None)
        deck.append(body)

    def process(self, xml_string: str) -> str:
        if not isinstance(xml_string, str):
            raise DeckXMLProcessingError("xml_string must be a string")
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            raise DeckXMLProcessingError("invalid MJCF XML") from exc
        if root.tag != "mujoco":
            raise DeckXMLProcessingError("root element must be <mujoco>")
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise DeckXMLProcessingError("MJCF is missing <worldbody>")
        custom = root.find("custom")
        if custom is not None and any(
            child.tag == "text" and child.get("name") == self.PROCESSOR_MARKER_NAME for child in custom
        ):
            raise DeckXMLProcessingError("dynamic deck processor was applied more than once")

        # Generated names are the duplicate-application marker.  Refusing any
        # collision also protects an environment that already has a body named
        # ``deck`` but did not opt into ShakeBench.
        body_index = {}
        for body, parent, ancestors in _body_entries(worldbody):
            name = body.get("name")
            if name is None:
                raise DeckXMLProcessingError("all bodies must be named before role-based reparenting")
            if name in body_index:
                raise DeckXMLProcessingError(f"duplicate body name {name!r}")
            body_index[name] = (body, parent, ancestors)
        if self.config.driver_body_name in body_index or self.config.deck_body_name in body_index:
            raise DeckXMLProcessingError("dynamic deck processor was applied more than once or names collide")

        self._validate_handles(body_index)
        role_world_poses = {
            role: self._world_pose(body, body_index, worldbody)
            for role, body_name in self.body_handles.items()
            for body, _, _ in (body_index[body_name],)
        }

        generated_joint_names = _named_elements(root, "joint")
        for name, element in _named_elements(root, "freejoint").items():
            if name in generated_joint_names:
                raise DeckXMLProcessingError(f"duplicate joint name {name!r} in XML")
            generated_joint_names[name] = element
        generated_site_names = _named_elements(root, "site")
        generated_equality_names = {}
        equality_root = root.find("equality")
        if equality_root is not None:
            for element in equality_root:
                name = element.get("name")
                if name is None:
                    continue
                if name in generated_equality_names:
                    raise DeckXMLProcessingError(f"duplicate equality name {name!r} in XML")
                generated_equality_names[name] = element
        for name, category in (
            (self.config.deck_freejoint_name, "joint"),
            (self.config.driver_site_name, "site"),
            (self.config.deck_site_name, "site"),
            (self.config.weld_name, "weld"),
        ):
            name_registry = {
                "joint": generated_joint_names,
                "site": generated_site_names,
                "weld": generated_equality_names,
            }[category]
            if name in name_registry:
                raise DeckXMLProcessingError(f"generated {category} name {name!r} already exists")

        deck_pose = (
            np.asarray(self.config.deck_pos_m, dtype=float),
            T.quat2mat(np.asarray(self.config.deck_quat_wxyz)[[1, 2, 3, 0]]),
        )
        driver = ET.Element(
            "body",
            {
                "name": self.config.driver_body_name,
                "mocap": "true",
                "pos": _format_vector(self.config.deck_pos_m),
                "quat": _format_vector(self.config.deck_quat_wxyz),
            },
        )
        ET.SubElement(
            driver,
            "site",
            {
                "name": self.config.driver_site_name,
                "pos": "0 0 0",
                "type": "sphere",
                "size": format(self.config.site_size_m, ".17g"),
                "rgba": "0.9 0.2 0.2 0",
            },
        )
        deck = ET.Element(
            "body",
            {
                "name": self.config.deck_body_name,
                "pos": _format_vector(self.config.deck_pos_m),
                "quat": _format_vector(self.config.deck_quat_wxyz),
            },
        )
        ET.SubElement(deck, "freejoint", {"name": self.config.deck_freejoint_name})
        ET.SubElement(
            deck,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": format(self.config.deck_mass_kg, ".17g"),
                "diaginertia": _format_vector(self.config.deck_inertia_kg_m2),
            },
        )
        ET.SubElement(
            deck,
            "site",
            {
                "name": self.config.deck_site_name,
                "pos": "0 0 0",
                "type": "sphere",
                "size": format(self.config.site_size_m, ".17g"),
                "rgba": "0.2 0.5 0.9 0",
            },
        )

        # Add generated roots first, then move explicitly selected bodies into
        # the dynamic deck.  Existing world-level bodies not named by a handle
        # remain untouched.
        worldbody.append(driver)
        worldbody.append(deck)
        for role, body_name in self.body_handles.items():
            body, old_parent, _ = body_index[body_name]
            self._reparent_body(body, old_parent, role_world_poses[role], deck_pose, deck)

        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            {
                "name": self.config.weld_name,
                "body1": self.config.driver_body_name,
                "body2": self.config.deck_body_name,
                "relpose": "0 0 0 1 0 0 0",
                "solref": _format_vector(self.config.eq_solref),
                "solimp": _format_vector(self.config.eq_solimp),
            },
        )
        if custom is None:
            custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "text", {"name": self.PROCESSOR_MARKER_NAME, "data": "v1"})
        result = ET.tostring(root, encoding="utf8").decode("utf8")
        audit_deck_xml(result, self.config, self.body_handles)
        return result

    @staticmethod
    def _world_pose(
        body: ET.Element,
        body_index: Mapping[str, tuple[ET.Element, ET.Element, tuple[ET.Element, ...]]],
        worldbody: ET.Element,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, parent, _ = body_index[body.get("name")]
        local = _local_body_pose(body)
        if parent is worldbody:
            return local
        parent_name = parent.get("name")
        if parent_name not in body_index:
            raise DeckXMLProcessingError(f"cannot resolve parent pose for body {body.get('name')!r}")
        return _pose_compose(ShakeBenchDeckXMLProcessor._world_pose(parent, body_index, worldbody), local)


def make_deck_xml_processor(
    config: Optional[DeckDriverConfig] = None,
    body_handles: Any = None,
    required_roles: Iterable[str] = (),
) -> ShakeBenchDeckXMLProcessor:
    """Build the explicit XML processor used by a ShakeBench environment."""

    return ShakeBenchDeckXMLProcessor(config=config, body_handles=body_handles, required_roles=required_roles)


def _audit_body_index(worldbody: ET.Element):
    entries = _body_entries(worldbody)
    by_name = {}
    for body, parent, ancestors in entries:
        name = body.get("name")
        if name is None:
            raise DeckAuditError("all bodies must have names")
        if name in by_name:
            raise DeckAuditError(f"duplicate body name {name!r}")
        by_name[name] = (body, parent, ancestors)
    return by_name


def audit_deck_xml(
    xml_string: str,
    config: Optional[DeckDriverConfig] = None,
    body_handles: Any = None,
) -> DeckXMLAudit:
    """Validate generated MJCF and return its inspectable parent/model audit."""

    config = config if config is not None else DeckDriverConfig()
    handles = _normalise_body_handles(body_handles)
    try:
        root = ET.fromstring(xml_string)
    except (TypeError, ET.ParseError) as exc:
        raise DeckAuditError("invalid MJCF XML") from exc
    worldbody = root.find("worldbody")
    if root.tag != "mujoco" or worldbody is None:
        raise DeckAuditError("MJCF must contain a <mujoco><worldbody> root")
    body_index = _audit_body_index(worldbody)
    for name in (config.driver_body_name, config.deck_body_name):
        if name not in body_index:
            raise DeckAuditError(f"missing generated body {name!r}")
    driver, driver_parent, _ = body_index[config.driver_body_name]
    deck, deck_parent, _ = body_index[config.deck_body_name]
    if driver_parent is not worldbody or deck_parent is not worldbody:
        raise DeckAuditError("driver and deck must both be direct worldbody children")
    if driver.get("mocap", "false").lower() != "true":
        raise DeckAuditError("deck driver must be a mocap body")
    driver_geom_names = tuple(element.get("name") for element in driver.iter("geom") if element.get("name") is not None)
    if driver_geom_names:
        raise DeckAuditError("deck driver must not contain geoms or task contact")
    if any(element.tag in {"joint", "freejoint"} for element in driver.iter()):
        raise DeckAuditError("deck driver must not contain dynamic joints")

    freejoint = _find_named_child(deck, "freejoint", config.deck_freejoint_name)
    if freejoint is None:
        raise DeckAuditError(f"deck is missing freejoint {config.deck_freejoint_name!r}")
    inertials = [child for child in list(deck) if child.tag == "inertial"]
    if len(inertials) != 1:
        raise DeckAuditError("deck must have exactly one explicit direct inertial")
    inertial = inertials[0]
    deck_mass = _finite_float("deck inertial mass", inertial.get("mass"), minimum=0.0, strict=True)
    deck_inertia = _parse_vector("deck inertial diaginertia", inertial.get("diaginertia"), 3)
    for name in (config.driver_site_name, config.deck_site_name):
        if _find_named_child(driver if name == config.driver_site_name else deck, "site", name) is None:
            raise DeckAuditError(f"missing generated site {name!r}")

    equality = root.find("equality")
    if equality is None:
        raise DeckAuditError("MJCF is missing <equality>")
    weld = _find_named_child(equality, "weld", config.weld_name)
    if weld is None:
        raise DeckAuditError(f"missing generated weld {config.weld_name!r}")
    if weld.get("body1") != config.driver_body_name or weld.get("body2") != config.deck_body_name:
        raise DeckAuditError("weld must connect driver body1 to dynamic deck body2")
    weld_solref = _parse_vector("weld solref", weld.get("solref"), 2)
    weld_solimp = _parse_vector("weld solimp", weld.get("solimp"), 5)
    if not np.allclose(weld_solref, config.eq_solref, rtol=0.0, atol=1e-14):
        raise DeckAuditError("compiled XML weld solref differs from requested configuration")
    if not np.allclose(weld_solimp, config.eq_solimp, rtol=0.0, atol=1e-14):
        raise DeckAuditError("compiled XML weld solimp differs from requested configuration")
    for role, body_name in handles.items():
        if body_name not in body_index:
            raise DeckAuditError(f"role {role!r} refers to missing body {body_name!r}")
        _, parent, _ = body_index[body_name]
        if parent is not deck:
            raise DeckAuditError(f"role {role!r} body {body_name!r} was not reparented under deck")

    parent_graph = {}
    for name, (_, parent, _) in body_index.items():
        parent_graph[name] = None if parent is worldbody else parent.get("name")
    return DeckXMLAudit(
        parent_graph=parent_graph,
        role_body_names=handles,
        driver_body_name=config.driver_body_name,
        deck_body_name=config.deck_body_name,
        freejoint_name=config.deck_freejoint_name,
        site_names=(config.driver_site_name, config.deck_site_name),
        weld_name=config.weld_name,
        weld_body1=weld.get("body1"),
        weld_body2=weld.get("body2"),
        eq_solref=weld_solref,
        eq_solimp=weld_solimp,
        deck_mass_kg=deck_mass,
        deck_inertia_kg_m2=deck_inertia,
        driver_geom_names=driver_geom_names,
    )


def _raw_model_and_data(sim_or_model: Any, data: Any = None):
    if data is None and hasattr(sim_or_model, "data"):
        data = sim_or_model.data
    model = getattr(sim_or_model, "model", sim_or_model)
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    return raw_model, raw_data


def _mujoco_name(raw_model: Any, objtype: Any, object_id: int) -> Optional[str]:
    value = mujoco.mj_id2name(raw_model, objtype, int(object_id))
    return None if value is None else str(value)


def _mujoco_id(raw_model: Any, objtype: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(raw_model, objtype, name))
    if object_id < 0:
        raise DeckAuditError(f"compiled model is missing {name!r}")
    return object_id


def audit_compiled_deck_model(
    sim_or_model: Any,
    config: Optional[DeckDriverConfig] = None,
    body_handles: Any = None,
) -> dict[str, Any]:
    """Audit parent graph, inertia, freejoint, sites, and equality parameters."""

    config = config if config is not None else DeckDriverConfig()
    handles = _normalise_body_handles(body_handles)
    raw_model, _ = _raw_model_and_data(sim_or_model)
    driver_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, config.driver_body_name)
    deck_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, config.deck_body_name)
    parent_graph = {}
    for body_id in range(int(raw_model.nbody)):
        parent_id = int(raw_model.body_parentid[body_id])
        parent_graph[_mujoco_name(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_id)] = (
            None if parent_id == 0 else _mujoco_name(raw_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        )
    if parent_graph[config.driver_body_name] is not None or parent_graph[config.deck_body_name] is not None:
        raise DeckAuditError("compiled driver and deck must be worldbody children")
    if int(raw_model.body_mocapid[driver_id]) < 0:
        raise DeckAuditError("compiled deck driver is not mocap")
    if int(raw_model.body_mocapid[deck_id]) >= 0:
        raise DeckAuditError("dynamic deck must not be mocap")

    freejoint_ids = [
        joint_id
        for joint_id in range(int(raw_model.njnt))
        if int(raw_model.jnt_bodyid[joint_id]) == deck_id
        and int(raw_model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    freejoint_names = tuple(_mujoco_name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in freejoint_ids)
    if freejoint_names != (config.deck_freejoint_name,):
        raise DeckAuditError("compiled deck must have exactly its generated freejoint")

    if not np.isclose(float(raw_model.body_mass[deck_id]), config.deck_mass_kg, rtol=0.0, atol=1e-10):
        raise DeckAuditError("compiled deck mass differs from explicit inertial")
    if not np.allclose(raw_model.body_inertia[deck_id], config.deck_inertia_kg_m2, rtol=0.0, atol=1e-10):
        raise DeckAuditError("compiled deck inertia differs from explicit inertial")
    driver_geom_names = tuple(
        _mujoco_name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(int(raw_model.ngeom))
        if int(raw_model.geom_bodyid[geom_id]) == driver_id
    )
    if driver_geom_names:
        raise DeckAuditError("compiled driver contains contact geometry")

    for site_name in (config.driver_site_name, config.deck_site_name):
        site_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        expected_body = driver_id if site_name == config.driver_site_name else deck_id
        if int(raw_model.site_bodyid[site_id]) != expected_body:
            raise DeckAuditError(f"site {site_name!r} is attached to the wrong body")

    weld_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, config.weld_name)
    if int(raw_model.eq_type[weld_id]) != int(mujoco.mjtEq.mjEQ_WELD):
        raise DeckAuditError("generated equality is not a weld")
    if int(raw_model.eq_obj1id[weld_id]) != driver_id or int(raw_model.eq_obj2id[weld_id]) != deck_id:
        raise DeckAuditError("compiled weld connects the wrong bodies")
    if not np.allclose(raw_model.eq_solref[weld_id], config.eq_solref, rtol=0.0, atol=1e-14):
        raise DeckAuditError("compiled equality solref differs from configuration")
    if not np.allclose(raw_model.eq_solimp[weld_id], config.eq_solimp, rtol=0.0, atol=1e-14):
        raise DeckAuditError("compiled equality solimp differs from configuration")

    for role, body_name in handles.items():
        body_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if int(raw_model.body_parentid[body_id]) != deck_id:
            raise DeckAuditError(f"compiled role {role!r} body is not a deck child")
    return {
        "parent_graph": parent_graph,
        "role_body_names": handles,
        "driver_body_id": driver_id,
        "deck_body_id": deck_id,
        "freejoint_names": list(freejoint_names),
        "driver_site_name": config.driver_site_name,
        "deck_site_name": config.deck_site_name,
        "weld_name": config.weld_name,
        "eq_solref": np.asarray(raw_model.eq_solref[weld_id], dtype=float).tolist(),
        "eq_solimp": np.asarray(raw_model.eq_solimp[weld_id], dtype=float).tolist(),
        "deck_mass_kg": float(raw_model.body_mass[deck_id]),
        "deck_inertia_kg_m2": np.asarray(raw_model.body_inertia[deck_id], dtype=float).tolist(),
        "driver_geom_names": list(driver_geom_names),
    }


def process_deck_xml(
    xml_string: str,
    config: Optional[DeckDriverConfig] = None,
    body_handles: Any = None,
    required_roles: Iterable[str] = (),
) -> str:
    """Apply the role-based processor in one call for environment setup code."""

    return ShakeBenchDeckXMLProcessor(
        config=config,
        body_handles=body_handles,
        required_roles=required_roles,
    )(xml_string)


@dataclass(frozen=True)
class DeckCommand:
    """One six-axis command in ``(translation, rotation-vector)`` coordinates.

    Attributes:
        pose: ``[tx, ty, tz, rx, ry, rz]`` position coordinates in meters and
            radians, relative to the configured deck pose.
        twist: Coordinate derivatives in meters/second and radians/second.
        acceleration: Coordinate second derivatives in meters/second² and
            radians/second². Rotation fields are converted through the SO(3)
            Jacobian before entering the trace's spatial fields.
    """

    pose: np.ndarray
    twist: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        for name in ("pose", "twist", "acceleration"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise DeckDriverError(f"DeckCommand.{name} must have shape (6,) and finite values")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def command_to_world_state(
    command: DeckCommand, config: Optional[DeckDriverConfig] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map one authored deck command to world pose and spatial derivatives.

    Args:
        command: Relative command whose translation is in meters and whose
            rotation is an SO(3) exponential-coordinate vector in radians.
            Its derivative fields are coordinate derivatives, not spatial
            angular quantities.
        config: Deck driver pose convention. If omitted, the default config is
            used.

    Returns:
        tuple: ``(position, quaternion_wxyz, twist, acceleration)``. The pose
            is world-frame position plus quaternion; twist and acceleration
            are world spatial ``[linear_xyz, angular_xyz]`` quantities at the
            deck origin.
    """

    if not isinstance(command, DeckCommand):
        raise DeckDriverError("command must be a DeckCommand")
    config = config if config is not None else DeckDriverConfig()
    base_position = np.asarray(config.deck_pos_m, dtype=float)
    base_rotation = T.quat2mat(np.asarray(config.deck_quat_wxyz)[[1, 2, 3, 0]])
    position = base_position + base_rotation.dot(command.pose[:3])
    relative_quaternion = rotation_vector_to_quat(command.pose[3:])
    quaternion = _quat_multiply_wxyz(np.asarray(config.deck_quat_wxyz, dtype=float), relative_quaternion)
    relative_angular_velocity = rotation_vector_to_spatial_angular_velocity(command.pose[3:], command.twist[3:])
    relative_angular_acceleration = rotation_vector_to_spatial_angular_acceleration(
        command.pose[3:], command.twist[3:], command.acceleration[3:]
    )
    twist = np.concatenate((base_rotation.dot(command.twist[:3]), base_rotation.dot(relative_angular_velocity)))
    acceleration = np.concatenate(
        (base_rotation.dot(command.acceleration[:3]), base_rotation.dot(relative_angular_acceleration))
    )
    return position, _normalise_quat_wxyz(quaternion), twist, acceleration


@dataclass(frozen=True)
class DeckDriverTrace:
    """Immutable snapshot of driver instrumentation for one simulation run.

    Field contract:

    * ``integration_target_time_s`` is the left-limit target-evaluation time
      ``t`` supplied to the trajectory; ``command_pose/twist/acceleration``
      describe the q(t) input used over ``[t, t+dt]``.
    * ``sample_target_time_s`` is the right-limit target-evaluation time
      ``t+dt``; ``sample_target_pose/twist/acceleration`` describe the input
      written before the post-integration refresh.
    * ``integration_application_time_s`` and ``sample_application_time_s``
      are the two corresponding mocap write times.
    * ``sample_time_s`` is the post-step ``data.time`` at which actual state
      and diagnostics were sampled.
    * All pose rows are world-frame deck-origin ``[x, y, z, qw, qx, qy, qz]``;
      position is meters and the quaternion is unitless.
    * All twist/acceleration rows are world spatial quantities at the deck
      body origin, ordered ``[linear_xyz, angular_xyz]`` in m/s and rad/s or
      m/s² and rad/s². Command angular fields use the analytic SO(3)
      exponential-coordinate mapping; actual fields use MuJoCo's global
      object API.
    * ``deck_tracking_pose_error`` is actual minus the right-limit command at
      the same sample time, ordered translation followed by world
      rotation-vector error, in meters and radians.
    * ``weld_constraint_residual_raw`` is MuJoCo ``efc_pos`` for this weld in
      MuJoCo's constraint-coordinate order, and
      ``weld_constraint_force_raw`` is MuJoCo ``efc_force`` in that same raw
      order. Neither field is a spatial wrench.
    * solver iteration and warning fields are post-step diagnostics. Warning
      counts are per-step deltas, while ``warning_lastinfo`` is MuJoCo's
      latest per-warning info code.

    All per-sample arrays have matching first dimensions. Six-axis vectors
    use ``(tx, ty, tz, rx, ry, rz)`` order.
    """

    integration_target_time_s: np.ndarray
    sample_target_time_s: np.ndarray
    integration_application_time_s: np.ndarray
    sample_application_time_s: np.ndarray
    sample_time_s: np.ndarray
    command_pose: np.ndarray
    actual_pose: np.ndarray
    command_twist: np.ndarray
    actual_twist: np.ndarray
    command_acceleration: np.ndarray
    actual_acceleration: np.ndarray
    sample_target_pose: np.ndarray
    sample_target_twist: np.ndarray
    sample_target_acceleration: np.ndarray
    deck_tracking_pose_error: np.ndarray
    weld_constraint_residual_raw: np.ndarray
    weld_constraint_force_raw: np.ndarray
    solver_iterations: np.ndarray
    solver_niter: np.ndarray
    warning_number_delta: np.ndarray
    warning_lastinfo: np.ndarray

    def __post_init__(self) -> None:
        sample_count = np.asarray(self.sample_time_s).size
        vector_fields = (
            ("command_pose", 7),
            ("actual_pose", 7),
            ("command_twist", 6),
            ("actual_twist", 6),
            ("command_acceleration", 6),
            ("actual_acceleration", 6),
            ("sample_target_pose", 7),
            ("sample_target_twist", 6),
            ("sample_target_acceleration", 6),
            ("deck_tracking_pose_error", 6),
            ("weld_constraint_residual_raw", 6),
            ("weld_constraint_force_raw", 6),
        )
        for name in (
            "integration_target_time_s",
            "sample_target_time_s",
            "integration_application_time_s",
            "sample_application_time_s",
            "sample_time_s",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (sample_count,) or not np.all(np.isfinite(value)):
                raise DeckDriverError(f"trace field {name} must have {sample_count} finite samples")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name, width in vector_fields:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (sample_count, width) or not np.all(np.isfinite(value)):
                raise DeckDriverError(f"trace field {name} must have shape ({sample_count}, {width}) and finite values")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("solver_iterations",):
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if value.shape != (sample_count,):
                raise DeckDriverError(f"trace field {name} must have {sample_count} samples")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("solver_niter", "warning_number_delta", "warning_lastinfo"):
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if value.ndim != 2 or value.shape[0] != sample_count:
                raise DeckDriverError(f"trace field {name} must be a two-dimensional sample array")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def axis_order(self) -> tuple[str, ...]:
        return AXES

    @property
    def command_timestamps_s(self) -> np.ndarray:
        """Compatibility view of the left-limit integration target times."""

        return self.integration_target_time_s

    @property
    def application_timestamps_s(self) -> np.ndarray:
        """Compatibility view of the left-limit mocap application times."""

        return self.integration_application_time_s

    @property
    def sample_timestamps_s(self) -> np.ndarray:
        """Compatibility view of post-integration sample times."""

        return self.sample_time_s

    @property
    def field_contract(self) -> Mapping[str, str]:
        """Return the machine-readable frame, origin, order, time, and unit contract."""

        return TRACE_FIELD_CONTRACT

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": TRACE_SCHEMA_ID,
            "schema_version": TRACE_SCHEMA_VERSION,
            "axis_order": list(AXES),
            "field_contract": dict(TRACE_FIELD_CONTRACT),
            "integration_target_time_s": self.integration_target_time_s.tolist(),
            "sample_target_time_s": self.sample_target_time_s.tolist(),
            "integration_application_time_s": self.integration_application_time_s.tolist(),
            "sample_application_time_s": self.sample_application_time_s.tolist(),
            "sample_time_s": self.sample_time_s.tolist(),
            "command_pose": self.command_pose.tolist(),
            "actual_pose": self.actual_pose.tolist(),
            "command_twist": self.command_twist.tolist(),
            "actual_twist": self.actual_twist.tolist(),
            "command_acceleration": self.command_acceleration.tolist(),
            "actual_acceleration": self.actual_acceleration.tolist(),
            "sample_target_pose": self.sample_target_pose.tolist(),
            "sample_target_twist": self.sample_target_twist.tolist(),
            "sample_target_acceleration": self.sample_target_acceleration.tolist(),
            "deck_tracking_pose_error": self.deck_tracking_pose_error.tolist(),
            "weld_constraint_residual_raw": self.weld_constraint_residual_raw.tolist(),
            "weld_constraint_force_raw": self.weld_constraint_force_raw.tolist(),
            "solver_iterations": self.solver_iterations.tolist(),
            "solver_niter": self.solver_niter.tolist(),
            "warning_number_delta": self.warning_number_delta.tolist(),
            "warning_lastinfo": self.warning_lastinfo.tolist(),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class DeckDriver:
    """Runtime mocap driver installed on a :class:`MujocoEnv` physics seam."""

    def __init__(
        self,
        trajectory: Optional[Any] = None,
        config: Optional[DeckDriverConfig] = None,
        body_handles: Any = None,
        required_roles: Iterable[str] = (),
        refresh_stride: int = 1,
    ) -> None:
        self.config = config if config is not None else DeckDriverConfig()
        self.trajectory = trajectory
        if isinstance(refresh_stride, (bool, np.bool_)):
            raise DeckDriverError("refresh_stride must be a positive integer")
        try:
            numeric_refresh_stride = float(refresh_stride)
            normalized_refresh_stride = int(numeric_refresh_stride)
        except (TypeError, ValueError) as exc:
            raise DeckDriverError("refresh_stride must be a positive integer") from exc
        if (
            not np.isfinite(numeric_refresh_stride)
            or normalized_refresh_stride < 1
            or numeric_refresh_stride != normalized_refresh_stride
        ):
            raise DeckDriverError("refresh_stride must be a positive integer")
        self.refresh_stride = normalized_refresh_stride
        self.processor = ShakeBenchDeckXMLProcessor(
            config=self.config,
            body_handles=body_handles,
            required_roles=required_roles,
        )
        self._sim = None
        self._sim_token = None
        self._driver_mocap_id = None
        self._deck_body_id = None
        self._deck_freejoint_dofadr = None
        self._weld_id = None
        self._environment = None
        self._pending_command = None
        self._records = {
            "integration_target_time_s": [],
            "sample_target_time_s": [],
            "integration_application_time_s": [],
            "sample_application_time_s": [],
            "sample_time_s": [],
            "command_pose": [],
            "actual_pose": [],
            "command_twist": [],
            "actual_twist": [],
            "command_acceleration": [],
            "actual_acceleration": [],
            "sample_target_pose": [],
            "sample_target_twist": [],
            "sample_target_acceleration": [],
            "deck_tracking_pose_error": [],
            "weld_constraint_residual_raw": [],
            "weld_constraint_force_raw": [],
            "solver_iterations": [],
            "solver_niter": [],
            "warning_number_delta": [],
            "warning_lastinfo": [],
        }
        self._last_warning_number = None

    @property
    def role_handles(self) -> dict[str, str]:
        return self.processor.role_handles

    def install(self, env: Any) -> "DeckDriver":
        """Attach XML and pre/post physics hooks to an environment.

        The call is valid only before the environment has compiled a model.
        Install the driver from a new environment's construction path (with
        ``load_model_on_init=False``), then let the first reset compile and
        audit the processed XML before a physics hook can run.

        Raises:
            DeckDriverError: If the environment already owns a compiled
                simulator.  A late install is rejected instead of leaving
                hooks attached to an unaudited model that would only become
                correct after a later reset.
        """

        if self._environment is not None and self._environment is not env:
            raise DeckDriverError("driver is already installed on another environment")
        if self._environment is None and getattr(env, "sim", None) is not None:
            raise DeckDriverError(
                "DeckDriver.install() must run before simulation initialization; "
                "construct a new environment with the driver installed before its first reset"
            )
        if self._environment is None:
            env.set_xml_processor(self.processor)
            env.add_sim_initialization_hook(self.bind)
            env.request_post_integration_refresh(stride=self.refresh_stride)
            env.add_pre_physics_step_hook(self.pre_physics_step)
            env.add_post_integration_refresh_hook(self.post_integration_refresh)
            env.add_post_physics_step_hook(self.post_physics_step)
            self._environment = env
        return self

    def bind(self, sim: Any) -> None:
        raw_model, _ = _raw_model_and_data(sim)
        # Validate the complete compiled contract before mutating any bind
        # state.  This keeps a failed bind fail-closed: no partially bound
        # driver can write mocap data or publish a misleading trace.
        audit_compiled_deck_model(sim, self.config, self.role_handles)
        driver_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, self.config.driver_body_name)
        deck_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_BODY, self.config.deck_body_name)
        mocap_id = int(raw_model.body_mocapid[driver_id])
        if mocap_id < 0:
            raise DeckDriverError("deck driver body is not mocap in the compiled model")
        model_dt = float(raw_model.opt.timestep)
        if self.config.physics_timestep_s is not None and not np.isclose(
            model_dt, self.config.physics_timestep_s, rtol=0.0, atol=1e-14
        ):
            raise DeckDriverError("compiled model timestep differs from DeckDriverConfig.physics_timestep_s")
        if self.config.eq_solref[0] < 2.0 * model_dt:
            raise DeckDriverError("eq_solref time constant must be at least 2 * compiled model timestep")
        if self._sim_token is not None and self._sim_token != id(sim):
            self.reset_trace()
        self._sim = sim
        self._sim_token = id(sim)
        self._driver_mocap_id = mocap_id
        self._deck_body_id = deck_id
        freejoint_ids = [
            joint_id
            for joint_id in range(int(raw_model.njnt))
            if int(raw_model.jnt_bodyid[joint_id]) == deck_id
            and int(raw_model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(freejoint_ids) != 1:
            raise DeckDriverError("compiled deck must have exactly one freejoint")
        self._deck_freejoint_dofadr = int(raw_model.jnt_dofadr[freejoint_ids[0]])
        self._weld_id = _mujoco_id(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, self.config.weld_name)

    def reset_trace(self) -> None:
        for values in self._records.values():
            values.clear()
        self._pending_command = None
        self._last_warning_number = None

    def _ensure_bound(self) -> tuple[Any, Any, Any]:
        if self._environment is None and self._sim is None:
            raise DeckDriverError("driver is not installed or bound to a simulation")
        sim = self._environment.sim if self._environment is not None else self._sim
        if self._sim_token != id(sim):
            if self._sim_token is not None:
                self.reset_trace()
            self.bind(sim)
        elif self._records["sample_time_s"] and float(sim.data.time) < self._records["sample_time_s"][-1]:
            # The same MjSim can be reset when hard_reset=False.
            self.reset_trace()
        raw_model, raw_data = _raw_model_and_data(sim)
        return sim, raw_model, raw_data

    @staticmethod
    def _coerce_six(name: str, value: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
        if value is None:
            return np.zeros(6, dtype=float) if default is None else np.array(default, copy=True)
        array = np.asarray(value, dtype=float)
        if array.shape == (1, 6):
            array = array[0]
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise DeckDriverError(f"trajectory {name} must evaluate to six finite values")
        return np.array(array, copy=True)

    def _evaluate_command(self, time_s: float) -> DeckCommand:
        if self.trajectory is None:
            return DeckCommand(np.zeros(6), np.zeros(6), np.zeros(6))
        trajectory = self.trajectory
        if hasattr(trajectory, "evaluate"):
            result = trajectory.evaluate(time_s)
        elif callable(trajectory):
            result = trajectory(time_s)
        else:
            result = trajectory
        if isinstance(result, DeckCommand):
            return result
        if isinstance(result, Mapping):
            pose = result.get("pose", result.get("q", result.get("position")))
            twist = result.get("twist", result.get("qdot", result.get("velocity")))
            acceleration = result.get("acceleration", result.get("qdd"))
            return DeckCommand(
                self._coerce_six("pose", pose),
                self._coerce_six("twist", twist),
                self._coerce_six("acceleration", acceleration),
            )
        if hasattr(result, "q"):
            return DeckCommand(
                self._coerce_six("pose", result.q),
                self._coerce_six("twist", getattr(result, "qdot", None)),
                self._coerce_six("acceleration", getattr(result, "qdd", None)),
            )
        if isinstance(result, (tuple, list)) and len(result) == 3:
            return DeckCommand(
                self._coerce_six("pose", result[0]),
                self._coerce_six("twist", result[1]),
                self._coerce_six("acceleration", result[2]),
            )
        return DeckCommand(self._coerce_six("pose", result), np.zeros(6), np.zeros(6))

    def _command_world_pose(self, command: DeckCommand) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return command_to_world_state(command, self.config)

    def pre_physics_step(self, physics_time_s: float, policy_step: bool = False) -> None:
        """Write command immediately before MuJoCo's next position phase."""

        time_s = _finite_float("physics_time_s", physics_time_s, minimum=0.0)
        sim, _, raw_data = self._ensure_bound()
        command = self._evaluate_command(time_s)
        position, quaternion, twist, acceleration = self._command_world_pose(command)
        raw_data.mocap_pos[self._driver_mocap_id] = position
        raw_data.mocap_quat[self._driver_mocap_id] = quaternion
        application_time_s = float(raw_data.time)
        self._pending_command = {
            "integration_target_time": time_s,
            "integration_application_time": application_time_s,
            "pose": np.concatenate((position, quaternion)),
            "twist": twist,
            "acceleration": acceleration,
        }

    def post_integration_refresh(self, sample_time_s: float, policy_step: bool = False) -> None:
        """Write and record the right-limit target before ``sim.forward()``.

        Args:
            sample_time_s: Integrated simulator time ``t + dt``.
            policy_step: Whether the enclosing loop is a policy action step.
        """

        if self._pending_command is None:
            raise DeckDriverError("post-integration refresh called without a preceding integration command")
        time_s = _finite_float("sample_target_time_s", sample_time_s, minimum=0.0)
        _, _, raw_data = self._ensure_bound()
        command = self._evaluate_command(time_s)
        position, quaternion, twist, acceleration = self._command_world_pose(command)
        raw_data.mocap_pos[self._driver_mocap_id] = position
        raw_data.mocap_quat[self._driver_mocap_id] = quaternion
        application_time_s = float(raw_data.time)
        if not np.isclose(application_time_s, time_s, rtol=0.0, atol=1e-14):
            raise DeckDriverError("sample target was not written at the integrated simulator time")
        self._pending_command.update(
            {
                "sample_target_time": time_s,
                "sample_application_time": application_time_s,
                "sample_pose": np.concatenate((position, quaternion)),
                "sample_twist": twist,
                "sample_acceleration": acceleration,
                "right_limit_refresh_performed": True,
            }
        )

    def _actual_state(self, raw_model: Any, raw_data: Any):
        body_id = self._deck_body_id
        position = np.array(raw_data.xpos[body_id], dtype=float, copy=True)
        quaternion = _normalise_quat_wxyz(raw_data.xquat[body_id], "actual deck quaternion")
        velocity_rotlin = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            raw_model,
            raw_data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity_rotlin,
            0,
        )
        # MuJoCo documents both object APIs as rot:lin and uses ``flg_local=0``
        # for the global frame.  The acceleration API consumes the complete
        # post-constraint RNE buffers; using it avoids relabelling the free
        # joint's body-local tangent qacc as a world angular acceleration.
        mujoco.mj_rnePostConstraint(raw_model, raw_data)
        acceleration_rotlin = np.zeros(6, dtype=float)
        mujoco.mj_objectAcceleration(
            raw_model,
            raw_data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            acceleration_rotlin,
            0,
        )
        acceleration = np.concatenate((acceleration_rotlin[3:], acceleration_rotlin[:3]))
        twist = np.concatenate((velocity_rotlin[3:], velocity_rotlin[:3]))
        return position, quaternion, twist, acceleration

    def _weld_constraint_values(self, raw_data: Any, field_name: str) -> np.ndarray:
        if self._weld_id is None or not hasattr(raw_data, "efc_type"):
            return np.zeros(6, dtype=float)
        rows = np.flatnonzero(
            (np.asarray(raw_data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (np.asarray(raw_data.efc_id) == int(self._weld_id))
        )
        if rows.size != 6:
            raise DeckDriverError(
                f"compiled weld must expose six equality rows in efc_{field_name}; found {rows.size}"
            )
        values = np.asarray(getattr(raw_data, f"efc_{field_name}"), dtype=float)[rows]
        if not np.all(np.isfinite(values)):
            raise DeckDriverError(f"efc_{field_name} contains non-finite weld diagnostics")
        return np.array(values, dtype=float, copy=True)

    @staticmethod
    def _warning_arrays(raw_data: Any) -> tuple[np.ndarray, np.ndarray]:
        warning = getattr(raw_data, "warning", None)
        if warning is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return (
            np.array(warning.number, dtype=np.int64, copy=True),
            np.array(warning.lastinfo, dtype=np.int64, copy=True),
        )

    def post_physics_step(self, sample_time_s: float, policy_step: bool = False) -> None:
        """Sample actual state and solver diagnostics after one MuJoCo step."""

        if self._pending_command is None:
            raise DeckDriverError("post_physics_step called without a preceding pre_physics_step")
        time_s = _finite_float("sample_time_s", sample_time_s, minimum=0.0)
        if "sample_target_time" not in self._pending_command:
            raise DeckDriverError(
                "post_physics_step requires the right-limit sample target to be written before refresh"
            )
        if not np.isclose(self._pending_command["sample_target_time"], time_s, rtol=0.0, atol=1e-14):
            raise DeckDriverError("sample target time and post-step sample time differ")
        _, raw_model, raw_data = self._ensure_bound()
        actual_position, actual_quaternion, actual_twist, actual_acceleration = self._actual_state(raw_model, raw_data)
        actual_pose = np.concatenate((actual_position, actual_quaternion))
        tracking_command_pose = self._pending_command["sample_pose"]
        tracking_pose_error = np.concatenate(
            (
                actual_position - tracking_command_pose[:3],
                quat_to_rotation_vector(
                    _quat_multiply_wxyz(actual_quaternion, _quat_inverse_wxyz(tracking_command_pose[3:]))
                ),
            )
        )
        warning_number, warning_lastinfo = self._warning_arrays(raw_data)
        if self._last_warning_number is None or self._last_warning_number.shape != warning_number.shape:
            warning_delta = np.array(warning_number, copy=True)
        else:
            warning_delta = warning_number - self._last_warning_number
            warning_delta = np.maximum(warning_delta, 0)
        self._last_warning_number = np.array(warning_number, copy=True)
        solver_niter = np.array(getattr(raw_data, "solver_niter", np.zeros(0)), dtype=np.int64, copy=True)
        solver_iterations = int(np.max(solver_niter)) if solver_niter.size else 0
        self._records["integration_target_time_s"].append(self._pending_command["integration_target_time"])
        self._records["integration_application_time_s"].append(self._pending_command["integration_application_time"])
        self._records["command_pose"].append(self._pending_command["pose"].copy())
        self._records["command_twist"].append(self._pending_command["twist"].copy())
        self._records["command_acceleration"].append(self._pending_command["acceleration"].copy())
        self._records["sample_target_time_s"].append(self._pending_command["sample_target_time"])
        self._records["sample_application_time_s"].append(self._pending_command["sample_application_time"])
        self._records["sample_time_s"].append(time_s)
        self._records["actual_pose"].append(actual_pose)
        self._records["actual_twist"].append(actual_twist)
        self._records["actual_acceleration"].append(actual_acceleration)
        self._records["sample_target_pose"].append(self._pending_command["sample_pose"])
        self._records["sample_target_twist"].append(self._pending_command["sample_twist"])
        self._records["sample_target_acceleration"].append(self._pending_command["sample_acceleration"])
        self._records["deck_tracking_pose_error"].append(tracking_pose_error)
        self._records["weld_constraint_residual_raw"].append(self._weld_constraint_values(raw_data, "pos"))
        self._records["weld_constraint_force_raw"].append(self._weld_constraint_values(raw_data, "force"))
        self._records["solver_iterations"].append(solver_iterations)
        self._records["solver_niter"].append(solver_niter)
        self._records["warning_number_delta"].append(warning_delta)
        self._records["warning_lastinfo"].append(warning_lastinfo)
        self._pending_command = None

    @property
    def trace(self) -> DeckDriverTrace:
        def array(name: str, shape: tuple[int, ...], dtype=float):
            values = self._records[name]
            if not values:
                return np.empty(shape, dtype=dtype)
            if name in {"solver_niter", "warning_number_delta", "warning_lastinfo"}:
                widths = {len(value) for value in values}
                if len(widths) != 1:
                    raise DeckDriverError(f"trace field {name} has inconsistent widths")
            return np.asarray(values, dtype=dtype)

        return DeckDriverTrace(
            integration_target_time_s=array("integration_target_time_s", (0,), float),
            sample_target_time_s=array("sample_target_time_s", (0,), float),
            integration_application_time_s=array("integration_application_time_s", (0,), float),
            sample_application_time_s=array("sample_application_time_s", (0,), float),
            sample_time_s=array("sample_time_s", (0,), float),
            command_pose=array("command_pose", (0, 7), float),
            actual_pose=array("actual_pose", (0, 7), float),
            command_twist=array("command_twist", (0, 6), float),
            actual_twist=array("actual_twist", (0, 6), float),
            command_acceleration=array("command_acceleration", (0, 6), float),
            actual_acceleration=array("actual_acceleration", (0, 6), float),
            sample_target_pose=array("sample_target_pose", (0, 7), float),
            sample_target_twist=array("sample_target_twist", (0, 6), float),
            sample_target_acceleration=array("sample_target_acceleration", (0, 6), float),
            deck_tracking_pose_error=array("deck_tracking_pose_error", (0, 6), float),
            weld_constraint_residual_raw=array("weld_constraint_residual_raw", (0, 6), float),
            weld_constraint_force_raw=array("weld_constraint_force_raw", (0, 6), float),
            solver_iterations=array("solver_iterations", (0,), np.int64),
            solver_niter=array("solver_niter", (0, 0), np.int64),
            warning_number_delta=array("warning_number_delta", (0, 0), np.int64),
            warning_lastinfo=array("warning_lastinfo", (0, 0), np.int64),
        )
