"""Shared high-precision ShakeBench quaternion helpers using MuJoCo's wxyz order."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import robosuite.utils.transform_utils as T


def normalize_wxyz(value: Any, *, name: str = "quaternion", error_type: type[Exception] = ValueError) -> np.ndarray:
    try:
        quaternion = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{name} must contain four finite values") from exc
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise error_type(f"{name} must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise error_type(f"{name} must not be the zero quaternion")
    return quaternion / norm


def multiply_wxyz(first: Any, second: Any) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(first, dtype=float)
    w2, x2, y2, z2 = np.asarray(second, dtype=float)
    return np.array((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dtype=float)


def inverse_wxyz(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=float)
    return np.array((value[0], -value[1], -value[2], -value[3]), dtype=float) / float(np.dot(value, value))


def wxyz_to_matrix(value: Any, *, name: str = "quaternion", error_type: type[Exception] = ValueError) -> np.ndarray:
    w, x, y, z = normalize_wxyz(value, name=name, error_type=error_type)
    return np.array((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=float)


def rotation_vector_to_wxyz(value: Any, *, name: str = "rotation_vector", error_type: type[Exception] = ValueError) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{name} must contain three finite values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise error_type(f"{name} must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-14:
        return np.array((1.0, 0.0, 0.0, 0.0), dtype=float)
    axis = vector / angle
    return np.concatenate(([math.cos(angle / 2.0)], axis * math.sin(angle / 2.0)))


def wxyz_to_rotation_vector(value: Any, *, name: str = "quaternion", error_type: type[Exception] = ValueError) -> np.ndarray:
    quaternion = normalize_wxyz(value, name=name, error_type=error_type)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    sine_half = float(np.linalg.norm(quaternion[1:]))
    if sine_half <= 1.0e-14:
        return 2.0 * quaternion[1:]
    angle = 2.0 * math.atan2(sine_half, float(np.clip(quaternion[0], -1.0, 1.0)))
    return quaternion[1:] * (angle / sine_half)


def matrix_to_wxyz(value: Any, *, name: str = "rotation", error_type: type[Exception] = ValueError) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise error_type(f"{name} must be a finite 3x3 matrix")
    return normalize_wxyz(T.mat2quat(matrix)[[3, 0, 1, 2]], name=name, error_type=error_type)


def xyzw_to_matrix(value: Any, *, name: str = "quaternion", error_type: type[Exception] = ValueError) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float).reshape(-1)
    if quaternion.shape != (4,):
        raise error_type(f"{name} must contain four finite values")
    return wxyz_to_matrix(quaternion[[3, 0, 1, 2]], name=name, error_type=error_type)


def matrix_to_rotation_vector(value: Any, *, name: str = "rotation", error_type: type[Exception] = ValueError) -> np.ndarray:
    return wxyz_to_rotation_vector(matrix_to_wxyz(value, name=name, error_type=error_type), name=name, error_type=error_type)
