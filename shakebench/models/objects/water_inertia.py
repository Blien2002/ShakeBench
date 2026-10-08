"""Bottom-weighted rigid water-bottle inertia; no visual/collision or fluid model change."""

import numpy as np

WATER_INERTIA_PROFILE = "bottom_weighted_water_v1"
LOWER_MASS_FRACTION = 0.40
LOWER_COLUMN_RADIUS_M = 0.030
LOWER_COLUMN_HEIGHT_M = 0.100
LOWER_COLUMN_BOTTOM_CLEARANCE_M = 0.004


def quaternion_matrix(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
        raise ValueError("principal-axis quaternion must be finite and nonzero")
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def bottom_weighted_inertial(
    com, principal_inertia, principal_quaternion, mass, bottom_z, *, lower_fraction=LOWER_MASS_FRACTION
):
    """Move a fraction of the fixed mass into a lower cylindrical content proxy.

    Remaining mass retains the old normalized distribution. Both components
    contribute their own COM inertia plus parallel-axis terms about the new COM.
    The default0.28kg lower column occupies0.283litres at roughly990kg/m3.
    This is an approximate rigid content distribution, not a sloshing model.
    """
    com = np.asarray(com, dtype=float)
    principal = np.asarray(principal_inertia, dtype=float)
    values = np.array([mass, bottom_z, lower_fraction], dtype=float)
    if com.shape != (3,) or principal.shape != (3,) or not np.isfinite(com).all() or not np.isfinite(principal).all():
        raise ValueError("COM/principal inertia must have three finite values")
    if not np.isfinite(values).all() or mass <= 0 or not 0 < lower_fraction < 1 or np.min(principal) <= 0:
        raise ValueError("mass/inertia must be positive and lower fraction must be within(0,1)")
    if 2 * np.max(principal) >= np.sum(principal):
        raise ValueError("source principal inertia violates the rigid-body triangle inequality")
    column_com = com.copy()
    column_com[2] = bottom_z + LOWER_COLUMN_BOTTOM_CLEARANCE_M + LOWER_COLUMN_HEIGHT_M / 2
    if column_com[2] >= com[2]:
        raise ValueError("lower content proxy must lie below the original COM")
    new_com = (1 - lower_fraction) * com + lower_fraction * column_com
    rotation = quaternion_matrix(principal_quaternion)
    old_tensor = rotation @ np.diag(principal) @ rotation.T
    radius, height = LOWER_COLUMN_RADIUS_M, LOWER_COLUMN_HEIGHT_M
    lower_tensor = lower_fraction * mass * np.diag([radius**2 / 4 + height**2 / 12] * 2 + [radius**2 / 2])
    separation = column_com - com
    parallel = (
        mass
        * lower_fraction
        * (1 - lower_fraction)
        * (np.dot(separation, separation) * np.eye(3) - np.outer(separation, separation))
    )
    tensor = (1 - lower_fraction) * old_tensor + lower_tensor + parallel
    tensor = (tensor + tensor.T) / 2
    eigenvalues = np.linalg.eigvalsh(tensor)
    if eigenvalues[0] <= 0 or 2 * eigenvalues[-1] >= np.sum(eigenvalues):
        raise ValueError("computed inertia must be positive definite and physically realizable")
    full = [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]
    return {
        "pos": " ".join(format(v, ".17g") for v in new_com),
        "mass": format(mass, ".17g"),
        "fullinertia": " ".join(format(v, ".17g") for v in full),
    }
