"""Newton/MJWarp runtime diagnostics used by benchmark acceptance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSnapshot:
    newton_shape_count: int
    mjwarp_collision_geometry_count: int
    mjwarp_candidate_pair_count: int
    active_contact_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PenetrationSample:
    """Worst monitored signed contact distance for one physics frame."""

    depth_m: float = 0.0
    pair: str = "none"
    shape0: str = ""
    shape1: str = ""

    @property
    def depth_mm(self) -> float:
        return 1000.0 * self.depth_m


def collision_shape_geometry(label_fragment: str) -> list[dict[str, object]]:
    """Return scaled local mesh bounds for matching Newton collision shapes."""

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    if model is None:
        raise RuntimeError("Newton/MJWarp must be initialized before inspecting shapes")
    scales = model.shape_scale.numpy()
    transforms = model.shape_transform.numpy()
    solver = NewtonManager._solver
    shape_to_geom = None
    geom_friction = None
    if solver is not None:
        mapping = getattr(solver, "newton_shape_to_mjc_geom", None)
        friction = getattr(solver.mjw_model, "geom_friction", None)
        if mapping is not None and friction is not None:
            shape_to_geom = np.asarray(mapping.numpy()).reshape(-1)
            geom_friction = np.asarray(friction.numpy())
            if geom_friction.ndim > 2:
                geom_friction = geom_friction.reshape(-1, geom_friction.shape[-1])
    results: list[dict[str, object]] = []
    for index, label in enumerate(model.shape_label):
        label = str(label or "")
        if label_fragment.lower() not in label.lower():
            continue
        scale = np.asarray(scales[index], dtype=np.float64)
        source = model.shape_source[index]
        vertices = getattr(source, "_vertices", None)
        extent = None
        center = None
        if vertices is not None:
            vertices_np = np.asarray(
                vertices.numpy() if hasattr(vertices, "numpy") else vertices,
                dtype=np.float64,
            )
            extent = ((vertices_np.max(axis=0) - vertices_np.min(axis=0)) * scale).tolist()
            center = (0.5 * (vertices_np.max(axis=0) + vertices_np.min(axis=0)) * scale).tolist()
        result = {
                "shape_id": index,
                "label": label,
                "scale": scale.tolist(),
                "mesh_extent_m": extent,
                "mesh_center_m": center,
                "shape_transform": np.asarray(transforms[index]).tolist(),
            }
        if shape_to_geom is not None and geom_friction is not None and index < len(shape_to_geom):
            geom_id = int(shape_to_geom[index])
            result["mjwarp_geom_id"] = geom_id
            if 0 <= geom_id < len(geom_friction):
                result["mjwarp_geom_friction"] = np.asarray(geom_friction[geom_id]).tolist()
        results.append(result)
    return results


def configure_mujoco_contact_solref(
    solref: tuple[float, float],
    friction_mu: float | None = None,
) -> dict[str, object]:
    """Apply and report a uniform MuJoCo contact response after conversion.

    Newton currently does not preserve authored NativeCCD margins when
    ``use_mujoco_contacts=True``.  A direct, disclosed ``geom_solref`` update
    is therefore the only backend-compatible stiffness control that leaves
    collision geometry and candidate-pair counts unchanged.
    """

    from isaaclab_newton.physics import NewtonManager

    solver = NewtonManager._solver
    if solver is None:
        raise RuntimeError("Newton/MJWarp must be initialized before configuring contact response")
    geom_solref = solver.mjw_model.geom_solref.numpy()
    previous = np.unique(geom_solref.reshape(-1, 2), axis=0).tolist()
    geom_friction = solver.mjw_model.geom_friction.numpy()
    geom_condim = solver.mjw_model.geom_condim.numpy()
    friction_rows = geom_friction.reshape(-1, geom_friction.shape[-1])
    previous_friction = np.unique(friction_rows, axis=0).tolist()
    geom_solref[..., 0] = solref[0]
    geom_solref[..., 1] = solref[1]
    solver.mjw_model.geom_solref.assign(geom_solref)
    if friction_mu is not None:
        if friction_mu <= 0.0:
            raise ValueError("friction_mu must be positive")
        # Instanceable robot collision prims reject a late USD material bind.
        # Set the converted MJWarp sliding coefficient directly, exactly as
        # the backend-compatible solref override above.  Torsional/rolling
        # coefficients remain those authored by the source assets.
        geom_friction[..., 0] = friction_mu
        solver.mjw_model.geom_friction.assign(geom_friction)
    applied_friction = solver.mjw_model.geom_friction.numpy()
    applied_rows = applied_friction.reshape(-1, applied_friction.shape[-1])
    return {
        "requested_margin_m": None,
        "nativeccd_margin_honored": False,
        "solref_timeconst_s": solref[0],
        "solref_dampratio": solref[1],
        "previous_unique_solref": previous,
        "previous_unique_geom_friction": previous_friction,
        "applied_unique_geom_friction": np.unique(applied_rows, axis=0).tolist(),
        "requested_sliding_friction": friction_mu,
        "unique_geom_condim": np.unique(geom_condim).tolist(),
    }


def classify_penetration_pair(shape0: str, shape1: str) -> str | None:
    """Map Newton USD shape labels to the V0 semantic contact pairs."""

    lhs, rhs = shape0.lower(), shape1.lower()

    def paired(a: str, b: str) -> bool:
        return (a in lhs and b in rhs) or (a in rhs and b in lhs)

    if paired("workpiece", "worktabletop"):
        return "workpiece<->worktable"
    if paired("workpiece", "panda_leftfinger"):
        return "workpiece<->left_fingertip"
    if paired("workpiece", "panda_rightfinger"):
        return "workpiece<->right_fingertip"
    if paired("workpiece", "targetbin"):
        return "workpiece<->target_bin"
    if ("worktableleg" in lhs and "vibrationfloor" in rhs) or (
        "worktableleg" in rhs and "vibrationfloor" in lhs
    ):
        return "worktable_leg<->platen"
    if paired("/robot/", "vibrationfloor"):
        return "robot_link<->platen"
    if paired("/robot/", "worktabletop"):
        return "robot_link<->worktable"
    if paired("/robot/", "controlknob"):
        return "finger<->knob"
    if paired("/robot/", "controllever"):
        return "finger<->lever"
    if paired("/robot/", "controlbutton"):
        return "finger<->button"
    if paired("/robot/", "controlpanel"):
        return "finger<->control_panel"
    if paired("control", "worktabletop"):
        return "control<->worktable"
    return None


def penetration_probe() -> PenetrationSample:
    """Read the worst monitored penetration from live MuJoCo-Warp contacts.

    MuJoCo contact ``dist`` is signed: negative values are penetration.  The
    active contact order is shared with Newton's converted shape-id buffers,
    letting this probe retain the exact USD labels for regression reports.
    """

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    solver = NewtonManager._solver
    contacts = NewtonManager._contacts
    if model is None or solver is None or contacts is None:
        raise RuntimeError("Newton/MJWarp must be initialized before probing penetration")
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count <= 0:
        return PenetrationSample()
    distances = solver.mjw_data.contact.dist.numpy()[:count]
    shape0_ids = contacts.rigid_contact_shape0.numpy()[:count]
    shape1_ids = contacts.rigid_contact_shape1.numpy()[:count]
    labels = list(model.shape_label)
    worst = PenetrationSample()
    for distance, shape0_id, shape1_id in zip(distances, shape0_ids, shape1_ids):
        if int(shape0_id) < 0 or int(shape1_id) < 0:
            continue
        shape0 = str(labels[int(shape0_id)] or "")
        shape1 = str(labels[int(shape1_id)] or "")
        pair = classify_penetration_pair(shape0, shape1)
        depth_m = max(0.0, -float(distance))
        if pair is not None and depth_m > worst.depth_m:
            worst = PenetrationSample(depth_m, pair, shape0, shape1)
    return worst


def contact_snapshot() -> ContactSnapshot:
    """Read stable model/contact counters from the active Newton manager."""

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    solver = NewtonManager._solver
    contacts = NewtonManager._contacts
    if model is None or solver is None or contacts is None:
        raise RuntimeError("Newton/MJWarp must be initialized before collecting contact diagnostics")
    mjw_model = solver.mjw_model
    candidate_pairs = getattr(mjw_model, "nxn_geom_pair_filtered", ())
    active = contacts.rigid_contact_count.numpy()
    return ContactSnapshot(
        newton_shape_count=int(model.shape_count),
        mjwarp_collision_geometry_count=int(mjw_model.ngeom),
        mjwarp_candidate_pair_count=int(len(candidate_pairs)),
        active_contact_count=int(active.sum()),
    )


def print_contact_snapshot(label: str) -> ContactSnapshot:
    snapshot = contact_snapshot()
    print(
        f"[CONTACT:{label}] newton_shapes={snapshot.newton_shape_count} "
        f"mjwarp_geometries={snapshot.mjwarp_collision_geometry_count} "
        f"candidate_pairs={snapshot.mjwarp_candidate_pair_count} "
        f"active_contacts={snapshot.active_contact_count}"
    )
    return snapshot
