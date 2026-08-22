"""Six-DOF worktable isolator used by the ShakeBench MuJoCo spike.

The robot stays hard-mounted to ``deck``.  The worktable remains a child of
that deck, but six compliant joints restore the support dynamics that the
single-coordinate model omits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET


TABLE_MASS_KG = 32.0
TABLE_INERTIA_KG_M2 = (1.713, 1.713, 3.413)
CUBE_PAYLOAD_KG = 0.07011
GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class IsolatorParameters:
    """Physical parameters for an isotropic-frequency six-DOF support."""

    natural_frequency_hz: float
    damping_ratio: float
    table_mass_kg: float = TABLE_MASS_KG
    table_inertia_kg_m2: tuple[float, float, float] = TABLE_INERTIA_KG_M2
    payload_mass_kg: float = CUBE_PAYLOAD_KG
    gravity_m_s2: float = GRAVITY_M_S2

    def __post_init__(self) -> None:
        if self.natural_frequency_hz <= 0.0:
            raise ValueError("natural_frequency_hz must be positive")
        if self.damping_ratio <= 0.0:
            raise ValueError("damping_ratio must be positive")
        if self.table_mass_kg <= 0.0:
            raise ValueError("table_mass_kg must be positive")
        if len(self.table_inertia_kg_m2) != 3 or any(
            value <= 0.0 for value in self.table_inertia_kg_m2
        ):
            raise ValueError("table_inertia_kg_m2 must contain three positive values")
        if self.payload_mass_kg < 0.0:
            raise ValueError("payload_mass_kg must be non-negative")

    @property
    def omega_n_rad_s(self) -> float:
        return 2.0 * math.pi * self.natural_frequency_hz

    @property
    def translational_stiffness_n_m(self) -> float:
        return self.table_mass_kg * self.omega_n_rad_s**2

    @property
    def translational_damping_n_s_m(self) -> float:
        return (
            2.0
            * self.damping_ratio
            * self.table_mass_kg
            * self.omega_n_rad_s
        )

    @property
    def rotational_stiffness_n_m_rad(self) -> tuple[float, float, float]:
        return tuple(
            inertia * self.omega_n_rad_s**2
            for inertia in self.table_inertia_kg_m2
        )

    @property
    def rotational_damping_n_m_s_rad(self) -> tuple[float, float, float]:
        return tuple(
            2.0 * self.damping_ratio * inertia * self.omega_n_rad_s
            for inertia in self.table_inertia_kg_m2
        )

    @property
    def vertical_springref_m(self) -> float:
        return (
            (self.table_mass_kg + self.payload_mass_kg) * self.gravity_m_s2
            / self.translational_stiffness_n_m
        )

    def as_dict(self) -> dict:
        return {
            "natural_frequency_hz": self.natural_frequency_hz,
            "damping_ratio": self.damping_ratio,
            "omega_n_rad_s": self.omega_n_rad_s,
            "table_mass_kg": self.table_mass_kg,
            "table_inertia_kg_m2": list(self.table_inertia_kg_m2),
            "payload_mass_kg": self.payload_mass_kg,
            "translational_stiffness_n_m": self.translational_stiffness_n_m,
            "translational_damping_n_s_m": self.translational_damping_n_s_m,
            "rotational_stiffness_n_m_rad": list(
                self.rotational_stiffness_n_m_rad
            ),
            "rotational_damping_n_m_s_rad": list(
                self.rotational_damping_n_m_s_rad
            ),
            "vertical_springref_m": self.vertical_springref_m,
        }


def absolute_transmissibility(
    natural_frequency_hz: float,
    damping_ratio: float,
    excitation_frequency_hz: float,
) -> float:
    """Analytic absolute displacement transmissibility for base excitation."""

    ratio = excitation_frequency_hz / natural_frequency_hz
    numerator = 1.0 + (2.0 * damping_ratio * ratio) ** 2
    denominator = (1.0 - ratio**2) ** 2 + (
        2.0 * damping_ratio * ratio
    ) ** 2
    return math.sqrt(numerator / denominator)


def add_isolator_joints(
    table_body: ET.Element,
    parameters: IsolatorParameters,
) -> None:
    """Insert compliant table joints without changing any contact geometry."""

    if table_body.get("name") != "table":
        raise ValueError("isolator joints must be added to the table body")
    if any(child.tag in {"joint", "freejoint"} for child in table_body):
        raise ValueError("table body already has joints")

    linear_k = parameters.translational_stiffness_n_m
    linear_c = parameters.translational_damping_n_s_m
    rotational_k = parameters.rotational_stiffness_n_m_rad
    rotational_c = parameters.rotational_damping_n_m_s_rad
    joint_specs = (
        ("table_iso_tx", "slide", "1 0 0", linear_k, linear_c, 0.0),
        ("table_iso_ty", "slide", "0 1 0", linear_k, linear_c, 0.0),
        (
            "table_iso_tz",
            "slide",
            "0 0 1",
            linear_k,
            linear_c,
            parameters.vertical_springref_m,
        ),
        ("table_iso_rx", "hinge", "1 0 0", rotational_k[0], rotational_c[0], 0.0),
        ("table_iso_ry", "hinge", "0 1 0", rotational_k[1], rotational_c[1], 0.0),
        ("table_iso_rz", "hinge", "0 0 1", rotational_k[2], rotational_c[2], 0.0),
    )
    for index, (name, kind, axis, stiffness, damping, springref) in enumerate(
        joint_specs
    ):
        joint = ET.Element(
            "joint",
            name=name,
            type=kind,
            axis=axis,
            stiffness=f"{stiffness:.17g}",
            damping=f"{damping:.17g}",
            springref=f"{springref:.17g}",
        )
        table_body.insert(index, joint)
