"""Immutable Phase 06 physics-profile loading and compilation helpers.

The benchmark has two deliberately different configuration paths:

* ``official`` is a package asset.  Its embedded SHA-256 is verified before
  any score-affecting value is used by an environment.
* ``probe`` / ``training`` is an explicit, non-scoreable compatibility
  profile.  It is useful for exploratory physics work and cannot silently be
  mistaken for the official benchmark configuration.

This module is intentionally independent of task success, controllers, and
policy observations.  It only owns environment physics and the provenance
needed to audit it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional
import xml.etree.ElementTree as ET

import numpy as np

from robosuite import models
from robosuite.utils.shakebench_isolator import AXES, IsolatorConfig, derive_isolator_parameters


PHYSICS_PROFILE_SCHEMA_ID = "shakebench.official.physics"
PHYSICS_PROFILE_SCHEMA_VERSION = 1
OFFICIAL_PHYSICS_PROFILE_FILENAME = "shakebench_official_physics.yaml"
SELECTION_PROTOCOL_FILENAME = "shakebench_selection_protocol.yaml"
SELECTION_PROTOCOL_V2_FILENAME = "shakebench_selection_protocol_v2.yaml"
SELECTION_PROTOCOL_V4_FILENAME = "shakebench_selection_protocol_v4.yaml"
SELECTION_PROTOCOL_V5_FILENAME = "shakebench_selection_protocol_v5.yaml"
PHASE06R_STATUS_FILENAME = "shakebench_phase_06r_status.json"
PHASE06R2_STATUS_FILENAME = "shakebench_phase_06r2_status.json"
PHASE06R3_STATUS_FILENAME = "shakebench_phase_06r3_v4_status.json"
PHASE06R4_STATUS_FILENAME = "shakebench_phase_06r4_v5_status.json"
PROBE_PHYSICS_PROFILE_ID = "shakebench.probe.physics.v1"
OFFICIAL_PHYSICS_PROFILE_ID = "shakebench.official.physics.v1"


class PhysicsProfileError(ValueError):
    """Base error raised for malformed or unsafe physics profiles."""


class PhysicsProfileIntegrityError(PhysicsProfileError):
    """Raised when a package profile or its embedded hash is tampered with."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_profile_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical hash representation of a profile.

    ``profile_sha256`` is the only excluded field because it points to this
    representation.  No provenance, candidate result, or update field is
    excluded.
    """

    if not isinstance(payload, Mapping):
        raise PhysicsProfileError("physics profile must be an object")
    content = copy.deepcopy(_json_ready(payload))
    content.pop("profile_sha256", None)
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def physics_profile_hash(payload: Mapping[str, Any]) -> str:
    """Compute the SHA-256 hash used by an embedded profile."""

    return hashlib.sha256(canonical_profile_payload(payload).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _load_yaml_text(text: str) -> Mapping[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(text)
    except ImportError:
        # The checked-in YAML is also valid JSON.  This fallback keeps wheel
        # consumers independent of PyYAML while retaining a human-readable
        # YAML source in the repository.
        value = json.loads(text)
    except Exception as exc:
        raise PhysicsProfileError(f"invalid physics profile YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PhysicsProfileError("physics profile root must be an object")
    return value


def _asset_path(filename: str) -> Path:
    path = Path(models.assets_root) / filename
    if not path.is_file():
        raise PhysicsProfileIntegrityError(f"package physics asset is missing: {path}")
    return path


def _number(name: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise PhysicsProfileError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PhysicsProfileError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise PhysicsProfileError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise PhysicsProfileError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise PhysicsProfileError(f"{name} must be non-negative")
    return result


def _vector(name: str, value: Any, length: int, *, positive: bool = False, nonnegative: bool = False) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise PhysicsProfileError(f"{name} must contain {length} finite values")
    try:
        values = tuple(_number(f"{name}[{index}]", item) for index, item in enumerate(value))
    except TypeError as exc:
        raise PhysicsProfileError(f"{name} must contain {length} finite values") from exc
    if len(values) != length:
        raise PhysicsProfileError(f"{name} must contain {length} finite values")
    if positive and any(item <= 0.0 for item in values):
        raise PhysicsProfileError(f"{name} must contain only positive values")
    if nonnegative and any(item < 0.0 for item in values):
        raise PhysicsProfileError(f"{name} must contain only non-negative values")
    return values


def _same_vector(actual: Any, expected: Any, *, atol: float = 1.0e-12) -> bool:
    try:
        return bool(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PhysicsProfile:
    """Validated immutable environment-owned physics configuration."""

    payload: Mapping[str, Any]
    source: str
    profile_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise PhysicsProfileError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze(_json_ready(self.payload)))
        if len(self.profile_sha256) != 64:
            raise PhysicsProfileIntegrityError("profile_sha256 must be a SHA-256 digest")

    @property
    def profile_id(self) -> str:
        return str(self.payload["profile_id"])

    @property
    def schema_id(self) -> str:
        return str(self.payload["schema_id"])

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def scoreable(self) -> bool:
        return bool(self.payload["scoreable"])

    @property
    def physics(self) -> Mapping[str, Any]:
        return self.payload["physics"]

    @property
    def timestep(self) -> Mapping[str, Any]:
        return self.physics["timestep"]

    @property
    def deck(self) -> Mapping[str, Any]:
        return self.physics["deck"]

    @property
    def isolator(self) -> Mapping[str, Any]:
        return self.physics["isolator"]

    @property
    def contact(self) -> Mapping[str, Any]:
        return self.physics["contact"]

    @property
    def scheduler(self) -> Mapping[str, Any]:
        return self.physics["scheduler"]

    @property
    def model_timestep_s(self) -> float:
        return float(self.timestep["physics_timestep_s"])

    @property
    def dt_s(self) -> float:
        """Short alias for the environment-owned model timestep."""

        return self.model_timestep_s

    @property
    def f_max_hz(self) -> float:
        return float(self.physics["excitation"]["f_max_hz"])

    @property
    def control_freq_hz(self) -> float:
        return float(self.scheduler["control_freq_hz"])

    @property
    def control_steps(self) -> int:
        return int(self.scheduler["control_steps"])

    @property
    def deck_eq_solref(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.deck["eq_solref"])

    @property
    def deck_eq_solimp(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.deck["eq_solimp"])

    @property
    def deck_mass_kg(self) -> float:
        return float(self.deck["mass_kg"])

    @property
    def deck_inertia_kg_m2(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.deck["inertia_kg_m2"])

    @property
    def isolator_fn_hz(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.isolator["fn_hz"])

    @property
    def isolator_zeta(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.isolator["zeta"])

    @property
    def isolator_k(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.isolator["k"])

    @property
    def isolator_c(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.isolator["c"])

    @property
    def isolator_springref(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.isolator["springref"])

    @property
    def fn_hz(self) -> tuple[float, ...]:
        return self.isolator_fn_hz

    @property
    def zeta(self) -> tuple[float, ...]:
        return self.isolator_zeta

    @property
    def k(self) -> tuple[float, ...]:
        return self.isolator_k

    @property
    def c(self) -> tuple[float, ...]:
        return self.isolator_c

    @property
    def profile_hash(self) -> str:
        return self.profile_sha256

    @property
    def contact_profile(self) -> Mapping[str, Any]:
        return self.contact

    @property
    def contact_condim(self) -> int:
        return int(self.contact["condim"])

    @property
    def contact_torsional_mu(self) -> float:
        return float(self.contact["torsional_mu"])

    @property
    def contact_rolling_mu(self) -> float:
        return float(self.contact["rolling_mu"])

    @property
    def contact_margin_m(self) -> float:
        return float(self.contact["margin_m"])

    @property
    def contact_gap_m(self) -> float:
        return float(self.contact["gap_m"])

    @property
    def contact_solref(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.contact["solref"])

    @property
    def contact_solimp(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.contact["solimp"])

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)

    def isolator_config(self) -> IsolatorConfig:
        return IsolatorConfig(
            fn_hz=self.isolator_fn_hz,
            zeta=self.isolator_zeta,
            mass_kg=float(self.isolator["mass_kg"]),
            inertia_kg_m2=tuple(float(value) for value in self.isolator["inertia_kg_m2"]),
            gravity_m_s2=float(self.isolator["gravity_m_s2"]),
            travel_limits_m=tuple(float(value) for value in self.isolator["travel_limits_m"]),
            angle_limits_rad=tuple(float(value) for value in self.isolator["angle_limits_rad"]),
        )

    def deck_driver_config(self):
        from robosuite.utils.shakebench_deck import DeckDriverConfig

        return DeckDriverConfig(
            driver_body_name=str(self.deck["driver_body_name"]),
            deck_body_name=str(self.deck["body_name"]),
            deck_freejoint_name=str(self.deck["freejoint_name"]),
            weld_name=str(self.deck["weld_name"]),
            driver_site_name=str(self.deck["driver_site_name"]),
            deck_site_name=str(self.deck["site_name"]),
            deck_mass_kg=self.deck_mass_kg,
            deck_inertia_kg_m2=self.deck_inertia_kg_m2,
            deck_pos_m=tuple(float(value) for value in self.deck["pos_m"]),
            deck_quat_wxyz=tuple(float(value) for value in self.deck["quat_wxyz"]),
            eq_solref=self.deck_eq_solref,
            eq_solimp=self.deck_eq_solimp,
            physics_timestep_s=self.model_timestep_s,
            site_size_m=float(self.deck["site_size_m"]),
        )

    def pair_attributes(self, sliding_mu: float) -> dict[str, str]:
        """Return explicit MuJoCo pair attributes for one contact interface."""

        sliding = _number("sliding_mu", sliding_mu, nonnegative=True)
        friction = (sliding, self.contact_torsional_mu, self.contact_rolling_mu, self.contact_rolling_mu, self.contact_rolling_mu)
        return {
            "friction": " ".join(format(value, ".17g") for value in friction),
            "condim": str(self.contact_condim),
            "margin": format(self.contact_margin_m, ".17g"),
            "gap": format(self.contact_gap_m, ".17g"),
            "solref": " ".join(format(value, ".17g") for value in self.contact_solref),
            "solimp": " ".join(format(value, ".17g") for value in self.contact_solimp),
        }

    def process_xml(self, xml_string: str) -> str:
        """Apply only the environment-owned MuJoCo option values."""

        if not isinstance(xml_string, str):
            raise PhysicsProfileError("xml_string must be a string")
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            raise PhysicsProfileError("invalid MJCF XML") from exc
        option = root.find("option")
        if option is None:
            option = ET.Element("option")
            root.insert(0, option)
        option.set("timestep", format(self.model_timestep_s, ".17g"))
        option.set("integrator", str(self.timestep["integrator"]))
        option.set("solver", str(self.timestep["solver"]))
        option.set("iterations", str(int(self.timestep["iterations"])))
        option.set("tolerance", format(float(self.timestep["tolerance"]), ".17g"))
        gravity = float(self.isolator["gravity_m_s2"])
        option.set("gravity", "0 0 " + format(-gravity, ".17g"))
        if "ls_iterations" in self.timestep:
            option.set("ls_iterations", str(int(self.timestep["ls_iterations"])))
        return ET.tostring(root, encoding="unicode")

    def audit(self) -> dict[str, Any]:
        parameters = derive_isolator_parameters(self.isolator_config())
        return {
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "source": self.source,
            "status": self.status,
            "scoreable": self.scoreable,
            "dt_condition": {
                "dt_s": self.model_timestep_s,
                "f_max_hz": self.f_max_hz,
                "limit_s": 1.0 / (20.0 * self.f_max_hz),
                "passed": self.model_timestep_s <= 1.0 / (20.0 * self.f_max_hz),
            },
            "eq_solref_condition": {
                "time_constant_s": self.deck_eq_solref[0],
                "minimum_s": 2.0 * self.model_timestep_s,
                "passed": self.deck_eq_solref[0] >= 2.0 * self.model_timestep_s,
            },
            "derived_isolator": {
                "k_matches": _same_vector(parameters.stiffness, self.isolator_k, atol=1.0e-10),
                "c_matches": _same_vector(parameters.damping, self.isolator_c, atol=1.0e-10),
                "springref_matches": _same_vector(parameters.springref, self.isolator_springref, atol=1.0e-12),
            },
            "scheduler": _thaw(self.scheduler),
            "contact": _thaw(self.contact),
        }

    def assert_valid(self) -> "PhysicsProfile":
        if self.profile_sha256 != physics_profile_hash(self.payload):
            raise PhysicsProfileIntegrityError("physics profile hash does not match its payload")
        if self.schema_id != PHYSICS_PROFILE_SCHEMA_ID:
            raise PhysicsProfileError("unexpected physics profile schema")
        if self.schema_version != PHYSICS_PROFILE_SCHEMA_VERSION:
            raise PhysicsProfileError("unsupported physics profile schema version")
        if not self.profile_id:
            raise PhysicsProfileError("physics profile id must not be empty")
        if not isinstance(self.scoreable, bool):
            raise PhysicsProfileError("scoreable must be boolean")
        if self.scoreable and self.status != "official_immutable":
            raise PhysicsProfileError("only official_immutable profiles may be scoreable")
        if self.scoreable and self.profile_id != OFFICIAL_PHYSICS_PROFILE_ID:
            raise PhysicsProfileError("only the package official profile may be scoreable")
        _number("physics_timestep_s", self.model_timestep_s, positive=True)
        if self.model_timestep_s > 1.0 / (20.0 * self.f_max_hz):
            raise PhysicsProfileError("physics timestep violates dt <= 1/(20*f_max)")
        if self.deck_eq_solref[0] < 2.0 * self.model_timestep_s:
            raise PhysicsProfileError("deck equality solref time constant is less than 2 * dt")
        if self.contact_solref[0] < 2.0 * self.model_timestep_s:
            raise PhysicsProfileError("contact solref time constant is less than 2 * dt")
        if self.contact_condim not in {1, 3, 4, 6}:
            raise PhysicsProfileError("contact condim is not a MuJoCo condim")
        _vector("deck_eq_solref", self.deck_eq_solref, 2, positive=True)
        _vector("deck_eq_solimp", self.deck_eq_solimp, 5, nonnegative=True)
        _vector("contact_solref", self.contact_solref, 2, positive=True)
        _vector("contact_solimp", self.contact_solimp, 5, nonnegative=True)
        for name, values in (
            ("isolator_fn_hz", self.isolator_fn_hz),
            ("isolator_zeta", self.isolator_zeta),
            ("isolator_k", self.isolator_k),
            ("isolator_c", self.isolator_c),
        ):
            _vector(name, values, 6, positive=True)
        if tuple(self.isolator["axis_order"]) != AXES:
            raise PhysicsProfileError("isolator axis order is not canonical")
        parameters = derive_isolator_parameters(self.isolator_config())
        if not _same_vector(parameters.stiffness, self.isolator_k, atol=1.0e-10):
            raise PhysicsProfileError("profile isolator k does not match its fn/mass derivation")
        if not _same_vector(parameters.damping, self.isolator_c, atol=1.0e-10):
            raise PhysicsProfileError("profile isolator c does not match its fn/zeta/mass derivation")
        if not _same_vector(parameters.springref, self.isolator_springref, atol=1.0e-12):
            raise PhysicsProfileError("profile isolator springref does not match nominal-table preload")
        if self.contact["sliding_mu"]["table_object"] != 0.30:
            raise PhysicsProfileError("table/object sliding friction must remain 0.30")
        if self.contact["sliding_mu"]["finger_object"] != 1.00:
            raise PhysicsProfileError("finger/object sliding friction must remain 1.00")
        if any(_number(f"contact.{name}", getattr(self, f"contact_{name}"), nonnegative=True) < 0.0 for name in ("torsional_mu", "rolling_mu", "margin_m", "gap_m")):
            raise PhysicsProfileError("contact friction/margin/gap must be non-negative")
        expected_control_steps = 1.0 / (self.control_freq_hz * self.model_timestep_s)
        if self.control_steps != round(expected_control_steps) or not np.isclose(
            expected_control_steps, self.control_steps, rtol=0.0, atol=1.0e-12
        ):
            raise PhysicsProfileError("scheduler control_steps is not an integer multiple of dt")
        return self

    def assert_matches_deck_config(self, config: Any) -> None:
        expected = self.deck_driver_config()
        fields = (
            "driver_body_name",
            "deck_body_name",
            "deck_freejoint_name",
            "weld_name",
            "driver_site_name",
            "deck_site_name",
            "deck_mass_kg",
            "deck_inertia_kg_m2",
            "deck_pos_m",
            "deck_quat_wxyz",
            "eq_solref",
            "eq_solimp",
            "physics_timestep_s",
            "site_size_m",
        )
        for field_name in fields:
            actual = getattr(config, field_name)
            wanted = getattr(expected, field_name)
            if isinstance(wanted, tuple):
                if not _same_vector(actual, wanted, atol=1.0e-12):
                    raise PhysicsProfileError(f"deck_config.{field_name} differs from the official physics profile")
            elif isinstance(wanted, float):
                if not np.isclose(float(actual), wanted, rtol=0.0, atol=1.0e-12):
                    raise PhysicsProfileError(f"deck_config.{field_name} differs from the official physics profile")
            elif actual != wanted:
                raise PhysicsProfileError(f"deck_config.{field_name} differs from the official physics profile")

    def assert_matches_isolator_config(self, config: IsolatorConfig) -> None:
        expected = self.isolator_config()
        for field_name in (
            "fn_hz",
            "zeta",
            "mass_kg",
            "inertia_kg_m2",
            "gravity_m_s2",
            "travel_limits_m",
            "angle_limits_rad",
        ):
            actual = getattr(config, field_name)
            wanted = getattr(expected, field_name)
            if isinstance(wanted, tuple):
                same = _same_vector(actual, wanted, atol=1.0e-12)
            else:
                same = bool(np.isclose(float(actual), float(wanted), rtol=0.0, atol=1.0e-12))
            if not same:
                raise PhysicsProfileError(f"isolator config {field_name} differs from the official physics profile")


def _validate_payload(payload: Mapping[str, Any], *, source: str, require_official: bool) -> PhysicsProfile:
    required = {"schema_id", "schema_version", "profile_id", "status", "scoreable", "physics", "profile_sha256"}
    missing = sorted(required - set(payload))
    if missing:
        raise PhysicsProfileError("physics profile is missing: " + ", ".join(missing))
    embedded = payload.get("profile_sha256")
    if not isinstance(embedded, str) or len(embedded) != 64:
        raise PhysicsProfileIntegrityError("profile_sha256 is missing or malformed")
    computed = physics_profile_hash(payload)
    if embedded.lower() != computed:
        raise PhysicsProfileIntegrityError(
            f"physics profile hash mismatch: embedded={embedded.lower()} computed={computed}"
        )
    profile = PhysicsProfile(payload=payload, source=source, profile_sha256=computed)
    profile.assert_valid()
    if require_official:
        if profile.status != "official_immutable" or profile.scoreable is not True:
            raise PhysicsProfileIntegrityError("official asset is not marked official_immutable and scoreable")
        if profile.profile_id != OFFICIAL_PHYSICS_PROFILE_ID:
            raise PhysicsProfileIntegrityError("official asset has an unexpected profile id")
    return profile


def load_official_physics_profile(path: Optional[str | Path] = None) -> PhysicsProfile:
    """Load and verify the package-owned official profile."""

    if path is not None:
        requested = Path(path).resolve()
        packaged = _asset_path(OFFICIAL_PHYSICS_PROFILE_FILENAME).resolve()
        if requested != packaged:
            raise PhysicsProfileIntegrityError(
                "scoreable official physics must be loaded from the packaged canonical asset"
            )
    # V1 through V4 are archived. A scoreable package profile may be enabled
    # only by the active V5 status; when that artifact is absent or blocked the
    # loader remains fail-closed.
    status_path = _asset_path(PHASE06R4_STATUS_FILENAME)
    try:
        selection_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicsProfileIntegrityError("Phase 06R4 V5 selection status is unreadable") from exc
    if selection_status.get("status") != "PASS":
        raise PhysicsProfileIntegrityError("official physics is blocked until Phase 06R4 V5 selection passes")
    profile_path = _asset_path(OFFICIAL_PHYSICS_PROFILE_FILENAME)
    try:
        payload = _load_yaml_text(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PhysicsProfileIntegrityError(f"cannot read physics profile: {profile_path}") from exc
    profile = _validate_payload(payload, source=str(profile_path), require_official=True)
    try:
        protocol_bytes = _asset_path(SELECTION_PROTOCOL_V5_FILENAME).read_bytes()
    except PhysicsProfileIntegrityError:
        raise
    expected_protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    if profile.payload.get("protocol_sha256") != expected_protocol_hash:
        raise PhysicsProfileIntegrityError("official profile does not authenticate the Phase 06R4 V5 selection protocol")
    return profile


def load_selection_protocol(path: Optional[str | Path] = None) -> tuple[Mapping[str, Any], str, str]:
    """Load the pre-registered selection protocol and return payload/path/hash."""

    protocol_path = _asset_path(SELECTION_PROTOCOL_FILENAME) if path is None else Path(path)
    try:
        raw = protocol_path.read_bytes()
        payload = _load_yaml_text(raw.decode("utf-8"))
    except OSError as exc:
        raise PhysicsProfileIntegrityError(f"cannot read selection protocol: {protocol_path}") from exc
    return payload, str(protocol_path), hashlib.sha256(raw).hexdigest()


def make_probe_physics_profile() -> PhysicsProfile:
    """Return the explicit non-scoreable Phase 03-compatible probe profile."""

    isolator = IsolatorConfig()
    parameters = derive_isolator_parameters(isolator)
    payload = {
        "schema_id": PHYSICS_PROFILE_SCHEMA_ID,
        "schema_version": PHYSICS_PROFILE_SCHEMA_VERSION,
        "profile_id": PROBE_PHYSICS_PROFILE_ID,
        "status": "probe_non_scoreable",
        "scoreable": False,
        "not_scoreable_reason": "exploratory/training profile; not the Phase 06 official freeze",
        "physics": {
            "excitation": {"f_max_hz": 8.87},
            "timestep": {
                "physics_timestep_s": 0.0002,
                "integrator": "Euler",
                "solver": "Newton",
                "iterations": 100,
                "tolerance": 1.0e-8,
            },
            "scheduler": {"control_freq_hz": 20.0, "control_steps": 250, "post_integration_refresh": True},
            "deck": {
                "driver_body_name": "deck_driver",
                "body_name": "deck",
                "freejoint_name": "deck_freejoint",
                "weld_name": "deck_weld",
                "driver_site_name": "deck_driver_site",
                "site_name": "deck_site",
                "mass_kg": 400.0,
                "inertia_kg_m2": [12.12, 14.083333333333334, 26.033333333333332],
                "pos_m": [0.0, 0.0, 0.0],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "eq_solref": [0.0004, 0.5],
                "eq_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
                "site_size_m": 0.01,
            },
            "isolator": {
                "axis_order": list(AXES),
                "fn_hz": list(parameters.fn_hz),
                "zeta": list(parameters.zeta),
                "mass_kg": parameters.mass_kg,
                "inertia_kg_m2": list(parameters.inertia_kg_m2),
                "gravity_m_s2": parameters.gravity_m_s2,
                "travel_limits_m": list(isolator.travel_limits_m),
                "angle_limits_rad": list(isolator.angle_limits_rad),
                "k": list(parameters.stiffness),
                "c": list(parameters.damping),
                "springref": list(parameters.springref),
            },
            "contact": {
                "sliding_mu": {"table_object": 0.30, "finger_object": 1.00},
                "condim": 3,
                "torsional_mu": 0.005,
                "rolling_mu": 0.0001,
                "margin_m": 0.0,
                "gap_m": 0.0,
                "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
                "pair_scope": "explicit_can_pairs",
            },
        },
    }
    payload["profile_sha256"] = physics_profile_hash(payload)
    return _validate_payload(payload, source="generated probe profile", require_official=False)


def resolve_physics_profile(profile: Any = None) -> PhysicsProfile:
    """Resolve an environment profile name or immutable profile object."""

    if profile is None or profile == "official":
        return load_official_physics_profile()
    if isinstance(profile, PhysicsProfile):
        resolved = profile.assert_valid()
        if resolved.scoreable:
            official = load_official_physics_profile()
            if resolved.profile_sha256 != official.profile_sha256:
                raise PhysicsProfileIntegrityError("scoreable in-memory profile is not the package official profile")
        return resolved
    if isinstance(profile, str) and profile in {"probe", "training", "probe_non_scoreable"}:
        return make_probe_physics_profile()
    if isinstance(profile, (str, Path)):
        profile_path = Path(profile)
        try:
            payload = _load_yaml_text(profile_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PhysicsProfileError(f"cannot read physics profile: {profile_path}") from exc
        resolved = _validate_payload(payload, source=str(profile_path), require_official=False)
        if resolved.scoreable:
            raise PhysicsProfileIntegrityError(
                "external physics profile paths must be explicitly non-scoreable"
            )
        return resolved
    if isinstance(profile, Mapping):
        payload = dict(profile)
        if payload.get("scoreable") is True:
            raise PhysicsProfileError("scoreable mappings must be loaded from the package official profile")
        if "profile_sha256" not in payload:
            payload["profile_sha256"] = physics_profile_hash(payload)
        return _validate_payload(payload, source="in-memory physics profile", require_official=False)
    raise PhysicsProfileError("physics_profile must be official, probe, a path, mapping, or PhysicsProfile")


def official_physics_profile_hash() -> str:
    """Return the verified official profile hash."""

    return load_official_physics_profile().profile_sha256


def load_physics_profile(profile: Any = None) -> PhysicsProfile:
    """Public alias for resolving official or explicit non-scoreable profiles."""

    return resolve_physics_profile(profile)


__all__ = [
    "OFFICIAL_PHYSICS_PROFILE_FILENAME",
    "OFFICIAL_PHYSICS_PROFILE_ID",
    "PHYSICS_PROFILE_SCHEMA_ID",
    "PHYSICS_PROFILE_SCHEMA_VERSION",
    "PROBE_PHYSICS_PROFILE_ID",
    "PhysicsProfile",
    "PhysicsProfileError",
    "PhysicsProfileIntegrityError",
    "canonical_profile_payload",
    "load_official_physics_profile",
    "load_physics_profile",
    "load_selection_protocol",
    "make_probe_physics_profile",
    "official_physics_profile_hash",
    "physics_profile_hash",
    "resolve_physics_profile",
]
