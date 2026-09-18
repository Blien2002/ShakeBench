"""Runtime verification for the compact, package-owned ShakeBench assets.

Runtime loading authenticates package-owned files only.  It never searches
for raw evidence, opens an evidence archive, contacts the network, or treats a
stored full-audit boolean as authority.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shakebench.utils.artifacts import write_json

RUNTIME_CONTRACT_FILENAME = "shakebench_runtime_contract.json"
OFFICIAL_PHYSICS_PROFILE_FILENAME = "shakebench_official_physics.yaml"
OFFICIAL_PHYSICS_PROFILE_ID = "shakebench.official.physics.v2"
SCENE_ASSET_FILENAMES = (
    "shakebench_geometry_world_fixed_arm_v1.json",
    "shakebench_scene_world_fixed_arm_v1.json",
    "arenas/shakebench_arena.xml",
    "arenas/shakebench_robot_support.xml",
)


class RuntimePublicationError(ValueError):
    """Raised when the compact package asset contract is invalid."""


def _read_json(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimePublicationError(f"unreadable compact runtime asset {filename}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimePublicationError(f"compact runtime asset {filename} is not an object")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimePublicationError(f"unreadable official profile: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimePublicationError("official profile is not an object")
    return value


def build_runtime_contract(asset_root: str | Path) -> dict[str, Any]:
    """Return the current contract binding for the package-owned assets."""

    root = Path(asset_root).resolve()
    profile_path = root / OFFICIAL_PHYSICS_PROFILE_FILENAME
    try:
        profile = _load_yaml(profile_path)
    except (OSError, RuntimePublicationError) as exc:
        raise RuntimePublicationError(f"unreadable official profile: {exc}") from exc
    contract: dict[str, Any] = {
        "schema_id": "shakebench.runtime.publication_contract",
        "schema_version": 1,
        "contract_id": "shakebench.runtime.publication.v2",
        "package_contract": {
            "raw_archive_required_for_runtime": False,
            "network_access_required": False,
            "full_audit_requires_explicit_evidence": True,
            "runtime_verification": "structural_asset_checks_only",
        },
        "physics_profile": {
            "asset_filename": profile_path.name,
            "profile_id": profile.get("profile_id"),
        },
        "scene_assets": [],
    }
    for name in SCENE_ASSET_FILENAMES:
        if not (root / name).is_file():
            raise RuntimePublicationError(f"current scene asset is missing: {name}")
        contract["scene_assets"].append(name)
    return contract


def write_runtime_contract(asset_root: str | Path) -> Path:
    """Rewrite the compact contract from the current assets."""

    root = Path(asset_root).resolve()
    path = root / RUNTIME_CONTRACT_FILENAME
    write_json(path, build_runtime_contract(root))
    return path


def verify_runtime_publication_bundle(asset_root: str | Path, *, include_profile: bool = False) -> dict[str, Any]:
    """Verify the compact contract, the official profile, and the scene assets."""

    root = Path(asset_root).resolve()
    errors: list[str] = []
    try:
        contract = _read_json(root, RUNTIME_CONTRACT_FILENAME)
    except RuntimePublicationError as exc:
        return {"passed": False, "errors": [str(exc)]}

    if contract.get("schema_id") != "shakebench.runtime.publication_contract" or contract.get("schema_version") != 1:
        return {"passed": False, "errors": ["runtime contract schema mismatch"]}
    package_contract = contract.get("package_contract")
    if not isinstance(package_contract, Mapping):
        errors.append("runtime package contract missing")
    else:
        if package_contract.get("raw_archive_required_for_runtime") is not False:
            errors.append("runtime contract requires raw archive")
        if package_contract.get("network_access_required") is not False:
            errors.append("runtime contract requires network")
        if package_contract.get("full_audit_requires_explicit_evidence") is not True:
            errors.append("full-audit explicit-evidence contract missing")

    profile_spec = contract.get("physics_profile")
    if not isinstance(profile_spec, Mapping):
        errors.append("runtime physics profile binding missing")
        return {"passed": False, "errors": sorted(set(errors))}

    profile_name = profile_spec.get("asset_filename")
    if not isinstance(profile_name, str):
        errors.append("runtime profile asset filename is missing")
        return {"passed": False, "errors": sorted(set(errors))}
    profile_path = root / profile_name
    try:
        profile = _load_yaml(profile_path)
    except RuntimePublicationError as exc:
        errors.append(str(exc))
        profile = {}
    if not profile_path.is_file():
        errors.append("official profile asset is missing")

    if profile:
        if profile.get("profile_id") != OFFICIAL_PHYSICS_PROFILE_ID:
            errors.append("official profile id mismatch")
        if profile.get("status") != "official_immutable" or profile.get("scoreable") is not True:
            errors.append("official profile is not immutable scoreable")

    scene_assets = contract.get("scene_assets")
    if not isinstance(scene_assets, list) or sorted(scene_assets) != sorted(SCENE_ASSET_FILENAMES):
        errors.append("runtime scene asset list mismatch")
    else:
        for name in scene_assets:
            if not (root / name).is_file():
                errors.append(f"scene asset is missing: {name}")

    result = {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "profile_id": profile_spec.get("profile_id"),
        "raw_archive_required_for_runtime": False,
        "network_access_required": False,
    }
    if include_profile and profile:
        result["profile_payload"] = dict(profile)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        default=str(Path(__file__).resolve().parents[1] / "models" / "assets"),
    )
    parser.add_argument("--write", action="store_true", help="rewrite the contract from the current assets")
    args = parser.parse_args(argv)
    if args.write:
        path = write_runtime_contract(args.asset_root)
        print(f"wrote {path}")
        return 0
    result = verify_runtime_publication_bundle(args.asset_root, include_profile=True)
    payload = {key: value for key, value in result.items() if key != "profile_payload"}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNTIME_CONTRACT_FILENAME",
    "RuntimePublicationError",
    "build_runtime_contract",
    "verify_runtime_publication_bundle",
    "write_runtime_contract",
]
