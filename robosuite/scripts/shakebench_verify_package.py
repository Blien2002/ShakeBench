"""Verify clean wheel and sdist installs without source-checkout leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

FORBIDDEN_COMPONENTS = {
    "__pycache__",
    "build",
    "cache",
    "dist",
    "package_install",
    "site-packages",
    "virtualenv",
    "venv",
}
FORBIDDEN_SUFFIXES = (".whl", ".tar.gz", ".tar.zst")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _child_code(fixture_path: Path, install_root: Path, repo_root: Path) -> str:
    return (
        r"""
import hashlib
import json
import os
import pathlib
import socket
import sys
import urllib.request

install_root = pathlib.Path(__INSTALL_ROOT__).resolve()
repo_root = pathlib.Path(__REPO_ROOT__).resolve()
fixture = pathlib.Path(__FIXTURE__).resolve()

def inside(path, root):
    try:
        pathlib.Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False

# -c normally leaves the current directory as an empty sys.path entry.  The
# subprocess cwd is an empty temporary directory, and all checkout paths are
# removed explicitly so a source tree cannot satisfy an import.
sys.path[:] = [str(install_root)] + [
    item for item in sys.path
    if item and not inside(item, repo_root) and pathlib.Path(item).resolve() != install_root
]

try:
    import robosuite
    from robosuite import models
    from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan
    from robosuite.scripts.shakebench_run_oracle import verify_run_artifact
    from robosuite.utils.shakebench_dev_states import verify_phase07_dev_state_artifact
    from robosuite.utils.shakebench_physics import load_official_physics_profile
    from robosuite.utils.shakebench_runtime_verifier import verify_runtime_publication_bundle

    module_names = (
        "robosuite",
        "robosuite.environments.manipulation.vibration_pick_place_can",
        "robosuite.scripts.shakebench_run_oracle",
        "robosuite.utils.shakebench_dev_states",
        "robosuite.utils.shakebench_physics",
        "robosuite.utils.shakebench_runtime_verifier",
    )
    module_paths = {}
    for name in module_names:
        module = __import__(name, fromlist=["__file__"])
        module_paths[name] = str(pathlib.Path(module.__file__).resolve())
    import_paths_clean = all(not inside(item, repo_root) for item in sys.path if item)
    imports_clean = all(inside(item, install_root) for item in module_paths.values())
    root = pathlib.Path(models.assets_root).resolve()
    state_path = root / "shakebench_states_dev.json"
    state_check = verify_phase07_dev_state_artifact(state_path)
    profile = load_official_physics_profile()
    runtime_check = verify_runtime_publication_bundle(root)
    artifact_check = verify_run_artifact(fixture)

    raw_asset_members = []
    for item in root.rglob("*"):
        lower = item.name.lower()
        if item.is_file() and ("raw" in lower or "diagnostic" in lower or "selected_diagnostics" in lower):
            raw_asset_members.append(str(item.relative_to(root)))

    original_socket = socket.socket
    original_connection = socket.create_connection
    original_urlopen = urllib.request.urlopen
    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden during runtime verification")
    socket.socket = blocked
    socket.create_connection = blocked
    urllib.request.urlopen = blocked
    try:
        env = robosuite.make(
            "VibrationPickPlaceCan", robots="Panda", has_renderer=False,
            has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False,
            physics_profile="official", observation_tier="V0", imu_mode="ideal_smoke",
            horizon=1, ignore_done=True, seed=0,
        )
        try:
            env.reset()
            environment_smoke = True
        finally:
            env.close()
    finally:
        socket.socket = original_socket
        socket.create_connection = original_connection
        urllib.request.urlopen = original_urlopen

    passed = bool(
        imports_clean
        and import_paths_clean
        and runtime_check.get("passed") is True
        and state_check.get("passed") is True
        and artifact_check.get("passed") is True
        and environment_smoke
        and not raw_asset_members
    )
    print(json.dumps({
        "module_path": module_paths["robosuite"],
        "module_paths": module_paths,
        "install_root": str(install_root),
        "source_checkout_in_import_path": not import_paths_clean,
        "imports_from_install_root": imports_clean,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "dev_state_path": str(state_path),
        "dev_state_verifier": state_check,
        "runtime_publication": runtime_check,
        "artifact_verifier": artifact_check,
        "environment_smoke": environment_smoke,
        "raw_asset_members": raw_asset_members,
        "passed": passed,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
    raise
""".replace(
            "__INSTALL_ROOT__", repr(str(install_root.resolve()))
        )
        .replace("__REPO_ROOT__", repr(str(repo_root.resolve())))
        .replace("__FIXTURE__", repr(str(fixture_path.resolve())))
    )


def _run_installed_check(install_path: Path, fixture_path: Path, repo_root: Path) -> dict[str, Any]:
    """Run the installed check in a fresh empty cwd and validate its result."""

    empty_cwd = Path(tempfile.mkdtemp(prefix="shakebench_package_check_cwd_"))
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment["PYTHONPATH"] = str(install_path.resolve())
    environment["SHAKEBENCH_INSTALL_ROOT"] = str(install_path.resolve())
    environment["SHAKEBENCH_REPO_ROOT"] = str(repo_root.resolve())
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _child_code(fixture_path, install_path, repo_root)],
            cwd=empty_cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        shutil.rmtree(empty_cwd, ignore_errors=True)
    lines = completed.stdout.splitlines()
    parsed: dict[str, Any] | None = None
    if lines:
        try:
            value = json.loads(lines[-1])
            if isinstance(value, dict):
                parsed = value
        except json.JSONDecodeError:
            parsed = None
    path_checks = False
    if isinstance(parsed, dict):
        module_paths = parsed.get("module_paths")
        path_checks = (
            parsed.get("imports_from_install_root") is True
            and parsed.get("source_checkout_in_import_path") is False
            and isinstance(module_paths, dict)
            and bool(module_paths)
            and all(_inside(path, install_path) for path in module_paths.values())
            and _inside(parsed.get("module_path", ""), install_path)
        )
    passed = bool(
        completed.returncode == 0 and isinstance(parsed, dict) and parsed.get("passed") is True and path_checks
    )
    return {
        "exit_code": completed.returncode,
        "stdout_summary": lines[-5:],
        "stderr_summary": completed.stderr.splitlines()[-10:],
        "check": parsed,
        "passed": passed,
    }


def _removed_paths(repo_root: Path) -> set[str]:
    path = repo_root / "docs" / "shakebench_evidence_removed_paths_v1.txt"
    if not path.is_file():
        return set()
    return {
        line.split("\t", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _forbidden_member(member: str, removed_paths: set[str]) -> str | None:
    normalized = member.replace("\\", "/").lstrip("./")
    parts = set(Path(normalized).parts)
    lower = normalized.lower()
    if normalized in removed_paths:
        return "removed-path list member"
    if parts & FORBIDDEN_COMPONENTS:
        return "install/build/cache member"
    if lower.endswith(FORBIDDEN_SUFFIXES):
        return "wheel/sdist/archive member"
    if "site-packages" in lower or lower.startswith("out/") or "/out/" in lower:
        return "generated evidence/install member"
    if "raw" in Path(normalized).name.lower() or "diagnostic" in Path(normalized).name.lower():
        return "raw/diagnostic evidence member"
    return None


def _inspect_artifact(artifact: Path, repo_root: Path) -> dict[str, Any]:
    removed_paths = _removed_paths(repo_root)
    members: list[str]
    try:
        if artifact.name.endswith(".whl"):
            with zipfile.ZipFile(artifact) as archive:
                members = archive.namelist()
        else:
            with tarfile.open(artifact, "r:*") as archive:
                members = archive.getnames()
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        return {"passed": False, "errors": [f"cannot inspect {artifact.name}: {exc}"], "member_count": 0}
    violations = []
    for member in members:
        reason = _forbidden_member(member, removed_paths)
        if reason:
            violations.append({"member": member, "reason": reason})
    return {"passed": not violations, "errors": violations, "member_count": len(members)}


def verify_packages(
    wheel: str | Path,
    sdist: str | Path,
    fixture: str | Path,
    output: str | Path,
    *,
    install_root: str | Path | None = None,
    repo_root: str | Path,
    keep_temp: bool = False,
) -> dict[str, Any]:
    """Install both artifacts into isolated roots and emit one evidence JSON."""

    repo = Path(repo_root).resolve()
    wheel_path = Path(wheel).resolve()
    sdist_path = Path(sdist).resolve()
    fixture_path = Path(fixture).resolve()
    output_path = Path(output).resolve()
    if not wheel_path.is_file() or not sdist_path.is_file() or not fixture_path.is_file():
        result = {
            "schema_id": "shakebench.phase07.package_evidence",
            "schema_version": 2,
            "records": [],
            "passed": False,
            "errors": ["wheel, sdist, and fixture must be explicit existing files"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return result
    owned_root = install_root is None
    root = Path(tempfile.mkdtemp(prefix="shakebench_package_install_")) if owned_root else Path(install_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if output_path.is_relative_to(root):
        raise ValueError("package evidence output must not be inside install root")
    records: list[dict[str, Any]] = []
    try:
        for kind, artifact in (("wheel", wheel_path), ("sdist", sdist_path)):
            target = root / kind
            target.mkdir(parents=True, exist_ok=True)
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--target",
                    str(target),
                    str(artifact),
                ],
                cwd=Path(tempfile.gettempdir()),
                capture_output=True,
                text=True,
                check=False,
            )
            content = _inspect_artifact(artifact, repo)
            check = None
            if install.returncode == 0 and content.get("passed") is True:
                check = _run_installed_check(target, fixture_path, repo)
            records.append(
                {
                    "kind": kind,
                    "artifact_path": str(artifact),
                    "artifact_filename": artifact.name,
                    "artifact_bytes": artifact.stat().st_size,
                    "artifact_sha256": _sha256(artifact),
                    "content_check": content,
                    "install_exit_code": install.returncode,
                    "install_stdout_summary": install.stdout.splitlines()[-5:],
                    "install_stderr_summary": install.stderr.splitlines()[-10:],
                    "check": check,
                    "passed": bool(
                        install.returncode == 0
                        and content.get("passed") is True
                        and check is not None
                        and check.get("passed") is True
                    ),
                }
            )
        result = {
            "schema_id": "shakebench.phase07.package_evidence",
            "schema_version": 2,
            "fixture_path": str(fixture_path),
            "fixture_sha256": _sha256(fixture_path),
            "records": records,
            "passed": bool(records) and all(record["passed"] for record in records),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if not keep_temp:
            if owned_root:
                shutil.rmtree(root, ignore_errors=True)
            else:
                for name in ("wheel", "sdist"):
                    shutil.rmtree(root / name, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--sdist", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--install-root", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    result = verify_packages(
        args.wheel,
        args.sdist,
        args.fixture,
        args.output,
        install_root=args.install_root,
        repo_root=repo_root,
        keep_temp=args.keep_temp,
    )
    print(json.dumps({"output": args.output, "passed": result["passed"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_run_installed_check", "verify_packages", "main"]
