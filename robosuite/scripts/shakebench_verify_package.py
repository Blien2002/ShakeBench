"""Clean-install wheel and sdist package evidence verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_installed_check(install_path: Path, fixture_path: Path, repo_root: Path) -> dict[str, Any]:
    code = r"""
import hashlib, json, pathlib
import sys
sys.meta_path[:] = [finder for finder in sys.meta_path if "robosuite" not in repr(finder).lower()]
import robosuite
from robosuite import models
from robosuite.utils.shakebench_oracle import OracleControllerProfile
from robosuite.utils.shakebench_dev_states import verify_phase07_dev_state_artifact
from robosuite.scripts.shakebench_run_oracle import verify_run_artifact

root = pathlib.Path(models.assets_root)
state_path = root / "shakebench_states_dev.json"
state_payload = json.loads(state_path.read_text(encoding="utf-8"))
state_hash = state_payload["artifact_lock"]["payload_sha256"]
state_check = verify_phase07_dev_state_artifact(state_path)
profile = OracleControllerProfile()
module_path = pathlib.Path(__import__("robosuite.utils.shakebench_oracle", fromlist=["__file__"]).__file__).resolve()
artifact = pathlib.Path(__FIXTURE__)
artifact_check = verify_run_artifact(artifact)
env = robosuite.make(
    "VibrationPickPlaceCan", robots="Panda", has_renderer=False,
    has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False,
    physics_profile="official", observation_tier="V0", imu_mode="ideal_smoke",
    horizon=1, ignore_done=True, seed=0,
)
try:
    env.reset()
    smoke = True
finally:
    env.close()
print(json.dumps({
    "module_path": str(module_path),
    "profile_sha256": profile.sha256,
    "dev_state_path": str(state_path),
    "dev_state_payload_sha256": state_hash,
    "dev_state_verifier": state_check,
    "artifact_verifier": artifact_check,
    "environment_smoke": smoke,
    "passed": bool(state_check.get("passed") and artifact_check.get("passed") and smoke),
}, sort_keys=True))
    """.replace(
        "__FIXTURE__", repr(str(fixture_path))
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_path.resolve())
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root / "out" / "phase07r2",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = completed.stdout.splitlines()
    parsed = None
    if lines:
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError:
            parsed = None
    return {
        "exit_code": completed.returncode,
        "stdout_summary": lines[-5:],
        "stderr_summary": completed.stderr.splitlines()[-10:],
        "check": parsed,
        "passed": completed.returncode == 0 and isinstance(parsed, dict),
    }


def verify_packages(
    wheel: str | Path,
    sdist: str | Path,
    fixture: str | Path,
    output: str | Path,
    *,
    install_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = Path(install_root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, artifact in (("wheel", Path(wheel)), ("sdist", Path(sdist))):
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
                str(artifact.resolve()),
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        check = None if install.returncode != 0 else _run_installed_check(target, Path(fixture).resolve(), repo)
        records.append(
            {
                "kind": kind,
                "artifact_path": str(artifact.resolve()),
                "artifact_sha256": _sha256(artifact),
                "install_path": str(target.resolve()),
                "install_exit_code": install.returncode,
                "install_stdout_summary": install.stdout.splitlines()[-5:],
                "install_stderr_summary": install.stderr.splitlines()[-10:],
                "check": check,
                "passed": install.returncode == 0 and bool(check and check.get("passed")),
            }
        )
    result = {
        "schema_id": "shakebench.phase07.package_evidence",
        "schema_version": 1,
        "fixture_path": str(Path(fixture).resolve()),
        "fixture_sha256": _sha256(Path(fixture)),
        "records": records,
        "passed": all(record["passed"] for record in records),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--sdist", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--install-root", required=True)
    args = parser.parse_args()
    result = verify_packages(
        args.wheel,
        args.sdist,
        args.fixture,
        args.output,
        install_root=args.install_root,
        repo_root=Path(__file__).resolve().parents[2],
    )
    print(json.dumps({"output": args.output, "passed": result["passed"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
