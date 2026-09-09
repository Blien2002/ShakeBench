"""Fail-closed Phase 08R handoff verifier.

This verifier admits Phase 09 only after nominal solvability, a positive-Gamma
diagnostic with an observed controller event, three-process replay, and clean
package evidence have all passed. It never runs a rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robosuite.scripts.shakebench_run_oracle import (
    load_dev_states,
    verify_determinism_manifest,
    verify_run_artifact,
)


def verify_phase08r_handoff(
    *, nominal_path: str | Path, positive_gamma_path: str | Path, determinism_path: str | Path, package_path: str | Path
) -> dict[str, Any]:
    """Verify all evidence required to reopen Phase 09."""

    errors: list[str] = []
    nominal = Path(nominal_path)
    positive = Path(positive_gamma_path)
    determinism = Path(determinism_path)
    package = Path(package_path)
    nominal_verdict = verify_run_artifact(nominal)
    positive_verdict = verify_run_artifact(positive)
    determinism_verdict = verify_determinism_manifest(determinism)
    try:
        package_payload = json.loads(package.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        package_payload = None

    if not nominal_verdict["passed"]:
        errors.append("nominal artifact verifier")
    else:
        nominal_payload = json.loads(nominal.read_text(encoding="utf-8"))
        episodes = nominal_payload.get("episodes", [])
        expected_ids = {state["state_id"] for state in load_dev_states("robosuite/models/assets/shakebench_states_dev.json")}
        if {episode.get("state_id") for episode in episodes} != expected_ids or len(episodes) != 10:
            errors.append("nominal ten-state completeness")
        if any(
            episode.get("episode_validity") != "valid"
            or episode.get("score_outcome") != "success"
            or episode.get("termination_cause") != "success_latched"
            for episode in episodes
        ):
            errors.append("nominal solvability")

    if not positive_verdict["passed"]:
        errors.append("positive-Gamma artifact verifier")
    else:
        positive_payload = json.loads(positive.read_text(encoding="utf-8"))
        if float(positive_payload.get("gamma_commanded", 0.0)) <= 0.0:
            errors.append("positive-Gamma diagnostic is not positive")
        if not any(episode.get("controller_events") for episode in positive_payload.get("episodes", [])):
            errors.append("positive-Gamma recovery event is missing")

    if not determinism_verdict["passed"]:
        errors.append("three-process determinism")
    if not isinstance(package_payload, dict) or package_payload.get("passed") is not True:
        errors.append("clean package evidence")

    return {
        "schema_id": "shakebench.phase08r.handoff_verdict",
        "schema_version": 1,
        "passed": not errors,
        "errors": sorted(set(errors)),
        "nominal": nominal_verdict,
        "positive_gamma": positive_verdict,
        "determinism": determinism_verdict,
        "package_passed": isinstance(package_payload, dict) and package_payload.get("passed") is True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", required=True, type=Path)
    parser.add_argument("--positive-gamma", required=True, type=Path)
    parser.add_argument("--determinism", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    verdict = verify_phase08r_handoff(
        nominal_path=args.nominal,
        positive_gamma_path=args.positive_gamma,
        determinism_path=args.determinism,
        package_path=args.package,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": verdict["passed"]}, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["verify_phase08r_handoff", "main"]
