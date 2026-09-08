"""Build a Phase 08 scorecard from existing raw episode JSON; never runs physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.scripts.shakebench_run_oracle import verify_run_artifact
from robosuite.utils.shakebench_artifacts import file_sha256
from robosuite.utils.shakebench_scoring import EpisodeResult, ScorecardError, build_scorecard, scorecard_payload_hash
from robosuite.scripts.shakebench_cpu_batch import verify_batch_aggregate


def _append_verified_raw(path: Path, rows: list[EpisodeResult], sources: list[dict[str, object]]) -> None:
    verdict = verify_run_artifact(path)
    if not verdict["passed"]:
        raise ScorecardError(f"raw artifact rejected: {path}: {', '.join(verdict['errors'])}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows.extend(EpisodeResult.from_verified_raw(episode, verdict) for episode in raw["episodes"])
    sources.append({"path": str(path), "sha256": file_sha256(path), "semantic_verifier": verdict})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--raw-artifact",
        type=Path,
        action="append",
        help="authenticated schema-v5 raw run artifact; repeat for each tier",
    )
    inputs.add_argument(
        "--batch-aggregate",
        type=Path,
        action="append",
        help="authenticated CPU batch aggregate; required for production Phase 9 scorecards",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete-tiers", action="store_true", default=True)
    args = parser.parse_args(argv)
    rows = []
    sources = []
    retry_ledgers = []
    conflict_ledgers = []
    missing_state_ids = []
    incomplete_groups = []
    resource_telemetry = []
    batch_sources = []
    if args.raw_artifact:
        if args.require_complete_tiers:
            parser.error("production complete scorecards require --batch-aggregate for retry and telemetry audit")
        for path in args.raw_artifact:
            _append_verified_raw(path, rows, sources)
    else:
        for aggregate_path in args.batch_aggregate:
            aggregate_verdict = verify_batch_aggregate(aggregate_path)
            if not aggregate_verdict["passed"]:
                raise ScorecardError(
                    f"batch aggregate rejected: {aggregate_path}: {', '.join(aggregate_verdict['errors'])}"
                )
            aggregate = aggregate_verdict["payload"]
            batch_sources.append({"path": str(aggregate_path), "sha256": file_sha256(aggregate_path)})
            resource_telemetry.append(
                {
                    "path": str(aggregate_path),
                    "before": aggregate["resource_before"],
                    "after": aggregate["resource_after"],
                    "complete": aggregate.get("telemetry_complete"),
                    "reasons": aggregate.get("resource_limit_reasons", []),
                }
            )
            incomplete_groups.extend(aggregate.get("incomplete_groups", []))
            for record in aggregate["records"]:
                ledger = record.get("retry_ledger")
                if isinstance(ledger, dict):
                    retry_ledgers.append(ledger)
                    for event in ledger.get("events", []):
                        if isinstance(event, dict) and isinstance(event.get("conflict_ledger"), str):
                            conflict_path = Path(event["conflict_ledger"])
                            if conflict_path.is_file():
                                conflict_ledgers.append(json.loads(conflict_path.read_text(encoding="utf-8")))
                if record.get("semantic_passed") is True:
                    output = record.get("output")
                    if not isinstance(output, str):
                        raise ScorecardError("successful batch record has no output path")
                    if record.get("output_sha256") != file_sha256(output):
                        raise ScorecardError("batch record output hash mismatch")
                    _append_verified_raw(Path(output), rows, sources)
                else:
                    missing_state_ids.append(record.get("science", {}).get("state_id"))
    scorecard = build_scorecard(rows, require_complete_tiers=args.require_complete_tiers)
    scorecard["raw_artifacts"] = sources
    scorecard["batch_aggregates"] = batch_sources
    scorecard["retry_ledgers"] = retry_ledgers
    scorecard["duplicate_conflict_ledgers"] = conflict_ledgers
    scorecard["missing_state_ids"] = sorted(str(state_id) for state_id in missing_state_ids if state_id is not None)
    scorecard["incomplete_groups"] = sorted(str(group) for group in incomplete_groups)
    scorecard["resource_telemetry"] = resource_telemetry
    scorecard["payload_sha256"] = scorecard_payload_hash(scorecard)
    write_json_atomic(args.output, scorecard)
    print(
        json.dumps(
            {"payload_sha256": scorecard["payload_sha256"], "state_count": scorecard["state_count"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
