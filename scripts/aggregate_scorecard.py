#!/usr/bin/env python3
"""Aggregate scorecard JSON files into the generated README leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shakebench.benchmark.scorecard import decision_point


POLICY_ORDER = (
    "oracle_full",
    "oracle_phase",
    "oracle_reactive",
    "classical",
    "scripted",
    "random",
)
START = "<!-- LEADERBOARD_START -->"
END = "<!-- LEADERBOARD_END -->"


def _number(value, digits: int = 3) -> str:
    if value is None:
        return "pending"
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return "pending"
    return f"{float(value):.{digits}f}"


def leaderboard(cards: list[dict]) -> str:
    by_policy: dict[str, list[dict]] = {}
    for card in cards:
        by_policy.setdefault(card["policy_name"], []).append(card)
    lines = [
        "| Policy | Privileged | control_freq | SR @Γ=0.5 | Γ_c | f_c | grip_excess |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_ORDER:
        candidates = by_policy.get(policy, [])
        if not candidates:
            lines.append(f"| `{policy}` | pending | pending | pending | pending | pending | pending |")
            continue
        # Prefer a card containing a critical value, then the largest valid set.
        card = max(
            candidates,
            key=lambda item: (
                (
                    item.get("metrics", {}).get("Gamma_c", {}).get("mean")
                    if isinstance(item.get("metrics", {}).get("Gamma_c"), dict)
                    else item.get("metrics", {}).get("Gamma_c")
                )
                is not None,
                item.get("n_valid", 0),
            ),
        )
        metrics = card.get("metrics", {})
        privileges = card.get("requires_privileged", [])
        privilege = ", ".join(privileges) if privileges else "none"
        lines.append(
            f"| `{policy}` | {privilege} | {card.get('control_freq', 'pending')} | "
            f"{_number(metrics.get('SR'))} | {_number(metrics.get('Gamma_c'))} | "
            f"{_number(metrics.get('f_c'))} | {_number(metrics.get('grip_excess'))} |"
        )
    return "\n".join(lines)


def update_readme(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{path} is missing leaderboard markers")
    prefix, remainder = text.split(START, 1)
    _, suffix = remainder.split(END, 1)
    path.write_text(f"{prefix}{START}\n{table}\n{END}{suffix}", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scorecards", nargs="*", type=Path)
    parser.add_argument("--readme", type=Path)
    parser.add_argument(
        "--decision-output",
        type=Path,
        help="write paired oracle_phase-vs-reactive decision statistics",
    )
    args = parser.parse_args()
    paths = args.scorecards or sorted(Path("out").glob("scorecard_*.json"))
    cards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    table = leaderboard(cards)
    print(table)
    if args.readme is not None:
        update_readme(args.readme, table)
    if args.decision_output is not None:
        phase = next((card for card in cards if card["policy_name"] == "oracle_phase"), None)
        reactive = next(
            (card for card in cards if card["policy_name"] == "oracle_reactive"), None
        )
        if phase is None or reactive is None:
            raise ValueError("decision output requires both oracle_phase and oracle_reactive cards")
        args.decision_output.parent.mkdir(parents=True, exist_ok=True)
        args.decision_output.write_text(
            json.dumps(decision_point(phase, reactive), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
