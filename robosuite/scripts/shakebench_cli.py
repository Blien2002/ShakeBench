"""Small command-line entry point for the Phase 00 ShakeBench config envelope."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Sequence

from robosuite.utils.shakebench_config import (
    EXTENSION_ID,
    EXTENSION_VERSION,
    PYTHON_PACKAGE,
    SCHEMA_VERSION,
    ShakeBenchConfigError,
    config_hash,
    schema,
    validate_config,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Phase 00 configuration commands."""

    parser = argparse.ArgumentParser(prog="shakebench", description="ShakeBench configuration utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print extension and package identity")
    subparsers.add_parser("print-schema", help="print the Phase 00 JSON schema")

    validate_parser = subparsers.add_parser("validate-config", help="validate a JSON configuration")
    validate_parser.add_argument(
        "config_path",
        nargs="?",
        help="JSON file to validate; use '-' or omit the path to read stdin",
    )
    validate_parser.add_argument(
        "--config",
        dest="config_option",
        help="JSON file to validate (alternative to the positional path)",
    )
    return parser


def _package_version() -> str:
    try:
        return metadata.version(PYTHON_PACKAGE)
    except metadata.PackageNotFoundError:
        return "unknown"


def _read_config(path: str | None) -> object:
    if path is None or path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_validate(args: argparse.Namespace) -> int:
    """Validate the selected configuration path and print its hash."""

    if args.config_path is not None and args.config_option is not None:
        print("error: specify either a positional path or --config, not both", file=sys.stderr)
        return 2

    path = args.config_option if args.config_option is not None else args.config_path
    try:
        config = validate_config(_read_config(path))
    except (OSError, json.JSONDecodeError, ShakeBenchConfigError, TypeError, ValueError) as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "valid": True,
                "config_hash": config_hash(config),
                "schema_version": config.schema_version,
                "scoreable": config.scoreable,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-style status code."""

    args = _build_parser().parse_args(argv)
    if args.command == "version":
        print(
            json.dumps(
                {
                    "extension_id": EXTENSION_ID,
                    "extension_version": EXTENSION_VERSION,
                    "package": PYTHON_PACKAGE,
                    "package_version": _package_version(),
                    "schema_version": SCHEMA_VERSION,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "print-schema":
        print(json.dumps(schema(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-config":
        return _run_validate(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
