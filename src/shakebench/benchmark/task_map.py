"""Immutable benchmark task descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    language: str
    config: dict[str, Any]
