"""Allow ``python -m vibench`` from an Isaac Lab environment."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
