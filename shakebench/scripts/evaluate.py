"""Evaluate any policy factory on ShakeBench states with one frozen result contract.

Command-line entry point; the implementation lives in :mod:`shakebench.evaluation.evaluate`.
"""

from shakebench.evaluation.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
