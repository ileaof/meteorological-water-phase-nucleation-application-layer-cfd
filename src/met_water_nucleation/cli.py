"""Command-line interface for the met_water_nucleation package.

Delegates to the engine's argparse CLI (``met_h2o_nucleation.main``), which is
loaded read-only.  Installed as the console script ``met-water-nucleation`` and
reachable as ``python -m met_water_nucleation``.
"""
import sys

from . import engine


def main(argv=None) -> int:
    """Run the meteorological nucleation CLI. Returns the process exit code."""
    return engine.main(argv)


if __name__ == "__main__":
    sys.exit(main())