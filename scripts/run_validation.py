#!/usr/bin/env python3
"""Run the full validation suite (core [1]-[21] + met-layer self-checks) and
the 24-test test suite. Exits 0 on success, 1 on any failure.

Usage:
    python scripts/run_validation.py
    # or just:  met-water-nucleation --validate  (lightweight in-process gate)
"""
import sys

from met_water_nucleation import engine


def main() -> int:
    print("=" * 78)
    print("1) in-process self-checks (--validate)")
    print("=" * 78)
    ok = engine.run_self_checks(verbose=True)
    if not ok:
        print("SELF-CHECKS FAILED")
        return 1

    print("\n" + "=" * 78)
    print("2) full test suite (tests/test_met_nucleation.py)")
    print("=" * 78)
    import os
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test = os.path.join(here, "tests", "test_met_nucleation.py")
    rc = subprocess.call([sys.executable, test])
    if rc != 0:
        print("TEST SUITE FAILED")
        return 1
    print("\nALL VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())