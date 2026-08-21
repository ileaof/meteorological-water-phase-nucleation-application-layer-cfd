#!/usr/bin/env python3
"""Regenerate the per-scenario example outputs into outputs/<scenario>/.

This runs every example script. It writes to `outputs/<scenario>/`
(gitignored) and does NOT touch the committed flat reference outputs at
`outputs/*.{json,csv,nc,png}` (which are kept as historical references).

Usage:
    python scripts/regenerate_outputs.py
"""
import os
import subprocess
import sys

EXAMPLES = [
    "single_state",
    "vertical_profile",
    "xarray_netcdf",
    "frontal_collision",
    "figures",
]


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ex_dir = os.path.join(here, "examples")
    rc_all = 0
    for name in EXAMPLES:
        script = os.path.join(ex_dir, name + ".py")
        print(f"\n--- {name} ---")
        rc = subprocess.call([sys.executable, script])
        if rc != 0:
            print(f"  !! {name} FAILED (rc={rc})")
            rc_all = rc
    print("\nDone. Outputs are under outputs/<scenario>/ (gitignored).")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())