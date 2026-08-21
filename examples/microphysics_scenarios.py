"""Run the four precip_microphysics reference scenarios and print the
comparison that proves nucleation and precipitation are distinct stages.

    python examples/microphysics_scenarios.py [--kernel] [--json OUT.json]

``--kernel`` evaluates the validated second-order nucleation rate for each case
(slower, one kernel call per scenario); without it the CCN/Fletcher activation
pathway is used and the pipeline runs kernel-free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from precip_microphysics import scenarios   # noqa: E402


def _rain_like(diag):
    """The most-developed precipitation category in a diagnostic."""
    best = max(diag["categories"], key=lambda c: (c["diagnostic_level"], c["confidence"]))
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", action="store_true", help="use the nucleation kernel")
    ap.add_argument("--json", default=None, help="write full diagnostics to this path")
    args = ap.parse_args(argv)

    results = scenarios.run_all(use_kernel=args.kernel)

    hdr = (f"{'scenario':<34} {'log10I':>7} {'maxLvl':>7} {'category':>8} "
           f"{'lvl':>3} {'conf':>5} {'confirmed':>9} {'accum_mm':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, diag in results.items():
        best = _rain_like(diag)
        log10I = diag.get("nucleation", {}).get("log10I_liquid", float("nan"))
        print(f"{name:<34} {log10I:>7.1f} "
              f"{diag['overall']['max_diagnostic_level']:>7} "
              f"{best['category']:>8} {best['diagnostic_level']:>3} "
              f"{best['confidence']:>5.2f} {str(best['confirmed']):>9} "
              f"{best['accumulation_mm']:>9.3f}")

    print("\nInterpretation:")
    print("  * Scenario 1 has an enormous nucleation rate yet reaches Level 1 only")
    print("    (thermodynamic favourability) and confirms NOTHING -- the caveat holds.")
    print("  * Scenarios 2-4 confirm precipitation ONLY because the growth,")
    print("    sedimentation and (hail) survival chain actually ran.")

    # hail detail
    hail = next(c for c in results["4_deep_convective_hail"]["categories"]
                if c["category"] == "hail")
    print("\nHail (scenario 4) detail:")
    for k in ("diagnostic_level_name", "confidence", "confirmed", "growth_regime",
              "max_diameter_m", "melting_fraction", "surface_survival_probability",
              "accumulation_mm"):
        print(f"    {k:<32} {hail.get(k)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\nFull diagnostics written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
