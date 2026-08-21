#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_met_single_state.py
===========================
One atmospheric condition -> a full meteorological nucleation report for both
phases (liquid & ice), printed as JSON and saved to out_met_nucleation/.

Scenario: a supercooled, slightly supersaturated mid-tropospheric parcel
(T = 260 K, P = 700 hPa, RH = 110% wrt water) with a modest updraft, some
supercooled LWC and IWC, and a microphysical timestep / cell volume so that
`expected_events` is determined.  Demonstrates:
  * auto phase selection (both phases computed, dominant reported);
  * the full 47-field mandatory schema + metadata + assumptions/warnings;
  * the precipitation diagnosis (favourability indices + diagnostic class);
  * expected_events = I * dt * V_cell.
"""
import io
import json
import os
import sys

# utf-8 console (Windows cp1252 cannot print the special chars)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

# Bootstrap the package import when running from a source checkout without
# `pip install -e .` (path resolved relative to this file, not to CWD).
try:
    import met_water_nucleation as M
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    import met_water_nucleation as M

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "single_state")
os.makedirs(OUTDIR, exist_ok=True)


def main():
    T = 260.0            # ambient temperature [K]
    P = 70000.0          # total pressure [Pa]
    RH = 110.0           # relative humidity [%] wrt water
    met = M.MetInput(
        T=T, P=P, RH=RH, rh_reference="water",
        phase_mode="both", mode="homogeneous",
        r_ref=M.R_REF_DEFAULT,
        w=2.0, LWC=5.0e-4, IWC=1.0e-4, cooling_rate=-2.0e-4,
        N_ccn=3.0e8, N_inp=1.0e4,
        dt_micro=60.0, cell_volume=1.0e6,        # so expected_events is determined
    )
    runner = M.MetNucleationRunner(met)
    pv, src, warns = M.resolve_humidity(met, T, P)
    print(f"resolved p_v = {pv:.4f} Pa (source: {src})")
    if warns:
        print("humidity warnings:", warns)

    reps = runner.evaluate_point(T, P, pv, dynamics={
        "w": 2.0, "LWC": 5.0e-4, "IWC": 1.0e-4, "cooling_rate": -2.0e-4,
        "N_ccn": 3.0e8, "N_inp": 1.0e4})

    print("\n" + "=" * 78)
    print("SINGLE-STATE METEOROLOGICAL NUCLEATION REPORT")
    print("=" * 78)
    for ph, r in reps.items():
        print(f"\n--- phase = {ph}  (status={r.status}, mode={r.nucleation_mode}) ---")
        d = r.to_dict()
        for c in M.MANDATORY_FIELDS:
            v = d[c]
            if isinstance(v, float):
                v = f"{v:.6e}" if (v and v == v and abs(v) not in (0,)) else (
                    "nan" if v != v else f"{v:.6e}")
            print(f"  {c:28s} = {v}")
        print("  assumptions: " + "; ".join(r.assumptions))
        if r.warnings:
            print("  warnings: " + "; ".join(r.warnings))
        print("  validity_flags: " + ", ".join(r.validity_flags))
        print("  favourability detail:")
        for k, fv in r.favorability_detail.items():
            print(f"     {k:7s} = {fv['value']:.3f}  conf={fv['confidence']:.2f}  "
                  f"contrib={fv['contributing_vars']}  missing={fv['missing_vars']}")
            if fv["caveat"]:
                print(f"            caveat: {fv['caveat']}")

    # save JSON + CSV + NetCDF
    jpath = os.path.join(OUTDIR, "single_state_report.json")
    M.to_json(reps, jpath)
    M.to_csv(reps, os.path.join(OUTDIR, "single_state_report.csv"))
    M.to_netcdf(reps, os.path.join(OUTDIR, "single_state_report.nc"))
    print(f"\nSaved: {jpath}  (+ .csv, .nc)")

    # a favourability bar figure for the liquid phase
    plot = M.MetNucleationPlotter(OUTDIR)
    plot.plot_favorability_bars(reps[M.PHASE_LIQUID])
    print(f"Saved: {os.path.join(OUTDIR, 'favorability_bars.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())