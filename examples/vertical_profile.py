#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_met_vertical_profile.py
===============================
A vertical profile (temperature, pressure, humidity, dynamics vs altitude) ->
per-level meteorological nucleation reports for both phases, a CSV table and
a vertical-profile figure.

The profile is a simple idealised mid-latitude ascent: temperature falls with
height, pressure follows the hydrostatic relation, humidity rises toward
saturation near cloud top, with a supercooled-liquid layer and an updraft.
Demonstrates the `evaluate_profile` array driver and the vertical-profile
plotter, and how subsaturated levels (S<1) are reported without crashing.
"""
import io
import os
import sys
import math

import numpy as np

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

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "vertical_profile")
os.makedirs(OUTDIR, exist_ok=True)


def hydrostatic_profile():
    z = np.linspace(500.0, 7000.0, 20)              # altitude [m]
    T0, P0 = 285.0, 95000.0                          # surface-ish
    lapse = 6.5e-3                                   # K/m
    T = T0 - lapse * z
    P = P0 * (T / T0) ** (9.81 / (287.0 * lapse))     # hydrostatic
    # humidity: 80% wrt water near ground, rising to ~108% (slight supersat)
    # in the supercooled layer (z 2500..4500 m), near-saturation aloft
    RH = 80.0 + (z / z[-1]) * 25.0
    RH[(z > 2500) & (z < 4500)] = 108.0
    RH[z > 5500] = 95.0
    w = np.where((z > 1500) & (z < 5000), 3.0, 0.5)  # updraft [m/s]
    LWC = np.where((z > 2500) & (z < 4500), 5.0e-4, 0.0)
    IWC = np.where(z > 4000, 2.0e-4, 0.0)
    cooling = np.where(w > 1.0, -3.0e-4, -5.0e-5)    # K/s
    return z, T, P, RH, w, LWC, IWC, cooling


def main():
    z, T, P, RH, w, LWC, IWC, cooling = hydrostatic_profile()
    met = M.MetInput(phase_mode="both", mode="homogeneous",
                     dt_micro=60.0, cell_volume=1.0e6)
    runner = M.MetNucleationRunner(met)

    all_reps = []
    rows = []
    for i in range(len(z)):
        Ti, Pi, RHi = float(T[i]), float(P[i]), float(RH[i])
        met_i = M.MetInput(T=Ti, P=Pi, RH=RHi, phase_mode="both",
                           dt_micro=60.0, cell_volume=1.0e6)
        runner_i = M.MetNucleationRunner(met_i)
        pv, _, _ = M.resolve_humidity(met_i, Ti, Pi)
        reps = runner_i.evaluate_point(Ti, Pi, pv, dynamics={
            "w": float(w[i]), "LWC": float(LWC[i]), "IWC": float(IWC[i]),
            "cooling_rate": float(cooling[i]), "z": float(z[i])})
        all_reps.append(reps)
        for ph, r in reps.items():
            rows.append({
                "z_m": float(z[i]), "phase": ph, "status": r.status,
                "T_ambient_K": r.T_ambient_K, "S_water": r.S_water,
                "S_ice": r.S_ice, "gradT_K_m": r.gradT_K_m,
                "r_critical_2nd_m": r.r_critical_2nd_m,
                "log10_nucleation_rate": r.log10_nucleation_rate,
                "diagnostic_class": r.diagnostic_class,
                "rain_fav": r.rain_favorability, "snow_fav": r.snow_favorability,
                "graupel_fav": r.graupel_favorability, "hail_fav": r.hail_favorability,
                "confidence": r.confidence,
            })

    # CSV table
    import csv
    cpath = os.path.join(OUTDIR, "vertical_profile.csv")
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"Saved profile table -> {cpath}  ({len(rows)} rows)")

    # vertical-profile figure (uses per-level log10I / r_C / favourability)
    plot = M.MetNucleationPlotter(OUTDIR)
    plot.plot_vertical_profile(all_reps, z)
    print(f"Saved -> {os.path.join(OUTDIR, 'vertical_profile.png')}")

    # JSON of all reports
    jpath = os.path.join(OUTDIR, "vertical_profile.json")
    M.to_json(all_reps, jpath)
    print(f"Saved -> {jpath}")

    # brief console summary
    print("\nz[m]   phase   status       S_w    S_i    log10I   class")
    for i, reps in enumerate(all_reps):
        for ph in (M.PHASE_LIQUID, M.PHASE_ICE):
            r = reps.get(ph)
            if r is None:
                continue
            print(f"{z[i]:6.0f}  {ph:6s}  {r.status:11s}  "
                  f"{r.S_water:5.2f}  {r.S_ice:5.2f}  "
                  f"{r.log10_nucleation_rate:7.2f}  {r.diagnostic_class}")
    return 0


if __name__ == "__main__":
    sys.exit(main())