#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_met_figures.py
======================
Generate the FULL visualization suite specified for the meteorological
nucleation module, writing PNGs to out_met_nucleation/:

  * P_eq,shift(T, gradT) 3-D surface ........... peq_shift_surface_{phase}.png
  * Gibbs-Thomson G^1/G^2 & r_C1/r_C2 vs gradT . gt_and_radii_{phase}.png
  * Free-energy decomposition vs radius ....... free_energy_vs_r.png
  * Nucleation rate log10 I vs T (liquid/ice) . rates_vs_T.png
  * Vertical profile (log10I / r_C / fav) ..... vertical_profile.png
  * Precipitation favourability bars .......... favorability_bars.png

Each figure exercises a distinct `MetNucleationPlotter` method on data
produced by the runner (sweeps over gradT and T) and, for the surface and
free-energy panels, the core models directly.  No new physics is introduced;
this is a visualization driver only.
"""
import io
import math
import os
import sys

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

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def _pv_supersaturated(T, P, frac=1.05):
    """Vapour partial pressure ~5% above water saturation (both phases admissible)."""
    sat = M.LiquidNucleationModel(M.AtmosphericInput(phase_mode="liquid"))
    return sat.Psat(T) * frac


def sweep_gradT(T=260.0, P=70000.0, ng=16):
    """Reports over a log-spaced gradT range at fixed T (both phases)."""
    met = M.MetInput(phase_mode="both", mode="homogeneous",
                     dt_micro=60.0, cell_volume=1.0e6)
    runner = M.MetNucleationRunner(met)
    p_v = _pv_supersaturated(T, P)
    gs = np.logspace(0, 4, ng)  # 1 .. 1e4 K/m
    out = []
    for g in gs:
        reps = runner.evaluate_point(T, P, p_v, grad_T=float(g))
        out.append((float(g), reps))
    return out


def sweep_T(P=70000.0, nT=18):
    """Reports over a temperature range (both phases), one list per phase."""
    met = M.MetInput(phase_mode="both", mode="homogeneous",
                     dt_micro=60.0, cell_volume=1.0e6)
    runner = M.MetNucleationRunner(met)
    Ts = np.linspace(236.0, 272.0, nT)
    reports = []
    for T in Ts:
        p_v = _pv_supersaturated(float(T), P)
        reps = runner.evaluate_point(float(T), P, p_v)
        reports.append(reps)
    return reports


def main():
    plot = M.MetNucleationPlotter(OUTDIR)

    # 1) P_eq,shift surface (both phases) -- core models, reduced grid for runtime
    print("Generating P_eq,shift surfaces ...")
    for ph in (M.PHASE_LIQUID, M.PHASE_ICE):
        path = plot.plot_peq_shift_surface(phase=ph, T_range=(240.0, 285.0),
                                           nT=12, g_range=(1.0, 1.0e4), ng=12)
        print(f"  -> {path}")

    # 2) Gibbs-Thomson & critical radii vs gradT (both phases)
    print("Sweeping gradT ...")
    reports_by_gradT = sweep_gradT()
    for ph in (M.PHASE_LIQUID, M.PHASE_ICE):
        path = plot.plot_gibbs_thomson_and_radii(reports_by_gradT, phase=ph)
        print(f"  -> {path}")

    # 3) Free-energy decomposition vs radius (liquid, homogeneous theta=pi)
    print("Generating free-energy decomposition ...")
    cfg = M.AtmosphericInput(phase_mode="liquid")
    model = M.LiquidNucleationModel(cfg)
    T0, P0 = 260.0, 70000.0
    p_v0 = _pv_supersaturated(T0, P0)
    path = plot.plot_free_energy(model, T0, P0, p_v0, theta=math.pi, n=40)
    print(f"  -> {path}")

    # 4) Nucleation rate vs T (liquid & ice)
    print("Sweeping T ...")
    reports_T = sweep_T()
    path = plot.plot_rates(reports_T, reports_T)
    print(f"  -> {path}")

    # 5) Vertical profile (reuse the sweep_T reports vs a synthetic z)
    z = np.linspace(500.0, 7000.0, len(reports_T))
    path = plot.plot_vertical_profile(reports_T, z)
    print(f"  -> {path}")

    # 6) Favourability bars from one representative report (liquid, mid-sweep)
    rep_liq = None
    for reps in reports_T:
        r = reps.get(M.PHASE_LIQUID)
        if r and r.status == "ok":
            rep_liq = r
            break
    if rep_liq is not None:
        path = plot.plot_favorability_bars(rep_liq)
        print(f"  -> {path}")

    print("\nAll figures written to", OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())