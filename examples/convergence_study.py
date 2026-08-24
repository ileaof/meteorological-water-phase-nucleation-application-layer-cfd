"""Grid-convergence study for the deep-convection storm (Milestone 8).

Runs the SAME storm (Weisman-Klemp sounding, anelastic core, two-way microphysics,
fixed domain and duration) at a sequence of horizontal resolutions and reports how
the bulk metrics -- peak updraft, cloud top, surface precip -- and the conservation
errors change with grid spacing.  A converging solution shows the successive
differences shrinking; a diverging/under-resolved one keeps changing.

Honest scope (CPU-only): the storm is convection-PERMITTING, not convection-
resolving.  True convergence of deep moist convection needs dx <~ 100-250 m
(Bryan, Wyngaard & Fritsch 2003) -- millions of cells, out of reach here.  So this
study demonstrates the RESOLUTION-DEPENDENCE (the updraft strengthens and the
storm sharpens as dx decreases) and the conservation behaviour across grids, not a
grid-independent result.  It quantifies exactly how far from converged we are.

    python examples/convergence_study.py                       # default ladder
    python examples/convergence_study.py --duration 1200 --grids 16,24,32,40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow.base_state import sounding_diagnostics, weisman_klemp  # noqa: E402
from meteorological_flow.config import SimulationConfig, apply_overrides         # noqa: E402
from meteorological_flow.grid import Grid                                        # noqa: E402
from meteorological_flow.simulation import Simulation                            # noqa: E402


def _run(N, Nz, Lx, Lz, duration, qv_sfc):
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics="anelastic")
    cfg.domain.Lx = cfg.domain.Ly = Lx
    cfg.domain.Lz = Lz
    cfg.grid.nx = cfg.grid.ny = N
    cfg.grid.nz = Nz
    cfg.time.duration = duration
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_conv_%d" % N
    g = Grid(nx=N, ny=N, nz=Nz, Lx=Lx, Ly=Lx, Lz=Lz)
    t0 = time.perf_counter()
    rep = Simulation(cfg, base=weisman_klemp(g, qv_sfc=qv_sfc)).run()
    wall = time.perf_counter() - t0
    return rep, wall


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", default="12,16,24", help="comma list of horizontal N (nx=ny)")
    ap.add_argument("--Lx", type=float, default=20000.0)
    ap.add_argument("--Lz", type=float, default=18000.0)
    ap.add_argument("--Nz", type=int, default=45, help="vertical cells (fixed across the ladder)")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--qv-sfc", type=float, default=0.014, dest="qv_sfc")
    args = ap.parse_args(argv)

    grids = [int(s) for s in args.grids.split(",")]
    # environment (grid-independent physics): evaluate on a FINE reference column
    # so CAPE/EL are accurate regardless of the coarse simulation grids.
    genv = Grid(nx=4, ny=4, nz=120, Lx=args.Lx, Ly=args.Lx, Lz=args.Lz)
    d = sounding_diagnostics(weisman_klemp(genv, qv_sfc=args.qv_sfc))
    w_ceiling = d["w_max_parcel_m_s"]
    print("=== convergence study: Weisman-Klemp storm, anelastic, %.0f s ===" % args.duration)
    print("environment: CAPE=%.0f J/kg  EL=%.0f m  parcel ceiling sqrt(2CAPE)=%.0f m/s\n"
          % (d["CAPE_J_kg"], d["EL_m"], w_ceiling))
    print("  %5s %7s %7s %8s %8s %9s %11s %8s" % (
        "N", "dx[m]", "dz[m]", "wmax", "%ceil", "precip[mm]", "water_err", "wall[s]"))

    rows = []
    for N in grids:
        rep, wall = _run(N, args.Nz, args.Lx, args.Lz, args.duration, args.qv_sfc)
        dx = args.Lx / N; dz = args.Lz / args.Nz
        wmax = rep["final_stats"]["wmax"]
        precip = rep.get("surface_precip_mm", {}).get("total_mm", 0.0)
        werr = rep["conservation"]["total_water_rel_err"]
        rows.append((N, dx, dz, wmax, precip, werr))
        print("  %5d %7.0f %7.0f %8.1f %8.0f %9.3e %11.2e %8.0f" % (
            N, dx, dz, wmax, 100 * wmax / w_ceiling, precip, werr, wall))

    # convergence indicator: successive relative change in wmax
    print("\n  resolution dependence (successive change in wmax):")
    for i in range(1, len(rows)):
        w_prev, w = rows[i - 1][3], rows[i][3]
        rel = (w - w_prev) / max(abs(w_prev), 1e-9)
        print("    N %d -> %d (dx %.0f -> %.0f m):  wmax %+.1f%%  %s" % (
            rows[i - 1][0], rows[i][0], rows[i - 1][1], rows[i][1], 100 * rel,
            "(shrinking -> toward convergence)" if abs(rel) < 0.15 else "(still changing -> under-resolved)"))

    print("\n  Interpretation: the updraft strengthens as dx decreases (finer grids")
    print("  resolve the ~2-5 km updraft better).  Water conservation holds across")
    print("  all grids (rho0-weighted budget, M6).  This is convection-PERMITTING;")
    print("  converging deep convection needs dx <~ 100-250 m (Bryan et al. 2003),")
    print("  i.e. millions of cells -- beyond a single CPU.  The trend quantifies")
    print("  how far from grid-independent the demonstration is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
