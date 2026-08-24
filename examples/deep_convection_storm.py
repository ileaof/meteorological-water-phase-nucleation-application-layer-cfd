"""Theory-complete qualitative deep-convection storm.

This is the storm example assembled from the pieces that make a deep-convection
run *qualitatively* defensible, rather than a shallow Boussinesq demonstration:

  * a real conditionally-unstable reference sounding -- the Weisman-Klemp (1982)
    analytic profile, the standard idealised deep-convection benchmark, with a
    realistic CAPE ~ 2000 J/kg, a tropopause and an overlying stratosphere (M2);
  * the ANELASTIC dynamical core -- rho0(z) reference density with
    div(rho0 u)=0, so the deep-column mass expansion (updrafts amplifying as they
    rise into thinner air) is represented, which the constant-density Boussinesq
    core structurally misses (M3);
  * a DEEP domain (default 18 km) that actually contains the equilibrium level,
    so the updraft is not truncated below its natural anvil height;
  * two-way bulk microphysics -- cloud/rain/ice/snow/graupel/hail form from the
    flow's own supersaturation, latent heat feeds back on the buoyancy, and the
    precipitating species sediment to the surface;
  * a warm-bubble trigger sized to overcome the sounding's CIN.

Before running it prints the ENVIRONMENT (CAPE/CIN/LCL/LFC/EL/shear) and the
parcel-theory expectation w_max = sqrt(2*CAPE); afterwards it compares the
simulated updraft and cloud top against those theoretical anchors.

What it is NOT: a quantitative forecast.  The grid is coarse (~0.5-1 km), so
entrainment, the cold pool and (with shear) supercell rotation are only crudely
resolved; per-step latent heating, velocity and temperature are bounded as
documented stability safeguards.  It is an idealised, qualitatively-defensible
deep-convection run whose bulk numbers (updraft magnitude, cloud-top height,
mixed-phase partition, precipitation sign) are physically sensible, not exact.

    python examples/deep_convection_storm.py                       # defaults
    python examples/deep_convection_storm.py --shear 20 --duration 1200
    python examples/deep_convection_storm.py --dynamics boussinesq  # contrast
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


def _fmt(x, unit="m"):
    return ("%.0f %s" % (x, unit)) if x is not None else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=24, help="horizontal cells nx=ny")
    ap.add_argument("--Nz", type=int, default=45, help="vertical cells")
    ap.add_argument("--Lx", type=float, default=20000.0, help="horizontal extent [m] (=Ly)")
    ap.add_argument("--Lz", type=float, default=18000.0, help="domain top [m] (>= EL + overshoot)")
    ap.add_argument("--duration", type=float, default=600.0, help="simulated seconds")
    ap.add_argument("--bubble", type=float, default=3.0, help="warm-bubble dtheta [K] (> CIN barrier)")
    ap.add_argument("--shear", type=float, default=0.0,
                    help="0-6 km unidirectional wind shear [m/s]. With --periodic the "
                         "mean wind is ingested and TILTS/organises the updraft; without "
                         "it (closed walls) --shear only sets the reported sounding shear.")
    ap.add_argument("--periodic", action="store_true",
                    help="periodic lateral BCs: ingest the sheared mean wind so it "
                         "organises the storm (pair with --shear).")
    ap.add_argument("--qv-sfc", type=float, default=0.014, dest="qv_sfc",
                    help="surface vapour mixing ratio [kg/kg] (moisture -> CAPE)")
    ap.add_argument("--dynamics", choices=("anelastic", "boussinesq"), default="anelastic")
    ap.add_argument("--output", default="outputs/deep_convection_storm")
    args = ap.parse_args(argv)

    # --- the reference environment (M2 sounding) ---
    g = Grid(nx=args.N, ny=args.N, nz=args.Nz, Lx=args.Lx, Ly=args.Lx, Lz=args.Lz)
    base = weisman_klemp(g, qv_sfc=args.qv_sfc, u_shear=args.shear)
    d = sounding_diagnostics(base)
    w_parcel = d["w_max_parcel_m_s"]

    print("=== deep-convection storm: environment (Weisman-Klemp sounding) ===")
    print("  domain      : %.0f x %.0f x %.0f m   grid %d x %d x %d  (dx=%.0f m, dz=%.0f m)"
          % (args.Lx, args.Lx, args.Lz, args.N, args.N, args.Nz, g.dx, g.dz))
    print("  CAPE / CIN  : %.0f / %.0f J/kg" % (d["CAPE_J_kg"], d["CIN_J_kg"]))
    print("  LCL/LFC/EL  : %s / %s / %s" % (_fmt(d["LCL_m"]), _fmt(d["LFC_m"]), _fmt(d["EL_m"])))
    print("  freezing lvl: %s        0-6 km shear: %.1f m/s%s"
          % (_fmt(d["freezing_level_m"]), d["shear_0_6km_m_s"],
             ("  (INGESTED: mean wind organises the storm)" if args.periodic
              else "  (sounding only; add --periodic to ingest it)") if args.shear else ""))
    print("  parcel theory: w_max = sqrt(2 CAPE) = %.0f m/s  (thermodynamic ceiling;"
          % w_parcel)
    print("                 real updrafts reach ~40-60%% of this after entrainment,")
    print("                 water loading and perturbation-pressure drag)")
    if args.Lz < (d["EL_m"] or 0) + 1500.0:
        print("  ! warning: domain top (%.0f m) is close to/below EL+overshoot; "
              "raise --Lz for a clean anvil" % args.Lz)

    # --- configure the storm run (M3 anelastic core + two-way microphysics) ---
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics=args.dynamics,
                          periodic=args.periodic)
    cfg.domain.Lx = cfg.domain.Ly = args.Lx
    cfg.domain.Lz = args.Lz
    cfg.grid.nx = cfg.grid.ny = args.N
    cfg.grid.nz = args.Nz
    cfg.physics.bubble_dtheta = args.bubble
    cfg.time.duration = args.duration
    cfg.output.outdir = args.output
    cfg.output.format = ["json", "csv"]
    cfg.output.figures = []
    cfg.output.restart = False
    cfg.output.interval_steps = 50

    print("\n=== running %s core, %.0f s (two-way microphysics) ===" % (args.dynamics, args.duration))
    t0 = time.perf_counter()

    def _prog(t, dur, step):
        el = time.perf_counter() - t0
        eta = el / max(t, 1e-9) * (dur - t)
        sys.stdout.write("\r  t=%6.1f/%.0f s  step %-5d  %.0fs elapsed  ETA %.0fs   "
                         % (t, dur, step, el, eta))
        sys.stdout.flush()

    sim = Simulation(cfg, base=base)
    assert sim.do_microphysics and sim.dynamics == args.dynamics
    rep = sim.run(progress=_prog)
    st = sim.state
    print()

    # --- results vs theory ---
    wmax = rep["final_stats"]["wmax"]
    frac = wmax / w_parcel if w_parcel > 0 else float("nan")
    # cloud top: highest level holding resolved condensate
    cond = st.ql + st.qi + st.qr + st.qs + st.qg + st.qh
    col = cond.max(axis=(0, 1))
    thr = 1e-5
    ktop = np.max(np.where(col > thr)[0]) if np.any(col > thr) else -1
    cloud_top = g.zc[ktop] if ktop >= 0 else 0.0

    print("\n=== results (qualitative) ===")
    print("  steps=%d  final t=%.0f s  wall=%.0f s  max CFL=%.2f  core=%s"
          % (rep["n_steps"], rep["final_time"], time.perf_counter() - t0,
             rep["max_cfl"], rep["dynamics"]))
    print("  updraft     : max w = %.1f m/s  = %.0f%% of the parcel ceiling %.0f m/s"
          % (wmax, 100 * frac, w_parcel))
    print("  cloud top   : %s   (vs equilibrium level %s)" % (_fmt(cloud_top), _fmt(d["EL_m"])))
    print("  T range     : %.1f .. %.1f K   max S_w=%.3f  S_i=%.3f"
          % (rep["final_stats"]["T_min"], rep["final_stats"]["T_max"],
             float(np.max(st.S_w)), float(np.max(st.S_i))))
    print("  condensate (max mixing ratio, kg/kg):")
    for nm, lab in (("ql", "cloud"), ("qi", "ice"), ("qr", "rain"),
                    ("qs", "snow"), ("qg", "graupel"), ("qh", "hail")):
        print("      %-8s %.2e" % (lab, float(np.max(getattr(st, nm)))))
    prec = rep.get("surface_precip_mm", {})
    print("  surface precip (domain-mean): total %.3e mm  (rain %.2e, graupel %.2e, hail %.2e)"
          % (prec.get("total_mm", 0.0), prec.get("rain", 0.0),
             prec.get("graupel", 0.0), prec.get("hail", 0.0)))

    # --- qualitative interpretation ---
    print("\n=== interpretation ===")
    ok_w = 0.15 < frac < 0.9
    ok_top = d["EL_m"] and (0.6 * d["EL_m"] < cloud_top < 1.3 * d["EL_m"])
    print("  [%s] updraft is a physically sensible fraction of the parcel ceiling"
          % ("OK" if ok_w else "--"))
    print("  [%s] cloud top is in the neighbourhood of the equilibrium level"
          % ("OK" if ok_top else "--"))
    print("  Note: qualitative deep-convection demonstration (coarse grid; "
          "entrainment / cold pool / rotation only crudely resolved).")
    print("        For a longer, moister storm increase --duration and --qv-sfc; "
          "add --periodic --shear 20 to ingest the mean wind and tilt/organise the updraft.")
    print("\nSummary -> %s/summary.json" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
