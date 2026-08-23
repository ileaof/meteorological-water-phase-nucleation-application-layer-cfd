"""Report the reference-sounding diagnostics (Milestone 2).

Builds an idealised Weisman-Klemp sounding (or reads a CSV/NetCDF sounding),
reports CAPE, CIN, LCL, LFC, EL, freezing level, the Brunt-Vaisala profile and
the 0-6 km bulk shear, and (optionally) verifies that the reference state stays
in equilibrium (no spurious storm) when run with no perturbation.

    python examples/sounding_diagnostics.py                       # Weisman-Klemp
    python examples/sounding_diagnostics.py --qv-sfc 0.016 --shear 25
    python examples/sounding_diagnostics.py --csv my_sounding.csv
    python examples/sounding_diagnostics.py --equilibrium         # + equilibrium check
    python examples/sounding_diagnostics.py --to-csv out.csv      # write the profile
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow import soundings                       # noqa: E402
from meteorological_flow.base_state import sounding_diagnostics, weisman_klemp  # noqa: E402
from meteorological_flow.config import SimulationConfig, apply_overrides        # noqa: E402
from meteorological_flow.grid import Grid                       # noqa: E402
from meteorological_flow.simulation import Simulation           # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="read a sounding CSV instead of Weisman-Klemp")
    ap.add_argument("--netcdf", default=None, help="read a sounding NetCDF")
    ap.add_argument("--qv-sfc", type=float, default=0.014, dest="qv_sfc")
    ap.add_argument("--shear", type=float, default=0.0, help="WK unidirectional shear [m/s]")
    ap.add_argument("--Lz", type=float, default=18000.0)
    ap.add_argument("--Nz", type=int, default=90)
    ap.add_argument("--to-csv", default=None, dest="to_csv")
    ap.add_argument("--equilibrium", action="store_true")
    args = ap.parse_args(argv)

    g = Grid(nx=8, ny=8, nz=args.Nz, Lx=16000, Ly=16000, Lz=args.Lz)
    if args.csv:
        base = soundings.from_csv(g, args.csv); label = "CSV %s" % args.csv
    elif args.netcdf:
        base = soundings.from_netcdf(g, args.netcdf); label = "NetCDF %s" % args.netcdf
    else:
        base = weisman_klemp(g, qv_sfc=args.qv_sfc, u_shear=args.shear)
        label = "Weisman-Klemp (qv_sfc=%.3f, shear=%.0f m/s)" % (args.qv_sfc, args.shear)

    d = sounding_diagnostics(base)
    print("=== sounding diagnostics: %s ===" % label)
    print("  surface     : T=%.1f K  p=%.0f Pa  qv=%.4f kg/kg" % (base.T0[0], base.p0[0], base.qv0[0]))
    _f = lambda x, u="m": ("%.0f %s" % (x, u)) if x is not None else "n/a"
    print("  CAPE        : %.0f J/kg      CIN: %.0f J/kg" % (d["CAPE_J_kg"], d["CIN_J_kg"]))
    print("  LCL / LFC / EL : %s / %s / %s" % (_f(d["LCL_m"]), _f(d["LFC_m"]), _f(d["EL_m"])))
    print("  freezing level : %s        parcel w_max (sqrt 2CAPE): %.0f m/s"
          % (_f(d["freezing_level_m"]), d["w_max_parcel_m_s"]))
    print("  Brunt-Vaisala  : mean N^2 = %.2e s^-2   0-6 km shear: %.1f m/s"
          % (d["N2_mean_1_s2"], d["shear_0_6km_m_s"]))

    if args.to_csv:
        soundings.to_csv(base, args.to_csv)
        print("  wrote sounding -> %s" % args.to_csv)

    if args.equilibrium:
        cfg = apply_overrides(SimulationConfig(), storm_scale=True)
        cfg.domain.Lz = args.Lz
        cfg.grid.nx = cfg.grid.ny = 12
        cfg.grid.nz = min(args.Nz, 40)
        cfg.physics.bubble_dtheta = 0.0
        cfg.nucleation.stage = "none"
        cfg.time.duration = 200.0
        cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
        cfg.output.outdir = "outputs/_sounding_eq"
        ge = Grid(nx=12, ny=12, nz=cfg.grid.nz, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
        rep = Simulation(cfg, base=weisman_klemp(ge, qv_sfc=args.qv_sfc, u_shear=args.shear)
                         if not (args.csv or args.netcdf) else base).run()
        print("  equilibrium : after 200 s, max|u|=%.3f m/s (residual imbalance; "
              "should be << storm 10-40 m/s)" % rep["final_stats"]["umax"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
