"""Refined convection visualization ("good" criterion) with streamlines.

A higher-resolution companion to ``visualize_convection.py`` aimed at a *clean*
convection picture: it resolves the updraft with ~10-12 points across
(dx ~ 250 m) and damps the 2-delta checkerboard mode with a stronger subgrid
eddy viscosity (nu=kappa), then renders the circulation as streamlines.

Grid criteria (16 x 16 x 10 km storm domain):

    good      : Nx=Ny=64, Nz=48   -> dx~250 m, dz~208 m, ~197 k cells   (hours)
    preview   : Nx=Ny=40, Nz=36   -> dx~400 m, dz~278 m,  ~58 k cells   (~minutes)

The "good" run is expensive (a mature storm needs ~10-20 min of simulated time
on a ~200 k-cell grid = a long CPU run); use ``--preview`` for a fast look, and
launch the full run in the background / overnight.

    python examples/visualize_convection_hires.py --preview          # fast (~58k cells)
    python examples/visualize_convection_hires.py                    # good criterion
    python examples/visualize_convection_hires.py --Nx 64 --Ny 64 --Nz 48 --sgs 200 --duration 1200

Equivalent CLI (flow solver only, no plotting):

    python -m meteorological_flow.cli --storm-scale --Nx 64 --Ny 64 --Nz 48 \
        --sgs 200 --duration 1200 --output outputs/flow_hires
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow.config import (          # noqa: E402
    SimulationConfig, apply_overrides, format_geometry,
)
from meteorological_flow.simulation import Simulation   # noqa: E402

# reuse the plotting from the sibling example (keep a single renderer)
_spec = importlib.util.spec_from_file_location(
    "visualize_convection", os.path.join(os.path.dirname(__file__), "visualize_convection.py"))
_vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vc)
plot_convection = _vc.plot_convection


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true",
                    help="fast reduced run (40x40x36) instead of the good 64x64x48")
    ap.add_argument("--Nx", type=int, default=None)
    ap.add_argument("--Ny", type=int, default=None)
    ap.add_argument("--Nz", type=int, default=None)
    ap.add_argument("--sgs", type=float, default=200.0,
                    help="subgrid eddy viscosity/diffusivity [m^2/s] (damps 2-delta noise)")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default="outputs/convection_hires")
    ap.add_argument("--vectors", action="store_true",
                    help="use quiver arrows instead of streamlines")
    args = ap.parse_args(argv)

    if args.preview:
        Nx, Ny, Nz = 40, 40, 36
        duration = args.duration if args.duration is not None else 400.0
        args.out = args.out.replace("hires", "hires_preview")
    else:
        Nx, Ny, Nz = 64, 64, 48                     # "good" criterion
        duration = args.duration if args.duration is not None else 1200.0
    if args.Nx:
        Nx = args.Nx
    if args.Ny:
        Ny = args.Ny
    if args.Nz:
        Nz = args.Nz

    cfg = apply_overrides(SimulationConfig(), storm_scale=True,
                          Nx=Nx, Ny=Ny, Nz=Nz, sgs=args.sgs)
    cfg.time.duration = duration
    cfg.output.outdir = args.out
    cfg.output.format = ["netcdf", "json"]
    cfg.output.figures = []
    cfg.output.restart = False
    cfg.output.interval_steps = 999999

    print("=== refined convection (%s criterion) ===" % ("preview" if args.preview else "good"))
    print(format_geometry(cfg))
    print("  SGS        : nu=kappa=%.0f m^2/s (damps 2-delta checkerboard)" % cfg.flow.nu)
    print("Running ... (this is a long CPU run at the good criterion)")

    sim = Simulation(cfg)
    sim.run()
    st = sim.state
    wc = 0.5 * (st.w[:, :, :-1] + st.w[:, :, 1:])
    print("  max |w| = %.1f m/s   max cloud = %.2e kg/kg   max q_r = %.2e kg/kg" % (
        float(np.max(np.abs(wc))),
        float(np.max(np.asarray(st.ql) + np.asarray(st.qi))),
        float(np.max(st.qr))))

    files = plot_convection(sim, args.out, streamlines=not args.vectors)
    print("Figures:")
    for f in files:
        print("  " + f)
    print("NetCDF (ncview/ParaView/xarray): " + os.path.join(args.out, "flow.nc"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
