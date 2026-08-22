"""Configure the CFD domain, grid, CFL and pressure forcing (new input params).

Demonstrates the resizing interface added in the CFD audit: domain dimensions
(Lx, Ly, Lz), grid (Nx, Ny, Nz), CFL / dt_max, pressure drop vs gradient, mesh
presets and precision -- via the Python API (``apply_overrides``) and the
derived-geometry helpers.  It prints, for each configuration, the derived
spacings, the cell and domain volumes, the cell count and the memory estimate;
with ``--run`` it advances a short case and reports the CFL limiter and the
post-projection divergence.

Equivalent command-line runs (the CLI mirrors every option):

    # recommended scientific mesh, explicit dimensions
    python -m meteorological_flow.cli --Lx 1000 --Ly 1000 --Lz 1000 \
        --Nx 50 --Ny 50 --Nz 50 --duration 1200 --cfl 0.4 --output outputs/flow_rec

    # non-cubic convective column, gradient-controlled forcing + two-way microphysics
    python -m meteorological_flow.cli --Lx 2000 --Ly 2000 --Lz 5000 \
        --Nx 50 --Ny 50 --Nz 125 --pressure-gradient 0.02 --two-way-coupling \
        --duration 1800 --output outputs/flow_col

    # named preset + memory guard + float32 performance mode
    python -m meteorological_flow.cli --preset advanced --float32 \
        --max-memory-gb 8 --dry-run

Usage:
    python examples/cfd_domain_configuration.py            # print geometries
    python examples/cfd_domain_configuration.py --run      # + short runs (metrics)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow.config import (          # noqa: E402
    SimulationConfig, apply_overrides, format_geometry, geometry,
)
from meteorological_flow.simulation import Simulation   # noqa: E402


# a few illustrative configurations exercising the new parameters
CONFIGS = {
    "explicit cubic (recommended, drop)": dict(
        Lx=1000, Ly=1000, Lz=1000, Nx=50, Ny=50, Nz=50, cfl=0.4,
        pressure_drop=60.0),
    "explicit NON-CUBIC (gradient-fixed)": dict(
        Lx=2000, Ly=2000, Lz=5000, Nx=40, Ny=40, Nz=100, cfl=0.4,
        pressure_gradient=0.02),
    "preset 'convective-column'": dict(preset="convective-column", cfl=0.4),
    "preset 'fast' + float32": dict(preset="fast", float32=True),
}


def _short_run(cfg):
    """Advance a small pure-flow case; return (wall, steps, max_cfl, limiter, divmax)."""
    c = cfg
    c.nucleation.stage = "none"
    c.time.duration = 6.0
    c.output.format = []; c.output.figures = []; c.output.restart = False
    c.output.outdir = "outputs/_cfd_demo"
    t0 = time.perf_counter()
    sim = Simulation(c)
    rep = sim.run()
    wall = time.perf_counter() - t0
    div = sim.grid.divergence(sim.state.u, sim.state.v, sim.state.w)
    return wall, rep["n_steps"], rep["max_cfl"], rep.get("cfl_limiter_last"), \
        float(np.max(np.abs(div)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="also advance a short case per configuration and report metrics")
    args = ap.parse_args(argv)

    for label, over in CONFIGS.items():
        cfg = apply_overrides(SimulationConfig(), **over)
        gm = geometry(cfg)
        print("=" * 70)
        print(label)
        print("-" * 70)
        print(format_geometry(cfg))
        pg = cfg.physics.pressure_gradient
        print("  forcing    : p_drop=%.3g Pa%s" % (
            cfg.flow.p_drop,
            "  (gradient %.4g Pa/m * Lx)" % pg if pg is not None else "  (explicit drop)"))
        if args.run:
            wall, steps, mcfl, lim, div = _short_run(cfg)
            print("  short run  : %.1fs, %d steps, max_CFL=%.3f (limiter=%s), divmax=%.1e"
                  % (wall, steps, mcfl, lim, div))

    print("=" * 70)
    print("Cell-centred convention: dx=Lx/Nx (Nx CELLS). V_cell = dx*dy*dz is the")
    print("LOCAL cell volume used in N_expected = J*V_cell*dt -- never V_domain.")
    print("Run with --run to see the anisotropic CFL and post-projection divergence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
