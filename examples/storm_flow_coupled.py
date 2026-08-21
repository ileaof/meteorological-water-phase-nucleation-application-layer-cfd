"""Two-way coupled run: the 3D fluid flow drives the bulk microphysics.

This is the Increment-2 coupling — the fluid flow **does** run here. The
``meteorological_flow`` 3D Boussinesq solver advances the resolved circulation
(warm/moist vs cold/dry inflow, buoyancy, projection), and each step the
``precip_microphysics`` scheme is applied to the whole grid: cloud/ice/rain/
snow/graupel/hail form from the flow's own supersaturation, the latent heat feeds
back into the transported potential temperature, and the precipitating
hydrometeors are transported and sediment to the surface.

It is demonstration-scale (a ~1 km chamber, seconds of simulated time), so the
surface accumulation is small — the point is that the flow and the microphysics
are genuinely coupled two-way, not that this reproduces a ~100 mm storm (which
needs a km-scale, long-lived storm circulation).

    python examples/storm_flow_coupled.py [--grid N] [--duration S] [--json OUT]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow.config import SimulationConfig, apply_overrides   # noqa: E402
from meteorological_flow.simulation import Simulation         # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=16, help="nx=ny (nz is 1.5x)")
    ap.add_argument("--duration", type=float, default=45.0,
                    help="simulated seconds (per-step microphysics; a minute of "
                         "sim time takes a few minutes of wall clock)")
    ap.add_argument("--storm-scale", action="store_true",
                    help="km-scale deep-convection storm (stratified sounding + "
                         "warm-bubble trigger) instead of the shallow chamber")
    ap.add_argument("--json", default="outputs/flow_coupled/summary.json")
    args = ap.parse_args(argv)

    if args.storm_scale:
        cfg = apply_overrides(SimulationConfig(), storm_scale=True)
        if args.duration == 45.0:                 # bump the default for a storm
            cfg.time.duration = 1200.0
        else:
            cfg.time.duration = args.duration
    else:
        cfg = SimulationConfig()
        cfg.grid.nx = cfg.grid.ny = args.grid
        cfg.grid.nz = int(args.grid * 1.5)
        cfg.domain.Lz = 1500.0                    # taller chamber for sedimentation
        cfg.time.duration = args.duration
        cfg.nucleation.stage = "hydrometeor"      # <-- two-way coupling
    cfg.output.outdir = os.path.dirname(args.json) or "outputs/flow_coupled"
    cfg.output.format = ["json", "csv"]
    cfg.output.figures = []
    cfg.output.restart = False
    cfg.output.interval_steps = 25

    print("Running the 3D flow solver with two-way microphysics coupling ...")
    print(f"  grid = {cfg.grid.nx}x{cfg.grid.ny}x{cfg.grid.nz}, "
          f"domain = {cfg.domain.Lx:.0f}x{cfg.domain.Ly:.0f}x{cfg.domain.Lz:.0f} m, "
          f"duration = {cfg.time.duration:.0f} s")
    sim = Simulation(cfg)
    assert sim.do_microphysics, "microphysics coupling not active"
    report = sim.run()
    st = sim.state

    def mx(name):
        return float(np.max(getattr(st, name)))

    print("\n=== the fluid flow ran ===")
    print(f"  steps = {report['n_steps']}, final t = {report['final_time']:.1f} s, "
          f"max CFL = {report['max_cfl']:.3f}")
    print(f"  T range = {report['final_stats']['T_min']:.1f} .. "
          f"{report['final_stats']['T_max']:.1f} K, "
          f"max |w| = {report['final_stats']['wmax']:.2f} m/s")
    print(f"  max S_w = {float(np.max(st.S_w)):.3f}, max S_i = {float(np.max(st.S_i)):.3f}")

    print("\n=== microphysics formed by the flow (max mixing ratio, kg/kg) ===")
    for name, lab in (("ql", "cloud liquid"), ("qi", "cloud ice"), ("qr", "rain"),
                      ("qs", "snow"), ("qg", "graupel"), ("qh", "hail")):
        print(f"  {lab:<13} {mx(name):.3e}")

    print("\n=== surface precipitation (domain-mean) ===")
    prec = report.get("surface_precip_mm", {})
    for c in ("rain", "snow", "graupel", "hail"):
        print(f"  {c:<9} {prec.get(c, 0.0)*1e3:.4f} um   ({prec.get(c, 0.0):.3e} mm)")
    print(f"  TOTAL     {prec.get('total_mm', 0.0):.3e} mm")

    print("\nNote: demonstration-scale (~1 km chamber, seconds); the coupling is "
          "two-way (flow -> microphysics -> latent heat + sedimentation -> flow),\n"
          "but the accumulation is small. A ~100 mm event needs a km-scale, "
          "long-lived storm circulation (see the 0-D storm example).")
    print(f"\nSummary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
