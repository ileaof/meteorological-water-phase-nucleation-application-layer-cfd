"""Anelastic vs Boussinesq dynamical core (Milestone 3).

Runs the SAME warm-bubble deep-convection trigger with each dynamical core and
compares the vertical structure of the updraft.  The physical signature of the
anelastic core is deep-column mass expansion: as an updraft rises into lower
reference density rho0(z), the anelastic mass constraint div(rho0 u)=0 lets the
vertical velocity amplify with height, where the constant-density Boussinesq
core cannot.

To isolate the CORE (not the microphysics), this runs dynamics-only (no latent
heat, no hydrometeors): a dry buoyant bubble.  The comparison is qualitative
(coarse grid, short time) -- it demonstrates the core is wired correctly, not a
quantitative storm.

    python examples/anelastic_vs_boussinesq.py
    python examples/anelastic_vs_boussinesq.py --duration 240 --Nz 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow.base_state import weisman_klemp                 # noqa: E402
from meteorological_flow.config import SimulationConfig, apply_overrides  # noqa: E402
from meteorological_flow.grid import Grid                                # noqa: E402
from meteorological_flow.simulation import Simulation                    # noqa: E402


def _run(dynamics, args):
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics=dynamics)
    cfg.domain.Lz = args.Lz
    cfg.grid.nx = cfg.grid.ny = args.N
    cfg.grid.nz = args.Nz
    cfg.nucleation.stage = "none"          # dynamics-only: isolate the core
    cfg.physics.bubble_dtheta = args.bubble
    cfg.time.duration = args.duration
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_anel_cmp_%s" % dynamics
    g = Grid(nx=args.N, ny=args.N, nz=args.Nz,
             Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
    sim = Simulation(cfg, base=weisman_klemp(g))
    rep = sim.run()
    st = sim.state
    # column-max updraft per level (positive w), interpolated to cell centres
    wc = 0.5 * (st.w[:, :, :-1] + st.w[:, :, 1:])         # (nx,ny,nz)
    w_up = np.maximum(wc, 0.0).max(axis=(0, 1))           # (nz,)
    rho0 = np.asarray(sim.rho0_c) if sim.rho0_c is not None \
        else np.asarray(weisman_klemp(g).rho0)
    return g.zc, w_up, rho0, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--Nz", type=int, default=40)
    ap.add_argument("--Lz", type=float, default=12000.0)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--bubble", type=float, default=4.0)
    args = ap.parse_args(argv)

    print("=== anelastic vs Boussinesq core (dry buoyant bubble) ===")
    print("grid %dx%dx%d  Lz=%.0f m  duration=%.0f s  bubble=%.1f K\n"
          % (args.N, args.N, args.Nz, args.Lz, args.duration, args.bubble))
    zc, w_b, rho0, rep_b = _run("boussinesq", args)
    _,  w_a, _,    rep_a = _run("anelastic", args)

    nz = zc.size
    eps = 1e-6

    def _ratio(wa, wb):
        return (wa / wb) if wb > eps else float("nan")

    print("  core        wmax [m/s]   height of wmax [m]")
    kb, ka = int(np.argmax(w_b)), int(np.argmax(w_a))
    print("  boussinesq  %8.3f      %8.0f" % (w_b.max(), zc[kb]))
    print("  anelastic   %8.3f      %8.0f" % (w_a.max(), zc[ka]))

    # vertical structure: how the updraft (and its ratio) grow with height
    print("\n  z [m]   rho0     w_bouss   w_anel   w_anel/w_bouss")
    for k in range(0, nz, max(1, nz // 10)):
        print("  %5.0f  %6.3f  %8.4f  %8.4f   %6.2f"
              % (zc[k], rho0[k], w_b[k], w_a[k], _ratio(w_a[k], w_b[k])))

    # signature 1: across the active column the ratio w_anel/w_bouss TRENDS
    # upward with height (the anelastic mass constraint expands the rising air
    # into thinner rho0).  Reported as the least-squares slope over the levels
    # where the Boussinesq plume is resolved, plus base-vs-top ratio.
    active = np.where(w_b > eps)[0]
    zk = zc[active]
    rk = w_a[active] / w_b[active]
    slope = float(np.polyfit(zk, rk, 1)[0]) if zk.size > 2 else float("nan")
    r_base = float(np.mean(rk[:2])); r_top = float(np.mean(rk[-2:]))
    # signature 2: penetration depth (highest level with a resolved updraft)
    thr = 0.05 * max(w_b.max(), w_a.max())
    top_b = zc[np.max(np.where(w_b > thr)[0])] if np.any(w_b > thr) else 0.0
    top_a = zc[np.max(np.where(w_a > thr)[0])] if np.any(w_a > thr) else 0.0
    print("\n  signature 1 -- updraft ratio trends up with height:")
    print("      w_anel/w_bouss: base %.2f -> plume top %.2f   (slope %+.2e /m)"
          % (r_base, r_top, slope))
    print("  signature 2 -- anelastic plume penetrates deeper:")
    print("      updraft top (w>%.3f m/s):  boussinesq %.0f m   anelastic %.0f m"
          % (thr, top_b, top_a))
    print("  rho0(surface)/rho0(top) = %.2f  (the anelastic mass-expansion scale)"
          % (rho0[0] / rho0[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
