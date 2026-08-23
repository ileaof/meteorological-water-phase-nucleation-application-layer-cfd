"""Visualise the convective flow from a storm-scale coupled run.

Runs the km-scale deep-convection scenario (or any config) and renders the
classic convection view: a vertical x-z slice through the domain centre showing
the vertical velocity ``w`` (updraft/downdraft) in colour, the cloud field
(``q_l + q_i``) as contours, and the ``(u, w)`` circulation as vectors; plus a
horizontal x-y slice of ``w`` at mid-height showing the updraft cell(s).

    python examples/visualize_convection.py                       # quiver arrows
    python examples/visualize_convection.py --streamlines         # streamlines instead
    python examples/visualize_convection.py --duration 900 --grid 20 --streamlines
    python examples/visualize_convection.py --out outputs/conv    # PNGs -> outputs/conv/

    # match a deep anelastic storm run (same grid/Lz/core):
    python examples/visualize_convection.py --streamlines --grid 24 --Nz 45 \
        --Lz 18000 --dynamics anelastic --duration 600 --out outputs/storm_anelastic_viz

The figures are written as PNGs (headless Agg backend); the run also writes
``flow.nc`` which can be animated in ncview / ParaView / xarray.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meteorological_flow.config import SimulationConfig, apply_overrides   # noqa: E402
from meteorological_flow.simulation import Simulation                      # noqa: E402


def _centres(st):
    uc = 0.5 * (st.u[:-1, :, :] + st.u[1:, :, :])
    vc = 0.5 * (st.v[:, :-1, :] + st.v[:, 1:, :])
    wc = 0.5 * (st.w[:, :, :-1] + st.w[:, :, 1:])
    cloud = np.asarray(st.ql) + np.asarray(st.qi)
    return uc, vc, wc, cloud


def plot_convection(sim, outdir: str, streamlines: bool = False) -> list:
    """Render the convection slices.  ``streamlines=True`` draws the circulation
    as continuous streamlines (streamplot) instead of quiver arrows."""
    os.makedirs(outdir, exist_ok=True)
    g = sim.grid
    st = sim.state
    uc, vc, wc, cloud = _centres(st)
    jmid, kmid = g.ny // 2, g.nz // 2
    tag = "stream" if streamlines else "vectors"
    files = []

    # --- vertical x-z slice: w (colour) + cloud (contours) + circulation ---
    w_xz = wc[:, jmid, :]
    u_xz = uc[:, jmid, :]
    c_xz = cloud[:, jmid, :] * 1000.0                      # g/kg
    X, Z = np.meshgrid(g.xc, g.zc, indexing="ij")
    wmax = max(float(np.max(np.abs(w_xz))), 1e-6)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(X, Z, w_xz, shading="auto", cmap="RdBu_r",
                       vmin=-wmax, vmax=wmax)
    fig.colorbar(im, ax=ax, label="w [m/s]  (red = updraft)")
    cmax = float(np.nanmax(c_xz))
    if cmax > 1e-2:
        thr = max(0.05, 0.1 * cmax)                       # cloud edge [g/kg]
        ax.contourf(X, Z, c_xz, levels=[thr, cmax + 1e-9],
                    colors=["#2ca02c"], alpha=0.22)        # shaded cloud region
        ax.contour(X, Z, c_xz, levels=[thr], colors="green", linewidths=1.2)
    spd = float(np.max(np.hypot(u_xz, w_xz)))
    if spd > 1e-6:
        if streamlines:
            # streamplot needs 1-D increasing x,z and (nz,nx) velocity arrays
            ax.streamplot(g.xc, g.zc, u_xz.T, w_xz.T, color="k",
                          density=1.4, linewidth=0.7, arrowsize=0.9)
        else:
            sx, sz = max(1, g.nx // 16), max(1, g.nz // 16)
            ax.quiver(X[::sx, ::sz], Z[::sx, ::sz], u_xz[::sx, ::sz], w_xz[::sx, ::sz],
                      color="k", alpha=0.6, scale=max(spd * 20.0, 1e-6))
    ax.set_title("Convection (x-z, y=mid, t=%.0f s): w + cloud + %s"
                 % (sim.t, "streamlines" if streamlines else "(u,w)"))
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    p = os.path.join(outdir, "convection_xz_%s.png" % tag)
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); files.append(p)

    # --- horizontal x-y slice of w at mid-height (+ optional streamlines) ---
    w_xy = wc[:, :, kmid]
    u_xy, v_xy = uc[:, :, kmid], vc[:, :, kmid]
    Xh, Yh = np.meshgrid(g.xc, g.yc, indexing="ij")
    wmax_h = max(float(np.max(np.abs(w_xy))), 1e-6)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(Xh, Yh, w_xy, shading="auto", cmap="RdBu_r",
                       vmin=-wmax_h, vmax=wmax_h)
    fig.colorbar(im, ax=ax, label="w [m/s]")
    if streamlines and float(np.max(np.hypot(u_xy, v_xy))) > 1e-6:
        ax.streamplot(g.xc, g.yc, u_xy.T, v_xy.T, color="k",
                      density=1.4, linewidth=0.7, arrowsize=0.9)
    ax.set_title("Horizontal flow (x-y, z=%.0f m, t=%.0f s)%s"
                 % (g.zc[kmid], sim.t, " + streamlines" if streamlines else ""))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    p = os.path.join(outdir, "convection_xy_%s.png" % tag)
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); files.append(p)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=400.0)
    ap.add_argument("--grid", type=int, default=16, help="nx=ny (nz scaled for the column)")
    ap.add_argument("--Nz", type=int, default=None, help="vertical cells (overrides the scaled default)")
    ap.add_argument("--Lz", type=float, default=None, help="domain top [m] (overrides the preset)")
    ap.add_argument("--dynamics", choices=("anelastic", "boussinesq"), default=None,
                    help="dynamical core (match your run; default = the storm preset's boussinesq)")
    ap.add_argument("--out", default="outputs/convection_viz")
    ap.add_argument("--streamlines", action="store_true",
                    help="draw the circulation as streamlines (streamplot) instead "
                         "of quiver arrows")
    ap.add_argument("--no-storm", action="store_true",
                    help="use the shallow mixing chamber instead of the storm scenario")
    args = ap.parse_args(argv)

    if args.no_storm:
        cfg = SimulationConfig()
        cfg.grid.nx = cfg.grid.ny = args.grid; cfg.grid.nz = int(args.grid * 1.5)
        cfg.domain.Lz = 1500.0
        cfg.nucleation.stage = "hydrometeor"
    else:
        cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics=args.dynamics)
        cfg.grid.nx = cfg.grid.ny = args.grid
        cfg.grid.nz = args.Nz if args.Nz is not None else max(24, int(args.grid * 1.6))
        if args.Lz is not None:
            cfg.domain.Lz = float(args.Lz)
    cfg.time.duration = args.duration
    cfg.output.outdir = args.out
    cfg.output.format = ["netcdf", "json"]        # flow.nc for external tools
    cfg.output.figures = []                        # we render our own below
    cfg.output.restart = False
    cfg.output.interval_steps = 999999

    print("Running %s [%s core] (%dx%dx%d, Lz=%.0f m, %.0f s) ..." % (
        "storm-scale convection" if not args.no_storm else "mixing chamber",
        cfg.physics.dynamics, cfg.grid.nx, cfg.grid.ny, cfg.grid.nz,
        cfg.domain.Lz, cfg.time.duration))
    sim = Simulation(cfg)
    sim.run()
    st = sim.state
    print("  max |w| = %.1f m/s   max cloud (q_l+q_i) = %.2e kg/kg   max q_r = %.2e" % (
        float(np.max(np.abs(0.5 * (st.w[:, :, :-1] + st.w[:, :, 1:])))),
        float(np.max(np.asarray(st.ql) + np.asarray(st.qi))),
        float(np.max(st.qr))))
    files = plot_convection(sim, args.out, streamlines=args.streamlines)
    print("Figures written:")
    for f in files:
        print("  " + f)
    print("NetCDF for ncview/ParaView/xarray: " + os.path.join(args.out, "flow.nc"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
