#!/usr/bin/env python3
"""test_scalar_conservation.py -- scalar conservation, positivity, convergence.

Covers spec tests:
  7   total water (q_v + q_l + q_i) is conserved in a CLOSED domain (div-free
      velocity tangent to all boundaries -> zero net boundary flux) to
      discretization error.
  8   the (thermal + kinetic + potential) energy budget bookkeeping closes
      at the initial instant (dimensionally consistent, J-scale).
  10  q_v stays non-negative under advection (positivity, upwind, CFL<=1).
  17  the upwind advection error converges at 1st order as the GRID is refined
      at fixed CFL (the dominant upwind error is spatial diffusion ~ O(dx)).
"""
import numpy as np

from meteorological_flow import advection as adv
from meteorological_flow import diagnostics as diag
from meteorological_flow.config import SimulationConfig
from meteorological_flow.grid import Grid
from meteorological_flow.state import FlowState


def _cfg(n=12):
    cfg = SimulationConfig()
    cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = n
    return cfg


def _grid(n, L=100.0):
    return Grid(nx=n, ny=n, nz=n, Lx=L, Ly=L, Lz=L)


def _closed_swirl_cell(g, U0=1.0):
    """Cell-centre velocities of a 2-D (x,y) swirling flow that VANISH at every
    boundary cell centre: Uc[i=0,nx-1]=0, Vc[j=0,ny-1]=0, Wc=0.  Because the v1
    advection reconstructs face velocities from cell-centre values
    (Uf[0]=Uc[0], etc.), vanishing boundary cell-centre velocities make ALL
    boundary fluxes zero -> the domain is CLOSED and the scalar total is
    conserved exactly under flux-form advection.  (Returned as Uc,Vc,Wc to be
    passed straight to advect_center, bypassing cell_velocity.)"""
    nx, ny, nz = g.nx, g.ny, g.nz
    I = np.arange(nx); J = np.arange(ny)
    sx = np.sin(np.pi * I / (nx - 1))           # 0 at i=0 and i=nx-1
    sy = np.sin(np.pi * J / (ny - 1))           # 0 at j=0 and j=ny-1
    cx = np.cos(np.pi * I / (nx - 1))
    cy = np.cos(np.pi * J / (ny - 1))
    Uc = U0 * (sx[:, None, None] * cy[None, :, None]) * np.ones((nx, ny, nz))
    Vc = U0 * (cx[:, None, None] * sy[None, :, None]) * np.ones((nx, ny, nz))
    Wc = np.zeros((nx, ny, nz))
    return Uc, Vc, Wc


def _phys_state(g, rng):
    """A state with physical T (~280 K) so diagnose() does not hit Psat(T=0)."""
    st = FlowState.zeros(g)
    st.theta = 280.0 + 5.0 * rng.random(g.center_shape)
    st.qv = 0.004 + 0.002 * rng.random(g.center_shape)
    st.ql = 0.0005 * rng.random(g.center_shape)
    st.qi = 0.0003 * rng.random(g.center_shape)
    return st


def test_07_total_water_conserved_closed_domain():
    """[num] In a closed domain (swirling flow vanishing at all boundary cell
    centres -> zero boundary flux under the v1 face reconstruction) the domain
    integral of q_v + q_l + q_i is conserved under advection to roundoff."""
    g = _grid(24, 100.0)
    rng = np.random.default_rng(3)
    st = _phys_state(g, rng)
    Uc, Vc, Wc = _closed_swirl_cell(g, U0=1.0)
    tw0 = st.total_water()
    dx = g.dx
    umax = max(abs(Uc).max(), abs(Vc).max(), abs(Wc).max())
    dt = 0.4 * dx / max(umax, 1e-12)         # CFL<=0.4
    for _ in range(30):
        st.qv = adv.advect_center(st.qv, Uc, Vc, Wc, g, dt, order=1)
        st.ql = adv.advect_center(st.ql, Uc, Vc, Wc, g, dt, order=1)
        st.qi = adv.advect_center(st.qi, Uc, Vc, Wc, g, dt, order=1)
    tw1 = st.total_water()
    assert abs(tw1 - tw0) / tw0 < 1e-10, f"total water not conserved: d={tw1-tw0}"


def test_08_energy_budget_closed_form():
    """[math] The diagnostic energy budget closes: total = KE + PE + thermal,
    each term positive / dimensionally sane, and the relative bookkeeping error
    is exactly zero at the initial instant (by construction)."""
    cfg = _cfg(10)
    g = _grid(10, 100.0)
    rng = np.random.default_rng(5)
    st = _phys_state(g, rng)
    st.u[:] = 1.0
    st.diagnose(cfg)
    rho0 = 1.0
    ib = diag.initial_budgets(st, rho0)
    cb = diag.conservation_budgets(st, ib, rho0)
    assert abs(cb["total_water_rel_err"]) < 1e-12
    assert abs(cb["total_energy_rel_err"]) < 1e-12
    assert cb["thermal_energy_J"] > 0 and cb["kinetic_energy_J"] > 0
    assert abs(cb["total_energy_J"] - (cb["kinetic_energy_J"]
                + cb["potential_energy_J"] + cb["thermal_energy_J"])) < 1e-6


def test_10_qv_positivity_under_advection():
    """[num] q_v stays non-negative under upwind advection with CFL<=1 and a
    divergence-free velocity."""
    g = _grid(32, 100.0)
    st = FlowState.zeros(g)
    Uc, Vc, Wc = _closed_swirl_cell(g, U0=2.0)
    X = g.xc.reshape(-1, 1, 1)
    st.qv = 0.01 * np.exp(-((X - 50.0) ** 2) / 50.0) * np.ones(g.center_shape)
    umax = max(abs(Uc).max(), abs(Vc).max(), abs(Wc).max())
    dt = 0.9 * g.dx / max(umax, 1e-12)
    for _ in range(30):
        st.qv = adv.advect_center(st.qv, Uc, Vc, Wc, g, dt, order=1)
    assert (st.qv >= -1e-14).all(), "q_v went negative under upwind"


def test_17_advection_error_converges_with_grid():
    """[num] Halving dx (at fixed CFL) halves the upwind advection ERROR
    INTEGRAL of a smooth pulse over a fixed physical time (1st-order spatial
    convergence -- the dominant upwind error is numerical diffusion ~ O(dx)).
    The error is the L1 integral = sum|numered-analytic| * cell_vol (the raw
    sum scales as 1/dx and so does NOT converge; the integral does)."""
    L = 64.0
    T_end = 6.0
    errs = []
    for n in (64, 128, 256):
        g = Grid(nx=n, ny=8, nz=8, Lx=L, Ly=8.0, Lz=8.0)
        st = FlowState.zeros(g)
        st.u[:] = 1.0
        X = g.xc.reshape(-1, 1, 1)
        sig2 = 16.0
        s0 = np.exp(-((X - 16.0) ** 2) / sig2) * np.ones(g.center_shape)
        Uc, Vc, Wc = adv.cell_velocity(st, g)
        dt = 0.5 * g.dx / 1.0          # fixed CFL=0.5
        nstep = round(T_end / dt)
        s = s0.copy()
        for _ in range(nstep):
            s = adv.advect_center(s, Uc, Vc, Wc, g, dt, order=1)
        analytic = np.exp(-((X - (16.0 + T_end)) ** 2) / sig2) * np.ones(g.center_shape)
        errs.append(np.abs(s - analytic).sum() * g.cell_vol)
    # successive ratios ~2 (1st order): coarse/medium and medium/fine both > 1.4
    r1 = errs[0] / max(errs[1], 1e-30)
    r2 = errs[1] / max(errs[2], 1e-30)
    assert r1 > 1.4 and r2 > 1.4, \
        f"error integral did not halve with grid refine: {errs} ratios={r1},{r2}"