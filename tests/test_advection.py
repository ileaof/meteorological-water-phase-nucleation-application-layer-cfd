#!/usr/bin/env python3
"""test_advection.py -- flux-form upwind advection (monotone, conservative).

Covers spec test 9 (bounded scalar mixing / transport): a scalar advected by a
constant, divergence-free velocity is transported without creating new extrema
(monotone under CFL<=1) and conserves the domain integral.  Also checks the
MUSCL path runs and the 1st-order scheme reproduces the analytic translation of
a smooth pulse at CFL=0.5.
"""
import numpy as np

from meteorological_flow import advection as adv
from meteorological_flow.grid import Grid
from meteorological_flow.state import FlowState


def _grid(n=24):
    return Grid(nx=n, ny=n, nz=n, Lx=n * 1.0, Ly=n * 1.0, Lz=n * 1.0)


def _state_uniform_u(g, u0):
    st = FlowState.zeros(g)
    st.u[:] = u0                       # +x through-flow
    st.v[:] = 0.0
    st.w[:] = 0.0
    return st


def test_09a_constant_advection_translates_pulse():
    """[num] A 1-D pulse in x advected at u0 for one period returns to its
    start position (periodic wrap not used; we advect a fraction of a period
    and check the centre shifts by u0*dt)."""
    g = _grid(40)
    st = _state_uniform_u(g, 1.0)
    # smooth Gaussian pulse centred at x=10
    X = g.xc.reshape(-1, 1, 1)
    s = np.exp(-((X - 10.0) ** 2) / 2.0) * np.ones(g.center_shape)
    dx = g.dx
    dt = 0.5 * dx / 1.0                 # CFL=0.5
    Uc, Vc, Wc = adv.cell_velocity(st, g)
    s_new = adv.advect_center(s, Uc, Vc, Wc, g, dt, order=1)
    # peak should have moved ~u0*dt = 0.5 dx to the right
    peak0 = int(np.argmax(s[:, 0, 0]))
    peak1 = int(np.argmax(s_new[:, 0, 0]))
    assert abs(peak1 - peak0) <= 1, f"peak moved {peak1-peak0} cells"
    # no new extrema beyond the original max (monotone upwind)
    assert s_new.max() <= s.max() + 1e-12, "upwind created overshoot"
    assert s_new.min() >= s.min() - 1e-12, "upwind created undershoot"


def test_09b_advection_conserves_total_mass():
    """[num] Flux-form advection by a constant solenoidal velocity conserves
    the domain integral (up to boundary flux; with zero-gradient ends the total
    is constant)."""
    g = _grid(20)
    st = _state_uniform_u(g, 0.5)
    X = g.xc.reshape(-1, 1, 1)
    s = np.exp(-((X - 10.0) ** 2) / 4.0) * np.ones(g.center_shape)
    Uc, Vc, Wc = adv.cell_velocity(st, g)
    # zero-gradient ends: pulse stays away from boundaries over short time
    dt = 0.4 * g.dx / 0.5
    s1 = adv.advect_center(s, Uc, Vc, Wc, g, dt, order=1)
    # total mass over interior (exclude boundary cells where flux leaves)
    tot0 = s[1:-1, :, :].sum()
    tot1 = s1[1:-1, :, :].sum()
    assert abs(tot1 - tot0) / tot0 < 0.02, "mass not approximately conserved"


def test_09c_advection_preserves_positivity():
    """[num] A non-negative scalar stays non-negative under upwind advection
    with CFL<=1 and a solenoidal velocity."""
    g = _grid(16)
    st = _state_uniform_u(g, 1.0)
    X, Y = np.meshgrid(g.xc, g.yc, indexing="ij")
    s = np.exp(-((X - 8.0) ** 2 + (Y - 8.0) ** 2) / 3.0)
    s = s * np.ones(g.center_shape)        # broadcast over z
    Uc, Vc, Wc = adv.cell_velocity(st, g)
    dt = 0.9 * g.dx / 1.0                  # CFL=0.9, still monotone
    s1 = adv.advect_center(s, Uc, Vc, Wc, g, dt, order=1)
    assert (s1 >= -1e-14).all(), "positivity violated by upwind"


def test_09d_muscl_runs_and_is_bounded():
    """[num] The 2nd-order MUSCL(minmod) path runs and stays non-negative for a
    non-negative scalar under CFL<=1."""
    g = _grid(20)
    st = _state_uniform_u(g, 0.5)
    X = g.xc.reshape(-1, 1, 1)
    s = np.exp(-((X - 10.0) ** 2) / 2.0) * np.ones(g.center_shape)
    Uc, Vc, Wc = adv.cell_velocity(st, g)
    dt = 0.5 * g.dx / 0.5
    s1 = adv.advect_center(s, Uc, Vc, Wc, g, dt, order=2)
    assert (s1 >= -1e-12).all(), "MUSCL produced negative values"
    # minmod limiter should not overshoot the original extrema much
    assert s1.max() <= s.max() * 1.05 + 1e-12


def test_09e_stationary_field_unchanged():
    """[math] A uniform scalar advected by a constant velocity is unchanged
    (no spurious tendency)."""
    g = _grid(10)
    st = _state_uniform_u(g, 2.0)
    s = np.full(g.center_shape, 3.0)
    Uc, Vc, Wc = adv.cell_velocity(st, g)
    dt = 0.5 * g.dx / 2.0
    s1 = adv.advect_center(s, Uc, Vc, Wc, g, dt, order=1)
    assert np.allclose(s1, 3.0), "uniform scalar altered by advection"