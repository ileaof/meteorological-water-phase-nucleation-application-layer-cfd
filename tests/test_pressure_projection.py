#!/usr/bin/env python3
"""test_pressure_projection.py -- Chorin projection (spec tests 4, 5).

  4  the projection reduces the divergence of an arbitrary face velocity to
     (near) machine zero (mass-balanced BCs -> mean(div)=0 -> div-free).
  5  a hydrostatically-balanced w-field (dw/dz = buoyancy-free) is left
     unchanged in its divergence-free part by the projection.
"""
import numpy as np

from meteorological_flow.config import SimulationConfig
from meteorological_flow.grid import Grid
from meteorological_flow.pressure_solver import PressureSolver
from meteorological_flow.state import FlowState


def _cfg():
    cfg = SimulationConfig()
    cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = 10
    return cfg


def _grid(cfg):
    return Grid(nx=cfg.grid.nx, ny=cfg.grid.ny, nz=cfg.grid.nz,
                Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)


def test_04_projection_kills_divergence():
    """[num] Projecting an arbitrary divergent face velocity yields
    div(u_new) ~ 0 (the all-Neumann + mass-balanced compatibility makes the
    projected field divergence-free, not merely constant-divergence)."""
    cfg = _cfg()
    g = _grid(cfg)
    rng = np.random.default_rng(42)
    st = FlowState.zeros(g)
    # random divergent velocity (mass-balanced so mean(div)=0 is achievable)
    st.u[:] = rng.standard_normal(g.u_shape)
    st.v[:] = rng.standard_normal(g.v_shape)
    st.w[:] = rng.standard_normal(g.w_shape)
    # impose zero net flux at boundaries so the Neumann system is compatible
    st.u[0, :, :] = 0.0; st.u[-1, :, :] = 0.0
    st.v[:, 0, :] = 0.0; st.v[:, -1, :] = 0.0
    st.w[:, :, 0] = 0.0; st.w[:, :, -1] = 0.0
    rho0 = 1.0
    solver = PressureSolver(g, method="direct")
    div0 = g.divergence(st.u, st.v, st.w)
    res, it = solver.project(st, 0.1, rho0)
    div1 = g.divergence(st.u, st.v, st.w)
    assert abs(div0).max() > 1e-3, "test setup not divergent"
    assert abs(div1).max() < 1e-9, f"div not removed: max={abs(div1).max()}"
    # the solver residual is small (regularised SPD system)
    assert res < 1e-8, f"projection residual large: {res}"


def test_04b_projection_preserves_boundary_velocity():
    """[num] With Neumann pressure BCs the projection does not change the
    boundary face velocities (dp/dn=0 -> no correction at the boundary)."""
    cfg = _cfg()
    g = _grid(cfg)
    st = FlowState.zeros(g)
    rng = np.random.default_rng(7)
    st.u[1:-1, :, :] = rng.standard_normal((g.nx - 1, g.ny, g.nz))
    st.u[0, :, :] = 5.0          # fixed inflow value
    st.u[-1, :, :] = -5.0        # fixed outflow value
    st.v[:] = 0.0; st.w[:] = 0.0
    solver = PressureSolver(g, method="direct")
    u0_west = st.u[0, 0, 0]; u0_east = st.u[-1, 0, 0]
    solver.project(st, 0.1, 1.0)
    # boundary face values are preserved (Neumann grad=0 => no correction)
    assert np.allclose(st.u[0, :, :], u0_west), "west boundary u changed"
    assert np.allclose(st.u[-1, :, :], u0_east), "east boundary u changed"


def test_05_projection_divfree_field_unchanged():
    """[num] A divergence-free velocity is left unchanged by the projection
    (div=0 => rhs=0 => p'=0 => no correction)."""
    cfg = _cfg()
    g = _grid(cfg)
    st = FlowState.zeros(g)
    # a constant translation u=(1,0,0) is divergence-free
    st.u[:] = 1.0
    solver = PressureSolver(g, method="direct")
    u_before = st.u.copy()
    solver.project(st, 0.1, 1.0)
    assert np.allclose(st.u, u_before), "div-free field altered by projection"


def test_05b_cg_method_also_removes_divergence():
    """[num] The iterative CG path (used for grids > 40^3) also reduces the
    divergence of a divergent field to the tolerance."""
    cfg = SimulationConfig()
    cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = 8
    g = _grid(cfg)
    rng = np.random.default_rng(1)
    st = FlowState.zeros(g)
    st.u[:] = rng.standard_normal(g.u_shape)
    st.v[:] = rng.standard_normal(g.v_shape)
    st.w[:] = rng.standard_normal(g.w_shape)
    st.u[0, :, :] = 0.0; st.u[-1, :, :] = 0.0
    st.v[:, 0, :] = 0.0; st.v[:, -1, :] = 0.0
    st.w[:, :, 0] = 0.0; st.w[:, :, -1] = 0.0
    solver = PressureSolver(g, method="cg", tol=1e-8, maxiter=2000)
    solver.project(st, 0.1, 1.0)
    div1 = g.divergence(st.u, st.v, st.w)
    assert abs(div1).max() < 1e-6, f"CG div not removed: {abs(div1).max()}"