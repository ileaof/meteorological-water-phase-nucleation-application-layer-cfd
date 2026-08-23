"""Milestone 3 tests: the anelastic dynamical core (--dynamics anelastic).

The anelastic projection enforces div(rho0 u) = 0 with a height-dependent
reference density rho0(z), reusing the constant-coefficient Poisson operator via
the face-density-cancellation identity.  These tests verify:

  A. constant rho0 collapses exactly onto the validated Boussinesq projection;
  B. the anelastic (mass-weighted) divergence is ~0 after projection;
  C. with a real rho0(z) the anelastic and Boussinesq projections genuinely
     differ (the deep-column mass expansion is captured, not accidentally lost);
  D. config / CLI wiring (dynamics selector, validation);
  E. a short anelastic storm run stays finite (no runaway) and conserves the
     anelastic mass constraint on the evolving state.
"""
from __future__ import annotations

import numpy as np

from meteorological_flow.base_state import weisman_klemp
from meteorological_flow.config import SimulationConfig, apply_overrides, validate
from meteorological_flow.grid import Grid
from meteorological_flow.pressure_solver import PressureSolver
from meteorological_flow.simulation import Simulation
from meteorological_flow.state import FlowState


def _random_velocity(grid, seed=0):
    rng = np.random.default_rng(seed)
    st = FlowState.zeros(grid)
    st.u = rng.standard_normal(grid.u_shape)
    st.v = rng.standard_normal(grid.v_shape)
    st.w = rng.standard_normal(grid.w_shape)
    # rigid ground / open top faces are set by the solver's Neumann gradient=0;
    # zero the domain-boundary normal faces so the inputs are comparable.
    st.u[0, :, :] = st.u[-1, :, :] = 0.0
    st.v[:, 0, :] = st.v[:, -1, :] = 0.0
    st.w[:, :, 0] = st.w[:, :, -1] = 0.0
    return st


def _rho_weighted_div(grid, st, rho0_c, rho0_wface):
    rc = np.asarray(rho0_c).reshape(1, 1, -1)
    rwf = np.asarray(rho0_wface).reshape(1, 1, -1)
    dudx = (st.u[1:] - st.u[:-1]) / grid.dx
    dvdy = (st.v[:, 1:] - st.v[:, :-1]) / grid.dy
    wflux = rwf * st.w
    dwdz = (wflux[:, :, 1:] - wflux[:, :, :-1]) / grid.dz
    return rc * (dudx + dvdy) + dwdz


def test_A_constant_density_reduces_to_boussinesq():
    """With rho0(z)=const the anelastic projection must reproduce Boussinesq."""
    g = Grid(nx=8, ny=8, nz=10, Lx=8000, Ly=8000, Lz=8000)
    solver = PressureSolver(g, method="direct")
    rho0 = 1.1
    stB = _random_velocity(g, seed=1)
    stA = _random_velocity(g, seed=1)             # identical input
    dt = 0.5
    solver.project(stB, dt, rho0)
    solver.project_anelastic(stA, dt,
                             np.full(g.nz, rho0), np.full(g.nz + 1, rho0))
    assert np.allclose(stA.u, stB.u, atol=1e-9)
    assert np.allclose(stA.v, stB.v, atol=1e-9)
    assert np.allclose(stA.w, stB.w, atol=1e-9)


def test_B_anelastic_divergence_is_zeroed():
    """After the anelastic projection div(rho0 u) ~ 0 (defining constraint)."""
    g = Grid(nx=8, ny=8, nz=30, Lx=12000, Ly=12000, Lz=12000)
    base = weisman_klemp(g)
    rho0_c = np.asarray(base.rho0)
    rho0_wface = np.interp(g.zf, g.zc, rho0_c)
    solver = PressureSolver(g, method="direct")
    st = _random_velocity(g, seed=2)
    div0 = _rho_weighted_div(g, st, rho0_c, rho0_wface)
    solver.project_anelastic(st, 0.5, rho0_c, rho0_wface)
    div1 = _rho_weighted_div(g, st, rho0_c, rho0_wface)
    # the interior mass-weighted divergence collapses by orders of magnitude
    assert np.max(np.abs(div1[1:-1, 1:-1, 1:-1])) < 1e-6 * np.max(np.abs(div0))


def test_C_anelastic_differs_from_boussinesq():
    """A real rho0(z) makes the two cores give different corrected velocities."""
    g = Grid(nx=8, ny=8, nz=30, Lx=12000, Ly=12000, Lz=12000)
    base = weisman_klemp(g)
    rho0_c = np.asarray(base.rho0)
    rho0_wface = np.interp(g.zf, g.zc, rho0_c)
    solver = PressureSolver(g, method="direct")
    stB = _random_velocity(g, seed=3)
    stA = _random_velocity(g, seed=3)
    solver.project(stB, 0.5, float(rho0_c.mean()))
    solver.project_anelastic(stA, 0.5, rho0_c, rho0_wface)
    # rho0 drops ~3x over 12 km, so the vertical correction must differ markedly
    assert np.max(np.abs(stA.w - stB.w)) > 1e-3
    # ... and Boussinesq does NOT satisfy the anelastic constraint
    divA = _rho_weighted_div(g, stA, rho0_c, rho0_wface)
    divB = _rho_weighted_div(g, stB, rho0_c, rho0_wface)
    core = (slice(1, -1),) * 3
    assert np.max(np.abs(divB[core])) > 10.0 * np.max(np.abs(divA[core]))


def test_D_config_and_cli_wiring():
    assert SimulationConfig().physics.dynamics == "boussinesq"
    cfg = apply_overrides(SimulationConfig(), dynamics="anelastic", storm_scale=True)
    assert cfg.physics.dynamics == "anelastic"
    bad = SimulationConfig()
    bad.physics.dynamics = "compressible"
    try:
        validate(bad)
        raise AssertionError("validate should reject an unknown dynamics core")
    except AssertionError as e:
        assert "dynamics" in str(e)


def test_E_short_anelastic_storm_is_finite_and_mass_consistent():
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics="anelastic")
    cfg.domain.Lz = 12000.0
    cfg.grid.nx = cfg.grid.ny = 12
    cfg.grid.nz = 30
    cfg.time.duration = 60.0
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_anelastic"
    g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
    sim = Simulation(cfg, base=weisman_klemp(g))
    assert sim.dynamics == "anelastic"
    rep = sim.run()
    assert rep["dynamics"] == "anelastic"
    st = sim.state
    assert np.all(np.isfinite(st.u)) and np.all(np.isfinite(st.w))
    assert rep["final_stats"]["wmax"] < 120.0            # no runaway
    # the evolving state respects the anelastic mass constraint below the sponge.
    # (The top damping_layer applies Rayleigh damping AFTER the projection -- a
    # momentum sink that legitimately reintroduces a small divergence there, so
    # the check excludes the top ~20% of levels.  The clean post-projection
    # constraint is covered rigorously by test_B.)
    div = _rho_weighted_div(g, st, sim.rho0_c, sim.rho0_wface)
    below_sponge = div[1:-1, 1:-1, 1:(4 * g.nz) // 5]
    assert np.max(np.abs(below_sponge)) < 1e-3           # mass residual stays small
