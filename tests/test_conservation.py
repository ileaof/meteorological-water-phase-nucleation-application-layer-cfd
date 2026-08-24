"""Milestone 4 tests: conservation diagnostics and the discrete equilibrium.

Covers the complete water budget (airborne + surface accumulation), the
(an)elastic mass-continuity residual, and that a short storm run keeps water,
energy and the mass constraint bounded -- i.e. the projection (not the limiters)
enforces continuity.  The exact reference-state equilibrium is tested in
test_soundings.test_reference_state_equilibrium.
"""
from __future__ import annotations

import numpy as np

from meteorological_flow import diagnostics as diag
from meteorological_flow.base_state import weisman_klemp
from meteorological_flow.config import SimulationConfig, apply_overrides
from meteorological_flow.grid import Grid
from meteorological_flow.simulation import Simulation
from meteorological_flow.state import FlowState


def _storm_cfg(dynamics="anelastic", duration=120.0):
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics=dynamics)
    cfg.domain.Lz = 16000.0
    cfg.grid.nx = cfg.grid.ny = 12
    cfg.grid.nz = 30
    cfg.time.duration = duration
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_cons"
    return cfg


def test_total_water_includes_surface_accumulation():
    g = Grid(nx=6, ny=6, nz=8, Lx=6000, Ly=6000, Lz=8000)
    st = FlowState.zeros(g)
    st.qv[:] = 1e-3
    airborne = float(st.total_water())
    assert diag.surface_water_kg(st) == 0.0
    # deposit 2 mm (=2 kg/m^2) of rain over the footprint
    st.surface_precip["rain"][:] = 2.0
    expected_surface = 2.0 * g.dx * g.dy * (g.nx * g.ny)
    assert np.isclose(diag.surface_water_kg(st), expected_surface)
    assert np.isclose(diag.total_water_kg(st), airborne + expected_surface)


def test_mass_residual_zero_at_rest():
    g = Grid(nx=6, ny=6, nz=10, Lx=6000, Ly=6000, Lz=10000)
    st = FlowState.zeros(g)                       # u=v=w=0
    rho0 = np.linspace(1.1, 0.4, g.nz)
    rwf = np.interp(g.zf, g.zc, rho0)
    assert diag.mass_continuity_residual(st, rho0, rwf)["abs_max"] == 0.0
    assert diag.mass_continuity_residual(st)["abs_max"] == 0.0


def test_storm_reports_small_mass_residual_and_water_budget():
    for dyn in ("boussinesq", "anelastic"):
        cfg = _storm_cfg(dyn)
        g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
        rep = Simulation(cfg, base=weisman_klemp(g)).run()
        c = rep["conservation"]
        # the projection enforces continuity: the normalised residual is small
        assert c["mass_continuity_residual_norm"] < 1e-2, dyn
        # water and energy stay bounded over the short run (closed-ish domain)
        assert abs(c["total_water_rel_err"]) < 1e-2, dyn
        assert abs(c["total_energy_rel_err"]) < 1e-2, dyn


def test_massflux_transport_conserves_and_positive():
    """The M5 conservative transport conserves int rho0 s exactly in a closed box
    (zero boundary flux) and preserves positivity (MUSCL/minmod, small CFL)."""
    from meteorological_flow import advection as adv
    g = Grid(nx=8, ny=8, nz=10, Lx=8000, Ly=8000, Lz=10000)
    rng = np.random.default_rng(0)
    uf = rng.standard_normal(g.u_shape); uf[0] = uf[-1] = 0.0      # closed walls
    vf = rng.standard_normal(g.v_shape); vf[:, 0] = vf[:, -1] = 0.0
    wf = rng.standard_normal(g.w_shape); wf[:, :, 0] = wf[:, :, -1] = 0.0
    rho_c = np.linspace(1.1, 0.4, g.nz)
    rho_wf = np.interp(g.zf, g.zc, rho_c)
    s = np.abs(rng.standard_normal(g.center_shape)) + 0.1          # positive
    m0 = float((rho_c[None, None, :] * s).sum())
    for order in (1, 2):
        s2 = adv.advect_center_massflux(s, uf, vf, wf, g, 1e-3, rho_c, rho_wf, order=order)
        m1 = float((rho_c[None, None, :] * s2).sum())
        assert abs(m1 - m0) < 1e-9 * abs(m0), order          # int rho0 s conserved
        assert np.min(s2) >= -1e-12, order                   # positivity (small dt)


def test_rho0_weighted_water_budget():
    """M6: total_water_kg weights the airborne inventory by the density profile,
    so it measures int rho0 q (what the anelastic transport conserves), not the
    unweighted sum."""
    g = Grid(nx=5, ny=5, nz=12, Lx=5000, Ly=5000, Lz=12000)
    st = FlowState.zeros(g)
    st.qv[:] = 1e-2
    st.qr[:, :, 3] = 2e-3
    rho0 = np.linspace(1.15, 0.35, g.nz)                       # decreasing with height
    q = st.qv + st.ql + st.qi + st.qr + st.qs + st.qg + st.qh
    expected = float((rho0[None, None, :] * q).sum() * g.cell_vol)
    assert np.isclose(diag.total_water_kg(st, rho0), expected)
    # unweighted (rho=None) is the plain mixing-ratio inventory and differs
    assert np.isclose(diag.total_water_kg(st), float(q.sum() * g.cell_vol))
    assert not np.isclose(diag.total_water_kg(st, rho0), diag.total_water_kg(st))
    # a scalar weighting also works (Boussinesq)
    assert np.isclose(diag.total_water_kg(st, 1.2), float((1.2 * q).sum() * g.cell_vol))


def test_vertical_stretching():
    """Vertical grid stretching: uniform (z_stretch=1) is unchanged; a stretched
    anelastic storm still enforces the mass constraint (tiny residual from the
    variable-dz projection), conserves water and stays stable."""
    gu = Grid(nx=4, ny=4, nz=10, Lx=4000, Ly=4000, Lz=10000)
    assert gu.stretched is False and np.allclose(gu.dz_c, gu.dz)
    gs = Grid(nx=4, ny=4, nz=10, Lx=4000, Ly=4000, Lz=10000, z_stretch=1.2)
    assert gs.stretched and gs.dz_c[0] < gs.dz_c[-1] and np.isclose(gs.dz_c.sum(), 10000.0)
    cfg = _storm_cfg("anelastic", duration=200.0)
    cfg.grid.z_stretch = 1.1
    g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz, z_stretch=1.1)
    rep = Simulation(cfg, base=weisman_klemp(g)).run()
    c = rep["conservation"]
    assert c["mass_continuity_residual_norm"] < 1e-2          # variable-dz projection OK
    assert abs(c["total_water_rel_err"]) < 1e-2               # conserves
    assert rep["max_cfl"] < 1.0 and np.isfinite(rep["final_stats"]["wmax"])


def test_periodic_shear_storm():
    """Mean-wind/shear ingestion: periodic lateral BCs give a divergence-free
    projection, ingest the environmental wind (persisting via the perturbation-
    relaxed Rayleigh drag), conserve water and stay stable."""
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics="anelastic", periodic=True)
    assert cfg.boundaries.x_west == "periodic" and cfg.boundaries.y == "periodic"
    cfg.domain.Lz = 16000.0
    cfg.grid.nx = cfg.grid.ny = 12
    cfg.grid.nz = 30
    cfg.time.duration = 200.0
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_periodic"
    g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=16000.0, periodic=True)
    assert g.periodic
    sim = Simulation(cfg, base=weisman_klemp(g, u_shear=20.0))
    assert float(sim.state.u.max()) > 15.0             # mean wind ingested at init (u0 top ~20)
    rep = sim.run()
    c = rep["conservation"]
    assert c["mass_continuity_residual_norm"] < 1e-2   # periodic projection divergence-free
    assert abs(c["total_water_rel_err"]) < 1e-2        # conserves
    assert rep["max_cfl"] < 1.0 and np.isfinite(rep["final_stats"]["wmax"])
    assert float(sim.state.u.max()) > 10.0             # mean wind persists (not damped away)


def test_water_measure_label_present():
    cfg = _storm_cfg("anelastic", duration=30.0)
    g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
    rep = Simulation(cfg, base=weisman_klemp(g)).run()
    assert "surface accumulation" in rep["conservation"]["water_measure"]
