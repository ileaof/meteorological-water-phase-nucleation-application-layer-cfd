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


def test_water_measure_label_present():
    cfg = _storm_cfg("anelastic", duration=30.0)
    g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
    rep = Simulation(cfg, base=weisman_klemp(g)).run()
    assert "surface accumulation" in rep["conservation"]["water_measure"]
