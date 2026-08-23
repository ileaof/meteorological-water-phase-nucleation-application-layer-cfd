"""CFD geometry / grid / CFL / memory / pressure audit tests.

Covers the resizing requirements: domain & cell volumes, cell count, cubic and
anisotropic spacing, non-cubic domains, cell-vs-node convention, the anisotropic
CFL and diffusive limit, pressure drop vs gradient, J*V_cell*dt, memory
estimation, invalid-config rejection, and backward compatibility.
"""
from __future__ import annotations

import numpy as np
import pytest

from meteorological_flow.config import (
    PRESETS, SimulationConfig, apply_overrides, estimate_memory_gb, from_dict,
    geometry, validate,
)
from meteorological_flow.grid import Grid
from meteorological_flow.simulation import Simulation


def _cfg(**over):
    return apply_overrides(SimulationConfig(), **over)


# --- 1-3: volumes and cell count -----------------------------------------
def test_domain_volume():
    gm = geometry(_cfg(Lx=1000, Ly=1000, Lz=1000, Nx=50, Ny=50, Nz=50))
    assert gm["domain_volume_m3"] == pytest.approx(1.0e9)


def test_cell_volume():
    gm = geometry(_cfg(Lx=1000, Ly=1000, Lz=1000, Nx=50, Ny=50, Nz=50))
    assert gm["cell_volume_m3"] == pytest.approx(8000.0)   # 20*20*20


def test_number_of_cells():
    gm = geometry(_cfg(Nx=50, Ny=50, Nz=125))
    assert gm["n_cells"] == 50 * 50 * 125


# --- 4-5: cubic and anisotropic spacing ----------------------------------
def test_cubic_spacing():
    gm = geometry(_cfg(Lx=1000, Ly=1000, Lz=1000, Nx=50, Ny=50, Nz=50))
    assert gm["cubic"] and gm["dx_m"] == gm["dy_m"] == gm["dz_m"] == 20.0


def test_anisotropic_spacing():
    gm = geometry(_cfg(Lx=1000, Ly=1000, Lz=1000, Nx=100, Ny=50, Nz=25))
    assert gm["dx_m"] == pytest.approx(10.0)
    assert gm["dy_m"] == pytest.approx(20.0)
    assert gm["dz_m"] == pytest.approx(40.0)
    assert not gm["cubic"]


# --- 6: non-cubic domain runs & stays divergence-free --------------------
def test_noncubic_domain_projection_divergence_free():
    cfg = _cfg(Lx=200, Ly=100, Lz=400, Nx=16, Ny=10, Nz=24, duration=6.0)
    cfg.nucleation.stage = "none"
    cfg.output.format = ["json"]; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_noncubic"
    sim = Simulation(cfg); sim.run()
    div = sim.grid.divergence(sim.state.u, sim.state.v, sim.state.w)
    assert float(np.max(np.abs(div))) < 1e-8


# --- 7: cell-centred convention (dx = Lx/Nx, Nx CELLS) -------------------
def test_cell_centered_convention():
    g = Grid(nx=50, ny=40, nz=25, Lx=1000, Ly=800, Lz=500)
    assert g.dx == pytest.approx(20.0) and g.xc[0] == pytest.approx(10.0)  # (i+0.5)dx
    assert g.cell_vol == pytest.approx(g.dx * g.dy * g.dz)
    # cell volume is NOT the domain volume unless there is a single cell
    assert g.cell_vol != pytest.approx(g.Lx * g.Ly * g.Lz)


# --- 8: anisotropic advective CFL ----------------------------------------
def test_anisotropic_cfl():
    cfg = _cfg(Lx=100, Ly=100, Lz=200, Nx=10, Ny=10, Nz=20, cfl=0.4, dt_max=100.0)
    cfg.nucleation.stage = "none"
    sim = Simulation(cfg)
    sim.state.u[:] = 10.0; sim.state.v[:] = 0.0; sim.state.w[:] = 0.0
    dt = sim._dt()
    inv = 10.0 / sim.grid.dx           # only u nonzero
    assert dt == pytest.approx(0.4 / (1.25 * inv), rel=1e-6)
    assert sim._dt_limiter == "advective"


# --- 9: diffusive limit dominates when viscosity is large ----------------
def test_diffusive_limit():
    cfg = _cfg(Nx=10, Ny=10, Nz=10, dt_max=1e9)
    cfg.flow.nu = cfg.flow.kappa = 1.0e4
    cfg.nucleation.stage = "none"
    sim = Simulation(cfg)
    sim._dt()
    assert sim._dt_limiter == "diffusive"


# --- 10: pressure drop vs gradient ---------------------------------------
def test_pressure_drop_vs_gradient():
    c1 = _cfg(Lx=1000, pressure_gradient=0.1)
    assert c1.flow.p_drop == pytest.approx(100.0)         # gradient * Lx
    c2 = _cfg(Lx=100, pressure_gradient=0.1)
    assert c2.flow.p_drop == pytest.approx(10.0)          # same gradient, 10x shorter
    with pytest.raises(ValueError):
        _cfg(pressure_drop=50.0, pressure_gradient=1.0)   # mutually exclusive


# --- 11: N_expected = J * V_cell * dt uses the LOCAL cell volume ----------
def test_expected_events_uses_cell_volume():
    from precip_microphysics import nucleation_source as ns
    from precip_microphysics.config import MicrophysicsConfig
    from precip_microphysics.state import MicrophysicsState
    import precip_microphysics.thermo as th
    T, P = 260.0, 70000.0
    qv = float(th.qsat_water(T, P) * 1.2)
    st = MicrophysicsState(T=T, P=P, rho=0.9, qv=qv)
    J, dt = 1.0e6, 60.0
    _, d1 = ns.embryo_source(st, MicrophysicsConfig(), dt, 8000.0, J_liquid=J)
    _, d2 = ns.embryo_source(st, MicrophysicsConfig(), dt, 16000.0, J_liquid=J)
    # N_expected scales linearly with the cell volume (J*dt*V) -> 2x for 2x volume
    assert float(np.max(d2["N_expected_liquid"])) == pytest.approx(
        2.0 * float(np.max(d1["N_expected_liquid"])), rel=1e-9)


# --- 21: invalid configs are rejected ------------------------------------
def test_reject_invalid_configs():
    with pytest.raises(AssertionError):
        _cfg(Nx=2)                                        # below stencil minimum
    with pytest.raises(AssertionError):
        validate(_cfg_bad_precision())


def _cfg_bad_precision():
    c = SimulationConfig(); c.physics.precision = "float16"
    return c


# --- 22: memory estimate scales and honours precision --------------------
def test_memory_estimate():
    c64 = _cfg(Nx=50, Ny=50, Nz=50)
    c32 = _cfg(Nx=50, Ny=50, Nz=50, float32=True)
    assert estimate_memory_gb(c64) > 0
    assert estimate_memory_gb(c32) == pytest.approx(estimate_memory_gb(c64) / 2.0)
    # 100^3 is ~8x the cells of 50^3
    assert estimate_memory_gb(_cfg(Nx=100, Ny=100, Nz=100)) == pytest.approx(
        8.0 * estimate_memory_gb(c64), rel=1e-6)


# --- 23: backward compatibility (old-style config, no new args) ----------
def test_backward_compatible_config():
    c = apply_overrides(SimulationConfig(), grid_resolution=20)   # legacy path
    assert c.grid.nx == c.grid.ny == c.grid.nz == 20
    c2 = from_dict({"grid": {"nx": 20, "ny": 20, "nz": 20}})      # old YAML
    assert c2.physics.precision == "float64" and c2.physics.scenario == "mixing_chamber"


# --- presets -------------------------------------------------------------
def test_presets_defined():
    assert set(PRESETS) == {"fast", "light", "recommended", "advanced", "convective-column",
                            "storm-quick", "storm", "storm-refined", "storm-fine", "storm-hires"}
    c = _cfg(preset="convective-column")
    gm = geometry(c)
    assert gm["n_cells"] == 50 * 50 * 125 and gm["Lz_m"] == 5000.0
    # the storm-* presets carry the storm marker and are deep (EL-containing)
    for p in ("storm-quick", "storm", "storm-refined", "storm-fine", "storm-hires"):
        assert PRESETS[p].get("storm") is True and PRESETS[p]["Lz"] >= 16000.0


# --- 17: small 25^3 smoke run records geometry ---------------------------
def test_25cubed_smoke_records_geometry():
    cfg = _cfg(preset="fast", Lx=200, Ly=200, Lz=200, Nx=25, Ny=25, Nz=25,
               duration=3.0)
    cfg.nucleation.stage = "none"
    cfg.output.format = ["json"]; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_25"
    report = Simulation(cfg).run()
    assert report["geometry"]["n_cells"] == 25 ** 3
    assert report["memory_estimate_gb"] > 0
    assert report["cfl_limiter_last"] in ("advective", "diffusive", "dt_max")
    assert report["max_cfl"] < 1.0
