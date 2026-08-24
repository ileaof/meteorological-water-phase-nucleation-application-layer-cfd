"""Milestone 2 tests: reference sounding, CAPE/CIN/LCL/LFC/EL diagnostics,
sounding I/O, and the reference-state equilibrium verification.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from meteorological_flow import soundings
from meteorological_flow.base_state import (
    sounding_diagnostics, weisman_klemp,
)
from meteorological_flow.config import SimulationConfig, apply_overrides
from meteorological_flow.grid import Grid
from meteorological_flow.simulation import Simulation


def _deep_grid(nz=90, Lz=18000.0):
    return Grid(nx=4, ny=4, nz=nz, Lx=16000, Ly=16000, Lz=Lz)


def test_weisman_klemp_cape_is_physical():
    d = sounding_diagnostics(weisman_klemp(_deep_grid()))
    # the WK sounding is ~2000-2500 J/kg -- a recognised idealised benchmark
    assert 1500.0 < d["CAPE_J_kg"] < 3500.0
    assert d["CIN_J_kg"] < 0.0
    assert d["w_max_parcel_m_s"] == np.sqrt(2.0 * d["CAPE_J_kg"])


def test_cape_robust_across_vertical_resolution():
    """CAPE/EL must not collapse at certain vertical resolutions (regression: the
    LFC==EL bug that zeroed CAPE when the layer straddling the LFC had a negative
    mean buoyancy).  Across coarse-to-fine Nz, CAPE stays physical and the EL near
    the tropopause."""
    for nz in (30, 45, 60, 90, 120):
        d = sounding_diagnostics(weisman_klemp(_deep_grid(nz=nz, Lz=18000.0)))
        assert 500.0 < d["CAPE_J_kg"] < 4000.0, nz          # never spuriously 0
        assert 10000.0 < d["EL_m"] < 14000.0, nz            # EL near the tropopause
        assert d["LFC_m"] < d["EL_m"], nz                   # never LFC == EL


def test_sounding_level_ordering():
    d = sounding_diagnostics(weisman_klemp(_deep_grid()))
    assert d["LCL_m"] < d["LFC_m"] < d["EL_m"]
    assert 3000.0 < d["freezing_level_m"] < 5500.0     # ~4-5 km for T_sfc~300 K
    assert 10000.0 < d["EL_m"] < 14000.0               # near the tropopause


def test_cape_increases_with_moisture():
    g = _deep_grid()
    dry = sounding_diagnostics(weisman_klemp(g, qv_sfc=0.011))["CAPE_J_kg"]
    moist = sounding_diagnostics(weisman_klemp(g, qv_sfc=0.016))["CAPE_J_kg"]
    assert moist > dry


def test_shear_diagnostic():
    d = sounding_diagnostics(weisman_klemp(_deep_grid(), u_shear=25.0, u_half=3000.0))
    assert 18.0 < d["shear_0_6km_m_s"] < 26.0          # ~tanh-ramped to ~25 m/s


def test_brunt_vaisala_positive_in_troposphere():
    d = sounding_diagnostics(weisman_klemp(_deep_grid()))
    N2 = np.asarray(d["N2_1_s2"])
    assert np.all(N2[:-2] > 0.0)                        # stably stratified theta0


def test_sounding_csv_roundtrip():
    g = _deep_grid(nz=40, Lz=16000.0)
    base = weisman_klemp(g, u_shear=15.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "snd.csv")
        soundings.to_csv(base, p)
        back = soundings.from_csv(g, p)
    assert np.allclose(back.T0, base.T0, rtol=1e-3)
    assert np.allclose(back.qv0, base.qv0, atol=1e-5)
    assert np.allclose(back.u0, base.u0, atol=1e-3)


def test_from_arrays_with_RH():
    g = _deep_grid(nz=30, Lz=12000.0)
    z = np.array([0.0, 1000.0, 5000.0, 10000.0, 12000.0])
    T = np.array([300.0, 293.0, 265.0, 230.0, 220.0])
    p = np.array([1.0e5, 8.9e4, 5.4e4, 2.6e4, 1.9e4])
    RH = np.array([80.0, 70.0, 40.0, 20.0, 15.0])
    base = soundings.from_arrays(g, z, T, p=p, RH=RH)
    assert np.all(base.qv0 >= 0.0) and base.qv0[0] > base.qv0[-1]
    assert base.theta0[-1] > base.theta0[0]            # stably stratified


def test_reference_state_equilibrium():
    """The stratified reference state (no perturbation, no microphysics) must be a
    true DISCRETE equilibrium: it stays exactly at rest.  This holds because M4
    (a) diffuses only the perturbation theta'=theta-theta0 and (b) preserves
    theta0(z)/qv0(z) at the z-boundaries (zero-gradient on the perturbation).
    Without those, diffusing / zero-gradient-copying the curved base profile
    injects a phantom theta' -> spurious buoyancy -> a ~1 m/s circulation even at
    rest, which the limiters would then have to suppress.  Exact rest confirms the
    limiters are NOT propping up the solution."""
    for dynamics in ("boussinesq", "anelastic"):
        cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics=dynamics)
        cfg.domain.Lz = 16000.0
        cfg.grid.nx = cfg.grid.ny = 12
        cfg.grid.nz = 30
        cfg.physics.bubble_dtheta = 0.0                 # NO perturbation
        cfg.nucleation.stage = "none"                   # dynamics-only
        cfg.time.duration = 120.0
        cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
        cfg.output.outdir = "outputs/_test_eq"
        g = Grid(nx=12, ny=12, nz=30, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)
        sim = Simulation(cfg, base=weisman_klemp(g))
        assert float(np.max(np.abs(sim.state.velocity_magnitude_center()))) == 0.0
        rep = sim.run()
        assert rep["final_stats"]["umax"] < 1e-6, dynamics    # exact discrete rest
        assert rep["final_stats"]["wmax"] < 1e-6, dynamics
