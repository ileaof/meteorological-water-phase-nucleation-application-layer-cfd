#!/usr/bin/env python3
"""test_boundary_conditions.py -- BC application, warm<cold density, grid-refine.

Covers spec tests:
  6   warm/moist inflow is less dense than the cold/dry inflow (buoyancy sign is
      physically correct: warm air rises).
  18  BC application is consistent under grid refinement (the imposed inflow
      scalars/velocities converge to the configured values as dx->0).
"""
import numpy as np

from meteorological_flow import boundary_conditions as bc
from meteorological_flow import thermodynamics as th
from meteorological_flow.config import SimulationConfig
from meteorological_flow.grid import Grid
from meteorological_flow.state import FlowState


def _cfg():
    cfg = SimulationConfig()
    return cfg


def test_06_warm_inflow_less_dense_than_cold():
    """[exp] The warm/moist west inflow is less dense than the cold/dry east
    inflow, so buoyancy drives the warm parcel upward (the physical sign that
    the mixing zone produces rising plumes)."""
    cfg = _cfg()
    P0 = cfg.physics.P0
    rho_warm = th.density_moist(P0, cfg.boundaries.warm_inflow.T,
                                th.q_v_from_p_v(0.9 * th.psat_water(cfg.boundaries.warm_inflow.T), P0))
    rho_cold = th.density_moist(P0, cfg.boundaries.cold_inflow.T,
                                th.q_v_from_p_v(0.3 * th.psat_water(cfg.boundaries.cold_inflow.T), P0))
    assert rho_warm < rho_cold, f"warm {rho_warm} not less dense than cold {rho_cold}"


def test_06b_inflow_state_consistent():
    """[ref] inflow_state returns (theta, q_v) consistent with the configured
    T/RH at the scenario P0 (round-trip T(theta, P0_REF, P0) ~ T)."""
    cfg = _cfg()
    P0 = cfg.physics.P0
    th_w, qv_w = bc.inflow_state(cfg.boundaries.warm_inflow, P0)
    # round-trip: T = theta * (P0/P0_REF)^(R/cp)
    T_back = float(th.T_from_theta(th_w, P0, th.P0_REF))
    assert abs(T_back - cfg.boundaries.warm_inflow.T) < 1e-6, "theta round-trip failed"
    # q_v from p_v round-trip
    pv = th.p_v_from_q_v(qv_w, P0)
    assert abs(pv - 0.9 * th.psat_water(cfg.boundaries.warm_inflow.T)) / pv < 1e-6


def test_bc_velocity_inflow_applied():
    """[num] apply_velocity_bcs imposes the configured inflow u on the west
    face and the (negative) cold inflow u on the east face."""
    cfg = _cfg()
    g = Grid(nx=10, ny=10, nz=10, Lx=100.0, Ly=100.0, Lz=100.0)
    st = FlowState.zeros(g)
    bc.apply_velocity_bcs(st, g, cfg)
    assert np.allclose(st.u[0, :, :], cfg.boundaries.warm_inflow.u)
    assert np.allclose(st.u[-1, :, :], -cfg.boundaries.cold_inflow.u)
    # mass-balanced top outflow w_top = (u_warm+u_cold)*Lz/Lx
    w_top = (cfg.boundaries.warm_inflow.u + cfg.boundaries.cold_inflow.u) * g.Lz / g.Lx
    assert np.allclose(st.w[:, :, -1], w_top)
    # bottom free-slip
    assert np.allclose(st.w[:, :, 0], 0.0)


def test_bc_scalar_inflow_applied():
    """[num] apply_scalar_bcs fixes the inflow scalars (theta, q_v) at the
    west/east inflow cells and zeroes the hydrometeors there."""
    cfg = _cfg()
    g = Grid(nx=10, ny=10, nz=10, Lx=100.0, Ly=100.0, Lz=100.0)
    st = FlowState.zeros(g)
    st.ql[:] = 1.0; st.qi[:] = 1.0
    bc.apply_scalar_bcs(st, g, cfg)
    th_w, qv_w = bc.inflow_state(cfg.boundaries.warm_inflow, cfg.physics.P0)
    assert np.allclose(st.theta[0, :, :], th_w)
    assert np.allclose(st.qv[0, :, :], qv_w)
    assert np.allclose(st.ql[0, :, :], 0.0) and np.allclose(st.qi[0, :, :], 0.0)


def test_bc_mass_balance_top_outflow():
    """[num] The mass-balanced top outflow makes the net boundary flux of the
    imposed velocity field zero (mean divergence of the boundary-driven field
    -> 0), which is the compatibility the all-Neumann projection needs."""
    cfg = _cfg()
    g = Grid(nx=20, ny=20, nz=20, Lx=100.0, Ly=100.0, Lz=100.0)
    st = FlowState.zeros(g)
    bc.apply_velocity_bcs(st, g, cfg)
    # net x-flux in = (u_warm - |u_cold|) * Ly*Lz (both into domain)
    # net x-flux in = (u_warm + u_cold) * Ly * Lz
    u_in = (cfg.boundaries.warm_inflow.u + cfg.boundaries.cold_inflow.u) * g.Ly * g.Lz
    # net z-flux out = w_top * Lx * Ly
    w_top = (cfg.boundaries.warm_inflow.u + cfg.boundaries.cold_inflow.u) * g.Lz / g.Lx
    w_out = w_top * g.Lx * g.Ly
    assert abs(u_in - w_out) < 1e-9, "top outflow does not balance the two inflows"


def test_18_bc_consistent_under_grid_refinement():
    """[reg] Under grid refinement the inflow-imposed cell values converge to
    the exact configured scalars (they are Dirichlet, so exact at any dx; the
    test guards against an accidental off-by-half-cell interpolation)."""
    cfg = _cfg()
    P0 = cfg.physics.P0
    th_w, qv_w = bc.inflow_state(cfg.boundaries.warm_inflow, P0)
    for n in (10, 20, 40):
        g = Grid(nx=n, ny=n, nz=n, Lx=100.0, Ly=100.0, Lz=100.0)
        st = FlowState.zeros(g)
        bc.apply_scalar_bcs(st, g, cfg)
        assert np.allclose(st.theta[0, :, :], th_w), f"n={n} west theta drift"
        assert np.allclose(st.qv[0, :, :], qv_w), f"n={n} west qv drift"