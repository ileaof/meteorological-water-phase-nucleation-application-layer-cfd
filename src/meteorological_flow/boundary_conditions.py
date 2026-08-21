"""Boundary conditions for the 3D mixing chamber.

Reference setup (two opposing inflows + open top, mass balanced by the
projection): warm/moist air enters the west face (+x), cold/dry air enters the
east face (-x); the top is an open/outflow (zero-gradient) boundary through
which the net inflow exits; the bottom is free-slip; the lateral y walls are
free-slip (periodic y is a documented option).  A uniform pressure-gradient
body force (the controlled pressure drop) is applied to the u-momentum in
:mod:`simulation`, NOT here.

Inflow scalars (potential temperature theta, vapour q_v) are fixed from the
inflow T/RH via :mod:`thermodynamics`; q_l = q_i = 0 at inflows.
"""
from __future__ import annotations

import numpy as np

from . import thermodynamics as th
from .config import InflowConfig, SimulationConfig
from .grid import Grid
from .state import FlowState


def inflow_state(inflow: InflowConfig, P0: float):
    """Return (theta, q_v) for an inflow given its T [K], RH_water [%], P0 [Pa]."""
    pv = (inflow.RH_water / 100.0) * th.psat_water(inflow.T)
    qv = float(th.q_v_from_p_v(pv, P0))
    theta = float(th.theta_from_T(inflow.T, P0, th.P0_REF))
    return theta, qv


def apply_velocity_bcs(state: FlowState, grid: Grid, cfg: SimulationConfig) -> None:
    """Enforce velocity boundary conditions in place (config-driven)."""
    b = cfg.boundaries
    # x faces: west
    if b.x_west == "inflow":
        state.u[0, :, :] = b.warm_inflow.u          # +x into domain
    elif b.x_west == "outflow":
        state.u[0, :, :] = state.u[1, :, :]         # zero normal gradient
    # x faces: east
    if b.x_east == "inflow":
        state.u[-1, :, :] = -b.cold_inflow.u        # -x into domain
    elif b.x_east == "outflow":
        state.u[-1, :, :] = state.u[-2, :, :]        # zero normal gradient
    # y walls: free-slip (no normal v)
    if b.y == "free_slip" or b.y == "wall":
        state.v[:, 0, :] = 0.0
        state.v[:, -1, :] = 0.0
    # z: bottom free-slip (w=0); top boundary
    if b.z_bottom in ("free_slip", "no_slip"):
        state.w[:, :, 0] = 0.0
    if b.z_top == "rigid_lid":
        state.w[:, :, -1] = 0.0
    elif b.z_top == "open":
        # mass-balancing outflow: the top must carry out exactly what the two
        # opposing inflows bring in, so net boundary flux = 0 and the all-Neumann
        # projection yields a divergence-free (hence monotone-advection) field.
        #   w_top * (Lx*Ly) = (u_warm + u_cold) * (Ly*Lz)  =>  w_top = (u_w+c)*Lz/Lx
        w_out = (b.warm_inflow.u + b.cold_inflow.u) * grid.Lz / grid.Lx
        state.w[:, :, -1] = w_out
    # top damping layer: gently relax w toward 0 in the top slab (Rayleigh damping)
    if b.z_top == "damping_layer":
        nz = grid.nz
        nd = max(2, nz // 10)
        damp = np.linspace(0.0, 1.0, nd) ** 2
        for d, coeff in enumerate(damp):
            state.w[:, :, -1 - d] *= (1.0 - 0.05 * coeff)


def apply_scalar_bcs(state: FlowState, grid: Grid, cfg: SimulationConfig) -> None:
    """Enforce scalar (theta, q_v, q_l, q_i) boundary conditions in place."""
    b = cfg.boundaries
    P0 = cfg.physics.P0
    th_w, qv_w = inflow_state(b.warm_inflow, P0)
    th_c, qv_c = inflow_state(b.cold_inflow, P0)
    # inflow cells: west (cell 0) and east (cell -1) fixed
    if b.x_west == "inflow":
        state.theta[0, :, :] = th_w
        state.qv[0, :, :] = qv_w
        state.ql[0, :, :] = 0.0
        state.qi[0, :, :] = 0.0
    if b.x_east == "inflow":
        state.theta[-1, :, :] = th_c
        state.qv[-1, :, :] = qv_c
        state.ql[-1, :, :] = 0.0
        state.qi[-1, :, :] = 0.0
    # outflow / open boundaries: zero-gradient (copy inner slab)
    if b.x_west == "outflow":
        state.theta[0, :, :] = state.theta[1, :, :]
        state.qv[0, :, :] = state.qv[1, :, :]
    if b.x_east == "outflow":
        state.theta[-1, :, :] = state.theta[-2, :, :]
        state.qv[-1, :, :] = state.qv[-2, :, :]
    # y walls: zero normal gradient
    if b.y in ("free_slip", "wall"):
        state.theta[:, 0, :] = state.theta[:, 1, :]
        state.theta[:, -1, :] = state.theta[:, -2, :]
        state.qv[:, 0, :] = state.qv[:, 1, :]
        state.qv[:, -1, :] = state.qv[:, -2, :]
    # z: bottom & top zero-gradient for scalars (open top lets moisture exit)
    state.theta[:, :, 0] = state.theta[:, :, 1]
    state.theta[:, :, -1] = state.theta[:, :, -2]
    state.qv[:, :, 0] = state.qv[:, :, 1]
    state.qv[:, :, -1] = state.qv[:, :, -2]
    # periodic y (option): copy slabs to make the field periodic for the
    # one-sided operators' boundary rows
    if b.y == "periodic":
        state.theta[:, 0, :] = state.theta[:, -2, :]
        state.theta[:, -1, :] = state.theta[:, 1, :]
        state.qv[:, 0, :] = state.qv[:, -2, :]
        state.qv[:, -1, :] = state.qv[:, 1, :]
        state.u[:, 0, :] = state.u[:, -2, :]
        state.u[:, -1, :] = state.u[:, 1, :]


__all__ = ["apply_scalar_bcs", "apply_velocity_bcs", "inflow_state"]