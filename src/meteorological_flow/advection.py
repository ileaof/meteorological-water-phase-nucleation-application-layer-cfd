"""Finite-volume advection (flux-form, conservative, positivity-preserving).

Default is 1st-order upwind (monotone -> preserves positivity under CFL<=1).
Optional 2nd-order MUSCL with the minmod limiter.  Scalars live at cell centres
and are advected by the cell-centre velocity.  Momentum is advected by
interpolating each staggered component to centres, advecting, and interpolating
back -- a documented v1 simplification (not a fully conservative staggered
momentum advection; the projection step corrects the divergence).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .state import FlowState


def _minmod(a, b):
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def cell_velocity(state: FlowState, grid: Grid):
    """Cell-centre velocity (Uc, Vc, Wc) from staggered u, v, w."""
    Uc = 0.5 * (state.u[:-1, :, :] + state.u[1:, :, :])
    Vc = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    Wc = 0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:])
    return Uc, Vc, Wc


def _face_flux_x(s, Uf, grid, order):
    """Advective flux Fx = Uf * s_face on x-faces, shape (nx+1,ny,nz)."""
    nx = grid.nx
    F = np.zeros((nx + 1, grid.ny, grid.nz))
    if order == 1:
        # interior faces 1..nx-1
        left = s[:-1, :, :]
        right = s[1:, :, :]
        sup = np.where(Uf[1:-1, :, :] > 0.0, left, right)
        F[1:-1, :, :] = Uf[1:-1, :, :] * sup
        # boundary faces: one-sided upwind using edge cell value
        F[0, :, :] = Uf[0, :, :] * s[0, :, :]
        F[-1, :, :] = Uf[-1, :, :] * s[-1, :, :]
    else:
        # MUSCL with minmod
        sx = _minmod(s[1:, :, :] - s[:-1, :, :], np.zeros_like(s[1:, :, :]))
        # left state at interior face i (between i-1 and i): s[i-1]+0.5 slope[i-1]
        sL = s[:-1, :, :] + 0.5 * sx
        sR = s[1:, :, :] - 0.5 * sx
        F[1:-1, :, :] = Uf[1:-1, :, :] * np.where(Uf[1:-1, :, :] > 0.0, sL, sR)
        F[0, :, :] = Uf[0, :, :] * s[0, :, :]
        F[-1, :, :] = Uf[-1, :, :] * s[-1, :, :]
    return F


def _face_flux_y(s, Vf, grid, order):
    F = np.zeros((grid.nx, grid.ny + 1, grid.nz))
    if order == 1:
        left = s[:, :-1, :]
        right = s[:, 1:, :]
        sup = np.where(Vf[:, 1:-1, :] > 0.0, left, right)
        F[:, 1:-1, :] = Vf[:, 1:-1, :] * sup
        F[:, 0, :] = Vf[:, 0, :] * s[:, 0, :]
        F[:, -1, :] = Vf[:, -1, :] * s[:, -1, :]
    else:
        sy = _minmod(s[:, 1:, :] - s[:, :-1, :], np.zeros_like(s[:, 1:, :]))
        sL = s[:, :-1, :] + 0.5 * sy
        sR = s[:, 1:, :] - 0.5 * sy
        F[:, 1:-1, :] = Vf[:, 1:-1, :] * np.where(Vf[:, 1:-1, :] > 0.0, sL, sR)
        F[:, 0, :] = Vf[:, 0, :] * s[:, 0, :]
        F[:, -1, :] = Vf[:, -1, :] * s[:, -1, :]
    return F


def _face_flux_z(s, Wf, grid, order):
    F = np.zeros((grid.nx, grid.ny, grid.nz + 1))
    if order == 1:
        left = s[:, :, :-1]
        right = s[:, :, 1:]
        sup = np.where(Wf[:, :, 1:-1] > 0.0, left, right)
        F[:, :, 1:-1] = Wf[:, :, 1:-1] * sup
        F[:, :, 0] = Wf[:, :, 0] * s[:, :, 0]
        F[:, :, -1] = Wf[:, :, -1] * s[:, :, -1]
    else:
        sz = _minmod(s[:, :, 1:] - s[:, :, :-1], np.zeros_like(s[:, :, 1:]))
        sL = s[:, :, :-1] + 0.5 * sz
        sR = s[:, :, 1:] - 0.5 * sz
        F[:, :, 1:-1] = Wf[:, :, 1:-1] * np.where(Wf[:, :, 1:-1] > 0.0, sL, sR)
        F[:, :, 0] = Wf[:, :, 0] * s[:, :, 0]
        F[:, :, -1] = Wf[:, :, -1] * s[:, :, -1]
    return F


def advect_center(s, Uc, Vc, Wc, grid: Grid, dt: float, order: int = 1) -> np.ndarray:
    """Return s advanced by dt under advection by the cell-centre velocity.

    Flux-form: ds/dt = -div(F), F = u_face * s_upwind.  Conservative & monotone
    (order 1) under CFL<=1, so positivity of q_v/q_l/q_i is preserved.
    """
    # face velocities (simple average of adjacent centres)
    Uf = np.zeros(grid.u_shape)
    Uf[1:-1, :, :] = 0.5 * (Uc[:-1, :, :] + Uc[1:, :, :])
    Uf[0, :, :] = Uc[0, :, :]
    Uf[-1, :, :] = Uc[-1, :, :]
    Vf = np.zeros(grid.v_shape)
    Vf[:, 1:-1, :] = 0.5 * (Vc[:, :-1, :] + Vc[:, 1:, :])
    Vf[:, 0, :] = Vc[:, 0, :]
    Vf[:, -1, :] = Vc[:, -1, :]
    Wf = np.zeros(grid.w_shape)
    Wf[:, :, 1:-1] = 0.5 * (Wc[:, :, :-1] + Wc[:, :, 1:])
    Wf[:, :, 0] = Wc[:, :, 0]
    Wf[:, :, -1] = Wc[:, :, -1]

    Fx = _face_flux_x(s, Uf, grid, order)
    Fy = _face_flux_y(s, Vf, grid, order)
    Fz = _face_flux_z(s, Wf, grid, order)
    tend = -((Fx[1:, :, :] - Fx[:-1, :, :]) / grid.dx
             + (Fy[:, 1:, :] - Fy[:, :-1, :]) / grid.dy
             + (Fz[:, :, 1:] - Fz[:, :, :-1]) / grid.dz)
    return s + dt * tend


def advect_momentum(state: FlowState, grid: Grid, dt: float, order: int = 1) -> None:
    """Apply advection tendency to u, v, w in place (v1 center round-trip)."""
    Uc, Vc, Wc = cell_velocity(state, grid)
    # advect centre-interpolated velocity components, push tendency back to faces
    for comp_face, comp_name in ((state.u, "u"), (state.v, "v"), (state.w, "w")):
        # interpolate component to centres
        if comp_name == "u":
            fc = 0.5 * (state.u[:-1, :, :] + state.u[1:, :, :])
        elif comp_name == "v":
            fc = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
        else:
            fc = 0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:])
        fc_new = advect_center(fc, Uc, Vc, Wc, grid, dt, order)
        tend_c = (fc_new - fc) / dt if dt > 0 else 0.0 * fc
        # distribute center tendency back to faces (each face gets avg of nbr cells)
        if comp_name == "u":
            tend_f = np.zeros(grid.u_shape)
            tend_f[1:-1, :, :] = 0.5 * (tend_c[:-1, :, :] + tend_c[1:, :, :])
            tend_f[0, :, :] = tend_c[0, :, :]
            tend_f[-1, :, :] = tend_c[-1, :, :]
            state.u += dt * tend_f
        elif comp_name == "v":
            tend_f = np.zeros(grid.v_shape)
            tend_f[:, 1:-1, :] = 0.5 * (tend_c[:, :-1, :] + tend_c[:, 1:, :])
            tend_f[:, 0, :] = tend_c[:, 0, :]
            tend_f[:, -1, :] = tend_c[:, -1, :]
            state.v += dt * tend_f
        else:
            tend_f = np.zeros(grid.w_shape)
            tend_f[:, :, 1:-1] = 0.5 * (tend_c[:, :, :-1] + tend_c[:, :, 1:])
            tend_f[:, :, 0] = tend_c[:, :, 0]
            tend_f[:, :, -1] = tend_c[:, :, -1]
            state.w += dt * tend_f


__all__ = ["_minmod", "advect_center", "advect_momentum", "cell_velocity"]