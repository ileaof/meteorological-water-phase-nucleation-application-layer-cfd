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


def _minmod(a, b, xp=np):
    return xp.where(a * b <= 0.0, 0.0, xp.where(xp.abs(a) < xp.abs(b), a, b))


def cell_velocity(state: FlowState, grid: Grid):
    """Cell-centre velocity (Uc, Vc, Wc) from staggered u, v, w."""
    Uc = 0.5 * (state.u[:-1, :, :] + state.u[1:, :, :])
    Vc = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    Wc = 0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:])
    return Uc, Vc, Wc


def _face_flux_x(s, Uf, grid, order):
    """Advective flux Fx = Uf * s_face on x-faces, shape (nx+1,ny,nz)."""
    xp = grid.xp
    nx = grid.nx
    F = xp.zeros((nx + 1, grid.ny, grid.nz))
    if order == 1:
        # interior faces 1..nx-1
        left = s[:-1, :, :]
        right = s[1:, :, :]
        sup = xp.where(Uf[1:-1, :, :] > 0.0, left, right)
        F[1:-1, :, :] = Uf[1:-1, :, :] * sup
        # boundary faces: one-sided upwind using edge cell value
        F[0, :, :] = Uf[0, :, :] * s[0, :, :]
        F[-1, :, :] = Uf[-1, :, :] * s[-1, :, :]
    else:
        # MUSCL with minmod
        sx = _minmod(s[1:, :, :] - s[:-1, :, :], xp.zeros_like(s[1:, :, :]), xp=xp)
        # left state at interior face i (between i-1 and i): s[i-1]+0.5 slope[i-1]
        sL = s[:-1, :, :] + 0.5 * sx
        sR = s[1:, :, :] - 0.5 * sx
        F[1:-1, :, :] = Uf[1:-1, :, :] * xp.where(Uf[1:-1, :, :] > 0.0, sL, sR)
        F[0, :, :] = Uf[0, :, :] * s[0, :, :]
        F[-1, :, :] = Uf[-1, :, :] * s[-1, :, :]
    return F


def _face_flux_y(s, Vf, grid, order):
    xp = grid.xp
    F = xp.zeros((grid.nx, grid.ny + 1, grid.nz))
    if order == 1:
        left = s[:, :-1, :]
        right = s[:, 1:, :]
        sup = xp.where(Vf[:, 1:-1, :] > 0.0, left, right)
        F[:, 1:-1, :] = Vf[:, 1:-1, :] * sup
        F[:, 0, :] = Vf[:, 0, :] * s[:, 0, :]
        F[:, -1, :] = Vf[:, -1, :] * s[:, -1, :]
    else:
        sy = _minmod(s[:, 1:, :] - s[:, :-1, :], xp.zeros_like(s[:, 1:, :]), xp=xp)
        sL = s[:, :-1, :] + 0.5 * sy
        sR = s[:, 1:, :] - 0.5 * sy
        F[:, 1:-1, :] = Vf[:, 1:-1, :] * xp.where(Vf[:, 1:-1, :] > 0.0, sL, sR)
        F[:, 0, :] = Vf[:, 0, :] * s[:, 0, :]
        F[:, -1, :] = Vf[:, -1, :] * s[:, -1, :]
    return F


def _face_flux_z(s, Wf, grid, order):
    xp = grid.xp
    F = xp.zeros((grid.nx, grid.ny, grid.nz + 1))
    if order == 1:
        left = s[:, :, :-1]
        right = s[:, :, 1:]
        sup = xp.where(Wf[:, :, 1:-1] > 0.0, left, right)
        F[:, :, 1:-1] = Wf[:, :, 1:-1] * sup
        F[:, :, 0] = Wf[:, :, 0] * s[:, :, 0]
        F[:, :, -1] = Wf[:, :, -1] * s[:, :, -1]
    else:
        sz = _minmod(s[:, :, 1:] - s[:, :, :-1], xp.zeros_like(s[:, :, 1:]), xp=xp)
        sL = s[:, :, :-1] + 0.5 * sz
        sR = s[:, :, 1:] - 0.5 * sz
        F[:, :, 1:-1] = Wf[:, :, 1:-1] * xp.where(Wf[:, :, 1:-1] > 0.0, sL, sR)
        F[:, :, 0] = Wf[:, :, 0] * s[:, :, 0]
        F[:, :, -1] = Wf[:, :, -1] * s[:, :, -1]
    return F


def advect_center(s, Uc, Vc, Wc, grid: Grid, dt: float, order: int = 1) -> np.ndarray:
    """Return s advanced by dt under advection by the cell-centre velocity.

    Flux-form: ds/dt = -div(F), F = u_face * s_upwind.  Conservative & monotone
    (order 1) under CFL<=1, so positivity of q_v/q_l/q_i is preserved.
    """
    xp = grid.xp
    # face velocities (simple average of adjacent centres)
    Uf = xp.zeros(grid.u_shape)
    Uf[1:-1, :, :] = 0.5 * (Uc[:-1, :, :] + Uc[1:, :, :])
    Uf[0, :, :] = Uc[0, :, :]
    Uf[-1, :, :] = Uc[-1, :, :]
    Vf = xp.zeros(grid.v_shape)
    Vf[:, 1:-1, :] = 0.5 * (Vc[:, :-1, :] + Vc[:, 1:, :])
    Vf[:, 0, :] = Vc[:, 0, :]
    Vf[:, -1, :] = Vc[:, -1, :]
    Wf = xp.zeros(grid.w_shape)
    Wf[:, :, 1:-1] = 0.5 * (Wc[:, :, :-1] + Wc[:, :, 1:])
    Wf[:, :, 0] = Wc[:, :, 0]
    Wf[:, :, -1] = Wc[:, :, -1]

    Fx = _face_flux_x(s, Uf, grid, order)
    Fy = _face_flux_y(s, Vf, grid, order)
    Fz = _face_flux_z(s, Wf, grid, order)
    dz = grid.dz if not grid.stretched else grid.dz_c[None, None, :]
    tend = -((Fx[1:, :, :] - Fx[:-1, :, :]) / grid.dx
             + (Fy[:, 1:, :] - Fy[:, :-1, :]) / grid.dy
             + (Fz[:, :, 1:] - Fz[:, :, :-1]) / dz)
    return s + dt * tend


def advect_center_massflux(s, uf, vf, wf, grid: Grid, dt: float,
                           rho_c, rho_wf, order: int = 2) -> np.ndarray:
    """Conservative flux-form advection using the PROJECTED staggered face
    velocities and the reference density (Milestone 5).

    Solves ``d(rho0 s)/dt = -div(rho0 u s)`` in flux form, so ``int rho0 s dV`` is
    conserved up to the boundary flux.  Unlike :func:`advect_center`, the face
    velocities are the solver's own divergence-free ``u, v, w`` (not a cell-centre
    round-trip), so closed walls (``u_face=0``) carry exactly zero flux -- no
    spurious leak -- and the anelastic ``rho0(z)`` weighting makes the mass-weighted
    budget conservative.  ``order=2`` reconstructs the face value with a MUSCL
    (minmod) limited slope -- monotone/TVD, positivity-preserving, and far less
    diffusive than 1st-order upwind (which over-damps convection on a coarse grid).

    Parameters
    ----------
    uf, vf, wf : staggered face velocities (grid.u_shape / v_shape / w_shape).
    rho_c  : (nz,) reference density at cell centres.
    rho_wf : (nz+1,) reference density on the z-faces.
    order  : 1 (upwind) or 2 (MUSCL/minmod, default).
    """
    xp = grid.xp
    rc = xp.asarray(rho_c, dtype=float)[None, None, :]        # (1,1,nz)
    rwf = xp.asarray(rho_wf, dtype=float)[None, None, :]      # (1,1,nz+1)
    sx = xp.empty(grid.u_shape); sy = xp.empty(grid.v_shape); sz = xp.empty(grid.w_shape)
    if order >= 2:
        # limited cell slopes (minmod of backward/forward diffs; 0 at edge cells),
        # then upwind-biased reconstruction to the interior faces.
        dsx = xp.zeros_like(s); dsy = xp.zeros_like(s); dsz = xp.zeros_like(s)
        dsx[1:-1, :, :] = _minmod(s[1:-1, :, :] - s[:-2, :, :], s[2:, :, :] - s[1:-1, :, :], xp=xp)
        dsy[:, 1:-1, :] = _minmod(s[:, 1:-1, :] - s[:, :-2, :], s[:, 2:, :] - s[:, 1:-1, :], xp=xp)
        dsz[:, :, 1:-1] = _minmod(s[:, :, 1:-1] - s[:, :, :-2], s[:, :, 2:] - s[:, :, 1:-1], xp=xp)
        sx[1:-1, :, :] = xp.where(uf[1:-1, :, :] > 0.0,
                                  s[:-1, :, :] + 0.5 * dsx[:-1, :, :],
                                  s[1:, :, :] - 0.5 * dsx[1:, :, :])
        sy[:, 1:-1, :] = xp.where(vf[:, 1:-1, :] > 0.0,
                                  s[:, :-1, :] + 0.5 * dsy[:, :-1, :],
                                  s[:, 1:, :] - 0.5 * dsy[:, 1:, :])
        sz[:, :, 1:-1] = xp.where(wf[:, :, 1:-1] > 0.0,
                                  s[:, :, :-1] + 0.5 * dsz[:, :, :-1],
                                  s[:, :, 1:] - 0.5 * dsz[:, :, 1:])
    else:
        sx[1:-1, :, :] = xp.where(uf[1:-1, :, :] > 0.0, s[:-1, :, :], s[1:, :, :])
        sy[:, 1:-1, :] = xp.where(vf[:, 1:-1, :] > 0.0, s[:, :-1, :], s[:, 1:, :])
        sz[:, :, 1:-1] = xp.where(wf[:, :, 1:-1] > 0.0, s[:, :, :-1], s[:, :, 1:])
    # lateral boundary faces: edge cell value (walls carry u_face=0 -> zero flux),
    # or upwind from the periodic wrap neighbour (face 0 == face nx).
    if getattr(grid, "periodic", False):
        sx[0, :, :] = xp.where(uf[0, :, :] > 0.0, s[-1, :, :], s[0, :, :]); sx[-1, :, :] = sx[0, :, :]
        sy[:, 0, :] = xp.where(vf[:, 0, :] > 0.0, s[:, -1, :], s[:, 0, :]); sy[:, -1, :] = sy[:, 0, :]
    else:
        sx[0, :, :] = s[0, :, :]; sx[-1, :, :] = s[-1, :, :]
        sy[:, 0, :] = s[:, 0, :]; sy[:, -1, :] = s[:, -1, :]
    sz[:, :, 0] = s[:, :, 0]; sz[:, :, -1] = s[:, :, -1]
    # mass fluxes rho0 * u_face * s_face
    Fx = rc * uf * sx
    Fy = rc * vf * sy
    Fz = rwf * wf * sz
    dz = grid.dz if not grid.stretched else grid.dz_c[None, None, :]
    div = ((Fx[1:, :, :] - Fx[:-1, :, :]) / grid.dx
           + (Fy[:, 1:, :] - Fy[:, :-1, :]) / grid.dy
           + (Fz[:, :, 1:] - Fz[:, :, :-1]) / dz)
    return s - dt * div / rc      # d(rho0 s)/dt = -div  ->  s -= dt div/rho0


def advect_momentum(state: FlowState, grid: Grid, dt: float, order: int = 1) -> None:
    """Apply advection tendency to u, v, w in place (v1 center round-trip)."""
    xp = grid.xp
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
            tend_f = xp.zeros(grid.u_shape)
            tend_f[1:-1, :, :] = 0.5 * (tend_c[:-1, :, :] + tend_c[1:, :, :])
            tend_f[0, :, :] = tend_c[0, :, :]
            tend_f[-1, :, :] = tend_c[-1, :, :]
            state.u += dt * tend_f
        elif comp_name == "v":
            tend_f = xp.zeros(grid.v_shape)
            tend_f[:, 1:-1, :] = 0.5 * (tend_c[:, :-1, :] + tend_c[:, 1:, :])
            tend_f[:, 0, :] = tend_c[:, 0, :]
            tend_f[:, -1, :] = tend_c[:, -1, :]
            state.v += dt * tend_f
        else:
            tend_f = xp.zeros(grid.w_shape)
            tend_f[:, :, 1:-1] = 0.5 * (tend_c[:, :, :-1] + tend_c[:, :, 1:])
            tend_f[:, :, 0] = tend_c[:, :, 0]
            tend_f[:, :, -1] = tend_c[:, :, -1]
            state.w += dt * tend_f


__all__ = ["_minmod", "advect_center", "advect_center_massflux",
           "advect_momentum", "cell_velocity"]