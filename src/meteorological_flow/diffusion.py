"""Explicit 2nd-order diffusion (viscosity / scalar diffusivity).

Uses the cell-centre Laplacian from :class:`grid.Grid`.  Stability requires
dt <= dx^2 / (2 * nu * 3) (3D); the CFL selector in :mod:`simulation` enforces
this.  Smagorinsky SGS is a documented Batch-2 extension (off here).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .state import FlowState


def diffuse_center(s: np.ndarray, grid: Grid, kappa: float, dt: float,
                   base: np.ndarray | None = None) -> np.ndarray:
    """s + dt * kappa * laplacian(s').

    If ``base`` (a reference profile field, e.g. theta0(z) or qv0(z)) is given,
    only the PERTURBATION ``s' = s - base`` is diffused and the reference state is
    left untouched.  Diffusing the full field would apply kappa*laplacian to the
    curved base profile, injecting a spurious source (for a stratified sounding
    this manufactures theta' != 0 -> spurious buoyancy -> a phantom circulation
    even at rest).  Diffusing the perturbation only removes that artefact, so the
    reference state is a true discrete equilibrium.
    """
    if kappa <= 0.0 or dt <= 0.0:
        return s
    field = s if base is None else (s - base)
    return s + dt * kappa * grid.laplacian(field)


def _center_to_faces(tend_c: np.ndarray, grid: Grid, axis: int) -> np.ndarray:
    """Map a cell-centre tendency to the staggered face grid by averaging
    adjacent cells (one-sided at boundaries)."""
    xp = grid.xp
    if axis == 0:
        out = xp.zeros(grid.u_shape)
        out[1:-1, :, :] = 0.5 * (tend_c[:-1, :, :] + tend_c[1:, :, :])
        out[0, :, :] = tend_c[0, :, :]
        out[-1, :, :] = tend_c[-1, :, :]
    elif axis == 1:
        out = xp.zeros(grid.v_shape)
        out[:, 1:-1, :] = 0.5 * (tend_c[:, :-1, :] + tend_c[:, 1:, :])
        out[:, 0, :] = tend_c[:, 0, :]
        out[:, -1, :] = tend_c[:, -1, :]
    else:
        out = xp.zeros(grid.w_shape)
        out[:, :, 1:-1] = 0.5 * (tend_c[:, :, :-1] + tend_c[:, :, 1:])
        out[:, :, 0] = tend_c[:, :, 0]
        out[:, :, -1] = tend_c[:, :, -1]
    return out


def diffuse_momentum(state: FlowState, grid: Grid, nu: float, dt: float) -> None:
    """Apply viscous diffusion du/dt = nu * laplacian(u) to u, v, w in place
    (v1: interpolate each component to centres, diffuse, map tendency back)."""
    if nu <= 0.0 or dt <= 0.0:
        return
    uc = 0.5 * (state.u[:-1, :, :] + state.u[1:, :, :])
    vc = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    wc = 0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:])
    du = nu * grid.laplacian(uc)
    dv = nu * grid.laplacian(vc)
    dw = nu * grid.laplacian(wc)
    state.u += dt * _center_to_faces(du, grid, 0)
    state.v += dt * _center_to_faces(dv, grid, 1)
    state.w += dt * _center_to_faces(dw, grid, 2)


__all__ = ["diffuse_center", "diffuse_momentum"]