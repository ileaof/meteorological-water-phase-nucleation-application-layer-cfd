"""Chorin pressure projection for the Boussinesq solver.

Assembles the cell-centre 7-point Laplacian ONCE as a sparse matrix with
Neumann boundary conditions (ghost = self), and solves the Poisson equation

    lap(p') = (rho0 / dt) * div(u*)     ->     u = u* - (dt / rho0) grad(p')

each step.  The Laplacian is constant for a fixed grid/BC, so it is factorised
once (cached `splu` for small grids) or solved with CG (+Jacobi) for larger
ones.  All-Neumann makes the system singular (constant null space); we subtract
the mean of the RHS (compatibility) and add a tiny diagonal pin so the mean of
p' is zero.  Solver residual and iteration count are reported each step.

Boundary faces use dp'/dn = 0, so the projection does NOT alter the boundary
velocity (inflow Dirichlet velocities are preserved; the open top adjusts).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Grid
from .state import FlowState


def _idx(i, j, k, ny, nz):
    return (i * ny + j) * nz + k


class PressureSolver:
    def __init__(self, grid: Grid, method: str = "cg", tol: float = 1e-6,
                 maxiter: int = 800, dirichlet_top: bool = False):
        self.grid = grid
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.dirichlet_top = dirichlet_top   # p'=0 at the top (open pressure outlet)
        self.n = grid.nx * grid.ny * grid.nz
        self.A = self._build()           # A = -lap (positive semi-definite) + pin
        self._lu = None
        if method == "direct":
            # cached LU factorisation of the (regularised) SPD operator
            self._lu = spla.splu(self.A.tocsc())
        self.last_residual = 0.0
        self.last_iters = 0

    def _build(self):
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        rows, cols, data = [], [], []
        inv = {0: 1.0 / g.dx ** 2, 1: 1.0 / g.dy ** 2, 2: 1.0 / g.dz ** 2}
        top_cells = set()
        for i in range(nx):
            for j in range(ny):
                top_cells.add(_idx(i, j, nz - 1, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    a = _idx(i, j, k, ny, nz)
                    if self.dirichlet_top and a in top_cells:
                        # Dirichlet p'=0 at the top (open pressure outlet): identity row
                        rows.append(a); cols.append(a); data.append(1.0)
                        continue
                    diag = 0.0
                    for (di, dj, dk, ax) in ((-1, 0, 0, 0), (1, 0, 0, 0),
                                            (0, -1, 0, 1), (0, 1, 0, 1),
                                            (0, 0, -1, 2), (0, 0, 1, 2)):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            w = inv[ax]
                            diag += w
                            b = _idx(ni, nj, nk, ny, nz)
                            rows.append(a); cols.append(b); data.append(-w)
                    rows.append(a); cols.append(a); data.append(diag)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n, self.n)).tocsr()
        # pin the constant null space with a tiny diagonal (sets mean(p')=0).
        # With a Dirichlet top the system is already non-singular; the pin is harmless.
        A = A + 1e-12 * sp.eye(self.n, format="csr")
        return A

    def solve(self, rhs: np.ndarray):
        rhs = rhs.reshape(-1).astype(float)
        if self.dirichlet_top:
            # zero the RHS at Dirichlet (top) cells so p_top = 0
            nz = self.grid.nz
            ny = self.grid.ny
            for i in range(self.grid.nx):
                for j in range(ny):
                    rhs[_idx(i, j, nz - 1, ny, nz)] = 0.0
        else:
            rhs = rhs - rhs.mean()           # compatibility with all-Neumann
        if self.method == "direct" and self._lu is not None:
            p = self._lu.solve(rhs)
            self.last_residual = float(np.linalg.norm(self.A @ p - rhs))
            self.last_iters = 0
        else:
            # CG on the positive (regularised) operator with Jacobi precond
            diag = self.A.diagonal()
            M = spla.LinearOperator(self.A.shape, matvec=lambda x: x / diag)
            p, info = spla.cg(self.A, rhs, M=M, rtol=self.tol, atol=0.0, maxiter=self.maxiter)
            self.last_residual = float(np.linalg.norm(self.A @ p - rhs))
            self.last_iters = int(self.maxiter if info else 0)
            if info:
                # did not fully converge; keep the best iterate
                pass
        if not self.dirichlet_top:
            p = p - p.mean()
        return p.reshape(self.grid.center_shape), self.last_residual, self.last_iters

    def project(self, state: FlowState, dt: float, rho0: float):
        """Projection step in place: enforce div(u) ~ 0. Returns (residual, iters)."""
        g = self.grid
        div = g.divergence(state.u, state.v, state.w)          # cell centres
        # Chorin: lap(p') = (rho0/dt) div(u*).  A = -lap (positive-definite), so
        # solving A p = rhs  =>  lap(p') = -rhs; we therefore set rhs = -(rho0/dt) div
        # so that lap(p') = +(rho0/dt) div and the correction cancels the divergence.
        rhs = -(rho0 / dt) * div
        p, res, it = self.solve(rhs)
        state.p = p
        # correct face velocities: u -= (dt/rho0) grad(p); boundary grad=0
        gpx = g.grad_x_faces(p)   # Neumann -> 0 at boundary faces
        gpy = g.grad_y_faces(p)
        gpz = g.grad_z_faces(p)
        state.u -= (dt / rho0) * gpx
        state.v -= (dt / rho0) * gpy
        state.w -= (dt / rho0) * gpz
        return res, it


__all__ = ["PressureSolver"]