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
        self.backend = grid.backend
        self.xp = grid.xp
        stretched = getattr(grid, "stretched", False)
        if self.backend.name == "gpu":
            if stretched:
                # CG is not guaranteed to converge on the asymmetric
                # stretched-grid vertical operator (see _build()'s comment
                # below) -- this is a property of the operator itself, not a
                # GPU-specific issue, so a stretched grid always uses the
                # direct CPU solve regardless of backend, exactly like the
                # CPU heuristic in _pressure_method() already does. Only the
                # small (n,)-sized RHS/solution vectors cross the host/device
                # boundary each step -- never the operator or any 3-D field.
                if method != "direct":
                    print(f"[pressure_solver] GPU backend + stretched grid: forcing "
                          f"the direct CPU solve (requested method={method!r}) -- CG "
                          f"is not appropriate for the asymmetric stretched-grid "
                          f"operator; only the small RHS/solution vectors cross the "
                          f"host/device boundary each step, not the 3-D fields.")
                method = "direct"
            elif method != "cg":
                # CuPy has no GPU-native direct sparse LU solver equivalent to
                # scipy's splu (only iterative solvers) -- explicit, logged,
                # one-time-per-run behavioural difference from the CPU
                # heuristic in _pressure_method(), not a silent fallback.
                print(f"[pressure_solver] GPU backend: using iterative CG for the "
                      f"pressure Poisson solve (requested method={method!r} has no "
                      f"GPU-native direct-solver equivalent to scipy's splu).")
                method = "cg"
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.dirichlet_top = dirichlet_top   # p'=0 at the top (open pressure outlet)
        self.n = grid.nx * grid.ny * grid.nz
        # the Laplacian is ALWAYS assembled on the host via scipy: _build() is a
        # one-time (not per-step) triple Python loop over cell indices -- index
        # bookkeeping, not math, so there is no vectorisation/GPU win from moving
        # it, and scipy's COO/CSR builders are the simplest correct way to do it.
        self.A = self._build()           # A = -lap (positive semi-definite) + pin
        self._lu = None
        if method == "direct":
            # cached LU factorisation of the (regularised) SPD operator.
            # Always CPU/scipy -- see the stretched-grid note above.
            self._lu = spla.splu(self.A.tocsc())
        if self.backend.name == "gpu" and method != "direct":
            # only move the operator to the GPU when the CG path actually
            # uses it there; the direct path keeps everything on the host.
            self.A = self.backend.sparse.csr_matrix(self.A)
        self.last_residual = 0.0
        self.last_iters = 0

    def _build(self):
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        rows, cols, data = [], [], []
        inv = {0: 1.0 / g.dx ** 2, 1: 1.0 / g.dy ** 2, 2: 1.0 / g.dz ** 2}
        # variable-dz vertical coefficients (reduce to 1/dz^2 for uniform).  The
        # z-operator (1/dz_c) d/dz(dp/dz) is asymmetric under stretching, so a
        # stretched grid uses the direct (splu) solver -- see _pick_method.
        stretched = getattr(g, "stretched", False)
        # this loop is host-only (see __init__'s comment); dz_c/dzc_f may be
        # GPU-resident, so pull the (tiny, (nz,)/(nz+1,)-sized) profiles to the
        # host once here rather than indexing a device array per cell below.
        dz_c = g.backend.to_cpu(g.dz_c) if stretched else None
        dzc_f = g.backend.to_cpu(g.dzc_f) if stretched else None
        periodic = getattr(g, "periodic", False)   # wrap x/y neighbours
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
                        if periodic and ax == 0:
                            ni %= nx            # periodic x wrap (face 0 == face nx)
                        if periodic and ax == 1:
                            nj %= ny            # periodic y wrap
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            if ax == 2 and stretched:
                                # neighbour k+1: 1/(dz_c[k]*dzc_f[k+1]); k-1: 1/(dz_c[k]*dzc_f[k])
                                w = 1.0 / (dz_c[k] * (dzc_f[k + 1] if dk == 1 else dzc_f[k]))
                            else:
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
        xp = self.xp
        # direct solve is always host/scipy (splu); when the rest of the
        # solver is on GPU (stretched grid, see __init__), only this small
        # (n,)-sized RHS/solution vector crosses the host/device boundary --
        # never the operator or any 3-D field.
        use_host = (self.method == "direct" and self._lu is not None
                   and self.backend.name == "gpu")
        work_xp = np if use_host else xp
        rhs = self.backend.to_cpu(rhs) if use_host else xp.asarray(rhs)
        rhs = work_xp.asarray(rhs).reshape(-1).astype(float)
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
            self.last_residual = float(work_xp.linalg.norm(self.A @ p - rhs))
            self.last_iters = 0
        else:
            # CG on the positive (regularised) operator with Jacobi precond
            diag = self.A.diagonal()
            M = self.backend.sparse_linalg.LinearOperator(self.A.shape, matvec=lambda x: x / diag)
            p, info = self.backend.sparse_linalg.cg(self.A, rhs, M=M, rtol=self.tol,
                                                     atol=0.0, maxiter=self.maxiter)
            self.last_residual = float(work_xp.linalg.norm(self.A @ p - rhs))
            self.last_iters = int(self.maxiter if info else 0)
            if info:
                # did not fully converge; keep the best iterate
                pass
        if not self.dirichlet_top:
            p = p - p.mean()
        if use_host:
            p = xp.asarray(p)   # back to the solver's normal backend for the caller
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

    def project_anelastic(self, state: FlowState, dt: float, rho0_c, rho0_wface):
        """Anelastic projection: enforce div(rho0 u) ~ 0 with rho0 = rho0(z).

        The reference density varies only with height, so the SAME constant-
        coefficient 7-point Laplacian ``A`` is reused.  Writing the velocity
        correction as ``u = u* - (dt/rho0_face) grad(p')`` makes the face density
        cancel the operator exactly (rho0_face * (1/rho0_face) = 1), so

            lap(p') = (1/dt) div(rho0 u*)         (density-weighted divergence)

        is solved with the cached factorisation and the corrected velocity is
        discretely divergence-free in the anelastic (mass-weighted) sense.  This
        captures the deep-convection mass expansion (updrafts amplify as they
        rise into lower rho0) that the constant-density Boussinesq mode misses.

        Parameters
        ----------
        rho0_c : (nz,) reference density at cell centres.
        rho0_wface : (nz+1,) reference density on the z-faces (w-points).
        """
        g = self.grid
        xp = self.xp
        rc = xp.asarray(rho0_c, dtype=float).reshape(1, 1, -1)      # (1,1,nz)
        rwf = xp.asarray(rho0_wface, dtype=float).reshape(1, 1, -1)  # (1,1,nz+1)
        # density-weighted divergence  div(rho0 u*).  rho0 = rho0(z) is constant
        # in x,y at each level, so the horizontal terms factor rho0_c(k); the
        # vertical term uses the face density on the staggered w-points.
        dudx = (state.u[1:, :, :] - state.u[:-1, :, :]) / g.dx
        dvdy = (state.v[:, 1:, :] - state.v[:, :-1, :]) / g.dy
        wflux = rwf * state.w                                        # (nx,ny,nz+1)
        # per-cell height under vertical stretching (must match the Poisson z-stencil)
        dz = g.dz if not getattr(g, "stretched", False) else g.dz_c[None, None, :]
        dwdz = (wflux[:, :, 1:] - wflux[:, :, :-1]) / dz
        div_rho = rc * (dudx + dvdy) + dwdz
        # A = -lap, so rhs = -(1/dt) div(rho0 u*) gives lap(p') = (1/dt) div(rho0 u*)
        rhs = -(1.0 / dt) * div_rho
        p, res, it = self.solve(rhs)
        state.p = p
        gpx = g.grad_x_faces(p)   # boundary faces = 0 (Neumann)
        gpy = g.grad_y_faces(p)
        gpz = g.grad_z_faces(p)
        # correct with the LOCAL face density (must match the weights above).
        # x/y-faces at level k carry rho0_c(k); z-faces carry rho0_wface.
        state.u -= (dt / rc) * gpx
        state.v -= (dt / rc) * gpy
        state.w -= (dt / rwf) * gpz
        return res, it


__all__ = ["PressureSolver"]