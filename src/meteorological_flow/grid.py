"""Staggered Arakawa C-grid for the meteorological_flow solver.

Cell-centred scalars: theta, q_v, q_l, q_i, p' (perturbation pressure), T, rho.
Staggered velocities: u on x-faces (nx+1,ny,nz), v on y-faces (nx,ny+1,nz),
w on z-faces (nx,ny,nz+1).  z is the vertical (gravity along -z).

Operators are plain numpy finite-difference stencils.  Boundary one-sided
derivatives are used only where needed; the pressure solver owns the BC-aware
Laplacian.  All array shapes are (nx, ny, nz) for cell centres unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass

from .backend import Backend, get_backend


@dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    Lz: float
    z_stretch: float = 1.0      # >1 clusters levels near the surface (finer dz low,
                                # coarser aloft); 1.0 = uniform (default, unchanged)
    periodic: bool = False      # periodic lateral (x,y) boundaries (mean-wind storm)
    backend: Backend | None = None   # None -> CPU (every existing call site unaffected)

    def __post_init__(self):
        if self.backend is None:
            self.backend = get_backend("cpu")
        xp = self.xp = self.backend.xp
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz               # uniform reference spacing [m]
        self.xc = (xp.arange(self.nx) + 0.5) * self.dx
        self.yc = (xp.arange(self.ny) + 0.5) * self.dy
        self.xf = xp.linspace(0.0, self.Lx, self.nx + 1)
        self.yf = xp.linspace(0.0, self.Ly, self.ny + 1)
        # vertical levels: uniform, or geometrically stretched (dz_k = dz0 * r^k).
        if self.z_stretch == 1.0:
            self.zf = xp.linspace(0.0, self.Lz, self.nz + 1)
            self.zc = (xp.arange(self.nz) + 0.5) * self.dz
        else:
            w = self.z_stretch ** xp.arange(self.nz)          # relative cell heights
            edges = xp.concatenate([xp.asarray([0.0]), xp.cumsum(w)])
            self.zf = self.Lz * edges / edges[-1]
            self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])
        self.stretched = self.z_stretch != 1.0
        # dz arrays: cell heights dz_c (nz) and centre-to-centre spacings on the
        # z-faces dzc_f (nz+1).  For uniform these equal the scalar dz exactly, so
        # every operator below is byte-identical to the previous scalar-dz version.
        self.dz_c = xp.diff(self.zf)                          # (nz,) cell heights
        self.dzc_f = xp.empty(self.nz + 1)                    # (nz+1,) centre spacings
        self.dzc_f[1:-1] = xp.diff(self.zc)
        self.dzc_f[0] = self.dz_c[0]                          # boundary (Neumann grad=0)
        self.dzc_f[-1] = self.dz_c[-1]
        self.cell_vol = self.dx * self.dy * self.dz           # scalar (uniform value)
        self.cell_vol_c = self.dx * self.dy * self.dz_c       # (nz,) per-cell volume

    # ---- shapes ----
    @property
    def center_shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def u_shape(self):
        return (self.nx + 1, self.ny, self.nz)

    @property
    def v_shape(self):
        return (self.nx, self.ny + 1, self.nz)

    @property
    def w_shape(self):
        return (self.nx, self.ny, self.nz + 1)

    def zeros_c(self):
        return self.xp.zeros(self.center_shape)

    # ---- divergence of a face velocity field -> cell centres ----
    def divergence(self, u, v, w):
        """div(u) = du/dx + dv/dy + dw/dz at cell centres, shape (nx,ny,nz)."""
        dudx = (u[1:, :, :] - u[:-1, :, :]) / self.dx
        dvdy = (v[:, 1:, :] - v[:, :-1, :]) / self.dy
        dz = self.dz if not self.stretched else self.dz_c[None, None, :]
        dwdz = (w[:, :, 1:] - w[:, :, :-1]) / dz
        return dudx + dvdy + dwdz

    # ---- gradient of a cell-centred scalar -> face gradients ----
    def grad_x_faces(self, p):
        """dp/dx on x-faces (nx+1,ny,nz).  Interior: central; boundary faces are
        0 (Neumann) by default, or the periodic wrap (p[0]-p[nx-1])/dx when the
        grid is periodic (face 0 == face nx)."""
        g = self.xp.zeros(self.u_shape)
        g[1:-1, :, :] = (p[1:, :, :] - p[:-1, :, :]) / self.dx
        if self.periodic:
            wrap = (p[0, :, :] - p[-1, :, :]) / self.dx
            g[0, :, :] = wrap; g[-1, :, :] = wrap
        return g

    def grad_y_faces(self, p):
        g = self.xp.zeros(self.v_shape)
        g[:, 1:-1, :] = (p[:, 1:, :] - p[:, :-1, :]) / self.dy
        if self.periodic:
            wrap = (p[:, 0, :] - p[:, -1, :]) / self.dy
            g[:, 0, :] = wrap; g[:, -1, :] = wrap
        return g

    def grad_z_faces(self, p):
        g = self.xp.zeros(self.w_shape)
        spacing = self.dz if not self.stretched else self.dzc_f[None, None, 1:-1]
        g[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / spacing
        return g

    # ---- interpolate cell-centre scalar to face centres (simple average) ----
    def interp_c_to_ufaces(self, f):
        out = self.xp.empty(self.u_shape)
        out[1:-1, :, :] = 0.5 * (f[:-1, :, :] + f[1:, :, :])
        out[0, :, :] = f[0, :, :]
        out[-1, :, :] = f[-1, :, :]
        return out

    def interp_c_to_vfaces(self, f):
        out = self.xp.empty(self.v_shape)
        out[:, 1:-1, :] = 0.5 * (f[:, :-1, :] + f[:, 1:, :])
        out[:, 0, :] = f[:, 0, :]
        out[:, -1, :] = f[:, -1, :]
        return out

    def interp_c_to_wfaces(self, f):
        out = self.xp.empty(self.w_shape)
        out[:, :, 1:-1] = 0.5 * (f[:, :, :-1] + f[:, :, 1:])
        out[:, :, 0] = f[:, :, 0]
        out[:, :, -1] = f[:, :, -1]
        return out

    # ---- temperature gradient magnitude at cell centres (|grad T|, K/m) ----
    def grad_magnitude(self, f):
        """|grad f| at cell centres using central differences with one-sided
        boundary stencils.  Returns array shape (nx,ny,nz)."""
        gx = self._central_x(f)
        gy = self._central_y(f)
        gz = self._central_z(f)
        return self.xp.sqrt(gx * gx + gy * gy + gz * gz)

    def _central_x(self, f):
        g = self.xp.zeros_like(f)
        g[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * self.dx)
        g[0, :, :] = (f[1, :, :] - f[0, :, :]) / self.dx
        g[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / self.dx
        return g

    def _central_y(self, f):
        g = self.xp.zeros_like(f)
        g[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * self.dy)
        g[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / self.dy
        g[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / self.dy
        return g

    def _central_z(self, f):
        g = self.xp.zeros_like(f)
        if not self.stretched:
            g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * self.dz)
            g[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / self.dz
            g[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / self.dz
        else:
            g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (self.zc[2:] - self.zc[:-2])[None, None, :]
            g[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / self.dzc_f[None, None, 1]
            g[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / self.dzc_f[None, None, -2]
        return g

    def laplacian(self, f):
        """5/7-point Laplacian of a cell-centred scalar (used by diffusion).
        Lateral boundaries: one-sided (Neumann) by default, or periodic wrap."""
        g = self.xp.zeros_like(f)
        g[1:-1, :, :] += (f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]) / self.dx ** 2
        g[:, 1:-1, :] += (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / self.dy ** 2
        if self.periodic:
            g[0, :, :] += (f[1, :, :] - 2 * f[0, :, :] + f[-1, :, :]) / self.dx ** 2
            g[-1, :, :] += (f[0, :, :] - 2 * f[-1, :, :] + f[-2, :, :]) / self.dx ** 2
            g[:, 0, :] += (f[:, 1, :] - 2 * f[:, 0, :] + f[:, -1, :]) / self.dy ** 2
            g[:, -1, :] += (f[:, 0, :] - 2 * f[:, -1, :] + f[:, -2, :]) / self.dy ** 2
        else:
            g[0, :, :] += (f[1, :, :] - f[0, :, :]) / self.dx ** 2
            g[-1, :, :] += (f[-1, :, :] - f[-2, :, :]) / self.dx ** 2
            g[:, 0, :] += (f[:, 1, :] - f[:, 0, :]) / self.dy ** 2
            g[:, -1, :] += (f[:, -1, :] - f[:, -2, :]) / self.dy ** 2
        if not self.stretched:
            g[:, :, 1:-1] += (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / self.dz ** 2
            g[:, :, 0] += (f[:, :, 1] - f[:, :, 0]) / self.dz ** 2
            g[:, :, -1] += (f[:, :, -1] - f[:, :, -2]) / self.dz ** 2
        else:                                   # variable-spacing z second derivative
            dzcf = self.dzc_f[None, None, 1:-1]                 # interior face spacings
            dzc = self.dz_c[None, None, :]
            fg = (f[:, :, 1:] - f[:, :, :-1]) / dzcf            # interior face gradients
            g[:, :, 1:-1] += (fg[:, :, 1:] - fg[:, :, :-1]) / dzc[:, :, 1:-1]
            g[:, :, 0] += fg[:, :, 0] / dzc[:, :, 0]
            g[:, :, -1] += fg[:, :, -1] / dzc[:, :, -1]
        return g


__all__ = ["Grid"]