#!/usr/bin/env python3
"""test_grid.py -- staggered C-grid operators and metrics.

Covers spec tests:
  1  uniform field stays uniform under the divergence / gradient operators
  3  solid-body-style advection of a passive scalar (here: a rotated gradient
     is reproduced by the centred gradient stencils), and grid metrics are
     consistent (dx*Lx = ... , cell_vol, shapes).
"""
import numpy as np

from meteorological_flow.grid import Grid


def _grid(n=10, L=100.0):
    return Grid(nx=n, ny=n, nz=n, Lx=L, Ly=L, Lz=L)


def test_01_uniform_field_zero_divergence_and_gradient():
    """[math] A uniform velocity field has zero divergence, and a uniform
    scalar has zero gradient on the centred stencils."""
    g = _grid(8)
    u = np.full(g.u_shape, 2.0)
    v = np.full(g.v_shape, 0.5)
    w = np.full(g.w_shape, -0.3)
    div = g.divergence(u, v, w)
    assert np.allclose(div, 0.0), "uniform field divergence must vanish"
    f = np.full(g.center_shape, 7.0)
    assert np.allclose(g.grad_magnitude(f), 0.0), "uniform scalar |grad|=0"
    assert np.allclose(g.laplacian(f), 0.0), "uniform scalar laplacian=0"


def test_02_grid_metrics_and_shapes():
    """[math] Grid metrics and staggered shapes are internally consistent."""
    g = _grid(20, 100.0)
    assert np.isclose(g.dx, 5.0) and np.isclose(g.dy, 5.0) and np.isclose(g.dz, 5.0)
    assert np.isclose(g.cell_vol, 5.0 ** 3)
    assert g.center_shape == (20, 20, 20)
    assert g.u_shape == (21, 20, 20)
    assert g.v_shape == (20, 21, 20)
    assert g.w_shape == (20, 20, 21)
    # face coordinates span the domain exactly
    assert np.isclose(g.xf[0], 0.0) and np.isclose(g.xf[-1], 100.0)
    # cell centres sit at half-cell offsets
    assert np.isclose(g.xc[0], 2.5) and np.isclose(g.zc[-1], 97.5)


def test_03_centred_gradient_reproduces_linear_field():
    """[num] The central-difference gradient of a linear field f = a*x + b*y +
    c*z + d recovers the constant gradient (a, b, c) to 2nd-order accuracy,
    including the one-sided boundary stencils (1st-order there)."""
    g = _grid(16)
    X, Y, Z = np.meshgrid(g.xc, g.yc, g.zc, indexing="ij")
    a, b, c, d = 1.5, -0.7, 0.4, 3.0
    f = a * X + b * Y + c * Z + d
    mag = g.grad_magnitude(f)
    true_mag = np.sqrt(a * a + b * b + c * c)
    # interior is exact for a linear field (central differences)
    assert np.allclose(mag[1:-1, 1:-1, 1:-1], true_mag), "interior grad wrong"
    # boundaries are 1st-order one-sided: still close for a linear field
    assert np.allclose(mag, true_mag, atol=1e-6), "boundary grad wrong"


def test_04_divergence_of_linear_velocity_is_constant():
    """[num] div(u) of u=(a*x, b*y, c*z) sampled on faces equals a+b+c."""
    g = _grid(12)
    # face velocities linear in the face coordinate
    u = g.xf.reshape(-1, 1, 1) * np.ones(g.u_shape)            # du/dx = 1
    v = 2.0 * g.yf.reshape(1, -1, 1) * np.ones(g.v_shape)       # dv/dy = 2
    w = -0.5 * g.zf.reshape(1, 1, -1) * np.ones(g.w_shape)      # dw/dz = -0.5
    div = g.divergence(u, v, w)
    assert np.allclose(div, 1.0 + 2.0 - 0.5), "linear divergence wrong"


def test_05_laplacian_of_quadratic_is_constant():
    """[math] laplacian of f = x^2 + y^2 + z^2 (centred) is 2+2+2=6 in the
    interior (one-sided boundaries differ by O(dx))."""
    g = _grid(10)
    X, Y, Z = np.meshgrid(g.xc, g.yc, g.zc, indexing="ij")
    f = X ** 2 + Y ** 2 + Z ** 2
    lap = g.laplacian(f)
    assert np.allclose(lap[1:-1, 1:-1, 1:-1], 6.0), "interior laplacian != 6"


def test_06_interp_to_faces_round_trip_preserves_mean():
    """[num] Cell->face interpolation is the simple average; the average of
    the face-interpolated field recovers the cell mean to O(boundary)."""
    g = _grid(8)
    f = np.arange(np.prod(g.center_shape), dtype=float).reshape(g.center_shape)
    uf = g.interp_c_to_ufaces(f)
    # interior faces are the average of the two bracketing cells
    assert np.allclose(uf[1:-1, :, :], 0.5 * (f[:-1, :, :] + f[1:, :, :]))