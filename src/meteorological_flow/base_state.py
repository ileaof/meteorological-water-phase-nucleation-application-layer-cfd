"""Hydrostatic, conditionally-unstable base state for the deep-convection
(storm-scale) scenario.

The mixing-chamber scenario assumes a uniform background pressure ``P0`` -- fine
for a shallow (~100 m) box.  A km-deep storm column needs a *stratified* base
state so that ``T = theta (p/P0_REF)^(R_d/c_p)`` is realistic at every height
(the pressure falls from ~1000 hPa at the surface to ~200 hPa at 12 km).

This builds an idealised sounding:

* ``theta0(z)`` -- a stably stratified potential temperature (``dtheta/dz > 0``);
* ``qv0(z)``    -- moist low levels, drying aloft (exponential scale height);
* ``p0(z)``     -- hydrostatically integrated from the surface upward, consistent
  with the virtual temperature ``T_v0 = T0 (1 + 0.61 qv0)``.

A lifted, moist near-surface parcel is stable to *dry* ascent but becomes
positively buoyant once it saturates and the microphysics releases latent heat
-- i.e. the environment is *conditionally* unstable, which is what lets a warm
bubble grow into a deep, precipitating updraft when coupled to the two-way
microphysics.

Boussinesq caveat: over a 10-12 km depth the density varies by ~2-3x, beyond the
Boussinesq approximation's strict validity.  The storm scenario is therefore a
**demonstration** (Boussinesq-stretched), not a quantitatively validated
deep-convection result; an anelastic/compressible core is future work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import thermodynamics as th


@dataclass
class BaseState:
    zc: np.ndarray          # cell-centre heights [m]           (nz,)
    theta0: np.ndarray      # base potential temperature [K]    (nz,)
    qv0: np.ndarray         # base vapour mixing ratio [kg/kg]  (nz,)
    p0: np.ndarray          # base pressure [Pa]                (nz,)
    T0: np.ndarray          # base temperature [K]              (nz,)
    rho0: np.ndarray        # base density [kg/m^3]             (nz,)

    def field(self, arr1d, shape):
        """Broadcast a (nz,) base profile to the (nx,ny,nz) grid."""
        return np.broadcast_to(arr1d.reshape(1, 1, -1), shape).copy()


def build_base_state(grid, *, T_sfc=301.0, p_sfc=101325.0, RH_sfc=0.85,
                     dtheta_dz=3.2e-3, q_scale_height=3400.0,
                     RH_min=0.45) -> BaseState:
    """Return a hydrostatic conditionally-unstable :class:`BaseState`.

    ``dtheta_dz`` [K/m] sets the (stable) stratification; ``q_scale_height`` [m]
    the moisture lapse.  Defaults give ~CAPE-supporting warm-sector conditions.
    """
    zc = np.asarray(grid.zc, dtype=float)
    nz = zc.size

    theta_sfc = float(th.theta_from_T(T_sfc, p_sfc, th.P0_REF))
    theta0 = theta_sfc + dtheta_dz * zc

    # surface vapour from RH, drying aloft (also capped by a minimum RH so the
    # mid-troposphere is not bone dry)
    qsat_sfc = float(th.q_v_from_p_v(th.psat_water(T_sfc), p_sfc))
    qv_sfc = RH_sfc * qsat_sfc
    qv0 = qv_sfc * np.exp(-zc / q_scale_height)

    # hydrostatic integration upward: dp/dz = -rho g, rho = p/(R_d T_v)
    p0 = np.empty(nz)
    T0 = np.empty(nz)
    # integrate on the cell-centre grid from the surface (p_sfc at z=0) upward
    p_prev, z_prev = p_sfc, 0.0
    for k in range(nz):
        dz = zc[k] - z_prev
        p_new = p_prev
        for _ in range(3):   # fixed-point: T depends on p depends on T_v
            Tk = float(th.T_from_theta(theta0[k], p_new, th.P0_REF))
            Tvk = Tk * (1.0 + 0.61 * qv0[k])
            p_new = p_prev * float(np.exp(-th.g0 * dz / (th.R_d * Tvk)))
        p0[k] = p_new
        T0[k] = float(th.T_from_theta(theta0[k], p0[k], th.P0_REF))
        p_prev, z_prev = p_new, zc[k]

    # cap dryness aloft at RH_min over ice/water (avoid unphysically dry cells)
    qsat0 = th.q_v_from_p_v(th.psat_water(T0), p0)
    qv0 = np.maximum(qv0, RH_min * qsat0)
    Tv0 = T0 * (1.0 + 0.61 * qv0)
    rho0 = p0 / (th.R_d * Tv0)
    return BaseState(zc=zc, theta0=theta0, qv0=qv0, p0=p0, T0=T0, rho0=rho0)


def warm_bubble(grid, *, dtheta=2.5, x_c=None, y_c=None, z_c=1500.0,
                radius=2000.0, z_radius=1500.0, moist_frac=0.0, qv_bump=0.0):
    """Return a (nx,ny,nz) potential-temperature perturbation (and optional
    vapour perturbation) for a Gaussian warm bubble that triggers convection."""
    x_c = 0.5 * grid.Lx if x_c is None else x_c
    y_c = 0.5 * grid.Ly if y_c is None else y_c
    X = grid.xc.reshape(-1, 1, 1)
    Y = grid.yc.reshape(1, -1, 1)
    Z = grid.zc.reshape(1, 1, -1)
    r2 = ((X - x_c) / radius) ** 2 + ((Y - y_c) / radius) ** 2 + ((Z - z_c) / z_radius) ** 2
    amp = np.exp(-r2)
    dtheta_pert = dtheta * amp
    dqv_pert = qv_bump * amp
    return np.broadcast_to(dtheta_pert, grid.center_shape).copy(), \
        np.broadcast_to(dqv_pert, grid.center_shape).copy()


__all__ = ["BaseState", "build_base_state", "warm_bubble"]
