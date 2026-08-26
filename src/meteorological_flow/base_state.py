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
    u0: np.ndarray = None   # base zonal wind [m/s]             (nz,)
    v0: np.ndarray = None   # base meridional wind [m/s]        (nz,)

    def __post_init__(self):
        nz = np.asarray(self.zc).size
        if self.u0 is None:
            self.u0 = np.zeros(nz)
        if self.v0 is None:
            self.v0 = np.zeros(nz)

    def field(self, arr1d, shape, xp=np):
        """Broadcast a (nz,) base profile to the (nx,ny,nz) grid.

        ``xp`` lets the caller request a GPU-resident (CuPy) field; the
        1-D sounding itself is always built on the host (see module docstring)
        and transferred once here, since it is tiny (nz elements)."""
        return xp.broadcast_to(xp.asarray(arr1d).reshape(1, 1, -1), shape).copy()


def build_base_state(grid, *, T_sfc=301.0, p_sfc=101325.0, RH_sfc=0.85,
                     dtheta_dz=3.2e-3, q_scale_height=3400.0, RH_min=0.45,
                     z_trop=12000.0, dtheta_dz_strat=1.6e-2) -> BaseState:
    """Return a hydrostatic conditionally-unstable :class:`BaseState`.

    ``dtheta_dz`` [K/m] sets the tropospheric stratification; above ``z_trop``
    the (much larger) ``dtheta_dz_strat`` gives a stable stratosphere so a lifted
    parcel is capped (a physical equilibrium level / finite CAPE).  For domains
    shallower than ``z_trop`` (e.g. the 10 km storm) the stratosphere is simply
    out of the domain and the profile is unchanged.
    """
    # the hydrostatic sounding is a small (nz,) 1-D calculation with scalar
    # Brent-style fixed-point loops -- deliberately CPU-only regardless of the
    # grid's backend (see base_state.py's module docstring / docs/architecture.md);
    # grid.zc may be GPU-resident, so it is pulled to the host here, once.
    zc = np.asarray(grid.backend.to_cpu(grid.zc), dtype=float)
    nz = zc.size

    theta_sfc = float(th.theta_from_T(T_sfc, p_sfc, th.P0_REF))
    theta0 = np.where(
        zc <= z_trop,
        theta_sfc + dtheta_dz * zc,
        theta_sfc + dtheta_dz * z_trop + dtheta_dz_strat * (zc - z_trop))

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
    xp = grid.xp
    x_c = 0.5 * grid.Lx if x_c is None else x_c
    y_c = 0.5 * grid.Ly if y_c is None else y_c
    X = grid.xc.reshape(-1, 1, 1)
    Y = grid.yc.reshape(1, -1, 1)
    Z = grid.zc.reshape(1, 1, -1)
    r2 = ((X - x_c) / radius) ** 2 + ((Y - y_c) / radius) ** 2 + ((Z - z_c) / z_radius) ** 2
    amp = xp.exp(-r2)
    dtheta_pert = dtheta * amp
    dqv_pert = qv_bump * amp
    return xp.broadcast_to(dtheta_pert, grid.center_shape).copy(), \
        xp.broadcast_to(dqv_pert, grid.center_shape).copy()


def weisman_klemp(grid, *, theta_sfc=300.0, theta_tr=343.0, T_tr=213.0,
                  z_tr=12000.0, qv_sfc=0.014, p_sfc=100000.0,
                  u_shear=0.0, u_half=3000.0) -> BaseState:
    """Weisman & Klemp (1982, MWR 110, 504) analytic sounding -- the standard
    idealised deep-convection reference (CAPE ~ 2000-2500 J/kg).

        theta(z) = theta_sfc + (theta_tr-theta_sfc)(z/z_tr)^(5/4),  z <= z_tr
                 = theta_tr exp[ g/(c_p T_tr) (z - z_tr) ],         z >  z_tr
        RH(z)    = 1 - 0.75 (z/z_tr)^(5/4)  (0.25 above z_tr),
        q_v(z)   = min(q_v,sfc, RH * q_sat)   (constant-mixing-ratio boundary layer)

    Optional unidirectional shear ``u_shear`` [m/s] ramped over ``u_half`` [m].
    """
    # see build_base_state: this sounding calculation is deliberately
    # CPU-only; grid.zc may be GPU-resident.
    z = np.asarray(grid.backend.to_cpu(grid.zc), dtype=float)
    r = np.clip(z / z_tr, 0.0, None)
    theta0 = np.where(z <= z_tr,
                      theta_sfc + (theta_tr - theta_sfc) * r ** 1.25,
                      theta_tr * np.exp(th.g0 / (th.cp_d * T_tr) * (z - z_tr)))
    RH = np.where(z <= z_tr, np.maximum(1.0 - 0.75 * r ** 1.25, 0.25), 0.25)

    # hydrostatic integration with the WK moisture rule
    nz = z.size
    p0 = np.empty(nz)
    T0 = np.empty(nz)
    qv0 = np.empty(nz)
    p_prev, z_prev = p_sfc, 0.0
    for k in range(nz):
        dz = z[k] - z_prev
        p_new = p_prev
        for _ in range(3):
            Tk = float(th.T_from_theta(theta0[k], p_new, th.P0_REF))
            qsat = float(th.q_v_from_p_v(th.psat_water(Tk), p_new))
            qv = min(qv_sfc, RH[k] * qsat)
            Tv = Tk * (1.0 + 0.61 * qv)
            p_new = p_prev * float(np.exp(-th.g0 * dz / (th.R_d * Tv)))
        p0[k] = p_new
        T0[k] = float(th.T_from_theta(theta0[k], p0[k], th.P0_REF))
        qv0[k] = min(qv_sfc, RH[k] * float(th.q_v_from_p_v(th.psat_water(T0[k]), p0[k])))
        p_prev, z_prev = p_new, z[k]

    rho0 = p0 / (th.R_d * T0 * (1.0 + 0.61 * qv0))
    u0 = u_shear * np.tanh(z / u_half)                 # unidirectional shear
    return BaseState(zc=z, theta0=theta0, qv0=qv0, p0=p0, T0=T0, rho0=rho0,
                     u0=u0, v0=np.zeros(nz))


# ---------------------------------------------------------------------------
# parcel ascent + sounding diagnostics (CAPE, CIN, LCL, LFC, EL, N^2, shear)
# ---------------------------------------------------------------------------
def _qsat(T, p):
    """Saturation mixing ratio over water [kg/kg] (engine, read-only)."""
    return float(th.q_v_from_p_v(th.psat_water(float(T)), float(p)))


def _moist_lapse(T, p):
    """Pseudo-adiabatic (saturated) lapse rate -dT/dz [K/m] (Bolton form)."""
    es = float(th.psat_water(float(T)))
    rs = th.EPS * es / max(p - es, 1.0)
    num = th.g0 * (1.0 + th.Lv * rs / (th.R_d * T))
    den = th.cp_d + th.Lv ** 2 * rs * th.EPS / (th.R_d * T ** 2)
    return num / den


def parcel_profile(base: "BaseState"):
    """Lift a surface parcel dry- then pseudo-moist-adiabatically on the model
    levels.  Returns (T_parcel[z], qv_parcel[z], lcl_index or None)."""
    z, p = np.asarray(base.zc), np.asarray(base.p0)
    nz = z.size
    theta_sfc = float(th.theta_from_T(base.T0[0], p[0], th.P0_REF))
    qv_sfc = float(base.qv0[0])
    Tp = np.empty(nz)
    qvp = np.empty(nz)
    lcl = None
    for k in range(nz):
        if lcl is None:                              # unsaturated -> dry adiabat
            Tk = float(th.T_from_theta(theta_sfc, p[k], th.P0_REF))
            if qv_sfc >= _qsat(Tk, p[k]):
                lcl = k
                Tp[k] = Tk
                qvp[k] = _qsat(Tk, p[k])
            else:
                Tp[k] = Tk
                qvp[k] = qv_sfc
        else:                                        # saturated -> moist adiabat
            dz = z[k] - z[k - 1]
            Tp[k] = Tp[k - 1] - _moist_lapse(Tp[k - 1], p[k - 1]) * dz
            qvp[k] = _qsat(Tp[k], p[k])
    return Tp, qvp, lcl


def _cross_level(z, f, target, decreasing=True):
    """Linear-interpolated height where profile f crosses `target`."""
    f = np.asarray(f)
    for k in range(1, len(z)):
        a, b = f[k - 1], f[k]
        if (a - target) * (b - target) <= 0 and a != b:
            w = (target - a) / (b - a)
            return float(z[k - 1] + w * (z[k] - z[k - 1]))
    return None


def _bulk_shear(base, z1=0.0, z2=6000.0):
    """|V(z2) - V(z1)| bulk vertical wind shear [m/s]."""
    u = np.interp([z1, z2], base.zc, base.u0)
    v = np.interp([z1, z2], base.zc, base.v0)
    return float(np.hypot(u[1] - u[0], v[1] - v[0]))


def sounding_diagnostics(base: "BaseState") -> dict:
    """CAPE/CIN [J/kg], LCL/LFC/EL/freezing level [m], Brunt-Vaisala N^2(z) and
    0-6 km bulk shear for a reference sounding."""
    z, p = np.asarray(base.zc), np.asarray(base.p0)
    Tp, qvp, lcl_k = parcel_profile(base)
    Tv_p = Tp * (1.0 + 0.61 * qvp)                    # pseudo-adiabatic (no loading)
    Tv_e = np.asarray(base.T0) * (1.0 + 0.61 * np.asarray(base.qv0))
    B = th.g0 * (Tv_p - Tv_e) / Tv_e                  # parcel buoyancy [m/s^2]

    cape = cin = 0.0
    lfc_k = el_k = None
    if lcl_k is not None:
        # LFC: first level above the LCL where the parcel is positively buoyant.
        for k in range(lcl_k + 1, len(z)):
            if B[k] > 0.0:
                lfc_k = k
                break
        if lfc_k is not None:
            # CAPE: integrate positive buoyancy from the LFC up to the EL (the
            # first level above the LFC where buoyancy returns to <= 0).  Using
            # B[k] (not the layer mean) to detect the EL avoids a spurious
            # collapse EL==LFC when the layer straddling the LFC has a negative
            # mean (deep CIN just below); the area still uses the clipped mean.
            for k in range(lfc_k + 1, len(z)):
                dz = z[k] - z[k - 1]
                bk = 0.5 * (B[k] + B[k - 1])
                if B[k] > 0.0:
                    cape += max(bk, 0.0) * dz
                else:
                    cape += max(bk, 0.0) * dz          # partial last (positive) layer
                    el_k = k
                    break
            else:
                el_k = len(z) - 1                      # buoyant to the domain top
            for k in range(1, lfc_k + 1):              # CIN: negative area below LFC
                dz = z[k] - z[k - 1]
                bk = 0.5 * (B[k] + B[k - 1])
                if bk < 0.0:
                    cin += bk * dz

    dth = np.gradient(np.asarray(base.theta0), z)
    N2 = th.g0 / np.asarray(base.theta0) * dth
    return {
        "CAPE_J_kg": float(cape),
        "CIN_J_kg": float(cin),
        "LCL_m": (float(z[lcl_k]) if lcl_k is not None else None),
        "LFC_m": (float(z[lfc_k]) if lfc_k is not None else None),
        "EL_m": (float(z[el_k]) if el_k is not None else None),
        "freezing_level_m": _cross_level(z, base.T0, 273.15),
        "w_max_parcel_m_s": float(np.sqrt(2.0 * max(cape, 0.0))),   # sqrt(2 CAPE)
        "N2_1_s2": N2,
        "N2_mean_1_s2": float(np.mean(N2)),
        "shear_0_6km_m_s": _bulk_shear(base, 0.0, 6000.0),
        "parcel_T_K": Tp,
        "parcel_buoyancy_m_s2": B,
    }


__all__ = [
    "BaseState", "build_base_state", "weisman_klemp", "warm_bubble",
    "parcel_profile", "sounding_diagnostics",
]
