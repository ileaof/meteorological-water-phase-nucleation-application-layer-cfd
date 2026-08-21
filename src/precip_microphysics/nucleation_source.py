"""Nucleation -> hydrometeor embryo source.

Bridges the validated second-order nucleation kernel to the bulk microphysics.
For each parcel the kernel supplies the nucleation *rate* J [m^-3 s^-1]
(favourability + speed); this module converts it into an embryo mass source for
cloud water (q_c) and cloud ice (q_i):

    N_expected = J * dV * dt            (extensive count over a cell/timestep)
    N_intensive = J * dt                (per m^3)

Two conversions are provided:

* **deterministic (mean-field)** -- ``N_intensive = J dt``;
* **Poisson (stochastic)**       -- draw ``Poisson(J dV dt)`` then divide by dV.

Because the kernel rate is astronomically large under supersaturation
(log10 I ~ 50), the *droplet/crystal number* is not J dt but the number the
environment can activate: it is capped by the available cloud-condensation /
ice-nucleating particle concentration (the physically correct single-moment
closure), and the resulting mass source is then **limited by the available
supersaturated vapour**.  ``N_expected`` is still reported for transparency.

Double counting is avoided by ``cfg.activation_pathway``:

* ``eq39``        -- droplet/ice number from the kernel rate (capped by CCN/IN);
* ``ccn``         -- Twomey CCN activation spectrum (kernel used only as a gate);
* ``homogeneous`` -- homogeneous-only (no heterogeneous IN for ice).

Exactly one pathway contributes, so aerosol activation and the Eq.39
shifted-equilibrium model are never summed.
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as C
from . import thermo as th
from .processes import Transfer

NC_MAX = 1.0e9    # m^-3, max activatable droplet number (~1000 cm^-3, maritime->continental)
NI_MAX = 1.0e6    # m^-3, max ice-crystal number (~1 L^-1)
# Twomey CCN activation spectrum N = C_ccn * s^k (s = supersaturation percent)
TWOMEY_C = 1.0e8  # m^-3
TWOMEY_K = 0.6


def _arr(x):
    return np.asarray(x, dtype=float)


def _fletcher_IN(T):
    """Heterogeneous ice-nucleating-particle number [m^-3] (Fletcher 1962)."""
    return C.FLETCHER_N0 * np.exp(C.FLETCHER_BETA * (C.T0 - _arr(T)))


def embryo_source(st, cfg, dt, cell_volume, J_liquid=None, J_ice=None, rng=None):
    """Return (transfers, diagnostics) for the nucleation embryo source.

    ``J_liquid`` / ``J_ice`` are the kernel nucleation rates [m^-3 s^-1] (finite
    where nucleation solved, else None/NaN).  ``cell_volume`` = dV [m^3].
    """
    T, P = st.T, st.P
    rho = _arr(st.rho)
    Sw = th.saturation_ratio_water(st.qv, T, P)
    Si = th.saturation_ratio_ice(st.qv, T, P)
    transfers, diag = [], {}

    # ---- liquid embryos ----
    r_l = cfg.embryo_radius_liquid
    m_l = (4.0 / 3.0) * math.pi * r_l ** 3 * C.rho_w        # kg per droplet
    sw_super = Sw > 1.0
    if np.any(sw_super):
        if cfg.activation_pathway in ("eq39", "homogeneous") and J_liquid is not None:
            Jl = np.where(np.isfinite(_arr(J_liquid)), _arr(J_liquid), 0.0)
            N_exp = Jl * dt                                  # intensive [m^-3]
            if cfg.stochastic_nucleation and rng is not None:
                lam = np.clip(Jl * cell_volume * dt, 0.0, 1.0e18)
                N_int = rng.poisson(lam) / max(cell_volume, C.TINY)
            else:
                N_int = N_exp
        else:  # CCN pathway (Twomey)
            s_pct = np.maximum((Sw - 1.0) * 100.0, 0.0)
            N_int = TWOMEY_C * s_pct ** TWOMEY_K
            N_exp = N_int
        N_act = np.where(sw_super, np.minimum(N_int, NC_MAX), 0.0)
        dq_raw = N_act * m_l / np.maximum(rho, C.TINY)
        avail = np.maximum(_arr(st.qv) - th.qsat_water(T, P), 0.0) if cfg.vapour_limited \
            else dq_raw
        dq = np.clip(np.minimum(dq_raw, avail), 0.0, None)
        if np.any(dq > 0):
            transfers.append(Transfer("qv", "qc", dq, "nucleation_liquid"))
        diag["N_expected_liquid"] = np.where(sw_super, N_exp * cell_volume, 0.0)
        diag["N_activated_liquid"] = dq * np.maximum(rho, C.TINY) / m_l

    # ---- ice embryos ----
    r_i = cfg.embryo_radius_ice
    m_i = (4.0 / 3.0) * math.pi * r_i ** 3 * C.rho_i
    cold = _arr(T) < C.T0
    si_super = (Si > 1.0) & cold
    if np.any(si_super):
        if cfg.activation_pathway == "eq39" and J_ice is not None:
            Ji = np.where(np.isfinite(_arr(J_ice)), _arr(J_ice), 0.0)
            N_exp = Ji * dt
            if cfg.stochastic_nucleation and rng is not None:
                lam = np.clip(Ji * cell_volume * dt, 0.0, 1.0e18)
                N_int = rng.poisson(lam) / max(cell_volume, C.TINY)
            else:
                N_int = N_exp
        else:  # heterogeneous IN spectrum (Fletcher) or homogeneous gate
            N_int = _fletcher_IN(T)
            N_exp = N_int
        N_act = np.where(si_super, np.minimum(N_int, NI_MAX), 0.0)
        dq_raw = N_act * m_i / np.maximum(rho, C.TINY)
        avail = np.maximum(_arr(st.qv) - th.qsat_ice(T, P), 0.0) if cfg.vapour_limited \
            else dq_raw
        dq = np.clip(np.minimum(dq_raw, avail), 0.0, None)
        if np.any(dq > 0):
            transfers.append(Transfer("qv", "qi", dq, "nucleation_ice"))
        diag["N_expected_ice"] = np.where(si_super, N_exp * cell_volume, 0.0)
        diag["N_activated_ice"] = dq * np.maximum(rho, C.TINY) / m_i

    return transfers, diag


__all__ = ["embryo_source", "NC_MAX", "NI_MAX"]
