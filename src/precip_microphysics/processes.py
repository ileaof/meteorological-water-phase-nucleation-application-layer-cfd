"""Bulk-microphysics process rates (single-moment).

Each process is a pure function ``f(state, cfg, dt) -> list[Transfer]``.  A
:class:`Transfer` moves a mass mixing ratio ``dq`` (>=0) from one water species
to another; because every process is expressed as a *transfer* (never a raw
source/sink), total water is conserved by construction and the orchestrator
(:mod:`scheme`) derives the latent-heat sign from the phase-rank change
(vapour < liquid < ice).

Parameterization sources (coefficients live in :mod:`constants`, flagged
EMPIRICAL there):

* warm rain      -- Kessler (1969); rain evaporation Rutledge & Hobbs (1983).
* diffusional    -- Byers (1965) diffusional growth denominator A_K + A_D.
* collection     -- continuous bulk collection of cloud water by an exponential
                    precipitation category (Lin83; Rutledge & Hobbs 1983).
* freezing       -- Bigg (1953) immersion (bulk form, Reisner et al. 1998).
* aggregation    -- q_i -> q_s autoconversion threshold (Lin83).
* melting        -- ventilated bulk melting (Rutledge & Hobbs 1984).

Growth of the *initial* embryos is not done here -- that is the nucleation
source (:mod:`nucleation_source`), kept separate so aerosol activation and the
Eq.39 shifted-equilibrium model are not double counted.

Every function reads ``xp = st.xp`` (numpy or cupy; see
:class:`precip_microphysics.state.MicrophysicsState`) from the state it is
given, rather than hardcoding numpy, so a GPU-resident state stays resident
throughout -- only ``meteorological_flow.microphysics_coupling`` decides which
backend a state actually uses.
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from . import constants as C
from . import size_distributions as sd
from . import thermo as th

# src -> dst transfer of dq [kg/kg]; ``name`` labels the process for diagnostics.
Transfer = namedtuple("Transfer", "src dst dq name")

_N0 = {"snow": C.N0_s, "graupel": C.N0_g, "hail": C.N0_h}
_VA, _VB = C.VT_A, C.VT_B


def _arr(x, xp=np):
    return xp.asarray(x, dtype=float)


def _cap(dq, avail, xp=np):
    """Clip a transfer to be non-negative and not exceed the available source."""
    return xp.clip(_arr(dq, xp), 0.0, xp.maximum(_arr(avail, xp), 0.0))


def _diffusional_denominator(T, P, phase, xp=np):
    """A_K + A_D for diffusional growth/evaporation (Byers 1965).

    A_K = (L/(K_t T))(L/(R_v T) - 1)  (thermal-conduction term)
    A_D = R_v T / (D_v e_sat)          (vapour-diffusion term)
    """
    T = _arr(T, xp)
    if phase == "water":
        L = C.Lv
        es = th.psat_water(T, xp=xp)
    else:
        L = C.Ls
        es = th.psat_ice(T, xp=xp)
    A_K = (L / (C.K_THERM * T)) * (L / (C.R_v * T) - 1.0)
    A_D = C.R_v * T / (C.DIFF_VAPOR * xp.maximum(es, C.TINY))
    return A_K + A_D


def _ventilated_capacitance(N0, lam, a, b):
    """Integral [0.78/lambda^2 + 0.31 Sc^(1/3) sqrt(a/nu) Gamma((b+5)/2)
    lambda^(-(b+5)/2)] for an exponential distribution (Rutledge & Hobbs 1983).

    lam is a (possibly GPU-resident) array; the Gamma-function factor is a
    Python-scalar constant (only a, b depend on the category, not the cell),
    so this stays a plain array expression in whatever backend lam already is.
    """
    term1 = 0.78 / lam ** 2
    term2 = (0.31 * C.SC ** (1.0 / 3.0) * math.sqrt(a / C.NU_AIR)
             * math.gamma((b + 5.0) / 2.0) * lam ** (-(b + 5.0) / 2.0))
    return N0 * (term1 + term2)


# ---------------------------------------------------------------------------
# warm rain (Kessler 1969)
# ---------------------------------------------------------------------------
def condensation_adjustment(st, cfg, dt):
    """Saturation adjustment: relax supersaturation over water into cloud water
    (condensation) or evaporate cloud water in subsaturated air.  One-step
    analytic increment with latent-heat feedback on q_sat (Soong & Ogura 1973):

        dq = (q_v - q_sat) / (1 + L_v^2 q_sat / (c_p R_v T^2))
    """
    if not cfg.processes.condensation:
        return []
    xp = st.xp
    T, P = st.T, st.P
    qsat = th.qsat_water(T, P, xp=xp)
    denom = 1.0 + C.Lv ** 2 * qsat / (C.cp_d * C.R_v * _arr(T, xp) ** 2)
    dq = (_arr(st.qv, xp) - qsat) / denom
    cond = xp.where(dq > 0.0, xp.minimum(dq, _arr(st.qv, xp)), 0.0)
    evap = xp.where(dq < 0.0, xp.minimum(-dq, _arr(st.qc, xp)), 0.0)
    out = []
    if xp.any(cond > 0):
        out.append(Transfer("qv", "qc", cond, "condensation"))
    if xp.any(evap > 0):
        out.append(Transfer("qc", "qv", evap, "cloud_evaporation"))
    return out


def autoconversion(st, cfg, dt):
    """q_c -> q_r (Kessler): rate = k1 (q_c - q_c,crit)_+ ."""
    if not cfg.processes.autoconversion:
        return []
    xp = st.xp
    excess = xp.maximum(_arr(st.qc, xp) - C.KESSLER_QC_CRIT, 0.0)
    dq = _cap(C.KESSLER_K1 * excess * dt, st.qc, xp)
    return [Transfer("qc", "qr", dq, "autoconversion")] if xp.any(dq > 0) else []


def accretion(st, cfg, dt):
    """Rain collects cloud water (Kessler): rate = k2 q_c q_r^0.875."""
    if not cfg.processes.accretion:
        return []
    xp = st.xp
    rate = C.KESSLER_K2 * _arr(st.qc, xp) * xp.maximum(_arr(st.qr, xp), 0.0) ** 0.875
    dq = _cap(rate * dt, st.qc, xp)
    return [Transfer("qc", "qr", dq, "accretion")] if xp.any(dq > 0) else []


def rain_evaporation(st, cfg, dt):
    """q_r -> q_v in subsaturated air (ventilated, Rutledge & Hobbs 1983)."""
    if not cfg.processes.rain_evaporation:
        return []
    xp = st.xp
    T, P = st.T, st.P
    Sw = th.saturation_ratio_water(st.qv, T, P, xp=xp)
    sub = Sw < 1.0
    if not xp.any(sub & (_arr(st.qr, xp) > C.QSMALL)):
        return []
    lam = sd.lambda_slope(st.qr, st.rho, "rain", xp)
    vent = _ventilated_capacitance(C.N0_r, lam, _VA["rain"], _VB["rain"])
    denom = _diffusional_denominator(T, P, "water", xp)
    # dq_r/dt = (2 pi / rho) (S_w - 1) / (A_K+A_D) * vent   (<0 when subsaturated)
    rate = (2.0 * math.pi / xp.maximum(_arr(st.rho, xp), C.TINY)) * (Sw - 1.0) / denom * vent
    evap = xp.where(xp.isfinite(rate) & (Sw < 1.0), -rate, 0.0)   # positive magnitude
    # do not evaporate past saturation
    to_sat = xp.maximum(th.qsat_water(T, P, xp=xp) - _arr(st.qv, xp), 0.0)
    dq = _cap(xp.minimum(evap * dt, to_sat), st.qr, xp)
    return [Transfer("qr", "qv", dq, "rain_evaporation")] if xp.any(dq > 0) else []


# ---------------------------------------------------------------------------
# collection kernel (continuous bulk collection of cloud water)
# ---------------------------------------------------------------------------
def _collect_cloud(qc, q_col, rho, cat, E, xp=np):
    """dq_c/dt for continuous collection of cloud water by precip category
    ``cat`` with exponential distribution:

        dq_c/dt = (pi/4) E q_c N0 a Gamma(3+b)/lambda^(3+b) (rho0/rho)^0.5
    """
    qc = _arr(qc, xp)
    valid = _arr(q_col, xp) > C.QSMALL
    lam = sd.lambda_slope(q_col, rho, cat, xp)
    a, b = _VA[cat], _VB[cat]
    kernel = (math.pi / 4.0) * E * _N0[cat] * a * math.gamma(3.0 + b) / lam ** (3.0 + b)
    dens = (C.RHO0_VT / xp.maximum(_arr(rho, xp), C.TINY)) ** 0.5
    rate = xp.where(valid, kernel * dens * qc, 0.0)
    return xp.where(xp.isfinite(rate), rate, 0.0)


def riming(st, cfg, dt):
    """Supercooled cloud water rimes onto snow, graupel and hail (liquid->ice,
    releases L_f).  Only where T < T0."""
    if not cfg.processes.riming:
        return []
    xp = st.xp
    cold = _arr(st.T, xp) < C.T0
    if not xp.any(cold):
        return []
    out = []
    remaining = _arr(st.qc, xp).copy()
    for cat, dst in (("snow", "qs"), ("graupel", "qg"), ("hail", "qh")):
        rate = _collect_cloud(remaining, getattr(st, dst), st.rho, cat, C.E_COLLECT, xp)
        dq = _cap(xp.where(cold, rate * dt, 0.0), remaining, xp)
        if xp.any(dq > 0):
            out.append(Transfer("qc", dst, dq, f"riming_{cat}"))
            remaining = remaining - dq
    return out


# ---------------------------------------------------------------------------
# freezing (Bigg 1953, bulk immersion)
# ---------------------------------------------------------------------------
def _bigg_rate(T, xp=np):
    cold = _arr(T, xp) < C.T0
    r = C.BIGG_B * (xp.exp(C.BIGG_A * (C.T0 - _arr(T, xp))) - 1.0)
    return xp.where(cold, xp.maximum(r, 0.0), 0.0)


def immersion_freezing(st, cfg, dt):
    """Cloud droplets freeze to cloud ice; supercooled rain freezes to graupel
    (Bigg volume law, bulk form).  liquid -> ice, releases L_f."""
    if not cfg.processes.ice_nucleation:
        return []
    xp = st.xp
    jr = _bigg_rate(st.T, xp)
    # normalise by a reference droplet volume so the bulk rate is a 1/s scale
    frac = 1.0 - xp.exp(-jr * dt * 1.0e-6)      # EMPIRICAL bulk volume scale
    frac = xp.clip(frac, 0.0, 1.0)
    out = []
    dqc = _cap(_arr(st.qc, xp) * frac, st.qc, xp)
    if xp.any(dqc > 0):
        out.append(Transfer("qc", "qi", dqc, "immersion_freezing_cloud"))
    dqr = _cap(_arr(st.qr, xp) * frac, st.qr, xp)
    if xp.any(dqr > 0):
        out.append(Transfer("qr", "qg", dqr, "freezing_rain"))
    return out


# ---------------------------------------------------------------------------
# deposition / sublimation (Bergeron-Findeisen; ice & snow)
# ---------------------------------------------------------------------------
def deposition(st, cfg, dt):
    """Vapour deposition onto / sublimation from cloud ice and snow (diffusional
    growth over ice).  vapour <-> ice, L_s."""
    if not cfg.processes.deposition:
        return []
    xp = st.xp
    T, P = st.T, st.P
    cold = _arr(T, xp) < C.T0
    Si = th.saturation_ratio_ice(st.qv, T, P, xp=xp)
    denom = _diffusional_denominator(T, P, "ice", xp)
    out = []
    # snow deposition (ventilated capacitance)
    for cat, sp in (("snow", "qs"),):
        has = _arr(getattr(st, sp), xp) > C.QSMALL
        lam = sd.lambda_slope(getattr(st, sp), st.rho, cat, xp)
        vent = _ventilated_capacitance(_N0[cat], lam, _VA[cat], _VB[cat])
        rate = (2.0 * math.pi / xp.maximum(_arr(st.rho, xp), C.TINY)) * (Si - 1.0) / denom * vent
        rate = xp.where(cold & has & xp.isfinite(rate), rate, 0.0)
        dep = xp.where(rate > 0, rate * dt, 0.0)
        sub = xp.where(rate < 0, -rate * dt, 0.0)
        # deposition limited by available vapour above ice saturation
        to_sat = xp.maximum(_arr(st.qv, xp) - th.qsat_ice(T, P, xp=xp), 0.0)
        dep = _cap(xp.minimum(dep, to_sat), st.qv, xp)
        sub = _cap(sub, getattr(st, sp), xp)
        if xp.any(dep > 0):
            out.append(Transfer("qv", sp, dep, f"deposition_{cat}"))
        if xp.any(sub > 0):
            out.append(Transfer(sp, "qv", sub, f"sublimation_{cat}"))
    # cloud-ice deposition (simple capacitance ~ crystal radius, monodisperse)
    has_i = _arr(st.qi, xp) > C.QSMALL
    ri = sd.ice_radius(st.qi, st.rho, xp=xp)
    Ni = sd.NI_DEFAULT
    cap_i = 4.0 * math.pi * xp.nan_to_num(ri) * Ni     # bulk capacitance
    rate_i = (1.0 / xp.maximum(_arr(st.rho, xp), C.TINY)) * (Si - 1.0) / denom * cap_i
    rate_i = xp.where(cold & has_i & xp.isfinite(rate_i), rate_i, 0.0)
    to_sat = xp.maximum(_arr(st.qv, xp) - th.qsat_ice(T, P, xp=xp), 0.0)
    dep_i = _cap(xp.minimum(xp.where(rate_i > 0, rate_i * dt, 0.0), to_sat), st.qv, xp)
    sub_i = _cap(xp.where(rate_i < 0, -rate_i * dt, 0.0), st.qi, xp)
    if xp.any(dep_i > 0):
        out.append(Transfer("qv", "qi", dep_i, "deposition_ice"))
    if xp.any(sub_i > 0):
        out.append(Transfer("qi", "qv", sub_i, "sublimation_ice"))
    return out


def aggregation(st, cfg, dt):
    """Cloud ice aggregates into snow (autoconversion threshold, Lin83)."""
    if not cfg.processes.aggregation:
        return []
    xp = st.xp
    excess = xp.maximum(_arr(st.qi, xp) - C.QI_CRIT_SNOW, 0.0)
    dq = _cap(C.QI_AUTO_RATE * excess * dt, st.qi, xp)
    return [Transfer("qi", "qs", dq, "aggregation")] if xp.any(dq > 0) else []


def graupel_conversion(st, cfg, dt):
    """Heavily rimed snow converts to graupel (Lin83 threshold)."""
    if not cfg.processes.graupel_conversion:
        return []
    xp = st.xp
    excess = xp.maximum(_arr(st.qs, xp) - C.QS_CRIT_GRAUPEL, 0.0)
    dq = _cap(C.RIME_TO_GRAUPEL_RATE * excess * dt, st.qs, xp)
    return [Transfer("qs", "qg", dq, "snow_to_graupel")] if xp.any(dq > 0) else []


# ---------------------------------------------------------------------------
# melting (ventilated, Rutledge & Hobbs 1984): frozen -> rain above 0 degC
# ---------------------------------------------------------------------------
def _melt(st, cfg, dt, cat, sp):
    xp = st.xp
    warm = _arr(st.T, xp) > C.T0
    has = _arr(getattr(st, sp), xp) > C.QSMALL
    if not xp.any(warm & has):
        return None
    lam = sd.lambda_slope(getattr(st, sp), st.rho, cat, xp)
    vent = _ventilated_capacitance(_N0[cat], lam, _VA[cat], _VB[cat])
    # dq/dt = (2 pi / (rho Lf)) K_t (T - T0) * vent    (>0 above 0 degC)
    rate = (2.0 * math.pi / (xp.maximum(_arr(st.rho, xp), C.TINY) * C.Lf)) \
        * C.K_THERM * (_arr(st.T, xp) - C.T0) * vent
    rate = xp.where(warm & has & xp.isfinite(rate), rate, 0.0)
    dq = _cap(rate * dt, getattr(st, sp), xp)
    return Transfer(sp, "qr", dq, f"melting_{cat}") if xp.any(dq > 0) else None


def snow_melting(st, cfg, dt):
    if not cfg.processes.graupel_melting:      # shares the melting switch family
        return []
    t = _melt(st, cfg, dt, "snow", "qs")
    return [t] if t else []


def graupel_melting(st, cfg, dt):
    if not cfg.processes.graupel_melting:
        return []
    t = _melt(st, cfg, dt, "graupel", "qg")
    return [t] if t else []


# ---------------------------------------------------------------------------
# hail (embryo + growth + melting)
# ---------------------------------------------------------------------------
def hail_embryo(st, cfg, dt):
    """Graupel converts to a hail embryo where the environment supports wet
    growth: graupel present, substantial supercooled LWC, and a strong updraft
    lofting the embryo into the growth zone (gates from constants)."""
    if not cfg.processes.hail_growth:
        return []
    xp = st.xp
    lwc = _arr(st.qc, xp) * _arr(st.rho, xp)                 # kg/m^3 supercooled water
    cold = _arr(st.T, xp) < C.T0
    gate = (cold & (_arr(st.qg, xp) > C.QG_CRIT_HAIL)
            & (lwc > C.HAIL_LWC_CRIT) & (_arr(st.w, xp) > C.HAIL_UPDRAFT_CRIT))
    excess = xp.where(gate, xp.maximum(_arr(st.qg, xp) - C.QG_CRIT_HAIL, 0.0), 0.0)
    dq = _cap(C.RIME_TO_GRAUPEL_RATE * excess * dt, st.qg, xp)
    return [Transfer("qg", "qh", dq, "hail_embryo")] if xp.any(dq > 0) else []


def hail_melting(st, cfg, dt):
    """Hail melts to rain above 0 degC (typically below the freezing level)."""
    if not cfg.processes.hail_melting:
        return []
    t = _melt(st, cfg, dt, "hail", "qh")
    return [t] if t else []


# ordered process list applied by the scheme each substep
PROCESS_ORDER = (
    condensation_adjustment,
    deposition,
    immersion_freezing,
    autoconversion,
    accretion,
    riming,
    aggregation,
    graupel_conversion,
    hail_embryo,
    snow_melting,
    graupel_melting,
    hail_melting,
    rain_evaporation,
)


__all__ = ["Transfer", "PROCESS_ORDER"] + [f.__name__ for f in PROCESS_ORDER]
