"""Thermodynamic helpers for the meteorological_flow package.

Pure-function conversions between the transported/ prognostic variables used by
the Boussinesq flow solver and the inputs required by the validated nucleation
kernel (``met_water_nucleation.un``): (T, P, p_v, |gradT|).

Saturation vapour pressures are taken from the engine's validated
``SaturationProperties`` (IAPWS Wagner for liquid, extended below the triple
point; Goff-Gratch for ice).  These curves are **not** reimplemented here -- the
engine is treated read-only, so the flow layer stays consistent with the
validated microphysics; ``psat_water``/``psat_ice`` below are a true vectorised
(ufunc) re-expression of the exact same closed-form equations the engine
computes scalar-wise (see the coefficient citation there), not a new
derivation -- pinned equal to the engine's per-element output by
``tests/test_thermodynamics_vectorization.py``.

Every function accepts an optional ``xp=`` array-module override (default
``numpy``) so the flow solver can call these on GPU-resident (CuPy) arrays
without a host round-trip; scalar/NumPy callers elsewhere in the codebase are
unaffected since the default is unchanged.

All quantities SI unless noted (RH in percent, theta in K).
"""
from __future__ import annotations

import math

import numpy as np

import met_water_nucleation as M

# --- Physical constants (standard atmospheric values) -----------------------
R_d = 287.058          # J kg^-1 K^-1  specific gas constant for dry air
cp_d = 1005.0          # J kg^-1 K^-1  specific heat of dry air at const pressure
g0 = 9.81              # m s^-2        gravitational acceleration
EPS = 0.62197          # ratio M_w / M_d  (water/dry-air molar masses)
RD_OVER_CP = R_d / cp_d        # ~0.2854, exponent in theta<->T
P0_REF = 100000.0     # Pa, reference pressure for potential temperature

# Latent heats [J kg^-1] (constant first-order values; T-dep upgrade is Batch 2)
Lv = 2.501e6           # vaporisation
Ls = 2.836e6           # sublimation (deposition)
Lf = Ls - Lv           # fusion (freezing) ~3.35e5

SaturationProperties = M.SaturationProperties


def theta_from_T(T, P, P0=P0_REF, xp=np):
    """Potential temperature [K]: theta = T (P0/P)^(R_d/cp_d)."""
    return xp.asarray(T) * (P0 / xp.asarray(P)) ** RD_OVER_CP


def T_from_theta(theta, P, P0=P0_REF, xp=np):
    """Temperature [K] from potential temperature: T = theta (P/P0)^(R_d/cp_d)."""
    return xp.asarray(theta) * (xp.asarray(P) / P0) ** RD_OVER_CP


def p_v_from_q_v(q_v, P, xp=np):
    """Water-vapour partial pressure [Pa] from mixing ratio q_v [kg/kg] and
    total pressure P [Pa].

    p_v = q_v P / (eps + (1 - eps) q_v)   (exact for the ideal-gas mixture).
    """
    q_v = xp.asarray(q_v, dtype=float)
    P = xp.asarray(P, dtype=float)
    return q_v * P / (EPS + (1.0 - EPS) * q_v)


def q_v_from_p_v(p_v, P, xp=np):
    """Inverse of :func:`p_v_from_q_v`: q_v [kg/kg] from p_v [Pa], P [Pa]."""
    p_v = xp.asarray(p_v, dtype=float)
    P = xp.asarray(P, dtype=float)
    return EPS * p_v / (P - (1.0 - EPS) * p_v)


# --- Saturation vapour pressure: true vectorised (ufunc) closed forms -------
# The engine's SaturationProperties.Psat_water/Psat_ice
# (src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/
# unified_h2o_nucleation_climate.py:258-298) evaluate these exact formulas
# scalar-wise with math.exp/math.log10. Wrapping them with np.vectorize (the
# previous approach here) is a hidden Python-level per-element loop -- slow on
# a full 3-D field, and it can never accept a CuPy array, which would force a
# silent GPU<->host round-trip every call. Reproduced below as real ufunc
# expressions (xp.exp/xp.log10/xp.power) using the SAME published coefficients
# the engine uses -- not re-derived -- with array support (incl. GPU) as a
# consequence, not an independent implementation to keep in sync by hand.
_WAGNER_A = (-7.85951783, 1.84408259, -11.7866497, 22.6807411,
             -15.9618719, 1.80122502)                          # IAPWS Wagner
_WAGNER_B = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
_TC = 647.096          # K,  critical temperature (== engine's Tc)
_PC = 22.064e6         # Pa, critical pressure    (== engine's Pc)
_PT = 611.657          # Pa, triple-point pressure (== engine's Pt)
_GG_TT = 273.16        # K,  triple-point temperature (== engine's Tt / Goff-Gratch anchor)
_GG_LOGE_REF = math.log10(_PT / 100.0)   # log10(Pt in hPa) -- anchors the ice curve at Pt


def psat_water(T, xp=np):
    """Saturation vapour pressure over liquid water [Pa] (IAPWS Wagner,
    extended below the triple point -- coefficients from the engine, which
    stays read-only). Fully vectorised; works for scalars and arrays."""
    T = xp.asarray(T, dtype=float)
    tau = 1.0 - T / _TC
    s = sum(a * tau ** b for a, b in zip(_WAGNER_A, _WAGNER_B))
    return _PC * xp.exp((_TC / T) * s)


def psat_ice(T, xp=np):
    """Saturation vapour pressure over ice [Pa] (Goff-Gratch, coefficients
    from the engine). Fully vectorised; works for scalars and arrays."""
    T = xp.asarray(T, dtype=float)
    ratio = _GG_TT / T
    log10e = (-9.09718 * (ratio - 1.0)
              - 3.56654 * xp.log10(ratio)
              + 0.876793 * (1.0 - T / _GG_TT)
              + _GG_LOGE_REF)
    return 100.0 * xp.power(10.0, log10e)   # hPa -> Pa


def saturation_ratios(T, p_v, xp=np):
    """Return (S_w, S_i, RH_w_percent, RH_i_percent) for T [K], p_v [Pa].

    S_* = p_v / P_sat,phase(T); RH_* = 100 * S_*.  Vectorised; works for scalars.
    """
    T = xp.asarray(T, dtype=float)
    p_v = xp.asarray(p_v, dtype=float)
    pw = psat_water(T, xp=xp)
    pi = psat_ice(T, xp=xp)
    S_w = p_v / pw
    S_i = p_v / pi
    RH_w = 100.0 * S_w
    RH_i = 100.0 * S_i
    return S_w, S_i, RH_w, RH_i


def density_dry(P, T, xp=np):
    """Dry-air density [kg m^-3] from the ideal gas law."""
    return xp.asarray(P, dtype=float) / (R_d * xp.asarray(T, dtype=float))


def density_moist(P, T, q_v, xp=np):
    """Moist-air density [kg m^-3] (Boussinesq reference uses dry; this is the
    virtual-temperature form rho = P/(R_d T_v), T_v = T(1 + 0.61 q_v))."""
    T = xp.asarray(T, dtype=float)
    q_v = xp.asarray(q_v, dtype=float)
    return xp.asarray(P, dtype=float) / (R_d * T * (1.0 + 0.61 * q_v))


def terminal_velocity_ice(q_i, rho=1.0, xp=np):
    """Very simple mass-weighted ice/graupel terminal fall speed [m/s] for
    Batch-2 sedimentation.  Placeholder linear law, documented as a
    parameterization (not a size-resolved terminal velocity)."""
    return 1.0 * xp.sqrt(xp.maximum(q_i, 0.0) / max(rho, 1e-12))


__all__ = [
    "EPS",
    "P0_REF",
    "RD_OVER_CP",
    "Lf",
    "Ls",
    "Lv",
    "R_d",
    "SaturationProperties",
    "T_from_theta",
    "cp_d",
    "density_dry",
    "density_moist",
    "g0",
    "p_v_from_q_v",
    "psat_ice",
    "psat_water",
    "q_v_from_p_v",
    "saturation_ratios",
    "terminal_velocity_ice",
    "theta_from_T",
]