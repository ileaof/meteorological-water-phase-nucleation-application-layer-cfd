"""Thermodynamic helpers for the meteorological_flow package.

Pure-function conversions between the transported/ prognostic variables used by
the Boussinesq flow solver and the inputs required by the validated nucleation
kernel (``met_water_nucleation.un``): (T, P, p_v, |gradT|).

Saturation vapour pressures are taken from the engine's validated
``SaturationProperties`` (IAPWS Wagner for liquid, extended below the triple
point; Goff-Gratch for ice).  These curves are **not** reimplemented here -- the
engine is treated read-only, so the flow layer stays consistent with the
validated microphysics.

All quantities SI unless noted (RH in percent, theta in K).
"""
from __future__ import annotations

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


def theta_from_T(T, P, P0=P0_REF):
    """Potential temperature [K]: theta = T (P0/P)^(R_d/cp_d)."""
    return np.asarray(T) * (P0 / np.asarray(P)) ** RD_OVER_CP


def T_from_theta(theta, P, P0=P0_REF):
    """Temperature [K] from potential temperature: T = theta (P/P0)^(R_d/cp_d)."""
    return np.asarray(theta) * (np.asarray(P) / P0) ** RD_OVER_CP


def p_v_from_q_v(q_v, P):
    """Water-vapour partial pressure [Pa] from mixing ratio q_v [kg/kg] and
    total pressure P [Pa].

    p_v = q_v P / (eps + (1 - eps) q_v)   (exact for the ideal-gas mixture).
    """
    q_v = np.asarray(q_v, dtype=float)
    P = np.asarray(P, dtype=float)
    return q_v * P / (EPS + (1.0 - EPS) * q_v)


def q_v_from_p_v(p_v, P):
    """Inverse of :func:`p_v_from_q_v`: q_v [kg/kg] from p_v [Pa], P [Pa]."""
    p_v = np.asarray(p_v, dtype=float)
    P = np.asarray(P, dtype=float)
    return EPS * p_v / (P - (1.0 - EPS) * p_v)


# The engine's Psat_* use scalar math.exp, so they cannot take arrays directly.
# We vectorise them here (the underlying validated equations are unchanged).
_psat_water_v = np.vectorize(
    lambda t: SaturationProperties.Psat_water(float(t), extended=True), otypes=[float])
_psat_ice_v = np.vectorize(
    lambda t: SaturationProperties.Psat_ice(float(t)), otypes=[float])


def psat_water(T):
    """Saturation vapour pressure over liquid water [Pa] (IAPWS Wagner,
    extended below the triple point -- engine, read-only). Vectorised."""
    T = np.asarray(T, dtype=float)
    if T.ndim == 0:
        return float(SaturationProperties.Psat_water(float(T), extended=True))
    return _psat_water_v(T)


def psat_ice(T):
    """Saturation vapour pressure over ice [Pa] (Goff-Gratch -- engine). Vectorised."""
    T = np.asarray(T, dtype=float)
    if T.ndim == 0:
        return float(SaturationProperties.Psat_ice(float(T)))
    return _psat_ice_v(T)


def saturation_ratios(T, p_v):
    """Return (S_w, S_i, RH_w_percent, RH_i_percent) for T [K], p_v [Pa].

    S_* = p_v / P_sat,phase(T); RH_* = 100 * S_*.  Vectorised; works for scalars.
    """
    T = np.asarray(T, dtype=float)
    p_v = np.asarray(p_v, dtype=float)
    pw = psat_water(T)
    pi = psat_ice(T)
    S_w = p_v / pw
    S_i = p_v / pi
    RH_w = 100.0 * S_w
    RH_i = 100.0 * S_i
    return S_w, S_i, RH_w, RH_i


def density_dry(P, T):
    """Dry-air density [kg m^-3] from the ideal gas law."""
    return np.asarray(P, dtype=float) / (R_d * np.asarray(T, dtype=float))


def density_moist(P, T, q_v):
    """Moist-air density [kg m^-3] (Boussinesq reference uses dry; this is the
    virtual-temperature form rho = P/(R_d T_v), T_v = T(1 + 0.61 q_v))."""
    T = np.asarray(T, dtype=float)
    q_v = np.asarray(q_v, dtype=float)
    return np.asarray(P, dtype=float) / (R_d * T * (1.0 + 0.61 * q_v))


def terminal_velocity_ice(q_i, rho=1.0):
    """Very simple mass-weighted ice/graupel terminal fall speed [m/s] for
    Batch-2 sedimentation.  Placeholder linear law, documented as a
    parameterization (not a size-resolved terminal velocity)."""
    return 1.0 * np.sqrt(np.maximum(q_i, 0.0) / max(rho, 1e-12))


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