"""Single-moment size-distribution closure.

Each precipitating category (rain, snow, graupel, hail) is assumed to follow an
exponential (Marshall-Palmer) distribution ``N(D) = N0 exp(-lambda D)`` with a
fixed intercept ``N0`` (``constants.N0_*``).  From the prognostic mass mixing
ratio ``q`` and the air density the slope, number concentration, characteristic
radius and mass-weighted terminal velocity follow in closed form:

    mass:   rho_air q = N0 pi rho_x / lambda^4
      =>    lambda = (pi rho_x N0 / (rho_air q))^(1/4)
    number: N = N0 / lambda
    V_mass = a * Gamma(4+b)/Gamma(4) * lambda^(-b) * (rho0/rho_air)^0.4

(the density correction is Foote & du Toit 1969).  Cloud water and cloud ice do
not sediment appreciably; their characteristic radius is taken monodisperse
from an assumed number concentration.

This is the documented single-moment closure: the intercept ``N0`` is fixed and
the number concentration is diagnostic, not prognostic.
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as C

_N0 = {"rain": C.N0_r, "snow": C.N0_s, "graupel": C.N0_g, "hail": C.N0_h}
_RHOX = {"rain": C.rho_w, "snow": C.rho_s, "graupel": C.rho_g, "hail": C.rho_h}
_VA = C.VT_A
_VB = C.VT_B

# default (diagnostic) number concentrations for the non-sedimenting classes
NC_DEFAULT = 1.0e8    # cloud droplets [m^-3]  (continental, ~100 cm^-3)
NI_DEFAULT = 1.0e4    # cloud-ice crystals [m^-3]


def _mask_small(q, xp=np):
    q = xp.asarray(q, dtype=float)
    return xp.where(q > C.QSMALL, q, xp.nan)


def lambda_slope(q, rho_air, category, xp=np):
    """Exponential slope lambda [1/m] for a precipitating category."""
    q = _mask_small(q, xp)
    rho_x = _RHOX[category]
    N0 = _N0[category]
    return (math.pi * rho_x * N0 / (xp.asarray(rho_air, dtype=float) * q)) ** 0.25


def number_conc(q, rho_air, category, xp=np):
    """Diagnostic number concentration N [m^-3] = N0 / lambda."""
    lam = lambda_slope(q, rho_air, category, xp)
    return _N0[category] / lam


def characteristic_radius(q, rho_air, category, xp=np):
    """Mean-diameter radius r = 1/(2 lambda) [m] (NaN where q ~ 0)."""
    lam = lambda_slope(q, rho_air, category, xp)
    return 1.0 / (2.0 * lam)


def mass_weighted_vt(q, rho_air, category, xp=np):
    """Mass-weighted terminal fall speed [m/s] (>=0; 0 where q ~ 0)."""
    q = xp.asarray(q, dtype=float)
    rho_air = xp.asarray(rho_air, dtype=float)
    lam = lambda_slope(q, rho_air, category, xp)
    a, b = _VA[category], _VB[category]
    gratio = math.gamma(4.0 + b) / C.GAMMA4
    dens_corr = (C.RHO0_VT / xp.maximum(rho_air, C.TINY)) ** 0.4
    vt = a * gratio * lam ** (-b) * dens_corr
    return xp.where(q > C.QSMALL, vt, 0.0)


def cloud_radius(qc, rho_air, Nc=None, xp=np):
    """Volume-mean cloud-droplet radius [m] from q_c and droplet number."""
    qc = xp.asarray(qc, dtype=float)
    Nc = NC_DEFAULT if Nc is None else xp.asarray(Nc, dtype=float)
    mass_per_drop = rho_air * qc / xp.maximum(Nc, C.TINY)     # kg per drop
    vol = mass_per_drop / C.rho_w
    r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    return xp.where(qc > C.QSMALL, r, xp.nan)


def ice_radius(qi, rho_air, Ni=None, xp=np):
    """Volume-mean cloud-ice radius [m] from q_i and crystal number."""
    qi = xp.asarray(qi, dtype=float)
    Ni = NI_DEFAULT if Ni is None else xp.asarray(Ni, dtype=float)
    mass_per_crystal = rho_air * qi / xp.maximum(Ni, C.TINY)
    vol = mass_per_crystal / C.rho_i
    r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    return xp.where(qi > C.QSMALL, r, xp.nan)


__all__ = [
    "lambda_slope", "number_conc", "characteristic_radius", "mass_weighted_vt",
    "cloud_radius", "ice_radius", "NC_DEFAULT", "NI_DEFAULT",
]
